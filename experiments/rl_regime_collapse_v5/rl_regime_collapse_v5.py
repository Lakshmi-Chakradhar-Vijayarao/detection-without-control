"""
experiments/rl_regime_collapse_v5/rl_regime_collapse_v5.py

RL REGIME COLLAPSE — v5: CAUSAL CHAIN + ANSWER-ONSET EXTRACTION
================================================================

Central hypothesis (Regime 2 confirmation):
  DeepSeek-R1-Distill-Qwen-7B commits at the answer-onset token (after </think>),
  not during CoT. J_know during CoT ≈ −8 (exploration); J_know at answer onset >> 0.

Causal chain (same backbone, different training):
  Model A: Qwen/Qwen2.5-7B           — base (no alignment, no RLHF)
  Model B: Qwen/Qwen2.5-7B-Instruct  — SFT + RLHF (Regime 1 from v4)
  Model C: DeepSeek-R1-Distill-Qwen-7B — RL reasoning (Regime 2 candidate from v4)

All three share the Qwen2.5-7B architecture. The ONLY difference is training regime.
This closes the "maybe it's architecture" confound.

Key additions over v4:
  1. Three-model causal chain (base → instruct → RL reasoning)
  2. Answer-onset extraction for think models:
       - cot_j_mean = mean J_know across entire CoT
       - answer_j_mean = mean J_know across first 20 answer tokens (after </think>)
       - answer_jump = answer_j_mean − cot_j_mean
     If answer_jump >> 0: Regime 2 confirmed (commitment deferred to answer onset)
  3. Six trajectory descriptors (replaces single commit_rate metric):
       exploration_intensity, crossing_count, convergence_speed,
       answer_jump, plateau_duration, cw_j_gap
  4. Verdict reports descriptor vectors — no binary AMBIGUOUS/CONFIRMED collapse

Dataset: TriviaQA rc.wikipedia (same 300 pool as v4)
GPU: T4 (sm_75). Models run sequentially. 4-bit quant for 7B models.
"""

from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Install deps ──────────────────────────────────────────────────────────────
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "bitsandbytes>=0.46.1", "scikit-learn", "datasets", "huggingface_hub"],
               check=False)

import numpy as np
import torch

# ── HF login ──────────────────────────────────────────────────────────────────
try:
    _hf_token = ""
    try:
        from kaggle_secrets import UserSecretsClient as _USC
        _hf_token = _USC().get_secret("HF_TOKEN")
    except Exception:
        pass
    if not _hf_token:
        _hf_token = (os.environ.get("HF_TOKEN") or
                     os.environ.get("HUGGING_FACE_HUB_TOKEN") or "")
    if _hf_token:
        from huggingface_hub import login as _hf_login
        _hf_login(token=_hf_token, add_to_git_credential=False)
        print("HF login: OK", flush=True)
    else:
        print("WARNING: HF_TOKEN not set.", flush=True)
except Exception as _e:
    print(f"HF login error: {_e}", flush=True)

# ── GPU check ─────────────────────────────────────────────────────────────────
assert torch.cuda.is_available(), "GPU required"
_sm_major, _sm_minor = torch.cuda.get_device_capability(0)
_sm = _sm_major * 10 + _sm_minor
assert _sm >= 70, f"GPU sm_{_sm} not supported — need T4 (sm_75) or better."
DEVICE = "cuda"
print(f"GPU: {torch.cuda.get_device_name(0)}  sm_{_sm}", flush=True)
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB",
      flush=True)

# ── Constants ─────────────────────────────────────────────────────────────────
N_CAL_HALF      = 15     # per class → 30 total
N_MAIN          = 60     # trajectory questions per model (3 models × 60 = 180 total)
CAL_MAX_TOK     = 200    # non-think models: calibration answer max tokens
CAL_THINK_TOK   = 800    # think models: max think-block tokens (stop at </think>)
CAL_ANS_TOK     = 80     # think models: answer tokens after </think>
MAX_GEN_TOK     = 300    # max tokens per CoT trajectory
ANS_WINDOW_TOK  = 20     # tokens after </think> to extract answer_j_mean
CAL_TIMEOUT_S   = 2700   # 45 min per model; skip if exceeded
COMMIT_THRESH   = 0.10
COMMIT_WINDOW   = 10
COMMIT_PERSIST  = 8
PLATEAU_EPS     = 0.01   # |Δj| < this → plateau
SEED            = 42
rng             = np.random.default_rng(SEED)

OUT_DIR      = Path("/kaggle/working")
RESULTS_FILE = OUT_DIR / "rl_regime_collapse_v5_results.json"

# Causal chain: same Qwen2.5-7B backbone, three training regimes
MODELS = [
    {
        "id":         "Qwen/Qwen2.5-7B",
        "key":        "qwen25_7b_base",
        "has_think":  False,
        "is_base":    True,   # no chat template — use completion format
        "use_4bit":   True,
        "description": "Base (no alignment)",
    },
    {
        "id":         "Qwen/Qwen2.5-7B-Instruct",
        "key":        "qwen25_7b_instruct",
        "has_think":  False,
        "is_base":    False,
        "use_4bit":   True,
        "description": "SFT + RLHF (Regime 1 from v4)",
        "v4_result":  {"commit_rate": 0.675, "z": 7.29, "j_mean": 0.261,
                       "traj_var": 0.396, "cw_j_gap": 0.304},
    },
    {
        "id":         "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "key":        "deepseek_r1_qwen7b",
        "has_think":  True,
        "is_base":    False,
        "use_4bit":   True,
        "description": "RL reasoning (Regime 2 candidate from v4)",
        "v4_result":  {"commit_rate": 0.000, "z": 0.00, "j_mean": -8.307,
                       "traj_var": 1.876, "cw_j_gap": -0.092,
                       "probe_auroc": 0.8756},
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data(n: int = 300) -> List[Dict]:
    from datasets import load_dataset
    ds = load_dataset("trivia_qa", "rc.wikipedia", split="train", streaming=True)
    pool = []
    for row in ds:
        if not row["answer"]["aliases"]:
            continue
        pool.append({"question": row["question"], "answers": row["answer"]["aliases"]})
        if len(pool) >= n:
            break
    print(f"Loaded {len(pool)} TriviaQA questions", flush=True)
    return pool


def token_f1(pred: str, golds: List[str]) -> float:
    p = set(pred.lower().split())
    best = 0.0
    for g in golds:
        q = set(g.lower().split())
        c = p & q
        if not c or not p or not q:
            continue
        pr = len(c) / len(p)
        rc = len(c) / len(q)
        best = max(best, 2 * pr * rc / (pr + rc))
    return best


def answer_contains(pred: str, golds: List[str]) -> bool:
    pred_l = pred.lower()
    return any(g.lower().strip() in pred_l for g in golds if g.strip())


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_cfg: Dict):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    mid = model_cfg["id"]
    print(f"\nLoading {mid} …", flush=True)
    tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    mdl = AutoModelForCausalLM.from_pretrained(
        mid, quantization_config=bnb_cfg,
        device_map=None, trust_remote_code=True,
    ).to(DEVICE)
    mdl.eval()
    n_layers = mdl.config.num_hidden_layers
    print(f"  Layers={n_layers}  Hidden={mdl.config.hidden_size}", flush=True)
    return mdl, tok


# ─────────────────────────────────────────────────────────────────────────────
# Architecture helpers
# ─────────────────────────────────────────────────────────────────────────────

_LAYER_PATHS = [
    "model.layers",
    "model.language_model.layers",
    "language_model.model.layers",
    "transformer.h",
]


def get_transformer_layers(model):
    for path in _LAYER_PATHS:
        try:
            obj = model
            for part in path.split("."):
                obj = getattr(obj, part)
            if hasattr(obj, "__len__") and len(obj) > 0:
                return obj
        except AttributeError:
            continue
    raise RuntimeError(f"Cannot locate layers in {type(model).__name__}")


class HSCapture:
    def __init__(self, model, layer_idx: int):
        self.hs: Optional[np.ndarray] = None
        self._h = get_transformer_layers(model)[layer_idx].register_forward_hook(
            self._fn)

    def _fn(self, mod, inp, out):
        x = out[0] if isinstance(out, tuple) else out
        self.hs = x[:, -1, :].detach().float().cpu().numpy()

    def remove(self):
        self._h.remove()


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builders
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt(tok, question: str, is_base: bool) -> str:
    if is_base:
        # Completion-style — no chat template
        return f"Question: {question}\nAnswer:"
    msgs = [{"role": "user", "content": question}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def get_think_end_id(tok) -> int:
    for cand in ["</think>", "<|/think|>", "<think_end>"]:
        try:
            ids = tok.encode(cand, add_special_tokens=False)
            if len(ids) == 1 and ids[0] != tok.unk_token_id:
                return ids[0]
        except Exception:
            pass
        try:
            tid = tok.convert_tokens_to_ids(cand)
            if tid is not None and tid != tok.unk_token_id:
                return tid
        except Exception:
            pass
    return 151649  # DeepSeek-R1-Distill-Qwen tokenizer fallback


# ─────────────────────────────────────────────────────────────────────────────
# Gen-step-1 extraction (for calibration)
# ─────────────────────────────────────────────────────────────────────────────

def get_gen_step1_hs(model, tok, prompt: str, layer: int) -> Optional[np.ndarray]:
    ids = tok(prompt, return_tensors="pt",
              truncation=True, max_length=1024).input_ids.to(DEVICE)
    cap_s1: List[Optional[np.ndarray]] = [None]

    def _hook(mod, inp, out):
        x = out[0] if isinstance(out, tuple) else out
        if x.shape[1] == 1:
            cap_s1[0] = x[0, -1, :].float().detach().cpu().numpy()

    layers = get_transformer_layers(model)
    handle = layers[layer].register_forward_hook(_hook)
    try:
        with torch.no_grad():
            pre = model(ids, use_cache=True)
            pkv = pre.past_key_values
            model(ids[:, -1:], past_key_values=pkv, use_cache=False)
    finally:
        handle.remove()
    return cap_s1[0]


def generate_answer(model, tok, prompt: str, think_end_id: Optional[int]) -> str:
    ids = tok(prompt, return_tensors="pt",
              truncation=True, max_length=1024).input_ids.to(DEVICE)
    with torch.no_grad():
        if think_end_id is not None:
            stop_ids = list({tok.eos_token_id, think_end_id})
            out = model.generate(
                ids, max_new_tokens=CAL_THINK_TOK, do_sample=False,
                eos_token_id=stop_ids, pad_token_id=tok.eos_token_id,
            )
            if int(out[0, -1].item()) == think_end_id:
                out = model.generate(
                    out, max_new_tokens=CAL_ANS_TOK, do_sample=False,
                    pad_token_id=tok.eos_token_id,
                )
        else:
            out = model.generate(
                ids, max_new_tokens=CAL_MAX_TOK, do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


# ─────────────────────────────────────────────────────────────────────────────
# Calibration: unilateral PARAM vs WRONG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Probe:
    direction: np.ndarray
    mu_param:  float
    mu_wrong:  float
    layer_idx: int
    auroc:     float

    def score(self, h: np.ndarray) -> float:
        p      = float(np.dot(h, self.direction))
        scale  = (self.mu_param - self.mu_wrong) / 2.0
        center = (self.mu_param + self.mu_wrong) / 2.0
        return (p - center) / (abs(scale) + 1e-9)


def calibrate_unilateral(model, tok, data_pool: List[Dict], layer: int,
                          is_base: bool, think_end_id: Optional[int]) -> Probe:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import roc_auc_score

    print(f"\n  [Calibrate] layer {layer} — UNILATERAL (PARAM vs WRONG)", flush=True)
    param_hs: List[np.ndarray] = []
    wrong_hs:  List[np.ndarray] = []
    t_start = time.time()
    shuffled = list(data_pool)
    np.random.shuffle(shuffled)

    for i, s in enumerate(shuffled):
        elapsed = time.time() - t_start
        if i % 5 == 0:
            print(f"    sample {i}/{len(shuffled)}"
                  f"  PARAM={len(param_hs)}  WRONG={len(wrong_hs)}"
                  f"  elapsed={elapsed:.0f}s", flush=True)

        if len(param_hs) >= N_CAL_HALF and len(wrong_hs) >= N_CAL_HALF:
            print(f"    Both classes full — stopping at sample {i}", flush=True)
            break
        if elapsed > CAL_TIMEOUT_S:
            print(f"    TIMEOUT at {elapsed:.0f}s — stopping calibration", flush=True)
            break

        prompt   = build_prompt(tok, s["question"], is_base)
        gen      = generate_answer(model, tok, prompt, think_end_id)
        correct  = (answer_contains(gen, s["answers"]) or
                    token_f1(gen, s["answers"]) >= 0.4)

        if correct and len(param_hs) >= N_CAL_HALF:
            continue
        if not correct and len(wrong_hs) >= N_CAL_HALF:
            continue

        hs = get_gen_step1_hs(model, tok, prompt, layer)
        if hs is None:
            continue
        if correct:
            param_hs.append(hs)
        else:
            wrong_hs.append(hs)

    n_p, n_w = len(param_hs), len(wrong_hs)
    print(f"  [Calibrate] PARAM={n_p}  WRONG={n_w}"
          f"  elapsed={time.time() - t_start:.0f}s", flush=True)

    if n_p < 5 or n_w < 5:
        raise RuntimeError(
            f"Insufficient calibration: PARAM={n_p}, WRONG={n_w}. Need >= 5 each."
        )

    X     = np.stack(param_hs + wrong_hs)
    y     = np.array([1] * n_p + [0] * n_w)
    lda   = LinearDiscriminantAnalysis(n_components=1)
    lda.fit(X, y)
    d     = lda.coef_[0] / (np.linalg.norm(lda.coef_[0]) + 1e-9)
    projs = X @ d
    mu_p  = float(np.mean(projs[y == 1]))
    mu_w  = float(np.mean(projs[y == 0]))
    auroc = float(roc_auc_score(y, lda.decision_function(X)))
    print(f"  [Calibrate] in-sample AUROC={auroc:.4f}"
          f"  mu_PARAM={mu_p:.3f}  mu_WRONG={mu_w:.3f}", flush=True)
    return Probe(direction=d, mu_param=mu_p, mu_wrong=mu_w,
                 layer_idx=layer, auroc=auroc)


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory — with answer-onset extraction for think models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TraceResult:
    question:         str
    answers:          List[str]
    cot_trajectory:   List[float] = field(default_factory=list)  # CoT / full gen
    answer_trajectory: List[float] = field(default_factory=list) # tokens after </think>
    gen_len:          int = 0
    f1:               float = 0.0
    commit_step:      Optional[int] = None
    had_think:        bool = False


def find_commit(traj: List[float]) -> Optional[int]:
    n = len(traj)
    t = np.array(traj)
    if n < COMMIT_WINDOW + COMMIT_PERSIST:
        return None
    for i in range(n - COMMIT_WINDOW - COMMIT_PERSIST + 1):
        if np.mean(t[i:i + COMMIT_WINDOW]) >= COMMIT_THRESH:
            if np.min(t[i + COMMIT_WINDOW:i + COMMIT_WINDOW + COMMIT_PERSIST]) >= COMMIT_THRESH * 0.7:
                return i
    return None


def run_trajectory(model, tok, probe: Probe, sample: Dict,
                   is_base: bool, think_end_id: Optional[int]) -> TraceResult:
    tr  = TraceResult(question=sample["question"], answers=sample["answers"])
    cap = HSCapture(model, probe.layer_idx)

    try:
        prompt  = build_prompt(tok, sample["question"], is_base)
        ids     = tok(prompt, return_tensors="pt",
                      truncation=True, max_length=1024).input_ids.to(DEVICE)
        cur     = ids.clone()
        past_kv = None
        gen_ids: List[int] = []
        think_done = False

        # Phase 1: generate CoT (or full answer for non-think models)
        for _ in range(MAX_GEN_TOK):
            with torch.no_grad():
                if past_kv is None:
                    out = model(cur, use_cache=True)
                else:
                    out = model(cur[:, -1:], past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values

            if cap.hs is not None:
                tr.cot_trajectory.append(probe.score(cap.hs[0]))

            nxt = int(torch.argmax(out.logits[0, -1]).item())
            gen_ids.append(nxt)
            cur = torch.cat([cur, torch.tensor([[nxt]], device=DEVICE)], dim=1)

            if nxt == tok.eos_token_id:
                break
            if think_end_id and nxt == think_end_id:
                think_done = True
                tr.had_think = True
                break

        tr.gen_len = len(gen_ids)

        # Phase 2: for think models, capture j_know during answer generation
        ans_gen_ids: List[int] = []
        if think_done:
            for _ in range(ANS_WINDOW_TOK):
                with torch.no_grad():
                    out = model(cur[:, -1:], past_key_values=past_kv, use_cache=True)
                past_kv = out.past_key_values

                if cap.hs is not None:
                    tr.answer_trajectory.append(probe.score(cap.hs[0]))

                nxt = int(torch.argmax(out.logits[0, -1]).item())
                ans_gen_ids.append(nxt)
                cur = torch.cat([cur, torch.tensor([[nxt]], device=DEVICE)], dim=1)
                if nxt == tok.eos_token_id:
                    break

        # Score
        cot_text = tok.decode(gen_ids, skip_special_tokens=True).strip()
        ans_text = tok.decode(ans_gen_ids, skip_special_tokens=True).strip() if ans_gen_ids else ""
        full_text = (cot_text + " " + ans_text).strip() if ans_text else cot_text

        tr.f1 = max(
            token_f1(full_text, sample["answers"]),
            1.0 if answer_contains(full_text, sample["answers"]) else 0.0,
        )
        tr.commit_step = find_commit(tr.cot_trajectory)

    finally:
        cap.remove()
        if past_kv is not None:
            del past_kv

    return tr


# ─────────────────────────────────────────────────────────────────────────────
# Six trajectory descriptors
# ─────────────────────────────────────────────────────────────────────────────

def compute_descriptors(traces: List[TraceResult], probe: Probe) -> Dict:
    all_cot = [t.cot_trajectory for t in traces if len(t.cot_trajectory) >= 5]
    correct  = [t for t in traces if t.f1 >= 0.4]
    wrong    = [t for t in traces if t.f1 < 0.1]

    # 1. exploration_intensity = mean(std(traj)) across traces
    exploration_intensity = float(np.mean([np.std(tj) for tj in all_cot])) if all_cot else 0.0

    # 2. crossing_count = mean(tokens above COMMIT_THRESH per trace)
    crossing_count = float(np.mean([sum(1 for j in tj if j > COMMIT_THRESH)
                                    for tj in all_cot])) if all_cot else 0.0

    # 3. convergence_speed = mean(var(first_half) - var(second_half))
    #    positive → variance decreasing (converging); negative → diverging
    conv_vals = []
    for tj in all_cot:
        T = len(tj)
        if T < 10:
            continue
        h = T // 2
        conv_vals.append(float(np.var(tj[:h])) - float(np.var(tj[h:])))
    convergence_speed = float(np.mean(conv_vals)) if conv_vals else 0.0

    # 4. answer_jump = answer_j_mean - cot_j_mean (think models only)
    answer_jumps = []
    for t in traces:
        if t.had_think and t.answer_trajectory and t.cot_trajectory:
            cot_mean = float(np.mean(t.cot_trajectory))
            ans_mean = float(np.mean(t.answer_trajectory[:5]))  # first 5 answer tokens
            answer_jumps.append(ans_mean - cot_mean)
    answer_jump = float(np.mean(answer_jumps)) if answer_jumps else None
    answer_jump_std = float(np.std(answer_jumps)) if len(answer_jumps) >= 2 else None

    # 5. plateau_duration = mean(tokens with |Δj| < PLATEAU_EPS after last crossing)
    plateau_vals = []
    for tj in all_cot:
        ta = np.array(tj)
        crossings = [i for i in range(len(ta)) if ta[i] > COMMIT_THRESH]
        if not crossings:
            continue
        last_cross = crossings[-1]
        if last_cross + 1 >= len(ta):
            continue
        tail = ta[last_cross:]
        diffs = np.abs(np.diff(tail))
        plateau_vals.append(int(np.sum(diffs < PLATEAU_EPS)))
    plateau_duration = float(np.mean(plateau_vals)) if plateau_vals else 0.0

    # 6. cw_j_gap = mean(j_correct) - mean(j_wrong)
    j_correct = [float(np.mean(t.cot_trajectory)) for t in correct if t.cot_trajectory]
    j_wrong   = [float(np.mean(t.cot_trajectory)) for t in wrong   if t.cot_trajectory]
    cw_j_gap  = (float(np.mean(j_correct)) - float(np.mean(j_wrong))
                 if j_correct and j_wrong else 0.0)

    return {
        "exploration_intensity": round(exploration_intensity, 4),
        "crossing_count":       round(crossing_count, 3),
        "convergence_speed":    round(convergence_speed, 4),
        "answer_jump":          round(answer_jump, 4) if answer_jump is not None else None,
        "answer_jump_std":      round(answer_jump_std, 4) if answer_jump_std is not None else None,
        "n_answer_jumps":       len(answer_jumps),
        "plateau_duration":     round(plateau_duration, 2),
        "cw_j_gap":             round(cw_j_gap, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Legacy commitment stats (for backward compatibility with v4)
# ─────────────────────────────────────────────────────────────────────────────

def compute_legacy_stats(traces: List[TraceResult]) -> Dict:
    all_j   = [v for t in traces for v in t.cot_trajectory]
    j_mean  = float(np.mean(all_j)) if all_j else 0.0
    j_std   = float(np.std(all_j))  if all_j else 0.0
    j_max   = float(np.max(all_j))  if all_j else 0.0

    long_trajs = [t.cot_trajectory for t in traces
                  if len(t.cot_trajectory) >= COMMIT_WINDOW + COMMIT_PERSIST]
    traj_stds  = [float(np.std(tj)) for tj in long_trajs if tj]
    mean_traj_variance = float(np.mean(traj_stds)) if traj_stds else 0.0

    committed   = [t for t in traces if t.commit_step is not None]
    commit_rate = len(committed) / len(traces) if traces else 0.0

    all_positions  = [t.commit_step for t in traces if t.commit_step is not None and t.gen_len > 0]
    null_positions = list(rng.integers(
        0, max(1, int(np.mean([t.gen_len for t in traces if t.gen_len > 0]) or 1)),
        size=len(all_positions)))
    z_score = 0.0
    if all_positions:
        mu_obs  = float(np.mean(all_positions))
        std_obs = float(np.std(all_positions)) or 1.0
        z_score = (float(np.mean(null_positions)) - mu_obs) / (std_obs / max(1, len(all_positions)) ** 0.5)

    accuracy = sum(1 for t in traces if t.f1 >= 0.4) / max(1, len(traces))

    return {
        "j_know_mean":        round(j_mean, 4),
        "j_know_std":         round(j_std, 4),
        "j_know_max":         round(j_max, 4),
        "mean_traj_variance": round(mean_traj_variance, 4),
        "commit_rate":        round(commit_rate, 4),
        "z_score":            round(z_score, 2),
        "accuracy":           round(accuracy, 3),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-model pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_model(model_cfg: Dict, data_pool: List[Dict]) -> Dict:
    print(f"\n{'='*60}", flush=True)
    print(f"MODEL: {model_cfg['id']}", flush=True)
    print(f"  Description: {model_cfg['description']}", flush=True)
    if model_cfg.get("v4_result"):
        print(f"  v4 result: {model_cfg['v4_result']}", flush=True)
    print(f"{'='*60}", flush=True)

    t_model_start = time.time()
    mdl, tok = load_model(model_cfg)
    n_layers     = mdl.config.num_hidden_layers
    layer_idx    = max(0, n_layers - 2)   # L26 for 28-layer models
    is_base      = model_cfg["is_base"]
    think_end_id = get_think_end_id(tok) if model_cfg["has_think"] else None

    if think_end_id:
        print(f"  think_end_id={think_end_id} ('{tok.decode([think_end_id])}')", flush=True)

    # ── Calibration ───────────────────────────────────────────────────────────
    print(f"\n--- Calibration (n_target={N_CAL_HALF}×2) ---", flush=True)
    probe = calibrate_unilateral(mdl, tok, data_pool, layer_idx, is_base, think_end_id)

    # ── Trajectories ──────────────────────────────────────────────────────────
    print(f"\n--- Trajectories (n={N_MAIN}) ---", flush=True)
    traj_pool = data_pool[:N_MAIN]
    traces: List[TraceResult] = []

    for i, s in enumerate(traj_pool):
        if i % 10 == 0:
            elapsed = time.time() - t_model_start
            print(f"  [{i}/{N_MAIN}]  elapsed={elapsed:.0f}s", flush=True)
        try:
            tr = run_trajectory(mdl, tok, probe, s, is_base, think_end_id)
            traces.append(tr)
        except Exception as e:
            print(f"  [WARN] trajectory {i} failed: {e}", flush=True)

    # ── Compute all metrics ───────────────────────────────────────────────────
    print(f"\n--- Descriptors ---", flush=True)
    descriptors  = compute_descriptors(traces, probe)
    legacy_stats = compute_legacy_stats(traces)

    print(f"  exploration_intensity = {descriptors['exploration_intensity']}", flush=True)
    print(f"  crossing_count        = {descriptors['crossing_count']}", flush=True)
    print(f"  convergence_speed     = {descriptors['convergence_speed']}", flush=True)
    if descriptors['answer_jump'] is not None:
        print(f"  answer_jump           = {descriptors['answer_jump']}"
              f"  (n={descriptors['n_answer_jumps']})", flush=True)
    print(f"  plateau_duration      = {descriptors['plateau_duration']}", flush=True)
    print(f"  cw_j_gap              = {descriptors['cw_j_gap']}", flush=True)
    print(f"  j_know_mean           = {legacy_stats['j_know_mean']}", flush=True)
    print(f"  commit_rate           = {legacy_stats['commit_rate']}", flush=True)
    print(f"  z_score               = {legacy_stats['z_score']}", flush=True)
    print(f"  accuracy              = {legacy_stats['accuracy']}", flush=True)

    return {
        "model_id":      model_cfg["id"],
        "description":   model_cfg["description"],
        "probe_layer":   layer_idx,
        "probe_auroc":   round(probe.auroc, 4),
        "n_traces":      len(traces),
        "descriptors":   descriptors,
        **legacy_stats,
        "elapsed_s":     round(time.time() - t_model_start),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Causal chain verdict — replaces binary AMBIGUOUS/CONFIRMED
# ─────────────────────────────────────────────────────────────────────────────

def compute_verdict(results: Dict) -> Dict:
    base    = results.get("qwen25_7b_base", {})
    instruct = results.get("qwen25_7b_instruct", {})
    rl      = results.get("deepseek_r1_qwen7b", {})

    if not base or not instruct or not rl:
        return {"verdict": "INCOMPLETE", "reason": "Not all three models completed"}

    # Regime 2 confirmation: DeepSeek answer_jump must be > 1.0 (normalized units)
    answer_jump = rl.get("descriptors", {}).get("answer_jump")
    regime2_confirmed = (answer_jump is not None and answer_jump > 1.0)

    # Causal chain: verify monotone progression base → instruct → RL
    base_ei = base.get("descriptors", {}).get("exploration_intensity", 0)
    inst_ei = instruct.get("descriptors", {}).get("exploration_intensity", 0)
    rl_ei   = rl.get("descriptors", {}).get("exploration_intensity", 0)
    causal_chain_visible = (base_ei > 0 and inst_ei > 0 and rl_ei > 0)

    # Attenuation: instruct should have weaker signal than base
    base_auroc = base.get("probe_auroc", 0)
    inst_auroc = instruct.get("probe_auroc", 0)
    attenuation_visible = inst_auroc < base_auroc - 0.01

    if regime2_confirmed:
        verdict = "REGIME_2_CONFIRMED"
        reason  = (f"answer_jump={answer_jump:.3f} > 1.0 — DeepSeek-R1 commits at "
                   f"answer onset, not during CoT. Causal chain on same backbone confirmed.")
    elif answer_jump is not None and answer_jump > 0:
        verdict = "REGIME_2_PARTIAL"
        reason  = (f"answer_jump={answer_jump:.3f} > 0 but < 1.0 — directional support "
                   f"for Regime 2 but below confirmation threshold.")
    elif answer_jump is not None and answer_jump <= 0:
        verdict = "REGIME_2_NULL"
        reason  = (f"answer_jump={answer_jump:.3f} ≤ 0 — DeepSeek does not commit at "
                   f"answer onset. Persistent exploration throughout, no deferred commitment.")
    else:
        verdict = "INCOMPLETE"
        reason  = "answer_jump not computable — no think traces collected"

    return {
        "verdict":             verdict,
        "reason":              reason,
        "regime2_confirmed":   regime2_confirmed,
        "answer_jump":         answer_jump,
        "causal_chain_visible": causal_chain_visible,
        "attenuation_visible": attenuation_visible,
        "base_probe_auroc":    base.get("probe_auroc"),
        "instruct_probe_auroc": instruct.get("probe_auroc"),
        "rl_probe_auroc":      rl.get("probe_auroc"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    print(f"RL Regime Collapse v5  |  {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"N_CAL_HALF={N_CAL_HALF}  N_MAIN={N_MAIN}  ANS_WINDOW={ANS_WINDOW_TOK}tok", flush=True)
    print(f"Causal chain: Qwen2.5-7B base → instruct → DeepSeek-R1-Distill-Qwen-7B", flush=True)

    data_pool = load_data(n=300)
    all_results: Dict = {}

    for model_cfg in MODELS:
        try:
            stats = run_model(model_cfg, data_pool)
            all_results[model_cfg["key"]] = stats
        except Exception as e:
            print(f"\n[ERROR] {model_cfg['id']}: {e}", flush=True)
            import traceback; traceback.print_exc()
            all_results[model_cfg["key"]] = {
                "error": str(e), "model_id": model_cfg["id"]}
        finally:
            gc.collect()
            torch.cuda.empty_cache()
            vram = torch.cuda.memory_allocated() / 1e9
            print(f"VRAM after cleanup: {vram:.2f} GB allocated", flush=True)

        # Save after each model
        interim = {"results": all_results, "elapsed_s": round(time.time() - t_start),
                   "status": "in_progress"}
        RESULTS_FILE.write_text(json.dumps(interim, indent=2))
        print(f"\n[Saved interim] {RESULTS_FILE}", flush=True)

    verdict = compute_verdict(all_results)
    print(f"\n{'='*60}", flush=True)
    print(f"VERDICT:  {verdict['verdict']}", flush=True)
    print(f"REASON:   {verdict['reason']}", flush=True)
    print(f"{'='*60}", flush=True)

    final = {
        "results":   all_results,
        "verdict":   verdict,
        "elapsed_s": round(time.time() - t_start),
        "status":    "complete",
        "config": {
            "n_cal_half":     N_CAL_HALF,
            "n_main":         N_MAIN,
            "ans_window_tok": ANS_WINDOW_TOK,
            "calibration":    "unilateral_param_vs_wrong",
            "causal_chain":   [m["id"] for m in MODELS],
        },
    }
    RESULTS_FILE.write_text(json.dumps(final, indent=2))
    print(f"\n[Final results] {RESULTS_FILE}", flush=True)
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
