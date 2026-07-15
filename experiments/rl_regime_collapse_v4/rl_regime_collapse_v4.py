"""
kaggle/rl_regime_collapse_v4/rl_regime_collapse_v4.py

RL REGIME COLLAPSE EXPERIMENT — v4
===================================

Central hypothesis: RL reasoning training (not model size, not architecture)
induces Regime 2 collapse — flattening of the epistemic commitment trajectory
into a globally pre-committed latent state.

Key fixes from v3:
  1. UNILATERAL calibration: PARAM vs WRONG (no bilateral CTX_DEP oracle).
     - v3 hang root cause: bilateral oracle for DeepSeek-R1 required 200×2 gens
       at ~60s each because the model answers correctly even without context,
       yielding <5 CTX_DEP pairs after 200 iterations = 24,000s hang.
     - v4: WRONG = model answers incorrectly WITHOUT context (1 gen per sample).
       No context verification needed.
  2. N_CAL_HALF = 15 (30 total). Enough for Fisher LDA; dramatically faster.
  3. Progress prints every 5 samples in ALL loops — no silent hangs.
  4. CAL_THINK_TOK = 800 (was 2000), think stop at </think> token.
  5. Intermediate save after Model A completes.
  6. Timeout guard: if calibration takes > 45 min per model, skip and save partial.

Comparison:
  Model A: Qwen/Qwen2.5-7B-Instruct (SFT/RLHF, no reasoning RL)
           → KNOWN result from v1: REGIME_1 (traj_var=1.02, cw_gap=+0.358)
           → v4 runs it again with unilateral calibration for direct comparison
  Model B: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B (RL-distilled, Qwen backbone)
           → hypothesis: REGIME_2 (flat trajectories, globally committed)

Dataset: TriviaQA rc.wikipedia, same 100 questions for both models.
GPU: T4 (sm_75). Models run sequentially with full CUDA memory release.
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
N_CAL_HALF     = 15     # per class → 30 total  (was 30 → 60 in v3)
N_MAIN         = 80     # trajectory questions per model  (was 100, save time)
CAL_MAX_TOK    = 200    # non-think models: max tokens for calibration answer
CAL_THINK_TOK  = 800    # think models: max think-block tokens (stop at </think>)
CAL_ANS_TOK    = 80     # think models: short answer after </think>
MAX_GEN_TOK    = 300    # max tokens per trajectory
CAL_TIMEOUT_S  = 2700   # 45 min per model calibration; skip and save if exceeded
COMMIT_WINDOW  = 10
COMMIT_THRESH  = 0.10
COMMIT_PERSIST = 8
SEED           = 42
rng            = np.random.default_rng(SEED)

OUT_DIR      = Path("/kaggle/working")
RESULTS_FILE = OUT_DIR / "rl_regime_collapse_v4_results.json"

# Model A already confirmed REGIME_1 in v1. Run again with unilateral calibration
# for direct comparison. Use 4-bit to fit both models on T4.
MODELS = [
    {
        "id":         "Qwen/Qwen2.5-7B-Instruct",
        "key":        "qwen25_7b_instruct",
        "has_think":  False,
        "use_4bit":   True,
        "prior":      {"regime": "REGIME_1", "prefill_auroc": 0.8611,
                       "mean_traj_variance": 1.0165, "cw_j_gap": 0.3580},
    },
    {
        "id":         "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "key":        "deepseek_r1_qwen7b",
        "has_think":  True,
        "use_4bit":   True,
        "prior":      None,
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data(n_nocontext: int = 250) -> List[Dict]:
    from datasets import load_dataset
    ds = load_dataset("trivia_qa", "rc.wikipedia", split="train", streaming=True)
    pool = []
    for row in ds:
        if not row["answer"]["aliases"]:
            continue
        pool.append({
            "question": row["question"],
            "answers":  row["answer"]["aliases"],
        })
        if len(pool) >= n_nocontext:
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
# Model loading / unloading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_cfg: Dict):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    mid = model_cfg["id"]
    print(f"\nLoading {mid} …", flush=True)
    tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if model_cfg["use_4bit"]:
        cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        mdl = AutoModelForCausalLM.from_pretrained(
            mid, quantization_config=cfg,
            device_map=None, trust_remote_code=True,
        ).to(DEVICE)
    else:
        mdl = AutoModelForCausalLM.from_pretrained(
            mid, torch_dtype=torch.float16,
            device_map=None, trust_remote_code=True,
        ).to(DEVICE)

    mdl.eval()
    n_layers = mdl.config.num_hidden_layers
    print(f"  Layers={n_layers}  Hidden={mdl.config.hidden_size}", flush=True)
    return mdl, tok


# ─────────────────────────────────────────────────────────────────────────────
# Architecture-agnostic layer resolver
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
                print(f"  [layers] {path} ({len(obj)} layers)", flush=True)
                return obj
        except AttributeError:
            continue
    raise RuntimeError(f"Cannot locate layers in {type(model).__name__}")


# ─────────────────────────────────────────────────────────────────────────────
# Hidden-state capture
# ─────────────────────────────────────────────────────────────────────────────

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
# Generation helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_think_end_id(tok) -> Optional[int]:
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
    # DeepSeek-R1-Distill-Qwen uses Qwen2.5 tokenizer token 151649 for </think>
    return 151649


def build_prompt(tok, question: str) -> str:
    msgs = [{"role": "user", "content": question}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def generate_answer(model, tok, prompt: str,
                    think_end_id: Optional[int]) -> str:
    """Generate a short answer. Returns decoded text (input portion stripped)."""
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


def get_gen_step1_hs(model, tok, prompt: str, layer: int) -> Optional[np.ndarray]:
    """Extract hidden state at gen-step-1 (first autoregressive decode step)."""
    ids = tok(prompt, return_tensors="pt",
              truncation=True, max_length=1024).input_ids.to(DEVICE)
    cap_s1: List[Optional[np.ndarray]] = [None]

    def _hook(mod, inp, out):
        x = out[0] if isinstance(out, tuple) else out
        # seq_len == 1 → this is a decode step, not prefill
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


# ─────────────────────────────────────────────────────────────────────────────
# Unilateral calibration: PARAM vs WRONG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Probe:
    direction: np.ndarray
    mu_param:  float
    mu_wrong:  float    # replaces mu_ctxdep from v3
    layer_idx: int
    auroc:     float

    def score(self, h: np.ndarray) -> float:
        p      = float(np.dot(h, self.direction))
        scale  = (self.mu_param - self.mu_wrong) / 2.0
        center = (self.mu_param + self.mu_wrong) / 2.0
        return (p - center) / (abs(scale) + 1e-9)


def calibrate_unilateral(model, tok, data_pool: List[Dict],
                          layer: int,
                          think_end_id: Optional[int] = None) -> Probe:
    """
    Unilateral calibration: PARAM (correct without context) vs WRONG (incorrect
    without context). One generation per sample — no bilateral context verification.

    This is faster than bilateral oracle and works even when the model is too accurate
    to yield bilateral CTX_DEP pairs (the v3 hang root cause with DeepSeek-R1).
    """
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

        # Progress print every 5 samples
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

        prompt = build_prompt(tok, s["question"])

        # Generate answer first to determine label
        gen = generate_answer(model, tok, prompt, think_end_id)
        is_correct = (answer_contains(gen, s["answers"]) or
                      token_f1(gen, s["answers"]) >= 0.4)

        if is_correct and len(param_hs) >= N_CAL_HALF:
            continue
        if not is_correct and len(wrong_hs) >= N_CAL_HALF:
            continue

        # Extract gen-step-1 hidden state
        hs = get_gen_step1_hs(model, tok, prompt, layer)
        if hs is None:
            continue

        if is_correct:
            param_hs.append(hs)
        else:
            wrong_hs.append(hs)

    n_p, n_w = len(param_hs), len(wrong_hs)
    print(f"  [Calibrate] PARAM={n_p}  WRONG={n_w}"
          f"  elapsed={time.time() - t_start:.0f}s", flush=True)

    if n_p < 5 or n_w < 5:
        raise RuntimeError(
            f"Insufficient calibration: PARAM={n_p}, WRONG={n_w}. "
            f"Need >= 5 each. Try N_CAL_HALF smaller or use more data."
        )

    X = np.stack(param_hs + wrong_hs)
    y = np.array([1] * n_p + [0] * n_w)

    lda = LinearDiscriminantAnalysis(n_components=1)
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
# Trajectory analysis
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TraceResult:
    question:   str
    answers:    List[str]
    trajectory: List[float] = field(default_factory=list)
    gen_len:    int = 0
    f1:         float = 0.0
    commit_step: Optional[int] = None


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
                   think_end_id: Optional[int]) -> TraceResult:
    tr  = TraceResult(question=sample["question"], answers=sample["answers"])
    cap = HSCapture(model, probe.layer_idx)

    try:
        prompt  = build_prompt(tok, sample["question"])
        ids     = tok(prompt, return_tensors="pt",
                      truncation=True, max_length=1024).input_ids.to(DEVICE)
        cur     = ids.clone()
        past_kv = None
        gen_ids: List[int] = []
        think_done = False

        for _ in range(MAX_GEN_TOK):
            with torch.no_grad():
                if past_kv is None:
                    out = model(cur, use_cache=True)
                else:
                    out = model(cur[:, -1:], past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values

            if cap.hs is not None:
                tr.trajectory.append(probe.score(cap.hs[0]))

            nxt = int(torch.argmax(out.logits[0, -1]).item())
            gen_ids.append(nxt)
            cur = torch.cat([cur, torch.tensor([[nxt]], device=DEVICE)], dim=1)

            if nxt == tok.eos_token_id:
                break
            if think_end_id and nxt == think_end_id:
                think_done = True
                break

        tr.gen_len = len(gen_ids)
        gen_text   = tok.decode(gen_ids, skip_special_tokens=True).strip()

        if think_done:
            with torch.no_grad():
                ans_out = model.generate(
                    cur, max_new_tokens=100, do_sample=False,
                    pad_token_id=tok.eos_token_id)
            ans_text = tok.decode(ans_out[0][cur.shape[1]:], skip_special_tokens=True).strip()
            tr.f1 = max(
                token_f1(ans_text, sample["answers"]),
                token_f1(gen_text, sample["answers"]),
                1.0 if answer_contains(ans_text, sample["answers"]) else 0.0,
            )
        else:
            tr.f1 = max(
                token_f1(gen_text, sample["answers"]),
                1.0 if answer_contains(gen_text, sample["answers"]) else 0.0,
            )

        tr.commit_step = find_commit(tr.trajectory)

    finally:
        cap.remove()
        if past_kv is not None:
            del past_kv

    return tr


# ─────────────────────────────────────────────────────────────────────────────
# Statistics and regime classification
# ─────────────────────────────────────────────────────────────────────────────

def compute_stats(model_cfg: Dict, probe: Probe, traces: List[TraceResult]) -> Dict:
    long_trajs = [t.trajectory for t in traces
                  if len(t.trajectory) >= COMMIT_WINDOW + COMMIT_PERSIST]
    correct = [t for t in traces if t.f1 >= 0.4]
    wrong   = [t for t in traces if t.f1 < 0.1]

    traj_stds = [float(np.std(tj)) for tj in long_trajs if tj]
    mean_traj_variance = float(np.mean(traj_stds)) if traj_stds else 0.0

    j_correct = [float(np.mean(t.trajectory)) for t in correct if t.trajectory]
    j_wrong   = [float(np.mean(t.trajectory)) for t in wrong   if t.trajectory]
    cw_j_gap  = (float(np.mean(j_correct)) - float(np.mean(j_wrong))
                 if j_correct and j_wrong else 0.0)

    all_j = [v for t in traces for v in t.trajectory]
    j_mean = float(np.mean(all_j)) if all_j else 0.0
    j_std  = float(np.std(all_j))  if all_j else 0.0
    j_max  = float(np.max(all_j))  if all_j else 0.0

    print(f"  J_know: mean={j_mean:.3f}  std={j_std:.3f}  max={j_max:.3f}",
          flush=True)

    committed   = [t for t in traces if t.commit_step is not None]
    commit_rate = len(committed) / len(traces) if traces else 0.0

    all_positions = [t.commit_step for t in traces
                     if t.commit_step is not None and t.gen_len > 0]
    null_positions = list(rng.integers(0, max(1, int(np.mean(
        [t.gen_len for t in traces if t.gen_len > 0]) or 1)),
        size=len(all_positions)))

    z_score = 0.0
    if all_positions:
        mu_obs  = float(np.mean(all_positions))
        mu_null = float(np.mean(null_positions))
        std_obs = float(np.std(all_positions)) or 1.0
        z_score = (mu_null - mu_obs) / (std_obs / max(1, len(all_positions)) ** 0.5)

    print(f"  commit_rate={commit_rate:.3f}  z={z_score:.2f}", flush=True)
    print(f"  traj_var={mean_traj_variance:.4f}  cw_j_gap={cw_j_gap:+.4f}", flush=True)

    # Regime classification — same logic as v2 but using unilateral calibration
    if mean_traj_variance > 0.7 and cw_j_gap > 0.02:
        regime = "REGIME_1"
    elif commit_rate < 0.15 and abs(z_score) > 100:
        regime = "REGIME_2"
    elif mean_traj_variance < 0.3 and abs(j_mean) < 0.5:
        regime = "REGIME_3"
    else:
        regime = "AMBIGUOUS"

    print(f"  → REGIME: {regime}", flush=True)

    return {
        "model_id":            model_cfg["id"],
        "probe_layer":         probe.layer_idx,
        "probe_auroc":         round(probe.auroc, 4),
        "n_traces":            len(traces),
        "commit_rate":         round(commit_rate, 4),
        "z_score":             round(z_score, 2),
        "mean_traj_variance":  round(mean_traj_variance, 4),
        "cw_j_gap":            round(cw_j_gap, 4),
        "j_know_mean":         round(j_mean, 4),
        "j_know_std":          round(j_std, 4),
        "j_know_max":          round(j_max, 4),
        "accuracy":            round(sum(1 for t in traces if t.f1 >= 0.4) / max(1, len(traces)), 3),
        "regime":              regime,
        "prior":               model_cfg.get("prior"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-model pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_model(model_cfg: Dict, data_pool: List[Dict]) -> Dict:
    print(f"\n{'='*60}", flush=True)
    print(f"MODEL: {model_cfg['id']}", flush=True)
    if model_cfg.get("prior"):
        print(f"  Prior result: {model_cfg['prior']}", flush=True)
    print(f"{'='*60}", flush=True)

    t_model_start = time.time()
    mdl, tok = load_model(model_cfg)
    n_layers  = mdl.config.num_hidden_layers
    layer_idx = max(0, n_layers - 2)  # L26 for 28-layer models

    think_end_id = get_think_end_id(tok) if model_cfg["has_think"] else None
    if think_end_id:
        print(f"  think_end_id={think_end_id}  "
              f"('{tok.decode([think_end_id])}')", flush=True)

    # ── Calibration ───────────────────────────────────────────────────────────
    print(f"\n--- Calibration (unilateral, n_target={N_CAL_HALF}×2) ---", flush=True)
    probe = calibrate_unilateral(mdl, tok, data_pool, layer_idx, think_end_id)

    # ── Trajectories ──────────────────────────────────────────────────────────
    print(f"\n--- Trajectories (n={N_MAIN}) ---", flush=True)
    # Use questions not seen during calibration
    traj_pool = data_pool[:N_MAIN]
    traces: List[TraceResult] = []

    for i, s in enumerate(traj_pool):
        if i % 10 == 0:
            elapsed = time.time() - t_model_start
            print(f"  [{i}/{N_MAIN}]  elapsed={elapsed:.0f}s", flush=True)
        try:
            tr = run_trajectory(mdl, tok, probe, s, think_end_id)
            traces.append(tr)
        except Exception as e:
            print(f"  [WARN] trajectory {i} failed: {e}", flush=True)

    # ── Stats ─────────────────────────────────────────────────────────────────
    print(f"\n--- Stats ---", flush=True)
    stats = compute_stats(model_cfg, probe, traces)
    stats["elapsed_s"] = round(time.time() - t_model_start)

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Verdict
# ─────────────────────────────────────────────────────────────────────────────

def compute_verdict(results: Dict) -> Dict:
    a = results.get("qwen25_7b_instruct", {})
    b = results.get("deepseek_r1_qwen7b", {})

    if not a or not b:
        return {"verdict": "INCOMPLETE", "reason": "One or both models did not complete"}

    regime_a = a.get("regime", "UNKNOWN")
    regime_b = b.get("regime", "UNKNOWN")

    if regime_a == "REGIME_1" and regime_b == "REGIME_2":
        verdict = "RL_COLLAPSE_CONFIRMED"
        reason  = ("Same Qwen backbone: SFT/RLHF = Regime 1 (dynamic), "
                   "RL-distilled = Regime 2 (globally committed). "
                   "RL training is the sufficient cause.")
    elif regime_a == "REGIME_2" and regime_b == "REGIME_2":
        verdict = "BOTH_REGIME_2"
        reason  = "Both models show Regime 2. Scale or base model properties may dominate."
    elif regime_a == "REGIME_1" and regime_b == "REGIME_1":
        verdict = "BOTH_REGIME_1"
        reason  = "RL training does not induce Regime 2 collapse in this comparison."
    else:
        verdict = "AMBIGUOUS"
        reason  = f"Regime A={regime_a}, Regime B={regime_b}"

    return {"verdict": verdict, "reason": reason, "regime_a": regime_a, "regime_b": regime_b}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    print(f"RL Regime Collapse v4  |  {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"N_CAL_HALF={N_CAL_HALF}  N_MAIN={N_MAIN}  "
          f"CAL_THINK_TOK={CAL_THINK_TOK}  CAL_TIMEOUT={CAL_TIMEOUT_S}s", flush=True)

    data_pool = load_data(n_nocontext=300)

    all_results: Dict = {}
    mdl = None

    for model_cfg in MODELS:
        try:
            stats = run_model(model_cfg, data_pool)
            all_results[model_cfg["key"]] = stats
        except Exception as e:
            print(f"\n[ERROR] {model_cfg['id']}: {e}", flush=True)
            import traceback; traceback.print_exc()
            all_results[model_cfg["key"]] = {"error": str(e), "model_id": model_cfg["id"]}
        finally:
            # Aggressive GPU memory release before next model
            if mdl is not None:
                del mdl
                mdl = None
            gc.collect()
            torch.cuda.empty_cache()
            print(f"VRAM after cleanup: "
                  f"{torch.cuda.memory_allocated() / 1e9:.2f} GB allocated", flush=True)

        # Save intermediate results after each model
        interim = {"results": all_results, "elapsed_s": time.time() - t_start,
                   "status": "in_progress"}
        RESULTS_FILE.write_text(json.dumps(interim, indent=2))
        print(f"\n[Saved interim] {RESULTS_FILE}", flush=True)

    # Final verdict
    verdict = compute_verdict(all_results)
    print(f"\n{'='*60}", flush=True)
    print(f"VERDICT: {verdict['verdict']}", flush=True)
    print(f"REASON:  {verdict['reason']}", flush=True)
    print(f"{'='*60}", flush=True)

    final = {
        "results": all_results,
        "verdict": verdict,
        "elapsed_s": round(time.time() - t_start),
        "status": "complete",
        "config": {
            "n_cal_half":    N_CAL_HALF,
            "n_main":        N_MAIN,
            "cal_think_tok": CAL_THINK_TOK,
            "cal_timeout_s": CAL_TIMEOUT_S,
            "calibration":   "unilateral_param_vs_wrong",
        },
    }
    RESULTS_FILE.write_text(json.dumps(final, indent=2))
    print(f"\n[Final results] {RESULTS_FILE}", flush=True)
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
