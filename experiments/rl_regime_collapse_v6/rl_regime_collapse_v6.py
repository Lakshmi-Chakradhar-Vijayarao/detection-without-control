"""
experiments/rl_regime_collapse_v6/rl_regime_collapse_v6.py

RL REGIME COLLAPSE — v6: DEEPSEEK ANSWER-ONSET (STABLE PROBE)
==============================================================

Purpose: Resolve the two open questions from v5:
  1. answer_jump test: does DeepSeek commit at answer onset after </think>?
     v5 had n_answer_jumps=3 (MAX_GEN_TOK=300 too small). v6 uses MAX_GEN_TOK=1500.
  2. probe stability: v4 gave j_mean=-8.307, v5 gave +1.838 with same calibration type
     but different N_CAL (20 vs 30). v6 uses N_CAL=50 per class and multiple seeds.

Model: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B only.
Architecture: Qwen2.5-7B (28 layers, hidden=3584, GQA). 4-bit quantization.
Dataset: TriviaQA rc.wikipedia.
GPU: T4 (sm_75). Expected time: ~90 minutes.

Design:
  - N_CAL=50 per class → total n=100 calibration points (vs 30 in v5)
  - Probe estimated with 5-fold CV to report stability
  - MAX_GEN_TOK=1500 per trajectory (covers most DeepSeek think blocks)
  - N_MAIN=20 trajectories (time budget: ~20 × 4min = ~80min)
  - ANS_WINDOW_TOK=50 (more answer tokens for better statistics)
  - Reports: answer_jump, n_answer_jumps, probe_cv_std

Central test:
  - If answer_jump > 1.0 (normalized units): REGIME_2_CONFIRMED — deferred commitment
  - If answer_jump ≤ 0: REGIME_2_NULL — globally non-committed
  - If n_answer_jumps < 5: BUDGET_INSUFFICIENT — increase MAX_GEN_TOK further
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

assert torch.cuda.is_available(), "GPU required"
DEVICE = "cuda"
_sm_major, _sm_minor = torch.cuda.get_device_capability(0)
_sm = _sm_major * 10 + _sm_minor
print(f"GPU: {torch.cuda.get_device_name(0)}  sm_{_sm}", flush=True)
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", flush=True)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_ID        = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
LAYER_IDX       = 26          # Qwen2.5-7B layer 26 (same as v4/v5)
N_CAL           = 50          # per class — 100 total (vs 30 in v5 — stability)
N_CAL_SEEDS     = 3           # probe stability: fit 3 times, report CV std
MAX_GEN_TOK     = 1500        # covers most DeepSeek think blocks (avg 1000-4000 tok)
CAL_THINK_TOK   = 2000        # calibration generation budget (longer for complete answers)
ANS_WINDOW_TOK  = 50          # answer tokens after </think> to capture
N_MAIN          = 20          # trajectory questions (time budget)
COMMIT_THRESH   = 0.10        # J_know commitment threshold
SEED            = 42
rng             = np.random.default_rng(SEED)

OUT_DIR  = Path("/kaggle/working")
OUT_FILE = OUT_DIR / "rl_regime_collapse_v6_results.json"


# ── Data ──────────────────────────────────────────────────────────────────────
def load_data(n: int = 500) -> List[Dict]:
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

def answer_contains(pred: str, golds: List[str]) -> bool:
    p = pred.lower()
    return any(g.lower().strip() in p for g in golds if g.strip())

def token_f1(pred: str, golds: List[str]) -> float:
    p = set(pred.lower().split())
    best = 0.0
    for g in golds:
        q = set(g.lower().split())
        c = p & q
        if not c or not p or not q:
            continue
        pr, rc = len(c)/len(p), len(c)/len(q)
        best = max(best, 2*pr*rc/(pr+rc))
    return best


# ── Model ─────────────────────────────────────────────────────────────────────
def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    print(f"\nLoading {MODEL_ID} …", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                              bnb_4bit_quant_type="nf4")
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map=None, trust_remote_code=True
    ).to(DEVICE).eval()
    n = mdl.config.num_hidden_layers
    print(f"  Layers={n}  Hidden={mdl.config.hidden_size}", flush=True)
    print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

    # Detect </think> token id
    think_end_id = None
    for cand in ["</think>", "<|endofthought|>"]:
        try:
            ids = tok.encode(cand, add_special_tokens=False)
            if ids:
                think_end_id = ids[-1]
                break
        except Exception:
            pass
    print(f"  think_end_id = {think_end_id}", flush=True)
    return mdl, tok, think_end_id


def get_step1_hs(model, tok, prompt: str) -> Optional[np.ndarray]:
    ids = tok(prompt, return_tensors="pt", truncation=True, max_length=1024).input_ids.to(DEVICE)
    captured = [None]
    def hook(mod, inp, out):
        x = out[0] if isinstance(out, tuple) else out
        if x.shape[1] == 1:
            captured[0] = x[0, -1, :].float().detach().cpu().numpy()
    h = model.model.layers[LAYER_IDX].register_forward_hook(hook)
    try:
        with torch.no_grad():
            pre = model(ids, use_cache=True)
            model(ids[:, -1:], past_key_values=pre.past_key_values, use_cache=False)
    finally:
        h.remove()
    return captured[0]


# ── Hook-based hidden state capture ───────────────────────────────────────────
class HiddenStateCapture:
    def __init__(self):
        self.hs: Optional[np.ndarray] = None
        self._h = None

    def attach(self, model):
        def hook(mod, inp, out):
            x = out[0] if isinstance(out, tuple) else out
            if x.shape[1] == 1:
                self.hs = x[0, -1, :].float().detach().cpu().numpy()
        self._h = model.model.layers[LAYER_IDX].register_forward_hook(hook)

    def detach(self):
        if self._h is not None:
            self._h.remove()
            self._h = None


# ── Fisher LDA probe ──────────────────────────────────────────────────────────
class FisherProbe:
    def __init__(self, direction: np.ndarray, mu_p: float, mu_w: float):
        self.d    = direction
        self.mu_p = mu_p
        self.mu_w = mu_w
        self.mu_thresh = (mu_p + mu_w) / 2

    def score(self, hs: np.ndarray) -> float:
        return float(hs @ self.d)

    def committed(self, hs: np.ndarray) -> bool:
        return self.score(hs) > COMMIT_THRESH


# ── Calibration (PARAM vs WRONG, N_CAL per class) ─────────────────────────────
def calibrate(model, tok, data: List[Dict]) -> Tuple[FisherProbe, float, float]:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    print(f"\n  Calibrating (PARAM vs WRONG, N_CAL={N_CAL} per class) …", flush=True)
    param_hs, wrong_hs = [], []
    shuffled = list(data)
    rng.shuffle(shuffled)
    t0 = time.time()

    for i, s in enumerate(shuffled):
        if len(param_hs) >= N_CAL and len(wrong_hs) >= N_CAL:
            print(f"    Done at sample {i}  elapsed={time.time()-t0:.0f}s", flush=True)
            break
        if i % 5 == 0:
            print(f"    [{i}] PARAM={len(param_hs)} WRONG={len(wrong_hs)} "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)

        msgs = [{"role": "user", "content": s["question"]}]
        try:
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt = f"Question: {s['question']}\nAnswer:"

        ids = tok(prompt, return_tensors="pt", truncation=True, max_length=1024).input_ids.to(DEVICE)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=CAL_THINK_TOK, do_sample=False,
                                  pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

        # Strip think block for answer evaluation
        if "</think>" in gen:
            gen_eval = gen.split("</think>", 1)[-1].strip()
        else:
            gen_eval = gen

        correct = answer_contains(gen_eval, s["answers"]) or token_f1(gen_eval, s["answers"]) >= 0.4

        if correct and len(param_hs) >= N_CAL:
            continue
        if not correct and len(wrong_hs) >= N_CAL:
            continue

        hs = get_step1_hs(model, tok, prompt)
        if hs is None:
            continue
        if correct:
            param_hs.append(hs)
        else:
            wrong_hs.append(hs)

    np_, nw = len(param_hs), len(wrong_hs)
    print(f"  Calibration done: PARAM={np_} WRONG={nw}  elapsed={time.time()-t0:.0f}s", flush=True)
    if np_ < 10 or nw < 10:
        raise RuntimeError(f"Insufficient: PARAM={np_}, WRONG={nw}")

    X = np.stack(param_hs + wrong_hs)
    y = np.array([1]*np_ + [0]*nw)

    # Multi-seed probe stability
    cv_aurocs = []
    probe_dirs = []
    for seed in range(N_CAL_SEEDS):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        fold_aurocs = []
        for tr_idx, val_idx in cv.split(X, y):
            lda = LinearDiscriminantAnalysis(n_components=1)
            lda.fit(X[tr_idx], y[tr_idx])
            scores = lda.decision_function(X[val_idx])
            fold_aurocs.append(roc_auc_score(y[val_idx], scores))
        cv_aurocs.append(float(np.mean(fold_aurocs)))

        lda_full = LinearDiscriminantAnalysis(n_components=1)
        lda_full.fit(X, y)
        d = lda_full.coef_[0] / (np.linalg.norm(lda_full.coef_[0]) + 1e-9)
        probe_dirs.append(d)

    probe_auroc_mean = float(np.mean(cv_aurocs))
    probe_auroc_std  = float(np.std(cv_aurocs))

    # Pairwise cosim between probe directions across seeds — stability metric
    cosims = []
    for a in range(len(probe_dirs)):
        for b in range(a+1, len(probe_dirs)):
            cosims.append(float(np.dot(probe_dirs[a], probe_dirs[b])))
    probe_dir_stability = float(np.mean(cosims))  # cosim across seeds

    # Use seed-0 direction for trajectories
    primary_dir = probe_dirs[0]
    lda_final   = LinearDiscriminantAnalysis(n_components=1)
    lda_final.fit(X, y)
    projs  = X @ primary_dir
    mu_p   = float(np.mean(projs[y == 1]))
    mu_w   = float(np.mean(projs[y == 0]))

    print(f"  CV AUROC: {probe_auroc_mean:.4f} ± {probe_auroc_std:.4f} (over {N_CAL_SEEDS} seeds)", flush=True)
    print(f"  Probe direction cosim across seeds: {probe_dir_stability:.4f}", flush=True)
    print(f"  mu_PARAM={mu_p:.3f}  mu_WRONG={mu_w:.3f}", flush=True)

    return (FisherProbe(primary_dir, mu_p, mu_w),
            probe_auroc_mean, probe_auroc_std, probe_dir_stability)


# ── Trajectory run ─────────────────────────────────────────────────────────────
@dataclass
class TraceResult:
    question:          str
    answers:           List[str]
    cot_trajectory:    List[float] = field(default_factory=list)
    answer_trajectory: List[float] = field(default_factory=list)
    gen_len:           int = 0
    f1:                float = 0.0
    had_think:         bool = False
    think_completed:   bool = False


def run_trajectory(model, tok, think_end_id, probe: FisherProbe,
                   s: Dict) -> TraceResult:
    tr = TraceResult(question=s["question"], answers=s["answers"])
    cap = HiddenStateCapture()
    cap.attach(model)

    msgs = [{"role": "user", "content": s["question"]}]
    try:
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        prompt = f"Question: {s['question']}\nAnswer:"

    ids = tok(prompt, return_tensors="pt", truncation=True, max_length=1024).input_ids.to(DEVICE)
    cur = ids

    try:
        with torch.no_grad():
            out = model(cur, use_cache=True)
        past_kv = out.past_key_values
        gen_ids  = []
        ans_gen_ids = []
        think_done = False

        # Phase 1: CoT generation
        for step in range(MAX_GEN_TOK):
            with torch.no_grad():
                out = model(cur[:, -1:], past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            if cap.hs is not None:
                tr.cot_trajectory.append(probe.score(cap.hs))
            nxt = int(torch.argmax(out.logits[0, -1]).item())
            gen_ids.append(nxt)
            cur = torch.cat([cur, torch.tensor([[nxt]], device=DEVICE)], dim=1)

            if nxt == tok.eos_token_id:
                break
            if think_end_id is not None and nxt == think_end_id:
                think_done = True
                tr.had_think = True
                tr.think_completed = True
                break

        # Phase 2: answer tokens after </think>
        if think_done:
            for _ in range(ANS_WINDOW_TOK):
                with torch.no_grad():
                    out = model(cur[:, -1:], past_key_values=past_kv, use_cache=True)
                past_kv = out.past_key_values
                if cap.hs is not None:
                    tr.answer_trajectory.append(probe.score(cap.hs))
                nxt = int(torch.argmax(out.logits[0, -1]).item())
                ans_gen_ids.append(nxt)
                cur = torch.cat([cur, torch.tensor([[nxt]], device=DEVICE)], dim=1)
                if nxt == tok.eos_token_id:
                    break

        # Decode answer
        all_gen = tok.decode(gen_ids, skip_special_tokens=False)
        if "</think>" in all_gen:
            ans_text = all_gen.split("</think>", 1)[-1]
        else:
            ans_text = all_gen
        if ans_gen_ids:
            ans_text += tok.decode(ans_gen_ids, skip_special_tokens=True)

        tr.gen_len = len(gen_ids)
        tr.f1 = max(
            1.0 if answer_contains(ans_text, s["answers"]) else 0.0,
            token_f1(ans_text, s["answers"])
        )
    except Exception as e:
        print(f"    [trace error] {e}", flush=True)
    finally:
        cap.detach()
        del cur

    return tr


# ── Descriptors ───────────────────────────────────────────────────────────────
def compute_descriptors(traces: List[TraceResult], probe: FisherProbe) -> Dict:
    cot_means = [float(np.mean(t.cot_trajectory)) for t in traces if t.cot_trajectory]
    cot_vars  = [float(np.var(t.cot_trajectory))  for t in traces if t.cot_trajectory]

    # Exploration intensity = mean trajectory variance
    exploration_intensity = float(np.mean(cot_vars)) if cot_vars else None

    # Crossing count = threshold crossings per trajectory
    crossing_counts = []
    for t in traces:
        if not t.cot_trajectory:
            continue
        above = [j > COMMIT_THRESH for j in t.cot_trajectory]
        n_cross = sum(1 for i in range(1, len(above)) if above[i] != above[i-1])
        crossing_counts.append(n_cross)

    # answer_jump (only from traces that completed </think>)
    answer_jumps = []
    for t in traces:
        if t.think_completed and t.answer_trajectory and t.cot_trajectory:
            cot_mean  = float(np.mean(t.cot_trajectory))
            ans_mean  = float(np.mean(t.answer_trajectory[:10]))  # first 10 answer tokens
            answer_jumps.append(ans_mean - cot_mean)

    # commit_rate and z_score
    if cot_means:
        commit_flags = [1.0 if m > COMMIT_THRESH else 0.0 for m in cot_means]
        commit_rate  = float(np.mean(commit_flags))
        se = float(np.std(commit_flags)) / (len(commit_flags)**0.5 + 1e-9)
        z_score = commit_rate / (se + 1e-9) if se > 1e-9 else 0.0
    else:
        commit_rate, z_score = 0.0, 0.0

    # cw_j_gap
    correct_means   = [float(np.mean(t.cot_trajectory)) for t in traces
                       if t.cot_trajectory and t.f1 >= 0.4]
    incorrect_means = [float(np.mean(t.cot_trajectory)) for t in traces
                       if t.cot_trajectory and t.f1 < 0.4]
    cw_j_gap = (float(np.mean(correct_means)) - float(np.mean(incorrect_means))
                if correct_means and incorrect_means else None)

    n_think_completed = sum(1 for t in traces if t.think_completed)

    return {
        "exploration_intensity": round(exploration_intensity, 4) if exploration_intensity is not None else None,
        "crossing_count":        round(float(np.mean(crossing_counts)), 3) if crossing_counts else None,
        "answer_jump":           round(float(np.mean(answer_jumps)), 4) if answer_jumps else None,
        "answer_jump_std":       round(float(np.std(answer_jumps)), 4) if len(answer_jumps) > 1 else None,
        "n_answer_jumps":        len(answer_jumps),
        "n_think_completed":     n_think_completed,
        "cw_j_gap":              round(cw_j_gap, 4) if cw_j_gap is not None else None,
    }


# ── Verdict ───────────────────────────────────────────────────────────────────
def compute_verdict(desc: Dict, commit_rate: float, z_score: float,
                    probe_auroc: float) -> Dict:
    n_jumps = desc["n_answer_jumps"]
    aj = desc["answer_jump"]

    if n_jumps < 5:
        verdict = "BUDGET_INSUFFICIENT"
        reason  = (f"n_answer_jumps={n_jumps} < 5 — most think blocks still exceed "
                   f"MAX_GEN_TOK={MAX_GEN_TOK}. Verdict invalid.")
    elif aj is not None and aj > 1.0:
        verdict = "REGIME_2_CONFIRMED"
        reason  = (f"answer_jump={aj:.3f} > 1.0 — DeepSeek commits at answer onset. "
                   "J_know rises significantly after </think>.")
    elif aj is not None and aj <= 0.0:
        verdict = "REGIME_2_NULL"
        reason  = (f"answer_jump={aj:.3f} ≤ 0 — no deferred commitment. "
                   "J_know does not rise at answer onset.")
    elif aj is not None and 0.0 < aj <= 1.0:
        verdict = "REGIME_2_PARTIAL"
        reason  = (f"answer_jump={aj:.3f} in (0, 1.0) — partial onset effect. "
                   "J_know rises at answer onset but below confirmation threshold.")
    else:
        verdict = "INCONCLUSIVE"
        reason  = "answer_jump could not be computed."

    return {
        "verdict":       verdict,
        "reason":        reason,
        "answer_jump":   aj,
        "n_answer_jumps": n_jumps,
        "commit_rate":   commit_rate,
        "z_score":       z_score,
        "probe_auroc":   probe_auroc,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print(f"RL Regime Collapse V6 — DeepSeek Answer-Onset (Stable Probe)", flush=True)
    print(f"Model: {MODEL_ID}", flush=True)
    print(f"N_CAL={N_CAL} | MAX_GEN_TOK={MAX_GEN_TOK} | N_MAIN={N_MAIN}", flush=True)
    print(f"Resolving: answer_jump test + probe stability from v5", flush=True)
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    data = load_data(500)
    model, tok, think_end_id = load_model()

    results: Dict = {
        "model_id":   MODEL_ID,
        "n_cal":      N_CAL,
        "n_main":     N_MAIN,
        "layer":      LAYER_IDX,
        "max_gen_tok": MAX_GEN_TOK,
        "status":     "in_progress",
    }

    # ── Calibration ─────────────────────────────────────────────────────────
    probe, probe_auroc_mean, probe_auroc_std, dir_stability = calibrate(model, tok, data)

    results["probe_auroc_mean"]   = round(probe_auroc_mean, 4)
    results["probe_auroc_std"]    = round(probe_auroc_std, 4)
    results["probe_dir_stability"] = round(dir_stability, 4)
    results["mu_param"]           = round(probe.mu_p, 3)
    results["mu_wrong"]           = round(probe.mu_w, 3)

    OUT_FILE.write_text(json.dumps(results, indent=2))
    print(f"\n  Probe AUROC: {probe_auroc_mean:.4f} ± {probe_auroc_std:.4f}", flush=True)
    print(f"  Direction stability (cosim across seeds): {dir_stability:.4f}", flush=True)

    if dir_stability < 0.5:
        print(f"  WARNING: low direction stability ({dir_stability:.4f}) — probe unreliable", flush=True)

    # ── Trajectories ─────────────────────────────────────────────────────────
    print(f"\n--- Trajectories (n={N_MAIN}, MAX_GEN_TOK={MAX_GEN_TOK}) ---", flush=True)
    traces = []
    shuffled = list(data)
    rng.shuffle(shuffled)
    t1 = time.time()

    for i, s in enumerate(shuffled[:N_MAIN]):
        if i % 5 == 0:
            print(f"  [{i}/{N_MAIN}]  think_completed={sum(t.think_completed for t in traces)} "
                  f"elapsed={time.time()-t1:.0f}s", flush=True)
        tr = run_trajectory(model, tok, think_end_id, probe, s)
        traces.append(tr)
        if i % 5 == 0:
            OUT_FILE.write_text(json.dumps(results | {"traces_done": i+1}, indent=2))

    # ── Descriptors ─────────────────────────────────────────────────────────
    cot_means = [float(np.mean(t.cot_trajectory)) for t in traces if t.cot_trajectory]
    commit_rate = float(np.mean([1.0 if m > COMMIT_THRESH else 0.0 for m in cot_means])) if cot_means else 0.0
    se = float(np.std([1.0 if m > COMMIT_THRESH else 0.0 for m in cot_means])) / (len(cot_means)**0.5 + 1e-9) if cot_means else 1e-9
    z_score = commit_rate / (se + 1e-9) if se > 1e-9 else 0.0

    j_know_mean = float(np.mean(cot_means)) if cot_means else 0.0
    j_know_std  = float(np.std(cot_means))  if cot_means else 0.0
    j_know_max  = float(max((max(t.cot_trajectory) for t in traces if t.cot_trajectory), default=0.0))
    accuracy    = float(np.mean([t.f1 for t in traces])) if traces else 0.0

    desc = compute_descriptors(traces, probe)

    print(f"\n--- Results ---", flush=True)
    print(f"  n_think_completed = {desc['n_think_completed']} / {len(traces)}", flush=True)
    print(f"  n_answer_jumps    = {desc['n_answer_jumps']}", flush=True)
    print(f"  answer_jump       = {desc['answer_jump']}", flush=True)
    print(f"  commit_rate       = {commit_rate:.4f}", flush=True)
    print(f"  z_score           = {z_score:.2f}", flush=True)
    print(f"  j_know_mean       = {j_know_mean:.4f}", flush=True)
    print(f"  exploration       = {desc['exploration_intensity']}", flush=True)

    verdict_d = compute_verdict(desc, commit_rate, z_score, probe_auroc_mean)

    results.update({
        "probe_auroc":   probe_auroc_mean,
        "commit_rate":   round(commit_rate, 4),
        "z_score":       round(z_score, 2),
        "j_know_mean":   round(j_know_mean, 4),
        "j_know_std":    round(j_know_std, 4),
        "j_know_max":    round(j_know_max, 4),
        "accuracy":      round(accuracy, 3),
        "descriptors":   desc,
        "verdict":       verdict_d,
        "elapsed_s":     round(time.time() - t0),
        "status":        "complete",
    })

    print(f"\n{'='*60}", flush=True)
    print(f"  VERDICT: {verdict_d['verdict']}", flush=True)
    print(f"  {verdict_d['reason']}", flush=True)
    print(f"{'='*60}", flush=True)

    OUT_FILE.write_text(json.dumps(results, indent=2))
    print(f"\n[Final] {OUT_FILE}", flush=True)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
