"""
experiments/answer_jump_v2/answer_jump_v2.py

ANSWER JUMP V2 — REPLICATION ON DEEPSEEK-R1-DISTILL-LLAMA-8B
=============================================================

Purpose: Replicate the answer_jump finding from rl_regime_collapse_v6 on a second
RL-trained reasoning model with a *different backbone architecture*.

v6 used: DeepSeek-R1-Distill-Qwen-7B (Qwen2.5-7B backbone, 28 layers, hidden=3584)
v2 uses: DeepSeek-R1-Distill-Llama-8B (Llama-3.1-8B backbone, 32 layers, hidden=4096)

Pre-registered claim being tested:
  answer_jump > 1.0 (normalized probe units) at the </think> boundary.
  This would confirm that the deferred-commitment pattern generalizes across
  RL-distilled models with different backbone architectures.

Pre-registered verdicts (same as v6):
  - REGIME_2_CONFIRMED: answer_jump > 1.0 (replication success)
  - REGIME_2_PARTIAL: 0.0 < answer_jump ≤ 1.0 (direction confirmed, weaker)
  - REGIME_2_NULL: answer_jump ≤ 0 (failure to replicate)
  - BUDGET_INSUFFICIENT: n_answer_jumps < 5

If REGIME_2_CONFIRMED: Claim 3(c) is now supported by two independent RL models
with different architectures. This substantially strengthens the paper.

Architecture differences to watch:
  - Llama-3.1-8B: 32 layers (vs 28 for Qwen2.5-7B), hidden=4096 (vs 3584)
  - We use LAYER_IDX=26 for comparability; this is 81% depth for Llama vs 93% for Qwen
  - If v2 needs a higher layer, run at LAYER_IDX=30 as a sensitivity check

Model: deepseek-ai/DeepSeek-R1-Distill-Llama-8B
Dataset: TriviaQA rc.wikipedia (same as v6)
GPU: T4 (16GB). 4-bit quantization. Expected time: ~90 minutes.

Controls included:
  - commit_rate and z_score as regime 2 markers
  - n_think_completed to validate think-block budget
  - cot_trajectory for exploration intensity check
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

assert torch.cuda.is_available(), "GPU required — T4 expected"
DEVICE = "cuda"
_sm_major, _sm_minor = torch.cuda.get_device_capability(0)
_sm = _sm_major * 10 + _sm_minor
print(f"GPU: {torch.cuda.get_device_name(0)}  sm_{_sm}", flush=True)
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", flush=True)
if _sm < 75:
    raise RuntimeError(f"sm_{_sm} < sm_75 — T4/V100 required for bitsandbytes INT4")

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_ID        = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
LAYER_IDX       = 26          # Llama-3.1-8B has 32 layers; L26 = 81% depth
N_CAL           = 40          # per class (40 PARAM + 40 WRONG = 80 total)
N_CAL_SEEDS     = 3           # probe direction stability over 3 seeds
MAX_GEN_TOK     = 1500        # covers most think blocks
CAL_THINK_TOK   = 1500        # calibration generation budget
ANS_WINDOW_TOK  = 50          # answer tokens after </think>
N_MAIN          = 20          # trajectory questions
COMMIT_THRESH   = 0.10        # J_know commitment threshold (normalized units)
SEED            = 42
rng             = np.random.default_rng(SEED)

OUT_DIR  = Path("/kaggle/working")
OUT_FILE = OUT_DIR / "answer_jump_v2_results.json"


# ── Data ──────────────────────────────────────────────────────────────────────
def load_data(n: int = 600) -> List[Dict]:
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
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                              bnb_4bit_quant_type="nf4")
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map="auto",
        low_cpu_mem_usage=True, trust_remote_code=False
    ).eval()
    n_layers = mdl.config.num_hidden_layers
    hidden_size = mdl.config.hidden_size
    print(f"  Layers={n_layers}  Hidden={hidden_size}", flush=True)
    print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

    if LAYER_IDX >= n_layers:
        raise RuntimeError(f"LAYER_IDX={LAYER_IDX} >= n_layers={n_layers}")

    # Detect </think> token id
    think_end_id = None
    for cand in ["</think>", "<|endofthought|>"]:
        try:
            ids = tok.encode(cand, add_special_tokens=False)
            if ids:
                think_end_id = ids[-1]
                print(f"  think_end_id = {think_end_id} (from '{cand}')", flush=True)
                break
        except Exception:
            pass
    if think_end_id is None:
        print("  WARNING: </think> token not found — answer_jump test may not work", flush=True)

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


# ── Hidden state capture hook ──────────────────────────────────────────────────
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

    def score(self, hs: np.ndarray) -> float:
        return float(hs @ self.d)


# ── Calibration ───────────────────────────────────────────────────────────────
def calibrate(model, tok, data: List[Dict]) -> Tuple[FisherProbe, float, float, float]:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    print(f"\n  Calibrating (PARAM=correct vs WRONG=incorrect, N_CAL={N_CAL}) …", flush=True)
    param_hs, wrong_hs = [], []
    shuffled = list(data)
    rng.shuffle(shuffled)
    t0 = time.time()

    for i, s in enumerate(shuffled):
        if len(param_hs) >= N_CAL and len(wrong_hs) >= N_CAL:
            print(f"    Done at item {i}  elapsed={time.time()-t0:.0f}s", flush=True)
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
        gen_eval = gen.split("</think>", 1)[-1].strip() if "</think>" in gen else gen
        correct = answer_contains(gen_eval, s["answers"]) or token_f1(gen_eval, s["answers"]) >= 0.4

        if correct and len(param_hs) >= N_CAL:
            continue
        if not correct and len(wrong_hs) >= N_CAL:
            continue

        hs = get_step1_hs(model, tok, prompt)
        if hs is None:
            continue
        (param_hs if correct else wrong_hs).append(hs)

    np_, nw = len(param_hs), len(wrong_hs)
    print(f"  Calibration done: PARAM={np_} WRONG={nw}  elapsed={time.time()-t0:.0f}s", flush=True)
    if np_ < 8 or nw < 8:
        raise RuntimeError(f"Insufficient cal samples: PARAM={np_}, WRONG={nw}")

    X = np.stack(param_hs + wrong_hs)
    y = np.array([1]*np_ + [0]*nw)

    cv_aurocs, probe_dirs = [], []
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
    cosims = [float(np.dot(probe_dirs[a], probe_dirs[b]))
              for a in range(len(probe_dirs)) for b in range(a+1, len(probe_dirs))]
    probe_dir_stability = float(np.mean(cosims))

    primary_dir = probe_dirs[0]
    projs = X @ primary_dir
    mu_p = float(np.mean(projs[y == 1]))
    mu_w = float(np.mean(projs[y == 0]))

    print(f"  CV AUROC: {probe_auroc_mean:.4f} ± {probe_auroc_std:.4f}", flush=True)
    print(f"  Probe stability (cosim across seeds): {probe_dir_stability:.4f}", flush=True)
    print(f"  mu_PARAM={mu_p:.3f}  mu_WRONG={mu_w:.3f}", flush=True)

    return FisherProbe(primary_dir, mu_p, mu_w), probe_auroc_mean, probe_auroc_std, probe_dir_stability


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


def run_trajectory(model, tok, think_end_id, probe: FisherProbe, s: Dict) -> TraceResult:
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
        gen_ids = []
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

        all_gen = tok.decode(gen_ids, skip_special_tokens=False)
        ans_text = all_gen.split("</think>", 1)[-1] if "</think>" in all_gen else all_gen
        if ans_gen_ids:
            ans_text += tok.decode(ans_gen_ids, skip_special_tokens=True)
        tr.gen_len = len(gen_ids)
        tr.f1 = max(1.0 if answer_contains(ans_text, s["answers"]) else 0.0,
                    token_f1(ans_text, s["answers"]))
    except Exception as e:
        print(f"    [trace error] {e}", flush=True)
    finally:
        cap.detach()
        del cur
    return tr


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print("ANSWER JUMP V2 — DeepSeek-R1-Distill-Llama-8B", flush=True)
    print(f"Model: {MODEL_ID}", flush=True)
    print(f"Layer: {LAYER_IDX} (Llama-3.1-8B has 32 layers; L26 = 81% depth)", flush=True)
    print(f"N_CAL={N_CAL} | N_MAIN={N_MAIN} | MAX_GEN_TOK={MAX_GEN_TOK}", flush=True)
    print(f"Pre-registered claim: answer_jump > 1.0 at </think> boundary", flush=True)
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    data = load_data(600)
    model, tok, think_end_id = load_model()

    results: Dict = {
        "model_id": MODEL_ID,
        "backbone": "Llama-3.1-8B",
        "layer": LAYER_IDX,
        "n_cal": N_CAL,
        "n_main": N_MAIN,
        "max_gen_tok": MAX_GEN_TOK,
        "preregistered_claim": "answer_jump > 1.0 at </think> → REGIME_2_CONFIRMED",
        "v6_result_to_replicate": {"model": "DeepSeek-R1-Distill-Qwen-7B",
                                   "answer_jump": 5.54, "n_answer_jumps": 17},
        "status": "in_progress",
    }

    probe, probe_auroc_mean, probe_auroc_std, dir_stability = calibrate(model, tok, data)

    results.update({
        "probe_auroc_mean": round(probe_auroc_mean, 4),
        "probe_auroc_std":  round(probe_auroc_std, 4),
        "probe_dir_stability": round(dir_stability, 4),
        "mu_param": round(probe.mu_p, 3),
        "mu_wrong": round(probe.mu_w, 3),
    })
    OUT_FILE.write_text(json.dumps(results, indent=2))
    print(f"\n  Probe AUROC: {probe_auroc_mean:.4f} ± {probe_auroc_std:.4f}", flush=True)
    print(f"  Direction stability: {dir_stability:.4f}", flush=True)

    # ── Trajectories ─────────────────────────────────────────────────────────
    print(f"\n--- Trajectories (n={N_MAIN}, MAX_GEN_TOK={MAX_GEN_TOK}) ---", flush=True)
    traces = []
    shuffled = list(data)
    rng.shuffle(shuffled)
    t1 = time.time()

    for i, s in enumerate(shuffled[:N_MAIN]):
        if i % 5 == 0:
            n_think = sum(t.think_completed for t in traces)
            print(f"  [{i}/{N_MAIN}]  think_completed={n_think}  elapsed={time.time()-t1:.0f}s",
                  flush=True)
        tr = run_trajectory(model, tok, think_end_id, probe, s)
        traces.append(tr)
        if i % 5 == 0:
            OUT_FILE.write_text(json.dumps(results | {"traces_done": i+1}, indent=2))

    # ── Compute answer_jump ───────────────────────────────────────────────────
    cot_means = [float(np.mean(t.cot_trajectory)) for t in traces if t.cot_trajectory]
    commit_flags = [1.0 if m > COMMIT_THRESH else 0.0 for m in cot_means]
    commit_rate = float(np.mean(commit_flags)) if commit_flags else 0.0
    se = float(np.std(commit_flags)) / (len(commit_flags)**0.5 + 1e-9) if commit_flags else 1e-9
    z_score = commit_rate / (se + 1e-9) if se > 1e-9 else 0.0

    answer_jumps = []
    for t in traces:
        if t.think_completed and t.answer_trajectory and t.cot_trajectory:
            cot_mean = float(np.mean(t.cot_trajectory))
            ans_mean = float(np.mean(t.answer_trajectory[:10]))
            answer_jumps.append(ans_mean - cot_mean)

    exploration_intensity = float(np.mean([float(np.var(t.cot_trajectory))
                                           for t in traces if t.cot_trajectory]))

    n_think_completed = sum(1 for t in traces if t.think_completed)
    answer_jump = float(np.mean(answer_jumps)) if answer_jumps else None
    answer_jump_std = float(np.std(answer_jumps)) if len(answer_jumps) > 1 else None

    print(f"\n--- RESULTS ---", flush=True)
    print(f"  commit_rate={commit_rate:.3f}  z_score={z_score:.2f}", flush=True)
    print(f"  exploration_intensity={exploration_intensity:.4f}", flush=True)
    print(f"  n_think_completed={n_think_completed}/{N_MAIN}", flush=True)
    print(f"  n_answer_jumps={len(answer_jumps)}", flush=True)
    if answer_jump is not None:
        print(f"  answer_jump={answer_jump:.4f} ± {answer_jump_std or 0:.4f}", flush=True)

    # Verdict
    n_j = len(answer_jumps)
    if n_j < 5:
        verdict = "BUDGET_INSUFFICIENT"
        reason = f"n_answer_jumps={n_j} < 5 — increase MAX_GEN_TOK"
    elif answer_jump is not None and answer_jump > 1.0:
        verdict = "REGIME_2_CONFIRMED"
        reason = f"answer_jump={answer_jump:.3f} > 1.0 — deferred commitment replicates on Llama backbone"
    elif answer_jump is not None and answer_jump <= 0.0:
        verdict = "REGIME_2_NULL"
        reason = f"answer_jump={answer_jump:.3f} ≤ 0 — failure to replicate"
    elif answer_jump is not None:
        verdict = "REGIME_2_PARTIAL"
        reason = f"answer_jump={answer_jump:.3f} in (0, 1.0) — direction correct, below threshold"
    else:
        verdict = "INCONCLUSIVE"
        reason = "answer_jump could not be computed"

    print(f"\n*** VERDICT: {verdict} ***", flush=True)
    print(f"    {reason}", flush=True)

    results.update({
        "status": "complete",
        "elapsed_s": round(time.time()-t0),
        "commit_rate": round(commit_rate, 4),
        "z_score": round(z_score, 2),
        "exploration_intensity": round(exploration_intensity, 4),
        "n_think_completed": n_think_completed,
        "n_answer_jumps": n_j,
        "answer_jump": round(answer_jump, 4) if answer_jump is not None else None,
        "answer_jump_std": round(answer_jump_std, 4) if answer_jump_std is not None else None,
        "verdict": verdict,
        "reason": reason,
        "v6_comparison": {
            "v6_model": "DeepSeek-R1-Distill-Qwen-7B",
            "v6_answer_jump": 5.54,
            "v2_answer_jump": round(answer_jump, 4) if answer_jump is not None else None,
            "replicated": verdict == "REGIME_2_CONFIRMED",
        }
    })
    OUT_FILE.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {OUT_FILE}", flush=True)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
