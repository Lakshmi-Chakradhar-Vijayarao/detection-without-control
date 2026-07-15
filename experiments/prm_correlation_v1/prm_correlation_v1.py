"""
kaggle/prm_correlation_v1/prm_correlation_v1.py

PROCESS REWARD CORRELATION — J_KNOW VS PRM SCORES
===================================================

Question: does J_know at reasoning step k correlate with process reward at step k?

If yes: hidden-state geometry is an unsupervised approximation of process reward.
This means you can train PRMs without human annotation — the geometry provides
the signal for free. Every lab training reasoning models needs this.

Design:
  1. Generate reasoning traces on MATH problems with DeepSeek-R1-Distill-Qwen-1.5B
  2. At each reasoning "step" (punctuation boundary), record mean J_know
  3. Score each step with a lightweight PRM proxy: continuation quality signal
     - PRM proxy: after step S, truncate and continue — does model reach correct answer?
     - This is "oracle process reward" — expensive but label-free and correct
  4. Compute corr(J_know_step_k, PRM_proxy_step_k) across steps and problems
  5. Also compute: does early J_know predict final answer correctness (simpler signal)

Note on PRM proxy:
  True PRM labeling requires human annotation per step.
  Oracle PRM = run model forward from each step checkpoint to see if it reaches correct answer.
  This is N×M forward passes (N steps × M continuation samples).
  We use M=1 greedy to keep T4 runtime feasible.

Model:  deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
Dataset: MATH (competition math, well-known labels)

Output:
  prm_correlation_v1_results.json
  prm_correlation_v1_figure.png
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_ID       = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
CAL_N          = 60
MAIN_N         = 80      # MATH problems (oracle PRM is expensive)
MAX_THINK_TOK  = 1200
MAX_ANS_TOK    = 150
CAL_MAX_TOK    = 500
STEP_MIN_TOKS  = 20      # minimum tokens per reasoning "step"
N_BOOTSTRAP    = 300
SEED           = 42

rng = np.random.default_rng(SEED)

OUT_DIR      = Path("/kaggle/working")
RESULTS_FILE = OUT_DIR / "prm_correlation_v1_results.json"
FIGURE_FILE  = OUT_DIR / "prm_correlation_v1_figure.png"

assert torch.cuda.is_available(), "T4 GPU required"
_sm_major, _sm_minor = torch.cuda.get_device_capability(0)
_sm = _sm_major * 10 + _sm_minor
assert _sm >= 70, f"GPU sm_{_sm} not supported — need T4 (sm_75) or better. Re-run on T4."
DEVICE = "cuda"
print(f"GPU: {torch.cuda.get_device_name(0)}  (sm_{_sm})", flush=True)
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data — MATH dataset
# ─────────────────────────────────────────────────────────────────────────────

def load_math(n: int = 200) -> List[Dict]:
    """Load competition math problems with verified answers."""
    from datasets import load_dataset
    ds = None
    for dataset_id, kwargs in [
        ("HuggingFaceH4/MATH-500",   {"split": "test",  "streaming": False}),
        ("EleutherAI/hendrycks_math", {"name": "algebra", "split": "test", "streaming": True}),
        ("lighteval/MATH-Hard",       {"split": "test",  "streaming": True}),
    ]:
        try:
            ds = load_dataset(dataset_id, **kwargs)
            print(f"Loaded dataset: {dataset_id}", flush=True)
            break
        except Exception as e:
            print(f"Dataset {dataset_id} failed: {e}", flush=True)
    if ds is None:
        raise RuntimeError("All MATH dataset sources failed — no data to run on.")

    out = []
    for row in ds:
        problem  = row.get("problem", row.get("question", ""))
        solution = row.get("solution", row.get("answer", ""))
        if not problem or not solution:
            continue
        # Extract final answer from solution box: \boxed{...}
        m = re.search(r"\\boxed\{([^}]+)\}", solution)
        answer = m.group(1).strip() if m else solution.strip()[-50:]
        out.append({"question": problem, "answer": answer})
        if len(out) >= n:
            break
    print(f"Loaded {len(out)} MATH problems", flush=True)
    return out


def math_correct(pred: str, gold: str) -> bool:
    """Check if prediction contains the correct answer."""
    pred = pred.lower().strip()
    gold = gold.lower().strip()
    return gold in pred or pred.endswith(gold) or gold in pred[-100:]


# ─────────────────────────────────────────────────────────────────────────────
# Model + hidden-state capture
# ─────────────────────────────────────────────────────────────────────────────

def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {MODEL_ID} …")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, trust_remote_code=True,
    ).to(DEVICE).eval()
    print(f"  Loaded. Layers: {mdl.config.num_hidden_layers}")
    return mdl, tok


def find_think_end_id(tok) -> Optional[int]:
    for s in ["</think>", "<|/think|>", "</thinking>"]:
        ids = tok.encode(s, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
    return None


class HSCapture:
    def __init__(self, model, layer_idx: int):
        self.hs: Optional[np.ndarray] = None
        self._h = model.model.layers[layer_idx].register_forward_hook(self._fn)

    def _fn(self, mod, inp, out):
        x = out[0] if isinstance(out, tuple) else out
        self.hs = x[:, -1, :].detach().float().cpu().numpy()

    def remove(self):
        self._h.remove()


# ─────────────────────────────────────────────────────────────────────────────
# Fisher LDA calibration (correct vs wrong on MATH)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Probe:
    direction:  np.ndarray
    mu_correct: float
    mu_wrong:   float
    layer_idx:  int
    auroc:      float

    def score(self, h: np.ndarray) -> float:
        p      = float(np.dot(h, self.direction))
        scale  = (self.mu_correct - self.mu_wrong) / 2.0
        center = (self.mu_correct + self.mu_wrong) / 2.0
        return (p - center) / (abs(scale) + 1e-9)


def calibrate(model, tok, samples: List[Dict], layer: int, n_target: int) -> Probe:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import roc_auc_score

    cor_hs, wrg_hs = [], []
    n_half = n_target // 2

    for s in samples:
        if len(cor_hs) >= n_half and len(wrg_hs) >= n_half:
            break
        msgs   = [{"role": "user", "content": f"Solve: {s['question']}"}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids    = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
        cap    = HSCapture(model, layer)
        with torch.no_grad():
            model(ids)
        hs = cap.hs[0].copy() if cap.hs is not None else None
        cap.remove()
        if hs is None:
            continue
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=CAL_MAX_TOK,
                                  do_sample=False, pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        ans = gen.split("</think>", 1)[1].strip() if "</think>" in gen else gen
        if math_correct(ans, s["answer"]) and len(cor_hs) < n_half:
            cor_hs.append(hs)
        elif not math_correct(ans, s["answer"]) and len(wrg_hs) < n_half:
            wrg_hs.append(hs)

    if len(cor_hs) < 4 or len(wrg_hs) < 4:
        raise RuntimeError(f"Too sparse: correct={len(cor_hs)}, wrong={len(wrg_hs)}")

    X = np.stack(cor_hs + wrg_hs)
    y = np.array([1] * len(cor_hs) + [0] * len(wrg_hs))
    lda = LinearDiscriminantAnalysis(n_components=1)
    lda.fit(X, y)
    d     = lda.coef_[0] / (np.linalg.norm(lda.coef_[0]) + 1e-9)
    projs = X @ d
    mu_c  = float(np.mean(projs[y == 1]))
    mu_w  = float(np.mean(projs[y == 0]))
    auroc = float(roc_auc_score(y, lda.decision_function(X)))
    print(f"  Layer {layer}: AUROC={auroc:.4f}  correct={len(cor_hs)}, wrong={len(wrg_hs)}")
    return Probe(direction=d, mu_correct=mu_c, mu_wrong=mu_w, layer_idx=layer, auroc=auroc)


def pick_layer(model, tok, samples, n_layers) -> Probe:
    candidates = [n_layers - 4, n_layers - 3, n_layers - 2, n_layers - 1]
    candidates = [l for l in candidates if 0 <= l < n_layers]
    best = None
    for l in candidates:
        try:
            p = calibrate(model, tok, samples[:80], l, n_target=CAL_N)
            if best is None or p.auroc > best.auroc:
                best = p
        except Exception as e:
            print(f"  Layer {l}: {e}")
    if best is None:
        raise RuntimeError("All layers failed")
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Step segmentation: split think block at sentence/step boundaries
# ─────────────────────────────────────────────────────────────────────────────

def segment_trajectory(trajectory: List[float], token_texts: List[str],
                        min_tokens: int = STEP_MIN_TOKS) -> List[Dict]:
    """
    Split trajectory into reasoning steps at sentence boundaries.
    Returns list of step dicts with mean J_know and step index.
    """
    # Find boundaries: tokens containing '.', '\n', '?', '!'
    boundaries = [0]
    for i, t in enumerate(token_texts):
        if any(c in t for c in (".\n", "\n\n", ". ", "! ", "? ")):
            if i - boundaries[-1] >= min_tokens:
                boundaries.append(i)
    if boundaries[-1] < len(trajectory) - 1:
        boundaries.append(len(trajectory))

    steps = []
    for j in range(len(boundaries) - 1):
        s, e = boundaries[j], boundaries[j + 1]
        step_traj = trajectory[s:e]
        if len(step_traj) < 3:
            continue
        steps.append({
            "step_idx":     j,
            "start_tok":    s,
            "end_tok":      e,
            "length":       e - s,
            "mean_j_know":  float(np.mean(step_traj)),
            "max_j_know":   float(np.max(step_traj)),
            "min_j_know":   float(np.min(step_traj)),
            "j_velocity":   float(np.mean(step_traj[-5:])) - float(np.mean(step_traj[:5])),
        })
    return steps


# ─────────────────────────────────────────────────────────────────────────────
# Main trace: generate + track J_know per token + oracle PRM per step
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MathTrace:
    question:     str
    answer_gold:  str
    think_traj:   List[float] = field(default_factory=list)
    tok_texts:    List[str]   = field(default_factory=list)
    answer_full:  str         = ""
    correct_full: bool        = False
    steps:        List[Dict]  = field(default_factory=list)
    # Oracle PRM: after each step, did the model reach correct answer?
    step_oracle:  List[bool]  = field(default_factory=list)
    # Pct of think tokens at each step
    step_pct:     List[float] = field(default_factory=list)


def run_trace(model, tok, probe: Probe, sample: Dict,
              think_end_id: Optional[int]) -> MathTrace:
    tr  = MathTrace(question=sample["question"], answer_gold=sample["answer"])
    cap = HSCapture(model, probe.layer_idx)

    try:
        msgs       = [{"role": "user", "content": f"Solve step by step: {sample['question']}"}]
        prompt     = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        prompt_ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
        cur        = prompt_ids.clone()
        think_ids: List[int] = []
        past_kv    = None

        # KV-cached loop: O(n) per step. Without cache: 1200 tokens × 80 Qs ≈ 7h.
        for _ in range(MAX_THINK_TOK):
            with torch.no_grad():
                if past_kv is None:
                    out = model(cur, use_cache=True)
                else:
                    out = model(cur[:, -1:], past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            if cap.hs is not None:
                tr.think_traj.append(probe.score(cap.hs[0]))

            nxt = int(torch.argmax(out.logits[0, -1]).item())
            tr.tok_texts.append(tok.decode([nxt]))
            think_ids.append(nxt)
            cur = torch.cat([cur, torch.tensor([[nxt]], device=DEVICE)], dim=1)
            if nxt == tok.eos_token_id:
                break
            if think_end_id is not None and nxt == think_end_id:
                break

        # Full answer
        with torch.no_grad():
            ans_ids = model.generate(cur, max_new_tokens=MAX_ANS_TOK,
                                     do_sample=False, pad_token_id=tok.eos_token_id)
        tr.answer_full  = tok.decode(ans_ids[0][cur.shape[1]:], skip_special_tokens=True).strip()
        tr.correct_full = math_correct(tr.answer_full, sample["answer"])

        # Segment think block into steps
        tr.steps = segment_trajectory(tr.think_traj, tr.tok_texts)

        # Oracle PRM: for each step boundary, truncate and continue, check correctness
        n_think = len(think_ids)
        for step in tr.steps:
            e_tok      = step["end_tok"]
            pct        = 100.0 * e_tok / max(n_think, 1)
            tr.step_pct.append(pct)
            trunc_ids  = think_ids[:e_tok]
            t_t        = torch.tensor([trunc_ids], device=DEVICE)
            end_t      = (torch.tensor([[think_end_id]], device=DEVICE)
                          if think_end_id is not None
                          else torch.zeros((1, 0), dtype=torch.long, device=DEVICE))
            cont_in    = torch.cat([prompt_ids, t_t, end_t], dim=1)
            with torch.no_grad():
                cont_out = model.generate(cont_in, max_new_tokens=MAX_ANS_TOK,
                                          do_sample=False, pad_token_id=tok.eos_token_id)
            cont_text = tok.decode(cont_out[0][cont_in.shape[1]:], skip_special_tokens=True)
            tr.step_oracle.append(math_correct(cont_text, sample["answer"]))

    finally:
        cap.remove()
    return tr


# ─────────────────────────────────────────────────────────────────────────────
# Statistics: compute corr(J_know, oracle_PRM) across steps
# ─────────────────────────────────────────────────────────────────────────────

def statistics(traces: List[MathTrace], probe: Probe) -> Dict:
    # Collect all (step_mean_j_know, oracle_label) pairs
    all_j, all_prm = [], []
    for tr in traces:
        for i, step in enumerate(tr.steps):
            if i < len(tr.step_oracle):
                all_j.append(step["mean_j_know"])
                all_prm.append(1.0 if tr.step_oracle[i] else 0.0)

    # Correlation
    if len(all_j) >= 10:
        from scipy.stats import pearsonr, spearmanr
        r_p, p_p = pearsonr(all_j, all_prm)
        r_s, p_s = spearmanr(all_j, all_prm)
    else:
        r_p = r_s = p_p = p_s = float("nan")

    # Simpler: early J_know vs final correctness
    early_j  = [np.mean(tr.think_traj[:50]) if len(tr.think_traj) >= 50 else np.mean(tr.think_traj)
                for tr in traces]
    final_ok = [1.0 if tr.correct_full else 0.0 for tr in traces]
    if len(early_j) >= 10:
        from scipy.stats import pearsonr
        r_early, p_early = pearsonr(early_j, final_ok)
    else:
        r_early = p_early = float("nan")

    correct_rate = float(np.mean(final_ok)) if final_ok else 0.0

    # Per-step trajectory statistics: mean J_know aggregated per step index across problems
    max_steps = max((len(tr.steps) for tr in traces if tr.steps), default=0)
    step_j_by_idx: Dict[int, List[float]] = {}
    for tr in traces:
        for step in tr.steps:
            idx = step["step_idx"]
            step_j_by_idx.setdefault(idx, []).append(step["mean_j_know"])
    mean_j_per_step = {
        idx: float(np.mean(vals))
        for idx, vals in sorted(step_j_by_idx.items())
    }

    # corr(J_prmproxy) per step: mean J for steps where oracle=1 vs oracle=0
    j_prm1 = [j for j, p in zip(all_j, all_prm) if p == 1.0]
    j_prm0 = [j for j, p in zip(all_j, all_prm) if p == 0.0]

    trajectory_stats = {
        "mean_j_know_per_step":    mean_j_per_step,
        "n_steps_observed":        len(all_j),
        "mean_j_know_prm1":        float(np.mean(j_prm1)) if j_prm1 else None,
        "mean_j_know_prm0":        float(np.mean(j_prm0)) if j_prm0 else None,
        "j_know_prm_gap":          float(np.mean(j_prm1) - np.mean(j_prm0))
                                   if j_prm1 and j_prm0 else None,
        "corr_j_prmproxy_pearson": float(r_p) if not np.isnan(r_p) else None,
        "corr_j_prmproxy_spearman": float(r_s) if not np.isnan(r_s) else None,
    }

    return {
        "model":              MODEL_ID,
        "probe_layer":        probe.layer_idx,
        "cal_auroc":          probe.auroc,
        "n_problems":         len(traces),
        "n_step_pairs":       len(all_j),
        "correct_rate":       correct_rate,
        "prm_corr_pearson":   float(r_p) if not np.isnan(r_p) else None,
        "prm_corr_spearman":  float(r_s) if not np.isnan(r_s) else None,
        "prm_corr_p_pearson": float(p_p) if not np.isnan(p_p) else None,
        "early_j_corr":       float(r_early) if not np.isnan(r_early) else None,
        "early_j_p":          float(p_early) if not np.isnan(p_early) else None,
        "trajectory_stats":   trajectory_stats,
        "verdict": (
            "PRM_SIGNAL_STRONG"   if not np.isnan(r_s) and abs(r_s) > 0.5 else
            "PRM_SIGNAL_MODERATE" if not np.isnan(r_s) and abs(r_s) > 0.3 else
            "PRM_SIGNAL_WEAK"
        ),
        "interpretation": (
            f"corr(J_know_step, oracle_PRM) = Pearson {r_p:.3f}, Spearman {r_s:.3f}. "
            f"Early J_know vs final correctness: r={r_early:.3f} (p={p_early:.4f}). "
            f"{'Hidden geometry approximates PRM signal without labels.' if abs(r_s) > 0.4 else 'Weak PRM correlation — step granularity may be too coarse.'}"
        ),
    }


def make_figure(traces: List[MathTrace], stats: Dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(2, 3, hspace=0.4, wspace=0.35)

    # Panel 1: J_know trajectory for correct vs wrong problems
    ax1 = fig.add_subplot(gs[0, 0])
    correct_t = [t for t in traces if t.correct_full and t.think_traj]
    wrong_t   = [t for t in traces if not t.correct_full and t.think_traj]
    N_BINS    = 40
    xs_pct    = np.linspace(0, 100, N_BINS)

    def bin_traj(tr_list):
        bins = [[] for _ in range(N_BINS)]
        for tr in tr_list:
            traj = tr.think_traj
            for i, v in enumerate(traj):
                b = min(int(i / len(traj) * N_BINS), N_BINS - 1)
                bins[b].append(v)
        return [np.mean(b) if b else np.nan for b in bins]

    if correct_t:
        ax1.plot(xs_pct, bin_traj(correct_t), "#27ae60", lw=2.5,
                 label=f"Correct (n={len(correct_t)})")
    if wrong_t:
        ax1.plot(xs_pct, bin_traj(wrong_t), "#c0392b", lw=2.5,
                 label=f"Wrong (n={len(wrong_t)})")
    ax1.axhline(0, color="gray", ls="--", lw=0.8)
    ax1.set_title("J_know through MATH reasoning trace", fontsize=10)
    ax1.set_xlabel("% through think block", fontsize=10)
    ax1.set_ylabel("J_know", fontsize=10)
    ax1.legend(fontsize=8)

    # Panel 2: J_know per step vs oracle PRM label
    ax2 = fig.add_subplot(gs[0, 1])
    j_vals, prm_vals = [], []
    for tr in traces:
        for i, step in enumerate(tr.steps):
            if i < len(tr.step_oracle):
                j_vals.append(step["mean_j_know"])
                prm_vals.append(1.0 if tr.step_oracle[i] else 0.0)
    if j_vals:
        cols = ["#2ecc71" if p == 1 else "#e74c3c" for p in prm_vals]
        ax2.scatter(j_vals, prm_vals, c=cols, alpha=0.4, s=20)
        if stats["prm_corr_pearson"]:
            ax2.set_title(
                f"J_know per step vs oracle PRM\nr={stats['prm_corr_pearson']:.3f}  "
                f"ρ={stats['prm_corr_spearman']:.3f}", fontsize=10
            )
        ax2.set_xlabel("Mean J_know at step", fontsize=10)
        ax2.set_ylabel("Oracle PRM (1=correct continuation)", fontsize=10)

    # Panel 3: Early J_know vs final correctness
    ax3 = fig.add_subplot(gs[0, 2])
    early_j  = [np.mean(t.think_traj[:50]) if len(t.think_traj) >= 50 else np.mean(t.think_traj)
                for t in traces if t.think_traj]
    final_ok = [1 if t.correct_full else 0 for t in traces if t.think_traj]
    if early_j:
        cols3 = ["#2ecc71" if f else "#e74c3c" for f in final_ok]
        ax3.scatter(early_j, final_ok, c=cols3, alpha=0.5, s=30)
        ax3.set_title(
            f"Early J_know (first 50 toks) vs final correctness\nr={stats['early_j_corr']:.3f}",
            fontsize=10
        )
        ax3.set_xlabel("Mean J_know (first 50 tokens)", fontsize=10)
        ax3.set_ylabel("Final answer correct", fontsize=10)

    # Panel 4: Step-level J_know distribution by oracle label
    ax4 = fig.add_subplot(gs[1, 0])
    j_corr = [j for j, p in zip(j_vals, prm_vals) if p == 1]
    j_wrng = [j for j, p in zip(j_vals, prm_vals) if p == 0]
    if j_corr:
        ax4.hist(j_corr, bins=20, alpha=0.6, color="#2ecc71",
                 label=f"PRM=1 n={len(j_corr)}", density=True)
    if j_wrng:
        ax4.hist(j_wrng, bins=20, alpha=0.6, color="#e74c3c",
                 label=f"PRM=0 n={len(j_wrng)}", density=True)
    ax4.set_title("J_know by oracle PRM label", fontsize=10)
    ax4.legend(fontsize=8)

    # Panel 5: Step position (%) vs oracle correctness
    ax5 = fig.add_subplot(gs[1, 1])
    pct_vals = []
    for tr in traces:
        for i, pct in enumerate(tr.step_pct):
            if i < len(tr.step_oracle):
                pct_vals.append((pct, 1 if tr.step_oracle[i] else 0))
    if pct_vals:
        xs, ys = zip(*pct_vals)
        ax5.scatter(xs, ys, alpha=0.4, s=15,
                    c=["#2ecc71" if y else "#e74c3c" for y in ys])
        ax5.set_title("Step position (%) vs oracle correctness", fontsize=10)
        ax5.set_xlabel("% through think block", fontsize=10)
        ax5.set_ylabel("Oracle PRM label", fontsize=10)

    # Panel 6: Summary
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    txt = (
        f"PRM CORRELATION v1\n"
        f"{'='*36}\n\n"
        f"Model:      {MODEL_ID}\n"
        f"N problems: {stats['n_problems']}\n"
        f"N steps:    {stats['n_step_pairs']}\n"
        f"Correct:    {stats['correct_rate']*100:.1f}%\n\n"
        f"PROBE\n"
        f"  Layer:  {stats['probe_layer']}\n"
        f"  AUROC:  {stats['cal_auroc']:.4f}\n\n"
        f"PRM CORRELATION\n"
        f"  Pearson:   {stats['prm_corr_pearson']:.3f}\n"
        f"  Spearman:  {stats['prm_corr_spearman']:.3f}\n\n"
        f"EARLY J_KNOW VS FINAL\n"
        f"  r = {stats['early_j_corr']:.3f}  p={stats['early_j_p']:.4f}\n\n"
        f"VERDICT: {stats['verdict']}\n"
    )
    ax6.text(0.04, 0.96, txt, transform=ax6.transAxes, fontsize=8.5,
             va="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="#f0f4f8", alpha=0.9))

    plt.suptitle(
        f"J_know vs Process Reward — DeepSeek-R1-Distill-Qwen-1.5B on MATH\n"
        f"Pearson r={stats['prm_corr_pearson']:.3f}  Spearman ρ={stats['prm_corr_spearman']:.3f}",
        fontsize=11, fontweight="bold",
    )
    fig.savefig(str(FIGURE_FILE), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure → {FIGURE_FILE}")


def main():
    t0 = time.time()
    all_s = load_math(n=200)
    rng.shuffle(all_s)
    cal_s  = all_s[:80]
    main_s = all_s[80: 80 + MAIN_N]

    model, tok = load_model()
    n_layers   = model.config.num_hidden_layers
    think_end  = find_think_end_id(tok)

    probe = pick_layer(model, tok, cal_s, n_layers)

    print(f"\nMain: {MAIN_N} problems (oracle PRM per step = slow) …")
    traces = []
    for i, s in enumerate(main_s):
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{MAIN_N}  ({elapsed:.0f}s)")
        try:
            traces.append(run_trace(model, tok, probe, s, think_end))
        except Exception as e:
            print(f"  Error {i}: {e}")

    stats = statistics(traces, probe)
    print(f"\nVERDICT: {stats['verdict']}")
    print(f"INTERP:  {stats['interpretation']}")

    make_figure(traces, stats)
    with open(RESULTS_FILE, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Results → {RESULTS_FILE}   Total: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
