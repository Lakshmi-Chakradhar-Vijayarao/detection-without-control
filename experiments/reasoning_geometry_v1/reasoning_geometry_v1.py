"""
kaggle/reasoning_geometry_v1/reasoning_geometry_v1.py

EPISTEMIC COMMITMENT MOMENT IN REASONING MODELS
================================================

Central question: When does a reasoning model's hidden state commit to its
answer INSIDE the <think>...</think> block — before the answer token appears?

The one number this experiment produces:
  "X% of think-block tokens are post-commitment elaboration"

If X > 50% with z > 3 vs shuffled null → the model already decided long before
it stopped thinking. Every lab running o1-style inference is burning GPU budget
on post-decision narration they currently cannot detect.

Measurements:
  1. J_know trajectory through every think-block token
  2. Commitment moment: first stable basin in J_know (window + persistence)
  3. Commitment percentage: % of think tokens AFTER commitment
  4. Null hypothesis: shuffle trajectories, z-score observed vs null
  5. Early exit test: truncate at commit step, F1 delta vs full generation
  6. Correct vs wrong trajectory separation (commitment gap)

Model:  deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
  — Open weights, T4 compatible at float16 (~3 GB)
  — Native <think>...</think> reasoning architecture
  — Qwen2.5 backbone, architecturally close to Alibaba/Google Qwen family

Dataset: TriviaQA rc.wikipedia (streaming, no download)

Output:
  reasoning_geometry_v1_results.json  — all statistics + verdict
  reasoning_geometry_v1_figure.png    — 6-panel figure
"""

from __future__ import annotations


import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_ID       = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
CAL_N          = 80      # calibration samples (target: 40 correct, 40 wrong)
MAIN_N         = 200     # main experiment questions
MAX_THINK_TOK  = 1500    # max think-block tokens per question
MAX_ANS_TOK    = 150     # max answer tokens after </think>
CAL_MAX_TOK    = 8192    # max think tokens in calibration — eos stops it naturally
COMMIT_WINDOW  = 10      # rolling window for commitment detection
COMMIT_THRESH  = 0.8     # J_know normalised threshold
COMMIT_PERSIST = 15      # must hold for this many steps
N_BOOTSTRAP    = 500     # shuffles for null hypothesis
SEED           = 42

rng = np.random.default_rng(SEED)

OUT_DIR      = Path("/kaggle/working")
RESULTS_FILE = OUT_DIR / "reasoning_geometry_v1_results.json"
FIGURE_FILE  = OUT_DIR / "reasoning_geometry_v1_figure.png"

# ── Device ────────────────────────────────────────────────────────────────────
assert torch.cuda.is_available(), "GPU required"
DEVICE = "cuda"
_gpu_name = torch.cuda.get_device_name(0)
_sm_major, _sm_minor = torch.cuda.get_device_capability(0)
_sm = _sm_major * 10 + _sm_minor
print(f"GPU: {_gpu_name}  (sm_{_sm})")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
# P100 = sm_60: current PyTorch (CUDA 12.x) has no kernel image for it.
# Fail fast so Kaggle re-queues rather than wasting 6 min loading the model.
assert _sm >= 70, f"GPU sm_{_sm} not supported — need T4 (sm_75) or better. Re-run to get a different GPU."


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def load_trivia_qa(n: int = 450) -> List[Dict]:
    from datasets import load_dataset
    ds = load_dataset("trivia_qa", "rc.wikipedia", split="train", streaming=True)
    out = []
    for row in ds:
        if not row["answer"]["aliases"]:
            continue
        # Grab first Wikipedia passage as context for bilateral oracle calibration
        pages = row.get("entity_pages", {})
        wiki  = pages.get("wiki_context", []) if pages else []
        ctx   = wiki[0][:1200] if wiki else ""
        out.append({
            "question": row["question"],
            "answers":  row["answer"]["aliases"],
            "context":  ctx,
        })
        if len(out) >= n:
            break
    print(f"Loaded {len(out)} TriviaQA samples")
    return out


def token_f1(pred: str, golds: List[str]) -> float:
    p = set(pred.lower().split())
    best = 0.0
    for g in golds:
        q = set(g.lower().split())
        if not p or not q:
            continue
        c = p & q
        if not c:
            continue
        prec = len(c) / len(p)
        rec  = len(c) / len(q)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


def answer_contains(pred: str, golds: List[str]) -> bool:
    """Check if any gold alias appears as a substring of pred (case-insensitive).
    More robust than token F1 for verbose reasoning-model answers where the
    correct fact is present but diluted by surrounding explanation text."""
    pred_l = pred.lower()
    for g in golds:
        g_l = g.lower().strip()
        if g_l and g_l in pred_l:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Model
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
    for word, idx in tok.get_vocab().items():
        if "</think>" in word:
            return tok.encode(word, add_special_tokens=False)[0]
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Hidden-state capture
# ─────────────────────────────────────────────────────────────────────────────

class HSCapture:
    def __init__(self, model, layer_idx: int):
        self.hs: Optional[np.ndarray] = None
        self._h = model.model.layers[layer_idx].register_forward_hook(self._fn)

    def _fn(self, mod, inp, out):
        x = out[0] if isinstance(out, tuple) else out
        self.hs = x[:, -1, :].detach().cpu().float().numpy()

    def remove(self):
        self._h.remove()


# ─────────────────────────────────────────────────────────────────────────────
# Calibration — Fisher LDA (correct vs wrong)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Probe:
    direction:  np.ndarray
    mu_correct: float
    mu_wrong:   float
    layer_idx:  int
    auroc:      float

    def score(self, h: np.ndarray) -> float:
        """Signed J_know: positive = leaning correct, negative = leaning wrong."""
        p      = float(np.dot(h, self.direction))
        scale  = (self.mu_correct - self.mu_wrong) / 2.0
        center = (self.mu_correct + self.mu_wrong) / 2.0
        return (p - center) / (abs(scale) + 1e-9)


def calibrate(model, tok, samples: List[Dict], layer: int,
              n_target: int = CAL_N) -> Probe:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import roc_auc_score

    print(f"  Calibrating at layer {layer} …")
    cor_hs, wrg_hs = [], []
    n_half = n_target // 2

    # Bilateral oracle using full reasoning generation.
    # Force-skipping think doesn't work reliably at 1.5B — the model ignores it.
    # Let it reason fully (CAL_MAX_TOK=1200), extract answer from after </think>.
    # PARAM  : F1 >= 0.4 without context  (model knows it from parametric memory)
    # CTX_DEP: F1 < 0.05 without context  (model fails from memory alone)
    # No context verification needed for CTX_DEP — failure is sufficient signal.
    # n_target=20 (10 per class) keeps calibration to ~50 full-generation samples.
    param_hs, ctxdep_hs = [], []
    n_half      = n_target // 2
    max_scanned = 200  # hard cap to avoid infinite loop

    for idx, s in enumerate(samples):
        if idx >= max_scanned:
            break
        if len(param_hs) >= n_half and len(ctxdep_hs) >= n_half:
            break

        if idx % 5 == 0:
            print(f"    cal [{idx}/200] PARAM={len(param_hs)}/{n_half}  CTX_DEP={len(ctxdep_hs)}/{n_half}",
                  flush=True)

        q       = s["question"]
        answers = s["answers"]

        msgs  = [{"role": "user", "content": q}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids   = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)

        # Capture prefill hidden state (before any reasoning tokens)
        cap = HSCapture(model, layer)
        with torch.no_grad():
            model(ids)
        hs = cap.hs[0].copy() if cap.hs is not None else None
        cap.remove()
        if hs is None:
            continue

        # Two-phase oracle generation for reasoning models.
        # Problem: fixed token budgets cut off reasoning mid-chain (think blocks
        # can exceed 2000+ tokens for distilled R1 models).
        # Fix: phase 1 stops exactly at </think> by treating it as eos_token_id.
        # Phase 2 generates the answer (150 tokens, always completes).
        think_close_id = 151649  # </think> — confirmed from logs

        # Phase 1: generate think block, stop when </think> is emitted
        with torch.no_grad():
            out_think = model.generate(
                ids, max_new_tokens=CAL_MAX_TOK, do_sample=False,
                pad_token_id=tok.eos_token_id,
                eos_token_id=[tok.eos_token_id, think_close_id],
            )
        n_think = out_think.shape[1] - ids.shape[1]
        think_closed = n_think < CAL_MAX_TOK  # stopped early = </think> found

        if think_closed:
            # Reattach </think> token (generate strips eos from output)
            close_t    = torch.tensor([[think_close_id]], device=DEVICE)
            full_ids   = torch.cat([out_think, close_t], dim=1)
            # Phase 2: generate answer (short, always fits)
            with torch.no_grad():
                out_ans = model.generate(
                    full_ids, max_new_tokens=150, do_sample=False,
                    pad_token_id=tok.eos_token_id,
                )
            ans = tok.decode(
                out_ans[0][full_ids.shape[1]:], skip_special_tokens=True
            ).strip()
        else:
            ans = ""  # think block never closed → model couldn't answer
        f1          = token_f1(ans, answers)
        has_answer  = answer_contains(ans, answers)
        if idx < 3:
            print(f"    DBG [{idx}] n_think={n_think} closed={think_closed} "
                  f"has_ans={has_answer} f1={f1:.3f} ans='{ans[:50]}'", flush=True)

        has_answer = answer_contains(ans, answers)
        if has_answer and len(param_hs) < n_half:
            param_hs.append(hs)
        elif not has_answer and f1 < 0.05 and len(ctxdep_hs) < n_half:
            ctxdep_hs.append(hs)

    cor_hs = param_hs
    wrg_hs = ctxdep_hs

    if len(cor_hs) < 5 or len(wrg_hs) < 5:
        raise RuntimeError(f"Calibration too sparse: correct={len(cor_hs)}, wrong={len(wrg_hs)}")

    X = np.stack(cor_hs + wrg_hs)
    y = np.array([1] * len(cor_hs) + [0] * len(wrg_hs))
    lda = LinearDiscriminantAnalysis(n_components=1)
    lda.fit(X, y)
    d      = lda.coef_[0] / (np.linalg.norm(lda.coef_[0]) + 1e-9)
    projs  = X @ d
    mu_c   = float(np.mean(projs[y == 1]))
    mu_w   = float(np.mean(projs[y == 0]))
    auroc  = float(roc_auc_score(y, lda.decision_function(X)))
    print(f"    Layer {layer}: AUROC={auroc:.4f}  mu_c={mu_c:.3f}  mu_w={mu_w:.3f}")
    return Probe(direction=d, mu_correct=mu_c, mu_wrong=mu_w,
                 layer_idx=layer, auroc=auroc)


def pick_layer(model, tok, samples, n_layers) -> Probe:
    # L26 = n_layers-2 is the empirically validated optimal layer across
    # Qwen/Llama families (from ESM layer sweep experiments). Try it first,
    # fall back to neighbours only if it fails.
    preferred = n_layers - 2
    candidates = [preferred, n_layers - 3, n_layers - 1, n_layers - 4]
    candidates = [l for l in candidates if 0 <= l < n_layers]
    for l in candidates:
        try:
            p = calibrate(model, tok, samples[:200], l, n_target=10)
            print(f"Best probe layer: {p.layer_idx} (AUROC={p.auroc:.4f})")
            return p
        except Exception as e:
            print(f"  Layer {l} failed: {e}")
    raise RuntimeError("All candidate layers failed calibration")


# ─────────────────────────────────────────────────────────────────────────────
# Commitment detection
# ─────────────────────────────────────────────────────────────────────────────

def find_commit(traj: List[float], window: int = COMMIT_WINDOW,
                thresh: float = COMMIT_THRESH, persist: int = COMMIT_PERSIST) -> Optional[int]:
    """
    First step where J_know enters a stable basin.
    Basin: rolling mean >= thresh AND stays >= thresh*0.8 for `persist` steps.
    """
    n = len(traj)
    t = np.array(traj)
    if n < window + persist:
        return None
    for i in range(n - window - persist + 1):
        if np.mean(t[i: i + window]) >= thresh:
            if np.min(t[i + window: i + window + persist]) >= thresh * 0.8:
                return i
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Reasoning trace generation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Trace:
    question:        str
    answers:         List[str]
    think_traj:      List[float] = field(default_factory=list)  # J_know per think token
    think_len:       int         = 0
    answer_full:     str         = ""
    answer_early:    str         = ""
    f1_full:         float       = 0.0
    f1_early:        float       = 0.0
    has_answer:      bool        = False
    commit_step:     Optional[int] = None
    commit_pct:      float         = 0.0   # % of think tokens POST-commit
    think_ended:     bool          = False


def run_trace(model, tok, probe: Probe, sample: Dict,
              think_end_id: Optional[int]) -> Trace:
    q, answers = sample["question"], sample["answers"]
    msgs   = [{"role": "user", "content": q}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    tr  = Trace(question=q, answers=answers)
    cap = HSCapture(model, probe.layer_idx)

    try:
        prompt_ids   = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
        current_ids  = prompt_ids.clone()
        think_tok_ids: List[int] = []

        for _ in range(MAX_THINK_TOK):
            with torch.no_grad():
                out = model(current_ids)
            if cap.hs is not None:
                tr.think_traj.append(probe.score(cap.hs[0]))

            next_tok = int(torch.argmax(out.logits[0, -1]).item())
            think_tok_ids.append(next_tok)
            current_ids = torch.cat(
                [current_ids, torch.tensor([[next_tok]], device=DEVICE)], dim=1
            )

            if next_tok == tok.eos_token_id:
                break
            if think_end_id is not None and next_tok == think_end_id:
                tr.think_ended = True
                break

        tr.think_len = len(think_tok_ids)

        # Generate answer
        with torch.no_grad():
            ans_ids = model.generate(current_ids, max_new_tokens=MAX_ANS_TOK,
                                     do_sample=False, pad_token_id=tok.eos_token_id)
        tr.answer_full = tok.decode(
            ans_ids[0][current_ids.shape[1]:], skip_special_tokens=True
        ).strip()
        tr.f1_full     = token_f1(tr.answer_full, answers)
        tr.has_answer  = answer_contains(tr.answer_full, answers)

        # Commitment moment
        commit = find_commit(tr.think_traj)
        tr.commit_step = commit
        if commit is not None and tr.think_len > 0:
            tr.commit_pct = 100.0 * (tr.think_len - commit) / tr.think_len

        # Simulate early exit
        if commit is not None and commit + COMMIT_WINDOW < tr.think_len - 5:
            exit_at     = commit + COMMIT_WINDOW
            trunc_ids   = think_tok_ids[:exit_at]
            think_t     = torch.tensor([trunc_ids], device=DEVICE)
            end_t       = (torch.tensor([[think_end_id]], device=DEVICE)
                           if think_end_id is not None else torch.zeros((1, 0), dtype=torch.long, device=DEVICE))
            early_input = torch.cat([prompt_ids, think_t, end_t], dim=1)
            with torch.no_grad():
                early_out = model.generate(early_input, max_new_tokens=MAX_ANS_TOK,
                                           do_sample=False, pad_token_id=tok.eos_token_id)
            tr.answer_early = tok.decode(
                early_out[0][early_input.shape[1]:], skip_special_tokens=True
            ).strip()
            tr.f1_early = token_f1(tr.answer_early, answers)
        else:
            tr.answer_early = tr.answer_full
            tr.f1_early     = tr.f1_full

    finally:
        cap.remove()

    return tr


# ─────────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────────

def statistics(traces: List[Trace]) -> Dict:
    correct = [t for t in traces if t.has_answer]
    wrong   = [t for t in traces if not t.has_answer and t.f1_full < 0.1]
    committed = [t for t in traces if t.commit_step is not None]

    all_cpct   = [t.commit_pct for t in committed]
    cor_cpct   = [t.commit_pct for t in correct if t.commit_step is not None]
    wrg_cpct   = [t.commit_pct for t in wrong   if t.commit_step is not None]

    obs_mean = float(np.mean(all_cpct)) if all_cpct else 0.0

    # Early exit
    exits     = [t for t in committed if t.answer_early != t.answer_full]
    f1_delta  = float(np.mean([t.f1_full - t.f1_early for t in exits])) if exits else 0.0
    tok_saved = float(np.mean([t.commit_pct for t in exits])) if exits else 0.0

    # Null hypothesis: shuffle trajectories, destroy temporal structure.
    # Each bootstrap sample shuffles ALL question trajectories and computes the
    # mean commit% across all questions — so null_std is the std of bootstrap MEANS,
    # not per-question variance. z = (obs_mean - null_mean) / null_std is then valid.
    all_trajs = [t.think_traj for t in traces if len(t.think_traj) > COMMIT_WINDOW + COMMIT_PERSIST]
    null_boot_means = []
    for _ in range(N_BOOTSTRAP):
        if not all_trajs:
            break
        boot_cpcts = []
        for traj in all_trajs:
            t = list(traj)
            rng.shuffle(t)
            c = find_commit(t)
            boot_cpcts.append(100.0 * (len(t) - c) / len(t) if c is not None else 0.0)
        null_boot_means.append(float(np.mean(boot_cpcts)))

    null_mean = float(np.mean(null_boot_means)) if null_boot_means else 0.0
    null_std  = float(np.std(null_boot_means))  if null_boot_means else 1.0
    z         = (obs_mean - null_mean) / max(null_std, 0.1)

    verdict = ("COMMITTED_EARLY" if z > 3 and obs_mean > 40 else
               "WEAK_SIGNAL"     if z > 2 and obs_mean > 20 else "NULL")

    return {
        "n_total":            len(traces),
        "n_correct":          len(correct),
        "n_wrong":            len(wrong),
        "commit_rate":        len(committed) / max(len(traces), 1),
        "mean_commit_pct":    obs_mean,
        "correct_commit_pct": float(np.mean(cor_cpct)) if cor_cpct else 0.0,
        "wrong_commit_pct":   float(np.mean(wrg_cpct)) if wrg_cpct else 0.0,
        "commit_gap":         (float(np.mean(cor_cpct)) - float(np.mean(wrg_cpct))
                               if cor_cpct and wrg_cpct else 0.0),
        "f1_delta":           f1_delta,
        "tokens_saved_pct":   tok_saved,
        "n_exits":            len(exits),
        "null_mean":          null_mean,
        "null_std":           null_std,
        "z_score":            z,
        "verdict":            verdict,
        "interpretation": (
            f"Mean {obs_mean:.1f}% of think tokens are post-commitment elaboration "
            f"(z={z:.2f} vs shuffled null). "
            f"Early exit F1 delta: {f1_delta:+.3f} ({tok_saved:.1f}% tokens saved)."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────────

def make_figure(traces: List[Trace], stats: Dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    correct = [t for t in traces if t.has_answer and t.think_traj]
    wrong   = [t for t in traces if not t.has_answer and t.f1_full < 0.1 and t.think_traj]

    fig = plt.figure(figsize=(20, 12))
    gs  = gridspec.GridSpec(2, 3, hspace=0.42, wspace=0.35)

    # ── Panel 1: Trajectories aligned to % of think block ─────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    N_BINS = 50
    xs_pct = np.linspace(0, 100, N_BINS)

    def bin_traj(tr_list, n_bins=N_BINS):
        bins = [[] for _ in range(n_bins)]
        for tr in tr_list:
            traj = tr.think_traj
            for i, v in enumerate(traj):
                b = min(int(i / len(traj) * n_bins), n_bins - 1)
                bins[b].append(v)
        return [np.mean(b) if b else np.nan for b in bins]

    for tr in correct[:8]:
        xs = np.linspace(0, 100, len(tr.think_traj))
        ax1.plot(xs, tr.think_traj, color="#2ecc71", alpha=0.25, lw=0.9)
    for tr in wrong[:8]:
        xs = np.linspace(0, 100, len(tr.think_traj))
        ax1.plot(xs, tr.think_traj, color="#e74c3c", alpha=0.25, lw=0.9)

    if correct:
        ax1.plot(xs_pct, bin_traj(correct), "#27ae60", lw=2.5, label=f"Correct avg (n={len(correct)})")
    if wrong:
        ax1.plot(xs_pct, bin_traj(wrong),   "#c0392b", lw=2.5, label=f"Wrong avg (n={len(wrong)})")

    ax1.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax1.set_xlabel("% through think block", fontsize=10)
    ax1.set_ylabel("J_know", fontsize=10)
    ax1.set_title("J_know through reasoning trace\nCorrect vs wrong runs", fontsize=10)
    ax1.legend(fontsize=8)

    # ── Panel 2: Commit % distribution ────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    bins = np.linspace(0, 100, 21)
    cor_c = [t.commit_pct for t in correct if t.commit_step is not None]
    wrg_c = [t.commit_pct for t in wrong   if t.commit_step is not None]
    if cor_c:
        ax2.hist(cor_c, bins=bins, alpha=0.6, color="#2ecc71",
                 label=f"Correct n={len(cor_c)}", density=True)
    if wrg_c:
        ax2.hist(wrg_c, bins=bins, alpha=0.6, color="#e74c3c",
                 label=f"Wrong n={len(wrg_c)}", density=True)
    ax2.axvline(stats["mean_commit_pct"], color="orange", ls="--", lw=2,
                label=f"Mean {stats['mean_commit_pct']:.1f}%")
    ax2.set_xlabel("% think tokens AFTER commitment", fontsize=10)
    ax2.set_title("Post-commitment token distribution\n— THE CORE FINDING —",
                  fontsize=10, fontweight="bold")
    ax2.legend(fontsize=8)

    # ── Panel 3: Early exit scatter ───────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    exits = [t for t in traces if t.answer_early != t.answer_full and t.commit_step is not None]
    if exits:
        saved  = [t.commit_pct for t in exits]
        deltas = [t.f1_full - t.f1_early for t in exits]
        sc = ax3.scatter(saved, deltas, c=[t.f1_full for t in exits],
                         cmap="RdYlGn", alpha=0.7, s=40, vmin=0, vmax=1)
        plt.colorbar(sc, ax=ax3, label="Full F1")
        ax3.axhline(0, color="gray", ls="--", lw=1)
        ax3.set_xlabel("% tokens saved by early exit", fontsize=10)
        ax3.set_ylabel("F1 delta  (full − early)", fontsize=10)
        ax3.set_title(
            f"Early exit impact\nMean Δ={stats['f1_delta']:+.3f}  "
            f"tokens saved={stats['tokens_saved_pct']:.1f}%", fontsize=10
        )
    else:
        ax3.text(0.5, 0.5, "No early exits simulated", ha="center", va="center")
        ax3.set_title("Early exit analysis", fontsize=10)

    # ── Panel 4: Null hypothesis ───────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    all_cpct = [t.commit_pct for t in traces if t.commit_step is not None]
    if all_cpct:
        ax4.hist(all_cpct, bins=20, alpha=0.7, color="#3498db",
                 label=f"Observed μ={stats['mean_commit_pct']:.1f}%", density=True, zorder=3)

    null_x = np.linspace(0, 100, 200)
    nm, ns = stats["null_mean"], max(stats["null_std"], 1e-3)
    ax4.plot(null_x,
             np.exp(-0.5 * ((null_x - nm) / ns) ** 2) / (ns * np.sqrt(2 * np.pi)),
             color="gray", ls="--", lw=2, label=f"Null (shuffled) μ={nm:.1f}%")
    ax4.axvline(stats["mean_commit_pct"], color="orange", lw=2)
    ax4.set_xlabel("% think tokens post-commit", fontsize=10)
    ax4.set_title(f"Observed vs shuffled null\nz = {stats['z_score']:.2f}", fontsize=10, fontweight="bold")
    ax4.legend(fontsize=8)

    # ── Panel 5: Commit % vs final F1 ────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    cp  = [t.commit_pct for t in traces if t.commit_step is not None]
    f1s = [t.f1_full    for t in traces if t.commit_step is not None]
    if cp:
        cols = ["#2ecc71" if f >= 0.4 else "#e74c3c" if f < 0.1 else "#95a5a6" for f in f1s]
        ax5.scatter(cp, f1s, c=cols, alpha=0.6, s=28)
        if len(cp) > 5:
            coef = np.polyfit(cp, f1s, 1)
            xf   = np.linspace(min(cp), max(cp), 100)
            ax5.plot(xf, np.polyval(coef, xf), "orange", lw=1.5, ls="--")
            rho = np.corrcoef(cp, f1s)[0, 1]
            ax5.set_title(f"Commit % vs final F1\nρ = {rho:.3f}", fontsize=10)
        ax5.set_xlabel("% tokens post-commit", fontsize=10)
        ax5.set_ylabel("Final answer F1", fontsize=10)

    # ── Panel 6: Summary ──────────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    txt = (
        f"REASONING GEOMETRY v1\n"
        f"{'='*38}\n\n"
        f"Model:    {MODEL_ID}\n"
        f"N:        {stats['n_total']}\n"
        f"Correct:  {stats['n_correct']}  "
        f"({100*stats['n_correct']/max(stats['n_total'],1):.0f}%)\n"
        f"Wrong:    {stats['n_wrong']}  "
        f"({100*stats['n_wrong']/max(stats['n_total'],1):.0f}%)\n\n"
        f"COMMITMENT MOMENT\n"
        f"  Commit rate:    {stats['commit_rate']*100:.1f}%\n"
        f"  Mean pct:       {stats['mean_commit_pct']:.1f}%\n"
        f"  Correct pct:    {stats['correct_commit_pct']:.1f}%\n"
        f"  Wrong pct:      {stats['wrong_commit_pct']:.1f}%\n"
        f"  Gap:            {stats['commit_gap']:+.1f}pp\n"
        f"  Null mean:      {stats['null_mean']:.1f}%\n"
        f"  z-score:        {stats['z_score']:.2f}\n\n"
        f"EARLY EXIT\n"
        f"  F1 delta:       {stats['f1_delta']:+.4f}\n"
        f"  Tokens saved:   {stats['tokens_saved_pct']:.1f}%\n\n"
        f"VERDICT: {stats['verdict']}\n"
    )
    ax6.text(0.04, 0.96, txt, transform=ax6.transAxes, fontsize=8.5,
             va="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="#f0f4f8", alpha=0.9))

    plt.suptitle(
        f"Epistemic Commitment in Reasoning Models — DeepSeek-R1-Distill-Qwen-1.5B\n"
        f"Mean {stats['mean_commit_pct']:.1f}% of think tokens are "
        f"post-commitment elaboration  (z = {stats['z_score']:.2f})",
        fontsize=12, fontweight="bold",
    )
    fig.savefig(str(FIGURE_FILE), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved → {FIGURE_FILE}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    all_samples = load_trivia_qa(n=450)
    rng.shuffle(all_samples)
    cal_samples  = all_samples[:200]
    main_samples = all_samples[200: 200 + MAIN_N]

    model, tok = load_model()
    n_layers   = model.config.num_hidden_layers
    think_end  = find_think_end_id(tok)
    print(f"</think> token id: {think_end}")

    probe = pick_layer(model, tok, cal_samples, n_layers)

    print(f"\nMain experiment: {MAIN_N} questions …")
    traces = []
    for i, s in enumerate(main_samples):
        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{MAIN_N}  ({elapsed:.0f}s  "
                  f"mean commit%={np.mean([t.commit_pct for t in traces if t.commit_step is not None] or [0]):.1f})")
        try:
            traces.append(run_trace(model, tok, probe, s, think_end))
        except Exception as e:
            print(f"  Error {i}: {e}")

    stats = statistics(traces)
    print(f"\n{'='*55}")
    print(f"VERDICT:  {stats['verdict']}")
    print(f"INTERP:   {stats['interpretation']}")
    print(f"z-score:  {stats['z_score']:.2f}")
    print(f"{'='*55}\n")

    make_figure(traces, stats)

    results = {
        "model":          MODEL_ID,
        "probe_layer":    probe.layer_idx,
        "cal_auroc":      probe.auroc,
        "stats":          stats,
        "elapsed_s":      time.time() - t0,
        "commit_window":  COMMIT_WINDOW,
        "commit_thresh":  COMMIT_THRESH,
        "commit_persist": COMMIT_PERSIST,
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results → {RESULTS_FILE}")
    print(f"Total: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
