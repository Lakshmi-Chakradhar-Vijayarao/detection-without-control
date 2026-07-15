"""
kaggle/reasoning_geometry_llama_v1/reasoning_geometry_llama_v1.py

EPISTEMIC COMMITMENT IN REASONING MODELS — LLAMA BACKBONE
===========================================================

Identical experiment to reasoning_geometry_v1, but on DeepSeek-R1-Distill-Llama-8B.

DeepSeek-R1-Distill-Qwen-1.5B uses the Qwen2.5 backbone.
DeepSeek-R1-Distill-Llama-8B uses the Llama 3 backbone.

Same RL reasoning fine-tuning, different base architecture.

If commitment moment detection works on BOTH:
  → the finding is not a Qwen artifact
  → it is a property of RL-trained reasoning, independent of base architecture
  → this is the cross-architecture validation needed for the paper

Architecture note:
  Llama-8B has 32 transformer layers, GQA attention, RoPE positional encoding.
  This is the same architecture family as Meta's Llama 3 production models.
  Proving the signal here gives Meta researchers direct relevance.

Model: deepseek-ai/DeepSeek-R1-Distill-Llama-8B
  — 8B at float16 = ~16 GB, exactly T4 VRAM budget
  — Use 4-bit quantization (bitsandbytes) for safety headroom
  — Llama 3 backbone: 32 layers, 4096 hidden dim, GQA

Output:
  reasoning_geometry_llama_v1_results.json
  reasoning_geometry_llama_v1_figure.png
"""

from __future__ import annotations

import os
# Reduce CUDA allocator fragmentation (reserved-but-unallocated memory)
# This is especially important during calibration where many KV caches are
# allocated and freed sequentially; expandable_segments returns freed memory
# back to the CUDA pool instead of holding it as reserved fragmentation.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                "bitsandbytes>=0.46.1"], check=True)

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_ID       = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
USE_4BIT       = True    # load in 4-bit to fit on T4
CAL_N          = 60
MAIN_N         = 100
MAX_THINK_TOK  = 1200
MAX_ANS_TOK    = 120
CAL_MAX_TOK    = 1200   # Match MAX_THINK_TOK: caps current_ids at ~1220 tokens so that
                        # model.generate(current_ids, ...) forward pass uses ≤0.19 GB attention.
                        # 6000-token sequences caused 4.33 GB allocation (32 × 5800² × 4B per layer).
COMMIT_WINDOW  = 10
# COMMIT_THRESH is not hardcoded — it is derived from calibration data as the p20 of
# correct-class probe scores (see calibrate()). This is architecture-adaptive: the
# threshold means "firmly in the PARAM region per this model's geometry," not an
# arbitrary scalar that was tuned on Qwen and blindly reused on Llama.
COMMIT_PERSIST = 15
N_BOOTSTRAP    = 400
SEED           = 42

rng = np.random.default_rng(SEED)

OUT_DIR      = Path("/kaggle/working")
RESULTS_FILE = OUT_DIR / "reasoning_geometry_llama_v1_results.json"
FIGURE_FILE  = OUT_DIR / "reasoning_geometry_llama_v1_figure.png"

assert torch.cuda.is_available(), "T4 GPU required"
_sm_major, _sm_minor = torch.cuda.get_device_capability(0)
_sm = _sm_major * 10 + _sm_minor
assert _sm >= 70, f"GPU sm_{_sm} not supported — need T4 (sm_75) or better. Re-run on T4."
DEVICE = "cuda"
print(f"GPU: {torch.cuda.get_device_name(0)}  (sm_{_sm})")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ─────────────────────────────────────────────────────────────────────────────
# Data  (identical to v1)
# ─────────────────────────────────────────────────────────────────────────────

def load_trivia_qa(n: int = 350) -> List[Dict]:
    from datasets import load_dataset
    ds  = load_dataset("trivia_qa", "rc.wikipedia", split="train", streaming=True)
    out = []
    for row in ds:
        if not row["answer"]["aliases"]:
            continue
        out.append({"question": row["question"], "answers": row["answer"]["aliases"]})
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
        pr = len(c) / len(p)
        rc = len(c) / len(q)
        best = max(best, 2 * pr * rc / (pr + rc))
    return best


def answer_contains(pred: str, golds: List[str]) -> bool:
    pred_l = pred.lower()
    for g in golds:
        g_l = g.lower().strip()
        if g_l and g_l in pred_l:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Model loading with optional 4-bit quantization
# ─────────────────────────────────────────────────────────────────────────────

def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {MODEL_ID} (4bit={USE_4BIT}) …")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if USE_4BIT:
        from transformers import BitsAndBytesConfig
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        mdl = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, quantization_config=bnb_cfg,
            device_map="auto", trust_remote_code=True,
        )
    else:
        mdl = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16, trust_remote_code=True,
        ).to(DEVICE)

    mdl.eval()
    print(f"  Loaded. Layers: {mdl.config.num_hidden_layers}  "
          f"Hidden: {mdl.config.hidden_size}")
    return mdl, tok


def find_think_end_id(tok) -> Optional[int]:
    for s in ["</think>", "<|/think|>", "</thinking>"]:
        ids = tok.encode(s, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Hidden-state capture (works with device_map="auto")
# ─────────────────────────────────────────────────────────────────────────────

class HSCapture:
    def __init__(self, model, layer_idx: int):
        self.hs: Optional[np.ndarray] = None
        # With device_map="auto", layer may be on any device — move to cpu in hook
        self._h = model.model.layers[layer_idx].register_forward_hook(self._fn)

    def _fn(self, mod, inp, out):
        x = out[0] if isinstance(out, tuple) else out
        self.hs = x[:, -1, :].detach().float().cpu().numpy()

    def remove(self):
        self._h.remove()


# ─────────────────────────────────────────────────────────────────────────────
# Fisher LDA calibration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Probe:
    direction:    np.ndarray
    mu_correct:   float
    mu_wrong:     float
    layer_idx:    int
    auroc:        float
    commit_thresh: float = 0.5   # set adaptively from calibration; see calibrate()

    def score(self, h: np.ndarray) -> float:
        p      = float(np.dot(h, self.direction))
        scale  = (self.mu_correct - self.mu_wrong) / 2.0
        center = (self.mu_correct + self.mu_wrong) / 2.0
        return (p - center) / (abs(scale) + 1e-9)


def calibrate_gen(model, tok, samples: List[Dict], layer: int, n_target: int) -> Probe:
    """
    Generation-time calibration using a semantic oracle (PARAM vs CTX_DEP).

    Previous approach (temporal split: first-5 vs last-10% tokens) failed on
    Llama-8B because the middle of the think block already scored positive
    (mean ≈ +0.25), making the threshold too permissive: commit_rate=100%,
    commit_gap=-6.4% (inverted — wrong answers committed earlier than correct).

    Root cause: position-in-sequence was the discriminant, not epistemic state.
    For Qwen-1.5B the two coincide; for Llama-8B they don't.

    This fix uses a semantic oracle:

      COMMITTED   (y=1): hidden states from the MIDDLE (20–80%) of think blocks
                         for PARAM questions (model answers correctly without context).
                         These represent "thinking about something you know."

      UNCOMMITTED (y=0): hidden states from the MIDDLE (20–80%) of think blocks
                         for CTX_DEP questions (model answers incorrectly).
                         These represent "thinking about something you don't know."

    Sampling from the middle of BOTH classes removes temporal position as a
    confound — the probe must learn the epistemic content axis, not just
    "early vs late token."

    Threshold: p20 of LATE (80%+) think tokens from held-out PARAM questions
    (not used in LDA fitting). Late PARAM tokens are the maximally committed
    endpoint; p20 gives a principled threshold without any temporal confound.
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import roc_auc_score

    think_end_id = find_think_end_id(tok)
    print(f"  Calibrating layer {layer} (semantic oracle, think_end_id={think_end_id}) …",
          flush=True)

    committed_hs:     List[np.ndarray] = []  # PARAM-middle tokens → LDA (capped at n_half)
    uncommitted_hs:   List[np.ndarray] = []  # CTX_DEP-middle tokens → LDA (capped at n_half)
    committed_hs_all: List[np.ndarray] = []  # PARAM-late tokens → threshold (uncapped)

    n_half         = n_target // 2
    MID_START_FRAC = 0.20   # middle window: removes early-token temporal bias
    MID_END_FRAC   = 0.80
    LATE_START_FRAC = 0.80  # late window: used for threshold estimation only
    N_PER_QUESTION  = 3     # evenly-spaced tokens per question for LDA (diversity > density)
    THRESH_N_COMMITTED = n_half + 30  # need ≥30 held-out late-PARAM states for robust p20

    for idx, s in enumerate(samples):
        lda_done    = len(committed_hs) >= n_half and len(uncommitted_hs) >= n_half
        thresh_done = len(committed_hs_all) >= THRESH_N_COMMITTED
        if lda_done and thresh_done:
            break

        msgs       = [{"role": "user", "content": s["question"]}]
        prompt     = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        prompt_ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)

        cap         = HSCapture(model, layer)
        all_hs:     List[np.ndarray] = []
        think_ids:  List[int]        = []
        current_ids = prompt_ids.clone()
        past_kv     = None
        out         = None

        with torch.no_grad():
            for _ in range(CAL_MAX_TOK):
                if past_kv is None:
                    out = model(current_ids, use_cache=True)
                else:
                    out = model(current_ids[:, -1:], past_key_values=past_kv, use_cache=True)
                past_kv = out.past_key_values
                if cap.hs is not None:
                    all_hs.append(cap.hs[0].copy())
                next_tok = int(torch.argmax(out.logits[0, -1]).item())
                think_ids.append(next_tok)
                current_ids = torch.cat(
                    [current_ids, torch.tensor([[next_tok]], device=DEVICE)], dim=1
                )
                if next_tok == tok.eos_token_id:
                    break
                if think_end_id is not None and next_tok == think_end_id:
                    break
        cap.remove()
        del past_kv, out
        torch.cuda.empty_cache()

        n_think      = len(all_hs)
        think_closed = (think_end_id is not None and
                        len(think_ids) > 0 and think_ids[-1] == think_end_id)

        with torch.no_grad():
            out_ans = model.generate(current_ids, max_new_tokens=150, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
        ans        = tok.decode(out_ans[0][current_ids.shape[1]:],
                                skip_special_tokens=True).strip()
        has_answer = answer_contains(ans, s["answers"])
        f1         = token_f1(ans, s["answers"])
        del out_ans, current_ids
        torch.cuda.empty_cache()

        if idx < 3:
            print(f"    DBG [{idx}] n_think={n_think} closed={think_closed} "
                  f"has_ans={has_answer} f1={f1:.3f} ans='{ans[:50]}'", flush=True)
        if idx % 5 == 0:
            print(f"    cal [{idx}] COMMITTED={len(committed_hs)}/{n_half}  "
                  f"UNCOMMITTED={len(uncommitted_hs)}/{n_half}  "
                  f"thresh_pool={len(committed_hs_all)}/{THRESH_N_COMMITTED}", flush=True)

        think_hs     = all_hs[1:]   # skip prompt's last token
        n_think_pure = len(think_hs)

        if n_think_pure < 25:
            continue   # too short to sample middle window reliably

        # Middle window: 20–80% of think block (removes temporal position artifact)
        mid_start = int(n_think_pure * MID_START_FRAC)
        mid_end   = int(n_think_pure * MID_END_FRAC)
        mid_hs    = think_hs[mid_start:mid_end]

        # Evenly-spaced subsample from the middle window
        step    = max(1, len(mid_hs) // N_PER_QUESTION)
        sampled = mid_hs[::step][:N_PER_QUESTION]

        if has_answer:
            # PARAM: middle tokens → LDA committed class
            for h in sampled:
                if len(committed_hs) < n_half:
                    committed_hs.append(h)
            # Late tokens → held-out pool for threshold estimation
            late_start = int(n_think_pure * LATE_START_FRAC)
            for h in think_hs[late_start:]:
                committed_hs_all.append(h)
        else:
            # CTX_DEP: middle tokens → LDA uncommitted class
            for h in sampled:
                if len(uncommitted_hs) < n_half:
                    uncommitted_hs.append(h)

    if len(committed_hs) < 5 or len(uncommitted_hs) < 5:
        raise RuntimeError(
            f"Too sparse: committed={len(committed_hs)}, uncommitted={len(uncommitted_hs)}"
        )

    print(f"    Fitting LDA: committed={len(committed_hs)}, "
          f"uncommitted={len(uncommitted_hs)}, dim={len(committed_hs[0])}", flush=True)

    X = np.stack(committed_hs + uncommitted_hs)
    y = np.array([1] * len(committed_hs) + [0] * len(uncommitted_hs))

    # Ledoit-Wolf shrinkage handles n << p (typically ~100 samples in 4096 dims) robustly.
    # Without shrinkage, Fisher LDA in high dimensions overfits the scatter matrix.
    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    lda.fit(X, y)
    d     = lda.coef_[0] / (np.linalg.norm(lda.coef_[0]) + 1e-9)
    projs = X @ d
    mu_c  = float(np.mean(projs[y == 1]))
    mu_w  = float(np.mean(projs[y == 0]))

    # Ensure committed class always has higher projection (sign convention)
    if mu_c < mu_w:
        d = -d
        projs = X @ d
        mu_c  = float(np.mean(projs[y == 1]))
        mu_w  = float(np.mean(projs[y == 0]))

    auroc = float(roc_auc_score(y, X @ d))

    scale  = (mu_c - mu_w) / 2.0
    center = (mu_c + mu_w) / 2.0

    # Threshold from late-PARAM states (held-out from LDA — LDA was trained on middle tokens).
    # Late PARAM tokens represent the maximally committed endpoint. p20 of these gives:
    # "80% of fully-committed states exceed this threshold" without any temporal confound,
    # since the LDA never saw late tokens (it was trained on middle-window tokens only).
    holdout_committed = committed_hs_all  # all late-PARAM tokens; no overlap with LDA training
    if len(holdout_committed) >= 10:
        cal_committed_scores = [(float(np.dot(h, d)) - center) / (abs(scale) + 1e-9)
                                for h in holdout_committed]
        commit_thresh = float(np.percentile(cal_committed_scores, 20))
        thresh_source = f"p20 of {len(holdout_committed)} held-out committed states"
    else:
        # Fallback: geometric midpoint between decision boundary (0) and committed pole (+1)
        cal_committed_scores = []
        commit_thresh = 0.5
        thresh_source = "geometric fallback (no held-out committed states)"
    commit_thresh = max(commit_thresh, 0.05)

    print(f"    Layer {layer}: AUROC={auroc:.4f}  mu_committed={mu_c:.4f}  "
          f"mu_uncommitted={mu_w:.4f}  scale={scale:.4f}  "
          f"commit_thresh={commit_thresh:.4f} ({thresh_source})", flush=True)
    return Probe(direction=d, mu_correct=mu_c, mu_wrong=mu_w,
                 layer_idx=layer, auroc=auroc, commit_thresh=commit_thresh)


def pick_layer(model, tok, samples, n_layers) -> Probe:
    """
    Semantic-oracle calibration at deep layers.
    n_target=20 (10 PARAM-middle + 10 CTX_DEP-middle for LDA).

    At N_PER_QUESTION=3 tokens per question and ~44% PARAM rate:
    need ~4 PARAM + ~4 CTX_DEP questions = ~9 total for LDA.
    Plus ~2 more PARAM questions for threshold pool (THRESH_N_COMMITTED=40).
    Total: ~15 questions × ~100s = ~25 min calibration, leaving ~7h for main.

    Layer priority: L28 (deepest semantic representations) → L18 (strong
    cross-model signal per layer sweep, n_layers - 14 for 32-layer Llama).
    """
    preferred  = n_layers - 4        # L28 for 32-layer Llama
    fallback   = n_layers - 14       # L18 (strong CS signal in layer sweep)
    candidates = [preferred, fallback]
    candidates = [l for l in dict.fromkeys(candidates) if 0 <= l < n_layers]
    for l in candidates:
        try:
            p = calibrate_gen(model, tok, samples[:150], l, n_target=20)
            print(f"Best: layer {p.layer_idx} AUROC={p.auroc:.4f}", flush=True)
            return p
        except Exception as e:
            print(f"  Layer {l} failed: {e}", flush=True)
    raise RuntimeError("All candidate layers failed calibration")


# ─────────────────────────────────────────────────────────────────────────────
# Commitment detection  (identical logic to v1)
# ─────────────────────────────────────────────────────────────────────────────

def find_commit(traj: List[float], thresh: float) -> Optional[int]:
    n = len(traj)
    t = np.array(traj)
    if n < COMMIT_WINDOW + COMMIT_PERSIST:
        return None
    for i in range(n - COMMIT_WINDOW - COMMIT_PERSIST + 1):
        if np.mean(t[i: i + COMMIT_WINDOW]) >= thresh:
            if np.min(t[i + COMMIT_WINDOW: i + COMMIT_WINDOW + COMMIT_PERSIST]) >= thresh * 0.8:
                return i
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Reasoning trace
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Trace:
    question:     str
    answers:      List[str]
    think_traj:   List[float] = field(default_factory=list)
    think_len:    int         = 0
    answer_full:  str         = ""
    answer_early: str         = ""
    f1_full:      float       = 0.0
    f1_early:     float       = 0.0
    has_answer:   bool        = False
    commit_step:  Optional[int] = None
    commit_pct:   float         = 0.0


def run_trace(model, tok, probe: Probe, sample: Dict,
              think_end_id: Optional[int]) -> Trace:
    msgs   = [{"role": "user", "content": sample["question"]}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    tr     = Trace(question=sample["question"], answers=sample["answers"])
    cap    = HSCapture(model, probe.layer_idx)

    try:
        prompt_ids  = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
        current_ids = prompt_ids.clone()
        think_ids: List[int] = []

        # KV-cached token-by-token loop — O(1) attention per step instead of O(n²).
        # First forward: full prompt, get KV cache.
        # Subsequent forwards: single new token + past_key_values.
        past_kv = None
        for _ in range(MAX_THINK_TOK):
            with torch.no_grad():
                if past_kv is None:
                    out     = model(current_ids, use_cache=True)
                else:
                    out     = model(current_ids[:, -1:], past_key_values=past_kv,
                                    use_cache=True)
            past_kv = out.past_key_values
            if cap.hs is not None:
                tr.think_traj.append(probe.score(cap.hs[0]))

            next_tok = int(torch.argmax(out.logits[0, -1]).item())
            think_ids.append(next_tok)
            current_ids = torch.cat(
                [current_ids, torch.tensor([[next_tok]], device=DEVICE)], dim=1
            )
            if next_tok == tok.eos_token_id:
                break
            if think_end_id is not None and next_tok == think_end_id:
                break

        tr.think_len = len(think_ids)

        with torch.no_grad():
            ans_ids = model.generate(current_ids, max_new_tokens=MAX_ANS_TOK,
                                     do_sample=False, pad_token_id=tok.eos_token_id)
        tr.answer_full = tok.decode(ans_ids[0][current_ids.shape[1]:],
                                    skip_special_tokens=True).strip()
        tr.f1_full    = token_f1(tr.answer_full, sample["answers"])
        tr.has_answer = answer_contains(tr.answer_full, sample["answers"])

        commit = find_commit(tr.think_traj, probe.commit_thresh)
        tr.commit_step = commit
        if commit is not None and tr.think_len > 0:
            tr.commit_pct = 100.0 * (tr.think_len - commit) / tr.think_len

        # Early exit simulation
        if commit is not None and commit + COMMIT_WINDOW < tr.think_len - 5:
            trunc      = think_ids[: commit + COMMIT_WINDOW]
            t_t        = torch.tensor([trunc], device=DEVICE)
            end_t      = (torch.tensor([[think_end_id]], device=DEVICE)
                          if think_end_id is not None
                          else torch.zeros((1, 0), dtype=torch.long, device=DEVICE))
            early_in   = torch.cat([prompt_ids, t_t, end_t], dim=1)
            with torch.no_grad():
                e_out = model.generate(early_in, max_new_tokens=MAX_ANS_TOK,
                                       do_sample=False, pad_token_id=tok.eos_token_id)
            tr.answer_early = tok.decode(e_out[0][early_in.shape[1]:],
                                         skip_special_tokens=True).strip()
            tr.f1_early = token_f1(tr.answer_early, sample["answers"])
        else:
            tr.answer_early = tr.answer_full
            tr.f1_early     = tr.f1_full

    finally:
        cap.remove()
    return tr


# ─────────────────────────────────────────────────────────────────────────────
# Statistics + figure  (mirrors v1 exactly for direct comparison)
# ─────────────────────────────────────────────────────────────────────────────

def statistics(traces: List[Trace], commit_thresh: float) -> Dict:
    correct   = [t for t in traces if t.has_answer]
    wrong     = [t for t in traces if not t.has_answer and t.f1_full < 0.1]
    committed = [t for t in traces if t.commit_step is not None]
    all_cpct  = [t.commit_pct for t in committed]
    cor_cpct  = [t.commit_pct for t in correct if t.commit_step is not None]
    wrg_cpct  = [t.commit_pct for t in wrong   if t.commit_step is not None]
    obs_mean  = float(np.mean(all_cpct)) if all_cpct else 0.0

    exits    = [t for t in committed if t.answer_early != t.answer_full]
    f1_delta = float(np.mean([t.f1_full - t.f1_early for t in exits])) if exits else 0.0
    tok_svd  = float(np.mean([t.commit_pct for t in exits])) if exits else 0.0

    all_trajs = [t.think_traj for t in traces if len(t.think_traj) > COMMIT_WINDOW + COMMIT_PERSIST]
    null_boot_means = []
    for _ in range(N_BOOTSTRAP):
        if not all_trajs:
            break
        boot_cpcts = []
        for traj in all_trajs:
            t = list(traj)
            rng.shuffle(t)
            c = find_commit(t, commit_thresh)
            boot_cpcts.append(100.0 * (len(t) - c) / len(t) if c is not None else 0.0)
        null_boot_means.append(float(np.mean(boot_cpcts)))

    null_mean = float(np.mean(null_boot_means)) if null_boot_means else 0.0
    null_std  = float(np.std(null_boot_means))  if null_boot_means else 1.0
    z         = (obs_mean - null_mean) / max(null_std, 0.1)

    return {
        "model":              MODEL_ID,
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
        "tokens_saved_pct":   tok_svd,
        "null_mean":          null_mean,
        "null_std":           null_std,
        "z_score":            z,
        "commit_thresh":      commit_thresh,   # adaptive; reported for reproducibility
        "verdict": ("COMMITTED_EARLY" if z > 3 and obs_mean > 40 else
                    "WEAK_SIGNAL"     if z > 2 and obs_mean > 20 else "NULL"),
    }


def make_figure(traces: List[Trace], stats: Dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    correct = [t for t in traces if t.has_answer and t.think_traj]
    wrong   = [t for t in traces if not t.has_answer and t.f1_full < 0.1 and t.think_traj]

    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(2, 3, hspace=0.42, wspace=0.35)

    N_BINS = 50
    xs_pct = np.linspace(0, 100, N_BINS)

    def bin_traj(tr_list):
        bins = [[] for _ in range(N_BINS)]
        for tr in tr_list:
            traj = tr.think_traj
            for i, v in enumerate(traj):
                b = min(int(i / len(traj) * N_BINS), N_BINS - 1)
                bins[b].append(v)
        return [np.mean(b) if b else np.nan for b in bins]

    ax1 = fig.add_subplot(gs[0, 0])
    for tr in correct[:6]:
        ax1.plot(np.linspace(0, 100, len(tr.think_traj)), tr.think_traj,
                 color="#2ecc71", alpha=0.25, lw=0.9)
    for tr in wrong[:6]:
        ax1.plot(np.linspace(0, 100, len(tr.think_traj)), tr.think_traj,
                 color="#e74c3c", alpha=0.25, lw=0.9)
    if correct:
        ax1.plot(xs_pct, bin_traj(correct), "#27ae60", lw=2.5,
                 label=f"Correct avg (n={len(correct)})")
    if wrong:
        ax1.plot(xs_pct, bin_traj(wrong), "#c0392b", lw=2.5,
                 label=f"Wrong avg (n={len(wrong)})")
    ax1.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax1.set_xlabel("% through think block", fontsize=10)
    ax1.set_ylabel("J_know", fontsize=10)
    ax1.set_title("Llama backbone — J_know trajectories", fontsize=10)
    ax1.legend(fontsize=8)

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
    ax2.set_title("Post-commitment % — Llama backbone", fontsize=10, fontweight="bold")
    ax2.legend(fontsize=8)

    ax3 = fig.add_subplot(gs[0, 2])
    exits = [t for t in traces if t.answer_early != t.answer_full]
    if exits:
        ax3.scatter([t.commit_pct for t in exits],
                    [t.f1_full - t.f1_early for t in exits],
                    alpha=0.6, s=35, color="#3498db")
        ax3.axhline(0, color="gray", ls="--")
        ax3.set_title(
            f"Early exit: Δ={stats['f1_delta']:+.3f}  "
            f"saved={stats['tokens_saved_pct']:.1f}%", fontsize=10)
        ax3.set_xlabel("% tokens saved", fontsize=10)
        ax3.set_ylabel("F1 delta", fontsize=10)

    ax4 = fig.add_subplot(gs[1, 0])
    all_c = [t.commit_pct for t in traces if t.commit_step is not None]
    if all_c:
        ax4.hist(all_c, bins=20, alpha=0.7, color="#3498db", density=True, zorder=3)
    null_x = np.linspace(0, 100, 200)
    nm, ns = stats["null_mean"], max(stats["null_std"], 1e-3)
    ax4.plot(null_x, np.exp(-0.5 * ((null_x - nm) / ns) ** 2) / (ns * np.sqrt(2 * np.pi)),
             "gray", ls="--", lw=2, label=f"Null μ={nm:.1f}%")
    ax4.axvline(stats["mean_commit_pct"], color="orange", lw=2)
    ax4.set_title(f"Null hypothesis  z={stats['z_score']:.2f}", fontsize=10, fontweight="bold")
    ax4.legend(fontsize=8)

    ax5 = fig.add_subplot(gs[1, 1])
    cp  = [t.commit_pct for t in traces if t.commit_step is not None]
    f1v = [t.f1_full    for t in traces if t.commit_step is not None]
    if cp:
        cols = ["#2ecc71" if f >= 0.4 else "#e74c3c" if f < 0.1 else "#95a5a6" for f in f1v]
        ax5.scatter(cp, f1v, c=cols, alpha=0.6, s=28)
        rho = np.corrcoef(cp, f1v)[0, 1] if len(cp) > 5 else 0.0
        ax5.set_title(f"Commit % vs F1  ρ={rho:.3f}", fontsize=10)
        ax5.set_xlabel("% tokens post-commit", fontsize=10)
        ax5.set_ylabel("F1", fontsize=10)

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    txt = (
        f"REASONING GEOMETRY — LLAMA BACKBONE\n"
        f"{'='*38}\n\n"
        f"Model:  {MODEL_ID}\n"
        f"N:      {stats['n_total']}\n"
        f"Commit rate:  {stats['commit_rate']*100:.1f}%\n"
        f"Mean pct:     {stats['mean_commit_pct']:.1f}%\n"
        f"Correct pct:  {stats['correct_commit_pct']:.1f}%\n"
        f"Wrong pct:    {stats['wrong_commit_pct']:.1f}%\n"
        f"z-score:      {stats['z_score']:.2f}\n"
        f"F1 delta:     {stats['f1_delta']:+.4f}\n"
        f"Tokens saved: {stats['tokens_saved_pct']:.1f}%\n\n"
        f"VERDICT: {stats['verdict']}\n"
    )
    ax6.text(0.04, 0.96, txt, transform=ax6.transAxes, fontsize=8.5,
             va="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="#f0f4f8", alpha=0.9))

    plt.suptitle(
        f"Epistemic Commitment — Llama Backbone ({MODEL_ID})\n"
        f"Cross-backbone validation: Qwen architecture vs Llama architecture",
        fontsize=11, fontweight="bold",
    )
    fig.savefig(str(FIGURE_FILE), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure → {FIGURE_FILE}")


def main():
    t0 = time.time()
    all_s = load_trivia_qa(350)
    rng.shuffle(all_s)
    cal_s  = all_s[:150]
    main_s = all_s[150: 150 + MAIN_N]

    model, tok = load_model()
    n_layers   = model.config.num_hidden_layers
    think_end  = find_think_end_id(tok)

    probe = pick_layer(model, tok, cal_s, n_layers)

    print(f"\nMain experiment: {MAIN_N} questions …", flush=True)
    traces = []
    for i, s in enumerate(main_s):
        try:
            tr = run_trace(model, tok, probe, s, think_end)
            traces.append(tr)
            # Probe-score diagnostic for first 3 questions — helps tune COMMIT_THRESH
            if i < 3 and tr.think_traj:
                t_arr = tr.think_traj
                print(f"  DBG traj[{i}] len={len(t_arr)} "
                      f"min={min(t_arr):.3f} max={max(t_arr):.3f} "
                      f"mean={float(np.mean(t_arr)):.3f} "
                      f"commit={tr.commit_step}", flush=True)
        except Exception as e:
            print(f"  Error {i}: {e}", flush=True)
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            committed = [t for t in traces if t.commit_step is not None]
            mean_cpct = float(np.mean([t.commit_pct for t in committed])) if committed else 0.0
            print(f"  {i+1}/{MAIN_N}  ({elapsed:.0f}s  mean commit%={mean_cpct:.1f})", flush=True)
            # Intermediate save — preserves partial results if run times out
            partial = {
                "partial": True,
                "traces_so_far": i + 1,
                "mean_commit_pct": mean_cpct,
                "commit_rate": len(committed) / max(i + 1, 1),
                "probe_layer": probe.layer_idx,
                "cal_auroc": probe.auroc,
                "commit_thresh": probe.commit_thresh,
                "elapsed_s": elapsed,
            }
            RESULTS_FILE.write_text(json.dumps(partial, indent=2))

    stats = statistics(traces, probe.commit_thresh)
    print(f"\nVERDICT: {stats['verdict']}  z={stats['z_score']:.2f}  "
          f"mean={stats['mean_commit_pct']:.1f}%  thresh={probe.commit_thresh:.4f}", flush=True)

    make_figure(traces, stats)
    results = {"stats": stats, "probe_layer": probe.layer_idx,
               "cal_auroc": probe.auroc, "elapsed_s": time.time() - t0}
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results → {RESULTS_FILE}   Total: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
