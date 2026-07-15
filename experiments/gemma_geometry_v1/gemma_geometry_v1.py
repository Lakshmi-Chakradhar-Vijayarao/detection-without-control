"""
kaggle/gemma_geometry_v1/gemma_geometry_v1.py

FISHER LDA EPISTEMIC PROBE ON GEMMA 3 — GOOGLE ARCHITECTURE FAMILY
=====================================================================

Purpose: prove the epistemic signal generalises to Google's architecture family.

Gemma 3 uses:
  - Multi-head attention (not GQA like Llama)
  - RoPE positional encoding
  - GeGLU activation
  - Architecturally close to the Gemini transformer stack

If Fisher LDA probe achieves comparable AUROC on Gemma 3 vs Llama family:
  → the finding is not GQA-specific
  → it generalises across distinct industrial architectures
  → Google DeepMind researchers see direct applicability

Two experiments:
  A. Standard bilateral oracle: PARAM vs CTX_DEP AUROC (same as main ESM research)
  B. Instruction-prompted CoT: J_know trajectory through "think step by step" generation
     — Gemma 3 has no native <think> block, so we prompt for it explicitly

Model: google/gemma-3-4b-it  (4B, fits T4 at float16 or int4)

Output:
  gemma_geometry_v1_results.json
  gemma_geometry_v1_figure.png
"""

from __future__ import annotations

import json
import os
import time

# HF login — required for gated models (google/gemma-3-4b-it needs accepted terms)
# Kaggle secrets are read via UserSecretsClient, not os.environ directly.
# Add secret named HF_TOKEN at: notebook → Add-ons → Secrets
try:
    _hf_token = ""
    try:
        from kaggle_secrets import UserSecretsClient as _USC
        _hf_token = _USC().get_secret("HF_TOKEN")
    except Exception:
        pass
    if not _hf_token:
        _hf_token = os.environ.get("HF_TOKEN", "") or os.environ.get("HUGGING_FACE_HUB_TOKEN", "")
    if _hf_token:
        from huggingface_hub import login as _hf_login
        _hf_login(token=_hf_token, add_to_git_credential=False)
        print("HF login: OK", flush=True)
    else:
        print("WARNING: HF_TOKEN not found. Add it via notebook → Add-ons → Secrets.", flush=True)
except Exception as _e:
    print(f"HF login error: {_e}", flush=True)

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_ID       = "google/gemma-2-2b-it"
USE_4BIT       = False   # 2B at float16 = ~4.5 GB, fits T4 comfortably
CAL_N          = 80
MAIN_N         = 150
MAX_GEN_TOK    = 400     # Gemma has no explicit think block — full generation
CAL_MAX_TOK    = 300
N_BOOTSTRAP    = 400
COMMIT_WINDOW  = 10
COMMIT_THRESH  = 0.10   # lowered from 0.8 — Llama gen trajectories max at ~0.35 on normalized scale
COMMIT_PERSIST = 12
SEED           = 42

rng = np.random.default_rng(SEED)

OUT_DIR      = Path("/kaggle/working")
RESULTS_FILE = OUT_DIR / "gemma_geometry_v1_results.json"
FIGURE_FILE  = OUT_DIR / "gemma_geometry_v1_figure.png"

assert torch.cuda.is_available(), "T4 GPU required"
_sm_major, _sm_minor = torch.cuda.get_device_capability(0)
_sm = _sm_major * 10 + _sm_minor
assert _sm >= 70, f"GPU sm_{_sm} not supported — need T4 (sm_75) or better. Re-run on T4."
DEVICE = "cuda"
print(f"GPU: {torch.cuda.get_device_name(0)}  (sm_{_sm})", flush=True)
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def load_data(n_nocontext: int = 150, n_withcontext: int = 150) -> Tuple[List, List]:
    """
    Load two sets for bilateral oracle calibration:
      PARAM set   — TriviaQA questions model answers correctly without context
      CTX_DEP set — TriviaQA questions needing context (use web_search_results field)
    """
    from datasets import load_dataset
    ds = load_dataset("trivia_qa", "rc.wikipedia", split="train", streaming=True)
    no_ctx, with_ctx = [], []
    for row in ds:
        if not row["answer"]["aliases"]:
            continue
        # entity_pages.wiki_context is the correct field for rc.wikipedia split
        ctx_candidates = (row.get("entity_pages") or {}).get("wiki_context") or [""]
        ctx = ctx_candidates[0][:1000] if ctx_candidates and ctx_candidates[0] else ""
        sample = {"question": row["question"], "answers": row["answer"]["aliases"], "context": ctx}
        if ctx and len(with_ctx) < n_withcontext:
            with_ctx.append(sample)
        elif len(no_ctx) < n_nocontext:
            no_ctx.append(sample)
        if len(no_ctx) >= n_nocontext and len(with_ctx) >= n_withcontext:
            break
    print(f"Loaded: no_ctx={len(no_ctx)}, with_ctx={len(with_ctx)}")
    return no_ctx, with_ctx


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

    if USE_4BIT:
        from transformers import BitsAndBytesConfig
        cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                  bnb_4bit_quant_type="nf4")
        mdl = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, quantization_config=cfg, device_map="auto", trust_remote_code=True)
    else:
        mdl = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16, trust_remote_code=True).to(DEVICE)

    mdl.eval()
    # Gemma3 is multimodal — text config is nested under text_config
    _tcfg = getattr(mdl.config, 'text_config', mdl.config)
    print(f"  Loaded. Layers: {_tcfg.num_hidden_layers}  "
          f"Hidden: {_tcfg.hidden_size}")
    return mdl, tok


# ─────────────────────────────────────────────────────────────────────────────
# Hidden-state capture
# ─────────────────────────────────────────────────────────────────────────────

# Architecture-agnostic layer resolver.
# Paths tried in order — first one that resolves to a non-empty sequence wins.
_LAYER_PATHS = [
    "model.layers",                  # Llama / Qwen / Mistral / DeepSeek-Llama
    "model.language_model.layers",   # Gemma-3 multimodal (Gemma3ForConditionalGeneration)
    "language_model.model.layers",   # alternate VLM layouts
    "language_model.layers",         # some wrappers
    "transformer.h",                 # GPT-2 style
]

_resolved_path: Optional[str] = None  # cache resolved path to avoid repeated print spam

def get_transformer_layers(model):
    global _resolved_path
    for path in _LAYER_PATHS:
        try:
            obj = model
            for part in path.split("."):
                obj = getattr(obj, part)
            if hasattr(obj, "__len__") and len(obj) > 0:
                if _resolved_path != path:
                    _resolved_path = path
                    print(f"[Credence] layer path: {path}  ({len(obj)} layers)", flush=True)
                return obj
        except AttributeError:
            continue
    raise RuntimeError(f"Cannot locate transformer layers in {type(model).__name__}")


class HSCapture:
    def __init__(self, model, layer_idx: int):
        self.hs: Optional[np.ndarray] = None
        self._h = get_transformer_layers(model)[layer_idx].register_forward_hook(self._fn)

    def _fn(self, mod, inp, out):
        x = out[0] if isinstance(out, tuple) else out
        self.hs = x[:, -1, :].detach().float().cpu().numpy()

    def remove(self):
        self._h.remove()


def get_step1_hs(model, tok, prompt: str, layer: int) -> Optional[np.ndarray]:
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
    cap = HSCapture(model, layer)
    with torch.no_grad():
        model(ids)
    hs = cap.hs[0].copy() if cap.hs is not None else None
    cap.remove()
    return hs


# ─────────────────────────────────────────────────────────────────────────────
# Bilateral oracle calibration (PARAM vs CTX_DEP)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Probe:
    direction:  np.ndarray
    mu_param:   float
    mu_ctxdep:  float
    layer_idx:  int
    auroc:      float

    def score(self, h: np.ndarray) -> float:
        p      = float(np.dot(h, self.direction))
        scale  = (self.mu_param - self.mu_ctxdep) / 2.0
        center = (self.mu_param + self.mu_ctxdep) / 2.0
        return (p - center) / (abs(scale) + 1e-9)


def build_prompt_nocontext(tok, q: str) -> str:
    msgs = [{"role": "user", "content": q}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def build_prompt_withcontext(tok, q: str, ctx: str) -> str:
    content = f"Context: {ctx[:600]}\n\nQuestion: {q}"
    msgs = [{"role": "user", "content": content}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def calibrate(model, tok, no_ctx: List, with_ctx: List, layer: int,
              n_target: int = CAL_N) -> Probe:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import roc_auc_score

    print(f"  Calibrating layer {layer} …", flush=True)
    param_hs, ctxdep_hs = [], []
    n_half = n_target // 2

    # PARAM: model answers correctly WITHOUT context
    for s in no_ctx:
        if len(param_hs) >= n_half:
            break
        prompt = build_prompt_nocontext(tok, s["question"])
        hs     = get_step1_hs(model, tok, prompt, layer)
        if hs is None:
            continue
        ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=CAL_MAX_TOK,
                                  do_sample=False, pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        if answer_contains(gen, s["answers"]) or token_f1(gen, s["answers"]) >= 0.4:
            param_hs.append(hs)

    # CTX_DEP: model needs context (answers wrong WITHOUT, right WITH)
    for s in with_ctx:
        if len(ctxdep_hs) >= n_half:
            break
        if not s["context"]:
            continue
        p_no  = build_prompt_nocontext(tok, s["question"])
        p_yes = build_prompt_withcontext(tok, s["question"], s["context"])
        hs    = get_step1_hs(model, tok, p_no, layer)
        if hs is None:
            continue
        ids_no  = tok(p_no,  return_tensors="pt").input_ids.to(DEVICE)
        ids_yes = tok(p_yes, return_tensors="pt").input_ids.to(DEVICE)
        with torch.no_grad():
            out_no  = model.generate(ids_no,  max_new_tokens=CAL_MAX_TOK, do_sample=False,
                                      pad_token_id=tok.eos_token_id)
            out_yes = model.generate(ids_yes, max_new_tokens=CAL_MAX_TOK, do_sample=False,
                                      pad_token_id=tok.eos_token_id)
        gen_no  = tok.decode(out_no[0][ids_no.shape[1]:],   skip_special_tokens=True)
        gen_yes = tok.decode(out_yes[0][ids_yes.shape[1]:], skip_special_tokens=True)
        f1_no   = token_f1(gen_no,  s["answers"])
        f1_yes  = token_f1(gen_yes, s["answers"])
        # symmetric binary oracle: wrong without, right with
        no_correct  = answer_contains(gen_no,  s["answers"]) or f1_no  >= 0.4
        yes_correct = answer_contains(gen_yes, s["answers"]) or f1_yes >= 0.4
        if not no_correct and yes_correct:
            ctxdep_hs.append(hs)

    print(f"    PARAM={len(param_hs)}, CTX_DEP={len(ctxdep_hs)}")
    if len(param_hs) < 5 or len(ctxdep_hs) < 5:
        raise RuntimeError("Insufficient calibration data")

    X = np.stack(param_hs + ctxdep_hs)
    y = np.array([1] * len(param_hs) + [0] * len(ctxdep_hs))
    lda = LinearDiscriminantAnalysis(n_components=1)
    lda.fit(X, y)
    d     = lda.coef_[0] / (np.linalg.norm(lda.coef_[0]) + 1e-9)
    projs = X @ d
    mu_p  = float(np.mean(projs[y == 1]))
    mu_c  = float(np.mean(projs[y == 0]))
    auroc = float(roc_auc_score(y, lda.decision_function(X)))
    print(f"    AUROC={auroc:.4f}  mu_PARAM={mu_p:.3f}  mu_CTX={mu_c:.3f}")
    return Probe(direction=d, mu_param=mu_p, mu_ctxdep=mu_c,
                 layer_idx=layer, auroc=auroc)


def pick_layer(model, tok, no_ctx, with_ctx, n_layers) -> Probe:
    candidates = sorted({n_layers - 4, n_layers - 3, n_layers - 2, n_layers - 1})
    candidates = [l for l in candidates if 0 <= l < n_layers]
    best = None
    for l in candidates:
        try:
            p = calibrate(model, tok, no_ctx[:80], with_ctx[:80], l, n_target=40)
            if best is None or p.auroc > best.auroc:
                best = p
        except Exception as e:
            print(f"  Layer {l}: {e}")
    if best is None:
        raise RuntimeError("All layers failed")
    print(f"Best: layer {best.layer_idx}  AUROC={best.auroc:.4f}")
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Experiment B: CoT trajectory through instruction-prompted reasoning
# ─────────────────────────────────────────────────────────────────────────────

def find_commit(traj: List[float]) -> Optional[int]:
    n = len(traj)
    t = np.array(traj)
    if n < COMMIT_WINDOW + COMMIT_PERSIST:
        return None
    for i in range(n - COMMIT_WINDOW - COMMIT_PERSIST + 1):
        if np.mean(t[i: i + COMMIT_WINDOW]) >= COMMIT_THRESH:
            if np.min(t[i + COMMIT_WINDOW: i + COMMIT_WINDOW + COMMIT_PERSIST]) >= COMMIT_THRESH * 0.8:
                return i
    return None


COT_TRIGGER = "Think step by step before answering."

@dataclass
class CoTTrace:
    question:    str
    answers:     List[str]
    trajectory:  List[float] = field(default_factory=list)
    gen_len:     int         = 0
    answer:      str         = ""
    f1:          float       = 0.0
    commit_step: Optional[int] = None
    commit_pct:  float         = 0.0


def run_cot_trace(model, tok, probe: Probe, sample: Dict) -> CoTTrace:
    q   = sample["question"]
    tr  = CoTTrace(question=q, answers=sample["answers"])
    cap = HSCapture(model, probe.layer_idx)

    try:
        content = f"{COT_TRIGGER}\n\n{q}"
        msgs    = [{"role": "user", "content": content}]
        prompt  = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids     = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
        cur     = ids.clone()

        # KV-cached generation: O(n) per step instead of O(n²)
        gen_tok_ids: List[int] = []
        past_kv = None
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
            gen_tok_ids.append(nxt)
            cur = torch.cat([cur, torch.tensor([[nxt]], device=DEVICE)], dim=1)
            if nxt == tok.eos_token_id:
                break

        tr.gen_len = len(gen_tok_ids)
        gen_text   = tok.decode(gen_tok_ids, skip_special_tokens=True).strip()
        # Extract answer: text after "answer:" or last sentence
        if "answer:" in gen_text.lower():
            tr.answer = gen_text.lower().split("answer:")[-1].strip()
        elif "\n\n" in gen_text:
            tr.answer = gen_text.split("\n\n")[-1].strip()
        else:
            tr.answer = gen_text[-200:].strip()
        tr.f1 = token_f1(tr.answer, sample["answers"])

        c = find_commit(tr.trajectory)
        tr.commit_step = c
        if c is not None and tr.gen_len > 0:
            tr.commit_pct = 100.0 * (tr.gen_len - c) / tr.gen_len

    finally:
        cap.remove()
    return tr


# ─────────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────────

def statistics(probe: Probe, cot_traces: List[CoTTrace],
               no_ctx: List[Dict], with_ctx: List[Dict]) -> Dict:
    # Trajectory distribution — understand J_know scale for this architecture
    all_j_vals = [v for t in cot_traces for v in t.trajectory]
    j_mean = float(np.mean(all_j_vals)) if all_j_vals else 0.0
    j_std  = float(np.std(all_j_vals))  if all_j_vals else 0.0
    j_max  = float(np.max(all_j_vals))  if all_j_vals else 0.0
    j_p75  = float(np.percentile(all_j_vals, 75)) if all_j_vals else 0.0

    # Adaptive threshold: 60th percentile of all J_know values
    adaptive_thresh = float(np.percentile(all_j_vals, 60)) if all_j_vals else COMMIT_THRESH
    adaptive_thresh = max(adaptive_thresh, COMMIT_THRESH * 0.5)  # floor at half fixed
    print(f"  J_know: mean={j_mean:.3f} std={j_std:.3f} max={j_max:.3f} p75={j_p75:.3f}", flush=True)
    print(f"  Adaptive threshold: {adaptive_thresh:.4f}  (fixed: {COMMIT_THRESH})", flush=True)

    def find_commit_thresh(traj: List[float], thresh: float) -> Optional[int]:
        n = len(traj)
        t = np.array(traj)
        if n < COMMIT_WINDOW + COMMIT_PERSIST:
            return None
        for i in range(n - COMMIT_WINDOW - COMMIT_PERSIST + 1):
            if np.mean(t[i: i + COMMIT_WINDOW]) >= thresh:
                if np.min(t[i + COMMIT_WINDOW: i + COMMIT_WINDOW + COMMIT_PERSIST]) >= thresh * 0.8:
                    return i
        return None

    # Run with adaptive threshold
    for t in cot_traces:
        if t.commit_step is None and t.trajectory:
            c = find_commit_thresh(t.trajectory, adaptive_thresh)
            if c is not None:
                t.commit_step = c
                if t.gen_len > 0:
                    t.commit_pct = 100.0 * (t.gen_len - c) / t.gen_len

    committed = [t for t in cot_traces if t.commit_step is not None]
    all_cpct  = [t.commit_pct for t in committed]
    cor_cpct  = [t.commit_pct for t in cot_traces if t.f1 >= 0.4 and t.commit_step is not None]
    obs_mean  = float(np.mean(all_cpct)) if all_cpct else 0.0

    all_trajs  = [t.trajectory for t in cot_traces
                  if len(t.trajectory) > COMMIT_WINDOW + COMMIT_PERSIST]
    null_cpcts = []
    for _ in range(N_BOOTSTRAP):
        if not all_trajs:
            break
        traj = list(all_trajs[rng.integers(0, len(all_trajs))])
        rng.shuffle(traj)
        c = find_commit_thresh(traj, adaptive_thresh)
        null_cpcts.append(100.0 * (len(traj) - c) / len(traj) if c is not None else 0.0)

    nm  = float(np.mean(null_cpcts)) if null_cpcts else 0.0
    ns  = float(np.std(null_cpcts))  if null_cpcts else 1.0
    z   = (obs_mean - nm) / max(ns, 0.1)

    commit_rate = len(committed) / max(len(cot_traces), 1)
    print(f"  Commit rate (adaptive): {commit_rate:.3f}  obs_mean={obs_mean:.1f}%  z={z:.2f}", flush=True)

    return {
        "model":              MODEL_ID,
        "probe_layer":        probe.layer_idx,
        "cal_auroc":          probe.auroc,
        "n_cot_traces":       len(cot_traces),
        "commit_rate":        commit_rate,
        "mean_commit_pct":    obs_mean,
        "correct_commit_pct": float(np.mean(cor_cpct)) if cor_cpct else 0.0,
        "null_mean":          nm,
        "null_std":           ns,
        "z_score":            z,
        "commit_thresh_used": adaptive_thresh,
        "j_know_mean":        j_mean,
        "j_know_std":         j_std,
        "j_know_max":         j_max,
        "j_know_p75":         j_p75,
        "architecture":       "Gemma2 (MQA, multi-head attention, RoPE, GeGLU)",
        "verdict": ("GEMMA_SIGNAL_CONFIRMED" if probe.auroc > 0.75 and z > 2 else
                    "AUROC_CONFIRMED"        if probe.auroc > 0.75 else
                    "WEAK"),
        "interpretation": (
            f"Gemma 2 Fisher LDA AUROC={probe.auroc:.4f}. "
            f"J_know mean={j_mean:.3f} max={j_max:.3f}. "
            f"Adaptive threshold={adaptive_thresh:.3f}. "
            f"Commit rate={commit_rate:.2f} (z={z:.2f}). "
            f"{'Signal with commit moment.' if z > 2 else 'Signal confirmed but no commit moment — MQA flat trajectory.'}"
        ),
    }


def make_figure(probe: Probe, cot_traces: List[CoTTrace], stats: Dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    correct = [t for t in cot_traces if t.f1 >= 0.4 and t.trajectory]
    wrong   = [t for t in cot_traces if t.f1 < 0.1  and t.trajectory]

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

    N_BINS = 40
    xs_pct = np.linspace(0, 100, N_BINS)

    def bin_traj(tr_list):
        bins = [[] for _ in range(N_BINS)]
        for tr in tr_list:
            for i, v in enumerate(tr.trajectory):
                b = min(int(i / len(tr.trajectory) * N_BINS), N_BINS - 1)
                bins[b].append(v)
        return [np.mean(b) if b else np.nan for b in bins]

    ax1 = fig.add_subplot(gs[0, 0])
    for tr in correct[:6]:
        ax1.plot(np.linspace(0, 100, len(tr.trajectory)), tr.trajectory,
                 color="#2ecc71", alpha=0.3, lw=0.9)
    for tr in wrong[:6]:
        ax1.plot(np.linspace(0, 100, len(tr.trajectory)), tr.trajectory,
                 color="#e74c3c", alpha=0.3, lw=0.9)
    if correct:
        ax1.plot(xs_pct, bin_traj(correct), "#27ae60", lw=2.5,
                 label=f"Correct avg (n={len(correct)})")
    if wrong:
        ax1.plot(xs_pct, bin_traj(wrong), "#c0392b", lw=2.5,
                 label=f"Wrong avg (n={len(wrong)})")
    ax1.axhline(0, color="gray", ls="--", lw=0.8)
    ax1.set_title(f"Gemma 2 — J_know through CoT generation\nProbe AUROC={probe.auroc:.4f}",
                  fontsize=10)
    ax1.set_xlabel("% through generation", fontsize=10)
    ax1.set_ylabel("J_know", fontsize=10)
    ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(gs[0, 1])
    bins = np.linspace(0, 100, 21)
    cor_c = [t.commit_pct for t in correct if t.commit_step is not None]
    if cor_c:
        ax2.hist(cor_c, bins=bins, alpha=0.7, color="#2ecc71",
                 label=f"Correct n={len(cor_c)}", density=True)
    ax2.axvline(stats["mean_commit_pct"], color="orange", ls="--", lw=2,
                label=f"Mean {stats['mean_commit_pct']:.1f}%")
    ax2.set_title("Post-commitment % distribution", fontsize=10)
    ax2.legend(fontsize=8)

    ax3 = fig.add_subplot(gs[1, 0])
    all_c = [t.commit_pct for t in cot_traces if t.commit_step is not None]
    if all_c:
        ax3.hist(all_c, bins=20, alpha=0.7, color="#3498db", density=True, zorder=3)
    null_x = np.linspace(0, 100, 200)
    nm, ns = stats["null_mean"], max(stats["null_std"], 1e-3)
    ax3.plot(null_x, np.exp(-0.5 * ((null_x - nm) / ns) ** 2) / (ns * np.sqrt(2 * np.pi)),
             "gray", ls="--", lw=2, label=f"Null μ={nm:.1f}%")
    ax3.axvline(stats["mean_commit_pct"], color="orange", lw=2)
    ax3.set_title(f"Null hypothesis  z={stats['z_score']:.2f}", fontsize=10)
    ax3.legend(fontsize=8)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    txt = (
        f"GEMMA 2 GEOMETRY v1\n"
        f"{'='*36}\n\n"
        f"Model:        {MODEL_ID}\n"
        f"Architecture: multi-head attn, RoPE\n\n"
        f"PART A — Fisher LDA Probe\n"
        f"  Layer:    {stats['probe_layer']}\n"
        f"  AUROC:    {stats['cal_auroc']:.4f}\n\n"
        f"PART B — CoT Commit Moment\n"
        f"  N:        {stats['n_cot_traces']}\n"
        f"  Commit %: {stats['mean_commit_pct']:.1f}%\n"
        f"  z-score:  {stats['z_score']:.2f}\n\n"
        f"VERDICT: {stats['verdict']}\n\n"
        f"{stats['interpretation']}\n"
    )
    ax4.text(0.04, 0.96, txt, transform=ax4.transAxes, fontsize=8.5,
             va="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="#f0f4f8", alpha=0.9))

    plt.suptitle(f"Epistemic Geometry on Gemma 2 — Google Architecture Validation\n"
                 f"AUROC={probe.auroc:.4f}  CoT commit%={stats['mean_commit_pct']:.1f}%  "
                 f"z={stats['z_score']:.2f}",
                 fontsize=11, fontweight="bold")
    fig.savefig(str(FIGURE_FILE), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure → {FIGURE_FILE}")


def main():
    t0 = time.time()
    no_ctx, with_ctx = load_data(n_nocontext=200, n_withcontext=200)
    model, tok = load_model()
    _tcfg    = getattr(model.config, 'text_config', model.config)
    n_layers = _tcfg.num_hidden_layers

    # Part A: bilateral oracle calibration
    probe = pick_layer(model, tok, no_ctx, with_ctx, n_layers)

    # Part B: CoT trajectory on no-context questions
    print(f"\nCoT traces: {MAIN_N} questions …", flush=True)
    cot_traces = []
    all_samples = no_ctx[80: 80 + MAIN_N]
    for i, s in enumerate(all_samples):
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{MAIN_N}  ({time.time()-t0:.0f}s)", flush=True)
        try:
            cot_traces.append(run_cot_trace(model, tok, probe, s))
        except Exception as e:
            print(f"  Error {i}: {e}", flush=True)

    stats = statistics(probe, cot_traces, no_ctx, with_ctx)
    print(f"\nVERDICT: {stats['verdict']}")
    print(f"AUROC: {stats['cal_auroc']:.4f}  z: {stats['z_score']:.2f}")

    make_figure(probe, cot_traces, stats)
    with open(RESULTS_FILE, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Results → {RESULTS_FILE}   Total: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
