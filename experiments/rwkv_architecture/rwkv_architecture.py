"""
experiments/rwkv_architecture/rwkv_architecture.py

TASK 5.2 — RWKV (LINEAR ATTENTION / RNN-STYLE) ARCHITECTURE TEST
=================================================================

Central question: Does the bilateral oracle signal (AUROC ≥ 0.82) survive
in RWKV, a linear-attention / recurrent hybrid architecture?

Background:
  - RWKV replaces softmax attention with a linear recurrence (WKV operator).
  - In "RNN mode" each token position produces a state; in generation mode
    the state is maintained as a running summary.
  - Architecturally distinct from both Transformers AND Mamba.
  - If HOLDS across SSM + RWKV + Transformer → signal is truly arch-agnostic.

Models:
  - RWKV/v6-Finch-1B6-HF (1.6B, HF-compatible, T4-safe)
  - Fallback: RWKV/v5-Eagle-7B is too large; use BlinkDL/rwkv-4-world-1.5B

Protocol: same bilateral oracle + Fisher+PCA64 as C3-v3
Output: rwkv_architecture_results.json
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import torch
from datasets import load_dataset
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_CANDIDATES = [
    "RWKV/v6-Finch-1B6-HF",
    "BlinkDL/rwkv-4-world-1.5b",
]
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
N_TARGET     = 80
POOL_SIZE    = 2000
PARAM_MIN    = 0.50
CTX_MIN_NC   = 0.05
CTX_MIN_CTX  = 0.50
N_BOOTSTRAP  = 500
RESULTS_FILE = "rwkv_architecture_results.json"


# ── Helpers ───────────────────────────────────────────────────────────────────
def token_f1(pred: str, gold: str) -> float:
    p = set(pred.lower().split()); g = set(gold.lower().split())
    if not p or not g: return 0.0
    prec = len(p & g) / len(p); rec = len(p & g) / len(g)
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0

def best_f1(pred: str, answers: list[str]) -> float:
    return max(token_f1(pred, a) for a in answers) if answers else 0.0

def fmt_prompt(q: str, ctx: str = None) -> str:
    if ctx:
        return f"Background: {ctx}\n\nQuestion: {q}\nAnswer:"
    return f"Question: {q}\nAnswer:"


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model():
    for model_id in MODEL_CANDIDATES:
        try:
            print(f"Trying {model_id}...")
            tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            if tok.pad_token_id is None:
                tok.pad_token_id = tok.eos_token_id or 0
            mdl = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.float16, trust_remote_code=True
            ).to(DEVICE).eval()
            print(f"Loaded {model_id}")
            return tok, mdl, model_id
        except Exception as e:
            print(f"  Failed ({type(e).__name__}: {e})")
    raise RuntimeError("No RWKV model loaded")


# ── HS extraction ─────────────────────────────────────────────────────────────
def _find_last_block(mdl):
    """Return the last RWKV block regardless of architecture variant."""
    if hasattr(mdl, "rwkv") and hasattr(mdl.rwkv, "blocks"):
        return mdl.rwkv.blocks[-1]
    if hasattr(mdl, "model") and hasattr(mdl.model, "blocks"):
        return mdl.model.blocks[-1]
    if hasattr(mdl, "blocks"):
        return mdl.blocks[-1]
    # TransformerLens-style
    for name, module in reversed(list(mdl.named_modules())):
        if "block" in name.lower() or "layer" in name.lower():
            return module
    raise AttributeError("Cannot find RWKV blocks")


def get_last_layer_hs(tok, mdl, prompt: str) -> np.ndarray:
    captured = [None]
    step_ctr = [0]

    last_block = _find_last_block(mdl)

    def hook_fn(module, inp_t, out):
        hs = out[0] if isinstance(out, tuple) else out
        if isinstance(hs, torch.Tensor) and hs.ndim == 3 and hs.shape[1] == 1:
            if step_ctr[0] == 0:
                captured[0] = hs[0, 0, :].detach().float().cpu().numpy()
            step_ctr[0] += 1

    handle = last_block.register_forward_hook(hook_fn)
    inputs  = tok(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        mdl.generate(
            **inputs, max_new_tokens=2, do_sample=False,
            pad_token_id=tok.pad_token_id
        )
    handle.remove()
    return captured[0]


# ── Oracle labeling ───────────────────────────────────────────────────────────
def build_oracle_pool(tok, mdl, dataset_iter):
    param_items, ctx_dep_items = [], []
    pool_seen = 0

    for ex in dataset_iter:
        if pool_seen >= POOL_SIZE: break
        pool_seen += 1

        q = ex["question"]
        answers = ex["answer"]["aliases"] if "aliases" in ex["answer"] else [ex["answer"]["value"]]
        ctx_parts = ex.get("entity_pages", {}).get("wiki_context", [])
        ctx = ctx_parts[0][:800] if ctx_parts else ""
        if not ctx: continue

        nc_prompt = fmt_prompt(q)
        nc_input  = tok(nc_prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            nc_out = mdl.generate(**nc_input, max_new_tokens=30, do_sample=False,
                                  pad_token_id=tok.pad_token_id)
        nc_text = tok.decode(nc_out[0, nc_input["input_ids"].shape[1]:], skip_special_tokens=True)
        nc_f1   = best_f1(nc_text, answers)

        if nc_f1 >= PARAM_MIN:
            label = "PARAM"
        elif nc_f1 <= CTX_MIN_NC:
            wc_prompt = fmt_prompt(q, ctx)
            wc_input  = tok(wc_prompt, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                wc_out = mdl.generate(**wc_input, max_new_tokens=30, do_sample=False,
                                      pad_token_id=tok.pad_token_id)
            wc_text = tok.decode(wc_out[0, wc_input["input_ids"].shape[1]:], skip_special_tokens=True)
            if best_f1(wc_text, answers) < CTX_MIN_CTX: continue
            label = "CTX_DEP"
        else:
            continue

        hs = get_last_layer_hs(tok, mdl, nc_prompt)
        if hs is None: continue

        item = {"hs": hs, "nc_f1": nc_f1}
        if label == "PARAM": param_items.append(item)
        else:                ctx_dep_items.append(item)

        n_p = len(param_items); n_c = len(ctx_dep_items)
        if (n_p + n_c) % 10 == 0:
            print(f"  pool={pool_seen} PARAM={n_p} CTX_DEP={n_c}")
        if n_p >= N_TARGET and n_c >= N_TARGET:
            break

    return param_items[:N_TARGET], ctx_dep_items[:N_TARGET]


# ── Probe ─────────────────────────────────────────────────────────────────────
def fit_fisher_pca64(X, y):
    n_comp = min(64, X.shape[1], X.shape[0] - 2)
    pca = PCA(n_components=n_comp, random_state=42)
    X_r = pca.fit_transform(X)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for tr, va in skf.split(X_r, y):
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        lda.fit(X_r[tr], y[tr])
        prob = lda.predict_proba(X_r[va])[:, 1]
        scores.append(roc_auc_score(y[va], prob))
    mean_auroc = float(np.mean(scores))

    full_lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    full_lda.fit(X_r, y)
    full_prob = full_lda.predict_proba(X_r)[:, 1]
    rng = np.random.default_rng(42)
    boot = [
        roc_auc_score(y[idx := rng.integers(0, len(y), len(y))], full_prob[idx])
        for _ in range(N_BOOTSTRAP)
        if len(np.unique(y[rng.integers(0, len(y), len(y))])) > 1
    ]
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5]) if boot else (0.0, 0.0)

    y_s = rng.permutation(y)
    shuf = []
    for tr, va in skf.split(X_r, y_s):
        l = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        l.fit(X_r[tr], y_s[tr])
        shuf.append(roc_auc_score(y_s[va], l.predict_proba(X_r[va])[:, 1]))

    return {
        "auroc": mean_auroc, "ci_lo": float(ci_lo), "ci_hi": float(ci_hi),
        "shuffled": float(np.mean(shuf)), "n": int(len(y)),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    tok, mdl, model_id = load_model()

    print("Loading TriviaQA...")
    ds = load_dataset("trivia_qa", "rc.wikipedia", split="validation",
                      streaming=True, trust_remote_code=True)

    print(f"Building oracle pool (target {N_TARGET}/class)...")
    param_items, ctx_dep_items = build_oracle_pool(tok, mdl, iter(ds))
    n_p = len(param_items); n_c = len(ctx_dep_items)
    n   = min(n_p, n_c)
    print(f"Pool: PARAM={n_p} CTX_DEP={n_c}")

    if n < 20:
        results = {"model_id": model_id, "verdict": "INSUFFICIENT_DATA",
                   "n_param": n_p, "n_ctx_dep": n_c}
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)
        print("INSUFFICIENT_DATA")
        return

    P = np.vstack([it["hs"] for it in param_items[:n]])
    C = np.vstack([it["hs"] for it in ctx_dep_items[:n]])
    X = np.vstack([P, C])
    y = np.array([1] * n + [0] * n)

    print("Fitting probe...")
    probe = fit_fisher_pca64(X, y)
    auroc, shuffled = probe["auroc"], probe["shuffled"]
    print(f"AUROC={auroc:.4f} [{probe['ci_lo']:.3f}-{probe['ci_hi']:.3f}] shuffled={shuffled:.4f}")

    baseline = 0.841
    if shuffled >= 0.70:
        verdict = "PROBE_DEGENERATE"
    elif auroc >= baseline - 0.05:
        verdict = "HOLDS"
    elif auroc >= 0.70:
        verdict = "PARTIAL"
    else:
        verdict = "ABSENT"

    print(f"\nVERDICT: {verdict}")

    results = {
        "model_id":      model_id,
        "architecture":  "RWKV/linear-attention",
        "n_param":       n_p, "n_ctx_dep": n_c,
        "probe":         probe,
        "baseline_qwen": baseline,
        "verdict":       verdict,
        "interpretation": {
            "HOLDS":   "RWKV hidden states encode bilateral oracle signal. Signal is arch-agnostic.",
            "PARTIAL": "Weaker signal in RWKV. Partial epistemic organization without full attention.",
            "ABSENT":  "RWKV does not encode bilateral oracle signal.",
            "PROBE_DEGENERATE": "Shuffled control too high — likely data sparsity issue.",
            "INSUFFICIENT_DATA": "Not enough labeled items.",
        }.get(verdict, ""),
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results → {RESULTS_FILE}")


if __name__ == "__main__":
    main()
