"""
experiments/head_attribution/head_attribution.py

TASK 4.1 — ATTENTION HEAD ATTRIBUTION ANALYSIS
===============================================

Central question: Which attention heads at L26 (and L20-L27) show the
strongest separation between PARAM and CTX_DEP classes at step-1?

Method:
  1. Bilateral oracle labeling (same as C3-v3)
  2. Hook each head's output separately at layers L20-L27
  3. Score each head by Bhattacharyya distance between class-conditional
     mean activation distributions
  4. Fisher score as secondary ranking metric
  5. Output: ranked head list + top-K head indices for Task 4.2

Why Bhattacharyya distance:
  - Measures distributional overlap, not just means
  - Head outputs are ~512-dim vectors; Bhattacharyya compresses to scalar
  - Robust to outlier heads that fire rarely

Output: head_attribution_results.json
  - ranked_heads: [{layer, head, bhatt_dist, fisher_score, n_param, n_ctx_dep}]
  - top_k_heads: [{layer, head}] (K=10 for Task 4.2)

Model: Qwen/Qwen2.5-1.5B-Instruct
"""

from __future__ import annotations

import json
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from datasets import load_dataset
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID   = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# Layers to analyse (deep layers, based on layer_sweep showing signal peaks L20-L28)
EVAL_LAYERS = list(range(20, 28))   # L20..L27 inclusive

N_TARGET    = 80    # per class (fast run)
POOL_SIZE   = 2000
PARAM_MIN   = 0.50
CTX_MIN_NC  = 0.05
CTX_MIN_CTX = 0.50
TOP_K       = 10    # heads to pass to Task 4.2

RESULTS_FILE = "head_attribution_results.json"

# ── Helpers ───────────────────────────────────────────────────────────────────
def token_f1(pred: str, gold: str) -> float:
    p = set(pred.lower().split()); g = set(gold.lower().split())
    if not p or not g: return 0.0
    prec = len(p & g) / len(p); rec = len(p & g) / len(g)
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0

def best_f1(pred: str, answers: list[str]) -> float:
    return max(token_f1(pred, a) for a in answers) if answers else 0.0

def fmt_nocontext(q: str) -> str:
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
    )

def fmt_withcontext(q: str, ctx: str) -> str:
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\nBackground: {ctx}\n\n{q}<|im_end|>\n<|im_start|>assistant\n"
    )


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model():
    print(f"Loading {MODEL_ID}...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, trust_remote_code=True
    ).to(DEVICE).eval()
    return tok, mdl


# ── Multi-layer head extraction ───────────────────────────────────────────────
def get_head_activations(tok, mdl, prompt: str) -> dict[int, np.ndarray]:
    """
    Returns {layer_idx: head_activations} where head_activations has shape
    (n_heads, head_dim). Captured at step-1 (first generated token).
    """
    head_acts: dict[int, Optional[np.ndarray]] = {l: None for l in EVAL_LAYERS}
    step_ctrs = {l: [0] for l in EVAL_LAYERS}

    config = mdl.config
    n_heads  = config.num_attention_heads
    head_dim = config.hidden_size // n_heads

    handles = []
    for l_idx in EVAL_LAYERS:
        attn_module = mdl.model.layers[l_idx].self_attn

        def make_hook(layer):
            def hook_fn(module, inp_t, out):
                # out[0] is (batch, seq_len, hidden_size) — the attention output
                hs = out[0] if isinstance(out, tuple) else out
                if hs.shape[1] != 1:  # prompt pass
                    return
                if step_ctrs[layer][0] == 0:
                    # Reshape to (n_heads, head_dim)
                    flat = hs[0, 0, :].detach().float().cpu().numpy()
                    head_acts[layer] = flat.reshape(n_heads, head_dim)
                step_ctrs[layer][0] += 1
            return hook_fn

        h = attn_module.register_forward_hook(make_hook(l_idx))
        handles.append(h)

    inputs = tok(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        mdl.generate(
            **inputs,
            max_new_tokens=2,
            do_sample=False,
            use_cache=True,
            pad_token_id=tok.pad_token_id,
        )

    for h in handles:
        h.remove()

    return head_acts


# ── Oracle labeling ───────────────────────────────────────────────────────────
def build_oracle_pool(tok, mdl, dataset_iter):
    param_items, ctx_dep_items = [], []
    pool_seen = 0

    for ex in dataset_iter:
        if pool_seen >= POOL_SIZE:
            break
        pool_seen += 1

        q = ex["question"]
        answers = ex["answer"]["aliases"] if "aliases" in ex["answer"] else [ex["answer"]["value"]]
        ctx_parts = ex.get("entity_pages", {}).get("wiki_context", [])
        ctx = ctx_parts[0][:800] if ctx_parts else ""
        if not ctx:
            continue

        nc_prompt = fmt_nocontext(q)
        nc_input  = tok(nc_prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            nc_out = mdl.generate(
                **nc_input, max_new_tokens=30, do_sample=False, pad_token_id=tok.pad_token_id
            )
        nc_text = tok.decode(nc_out[0, nc_input["input_ids"].shape[1]:], skip_special_tokens=True)
        nc_f1   = best_f1(nc_text, answers)

        # Oracle check first — skip HS extraction if item will be skipped
        if nc_f1 < PARAM_MIN and not (nc_f1 <= CTX_MIN_NC):
            continue

        if nc_f1 <= CTX_MIN_NC:
            wc_prompt = fmt_withcontext(q, ctx)
            wc_input  = tok(wc_prompt, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                wc_out = mdl.generate(
                    **wc_input, max_new_tokens=30, do_sample=False, pad_token_id=tok.pad_token_id
                )
            wc_text = tok.decode(wc_out[0, wc_input["input_ids"].shape[1]:], skip_special_tokens=True)
            wc_f1   = best_f1(wc_text, answers)
            if wc_f1 < CTX_MIN_CTX:
                continue
            label = "CTX_DEP"
        else:
            label = "PARAM"

        # Extract head activations from nocontext pass
        head_acts = get_head_activations(tok, mdl, nc_prompt)

        # Skip if any layer extraction failed
        if any(v is None for v in head_acts.values()):
            continue

        item = {"head_acts": head_acts, "nc_f1": nc_f1, "label": label}

        if label == "PARAM":
            param_items.append(item)
        else:
            ctx_dep_items.append(item)

        n_p = len(param_items); n_c = len(ctx_dep_items)
        if (n_p + n_c) % 10 == 0:
            print(f"  pool={pool_seen} PARAM={n_p} CTX_DEP={n_c}")

        if n_p >= N_TARGET and n_c >= N_TARGET:
            break

    return param_items[:N_TARGET], ctx_dep_items[:N_TARGET]


# ── Head scoring ──────────────────────────────────────────────────────────────
def bhattacharyya_distance(mu1: np.ndarray, cov1: np.ndarray,
                           mu2: np.ndarray, cov2: np.ndarray) -> float:
    """Bhattacharyya distance between two Gaussians."""
    cov_avg = (cov1 + cov2) / 2.0
    delta   = mu2 - mu1
    try:
        inv_cov = np.linalg.pinv(cov_avg)
        mahal   = 0.125 * float(delta @ inv_cov @ delta)
        sign1, ld1 = np.linalg.slogdet(cov_avg)
        sign2, ld2 = np.linalg.slogdet(cov1)
        sign3, ld3 = np.linalg.slogdet(cov2)
        if sign1 <= 0 or sign2 <= 0 or sign3 <= 0:
            return mahal  # fallback: Mahalanobis only
        det_term = 0.5 * (ld1 - 0.5 * (ld2 + ld3))
        return mahal + det_term
    except Exception:
        return 0.0


def score_head(param_hs: list[np.ndarray], ctx_dep_hs: list[np.ndarray]) -> dict:
    """Score a single head (each hs is head_dim vector)."""
    P = np.vstack(param_hs)
    C = np.vstack(ctx_dep_hs)

    mu_p = P.mean(axis=0); mu_c = C.mean(axis=0)
    cov_p = np.cov(P, rowvar=False) + 1e-6 * np.eye(P.shape[1])
    cov_c = np.cov(C, rowvar=False) + 1e-6 * np.eye(C.shape[1])

    bhatt = bhattacharyya_distance(mu_p, cov_p, mu_c, cov_c)

    # Fisher score: between-class / within-class variance (trace-ratio)
    between = np.outer(mu_p - mu_c, mu_p - mu_c)
    within  = cov_p + cov_c
    try:
        inv_w  = np.linalg.pinv(within)
        fisher = float(np.trace(inv_w @ between))
    except Exception:
        fisher = 0.0

    # LDA AUROC (1D)
    X = np.vstack([P, C])
    y = np.array([1] * len(P) + [0] * len(C))
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    try:
        lda.fit(X, y)
        prob = lda.predict_proba(X)[:, 1]
        auroc = float(roc_auc_score(y, prob))
    except Exception:
        auroc = 0.5

    return {"bhatt_dist": float(bhatt), "fisher_score": float(fisher), "lda_auroc": auroc}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    tok, mdl = load_model()
    n_heads  = mdl.config.num_attention_heads

    print("Loading TriviaQA...")
    ds = load_dataset(
        "trivia_qa", "rc.wikipedia", split="validation",
        streaming=True, trust_remote_code=True
    )

    print(f"Building oracle pool (target {N_TARGET}/class)...")
    param_items, ctx_dep_items = build_oracle_pool(tok, mdl, iter(ds))
    n_p = len(param_items); n_c = len(ctx_dep_items)
    n   = min(n_p, n_c)
    print(f"Pool complete: PARAM={n_p} CTX_DEP={n_c}")

    # ── Score every head at every eval layer ──────────────────────────────────
    ranked_heads = []
    for l_idx in EVAL_LAYERS:
        print(f"\nScoring heads at L{l_idx:02d}...")
        for h_idx in range(n_heads):
            p_vecs = [item["head_acts"][l_idx][h_idx] for item in param_items[:n]]
            c_vecs = [item["head_acts"][l_idx][h_idx] for item in ctx_dep_items[:n]]
            scores = score_head(p_vecs, c_vecs)
            ranked_heads.append({
                "layer":        l_idx,
                "head":         h_idx,
                "bhatt_dist":   scores["bhatt_dist"],
                "fisher_score": scores["fisher_score"],
                "lda_auroc":    scores["lda_auroc"],
            })
            if h_idx % 4 == 0:
                print(f"  h{h_idx:02d} bhatt={scores['bhatt_dist']:.3f} auroc={scores['lda_auroc']:.3f}")

    # Sort by Bhattacharyya distance (primary) then LDA AUROC (secondary)
    ranked_heads.sort(key=lambda x: (-x["bhatt_dist"], -x["lda_auroc"]))
    top_k = [{"layer": h["layer"], "head": h["head"]} for h in ranked_heads[:TOP_K]]

    print(f"\n{'='*60}")
    print(f"TOP {TOP_K} DISCRIMINATING HEADS:")
    for i, h in enumerate(ranked_heads[:TOP_K]):
        print(f"  {i+1:2d}. L{h['layer']:02d} H{h['head']:02d} "
              f"bhatt={h['bhatt_dist']:.4f} auroc={h['lda_auroc']:.4f}")
    print(f"{'='*60}")

    results = {
        "model":        MODEL_ID,
        "eval_layers":  EVAL_LAYERS,
        "n_heads":      n_heads,
        "n_param":      n_p,
        "n_ctx_dep":    n_c,
        "ranked_heads": ranked_heads,
        "top_k_heads":  top_k,
        "top_k":        TOP_K,
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results → {RESULTS_FILE}")
    print(f"Top-K heads saved. Pass head_attribution_results.json to Task 4.2.")


if __name__ == "__main__":
    main()
