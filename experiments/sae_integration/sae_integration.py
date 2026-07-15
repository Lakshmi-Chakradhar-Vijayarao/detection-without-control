"""
experiments/sae_integration/sae_integration.py

TASK 4.3 — SAE FEATURE INTEGRATION
====================================

Central question: Do PARAM and CTX_DEP items activate distinct sets of
sparse autoencoder (SAE) features at step-1?

Strategy:
  Option A (primary): Use Gemma Scope SAEs on google/gemma-2-2b-it
    - Gemma Scope is the only production-quality SAE release for small instruct models
    - AUROC baseline will differ from Qwen (different architecture)
    - Tests: do SAE features cluster by bilateral oracle label?

  Option B (fallback): Approximate SAE with MLP neuron decomposition on Qwen
    - Hook MLP intermediate activations (post-activation) at L26
    - Each MLP neuron is an approximate "feature"
    - Sparse: many neurons near zero per token
    - Less clean than true SAE but available on any model

  This script runs Option A if gemma-2-2b-it is available, else Option B.

Measurements:
  1. Binary feature activation rate per class (PARAM vs CTX_DEP)
  2. Top discriminating features by chi-square test
  3. Feature AUROC (single-feature classifiers)
  4. SAE-reconstruction quality relative to Fisher+PCA64 AUROC

Output: sae_integration_results.json
"""

from __future__ import annotations

import json
import os
import warnings
from typing import Optional

import numpy as np
import torch
from datasets import load_dataset
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy.stats import chi2_contingency
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
# Option A: Gemma + Gemma Scope
GEMMA_MODEL_ID = "google/gemma-2-2b-it"
GEMMA_SAE_ID   = "google/gemma-scope-2b-pt-res"  # residual stream SAE

# Option B: Qwen MLP fallback
QWEN_MODEL_ID  = "Qwen/Qwen2.5-1.5B-Instruct"

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
LAYER_IDX   = 26
N_TARGET    = 60
POOL_SIZE   = 2000
PARAM_MIN   = 0.50
CTX_MIN_NC  = 0.05
CTX_MIN_CTX = 0.50

# Feature activation threshold (below this → "inactive")
ACTIVATION_THRESHOLD = 0.01
TOP_FEATURES_K       = 50   # report top-K discriminating features

RESULTS_FILE = "/kaggle/working/sae_integration_results.json"


# ── Helpers ───────────────────────────────────────────────────────────────────
def token_f1(pred: str, gold: str) -> float:
    p = set(pred.lower().split()); g = set(gold.lower().split())
    if not p or not g: return 0.0
    prec = len(p & g) / len(p); rec = len(p & g) / len(g)
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0

def best_f1(pred: str, answers: list[str]) -> float:
    return max(token_f1(pred, a) for a in answers) if answers else 0.0

def fmt_nc_qwen(q: str) -> str:
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
    )

def fmt_wc_qwen(q: str, ctx: str) -> str:
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\nBackground: {ctx}\n\n{q}<|im_end|>\n<|im_start|>assistant\n"
    )

def fmt_nc_gemma(q: str) -> str:
    return f"<start_of_turn>user\n{q}<end_of_turn>\n<start_of_turn>model\n"

def fmt_wc_gemma(q: str, ctx: str) -> str:
    return f"<start_of_turn>user\nBackground: {ctx}\n\n{q}<end_of_turn>\n<start_of_turn>model\n"


# ── Model loading ─────────────────────────────────────────────────────────────
def try_load_gemma():
    """Attempt to load Gemma-2 + Gemma Scope SAE. Returns (tok, mdl, sae) or None."""
    try:
        from transformer_lens import HookedTransformer
        from sae_lens import SAE
    except ImportError:
        print("sae_lens / transformer_lens not available — falling back to Option B")
        return None

    try:
        print(f"Loading Gemma Scope SAE: {GEMMA_SAE_ID}...")
        sae, cfg_dict, _ = SAE.from_pretrained(
            release=GEMMA_SAE_ID,
            sae_id="layer_20/width_16k/average_l0_71",   # layer 20, 16k features
        )
        sae = sae.to(DEVICE)

        print(f"Loading {GEMMA_MODEL_ID} via TransformerLens...")
        mdl = HookedTransformer.from_pretrained(GEMMA_MODEL_ID, dtype=torch.float16)
        mdl = mdl.to(DEVICE).eval()

        tok = AutoTokenizer.from_pretrained(GEMMA_MODEL_ID, trust_remote_code=True)
        return tok, mdl, sae
    except Exception as e:
        print(f"Gemma Scope load failed ({e}) — falling back to Option B")
        return None


def load_qwen_fallback():
    print(f"Loading {QWEN_MODEL_ID} (MLP neuron fallback)...")
    tok = AutoTokenizer.from_pretrained(QWEN_MODEL_ID, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    mdl = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_ID, torch_dtype=torch.float16, trust_remote_code=True
    ).to(DEVICE).eval()
    return tok, mdl


# ── Feature extraction (Option B: MLP neurons) ───────────────────────────────
def get_mlp_features_qwen(tok, mdl, prompt: str) -> Optional[np.ndarray]:
    """Extract MLP intermediate activations at L26 step-1."""
    captured = [None]
    step_ctr = [0]
    mlp = mdl.model.layers[LAYER_IDX].mlp

    def hook_fn(module, inp_t, out):
        if isinstance(out, tuple): out = out[0]
        if out.shape[1] != 1: return
        if step_ctr[0] == 0:
            # For SwiGLU-style MLPs, the gate activation is after gate_proj × up_proj
            # We hook the output of the activation function (post-act features)
            captured[0] = out[0, 0, :].detach().float().cpu().numpy()
        step_ctr[0] += 1

    # Hook the down_proj input (= post-activation intermediate features)
    handle = mlp.down_proj.register_forward_hook(
        lambda mod, inp, out: hook_fn(mod, inp, inp[0] if isinstance(inp, tuple) else inp)
    )
    inputs = tok(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        mdl.generate(**inputs, max_new_tokens=2, do_sample=False,
                     pad_token_id=tok.pad_token_id)
    handle.remove()
    return captured[0]


# ── Feature extraction (Option A: SAE features) ──────────────────────────────
def get_sae_features_gemma(tok, mdl, sae, prompt: str) -> Optional[np.ndarray]:
    """Extract SAE feature activations using TransformerLens + SAE Lens."""
    try:
        inputs = tok(prompt, return_tensors="pt").to(DEVICE)
        prompt_len = inputs["input_ids"].shape[1]

        # Generate one token then extract HS at that position
        with torch.no_grad():
            _, cache = mdl.run_with_cache(inputs["input_ids"])

        # Get residual stream at hook_resid_post for layer 20
        resid = cache["blocks.20.hook_resid_post"]   # (1, seq, d_model)
        # Take last prompt token (pre-generation) as proxy
        resid_vec = resid[0, -1, :].unsqueeze(0)   # (1, d_model)

        # Encode through SAE
        with torch.no_grad():
            feat_acts = sae.encode(resid_vec.to(sae.W_enc.dtype))   # (1, n_features)

        return feat_acts[0].detach().float().cpu().numpy()
    except Exception as e:
        return None


# ── Oracle labeling ───────────────────────────────────────────────────────────
def build_oracle_pool(tok, mdl, fmt_nc, fmt_wc, feat_fn, dataset_iter):
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

        nc_prompt = fmt_nc(q)
        nc_input  = tok(nc_prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            nc_out = mdl.generate(**nc_input, max_new_tokens=30, do_sample=False,
                                  pad_token_id=tok.pad_token_id) if hasattr(mdl, "generate") else \
                     mdl.generate(**nc_input, max_new_tokens=30, do_sample=False)
        nc_text = tok.decode(nc_out[0, nc_input["input_ids"].shape[1]:], skip_special_tokens=True)
        nc_f1   = best_f1(nc_text, answers)

        if nc_f1 >= PARAM_MIN:
            label = "PARAM"
        elif nc_f1 <= CTX_MIN_NC:
            wc_prompt = fmt_wc(q, ctx)
            wc_input  = tok(wc_prompt, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                wc_out = mdl.generate(**wc_input, max_new_tokens=30, do_sample=False,
                                      pad_token_id=tok.pad_token_id) if hasattr(mdl, "generate") else \
                         mdl.generate(**wc_input, max_new_tokens=30, do_sample=False)
            wc_text = tok.decode(wc_out[0, wc_input["input_ids"].shape[1]:], skip_special_tokens=True)
            if best_f1(wc_text, answers) < CTX_MIN_CTX: continue
            label = "CTX_DEP"
        else:
            continue

        feats = feat_fn(nc_prompt)
        if feats is None: continue

        item = {"features": feats, "nc_f1": nc_f1, "label": label}
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


# ── Feature analysis ──────────────────────────────────────────────────────────
def analyze_features(param_items, ctx_dep_items):
    n = min(len(param_items), len(ctx_dep_items))
    P = np.vstack([it["features"] for it in param_items[:n]])
    C = np.vstack([it["features"] for it in ctx_dep_items[:n]])
    X = np.vstack([P, C])
    y = np.array([1] * n + [0] * n)

    n_feats = X.shape[1]
    print(f"  Feature dimension: {n_feats}")

    # 1. Sparsity
    sparsity = float(np.mean(np.abs(X) < ACTIVATION_THRESHOLD))
    print(f"  Global sparsity: {sparsity:.3f}")

    # 2. Per-feature chi-square discrimination
    X_bin = (np.abs(X) > ACTIVATION_THRESHOLD).astype(int)
    chisq_stats = []
    for f_idx in range(n_feats):
        p_act = int(np.sum(X_bin[:n, f_idx]))
        c_act = int(np.sum(X_bin[n:, f_idx]))
        table = np.array([[p_act, n - p_act], [c_act, n - c_act]])
        try:
            chi2, pval, _, _ = chi2_contingency(table)
            chisq_stats.append({"feature": f_idx, "chi2": float(chi2), "pval": float(pval),
                                  "param_rate": p_act / n, "ctx_dep_rate": c_act / n})
        except Exception:
            chisq_stats.append({"feature": f_idx, "chi2": 0.0, "pval": 1.0,
                                  "param_rate": 0.0, "ctx_dep_rate": 0.0})

    chisq_stats.sort(key=lambda x: -x["chi2"])
    top_features = chisq_stats[:TOP_FEATURES_K]

    # 3. Ensemble Fisher+PCA64 AUROC using full feature vector
    n_comp = min(64, n_feats, 2 * n - 2)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_comp, random_state=42)
    X_r = pca.fit_transform(X)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aurocs = []
    for tr, va in skf.split(X_r, y):
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        lda.fit(X_r[tr], y[tr])
        prob = lda.predict_proba(X_r[va])[:, 1]
        aurocs.append(roc_auc_score(y[va], prob))
    ensemble_auroc = float(np.mean(aurocs))

    return {
        "n_features": n_feats,
        "sparsity":   sparsity,
        "top_features": top_features,
        "ensemble_auroc": ensemble_auroc,
        "n_per_class": n,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    mode = "A"
    gemma_loaded = try_load_gemma()

    if gemma_loaded is not None:
        tok, mdl, sae = gemma_loaded
        fmt_nc, fmt_wc = fmt_nc_gemma, fmt_wc_gemma
        feat_fn = lambda p: get_sae_features_gemma(tok, mdl, sae, p)
        model_used = GEMMA_MODEL_ID
        print("Mode A: Gemma Scope SAE features")
    else:
        mode = "B"
        tok, mdl = load_qwen_fallback()
        fmt_nc, fmt_wc = fmt_nc_qwen, fmt_wc_qwen
        feat_fn = lambda p: get_mlp_features_qwen(tok, mdl, p)
        model_used = QWEN_MODEL_ID
        print("Mode B: Qwen MLP neuron features (fallback)")

    print("Loading TriviaQA...")
    ds = load_dataset("trivia_qa", "rc.wikipedia", split="validation",
                      streaming=True, trust_remote_code=True)

    print(f"Building oracle pool (target {N_TARGET}/class)...")
    param_items, ctx_dep_items = build_oracle_pool(
        tok, mdl, fmt_nc, fmt_wc, feat_fn, iter(ds)
    )
    n_p = len(param_items); n_c = len(ctx_dep_items)
    print(f"Pool: PARAM={n_p} CTX_DEP={n_c}")

    print("\nAnalyzing feature distributions...")
    analysis = analyze_features(param_items, ctx_dep_items)

    ensemble_auroc = analysis["ensemble_auroc"]
    sparsity       = analysis["sparsity"]
    c3v3_baseline  = 0.841

    if ensemble_auroc >= c3v3_baseline - 0.03:
        verdict = "SAE_EQUIVALENT"       # SAE features match Fisher HS performance
    elif ensemble_auroc >= 0.70:
        verdict = "SAE_PARTIAL"          # some signal but weaker
    else:
        verdict = "SAE_INSUFFICIENT"     # features don't capture bilateral oracle signal

    print(f"\n{'='*60}")
    print(f"VERDICT: {verdict} (mode={mode})")
    print(f"Ensemble AUROC: {ensemble_auroc:.4f}  Baseline: {c3v3_baseline}")
    print(f"Sparsity: {sparsity:.3f}")
    print(f"\nTop 10 discriminating features:")
    for i, f in enumerate(analysis["top_features"][:10]):
        print(f"  {i+1:2d}. feat={f['feature']:5d} chi2={f['chi2']:.2f} "
              f"PARAM_rate={f['param_rate']:.3f} CTX_DEP_rate={f['ctx_dep_rate']:.3f}")
    print(f"{'='*60}")

    results = {
        "model_used":      model_used,
        "mode":            mode,
        "layer_idx":       LAYER_IDX,
        "n_param":         n_p,
        "n_ctx_dep":       n_c,
        "baseline_c3v3":   c3v3_baseline,
        "analysis":        analysis,
        "verdict":         verdict,
        "interpretation": {
            "SAE_EQUIVALENT":   "SAE/MLP features fully capture bilateral oracle signal. Hidden state geometry ≈ feature geometry.",
            "SAE_PARTIAL":      "Partial capture. Fisher on raw HS outperforms SAE features — geometry beyond sparse features.",
            "SAE_INSUFFICIENT": "SAE features don't separate PARAM/CTX_DEP. The signal may be in HS geometry, not sparse activations.",
        }.get(verdict, ""),
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results → {RESULTS_FILE}")


if __name__ == "__main__":
    main()
