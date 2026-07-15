"""
experiments/mamba_architecture/mamba_architecture.py

TASK 5.1 — MAMBA (SSM) ARCHITECTURE TEST
=========================================

Central question: Does the bilateral oracle signal (AUROC ≥ 0.82) survive
in a State Space Model (Mamba) which has NO attention mechanism?

Background:
  - All prior confirmed results (C001-C004) are on Qwen/Llama: Transformer
    architectures with multi-head attention.
  - Mamba replaces attention with selective state spaces (S6 operator).
  - Hidden states are conceptually different: no key-value mechanism,
    state is a running summary of prior tokens.
  - If signal holds → epistemic organization is not attention-specific.
  - If signal fails → attention is mechanistically required.

Protocol:
  - Bilateral oracle: same thresholds as C3-v3 (PARAM≥0.50, CTX_DEP≤0.05/≥0.50)
  - Hidden state extraction: hook the mixer output at last Mamba layer
  - Fisher+PCA64 probe (same as C001)
  - Shuffled control included

Model: state-spaces/mamba-2.8b-hf  (fits T4 at float16: ~5.6 GB)
  OR fallback: state-spaces/mamba-370m-hf (~0.74 GB) if quota is tight

Dataset: TriviaQA rc.wikipedia (streaming)
Output: mamba_architecture_results.json
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
from transformers import AutoModelForCausalLM, AutoTokenizer, MambaConfig

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
# Try large model first; fallback to small if OOM
MODEL_CANDIDATES = [
    "state-spaces/mamba-2.8b-hf",
    "state-spaces/mamba-370m-hf",
]
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
N_TARGET    = 80
POOL_SIZE   = 2000
PARAM_MIN   = 0.50
CTX_MIN_NC  = 0.05
CTX_MIN_CTX = 0.50
N_BOOTSTRAP = 500
RESULTS_FILE = "mamba_architecture_results.json"


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
                tok.pad_token_id = tok.eos_token_id

            mdl = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                trust_remote_code=True,
            ).to(DEVICE).eval()
            print(f"Loaded {model_id}")
            return tok, mdl, model_id
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            print(f"  Failed ({e}), trying next...")
    raise RuntimeError("No Mamba model loaded successfully")


# ── HS extraction ─────────────────────────────────────────────────────────────
def get_last_layer_hs(tok, mdl, prompt: str) -> np.ndarray:
    """
    Mamba HF implementation: model.backbone.layers[-1] is the last mixer layer.
    We hook its output and capture the hidden state at step-1 (first generated token).
    """
    captured = [None]
    step_ctr = [0]

    # Find the last Mamba layer
    layers = mdl.backbone.layers if hasattr(mdl, "backbone") else mdl.model.layers
    last_layer = layers[-1]

    def hook_fn(module, inp_t, out):
        hs = out[0] if isinstance(out, tuple) else out
        if hs.shape[1] != 1:   # prompt pass
            return
        if step_ctr[0] == 0:
            captured[0] = hs[0, 0, :].detach().float().cpu().numpy()
        step_ctr[0] += 1

    handle = last_layer.register_forward_hook(hook_fn)
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
            wc_prompt = fmt_withcontext(q, ctx)
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

def fmt_withcontext(q: str, ctx: str) -> str:
    return f"Background: {ctx}\n\nQuestion: {q}\nAnswer:"


# ── Probe ─────────────────────────────────────────────────────────────────────
def fit_fisher_pca64(X: np.ndarray, y: np.ndarray):
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

    # Bootstrap CI
    full_lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    full_lda.fit(X_r, y)
    full_prob = full_lda.predict_proba(X_r)[:, 1]
    rng = np.random.default_rng(42)
    boot = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2: continue
        boot.append(roc_auc_score(y[idx], full_prob[idx]))
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])

    # Shuffled null
    y_shuf = rng.permutation(y)
    shuf = []
    for tr, va in skf.split(X_r, y_shuf):
        lda_s = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        lda_s.fit(X_r[tr], y_shuf[tr])
        p = lda_s.predict_proba(X_r[va])[:, 1]
        shuf.append(roc_auc_score(y_shuf[va], p))

    return {
        "auroc":    mean_auroc,
        "ci_lo":    float(ci_lo),
        "ci_hi":    float(ci_hi),
        "shuffled": float(np.mean(shuf)),
        "n":        int(len(y)),
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

    P = np.vstack([it["hs"] for it in param_items[:n]])
    C = np.vstack([it["hs"] for it in ctx_dep_items[:n]])
    X = np.vstack([P, C])
    y = np.array([1] * n + [0] * n)

    print("Fitting Fisher+PCA64...")
    probe = fit_fisher_pca64(X, y)
    print(f"AUROC={probe['auroc']:.4f} [{probe['ci_lo']:.3f}-{probe['ci_hi']:.3f}] "
          f"shuffled={probe['shuffled']:.4f}")

    baseline = 0.841  # C3-v3 Qwen
    auroc    = probe["auroc"]
    shuffled = probe["shuffled"]

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
        "model_id":        model_id,
        "architecture":    "SSM/Mamba",
        "n_param":         n_p,
        "n_ctx_dep":       n_c,
        "probe":           probe,
        "baseline_qwen":   baseline,
        "verdict":         verdict,
        "interpretation": {
            "HOLDS":            "SSM hidden states encode bilateral oracle signal as well as Transformers. Signal is arch-agnostic.",
            "PARTIAL":          "Weaker signal in SSMs. Some epistemic organization but attention-like heads may amplify it.",
            "ABSENT":           "SSMs do not encode bilateral oracle signal. Attention mechanism required for this geometry.",
            "PROBE_DEGENERATE": "Shuffled control too high — probe is degenerate. Check N and Fisher PCA config.",
        }.get(verdict, ""),
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results → {RESULTS_FILE}")


if __name__ == "__main__":
    main()
