#!/usr/bin/env python3
"""
experiments/borderline_geometry/borderline_geometry.py

Task 1.2 — Borderline Population Geometry
==========================================
CLAIM UNDER TEST: C010 (EXPLORATORY) + new hypothesis
  Is bilateral oracle binary classification thresholding a continuous latent
  accessibility variable, or are PARAM/CTX_DEP genuinely categorical states?

EXPERIMENT DESIGN
-----------------
Collect 5 groups from TriviaQA based on nocontext F1 thresholds:

  STRONG_PARAM  : nocontext_F1 >= 0.80                                (N=60)
  WEAK_PARAM    : nocontext_F1 in [0.20, 0.50)                       (N=60)
  BORDERLINE    : nocontext_F1 in [0.05, 0.20)                       (N=60)
  WEAK_CTX_DEP  : nocontext_F1 in [0.00, 0.10) AND ctx_F1 >= 0.50   (N=60)
  STRONG_CTX_DEP: nocontext_F1 <= 0.05 AND ctx_F1 >= 0.80            (N=60)

For each item: extract step-1 hidden states at layer LAYER_IDX (nocontext pass).

Train Fisher+PCA64 probe on STRONG_PARAM vs STRONG_CTX_DEP only.
Apply probe to all 5 groups — measure decision value distribution.

Outputs:
  - Per-group LDA decision value statistics (mean, std, histogram)
  - PCA-2D projection coordinates for all 5 groups (for visualization)
  - Kolmogorov-Smirnov test: are adjacent groups statistically distinguishable?
  - Verdict: CONTINUOUS_MANIFOLD or DISCRETE_CLUSTERS

Expected outcomes:
  CONTINUOUS_MANIFOLD: decision values ordered STRONG_PARAM > WEAK_PARAM >
    BORDERLINE > WEAK_CTX_DEP > STRONG_CTX_DEP with significant overlap.
    → bilateral oracle is thresholding a continuous scalar.
  DISCRETE_CLUSTERS: STRONG_PARAM and STRONG_CTX_DEP clusters with BORDERLINE
    randomly distributed between them (bimodal).
    → epistemic state is genuinely categorical.

REGISTRY: EXP_T1C_BORDERLINE (PENDING → COMPLETE when run)
"""

from __future__ import annotations
import gc, json, os, random, sys, time
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp

# ── Config ────────────────────────────────────────────────────────────────────
SEED         = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
if DEVICE == "cpu":
    raise RuntimeError("GPU required. Exiting.")

LAYER_IDX    = 26
PCA_DIM      = 64
N_PER_GROUP  = 60
POOL_SIZE    = 5000   # with corrected thresholds (~3.75% yield per STRONG group)
MAX_GEN      = 60
MAX_CTX      = 800
OUTPUT_FILE  = "borderline_geometry_results.json"

MODEL_CFG = {"name": "qwen25_1.5b_instruct", "model_id": "Qwen/Qwen2.5-1.5B-Instruct"}

# Group thresholds — aligned with C3-v3 bilateral oracle for STRONG groups
# v3 fix: STRONG thresholds use standard bilateral oracle (0.50/0.50), not (0.80/0.80)
# v3 fix: GROUP_ORDER puts STRONG_CTX_DEP before WEAK_CTX_DEP so high-confidence items
#         are captured first. WEAK_CTX_DEP = nocontext<0.05 AND withcontext in [0.20,0.50)
GROUPS = {
    "STRONG_PARAM":   {"nocontext_min": 0.50, "nocontext_max": 1.01, "ctx_min": None},
    "WEAK_PARAM":     {"nocontext_min": 0.15, "nocontext_max": 0.50, "ctx_min": None},
    "BORDERLINE":     {"nocontext_min": 0.05, "nocontext_max": 0.15, "ctx_min": None},
    "STRONG_CTX_DEP": {"nocontext_min": 0.00, "nocontext_max": 0.05, "ctx_min": 0.50},
    "WEAK_CTX_DEP":   {"nocontext_min": 0.00, "nocontext_max": 0.10, "ctx_min": 0.20},
}
GROUP_ORDER = ["STRONG_PARAM", "WEAK_PARAM", "BORDERLINE", "STRONG_CTX_DEP", "WEAK_CTX_DEP"]


# ── Dataset ───────────────────────────────────────────────────────────────────
def load_trivia():
    ds = load_dataset("trivia_qa", "rc.wikipedia", split="validation",
                      trust_remote_code=True)
    items = []
    for ex in ds:
        ctx = ""
        ep = ex.get("entity_pages", {})
        if ep and ep.get("wiki_context"):
            ctx = ep["wiki_context"][0][:MAX_CTX]
        ans = ex["answer"]["aliases"] or [ex["answer"]["value"]]
        items.append({"question": ex["question"], "context": ctx, "answers": ans})
    random.shuffle(items)
    return items


# ── Text helpers ──────────────────────────────────────────────────────────────
def fmt_prompt(q, ctx=None):
    if ctx:
        return f"Context: {ctx}\n\nAnswer the following in one short phrase.\nQuestion: {q}\nAnswer:"
    return f"Answer the following in one short phrase.\nQuestion: {q}\nAnswer:"

def generate_text(model, tok, prompt, max_new=MAX_GEN):
    inp = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id, use_cache=True)
    return tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def token_f1(pred, golds):
    pred_tok = set(pred.lower().split())
    best = 0.0
    for g in golds:
        g_tok = set(g.lower().split())
        if not g_tok or not pred_tok:
            continue
        common = pred_tok & g_tok
        if not common:
            continue
        p = len(common) / len(pred_tok)
        r = len(common) / len(g_tok)
        best = max(best, 2 * p * r / (p + r))
    return best


# ── Group Labeling ────────────────────────────────────────────────────────────
def assign_group(model, tok, item):
    q, ctx, ans = item["question"], item["context"], item["answers"]
    pred_no = generate_text(model, tok, fmt_prompt(q))
    f1_no   = token_f1(pred_no, ans)

    f1_ctx = None  # lazy: only computed when a ctx-requiring group is reached

    for gname in GROUP_ORDER:
        g = GROUPS[gname]
        in_nocontext = g["nocontext_min"] <= f1_no < g["nocontext_max"]
        needs_ctx    = g["ctx_min"] is not None
        if in_nocontext:
            if needs_ctx:
                if f1_ctx is None and ctx:   # compute ctx once, lazily
                    pred_ctx = generate_text(model, tok, fmt_prompt(q, ctx))
                    f1_ctx   = token_f1(pred_ctx, ans)
                if ctx and f1_ctx is not None and f1_ctx >= g["ctx_min"]:
                    return gname, f1_no, f1_ctx
            else:
                return gname, f1_no, f1_ctx

    return "SKIP", f1_no, f1_ctx


# ── HS Extraction (step-1) ────────────────────────────────────────────────────
def extract_step1_hs(model, tok, q, layer_idx):
    prompt = fmt_prompt(q)
    inp    = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    captured = {}

    def hook_fn(module, inp_t, out):
        hs = out[0] if isinstance(out, tuple) else out
        if hs.shape[1] != 1:
            return
        if "hs" in captured:
            return
        captured["hs"] = hs[0, 0, :].detach().float().cpu().numpy()

    handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            model.generate(**inp, max_new_tokens=2, do_sample=False,
                           pad_token_id=tok.eos_token_id, use_cache=True)
    finally:
        handle.remove()

    return captured.get("hs")


# ── Probe ─────────────────────────────────────────────────────────────────────
def fit_probe(hs_pos, hs_neg):
    X = np.vstack([hs_pos, hs_neg]).astype(np.float32)
    y = np.array([1]*len(hs_pos) + [0]*len(hs_neg))
    pca = PCA(n_components=min(PCA_DIM, X.shape[1], X.shape[0]-1), random_state=SEED)
    X_r = pca.fit_transform(X)
    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    lda.fit(X_r, y)
    scores = lda.decision_function(X_r)
    auroc  = float(roc_auc_score(y, scores))
    return pca, lda, auroc

def apply_probe(pca, lda, hs_list):
    if not hs_list:
        return np.array([])
    X = np.array(hs_list, dtype=np.float32)
    X_r = pca.transform(X)
    return lda.decision_function(X_r)

def pca2d_project(pca_fitted, hs_dict):
    """Project all groups to 2D PCA space for visualization."""
    all_vecs = []
    group_indices = {}
    idx = 0
    for gname in GROUP_ORDER:
        vecs = hs_dict.get(gname, [])
        if vecs:
            group_indices[gname] = list(range(idx, idx + len(vecs)))
            all_vecs.extend(vecs)
            idx += len(vecs)
    if not all_vecs:
        return {}
    X = np.array(all_vecs, dtype=np.float32)
    X_r = pca_fitted.transform(X)
    # Keep first 2 components
    coords = X_r[:, :2].tolist()
    result = {}
    for gname, idxs in group_indices.items():
        result[gname] = [coords[i] for i in idxs]
    return result


# ── Main ──────────────────────────────────────────────────────────────────────
def run():
    t0 = time.time()
    results = {
        "config": {
            "model": MODEL_CFG["name"],
            "layer_idx": LAYER_IDX,
            "n_per_group": N_PER_GROUP,
            "groups": list(GROUPS.keys()),
        },
        "group_stats": {},
        "ks_tests": {},
        "pca2d_coords": {},
        "verdict": "UNKNOWN",
        "elapsed_s": 0,
    }

    print(f"\n{'='*60}")
    print(f"Borderline Population Geometry — {MODEL_CFG['name']}")
    print(f"{'='*60}")

    print("Loading dataset...")
    all_items = load_trivia()[:POOL_SIZE]

    print(f"Loading model: {MODEL_CFG['model_id']}")
    tok = AutoTokenizer.from_pretrained(MODEL_CFG["model_id"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_CFG["model_id"], torch_dtype=torch.float16,
        device_map=None, low_cpu_mem_usage=True,
    ).to(DEVICE).eval()

    # ── Phase 1: Group Labeling ─────────────────────────────────────────────────
    print("\nPhase 1: Collecting items per group...")
    group_items = {g: [] for g in GROUP_ORDER}
    done_count  = {g: 0 for g in GROUP_ORDER}

    for i, item in enumerate(all_items):
        if all(len(group_items[g]) >= N_PER_GROUP for g in GROUP_ORDER):
            break
        if i % 200 == 0:
            summary = "  ".join(f"{g[:2]}={len(group_items[g])}" for g in GROUP_ORDER)
            print(f"  [{i}/{POOL_SIZE}]  {summary}", flush=True)

        gname, f1_no, f1_ctx = assign_group(model, tok, item)
        if gname != "SKIP" and len(group_items[gname]) < N_PER_GROUP:
            group_items[gname].append(item)

    for g in GROUP_ORDER:
        print(f"  {g}: {len(group_items[g])} items")

    # ── Phase 2: HS Extraction ──────────────────────────────────────────────────
    print("\nPhase 2: Extracting step-1 hidden states...")
    group_hs = {g: [] for g in GROUP_ORDER}

    for g in GROUP_ORDER:
        print(f"  Extracting {g}...")
        for idx, item in enumerate(group_items[g]):
            vec = extract_step1_hs(model, tok, item["question"], LAYER_IDX)
            if vec is not None:
                group_hs[g].append(vec)
        print(f"    → {len(group_hs[g])} HS vectors extracted")

    # ── Phase 3: Fit Probe on Extreme Groups ────────────────────────────────────
    print("\nPhase 3: Fitting Fisher+PCA64 probe on STRONG groups...")
    sp_hs  = np.array(group_hs["STRONG_PARAM"],   dtype=np.float32)
    sc_hs  = np.array(group_hs["STRONG_CTX_DEP"], dtype=np.float32)

    if len(sp_hs) < 10 or len(sc_hs) < 10:
        raise RuntimeError("Insufficient STRONG group data for probe fitting.")

    pca_fitted, lda_fitted, train_auroc = fit_probe(sp_hs, sc_hs)
    print(f"  Training AUROC (STRONG groups): {train_auroc:.4f}")

    # ── Phase 4: Apply Probe to All Groups ──────────────────────────────────────
    print("\nPhase 4: Applying probe to all groups...")
    group_scores = {}
    for g in GROUP_ORDER:
        scores = apply_probe(pca_fitted, lda_fitted, group_hs[g])
        group_scores[g] = scores.tolist()
        if len(scores) > 0:
            print(f"  {g:20s}: mean={scores.mean():+.3f}  std={scores.std():.3f}  n={len(scores)}")

    # ── Phase 5: Statistics ──────────────────────────────────────────────────────
    print("\nPhase 5: Computing group statistics and KS tests...")
    group_stats = {}
    for g in GROUP_ORDER:
        sc = np.array(group_scores[g])
        if len(sc) == 0:
            group_stats[g] = {"n": 0}
            continue
        group_stats[g] = {
            "n":        len(sc),
            "mean":     float(np.mean(sc)),
            "std":      float(np.std(sc)),
            "median":   float(np.median(sc)),
            "p25":      float(np.percentile(sc, 25)),
            "p75":      float(np.percentile(sc, 75)),
        }

    # KS tests between adjacent groups
    ks_tests = {}
    for i in range(len(GROUP_ORDER) - 1):
        g1, g2 = GROUP_ORDER[i], GROUP_ORDER[i+1]
        sc1 = np.array(group_scores[g1])
        sc2 = np.array(group_scores[g2])
        if len(sc1) > 4 and len(sc2) > 4:
            stat, pval = ks_2samp(sc1, sc2)
            ks_tests[f"{g1}_vs_{g2}"] = {"ks_stat": round(float(stat), 4),
                                           "p_value": round(float(pval), 6)}
            print(f"  KS {g1} vs {g2}: D={stat:.4f}  p={pval:.4f}")

    # ── Phase 6: PCA-2D Projection ───────────────────────────────────────────────
    print("\nPhase 6: PCA-2D projection for visualization...")
    pca2d_coords = pca2d_project(pca_fitted, group_hs)

    # ── Verdict ────────────────────────────────────────────────────────────────
    means = [(g, group_stats[g].get("mean", 0)) for g in GROUP_ORDER if group_stats[g].get("n", 0) > 0]
    expected_order = all(
        means[i][1] > means[i+1][1]
        for i in range(len(means)-1)
    )

    borderline_stats = group_stats.get("BORDERLINE", {})
    sp_stats = group_stats.get("STRONG_PARAM", {})
    sc_stats = group_stats.get("STRONG_CTX_DEP", {})

    if (expected_order and borderline_stats.get("n", 0) > 0 and
            sp_stats.get("mean") is not None and sc_stats.get("mean") is not None):
        b_mean = borderline_stats.get("mean", 0)
        sp_mean = sp_stats.get("mean", 1)
        sc_mean = sc_stats.get("mean", -1)
        between = sc_mean < b_mean < sp_mean
        verdict = "CONTINUOUS_MANIFOLD" if between else "PARTIAL_ORDER"
    else:
        verdict = "DISCRETE_CLUSTERS" if not expected_order else "PARTIAL_ORDER"

    print(f"\nVerdict: {verdict}")
    if expected_order:
        print("  Group means in expected order: STRONG_PARAM > ... > STRONG_CTX_DEP")
    else:
        print("  Group means NOT in expected order — check results.")

    results["group_stats"]   = group_stats
    results["group_scores"]  = {g: [round(v, 4) for v in group_scores[g]] for g in GROUP_ORDER}
    results["ks_tests"]      = ks_tests
    results["pca2d_coords"]  = pca2d_coords
    results["train_auroc"]   = round(train_auroc, 4)
    results["verdict"]       = verdict
    results["elapsed_s"]     = round(time.time() - t0, 1)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_FILE}")
    print(f"Total elapsed: {results['elapsed_s']}s")


if __name__ == "__main__":
    run()
