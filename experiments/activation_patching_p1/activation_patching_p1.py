#!/usr/bin/env python3
"""
Activation Patching P1 — Bilateral Oracle Direction
=====================================================
Tests whether the PARAM/CTX_DEP discriminant axis at layer 26 causally
constrains generation.

Exp A (CTX_DEP → PARAM):
  Patch CTX_DEP items toward PARAM centroid direction.
  If F1 increases monotonically with alpha: geometry is causally load-bearing.

Exp B (PARAM → CTX_DEP):
  Patch PARAM items toward CTX_DEP centroid direction.
  If F1 drops: the direction is necessary for retrieval, not just correlated.

Controls (at alpha=1.0):
  - Shuffled: random direction, same magnitude as centroid distance
  - Orthogonal: direction orthogonal to Fisher axis, same magnitude

Alpha sweep: {0.0, 0.25, 0.5, 1.0, 1.5, 2.0} × centroid distance
  alpha=1.0 → move exactly one centroid-distance step

Both experiments on Qwen2.5-1.5B-Instruct and Llama-3.2-3B-Instruct.
Probe fitting: Fisher+PCA64 on N_CAL=80/class from calibration pool.
Test set: N_TEST=40/class from a separate, non-overlapping pool.
"""

import json, os, time, random
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score

# ── Config ────────────────────────────────────────────────────────────────────
SEED            = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
if DEVICE == "cpu":
    print("WARNING: No GPU detected. Run will be very slow.")

LAYER_IDX       = 26           # penultimate layer (n_layers=28 for both models)
PCA_DIM         = 64           # PCA components before LDA
N_CAL_TARGET    = 80           # calibration set per class (probe fitting)
N_TEST_TARGET   = 40           # test set per class (patching experiment)
POOL_CALIB      = 2000         # TriviaQA items to search for calibration
POOL_TEST       = 1500         # TriviaQA items to search for test (non-overlapping)
ALPHA_VALUES    = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]
ALPHA_CONTROL   = 1.0          # alpha used for shuffled/orthogonal controls
MAX_GEN         = 50
MAX_CONTEXT     = 800          # chars, for wiki context truncation
OUTPUT_FILE     = "activation_patching_results.json"

def _find_llama_path():
    """Discover Llama model path from Kaggle model sources mount."""
    candidates = [
        "/kaggle/input/llama-3.2/transformers/3b-instruct/1",
        "/kaggle/input/llama3.2/transformers/3b-instruct/1",
        "/kaggle/input/llama-3.2-3b-instruct/transformers/default/1",
    ]
    # Also scan /kaggle/input/ for any directory containing llama
    try:
        for d in os.listdir("/kaggle/input"):
            if "llama" in d.lower():
                base = f"/kaggle/input/{d}"
                print(f"  Found llama candidate: {base}")
                # Walk two levels
                for sub in ["", "/transformers/3b-instruct/1", "/1"]:
                    full = base + sub
                    if os.path.exists(full) and os.path.isdir(full):
                        candidates.insert(0, full)
    except Exception:
        pass
    for c in candidates:
        if os.path.exists(c):
            print(f"  Llama path resolved: {c}")
            return c
    print(f"  /kaggle/input contents: {os.listdir('/kaggle/input') if os.path.exists('/kaggle/input') else 'N/A'}")
    return "meta-llama/Llama-3.2-3B-Instruct"  # fallback (will 401 without HF token)

_LLAMA_ID = _find_llama_path()

MODELS = [
    {"name": "qwen25_1.5b_instruct", "model_id": "Qwen/Qwen2.5-1.5B-Instruct"},
    {"name": "llama3.2_3b_instruct",  "model_id": _LLAMA_ID},
]

# ── Dataset ───────────────────────────────────────────────────────────────────
def load_trivia():
    ds = load_dataset("trivia_qa", "rc.wikipedia", split="validation", trust_remote_code=True)
    items = []
    for ex in ds:
        ctx = ""
        ep = ex.get("entity_pages", {})
        if ep and ep.get("wiki_context"):
            ctx = ep["wiki_context"][0][:MAX_CONTEXT]
        ans = ex["answer"]["aliases"] or [ex["answer"]["value"]]
        items.append({"question": ex["question"], "context": ctx, "answers": ans})
    random.shuffle(items)
    return items

# ── Text helpers ──────────────────────────────────────────────────────────────
def fmt_prompt(q, ctx=None):
    if ctx:
        return f"Context: {ctx}\n\nAnswer the following in one short phrase.\nQuestion: {q}\nAnswer:"
    return f"Answer the following in one short phrase.\nQuestion: {q}\nAnswer:"

def generate(model, tok, prompt, max_new=MAX_GEN):
    inp = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **inp, max_new_tokens=max_new, do_sample=False,
            pad_token_id=tok.eos_token_id, use_cache=True,
        )
    return tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def token_f1(pred, golds):
    pt = pred.lower().split()
    if not pt:
        return 0.0
    best = 0.0
    for g in golds:
        gt = g.lower().split()
        common = set(pt) & set(gt)
        if not common:
            continue
        p = len(common) / len(pt)
        r = len(common) / len(gt)
        best = max(best, 2 * p * r / (p + r))
    return best

def answer_contains(pred, golds):
    pl = pred.lower()
    return any(g.lower() in pl or pl in g.lower() for g in golds)

# ── Bilateral oracle ──────────────────────────────────────────────────────────
def bilateral_label(model, tok, q, ctx, ans):
    """Two-pass oracle: PARAM (model knows), CTX_DEP (needs context), SKIP."""
    nc = generate(model, tok, fmt_prompt(q, ctx=None))
    nc_f1 = token_f1(nc, ans)
    if nc_f1 >= 0.50 or answer_contains(nc, ans):
        return "PARAM"
    if nc_f1 > 0.05:
        return "SKIP"
    wc = generate(model, tok, fmt_prompt(q, ctx=ctx))
    if token_f1(wc, ans) >= 0.50 or answer_contains(wc, ans):
        return "CTX_DEP"
    return "SKIP"

# ── Step-1 hidden state extraction ───────────────────────────────────────────
def extract_step1_hs(model, tok, q, layer_idx):
    """
    Extract hidden state at layer_idx after the FIRST generated token.
    Uses shape-based dispatch: during prompt processing, seq_len > 1.
    During generation steps, seq_len == 1. First such step = step-1.
    """
    captured = {}

    def hook_fn(module, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        if hs.shape[1] != 1:
            return                                  # prompt pass, skip
        if "hs" in captured:
            return                                  # already captured step-1
        captured["hs"] = hs[0, 0, :].detach().float().cpu().numpy()

    h = model.model.layers[layer_idx].register_forward_hook(hook_fn)
    inp = tok(fmt_prompt(q), return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        # max_new_tokens=2: prompt pass has shape[1]=prompt_len (skipped);
        # token-1 pass has shape[1]=1 (captured). With max_new_tokens=1,
        # HF never makes a separate generation-phase forward call.
        model.generate(**inp, max_new_tokens=2, do_sample=False,
                       pad_token_id=tok.eos_token_id, use_cache=True)
    h.remove()
    return captured.get("hs")

# ── Probe fitting ─────────────────────────────────────────────────────────────
def fit_probe_and_direction(hs_param, hs_ctxdep):
    """
    Fisher+PCA64 probe.
    Returns:
      - param_dir: unit vector pointing from CTX_DEP centroid toward PARAM centroid
      - centroid_dist: L2 distance between centroids in full hidden space
      - auroc: Fisher+PCA64 probe AUROC on all calibration data
    alpha=1.0 in patching moves by exactly one centroid_dist step.
    """
    if len(hs_param) == 0 or len(hs_ctxdep) == 0:
        raise ValueError(f"Empty HS arrays: param={len(hs_param)}, ctxdep={len(hs_ctxdep)}. "
                         "Check hook dispatch — max_new_tokens must be >= 2.")
    X = np.vstack([hs_param, hs_ctxdep]).astype(np.float32)
    y = np.array([1] * len(hs_param) + [0] * len(hs_ctxdep))

    n_comp = min(PCA_DIM, X.shape[0] - 2, X.shape[1])
    pca = PCA(n_components=n_comp, random_state=SEED)
    X_pca = pca.fit_transform(X)

    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    lda.fit(X_pca, y)
    auroc = float(roc_auc_score(y, lda.decision_function(X_pca)))

    mu_p = X_pca[y == 1].mean(axis=0)
    mu_c = X_pca[y == 0].mean(axis=0)
    diff_pca  = mu_p - mu_c                          # PARAM - CTX_DEP in PCA space
    diff_full = diff_pca @ pca.components_            # (hidden_dim,)
    cdist = float(np.linalg.norm(diff_full))
    if cdist < 1e-8:
        diff_full = np.random.RandomState(SEED).randn(X.shape[1])
        cdist = float(np.linalg.norm(diff_full))
    param_dir = diff_full / cdist                     # unit vector

    return param_dir, cdist, auroc

def make_orthogonal_dir(d):
    """Unit vector orthogonal to d (for control experiment)."""
    rng = np.random.RandomState(SEED + 1)
    v = rng.randn(len(d))
    v -= (v @ d) * d
    return v / np.linalg.norm(v)

def make_random_dir(dim):
    """Random unit vector (shuffled direction control)."""
    rng = np.random.RandomState(SEED + 2)
    v = rng.randn(dim)
    return v / np.linalg.norm(v)

# ── Patched generation ────────────────────────────────────────────────────────
def generate_with_patch(model, tok, q, patch_vec, layer_idx, ctx=None):
    """
    Generate answer with patch_vec added to step-1 hidden state at layer_idx.
    patch_vec = alpha * centroid_dist * unit_direction  (caller scales it)
    """
    patch_np = patch_vec.astype(np.float32)
    patched = {"done": False}

    def hook_fn(module, inp, out):
        if patched["done"]:
            return
        hs = out[0] if isinstance(out, tuple) else out
        if hs.shape[1] != 1:
            return                                  # prompt pass, not generation
        patched["done"] = True
        patch_t = torch.from_numpy(patch_np).to(hs.device).to(hs.dtype)
        hs_mod  = hs + patch_t.unsqueeze(0).unsqueeze(0)
        if isinstance(out, tuple):
            return (hs_mod,) + out[1:]
        return hs_mod

    h = model.model.layers[layer_idx].register_forward_hook(hook_fn)
    inp = tok(fmt_prompt(q, ctx=ctx), return_tensors="pt",
              truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **inp, max_new_tokens=MAX_GEN, do_sample=False,
            pad_token_id=tok.eos_token_id, use_cache=True,
        )
    h.remove()
    return tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

# ── Experiment runner ─────────────────────────────────────────────────────────
def run_experiment(model, tok, items, param_dir, cdist, orth_dir, shuf_dir, layer_idx):
    """
    Run one patching experiment direction sweep + controls.
    param_dir: unit vector pointing in the patch direction
    cdist: centroid distance (scales alpha)
    Items are the test samples to patch.
    """
    records = []
    for item in items:
        q, ans = item["question"], item["answers"]
        rec = {"question": q[:80], "alpha_results": {}, "controls": {}}

        for alpha in ALPHA_VALUES:
            patch_vec = param_dir * (alpha * cdist)
            a = generate_with_patch(model, tok, q, patch_vec, layer_idx)
            rec["alpha_results"][str(alpha)] = {
                "f1":       round(token_f1(a, ans), 4),
                "contains": answer_contains(a, ans),
                "answer":   a[:100],
            }

        # Shuffled control at alpha=1.0
        shuf_vec = shuf_dir * ALPHA_CONTROL * cdist
        a_shuf = generate_with_patch(model, tok, q, shuf_vec, layer_idx)
        rec["controls"]["shuffled"] = {
            "f1": round(token_f1(a_shuf, ans), 4),
            "contains": answer_contains(a_shuf, ans),
            "answer": a_shuf[:80],
        }

        # Orthogonal control at alpha=1.0
        orth_vec = orth_dir * ALPHA_CONTROL * cdist
        a_orth = generate_with_patch(model, tok, q, orth_vec, layer_idx)
        rec["controls"]["orthogonal"] = {
            "f1": round(token_f1(a_orth, ans), 4),
            "contains": answer_contains(a_orth, ans),
            "answer": a_orth[:80],
        }

        records.append(rec)
    return records

# ── Aggregation ───────────────────────────────────────────────────────────────
def aggregate(records):
    """Compute dose-response stats and control deltas from experiment records."""
    baseline_f1 = float(np.mean([r["alpha_results"]["0.0"]["f1"] for r in records]))
    dose = {}
    for alpha in ALPHA_VALUES:
        vals = [r["alpha_results"][str(alpha)]["f1"] for r in records]
        cr   = [r["alpha_results"][str(alpha)]["contains"] for r in records]
        dose[str(alpha)] = {
            "f1_mean":      round(float(np.mean(vals)), 4),
            "f1_std":       round(float(np.std(vals)),  4),
            "contains_rate":round(float(np.mean(cr)),   4),
            "delta":        round(float(np.mean(vals)) - baseline_f1, 4),
        }
    shuf_f1  = float(np.mean([r["controls"]["shuffled"]["f1"]   for r in records]))
    orth_f1  = float(np.mean([r["controls"]["orthogonal"]["f1"] for r in records]))
    patch_f1 = float(np.mean([r["alpha_results"]["1.0"]["f1"]   for r in records]))
    dose["controls"] = {
        "baseline_f1":      round(baseline_f1, 4),
        "patch_1.0_f1":     round(patch_f1,    4),
        "patch_1.0_delta":  round(patch_f1 - baseline_f1, 4),
        "shuffled_f1":      round(shuf_f1,     4),
        "shuffled_delta":   round(shuf_f1 - baseline_f1,  4),
        "orthogonal_f1":    round(orth_f1,     4),
        "orthogonal_delta": round(orth_f1 - baseline_f1,  4),
    }
    return dose

def compute_verdict(agg_a, agg_b):
    """
    CAUSAL_BIDIRECTIONAL: Exp A F1 rises AND controls don't AND Exp B F1 drops.
    CAUSAL_A_ONLY: Exp A causal, Exp B inconclusive.
    CAUSAL_B_ONLY: Exp B causal, Exp A inconclusive.
    EPIPHENOMENAL: neither.
    Causal threshold: delta > 0.05 for A, < -0.05 for B.
    Specificity: patch - shuffled > 0.05 AND patch - orthogonal > 0.05.
    """
    pa = agg_a["controls"]["patch_1.0_delta"]
    sa = agg_a["controls"]["shuffled_delta"]
    oa = agg_a["controls"]["orthogonal_delta"]
    pb = agg_b["controls"]["patch_1.0_delta"]

    causal_a = (pa > 0.05) and ((pa - sa) > 0.05) and ((pa - oa) > 0.05)
    causal_b = (pb < -0.05)

    if causal_a and causal_b:
        return "CAUSAL_BIDIRECTIONAL"
    if causal_a:
        return "CAUSAL_A_ONLY"
    if causal_b:
        return "CAUSAL_B_ONLY"
    return "EPIPHENOMENAL"

# ── Per-model pipeline ────────────────────────────────────────────────────────
def run_model(model_cfg, trivia):
    name = model_cfg["name"]
    print(f"\n{'='*60}\n{name}\n{'='*60}")
    t_total = time.time()

    tok = AutoTokenizer.from_pretrained(model_cfg["model_id"], trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["model_id"],
        torch_dtype=torch.float16,
        device_map=None,
        trust_remote_code=True,
    ).to(DEVICE).eval()

    n_layers  = model.config.num_hidden_layers
    layer_idx = min(LAYER_IDX, n_layers - 2)
    print(f"  n_layers={n_layers}, patching layer {layer_idx}")

    # ── Phase 1: Calibration collection ──────────────────────────────────────
    print(f"\n  [Phase 1] Calibration bilateral oracle (target {N_CAL_TARGET}/class)...")
    t0 = time.time()
    cal_param, cal_ctxdep = [], []
    for i, item in enumerate(trivia[:POOL_CALIB]):
        if len(cal_param) >= N_CAL_TARGET and len(cal_ctxdep) >= N_CAL_TARGET:
            break
        if not item["context"]:
            continue
        lbl = bilateral_label(model, tok, item["question"], item["context"], item["answers"])
        if lbl == "PARAM"   and len(cal_param)   < N_CAL_TARGET:
            cal_param.append(item)
        elif lbl == "CTX_DEP" and len(cal_ctxdep) < N_CAL_TARGET:
            cal_ctxdep.append(item)
        if i % 200 == 0:
            print(f"    [{i}/{POOL_CALIB}] P={len(cal_param)} C={len(cal_ctxdep)}")
    n_cal = min(len(cal_param), len(cal_ctxdep))
    cal_param, cal_ctxdep = cal_param[:n_cal], cal_ctxdep[:n_cal]
    print(f"  Calibration: {n_cal}/class in {time.time()-t0:.0f}s")

    if n_cal < 20:
        return {"name": name, "error": f"Insufficient calibration samples: {n_cal}/class"}

    # ── Phase 2: Test collection (non-overlapping pool) ───────────────────────
    print(f"\n  [Phase 2] Test bilateral oracle (target {N_TEST_TARGET}/class)...")
    t0 = time.time()
    tst_param, tst_ctxdep = [], []
    for i, item in enumerate(trivia[POOL_CALIB: POOL_CALIB + POOL_TEST]):
        if len(tst_param) >= N_TEST_TARGET and len(tst_ctxdep) >= N_TEST_TARGET:
            break
        if not item["context"]:
            continue
        lbl = bilateral_label(model, tok, item["question"], item["context"], item["answers"])
        if lbl == "PARAM"   and len(tst_param)   < N_TEST_TARGET:
            tst_param.append(item)
        elif lbl == "CTX_DEP" and len(tst_ctxdep) < N_TEST_TARGET:
            tst_ctxdep.append(item)
        if i % 100 == 0:
            print(f"    [{i}/{POOL_TEST}] P={len(tst_param)} C={len(tst_ctxdep)}")
    n_test = min(len(tst_param), len(tst_ctxdep))
    tst_param, tst_ctxdep = tst_param[:n_test], tst_ctxdep[:n_test]
    print(f"  Test: {n_test}/class in {time.time()-t0:.0f}s")

    if n_test < 10:
        return {"name": name, "error": f"Insufficient test samples: {n_test}/class"}

    # ── Phase 3: Step-1 HS extraction for probe ───────────────────────────────
    print(f"\n  [Phase 3] Extracting step-1 hidden states ({n_cal*2} total)...")
    t0 = time.time()
    hs_param  = [extract_step1_hs(model, tok, x["question"], layer_idx) for x in cal_param]
    hs_ctxdep = [extract_step1_hs(model, tok, x["question"], layer_idx) for x in cal_ctxdep]
    hs_param  = [h for h in hs_param  if h is not None]
    hs_ctxdep = [h for h in hs_ctxdep if h is not None]
    print(f"  HS extracted: {len(hs_param)}P + {len(hs_ctxdep)}C in {time.time()-t0:.0f}s")

    # ── Phase 4: Probe fitting and direction computation ──────────────────────
    print(f"\n  [Phase 4] Fitting Fisher+PCA{PCA_DIM} probe...")
    param_dir, cdist, auroc = fit_probe_and_direction(hs_param, hs_ctxdep)
    orth = make_orthogonal_dir(param_dir)
    shuf = make_random_dir(len(param_dir))
    print(f"  Probe AUROC={auroc:.4f}, centroid_dist={cdist:.4f}")
    print(f"  alpha=1.0 patch magnitude: {cdist:.4f} (one full centroid step)")

    # ── Phase 5: Patching experiments ────────────────────────────────────────
    print(f"\n  [Phase 5] Running Exp A: CTX_DEP ({n_test} items) → PARAM direction...")
    t0 = time.time()
    raw_a = run_experiment(model, tok, tst_ctxdep, param_dir,  cdist, orth, shuf, layer_idx)

    print(f"  Running Exp B: PARAM ({n_test} items) → CTX_DEP direction...")
    raw_b = run_experiment(model, tok, tst_param, -param_dir, cdist, orth, shuf, layer_idx)
    print(f"  Patching done in {time.time()-t0:.0f}s")

    # ── Phase 5b: Layer sweep (Exp A only, α=1.0, n=10, find causal layer) ─────
    SWEEP_LAYERS = [4, 8, 12, 16, 20, 22, 24]
    sweep_items  = tst_ctxdep[:10]           # small subset for speed
    print(f"\n  [Phase 5b] Layer sweep: Exp A at α=1.0 across layers {SWEEP_LAYERS}...")
    sweep_results = {}
    for sl in SWEEP_LAYERS:
        if sl >= n_layers - 1:
            continue
        # Re-fit probe at this layer using stored HS... but we only have L26 HS.
        # Instead: just test with the same direction (centroid from L26) but inject at sl.
        # This measures "does inserting the L26-derived direction at layer sl change behavior?"
        t_sl = time.time()
        sl_records = []
        for item in sweep_items:
            q, ans = item["question"], item["answers"]
            # baseline (alpha=0)
            a0 = generate_with_patch(model, tok, q, param_dir * 0, sl)
            f0 = token_f1(a0, ans)
            # patch at alpha=1.0
            patch_vec = param_dir * (1.0 * cdist)
            a1 = generate_with_patch(model, tok, q, patch_vec, sl)
            f1_val = token_f1(a1, ans)
            # shuffled control
            a_shuf = generate_with_patch(model, tok, q, shuf * (1.0 * cdist), sl)
            f_shuf = token_f1(a_shuf, ans)
            sl_records.append({"baseline": f0, "patch": f1_val, "shuffled": f_shuf})
        mean_baseline = float(np.mean([r["baseline"] for r in sl_records]))
        mean_patch    = float(np.mean([r["patch"]    for r in sl_records]))
        mean_shuf     = float(np.mean([r["shuffled"] for r in sl_records]))
        sweep_results[str(sl)] = {
            "baseline_f1": round(mean_baseline, 4),
            "patch_f1":    round(mean_patch,    4),
            "delta":       round(mean_patch - mean_baseline, 4),
            "shuffled_f1": round(mean_shuf,     4),
            "specific_delta": round((mean_patch - mean_baseline) - (mean_shuf - mean_baseline), 4),
        }
        print(f"    L{sl:02d}: baseline={mean_baseline:.4f}  patch={mean_patch:.4f}  "
              f"Δ={mean_patch-mean_baseline:+.4f}  shuffled_Δ={mean_shuf-mean_baseline:+.4f}  "
              f"specific={sweep_results[str(sl)]['specific_delta']:+.4f}  ({time.time()-t_sl:.0f}s)")

    # ── Phase 6: Analysis ─────────────────────────────────────────────────────
    agg_a   = aggregate(raw_a)
    agg_b   = aggregate(raw_b)
    verdict = compute_verdict(agg_a, agg_b)

    print(f"\n  ══ RESULTS ══")
    print(f"  Probe AUROC: {auroc:.4f}")
    print(f"  Exp A — CTX_DEP items patched toward PARAM:")
    for alpha in ALPHA_VALUES:
        s = agg_a[str(alpha)]
        print(f"    α={alpha:4.2f}: F1={s['f1_mean']:.4f}  Δ={s['delta']:+.4f}  contains={s['contains_rate']:.3f}")
    ctrl = agg_a["controls"]
    print(f"    Shuffled α=1.0:    Δ={ctrl['shuffled_delta']:+.4f}")
    print(f"    Orthogonal α=1.0:  Δ={ctrl['orthogonal_delta']:+.4f}")
    print(f"  Exp B — PARAM items patched toward CTX_DEP:")
    print(f"    Baseline: {agg_b['controls']['baseline_f1']:.4f}  "
          f"Patch@1.0: {agg_b['controls']['patch_1.0_f1']:.4f}  "
          f"Δ={agg_b['controls']['patch_1.0_delta']:+.4f}")
    print(f"\n  CAUSAL VERDICT: {verdict}")
    print(f"  Total time: {time.time()-t_total:.0f}s")

    del model
    torch.cuda.empty_cache()

    return {
        "name":          name,
        "model_id":      model_cfg["model_id"],
        "n_layers":      int(n_layers),
        "layer_idx":     int(layer_idx),
        "layer_sweep":   sweep_results,
        "n_cal":         int(n_cal),
        "n_test":        int(n_test),
        "probe_auroc":   round(auroc, 4),
        "centroid_dist": round(cdist, 4),
        "exp_a":         agg_a,
        "exp_b":         agg_b,
        "verdict":       verdict,
        "raw_a":         raw_a,
        "raw_b":         raw_b,
    }

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading TriviaQA rc.wikipedia...")
    trivia = load_trivia()
    print(f"Loaded {len(trivia)} items (shuffled)")

    all_results = {}
    for cfg in MODELS:
        try:
            result = run_model(cfg, trivia)
            all_results[cfg["name"]] = result
        except Exception as e:
            import traceback; traceback.print_exc()
            all_results[cfg["name"]] = {"error": str(e), "name": cfg["name"]}
        # Save after each model completes
        with open(OUTPUT_FILE, "w") as fh:
            json.dump({"models": all_results, "status": "partial"}, fh, indent=2)
        print(f"[saved intermediate → {OUTPUT_FILE}]")

    with open(OUTPUT_FILE, "w") as fh:
        json.dump({"models": all_results, "status": "complete"}, fh, indent=2)
    print(f"\nComplete. Results: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
