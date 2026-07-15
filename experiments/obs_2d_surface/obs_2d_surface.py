#!/usr/bin/env python3
"""
experiments/obs_2d_surface/obs_2d_surface.py

Task 2.1 — 2D Observability Surface (layer × generation step)
==============================================================
CENTRAL QUESTION: Where in computation does epistemic accessibility live?

Measures bilateral oracle AUROC across a 2D grid:
  - LAYER dimension: {0, 4, 8, 12, 16, 20, 22, 24, 26, 27}
  - STEP dimension:  {0, 1, 2, 5, 10, 25}

For each (layer, step) cell: fit Fisher+PCA64 probe on calibration set,
evaluate AUROC on test set, record shuffled control.

Output: 10×6 AUROC matrix — heatmap data for the flagship figure.

EXPECTED OUTCOMES
-----------------
  DEEP_EARLY_PEAK: AUROC peaks at deep layers (L24-L26) and early steps (0-1).
    → epistemic accessibility is localized in final layers at generation onset.
  BROAD_DISTRIBUTION: high AUROC across many (layer, step) cells.
    → epistemic state is distributed throughout the residual stream.
  LAYER_INVARIANT_STEP_SENSITIVE: similar AUROC across layers at each step.
    → step timing matters more than layer depth.

IMPLEMENTATION NOTE
-------------------
Uses simultaneous multi-layer hooks. For each item:
  - Register one hook per selected layer
  - Generate up to MAX_STEPS tokens
  - Each hook captures HS at each generation step using step counter
  - Remove all hooks after generation

Memory: N_items × N_layers × N_steps × hidden_dim × 4 bytes
  = 160 × 10 × 26 × 1536 × 4 ≈ 25 MB (manageable on T4 16GB)

REGISTRY: EXP_T2A_2D_SURFACE (PENDING → COMPLETE when run)
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

# ── Config ────────────────────────────────────────────────────────────────────
SEED        = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
if DEVICE == "cpu":
    raise RuntimeError("GPU required. Exiting.")

EVAL_LAYERS = [0, 4, 8, 12, 16, 20, 22, 24, 26, 27]
EVAL_STEPS  = [0, 1, 2, 5, 10]   # step-25 excluded: TriviaQA items EOS before step-10+
MAX_STEPS   = 10
MAX_GEN     = MAX_STEPS + 2
PCA_DIM     = 64
N_CAL       = 80   # per class; larger → more stable probe per cell
N_TEST      = 40
POOL_CAL    = 3000
POOL_TEST   = 2000
MAX_GEN_LABEL = 60
MAX_CTX     = 800
PARAM_MIN_F1 = 0.50
CTX_MAX_F1   = 0.05
CTX_MIN_CTX  = 0.50
OUTPUT_FILE  = "obs_2d_surface_results.json"

MODEL_CFG = {"name": "qwen25_1.5b_instruct", "model_id": "Qwen/Qwen2.5-1.5B-Instruct", "n_layers": 28}


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

def generate_text(model, tok, prompt, max_new=MAX_GEN_LABEL):
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


# ── Oracle labeling ───────────────────────────────────────────────────────────
def label_item(model, tok, item):
    q, ctx, ans = item["question"], item["context"], item["answers"]
    pred_no = generate_text(model, tok, fmt_prompt(q))
    f1_no   = token_f1(pred_no, ans)
    if f1_no >= PARAM_MIN_F1:
        return "PARAM"
    if ctx and f1_no <= CTX_MAX_F1:
        pred_ctx = generate_text(model, tok, fmt_prompt(q, ctx))
        if token_f1(pred_ctx, ans) >= CTX_MIN_CTX:
            return "CTX_DEP"
    return "SKIP"


# ── Multi-layer Multi-step HS Extraction ──────────────────────────────────────
def extract_all_layers_all_steps(model, tok, q, layers=EVAL_LAYERS, max_steps=MAX_STEPS):
    """
    Extract hidden states at all specified layers and all generation steps.
    Returns: { layer_idx: { step_idx: np.ndarray(hidden_dim,) } }
    """
    prompt = fmt_prompt(q)
    inp    = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)

    # captured[layer_idx][step_idx] = vector
    captured  = {l: {} for l in layers}
    step_ctr  = {l: [0] for l in layers}
    handles   = []

    for l_idx in layers:
        def make_hook(layer):
            def hook_fn(module, inp_t, out):
                hs = out[0] if isinstance(out, tuple) else out
                if hs.shape[1] != 1:   # prompt pass
                    return
                s = step_ctr[layer][0]
                if s <= max_steps:
                    captured[layer][s] = hs[0, 0, :].detach().float().cpu().numpy()
                step_ctr[layer][0] += 1
            return hook_fn
        h = model.model.layers[l_idx].register_forward_hook(make_hook(l_idx))
        handles.append(h)

    try:
        with torch.no_grad():
            model.generate(**inp, max_new_tokens=MAX_GEN, do_sample=False,
                           pad_token_id=tok.eos_token_id, use_cache=True)
    finally:
        for h in handles:
            h.remove()

    return captured


# ── Probe ─────────────────────────────────────────────────────────────────────
def fit_and_eval(hs_param_train, hs_ctxdep_train, hs_param_test, hs_ctxdep_test):
    """Fit Fisher+PCA64 on train, evaluate on test. Returns (auroc, shuf_auroc)."""
    if (len(hs_param_train) < 4 or len(hs_ctxdep_train) < 4 or
            len(hs_param_test) < 4 or len(hs_ctxdep_test) < 4):
        return None, None

    X_tr = np.vstack([hs_param_train, hs_ctxdep_train]).astype(np.float32)
    y_tr = np.array([1]*len(hs_param_train) + [0]*len(hs_ctxdep_train))
    X_te = np.vstack([hs_param_test,  hs_ctxdep_test]).astype(np.float32)
    y_te = np.array([1]*len(hs_param_test) + [0]*len(hs_ctxdep_test))

    n_comp = min(PCA_DIM, X_tr.shape[1], X_tr.shape[0]-1)
    pca    = PCA(n_components=n_comp, random_state=SEED)
    X_tr_r = pca.fit_transform(X_tr)
    X_te_r = pca.transform(X_te)

    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    lda.fit(X_tr_r, y_tr)

    scores_real = lda.decision_function(X_te_r)
    auroc       = float(roc_auc_score(y_te, scores_real))

    y_shuf = y_te.copy(); np.random.shuffle(y_shuf)
    shuf_auroc = float(roc_auc_score(y_shuf, scores_real))

    return round(auroc, 4), round(shuf_auroc, 4)


# ── Main ──────────────────────────────────────────────────────────────────────
def run():
    t0 = time.time()
    results = {
        "config": {
            "model": MODEL_CFG["name"],
            "eval_layers": EVAL_LAYERS,
            "eval_steps": EVAL_STEPS,
            "pca_dim": PCA_DIM,
            "n_cal": N_CAL,
            "n_test": N_TEST,
        },
        "surface": {},   # surface[layer][step] = {auroc, shuf_auroc}
        "elapsed_s": 0,
    }

    print(f"\n{'='*60}")
    print(f"2D Observability Surface — {MODEL_CFG['name']}")
    print(f"Layers: {EVAL_LAYERS}")
    print(f"Steps:  {EVAL_STEPS}")
    print(f"{'='*60}")

    print("Loading dataset...")
    all_items = load_trivia()

    print(f"Loading model: {MODEL_CFG['model_id']}")
    tok = AutoTokenizer.from_pretrained(MODEL_CFG["model_id"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_CFG["model_id"], torch_dtype=torch.float16,
        device_map=None, low_cpu_mem_usage=True,
    ).to(DEVICE).eval()

    # ── Phase 1: Oracle Labeling ────────────────────────────────────────────────
    print("\nPhase 1: Bilateral oracle labeling (calibration)...")
    cal_param, cal_ctxdep = [], []
    for i, item in enumerate(all_items[:POOL_CAL]):
        if len(cal_param) >= N_CAL and len(cal_ctxdep) >= N_CAL:
            break
        if i % 100 == 0:
            print(f"  [{i}] P={len(cal_param)} C={len(cal_ctxdep)}", flush=True)
        lbl = label_item(model, tok, item)
        if lbl == "PARAM" and len(cal_param) < N_CAL:
            cal_param.append(item)
        elif lbl == "CTX_DEP" and len(cal_ctxdep) < N_CAL:
            cal_ctxdep.append(item)

    print("\nPhase 1b: Test labeling...")
    tst_param, tst_ctxdep = [], []
    for i, item in enumerate(all_items[POOL_CAL:POOL_CAL + POOL_TEST]):
        if len(tst_param) >= N_TEST and len(tst_ctxdep) >= N_TEST:
            break
        if i % 100 == 0:
            print(f"  1b [{i}/{POOL_TEST}] TP={len(tst_param)} TC={len(tst_ctxdep)}", flush=True)
        lbl = label_item(model, tok, item)
        if lbl == "PARAM" and len(tst_param) < N_TEST:
            tst_param.append(item)
        elif lbl == "CTX_DEP" and len(tst_ctxdep) < N_TEST:
            tst_ctxdep.append(item)

    print(f"  Cal P={len(cal_param)} C={len(cal_ctxdep)}  |  Test P={len(tst_param)} C={len(tst_ctxdep)}")

    # ── Phase 2: Multi-Layer Multi-Step HS Extraction ───────────────────────────
    # Storage: items_hs[split][class] = { layer: { step: list of vecs } }
    print("\nPhase 2: Multi-layer multi-step HS extraction...")

    def extract_set(items, label):
        set_hs = {l: {s: [] for s in range(MAX_STEPS + 1)} for l in EVAL_LAYERS}
        for idx, item in enumerate(items):
            if idx % 20 == 0:
                print(f"  {label} [{idx}/{len(items)}]", flush=True)
            layer_step = extract_all_layers_all_steps(model, tok, item["question"])
            for l in EVAL_LAYERS:
                for s, vec in layer_step[l].items():
                    if s in set_hs[l]:
                        set_hs[l][s].append(vec)
        return set_hs

    cal_p_hs = extract_set(cal_param,  "CAL_PARAM")
    cal_c_hs = extract_set(cal_ctxdep, "CAL_CTX_DEP")
    tst_p_hs = extract_set(tst_param,  "TST_PARAM")
    tst_c_hs = extract_set(tst_ctxdep, "TST_CTX_DEP")

    # ── Phase 3: Probe per (Layer, Step) ─────────────────────────────────────
    print("\nPhase 3: Fitting probes for all (layer, step) cells...")
    surface = {}

    for l in EVAL_LAYERS:
        surface[l] = {}
        for s in EVAL_STEPS:
            hp_tr = cal_p_hs[l].get(s, [])
            hc_tr = cal_c_hs[l].get(s, [])
            hp_te = tst_p_hs[l].get(s, [])
            hc_te = tst_c_hs[l].get(s, [])

            auroc, shuf_auroc = fit_and_eval(hp_tr, hc_tr, hp_te, hc_te)

            surface[l][s] = {
                "auroc": auroc,
                "shuf_auroc": shuf_auroc,
                "n_cal": (len(hp_tr), len(hc_tr)),
                "n_tst": (len(hp_te), len(hc_te)),
            }
            a_str = f"{auroc:.4f}" if auroc is not None else " SKIP"
            s_str = f"{shuf_auroc:.4f}" if shuf_auroc is not None else "   --"
            print(f"  L{l:02d} step={s:2d}: AUROC={a_str}  shuffled={s_str}  "
                  f"cal=({len(hp_tr)},{len(hc_tr)})  tst=({len(hp_te)},{len(hc_te)})", flush=True)

    # ── Peak Analysis ──────────────────────────────────────────────────────────
    print("\nPeak Analysis:")
    max_auroc, peak_layer, peak_step = -1, None, None
    for l in EVAL_LAYERS:
        for s in EVAL_STEPS:
            a = surface[l][s].get("auroc")
            if a is not None and a > max_auroc:
                max_auroc, peak_layer, peak_step = a, l, s

    print(f"  Global peak: AUROC={max_auroc:.4f} at Layer={peak_layer}, Step={peak_step}")

    # Check if step-1 at deep layers is the peak
    step1_deep = [surface[l].get(1, {}).get("auroc") for l in [24, 26, 27]
                  if surface[l].get(1, {}).get("auroc") is not None]
    step1_mean_deep = np.mean(step1_deep) if step1_deep else 0

    all_valid = [surface[l][s]["auroc"] for l in EVAL_LAYERS for s in EVAL_STEPS
                 if surface[l][s].get("auroc") is not None]
    overall_mean = np.mean(all_valid) if all_valid else 0

    if peak_step == 1 and peak_layer in [24, 26, 27]:
        verdict = "DEEP_EARLY_PEAK"
    elif step1_mean_deep > overall_mean + 0.05:
        verdict = "STEP1_DEEP_DOMINANT"
    elif max(EVAL_LAYERS.index(peak_layer), 0) < 5:
        verdict = "SHALLOW_PEAK"
    else:
        verdict = "BROAD_DISTRIBUTION"

    print(f"  Verdict: {verdict}")

    results["surface"]    = {str(l): {str(s): v for s, v in sv.items()} for l, sv in surface.items()}
    results["peak"]       = {"layer": peak_layer, "step": peak_step, "auroc": round(max_auroc, 4)}
    results["verdict"]    = verdict
    results["elapsed_s"]  = round(time.time() - t0, 1)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_FILE}")
    print(f"Total elapsed: {results['elapsed_s']}s")


if __name__ == "__main__":
    run()
