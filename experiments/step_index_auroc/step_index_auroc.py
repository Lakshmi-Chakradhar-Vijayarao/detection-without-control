#!/usr/bin/env python3
"""
experiments/step_index_auroc/step_index_auroc.py  — v3

Task 1.1 — Step-Index AUROC Profile
=====================================
CLAIM UNDER TEST: C006 (SUPPORTED)
  Step-1 (or step-0) is the privileged extraction point. AUROC peaks early
  and decays at later generation steps.

v3 changes vs v2:
  - EOS FILTER: only accept items where nocontext generation produces ≥ MIN_GEN_TOKENS
    tokens in BOTH classes. Eliminates the EOS confound (PARAM items terminating
    at step 2-4 with short answers).
  - 5-FOLD CV instead of 75/25 hold-out: better AUROC estimate from small n.
  - POOL=4000 to collect n≥50 CTX_DEP items.
  - MIN_GEN_TOKENS=8: items must generate at least 8 tokens nocontext.
  - Removed MAX_STEPS=25 steps that had 0 PARAM items in v2.

v2 finding: PARAM items terminate at step 2-4 (short answers). Step-0 had
  AUROC=0.781, step-1=0.609, step-2 degenerate. EOS is the confound.

REGISTRY: EXP_T1D_STEP_INDEX_V3
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
from sklearn.model_selection import StratifiedKFold

# ── Config ────────────────────────────────────────────────────────────────────
SEED        = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
p = lambda *a, **k: print(*a, **k, flush=True)
p(f"Device: {DEVICE}")
if DEVICE == "cpu":
    raise RuntimeError("GPU required.")

LAYER_IDX    = 26
PCA_DIM      = 64
POOL         = 4000            # larger pool for CTX_DEP yield
N_CV_FOLDS   = 5               # cross-validation folds
MAX_STEPS    = 10              # evaluate through step 10 (EOS-filtered items go this far)
EVAL_STEPS   = [0, 1, 2, 5, 10]
MAX_GEN      = MAX_STEPS + 2   # = 12, forces separate generation calls
MIN_GEN_TOKENS = 8             # EOS filter: item must generate >= this many tokens
MAX_CTX      = 800
PARAM_MIN_F1 = 0.50
CTX_MAX_F1   = 0.05
CTX_MIN_CTX  = 0.50
LABEL_GEN    = 30              # tokens for oracle labeling
OUTPUT_FILE  = "step_index_results.json"

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


# ── Dataset ───────────────────────────────────────────────────────────────────
def load_trivia():
    p("Loading TriviaQA...")
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
    p(f"Loaded {len(items)} items.")
    return items


# ── Text helpers ──────────────────────────────────────────────────────────────
def fmt_prompt(q, ctx=None):
    if ctx:
        return f"Context: {ctx}\n\nAnswer the following in one short phrase.\nQuestion: {q}\nAnswer:"
    return f"Answer the following in one short phrase.\nQuestion: {q}\nAnswer:"

def generate_with_ids(model, tok, prompt, max_new):
    inp = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id, use_cache=True)
    gen_ids = out[0][inp["input_ids"].shape[1]:]
    text    = tok.decode(gen_ids, skip_special_tokens=True).strip()
    return text, len(gen_ids)

def token_f1(pred, golds):
    pred_tok = set(pred.lower().split())
    best = 0.0
    for g in golds:
        g_tok = set(g.lower().split())
        if not g_tok or not pred_tok: continue
        common = pred_tok & g_tok
        if not common: continue
        prec = len(common)/len(pred_tok); rec = len(common)/len(g_tok)
        best = max(best, 2*prec*rec/(prec+rec))
    return best


# ── Bilateral Oracle + EOS filter ────────────────────────────────────────────
def label_item(model, tok, item):
    """
    Returns (label, nc_gen_len) where nc_gen_len is the number of tokens
    generated in the nocontext pass. EOS filter: nc_gen_len >= MIN_GEN_TOKENS.
    """
    q, ctx, ans = item["question"], item["context"], item["answers"]
    pred_no, nc_len = generate_with_ids(model, tok, fmt_prompt(q), LABEL_GEN)
    f1_no = token_f1(pred_no, ans)

    if f1_no >= PARAM_MIN_F1:
        return "PARAM", nc_len
    if ctx and f1_no <= CTX_MAX_F1:
        pred_ctx, _ = generate_with_ids(model, tok, fmt_prompt(q, ctx), LABEL_GEN)
        if token_f1(pred_ctx, ans) >= CTX_MIN_CTX:
            return "CTX_DEP", nc_len
    return "SKIP", nc_len


# ── Multi-Step HS Extraction ──────────────────────────────────────────────────
def extract_hs_all_steps(model, tok, q, layer_idx):
    """
    Capture HS at each generation step 0..MAX_STEPS via shape-based dispatch.
    MAX_GEN = MAX_STEPS + 2 guarantees all steps fire as separate calls.
    Returns: {step_idx: np.ndarray(hidden_dim,)}
    """
    prompt   = fmt_prompt(q)
    inp      = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    step_hs  = {}
    step_ctr = [0]

    def hook_fn(module, inp_t, out):
        hs = out[0] if isinstance(out, tuple) else out
        if hs.shape[1] != 1: return   # prompt pass
        s = step_ctr[0]
        if s <= MAX_STEPS:
            step_hs[s] = hs[0, 0, :].detach().float().cpu().numpy()
        step_ctr[0] += 1

    handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            model.generate(**inp, max_new_tokens=MAX_GEN, do_sample=False,
                           pad_token_id=tok.eos_token_id, use_cache=True)
    finally:
        handle.remove()
    return step_hs


# ── Probe: 5-fold CV ─────────────────────────────────────────────────────────
def cv_auroc(hs_param, hs_ctxdep, n_folds=N_CV_FOLDS):
    """5-fold CV AUROC + shuffled control."""
    n = min(len(hs_param), len(hs_ctxdep))
    if n < n_folds * 2:
        return None, None, n

    X = np.vstack(hs_param[:n] + hs_ctxdep[:n]).astype(np.float32)
    y = np.array([1]*n + [0]*n)

    skf    = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    scores, shuf_scores = [], []

    for tr, va in skf.split(X, y):
        n_comp = min(PCA_DIM, X.shape[1], len(tr) - 1)  # clamp to fold size
        pca = PCA(n_components=n_comp, random_state=SEED)
        X_tr = pca.fit_transform(X[tr])
        X_va = pca.transform(X[va])

        lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        lda.fit(X_tr, y[tr])

        sc = lda.decision_function(X_va)
        scores.append(float(roc_auc_score(y[va], sc)))

        y_s = y[va].copy(); np.random.shuffle(y_s)
        shuf_scores.append(float(roc_auc_score(y_s, sc)))

    return float(np.mean(scores)), float(np.mean(shuf_scores)), n


# ── Main ──────────────────────────────────────────────────────────────────────
def run():
    t0 = time.time()
    p(f"\n{'='*60}")
    p(f"Step-Index AUROC Profile v3 — {MODEL_ID}")
    p(f"POOL={POOL}  MIN_GEN_TOKENS={MIN_GEN_TOKENS}  EVAL_STEPS={EVAL_STEPS}")
    p(f"CV_FOLDS={N_CV_FOLDS}  MAX_GEN={MAX_GEN}")
    p(f"{'='*60}")

    all_items = load_trivia()[:POOL]

    p(f"Loading model: {MODEL_ID}")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16,
        device_map=None, low_cpu_mem_usage=True,
    ).to(DEVICE).eval()
    p("Model loaded.")

    # ── Phase 1: Oracle labeling with EOS filter ──────────────────────────────
    p(f"\nPhase 1: Oracle labeling + EOS filter (pool={POOL}, min_gen={MIN_GEN_TOKENS})...")
    param_items, ctxdep_items = [], []
    eos_rejected = 0

    for i, item in enumerate(all_items):
        lbl, nc_len = label_item(model, tok, item)
        if lbl in ("PARAM", "CTX_DEP"):
            if nc_len < MIN_GEN_TOKENS:
                eos_rejected += 1
                continue
            if lbl == "PARAM":   param_items.append(item)
            else:                ctxdep_items.append(item)
        if i % 20 == 0:
            p(f"  [{i:4d}/{POOL}] PARAM={len(param_items)} CTX_DEP={len(ctxdep_items)} "
              f"EOS_rejected={eos_rejected}  ({time.time()-t0:.0f}s)")

    n_p = len(param_items); n_c = len(ctxdep_items)
    p(f"\nLabeling complete: PARAM={n_p}  CTX_DEP={n_c}  EOS_rejected={eos_rejected}")

    if n_c < N_CV_FOLDS * 2:
        raise RuntimeError(f"Insufficient CTX_DEP items (n={n_c}). Need >={N_CV_FOLDS*2}.")

    n = min(n_p, n_c)
    param_items  = param_items[:n]
    ctxdep_items = ctxdep_items[:n]
    p(f"Balanced to n={n}/class.")

    # ── Phase 2: Multi-step HS extraction ────────────────────────────────────
    p(f"\nPhase 2: Extracting HS at all steps ({n}P + {n}C items, max_gen={MAX_GEN})...")
    step_hs = {s: {"param": [], "ctxdep": []} for s in range(MAX_STEPS + 1)}

    for idx, item in enumerate(param_items):
        if idx % 10 == 0:
            p(f"  PARAM [{idx}/{n}]  ({time.time()-t0:.0f}s)")
        hs = extract_hs_all_steps(model, tok, item["question"], LAYER_IDX)
        for s, vec in hs.items():
            if s in step_hs:
                step_hs[s]["param"].append(vec)

    for idx, item in enumerate(ctxdep_items):
        if idx % 10 == 0:
            p(f"  CTX_DEP [{idx}/{n}]  ({time.time()-t0:.0f}s)")
        hs = extract_hs_all_steps(model, tok, item["question"], LAYER_IDX)
        for s, vec in hs.items():
            if s in step_hs:
                step_hs[s]["ctxdep"].append(vec)

    # ── Phase 3: 5-fold CV AUROC per step ────────────────────────────────────
    p(f"\nPhase 3: 5-fold CV AUROC per step...")
    step_results = {}

    for s in EVAL_STEPS:
        hp = step_hs[s]["param"]
        hc = step_hs[s]["ctxdep"]
        auroc, shuf, n_used = cv_auroc(hp, hc)

        if auroc is None:
            p(f"  step={s:3d}: SKIP ({len(hp)}P / {len(hc)}C — too few for {N_CV_FOLDS}-fold CV)")
            step_results[s] = {"auroc": None, "n_param": len(hp), "n_ctxdep": len(hc)}
            continue

        step_results[s] = {
            "auroc":    round(auroc, 4),
            "shuffled": round(shuf, 4),
            "n_param":  len(hp),
            "n_ctxdep": len(hc),
            "n_used":   n_used,
        }
        p(f"  step={s:3d}: AUROC={auroc:.4f}  shuffled={shuf:.4f}  "
          f"n={len(hp)}P/{len(hc)}C")

    # ── Verdict ───────────────────────────────────────────────────────────────
    valid = [(s, r["auroc"]) for s, r in step_results.items() if r.get("auroc") is not None]
    valid.sort()

    verdict = "UNKNOWN"
    if len(valid) >= 2:
        by_step = dict(valid)
        s0 = by_step.get(0); s1 = by_step.get(1)
        later = [a for s, a in valid if s > 1]

        if s0 and s1 and s0 > s1 and (not later or s0 > max(later)):
            verdict = "PEAK_AT_0"
        elif s1 and s0 and s1 > s0 and (not later or s1 > max(later)):
            verdict = "PEAK_AT_1"
        elif s1 and later and all(abs(s1 - a) < 0.05 for a in later):
            verdict = "FLAT_FROM_1"
        elif valid and all(valid[i][1] >= valid[i+1][1] - 0.02 for i in range(len(valid)-1)):
            verdict = "MONOTONE_DECAY"
        else:
            verdict = "IRREGULAR"

    p(f"\n{'='*60}")
    p(f"VERDICT: {verdict}")
    p(f"Step-AUROC: { {s: r.get('auroc') for s, r in step_results.items()} }")
    p(f"{'='*60}")

    results = {
        "version":   "v3",
        "config":    {
            "model":          MODEL_ID,
            "layer_idx":      LAYER_IDX,
            "pca_dim":        PCA_DIM,
            "pool":           POOL,
            "n_cv_folds":     N_CV_FOLDS,
            "min_gen_tokens": MIN_GEN_TOKENS,
            "eval_steps":     EVAL_STEPS,
            "max_steps":      MAX_STEPS,
            "max_gen":        MAX_GEN,
        },
        "n_param":       n_p,
        "n_ctxdep":      n_c,
        "eos_rejected":  eos_rejected,
        "n_per_class":   n,
        "step_auroc":    step_results,
        "verdict":       verdict,
        "elapsed_s":     round(time.time() - t0, 1),
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    p(f"\nResults → {OUTPUT_FILE}")
    p(f"Total elapsed: {results['elapsed_s']}s")


if __name__ == "__main__":
    run()
