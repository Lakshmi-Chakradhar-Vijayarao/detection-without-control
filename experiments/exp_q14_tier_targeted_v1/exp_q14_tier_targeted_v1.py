#!/usr/bin/env python3
"""
exp_q14_tier_targeted_v1.py — DIFFICULTY CONTROL (TIER-TARGETED COLLECTION)
=============================================================================
REDESIGN OVER Q14v3:

  Q14v3 failed because:
    1. TriviaQA validation has 7993 items total — pool exhausted
    2. Easy PARAM (context_f1 ≥ 0.80) yield = 20/200 PARAM = 10%
    3. Tier probes need N≥30/class; got N=20 → AUROC=N/A

  This version fixes all three:
    A) TriviaQA rc.wikipedia TRAIN split (~61k items)
    B) Tier-targeted collection: keep scanning past 200 PARAM total
       until each tier bucket has ≥ N_TIER_TARGET items
    C) Global PCA basis (fit on all collected items, N≈500+):
       tier LDA only needs N≥15/class — avoids the small-N PCA failure
    D) 5-fold stratified CV for tier probes (not fixed 75/25 split)
    E) Adds Phase 5: within-CTX_DEP difficulty test (absent in Q14v3)

COLLECTION TARGETS (tier-targeted):
  Easy PARAM   (context_f1 ≥ 0.80)          → target 30 items
  Medium PARAM (context_f1 ∈ [0.50, 0.80))  → target 30 items
  Hard PARAM   (context_f1 < 0.50)           → cap at 300
  CTX_DEP total                              → target 150 items

PHASES:
  1. Bilateral oracle collection (TRAIN split, tier-targeted)
  2. Full unmatched probe (C001 replication, global PCA64 + LDA 75/25)
  3. Difficulty-matched tier probes (global PCA64 + LDA, 5-fold CV):
       Easy tier  (context_f1 ≥ 0.80): PARAM_easy vs CTX_DEP_easy
       Medium tier (context_f1 ∈ [0.50, 0.80)): PARAM_med vs CTX_DEP_med
  4. Within-PARAM difficulty (nocontext_f1 tiers, global PCA64 + LDA 5-fold CV)
  5. Within-CTX_DEP difficulty (context_f1 tiers, global PCA64 + LDA 5-fold CV)

VERDICT PRE-REGISTRATION:
  H-EPISTEMIC (probe reads knowledge source, not difficulty):
    AUROC_easy_tier ≥ 0.60 AND AUROC_med_tier ≥ 0.55 AND
    AUROC_within_param ≤ 0.60 AND AUROC_within_ctxdep ≤ 0.60
  H-DIFFICULTY (probe is a difficulty detector):
    AUROC_easy_tier < 0.55 OR AUROC_within_param ≥ 0.65

GPU: T4 (~7h for MAX_SCAN=15000)
"""

from __future__ import annotations
import gc, json, os, random, time
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# ── Config ─────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}", flush=True)
if DEVICE == "cpu":
    raise RuntimeError("GPU required. Exiting.")

MODEL_ID        = "Qwen/Qwen2.5-1.5B-Instruct"
LAYER_IDX       = 26

# Tier-targeted collection
N_EASY_PARAM    = 30     # context_f1 ≥ 0.80 (rare — keeps scanning for these)
N_MED_PARAM     = 30     # context_f1 ∈ [0.50, 0.80)
N_HARD_PARAM    = 300    # context_f1 < 0.50 (capped — easy to collect)
N_CTX_TARGET    = 150    # CTX_DEP total (tier-assigned post-hoc)
MAX_SCAN        = 15_000 # ~7h on T4 (was 12k; increased to improve easy/med tier yield)

# Oracle thresholds (same as C001/Q14v3)
PARAM_MIN_F1    = 0.50
CTX_MAX_F1      = 0.05
CTX_MIN_CTX     = 0.50

# Tier thresholds
TIER_EASY_MIN   = 0.80
TIER_MED_MIN    = 0.50
TIER_MED_MAX    = 0.80

# Probe config
PCA_DIM_FULL    = 64     # Global PCA basis — fit on all N items
N_TIER_MIN      = 15     # Min N/class to run tier probe
N_CV_FOLDS      = 5      # Stratified CV folds for tier probes
N_SHUFFLED      = 3      # Shuffled controls per probe
TRAIN_FRAC      = 0.75   # For full unmatched probe (large N)
MAX_GEN         = 60
MAX_CTX         = 800

OUTPUT_FILE        = "/kaggle/working/exp_q14_tier_targeted_v1_results.json"
SAVE_INTERVAL      = 1000   # progress print interval
CHECKPOINT_INTERVAL = 2000  # save numpy arrays to disk (crash protection)


# ── Helpers ────────────────────────────────────────────────────────────────────
def token_f1(pred: str, gold_list: list[str]) -> float:
    pred_tokens = set(pred.lower().split())
    if not pred_tokens:
        return 0.0
    best = 0.0
    for gold in gold_list:
        gold_tokens = set(gold.lower().split())
        if not gold_tokens:
            continue
        common = len(pred_tokens & gold_tokens)
        if common == 0:
            continue
        p = common / len(pred_tokens)
        r = common / len(gold_tokens)
        f1 = 2 * p * r / (p + r)
        best = max(best, f1)
    return best


def _get_hf_token():
    t = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return t if t and t.startswith("hf_") else None


def _save_checkpoint(easy_p, med_p, hard_p, ctx, n_scanned):
    """Save hidden state arrays to disk for crash recovery."""
    info = {
        "phase": "collection_in_progress", "scanned": n_scanned,
        "easy_param": len(easy_p), "med_param": len(med_p),
        "hard_param": len(hard_p), "ctxdep": len(ctx),
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(info, f)
    for name, bucket in [("easy", easy_p), ("med", med_p), ("hard", hard_p), ("ctx", ctx)]:
        if bucket:
            np.save(f"/kaggle/working/q14tt_ckpt_{name}.npy",
                    np.stack([x["hs"] for x in bucket]))
    print(f"  [ckpt n={n_scanned}: easy={len(easy_p)} med={len(med_p)} hard={len(hard_p)} ctx={len(ctx)}]",
          flush=True)


# ── Dataset ────────────────────────────────────────────────────────────────────
def load_triviaqa_train():
    print("Loading TriviaQA TRAIN split...", flush=True)
    ds = load_dataset("trivia_qa", "rc.wikipedia", split="train", trust_remote_code=True)
    items = []
    for ex in ds:
        q = ex["question"]
        answers = ex["answer"]["aliases"] if ex["answer"]["aliases"] else [ex["answer"]["value"]]
        passage = ""
        if "search_results" in ex:
            for sr in ex.get("search_results", {}).get("search_context", []):
                if sr:
                    passage = sr
                    break
        if not passage:
            pages = ex.get("entity_pages", {}).get("wiki_context", [])
            if pages:
                passage = pages[0]
        items.append({"question": q, "answers": answers, "passage": passage})
    random.shuffle(items)
    print(f"  Loaded {len(items)} TriviaQA TRAIN items", flush=True)
    return items


# ── Prompts ────────────────────────────────────────────────────────────────────
def fmt_nc(q: str, tok) -> str:
    msgs = [{"role": "user", "content": f"Answer this question in a few words: {q}"}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def fmt_ctx(q: str, passage: str, tok) -> str:
    ctx = passage[:MAX_CTX]
    msgs = [{"role": "user", "content": f"Context: {ctx}\n\nAnswer this question in a few words: {q}"}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


# ── Generation + HS hook ───────────────────────────────────────────────────────
def generate_with_hs(prompt: str, model, tok, layer_idx: int, step: int = 1):
    """Run generation; capture hidden state at `step` (1-indexed) from layer `layer_idx`."""
    inputs = tok(prompt, return_tensors="pt").to(DEVICE)
    captured = {}

    def hook_fn(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        # With KV cache each decode step has shape[1]==1; prefill has shape[1]==input_len.
        # Capture only once at step-1 (first new token, consistent with C001/exp_b).
        if "hs" not in captured and h.shape[1] == 1:
            captured["hs"] = h[0, -1, :].detach().float().cpu().numpy()

    handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_GEN,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tok.eos_token_id,
            )
    finally:
        handle.remove()

    gen_text = tok.decode(out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    hs = captured.get("hs", None)
    return gen_text, hs


def generate_text_only(prompt: str, model, tok):
    """Context pass — text only, no hidden state capture (saves memory)."""
    inputs = tok(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_GEN,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


# ── Probe (global PCA basis) ───────────────────────────────────────────────────
def fit_global_pca(all_hs: np.ndarray, pca_dim: int) -> PCA:
    """Fit PCA on all collected hidden states. Used as shared basis for all probes."""
    pca = PCA(n_components=min(pca_dim, all_hs.shape[1], all_hs.shape[0] - 1))
    pca.fit(all_hs)
    return pca


def run_probe_full(X: np.ndarray, y: np.ndarray, pca: PCA,
                   label: str, train_frac: float = TRAIN_FRAC) -> dict:
    """
    Full unmatched probe (C001 replication).
    Uses fixed 75/25 train/test split — N is large enough here.
    """
    rng = np.random.default_rng(SEED)
    n = len(y)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(train_frac * n)
    tr, te = idx[:n_train], idx[n_train:]

    X_pca = pca.transform(X)
    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    lda.fit(X_pca[tr], y[tr])
    scores = lda.decision_function(X_pca[te])
    auroc = float(roc_auc_score(y[te], scores))

    # Shuffled control
    shuf_aurocs = []
    for _ in range(N_SHUFFLED):
        y_s = y.copy()
        rng.shuffle(y_s)
        lda_s = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        lda_s.fit(X_pca[tr], y_s[tr])
        sc_s = lda_s.decision_function(X_pca[te])
        shuf_aurocs.append(float(roc_auc_score(y_s[te], sc_s)))

    result = {
        "auroc": round(auroc, 4),
        "shuffled_max": round(max(shuf_aurocs), 4),
        "n_total": n,
        "n_test": len(te),
    }
    print(f"  [{label}] AUROC={auroc:.4f}  shuffled_max={max(shuf_aurocs):.4f}  n={n}", flush=True)
    return result


def run_probe_cv(X: np.ndarray, y: np.ndarray, pca: PCA,
                 label: str, n_folds: int = N_CV_FOLDS) -> dict:
    """
    Tier / within-class probe using stratified K-fold CV.
    Uses global PCA (already fitted). Only the LDA step is trained on tier data.
    Best for small N — avoids the unreliable 75/25 split problem.
    """
    n_min = int(min(np.sum(y == 0), np.sum(y == 1)))
    if n_min < N_TIER_MIN:
        print(f"  [{label}] SKIP — n_min={n_min} < {N_TIER_MIN}", flush=True)
        return {"auroc": None, "n_min": n_min,
                "reason": f"n_min={n_min} < N_TIER_MIN={N_TIER_MIN}"}

    X_pca = pca.transform(X)
    actual_folds = min(n_folds, n_min)
    cv = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=SEED)

    fold_aurocs = []
    for tr_idx, te_idx in cv.split(X_pca, y):
        lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        lda.fit(X_pca[tr_idx], y[tr_idx])
        scores = lda.decision_function(X_pca[te_idx])
        fold_aurocs.append(float(roc_auc_score(y[te_idx], scores)))

    auroc_cv = float(np.mean(fold_aurocs))
    auroc_std = float(np.std(fold_aurocs))

    # Shuffled controls (same CV)
    rng = np.random.default_rng(SEED)
    shuf_aurocs = []
    for _ in range(N_SHUFFLED):
        y_s = y.copy(); rng.shuffle(y_s)
        s_folds = []
        for tr_idx, te_idx in cv.split(X_pca, y_s):
            lda_s = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
            lda_s.fit(X_pca[tr_idx], y_s[tr_idx])
            sc_s = lda_s.decision_function(X_pca[te_idx])
            s_folds.append(float(roc_auc_score(y_s[te_idx], sc_s)))
        shuf_aurocs.append(float(np.mean(s_folds)))

    result = {
        "auroc_cv": round(auroc_cv, 4),
        "auroc_std": round(auroc_std, 4),
        "shuffled_max": round(max(shuf_aurocs), 4),
        "n_per_class": n_min,
        "n_folds": actual_folds,
    }
    print(
        f"  [{label}] AUROC_CV={auroc_cv:.4f} ±{auroc_std:.4f}"
        f"  shuffled_max={max(shuf_aurocs):.4f}  n={n_min}  folds={actual_folds}",
        flush=True
    )
    return result


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    t_start = time.time()

    # — Load model —
    hf_tok = _get_hf_token()
    print(f"Loading model {MODEL_ID}...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_tok)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16,
        device_map=None, token=hf_tok
    ).to(DEVICE)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  Loaded. {n_params:.2f}B params", flush=True)

    # — Load dataset (TRAIN split) —
    items = load_triviaqa_train()

    # ── Phase 1: Tier-targeted bilateral oracle collection ──────────────────────
    # Tier buckets: collect PARAM until each tier is satisfied
    easy_param  = []  # context_f1 ≥ 0.80
    med_param   = []  # context_f1 ∈ [0.50, 0.80)
    hard_param  = []  # context_f1 < 0.50
    ctxdep_items = []

    scanned = 0
    t0 = time.time()

    def tiers_satisfied():
        return (
            len(easy_param)  >= N_EASY_PARAM and
            len(med_param)   >= N_MED_PARAM   and
            len(ctxdep_items) >= N_CTX_TARGET
        )

    def n_param():
        return len(easy_param) + len(med_param) + len(hard_param)

    print(
        f"\nRunning tier-targeted bilateral oracle (pool={MAX_SCAN}, train split)...",
        flush=True
    )
    print(
        f"  Targets: easy_param={N_EASY_PARAM}  med_param={N_MED_PARAM}"
        f"  hard_param≤{N_HARD_PARAM}  ctxdep={N_CTX_TARGET}",
        flush=True
    )

    for item in items:
        if tiers_satisfied() or scanned >= MAX_SCAN:
            break
        scanned += 1

        q = item["question"]
        answers = item["answers"]
        passage = item.get("passage", "")

        # No-context pass (always)
        prompt_nc = fmt_nc(q, tok)
        text_nc, hs = generate_with_hs(prompt_nc, model, tok, LAYER_IDX)
        if hs is None:
            continue
        f1_nc = token_f1(text_nc, answers)

        # — PARAM candidate —
        if f1_nc >= PARAM_MIN_F1:
            # Run context pass to get context_f1 (needed for tier assignment)
            if passage:
                prompt_ctx = fmt_ctx(q, passage, tok)
                text_ctx = generate_text_only(prompt_ctx, model, tok)
                ctx_f1 = token_f1(text_ctx, answers)
            else:
                ctx_f1 = f1_nc  # No passage — treat as hard PARAM

            entry = {"hs": hs, "nocontext_f1": f1_nc, "context_f1": ctx_f1,
                     "label": "PARAM"}

            if ctx_f1 >= TIER_EASY_MIN and len(easy_param) < N_EASY_PARAM:
                easy_param.append(entry)
            elif TIER_MED_MIN <= ctx_f1 < TIER_MED_MAX and len(med_param) < N_MED_PARAM:
                med_param.append(entry)
            elif ctx_f1 < TIER_MED_MIN and len(hard_param) < N_HARD_PARAM:
                hard_param.append(entry)

        # — CTX_DEP candidate —
        elif f1_nc <= CTX_MAX_F1 and len(text_nc.split()) >= 2:
            if passage and len(ctxdep_items) < N_CTX_TARGET:
                prompt_ctx = fmt_ctx(q, passage, tok)
                text_ctx = generate_text_only(prompt_ctx, model, tok)
                ctx_f1 = token_f1(text_ctx, answers)
                if ctx_f1 >= CTX_MIN_CTX:
                    ctxdep_items.append({"hs": hs, "nocontext_f1": f1_nc,
                                         "context_f1": ctx_f1, "label": "CTX_DEP"})

        if scanned % 500 == 0 or (scanned <= 2000 and scanned % 100 == 0):
            elapsed = time.time() - t0
            print(
                f"  scanned={scanned}  "
                f"easy_p={len(easy_param)}/{N_EASY_PARAM}  "
                f"med_p={len(med_param)}/{N_MED_PARAM}  "
                f"hard_p={len(hard_param)}  "
                f"ctxdep={len(ctxdep_items)}/{N_CTX_TARGET}  "
                f"({elapsed:.0f}s)",
                flush=True
            )

        # Intermediate checkpoint with HS arrays (crash protection)
        if scanned % CHECKPOINT_INTERVAL == 0:
            _save_checkpoint(easy_param, med_param, hard_param, ctxdep_items, scanned)

    print(
        f"\nCollection complete: "
        f"easy_param={len(easy_param)}  med_param={len(med_param)}  "
        f"hard_param={len(hard_param)}  ctxdep={len(ctxdep_items)}  "
        f"scanned={scanned}",
        flush=True
    )

    # ── Build combined arrays ───────────────────────────────────────────────────
    all_param = easy_param + med_param + hard_param
    all_items = all_param + ctxdep_items

    if len(all_param) < 10 or len(ctxdep_items) < 10:
        print("FATAL: insufficient data to proceed.", flush=True)
        return

    X_all  = np.stack([x["hs"] for x in all_items])
    y_all  = np.array([0 if x["label"] == "CTX_DEP" else 1 for x in all_items])

    X_param = np.stack([x["hs"] for x in all_param])
    X_ctx   = np.stack([x["hs"] for x in ctxdep_items])

    # Save hidden states
    np.save("/kaggle/working/q14tt_X_param.npy", X_param)
    np.save("/kaggle/working/q14tt_X_ctx.npy",   X_ctx)
    print(f"  HS saved: param={X_param.shape}  ctxdep={X_ctx.shape}", flush=True)

    # Free model memory before probing
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ── Fit global PCA on ALL items ─────────────────────────────────────────────
    print("\nFitting global PCA on all collected items...", flush=True)
    pca_global = fit_global_pca(X_all, PCA_DIM_FULL)
    print(f"  PCA fit: input={X_all.shape}  output_dim={pca_global.n_components_}", flush=True)

    results = {
        "collection": {
            "scanned": scanned,
            "easy_param": len(easy_param),
            "med_param": len(med_param),
            "hard_param": len(hard_param),
            "ctxdep_total": len(ctxdep_items),
            "model": MODEL_ID,
            "layer": LAYER_IDX,
            "dataset": "triviaqa_rc_wikipedia_TRAIN",
        },
        "phases": {}
    }

    # ── Phase 2: Full unmatched (C001 replication) ──────────────────────────────
    print("\n" + "="*60, flush=True)
    print("Phase 2: Full unmatched probe (C001 replication)...", flush=True)
    results["phases"]["full_unmatched"] = run_probe_full(X_all, y_all, pca_global, "full_unmatched")

    # ── Phase 3: Difficulty-matched tier probes ──────────────────────────────────
    print("\n" + "="*60, flush=True)
    print("Phase 3: Difficulty-matched tier probes (5-fold CV)...", flush=True)

    # Build CTX_DEP tier arrays
    ctx_f1_arr = np.array([x["context_f1"] for x in ctxdep_items])
    easy_ctx_mask = ctx_f1_arr >= TIER_EASY_MIN
    med_ctx_mask  = (ctx_f1_arr >= TIER_MED_MIN) & (ctx_f1_arr < TIER_MED_MAX)

    easy_ctx_hs  = X_ctx[easy_ctx_mask]
    med_ctx_hs   = X_ctx[med_ctx_mask]

    print(f"  CTX_DEP tiers: easy={easy_ctx_hs.shape[0]}  medium={med_ctx_hs.shape[0]}", flush=True)

    # Easy tier
    n_easy = min(len(easy_param), easy_ctx_hs.shape[0])
    print(f"\n  [Easy tier] PARAM={len(easy_param)}  CTX_DEP={easy_ctx_hs.shape[0]}  using={n_easy}/class", flush=True)
    if n_easy >= N_TIER_MIN:
        ep_hs = np.stack([x["hs"] for x in easy_param[:n_easy]])
        X_easy = np.vstack([ep_hs, easy_ctx_hs[:n_easy]])
        y_easy = np.array([1]*n_easy + [0]*n_easy)
        results["phases"]["tier_easy"] = run_probe_cv(X_easy, y_easy, pca_global, "tier_easy")
    else:
        print(f"  [tier_easy] SKIP — n_easy={n_easy} < {N_TIER_MIN}", flush=True)
        results["phases"]["tier_easy"] = {"auroc": None, "reason": f"n_easy={n_easy} < N_TIER_MIN"}

    # Medium tier
    n_med = min(len(med_param), med_ctx_hs.shape[0])
    print(f"\n  [Medium tier] PARAM={len(med_param)}  CTX_DEP={med_ctx_hs.shape[0]}  using={n_med}/class", flush=True)
    if n_med >= N_TIER_MIN:
        mp_hs = np.stack([x["hs"] for x in med_param[:n_med]])
        X_med = np.vstack([mp_hs, med_ctx_hs[:n_med]])
        y_med = np.array([1]*n_med + [0]*n_med)
        results["phases"]["tier_med"] = run_probe_cv(X_med, y_med, pca_global, "tier_med")
    else:
        print(f"  [tier_med] SKIP — n_med={n_med} < {N_TIER_MIN}", flush=True)
        results["phases"]["tier_med"] = {"auroc": None, "reason": f"n_med={n_med} < N_TIER_MIN"}

    # ── Phase 4: Within-PARAM difficulty (nocontext_f1 tiers) ───────────────────
    print("\n" + "="*60, flush=True)
    print("Phase 4: Within-PARAM difficulty test...", flush=True)

    nc_f1_arr = np.array([x["nocontext_f1"] for x in all_param])
    easy_p_mask = nc_f1_arr >= 0.90
    hard_p_mask = (nc_f1_arr >= 0.50) & (nc_f1_arr < 0.70)

    n_easy_p = int(np.sum(easy_p_mask))
    n_hard_p = int(np.sum(hard_p_mask))
    print(f"  Within-PARAM: easy(≥0.90)={n_easy_p}  hard([0.50,0.70))={n_hard_p}", flush=True)

    n_wp = min(n_easy_p, n_hard_p)
    if n_wp >= N_TIER_MIN:
        ep_wp = X_param[easy_p_mask][:n_wp]
        hp_wp = X_param[hard_p_mask][:n_wp]
        X_wp  = np.vstack([ep_wp, hp_wp])
        y_wp  = np.array([1]*n_wp + [0]*n_wp)
        results["phases"]["within_param"] = run_probe_cv(X_wp, y_wp, pca_global, "within_PARAM")
    else:
        print(f"  [within_PARAM] SKIP — n={n_wp} < {N_TIER_MIN}", flush=True)
        results["phases"]["within_param"] = {"auroc": None, "reason": f"n_wp={n_wp} < N_TIER_MIN"}

    # ── Phase 5: Within-CTX_DEP difficulty (context_f1 tiers) ───────────────────
    print("\n" + "="*60, flush=True)
    print("Phase 5: Within-CTX_DEP difficulty test...", flush=True)

    # Easy CTX_DEP (context_f1 ≥ 0.90) vs Hard CTX_DEP (context_f1 ∈ [0.50, 0.70))
    hard_ctx_mask = (ctx_f1_arr >= TIER_MED_MIN) & (ctx_f1_arr < 0.70)
    very_easy_ctx_mask = ctx_f1_arr >= 0.90

    n_easy_c = int(np.sum(very_easy_ctx_mask))
    n_hard_c = int(np.sum(hard_ctx_mask))
    print(f"  Within-CTX_DEP: easy(≥0.90)={n_easy_c}  hard([0.50,0.70))={n_hard_c}", flush=True)

    n_wc = min(n_easy_c, n_hard_c)
    if n_wc >= N_TIER_MIN:
        ec_wc = X_ctx[very_easy_ctx_mask][:n_wc]
        hc_wc = X_ctx[hard_ctx_mask][:n_wc]
        X_wc  = np.vstack([ec_wc, hc_wc])
        y_wc  = np.array([1]*n_wc + [0]*n_wc)
        results["phases"]["within_ctxdep"] = run_probe_cv(X_wc, y_wc, pca_global, "within_CTX_DEP")
    else:
        print(f"  [within_CTX_DEP] SKIP — n={n_wc} < {N_TIER_MIN}", flush=True)
        results["phases"]["within_ctxdep"] = {"auroc": None, "reason": f"n_wc={n_wc} < N_TIER_MIN"}

    # ── Descriptive stats ───────────────────────────────────────────────────────
    param_nc_f1  = [x["nocontext_f1"] for x in all_param]
    param_ctx_f1 = [x["context_f1"] for x in all_param]
    ctx_f1_list  = [x["context_f1"] for x in ctxdep_items]
    results["descriptive"] = {
        "param_nocontext_f1_mean": round(float(np.mean(param_nc_f1)), 3),
        "param_nocontext_f1_std":  round(float(np.std(param_nc_f1)), 3),
        "param_context_f1_mean":   round(float(np.mean(param_ctx_f1)), 3),
        "param_context_f1_std":    round(float(np.std(param_ctx_f1)), 3),
        "ctxdep_context_f1_mean":  round(float(np.mean(ctx_f1_list)), 3),
        "ctxdep_context_f1_std":   round(float(np.std(ctx_f1_list)), 3),
    }

    # ── Verdict ──────────────────────────────────────────────────────────────────
    def get_auroc(phase_key, field="auroc_cv"):
        r = results["phases"].get(phase_key, {})
        return r.get(field) or r.get("auroc")

    au_easy   = get_auroc("tier_easy")
    au_med    = get_auroc("tier_med")
    au_wp     = get_auroc("within_param")
    au_wc     = get_auroc("within_ctxdep")
    au_full   = results["phases"].get("full_unmatched", {}).get("auroc")

    if au_easy is None or au_med is None:
        verdict = "INSUFFICIENT_DATA"
        detail = "One or more tier probes could not run (N < N_TIER_MIN)"
    elif au_easy >= 0.60 and au_med >= 0.55 and (au_wp or 1.0) <= 0.62 and (au_wc or 1.0) <= 0.62:
        verdict = "H_EPISTEMIC"
        detail = f"Signal persists at matched difficulty: easy={au_easy:.3f} med={au_med:.3f}"
    elif au_easy < 0.55 or (au_wp is not None and au_wp >= 0.65):
        verdict = "H_DIFFICULTY"
        detail = f"Difficulty confound detected: easy_tier={au_easy} within_param={au_wp}"
    else:
        verdict = "AMBIGUOUS"
        detail = f"easy={au_easy:.3f} med={au_med:.3f} wp={au_wp} wc={au_wc}"

    results["verdict"] = verdict
    results["verdict_detail"] = detail
    results["runtime_seconds"] = round(time.time() - t_start, 1)

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "="*60, flush=True)
    print("RESULTS SUMMARY", flush=True)
    print("="*60, flush=True)
    d = results["descriptive"]
    print(f"  PARAM  nocontext_f1: mean={d['param_nocontext_f1_mean']}  std={d['param_nocontext_f1_std']}", flush=True)
    print(f"  PARAM  context_f1:   mean={d['param_context_f1_mean']}  std={d['param_context_f1_std']}", flush=True)
    print(f"  CTX_DEP context_f1:  mean={d['ctxdep_context_f1_mean']}  std={d['ctxdep_context_f1_std']}", flush=True)
    print(f"\n  Full unmatched AUROC:  {au_full}", flush=True)
    print(f"  Easy tier AUROC_CV:    {au_easy}", flush=True)
    print(f"  Medium tier AUROC_CV:  {au_med}", flush=True)
    print(f"  Within-PARAM AUROC_CV: {au_wp}", flush=True)
    print(f"  Within-CTX_DEP:        {au_wc}", flush=True)
    print(f"\n  VERDICT: {verdict}", flush=True)
    print(f"  {detail}", flush=True)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Output: {OUTPUT_FILE}", flush=True)
    print("="*60, flush=True)


if __name__ == "__main__":
    main()
