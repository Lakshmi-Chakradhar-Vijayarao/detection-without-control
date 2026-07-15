#!/usr/bin/env python3
"""
false_certainty_llama_v1.py — EXP-A Llama Replication: False Certainty Detection

SCIENTIFIC QUESTION:
  C017: Does Fisher+PCA64 AUROC > 0.70 separate CC vs CW in the entropy-matched
        confident zone for Llama-3.2-3B-Instruct?
  C018: Does the bilateral oracle (PARAM vs CTX_DEP) probe transfer to CC/CW
        (BO_Transfer AUROC > 0.70)?

Reference (Qwen2.5-1.5B-Instruct):
  Fisher=0.8544, Entropy=0.6144, BO_Transfer=0.8800, Gap=0.240

DESIGN:
  THETA_CONF = 1.2000  (starting estimate for Llama confident-zone entropy)
  Entropy window: [THETA_CONF - ENT_HALF, THETA_CONF + ENT_HALF]  i.e. [0.90, 1.50]
  CC (CONFIDENT_CORRECT): correct answers (F1 >= 0.50) with entropy in window
  CW (CONFIDENT_WRONG):   wrong answers   (F1 <= 0.05) with entropy in window

  Phase 1 — Bilateral Oracle Calibration (first 2000 items):
    Collect PARAM (correct nocontext) and CTX_DEP (wrong nocontext / right with ctx).
    Fit PCA(64)+LDA probe on PARAM vs CTX_DEP.
    Sanity check: bilateral oracle AUROC expected ~0.73 (from Qwen reference).

  Phase 2 — CC/CW Collection (all 5000 items):
    Per item: extract step-1 entropy at L26.
    If entropy in [0.90, 1.50]: generate answer, compute token F1.
    Classify as CC or CW; stop at N_CC_TARGET=80 and N_CW_TARGET=80.
    Intermediate saves every 20 collected items.

  Phase 3 — Head-to-Head:
    Train/test split (TRAIN_FRAC=0.75).
    a. Fisher+PCA64 AUROC (new probe fitted on CC/CW train set)
    b. Entropy AUROC     (direct entropy discrimination; ~0.50 = window matched)
    c. gap = Fisher - Entropy
    d. BO_Transfer AUROC (Phase-1 bilateral oracle probe applied to CC/CW test)

VERDICT:
  SUPPORTED     : Fisher AUROC >= 0.70 AND gap >= 0.10
  NOT_SUPPORTED : otherwise

Probe layer: L26 (0-indexed), hook on model.model.layers[26].
GPU: T4. Expected ~2-3h.
"""

from __future__ import annotations
import gc, json, os, random, sys, time
import numpy as np
import torch
from datasets import load_dataset
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_ID       = "meta-llama/Llama-3.2-3B-Instruct"
THETA_CONF     = 1.2000        # confident-zone entropy threshold
ENT_HALF       = 0.30          # half-width of entropy window
ENT_LO         = THETA_CONF - ENT_HALF    # 0.90
ENT_HI         = THETA_CONF + ENT_HALF    # 1.50
LAYER_IDX      = 26            # probe layer (0-indexed); Llama-3.2-3B has 28 layers
N_CC_TARGET    = 80            # max CONFIDENT_CORRECT items to collect
N_CW_TARGET    = 80            # max CONFIDENT_WRONG items to collect
POOL_SIZE      = 5_000         # total TriviaQA items to load
N_BO_EACH      = 100           # per-class items for bilateral oracle calibration
BO_POOL_SLICE  = 2_000         # first N pool items used for Phase 1
TRAIN_FRAC     = 0.75
PCA_DIM        = 64
SEED           = 42
MAX_NEW        = 60

PARAM_MIN_F1   = 0.50          # CC / PARAM threshold
CW_MAX_F1      = 0.05          # CW / false-certainty threshold
CTX_MIN_F1     = 0.50          # CTX_DEP withcontext threshold

SAVE_PATH      = "/kaggle/working/exp_false_certainty_llama_results.json"

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    raise RuntimeError("GPU required — no CUDA device found.")
print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB",
      flush=True)
print(f"Entropy window: [{ENT_LO:.4f}, {ENT_HI:.4f}]  (THETA_CONF={THETA_CONF})",
      flush=True)


# ── HF token / model path resolution ──────────────────────────────────────────
def _get_hf_token() -> str | None:
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        v = os.environ.get(k)
        if v:
            return v
    try:
        from kaggle_secrets import UserSecretsClient
        v = UserSecretsClient().get_secret("HF_TOKEN")
        if v:
            return v
    except Exception:
        pass
    return None


def _resolve_model_path() -> str:
    """Return local Kaggle model mount path or fall back to HF Hub id."""
    if os.path.exists("/kaggle/input"):
        for d in sorted(os.listdir("/kaggle/input")):
            if "llama" in d.lower():
                for sub in [
                    "",
                    "/transformers/3b-instruct/1",
                    "/transformers/8b-instruct/1",
                    "/1",
                ]:
                    p = f"/kaggle/input/{d}{sub}"
                    if os.path.isdir(p):
                        print(f"  Found local model at {p}", flush=True)
                        return p
    tok = _get_hf_token()
    if tok:
        from huggingface_hub import login
        login(token=tok, add_to_git_credential=False)
    return MODEL_ID


# ── Model setup ───────────────────────────────────────────────────────────────
def setup_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = _resolve_model_path()
    print(f"\nLoading {model_path} …", flush=True)

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    mdl = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=None,
        trust_remote_code=True,
    ).to(DEVICE).eval()

    n_layers = mdl.config.num_hidden_layers
    d_model  = mdl.config.hidden_size
    print(f"  n_layers={n_layers}  d_model={d_model}", flush=True)
    assert LAYER_IDX < n_layers, (
        f"LAYER_IDX={LAYER_IDX} >= n_layers={n_layers}"
    )
    return mdl, tok


# ── Prompt formatting for Llama-3.2-3B-Instruct ───────────────────────────────
def fmt_prompt(q: str) -> str:
    return (
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{q}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def fmt_ctx_prompt(q: str, ctx: str) -> str:
    body = f"Context: {ctx}\n\nAnswer in one short phrase.\n{q}"
    return (
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{body}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )


# ── Data loading ──────────────────────────────────────────────────────────────
def load_pool(n: int) -> list:
    ds = load_dataset("trivia_qa", "rc.wikipedia", split="validation")
    items = []
    for ex in ds:
        ep  = ex.get("entity_pages", {})
        ctx = (ep.get("wiki_context") or [""])[0][:800] if ep else ""
        ans = ex["answer"]["aliases"] or [ex["answer"]["value"]]
        items.append({"question": ex["question"], "context": ctx, "answers": ans})
        if len(items) >= n:
            break
    random.shuffle(items)
    print(f"Pool loaded: {len(items)} items", flush=True)
    return items


def normalize(s: str) -> str:
    import re
    return re.sub(r"[^\w\s]", " ", s.lower()).strip()


def token_f1(pred: str, golds: list) -> float:
    pt   = set(normalize(pred).split())
    best = 0.0
    for g in golds:
        gt = set(normalize(g).split())
        if not pt or not gt:
            continue
        c = pt & gt
        if not c:
            continue
        p    = len(c) / len(pt)
        r    = len(c) / len(gt)
        best = max(best, 2 * p * r / (p + r))
    return best


def gen_text(model, tok, prompt: str) -> str:
    inp = tok(prompt, return_tensors="pt", truncation=True,
               max_length=512).to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **inp,
            max_new_tokens=MAX_NEW,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
            use_cache=True,
        )
    return tok.decode(
        out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()


# ── Hidden state + entropy extraction at L26, step-1 ─────────────────────────
def extract_step1_hs_entropy(model, tok, question: str) -> dict:
    """
    Explicit 2-step forward pass to extract step-1 hidden state at LAYER_IDX.
    Pass 1: prefill (no HS) → KV cache + first-token prediction.
    Pass 2: decode step-1 token with KV cache (output_hidden_states=True).

    Returns:
      hs      : np.ndarray shape (d_model,) or None on failure
      entropy : Shannon entropy of the step-1 output distribution
    """
    prompt = fmt_prompt(question)
    inp    = tok(prompt, return_tensors="pt", truncation=True,
                  max_length=512).to(DEVICE)
    try:
        with torch.no_grad():
            # Pass 1: prefill — get KV cache + first predicted token
            prefill = model(
                **inp,
                output_hidden_states=False,
                use_cache=True,
                return_dict=True,
            )
            next_tok = prefill.logits[0, -1, :].argmax().unsqueeze(0).unsqueeze(0)
            # Pass 2: step-1 generation with KV cache
            gen = model(
                input_ids=next_tok,
                past_key_values=prefill.past_key_values,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        # hidden_states[0]=embeddings, [k]=output of layer k-1 (0-indexed)
        # LAYER_IDX=26 → index 27
        hs = gen.hidden_states[LAYER_IDX + 1][0, 0, :].detach().float().cpu().numpy()
        # Entropy from step-1 logits
        logits  = gen.logits[0, 0, :].float()
        probs   = torch.softmax(logits, dim=-1).clamp(min=1e-10)
        entropy = float(-torch.sum(probs * torch.log(probs)).item())
        return {"hs": hs, "entropy": entropy}
    except Exception as e:
        print(f"  WARN: extract_step1_hs_entropy failed: {e}", flush=True)
        return {"hs": None, "entropy": None}


# ── Probe utilities ────────────────────────────────────────────────────────────
def pca_lda_auroc(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
):
    """
    Fit PCA(PCA_DIM)+LDA on training data; evaluate on test data.
    Returns (auroc, shuffled_control, pca_obj, lda_obj).
    """
    n_comp = min(PCA_DIM, X_tr.shape[1], X_tr.shape[0] - 1)
    pca    = PCA(n_components=n_comp, random_state=SEED)
    X_tr_r = pca.fit_transform(X_tr.astype(np.float32))
    X_te_r = pca.transform(X_te.astype(np.float32))

    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    lda.fit(X_tr_r, y_tr)
    scores = lda.decision_function(X_te_r)

    auroc  = float(roc_auc_score(y_te, scores))
    y_shuf = y_te.copy()
    np.random.shuffle(y_shuf)
    shuffled = float(roc_auc_score(y_shuf, scores))

    return round(auroc, 4), round(shuffled, 4), pca, lda


def entropy_auroc(ent_cc: list, ent_cw: list) -> float:
    """
    Direct entropy discrimination: lower entropy favours CC (label=1).
    Score = -entropy so that higher score = more likely CC.
    """
    y  = np.array([1] * len(ent_cc) + [0] * len(ent_cw))
    sc = np.concatenate([-np.array(ent_cc), -np.array(ent_cw)])
    return round(float(roc_auc_score(y, sc)), 4)


# ── Intermediate save ─────────────────────────────────────────────────────────
def _save_partial(data: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ── Phase 1: Bilateral Oracle Calibration ─────────────────────────────────────
def phase1_bilateral_oracle(model, tok, pool: list) -> dict:
    """
    Scan pool[:BO_POOL_SLICE] for PARAM and CTX_DEP items.
    Fit PCA(64)+LDA probe on PARAM (label=1) vs CTX_DEP (label=0).
    """
    print(f"\n=== Phase 1: Bilateral Oracle Calibration ===", flush=True)
    print(f"  Using first {min(BO_POOL_SLICE, len(pool))} items "
          f"(N_BO_EACH={N_BO_EACH} per class)", flush=True)
    t0 = time.time()

    param_hs:  list = []
    ctxdep_hs: list = []

    for i, item in enumerate(pool[:BO_POOL_SLICE]):
        n_before = len(param_hs) + len(ctxdep_hs)

        if len(param_hs) >= N_BO_EACH and len(ctxdep_hs) >= N_BO_EACH:
            print(f"  Targets reached at item {i}.", flush=True)
            break

        if i % 50 == 0:
            print(
                f"  [{i}/{min(BO_POOL_SLICE, len(pool))}] "
                f"PARAM={len(param_hs)} CTX_DEP={len(ctxdep_hs)} "
                f"elapsed={time.time()-t0:.0f}s",
                flush=True,
            )

        # Generate answer without context
        pred_nc = gen_text(model, tok, fmt_prompt(item["question"]))
        f1_nc   = token_f1(pred_nc, item["answers"])

        item_added = False

        # PARAM check: correct without context
        if f1_nc >= PARAM_MIN_F1 and len(param_hs) < N_BO_EACH:
            sig = extract_step1_hs_entropy(model, tok, item["question"])
            if sig["hs"] is not None:
                param_hs.append(sig["hs"])
                item_added = True

        # CTX_DEP check: wrong without context, right with context
        elif f1_nc <= CW_MAX_F1 and item["context"] and len(ctxdep_hs) < N_BO_EACH:
            pred_ctx = gen_text(
                model, tok,
                fmt_ctx_prompt(item["question"], item["context"]),
            )
            if token_f1(pred_ctx, item["answers"]) >= CTX_MIN_F1:
                sig = extract_step1_hs_entropy(model, tok, item["question"])
                if sig["hs"] is not None:
                    ctxdep_hs.append(sig["hs"])
                    item_added = True

        # Intermediate save every 20 collected (guard: only when something was added)
        if item_added:
            n_after = len(param_hs) + len(ctxdep_hs)
            if n_after > n_before and n_after % 20 == 0:
                _save_partial(
                    {
                        "phase":    "phase1_partial",
                        "n_param":  len(param_hs),
                        "n_ctxdep": len(ctxdep_hs),
                        "elapsed_s": round(time.time() - t0, 1),
                    },
                    SAVE_PATH,
                )
                print(
                    f"  [SAVE] phase1 n_param={len(param_hs)} "
                    f"n_ctxdep={len(ctxdep_hs)}",
                    flush=True,
                )

    n_p, n_c = len(param_hs), len(ctxdep_hs)
    print(
        f"  Final: PARAM={n_p}  CTX_DEP={n_c}  elapsed={time.time()-t0:.0f}s",
        flush=True,
    )

    if n_p < 10 or n_c < 10:
        print("  WARN: insufficient samples for bilateral oracle — skipping.", flush=True)
        return {"ok": False, "n_param": n_p, "n_ctxdep": n_c}

    n_tr_p = int(n_p * TRAIN_FRAC)
    n_tr_c = int(n_c * TRAIN_FRAC)
    X_tr   = np.vstack(param_hs[:n_tr_p] + ctxdep_hs[:n_tr_c])
    y_tr   = np.array([1] * n_tr_p + [0] * n_tr_c)
    X_te   = np.vstack(param_hs[n_tr_p:] + ctxdep_hs[n_tr_c:])
    y_te   = np.array([1] * (n_p - n_tr_p) + [0] * (n_c - n_tr_c))

    bo_auroc, bo_shuf, bo_pca, bo_lda = pca_lda_auroc(X_tr, y_tr, X_te, y_te)
    print(
        f"  Bilateral Oracle AUROC: {bo_auroc:.4f}  shuffled={bo_shuf:.4f}",
        flush=True,
    )

    return {
        "ok":          True,
        "n_param":     n_p,
        "n_ctxdep":    n_c,
        "bo_auroc":    bo_auroc,
        "bo_shuffled": bo_shuf,
        "pca":         bo_pca,
        "lda":         bo_lda,
    }


# ── Phase 2: CC/CW Collection ─────────────────────────────────────────────────
def phase2_collect_cc_cw(model, tok, pool: list) -> dict:
    """
    Scan all POOL_SIZE items.
    For each item:
      1. Extract step-1 entropy at L26.
      2. If entropy in [ENT_LO, ENT_HI]: generate answer, compute F1.
      3. Classify as CC (F1 >= 0.50) or CW (F1 <= 0.05).
    Stop when N_CC_TARGET and N_CW_TARGET are both reached.
    Intermediate save every 20 collected items.
    """
    print(f"\n=== Phase 2: CC/CW Collection ===", flush=True)
    print(
        f"  Entropy window: [{ENT_LO:.4f}, {ENT_HI:.4f}]  "
        f"Targets: CC={N_CC_TARGET} CW={N_CW_TARGET}",
        flush=True,
    )
    t0 = time.time()

    cc_recs:  list = []
    cw_recs:  list = []
    n_scanned  = 0
    n_in_window = 0

    for i, item in enumerate(pool):
        n_before = len(cc_recs) + len(cw_recs)

        if len(cc_recs) >= N_CC_TARGET and len(cw_recs) >= N_CW_TARGET:
            print(f"  Both targets reached at item {i}.", flush=True)
            break

        if i % 50 == 0:
            print(
                f"  [{i}/{len(pool)}] CC={len(cc_recs)} CW={len(cw_recs)} "
                f"in_window={n_in_window} elapsed={time.time()-t0:.0f}s",
                flush=True,
            )

        n_scanned += 1

        # Step 1: extract step-1 entropy (and hidden state)
        sig = extract_step1_hs_entropy(model, tok, item["question"])
        if sig["hs"] is None:
            continue
        ent = sig["entropy"]

        # Step 2: entropy window filter
        if not (ENT_LO <= ent <= ENT_HI):
            continue
        n_in_window += 1

        # Step 3: generate answer and compute F1
        pred = gen_text(model, tok, fmt_prompt(item["question"]))
        f1   = token_f1(pred, item["answers"])

        # Step 4: classify
        item_added = False
        if f1 >= PARAM_MIN_F1 and len(cc_recs) < N_CC_TARGET:
            cc_recs.append({"hs": sig["hs"], "entropy": ent, "f1": f1})
            item_added = True
        elif f1 <= CW_MAX_F1 and len(cw_recs) < N_CW_TARGET:
            cw_recs.append({"hs": sig["hs"], "entropy": ent, "f1": f1})
            item_added = True

        # Intermediate save every 20 collected (guard: only when something was added)
        if item_added:
            n_after = len(cc_recs) + len(cw_recs)
            if n_after > n_before and n_after % 20 == 0:
                _save_partial(
                    {
                        "phase":       "phase2_partial",
                        "n_cc":        len(cc_recs),
                        "n_cw":        len(cw_recs),
                        "n_scanned":   n_scanned,
                        "n_in_window": n_in_window,
                        "elapsed_s":   round(time.time() - t0, 1),
                    },
                    SAVE_PATH,
                )
                print(
                    f"  [SAVE] phase2 CC={len(cc_recs)} CW={len(cw_recs)}",
                    flush=True,
                )

    n_cc, n_cw = len(cc_recs), len(cw_recs)
    print(
        f"\n  CC={n_cc}  CW={n_cw}  scanned={n_scanned} "
        f"in_window={n_in_window}  elapsed={time.time()-t0:.0f}s",
        flush=True,
    )

    if n_cc < 20 or n_cw < 20:
        print(
            f"  WARN: insufficient samples — CC={n_cc} CW={n_cw}. "
            f"Consider widening ENT_HALF or increasing POOL_SIZE.",
            flush=True,
        )
        return {
            "ok":          False,
            "n_cc":        n_cc,
            "n_cw":        n_cw,
            "n_scanned":   n_scanned,
            "n_in_window": n_in_window,
        }

    cc_ent = [r["entropy"] for r in cc_recs]
    cw_ent = [r["entropy"] for r in cw_recs]
    print(
        f"  CC entropy: mean={np.mean(cc_ent):.4f}  std={np.std(cc_ent):.4f}",
        flush=True,
    )
    print(
        f"  CW entropy: mean={np.mean(cw_ent):.4f}  std={np.std(cw_ent):.4f}",
        flush=True,
    )

    return {
        "ok":              True,
        "cc_recs":         cc_recs,
        "cw_recs":         cw_recs,
        "n_cc":            n_cc,
        "n_cw":            n_cw,
        "n_scanned":       n_scanned,
        "n_in_window":     n_in_window,
        "cc_entropy_mean": round(float(np.mean(cc_ent)), 4),
        "cw_entropy_mean": round(float(np.mean(cw_ent)), 4),
    }


# ── Phase 3: Head-to-Head Comparison ──────────────────────────────────────────
def phase3_compare(p2: dict, p1: dict) -> dict:
    """
    a. Fisher+PCA64 AUROC — new probe fitted on CC/CW train set
    b. Entropy AUROC      — direct entropy discrimination
    c. gap = Fisher - Entropy
    d. BO_Transfer AUROC  — Phase-1 bilateral oracle probe applied to CC/CW test
    """
    print(f"\n=== Phase 3: Head-to-Head Comparison ===", flush=True)

    cc_recs, cw_recs = p2["cc_recs"], p2["cw_recs"]
    n_cc, n_cw       = len(cc_recs), len(cw_recs)
    n_tr_cc          = int(n_cc * TRAIN_FRAC)
    n_tr_cw          = int(n_cw * TRAIN_FRAC)
    n_te_cc          = n_cc - n_tr_cc
    n_te_cw          = n_cw - n_tr_cw

    print(
        f"  Split: train CC={n_tr_cc} CW={n_tr_cw}  "
        f"test CC={n_te_cc} CW={n_te_cw}",
        flush=True,
    )

    # Build arrays
    hs_cc_tr = np.vstack([r["hs"] for r in cc_recs[:n_tr_cc]])
    hs_cw_tr = np.vstack([r["hs"] for r in cw_recs[:n_tr_cw]])
    hs_cc_te = np.vstack([r["hs"] for r in cc_recs[n_tr_cc:]])
    hs_cw_te = np.vstack([r["hs"] for r in cw_recs[n_tr_cw:]])

    X_tr = np.vstack([hs_cc_tr, hs_cw_tr])
    y_tr = np.array([1] * n_tr_cc + [0] * n_tr_cw)
    X_te = np.vstack([hs_cc_te, hs_cw_te])
    y_te = np.array([1] * n_te_cc + [0] * n_te_cw)

    # (a) Fisher+PCA64 — new probe on CC/CW
    fisher_auroc_val, fisher_shuf, _, _ = pca_lda_auroc(X_tr, y_tr, X_te, y_te)
    print(
        f"  Fisher+PCA64 AUROC: {fisher_auroc_val:.4f}  "
        f"shuffled={fisher_shuf:.4f}",
        flush=True,
    )

    # (b) Entropy AUROC (should be ~0.50 if window matched correctly)
    ent_cc_te = [r["entropy"] for r in cc_recs[n_tr_cc:]]
    ent_cw_te = [r["entropy"] for r in cw_recs[n_tr_cw:]]
    entropy_auroc_val = entropy_auroc(ent_cc_te, ent_cw_te)
    print(
        f"  Entropy AUROC: {entropy_auroc_val:.4f}  "
        f"(~0.50 = window matched; >0.65 = window too wide)",
        flush=True,
    )

    # (c) Gap
    gap = round(fisher_auroc_val - entropy_auroc_val, 4)
    print(f"  Gap (Fisher - Entropy): {gap:.4f}", flush=True)

    # (d) BO_Transfer — bilateral oracle probe from Phase 1
    bo_transfer = None
    if p1.get("ok"):
        try:
            X_te_bo   = p1["pca"].transform(X_te.astype(np.float32))
            bo_scores = p1["lda"].decision_function(X_te_bo)
            bo_transfer = round(float(roc_auc_score(y_te, bo_scores)), 4)
            print(
                f"  BO_Transfer AUROC: {bo_transfer:.4f}  "
                f"(bilateral oracle probe applied to CC/CW test)",
                flush=True,
            )
        except Exception as e:
            print(f"  BO_Transfer error: {e}", flush=True)
    else:
        print("  BO_Transfer: skipped (Phase 1 failed).", flush=True)

    # Verdict
    c017_supported = fisher_auroc_val >= 0.70 and gap >= 0.10
    c018_supported = bo_transfer is not None and bo_transfer >= 0.70

    c017 = "SUPPORTED" if c017_supported else "NOT_SUPPORTED"
    c018 = "SUPPORTED" if c018_supported else "NOT_SUPPORTED"
    verdict = "SUPPORTED" if c017_supported else "NOT_SUPPORTED"

    print(f"\n  VERDICT: {verdict}", flush=True)
    print(
        f"  C017 (Fisher>=0.70 AND gap>=0.10): {c017}  "
        f"[Fisher={fisher_auroc_val:.4f} gap={gap:.4f}]",
        flush=True,
    )
    print(
        f"  C018 (BO_Transfer>=0.70):          {c018}  "
        f"[BO_Transfer={bo_transfer}]",
        flush=True,
    )

    # Compare with Qwen reference
    print(f"\n  --- Qwen reference vs Llama ---", flush=True)
    print(f"  Fisher:       Qwen=0.8544  Llama={fisher_auroc_val:.4f}", flush=True)
    print(f"  Entropy:      Qwen=0.6144  Llama={entropy_auroc_val:.4f}", flush=True)
    print(f"  Gap:          Qwen=0.2400  Llama={gap:.4f}", flush=True)
    print(f"  BO_Transfer:  Qwen=0.8800  Llama={bo_transfer}", flush=True)

    return {
        "n_cc":              n_cc,
        "n_cw":              n_cw,
        "n_train_per_class": min(n_tr_cc, n_tr_cw),
        "n_test_per_class":  min(n_te_cc, n_te_cw),
        "fisher_auroc":      fisher_auroc_val,
        "fisher_shuffled":   fisher_shuf,
        "entropy_auroc":     entropy_auroc_val,
        "gap":               gap,
        "bo_transfer_auroc": bo_transfer,
        "verdict":           verdict,
        "c017":              c017,
        "c018":              c018,
    }


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*65}", flush=True)
    print(f"EXP-A Llama Replication: False Certainty Detection", flush=True)
    print(f"Model:  {MODEL_ID}", flush=True)
    print(f"Config: THETA_CONF={THETA_CONF}  LAYER={LAYER_IDX}  "
          f"PCA_DIM={PCA_DIM}  SEED={SEED}", flush=True)
    print(f"{'='*65}", flush=True)
    t_start = time.time()

    # Load model
    model, tok = setup_model()
    gc.collect()
    torch.cuda.empty_cache()

    # Load pool
    pool = load_pool(POOL_SIZE)

    # Phase 1 — Bilateral Oracle Calibration
    p1 = phase1_bilateral_oracle(model, tok, pool)
    gc.collect()

    # Phase 2 — CC/CW Collection
    p2 = phase2_collect_cc_cw(model, tok, pool)
    del pool
    gc.collect()

    if not p2["ok"]:
        print("FATAL: Phase 2 failed to collect enough items.", flush=True)
        partial = {
            "experiment":   "EXP_A_LLAMA_REPLICATION",
            "status":       "INCOMPLETE — insufficient CC/CW items",
            "phase1_ok":    p1.get("ok", False),
            "phase2_n_cc":  p2.get("n_cc", 0),
            "phase2_n_cw":  p2.get("n_cw", 0),
            "phase2_n_scanned":   p2.get("n_scanned", 0),
            "phase2_n_in_window": p2.get("n_in_window", 0),
            "suggestion":   (
                "Widen ENT_HALF (currently 0.30) or increase POOL_SIZE; "
                "reported n_in_window shows how many items passed the filter."
            ),
        }
        _save_partial(partial, SAVE_PATH)
        print(f"Partial results saved to {SAVE_PATH}", flush=True)
        sys.exit(1)

    # Phase 3 — Head-to-Head Comparison
    p3 = phase3_compare(p2, p1)

    elapsed = time.time() - t_start
    print(f"\nTotal elapsed: {elapsed / 60:.1f} min", flush=True)

    # Assemble and save final results
    results = {
        "experiment": "EXP_A_LLAMA_REPLICATION",
        "model":      MODEL_ID,
        "config": {
            "theta_conf":   THETA_CONF,
            "ent_half":     ENT_HALF,
            "ent_lo":       ENT_LO,
            "ent_hi":       ENT_HI,
            "layer_idx":    LAYER_IDX,
            "n_cc_target":  N_CC_TARGET,
            "n_cw_target":  N_CW_TARGET,
            "pool_size":    POOL_SIZE,
            "bo_pool_slice": BO_POOL_SLICE,
            "n_bo_each":    N_BO_EACH,
            "train_frac":   TRAIN_FRAC,
            "pca_dim":      PCA_DIM,
            "seed":         SEED,
            "max_new":      MAX_NEW,
        },
        "phase1": {
            "ok":          p1.get("ok"),
            "n_param":     p1.get("n_param"),
            "n_ctxdep":    p1.get("n_ctxdep"),
            "bo_auroc":    p1.get("bo_auroc"),
            "bo_shuffled": p1.get("bo_shuffled"),
        },
        "phase2": {
            "n_cc":            p2["n_cc"],
            "n_cw":            p2["n_cw"],
            "n_scanned":       p2.get("n_scanned"),
            "n_in_window":     p2.get("n_in_window"),
            "cc_entropy_mean": p2.get("cc_entropy_mean"),
            "cw_entropy_mean": p2.get("cw_entropy_mean"),
        },
        "phase3":      p3,
        "verdict":     p3["verdict"],
        "c017":        p3["c017"],
        "c018":        p3["c018"],
        "elapsed_min": round(elapsed / 60, 1),
        "reference_qwen": {
            "model":        "Qwen2.5-1.5B-Instruct",
            "fisher_auroc": 0.8544,
            "entropy_auroc": 0.6144,
            "bo_transfer":  0.8800,
            "gap":          0.240,
        },
    }

    _save_partial(results, SAVE_PATH)
    print(f"\nResults saved to {SAVE_PATH}", flush=True)

    print("\n--- FINAL SUMMARY ---", flush=True)
    summary = {
        "fisher_auroc":      p3["fisher_auroc"],
        "entropy_auroc":     p3["entropy_auroc"],
        "gap":               p3["gap"],
        "bo_transfer_auroc": p3["bo_transfer_auroc"],
        "verdict":           p3["verdict"],
        "c017":              p3["c017"],
        "c018":              p3["c018"],
    }
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
