#!/usr/bin/env python3
"""
perturbation_battery_v1.py — EXP-J Replication: Perturbation Invariance on Llama-3.2-3B-Instruct

SCIENTIFIC QUESTION:
  Is the Fisher+PCA64 decision score at L26 step-1 invariant under surface-level
  question perturbations on Llama-3.2-3B-Instruct? C025 established ICC=0.913 on
  Qwen2.5-1.5B-Instruct (N=80/class). This run replicates on a second independent
  architecture to strengthen C025 toward CONFIRMED status.

KILL CRITERION:
  within-question score variance >= 50% of between-question variance
  → C025 architecture generalization fails.

DESIGN:
  Phase 1 (Bilateral Oracle Calibration, POOL_SIZE items):
    Collect N_PARAM_TARGET PARAM items (F1 >= 0.50 without context) and
    N_CTX_DEP_TARGET CTX_DEP items (fails without, succeeds with context).
    Extract L26 step-1 hidden states via forward hook (max_new_tokens=1).
    Fit Fisher+PCA64 (PCA then LDA) probe on train split.
    Record calibration AUROC as sanity check (expect ~0.74 for Llama-3.2-3B).

  Phase 2 (Perturbation Battery):
    For each collected item:
      - Original prompt → L26 step-1 HS → Fisher score
      - 4 perturbations (REPHRASE, LOWERCASE, APPEND, TYPO) → HS → score each
    Record all 5 scores per item.
    Intermediate save every 20 items.

  Phase 3 (Analysis):
    - ICC = between_var / (between_var + mean_within_var)
    - Paired t-test: PARAM vs CTX_DEP variant scores remain separated
    - Per-variant correlation with original score
    - Verdict: ROBUST (ICC >= 0.70) / BORDERLINE (>= 0.50) / FRAGILE (< 0.50)
    - Kill criterion: mean_within >= 0.50 * between_var

PERTURBATION TYPES:
  REPHRASE  — surface-level question reword (no LLM needed)
  LOWERCASE — entire question in lowercase
  APPEND    — neutral suffix "Please answer briefly."
  TYPO      — drop 3rd character of the longest word

CONFIG:
  MODEL_ID         = "meta-llama/Llama-3.2-3B-Instruct"
  POOL_SIZE        = 3000
  N_PARAM_TARGET   = 80    (per class)
  N_CTX_DEP_TARGET = 80    (per class)
  N_VARIANTS       = 4
  LAYER_IDX        = 26
  PCA_DIM          = 64
  SEED             = 42

OUTPUT: /kaggle/working/exp_j_llama_results.json

GPU: T4. Expected ~3-4h.
"""

from __future__ import annotations
import gc, json, os, random, re, sys, time
import numpy as np
import torch
from datasets import load_dataset
from scipy.stats import ttest_ind
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_ID         = "meta-llama/Llama-3.2-3B-Instruct"
POOL_SIZE        = 3_000
N_PARAM_TARGET   = 80
N_CTX_DEP_TARGET = 80
N_VARIANTS       = 4
LAYER_IDX        = 26
PCA_DIM          = 64
SEED             = 42
MAX_NEW          = 60
PARAM_MIN_F1     = 0.50
CTX_MAX_NC       = 0.05
CTX_MIN_CTX      = 0.50
TRAIN_FRAC       = 0.75
OUT_PATH         = "/kaggle/working/exp_j_llama_results.json"

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_TOK = None  # set in setup_model(); used by fmt_prompt for chat template
if DEVICE == "cpu":
    raise RuntimeError("GPU required")
print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB", flush=True)


# ── HF Token ───────────────────────────────────────────────────────────────────
def _get_hf_token():
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


# ── Model setup ────────────────────────────────────────────────────────────────
def setup_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok_hf = _get_hf_token()
    if tok_hf:
        from huggingface_hub import login
        login(token=tok_hf, add_to_git_credential=False)
    print(f"\nLoading {MODEL_ID} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map=None, trust_remote_code=True
    ).to(DEVICE).eval()
    print(f"  n_layers={mdl.config.num_hidden_layers}  d={mdl.config.hidden_size}", flush=True)
    global _TOK
    _TOK = tok
    has_chat = hasattr(tok, "apply_chat_template") and tok.chat_template is not None
    print(f"  chat_template={'YES' if has_chat else 'NO'}", flush=True)
    return mdl, tok


# ── Data ───────────────────────────────────────────────────────────────────────
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
    print(f"Pool: {len(items)} items", flush=True)
    return items


def normalize(s: str) -> str:
    """Lowercase and strip punctuation for fuzzy matching."""
    return re.sub(r"[^\w\s]", " ", s.lower()).strip()


def token_f1(pred: str, golds: list) -> float:
    pt = set(normalize(pred).split())
    best = 0.0
    for g in golds:
        gt = set(normalize(g).split())
        if not pt or not gt:
            continue
        c = pt & gt
        if not c:
            continue
        p = len(c) / len(pt)
        r = len(c) / len(gt)
        best = max(best, 2 * p * r / (p + r))
    return best


def answer_contains(pred: str, golds: list) -> bool:
    pn = normalize(pred)
    return any(normalize(g) in pn for g in golds)


def fmt_prompt(q: str, ctx: str = "") -> str:
    """Raw completion prompt for hidden-state extraction — maintains format parity with large_n_v2."""
    if ctx:
        return f"Context: {ctx}\n\nQuestion: {q}\nAnswer:"
    return f"Question: {q}\nAnswer:"


def fmt_oracle(q: str, ctx: str = "") -> str:
    """Chat prompt for oracle labeling. No brevity system message — lets model reason normally."""
    content = q if not ctx else f"Context: {ctx}\n\nQuestion: {q}"
    if _TOK is not None and hasattr(_TOK, "apply_chat_template") and _TOK.chat_template is not None:
        msgs = [{"role": "user", "content": content}]
        return _TOK.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return f"Question: {q}\nAnswer:"


def gen_oracle(model, tok, prompt: str) -> str:
    """Short-answer generation for oracle labeling (max 20 tokens to keep precision high)."""
    inp = tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **inp, max_new_tokens=20, do_sample=False,
            pad_token_id=tok.eos_token_id, use_cache=True
        )
    return tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def gen_text(model, tok, prompt: str) -> str:
    inp = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **inp, max_new_tokens=MAX_NEW, do_sample=False,
            pad_token_id=tok.eos_token_id, use_cache=True
        )
    return tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()


# ── Perturbation generation ────────────────────────────────────────────────────
def make_variants(q: str) -> list[str]:
    """Generate N_VARIANTS surface perturbations of question q."""
    variants = []

    # REPHRASE: swap common question openers
    if q.lower().startswith("what is"):
        variants.append("Can you name the" + q[7:])
    elif q.lower().startswith("what was"):
        variants.append("Can you name the" + q[8:])
    elif q.lower().startswith("who"):
        variants.append("Name the person:" + q[3:])
    else:
        variants.append("Please tell me: " + q)

    # LOWERCASE
    variants.append(q.lower())

    # APPEND: neutral filler suffix
    variants.append(q + " Please answer briefly.")

    # TYPO: drop 3rd character of the longest word
    words = q.split()
    longest = max(words, key=len)
    if len(longest) > 4:
        idx  = q.index(longest)
        typo = longest[:2] + longest[3:]
        variants.append(q[:idx] + typo + q[idx + len(longest):])
    else:
        variants.append(q + "?")  # fallback for short questions

    return variants[:N_VARIANTS]


# ── Hidden-state extraction via explicit 2-step forward pass ──────────────────
def extract_hs(model, tok, question: str) -> np.ndarray | None:
    """
    Extract L{LAYER_IDX} step-1 hidden state using explicit prefill + generation.

    Two forward passes:
      1. Prefill: process the full prompt, get KV cache + first-token logits.
      2. Step-1: run the first generated token through the model with KV cache,
         capture output_hidden_states at LAYER_IDX.

    This is equivalent to the original hook approach but avoids hook-firing
    edge cases with GQA / newer transformers versions.
    """
    prompt = fmt_prompt(question)
    inp    = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    try:
        with torch.no_grad():
            # Pass 1: prefill — get KV cache and first-token prediction
            prefill = model(
                **inp,
                output_hidden_states=False,
                use_cache=True,
                return_dict=True,
            )
            # Greedy first token (matches do_sample=False in original hook approach)
            next_tok = prefill.logits[0, -1, :].argmax().unsqueeze(0).unsqueeze(0)

            # Pass 2: generation step-1 — process first generated token with KV cache
            gen = model(
                input_ids=next_tok,
                past_key_values=prefill.past_key_values,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        # hidden_states: tuple of length n_layers+1
        #   [0] = embedding output, [k] = output of layer k-1 (0-indexed)
        # LAYER_IDX=26 (0-indexed) → index LAYER_IDX+1 = 27
        hs = gen.hidden_states[LAYER_IDX + 1]  # [1, 1, d]
        return hs[0, 0, :].detach().float().cpu().numpy()
    except Exception as e:
        print(f"    WARN: extract_hs failed: {e}", flush=True)
        return None


# ── Probe utilities ────────────────────────────────────────────────────────────
def fit_probe(X_tr: np.ndarray, y_tr: np.ndarray,
              X_te: np.ndarray, y_te: np.ndarray):
    """Fit PCA64 + LDA probe, return (auroc, shuffled_auroc, pca, lda)."""
    n_comp = min(PCA_DIM, X_tr.shape[1], X_tr.shape[0] - 1)
    pca    = PCA(n_components=n_comp, random_state=SEED)
    X_tr_r = pca.fit_transform(X_tr.astype(np.float32))
    X_te_r = pca.transform(X_te.astype(np.float32))
    lda    = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    lda.fit(X_tr_r, y_tr)
    scores   = lda.decision_function(X_te_r)
    auroc    = float(roc_auc_score(y_te, scores))
    y_shuf   = y_te.copy(); np.random.shuffle(y_shuf)
    shuffled = float(roc_auc_score(y_shuf, scores))
    return round(auroc, 4), round(shuffled, 4), pca, lda


def apply_probe(pca, lda, hs: np.ndarray) -> float:
    """Return scalar decision score for a single hidden-state vector."""
    x  = hs.astype(np.float32).reshape(1, -1)
    xr = pca.transform(x)
    return float(lda.decision_function(xr)[0])


# ── Phase 1: Bilateral oracle calibration ─────────────────────────────────────
def phase1_calibrate(model, tok, pool: list) -> dict:
    """
    Collect PARAM and CTX_DEP items via bilateral oracle protocol.
    Extract L26 step-1 hidden states. Fit Fisher+PCA64 probe.
    """
    print("\n=== Phase 1: Bilateral Oracle Calibration ===", flush=True)
    t0 = time.time()

    param_items  = []   # each: {question, hs, answers, context, label=1}
    ctxdep_items = []   # each: {question, hs, answers, context, label=0}
    n_skip       = 0

    for i, item in enumerate(pool):
        if len(param_items) >= N_PARAM_TARGET and len(ctxdep_items) >= N_CTX_DEP_TARGET:
            break
        if i % 50 == 0:
            print(f"  [{i}] PARAM={len(param_items)} CTX_DEP={len(ctxdep_items)}"
                  f"  elapsed={time.time()-t0:.0f}s", flush=True)

        # Nocontext pass for oracle labeling (short-answer chat format)
        pred_nc = gen_oracle(model, tok, fmt_oracle(item["question"]))
        f1_nc   = token_f1(pred_nc, item["answers"])

        # Debug: print first 5 items to confirm model output format
        if i < 5:
            ac = answer_contains(pred_nc, item["answers"])
            print(f"  DEBUG_NC[{i}]: q='{item['question'][:50]}'"
                  f"  pred='{pred_nc[:50]}'"
                  f"  ans={item['answers'][:2]}"
                  f"  f1={f1_nc:.3f}  ac={ac}", flush=True)

        # PARAM: model knows the answer without context
        if (f1_nc >= PARAM_MIN_F1 or answer_contains(pred_nc, item["answers"])) \
                and len(param_items) < N_PARAM_TARGET:
            hs = extract_hs(model, tok, item["question"])
            if hs is not None:
                param_items.append({
                    "question": item["question"],
                    "hs":       hs,
                    "answers":  item["answers"],
                    "context":  item["context"],
                    "label":    1,
                })
            continue

        # CTX_DEP: model fails without context, succeeds with context
        if f1_nc <= CTX_MAX_NC and item["context"] and len(ctxdep_items) < N_CTX_DEP_TARGET:
            pred_ctx = gen_oracle(model, tok, fmt_oracle(item["question"], item["context"]))
            f1_ctx   = token_f1(pred_ctx, item["answers"])
            if i < 5:
                ac_ctx = answer_contains(pred_ctx, item["answers"])
                print(f"  DEBUG_CTX[{i}]: pred='{pred_ctx[:50]}'"
                      f"  f1_ctx={f1_ctx:.3f}  ac_ctx={ac_ctx}", flush=True)
            if f1_ctx >= CTX_MIN_CTX or answer_contains(pred_ctx, item["answers"]):
                hs = extract_hs(model, tok, item["question"])
                if hs is not None:
                    ctxdep_items.append({
                        "question": item["question"],
                        "hs":       hs,
                        "answers":  item["answers"],
                        "context":  item["context"],
                        "label":    0,
                    })
                continue

        n_skip += 1

    n_p, n_c = len(param_items), len(ctxdep_items)
    print(f"  Collected: PARAM={n_p}  CTX_DEP={n_c}  SKIP={n_skip}", flush=True)

    if n_p < 10 or n_c < 10:
        print("  WARN: insufficient calibration samples", flush=True)
        return {"ok": False}

    # Train/test split
    n_tr_p = int(n_p * TRAIN_FRAC)
    n_tr_c = int(n_c * TRAIN_FRAC)
    X_tr = np.vstack(
        [r["hs"] for r in param_items[:n_tr_p]] +
        [r["hs"] for r in ctxdep_items[:n_tr_c]]
    )
    y_tr = np.array([1] * n_tr_p + [0] * n_tr_c)
    X_te = np.vstack(
        [r["hs"] for r in param_items[n_tr_p:]] +
        [r["hs"] for r in ctxdep_items[n_tr_c:]]
    )
    y_te = np.array([1] * (n_p - n_tr_p) + [0] * (n_c - n_tr_c))

    cal_auroc, cal_shuf, pca, lda = fit_probe(X_tr, y_tr, X_te, y_te)
    print(f"  Calibration AUROC: {cal_auroc:.4f}  shuffled={cal_shuf:.4f}", flush=True)

    return {
        "ok":          True,
        "pca":         pca,
        "lda":         lda,
        "cal_auroc":   cal_auroc,
        "cal_shuf":    cal_shuf,
        "n_param":     n_p,
        "n_ctxdep":    n_c,
        "param_items":  param_items,
        "ctxdep_items": ctxdep_items,
    }


# ── Intermediate save ──────────────────────────────────────────────────────────
def _save_intermediate(battery: list, p1: dict) -> None:
    partial = {
        "experiment": "EXP_J_PERTURBATION_BATTERY_V1",
        "status":     "PARTIAL",
        "n_items":    len(battery),
        "cal_auroc":  p1.get("cal_auroc"),
        "battery":    [
            {
                "question":       r["question"],
                "label":          r["label"],
                "label_str":      r["label_str"],
                "original_score": r["original_score"],
                "variant_scores": r["variant_scores"],
                "all_scores":     r["all_scores"],
                "variants":       r["variants"],
            }
            for r in battery
        ],
    }
    try:
        os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
        with open(OUT_PATH, "w") as f:
            json.dump(partial, f, indent=2)
    except Exception as e:
        print(f"    Intermediate save failed: {e}", flush=True)


# ── Phase 2: Perturbation battery ─────────────────────────────────────────────
def phase2_battery(model, tok, p1: dict) -> list:
    """
    For every labeled item from Phase 1:
      - Extract probe score for the original question (hs already captured)
      - Generate N_VARIANTS perturbations, extract HS and score each
    Returns list of battery records.
    """
    print("\n=== Phase 2: Perturbation Battery ===", flush=True)
    t0 = time.time()

    pca, lda  = p1["pca"], p1["lda"]
    all_items = p1["param_items"] + p1["ctxdep_items"]
    random.shuffle(all_items)

    battery = []

    for i, item in enumerate(all_items):
        if i % 50 == 0:
            print(f"  [{i}/{len(all_items)}]  processed={len(battery)}"
                  f"  elapsed={time.time()-t0:.0f}s", flush=True)

        collected_before = len(battery)

        # Original score: hs was captured in Phase 1
        orig_score = apply_probe(pca, lda, item["hs"])

        # Variant scores
        variants       = make_variants(item["question"])
        variant_scores = []
        for v_q in variants:
            v_hs = extract_hs(model, tok, v_q)
            if v_hs is None:
                variant_scores.append(None)
            else:
                variant_scores.append(apply_probe(pca, lda, v_hs))

        # Skip items with any missing variant extraction
        if any(s is None for s in variant_scores):
            print(f"    WARN: item {i} has missing variant HS — skipping", flush=True)
            continue

        battery.append({
            "question":       item["question"],
            "label":          item["label"],
            "label_str":      "PARAM" if item["label"] == 1 else "CTX_DEP",
            "original_score": orig_score,
            "variant_scores": variant_scores,
            "all_scores":     [orig_score] + variant_scores,
            "variants":       variants,
        })

        collected_after = len(battery)
        if collected_after > collected_before and collected_after % 20 == 0:
            _save_intermediate(battery, p1)
            print(f"    Intermediate save at {collected_after} items", flush=True)

    print(f"  Battery complete: {len(battery)} items", flush=True)
    return battery


# ── Phase 3: Analysis ──────────────────────────────────────────────────────────
def phase3_analyze(battery: list) -> dict:
    """
    Compute ICC, paired t-test, per-variant stats, and overall verdict.
    """
    print("\n=== Phase 3: Analysis ===", flush=True)

    n_items = len(battery)
    if n_items < 10:
        print("  WARN: too few items for meaningful analysis", flush=True)
        return {"error": "too few items", "n_items": n_items}

    original_scores = np.array([r["original_score"] for r in battery])
    all_scores_arr  = [np.array(r["all_scores"]) for r in battery]

    # ── ICC ──────────────────────────────────────────────────────────────────
    between_var    = float(np.var(original_scores))
    within_vars    = [float(np.var(all_scores_arr[i])) for i in range(n_items)]
    mean_within    = float(np.mean(within_vars))
    icc            = (between_var / (between_var + mean_within)
                      if (between_var + mean_within) > 0 else 0.0)
    kill_triggered = mean_within >= 0.50 * between_var
    verdict        = ("ROBUST" if icc >= 0.70 else
                      ("BORDERLINE" if icc >= 0.50 else "FRAGILE"))

    print(f"  n_items:         {n_items}", flush=True)
    print(f"  between_var:     {between_var:.6f}", flush=True)
    print(f"  mean_within_var: {mean_within:.6f}", flush=True)
    print(f"  ICC:             {icc:.4f}", flush=True)
    print(f"  kill_triggered:  {kill_triggered}", flush=True)
    print(f"  VERDICT:         {verdict}", flush=True)

    # ── Paired t-test: PARAM vs CTX_DEP separation after perturbation ────────
    param_variant_scores  = []
    ctxdep_variant_scores = []
    for r in battery:
        if r["label"] == 1:
            param_variant_scores.extend(r["variant_scores"])
        else:
            ctxdep_variant_scores.extend(r["variant_scores"])

    t_stat, p_val    = ttest_ind(param_variant_scores, ctxdep_variant_scores)
    sep_preserved    = (p_val < 0.05 and
                        float(np.mean(param_variant_scores)) > float(np.mean(ctxdep_variant_scores)))

    print(f"\n  Separation under perturbation:", flush=True)
    print(f"    PARAM variant mean:   {np.mean(param_variant_scores):.4f}"
          f"  (n={len(param_variant_scores)})", flush=True)
    print(f"    CTX_DEP variant mean: {np.mean(ctxdep_variant_scores):.4f}"
          f"  (n={len(ctxdep_variant_scores)})", flush=True)
    print(f"    t={t_stat:.3f}  p={p_val:.4f}  sep_preserved={sep_preserved}", flush=True)

    # ── Per-variant breakdown ─────────────────────────────────────────────────
    variant_names = ["REPHRASE", "LOWERCASE", "APPEND", "TYPO"]
    per_variant   = {}
    orig_scores_list = [r["original_score"] for r in battery]

    for vi, vname in enumerate(variant_names):
        v_scores   = [r["variant_scores"][vi] for r in battery]
        corr_mat   = np.corrcoef(orig_scores_list, v_scores) if len(v_scores) > 2 else None
        corr       = float(corr_mat[0, 1]) if corr_mat is not None else None
        mean_delta = float(np.mean([abs(v - o) for v, o in zip(v_scores, orig_scores_list)]))
        per_variant[vname] = {
            "mean_score":       round(float(np.mean(v_scores)), 4),
            "std_score":        round(float(np.std(v_scores)), 4),
            "corr_with_orig":   round(corr, 4) if corr is not None else None,
            "mean_abs_delta":   round(mean_delta, 4),
        }
        corr_str = f"{corr:.3f}" if corr is not None else "N/A"
        print(f"  {vname:10s}: mean={np.mean(v_scores):.4f}  std={np.std(v_scores):.4f}"
              f"  corr={corr_str}  |delta|={mean_delta:.4f}", flush=True)

    # ── Class counts ─────────────────────────────────────────────────────────
    n_param  = sum(1 for r in battery if r["label"] == 1)
    n_ctxdep = sum(1 for r in battery if r["label"] == 0)
    print(f"\n  n_PARAM={n_param}  n_CTX_DEP={n_ctxdep}", flush=True)

    return {
        "n_items":              n_items,
        "n_param":              n_param,
        "n_ctxdep":             n_ctxdep,
        "between_var":          round(between_var, 6),
        "mean_within_var":      round(mean_within, 6),
        "icc":                  round(icc, 4),
        "kill_triggered":       kill_triggered,
        "verdict":              verdict,
        "t_stat":               round(float(t_stat), 4),
        "p_value":              round(float(p_val), 6),
        "sep_preserved":        sep_preserved,
        "param_variant_mean":   round(float(np.mean(param_variant_scores)), 4),
        "ctxdep_variant_mean":  round(float(np.mean(ctxdep_variant_scores)), 4),
        "per_variant":          per_variant,
    }


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*60}", flush=True)
    print(f"EXP-J: Perturbation Battery v1", flush=True)
    print(f"  MODEL:    {MODEL_ID}", flush=True)
    print(f"  POOL:     {POOL_SIZE}", flush=True)
    print(f"  TARGETS:  PARAM={N_PARAM_TARGET}  CTX_DEP={N_CTX_DEP_TARGET}", flush=True)
    print(f"  LAYER:    {LAYER_IDX}  PCA_DIM={PCA_DIM}  VARIANTS={N_VARIANTS}", flush=True)
    print(f"  OUT:      {OUT_PATH}", flush=True)
    print(f"{'='*60}", flush=True)

    model, tok = setup_model()

    pool = load_pool(POOL_SIZE)

    # Phase 1: Bilateral oracle calibration + probe fitting
    p1 = phase1_calibrate(model, tok, pool)
    del pool; gc.collect()

    if not p1["ok"]:
        print("FATAL: Phase 1 calibration failed to collect enough items.", flush=True)
        sys.exit(1)

    # Phase 2: Perturbation battery
    battery = phase2_battery(model, tok, p1)

    if len(battery) < 10:
        print("FATAL: battery has fewer than 10 complete items.", flush=True)
        sys.exit(1)

    # Phase 3: Analysis
    analysis = phase3_analyze(battery)

    # Assemble final results
    results = {
        "experiment":     "EXP_J_PERTURBATION_BATTERY_V1",
        "model_id":       MODEL_ID,
        "status":         "COMPLETE",
        "config": {
            "pool_size":        POOL_SIZE,
            "n_param_target":   N_PARAM_TARGET,
            "n_ctxdep_target":  N_CTX_DEP_TARGET,
            "n_variants":       N_VARIANTS,
            "layer_idx":        LAYER_IDX,
            "pca_dim":          PCA_DIM,
            "seed":             SEED,
            "max_new":          MAX_NEW,
            "param_min_f1":     PARAM_MIN_F1,
            "ctx_max_nc":       CTX_MAX_NC,
            "ctx_min_ctx":      CTX_MIN_CTX,
            "train_frac":       TRAIN_FRAC,
        },
        "calibration": {
            "cal_auroc":  p1["cal_auroc"],
            "cal_shuf":   p1["cal_shuf"],
            "n_param":    p1["n_param"],
            "n_ctxdep":   p1["n_ctxdep"],
        },
        "battery_n_items": len(battery),
        "analysis":        analysis,
        "verdict":         analysis.get("verdict"),
        "kill_triggered":  analysis.get("kill_triggered"),
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {OUT_PATH}", flush=True)
    print(f"\n{'='*60}", flush=True)
    print(f"FINAL VERDICT:   {analysis.get('verdict')}", flush=True)
    print(f"kill_triggered:  {analysis.get('kill_triggered')}", flush=True)
    print(f"ICC:             {analysis.get('icc'):.4f}"
          f"  (>= 0.70 = ROBUST, >= 0.50 = BORDERLINE)", flush=True)
    print(f"sep_preserved:   {analysis.get('sep_preserved')}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
