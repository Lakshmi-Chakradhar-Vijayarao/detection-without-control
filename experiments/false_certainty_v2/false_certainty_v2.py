#!/usr/bin/env python3
"""
false_certainty_v2.py — EXP-A: False Certainty Detection v2

SCIENTIFIC QUESTION:
  When a model outputs with LOW ENTROPY (confident) but is WRONG,
  does the Fisher+PCA64 probe at L26 step-1 carry signal beyond entropy?
  Both classes have MATCHED entropy by construction, so entropy AUROC ≈ 0.50
  is the expected baseline. Fisher AUROC > 0.65 means hidden state has
  independent signal. Fisher AUROC ≈ 0.50 means step-1 is blind to confabulation.

TWO-PHASE DESIGN:
  Phase 1 (Bilateral Oracle Calibration, 5000 items):
    Collect PARAM and CTX_DEP items, extract L26 step-1 HS.
    Fit Fisher+PCA64 probe on PARAM vs CTX_DEP.
    Sanity check: bilateral oracle AUROC should be ~0.73 for Qwen.

  Phase 2 (Entropy-Matched False Certainty Collection, 15000 items):
    For each item: nocontext pass, compute entropy at step-1.
    θ_conf = 30th percentile of all entropies  (the "confident zone").
    Collect:
      CONFIDENT_CORRECT: entropy < θ_conf AND F1 ≥ 0.50   (N_TARGET)
      CONFIDENT_WRONG:   entropy < θ_conf AND F1 ≤ 0.05   (N_TARGET)
    Extract L26 and L8 step-1 HS for each.

  Phase 3 (Head-to-Head):
    Train/test split (TRAIN_FRAC=0.75).
    1. FC_Fisher  — Fisher+PCA64 fitted on CC/CW train, evaluated on test
    2. FC_Entropy — entropy AUROC on test (≈0.50 = matching succeeded)
    3. FC_JV      — J_velocity (L8→L26 shift) AUROC on test
    4. FC_Combined — LR([Fisher, -entropy, J_velocity]) AUROC
    5. BO_Transfer — bilateral oracle probe from Phase 1 applied to CC/CW test

DECISION GATE:
  HIDDEN_STATE_ESSENTIAL : FC_Fisher ≥ 0.65 AND FC_Entropy ≤ 0.55
  STEP_1_BLIND           : FC_Fisher ≤ 0.55 AND FC_Entropy ≤ 0.55
  JV_PARTIAL             : FC_JV ≥ 0.65 AND FC_Fisher ≤ 0.55
  ENTROPY_NOT_MATCHED    : FC_Entropy ≥ 0.65 (tighten θ_conf and rerun)

MODELS (run one per kernel invocation — set MODEL_IDX):
  0: Qwen/Qwen2.5-1.5B-Instruct
  1: meta-llama/Llama-3.2-3B-Instruct

GPU: T4. Expected ~3-4h per model.
"""

from __future__ import annotations
import gc, json, os, random, sys, time
import numpy as np
import torch
from datasets import load_dataset
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_IDX       = 0                     # 0=Qwen, 1=Llama
POOL_PHASE1     = 5_000                 # bilateral oracle calibration pool
POOL_PHASE2     = 15_000               # false certainty collection pool
N_CAL_EACH      = 100                   # per class for bilateral oracle
N_TARGET        = 100                   # per class for false certainty
ENTROPY_PCT     = 30                    # confident zone = bottom 30th pct entropy
TRAIN_FRAC      = 0.75
LAYER_IDX       = 26                    # penultimate layer (0-indexed)
LAYER_SHALLOW   = 8                     # for J_velocity
PCA_DIM         = 64
PARAM_MIN_F1    = 0.50
CTX_MAX_NC      = 0.05                  # nocontext F1 threshold for CTX_DEP
CTX_MIN_CTX     = 0.50                  # withcontext F1 threshold for CTX_DEP
SEED            = 42

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    raise RuntimeError("GPU required")
print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB", flush=True)

MODELS = [
    {"name": "qwen25_1.5b_instruct", "model_id": "Qwen/Qwen2.5-1.5B-Instruct", "n_layers": 28},
    {"name": "llama32_3b_instruct",  "model_id": None,                           "n_layers": 28},
]


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


def _resolve_llama():
    for d in (os.listdir("/kaggle/input") if os.path.exists("/kaggle/input") else []):
        if "llama" in d.lower():
            for sub in ["", "/transformers/3b-instruct/1", "/1"]:
                p = f"/kaggle/input/{d}{sub}"
                if os.path.isdir(p):
                    return p
    tok = _get_hf_token()
    if tok:
        from huggingface_hub import login
        login(token=tok, add_to_git_credential=False)
    return "meta-llama/Llama-3.2-3B-Instruct"


def setup_model(cfg):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = cfg["model_id"] or _resolve_llama()
    print(f"\nLoading {model_id} …", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map=None, trust_remote_code=True
    ).to(DEVICE).eval()
    print(f"  n_layers={mdl.config.num_hidden_layers}  d={mdl.config.hidden_size}", flush=True)
    return mdl, tok


# ── Data ───────────────────────────────────────────────────────────────────────
def load_pool(n: int):
    ds = load_dataset("trivia_qa", "rc.wikipedia", split="validation",
                      trust_remote_code=True)
    items = []
    for ex in ds:
        ep = ex.get("entity_pages", {})
        ctx = (ep.get("wiki_context") or [""])[0][:800] if ep else ""
        ans = ex["answer"]["aliases"] or [ex["answer"]["value"]]
        items.append({"question": ex["question"], "context": ctx, "answers": ans})
        if len(items) >= n:
            break
    random.shuffle(items)
    print(f"Pool: {len(items)} items", flush=True)
    return items


def token_f1(pred: str, golds: list) -> float:
    pt = set(pred.lower().split())
    best = 0.0
    for g in golds:
        gt = set(g.lower().split())
        if not pt or not gt:
            continue
        c = pt & gt
        if not c:
            continue
        p = len(c) / len(pt)
        r = len(c) / len(gt)
        best = max(best, 2 * p * r / (p + r))
    return best


def fmt_prompt(q: str, ctx: str = "") -> str:
    if ctx:
        return f"Context: {ctx}\n\nAnswer in one short phrase.\nQuestion: {q}\nAnswer:"
    return f"Answer in one short phrase.\nQuestion: {q}\nAnswer:"


def gen_text(model, tok, prompt: str, max_new: int = 60) -> str:
    inp = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id, use_cache=True)
    return tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()


# ── HS + entropy extraction ────────────────────────────────────────────────────
def extract_step1(model, tok, question: str, layer_idx: int,
                  layer_shallow: int | None = None) -> dict:
    """
    Single nocontext pass. Returns:
      hs_deep   : L{layer_idx} step-1 hidden state
      hs_shallow: L{layer_shallow} step-1 hidden state (if provided)
      entropy   : Shannon entropy of step-1 output distribution
    """
    prompt = fmt_prompt(question)
    inp = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)

    captured = {}

    def _make_hook(key):
        def _hook(mod, inp_t, out):
            hs = out[0] if isinstance(out, tuple) else out
            if hs.shape[1] != 1 or key in captured:
                return
            captured[key] = hs[0, 0, :].detach().float().cpu().numpy()
        return _hook

    handles = []
    handles.append(model.model.layers[layer_idx].register_forward_hook(_make_hook("hs_deep")))
    if layer_shallow is not None:
        handles.append(model.model.layers[layer_shallow].register_forward_hook(_make_hook("hs_shallow")))

    try:
        with torch.no_grad():
            out = model.generate(
                **inp, max_new_tokens=2, do_sample=False,
                pad_token_id=tok.eos_token_id, use_cache=True,
                output_scores=True, return_dict_in_generate=True,
            )
    finally:
        for h in handles:
            h.remove()

    logits = out.scores[0][0].float()
    probs  = torch.softmax(logits, dim=-1).clamp(min=1e-10)
    entropy = float(-torch.sum(probs * torch.log(probs)).item())

    return {
        "hs_deep":    captured.get("hs_deep"),
        "hs_shallow": captured.get("hs_shallow"),
        "entropy":    entropy,
    }


# ── Probe utilities ────────────────────────────────────────────────────────────
def _pca_lda(X_tr, y_tr, X_te, y_te):
    n_comp = min(PCA_DIM, X_tr.shape[1], X_tr.shape[0] - 1)
    pca    = PCA(n_components=n_comp, random_state=SEED)
    X_tr_r = pca.fit_transform(X_tr.astype(np.float32))
    X_te_r = pca.transform(X_te.astype(np.float32))
    lda    = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    lda.fit(X_tr_r, y_tr)
    scores = lda.decision_function(X_te_r)
    auroc  = float(roc_auc_score(y_te, scores))
    y_shuf = y_te.copy(); np.random.shuffle(y_shuf)
    shuf   = float(roc_auc_score(y_shuf, scores))
    return round(auroc, 4), round(shuf, 4), pca, lda


def entropy_auroc(ent_0, ent_1):
    """ent_0 = class-0 entropies, ent_1 = class-1 entropies.
    Score = -entropy (higher entropy → class 0).
    """
    y    = np.array([1]*len(ent_1) + [0]*len(ent_0))
    sc   = np.concatenate([-np.array(ent_1), -np.array(ent_0)])
    return round(float(roc_auc_score(y, sc)), 4)


def jv_auroc(jv_cc: list, jv_cw: list):
    y  = np.array([1]*len(jv_cc) + [0]*len(jv_cw))
    sc = np.concatenate([jv_cc, jv_cw])
    if len(np.unique(y)) < 2 or len(sc) != len(y):
        return None
    return round(float(roc_auc_score(y, sc)), 4)


def combined_auroc(f_cc_te, f_cw_te, e_cc_te, e_cw_te, jv_cc_te, jv_cw_te,
                   f_cc_tr, f_cw_tr, e_cc_tr, e_cw_tr, jv_cc_tr, jv_cw_tr):
    """LR on [Fisher_score, -entropy, J_velocity]."""
    f_tr = np.concatenate([f_cc_tr, f_cw_tr]).reshape(-1, 1)
    e_tr = np.concatenate([e_cc_tr, e_cw_tr]).reshape(-1, 1)
    jv_tr = np.concatenate([jv_cc_tr, jv_cw_tr]).reshape(-1, 1)
    X_tr = np.hstack([f_tr, -e_tr, jv_tr])
    y_tr = np.array([1]*len(f_cc_tr) + [0]*len(f_cw_tr))

    f_te  = np.concatenate([f_cc_te, f_cw_te]).reshape(-1, 1)
    e_te  = np.concatenate([e_cc_te, e_cw_te]).reshape(-1, 1)
    jv_te = np.concatenate([jv_cc_te, jv_cw_te]).reshape(-1, 1)
    X_te  = np.hstack([f_te, -e_te, jv_te])
    y_te  = np.array([1]*len(f_cc_te) + [0]*len(f_cw_te))

    lr = LogisticRegression(max_iter=500, C=1.0)
    lr.fit(X_tr, y_tr)
    sc = lr.predict_proba(X_te)[:, 1]
    return round(float(roc_auc_score(y_te, sc)), 4)


# ── Phase 1: Bilateral oracle calibration ─────────────────────────────────────
def phase1_bilateral_oracle(model, tok, pool: list) -> dict:
    """Collect PARAM + CTX_DEP items, fit Fisher+PCA64 probe."""
    print("\n=== Phase 1: Bilateral Oracle Calibration ===", flush=True)
    t0 = time.time()

    param_hs, ctxdep_hs = [], []
    n_skip = 0

    for i, item in enumerate(pool):
        if len(param_hs) >= N_CAL_EACH and len(ctxdep_hs) >= N_CAL_EACH:
            break
        if i % 50 == 0:
            print(f"  [{i}] PARAM={len(param_hs)} CTX_DEP={len(ctxdep_hs)}"
                  f"  elapsed={time.time()-t0:.0f}s", flush=True)

        # Check if PARAM
        pred_nc = gen_text(model, tok, fmt_prompt(item["question"]))
        f1_nc   = token_f1(pred_nc, item["answers"])

        if f1_nc >= PARAM_MIN_F1 and len(param_hs) < N_CAL_EACH:
            sig = extract_step1(model, tok, item["question"], LAYER_IDX)
            if sig["hs_deep"] is not None:
                param_hs.append(sig["hs_deep"])
            continue

        if f1_nc <= CTX_MAX_NC and item["context"]:
            pred_ctx = gen_text(model, tok, fmt_prompt(item["question"], item["context"]))
            if token_f1(pred_ctx, item["answers"]) >= CTX_MIN_CTX and len(ctxdep_hs) < N_CAL_EACH:
                sig = extract_step1(model, tok, item["question"], LAYER_IDX)
                if sig["hs_deep"] is not None:
                    ctxdep_hs.append(sig["hs_deep"])
                continue

        n_skip += 1

    n_p, n_c = len(param_hs), len(ctxdep_hs)
    print(f"  Collected: PARAM={n_p}  CTX_DEP={n_c}  SKIP={n_skip}", flush=True)

    if n_p < 10 or n_c < 10:
        print("  WARN: insufficient samples for bilateral oracle calibration", flush=True)
        return {"ok": False}

    n_tr_p = int(n_p * TRAIN_FRAC)
    n_tr_c = int(n_c * TRAIN_FRAC)
    X_tr = np.vstack(param_hs[:n_tr_p] + ctxdep_hs[:n_tr_c])
    y_tr = np.array([1]*n_tr_p + [0]*n_tr_c)
    X_te = np.vstack(param_hs[n_tr_p:] + ctxdep_hs[n_tr_c:])
    y_te = np.array([1]*(n_p - n_tr_p) + [0]*(n_c - n_tr_c))

    bo_auroc, bo_shuf, bo_pca, bo_lda = _pca_lda(X_tr, y_tr, X_te, y_te)
    print(f"  Bilateral Oracle AUROC: {bo_auroc:.4f}  shuffled={bo_shuf:.4f}", flush=True)

    return {
        "ok": True,
        "n_param": n_p, "n_ctxdep": n_c,
        "bo_auroc": bo_auroc, "bo_shuffled": bo_shuf,
        "pca": bo_pca, "lda": bo_lda,
    }


# ── Phase 2: Entropy-matched false certainty collection ───────────────────────
def phase2_collect_fc(model, tok, pool: list) -> dict:
    """
    Collect (entropy, hs_deep, hs_shallow, f1) for all items in pool.
    Determine θ_conf, then select N_TARGET CONFIDENT_CORRECT and CONFIDENT_WRONG.
    """
    print("\n=== Phase 2: Entropy-Matched Collection ===", flush=True)
    t0 = time.time()

    all_records = []   # list of {entropy, hs_deep, hs_shallow, f1}

    for i, item in enumerate(pool):
        if i % 100 == 0:
            print(f"  [{i}/{len(pool)}]  collected={len(all_records)}"
                  f"  elapsed={time.time()-t0:.0f}s", flush=True)

        sig = extract_step1(model, tok, item["question"], LAYER_IDX, LAYER_SHALLOW)
        if sig["hs_deep"] is None:
            continue

        pred = gen_text(model, tok, fmt_prompt(item["question"]))
        f1   = token_f1(pred, item["answers"])

        all_records.append({
            "entropy":    sig["entropy"],
            "hs_deep":    sig["hs_deep"],
            "hs_shallow": sig["hs_shallow"],
            "f1":         f1,
        })

    print(f"  Total records: {len(all_records)}", flush=True)

    # Determine θ_conf
    all_entropies = [r["entropy"] for r in all_records]
    theta_conf = float(np.percentile(all_entropies, ENTROPY_PCT))
    print(f"  θ_conf ({ENTROPY_PCT}th pct): {theta_conf:.4f}", flush=True)
    print(f"  Items in confident zone: "
          f"{sum(1 for r in all_records if r['entropy'] < theta_conf)}", flush=True)

    # Select CC and CW
    cc_recs = [r for r in all_records if r["entropy"] < theta_conf and r["f1"] >= PARAM_MIN_F1]
    cw_recs = [r for r in all_records if r["entropy"] < theta_conf and r["f1"] <= CTX_MAX_NC]
    random.shuffle(cc_recs); random.shuffle(cw_recs)
    cc_recs = cc_recs[:N_TARGET]
    cw_recs = cw_recs[:N_TARGET]

    print(f"  CONFIDENT_CORRECT: {len(cc_recs)}", flush=True)
    print(f"  CONFIDENT_WRONG:   {len(cw_recs)}", flush=True)

    if len(cc_recs) < 20 or len(cw_recs) < 20:
        print("  WARN: insufficient CC or CW samples", flush=True)
        return {"ok": False, "theta_conf": theta_conf}

    # Entropy stats
    cc_ent = [r["entropy"] for r in cc_recs]
    cw_ent = [r["entropy"] for r in cw_recs]
    print(f"  CC entropy: mean={np.mean(cc_ent):.3f}  std={np.std(cc_ent):.3f}", flush=True)
    print(f"  CW entropy: mean={np.mean(cw_ent):.3f}  std={np.std(cw_ent):.3f}", flush=True)

    return {
        "ok":         True,
        "theta_conf": theta_conf,
        "cc_recs":    cc_recs,
        "cw_recs":    cw_recs,
        "cc_entropy_mean": round(float(np.mean(cc_ent)), 4),
        "cw_entropy_mean": round(float(np.mean(cw_ent)), 4),
        "n_pool": len(all_records),
    }


# ── Phase 3: Head-to-head comparison ──────────────────────────────────────────
def phase3_compare(p2: dict, p1: dict) -> dict:
    """Run all probes on the entropy-matched CC vs CW items."""
    print("\n=== Phase 3: Head-to-Head Comparison ===", flush=True)

    cc_recs, cw_recs = p2["cc_recs"], p2["cw_recs"]
    n_cc, n_cw = len(cc_recs), len(cw_recs)
    n_tr_cc = int(n_cc * TRAIN_FRAC)
    n_tr_cw = int(n_cw * TRAIN_FRAC)
    n_te_cc = n_cc - n_tr_cc
    n_te_cw = n_cw - n_tr_cw

    # Arrays
    hs_cc_tr = np.vstack([r["hs_deep"] for r in cc_recs[:n_tr_cc]])
    hs_cw_tr = np.vstack([r["hs_deep"] for r in cw_recs[:n_tr_cw]])
    hs_cc_te = np.vstack([r["hs_deep"] for r in cc_recs[n_tr_cc:]])
    hs_cw_te = np.vstack([r["hs_deep"] for r in cw_recs[n_tr_cw:]])

    X_tr = np.vstack([hs_cc_tr, hs_cw_tr])
    y_tr = np.array([1]*n_tr_cc + [0]*n_tr_cw)
    X_te = np.vstack([hs_cc_te, hs_cw_te])
    y_te = np.array([1]*n_te_cc + [0]*n_te_cw)

    # 1. Fisher+PCA64 (new probe on CC/CW)
    fc_fisher, fc_shuf, _, _ = _pca_lda(X_tr, y_tr, X_te, y_te)
    print(f"  FC_Fisher AUROC: {fc_fisher:.4f}  shuffled={fc_shuf:.4f}", flush=True)

    # 2. Entropy AUROC (validation: should be ~0.50)
    ent_cc_te = [r["entropy"] for r in cc_recs[n_tr_cc:]]
    ent_cw_te = [r["entropy"] for r in cw_recs[n_tr_cw:]]
    fc_entropy = entropy_auroc(ent_cw_te, ent_cc_te)
    print(f"  FC_Entropy AUROC: {fc_entropy:.4f}  (≈0.50 = matching succeeded)", flush=True)

    # 3. J_velocity AUROC
    def get_jv(recs):
        jvs = []
        for r in recs:
            if r["hs_shallow"] is not None and r["hs_deep"] is not None:
                # J_velocity = L_deep score - L_shallow score (raw dot product diff)
                jvs.append(float(np.linalg.norm(r["hs_deep"] - r["hs_shallow"])))
            else:
                jvs.append(0.0)
        return jvs

    jv_cc_te = get_jv(cc_recs[n_tr_cc:])
    jv_cw_te = get_jv(cw_recs[n_tr_cw:])
    jv_cc_tr = get_jv(cc_recs[:n_tr_cc])
    jv_cw_tr = get_jv(cw_recs[:n_tr_cw])
    fc_jv = jv_auroc(jv_cc_te, jv_cw_te)
    print(f"  FC_JV AUROC: {fc_jv}", flush=True)

    # 4. Combined LR
    _, _, pca_c, lda_c = _pca_lda(X_tr, y_tr, X_te, y_te)  # re-fit to get decision scores
    n_comp = min(PCA_DIM, X_tr.shape[1], X_tr.shape[0] - 1)
    pca_c2 = PCA(n_components=n_comp, random_state=SEED)
    X_tr_r = pca_c2.fit_transform(X_tr.astype(np.float32))
    X_te_r = pca_c2.transform(X_te.astype(np.float32))
    lda_c2 = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    lda_c2.fit(X_tr_r, y_tr)
    f_tr_scores = lda_c2.decision_function(X_tr_r)
    f_te_scores = lda_c2.decision_function(X_te_r)

    f_cc_tr = f_tr_scores[:n_tr_cc].tolist()
    f_cw_tr = f_tr_scores[n_tr_cc:].tolist()
    f_cc_te = f_te_scores[:n_te_cc].tolist()
    f_cw_te = f_te_scores[n_te_cc:].tolist()

    ent_cc_tr = [r["entropy"] for r in cc_recs[:n_tr_cc]]
    ent_cw_tr = [r["entropy"] for r in cw_recs[:n_tr_cw]]

    if fc_jv is not None:
        fc_combined = combined_auroc(
            f_cc_te, f_cw_te, ent_cc_te, ent_cw_te, jv_cc_te, jv_cw_te,
            f_cc_tr, f_cw_tr, ent_cc_tr, ent_cw_tr, jv_cc_tr, jv_cw_tr
        )
        print(f"  FC_Combined AUROC: {fc_combined:.4f}", flush=True)
    else:
        fc_combined = None
        print("  FC_Combined: skipped (no J_velocity)", flush=True)

    # 5. Bilateral oracle probe transfer (if Phase 1 succeeded)
    bo_transfer = None
    if p1.get("ok"):
        n_comp_bo = min(PCA_DIM, X_te.shape[1], p1["pca"].n_components_)
        try:
            X_te_bo = p1["pca"].transform(X_te.astype(np.float32))
            bo_scores = p1["lda"].decision_function(X_te_bo)
            bo_transfer = round(float(roc_auc_score(y_te, bo_scores)), 4)
            print(f"  BO_Transfer AUROC: {bo_transfer:.4f}  "
                  f"(bilateral oracle probe applied to CC/CW)", flush=True)
        except Exception as e:
            print(f"  BO_Transfer failed: {e}", flush=True)

    # Decision gate
    def verdict():
        if fc_entropy >= 0.65:
            return "ENTROPY_NOT_MATCHED"
        if fc_fisher >= 0.65 and fc_entropy <= 0.55:
            return "HIDDEN_STATE_ESSENTIAL"
        if fc_fisher <= 0.55 and fc_entropy <= 0.55:
            return "STEP_1_BLIND"
        if fc_jv is not None and fc_jv >= 0.65 and fc_fisher <= 0.55:
            return "JV_PARTIAL"
        return "AMBIGUOUS"

    verd = verdict()
    print(f"\n  VERDICT: {verd}", flush=True)
    print(f"  Summary: Fisher={fc_fisher:.4f}  Entropy={fc_entropy:.4f}"
          f"  JV={fc_jv}  Combined={fc_combined}  BO_Transfer={bo_transfer}", flush=True)

    return {
        "n_cc": n_cc, "n_cw": n_cw,
        "n_train_per_class": min(n_tr_cc, n_tr_cw),
        "n_test_per_class":  min(n_te_cc, n_te_cw),
        "fc_fisher_auroc":   fc_fisher,
        "fc_fisher_shuffled": fc_shuf,
        "fc_entropy_auroc":  fc_entropy,
        "fc_jv_auroc":       fc_jv,
        "fc_combined_auroc": fc_combined,
        "bo_transfer_auroc": bo_transfer,
        "verdict":           verd,
    }


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    cfg = MODELS[MODEL_IDX]
    print(f"\n{'='*60}", flush=True)
    print(f"EXP-A: False Certainty v2 — {cfg['name']}", flush=True)
    print(f"{'='*60}", flush=True)

    model, tok = setup_model(cfg)

    # Phase 1
    p1_pool = load_pool(POOL_PHASE1)
    p1 = phase1_bilateral_oracle(model, tok, p1_pool)
    del p1_pool; gc.collect()

    # Phase 2
    p2_pool = load_pool(POOL_PHASE2)
    p2 = phase2_collect_fc(model, tok, p2_pool)
    del p2_pool; gc.collect()

    if not p2["ok"]:
        print("FATAL: Phase 2 failed to collect enough items.", flush=True)
        sys.exit(1)

    # Phase 3
    p3 = phase3_compare(p2, p1)

    # Assemble results
    results = {
        "experiment":    "EXP_A_FALSE_CERTAINTY_V2",
        "model_name":    cfg["name"],
        "config": {
            "pool_phase1":   POOL_PHASE1,
            "pool_phase2":   POOL_PHASE2,
            "n_cal_each":    N_CAL_EACH,
            "n_target":      N_TARGET,
            "entropy_pct":   ENTROPY_PCT,
            "train_frac":    TRAIN_FRAC,
            "layer_idx":     LAYER_IDX,
            "layer_shallow": LAYER_SHALLOW,
            "pca_dim":       PCA_DIM,
            "seed":          SEED,
        },
        "phase1": {
            "bo_auroc":    p1.get("bo_auroc"),
            "bo_shuffled": p1.get("bo_shuffled"),
            "n_param":     p1.get("n_param"),
            "n_ctxdep":    p1.get("n_ctxdep"),
        },
        "phase2": {
            "theta_conf":      p2["theta_conf"],
            "n_pool":          p2.get("n_pool"),
            "n_cc":            len(p2["cc_recs"]),
            "n_cw":            len(p2["cw_recs"]),
            "cc_entropy_mean": p2["cc_entropy_mean"],
            "cw_entropy_mean": p2["cw_entropy_mean"],
        },
        "phase3":  p3,
        "verdict": p3["verdict"],
        "observability_ladder_level": (
            "3" if p3["verdict"] == "HIDDEN_STATE_ESSENTIAL" else "1"
        ),
    }

    out_path = "false_certainty_v2_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {out_path}", flush=True)
    print(json.dumps({k: v for k, v in results["phase3"].items()
                      if k != "cc_recs" and k != "cw_recs"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
