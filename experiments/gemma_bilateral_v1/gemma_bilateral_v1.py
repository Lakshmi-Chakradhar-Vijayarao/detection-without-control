#!/usr/bin/env python3
"""
gemma_bilateral_v1.py — EXP_GEMMA_BILATERAL_V1
Third-architecture replication of L1 + L2 on Gemma-2-2B-IT (Google family)

SCIENTIFIC QUESTION:
  Does bilateral oracle Fisher+PCA64 replicate on Gemma-2-2B-IT?
  L1: PARAM vs CTX_DEP AUROC >= 0.70 (C001/C003 architecture generalization)
  L2: CC vs CW Fisher gap >= 0.15 within entropy-matched zone (C017 generalization)

DESIGN:
  Phase 1 — Bilateral Oracle L1 (N=200/class, pool=15000, shuffled control):
    Collect PARAM (nocontext F1 >= 0.50) and CTX_DEP (nocontext F1 <= 0.05,
    withcontext F1 >= 0.50) items.
    Fisher+PCA64 at penultimate layer step-1.
    Shuffled control: permute labels, rerun probe, compare AUROC.
    Bootstrap 95% CI (n=1000).

  Phase 2 — L2 False Certainty (N=100/class, entropy-matched):
    From remaining pool, collect CC (correct, low entropy) and CW (wrong, low entropy)
    items with entropy in calibrated window [THETA_LO, THETA_HI].
    Fisher+PCA64 vs entropy baseline. Gap = Fisher - Entropy.

  Phase 3 — Bilateral Oracle Transfer (BO_Transfer):
    Apply Phase 1 probe directly to CC/CW test items (transfer AUROC).

TERMINATION RULES (from research_plan_v2.md):
  T1_PARTIAL: L1 < 0.65 → characterize as architecture-specific, note in paper
  T2: L2 gap < 0.05 → L2 does not generalize to Gemma, note scope contraction

GPU: T4 (~10 hrs)
Model: google/gemma-2-2b-it
"""

from __future__ import annotations
import gc, json, os, random, time
import numpy as np
import torch

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ── Config ──────────────────────────────────────────────────────────────────────
MODEL_ID         = "google/gemma-2-2b-it"
LAYER_IDX        = None          # auto-selected as n_layers - 2 (penultimate)
N_L1_TARGET      = 200           # per-class for L1 bilateral oracle
N_L2_TARGET      = 100           # per-class for L2 CC/CW
POOL_SIZE         = 15_000
N_BOOTSTRAP      = 1000
TRAIN_FRAC       = 0.75
PCA_DIM          = 64
MAX_NEW          = 60
PARAM_MIN_F1     = 0.50
CTX_MAX_F1_NC    = 0.05
CTX_MIN_F1_WC    = 0.50
CW_MAX_F1        = 0.05
ENT_HALF         = 0.35          # half-width of entropy window; calibrated below

SAVE_PATH        = "/kaggle/working/gemma_bilateral_v1_results.json"
INTERMEDIATE     = "/kaggle/working/gemma_bilateral_v1_intermediate.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    raise RuntimeError("GPU required — no CUDA device found.")
print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB", flush=True)


# ── HF token ────────────────────────────────────────────────────────────────────
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

_tok_val = _get_hf_token()
if _tok_val:
    from huggingface_hub import login as _hf_login
    _hf_login(token=_tok_val, add_to_git_credential=False)
    print("HF login: OK", flush=True)
else:
    print("WARNING: HF_TOKEN not found", flush=True)


# ── Data ─────────────────────────────────────────────────────────────────────────
def load_pool(n: int = POOL_SIZE):
    from datasets import load_dataset
    ds = load_dataset("trivia_qa", "rc.wikipedia", split="train", streaming=True)
    items = []
    for row in ds:
        if not row["answer"]["aliases"]:
            continue
        ctx_list = (row.get("entity_pages") or {}).get("wiki_context") or [""]
        ctx = ctx_list[0][:1000] if ctx_list and ctx_list[0] else ""
        items.append({
            "question": row["question"],
            "answers":  row["answer"]["aliases"],
            "context":  ctx,
        })
        if len(items) >= n:
            break
    np.random.shuffle(items)
    print(f"Loaded pool: {len(items)} items", flush=True)
    return items


def token_f1(pred: str, golds) -> float:
    p = set(pred.lower().split())
    best = 0.0
    for g in golds:
        q = set(g.lower().split())
        c = p & q
        if c and p and q:
            pr = len(c)/len(p); rc = len(c)/len(q)
            best = max(best, 2*pr*rc/(pr+rc))
    return best


def answer_contains(pred: str, golds) -> bool:
    pl = pred.lower()
    return any(g.lower().strip() and g.lower().strip() in pl for g in golds)


# ── Model ────────────────────────────────────────────────────────────────────────
def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {MODEL_ID} …", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, trust_remote_code=True
    ).to(DEVICE)
    model.eval()
    cfg = getattr(model.config, 'text_config', model.config)
    n_layers = cfg.num_hidden_layers
    hidden = cfg.hidden_size
    print(f"  Loaded: {n_layers} layers, hidden={hidden}", flush=True)
    return model, tokenizer, n_layers


# ── Layer resolver (Gemma-2 uses model.model.layers) ────────────────────────────
_LAYER_PATHS = [
    "model.layers",
    "model.language_model.layers",
    "language_model.model.layers",
    "language_model.layers",
    "transformer.h",
]

def get_layers(model):
    for path in _LAYER_PATHS:
        try:
            obj = model
            for part in path.split("."):
                obj = getattr(obj, part)
            if hasattr(obj, "__len__") and len(obj) > 0:
                print(f"[layer path] {path} ({len(obj)} layers)", flush=True)
                return obj
        except AttributeError:
            continue
    raise RuntimeError(f"Cannot find transformer layers in {type(model).__name__}")


# ── Hidden-state extraction ──────────────────────────────────────────────────────
def get_step1_hs(model, tokenizer, prompt: str, layer_idx: int):
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    hs_out = [None]

    def hook(mod, inp, out):
        x = out[0] if isinstance(out, tuple) else out
        hs_out[0] = x[:, -1, :].detach().float().cpu().numpy()

    layers = get_layers(model)
    h = layers[layer_idx].register_forward_hook(hook)
    with torch.no_grad():
        model(ids)
    h.remove()
    return hs_out[0][0] if hs_out[0] is not None else None


def get_step1_entropy(model, tokenizer, prompt: str):
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad():
        out = model(ids)
    logits = out.logits[0, -1, :]
    probs = torch.softmax(logits, dim=-1)
    ent = float(-torch.sum(probs * torch.log(probs + 1e-10)).item())
    return ent


def generate(model, tokenizer, prompt: str):
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=MAX_NEW, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


# ── Prompts ──────────────────────────────────────────────────────────────────────
def prompt_nocontext(tokenizer, q: str) -> str:
    msgs = [{"role": "user", "content": q}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def prompt_withcontext(tokenizer, q: str, ctx: str) -> str:
    content = f"Context: {ctx[:600]}\n\nQuestion: {q}"
    msgs = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


# ── Probe ─────────────────────────────────────────────────────────────────────────
def fit_probe(X, y):
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import roc_auc_score
    pca = PCA(n_components=min(PCA_DIM, X.shape[1], X.shape[0]-1))
    Xp = pca.fit_transform(X)
    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    lda.fit(Xp, y)
    scores = lda.decision_function(Xp)
    auroc = float(roc_auc_score(y, scores))
    return pca, lda, auroc, scores


def eval_probe(pca, lda, X_test, y_test):
    from sklearn.metrics import roc_auc_score
    Xp = pca.transform(X_test)
    scores = lda.decision_function(Xp)
    return float(roc_auc_score(y_test, scores)), scores


def bootstrap_ci(pca, lda, X, y, n_bootstrap=N_BOOTSTRAP):
    from sklearn.metrics import roc_auc_score
    Xp = pca.transform(X)
    scores = lda.decision_function(Xp)
    aurocs = []
    n = len(y)
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        try:
            aurocs.append(float(roc_auc_score(y[idx], scores[idx])))
        except Exception:
            pass
    if not aurocs:
        return (0.0, 0.0)
    return (float(np.percentile(aurocs, 2.5)), float(np.percentile(aurocs, 97.5)))


def shuffled_auroc(pca, lda, X, y):
    from sklearn.metrics import roc_auc_score
    Xp = pca.transform(X)
    scores = lda.decision_function(Xp)
    y_shuf = y.copy(); np.random.shuffle(y_shuf)
    try:
        return float(roc_auc_score(y_shuf, scores))
    except Exception:
        return 0.5


# ── Phase 1: Bilateral Oracle L1 ────────────────────────────────────────────────
def run_phase1(model, tokenizer, pool, layer_idx):
    print("\n=== Phase 1: Bilateral Oracle L1 ===", flush=True)
    param_hs, ctxdep_hs = [], []
    n_scanned = 0

    for item in pool:
        if len(param_hs) >= N_L1_TARGET and len(ctxdep_hs) >= N_L1_TARGET:
            break
        n_scanned += 1
        q = item["question"]
        ans = item["answers"]
        ctx = item["context"]

        pnc = prompt_nocontext(tokenizer, q)
        hs = get_step1_hs(model, tokenizer, pnc, layer_idx)
        if hs is None:
            continue

        # Generate nocontext response
        gen_nc = generate(model, tokenizer, pnc)
        f1_nc = token_f1(gen_nc, ans)
        ok_nc = answer_contains(gen_nc, ans) or f1_nc >= PARAM_MIN_F1

        if ok_nc and len(param_hs) < N_L1_TARGET:
            param_hs.append(hs)
        elif not ok_nc and f1_nc <= CTX_MAX_F1_NC and ctx and len(ctxdep_hs) < N_L1_TARGET:
            # Check withcontext
            pwc = prompt_withcontext(tokenizer, q, ctx)
            gen_wc = generate(model, tokenizer, pwc)
            f1_wc = token_f1(gen_wc, ans)
            ok_wc = answer_contains(gen_wc, ans) or f1_wc >= CTX_MIN_F1_WC
            if ok_wc:
                ctxdep_hs.append(hs)

        if n_scanned % 200 == 0 or (n_scanned <= 50 and n_scanned % 10 == 0):
            print(f"  scanned={n_scanned} PARAM={len(param_hs)} CTX_DEP={len(ctxdep_hs)}", flush=True)

    print(f"Phase 1 done: scanned={n_scanned} PARAM={len(param_hs)} CTX_DEP={len(ctxdep_hs)}", flush=True)
    if len(param_hs) < 20 or len(ctxdep_hs) < 20:
        raise RuntimeError(f"Insufficient L1 data: PARAM={len(param_hs)} CTX_DEP={len(ctxdep_hs)}")

    n_min = min(len(param_hs), len(ctxdep_hs))
    X = np.stack(param_hs[:n_min] + ctxdep_hs[:n_min])
    y = np.array([1]*n_min + [0]*n_min)

    # Train/test split
    n_train = int(n_min * TRAIN_FRAC)
    idx = np.random.permutation(len(X))
    tr_idx, te_idx = idx[:n_train*2], idx[n_train*2:]
    # Balanced split
    param_idx = np.where(y == 1)[0]; ctx_idx = np.where(y == 0)[0]
    ptr = param_idx[:n_train]; pte = param_idx[n_train:n_min]
    ctr = ctx_idx[:n_train];   cte = ctx_idx[n_train:n_min]
    X_train = np.concatenate([X[ptr], X[ctr]])
    y_train = np.concatenate([np.ones(len(ptr)), np.zeros(len(ctr))])
    X_test  = np.concatenate([X[pte], X[cte]])
    y_test  = np.concatenate([np.ones(len(pte)), np.zeros(len(cte))])

    pca, lda, train_auroc, _ = fit_probe(X_train, y_train)
    test_auroc, _ = eval_probe(pca, lda, X_test, y_test)
    ci = bootstrap_ci(pca, lda, X_test, y_test)
    shuf = shuffled_auroc(pca, lda, X_test, y_test)

    shuf_status = "CLEAN" if shuf < 0.62 else ("WARN" if shuf < 0.70 else "FAIL")
    print(f"L1 test AUROC={test_auroc:.4f}  shuffled={shuf:.4f} ({shuf_status})", flush=True)
    print(f"L1 CI=[{ci[0]:.3f}, {ci[1]:.3f}]", flush=True)

    return {
        "n_param": len(param_hs), "n_ctxdep": len(ctxdep_hs),
        "n_scanned": n_scanned,
        "train_auroc": train_auroc, "test_auroc": test_auroc,
        "ci_95": list(ci), "shuffled_auroc": shuf,
        "shuffled_status": shuf_status,
        "l1_verdict": ("PASS" if test_auroc >= 0.65 and shuf_status != "FAIL" else "FAIL"),
    }, pca, lda, X, y


# ── Phase 2: L2 False Certainty ──────────────────────────────────────────────────
def run_phase2(model, tokenizer, pool, layer_idx, bo_pca, bo_lda):
    print("\n=== Phase 2: L2 False Certainty (CC vs CW) ===", flush=True)

    # Calibrate theta_conf: find 30th percentile entropy on pool sample
    print("  Calibrating entropy threshold …", flush=True)
    sample_ents = []
    for item in pool[:500]:
        pnc = prompt_nocontext(tokenizer, item["question"])
        ent = get_step1_entropy(model, tokenizer, pnc)
        sample_ents.append(ent)
    theta_conf = float(np.percentile(sample_ents, 30))
    ent_lo = theta_conf - ENT_HALF
    ent_hi = theta_conf + ENT_HALF
    print(f"  theta_conf={theta_conf:.4f}  window=[{ent_lo:.4f},{ent_hi:.4f}]", flush=True)

    cc_hs, cw_hs = [], []
    cc_ents, cw_ents = [], []
    n_scanned = 0

    for item in pool:
        if len(cc_hs) >= N_L2_TARGET and len(cw_hs) >= N_L2_TARGET:
            break
        n_scanned += 1
        q = item["question"]
        ans = item["answers"]

        pnc = prompt_nocontext(tokenizer, q)
        ent = get_step1_entropy(model, tokenizer, pnc)
        if not (ent_lo <= ent <= ent_hi):
            continue

        hs = get_step1_hs(model, tokenizer, pnc, layer_idx)
        if hs is None:
            continue

        gen_nc = generate(model, tokenizer, pnc)
        f1_nc = token_f1(gen_nc, ans)
        ok = answer_contains(gen_nc, ans) or f1_nc >= PARAM_MIN_F1

        if ok and len(cc_hs) < N_L2_TARGET:
            cc_hs.append(hs); cc_ents.append(ent)
        elif f1_nc <= CW_MAX_F1 and len(cw_hs) < N_L2_TARGET:
            cw_hs.append(hs); cw_ents.append(ent)

        if n_scanned % 500 == 0:
            print(f"  scanned={n_scanned} CC={len(cc_hs)} CW={len(cw_hs)}", flush=True)

        # Intermediate save
        if (len(cc_hs) + len(cw_hs)) % 20 == 0 and (len(cc_hs) + len(cw_hs)) > 0:
            try:
                with open(INTERMEDIATE, "w") as f:
                    json.dump({"cc": len(cc_hs), "cw": len(cw_hs), "scanned": n_scanned}, f)
            except Exception:
                pass

    print(f"Phase 2 done: scanned={n_scanned} CC={len(cc_hs)} CW={len(cw_hs)}", flush=True)
    if len(cc_hs) < 10 or len(cw_hs) < 10:
        print("WARNING: Insufficient L2 data. Skipping L2.", flush=True)
        return {"l2_skip": True, "l2_reason": "Insufficient items in entropy window"}

    n_min = min(len(cc_hs), len(cw_hs))
    X = np.stack(cc_hs[:n_min] + cw_hs[:n_min])
    y = np.array([1]*n_min + [0]*n_min)
    ents = np.array(cc_ents[:n_min] + cw_ents[:n_min])

    n_train = int(n_min * TRAIN_FRAC)
    cc_idx = np.where(y == 1)[0]; cw_idx = np.where(y == 0)[0]
    ptr = cc_idx[:n_train]; pte = cc_idx[n_train:n_min]
    ctr = cw_idx[:n_train]; cte = cw_idx[n_train:n_min]
    X_train = np.concatenate([X[ptr], X[ctr]])
    y_train = np.concatenate([np.ones(len(ptr)), np.zeros(len(ctr))])
    X_test  = np.concatenate([X[pte], X[cte]])
    y_test  = np.concatenate([np.ones(len(pte)), np.zeros(len(cte))])

    # Fisher AUROC on L2 task
    pca2, lda2, _, _ = fit_probe(X_train, y_train)
    fisher_auroc, _ = eval_probe(pca2, lda2, X_test, y_test)
    fisher_ci = bootstrap_ci(pca2, lda2, X_test, y_test)
    fisher_shuf = shuffled_auroc(pca2, lda2, X_test, y_test)

    # Entropy AUROC on L2 task
    from sklearn.metrics import roc_auc_score
    ents_test = ents[np.concatenate([pte, cte])]
    # Lower entropy → more likely CC (correct)
    try:
        ent_auroc = float(roc_auc_score(y_test, -ents_test))
    except Exception:
        ent_auroc = 0.5

    gap = fisher_auroc - ent_auroc

    # BO Transfer AUROC: apply Phase 1 probe to CC/CW test
    try:
        bo_transfer, _ = eval_probe(bo_pca, bo_lda, X_test, y_test)
    except Exception:
        bo_transfer = None

    shuf_status = "CLEAN" if fisher_shuf < 0.62 else "WARN"
    l2_verdict = "SUPPORTED" if fisher_auroc >= 0.70 and gap >= 0.10 else "NOT_SUPPORTED"

    cc_ent_mean = float(np.mean(cc_ents[:n_min]))
    cw_ent_mean = float(np.mean(cw_ents[:n_min]))

    print(f"L2 Fisher={fisher_auroc:.4f}  Entropy={ent_auroc:.4f}  Gap={gap:.4f}", flush=True)
    print(f"L2 shuffled={fisher_shuf:.4f} ({shuf_status})", flush=True)
    print(f"L2 BO_Transfer={bo_transfer:.4f}" if bo_transfer else "L2 BO_Transfer=N/A", flush=True)
    print(f"L2 VERDICT: {l2_verdict}", flush=True)

    return {
        "n_cc": n_min, "n_cw": n_min, "n_scanned": n_scanned,
        "theta_conf": theta_conf, "ent_window": [ent_lo, ent_hi],
        "cc_ent_mean": cc_ent_mean, "cw_ent_mean": cw_ent_mean,
        "fisher_auroc": fisher_auroc, "fisher_ci_95": list(fisher_ci),
        "fisher_shuffled": fisher_shuf, "shuffled_status": shuf_status,
        "entropy_auroc": ent_auroc,
        "gap": gap,
        "bo_transfer_auroc": bo_transfer,
        "l2_verdict": l2_verdict,
    }


# ── Main ──────────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    pool = load_pool(POOL_SIZE)
    model, tokenizer, n_layers = load_model()

    # Set layer index to penultimate
    global LAYER_IDX
    LAYER_IDX = n_layers - 2
    print(f"Probe layer: {LAYER_IDX} (penultimate of {n_layers})", flush=True)

    # Phase 1: L1
    l1_results, bo_pca, bo_lda, X_l1, y_l1 = run_phase1(model, tokenizer, pool, LAYER_IDX)

    # Phase 2: L2
    l2_results = run_phase2(model, tokenizer, pool, LAYER_IDX, bo_pca, bo_lda)

    # Summary
    results = {
        "experiment":    "EXP_GEMMA_BILATERAL_V1",
        "model":         MODEL_ID,
        "layer_idx":     LAYER_IDX,
        "n_layers":      n_layers,
        "pca_dim":       PCA_DIM,
        "n_l1_target":   N_L1_TARGET,
        "n_l2_target":   N_L2_TARGET,
        "pool_size":     POOL_SIZE,
        "l1": l1_results,
        "l2": l2_results,
        "elapsed_min":   (time.time() - t0) / 60,
        "architecture_note": "Gemma-2 (MQA variant, RoPE, GeGLU, Google family)",
    }

    # Termination rule check
    l1_auroc = l1_results.get("test_auroc", 0)
    l2_gap   = l2_results.get("gap", 0) if not l2_results.get("l2_skip") else 0
    if l1_auroc < 0.65:
        results["termination"] = "T1_PARTIAL: L1 < 0.65; characterize as architecture-specific"
    elif l2_gap < 0.05 and not l2_results.get("l2_skip"):
        results["termination"] = "T2_L2: L2 gap < 0.05; L2 does not generalize to Gemma"
    else:
        results["termination"] = "NO_TERMINATION"

    print(f"\n{'='*50}", flush=True)
    print(f"EXP_GEMMA_BILATERAL_V1 RESULTS", flush=True)
    print(f"  L1 AUROC = {l1_auroc:.4f}  shuffled={l1_results.get('shuffled_auroc',0):.4f}", flush=True)
    if not l2_results.get("l2_skip"):
        print(f"  L2 Fisher = {l2_results.get('fisher_auroc',0):.4f}  gap={l2_gap:.4f}", flush=True)
    print(f"  Termination: {results['termination']}", flush=True)
    print(f"  Elapsed: {results['elapsed_min']:.1f} min", flush=True)

    with open(SAVE_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults → {SAVE_PATH}", flush=True)


if __name__ == "__main__":
    main()
