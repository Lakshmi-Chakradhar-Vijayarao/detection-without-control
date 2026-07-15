#!/usr/bin/env python3
"""
l2_large_n_v1.py — EXP_L2_LARGE_N_V1

MOTIVATED BY: C034 variance concern.
  Prior L2 experiments (N=100/class, single 75/25 split) show AUROC ranging from
  0.670 to 0.854 across different TriviaQA subsets. Small n_test=25-50 gives wide
  bootstrap CIs. The true mean Fisher L2 AUROC is unknown.

SCIENTIFIC QUESTION:
  What is the stable cross-validated Fisher L2 AUROC and gap on TriviaQA?
  Does the item-level distribution explain the 0.670-0.854 variance?

DESIGN:
  Collect N=500/class CC/CW items (entropy-matched, CO-style: entropy ≤ theta_conf).
  Run 5-fold stratified cross-validation:
    Each fold: train on 400/class, test on 100/class.
    Report mean ± std AUROC across folds.
  Also report:
    - Item-level Fisher score distribution (CC vs CW histogram)
    - Per-decile entropy breakdown: does Fisher gap vary across confidence levels?
    - Entropy AUROC as baseline (same cross-validation)

  Use CO-style collection (entropy ≤ theta_conf) since C034 shows CO ≈ BO.

VERDICTS:
  STABLE_SIGNAL: CV AUROC mean ≥ 0.75, std ≤ 0.05
  MODERATE_SIGNAL: CV AUROC mean ∈ [0.65, 0.75)
  WEAK_SIGNAL: CV AUROC mean < 0.65

GPU: T4 (~12h for N=500/class collection)
Model: Qwen/Qwen2.5-1.5B-Instruct
"""

from __future__ import annotations
import gc, json, os, random, time
import numpy as np
import torch

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

MODEL_ID        = "Qwen/Qwen2.5-1.5B-Instruct"
LAYER_IDX       = 26
N_TARGET        = 500    # per class
N_FOLDS         = 5
POOL_SIZE       = 20_000
PCA_DIM         = 64
MAX_NEW         = 60
PARAM_MIN_F1    = 0.50
CW_MAX_F1       = 0.05

SAVE_PATH       = "/kaggle/working/l2_large_n_v1_results.json"
INTERMEDIATE    = "/kaggle/working/l2_large_n_v1_intermediate.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    raise RuntimeError("GPU required.")
print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB", flush=True)


# ── HF token ─────────────────────────────────────────────────────────────────────
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


# ── Data ─────────────────────────────────────────────────────────────────────────
def load_pool(n: int = POOL_SIZE):
    from datasets import load_dataset
    ds = load_dataset("trivia_qa", "rc.wikipedia", split="train", streaming=True)
    items = []
    for row in ds:
        if not row["answer"]["aliases"]:
            continue
        items.append({"question": row["question"], "answers": row["answer"]["aliases"]})
        if len(items) >= n:
            break
    np.random.shuffle(items)
    print(f"Pool loaded: {len(items)} items", flush=True)
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


# ── Model ─────────────────────────────────────────────────────────────────────────
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
    print(f"Loaded: {model.config.num_hidden_layers} layers", flush=True)
    return model, tokenizer


def prompt_nc(tokenizer, q: str) -> str:
    msgs = [{"role": "user", "content": q}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def get_hs_and_entropy(model, tokenizer, prompt: str):
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    hs_out = [None]

    def hook(mod, inp, out):
        x = out[0] if isinstance(out, tuple) else out
        hs_out[0] = x[:, -1, :].detach().float().cpu().numpy()

    h = model.model.layers[LAYER_IDX].register_forward_hook(hook)
    with torch.no_grad():
        out = model(ids)
    h.remove()

    logits = out.logits[0, -1, :].float()
    logits = torch.nan_to_num(logits, nan=0.0, posinf=80.0, neginf=-80.0)
    probs  = torch.softmax(logits, dim=-1)
    ent    = float(-torch.sum(probs * torch.log(probs + 1e-10)).item())
    if not np.isfinite(ent):
        ent = 0.0
    hs = hs_out[0][0] if hs_out[0] is not None else None
    return hs, ent


def generate(model, tokenizer, prompt: str):
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=MAX_NEW, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


# ── Cross-validated probe ─────────────────────────────────────────────────────────
def cross_validate(X, y, n_folds=N_FOLDS):
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    fold_aurocs = []
    fold_ent_aurocs = []

    # Build entropy array (stored separately alongside X)
    # X has last column as entropy — we split it off here
    X_hs = X[:, :-1]
    ents = X[:, -1]

    for fold_i, (tr_idx, te_idx) in enumerate(skf.split(X_hs, y)):
        pca = PCA(n_components=min(PCA_DIM, X_hs.shape[1], len(tr_idx)-1))
        Xp_tr = pca.fit_transform(X_hs[tr_idx])
        lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        lda.fit(Xp_tr, y[tr_idx])

        Xp_te = pca.transform(X_hs[te_idx])
        scores = lda.decision_function(Xp_te)
        auroc = float(roc_auc_score(y[te_idx], scores))
        fold_aurocs.append(auroc)

        try:
            ent_auroc = float(roc_auc_score(y[te_idx], -ents[te_idx]))
        except Exception:
            ent_auroc = 0.5
        fold_ent_aurocs.append(ent_auroc)

        print(f"  Fold {fold_i+1}/{n_folds}: Fisher={auroc:.4f}  Entropy={ent_auroc:.4f}", flush=True)

    return {
        "fold_fisher_aurocs": fold_aurocs,
        "fold_entropy_aurocs": fold_ent_aurocs,
        "mean_fisher": float(np.mean(fold_aurocs)),
        "std_fisher": float(np.std(fold_aurocs)),
        "mean_entropy": float(np.mean(fold_ent_aurocs)),
        "std_entropy": float(np.std(fold_ent_aurocs)),
        "mean_gap": float(np.mean(fold_aurocs) - np.mean(fold_ent_aurocs)),
    }


# ── Per-entropy-decile breakdown ─────────────────────────────────────────────────
def decile_breakdown(X_hs, ents, y):
    """Fisher AUROC within each entropy decile."""
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import roc_auc_score

    decile_results = []
    decile_edges = np.percentile(ents, np.arange(0, 110, 10))
    for i in range(10):
        lo, hi = decile_edges[i], decile_edges[i+1]
        mask = (ents >= lo) & (ents < hi) if i < 9 else (ents >= lo) & (ents <= hi)
        X_d, y_d = X_hs[mask], y[mask]
        n_min = min(np.sum(y_d==0), np.sum(y_d==1))
        if n_min < 10:
            decile_results.append({"decile": i, "n": int(mask.sum()), "auroc": None})
            continue
        idx0 = np.where(y_d==0)[0][:n_min]; idx1 = np.where(y_d==1)[0][:n_min]
        Xb = np.concatenate([X_d[idx0], X_d[idx1]])
        yb = np.array([0]*n_min + [1]*n_min)
        try:
            pca = PCA(n_components=min(PCA_DIM, Xb.shape[1], len(yb)-1))
            Xp = pca.fit_transform(Xb)
            lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
            lda.fit(Xp, yb)
            sc = lda.decision_function(Xp)
            au = float(roc_auc_score(yb, sc))
        except Exception:
            au = None
        decile_results.append({
            "decile": i, "n": int(mask.sum()), "ent_lo": float(lo),
            "ent_hi": float(hi), "auroc": au
        })
    return decile_results


# ── Collection ────────────────────────────────────────────────────────────────────
def collect_items(model, tokenizer, pool, theta_conf):
    print(f"\n=== Collecting N={N_TARGET}/class (confident filter: ent ≤ {theta_conf:.4f}) ===", flush=True)
    cc_hs, cc_ents = [], []
    cw_hs, cw_ents = [], []
    n_scanned = 0

    for item in pool:
        if len(cc_hs) >= N_TARGET and len(cw_hs) >= N_TARGET:
            break
        n_scanned += 1
        q = item["question"]
        ans = item["answers"]
        pnc = prompt_nc(tokenizer, q)

        hs, ent = get_hs_and_entropy(model, tokenizer, pnc)
        if hs is None or ent > theta_conf:
            continue

        gen = generate(model, tokenizer, pnc)
        f1 = token_f1(gen, ans)
        ok = answer_contains(gen, ans) or f1 >= PARAM_MIN_F1

        if ok and len(cc_hs) < N_TARGET:
            cc_hs.append(hs); cc_ents.append(ent)
        elif f1 <= CW_MAX_F1 and len(cw_hs) < N_TARGET:
            cw_hs.append(hs); cw_ents.append(ent)

        if n_scanned % 500 == 0:
            print(f"  scanned={n_scanned} CC={len(cc_hs)} CW={len(cw_hs)}", flush=True)

        if (len(cc_hs) + len(cw_hs)) % 100 == 0 and (len(cc_hs) + len(cw_hs)) > 0:
            try:
                with open(INTERMEDIATE, "w") as f:
                    json.dump({"cc": len(cc_hs), "cw": len(cw_hs), "scanned": n_scanned}, f)
            except Exception:
                pass

    print(f"Collection done: scanned={n_scanned} CC={len(cc_hs)} CW={len(cw_hs)}", flush=True)
    if len(cc_hs) < 100 or len(cw_hs) < 100:
        raise RuntimeError(f"Insufficient data: CC={len(cc_hs)} CW={len(cw_hs)}")

    n_min = min(len(cc_hs), len(cw_hs))
    return cc_hs[:n_min], cc_ents[:n_min], cw_hs[:n_min], cw_ents[:n_min], n_scanned


# ── Main ──────────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    pool = load_pool(POOL_SIZE)
    model, tokenizer = load_model()

    # Calibrate theta_conf
    print("Calibrating entropy threshold …", flush=True)
    calib_ents = []
    for item in pool[:800]:
        _, ent = get_hs_and_entropy(model, tokenizer, prompt_nc(tokenizer, item["question"]))
        calib_ents.append(ent)
    calib_ents = [e for e in calib_ents if np.isfinite(e) and e > 0]
    if not calib_ents:
        raise RuntimeError("All calibration entropies invalid")
    theta_conf = float(np.percentile(calib_ents, 30))
    print(f"  theta_conf={theta_conf:.4f} (30th pctile of {len(calib_ents)} samples)", flush=True)

    # Collect
    cc_hs, cc_ents, cw_hs, cw_ents, n_scanned = collect_items(
        model, tokenizer, pool[800:], theta_conf
    )
    n = min(len(cc_hs), len(cw_hs))
    print(f"\n  Final N: {n}/class", flush=True)

    # Build feature matrix (hidden states + entropy as last column for CV)
    X_hs = np.stack(cc_hs[:n] + cw_hs[:n])
    ents_all = np.array(cc_ents[:n] + cw_ents[:n])
    y = np.array([1]*n + [0]*n)
    X = np.concatenate([X_hs, ents_all.reshape(-1, 1)], axis=1)

    # Cross-validation
    print(f"\n=== {N_FOLDS}-fold cross-validation ===", flush=True)
    cv_result = cross_validate(X, y, N_FOLDS)
    print(f"\nCV Fisher: mean={cv_result['mean_fisher']:.4f}  std={cv_result['std_fisher']:.4f}", flush=True)
    print(f"CV Entropy: mean={cv_result['mean_entropy']:.4f}  std={cv_result['std_entropy']:.4f}", flush=True)
    print(f"CV Gap: {cv_result['mean_gap']:.4f}", flush=True)

    # Verdict
    if cv_result["mean_fisher"] >= 0.75 and cv_result["std_fisher"] <= 0.05:
        verdict = "STABLE_SIGNAL"
    elif cv_result["mean_fisher"] >= 0.65:
        verdict = "MODERATE_SIGNAL"
    else:
        verdict = "WEAK_SIGNAL"

    # Decile breakdown
    print("\n=== Per-entropy-decile breakdown ===", flush=True)
    deciles = decile_breakdown(X_hs, ents_all, y)
    for d in deciles:
        if d["auroc"] is not None:
            print(f"  Decile {d['decile']}: ent=[{d.get('ent_lo', 0):.3f},{d.get('ent_hi', 0):.3f}] n={d['n']} Fisher={d['auroc']:.4f}", flush=True)

    results = {
        "experiment": "EXP_L2_LARGE_N_V1",
        "model": MODEL_ID,
        "layer_idx": LAYER_IDX,
        "n_per_class": n,
        "n_folds": N_FOLDS,
        "theta_conf": theta_conf,
        "n_scanned": n_scanned,
        "cc_ent_mean": float(np.mean(cc_ents[:n])),
        "cw_ent_mean": float(np.mean(cw_ents[:n])),
        "cv": cv_result,
        "decile_breakdown": deciles,
        "verdict": verdict,
        "elapsed_min": (time.time() - t0) / 60,
    }

    print(f"\n{'='*50}", flush=True)
    print(f"EXP_L2_LARGE_N_V1: {verdict}", flush=True)
    print(f"  CV Fisher AUROC = {cv_result['mean_fisher']:.4f} ± {cv_result['std_fisher']:.4f}", flush=True)
    print(f"  CV Gap          = {cv_result['mean_gap']:.4f}", flush=True)
    print(f"  Elapsed: {results['elapsed_min']:.1f} min", flush=True)

    with open(SAVE_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults → {SAVE_PATH}", flush=True)


if __name__ == "__main__":
    main()
