#!/usr/bin/env python3
"""
co_l2_v1.py — EXP_CO_L2_V1

MOTIVATED BY: EXP_0_BO_ABLATION_V1 result — CO_BETTER (AUROC 0.885 vs 0.806)
  At L1, Fisher+PCA64 separates PARAM from any-wrong BETTER than from CTX_DEP alone.
  CTX_DEP items are geometrically closest to PARAM (borderline cases).
  This experiment tests whether the same holds at L2.

SCIENTIFIC QUESTION:
  Does CO-style labeling (confident-correct vs confident-wrong, no entropy matching)
  yield a larger Fisher gap than the BO-style L2 (entropy-matched CC/CW)?

  BO-L2 (current): items MUST fall in calibrated entropy window [θ-Δ, θ+Δ].
    → Selects hardest cases; confirmed gap 0.240 (Qwen) / 0.365 (Llama).
  CO-L2 (this experiment): items only need entropy ≤ θ_conf (30th pctile).
    → Broader "confident" set; no entropy matching between CC/CW groups.
    → Prediction: higher AUROC because CO avoids selecting borderline cases.

DESIGN:
  Phase 1 — CO-L2 collection (N=200/class):
    CC items: entropy ≤ θ_conf AND correct (answer_contains or F1 ≥ 0.50)
    CW items: entropy ≤ θ_conf AND wrong (F1 ≤ 0.05)
    No entropy window — just confident filter.

  Phase 2 — BO-L2 collection (N=100/class, same pool continuation):
    CC/CW items in entropy window [θ_conf - 0.30, θ_conf + 0.30].
    Direct comparison with CO-L2 on same model/pool.

  Phase 3 — Probe comparison:
    Fisher+PCA64 on CO-L2 vs BO-L2 items.
    Entropy AUROC as baseline for each.
    Gaps: CO_gap = Fisher_CO - Entropy_CO; BO_gap = Fisher_BO - Entropy_BO.

VERDICTS:
  CO_LARGER: CO_gap > BO_gap + 0.05 → CO labeling is strictly better
  BO_LARGER: BO_gap > CO_gap + 0.05 → entropy matching helps (unexpected)
  EQUIVALENT: |CO_gap - BO_gap| ≤ 0.05 → labeling strategy doesn't matter

GPU: T4 (~6h)
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
N_CO_TARGET     = 200   # per class for CO-L2 (no entropy matching)
N_BO_TARGET     = 100   # per class for BO-L2 (entropy matched)
POOL_SIZE       = 12_000
N_BOOTSTRAP     = 1000
TRAIN_FRAC      = 0.75
PCA_DIM         = 64
MAX_NEW         = 60
PARAM_MIN_F1    = 0.50
CW_MAX_F1       = 0.05
ENT_HALF        = 0.30   # for BO-L2 window

SAVE_PATH       = "/kaggle/working/co_l2_v1_results.json"
INTERMEDIATE    = "/kaggle/working/co_l2_v1_intermediate.json"

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
        items.append({
            "question": row["question"],
            "answers":  row["answer"]["aliases"],
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
    n_layers = model.config.num_hidden_layers
    print(f"Loaded: {n_layers} layers", flush=True)
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


# ── Probe ─────────────────────────────────────────────────────────────────────────
def fit_and_eval(X, y, train_frac=TRAIN_FRAC, n_bootstrap=N_BOOTSTRAP):
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import roc_auc_score

    n_min_class = min(np.sum(y == 0), np.sum(y == 1))
    n_train = int(n_min_class * train_frac)
    idx0 = np.where(y == 0)[0]; idx1 = np.where(y == 1)[0]
    tr = np.concatenate([idx0[:n_train], idx1[:n_train]])
    te = np.concatenate([idx0[n_train:n_min_class], idx1[n_train:n_min_class]])

    pca = PCA(n_components=min(PCA_DIM, X.shape[1], len(tr)-1))
    Xp_tr = pca.fit_transform(X[tr])
    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    lda.fit(Xp_tr, y[tr])

    Xp_te = pca.transform(X[te])
    scores = lda.decision_function(Xp_te)
    auroc = float(roc_auc_score(y[te], scores))

    # shuffled control
    y_shuf = y[te].copy(); np.random.shuffle(y_shuf)
    try:
        shuf = float(roc_auc_score(y_shuf, scores))
    except Exception:
        shuf = 0.5

    # bootstrap CI
    aurocs = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(te), len(te), replace=True)
        try:
            aurocs.append(float(roc_auc_score(y[te][idx], scores[idx])))
        except Exception:
            pass
    ci = (float(np.percentile(aurocs, 2.5)), float(np.percentile(aurocs, 97.5))) if aurocs else (0.0, 0.0)

    return auroc, shuf, ci, pca, lda


def entropy_auroc(ents, y):
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(y, -np.array(ents)))
    except Exception:
        return 0.5


# ── Phase 1: CO-L2 collection ────────────────────────────────────────────────────
def collect_co_l2(model, tokenizer, pool, theta_conf):
    print(f"\n=== Phase 1: CO-L2 collection (N={N_CO_TARGET}/class) ===", flush=True)
    print(f"  Confident filter: entropy ≤ {theta_conf:.4f}", flush=True)

    cc_hs, cc_ents = [], []
    cw_hs, cw_ents = [], []
    n_scanned = 0

    for item in pool:
        if len(cc_hs) >= N_CO_TARGET and len(cw_hs) >= N_CO_TARGET:
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

        if ok and len(cc_hs) < N_CO_TARGET:
            cc_hs.append(hs); cc_ents.append(ent)
        elif f1 <= CW_MAX_F1 and len(cw_hs) < N_CO_TARGET:
            cw_hs.append(hs); cw_ents.append(ent)

        if n_scanned % 200 == 0:
            print(f"  scanned={n_scanned} CC={len(cc_hs)} CW={len(cw_hs)}", flush=True)
        if (len(cc_hs) + len(cw_hs)) % 50 == 0 and (len(cc_hs) + len(cw_hs)) > 0:
            try:
                with open(INTERMEDIATE, "w") as f:
                    json.dump({"phase": "co_l2", "cc": len(cc_hs), "cw": len(cw_hs)}, f)
            except Exception:
                pass

    print(f"CO-L2 done: scanned={n_scanned} CC={len(cc_hs)} CW={len(cw_hs)}", flush=True)
    if len(cc_hs) < 20 or len(cw_hs) < 20:
        raise RuntimeError(f"Insufficient CO-L2 data: CC={len(cc_hs)} CW={len(cw_hs)}")

    return cc_hs, cc_ents, cw_hs, cw_ents, n_scanned


# ── Phase 2: BO-L2 collection (entropy-matched) ───────────────────────────────────
def collect_bo_l2(model, tokenizer, pool, theta_conf, co_start_idx):
    print(f"\n=== Phase 2: BO-L2 collection (N={N_BO_TARGET}/class, entropy-matched) ===", flush=True)
    ent_lo = theta_conf - ENT_HALF
    ent_hi = theta_conf + ENT_HALF
    print(f"  Entropy window: [{ent_lo:.4f}, {ent_hi:.4f}]", flush=True)

    cc_hs, cc_ents = [], []
    cw_hs, cw_ents = [], []
    n_scanned = 0

    for item in pool[co_start_idx:]:
        if len(cc_hs) >= N_BO_TARGET and len(cw_hs) >= N_BO_TARGET:
            break
        n_scanned += 1
        q = item["question"]
        ans = item["answers"]
        pnc = prompt_nc(tokenizer, q)

        hs, ent = get_hs_and_entropy(model, tokenizer, pnc)
        if hs is None or not (ent_lo <= ent <= ent_hi):
            continue

        gen = generate(model, tokenizer, pnc)
        f1 = token_f1(gen, ans)
        ok = answer_contains(gen, ans) or f1 >= PARAM_MIN_F1

        if ok and len(cc_hs) < N_BO_TARGET:
            cc_hs.append(hs); cc_ents.append(ent)
        elif f1 <= CW_MAX_F1 and len(cw_hs) < N_BO_TARGET:
            cw_hs.append(hs); cw_ents.append(ent)

        if n_scanned % 200 == 0:
            print(f"  scanned={n_scanned} CC={len(cc_hs)} CW={len(cw_hs)}", flush=True)

    print(f"BO-L2 done: scanned={n_scanned} CC={len(cc_hs)} CW={len(cw_hs)}", flush=True)
    if len(cc_hs) < 10 or len(cw_hs) < 10:
        print("WARNING: Insufficient BO-L2 data — skipping comparison", flush=True)
        return None, None, None, None, n_scanned

    return cc_hs, cc_ents, cw_hs, cw_ents, n_scanned


# ── Main ──────────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    pool = load_pool(POOL_SIZE)
    model, tokenizer = load_model()

    # Calibrate theta_conf
    print("Calibrating entropy threshold …", flush=True)
    calib_ents = []
    for item in pool[:500]:
        _, ent = get_hs_and_entropy(model, tokenizer, prompt_nc(tokenizer, item["question"]))
        calib_ents.append(ent)
    calib_ents = [e for e in calib_ents if np.isfinite(e) and e > 0]
    if not calib_ents:
        raise RuntimeError("All calibration entropies NaN")
    theta_conf = float(np.percentile(calib_ents, 30))
    print(f"  theta_conf={theta_conf:.4f} (30th pctile of {len(calib_ents)} finite samples)", flush=True)

    # Phase 1: CO-L2
    co_cc_hs, co_cc_ents, co_cw_hs, co_cw_ents, co_scanned = collect_co_l2(
        model, tokenizer, pool, theta_conf
    )

    n_co = min(len(co_cc_hs), len(co_cw_hs))
    X_co = np.stack(co_cc_hs[:n_co] + co_cw_hs[:n_co])
    y_co = np.array([1]*n_co + [0]*n_co)
    ents_co = np.array(co_cc_ents[:n_co] + co_cw_ents[:n_co])

    co_fisher_auroc, co_shuf, co_ci, _, _ = fit_and_eval(X_co, y_co)
    co_entropy_auroc = entropy_auroc(ents_co, y_co)
    co_gap = co_fisher_auroc - co_entropy_auroc

    print(f"\nCO-L2: Fisher={co_fisher_auroc:.4f}  Entropy={co_entropy_auroc:.4f}  Gap={co_gap:.4f}", flush=True)
    print(f"CO-L2: shuffled={co_shuf:.4f}  CI={co_ci}", flush=True)

    co_result = {
        "n_cc": n_co, "n_cw": n_co,
        "n_scanned": co_scanned,
        "theta_conf": theta_conf,
        "cc_ent_mean": float(np.mean(co_cc_ents[:n_co])),
        "cw_ent_mean": float(np.mean(co_cw_ents[:n_co])),
        "ent_diff_nats": float(np.mean(co_cw_ents[:n_co]) - np.mean(co_cc_ents[:n_co])),
        "fisher_auroc": co_fisher_auroc,
        "fisher_ci_95": list(co_ci),
        "fisher_shuffled": co_shuf,
        "entropy_auroc": co_entropy_auroc,
        "gap": co_gap,
    }

    # Phase 2: BO-L2 (entropy-matched, continuation of pool)
    bo_cc_hs, bo_cc_ents, bo_cw_hs, bo_cw_ents, bo_scanned = collect_bo_l2(
        model, tokenizer, pool, theta_conf, co_scanned
    )

    bo_result = {"skipped": True}
    if bo_cc_hs is not None:
        n_bo = min(len(bo_cc_hs), len(bo_cw_hs))
        X_bo = np.stack(bo_cc_hs[:n_bo] + bo_cw_hs[:n_bo])
        y_bo = np.array([1]*n_bo + [0]*n_bo)
        ents_bo = np.array(bo_cc_ents[:n_bo] + bo_cw_ents[:n_bo])

        bo_fisher_auroc, bo_shuf, bo_ci, _, _ = fit_and_eval(X_bo, y_bo)
        bo_entropy_auroc = entropy_auroc(ents_bo, y_bo)
        bo_gap = bo_fisher_auroc - bo_entropy_auroc

        print(f"\nBO-L2: Fisher={bo_fisher_auroc:.4f}  Entropy={bo_entropy_auroc:.4f}  Gap={bo_gap:.4f}", flush=True)

        gap_diff = co_gap - bo_gap
        if gap_diff > 0.05:
            verdict = "CO_LARGER"
        elif gap_diff < -0.05:
            verdict = "BO_LARGER"
        else:
            verdict = "EQUIVALENT"

        bo_result = {
            "n_cc": n_bo, "n_cw": n_bo,
            "n_scanned": bo_scanned,
            "cc_ent_mean": float(np.mean(bo_cc_ents[:n_bo])),
            "cw_ent_mean": float(np.mean(bo_cw_ents[:n_bo])),
            "fisher_auroc": bo_fisher_auroc,
            "fisher_ci_95": list(bo_ci),
            "fisher_shuffled": bo_shuf,
            "entropy_auroc": bo_entropy_auroc,
            "gap": bo_gap,
        }
    else:
        verdict = "BO_SKIPPED"
        gap_diff = None

    print(f"\n{'='*50}", flush=True)
    print(f"EXP_CO_L2_V1 RESULTS", flush=True)
    print(f"  CO-L2: Fisher={co_fisher_auroc:.4f}  Gap={co_gap:.4f}", flush=True)
    if not bo_result.get("skipped"):
        print(f"  BO-L2: Fisher={bo_result['fisher_auroc']:.4f}  Gap={bo_result['gap']:.4f}", flush=True)
        print(f"  Gap diff (CO-BO): {gap_diff:+.4f}  VERDICT: {verdict}", flush=True)
    print(f"  Elapsed: {(time.time()-t0)/60:.1f} min", flush=True)

    results = {
        "experiment": "EXP_CO_L2_V1",
        "model": MODEL_ID,
        "layer_idx": LAYER_IDX,
        "motivation": "CO_BETTER from EXP_0_BO_ABLATION_V1 (CO AUROC=0.885 vs BO AUROC=0.806)",
        "co_l2": co_result,
        "bo_l2": bo_result,
        "verdict": verdict,
        "gap_diff_co_minus_bo": gap_diff,
        "elapsed_min": (time.time() - t0) / 60,
    }

    with open(SAVE_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults → {SAVE_PATH}", flush=True)


if __name__ == "__main__":
    main()
