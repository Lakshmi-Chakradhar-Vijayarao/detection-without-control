#!/usr/bin/env python3
"""
bo_ablation_v1.py — EXP_0_BO_ABLATION_V1
EIG rank #1. Tests whether the two-pass bilateral oracle labeling adds over
simpler correctness-only labeling.

SCIENTIFIC QUESTION:
  Do bilateral oracle labels (PARAM vs CTX_DEP, pure knowledge-source separation)
  yield a better probe than correctness labels (CORRECT vs INCORRECT_NOCONTEXT)?

DESIGN — two probes, same hidden states:
  Condition A — Bilateral Oracle (BO):
    PARAM:   nocontext F1 >= 0.50
    CTX_DEP: nocontext F1 <= 0.05 AND withcontext F1 >= 0.50

  Condition B — Correctness-Only (CO):
    CORRECT:   nocontext F1 >= 0.50
    INCORRECT: nocontext F1 <= 0.05 (any reason — CONFAB or genuinely missing ctx)
    (Superset of CTX_DEP — includes items that would fail withcontext too)

  Same N, same items for CORRECT/PARAM class.
  Difference: CO "wrong" class includes items that lack context AND still can't answer.
  BO "CTX_DEP" class requires withcontext recovery.

HYPOTHESIS:
  If bilateral oracle adds: A_AUROC > B_AUROC by margin > 0.03
  If equal: bilateral oracle is redundant protocol — scope contraction for Paper A

SETUP:
  Model: Qwen2.5-1.5B-Instruct (architecture already validated, fast)
  Layer: 26 (established optimal for Qwen)
  N=200/class
  Pool=10000
  Bootstrap CI (n=1000)
  Shuffled control for both probes

GPU: T4 (~3h)
"""

from __future__ import annotations
import gc, json, os, random, time
import numpy as np
import torch

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

MODEL_ID        = "Qwen/Qwen2.5-1.5B-Instruct"
LAYER_IDX       = 26
N_TARGET        = 200
POOL_SIZE       = 10_000
N_BOOTSTRAP     = 1000
TRAIN_FRAC      = 0.75
PCA_DIM         = 64
MAX_NEW         = 60
PARAM_MIN_F1    = 0.50
CW_MAX_F1_NC    = 0.05
CTX_MIN_F1_WC   = 0.50

SAVE_PATH       = "/kaggle/working/bo_ablation_v1_results.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    raise RuntimeError("GPU required.")
print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)


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
    n_layers = model.config.num_hidden_layers
    print(f"  Loaded: {n_layers} layers", flush=True)
    return model, tokenizer


def get_step1_hs(model, tokenizer, prompt: str, layer_idx: int):
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    hs_out = [None]

    def hook(mod, inp, out):
        x = out[0] if isinstance(out, tuple) else out
        hs_out[0] = x[:, -1, :].detach().float().cpu().numpy()

    h = model.model.layers[layer_idx].register_forward_hook(hook)
    with torch.no_grad():
        model(ids)
    h.remove()
    return hs_out[0][0] if hs_out[0] is not None else None


def generate(model, tokenizer, prompt: str):
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=MAX_NEW, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


def prompt_nocontext(tokenizer, q: str) -> str:
    msgs = [{"role": "user", "content": q}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def prompt_withcontext(tokenizer, q: str, ctx: str) -> str:
    content = f"Context: {ctx[:600]}\n\nQuestion: {q}"
    msgs = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


# ── Probe ─────────────────────────────────────────────────────────────────────────
def fit_and_eval_probe(X_train, y_train, X_test, y_test, n_bootstrap=N_BOOTSTRAP):
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import roc_auc_score

    pca = PCA(n_components=min(PCA_DIM, X_train.shape[1], X_train.shape[0]-1))
    Xtr_p = pca.fit_transform(X_train)
    Xte_p = pca.transform(X_test)

    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    lda.fit(Xtr_p, y_train)

    scores_te = lda.decision_function(Xte_p)
    auroc = float(roc_auc_score(y_test, scores_te))

    # Bootstrap CI
    aurocs = []
    n = len(y_test)
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        try:
            aurocs.append(float(roc_auc_score(y_test[idx], scores_te[idx])))
        except Exception:
            pass
    ci = (float(np.percentile(aurocs, 2.5)), float(np.percentile(aurocs, 97.5))) if aurocs else (0., 0.)

    # Shuffled control
    y_shuf = y_test.copy(); np.random.shuffle(y_shuf)
    try:
        shuf_auroc = float(roc_auc_score(y_shuf, scores_te))
    except Exception:
        shuf_auroc = 0.5

    return auroc, ci, shuf_auroc


# ── Data collection ───────────────────────────────────────────────────────────────
def collect_all(model, tokenizer, pool):
    """
    Single pass — collect hidden states for all four item classes:
      - param_hs:    nocontext correct
      - ctxdep_hs:   nocontext wrong + withcontext correct (bilateral oracle)
      - incorrect_hs: nocontext wrong (any reason — correctness-only superset)
    """
    print("\n=== Data Collection ===", flush=True)
    param_hs     = []
    ctxdep_hs    = []
    incorrect_hs = []
    n_scanned = 0

    for item in pool:
        if (len(param_hs) >= N_TARGET
                and len(ctxdep_hs) >= N_TARGET
                and len(incorrect_hs) >= N_TARGET):
            break
        n_scanned += 1
        q   = item["question"]
        ans = item["answers"]
        ctx = item["context"]

        pnc = prompt_nocontext(tokenizer, q)
        hs  = get_step1_hs(model, tokenizer, pnc, LAYER_IDX)
        if hs is None:
            continue

        gen_nc = generate(model, tokenizer, pnc)
        f1_nc  = token_f1(gen_nc, ans)
        ok_nc  = answer_contains(gen_nc, ans) or f1_nc >= PARAM_MIN_F1

        if ok_nc:
            if len(param_hs) < N_TARGET:
                param_hs.append(hs)
        else:
            # Wrong nocontext — qualifies for CO incorrect class
            if f1_nc <= CW_MAX_F1_NC and len(incorrect_hs) < N_TARGET:
                incorrect_hs.append(hs)

            # Check BO bilateral condition (withcontext recovery)
            if f1_nc <= CW_MAX_F1_NC and ctx and len(ctxdep_hs) < N_TARGET:
                pwc    = prompt_withcontext(tokenizer, q, ctx)
                gen_wc = generate(model, tokenizer, pwc)
                f1_wc  = token_f1(gen_wc, ans)
                ok_wc  = answer_contains(gen_wc, ans) or f1_wc >= CTX_MIN_F1_WC
                if ok_wc:
                    ctxdep_hs.append(hs)

        if n_scanned % 200 == 0:
            print(f"  scanned={n_scanned} PARAM={len(param_hs)} "
                  f"CTX_DEP={len(ctxdep_hs)} CO_INCORR={len(incorrect_hs)}", flush=True)

    print(f"Collection done: scanned={n_scanned} "
          f"PARAM={len(param_hs)} CTX_DEP={len(ctxdep_hs)} CO_INCORR={len(incorrect_hs)}", flush=True)
    return param_hs, ctxdep_hs, incorrect_hs


# ── Probe comparison ─────────────────────────────────────────────────────────────
def compare_probes(param_hs, ctxdep_hs, incorrect_hs):
    print("\n=== Probe Comparison ===", flush=True)
    n_bo = min(len(param_hs), len(ctxdep_hs))
    n_co = min(len(param_hs), len(incorrect_hs))

    if n_bo < 20 or n_co < 20:
        return {"error": f"Insufficient data: n_bo={n_bo} n_co={n_co}"}

    def split(pos_hs, neg_hs, n):
        X = np.stack(pos_hs[:n] + neg_hs[:n])
        y = np.array([1]*n + [0]*n)
        n_train = int(n * TRAIN_FRAC)
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        ptr = pos_idx[:n_train]; pte = pos_idx[n_train:n]
        ntr = neg_idx[:n_train]; nte = neg_idx[n_train:n]
        X_train = np.concatenate([X[ptr], X[ntr]])
        y_train = np.concatenate([np.ones(len(ptr)), np.zeros(len(ntr))])
        X_test  = np.concatenate([X[pte], X[nte]])
        y_test  = np.concatenate([np.ones(len(pte)), np.zeros(len(nte))])
        return X_train, y_train, X_test, y_test

    # Condition A: Bilateral Oracle (PARAM vs CTX_DEP)
    X_tr_bo, y_tr_bo, X_te_bo, y_te_bo = split(param_hs, ctxdep_hs, n_bo)
    auroc_bo, ci_bo, shuf_bo = fit_and_eval_probe(X_tr_bo, y_tr_bo, X_te_bo, y_te_bo)

    # Condition B: Correctness-Only (PARAM vs INCORRECT)
    X_tr_co, y_tr_co, X_te_co, y_te_co = split(param_hs, incorrect_hs, n_co)
    auroc_co, ci_co, shuf_co = fit_and_eval_probe(X_tr_co, y_tr_co, X_te_co, y_te_co)

    delta = auroc_bo - auroc_co
    verdict = ("BO_ADDS" if delta > 0.03 else
               ("BO_EQUIVALENT" if abs(delta) <= 0.03 else "CO_BETTER"))

    print(f"BO  AUROC={auroc_bo:.4f}  CI=[{ci_bo[0]:.3f},{ci_bo[1]:.3f}]  shuf={shuf_bo:.4f}", flush=True)
    print(f"CO  AUROC={auroc_co:.4f}  CI=[{ci_co[0]:.3f},{ci_co[1]:.3f}]  shuf={shuf_co:.4f}", flush=True)
    print(f"Delta (BO-CO)={delta:+.4f}  VERDICT: {verdict}", flush=True)

    return {
        "n_bo": n_bo, "n_co": n_co,
        "bo_auroc": auroc_bo, "bo_ci_95": list(ci_bo), "bo_shuffled": shuf_bo,
        "co_auroc": auroc_co, "co_ci_95": list(ci_co), "co_shuffled": shuf_co,
        "delta_bo_minus_co": delta,
        "verdict": verdict,
    }


# ── Main ──────────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    pool = load_pool(POOL_SIZE)
    model, tokenizer = load_model()

    param_hs, ctxdep_hs, incorrect_hs = collect_all(model, tokenizer, pool)
    comparison = compare_probes(param_hs, ctxdep_hs, incorrect_hs)

    results = {
        "experiment":   "EXP_0_BO_ABLATION_V1",
        "model":        MODEL_ID,
        "layer_idx":    LAYER_IDX,
        "n_target":     N_TARGET,
        "pool_size":    POOL_SIZE,
        "comparison":   comparison,
        "elapsed_min":  (time.time() - t0) / 60,
        "interpretation": {
            "BO_ADDS":        "Bilateral oracle two-pass design is essential — C003 upheld",
            "BO_EQUIVALENT":  "BO ≈ CO — withcontext check is redundant; scope contraction noted",
            "CO_BETTER":      "Surprising — correctness labels outperform BO; investigate label noise",
        }.get(comparison.get("verdict", ""), ""),
    }

    with open(SAVE_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults → {SAVE_PATH}", flush=True)
    print(f"Elapsed: {results['elapsed_min']:.1f} min", flush=True)


if __name__ == "__main__":
    main()
