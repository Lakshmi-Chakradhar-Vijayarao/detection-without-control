#!/usr/bin/env python3
"""
experiments/entropy_baseline/entropy_baseline.py

C008 Follow-up — Entropy Baseline Comparison
=============================================
QUESTION: Does Fisher+PCA64 add discriminative value beyond output entropy alone?

CONTEXT
-------
C008 was FALSIFIED: Spearman r(Fisher_decision, entropy) = -0.225 (Qwen) / -0.544 (Llama).
The signals are correlated. But r²=0.05-0.30 means Fisher is NOT identical to entropy.

This experiment resolves the remaining question: is the non-entropy component of Fisher
discriminative? If AUROC(Fisher) >> AUROC(entropy), Fisher carries signal beyond entropy.
If AUROC(Fisher) ≈ AUROC(entropy), Fisher is largely redundant.

PROTOCOL
--------
1. Bilateral oracle on TriviaQA (pool=5000, N_TARGET=150/class)
2. For each item at step-1, L26, simultaneously extract:
   - Hidden state vector (for Fisher+PCA64)
   - Output logits → compute Shannon entropy
3. Train/test split (TRAIN_FRAC=0.75 → test=37-38/class)
4. Fit Fisher+PCA64 probe on train HS, evaluate AUROC on test
5. Compute AUROC(entropy_alone) using -entropy as score (higher entropy → CTX_DEP)
6. Fit LR(entropy) on train entropy values, evaluate AUROC on test
7. Fit Fisher+entropy combined (LR on [Fisher_score, entropy]) on test
8. Report: AUROC_fisher, AUROC_entropy_raw, AUROC_entropy_lr, AUROC_combined

MODELS: Qwen2.5-1.5B-Instruct (Llama optional if HF_TOKEN available)
OUTPUT: entropy_baseline_results.json

REGISTRY: C008 follow-up (entropy-only baseline)
"""

from __future__ import annotations
import gc, json, os, random, sys, time
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# ── Config ────────────────────────────────────────────────────────────────────
SEED         = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}", flush=True)
if DEVICE == "cpu":
    raise RuntimeError("GPU required. Exiting.")

LAYER_IDX    = 26
PCA_DIM      = 64
N_TARGET     = 150
TRAIN_FRAC   = 0.75
POOL_SIZE    = 5000
MAX_GEN      = 2       # step-1 only
MAX_CTX      = 800
PARAM_MIN_F1 = 0.50
CTX_MAX_F1   = 0.05
CTX_MIN_CTX  = 0.50
OUTPUT_FILE  = "entropy_baseline_results.json"


def _get_hf_token():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("HF_TOKEN")
        if token:
            return token
    except Exception:
        pass
    return None


def _find_llama_path():
    candidates = [
        "/kaggle/input/llama-3.2/transformers/3b-instruct/1",
        "/kaggle/input/llama3.2/transformers/3b-instruct/1",
    ]
    try:
        for d in os.listdir("/kaggle/input"):
            if "llama" in d.lower():
                base = f"/kaggle/input/{d}"
                for sub in ["", "/transformers/3b-instruct/1", "/1"]:
                    full = base + sub
                    if os.path.exists(full) and os.path.isdir(full):
                        candidates.insert(0, full)
    except Exception:
        pass
    for c in candidates:
        if os.path.exists(c):
            return c
    hf_token = _get_hf_token()
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)
    return "meta-llama/Llama-3.2-3B-Instruct"


MODELS = [
    {"name": "qwen25_1.5b_instruct", "model_id": "Qwen/Qwen2.5-1.5B-Instruct", "n_layers": 28},
    {"name": "llama3.2_3b_instruct",  "model_id": _find_llama_path(),             "n_layers": 28},
]


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


def fmt_prompt(q, ctx=None):
    if ctx:
        return f"Context: {ctx}\n\nAnswer the following in one short phrase.\nQuestion: {q}\nAnswer:"
    return f"Answer the following in one short phrase.\nQuestion: {q}\nAnswer:"


def generate_text(model, tok, prompt, max_new=60):
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


# ── HS + entropy extraction (single pass) ────────────────────────────────────
def extract_hs_and_entropy(model, tok, q, layer_idx):
    """Extract step-1 hidden state at layer_idx AND output entropy simultaneously."""
    prompt  = fmt_prompt(q)
    inp     = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    captured = {}

    def hook_fn(module, inp_t, out):
        hs = out[0] if isinstance(out, tuple) else out
        if hs.shape[1] != 1:
            return
        if "hs" in captured:
            return
        captured["hs"] = hs[0, 0, :].detach().float().cpu().numpy()

    handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            out = model.generate(
                **inp, max_new_tokens=MAX_GEN, do_sample=False,
                pad_token_id=tok.eos_token_id, use_cache=True,
                output_scores=True, return_dict_in_generate=True,
            )
    finally:
        handle.remove()

    hs_vec = captured.get("hs")

    # output_scores[0] = logits at step-1 token
    logits  = out.scores[0][0].float()
    probs   = torch.softmax(logits, dim=-1)
    probs   = torch.clamp(probs, min=1e-10)
    entropy = float(-torch.sum(probs * torch.log(probs)).item())

    return hs_vec, entropy


# ── Probes ────────────────────────────────────────────────────────────────────
def fisher_auroc(hs_p_tr, hs_c_tr, hs_p_te, hs_c_te):
    """Fisher+PCA64 on HS train → eval on test."""
    X_tr = np.vstack([hs_p_tr, hs_c_tr]).astype(np.float32)
    y_tr = np.array([1]*len(hs_p_tr) + [0]*len(hs_c_tr))
    X_te = np.vstack([hs_p_te, hs_c_te]).astype(np.float32)
    y_te = np.array([1]*len(hs_p_te) + [0]*len(hs_c_te))

    n_comp = min(PCA_DIM, X_tr.shape[1], X_tr.shape[0] - 1)
    pca    = PCA(n_components=n_comp, random_state=SEED)
    X_tr_r = pca.fit_transform(X_tr)
    X_te_r = pca.transform(X_te)

    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    lda.fit(X_tr_r, y_tr)
    scores = lda.decision_function(X_te_r)
    auroc  = float(roc_auc_score(y_te, scores))
    y_shuf = y_te.copy(); np.random.shuffle(y_shuf)
    shuf   = float(roc_auc_score(y_shuf, scores))
    return round(auroc, 4), round(shuf, 4), lda.decision_function(X_te_r)


def entropy_raw_auroc(ent_p_te, ent_c_te):
    """-entropy as PARAM score (lower entropy = more PARAM)."""
    y_te   = np.array([1]*len(ent_p_te) + [0]*len(ent_c_te))
    scores = np.concatenate([-np.array(ent_p_te), -np.array(ent_c_te)])
    return round(float(roc_auc_score(y_te, scores)), 4)


def entropy_lr_auroc(ent_p_tr, ent_c_tr, ent_p_te, ent_c_te):
    """LR(entropy) trained on train split, evaluated on test split."""
    X_tr = np.array(ent_p_tr + ent_c_tr).reshape(-1, 1)
    y_tr = np.array([1]*len(ent_p_tr) + [0]*len(ent_c_tr))
    X_te = np.array(ent_p_te + ent_c_te).reshape(-1, 1)
    y_te = np.array([1]*len(ent_p_te) + [0]*len(ent_c_te))
    lr   = LogisticRegression(max_iter=200)
    lr.fit(X_tr, y_tr)
    scores = lr.predict_proba(X_te)[:, 1]
    return round(float(roc_auc_score(y_te, scores)), 4)


def combined_auroc(ent_p_te, ent_c_te, hs_p_tr, hs_c_tr, ent_p_tr, ent_c_tr,
                   hs_p_te, hs_c_te):
    """LR([Fisher_score, -entropy]) trained on train combined features."""
    # Re-fit Fisher on train to get train-set decision scores
    X_tr_hs = np.vstack([hs_p_tr, hs_c_tr]).astype(np.float32)
    y_tr    = np.array([1]*len(hs_p_tr) + [0]*len(hs_c_tr))
    n_comp  = min(PCA_DIM, X_tr_hs.shape[1], X_tr_hs.shape[0] - 1)
    pca     = PCA(n_components=n_comp, random_state=SEED)
    X_tr_r  = pca.fit_transform(X_tr_hs)
    X_te_r  = pca.transform(np.vstack([hs_p_te, hs_c_te]).astype(np.float32))
    lda     = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    lda.fit(X_tr_r, y_tr)

    f_tr   = lda.decision_function(X_tr_r).reshape(-1, 1)
    e_tr   = (-np.array(ent_p_tr + ent_c_tr)).reshape(-1, 1)
    f_te   = lda.decision_function(X_te_r).reshape(-1, 1)
    e_te   = (-np.array(ent_p_te + ent_c_te)).reshape(-1, 1)

    X_tr_c = np.hstack([f_tr, e_tr])
    X_te_c = np.hstack([f_te, e_te])
    y_te   = np.array([1]*len(hs_p_te) + [0]*len(hs_c_te))

    lr = LogisticRegression(max_iter=200)
    lr.fit(X_tr_c, y_tr)
    scores = lr.predict_proba(X_te_c)[:, 1]
    return round(float(roc_auc_score(y_te, scores)), 4)


# ── Per-model run ─────────────────────────────────────────────────────────────
def run_model(model_cfg, all_items):
    t0   = time.time()
    name = model_cfg["name"]
    print(f"\n{'='*60}", flush=True)
    print(f"Model: {name}", flush=True)
    print(f"{'='*60}", flush=True)

    tok = AutoTokenizer.from_pretrained(model_cfg["model_id"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["model_id"], torch_dtype=torch.float16,
        device_map=None, low_cpu_mem_usage=True,
    ).to(DEVICE).eval()

    # ── Oracle labeling ───────────────────────────────────────────────────────
    print(f"\nBilateral oracle (pool={POOL_SIZE})...", flush=True)
    param_items, ctxdep_items = [], []
    for i, item in enumerate(all_items[:POOL_SIZE]):
        if len(param_items) >= N_TARGET and len(ctxdep_items) >= N_TARGET:
            break
        if i % 200 == 0:
            print(f"  [{i}/{POOL_SIZE}] P={len(param_items)} C={len(ctxdep_items)}  "
                  f"({round(time.time()-t0)}s)", flush=True)
        lbl = label_item(model, tok, item)
        if lbl == "PARAM" and len(param_items) < N_TARGET:
            param_items.append(item)
        elif lbl == "CTX_DEP" and len(ctxdep_items) < N_TARGET:
            ctxdep_items.append(item)

    n_use = min(len(param_items), len(ctxdep_items))
    print(f"  Final: P={len(param_items)}  C={len(ctxdep_items)}  → using {n_use}/class", flush=True)

    # ── HS + entropy extraction ───────────────────────────────────────────────
    print(f"\nExtracting HS + entropy at layer {LAYER_IDX}...", flush=True)
    hs_param, hs_ctxdep   = [], []
    ent_param, ent_ctxdep = [], []

    for idx, item in enumerate(param_items[:n_use]):
        if idx % 50 == 0:
            print(f"  PARAM [{idx}/{n_use}]", flush=True)
        hs, ent = extract_hs_and_entropy(model, tok, item["question"], LAYER_IDX)
        if hs is not None:
            hs_param.append(hs)
            ent_param.append(ent)

    for idx, item in enumerate(ctxdep_items[:n_use]):
        if idx % 50 == 0:
            print(f"  CTX_DEP [{idx}/{n_use}]", flush=True)
        hs, ent = extract_hs_and_entropy(model, tok, item["question"], LAYER_IDX)
        if hs is not None:
            hs_ctxdep.append(hs)
            ent_ctxdep.append(ent)

    n_final = min(len(hs_param), len(hs_ctxdep))
    print(f"  Extracted: P={len(hs_param)}  C={len(hs_ctxdep)}", flush=True)

    # ── Train/test split ──────────────────────────────────────────────────────
    n_train = int(n_final * TRAIN_FRAC)
    n_test  = n_final - n_train

    hs_p_tr,  hs_p_te  = hs_param[:n_train],   hs_param[n_train:n_final]
    hs_c_tr,  hs_c_te  = hs_ctxdep[:n_train],  hs_ctxdep[n_train:n_final]
    ent_p_tr, ent_p_te = ent_param[:n_train],   ent_param[n_train:n_final]
    ent_c_tr, ent_c_te = ent_ctxdep[:n_train],  ent_ctxdep[n_train:n_final]

    print(f"  Split: train={n_train}/class  test={n_test}/class", flush=True)

    # ── Evaluate probes ───────────────────────────────────────────────────────
    print(f"\nEvaluating probes...", flush=True)

    f_auroc, f_shuf, _ = fisher_auroc(hs_p_tr, hs_c_tr, hs_p_te, hs_c_te)
    e_raw              = entropy_raw_auroc(ent_p_te, ent_c_te)
    e_lr               = entropy_lr_auroc(ent_p_tr, ent_c_tr, ent_p_te, ent_c_te)
    comb               = combined_auroc(ent_p_te, ent_c_te,
                                        hs_p_tr, hs_c_tr, ent_p_tr, ent_c_tr,
                                        hs_p_te, hs_c_te)

    mean_ent_p = round(float(np.mean(ent_param[:n_final])), 4)
    mean_ent_c = round(float(np.mean(ent_ctxdep[:n_final])), 4)

    fisher_adds_value = f_auroc > e_lr + 0.03
    verdict = "FISHER_ADDS_VALUE" if fisher_adds_value else (
        "FISHER_REDUNDANT" if f_auroc < e_lr + 0.03 else "MARGINAL"
    )

    print(f"\n  Fisher AUROC:       {f_auroc}  (shuffled={f_shuf})", flush=True)
    print(f"  Entropy-raw AUROC:  {e_raw}   (using -entropy as score directly)", flush=True)
    print(f"  Entropy-LR AUROC:   {e_lr}   (LR trained on entropy)", flush=True)
    print(f"  Combined AUROC:     {comb}   (LR on [Fisher, entropy])", flush=True)
    print(f"  Mean entropy PARAM: {mean_ent_p} nats", flush=True)
    print(f"  Mean entropy CTX:   {mean_ent_c} nats", flush=True)
    print(f"  Verdict: {verdict}", flush=True)
    print(f"  (Fisher > Entropy-LR by {round(f_auroc - e_lr, 4)})", flush=True)

    del model; gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()

    return {
        "name": name,
        "n_use": n_use, "n_train": n_train, "n_test": n_test,
        "fisher_auroc": f_auroc, "fisher_shuffled": f_shuf,
        "entropy_raw_auroc": e_raw,
        "entropy_lr_auroc": e_lr,
        "combined_auroc": comb,
        "mean_entropy_param": mean_ent_p,
        "mean_entropy_ctxdep": mean_ent_c,
        "fisher_minus_entropy_lr": round(f_auroc - e_lr, 4),
        "verdict": verdict,
        "elapsed_s": round(time.time() - t0, 1),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def run():
    t0 = time.time()
    print(f"\n{'='*60}", flush=True)
    print("Entropy Baseline Comparison — C008 follow-up", flush=True)
    print(f"N_TARGET={N_TARGET}/class  POOL={POOL_SIZE}  L={LAYER_IDX}", flush=True)
    print(f"{'='*60}", flush=True)

    all_items = load_trivia()
    all_results = []

    for model_cfg in MODELS:
        try:
            result = run_model(model_cfg, all_items)
            all_results.append(result)
        except Exception as e:
            print(f"  FAILED {model_cfg['name']}: {e}", flush=True)
            all_results.append({"name": model_cfg["name"], "status": f"FAILED: {str(e)[:100]}"})

        # intermediate save
        with open(OUTPUT_FILE, "w") as f:
            json.dump({"results": all_results, "elapsed_s": round(time.time()-t0, 1)}, f, indent=2)
        print(f"[checkpoint] saved after {model_cfg['name']}", flush=True)

    print(f"\n{'='*60}", flush=True)
    print("ENTROPY BASELINE RESULTS", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"{'Model':<28} {'Fisher':<10} {'Ent-raw':<10} {'Ent-LR':<10} {'Combined':<10} {'Verdict'}", flush=True)
    print(f"{'-'*28} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*20}", flush=True)
    for r in all_results:
        if "fisher_auroc" in r:
            print(f"  {r['name']:<26} {r['fisher_auroc']:<10} {r['entropy_raw_auroc']:<10} "
                  f"{r['entropy_lr_auroc']:<10} {r['combined_auroc']:<10} {r['verdict']}", flush=True)

    with open(OUTPUT_FILE, "w") as f:
        json.dump({"results": all_results, "elapsed_s": round(time.time()-t0, 1)}, f, indent=2)
    print(f"\nResults saved to {OUTPUT_FILE}", flush=True)
    print(f"Total elapsed: {round(time.time()-t0, 1)}s", flush=True)


if __name__ == "__main__":
    run()
