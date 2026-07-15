#!/usr/bin/env python3
"""
experiments/structural_independence_v2/structural_independence_v2.py

C008 Fix — Structural Independence (correct metric)
====================================================
CLAIM UNDER TEST: C008 (EXPLORATORY)
  corr(output_entropy, Fisher_decision_score) ≈ 0

CONTEXT
-------
Prior result: corr(entropy, j_score) = 0.0039 — but j_score was the MSCP metric,
NOT the bilateral oracle Fisher+PCA64 decision score. This experiment re-runs the
correlation with the correct metric.

WHAT WE MEASURE
---------------
- output_entropy: Shannon entropy of the softmax distribution at step-1, over
  the full vocabulary. High = uncertain logits; Low = confident logits.
- fisher_score: LDA decision function value from Fisher+PCA64 at L26 step-1.
  Positive = PARAM direction; Negative = CTX_DEP direction.

Independence claim: epistemic geometry (what kind of knowledge the model uses)
is orthogonal to output confidence (how certain the logits look). If r ≈ 0,
they're measuring different things.

PROTOCOL
--------
1. Bilateral oracle on TriviaQA (pool=5000, N_TARGET=150/class per model)
2. Extract step-1 HS + logits at L26 simultaneously
3. Fit Fisher+PCA64 on train split (0.75), evaluate on test split (0.25)
4. For each test item: get fisher_score + output_entropy
5. Compute Spearman + Pearson r(fisher_score, entropy)

MODELS: Qwen2.5-1.5B-Instruct, Llama-3.2-3B-Instruct
OUTPUT: structural_independence_v2_results.json

REGISTRY: C008 re-run (measurement identity fix)
"""

from __future__ import annotations
import gc, json, os, random, sys, time
import numpy as np
import torch
from scipy.stats import spearmanr, pearsonr
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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
MAX_GEN      = 60
MAX_CTX      = 800
PARAM_MIN_F1 = 0.50
CTX_MAX_F1   = 0.05
CTX_MIN_CTX  = 0.50
OUTPUT_FILE  = "structural_independence_v2_results.json"

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
        "/kaggle/input/llama-3.2-3b-instruct/transformers/default/1",
    ]
    try:
        for d in os.listdir("/kaggle/input"):
            if "llama" in d.lower():
                base = f"/kaggle/input/{d}"
                for sub in ["", "/transformers/3b-instruct/1", "/transformers/3b-instruct", "/1"]:
                    full = base + sub
                    if os.path.exists(full) and os.path.isdir(full):
                        candidates.insert(0, full)
    except Exception:
        pass
    for c in candidates:
        if os.path.exists(c):
            print(f"  Llama path: {c}", flush=True)
            return c
    hf_token = _get_hf_token()
    if hf_token:
        print(f"  Llama: logging in to HuggingFace via token", flush=True)
        from huggingface_hub import login
        login(token=hf_token)
    else:
        print("  WARNING: no HF_TOKEN found — Llama load will likely fail", flush=True)
    return "meta-llama/Llama-3.2-3B-Instruct"

MODELS = [
    # Qwen completed (v1): r=-0.2251, CORRELATED. Skip to save GPU time.
    # {"name": "qwen25_1.5b_instruct", "model_id": "Qwen/Qwen2.5-1.5B-Instruct", "n_layers": 28},
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


# ── Text helpers ──────────────────────────────────────────────────────────────
def fmt_prompt(q, ctx=None):
    if ctx:
        return f"Context: {ctx}\n\nAnswer the following in one short phrase.\nQuestion: {q}\nAnswer:"
    return f"Answer the following in one short phrase.\nQuestion: {q}\nAnswer:"

def generate_text(model, tok, prompt):
    inp = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=MAX_GEN, do_sample=False,
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


# ── Oracle labeling ───────────────────────────────────────────────────────────
def label_item(model, tok, item):
    q, ctx, ans = item["question"], item["context"], item["answers"]
    pred_no = generate_text(model, tok, fmt_prompt(q))
    f1_no   = token_f1(pred_no, ans)
    if f1_no >= PARAM_MIN_F1:
        return "PARAM", f1_no, None
    if ctx and f1_no <= CTX_MAX_F1:
        pred_ctx = generate_text(model, tok, fmt_prompt(q, ctx))
        f1_ctx   = token_f1(pred_ctx, ans)
        if f1_ctx >= CTX_MIN_CTX:
            return "CTX_DEP", f1_no, f1_ctx
    return "SKIP", f1_no, None


# ── HS + entropy extraction (single forward pass) ─────────────────────────────
def extract_hs_and_entropy(model, tok, q, layer_idx):
    """
    Returns (hidden_state_vec, output_entropy) at generation step 1.
    Both extracted in a single forward pass to avoid running the model twice.
    """
    prompt = fmt_prompt(q)
    inp    = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    captured = {}

    def hook_fn(module, inp_t, out):
        hs = out[0] if isinstance(out, tuple) else out
        if hs.shape[1] != 1:
            return
        if "hs" in captured:
            return
        captured["hs"] = hs[0, 0, :].detach().float().cpu()

    handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            out = model.generate(
                **inp,
                max_new_tokens=2,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
                use_cache=True,
                output_scores=True,
                return_dict_in_generate=True,
            )
    finally:
        handle.remove()

    hs_vec = captured.get("hs")
    entropy = None

    if out.scores and len(out.scores) >= 1:
        # scores[0] = logits at step 1, shape [batch, vocab_size]
        logits = out.scores[0][0].float()
        probs  = torch.softmax(logits, dim=-1)
        # clip tiny probs to avoid log(0)
        probs  = torch.clamp(probs, min=1e-10)
        entropy = float(-torch.sum(probs * torch.log(probs)).item())

    return (hs_vec.numpy() if hs_vec is not None else None), entropy


# ── Probe fitting ─────────────────────────────────────────────────────────────
def fit_fisher_pca64(hs_param, hs_ctxdep):
    X = np.vstack([hs_param, hs_ctxdep]).astype(np.float32)
    y = np.array([1]*len(hs_param) + [0]*len(hs_ctxdep))
    pca = PCA(n_components=min(PCA_DIM, X.shape[1], X.shape[0]-1), random_state=SEED)
    X_r = pca.fit_transform(X)
    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    lda.fit(X_r, y)
    return pca, lda


# ── Per-Model Run ─────────────────────────────────────────────────────────────
def run_model(model_cfg, all_items):
    print(f"\n{'='*60}", flush=True)
    print(f"Model: {model_cfg['name']}", flush=True)
    print(f"{'='*60}", flush=True)

    result = {"name": model_cfg["name"], "model_id": model_cfg["model_id"]}

    print(f"Loading {model_cfg['model_id']}...", flush=True)
    try:
        tok = AutoTokenizer.from_pretrained(model_cfg["model_id"])
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(
            model_cfg["model_id"], torch_dtype=torch.float16,
            device_map=None, low_cpu_mem_usage=True,
        ).to(DEVICE).eval()
    except Exception as e:
        result["status"] = f"FAILED_LOAD: {e}"
        return result

    layer_idx = model_cfg.get("n_layers", 28) - 2

    # Oracle labeling
    print(f"\nBilateral oracle labeling (pool={POOL_SIZE})...", flush=True)
    param_items, ctxdep_items = [], []
    t0 = time.time()

    for i, item in enumerate(all_items[:POOL_SIZE]):
        if len(param_items) >= N_TARGET and len(ctxdep_items) >= N_TARGET:
            break
        if i % 200 == 0:
            print(f"  [{i}/{POOL_SIZE}] P={len(param_items)} C={len(ctxdep_items)}  ({round(time.time()-t0)}s)", flush=True)
        lbl, _, _ = label_item(model, tok, item)
        if lbl == "PARAM" and len(param_items) < N_TARGET:
            param_items.append(item)
        elif lbl == "CTX_DEP" and len(ctxdep_items) < N_TARGET:
            ctxdep_items.append(item)

    n_per_class = min(len(param_items), len(ctxdep_items))
    print(f"  Final: P={len(param_items)}  C={len(ctxdep_items)}  (using {n_per_class}/class)", flush=True)

    result["n_param"]    = len(param_items)
    result["n_ctxdep"]   = len(ctxdep_items)
    result["n_per_class"] = n_per_class

    if n_per_class < 40:
        result["status"] = "INSUFFICIENT_ITEMS"
        del model; gc.collect()
        return result

    # HS + entropy extraction
    print(f"\nExtracting HS + entropy at layer {layer_idx}...", flush=True)
    hs_param, hs_ctxdep = [], []
    ent_param, ent_ctxdep = [], []

    for idx, item in enumerate(param_items[:n_per_class]):
        if idx % 50 == 0:
            print(f"  PARAM [{idx}/{n_per_class}]", flush=True)
        hs, ent = extract_hs_and_entropy(model, tok, item["question"], layer_idx)
        if hs is not None and ent is not None:
            hs_param.append(hs)
            ent_param.append(ent)

    for idx, item in enumerate(ctxdep_items[:n_per_class]):
        if idx % 50 == 0:
            print(f"  CTX_DEP [{idx}/{n_per_class}]", flush=True)
        hs, ent = extract_hs_and_entropy(model, tok, item["question"], layer_idx)
        if hs is not None and ent is not None:
            hs_ctxdep.append(hs)
            ent_ctxdep.append(ent)

    n_use   = min(len(hs_param), len(hs_ctxdep))
    n_train = int(n_use * TRAIN_FRAC)
    n_test  = n_use - n_train
    print(f"  Extracted: P={len(hs_param)}  C={len(hs_ctxdep)}  → train={n_train}/class  test={n_test}/class", flush=True)

    if n_test < 10:
        result["status"] = "INSUFFICIENT_TEST_ITEMS"
        del model; gc.collect()
        return result

    # Probe fitting
    hp_train = np.array(hs_param[:n_train],       dtype=np.float32)
    hc_train = np.array(hs_ctxdep[:n_train],      dtype=np.float32)
    hp_test  = np.array(hs_param[n_train:n_use],  dtype=np.float32)
    hc_test  = np.array(hs_ctxdep[n_train:n_use], dtype=np.float32)

    pca, lda = fit_fisher_pca64(hp_train, hc_train)

    X_test = np.vstack([hp_test, hc_test])
    y_test = np.array([1]*len(hp_test) + [0]*len(hc_test))

    # Fisher AUROC
    X_test_r    = pca.transform(X_test)
    fisher_scores = lda.decision_function(X_test_r)
    fisher_auroc  = float(roc_auc_score(y_test, fisher_scores))

    # Entropy for test items
    ent_test = np.array(ent_param[n_train:n_use] + ent_ctxdep[n_train:n_use], dtype=np.float32)

    # Correlations
    sp_r, sp_p = spearmanr(fisher_scores, ent_test)
    pe_r, pe_p = pearsonr(fisher_scores,  ent_test)

    # PARAM vs CTX_DEP entropy comparison
    ent_p_test = np.array(ent_param[n_train:n_use])
    ent_c_test = np.array(ent_ctxdep[n_train:n_use])
    mean_ent_param  = float(np.mean(ent_p_test))
    mean_ent_ctxdep = float(np.mean(ent_c_test))

    print(f"\n  Fisher AUROC:          {fisher_auroc:.4f}")
    print(f"  Spearman r(Fisher,Ent): {sp_r:.4f}  p={sp_p:.4f}")
    print(f"  Pearson  r(Fisher,Ent): {pe_r:.4f}  p={pe_p:.4f}")
    print(f"  Mean entropy PARAM:    {mean_ent_param:.4f}")
    print(f"  Mean entropy CTX_DEP:  {mean_ent_ctxdep:.4f}")

    verdict = "INDEPENDENT" if abs(sp_r) < 0.05 else ("WEAK_CORR" if abs(sp_r) < 0.15 else "CORRELATED")
    print(f"  Verdict: {verdict}", flush=True)

    result.update({
        "status": "COMPLETE",
        "layer_idx": layer_idx,
        "n_train_per_class": n_train,
        "n_test_per_class":  n_test,
        "fisher_auroc": round(fisher_auroc, 4),
        "correlation": {
            "spearman_r": round(float(sp_r), 4),
            "spearman_p": round(float(sp_p), 4),
            "pearson_r":  round(float(pe_r), 4),
            "pearson_p":  round(float(pe_p), 4),
            "verdict":    verdict,
        },
        "entropy_by_class": {
            "mean_param":   round(mean_ent_param, 4),
            "mean_ctxdep":  round(mean_ent_ctxdep, 4),
            "delta":        round(mean_ent_param - mean_ent_ctxdep, 4),
        },
    })

    del model; gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return result


# ── Main ──────────────────────────────────────────────────────────────────────
def run():
    t0 = time.time()
    results = {
        "experiment": "structural_independence_v2",
        "claim": "C008",
        "note": "C008 re-run with correct metric: Fisher+PCA64 decision score (not MSCP j_score)",
        "config": {
            "n_target": N_TARGET,
            "pool_size": POOL_SIZE,
            "layer_idx": LAYER_IDX,
            "pca_dim": PCA_DIM,
            "train_frac": TRAIN_FRAC,
        },
        "models": {},
        "elapsed_s": 0,
    }

    print("Loading dataset...", flush=True)
    all_items = load_trivia()

    for model_cfg in MODELS:
        res = run_model(model_cfg, all_items)
        results["models"][model_cfg["name"]] = res
        with open(OUTPUT_FILE, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[checkpoint] saved after {model_cfg['name']}", flush=True)

    # Summary
    print(f"\n{'='*60}", flush=True)
    print("STRUCTURAL INDEPENDENCE RESULTS", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"{'Model':<25} {'Fisher AUROC':<14} {'Spearman r':<12} {'p':<8} {'Verdict'}", flush=True)
    print(f"{'-'*25} {'-'*14} {'-'*12} {'-'*8} {'-'*12}", flush=True)

    for mname, mres in results["models"].items():
        if mres.get("status") != "COMPLETE":
            print(f"{mname:<25} FAILED: {mres.get('status','?')}", flush=True)
            continue
        corr = mres["correlation"]
        print(f"{mname:<25} {mres['fisher_auroc']:<14.4f} {corr['spearman_r']:<12.4f} "
              f"{corr['spearman_p']:<8.4f} {corr['verdict']}", flush=True)

    results["elapsed_s"] = round(time.time() - t0, 1)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_FILE}", flush=True)
    print(f"Total elapsed: {results['elapsed_s']}s", flush=True)


if __name__ == "__main__":
    run()
