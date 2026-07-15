#!/usr/bin/env python3
"""
experiments/large_n_validation/large_n_validation.py

Task 1.3 — Large-N Validation (N=200/class)
============================================
CLAIMS UNDER TEST: C001, C002, C003 (currently CONFIRMED at N=128-150)
  Replicate bilateral oracle AUROC measurement at N≥200/class with
  bootstrap 95% CIs tight enough to make table 1 of Paper 1.

EXPERIMENT DESIGN
-----------------
For each model (Qwen2.5-1.5B-Instruct, Llama-3.2-3B-Instruct):
  - Run bilateral oracle labeling on TriviaQA (pool=5000)
  - Target N=200/class
  - Extract step-1 hidden states at layer LAYER_IDX
  - Fit Fisher+PCA64 probe (N_train = 0.75 × N per class)
  - Evaluate on held-out test set
  - Bootstrap 1000 samples for 95% CI
  - Run shuffled control (label permutation)

Output: clean benchmark table for Paper 1, Section 3.

REGISTRY: EXP_T1A_LARGE_N (PENDING → COMPLETE when run)
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
print(f"Device: {DEVICE}")
if DEVICE == "cpu":
    raise RuntimeError("GPU required. Exiting.")

LAYER_IDX    = 26
PCA_DIM      = 64
N_TARGET     = 200
TRAIN_FRAC   = 0.75
POOL_SIZE    = 10000  # v2: increased from 5000; CTX_DEP yield ~2.4% → need ~8333 items
N_BOOTSTRAP  = 1000
MAX_GEN      = 60
MAX_CTX      = 800
PARAM_MIN_F1 = 0.50
CTX_MAX_F1   = 0.05
CTX_MIN_CTX  = 0.50
OUTPUT_FILE  = "large_n_validation_results.json"

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
    # Qwen completed (v3): AUROC=0.7312, CI=[0.6256,0.8283], N=197/class — skip to save GPU time
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

def generate_text(model, tok, prompt, max_new=MAX_GEN):
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


# ── Oracle + HS extraction ────────────────────────────────────────────────────
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

def extract_step1_hs(model, tok, q, layer_idx):
    prompt = fmt_prompt(q)
    inp    = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
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
            model.generate(**inp, max_new_tokens=2, do_sample=False,
                           pad_token_id=tok.eos_token_id, use_cache=True)
    finally:
        handle.remove()

    return captured.get("hs")


# ── Probe + Bootstrap ─────────────────────────────────────────────────────────
def fit_fisher_pca64(hs_param, hs_ctxdep, pca_dim=PCA_DIM):
    X = np.vstack([hs_param, hs_ctxdep]).astype(np.float32)
    y = np.array([1]*len(hs_param) + [0]*len(hs_ctxdep))
    pca = PCA(n_components=min(pca_dim, X.shape[1], X.shape[0]-1), random_state=SEED)
    X_r = pca.fit_transform(X)
    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    lda.fit(X_r, y)
    return pca, lda

def fit_logreg(hs_param, hs_ctxdep):
    X = np.vstack([hs_param, hs_ctxdep]).astype(np.float32)
    y = np.array([1]*len(hs_param) + [0]*len(hs_ctxdep))
    lr = LogisticRegression(C=1.0, max_iter=500, random_state=SEED)
    lr.fit(X, y)
    return lr

def eval_auroc(model_probe, X_test, y_test, probe_type="fisher"):
    if probe_type == "fisher":
        pca, lda = model_probe
        X_r = pca.transform(X_test.astype(np.float32))
        scores = lda.decision_function(X_r)
    else:
        scores = model_probe.decision_function(X_test.astype(np.float32))
    return float(roc_auc_score(y_test, scores))

def bootstrap_auroc(model_probe, X_test, y_test, n_boot=N_BOOTSTRAP, probe_type="fisher"):
    n = len(y_test)
    rng = np.random.RandomState(SEED)
    boot_aurocs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        X_b, y_b = X_test[idx], y_test[idx]
        if len(np.unique(y_b)) < 2:
            continue
        try:
            boot_aurocs.append(eval_auroc(model_probe, X_b, y_b, probe_type))
        except Exception:
            continue
    ci_lo = float(np.percentile(boot_aurocs, 2.5))
    ci_hi = float(np.percentile(boot_aurocs, 97.5))
    return ci_lo, ci_hi


# ── Per-Model Run ─────────────────────────────────────────────────────────────
def run_model(model_cfg, all_items):
    print(f"\n{'='*60}")
    print(f"Model: {model_cfg['name']}")
    print(f"{'='*60}")

    model_result = {"name": model_cfg["name"], "model_id": model_cfg["model_id"]}

    # Load model
    print(f"Loading {model_cfg['model_id']}...")
    try:
        tok = AutoTokenizer.from_pretrained(model_cfg["model_id"])
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(
            model_cfg["model_id"], torch_dtype=torch.float16,
            device_map=None, low_cpu_mem_usage=True,
        ).to(DEVICE).eval()
    except Exception as e:
        model_result["status"] = f"FAILED_LOAD: {e}"
        return model_result

    layer_idx = model_cfg.get("n_layers", 28) - 2  # penultimate layer

    # Oracle labeling
    print(f"\nBilateral oracle labeling (pool={POOL_SIZE})...")
    param_items, ctxdep_items = [], []
    t_label = time.time()

    for i, item in enumerate(all_items[:POOL_SIZE]):
        if len(param_items) >= N_TARGET and len(ctxdep_items) >= N_TARGET:
            break
        if i % 200 == 0:
            elapsed = round(time.time() - t_label)
            print(f"  [{i}/{POOL_SIZE}] P={len(param_items)} C={len(ctxdep_items)}  ({elapsed}s)", flush=True)
        lbl, _, _ = label_item(model, tok, item)
        if lbl == "PARAM" and len(param_items) < N_TARGET:
            param_items.append(item)
        elif lbl == "CTX_DEP" and len(ctxdep_items) < N_TARGET:
            ctxdep_items.append(item)

    n_param = len(param_items)
    n_ctxdep = len(ctxdep_items)
    n_per_class = min(n_param, n_ctxdep)
    print(f"  Final: P={n_param}  C={n_ctxdep}  (using {n_per_class} per class)")

    model_result["n_param"]    = n_param
    model_result["n_ctxdep"]   = n_ctxdep
    model_result["n_per_class"] = n_per_class

    if n_per_class < 40:
        model_result["status"] = "INSUFFICIENT_ITEMS"
        return model_result

    # HS extraction
    print(f"\nExtracting step-1 HS at layer {layer_idx}...")
    hs_param, hs_ctxdep = [], []

    for idx, item in enumerate(param_items[:n_per_class]):
        if idx % 50 == 0:
            print(f"  PARAM [{idx}/{n_per_class}]", flush=True)
        vec = extract_step1_hs(model, tok, item["question"], layer_idx)
        if vec is not None:
            hs_param.append(vec)

    for idx, item in enumerate(ctxdep_items[:n_per_class]):
        if idx % 50 == 0:
            print(f"  CTX_DEP [{idx}/{n_per_class}]", flush=True)
        vec = extract_step1_hs(model, tok, item["question"], layer_idx)
        if vec is not None:
            hs_ctxdep.append(vec)

    print(f"  Extracted: P={len(hs_param)}  C={len(hs_ctxdep)}")

    # Train/test split
    n_use    = min(len(hs_param), len(hs_ctxdep))
    n_train  = int(n_use * TRAIN_FRAC)
    n_test   = n_use - n_train

    hp_train = np.array(hs_param[:n_train],  dtype=np.float32)
    hc_train = np.array(hs_ctxdep[:n_train], dtype=np.float32)
    hp_test  = np.array(hs_param[n_train:n_use],  dtype=np.float32)
    hc_test  = np.array(hs_ctxdep[n_train:n_use], dtype=np.float32)

    X_test = np.vstack([hp_test, hc_test])
    y_test = np.array([1]*len(hp_test) + [0]*len(hc_test))

    print(f"\nProbe evaluation: train={n_train}/class  test={n_test}/class")

    # Fit Fisher+PCA64
    pca, lda = fit_fisher_pca64(hp_train, hc_train)
    fp_auroc = eval_auroc((pca, lda), X_test, y_test, "fisher")
    fp_ci_lo, fp_ci_hi = bootstrap_auroc((pca, lda), X_test, y_test, N_BOOTSTRAP, "fisher")

    # Shuffled control (Fisher)
    y_shuf = y_test.copy(); np.random.shuffle(y_shuf)
    fp_shuf_auroc = eval_auroc((pca, lda), X_test, y_shuf, "fisher")

    # Fit LR L2 (linear baseline for comparison)
    lr = fit_logreg(hp_train, hc_train)
    lr_auroc = eval_auroc(lr, X_test, y_test, "logreg")
    lr_ci_lo, lr_ci_hi = bootstrap_auroc(lr, X_test, y_test, N_BOOTSTRAP, "logreg")

    print(f"\n  Fisher+PCA64: AUROC={fp_auroc:.4f}  CI=[{fp_ci_lo:.4f},{fp_ci_hi:.4f}]  "
          f"shuffled={fp_shuf_auroc:.4f}")
    print(f"  LR L2:        AUROC={lr_auroc:.4f}  CI=[{lr_ci_lo:.4f},{lr_ci_hi:.4f}]")

    shuf_status = "CLEAN" if fp_shuf_auroc < 0.60 else "WARN" if fp_shuf_auroc < 0.70 else "FAIL"
    print(f"  Shuffled status: {shuf_status}")

    model_result.update({
        "status": "COMPLETE",
        "layer_idx": layer_idx,
        "n_train_per_class": n_train,
        "n_test_per_class": n_test,
        "fisher_pca64": {
            "auroc": round(fp_auroc, 4),
            "ci_lo": round(fp_ci_lo, 4),
            "ci_hi": round(fp_ci_hi, 4),
            "shuffled_auroc": round(fp_shuf_auroc, 4),
            "shuffled_status": shuf_status,
        },
        "logistic_l2": {
            "auroc": round(lr_auroc, 4),
            "ci_lo": round(lr_ci_lo, 4),
            "ci_hi": round(lr_ci_hi, 4),
        },
    })

    del model; gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return model_result


# ── Main ──────────────────────────────────────────────────────────────────────
def run():
    t0 = time.time()
    results = {
        "config": {
            "n_target": N_TARGET,
            "pool_size": POOL_SIZE,
            "layer_idx": LAYER_IDX,
            "pca_dim": PCA_DIM,
            "train_frac": TRAIN_FRAC,
            "n_bootstrap": N_BOOTSTRAP,
            "oracle": {
                "param_min_f1": PARAM_MIN_F1,
                "ctxdep_max_f1": CTX_MAX_F1,
                "ctxdep_min_ctx_f1": CTX_MIN_CTX,
            },
        },
        "models": {},
        "elapsed_s": 0,
    }

    print("Loading dataset...")
    all_items = load_trivia()

    for model_cfg in MODELS:
        res = run_model(model_cfg, all_items)
        results["models"][model_cfg["name"]] = res
        # intermediate save — preserves completed model if second model times out
        with open(OUTPUT_FILE, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[checkpoint] saved after {model_cfg['name']}", flush=True)

    # ── Summary Table ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("BENCHMARK TABLE — Large-N Validation")
    print(f"{'='*60}")
    print(f"{'Model':<25} {'N/class':<8} {'Fisher AUROC':<14} {'95% CI':<18} {'Shuf':<8} {'LR':<8}")
    print(f"{'-'*25} {'-'*8} {'-'*14} {'-'*18} {'-'*8} {'-'*8}")

    for mname, mres in results["models"].items():
        if mres.get("status") != "COMPLETE":
            print(f"{mname:<25} FAILED: {mres.get('status', '?')}")
            continue
        fp = mres["fisher_pca64"]
        lr = mres["logistic_l2"]
        ci = f"[{fp['ci_lo']:.3f},{fp['ci_hi']:.3f}]"
        print(f"{mname:<25} {mres['n_per_class']:<8} {fp['auroc']:<14.4f} {ci:<18} "
              f"{fp['shuffled_auroc']:<8.4f} {lr['auroc']:<8.4f}")

    results["elapsed_s"] = round(time.time() - t0, 1)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_FILE}")
    print(f"Total elapsed: {results['elapsed_s']}s")


if __name__ == "__main__":
    run()
