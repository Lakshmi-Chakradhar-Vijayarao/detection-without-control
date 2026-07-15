#!/usr/bin/env python3
"""
experiments/pythia_sweep/pythia_sweep.py

Task 3.1 + 3.2 — Pythia Observability Sweep
=============================================
CLAIM UNDER TEST: C011 (EXPLORATORY — untested)
  Epistemic accessibility changes predictably during LLM training.

Two sweep modes (controlled by SWEEP_MODE):

MODE A — Training Dynamics (fixed size, varying checkpoint):
  Model: EleutherAI/pythia-1.4b
  Checkpoints: {512, 2000, 8000, 16000, 32000, 64000, 128000, 143000}
  Question: Does bilateral oracle AUROC change during training?
  Look for: monotonic growth, phase transition, flat, or non-monotone.

MODE B — Capability Scaling (fixed final checkpoint, varying model size):
  Sizes: pythia-70m, 160m, 410m, 1b, 1.4b, 2.8b, 6.9b (12b if GPU allows)
  Question: Does observability scale with model size?

PYTHIA-SPECIFIC NOTES
---------------------
Pythia models are GPT-style base models (no instruction tuning).
The standard instruction prompt format will fail — models generate continuations
not answers. Use a completion-style prompt with answer cue.

Protocol adaptations for base models:
  1. Prompt format: "Q: {question}\nA:" (no instruction wrapper)
  2. Oracle thresholds are relaxed: param_min_f1=0.30, ctxdep_min_ctx_f1=0.30
     (base models rarely achieve F1>=0.50 with short completions)
  3. Pool size = 5000 per checkpoint (base yield is much lower)
  4. N_target = 50 per class (feasible for base models)
  5. Answer: check if any gold answer token is in first 10 generated tokens

Layer selection: second-to-last transformer block (layer_idx = n_layers - 2)
  pythia-70m:  6 layers  → L4
  pythia-160m: 12 layers → L10
  pythia-410m: 24 layers → L22
  pythia-1b:   16 layers → L14
  pythia-1.4b: 24 layers → L22
  pythia-2.8b: 32 layers → L30
  pythia-6.9b: 32 layers → L30

REGISTRY: EXP_T3A_PYTHIA (PENDING → COMPLETE when run)
"""

from __future__ import annotations
import gc, json, os, random, sys, time
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score

# ── Sweep Configuration ───────────────────────────────────────────────────────
SWEEP_MODE   = "A"  # "A" = training dynamics, "B" = model size scaling

# Mode A config
DYNAMICS_MODEL = "EleutherAI/pythia-1.4b"
CHECKPOINTS    = [512, 2000, 8000, 16000, 32000, 64000, 128000, 143000]

# Mode B config
SIZE_MODELS = [
    {"name": "pythia-70m",   "model_id": "EleutherAI/pythia-70m",   "n_layers": 6},
    {"name": "pythia-160m",  "model_id": "EleutherAI/pythia-160m",  "n_layers": 12},
    {"name": "pythia-410m",  "model_id": "EleutherAI/pythia-410m",  "n_layers": 24},
    {"name": "pythia-1b",    "model_id": "EleutherAI/pythia-1b",    "n_layers": 16},
    {"name": "pythia-1.4b",  "model_id": "EleutherAI/pythia-1.4b",  "n_layers": 24},
    {"name": "pythia-2.8b",  "model_id": "EleutherAI/pythia-2.8b",  "n_layers": 32},
    {"name": "pythia-6.9b",  "model_id": "EleutherAI/pythia-6.9b",  "n_layers": 32},
]

# Oracle + probe config (base-model relaxed thresholds)
SEED           = 42
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
PCA_DIM        = 64
N_TARGET       = 40    # per class per checkpoint — reduced for T4 time budget
POOL_SIZE      = 2000  # 2000 × 8 checkpoints × ~2s ≈ 32000s (fits in 9h)
TRAIN_FRAC     = 0.75  # 30 train / 10 test per class
MAX_GEN        = 20    # base models: short completion
MAX_CTX        = 600
PARAM_MIN_F1   = 0.30  # relaxed for base models
CTX_MAX_F1     = 0.05
CTX_MIN_CTX    = 0.30  # relaxed for base models
OUTPUT_FILE    = f"pythia_sweep_{SWEEP_MODE}_results.json"

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
print(f"Device: {DEVICE}", flush=True)
if DEVICE == "cpu":
    raise RuntimeError("GPU required.")


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


# ── Prompt format (base model compatible) ─────────────────────────────────────
def fmt_prompt_base(q, ctx=None):
    """Completion-style prompt suitable for non-instruction-tuned models."""
    if ctx:
        return f"Background: {ctx}\n\nQ: {q}\nA:"
    return f"Q: {q}\nA:"

def generate_text(model, tok, prompt, max_new=MAX_GEN):
    inp = tok(prompt, return_tensors="pt", truncation=True, max_length=400).to(DEVICE)
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


# ── Oracle labeling (base-model adapted) ──────────────────────────────────────
def label_item(model, tok, item):
    q, ctx, ans = item["question"], item["context"], item["answers"]
    pred_no = generate_text(model, tok, fmt_prompt_base(q))
    f1_no   = token_f1(pred_no, ans)
    if f1_no >= PARAM_MIN_F1:
        return "PARAM", f1_no, None
    if ctx and f1_no <= CTX_MAX_F1:
        pred_ctx = generate_text(model, tok, fmt_prompt_base(q, ctx))
        f1_ctx   = token_f1(pred_ctx, ans)
        if f1_ctx >= CTX_MIN_CTX:
            return "CTX_DEP", f1_no, f1_ctx
    return "SKIP", f1_no, None


# ── Architecture-aware layer accessor ────────────────────────────────────────
def get_layers(model):
    """Return the list of transformer blocks regardless of model family."""
    # GPT-NeoX (Pythia)
    if hasattr(model, "gpt_neox"):
        return model.gpt_neox.layers
    # LLaMA / Qwen / Mistral / Gemma
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise AttributeError(f"Cannot find layers on {type(model).__name__}")


# ── HS Extraction ─────────────────────────────────────────────────────────────
def extract_step1_hs(model, tok, q, layer_idx):
    prompt   = fmt_prompt_base(q)
    inp      = tok(prompt, return_tensors="pt", truncation=True, max_length=400).to(DEVICE)
    captured = {}

    def hook_fn(module, inp_t, out):
        hs = out[0] if isinstance(out, tuple) else out
        if hs.shape[1] != 1:
            return
        if "hs" in captured:
            return
        captured["hs"] = hs[0, 0, :].detach().float().cpu().numpy()

    handle = get_layers(model)[layer_idx].register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            model.generate(**inp, max_new_tokens=2, do_sample=False,
                           pad_token_id=tok.eos_token_id, use_cache=True)
    finally:
        handle.remove()

    return captured.get("hs")


# ── Probe ─────────────────────────────────────────────────────────────────────
def compute_auroc(hs_param, hs_ctxdep):
    """Fisher+PCA64 with train/test split. Returns (test_auroc, shuf_auroc)."""
    n = min(len(hs_param), len(hs_ctxdep))
    if n < 8:
        return None, None

    n_train = max(4, int(n * TRAIN_FRAC))
    n_test  = n - n_train
    if n_test < 4:
        return None, None

    p_tr = hs_param[:n_train];  p_te = hs_param[n_train:n]
    c_tr = hs_ctxdep[:n_train]; c_te = hs_ctxdep[n_train:n]

    X_tr = np.vstack([p_tr, c_tr]).astype(np.float32)
    y_tr = np.array([1]*len(p_tr) + [0]*len(c_tr))
    X_te = np.vstack([p_te, c_te]).astype(np.float32)
    y_te = np.array([1]*len(p_te) + [0]*len(c_te))

    n_comp = min(PCA_DIM, X_tr.shape[1], X_tr.shape[0]-1)
    pca    = PCA(n_components=n_comp, random_state=SEED)
    X_tr_r = pca.fit_transform(X_tr)
    X_te_r = pca.transform(X_te)

    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    lda.fit(X_tr_r, y_tr)
    scores = lda.decision_function(X_te_r)
    auroc  = float(roc_auc_score(y_te, scores))
    y_shuf = y_te.copy(); np.random.shuffle(y_shuf)
    shuf   = float(roc_auc_score(y_shuf, scores))
    return round(auroc, 4), round(shuf, 4)


# ── Single checkpoint/model run ───────────────────────────────────────────────
def run_single(model_id, revision, n_layers, all_items, label):
    """Run bilateral oracle + Fisher+PCA64 for one checkpoint or model size."""
    print(f"\n  [{label}] model={model_id}  revision={revision or 'final'}", flush=True)

    result = {"label": label, "model_id": model_id, "revision": revision}

    try:
        tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision,
            torch_dtype=torch.float16,
            device_map=None, low_cpu_mem_usage=True,
        ).to(DEVICE).eval()
    except Exception as e:
        print(f"    LOAD_FAILED: {e}", flush=True)
        result["status"] = f"LOAD_FAILED: {str(e)[:100]}"
        return result

    layer_idx = n_layers - 2   # penultimate layer

    # Oracle labeling
    param_items, ctxdep_items = [], []
    n_skip = 0
    t_label = time.time()

    for i, item in enumerate(all_items[:POOL_SIZE]):
        if len(param_items) >= N_TARGET and len(ctxdep_items) >= N_TARGET:
            break
        if i % 200 == 0:
            elapsed = round(time.time() - t_label)
            print(f"    [{i}/{POOL_SIZE}] P={len(param_items)} C={len(ctxdep_items)} "
                  f"SKIP={n_skip}  ({elapsed}s)", flush=True)
        lbl, _, _ = label_item(model, tok, item)
        if lbl == "PARAM" and len(param_items) < N_TARGET:
            param_items.append(item)
        elif lbl == "CTX_DEP" and len(ctxdep_items) < N_TARGET:
            ctxdep_items.append(item)
        else:
            n_skip += 1

    n_param  = len(param_items)
    n_ctxdep = len(ctxdep_items)
    print(f"    Labeled: P={n_param}  C={n_ctxdep}  SKIP={n_skip}", flush=True)

    result.update({"n_param": n_param, "n_ctxdep": n_ctxdep, "n_skip": n_skip,
                   "layer_idx": layer_idx})

    if min(n_param, n_ctxdep) < 10:
        result["status"] = "INSUFFICIENT"
        del model; gc.collect()
        if DEVICE == "cuda": torch.cuda.empty_cache()
        return result

    # HS extraction
    hs_param, hs_ctxdep = [], []
    n_use = min(n_param, n_ctxdep)

    for item in param_items[:n_use]:
        v = extract_step1_hs(model, tok, item["question"], layer_idx)
        if v is not None: hs_param.append(v)

    for item in ctxdep_items[:n_use]:
        v = extract_step1_hs(model, tok, item["question"], layer_idx)
        if v is not None: hs_ctxdep.append(v)

    print(f"    HS extracted: P={len(hs_param)}  C={len(hs_ctxdep)}", flush=True)

    auroc, shuf_auroc = compute_auroc(hs_param, hs_ctxdep)
    shuf_status = "CLEAN" if (shuf_auroc or 0) < 0.60 else "WARN" if (shuf_auroc or 0) < 0.70 else "FAIL"

    print(f"    AUROC={auroc}  shuffled={shuf_auroc}  [{shuf_status}]", flush=True)

    result.update({
        "status": "COMPLETE",
        "auroc": auroc,
        "shuffled_auroc": shuf_auroc,
        "shuffled_status": shuf_status,
        "n_extracted": (len(hs_param), len(hs_ctxdep)),
    })

    del model; gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()
    return result


# ── Mode A: Training Dynamics ─────────────────────────────────────────────────
def run_mode_a(all_items):
    print(f"\n{'='*60}")
    print(f"Mode A: Training Dynamics — {DYNAMICS_MODEL}")
    print(f"Checkpoints: {CHECKPOINTS}")
    print(f"{'='*60}")

    # Infer n_layers for 1.4b
    n_layers_1_4b = 24
    _t0_mode = time.time()
    results = []
    for ckpt in CHECKPOINTS:
        revision = f"step{ckpt}"
        r = run_single(DYNAMICS_MODEL, revision, n_layers_1_4b, all_items,
                       label=f"step{ckpt}")
        results.append(r)
        # intermediate save after each checkpoint
        with open(OUTPUT_FILE, "w") as f:
            json.dump({"sweep_mode": SWEEP_MODE, "results": results,
                       "verdict": "IN_PROGRESS", "elapsed_s": round(time.time() - _t0_mode, 1)}, f, indent=2)
        print(f"  [checkpoint] saved after step{ckpt}", flush=True)

    # Print curve
    print(f"\n{'='*60}")
    print("Training Dynamics Curve:")
    print(f"{'Step':<12} {'AUROC':<10} {'Shuffled':<10} {'Status'}")
    print(f"{'-'*12} {'-'*10} {'-'*10} {'-'*10}")
    for r in results:
        ckpt_str = r.get("label", "?")
        auroc    = r.get("auroc", "FAIL")
        shuf     = r.get("shuffled_auroc", "FAIL")
        status   = r.get("status", "?")
        print(f"  {ckpt_str:<12} {str(auroc):<10} {str(shuf):<10} {status}")

    # Verdict
    valid = [(r["label"], r["auroc"]) for r in results
             if r.get("status") == "COMPLETE" and r.get("auroc") is not None]
    verdict = "UNKNOWN"
    if len(valid) >= 3:
        aurocs = [a for _, a in valid]
        diffs  = [aurocs[i+1] - aurocs[i] for i in range(len(aurocs)-1)]
        if all(d >= -0.01 for d in diffs):
            verdict = "MONOTONIC_GROWTH"
        elif all(d <= 0.01 for d in diffs):
            verdict = "MONOTONIC_DECAY"
        elif max(aurocs) - min(aurocs) < 0.05:
            verdict = "FLAT"
        else:
            mid_peak = np.argmax(aurocs)
            if 0 < mid_peak < len(aurocs)-1:
                verdict = "INVERTED_U"
            else:
                max_change = max(abs(d) for d in diffs)
                if max_change > 0.08:
                    verdict = "PHASE_TRANSITION"
                else:
                    verdict = "NOISY"

    print(f"\nVerdict: {verdict}")
    return results, verdict


# ── Mode B: Model Size Scaling ────────────────────────────────────────────────
def run_mode_b(all_items):
    print(f"\n{'='*60}")
    print(f"Mode B: Model Size Scaling")
    print(f"Models: {[m['name'] for m in SIZE_MODELS]}")
    print(f"{'='*60}")

    results = []
    for m_cfg in SIZE_MODELS:
        r = run_single(m_cfg["model_id"], None, m_cfg["n_layers"], all_items,
                       label=m_cfg["name"])
        results.append(r)

    print(f"\n{'='*60}")
    print("Size Scaling Results:")
    print(f"{'Model':<15} {'N_layers':<10} {'AUROC':<10} {'Shuffled':<10}")
    print(f"{'-'*15} {'-'*10} {'-'*10} {'-'*10}")
    for r, cfg in zip(results, SIZE_MODELS):
        auroc = r.get("auroc", "FAIL")
        shuf  = r.get("shuffled_auroc", "FAIL")
        print(f"  {cfg['name']:<15} {cfg['n_layers']:<10} {str(auroc):<10} {str(shuf):<10}")

    valid = [(r["label"], r["auroc"]) for r in results
             if r.get("status") == "COMPLETE" and r.get("auroc") is not None]
    verdict = "UNKNOWN"
    if len(valid) >= 3:
        aurocs = [a for _, a in valid]
        if aurocs[-1] > aurocs[0] + 0.05:
            verdict = "SCALING_POSITIVE"
        elif abs(aurocs[-1] - aurocs[0]) < 0.05:
            verdict = "SCALING_INVARIANT"
        else:
            verdict = "SCALING_NEGATIVE"

    print(f"\nVerdict: {verdict}")
    return results, verdict


# ── Main ──────────────────────────────────────────────────────────────────────
def run():
    t0 = time.time()

    print(f"\n{'='*60}")
    print(f"Pythia Observability Sweep — Mode {SWEEP_MODE}")
    print(f"Oracle thresholds: PARAM≥{PARAM_MIN_F1}  CTX_DEP≤{CTX_MAX_F1}/≥{CTX_MIN_CTX}")
    print(f"N_target={N_TARGET}/class  pool={POOL_SIZE}")
    print(f"{'='*60}")

    print("Loading dataset...")
    all_items = load_trivia()

    if SWEEP_MODE == "A":
        sweep_results, verdict = run_mode_a(all_items)
    elif SWEEP_MODE == "B":
        sweep_results, verdict = run_mode_b(all_items)
    else:
        raise ValueError(f"Unknown SWEEP_MODE: {SWEEP_MODE}")

    results = {
        "sweep_mode": SWEEP_MODE,
        "config": {
            "pca_dim": PCA_DIM,
            "n_target": N_TARGET,
            "pool_size": POOL_SIZE,
            "param_min_f1": PARAM_MIN_F1,
            "ctxdep_max_f1": CTX_MAX_F1,
            "ctxdep_min_ctx_f1": CTX_MIN_CTX,
            "checkpoints": CHECKPOINTS if SWEEP_MODE == "A" else None,
            "models": [m["name"] for m in SIZE_MODELS] if SWEEP_MODE == "B" else None,
        },
        "results": sweep_results,
        "verdict": verdict,
        "elapsed_s": round(time.time() - t0, 1),
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_FILE}")
    print(f"Total elapsed: {results['elapsed_s']}s")


if __name__ == "__main__":
    run()
