"""
experiments/nonlinear_probe_v2/nonlinear_probe_v2.py

C3-v2: NONLINEAR PROBE RECOVERY — BILATERAL ORACLE
====================================================

WHY THIS IS DIFFERENT FROM C3-v1
----------------------------------
C3-v1 used PARAM vs WRONG labels (correct vs incorrect output).
That task leaks output quality signals: fluency, confidence, formatting, answer length.
More importantly, the Qwen base model couldn't answer in instruction format → severe
class imbalance (66/800 PARAM), making the primary base vs instruct comparison invalid.

C3-v2 uses bilateral oracle labels:
  PARAM:   model answers correctly WITHOUT context (pure parametric knowledge)
  CTX_DEP: model fails WITHOUT context but succeeds WITH context (contextual knowledge)

Bilateral oracle properties:
1. Works for base models: correct is detected via token_f1 against gold + answer_contains
2. Directly comparable to prior AUROC numbers (0.899/0.864/0.629 all used this labeling)
3. Isolates knowledge-source separability, not output quality
4. The probe asks: "from the model's generation-step-1 HS on the bare question,
   can we tell whether it's drawing on parametric vs contextual knowledge?"
   That is the real epistemic question.

STEP-1 HS: always extracted from the NOCONTEXT pass (both PARAM and CTX_DEP).
The probe asks about the model's epistemic state before it even sees the context.

THE SCIENTIFIC QUESTION (sharpened after C3-v1)
-------------------------------------------------
C3-v1 showed Qwen instruct PARAM/WRONG signal is linearly organized (Fisher 0.804 > MLP 0.776).
This leaves open: is the attenuation under RLHF also linear in structure?

Three possibilities C3-v2 can distinguish:
  A. Linear dominates: Fisher already optimal on bilateral oracle too → RLHF reduces
     magnitude but doesn't reorganize geometry into nonlinear structure
  B. Architecture-dependent: Qwen stays linear, Llama shows nonlinear structure
  C. Task-dependent: bilateral oracle (harder separation) surfaces nonlinear structure
     that PARAM/WRONG (easier separation) doesn't require

SVM FIX: PCA(256) applied before SVM to prevent high-dimensional memorization.
  (C3-v1 Llama SVM shuffled=0.707 was a d=3072 dimensional memorization artifact.)

Models:
  Qwen2.5-1.5B base     — bilateral oracle AUROC prior: 0.899
  Qwen2.5-1.5B instruct — bilateral oracle AUROC prior: 0.864, Δ=-0.036 (attenuation)
  Llama-3.2-3B instruct — bilateral oracle AUROC prior: 0.629

Protocol:
  N_CAL = 80 per class (bilateral oracle is slower: 2 passes per CTX_DEP question)
  Pool: 2000 questions from TriviaQA rc.wikipedia (has entity_pages.wiki_context)
  Context: first 800 chars of entity_pages.wiki_context[0]
  Layer: n_layers - 2 (L26 for 28-layer models)
  Probes: Fisher LDA, LR L2, SVM RBF+PCA(256), MLP-2 (256→128), MLP-3 (512→256→128)
  Shuffled controls: all probes
  Bootstrap CI: 1000 iterations

Recovery threshold: MLP(instruct) - Fisher(instruct) > 0.05 → RECOVERY
"""

from __future__ import annotations

import functools
import builtins
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

builtins.print = functools.partial(builtins.print, flush=True)

# ── Install deps ──────────────────────────────────────────────────────────────
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "bitsandbytes>=0.46.1", "scikit-learn", "datasets",
                "huggingface_hub", "scipy"],
               check=False)

import numpy as np
import torch

# ── HF login ──────────────────────────────────────────────────────────────────
try:
    _hf_token = ""
    try:
        from kaggle_secrets import UserSecretsClient as _USC
        _hf_token = _USC().get_secret("HF_TOKEN")
    except Exception:
        pass
    if not _hf_token:
        _hf_token = (os.environ.get("HF_TOKEN") or
                     os.environ.get("HUGGING_FACE_HUB_TOKEN") or "")
    if _hf_token:
        from huggingface_hub import login as _hf_login
        _hf_login(token=_hf_token, add_to_git_credential=False)
        print("HF login: OK")
    else:
        print("WARNING: HF_TOKEN not set.")
except Exception as _e:
    print(f"HF login error: {_e}")

# ── GPU check ─────────────────────────────────────────────────────────────────
assert torch.cuda.is_available(), "GPU required"
_sm_major, _sm_minor = torch.cuda.get_device_capability(0)
_sm = _sm_major * 10 + _sm_minor
assert _sm >= 70, f"GPU sm_{_sm} not supported."
DEVICE = "cuda"
print(f"GPU: {torch.cuda.get_device_name(0)}  sm_{_sm}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ── Config ────────────────────────────────────────────────────────────────────
N_CAL          = 80     # per class (PARAM + CTX_DEP)
TRAIN_FRAC     = 0.75   # 60 train / 20 test per class
MAX_GEN        = 60     # tokens for answer generation
N_BOOTSTRAP    = 1000
SEED           = 42
MAX_CONTEXT_CHARS = 800   # characters of wiki context to include
POOL_SIZE      = 2000     # question pool size — bilateral oracle has ~15-25% yield

OUT_DIR        = Path("/kaggle/working")
RESULTS_FILE   = OUT_DIR / "nonlinear_probe_v2_results.json"

RECOVERY_THRESHOLD   = 0.05
NORECOVERY_THRESHOLD = 0.02

rng = np.random.default_rng(SEED)

MODELS = [
    {
        "name":       "qwen25_1.5b_base",
        "model_id":   "Qwen/Qwen2.5-1.5B",
        "role":       "base",
        "family":     "Qwen2.5-1.5B",
        "prior_fisher_auroc": 0.899,
        "note": "Primary comparison: base line for RLHF attenuation test.",
    },
    {
        "name":       "qwen25_1.5b_instruct",
        "model_id":   "Qwen/Qwen2.5-1.5B-Instruct",
        "role":       "instruct",
        "family":     "Qwen2.5-1.5B",
        "prior_fisher_auroc": 0.864,
        "note": "Primary C3 target. Prior attenuation Δ=-0.036. Does nonlinear probe recover?",
    },
    {
        "name":       "llama3.2_3b_instruct",
        "model_id":   "meta-llama/Llama-3.2-3B-Instruct",
        "role":       "instruct_weak",
        "family":     "Llama-3.2-3B",
        "prior_fisher_auroc": 0.629,
        "note": "Weak bilateral oracle baseline. answer_jump=0.101. Linear vs distributed structure?",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading — TriviaQA rc.wikipedia (has entity_pages.wiki_context)
# ─────────────────────────────────────────────────────────────────────────────

def load_data(n: int = POOL_SIZE) -> List[Dict]:
    from datasets import load_dataset
    ds = load_dataset("trivia_qa", "rc.wikipedia", split="train", streaming=True)
    pool: List[Dict] = []
    for row in ds:
        if not row["answer"]["aliases"]:
            continue
        # Get context from entity_pages.wiki_context (correct field per prior experiment notes)
        context = ""
        ep = row.get("entity_pages", {})
        if ep and ep.get("wiki_context"):
            pages = ep["wiki_context"]
            if pages:
                context = pages[0][:MAX_CONTEXT_CHARS]
        if not context:
            # Fallback to search_results
            sr = row.get("search_results", {})
            if sr and sr.get("search_context"):
                contexts = sr["search_context"]
                if contexts:
                    context = contexts[0][:MAX_CONTEXT_CHARS]
        if not context:
            continue  # skip questions with no context (can't run bilateral oracle)
        pool.append({
            "question": row["question"],
            "answers":  row["answer"]["aliases"],
            "context":  context,
        })
        if len(pool) >= n:
            break
    print(f"Loaded {len(pool)} TriviaQA questions with context")
    return pool


def token_f1(pred: str, golds: List[str]) -> float:
    p = set(pred.lower().split())
    best = 0.0
    for g in golds:
        q = set(g.lower().split())
        c = p & q
        if not c or not p or not q:
            continue
        pr = len(c) / len(p)
        rc = len(c) / len(q)
        best = max(best, 2 * pr * rc / (pr + rc))
    return best


def answer_contains(pred: str, golds: List[str]) -> bool:
    pred_l = pred.lower()
    return any(g.lower().strip() in pred_l for g in golds if g.strip())


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_id: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    print(f"\nLoading {model_id} …")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb,
        device_map=None, low_cpu_mem_usage=True, trust_remote_code=True,
    ).to(DEVICE).eval()
    n_layers = mdl.config.num_hidden_layers
    hidden   = mdl.config.hidden_size
    layer    = max(0, n_layers - 2)
    print(f"  {model_id}: n_layers={n_layers}  hidden={hidden}  probe_layer={layer}")
    return mdl, tok, layer


def get_layers(model) -> list:
    for path in ["model.layers", "model.language_model.layers", "transformer.h"]:
        try:
            obj = model
            for p in path.split("."):
                obj = getattr(obj, p)
            if hasattr(obj, "__len__") and len(obj) > 0:
                return list(obj)
        except AttributeError:
            continue
    raise RuntimeError(f"Cannot find layers in {type(model).__name__}")


# ─────────────────────────────────────────────────────────────────────────────
# Generation and step-1 HS extraction
# ─────────────────────────────────────────────────────────────────────────────

def _build_prompt(tok, question: str, context: Optional[str] = None) -> str:
    """Build prompt with or without context. Handles chat vs base models."""
    if context:
        user_content = f"Context:\n{context}\n\nQuestion: {question}"
    else:
        user_content = question

    try:
        msgs = [{"role": "user", "content": user_content}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        if context:
            return f"Context: {context}\n\nQ: {question}\nA:"
        return f"Q: {question}\nA:"


def generate_answer(model, tok, question: str,
                    context: Optional[str] = None) -> str:
    prompt = _build_prompt(tok, question, context)
    ids = tok(prompt, return_tensors="pt",
              truncation=True, max_length=768).input_ids.to(DEVICE)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=MAX_GEN, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


def get_step1_hs(model, tok, question: str, layer_idx: int) -> Optional[np.ndarray]:
    """
    Step-1 hidden state from the NOCONTEXT pass.
    Always nocontext — the probe asks about parametric vs contextual organization
    before any context is given.
    """
    prompt = _build_prompt(tok, question, context=None)
    ids = tok(prompt, return_tensors="pt",
              truncation=True, max_length=512).input_ids.to(DEVICE)

    captured: List = [None]

    def _hook(mod, inp, out):
        x = out[0] if isinstance(out, tuple) else out
        if x.shape[1] == 1:
            captured[0] = x[0, -1, :].float().detach().cpu().numpy()

    layers = get_layers(model)
    handle = layers[layer_idx].register_forward_hook(_hook)
    try:
        with torch.no_grad():
            pre = model(ids, use_cache=True)
            pkv = pre.past_key_values
            model(ids[:, -1:], past_key_values=pkv, use_cache=False)
    finally:
        handle.remove()
    return captured[0]


# ─────────────────────────────────────────────────────────────────────────────
# Bilateral oracle labeling
# ─────────────────────────────────────────────────────────────────────────────

def bilateral_oracle_label(model, tok, question: str, context: str,
                            answers: List[str]) -> Tuple[str, float, float]:
    """
    Returns (label, nocontext_f1, withcontext_f1).
    label: "PARAM" | "CTX_DEP" | "SKIP"

    Protocol:
      PARAM:   nocontext_f1 >= 0.50 OR answer_contains(nocontext)
      CTX_DEP: nocontext clearly wrong (f1 <= 0.05 AND NOT answer_contains)
               AND withcontext correct (f1 >= 0.50 OR answer_contains)
      SKIP:    everything else (partial, both wrong, both right)

    Step-1 HS for the probe is ALWAYS from nocontext pass.
    """
    nc_ans = generate_answer(model, tok, question, context=None)
    nc_f1  = token_f1(nc_ans, answers)
    nc_ac  = answer_contains(nc_ans, answers)

    if nc_f1 >= 0.50 or nc_ac:
        return "PARAM", nc_f1, -1.0

    if nc_f1 > 0.05 or nc_ac:
        return "SKIP", nc_f1, -1.0

    # Model is clearly wrong without context — check with context
    wc_ans = generate_answer(model, tok, question, context=context)
    wc_f1  = token_f1(wc_ans, answers)
    wc_ac  = answer_contains(wc_ans, answers)

    if wc_f1 >= 0.50 or wc_ac:
        return "CTX_DEP", nc_f1, wc_f1

    return "SKIP", nc_f1, wc_f1


# ─────────────────────────────────────────────────────────────────────────────
# Calibration — collect PARAM and CTX_DEP hidden states
# ─────────────────────────────────────────────────────────────────────────────

def collect_states(model, tok, layer_idx: int,
                   data_pool: List[Dict]) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Returns (param_states, ctxdep_states, stats) each shaped (N_CAL, hidden_dim).
    Both use nocontext step-1 hidden states.
    """
    param_hs:  List[np.ndarray] = []
    ctxdep_hs: List[np.ndarray] = []
    stats = {"n_param": 0, "n_ctxdep": 0, "n_skip": 0, "n_total": 0}

    shuffled = list(data_pool)
    rng.shuffle(shuffled)

    t0 = time.time()
    for i, sample in enumerate(shuffled):
        if len(param_hs) >= N_CAL and len(ctxdep_hs) >= N_CAL:
            break

        elapsed = time.time() - t0
        if i % 20 == 0:
            print(f"    [{i}/{len(shuffled)}] PARAM={len(param_hs)} "
                  f"CTX_DEP={len(ctxdep_hs)}  elapsed={elapsed:.0f}s")

        if elapsed > 3000:
            print(f"    TIMEOUT at {elapsed:.0f}s")
            break

        if len(param_hs) >= N_CAL and len(ctxdep_hs) >= N_CAL:
            break

        label, nc_f1, wc_f1 = bilateral_oracle_label(
            model, tok, sample["question"], sample["context"], sample["answers"])

        stats["n_total"] += 1

        if label == "SKIP":
            stats["n_skip"] += 1
            continue

        if label == "PARAM" and len(param_hs) >= N_CAL:
            continue
        if label == "CTX_DEP" and len(ctxdep_hs) >= N_CAL:
            continue

        # Extract step-1 HS from nocontext pass
        hs = get_step1_hs(model, tok, sample["question"], layer_idx)
        if hs is None:
            continue

        if label == "PARAM":
            param_hs.append(hs)
            stats["n_param"] += 1
        else:
            ctxdep_hs.append(hs)
            stats["n_ctxdep"] += 1

    n_p, n_c = len(param_hs), len(ctxdep_hs)
    stats["elapsed_s"] = round(time.time() - t0)
    print(f"  Calibration done: PARAM={n_p}  CTX_DEP={n_c}  "
          f"skip_rate={stats['n_skip']}/{stats['n_total']}  "
          f"elapsed={stats['elapsed_s']}s")

    if n_p < 20 or n_c < 20:
        raise RuntimeError(f"Insufficient data: PARAM={n_p}, CTX_DEP={n_c}")

    return np.stack(param_hs), np.stack(ctxdep_hs), stats


# ─────────────────────────────────────────────────────────────────────────────
# Probe training and evaluation
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_auroc(y_true: np.ndarray, y_score: np.ndarray,
                    n_boot: int = N_BOOTSTRAP, seed: int = SEED) -> Tuple[float, float, float]:
    from sklearn.metrics import roc_auc_score
    rng_b = np.random.default_rng(seed)
    n = len(y_true)
    aurocs = []
    for _ in range(n_boot):
        idx = rng_b.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aurocs.append(roc_auc_score(y_true[idx], y_score[idx]))
    aurocs = np.array(aurocs)
    return float(np.mean(aurocs)), float(np.percentile(aurocs, 2.5)), float(np.percentile(aurocs, 97.5))


def train_test_split_arrays(param: np.ndarray, ctxdep: np.ndarray,
                             train_frac: float = TRAIN_FRAC):
    n_p, n_c = len(param), len(ctxdep)
    n_p_train = int(n_p * train_frac)
    n_c_train = int(n_c * train_frac)

    idx_p = rng.permutation(n_p)
    idx_c = rng.permutation(n_c)

    p_train, p_test = param[idx_p[:n_p_train]], param[idx_p[n_p_train:]]
    c_train, c_test = ctxdep[idx_c[:n_c_train]], ctxdep[idx_c[n_c_train:]]

    X_train = np.concatenate([p_train, c_train])
    y_train = np.array([1] * len(p_train) + [0] * len(c_train))
    X_test  = np.concatenate([p_test, c_test])
    y_test  = np.array([1] * len(p_test) + [0] * len(c_test))

    print(f"  Split: train={len(X_train)} ({len(p_train)}P/{len(c_train)}C)  "
          f"test={len(X_test)} ({len(p_test)}P/{len(c_test)}C)")
    return X_train, y_train, X_test, y_test


def run_all_probes(X_train: np.ndarray, y_train: np.ndarray,
                   X_test: np.ndarray,  y_test: np.ndarray) -> Dict:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline

    results: Dict = {}

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    # PCA-reduced versions for SVM (prevents high-dim memorization)
    pca_dim = min(256, X_tr.shape[0] - 1, X_tr.shape[1])
    pca = PCA(n_components=pca_dim, random_state=SEED)
    X_tr_pca = pca.fit_transform(X_tr)
    X_te_pca = pca.transform(X_te)
    print(f"  PCA: {X_tr.shape[1]}d → {pca_dim}d "
          f"(var={pca.explained_variance_ratio_.sum():.3f})")

    def eval_probe(name: str, clf, X_tr_, X_te_):
        print(f"  Training {name} …")
        t0 = time.time()
        clf.fit(X_tr_, y_train)

        if hasattr(clf, "decision_function"):
            scores = clf.decision_function(X_te_)
        else:
            scores = clf.predict_proba(X_te_)[:, 1]

        base_auroc = float(roc_auc_score(y_test, scores))
        mean, lo, hi = bootstrap_auroc(y_test, scores)

        # Shuffled control
        y_shuf = rng.permutation(y_train)
        clf.fit(X_tr_, y_shuf)
        if hasattr(clf, "decision_function"):
            shuf_scores = clf.decision_function(X_te_)
        else:
            shuf_scores = clf.predict_proba(X_te_)[:, 1]
        shuf_auroc = float(roc_auc_score(y_test, shuf_scores))

        print(f"    {name}: AUROC={base_auroc:.4f} [{lo:.4f},{hi:.4f}]  "
              f"shuffled={shuf_auroc:.4f}  ({time.time()-t0:.1f}s)")

        return {
            "auroc": base_auroc,
            "auroc_ci_lo": lo,
            "auroc_ci_hi": hi,
            "shuffled_auroc": shuf_auroc,
        }

    # 1. Fisher LDA (linear baseline — comparable to prior AUROC numbers)
    results["fisher_lda"] = eval_probe(
        "Fisher LDA",
        LinearDiscriminantAnalysis(n_components=1),
        X_tr, X_te
    )

    # 2. Logistic Regression L2 (linear, regularized)
    results["logistic_l2"] = eval_probe(
        "Logistic L2",
        LogisticRegression(C=1.0, max_iter=1000, random_state=SEED),
        X_tr, X_te
    )

    # 3. Kernel SVM RBF on PCA-reduced features (fixes high-dim memorization from v1)
    results["svm_rbf_pca"] = eval_probe(
        "SVM RBF (PCA-256)",
        SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=SEED),
        X_tr_pca, X_te_pca
    )

    # 4. MLP 2-layer (nonlinear — main recovery test)
    results["mlp_2layer"] = eval_probe(
        "MLP-2 (256→128)",
        MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=SEED,
            alpha=1e-3,
        ),
        X_tr, X_te
    )

    # 5. MLP 3-layer (deeper nonlinear)
    results["mlp_3layer"] = eval_probe(
        "MLP-3 (512→256→128)",
        MLPClassifier(
            hidden_layer_sizes=(512, 256, 128),
            activation="relu",
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=SEED,
            alpha=1e-3,
        ),
        X_tr, X_te
    )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Per-model analysis
# ─────────────────────────────────────────────────────────────────────────────

def run_model(model_cfg: Dict, data_pool: List[Dict]) -> Dict:
    name      = model_cfg["name"]
    model_id  = model_cfg["model_id"]
    role      = model_cfg["role"]
    prior_auc = model_cfg["prior_fisher_auroc"]

    print(f"\n{'='*70}")
    print(f"MODEL: {name}  ({role})")
    print(f"  model_id: {model_id}")
    print(f"  prior Fisher AUROC (bilateral oracle): {prior_auc:.3f}")
    print(f"{'='*70}")

    try:
        mdl, tok, layer_idx = load_model(model_id)
    except Exception as e:
        print(f"[ERROR] Failed to load {model_id}: {e}")
        return {"name": name, "status": "LOAD_ERROR", "error": str(e)}

    try:
        param_states, ctxdep_states, cal_stats = collect_states(
            mdl, tok, layer_idx, data_pool)
    except Exception as e:
        print(f"[ERROR] Calibration failed for {name}: {e}")
        del mdl; gc.collect(); torch.cuda.empty_cache()
        return {"name": name, "status": "CALIBRATION_ERROR", "error": str(e)}

    del mdl
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  GPU freed. Training probes on CPU …")

    X_train, y_train, X_test, y_test = train_test_split_arrays(param_states, ctxdep_states)

    try:
        probe_results = run_all_probes(X_train, y_train, X_test, y_test)
    except Exception as e:
        print(f"[ERROR] Probe training failed: {e}")
        import traceback; traceback.print_exc()
        return {"name": name, "status": "PROBE_ERROR", "error": str(e)}

    fisher_auroc   = probe_results["fisher_lda"]["auroc"]
    mlp2_auroc     = probe_results["mlp_2layer"]["auroc"]
    mlp3_auroc     = probe_results["mlp_3layer"]["auroc"]
    svm_auroc      = probe_results["svm_rbf_pca"]["auroc"]
    best_nonlinear = max(mlp2_auroc, mlp3_auroc, svm_auroc)
    recovery_delta = best_nonlinear - fisher_auroc

    if recovery_delta > RECOVERY_THRESHOLD:
        recovery_verdict = "RECOVERY"
        recovery_note    = (f"Nonlinear probe +{recovery_delta:.4f} above Fisher. "
                            f"Epistemic signal has nonlinear structure Fisher cannot capture.")
    elif recovery_delta < NORECOVERY_THRESHOLD:
        recovery_verdict = "NO_RECOVERY"
        recovery_note    = (f"Nonlinear probe {recovery_delta:+.4f} above Fisher (noise-level). "
                            f"Epistemic signal is linearly organized — Fisher is already optimal.")
    else:
        recovery_verdict = "PARTIAL_RECOVERY"
        recovery_note    = (f"Nonlinear probe +{recovery_delta:.4f} above Fisher. "
                            f"Marginal nonlinear advantage — needs larger N for definitive result.")

    fisher_delta = fisher_auroc - prior_auc
    consistency  = "CONSISTENT" if abs(fisher_delta) < 0.05 else "INCONSISTENT"

    result = {
        "name":             name,
        "model_id":         model_id,
        "role":             role,
        "family":           model_cfg["family"],
        "status":           "COMPLETE",
        "calibration":      cal_stats,
        "n_param":          len(param_states),
        "n_ctxdep":         len(ctxdep_states),
        "layer_idx":        layer_idx,
        "hidden_dim":       param_states.shape[1],
        "prior_fisher_auroc": prior_auc,
        "probes":           probe_results,
        "fisher_auroc":         fisher_auroc,
        "best_nonlinear_auroc": best_nonlinear,
        "recovery_delta":       round(recovery_delta, 4),
        "recovery_verdict":     recovery_verdict,
        "recovery_note":        recovery_note,
        "fisher_vs_prior":      round(fisher_delta, 4),
        "fisher_consistency":   consistency,
    }

    print(f"\n  ─── PROBE SUMMARY for {name} ───")
    for k, v in probe_results.items():
        print(f"    {k:<24} AUROC={v['auroc']:.4f} [{v['auroc_ci_lo']:.4f},{v['auroc_ci_hi']:.4f}]"
              f"  shuffled={v['shuffled_auroc']:.4f}")
    print(f"\n  RECOVERY DELTA (best_nonlinear - Fisher): {recovery_delta:+.4f}")
    print(f"  VERDICT: {recovery_verdict}")
    print(f"  {recovery_note}")
    print(f"  Fisher vs prior: {fisher_auroc:.4f} vs {prior_auc:.3f} "
          f"(Δ={fisher_delta:+.4f})  {consistency}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Cross-model interpretation
# ─────────────────────────────────────────────────────────────────────────────

def interpret_cross_model(results: List[Dict]) -> Dict:
    completed = {r["name"]: r for r in results if r.get("status") == "COMPLETE"}
    interp: Dict = {}

    # ── C3 Primary: Qwen base vs instruct ────────────────────────────────────
    base_r = completed.get("qwen25_1.5b_base")
    inst_r = completed.get("qwen25_1.5b_instruct")

    if base_r and inst_r:
        fb  = base_r["fisher_auroc"]
        fi  = inst_r["fisher_auroc"]
        nb  = base_r["best_nonlinear_auroc"]
        ni  = inst_r["best_nonlinear_auroc"]

        # Fisher attenuation under RLHF
        fisher_delta = fi - fb   # negative = attenuation (expected ~-0.036)

        # Nonlinear probe attenuation under RLHF
        nl_delta = ni - nb

        # Recovery advantage: does nonlinear retain MORE than linear under RLHF?
        # Positive = nonlinear probes preserve more signal across RLHF
        recovery_advantage = nl_delta - fisher_delta

        # Per-model: is base linear or nonlinear?
        base_verdict = base_r["recovery_verdict"]
        inst_verdict = inst_r["recovery_verdict"]

        if recovery_advantage > RECOVERY_THRESHOLD:
            c3_verdict = "C3_RECOVERY"
            c3_note    = (f"Nonlinear probes retain {recovery_advantage:+.4f} more AUROC "
                          f"across RLHF training than Fisher LDA. Information reorganizes "
                          f"to nonlinear structure under RLHF — 'information ≠ accessibility' "
                          f"thesis has direct evidence.")
        elif recovery_advantage < -RECOVERY_THRESHOLD:
            c3_verdict = "C3_LINEAR_DOMINATES"
            c3_note    = (f"Nonlinear probes attenuate MORE than Fisher under RLHF "
                          f"(advantage={recovery_advantage:+.4f}). Epistemic structure "
                          f"remains linearly organized throughout — RLHF reduces "
                          f"magnitude without reorganizing geometry. "
                          f"Possibility A confirmed for Qwen family.")
        elif abs(recovery_advantage) < NORECOVERY_THRESHOLD:
            c3_verdict = "C3_NEUTRAL"
            c3_note    = (f"Nonlinear and linear probes attenuate equally under RLHF "
                          f"(advantage={recovery_advantage:+.4f}). "
                          f"No evidence for geometric reorganization.")
        else:
            c3_verdict = "C3_PARTIAL"
            c3_note    = (f"Small differential ({recovery_advantage:+.4f}). "
                          f"Inconclusive at N={N_CAL}/class.")

        interp["qwen_base_vs_instruct"] = {
            "fisher_base":          round(fb, 4),
            "fisher_instruct":      round(fi, 4),
            "fisher_attenuation":   round(fisher_delta, 4),
            "fisher_prior_delta":   -0.036,
            "fisher_matches_prior": abs(fisher_delta - (-0.036)) < 0.04,
            "best_nl_base":         round(nb, 4),
            "best_nl_instruct":     round(ni, 4),
            "nl_attenuation":       round(nl_delta, 4),
            "recovery_advantage":   round(recovery_advantage, 4),
            "base_geometry":        base_verdict,
            "instruct_geometry":    inst_verdict,
            "verdict":              c3_verdict,
            "note":                 c3_note,
        }

    # ── C3 Secondary: Llama structural question ───────────────────────────────
    llama_r = completed.get("llama3.2_3b_instruct")
    if llama_r:
        lf = llama_r["fisher_auroc"]
        ln = llama_r["best_nonlinear_auroc"]
        ld = ln - lf

        prior_fisher = 0.629
        fisher_delta_from_prior = lf - prior_fisher

        if ld > RECOVERY_THRESHOLD:
            llama_verdict = "DISTRIBUTED_STRUCTURE"
            llama_note    = (f"MLP ({ln:.4f}) >> Fisher ({lf:.4f}) by +{ld:.4f}. "
                             f"Llama has nonlinear/distributed epistemic organization. "
                             f"Weak answer_jump (0.101) reflects LINEAR inaccessibility, "
                             f"not organizational absence. Linear probes underestimate "
                             f"Llama's actual Level-1 epistemic structure.")
        elif ld < NORECOVERY_THRESHOLD:
            llama_verdict = "GENUINELY_WEAK"
            llama_note    = (f"MLP ({ln:.4f}) ≈ Fisher ({lf:.4f}). "
                             f"Llama has predominantly linear epistemic structure at L26. "
                             f"Weak answer_jump genuinely reflects weak organizational strength, "
                             f"not linear inaccessibility of distributed signal.")
        else:
            llama_verdict = "MARGINAL"
            llama_note    = (f"Marginal nonlinear advantage (+{ld:.4f}). Inconclusive.")

        interp["llama_structural"] = {
            "fisher_auroc":              round(lf, 4),
            "best_nl_auroc":             round(ln, 4),
            "nonlinear_delta":           round(ld, 4),
            "fisher_vs_prior_0.629":     round(fisher_delta_from_prior, 4),
            "prior_used_bilateral_oracle": True,
            "answer_jump_prior":         0.101,
            "commit_rate_prior":         0.10,
            "verdict":                   llama_verdict,
            "note":                      llama_note,
        }

    # ── Overall: which of the 3 possibilities? ───────────────────────────────
    c3_prim = interp.get("qwen_base_vs_instruct", {}).get("verdict", "INCOMPLETE")

    if c3_prim == "C3_RECOVERY":
        thesis = "POSSIBILITY_C_TASK_DEPENDENT"
        thesis_note = ("Bilateral oracle surfaces nonlinear recovery that PARAM/WRONG missed. "
                       "The real epistemic signal (knowledge-source discrimination) has nonlinear "
                       "structure under RLHF. Possibility C confirmed: task-dependent nonlinearity. "
                       "'Information ≠ accessibility' thesis supported.")
    elif c3_prim == "C3_LINEAR_DOMINATES":
        thesis = "POSSIBILITY_A_LINEAR_DOMINATES"
        thesis_note = ("Epistemic geometry is predominantly linear throughout RLHF training. "
                       "RLHF reduces separability magnitude without reorganizing structure. "
                       "The project's scientific question sharpens to: "
                       "'under what conditions does linear separability degrade, and is that "
                       "degradation uniform or topology-preserving?'")
    elif c3_prim in ("C3_NEUTRAL", "C3_PARTIAL"):
        thesis = "INCONCLUSIVE_INCREASE_N"
        thesis_note = ("Marginal or neutral result. Run with N_CAL >= 150 for definitiveness.")
    else:
        thesis = "INCOMPLETE"
        thesis_note = "Not all models completed."

    interp["overall_verdict"] = thesis
    interp["overall_note"]    = thesis_note

    return interp


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    print(f"C3-v2: NONLINEAR PROBE RECOVERY — BILATERAL ORACLE  |  "
          f"{time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Labels: PARAM (parametric) vs CTX_DEP (contextual) — bilateral oracle protocol")
    print(f"N_CAL={N_CAL}/class  pool={POOL_SIZE}  layer=n_layers-2  PCA(256) for SVM")

    data_pool = load_data(n=POOL_SIZE)

    all_results: Dict = {
        "config": {
            "n_cal": N_CAL,
            "pool_size": POOL_SIZE,
            "train_frac": TRAIN_FRAC,
            "n_bootstrap": N_BOOTSTRAP,
            "recovery_threshold": RECOVERY_THRESHOLD,
            "norecovery_threshold": NORECOVERY_THRESHOLD,
            "max_gen": MAX_GEN,
            "max_context_chars": MAX_CONTEXT_CHARS,
            "seed": SEED,
            "label_type": "bilateral_oracle_PARAM_vs_CTX_DEP",
            "svm_pca_dim": 256,
        },
        "models": {},
        "interpretation": {},
        "status": "running",
    }

    model_results = []
    for cfg in MODELS:
        try:
            r = run_model(cfg, data_pool)
        except Exception as e:
            print(f"\n[FATAL] {cfg['name']}: {e}")
            import traceback; traceback.print_exc()
            r = {"name": cfg["name"], "status": "FATAL_ERROR", "error": str(e)}

        model_results.append(r)
        all_results["models"][cfg["name"]] = r
        all_results["elapsed_s"] = round(time.time() - t_start)
        RESULTS_FILE.write_text(json.dumps(all_results, indent=2, default=str))
        print(f"\n[Saved] {RESULTS_FILE}")

    # Cross-model interpretation
    print(f"\n{'='*70}")
    print(f"CROSS-MODEL INTERPRETATION")
    print(f"{'='*70}")
    interpretation = interpret_cross_model(model_results)
    all_results["interpretation"] = interpretation
    all_results["status"] = "complete"
    all_results["elapsed_s"] = round(time.time() - t_start)

    print(f"\n{'='*70}")
    print(f"C3-v2 FINAL VERDICT: {interpretation.get('overall_verdict', 'INCOMPLETE')}")
    print(f"{'='*70}")
    print(interpretation.get("overall_note", ""))

    qwen_cross = interpretation.get("qwen_base_vs_instruct", {})
    if qwen_cross:
        print(f"\nQwen base vs instruct (bilateral oracle):")
        print(f"  Fisher attenuation: {qwen_cross.get('fisher_base','?'):.4f} → "
              f"{qwen_cross.get('fisher_instruct','?'):.4f}  "
              f"(Δ={qwen_cross.get('fisher_attenuation',0):+.4f}  prior: -0.036)")
        print(f"  NL attenuation:     {qwen_cross.get('best_nl_base','?'):.4f} → "
              f"{qwen_cross.get('best_nl_instruct','?'):.4f}  "
              f"(Δ={qwen_cross.get('nl_attenuation',0):+.4f})")
        print(f"  Recovery advantage: {qwen_cross.get('recovery_advantage',0):+.4f}")
        print(f"  Base geometry: {qwen_cross.get('base_geometry','?')}")
        print(f"  Instruct geometry: {qwen_cross.get('instruct_geometry','?')}")
        print(f"  Verdict: {qwen_cross.get('verdict','?')}")

    llama_cross = interpretation.get("llama_structural", {})
    if llama_cross:
        print(f"\nLlama structural question (bilateral oracle):")
        print(f"  Fisher AUROC: {llama_cross.get('fisher_auroc','?'):.4f}  "
              f"(prior 0.629, Δ={llama_cross.get('fisher_vs_prior_0.629',0):+.4f})")
        print(f"  Best NL AUROC: {llama_cross.get('best_nl_auroc','?'):.4f}")
        print(f"  Nonlinear delta: {llama_cross.get('nonlinear_delta',0):+.4f}")
        print(f"  Verdict: {llama_cross.get('verdict','?')}")

    RESULTS_FILE.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\n[Final saved] {RESULTS_FILE}")
    print(f"Total elapsed: {time.time() - t_start:.0f}s")
    print(json.dumps(all_results, indent=2, default=str))


if __name__ == "__main__":
    main()
