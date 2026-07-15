"""
experiments/nonlinear_probe_v3/nonlinear_probe_v3.py

C3-v3: NONLINEAR PROBE RECOVERY — BILATERAL ORACLE (FIXED)
============================================================

WHAT C3-v2 EXPOSED
-------------------
C3-v2 revealed a critical estimator pathology: Fisher LDA with n=60 train per class
in d=1536 dimensions produces completely degenerate results. The shuffled control for
Qwen instruct Fisher reached 0.713 — higher than the real AUROC of 0.708. This is not
signal failure; it is probe failure. The covariance matrix is unidentifiable when n << d.

This is a known statistical pathology, not a problem specific to this project. Most
probing papers do not catch it because they don't run shuffled controls aggressively.

The "C3_RECOVERY" verdict in v2 (+0.085) was an artifact: Fisher(instruct) was noise
(0.708, shuffled=0.713), so any working probe appeared to "recover" above it. The actual
comparison of clean probes (LR L2, SVM PCA) showed base ≈ instruct (0.805 vs 0.793,
attenuation only −0.012, CIs fully overlapping at n=20 test per class).

THREE FIXES FOR C3-v3
----------------------

FIX 1: N_CAL = 150 per class.
  Gives n=112 train / 38 test per class (224/76 total). This is the minimum for
  meaningful AUROC estimation (CI width ~±0.08 at AUROC=0.80). At N=80 (n=20 test),
  CI width was ~±0.15 — far too wide. Prior experiments used n=150.

FIX 2: Regularized Fisher LDA (Ledoit-Wolf shrinkage).
  Replace raw Fisher LDA with LDA(solver='lsqr', shrinkage='auto'). Ledoit-Wolf
  analytically estimates the optimal shrinkage parameter, regularizing the covariance
  matrix when n << d. This is the principled fix — no arbitrary PCA dimension choice.
  Also: Fisher+PCA(64) as an additional variant to compare.

FIX 3: Pool = 4000 questions.
  Qwen base needs ~878 questions for 80 PARAM at 9.1% yield. For 150 PARAM, need
  ~1650 questions. 4000 gives sufficient headroom for all three models.

THE SCIENTIFIC QUESTION (now sharper)
--------------------------------------
After v1 (PARAM/WRONG) and v2 (bilateral oracle, small-N), the question is:

  At adequate statistical power (N=150/class), using probes that are stable in n<<d
  regimes, does the Qwen base→instruct bilateral oracle signal show:
    (a) Meaningful attenuation (Fisher_instruct << Fisher_base, matching prior −0.036)?
    (b) Nonlinear recovery (SVM/MLP significantly above regularized linear probe)?

  And for Llama: does bilateral oracle AUROC converge to prior 0.629 at N=150?

Note on the comparison baseline:
  Prior AUROC numbers (0.899 base, 0.864 instruct, 0.629 Llama) used large-N calibration.
  If regularized Fisher still disagrees at N=150, the discrepancy is real (not estimator noise).

PROBES IN THIS VERSION
-----------------------
  1. Fisher LDA (Ledoit-Wolf) — regularized Fisher, principled fix for n<<d
  2. Fisher LDA + PCA(64)      — aggressive dim-reduction variant for comparison
  3. Logistic Regression L2    — always-reliable linear baseline (C=1.0)
  4. SVM RBF + PCA(64)         — nonlinear, PCA preprocessing
  5. MLP-2 (256→128)           — nonlinear, stronger regularization (alpha=0.01)
  6. MLP-3 (512→256→128)       — deeper nonlinear

PRIMARY COMPARISON: regularized-Fisher vs LR vs SVM vs MLP
  Not: raw Fisher vs everything else (v2's mistake)

RECOVERY defined as: best_nonlinear - best_linear > 0.05
  where best_linear = max(fisher_lw_auroc, logistic_l2_auroc)
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

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "bitsandbytes>=0.46.1", "scikit-learn", "datasets",
                "huggingface_hub", "scipy"],
               check=False)

import numpy as np
import torch

try:
    _hf_token = ""
    try:
        from kaggle_secrets import UserSecretsClient as _USC
        _hf_token = _USC().get_secret("HF_TOKEN")
    except Exception:
        pass
    if not _hf_token:
        _hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""
    if _hf_token:
        from huggingface_hub import login as _hf_login
        _hf_login(token=_hf_token, add_to_git_credential=False)
        print("HF login: OK")
    else:
        print("WARNING: HF_TOKEN not set.")
except Exception as _e:
    print(f"HF login error: {_e}")

assert torch.cuda.is_available(), "GPU required"
_sm = torch.cuda.get_device_capability(0)
_sm = _sm[0] * 10 + _sm[1]
assert _sm >= 70, f"GPU sm_{_sm} not supported"
DEVICE = "cuda"
print(f"GPU: {torch.cuda.get_device_name(0)}  sm_{_sm}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ── Config ────────────────────────────────────────────────────────────────────
N_CAL             = 150    # per class — gives 112 train / 38 test
TRAIN_FRAC        = 0.75
MAX_GEN           = 60
N_BOOTSTRAP       = 1000
SEED              = 42
MAX_CONTEXT_CHARS = 800
POOL_SIZE         = 4000   # large enough for Qwen base at ~9% yield

OUT_DIR       = Path("/kaggle/working")
RESULTS_FILE  = OUT_DIR / "nonlinear_probe_v3_results.json"

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
        "note": "baseline; prior bilateral oracle AUROC 0.899 (large-N)",
    },
    {
        "name":       "qwen25_1.5b_instruct",
        "model_id":   "Qwen/Qwen2.5-1.5B-Instruct",
        "role":       "instruct",
        "family":     "Qwen2.5-1.5B",
        "prior_fisher_auroc": 0.864,
        "note": "RLHF-trained; prior bilateral oracle 0.864, Δ=-0.036",
    },
    {
        "name":       "llama3.2_3b_instruct",
        "model_id":   "meta-llama/Llama-3.2-3B-Instruct",
        "role":       "instruct_weak",
        "family":     "Llama-3.2-3B",
        "prior_fisher_auroc": 0.629,
        "note": "weak baseline; prior 0.629. Does N=150 converge to this?",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def load_data(n: int = POOL_SIZE) -> List[Dict]:
    from datasets import load_dataset
    ds = load_dataset("trivia_qa", "rc.wikipedia", split="train", streaming=True)
    pool: List[Dict] = []
    for row in ds:
        if not row["answer"]["aliases"]:
            continue
        context = ""
        ep = row.get("entity_pages", {})
        if ep and ep.get("wiki_context"):
            pages = ep["wiki_context"]
            if pages:
                context = pages[0][:MAX_CONTEXT_CHARS]
        if not context:
            sr = row.get("search_results", {})
            if sr and sr.get("search_context"):
                ctxs = sr["search_context"]
                if ctxs:
                    context = ctxs[0][:MAX_CONTEXT_CHARS]
        if not context:
            continue
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
        pr_ = len(c) / len(p)
        rc_ = len(c) / len(q)
        best = max(best, 2 * pr_ * rc_ / (pr_ + rc_))
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
# Generation
# ─────────────────────────────────────────────────────────────────────────────

def _build_prompt(tok, question: str, context: Optional[str] = None) -> str:
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
    """Step-1 hidden state from the NOCONTEXT pass — always."""
    prompt = _build_prompt(tok, question, context=None)
    ids = tok(prompt, return_tensors="pt",
              truncation=True, max_length=512).input_ids.to(DEVICE)
    captured = [None]
    def _hook(mod, inp, out):
        x = out[0] if isinstance(out, tuple) else out
        if x.shape[1] == 1:
            captured[0] = x[0, -1, :].float().detach().cpu().numpy()
    layers = get_layers(model)
    handle = layers[layer_idx].register_forward_hook(_hook)
    try:
        with torch.no_grad():
            pre = model(ids, use_cache=True)
            model(ids[:, -1:], past_key_values=pre.past_key_values, use_cache=False)
    finally:
        handle.remove()
    return captured[0]


# ─────────────────────────────────────────────────────────────────────────────
# Bilateral oracle labeling
# ─────────────────────────────────────────────────────────────────────────────

def bilateral_oracle_label(model, tok, question: str, context: str,
                            answers: List[str]) -> str:
    nc_ans = generate_answer(model, tok, question, context=None)
    nc_f1  = token_f1(nc_ans, answers)
    nc_ac  = answer_contains(nc_ans, answers)

    if nc_f1 >= 0.50 or nc_ac:
        return "PARAM"

    if nc_f1 > 0.05 or nc_ac:
        return "SKIP"  # partial — ambiguous

    wc_ans = generate_answer(model, tok, question, context=context)
    wc_f1  = token_f1(wc_ans, answers)
    wc_ac  = answer_contains(wc_ans, answers)

    if wc_f1 >= 0.50 or wc_ac:
        return "CTX_DEP"

    return "SKIP"


# ─────────────────────────────────────────────────────────────────────────────
# Calibration
# ─────────────────────────────────────────────────────────────────────────────

def collect_states(model, tok, layer_idx: int,
                   data_pool: List[Dict]) -> Tuple[np.ndarray, np.ndarray, Dict]:
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

        if elapsed > 4000:
            print(f"    TIMEOUT at {elapsed:.0f}s")
            break

        label = bilateral_oracle_label(
            model, tok, sample["question"], sample["context"], sample["answers"])

        stats["n_total"] += 1

        if label == "SKIP":
            stats["n_skip"] += 1
            continue
        if label == "PARAM" and len(param_hs) >= N_CAL:
            continue
        if label == "CTX_DEP" and len(ctxdep_hs) >= N_CAL:
            continue

        hs = get_step1_hs(model, tok, sample["question"], layer_idx)
        if hs is None:
            continue

        if label == "PARAM":
            param_hs.append(hs)
            stats["n_param"] += 1
        else:
            ctxdep_hs.append(hs)
            stats["n_ctxdep"] += 1

    stats["elapsed_s"] = round(time.time() - t0)
    n_p, n_c = len(param_hs), len(ctxdep_hs)
    print(f"  Calibration done: PARAM={n_p}  CTX_DEP={n_c}  "
          f"skip={stats['n_skip']}/{stats['n_total']}  elapsed={stats['elapsed_s']}s")

    if n_p < 30 or n_c < 30:
        raise RuntimeError(f"Insufficient data: PARAM={n_p}, CTX_DEP={n_c}")

    # Balance to equal n
    n_min = min(n_p, n_c)
    if n_min < max(n_p, n_c):
        print(f"  Balancing to {n_min}/class (was {n_p}P/{n_c}C)")
    return np.stack(param_hs[:n_min]), np.stack(ctxdep_hs[:n_min]), stats


# ─────────────────────────────────────────────────────────────────────────────
# Probes
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_auroc(y_true: np.ndarray, y_score: np.ndarray,
                    n_boot: int = N_BOOTSTRAP, seed: int = SEED
                    ) -> Tuple[float, float, float]:
    from sklearn.metrics import roc_auc_score
    rng_b = np.random.default_rng(seed)
    n = len(y_true)
    aurocs = []
    for _ in range(n_boot):
        idx = rng_b.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aurocs.append(roc_auc_score(y_true[idx], y_score[idx]))
    return (float(np.mean(aurocs)),
            float(np.percentile(aurocs, 2.5)),
            float(np.percentile(aurocs, 97.5)))


def train_test_split_arrays(param: np.ndarray, ctxdep: np.ndarray,
                             train_frac: float = TRAIN_FRAC):
    n = len(param)
    n_train = int(n * train_frac)

    idx_p = rng.permutation(n)
    idx_c = rng.permutation(n)

    p_tr, p_te = param[idx_p[:n_train]],  param[idx_p[n_train:]]
    c_tr, c_te = ctxdep[idx_c[:n_train]], ctxdep[idx_c[n_train:]]

    X_train = np.concatenate([p_tr, c_tr])
    y_train = np.array([1] * len(p_tr) + [0] * len(c_tr))
    X_test  = np.concatenate([p_te, c_te])
    y_test  = np.array([1] * len(p_te) + [0] * len(c_te))

    print(f"  Split: train={len(X_train)} ({len(p_tr)}P/{len(c_tr)}C)  "
          f"test={len(X_test)} ({len(p_te)}P/{len(c_te)}C)")
    return X_train, y_train, X_test, y_test


def run_all_probes(X_train, y_train, X_test, y_test) -> Dict:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    results: Dict = {}

    # Normalize
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    # PCA(64) for SVM and Fisher+PCA variant
    # 64 << n_train (112), so covariance is well-conditioned in PCA space
    pca64 = PCA(n_components=64, random_state=SEED)
    X_tr_pca64 = pca64.fit_transform(X_tr)
    X_te_pca64 = pca64.transform(X_te)
    print(f"  PCA(64): var={pca64.explained_variance_ratio_.sum():.3f}")

    def eval_probe(name: str, clf, X_tr_, X_te_) -> Dict:
        print(f"  Training {name} …")
        t0 = time.time()
        clf.fit(X_tr_, y_train)

        if hasattr(clf, "decision_function"):
            scores = clf.decision_function(X_te_)
        else:
            scores = clf.predict_proba(X_te_)[:, 1]

        base_auc = float(roc_auc_score(y_test, scores))
        mean, lo, hi = bootstrap_auroc(y_test, scores)

        # Shuffled control
        y_shuf = rng.permutation(y_train)
        clf.fit(X_tr_, y_shuf)
        if hasattr(clf, "decision_function"):
            shuf_scores = clf.decision_function(X_te_)
        else:
            shuf_scores = clf.predict_proba(X_te_)[:, 1]
        shuf_auc = float(roc_auc_score(y_test, shuf_scores))

        clean = "CLEAN" if shuf_auc < 0.55 else ("WARN" if shuf_auc < 0.65 else "FAIL")
        print(f"    {name}: AUROC={base_auc:.4f} [{lo:.4f},{hi:.4f}]  "
              f"shuffled={shuf_auc:.4f}  [{clean}]  ({time.time()-t0:.1f}s)")

        return {
            "auroc": base_auc,
            "auroc_ci_lo": lo,
            "auroc_ci_hi": hi,
            "shuffled_auroc": shuf_auc,
            "shuffled_status": clean,
        }

    # 1. Fisher LDA — Ledoit-Wolf regularization (FIX: principled n<<d solution)
    results["fisher_lda_lw"] = eval_probe(
        "Fisher LDA (Ledoit-Wolf)",
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        X_tr, X_te
    )

    # 2. Fisher LDA + PCA(64) — aggressive dim-reduction variant for comparison
    results["fisher_lda_pca64"] = eval_probe(
        "Fisher LDA + PCA(64)",
        LinearDiscriminantAnalysis(n_components=1),
        X_tr_pca64, X_te_pca64
    )

    # 3. Logistic Regression L2 — always-reliable linear baseline
    results["logistic_l2"] = eval_probe(
        "Logistic L2",
        LogisticRegression(C=1.0, max_iter=1000, random_state=SEED),
        X_tr, X_te
    )

    # 4. SVM RBF + PCA(64) — nonlinear probe
    results["svm_rbf_pca64"] = eval_probe(
        "SVM RBF + PCA(64)",
        SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=SEED),
        X_tr_pca64, X_te_pca64
    )

    # 5. MLP-2 — stronger regularization (alpha=0.01) for better behavior at N=150
    results["mlp_2layer"] = eval_probe(
        "MLP-2 (256→128, alpha=0.01)",
        MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=SEED,
            alpha=1e-2,
        ),
        X_tr, X_te
    )

    # 6. MLP-3 — deeper
    results["mlp_3layer"] = eval_probe(
        "MLP-3 (512→256→128, alpha=0.01)",
        MLPClassifier(
            hidden_layer_sizes=(512, 256, 128),
            activation="relu",
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=SEED,
            alpha=1e-2,
        ),
        X_tr, X_te
    )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Per-model run
# ─────────────────────────────────────────────────────────────────────────────

def run_model(model_cfg: Dict, data_pool: List[Dict]) -> Dict:
    name      = model_cfg["name"]
    model_id  = model_cfg["model_id"]
    role      = model_cfg["role"]
    prior_auc = model_cfg["prior_fisher_auroc"]

    print(f"\n{'='*70}")
    print(f"MODEL: {name}  ({role})")
    print(f"  prior bilateral oracle Fisher AUROC: {prior_auc:.3f}")
    print(f"{'='*70}")

    try:
        mdl, tok, layer_idx = load_model(model_id)
    except Exception as e:
        print(f"[ERROR] Load failed: {e}")
        return {"name": name, "status": "LOAD_ERROR", "error": str(e)}

    try:
        param_st, ctxdep_st, cal_stats = collect_states(mdl, tok, layer_idx, data_pool)
    except Exception as e:
        print(f"[ERROR] Calibration failed: {e}")
        del mdl; gc.collect(); torch.cuda.empty_cache()
        return {"name": name, "status": "CALIBRATION_ERROR", "error": str(e)}

    del mdl; gc.collect(); torch.cuda.empty_cache()
    print(f"  GPU freed. Running probes on CPU …")

    X_train, y_train, X_test, y_test = train_test_split_arrays(param_st, ctxdep_st)

    try:
        probe_results = run_all_probes(X_train, y_train, X_test, y_test)
    except Exception as e:
        print(f"[ERROR] Probe training failed: {e}")
        import traceback; traceback.print_exc()
        return {"name": name, "status": "PROBE_ERROR", "error": str(e)}

    # Identify best linear and best nonlinear among clean probes
    linear_keys    = ["fisher_lda_lw", "fisher_lda_pca64", "logistic_l2"]
    nonlinear_keys = ["svm_rbf_pca64", "mlp_2layer", "mlp_3layer"]

    def best_clean(keys):
        best_auc, best_key = 0.0, None
        for k in keys:
            r = probe_results.get(k, {})
            if r.get("shuffled_status") in ("CLEAN", "WARN") and r.get("auroc", 0) > best_auc:
                best_auc, best_key = r["auroc"], k
        return best_auc, best_key

    best_linear_auc,    best_linear_key    = best_clean(linear_keys)
    best_nonlinear_auc, best_nonlinear_key = best_clean(nonlinear_keys)
    recovery_delta = best_nonlinear_auc - best_linear_auc

    if recovery_delta > RECOVERY_THRESHOLD:
        recovery_verdict = "RECOVERY"
        recovery_note    = (f"Best nonlinear ({best_nonlinear_key}={best_nonlinear_auc:.4f}) "
                            f"+{recovery_delta:.4f} above best linear ({best_linear_key}={best_linear_auc:.4f}). "
                            f"Nonlinear structure exists beyond linear accessibility.")
    elif recovery_delta < NORECOVERY_THRESHOLD:
        recovery_verdict = "NO_RECOVERY"
        recovery_note    = (f"Best nonlinear ({best_nonlinear_auc:.4f}) ≈ "
                            f"best linear ({best_linear_auc:.4f}). "
                            f"Epistemic signal is linearly organized.")
    else:
        recovery_verdict = "PARTIAL_RECOVERY"
        recovery_note    = (f"Marginal recovery (+{recovery_delta:.4f}). "
                            f"Inconclusive — borderline.")

    # Fisher LW consistency with prior
    fisher_lw_auc = probe_results.get("fisher_lda_lw", {}).get("auroc", 0.0)
    fisher_delta  = fisher_lw_auc - prior_auc

    result = {
        "name":             name,
        "model_id":         model_id,
        "role":             role,
        "family":           model_cfg["family"],
        "status":           "COMPLETE",
        "calibration":      cal_stats,
        "n_per_class":      len(param_st),
        "layer_idx":        layer_idx,
        "hidden_dim":       param_st.shape[1],
        "prior_fisher_auroc": prior_auc,
        "probes":           probe_results,
        "best_linear_auroc":    best_linear_auc,
        "best_linear_probe":    best_linear_key,
        "best_nonlinear_auroc": best_nonlinear_auc,
        "best_nonlinear_probe": best_nonlinear_key,
        "recovery_delta":       round(recovery_delta, 4),
        "recovery_verdict":     recovery_verdict,
        "recovery_note":        recovery_note,
        "fisher_lw_vs_prior":   round(fisher_delta, 4),
        "fisher_lw_auroc":      fisher_lw_auc,
    }

    print(f"\n  ─── PROBE SUMMARY for {name} ───")
    for k, v in probe_results.items():
        flag = "  ← best linear" if k == best_linear_key else (
               "  ← best NL" if k == best_nonlinear_key else "")
        print(f"    {k:<36} AUROC={v['auroc']:.4f} [{v['auroc_ci_lo']:.4f},{v['auroc_ci_hi']:.4f}]"
              f"  shuffled={v['shuffled_auroc']:.4f} [{v['shuffled_status']}]{flag}")
    print(f"\n  Best linear:    {best_linear_key} = {best_linear_auc:.4f}")
    print(f"  Best nonlinear: {best_nonlinear_key} = {best_nonlinear_auc:.4f}")
    print(f"  Recovery delta: {recovery_delta:+.4f}  →  {recovery_verdict}")
    print(f"  Fisher LW vs prior: {fisher_lw_auc:.4f} vs {prior_auc:.3f} (Δ={fisher_delta:+.4f})")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Cross-model interpretation
# ─────────────────────────────────────────────────────────────────────────────

def interpret_cross_model(results: List[Dict]) -> Dict:
    completed = {r["name"]: r for r in results if r.get("status") == "COMPLETE"}
    interp: Dict = {}

    base_r = completed.get("qwen25_1.5b_base")
    inst_r = completed.get("qwen25_1.5b_instruct")

    if base_r and inst_r:
        # Use LR L2 for the clean attenuation estimate (most reliable single linear probe)
        lr_base = base_r["probes"].get("logistic_l2", {}).get("auroc", 0)
        lr_inst = inst_r["probes"].get("logistic_l2", {}).get("auroc", 0)
        lr_attenuation = lr_inst - lr_base

        # Use best clean linear for overall comparison
        bl_base = base_r["best_linear_auroc"]
        bl_inst = inst_r["best_linear_auroc"]
        nl_base = base_r["best_nonlinear_auroc"]
        nl_inst = inst_r["best_nonlinear_auroc"]

        linear_attenuation    = bl_inst - bl_base
        nonlinear_attenuation = nl_inst - nl_base
        recovery_advantage    = nonlinear_attenuation - linear_attenuation

        # Attenuation verdict
        if lr_attenuation < -0.03:
            atten_verdict = "ATTENUATION_CONFIRMED"
            atten_note    = (f"LR attenuation = {lr_attenuation:+.4f} (≤ -0.03 threshold). "
                             f"RLHF reduces bilateral oracle signal. "
                             f"Consistent with prior Δ=-0.036.")
        elif lr_attenuation > 0.01:
            atten_verdict = "NO_ATTENUATION"
            atten_note    = (f"LR attenuation = {lr_attenuation:+.4f} (positive). "
                             f"No evidence of RLHF reducing bilateral oracle accessibility. "
                             f"Prior Δ=-0.036 may have been small-N artifact or dataset confound.")
        else:
            atten_verdict = "ATTENUATION_WEAK"
            atten_note    = (f"LR attenuation = {lr_attenuation:+.4f} (small). "
                             f"Inconclusive — within noise.")

        # Recovery verdict (using clean probes comparison)
        if recovery_advantage > RECOVERY_THRESHOLD:
            recov_verdict = "C3_RECOVERY"
            recov_note    = (f"Nonlinear probes retain {recovery_advantage:+.4f} more AUROC "
                             f"across RLHF than linear probes. "
                             f"Nonlinear reorganization under RLHF is real.")
        elif recovery_advantage < -RECOVERY_THRESHOLD:
            recov_verdict = "C3_LINEAR_DOMINATES"
            recov_note    = (f"Linear and nonlinear probes attenuate equivalently "
                             f"(advantage={recovery_advantage:+.4f}). "
                             f"Geometry remains linearly organized throughout RLHF.")
        else:
            recov_verdict = "C3_NEUTRAL"
            recov_note    = (f"Recovery advantage = {recovery_advantage:+.4f}. "
                             f"No differential between linear and nonlinear under RLHF.")

        interp["qwen_base_vs_instruct"] = {
            "lr_base":           round(lr_base, 4),
            "lr_instruct":       round(lr_inst, 4),
            "lr_attenuation":    round(lr_attenuation, 4),
            "lr_prior_delta":    -0.036,
            "best_linear_base":  round(bl_base, 4),
            "best_linear_inst":  round(bl_inst, 4),
            "best_nl_base":      round(nl_base, 4),
            "best_nl_inst":      round(nl_inst, 4),
            "linear_attenuation":    round(linear_attenuation, 4),
            "nonlinear_attenuation": round(nonlinear_attenuation, 4),
            "recovery_advantage":    round(recovery_advantage, 4),
            "attenuation_verdict": atten_verdict,
            "attenuation_note":    atten_note,
            "recovery_verdict":    recov_verdict,
            "recovery_note":       recov_note,
        }

    llama_r = completed.get("llama3.2_3b_instruct")
    if llama_r:
        lr_llama = llama_r["probes"].get("logistic_l2", {}).get("auroc", 0)
        fw_llama = llama_r["fisher_lw_auroc"]
        nl_llama = llama_r["best_nonlinear_auroc"]
        nl_delta = nl_llama - max(lr_llama, fw_llama)
        prior    = 0.629

        if nl_delta > RECOVERY_THRESHOLD:
            llama_verdict = "DISTRIBUTED_STRUCTURE"
        elif nl_delta < NORECOVERY_THRESHOLD:
            llama_verdict = "GENUINELY_WEAK_LINEAR"
        else:
            llama_verdict = "MARGINAL"

        interp["llama_structural"] = {
            "fisher_lw_auroc":    round(fw_llama, 4),
            "logistic_l2_auroc":  round(lr_llama, 4),
            "best_nl_auroc":      round(nl_llama, 4),
            "nonlinear_delta":    round(nl_delta, 4),
            "prior_0.629_delta":  round(fw_llama - prior, 4),
            "verdict":            llama_verdict,
        }

    # Overall
    c3_prim = interp.get("qwen_base_vs_instruct", {})
    rv = c3_prim.get("recovery_verdict", "INCOMPLETE")
    av = c3_prim.get("attenuation_verdict", "INCOMPLETE")

    if rv == "C3_RECOVERY" and av in ("ATTENUATION_CONFIRMED", "ATTENUATION_WEAK"):
        thesis = "INFORMATION_NEQ_ACCESSIBILITY"
        thesis_note = ("RLHF attenuates linear accessibility while nonlinear probes "
                       "recover more — information is reorganized, not destroyed. "
                       "'Information ≠ Accessibility' thesis has direct evidence.")
    elif rv in ("C3_LINEAR_DOMINATES", "C3_NEUTRAL") and av == "ATTENUATION_CONFIRMED":
        thesis = "LINEAR_ATTENUATION_ONLY"
        thesis_note = ("RLHF reduces the magnitude of linearly accessible signal. "
                       "Nonlinear probes show equivalent attenuation — no hidden structure. "
                       "Correct framing: RLHF degrades epistemic signal linearly.")
    elif rv in ("C3_LINEAR_DOMINATES", "C3_NEUTRAL") and av in ("NO_ATTENUATION", "ATTENUATION_WEAK"):
        thesis = "SIGNAL_PRESERVED_LINEAR"
        thesis_note = ("Neither significant attenuation nor nonlinear recovery. "
                       "Bilateral oracle signal is largely preserved under RLHF "
                       "and remains linearly accessible. Prior Δ=-0.036 may have been noise.")
    elif rv == "C3_RECOVERY" and av == "NO_ATTENUATION":
        thesis = "NONLINEAR_ENRICHMENT"
        thesis_note = ("Unexpected: nonlinear probes recover MORE on instruct than base. "
                       "RLHF may be enriching nonlinear structure. Verify carefully.")
    else:
        thesis = "INCONCLUSIVE"
        thesis_note = "Insufficient data or incomplete results."

    interp["overall_verdict"] = thesis
    interp["overall_note"]    = thesis_note

    return interp


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    print(f"C3-v3: BILATERAL ORACLE — FIXED PROBES (N=150, Ledoit-Wolf LDA)  |  "
          f"{time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Fixes: N={N_CAL}/class, LDA(shrinkage=auto), PCA(64)+SVM, pool={POOL_SIZE}")

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
            "probe_fixes": ["ledoit_wolf_lda", "pca64_before_fisher", "alpha_0.01_mlp"],
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

    print(f"\n{'='*70}")
    print(f"CROSS-MODEL INTERPRETATION")
    print(f"{'='*70}")
    interpretation = interpret_cross_model(model_results)
    all_results["interpretation"] = interpretation
    all_results["status"] = "complete"
    all_results["elapsed_s"] = round(time.time() - t_start)

    print(f"\n{'='*70}")
    print(f"C3-v3 FINAL VERDICT: {interpretation.get('overall_verdict', 'INCOMPLETE')}")
    print(f"{'='*70}")
    print(interpretation.get("overall_note", ""))

    q = interpretation.get("qwen_base_vs_instruct", {})
    if q:
        print(f"\nQwen LR attenuation: {q.get('lr_base','?'):.4f} → {q.get('lr_instruct','?'):.4f}"
              f"  (Δ={q.get('lr_attenuation',0):+.4f}  prior: -0.036)")
        print(f"  Attenuation verdict: {q.get('attenuation_verdict','?')}")
        print(f"  Recovery advantage: {q.get('recovery_advantage',0):+.4f}")
        print(f"  Recovery verdict: {q.get('recovery_verdict','?')}")

    ll = interpretation.get("llama_structural", {})
    if ll:
        print(f"\nLlama Fisher LW: {ll.get('fisher_lw_auroc','?'):.4f}  "
              f"(prior 0.629, Δ={ll.get('prior_0.629_delta',0):+.4f})")
        print(f"  NL delta: {ll.get('nonlinear_delta',0):+.4f}  "
              f"Verdict: {ll.get('verdict','?')}")

    RESULTS_FILE.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\n[Final saved] {RESULTS_FILE}")
    print(f"Total elapsed: {time.time() - t_start:.0f}s")
    print(json.dumps(all_results, indent=2, default=str))


if __name__ == "__main__":
    main()
