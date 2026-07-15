"""
experiments/nonlinear_probe_v1/nonlinear_probe_v1.py

NONLINEAR PROBE RECOVERY TEST (C3)
====================================

Central question: Does information survive RLHF in a form that nonlinear probes
can recover even when Fisher LDA cannot?

This is the EXISTENTIAL experiment for the project's core thesis.

Two outcomes:

  RECOVERY: MLP AUROC (instruct) significantly > Fisher AUROC (instruct)
    → RLHF reorganizes the accessibility SURFACE, not the information content
    → "Information ≠ Accessibility" thesis gets direct evidence
    → RLHF is a transparency problem (information survives, window closes)
    → Safety and governance implications are real

  NO_RECOVERY: MLP AUROC (instruct) ≈ Fisher AUROC (instruct), both attenuated
    → RLHF may partially destroy the relevant information, not just reorganize it
    → The thesis requires revision: "RLHF reduces accessible information"
    → Safety implications shift: not a transparency problem but a representation problem

Secondary question (Llama):
  Does Llama-3.2-3B have weak Level-1 epistemic organization, or just weak linear
  accessibility? If MLP(Llama-instruct) >> Fisher(Llama-instruct), then Llama has
  distributed/nonlinear structure — commit_rate=0.10 is a linear-probe artifact,
  not a genuine organizational weakness. This changes the backbone-stratified
  interpretation of answer_jump_v2.

Models:
  1. Qwen2.5-1.5B base (baseline = 0.899 from rlhf_attenuation)
  2. Qwen2.5-1.5B instruct (attenuated = 0.864, Δ = -0.036)
  3. Llama-3.2-3B instruct (weak baseline = 0.629)

Labels: PARAM (correct answer, F1 >= 0.4) vs WRONG (incorrect, F1 < 0.1)
  [Consistent with rlhf_attenuation experiments. No context passage needed.]

Probe types:
  - Fisher LDA (linear baseline — matches existing AUROC numbers)
  - Logistic Regression L2 (linear, regularized — additional linear baseline)
  - Kernel SVM RBF (nonlinear, standard kernel method)
  - MLP-2 (2 hidden layers: 256→128, ReLU, dropout 0.3)
  - MLP-3 (3 hidden layers: 512→256→128, ReLU, dropout 0.3)

Protocol:
  - N_CAL = 150 per class (PARAM + WRONG)
  - Train/test split: 112 train / 38 test per class (75/25)
  - Bootstrap CI: 1000 iterations on test-set AUROC
  - Shuffled control: permute y labels, retrain, confirm AUROC ≈ 0.50
  - Layer: n_layers - 2 (L26 equivalent for 28-layer models)

Recovery threshold: MLP-2(instruct) - Fisher(instruct) > 0.05 → RECOVERY
No-recovery: difference < 0.02 → NO_RECOVERY
Partial: 0.02–0.05 → PARTIAL_RECOVERY

GPU: T4 (sm_75). All models at 4-bit NF4.
Expected time: ~50 min total (3 models × ~12 min calibration + probe training on CPU).
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
from dataclasses import dataclass, field
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
N_CAL          = 150    # per class (PARAM + WRONG)
TRAIN_FRAC     = 0.75   # 112 train + 38 test per class
MAX_GEN        = 80     # tokens for answer generation (short factual answers)
N_BOOTSTRAP    = 1000   # bootstrap iterations for AUROC CI
SEED           = 42
OUT_DIR        = Path("/kaggle/working")
RESULTS_FILE   = OUT_DIR / "nonlinear_probe_v1_results.json"
RECOVERY_THRESHOLD   = 0.05   # MLP - Fisher > this → RECOVERY
NORECOVERY_THRESHOLD = 0.02   # MLP - Fisher < this → NO_RECOVERY

rng = np.random.default_rng(SEED)

# Models to test: (name, model_id, role)
# role: "base" | "instruct" | "instruct_weak"
MODELS = [
    {
        "name":    "qwen25_1.5b_base",
        "model_id": "Qwen/Qwen2.5-1.5B",
        "role":    "base",
        "family":  "Qwen2.5-1.5B",
        "prior_fisher_auroc": 0.899,
        "note":    "Strong baseline. Attenuation: 0.899 → 0.864 under RLHF.",
    },
    {
        "name":    "qwen25_1.5b_instruct",
        "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "role":    "instruct",
        "family":  "Qwen2.5-1.5B",
        "prior_fisher_auroc": 0.864,
        "note":    "RLHF-attenuated. Primary C3 test: does MLP recover above 0.864?",
    },
    {
        "name":    "llama3.2_3b_instruct",
        "model_id": "meta-llama/Llama-3.2-3B-Instruct",
        "role":    "instruct_weak",
        "family":  "Llama-3.2-3B",
        "prior_fisher_auroc": 0.629,
        "note":    "Weak baseline. answer_jump_v2=0.101. Does MLP reveal hidden structure?",
    },
]

# Prior confirmed results for reference
CONFIRMED_PRIOR = {
    "qwen25_1.5b_delta": -0.036,
    "llama3.2_3b_instruct": 0.629,
    "qwen25_1.5b_base": 0.899,
    "qwen25_1.5b_instruct": 0.864,
}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading — TriviaQA (no context needed for PARAM vs WRONG)
# ─────────────────────────────────────────────────────────────────────────────

def load_data(n: int = 600) -> List[Dict]:
    from datasets import load_dataset
    ds = load_dataset("trivia_qa", "rc.wikipedia", split="train", streaming=True)
    pool: List[Dict] = []
    for row in ds:
        if not row["answer"]["aliases"]:
            continue
        pool.append({
            "question": row["question"],
            "answers":  row["answer"]["aliases"],
        })
        if len(pool) >= n:
            break
    print(f"Loaded {len(pool)} TriviaQA questions")
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
# Gen-step-1 hidden state extraction
# ─────────────────────────────────────────────────────────────────────────────

def get_step1_hs(model, tok, question: str, layer_idx: int) -> Optional[np.ndarray]:
    msgs = [{"role": "user", "content": question}]
    try:
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        prompt = f"Q: {question}\nA:"

    ids = tok(prompt, return_tensors="pt",
              truncation=True, max_length=512).input_ids.to(DEVICE)

    captured: List = [None]

    def _hook(mod, inp, out):
        x = out[0] if isinstance(out, tuple) else out
        # We want the FIRST generated token (step-1), shape (1, 1, hidden)
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


def generate_answer(model, tok, question: str) -> str:
    msgs = [{"role": "user", "content": question}]
    try:
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        prompt = f"Q: {question}\nA:"

    ids = tok(prompt, return_tensors="pt",
              truncation=True, max_length=512).input_ids.to(DEVICE)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=MAX_GEN, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


# ─────────────────────────────────────────────────────────────────────────────
# Calibration — collect PARAM and WRONG hidden states
# ─────────────────────────────────────────────────────────────────────────────

def collect_states(model, tok, layer_idx: int,
                   data_pool: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (param_states, wrong_states) each shaped (N_CAL, hidden_dim).
    PARAM = correct (F1 >= 0.4 OR answer_contains).
    WRONG = clearly incorrect (F1 < 0.1 AND NOT answer_contains).
    """
    param_hs: List[np.ndarray] = []
    wrong_hs:  List[np.ndarray] = []

    shuffled = list(data_pool)
    rng.shuffle(shuffled)

    t0 = time.time()
    for i, sample in enumerate(shuffled):
        if len(param_hs) >= N_CAL and len(wrong_hs) >= N_CAL:
            break

        elapsed = time.time() - t0
        if i % 10 == 0:
            print(f"    [{i}/{len(shuffled)}] PARAM={len(param_hs)} WRONG={len(wrong_hs)}"
                  f"  elapsed={elapsed:.0f}s")

        if elapsed > 2400:
            print(f"    TIMEOUT at {elapsed:.0f}s")
            break

        gen = generate_answer(model, tok, sample["question"])
        f1  = token_f1(gen, sample["answers"])
        ac  = answer_contains(gen, sample["answers"])
        is_correct   = (f1 >= 0.4 or ac)
        is_incorrect = (f1 < 0.1 and not ac)

        # Only use clear-cut cases to avoid label noise
        if not is_correct and not is_incorrect:
            continue

        if is_correct and len(param_hs) >= N_CAL:
            continue
        if is_incorrect and len(wrong_hs) >= N_CAL:
            continue

        hs = get_step1_hs(model, tok, sample["question"], layer_idx)
        if hs is None:
            continue

        if is_correct:
            param_hs.append(hs)
        else:
            wrong_hs.append(hs)

    n_p, n_w = len(param_hs), len(wrong_hs)
    print(f"  Calibration done: PARAM={n_p}  WRONG={n_w}  elapsed={time.time()-t0:.0f}s")

    if n_p < 20 or n_w < 20:
        raise RuntimeError(f"Insufficient data: PARAM={n_p}, WRONG={n_w}")

    return np.stack(param_hs), np.stack(wrong_hs)


# ─────────────────────────────────────────────────────────────────────────────
# Probe training and evaluation
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_auroc(y_true: np.ndarray, y_score: np.ndarray,
                    n_boot: int = N_BOOTSTRAP, seed: int = SEED) -> Tuple[float, float, float]:
    """Returns (mean, ci_lo, ci_hi) via percentile bootstrap."""
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


def train_test_split_arrays(param: np.ndarray, wrong: np.ndarray,
                             train_frac: float = TRAIN_FRAC
                             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns X_train, y_train, X_test, y_test."""
    n_p, n_w = len(param), len(wrong)
    n_p_train = int(n_p * train_frac)
    n_w_train = int(n_w * train_frac)

    idx_p = rng.permutation(n_p)
    idx_w = rng.permutation(n_w)

    p_train, p_test = param[idx_p[:n_p_train]], param[idx_p[n_p_train:]]
    w_train, w_test = wrong[idx_w[:n_w_train]], wrong[idx_w[n_w_train:]]

    X_train = np.concatenate([p_train, w_train])
    y_train = np.array([1] * len(p_train) + [0] * len(w_train))
    X_test  = np.concatenate([p_test, w_test])
    y_test  = np.array([1] * len(p_test) + [0] * len(w_test))

    print(f"  Split: train={len(X_train)} ({len(p_train)}P/{len(w_train)}W)  "
          f"test={len(X_test)} ({len(p_test)}P/{len(w_test)}W)")
    return X_train, y_train, X_test, y_test


@dataclass
class ProbeResult:
    name:        str
    auroc:       float
    auroc_ci_lo: float
    auroc_ci_hi: float
    shuffled_auroc: float     # sanity check — should be ~0.50


def run_all_probes(X_train: np.ndarray, y_train: np.ndarray,
                   X_test: np.ndarray,  y_test: np.ndarray,
                   hidden_dim: int) -> Dict[str, ProbeResult]:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    results: Dict[str, ProbeResult] = {}

    # Normalize features — critical for SVM and MLP
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    def eval_probe(name: str, clf, X_tr_: np.ndarray, X_te_: np.ndarray) -> ProbeResult:
        print(f"  Training {name} …")
        t0 = time.time()
        clf.fit(X_tr_, y_train)

        if hasattr(clf, "decision_function"):
            scores = clf.decision_function(X_te_)
        elif hasattr(clf, "predict_proba"):
            scores = clf.predict_proba(X_te_)[:, 1]
        else:
            raise RuntimeError(f"Cannot get scores from {name}")

        base_auroc  = float(roc_auc_score(y_test, scores))
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
        return ProbeResult(name, base_auroc, lo, hi, shuf_auroc)

    # 1. Fisher LDA (linear — baseline, matches existing AUROC numbers)
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

    # 3. Kernel SVM RBF (nonlinear — critical test)
    results["svm_rbf"] = eval_probe(
        "SVM RBF",
        SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=SEED),
        X_tr, X_te
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
            alpha=1e-3,   # L2 regularization
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
    print(f"  prior Fisher AUROC: {prior_auc:.3f}")
    print(f"{'='*70}")

    try:
        mdl, tok, layer_idx = load_model(model_id)
    except Exception as e:
        print(f"[ERROR] Failed to load {model_id}: {e}")
        return {"name": name, "status": "LOAD_ERROR", "error": str(e)}

    hidden_dim = mdl.config.hidden_size

    try:
        param_states, wrong_states = collect_states(mdl, tok, layer_idx, data_pool)
    except Exception as e:
        print(f"[ERROR] Calibration failed for {name}: {e}")
        del mdl; gc.collect(); torch.cuda.empty_cache()
        return {"name": name, "status": "CALIBRATION_ERROR", "error": str(e)}

    # Free GPU — probes run on CPU
    del mdl
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  GPU freed. Training probes on CPU …")

    X_train, y_train, X_test, y_test = train_test_split_arrays(param_states, wrong_states)

    try:
        probe_results = run_all_probes(X_train, y_train, X_test, y_test, hidden_dim)
    except Exception as e:
        print(f"[ERROR] Probe training failed: {e}")
        import traceback; traceback.print_exc()
        return {"name": name, "status": "PROBE_ERROR", "error": str(e)}

    # ── Compute recovery verdict ───────────────────────────────────────────
    fisher_auroc = probe_results["fisher_lda"].auroc
    mlp2_auroc   = probe_results["mlp_2layer"].auroc
    mlp3_auroc   = probe_results["mlp_3layer"].auroc
    svm_auroc    = probe_results["svm_rbf"].auroc

    best_nonlinear = max(mlp2_auroc, mlp3_auroc, svm_auroc)
    recovery_delta = best_nonlinear - fisher_auroc

    if recovery_delta > RECOVERY_THRESHOLD:
        recovery_verdict = "RECOVERY"
        recovery_note    = (f"Nonlinear probe {recovery_delta:+.4f} above Fisher. "
                            f"Information survives in nonlinear form → accessibility thesis supported.")
    elif recovery_delta < NORECOVERY_THRESHOLD:
        recovery_verdict = "NO_RECOVERY"
        recovery_note    = (f"Nonlinear probe {recovery_delta:+.4f} above Fisher (noise-level). "
                            f"Information may be partially destroyed, not just reorganized.")
    else:
        recovery_verdict = "PARTIAL_RECOVERY"
        recovery_note    = (f"Nonlinear probe {recovery_delta:+.4f} above Fisher. "
                            f"Partial recovery — interpretation requires more data.")

    # ── Consistency with prior Fisher AUROC ───────────────────────────────
    fisher_consistency = abs(fisher_auroc - prior_auc)
    consistency_note   = (f"Fisher AUROC {fisher_auroc:.4f} vs prior {prior_auc:.3f} "
                          f"(Δ={fisher_auroc - prior_auc:+.4f}). "
                          f"{'Consistent (< 0.05)' if fisher_consistency < 0.05 else 'INCONSISTENT — check calibration'}")

    result = {
        "name":             name,
        "model_id":         model_id,
        "role":             role,
        "family":           model_cfg["family"],
        "status":           "COMPLETE",
        "n_param":          len(param_states),
        "n_wrong":          len(wrong_states),
        "layer_idx":        layer_idx,
        "hidden_dim":       hidden_dim,
        "prior_fisher_auroc": prior_auc,
        "probes": {
            k: {
                "auroc":          v.auroc,
                "auroc_ci_lo":    v.auroc_ci_lo,
                "auroc_ci_hi":    v.auroc_ci_hi,
                "shuffled_auroc": v.shuffled_auroc,
            }
            for k, v in probe_results.items()
        },
        "fisher_auroc":         fisher_auroc,
        "best_nonlinear_auroc": best_nonlinear,
        "recovery_delta":       round(recovery_delta, 4),
        "recovery_verdict":     recovery_verdict,
        "recovery_note":        recovery_note,
        "fisher_consistency":   consistency_note,
    }

    # Print summary
    print(f"\n  ─── PROBE SUMMARY for {name} ───")
    for k, v in probe_results.items():
        print(f"    {k:<20} AUROC={v.auroc:.4f} [{v.auroc_ci_lo:.4f},{v.auroc_ci_hi:.4f}]"
              f"  shuffled={v.shuffled_auroc:.4f}")
    print(f"\n  RECOVERY DELTA (best_nonlinear - Fisher): {recovery_delta:+.4f}")
    print(f"  VERDICT: {recovery_verdict}")
    print(f"  {recovery_note}")
    print(f"  {consistency_note}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def interpret_cross_model(results: List[Dict]) -> Dict:
    """
    Cross-model interpretation after all three models complete.

    Key comparisons:
    1. Qwen base vs Qwen instruct: does nonlinear probe recover RLHF-lost signal?
    2. Qwen base vs Qwen instruct: does Fisher AUROC drop match prior (-0.036)?
    3. Llama instruct: does MLP reveal distributed structure vs Fisher underestimate?
    """
    completed = {r["name"]: r for r in results if r.get("status") == "COMPLETE"}

    interp: Dict = {"cross_model_analysis": {}}

    # ── C3 Primary: Qwen base vs instruct ────────────────────────────────────
    base_r = completed.get("qwen25_1.5b_base")
    inst_r = completed.get("qwen25_1.5b_instruct")

    if base_r and inst_r:
        fisher_base    = base_r["fisher_auroc"]
        fisher_inst    = inst_r["fisher_auroc"]
        best_nl_base   = base_r["best_nonlinear_auroc"]
        best_nl_inst   = inst_r["best_nonlinear_auroc"]

        fisher_delta   = fisher_inst - fisher_base   # negative = attenuation
        nl_delta       = best_nl_inst - best_nl_base  # if less negative than Fisher: recovery

        # Recovery = nonlinear recovers more than linear under RLHF
        # If nl_delta > fisher_delta: nonlinear retained more under RLHF
        recovery_advantage = nl_delta - fisher_delta

        if recovery_advantage > RECOVERY_THRESHOLD:
            c3_verdict = "C3_RECOVERY"
            c3_note    = (f"Nonlinear probes retain {recovery_advantage:+.4f} more AUROC across "
                          f"RLHF than Fisher LDA. Information survives RLHF in nonlinear form. "
                          f"'Information ≠ Accessibility' thesis has direct evidence.")
        elif recovery_advantage < NORECOVERY_THRESHOLD:
            c3_verdict = "C3_NO_RECOVERY"
            c3_note    = (f"Nonlinear probes degrade as much as Fisher LDA under RLHF "
                          f"(advantage={recovery_advantage:+.4f}). "
                          f"Information may be partially destroyed, not just reorganized. "
                          f"Thesis requires revision.")
        else:
            c3_verdict = "C3_PARTIAL"
            c3_note    = (f"Partial recovery (advantage={recovery_advantage:+.4f}). "
                          f"Interpretation ambiguous at this sample size. "
                          f"Increase N_CAL for definitive result.")

        interp["cross_model_analysis"]["qwen_base_vs_instruct"] = {
            "fisher_base":        fisher_base,
            "fisher_instruct":    fisher_inst,
            "fisher_delta":       round(fisher_delta, 4),
            "fisher_prior_delta": -0.036,
            "fisher_matches_prior": abs(fisher_delta - (-0.036)) < 0.03,
            "best_nl_base":       best_nl_base,
            "best_nl_instruct":   best_nl_inst,
            "nl_delta":           round(nl_delta, 4),
            "recovery_advantage": round(recovery_advantage, 4),
            "verdict":            c3_verdict,
            "note":               c3_note,
        }

    # ── C3 Secondary: Llama structural question ───────────────────────────────
    llama_r = completed.get("llama3.2_3b_instruct")
    if llama_r:
        llama_fisher = llama_r["fisher_auroc"]
        llama_nl     = llama_r["best_nonlinear_auroc"]
        llama_delta  = llama_nl - llama_fisher

        if llama_delta > RECOVERY_THRESHOLD:
            llama_verdict = "DISTRIBUTED_STRUCTURE"
            llama_note    = (f"MLP ({llama_nl:.4f}) >> Fisher ({llama_fisher:.4f}) by {llama_delta:+.4f}. "
                             f"Llama has rich nonlinear epistemic structure. "
                             f"Weak answer_jump (0.101) reflects LINEAR inaccessibility, not organizational weakness. "
                             f"Backbone-stratified magnitude interpretation changes: Llama has comparable Level-1, "
                             f"weaker Level-2 linear surface.")
        elif llama_delta < NORECOVERY_THRESHOLD:
            llama_verdict = "GENUINELY_WEAK"
            llama_note    = (f"MLP ({llama_nl:.4f}) ≈ Fisher ({llama_fisher:.4f}). "
                             f"Llama genuinely has weaker Level-1 epistemic organization. "
                             f"Weak answer_jump (0.101) reflects genuine organizational weakness, "
                             f"not just linear inaccessibility. Backbone-stratified hypothesis confirmed at Level-1.")
        else:
            llama_verdict = "AMBIGUOUS"
            llama_note    = (f"Partial nonlinear recovery ({llama_delta:+.4f}). Inconclusive.")

        interp["cross_model_analysis"]["llama_structural"] = {
            "fisher_auroc":       llama_fisher,
            "best_nl_auroc":      llama_nl,
            "nonlinear_delta":    round(llama_delta, 4),
            "answer_jump_prior":  0.101,
            "commit_rate_prior":  0.10,
            "verdict":            llama_verdict,
            "note":               llama_note,
        }

    # ── Overall thesis verdict ────────────────────────────────────────────────
    c3_primary = interp["cross_model_analysis"].get(
        "qwen_base_vs_instruct", {}).get("verdict", "INCOMPLETE")

    if c3_primary == "C3_RECOVERY":
        thesis_verdict = "THESIS_SUPPORTED"
        thesis_note    = ("Information ≠ Accessibility has direct evidence. "
                          "RLHF closes the linear accessibility window without destroying the information. "
                          "Four-level hierarchy Level 1→2 distinction is empirically real.")
    elif c3_primary == "C3_NO_RECOVERY":
        thesis_verdict = "THESIS_REQUIRES_REVISION"
        thesis_note    = ("RLHF attenuates both linear and nonlinear accessibility equally. "
                          "The 'information survives' framing may be incorrect. "
                          "Revise to: 'RLHF partially degrades epistemic information content.' "
                          "Four-level hierarchy collapses Levels 1-2 for RLHF-trained models.")
    elif c3_primary == "C3_PARTIAL":
        thesis_verdict = "THESIS_UNCERTAIN"
        thesis_note    = ("Partial recovery observed. Run with N_CAL >= 300 for definitive result.")
    else:
        thesis_verdict = "INCOMPLETE"
        thesis_note    = "Not all models completed."

    interp["overall_thesis_verdict"] = thesis_verdict
    interp["overall_thesis_note"]    = thesis_note

    return interp


def main():
    t_start = time.time()
    print(f"NONLINEAR PROBE RECOVERY TEST (C3)  |  "
          f"{time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Existential question: Does information survive RLHF in nonlinear form?")
    print(f"Recovery threshold: {RECOVERY_THRESHOLD:.2f}  |  "
          f"No-recovery threshold: {NORECOVERY_THRESHOLD:.2f}")
    print(f"N_CAL={N_CAL}/class  TRAIN_FRAC={TRAIN_FRAC}  N_BOOTSTRAP={N_BOOTSTRAP}")

    data_pool = load_data(n=800)

    all_results: Dict = {
        "config": {
            "n_cal": N_CAL,
            "train_frac": TRAIN_FRAC,
            "n_bootstrap": N_BOOTSTRAP,
            "recovery_threshold": RECOVERY_THRESHOLD,
            "norecovery_threshold": NORECOVERY_THRESHOLD,
            "max_gen": MAX_GEN,
            "seed": SEED,
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

        # Save after each model
        RESULTS_FILE.write_text(json.dumps(all_results, indent=2, default=str))
        print(f"\n[Saved] {RESULTS_FILE}")

    # ── Cross-model interpretation ────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"CROSS-MODEL INTERPRETATION")
    print(f"{'='*70}")
    interpretation = interpret_cross_model(model_results)
    all_results["interpretation"] = interpretation
    all_results["status"] = "complete"
    all_results["elapsed_s"] = round(time.time() - t_start)

    # ── Final summary print ───────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"C3 FINAL VERDICT: {interpretation.get('overall_thesis_verdict', 'INCOMPLETE')}")
    print(f"{'='*70}")
    print(interpretation.get("overall_thesis_note", ""))

    qwen_cross = interpretation.get("cross_model_analysis", {}).get("qwen_base_vs_instruct", {})
    if qwen_cross:
        print(f"\nQwen base vs instruct:")
        print(f"  Fisher: {qwen_cross.get('fisher_base', '?'):.4f} → "
              f"{qwen_cross.get('fisher_instruct', '?'):.4f}  "
              f"(Δ={qwen_cross.get('fisher_delta', 0):+.4f},  prior: -0.036)")
        print(f"  Best NL: {qwen_cross.get('best_nl_base', '?'):.4f} → "
              f"{qwen_cross.get('best_nl_instruct', '?'):.4f}")
        print(f"  Recovery advantage: {qwen_cross.get('recovery_advantage', 0):+.4f}")
        print(f"  C3 verdict: {qwen_cross.get('verdict', '?')}")

    llama_cross = interpretation.get("cross_model_analysis", {}).get("llama_structural", {})
    if llama_cross:
        print(f"\nLlama structural question:")
        print(f"  Fisher AUROC: {llama_cross.get('fisher_auroc', '?'):.4f}")
        print(f"  Best NL AUROC: {llama_cross.get('best_nl_auroc', '?'):.4f}")
        print(f"  Nonlinear delta: {llama_cross.get('nonlinear_delta', 0):+.4f}")
        print(f"  Verdict: {llama_cross.get('verdict', '?')}")

    RESULTS_FILE.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\n[Final saved] {RESULTS_FILE}")
    print(f"Total elapsed: {time.time() - t_start:.0f}s")
    print(json.dumps(all_results, indent=2, default=str))


if __name__ == "__main__":
    main()
