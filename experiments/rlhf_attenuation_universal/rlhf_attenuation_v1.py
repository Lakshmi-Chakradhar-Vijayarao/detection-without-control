"""
kaggle/rlhf_attenuation_universal/rlhf_attenuation_v1.py

RLHF ATTENUATION — UNIVERSALITY TEST
======================================

Central hypothesis: RLHF instruction tuning attenuates residual-stream
epistemic geometry by a UNIVERSAL constant (Δ ≈ 0.036 AUROC) independent of:
  - model family (Qwen, Llama, Gemma, Phi, Mistral)
  - organization (Alibaba, Meta, Google, Microsoft, Mistral AI)
  - training recipe (different RLHF implementations)

Prior confirmations:
  Llama-3.2-3B base (0.665) → instruct (0.629): Δ = -0.036
  Qwen2.5-1.5B base (0.899) → instruct (0.864): Δ = -0.036

This experiment tests THREE new families:
  Gemma-2-2B base vs Gemma-2-2B-it    (Google, MQA architecture)
  Phi-3-mini-4k base vs instruct       (Microsoft, SWA architecture)
  Mistral-7B v0.1 vs Mistral-7B-Instruct-v0.3  (Mistral AI, SWA)

If all three show Δ ≈ 0.03–0.04:
  → RLHF universally attenuates epistemic geometry across transformer families.
  → This is a law-like property of post-training alignment.
  → The paper becomes an alignment science paper, not an interpretability paper.

If one or more show Δ ≠ 0.036:
  → Architecture dependence (MQA vs GQA vs SWA may explain variation)
  → Still publishable: "architecture moderates RLHF attenuation magnitude"

Protocol:
  - Bilateral oracle calibration on TriviaQA (n=30 PARAM + n=30 WRONG, unilateral)
  - Extract gen-step-1 hidden states at probe layer (n_layers-2)
  - Fisher LDA, 3-fold CV AUROC
  - Report: base_auroc, instruct_auroc, delta, cosim(base_dir, instruct_dir)
  - Save all results + intermediate per family

GPU: T4 (sm_75). All models run at 4-bit NF4 for VRAM budget.
Expected time: ~45-60 min per family = ~3h total for all three.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Force flush on all prints — prevents silent buffering on Kaggle T4
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
        print("HF login: OK", flush=True)
    else:
        print("WARNING: HF_TOKEN not set.", flush=True)
except Exception as _e:
    print(f"HF login error: {_e}", flush=True)

# ── GPU check ─────────────────────────────────────────────────────────────────
assert torch.cuda.is_available(), "GPU required"
_sm_major, _sm_minor = torch.cuda.get_device_capability(0)
_sm = _sm_major * 10 + _sm_minor
assert _sm >= 70, f"GPU sm_{_sm} not supported."
DEVICE = "cuda"
print(f"GPU: {torch.cuda.get_device_name(0)}  sm_{_sm}", flush=True)
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", flush=True)

# ── Constants ─────────────────────────────────────────────────────────────────
N_CAL      = 30     # samples per class (PARAM + WRONG) per model
MAX_GEN    = 150    # max tokens for calibration answer generation
SEED       = 42
OUT_DIR    = Path("/kaggle/working")
RESULTS    = OUT_DIR / "rlhf_attenuation_v1_results.json"

rng = np.random.default_rng(SEED)

# ── Kaggle model hub paths (avoid HF Hub throttling) ─────────────────────────
_KG_GEMMA_BASE = "/kaggle/input/gemma-2/transformers/gemma-2-2b/2"
_KG_MISTRAL    = "/kaggle/input/mistral/transformers/7b-v0.1/1"

# ── Model pairs ───────────────────────────────────────────────────────────────
# Each pair: (family_name, base_model_id, instruct_model_id, architecture)
MODEL_PAIRS = [
    {
        "family":    "gemma2",
        "base":      _KG_GEMMA_BASE if os.path.exists(_KG_GEMMA_BASE) else "google/gemma-2-2b",
        "instruct":  "google/gemma-2-2b-it",
        "arch":      "MQA",
        "prior":     None,
        "note":      "Google, MQA architecture — different from confirmed GQA families",
    },
    {
        "family":    "olmo",
        "base":      "allenai/OLMo-7B-hf",
        "instruct":  "allenai/OLMo-7B-Instruct-hf",
        "arch":      "GQA",
        "prior":     None,
        "note":      "AllenAI, GQA architecture — open-data training, clean base/instruct pair",
    },
    {
        "family":    "mistral",
        "base":      _KG_MISTRAL if os.path.exists(_KG_MISTRAL) else "mistralai/Mistral-7B-v0.1",
        "instruct":  "mistralai/Mistral-7B-Instruct-v0.3",
        "arch":      "SWA",
        "prior":     None,
        "note":      "Mistral AI, SWA architecture — v0.1 base vs v0.3 instruct",
    },
]

# Prior confirmed results (from main experiment series)
CONFIRMED_PRIOR = {
    "llama3": {"base": 0.665, "instruct": 0.629, "delta": -0.036, "arch": "GQA"},
    "qwen25": {"base": 0.899, "instruct": 0.864, "delta": -0.036, "arch": "GQA"},
}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data(n: int = 300) -> List[Dict]:
    from datasets import load_dataset
    ds = load_dataset("trivia_qa", "rc.wikipedia", split="train", streaming=True)
    pool = []
    for row in ds:
        if not row["answer"]["aliases"]:
            continue
        pool.append({"question": row["question"], "answers": row["answer"]["aliases"]})
        if len(pool) >= n:
            break
    print(f"Loaded {len(pool)} TriviaQA questions", flush=True)
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
    print(f"\nLoading {model_id} …", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=cfg,
        device_map=None, trust_remote_code=True,
    ).to(DEVICE).eval()

    n_layers = mdl.config.num_hidden_layers
    hidden   = mdl.config.hidden_size
    print(f"  {model_id}: n_layers={n_layers}  hidden={hidden}", flush=True)
    return mdl, tok


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

    captured: list = [None]

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
# Calibration and AUROC
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelResult:
    model_id:    str
    variant:     str  # "base" or "instruct"
    arch:        str
    layer_idx:   int
    n_param:     int
    n_wrong:     int
    auroc:       float
    probe_dir:   np.ndarray


def run_single_model(model_id: str, variant: str, arch: str,
                     data_pool: List[Dict]) -> Optional[ModelResult]:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    print(f"\n{'─'*60}", flush=True)
    print(f"  {variant.upper()}: {model_id}", flush=True)
    t0 = time.time()

    try:
        mdl, tok = load_model(model_id)
    except Exception as e:
        print(f"  [ERROR] Failed to load: {e}", flush=True)
        return None

    n_layers  = mdl.config.num_hidden_layers
    layer_idx = max(0, n_layers - 2)
    print(f"  Probe layer: {layer_idx}", flush=True)

    param_hs: List[np.ndarray] = []
    wrong_hs:  List[np.ndarray] = []
    shuffled   = list(data_pool)
    np.random.shuffle(shuffled)

    for i, s in enumerate(shuffled):
        if len(param_hs) >= N_CAL and len(wrong_hs) >= N_CAL:
            break
        elapsed = time.time() - t0
        if i % 5 == 0:
            print(f"    [{i}/{len(shuffled)}] PARAM={len(param_hs)}  WRONG={len(wrong_hs)}"
                  f"  elapsed={elapsed:.0f}s", flush=True)
        if elapsed > 2400:
            print(f"    TIMEOUT at {elapsed:.0f}s", flush=True)
            break

        gen = generate_answer(mdl, tok, s["question"])
        is_correct = (answer_contains(gen, s["answers"]) or
                      token_f1(gen, s["answers"]) >= 0.4)

        if is_correct and len(param_hs) >= N_CAL:
            continue
        if not is_correct and len(wrong_hs) >= N_CAL:
            continue

        hs = get_step1_hs(mdl, tok, s["question"], layer_idx)
        if hs is None:
            continue

        if is_correct:
            param_hs.append(hs)
        else:
            wrong_hs.append(hs)

    n_p, n_w = len(param_hs), len(wrong_hs)
    print(f"  Calibration complete: PARAM={n_p}  WRONG={n_w}", flush=True)

    if n_p < 5 or n_w < 5:
        print(f"  [SKIP] Insufficient data.", flush=True)
        del mdl
        gc.collect()
        torch.cuda.empty_cache()
        return None

    X = np.stack(param_hs + wrong_hs)
    y = np.array([1] * n_p + [0] * n_w)
    lda = LinearDiscriminantAnalysis(n_components=1)

    n_min = min(n_p, n_w)
    folds = min(3, n_min)
    try:
        cv_auroc = float(np.mean(cross_val_score(
            lda, X, y,
            cv=StratifiedKFold(folds, shuffle=True, random_state=SEED),
            scoring="roc_auc"
        )))
        print(f"  CV AUROC ({folds}-fold): {cv_auroc:.4f}", flush=True)
    except Exception as e:
        print(f"  [WARN] CV failed: {e}. Using in-sample.", flush=True)
        lda.fit(X, y)
        cv_auroc = float(roc_auc_score(y, lda.decision_function(X)))

    lda.fit(X, y)
    d = lda.coef_[0] / (np.linalg.norm(lda.coef_[0]) + 1e-9)

    result = ModelResult(
        model_id  = model_id,
        variant   = variant,
        arch      = arch,
        layer_idx = layer_idx,
        n_param   = n_p,
        n_wrong   = n_w,
        auroc     = round(cv_auroc, 4),
        probe_dir = d,
    )
    print(f"  → AUROC {variant}: {cv_auroc:.4f}  (elapsed {time.time()-t0:.0f}s)",
          flush=True)

    del mdl
    gc.collect()
    torch.cuda.empty_cache()
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Per-family analysis
# ─────────────────────────────────────────────────────────────────────────────

def run_family(pair: Dict, data_pool: List[Dict]) -> Dict:
    family  = pair["family"]
    arch    = pair["arch"]
    print(f"\n{'='*60}", flush=True)
    print(f"FAMILY: {family.upper()}  ({arch})", flush=True)
    print(f"  Base:     {pair['base']}", flush=True)
    print(f"  Instruct: {pair['instruct']}", flush=True)
    print(f"{'='*60}", flush=True)

    base_result     = run_single_model(pair["base"],     "base",     arch, data_pool)
    instruct_result = run_single_model(pair["instruct"], "instruct", arch, data_pool)

    result: Dict = {
        "family":   family,
        "arch":     arch,
        "base_id":  pair["base"],
        "inst_id":  pair["instruct"],
        "note":     pair["note"],
    }

    if base_result is None or instruct_result is None:
        result["status"] = "INCOMPLETE"
        result["base_auroc"] = base_result.auroc if base_result else None
        result["inst_auroc"] = instruct_result.auroc if instruct_result else None
        result["delta"] = None
        return result

    delta  = instruct_result.auroc - base_result.auroc
    cosim  = float(np.dot(base_result.probe_dir, instruct_result.probe_dir))

    result.update({
        "status":      "COMPLETE",
        "base_auroc":  base_result.auroc,
        "inst_auroc":  instruct_result.auroc,
        "delta":       round(delta, 4),
        "cosim_dirs":  round(cosim, 4),
        "n_base_param": base_result.n_param,
        "n_base_wrong": base_result.n_wrong,
        "n_inst_param": instruct_result.n_param,
        "n_inst_wrong": instruct_result.n_wrong,
    })

    # Verdict: does this match the Δ≈0.036 law?
    delta_abs = abs(delta)
    if 0.02 <= delta_abs <= 0.06:
        result["rlhf_law"] = "CONFIRMED"
        result["verdict"]  = f"Δ={delta:.3f} — within [−0.02, −0.06] range → RLHF law holds"
    elif delta < -0.06:
        result["rlhf_law"] = "STRONGER_ATTENUATION"
        result["verdict"]  = f"Δ={delta:.3f} — stronger than GQA baseline"
    elif delta_abs < 0.02:
        result["rlhf_law"] = "WEAKER_ATTENUATION"
        result["verdict"]  = f"Δ={delta:.3f} — weaker attenuation, may reflect architecture"
    else:
        result["rlhf_law"] = "UNEXPECTED"
        result["verdict"]  = f"Δ={delta:.3f} — unexpected direction"

    print(f"\n  RESULT {family.upper()}: base={base_result.auroc:.4f}  "
          f"instruct={instruct_result.auroc:.4f}  Δ={delta:+.4f}", flush=True)
    print(f"  cosim(probe_base, probe_instruct) = {cosim:.4f}", flush=True)
    print(f"  {result['verdict']}", flush=True)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    print(f"RLHF Attenuation Universality Test v1  |  "
          f"{time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"Hypothesis: Δ ≈ 0.036 AUROC across all RLHF-tuned families\n", flush=True)

    data_pool = load_data(n=300)

    all_results: Dict = {
        "confirmed_prior": CONFIRMED_PRIOR,
        "new_families": {},
        "config": {"n_cal": N_CAL, "max_gen": MAX_GEN},
    }

    for pair in MODEL_PAIRS:
        try:
            fam_result = run_family(pair, data_pool)
        except Exception as e:
            print(f"\n[ERROR] Family {pair['family']}: {e}", flush=True)
            import traceback; traceback.print_exc()
            fam_result = {"family": pair["family"], "status": "ERROR", "error": str(e)}

        all_results["new_families"][pair["family"]] = fam_result

        # Save after each family
        all_results["elapsed_s"] = round(time.time() - t_start)
        RESULTS.write_text(json.dumps(all_results, indent=2))
        print(f"\n[Saved] {RESULTS}", flush=True)

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print(f"FINAL SUMMARY — RLHF Attenuation Universality", flush=True)
    print(f"{'='*60}", flush=True)

    print(f"\nPRIOR (confirmed):")
    for fam, r in CONFIRMED_PRIOR.items():
        print(f"  {fam:<12}  base={r['base']:.3f}  inst={r['instruct']:.3f}  "
              f"Δ={r['delta']:+.3f}  ({r['arch']})", flush=True)

    print(f"\nNEW FAMILIES:")
    all_deltas = [v["delta"] for v in all_results["new_families"].values()
                  if v.get("delta") is not None]
    for fam, r in all_results["new_families"].items():
        if r.get("status") == "COMPLETE":
            print(f"  {fam:<12}  base={r['base_auroc']:.3f}  inst={r['inst_auroc']:.3f}  "
                  f"Δ={r['delta']:+.3f}  ({r['arch']})  {r['rlhf_law']}", flush=True)
        else:
            print(f"  {fam:<12}  STATUS={r.get('status')}  "
                  f"error={r.get('error', 'N/A')}", flush=True)

    if all_deltas:
        all_deltas_combined = list(CONFIRMED_PRIOR.values()) + [
            {"delta": d} for d in all_deltas
        ]
        all_delta_vals = [v["delta"] for v in all_deltas_combined if "delta" in v]
        mean_delta = np.mean(all_delta_vals)
        std_delta  = np.std(all_delta_vals)
        print(f"\n  Combined Δ: mean={mean_delta:+.4f}  std={std_delta:.4f}  "
              f"n_families={len(all_delta_vals)}", flush=True)

        if std_delta < 0.01:
            print(f"\n  VERDICT: UNIVERSAL LAW — Δ is consistent across {len(all_delta_vals)} "
                  f"independent families.", flush=True)
        elif std_delta < 0.02:
            print(f"\n  VERDICT: NEAR-UNIVERSAL — small variation, likely architecture-dependent.",
                  flush=True)
        else:
            print(f"\n  VERDICT: ARCHITECTURE-DEPENDENT — Δ varies across families.",
                  flush=True)

    all_results["elapsed_s"] = round(time.time() - t_start)
    all_results["status"]    = "complete"
    RESULTS.write_text(json.dumps(all_results, indent=2))
    print(f"\n[Final] {RESULTS}", flush=True)
    print(json.dumps(all_results, indent=2), flush=True)


if __name__ == "__main__":
    main()
