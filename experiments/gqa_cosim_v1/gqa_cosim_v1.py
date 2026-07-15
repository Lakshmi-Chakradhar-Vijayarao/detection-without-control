"""
GQA cosim measurement v2 (Program B — Mechanism A validation).

v2 fixes vs v1:
  - Instruct model uses chat template (apply_chat_template) for both F1 eval
    and step1_hidden extraction. Raw completion prompt caused verbose outputs
    → ~2% PARAM hit rate → experiment hung. Chat template gives terse answers
    → ~40-50% PARAM hit rate, matching base model behavior.
  - Added MAX_SCAN=2000 per model as safety valve against silent hangs.
  - Scientific justification: each model evaluated in its natural operating
    mode (base=completion, instruct=chat). This is the correct comparison.

Protocol:
  1. Load Qwen2.5-7B (base) — bilateral oracle on TriviaQA — fit Fisher LDA
     — save probe direction coef_[0]
  2. Unload base, load Qwen2.5-7B-Instruct — same protocol with chat template
     — save probe dir
  3. Compute cosim(base_dir, instruct_dir)
  4. Report: base AUROC, instruct AUROC, Δ, cosim → verdict

Verdict:
  cosim > 0.70 → ATTENUATION_CONFIRMED  (Mechanism A valid for Qwen)
  0.35–0.70    → ROTATION_PARTIAL        (partial rotation, taxonomy needs revision)
  < 0.35       → ROTATION_CONFIRMED     (Mechanism A invalid — revision required)
"""

import subprocess
print("[init] pip install bitsandbytes...", flush=True)
subprocess.run(["pip", "install", "-q", "-U", "bitsandbytes>=0.46.1"], check=True)
print("[init] done.", flush=True)

import os, sys, gc, json, time
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

# Force flush on all prints (avoids silent buffering on Kaggle)
import functools, builtins
builtins.print = functools.partial(builtins.print, flush=True)

# ── Config ────────────────────────────────────────────────────────────────────
_KG_BASE     = "/kaggle/input/qwen2.5/transformers/7b/1"
_KG_INSTRUCT = "/kaggle/input/qwen2.5/transformers/7b-instruct/1"
BASE_MODEL_ID     = _KG_BASE     if os.path.exists(_KG_BASE)     else "Qwen/Qwen2.5-7B"
INSTRUCT_MODEL_ID = _KG_INSTRUCT if os.path.exists(_KG_INSTRUCT) else "Qwen/Qwen2.5-7B-Instruct"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
PROBE_LAYER  = 26
N_CAL        = 50   # per class, same as all other runs
N_CAL_SEEDS  = 3    # for probe stability check
MAX_GEN_TOK  = 64
MAX_SCAN     = 2000  # safety valve: abort if dataset exhausted before N_CAL
SEED         = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

print(f"[0s] Base:        {BASE_MODEL_ID}")
print(f"[0s] Instruct:    {INSTRUCT_MODEL_ID}")
print(f"[0s] Device:      {DEVICE}  Layer={PROBE_LAYER}  N_CAL={N_CAL}")

# ── Helpers ───────────────────────────────────────────────────────────────────
def make_prompt(tok, question: str, use_chat_template: bool) -> str:
    if use_chat_template:
        return tok.apply_chat_template(
            [{"role": "user", "content": f"Answer in one word or short phrase: {question}"}],
            tokenize=False, add_generation_prompt=True
        )
    return f"Answer briefly: {question}\nAnswer:"


def get_step1_hidden(model, tok, prompt: str):
    inputs = tok(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**inputs)
    h = out.hidden_states[PROBE_LAYER][0, -1].float().cpu().numpy()
    return h


def eval_nocontext_f1(model, tok, question: str, answer: str,
                      use_chat_template: bool) -> float:
    prompt = make_prompt(tok, question, use_chat_template)
    inputs = tok(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        ids = model.generate(
            **inputs, max_new_tokens=MAX_GEN_TOK,
            do_sample=False, temperature=1.0, top_p=1.0,
            pad_token_id=tok.eos_token_id
        )
    gen = tok.decode(ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    pred = set(gen.lower().split())
    gold = set(str(answer).lower().split())
    if not gold:
        return 0.0
    tp = len(pred & gold)
    if tp == 0:
        return 0.0
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def oracle_label(f1_nocontext: float) -> str:
    if f1_nocontext >= 0.50:
        return "PARAM"
    if f1_nocontext <= 0.05:
        return "CTX_DEP"
    return "SKIP"


def run_model(model_id: str, ds, label: str, use_chat_template: bool = False):
    """
    Load model, collect N_CAL PARAM + N_CAL CTX_DEP samples,
    fit Fisher LDA (N_CAL_SEEDS seeds for stability), return results dict.
    use_chat_template=True for instruct models (prevents verbose outputs
    that would cause near-zero PARAM hit rate on raw completion prompts).
    """
    print(f"\n{'='*60}")
    print(f"  Running: {label}  ({model_id})")
    print(f"  chat_template: {use_chat_template}")
    print(f"{'='*60}")

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb, device_map="auto",
        output_hidden_states=True
    )
    model.eval()
    print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    param_h, ctx_h = [], []
    t0 = time.time()

    for i, row in enumerate(ds):
        if len(param_h) >= N_CAL and len(ctx_h) >= N_CAL:
            break
        if i >= MAX_SCAN:
            print(f"  MAX_SCAN={MAX_SCAN} reached. PARAM={len(param_h)} CTX_DEP={len(ctx_h)}")
            break
        q = row["question"]
        a = row["answer"]["value"]
        f1 = eval_nocontext_f1(model, tok, q, a, use_chat_template)
        lbl = oracle_label(f1)
        if lbl == "SKIP":
            continue
        if lbl == "PARAM" and len(param_h) >= N_CAL:
            continue
        if lbl == "CTX_DEP" and len(ctx_h) >= N_CAL:
            continue

        prompt = make_prompt(tok, q, use_chat_template)
        h = get_step1_hidden(model, tok, prompt)

        if lbl == "PARAM":
            param_h.append(h)
        else:
            ctx_h.append(h)

        if (i + 1) % 20 == 0:
            print(f"    [{i+1}] PARAM={len(param_h)} CTX_DEP={len(ctx_h)} elapsed={time.time()-t0:.0f}s")

    n_p, n_c = len(param_h), len(ctx_h)
    print(f"  Collected: PARAM={n_p} CTX_DEP={n_c}")

    if n_p < N_CAL or n_c < N_CAL:
        print(f"  INSUFFICIENT samples. Exiting.")
        del model
        gc.collect()
        torch.cuda.empty_cache()
        return None

    # ── Fit Fisher LDA (N_CAL_SEEDS seeds for probe stability) ───────────────
    X_cal = np.array(param_h + ctx_h)
    y_cal = np.array([1]*n_p + [0]*n_c)

    probe_dirs = []
    aurocs = []
    for seed_i in range(N_CAL_SEEDS):
        rng = np.random.RandomState(seed_i)
        idx = rng.choice(n_p, N_CAL, replace=False)
        X_seed = np.concatenate([X_cal[idx], X_cal[n_p + rng.choice(n_c, N_CAL, replace=False)]])
        y_seed = np.array([1]*N_CAL + [0]*N_CAL)
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        lda.fit(X_seed, y_seed)
        scores = lda.decision_function(X_cal)
        auroc = roc_auc_score(y_cal, scores)
        probe_dirs.append(lda.coef_[0])
        aurocs.append(auroc)

    # probe stability: cosim between seed 0 and 1
    d0 = probe_dirs[0] / (np.linalg.norm(probe_dirs[0]) + 1e-8)
    d1 = probe_dirs[1] / (np.linalg.norm(probe_dirs[1]) + 1e-8)
    probe_stability = float(np.dot(d0, d1))

    # best direction (seed with highest AUROC)
    best_seed = int(np.argmax(aurocs))
    best_dir = probe_dirs[best_seed]

    result = {
        "model_id": model_id,
        "label": label,
        "n_param": n_p,
        "n_ctx": n_c,
        "auroc_mean": float(np.mean(aurocs)),
        "auroc_std": float(np.std(aurocs)),
        "probe_stability_cosim": probe_stability,
        "probe_dir": best_dir.tolist(),  # save direction for cosim computation
        "elapsed_s": int(time.time() - t0)
    }

    print(f"  AUROC: {result['auroc_mean']:.4f} ± {result['auroc_std']:.4f}")
    print(f"  Probe stability: {probe_stability:.4f}")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  Model unloaded. VRAM freed.")

    return result


# ── Load dataset once ─────────────────────────────────────────────────────────
print("\n  Loading TriviaQA...")
ds_full = load_dataset("trivia_qa", "rc.wikipedia", split="validation")
# shuffle with seed so both models see same order (same samples → fair comparison)
ds_full = ds_full.shuffle(seed=SEED)

# ── Run base model ────────────────────────────────────────────────────────────
base_result = run_model(BASE_MODEL_ID, ds_full, "BASE", use_chat_template=False)
if base_result is None:
    print("BASE model failed. Exiting.")
    sys.exit(1)

# ── Run instruct model ────────────────────────────────────────────────────────
# Re-shuffle with same seed so instruct model sees the same question order.
# use_chat_template=True: instruct model needs chat format to produce terse
# answers that score F1>=0.50 and yield adequate PARAM hit rate.
ds_full = ds_full.shuffle(seed=SEED)
instruct_result = run_model(INSTRUCT_MODEL_ID, ds_full, "INSTRUCT", use_chat_template=True)
if instruct_result is None:
    print("INSTRUCT model failed. Exiting.")
    sys.exit(1)

# ── Compute cosim between probe directions ────────────────────────────────────
base_dir    = np.array(base_result["probe_dir"])
instruct_dir = np.array(instruct_result["probe_dir"])

base_dir_n    = base_dir    / (np.linalg.norm(base_dir)    + 1e-8)
instruct_dir_n = instruct_dir / (np.linalg.norm(instruct_dir) + 1e-8)
cosim = float(np.dot(base_dir_n, instruct_dir_n))

base_auroc    = base_result["auroc_mean"]
instruct_auroc = instruct_result["auroc_mean"]
delta = instruct_auroc - base_auroc

# ── Verdict ───────────────────────────────────────────────────────────────────
if cosim > 0.70:
    verdict = "ATTENUATION_CONFIRMED"    # Mechanism A holds for Qwen GQA
elif cosim > 0.35:
    verdict = "ROTATION_PARTIAL"         # taxonomy needs nuancing
else:
    verdict = "ROTATION_CONFIRMED"       # Mechanism A invalid — paper must be revised

print("\n" + "="*60)
print("  GQA COSIM RESULTS")
print("="*60)
print(f"  Base AUROC:     {base_auroc:.4f}")
print(f"  Instruct AUROC: {instruct_auroc:.4f}")
print(f"  Delta:          {delta:+.4f}")
print(f"  cosim(base_probe, instruct_probe): {cosim:.4f}")
print(f"  Verdict: {verdict}")

# ── Save ──────────────────────────────────────────────────────────────────────
results = {
    "model_base": BASE_MODEL_ID,
    "model_instruct": INSTRUCT_MODEL_ID,
    "probe_layer": PROBE_LAYER,
    "n_cal": N_CAL,
    "n_cal_seeds": N_CAL_SEEDS,
    "base_auroc": base_auroc,
    "base_auroc_std": base_result["auroc_std"],
    "base_probe_stability": base_result["probe_stability_cosim"],
    "instruct_auroc": instruct_auroc,
    "instruct_auroc_std": instruct_result["auroc_std"],
    "instruct_probe_stability": instruct_result["probe_stability_cosim"],
    "delta": delta,
    "cosim_probe_directions": cosim,
    "verdict": verdict,
    "interpretation": (
        "ATTENUATION_CONFIRMED: RLHF preserves epistemic probe direction (cosim>0.70). "
        "Signal amplitude weakens (Δ negative) but geometry is maintained. Mechanism A validated."
        if verdict == "ATTENUATION_CONFIRMED" else
        "ROTATION_CONFIRMED: RLHF rotates probe direction near-orthogonally (cosim<0.35). "
        "Mechanism A is INVALID for Qwen — paper taxonomy must be revised."
        if verdict == "ROTATION_CONFIRMED" else
        "ROTATION_PARTIAL: partial rotation. Mechanism A holds weakly. Taxonomy needs nuancing."
    )
}

# Remove raw probe directions from saved output (too large)
out_path = "/kaggle/working/gqa_cosim_v1_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n[Final] {out_path}")
print(json.dumps(results, indent=2))
