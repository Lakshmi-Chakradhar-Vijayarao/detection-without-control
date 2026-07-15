"""
NQ-Open Blind Prediction — Law 1 Dataset Generalization
========================================================

PRE-REGISTERED PREDICTION (written 2026-07-13, before any NQ-Open results seen):
  Fisher+PCA64 bilateral oracle on Natural Questions Open achieves
  L1 AUROC >= 0.70 for Qwen2.5-1.5B-Instruct at L26 step-1,
  N >= 150/class, clean shuffled control.

  Kill criterion: AUROC < 0.65 at N >= 150/class.
  Expected range: 0.68 - 0.82.

This is a BLIND TEST of Law 1's dataset generalization claim.
The probe is retrained on NQ items — no TriviaQA data used.

NOTE ON NQ-OPEN PROTOCOL:
NQ-Open has no context passages, so we cannot run bilateral oracle (PARAM vs CTX_DEP).
Instead we use the standard L1 framing directly:
  PARAM  = model gets the answer correct (max F1 >= 0.50)
  CONFAB = model confidently wrong (max F1 < 0.15, non-abstention)
This is the direct L1 labeling used for all no-context evaluations.
The probe should separate parameterized knowledge from confabulation.
"""

from __future__ import annotations
import subprocess
print("[init] installing bitsandbytes...", flush=True)
subprocess.run(["pip", "install", "-q", "-U", "bitsandbytes>=0.46.1"], check=False)

import os, json, time, string
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import functools, builtins
builtins.print = functools.partial(builtins.print, flush=True)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID    = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
PROBE_LAYER = 26
N_TARGET      = 100    # per class (reduced from 200 to stay within 12h T4 budget)
MAX_SCAN      = 3600   # NQ-Open validation set size
N_PCA         = 64
SEED          = 42
F1_PARAM      = 0.50   # correct answer threshold
F1_CONFAB_MAX = 0.15   # max F1 to count as "wrong" (confabulation)
MAX_NEW_TOKENS = 40    # reduced from 60 — NQ answers are short, saves ~5s/item
N_BOOTSTRAP    = 1000
CHECKPOINT_INTERVAL = 200   # save partial results every N items
OUTPUT_FILE    = "/kaggle/working/nq_blind_prediction_v1.json"
CKPT_FILE      = "/kaggle/working/nq_blind_prediction_v1_ckpt.json"

np.random.seed(SEED)
torch.manual_seed(SEED)
T0 = time.time()
def ts(): return f"[{int(time.time()-T0):5d}s]"

print(f"{ts()} === NQ-OPEN BLIND PREDICTION v1 ===")
print(f"{ts()} PRE-REGISTERED: Law 1 AUROC >= 0.70 on Natural Questions Open")
print(f"{ts()} Kill criterion: AUROC < 0.65")
print(f"{ts()} Model={MODEL_ID}  Layer={PROBE_LAYER}  N_TARGET={N_TARGET}")
print(f"{ts()} Device={DEVICE}")

# ── F1 helpers ────────────────────────────────────────────────────────────────
def normalize(s: str) -> str:
    s = s.lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())

def token_f1(pred: str, gold: str) -> float:
    p_toks = normalize(pred).split()
    g_toks = normalize(gold).split()
    if not p_toks or not g_toks:
        return float(p_toks == g_toks)
    common = sum(min(p_toks.count(t), g_toks.count(t)) for t in set(p_toks) & set(g_toks))
    if common == 0:
        return 0.0
    prec = common / len(p_toks)
    rec  = common / len(g_toks)
    return 2 * prec * rec / (prec + rec)

def max_f1(pred: str, answers: list[str]) -> float:
    """NQ has multiple valid answers — take max F1 across all."""
    if not answers:
        return 0.0
    return max(token_f1(pred, a) for a in answers)

# ── Model ─────────────────────────────────────────────────────────────────────
def _get_hf_token():
    t = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if t and t.startswith("hf_"):
        return t
    try:
        from kaggle_secrets import UserSecretsClient
        t = UserSecretsClient().get_secret("HF_TOKEN")
        if t and t.startswith("hf_"):
            return t
    except Exception:
        pass
    return None

print(f"{ts()} Loading tokenizer...")
_hf_tok = _get_hf_token()
if _hf_tok:
    from huggingface_hub import login
    login(token=_hf_tok, add_to_git_credential=False)
    print(f"{ts()} HF login OK")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=_hf_tok, trust_remote_code=True)
print(f"{ts()} Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map=None,
    token=_hf_tok,
    trust_remote_code=True,
)
model = model.to(DEVICE)
model.eval()
print(f"{ts()} Model loaded. layers={model.config.num_hidden_layers}  d={model.config.hidden_size}")

# ── Dataset ───────────────────────────────────────────────────────────────────
print(f"{ts()} Loading NQ-Open validation set...")
nq = load_dataset("nq_open", split="validation", trust_remote_code=True)
print(f"{ts()} NQ-Open validation: {len(nq)} items")

def format_nocontext(question: str) -> str:
    return (f"<|im_start|>user\nAnswer the following question with a short phrase "
            f"(1-5 words). If you don't know, say 'I don't know'.\n\n"
            f"Question: {question}\n<|im_end|>\n<|im_start|>assistant\n")

def generate_and_extract(prompt: str, collect_hs: bool = False):
    """Generate answer; optionally collect hidden state at L26 step-1."""
    enc = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_len = enc.input_ids.shape[1]

    hs_vec = None
    if collect_hs:
        hook_storage = {}
        def make_hook(layer_idx):
            def hook(module, inp, out):
                if isinstance(out, tuple):
                    hs = out[0]
                else:
                    hs = out
                # With KV cache each decode step has shape[1]==1; prefill has shape[1]==input_len.
                # Capture only once at step-1 (first new token).
                if "hs" not in hook_storage and hs.shape[1] == 1:
                    hook_storage["hs"] = hs[:, -1, :].detach().float().cpu()
            return hook
        handle = model.model.layers[PROBE_LAYER].register_forward_hook(make_hook(PROBE_LAYER))

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=1.0,
            output_hidden_states=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    if collect_hs:
        handle.remove()
        hs_vec = hook_storage.get("hs", None)
        if hs_vec is not None:
            hs_vec = hs_vec.squeeze(0).numpy()

    new_tokens = out[0, input_len:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return answer, hs_vec

# ── Scan ──────────────────────────────────────────────────────────────────────
# L1 direct labeling: PARAM (F1 >= 0.50) vs CONFAB (F1 < 0.15, non-abstain)
# NQ-Open has no context, so we skip bilateral oracle and use single-pass labeling.
param_items  = []  # correct answers
confab_items = []  # confident wrong answers

n_scanned = 0
n_param_found = 0
n_confab_found = 0

print(f"{ts()} Starting L1 scan on NQ-Open (target {N_TARGET}/class)...")
print(f"{ts()} PARAM: F1 >= {F1_PARAM}  |  CONFAB: F1 < {F1_CONFAB_MAX} + non-abstain")
print(f"{ts()} MAX_SCAN={MAX_SCAN}  MAX_NEW_TOKENS={MAX_NEW_TOKENS}  CHECKPOINT_INTERVAL={CHECKPOINT_INTERVAL}")

t_scan_start = time.time()

for item in nq:
    if n_param_found >= N_TARGET and n_confab_found >= N_TARGET:
        break
    if n_scanned >= MAX_SCAN:
        break

    question = item["question"].strip()
    answers  = [a.strip() for a in item["answer"]]

    n_scanned += 1
    t_item = time.time()
    prompt = format_nocontext(question)
    answer, hs = generate_and_extract(prompt, collect_hs=True)
    f1 = max_f1(answer, answers)
    item_sec = time.time() - t_item

    # PARAM: model answers correctly
    if f1 >= F1_PARAM and n_param_found < N_TARGET:
        param_items.append({"hs": hs, "f1": f1, "question": question, "answer": answer})
        n_param_found += 1

    # CONFAB: model gives a wrong confident answer (not "I don't know")
    elif (f1 < F1_CONFAB_MAX
          and "don't know" not in answer.lower()
          and "i don't" not in answer.lower()
          and "unknown" not in answer.lower()
          and len(answer.split()) >= 2
          and n_confab_found < N_TARGET):
        confab_items.append({"hs": hs, "f1": f1, "question": question, "answer": answer})
        n_confab_found += 1

    # Dense logging: every 10 items
    if n_scanned % 10 == 0:
        elapsed = time.time() - t_scan_start
        rate = n_scanned / elapsed if elapsed > 0 else 0
        eta = (MAX_SCAN - n_scanned) / rate if rate > 0 else 0
        print(f"{ts()} [{n_scanned}/{MAX_SCAN}] PARAM={n_param_found}/{N_TARGET} "
              f"CONFAB={n_confab_found}/{N_TARGET} | {item_sec:.1f}s/item | "
              f"rate={rate:.2f}/s ETA={eta/60:.0f}min")

    # Checkpoint save every CHECKPOINT_INTERVAL items
    if n_scanned % CHECKPOINT_INTERVAL == 0:
        ckpt = {
            "phase": "scan_in_progress",
            "n_scanned": n_scanned, "n_param": n_param_found, "n_confab": n_confab_found,
            "elapsed_s": round(time.time() - t_scan_start, 1),
        }
        with open(CKPT_FILE, "w") as f:
            json.dump(ckpt, f)
        print(f"{ts()} CHECKPOINT saved: {n_scanned} items | PARAM={n_param_found} CONFAB={n_confab_found}")

print(f"{ts()} Scan complete: {n_scanned} scanned, PARAM={n_param_found}, CONFAB={n_confab_found}")

# ── Probe ─────────────────────────────────────────────────────────────────────
n_available = min(n_param_found, n_confab_found)
if n_available < 50:
    result = {
        "status": "PROTOCOL_FAILURE",
        "reason": f"Insufficient items: PARAM={n_param_found}, CONFAB={n_confab_found}",
        "n_scanned": n_scanned,
        "pre_registered_prediction": "AUROC >= 0.70",
        "verdict": "PROTOCOL_FAILURE",
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(result, f, indent=2)
    print(f"{ts()} PROTOCOL_FAILURE — too few items collected.")
    raise SystemExit(0)

# Balanced set
param_hs  = np.array([x["hs"] for x in param_items[:n_available]])
ctxdep_hs = np.array([x["hs"] for x in confab_items[:n_available]])
X = np.vstack([param_hs, ctxdep_hs])
y = np.array([1]*n_available + [0]*n_available)

# Train/test split
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.35, random_state=SEED, stratify=y
)
print(f"{ts()} Split: train={len(X_tr)} test={len(X_te)}")

# Fisher+PCA64
pca = PCA(n_components=N_PCA, random_state=SEED)
X_tr_pca = pca.fit_transform(X_tr)
X_te_pca = pca.transform(X_te)

lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
lda.fit(X_tr_pca, y_tr)
scores_real = lda.decision_function(X_te_pca)
auroc_real  = roc_auc_score(y_te, scores_real)

# Shuffled control
y_shuf = y_te.copy(); np.random.shuffle(y_shuf)
auroc_shuf = roc_auc_score(y_shuf, scores_real)

# Bootstrap CI
def bootstrap_auroc(scores, labels, n=N_BOOTSTRAP, seed=SEED):
    rng = np.random.default_rng(seed)
    vals = [roc_auc_score(labels[idx := rng.choice(len(labels), len(labels))], scores[idx])
            for _ in range(n)]
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

bs_mean, bs_lo, bs_hi = bootstrap_auroc(scores_real, y_te)
shuffled_clean = (auroc_shuf < auroc_real - 0.05)

# Verdict
if auroc_real >= 0.70 and shuffled_clean:
    verdict = "LAW1_GENERALIZED"
elif auroc_real >= 0.65 and shuffled_clean:
    verdict = "LAW1_WEAK"
elif auroc_real < 0.65:
    verdict = "LAW1_FALSIFIED_FOR_NQ"
else:
    verdict = "SHUFFLED_WARN"

print(f"{ts()} === RESULTS ===")
print(f"{ts()} AUROC (real)     = {auroc_real:.4f}")
print(f"{ts()} AUROC (shuffled) = {auroc_shuf:.4f}  clean={shuffled_clean}")
print(f"{ts()} Bootstrap mean   = {bs_mean:.4f}  CI=[{bs_lo:.4f}, {bs_hi:.4f}]")
print(f"{ts()} VERDICT          = {verdict}")
print(f"{ts()} PRE-REGISTERED   = AUROC >= 0.70")
print(f"{ts()} N collected      = {n_available}/class")

result = {
    "experiment": "EXP_NQ_BLIND_PREDICTION_V1",
    "pre_registered_prediction": "AUROC >= 0.70",
    "kill_criterion": "AUROC < 0.65",
    "model": MODEL_ID,
    "dataset": "nq_open",
    "layer": PROBE_LAYER,
    "step": 1,
    "n_per_class": n_available,
    "n_scanned": n_scanned,
    "n_param_found": n_param_found,
    "n_confab_found": n_confab_found,
    "n_train": len(X_tr),
    "n_test": len(X_te),
    "auroc_real": float(auroc_real),
    "auroc_shuffled": float(auroc_shuf),
    "shuffled_clean": bool(shuffled_clean),
    "bootstrap_mean": float(bs_mean),
    "bootstrap_ci_lo": float(bs_lo),
    "bootstrap_ci_hi": float(bs_hi),
    "verdict": verdict,
    "prediction_met": bool(auroc_real >= 0.70),
    "kill_triggered": bool(auroc_real < 0.65),
    "protocol_note": "L1 direct labeling (PARAM vs CONFAB). NQ-Open has no context passages so bilateral oracle not used. CONFAB = F1 < 0.15, non-abstain answer.",
}

with open(OUTPUT_FILE, "w") as f:
    json.dump(result, f, indent=2)

print(f"{ts()} Results saved to {OUTPUT_FILE}")
print(f"{ts()} DONE — verdict={verdict}  auroc={auroc_real:.4f}  N={n_available}/class")
