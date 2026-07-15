"""
SeaKR direct comparison v2 — chat template fix.

v1 bug: eval_f1 used raw text prompt ("Answer briefly: {q}\nAnswer:") for
Qwen2.5-7B-Instruct, which needs a chat template.  Result: <3% CORRECT class,
biased sample.

v2 fix: apply_chat_template for both oracle (eval_f1) and probe (get_signals).
N_TARGET reduced to 100/class to keep within T4 time budget.
"""

import subprocess
print("[init] pip install bitsandbytes...", flush=True)
subprocess.run(["pip", "install", "-q", "-U", "bitsandbytes>=0.46.1"], check=True)
print("[init] done.", flush=True)

import os, sys, json, time
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score

import functools, builtins
builtins.print = functools.partial(builtins.print, flush=True)

# ── Config ────────────────────────────────────────────────────────────────────
_KG_INSTRUCT = "/kaggle/input/qwen2.5/transformers/7b-instruct/1"
MODEL_ID    = _KG_INSTRUCT if os.path.exists(_KG_INSTRUCT) else "Qwen/Qwen2.5-7B-Instruct"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
PROBE_LAYER = 26
N_TARGET    = 100   # per class — reduced from 200 to ensure T4 completion
N_CAL       = 40    # calibration per class for Fisher probe
MAX_GEN_TOK = 32    # short answers; instruct model is concise with chat template
SEED        = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

print(f"  Model:  {MODEL_ID}")
print(f"  Device: {DEVICE}")
print(f"  Target: n={N_TARGET} per class (N_CAL={N_CAL})")

# ── Load model ────────────────────────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
print("\n  Loading model...")
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=bnb_config, device_map="auto",
    output_hidden_states=True
)
model.eval()
print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# ── Chat template helper ───────────────────────────────────────────────────────
def make_chat_prompt(question: str) -> str:
    """Format question with Qwen chat template for proper instruct-model generation."""
    messages = [{"role": "user", "content": f"Answer in one word or a short phrase: {question}"}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# ── Signal extraction ─────────────────────────────────────────────────────────
def get_signals(question: str):
    """
    Extract probing signals at generation step 1.
    Returns h_L26, h_last, entropy.
    Uses chat-formatted prompt so instruct model sees proper format.
    """
    prompt = make_chat_prompt(question)
    inputs = tok(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    h_L26  = out.hidden_states[PROBE_LAYER][0, -1].float().cpu().numpy()
    h_last = out.hidden_states[-1][0, -1].float().cpu().numpy()
    logits = out.logits[0, -1].float()
    probs  = torch.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
    return h_L26, h_last, entropy


def eval_f1(question: str, answer: str) -> float:
    """Token-level F1. Chat template used so instruct model generates correctly."""
    prompt = make_chat_prompt(question)
    inputs = tok(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        ids = model.generate(
            **inputs, max_new_tokens=MAX_GEN_TOK,
            do_sample=False, temperature=1.0, top_p=1.0,
            pad_token_id=tok.eos_token_id
        )
    gen = tok.decode(ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
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

# ── Load TriviaQA ─────────────────────────────────────────────────────────────
print("\n  Loading TriviaQA...")
ds = load_dataset("trivia_qa", "rc.wikipedia", split="validation")
ds = ds.shuffle(seed=SEED)

# ── Collect CORRECT / INCORRECT samples ──────────────────────────────────────
correct_h26, correct_hlast, correct_ent = [], [], []
incorr_h26,  incorr_hlast,  incorr_ent  = [], [], []
t0 = time.time()

for i, row in enumerate(ds):
    if len(correct_h26) >= N_TARGET and len(incorr_h26) >= N_TARGET:
        break

    q = row["question"]
    a = row["answer"]["value"]
    f1 = eval_f1(q, a)
    label = "CORRECT" if f1 >= 0.50 else "INCORRECT" if f1 <= 0.05 else "SKIP"

    if label == "SKIP":
        continue
    if label == "CORRECT" and len(correct_h26) >= N_TARGET:
        continue
    if label == "INCORRECT" and len(incorr_h26) >= N_TARGET:
        continue

    h26, hlast, ent = get_signals(q)

    if label == "CORRECT":
        correct_h26.append(h26); correct_hlast.append(hlast); correct_ent.append(ent)
    else:
        incorr_h26.append(h26);  incorr_hlast.append(hlast);  incorr_ent.append(ent)

    n_c, n_i = len(correct_h26), len(incorr_h26)
    if (i + 1) % 20 == 0:
        print(f"    [{i+1}] CORRECT={n_c} INCORRECT={n_i} elapsed={time.time()-t0:.0f}s")

n_c, n_i = len(correct_h26), len(incorr_h26)
print(f"\n  Collected: CORRECT={n_c} INCORRECT={n_i}")
if n_c < 30 or n_i < 30:
    print("  INSUFFICIENT samples. Exiting.")
    sys.exit(1)

# ── Signal computation ────────────────────────────────────────────────────────
all_y     = np.array([1]*n_c + [0]*n_i)
all_h26   = np.array(correct_h26 + incorr_h26)
all_hlast = np.array(correct_hlast + incorr_hlast)
all_ent   = np.array(correct_ent + incorr_ent)

# SeaKR signal 1: L2-norm of last hidden state
seakr_l2 = np.linalg.norm(all_hlast, axis=1)

# SeaKR signal 2: cosine similarity to centroid
mean_last   = all_hlast.mean(axis=0)
mean_last_n = mean_last / (np.linalg.norm(mean_last) + 1e-8)
seakr_cos   = all_hlast @ mean_last_n

# Fisher LDA probe at L26
n_cal = min(N_CAL, n_c, n_i)
X_cal = np.concatenate([all_h26[:n_cal], all_h26[n_c:n_c+n_cal]])
y_cal = np.array([1]*n_cal + [0]*n_cal)
lda   = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
lda.fit(X_cal, y_cal)
fisher_scores = lda.decision_function(all_h26)

# ── AUROC comparison ──────────────────────────────────────────────────────────
fisher_auroc    = roc_auc_score(all_y, fisher_scores)
seakr_l2_auroc  = roc_auc_score(all_y, seakr_l2)
seakr_cos_auroc = roc_auc_score(all_y, seakr_cos)
entropy_auroc   = roc_auc_score(all_y, -all_ent)

seakr_l2_auroc  = max(seakr_l2_auroc,  1 - seakr_l2_auroc)
seakr_cos_auroc = max(seakr_cos_auroc, 1 - seakr_cos_auroc)
entropy_auroc   = max(entropy_auroc,   1 - entropy_auroc)

print("\n" + "="*60)
print("  SEAKR vs. FISHER COMPARISON")
print("="*60)
print(f"  Fisher LDA (L{PROBE_LAYER})     : {fisher_auroc:.4f}")
print(f"  SeaKR L2-norm (last layer): {seakr_l2_auroc:.4f}")
print(f"  SeaKR cosim (last layer)  : {seakr_cos_auroc:.4f}")
print(f"  Entropy (output-space)    : {entropy_auroc:.4f}")

best_seakr = max(seakr_l2_auroc, seakr_cos_auroc)
gap = fisher_auroc - best_seakr

if gap > 0.05:
    verdict = "FISHER_BETTER"
elif gap > -0.05:
    verdict = "COMPETITIVE"
else:
    verdict = "SEAKR_BETTER"

print(f"\n  Gap (Fisher - best SeaKR): {gap:+.4f}")
print(f"  Verdict: {verdict}")

# ── Save results ──────────────────────────────────────────────────────────────
results = {
    "version": "v2",
    "fix": "chat_template_applied",
    "model": MODEL_ID,
    "dataset": "TriviaQA rc.wikipedia",
    "oracle": "CORRECT(F1>=0.50) vs INCORRECT(F1<=0.05) — SeaKR-compatible oracle",
    "n_correct": n_c,
    "n_incorrect": n_i,
    "n_total": n_c + n_i,
    "fisher_l26_auroc": float(fisher_auroc),
    "seakr_l2_auroc": float(seakr_l2_auroc),
    "seakr_cosim_auroc": float(seakr_cos_auroc),
    "entropy_auroc": float(entropy_auroc),
    "best_seakr_auroc": float(best_seakr),
    "gap_fisher_minus_seakr": float(gap),
    "verdict": verdict,
    "elapsed_s": int(time.time() - t0),
    "note": (
        "v2 uses apply_chat_template for both oracle F1 and probe extraction. "
        "v1 used raw text prompt which caused <3% CORRECT rate on instruct model. "
        "SeaKR signals are reconstructed from paper description (L2-norm and cosim "
        "of last hidden state). Oracle: CORRECT vs INCORRECT (SeaKR-compatible), "
        "not bilateral PARAM/CTX_DEP."
    )
}

out_path = "/kaggle/working/seakr_comparison_v2_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n[Final] {out_path}")
print(json.dumps(results, indent=2))
