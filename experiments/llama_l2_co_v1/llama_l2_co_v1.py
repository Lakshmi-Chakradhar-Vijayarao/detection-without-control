"""
Llama L2 Plain-CO — L2 False Certainty on Llama-3.2-3B-Instruct
================================================================

MOTIVATION:
  EXP_A_LAW3_LLAMA_V7 showed n_cc=0, n_cw=0 for Llama L2 with
  entropy-matched CO (theta_conf=0.50). Llama's entropy distribution
  is broader/shifted vs Qwen — almost no items fall in the 0.50 window.

  This experiment uses PLAIN CO labeling (no entropy threshold):
    CC = correct answer (F1 >= 0.50), any entropy
    CW = wrong  answer  (F1 <= 0.05), non-abstain, any entropy

  This is the same protocol that confirmed C036 on Gemma + Mistral.

PRE-REGISTERED INTERPRETATION (written before run):
  C036 extension: CO labeling recovers L2 for Llama-3.2-3B-Instruct.
  Expected: Fisher+PCA64 AUROC >= 0.70 (gap >= 0.10 vs entropy baseline).
  Kill criterion: Fisher AUROC < 0.60 AND gap < 0.05.

Model: meta-llama/Llama-3.2-3B-Instruct
Layer: 26 (penultimate, 0-indexed of 28 total)
Dataset: TriviaQA rc.nocontext validation
"""

from __future__ import annotations
import subprocess
print("[init] installing bitsandbytes...", flush=True)
subprocess.run(["pip", "install", "-q", "-U", "bitsandbytes>=0.46.1"], check=False)

import json, os, string, time

class _NpEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as _np
        if isinstance(obj, _np.integer): return int(obj)
        if isinstance(obj, _np.floating): return float(obj)
        if isinstance(obj, _np.bool_): return bool(obj)
        if isinstance(obj, _np.ndarray): return obj.tolist()
        return super().default(obj)
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
MODEL_ID       = "meta-llama/Llama-3.2-3B-Instruct"
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
PROBE_LAYER    = 26
N_TARGET       = 100     # per class
MAX_SCAN       = 8000
N_PCA          = 64
SEED           = 42
F1_CC          = 0.50    # correct threshold
F1_CW_MAX      = 0.05    # wrong threshold
MAX_NEW_TOKENS = 60
N_BOOTSTRAP    = 1000
OUTPUT_FILE    = "/kaggle/working/llama_l2_co_v1.json"

np.random.seed(SEED)
torch.manual_seed(SEED)
T0 = time.time()
def ts(): return f"[{int(time.time()-T0):5d}s]"

print(f"{ts()} === LLAMA L2 PLAIN-CO v1 ===")
print(f"{ts()} Model={MODEL_ID}  Layer={PROBE_LAYER}")
print(f"{ts()} Protocol: plain CO (no entropy threshold)")
print(f"{ts()} CC: F1 >= {F1_CC}  |  CW: F1 <= {F1_CW_MAX}, non-abstain")
print(f"{ts()} Device={DEVICE}")

# ── F1 helpers ────────────────────────────────────────────────────────────────
def normalize(s: str) -> str:
    s = s.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())

def token_f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = sum(min(p.count(t), g.count(t)) for t in set(p) & set(g))
    if common == 0:
        return 0.0
    pr, rc = common / len(p), common / len(g)
    return 2 * pr * rc / (pr + rc)

def max_f1_list(pred: str, answers: list[str]) -> float:
    return max((token_f1(pred, a) for a in answers), default=0.0)

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
print(f"{ts()} Loading TriviaQA validation...")
ds = load_dataset("trivia_qa", "rc.nocontext", split="validation", trust_remote_code=True)
print(f"{ts()} {len(ds)} items")

def format_q(question: str) -> str:
    return (f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
            f"Answer the following question with a short phrase (1-5 words). "
            f"If you don't know, say 'I don't know'.\n\n"
            f"Question: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n")

def run_item(prompt: str):
    enc = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_len = enc.input_ids.shape[1]
    hs_storage = {}

    def hook(module, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        # With KV cache each decode step has shape[1]==1; prefill has shape[1]==input_len.
        # Capture only once at step-1 (first new token).
        if "hs" not in hs_storage and hs.shape[1] == 1:
            hs_storage["hs"] = hs[:, -1, :].detach().float().cpu()

    handle = model.model.layers[PROBE_LAYER].register_forward_hook(hook)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=1.0,
            output_hidden_states=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    handle.remove()
    hs_vec = hs_storage.get("hs", None)
    if hs_vec is not None:
        hs_vec = hs_vec.squeeze(0).numpy()
    answer = tokenizer.decode(out[0, input_len:], skip_special_tokens=True).strip()
    return answer, hs_vec

def get_answers(item) -> list[str]:
    ans = item.get("answer", {})
    aliases = ans.get("aliases", []) if isinstance(ans, dict) else []
    norm_val = ans.get("normalized_value", "") if isinstance(ans, dict) else str(ans)
    val = ans.get("value", "") if isinstance(ans, dict) else str(ans)
    return list({a for a in [norm_val, val] + aliases if a})

# ── Scan ──────────────────────────────────────────────────────────────────────
cc_items, cw_items = [], []
n_scanned = n_cc = n_cw = 0

print(f"{ts()} Starting plain-CO scan (target {N_TARGET}/class)...")

for item in ds:
    if n_cc >= N_TARGET and n_cw >= N_TARGET:
        break
    if n_scanned >= MAX_SCAN:
        break

    question = item["question"].strip()
    answers  = get_answers(item)
    if not answers:
        continue

    n_scanned += 1
    prompt = format_q(question)
    answer, hs = run_item(prompt)
    f1 = max_f1_list(answer, answers)

    if f1 >= F1_CC and n_cc < N_TARGET:
        cc_items.append({"hs": hs, "f1": f1})
        n_cc += 1
        if n_cc % 10 == 0:
            print(f"{ts()} CC={n_cc}/{N_TARGET}  CW={n_cw}/{N_TARGET}  scanned={n_scanned}")

    elif (f1 <= F1_CW_MAX
          and "don't know" not in answer.lower()
          and "i don't" not in answer.lower()
          and len(answer.split()) >= 2
          and n_cw < N_TARGET):
        cw_items.append({"hs": hs, "f1": f1, "answer": answer})
        n_cw += 1
        if n_cw % 10 == 0:
            print(f"{ts()} CC={n_cc}/{N_TARGET}  CW={n_cw}/{N_TARGET}  scanned={n_scanned}")

    if n_scanned % 200 == 0:
        print(f"{ts()} scan {n_scanned}/{MAX_SCAN}  CC={n_cc}  CW={n_cw}")

print(f"{ts()} Scan done: CC={n_cc}, CW={n_cw}, scanned={n_scanned}")

# ── Probe ─────────────────────────────────────────────────────────────────────
n = min(n_cc, n_cw)
if n < 40:
    result = {
        "status": "PROTOCOL_FAILURE",
        "reason": f"Insufficient: CC={n_cc}, CW={n_cw}",
        "model": MODEL_ID,
        "verdict": "PROTOCOL_FAILURE",
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(result, f, indent=2)
    print(f"{ts()} PROTOCOL_FAILURE")
    raise SystemExit(0)

X = np.vstack([
    np.array([x["hs"] for x in cc_items[:n]]),
    np.array([x["hs"] for x in cw_items[:n]])
])
y = np.array([1]*n + [0]*n)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.35, random_state=SEED, stratify=y)

# Fisher+PCA64
pca = PCA(n_components=N_PCA, random_state=SEED)
X_tr_pca = pca.fit_transform(X_tr)
X_te_pca = pca.transform(X_te)
lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
lda.fit(X_tr_pca, y_tr)
scores = lda.decision_function(X_te_pca)
auroc_fisher = roc_auc_score(y_te, scores)

# Shuffled control
y_shuf = y_te.copy(); np.random.shuffle(y_shuf)
auroc_shuf = roc_auc_score(y_shuf, scores)

# Bootstrap CI
rng = np.random.default_rng(SEED)
bs_vals = [roc_auc_score(y_te[idx := rng.choice(len(y_te), len(y_te))], scores[idx])
           for _ in range(N_BOOTSTRAP)]
bs_lo, bs_hi = float(np.percentile(bs_vals, 2.5)), float(np.percentile(bs_vals, 97.5))

gap = auroc_fisher - 0.50  # gap vs chance (no entropy baseline available in plain CO)
shuffled_clean = auroc_shuf < auroc_fisher - 0.05

if auroc_fisher >= 0.70 and shuffled_clean:
    verdict = "SUPPORTED"
elif auroc_fisher >= 0.60 and shuffled_clean:
    verdict = "WEAK"
elif auroc_fisher < 0.60:
    verdict = "NOT_SUPPORTED"
else:
    verdict = "SHUFFLED_WARN"

kill_triggered = auroc_fisher < 0.60 and gap < 0.05

print(f"{ts()} === RESULTS ===")
print(f"{ts()} Fisher AUROC  = {auroc_fisher:.4f}  CI=[{bs_lo:.4f}, {bs_hi:.4f}]")
print(f"{ts()} Shuffled AUROC = {auroc_shuf:.4f}  clean={shuffled_clean}")
print(f"{ts()} VERDICT = {verdict}  kill={kill_triggered}")

result = {
    "experiment": "LLAMA_L2_CO_V1",
    "model": MODEL_ID,
    "protocol": "plain_CO_no_entropy_threshold",
    "layer": PROBE_LAYER,
    "n_per_class": n,
    "n_cc": n_cc,
    "n_cw": n_cw,
    "n_scanned": n_scanned,
    "n_train": len(X_tr),
    "n_test": len(X_te),
    "auroc_fisher": float(auroc_fisher),
    "auroc_shuffled": float(auroc_shuf),
    "shuffled_clean": bool(shuffled_clean),
    "bootstrap_ci_lo": bs_lo,
    "bootstrap_ci_hi": bs_hi,
    "gap_vs_chance": float(gap),
    "verdict": verdict,
    "kill_triggered": kill_triggered,
    "c036_extension": auroc_fisher >= 0.70,
    "note": "Plain CO labeling (no entropy threshold). Fixes entropy-matched CO failure (n_cc=0) from EXP_A_LAW3_LLAMA_V7.",
}

with open(OUTPUT_FILE, "w") as f:
    json.dump(result, f, indent=2, cls=_NpEncoder)

print(f"{ts()} Saved to {OUTPUT_FILE}")
print(f"{ts()} DONE — verdict={verdict}  auroc={auroc_fisher:.4f}  N={n}/class")
