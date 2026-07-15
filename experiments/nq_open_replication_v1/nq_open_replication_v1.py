"""
NQ-Open replication of r=0.0039 independence result (Program A).

Goal: replicate corr(J_know, entropy) ≈ 0 on Natural Questions Open
      — a second single-hop factual dataset independent of TriviaQA.

Design mirrors TriviaQA primary experiment exactly:
  - Same model: Qwen2.5-7B-Instruct
  - Same probe: Fisher LDA at layer 26, gen step-1
  - Same bilateral oracle: PARAM (F1>=0.50 no-context) vs CTX_DEP (F1<=0.05 no-context)
  - Same n: N_TARGET=50 per class, N_CAL=30 for probe fit

NQ-Open is appropriate because:
  - Single-hop factual questions (same type as TriviaQA)
  - Open-ended short answers (no MCQ format collapse)
  - Qwen2.5-7B-Instruct has ~50-60% parametric hit rate
  - Clean PARAM/CTX_DEP split expected under bilateral oracle

Expected:
  r(J_know, entropy) ≈ 0.00 (CEILING_REPLICATED)
  Fisher AUROC > 0.85
  Behavioral AUROC ≈ 0.50–0.60 (below 0.65 threshold)
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
from scipy.stats import pearsonr

# Force flush on all prints
import functools, builtins
builtins.print = functools.partial(builtins.print, flush=True)

# ── Config ────────────────────────────────────────────────────────────────────
_KG_PATH = "/kaggle/input/qwen2.5/transformers/7b-instruct/1"
MODEL_ID    = _KG_PATH if os.path.exists(_KG_PATH) else "Qwen/Qwen2.5-7B-Instruct"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
PROBE_LAYER = 26
N_TARGET    = 50    # per class (PARAM / CTX_DEP) — matches TriviaQA primary
N_CAL       = 30    # probe calibration per class
MAX_SCAN    = 1500  # NQ-Open validation has ~3600 items; bilateral oracle is strict
SEED        = 42

np.random.seed(SEED)
torch.manual_seed(SEED)
T0 = time.time()
def ts(): return f"[{int(time.time()-T0)}s]"

print(f"{ts()} Model={MODEL_ID}")
print(f"{ts()} Layer={PROBE_LAYER}  N_TARGET={N_TARGET}  N_CAL={N_CAL}  MAX_SCAN={MAX_SCAN}")

# ── Load model ────────────────────────────────────────────────────────────────
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
print(f"{ts()} Loading model...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=bnb, device_map="auto", output_hidden_states=True
)
model.eval()
print(f"{ts()} Loaded. VRAM={torch.cuda.memory_allocated()/1e9:.2f}GB")

# ── Helpers ───────────────────────────────────────────────────────────────────
def make_chat_prompt(question: str) -> str:
    msgs = [{"role": "user", "content": f"Answer in one word or a short phrase: {question}"}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def generate(prompt: str, max_new: int = 64) -> str:
    inp = tok(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        ids = model.generate(
            **inp, max_new_tokens=max_new,
            do_sample=False, temperature=1.0, top_p=1.0,
            pad_token_id=tok.eos_token_id
        )
    return tok.decode(ids[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()

def step1_signals(prompt: str):
    inp = tok(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**inp)
    h   = out.hidden_states[PROBE_LAYER][0, -1].float().cpu().numpy()
    p   = torch.softmax(out.logits[0, -1].float(), dim=-1)
    ent = -torch.sum(p * torch.log(p + 1e-10)).item()
    return h, ent

def token_f1(pred: str, gold) -> float:
    # NQ-Open answers can be a list; take best F1 across all gold answers
    golds = gold if isinstance(gold, list) else [str(gold)]
    best = 0.0
    p_t = set(pred.lower().split())
    for g in golds:
        g_t = set(str(g).lower().split())
        if not g_t:
            continue
        tp = len(p_t & g_t)
        if tp == 0:
            continue
        pr = tp / len(p_t) if p_t else 0.0
        rc = tp / len(g_t)
        f1 = 2 * pr * rc / (pr + rc) if (pr + rc) > 0 else 0.0
        best = max(best, f1)
    return best

# ── Load NQ-Open ──────────────────────────────────────────────────────────────
print(f"\n{ts()} Loading NQ-Open validation split...")
ds = load_dataset("nq_open", split="validation").shuffle(seed=SEED)
print(f"{ts()} NQ-Open loaded: {len(ds)} items")

# ── Bilateral oracle collection ───────────────────────────────────────────────
# Same protocol as TriviaQA primary:
#   PARAM:   F1(no-context) >= 0.50  (model knows the answer parametrically)
#   CTX_DEP: F1(no-context) <= 0.05  (model fails without context)
#   SKIP:    0.05 < F1 < 0.50
# For NQ-Open, we use no withcontext check (single-pass oracle),
# matching the simplified oracle used in TriviaQA (n=800 primary run).

param_h, param_ent = [], []
ctxdep_h, ctxdep_ent = [], []
scanned = 0

print(f"{ts()} Starting collection (PARAM F1>=0.50 / CTX_DEP F1<=0.05)...")

for row in ds:
    if len(param_h) >= N_TARGET and len(ctxdep_h) >= N_TARGET:
        break
    if scanned >= MAX_SCAN:
        print(f"{ts()} MAX_SCAN={MAX_SCAN} reached")
        break

    q    = row["question"]
    gold = row["answer"]  # NQ-Open: list of strings

    prompt  = make_chat_prompt(q)
    pred    = generate(prompt)
    f1_nc   = token_f1(pred, gold)
    scanned += 1

    if f1_nc >= 0.50 and len(param_h) < N_TARGET:
        h, ent = step1_signals(prompt)
        param_h.append(h); param_ent.append(ent)
    elif f1_nc <= 0.05 and len(ctxdep_h) < N_TARGET:
        h, ent = step1_signals(prompt)
        ctxdep_h.append(h); ctxdep_ent.append(ent)

    if scanned % 50 == 0:
        print(f"{ts()} scanned={scanned}  PARAM={len(param_h)}  CTX_DEP={len(ctxdep_h)}")

n_p, n_c = len(param_h), len(ctxdep_h)
print(f"\n{ts()} Collected: PARAM={n_p}  CTX_DEP={n_c}  scanned={scanned}")

if n_p < 10 or n_c < 10:
    print(f"{ts()} INSUFFICIENT — need >=10 per class. Exiting.")
    sys.exit(1)

# ── Fisher LDA probe ──────────────────────────────────────────────────────────
all_h   = np.array(param_h + ctxdep_h)
all_ent = np.array(param_ent + ctxdep_ent)
all_y   = np.array([1]*n_p + [0]*n_c)

n_cal = min(N_CAL, n_p, n_c)
X_cal = np.concatenate([all_h[:n_cal], all_h[n_p:n_p+n_cal]])
y_cal = np.array([1]*n_cal + [0]*n_cal)

lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
lda.fit(X_cal, y_cal)
j_know = lda.decision_function(all_h)

# AUROC on held-out
Xh = np.concatenate([all_h[n_cal:n_p], all_h[n_p+n_cal:]])
yh = np.array([1]*(n_p - n_cal) + [0]*(n_c - n_cal))
if len(set(yh)) < 2 or len(Xh) < 4:
    from sklearn.model_selection import cross_val_score
    fisher_auroc = cross_val_score(
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        X_cal, y_cal, cv=3, scoring="roc_auc").mean()
else:
    fisher_auroc = roc_auc_score(yh, lda.decision_function(Xh))

# Pearson r
r_val, p_val = pearsonr(j_know, all_ent)

# Behavioral AUROC (entropy alone)
beh = roc_auc_score(all_y, -all_ent)
beh = max(beh, 1 - beh)

# Verdict — same threshold as TriviaQA primary
ceiling = abs(r_val) < 0.05 and beh < 0.65 and fisher_auroc > 0.60
verdict = "CEILING_REPLICATED" if ceiling else "DOMAIN_DEPENDENT"

print(f"\n{ts()} ══ NQ-OPEN RESULTS ══")
print(f"{ts()} r(J_know, entropy) = {r_val:.4f}  (p={p_val:.4f})")
print(f"{ts()} Fisher AUROC       = {fisher_auroc:.4f}")
print(f"{ts()} Behavioral AUROC   = {beh:.4f}")
print(f"{ts()} Gap (Fisher-Behav) = {fisher_auroc - beh:+.4f}")
print(f"{ts()} Verdict            = {verdict}")
print(f"\n{ts()} Reference TriviaQA: r=0.0039  Fisher=0.989  Behav=0.51  n=800")

# ── Save ──────────────────────────────────────────────────────────────────────
results = {
    "dataset": "nq_open",
    "model_id": MODEL_ID,
    "probe_layer": PROBE_LAYER,
    "n_param": n_p,
    "n_ctxdep": n_c,
    "scanned": scanned,
    "n_cal": n_cal,
    "r_j_know_entropy": float(r_val),
    "r_pvalue": float(p_val),
    "fisher_auroc": float(fisher_auroc),
    "behavioral_auroc": float(beh),
    "gap": float(fisher_auroc - beh),
    "ceiling_replicated": bool(ceiling),
    "verdict": verdict,
    "reference_triviaqa": {
        "r": 0.0039, "fisher": 0.989, "behavioral": 0.51, "n": 800
    },
    "elapsed_s": int(time.time() - T0)
}

out = "/kaggle/working/nq_open_replication_v1_results.json"
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"{ts()} Saved: {out}")
print(json.dumps(results, indent=2))
