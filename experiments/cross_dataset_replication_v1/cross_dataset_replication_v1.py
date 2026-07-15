"""
Cross-dataset replication of r=0.0039 (Program A).
v9 fixes vs v8:
  - HotpotQA: lower KNOWS threshold F1>=0.40 → F1>=0.20.
    Rationale: multi-hop questions almost never score F1>=0.40 from parametric
    memory alone (hit rate was 0.67%: 3/450 scanned). F1>=0.20 captures partial
    parametric knowledge (model knows one of the two sub-facts). DOESNT_KNOW
    tightened to F1<=0.05 to maintain clean separation.
  - HotpotQA MAX_SCAN: 800 → 1500 as safety margin.
v8: Kaggle model hub loading (95s vs 7000s stall).
v7: HotpotQA single-pass oracle. MMLU r=-0.5447 confirmed (domain-dependent).
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

import functools, builtins
builtins.print = functools.partial(builtins.print, flush=True)

# ── Config ─────────────────────────────────────────────────────────────────────
_KAGGLE_PATH = "/kaggle/input/qwen2.5/transformers/7b-instruct/1"
MODEL_ID = _KAGGLE_PATH if os.path.exists(_KAGGLE_PATH) else "Qwen/Qwen2.5-7B-Instruct"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
PROBE_LAYER = 26
N_TARGET       = 30    # per class
N_CAL          = 20
MAX_SCAN_MMLU  = 800
MAX_SCAN_HPQA  = 1500  # HotpotQA needs more scans; multi-hop is parametrically sparse
SEED           = 42

np.random.seed(SEED); torch.manual_seed(SEED)
T0 = time.time()
def ts(): return f"[{int(time.time()-T0)}s]"

print(f"{ts()} Model={MODEL_ID} Layer={PROBE_LAYER} N_TARGET={N_TARGET} MAX_SCAN_MMLU={MAX_SCAN_MMLU} MAX_SCAN_HPQA={MAX_SCAN_HPQA}", flush=True)

# ── Load model ─────────────────────────────────────────────────────────────────
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
print(f"{ts()} Loading model...", flush=True)
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=bnb, device_map="auto", output_hidden_states=True
)
model.eval()
print(f"{ts()} Loaded. VRAM={torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)

# ── Helpers ────────────────────────────────────────────────────────────────────
def step1_signals(prompt: str):
    """Forward pass on prompt → (hidden_L26 numpy, output_entropy float)."""
    inp = tok(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**inp)
    h   = out.hidden_states[PROBE_LAYER][0, -1].float().cpu().numpy()
    p   = torch.softmax(out.logits[0, -1].float(), dim=-1)
    ent = -torch.sum(p * torch.log(p + 1e-10)).item()
    return h, ent


def generate(prompt: str, max_new: int = 64) -> str:
    inp = tok(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        ids = model.generate(
            **inp, max_new_tokens=max_new,
            do_sample=False, temperature=1.0, top_p=1.0,
            pad_token_id=tok.eos_token_id
        )
    return tok.decode(ids[0][inp.input_ids.shape[1]:], skip_special_tokens=True)


def token_f1(pred: str, gold: str) -> float:
    p_t = set(pred.lower().split())
    g_t = set(str(gold).lower().split())
    if not g_t: return 0.0
    tp = len(p_t & g_t)
    if tp == 0: return 0.0
    pr = tp / len(p_t) if p_t else 0.0
    rc = tp / len(g_t)
    return 2*pr*rc/(pr+rc) if (pr+rc) > 0 else 0.0

# ── MMLU: MCQ format, letter check ────────────────────────────────────────────
def run_mmlu():
    print(f"\n{ts()} ── MMLU (MCQ format: A/B/C/D letter check) ──", flush=True)
    ds = load_dataset("cais/mmlu", "all", split="test", trust_remote_code=True)
    ds = ds.shuffle(seed=SEED)
    print(f"{ts()} MMLU loaded: {len(ds)} items", flush=True)

    LETTERS = ["A", "B", "C", "D"]
    pos_h, pos_ent = [], []   # CORRECT
    neg_h, neg_ent = [], []   # INCORRECT
    scanned = 0

    for row in ds:
        if len(pos_h) >= N_TARGET and len(neg_h) >= N_TARGET:
            break
        if scanned >= MAX_SCAN_MMLU:
            print(f"{ts()} MMLU: MAX_SCAN_MMLU={MAX_SCAN_MMLU} reached", flush=True)
            break

        q           = row["question"]
        choices     = row["choices"]
        correct_idx = row["answer"]
        correct_ltr = LETTERS[correct_idx]

        # Build MCQ prompt (same prompt used for both generation and step1_signals)
        opts = "\n".join(f"{LETTERS[i]}) {c}" for i, c in enumerate(choices))
        prompt = f"Question: {q}\n{opts}\nAnswer with one letter (A/B/C/D):"

        # Generate — only 5 tokens needed for a letter
        gen_out = generate(prompt, max_new=5).strip()
        pred_ltr = gen_out[0].upper() if gen_out else "?"
        is_correct = (pred_ltr == correct_ltr)
        scanned += 1

        if is_correct and len(pos_h) < N_TARGET:
            h, ent = step1_signals(prompt)
            pos_h.append(h); pos_ent.append(ent)
        elif not is_correct and len(neg_h) < N_TARGET:
            h, ent = step1_signals(prompt)
            neg_h.append(h); neg_ent.append(ent)

        if scanned % 50 == 0:
            print(f"{ts()} MMLU scanned={scanned} CORRECT={len(pos_h)} INCORRECT={len(neg_h)}", flush=True)

    return _compute("MMLU", pos_h, pos_ent, neg_h, neg_ent, scanned, "CORRECT", "INCORRECT")

# ── HotpotQA: single-pass KNOWS / DOESNT_KNOW ────────────────────────────────
def run_hotpotqa():
    print(f"\n{ts()} ── HotpotQA (single-pass: KNOWS f1>=0.40 / DOESNT_KNOW f1<=0.10) ──", flush=True)
    # trust_remote_code deprecated; load without it
    ds = load_dataset("hotpot_qa", "distractor", split="validation")
    ds = ds.shuffle(seed=SEED)
    print(f"{ts()} HotpotQA loaded: {len(ds)} items", flush=True)

    # Thresholds: relaxed vs TriviaQA because HotpotQA multi-hop is parametrically sparse.
    # F1>=0.20 = model knows one sub-fact (partial parametric knowledge).
    # F1<=0.05 = complete failure. Gap 0.05–0.20 is ambiguous SKIP zone.
    HPQA_KNOWS_THR = 0.20
    HPQA_DNKNOW_THR = 0.05

    pos_h, pos_ent = [], []   # KNOWS (partial parametric knowledge, F1>=0.20)
    neg_h, neg_ent = [], []   # DOESNT_KNOW (complete failure, F1<=0.05)
    scanned = 0

    for row in ds:
        if len(pos_h) >= N_TARGET and len(neg_h) >= N_TARGET:
            break
        if scanned >= MAX_SCAN_HPQA:
            print(f"{ts()} HotpotQA: MAX_SCAN_HPQA={MAX_SCAN_HPQA} reached", flush=True)
            break

        q    = row["question"]
        gold = row["answer"]

        # Single generate call — no context (same open-ended format as TriviaQA)
        prompt = f"Answer briefly: {q}\nAnswer:"
        f1_nc  = token_f1(generate(prompt), gold)
        scanned += 1

        if f1_nc >= HPQA_KNOWS_THR and len(pos_h) < N_TARGET:
            h, ent = step1_signals(prompt)
            pos_h.append(h); pos_ent.append(ent)
        elif f1_nc <= HPQA_DNKNOW_THR and len(neg_h) < N_TARGET:
            h, ent = step1_signals(prompt)
            neg_h.append(h); neg_ent.append(ent)

        if scanned % 50 == 0:
            print(f"{ts()} HotpotQA scanned={scanned} KNOWS(f1>={HPQA_KNOWS_THR})={len(pos_h)} DOESNT_KNOW(f1<={HPQA_DNKNOW_THR})={len(neg_h)}", flush=True)

    return _compute("HotpotQA", pos_h, pos_ent, neg_h, neg_ent, scanned,
                    f"KNOWS(f1>={HPQA_KNOWS_THR})", f"DOESNT_KNOW(f1<={HPQA_DNKNOW_THR})")

# ── Compute r / AUROC ──────────────────────────────────────────────────────────
def _compute(name, pos_h, pos_ent, neg_h, neg_ent, scanned, lp, ln):
    n_p, n_n = len(pos_h), len(neg_h)
    print(f"\n{ts()} [{name}] Collected: {lp}={n_p}  {ln}={n_n}  scanned={scanned}", flush=True)

    if n_p < 10 or n_n < 10:
        print(f"{ts()} [{name}] INSUFFICIENT — need ≥10 per class", flush=True)
        return {"dataset": name, "status": "INSUFFICIENT", "n_pos": n_p, "n_neg": n_n,
                "scanned": scanned}

    all_h   = np.array(pos_h + neg_h)
    all_ent = np.array(pos_ent + neg_ent)
    all_y   = np.array([1]*n_p + [0]*n_n)

    # Fisher LDA
    n_cal = min(N_CAL, n_p, n_n)
    X_cal = np.concatenate([all_h[:n_cal], all_h[n_p:n_p+n_cal]])
    y_cal = np.array([1]*n_cal + [0]*n_cal)
    lda   = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    lda.fit(X_cal, y_cal)
    j_know = lda.decision_function(all_h)

    # AUROC on held-out
    Xh = np.concatenate([all_h[n_cal:n_p], all_h[n_p+n_cal:]])
    yh = np.array([1]*(n_p-n_cal) + [0]*(n_n-n_cal))
    if len(set(yh)) < 2 or len(Xh) < 4:
        from sklearn.model_selection import cross_val_score
        fisher_auroc = cross_val_score(
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
            X_cal, y_cal, cv=3, scoring="roc_auc").mean()
    else:
        fisher_auroc = roc_auc_score(yh, lda.decision_function(Xh))

    # Pearson r
    r_val, p_val = pearsonr(j_know, all_ent)

    # Behavioral AUROC
    beh = roc_auc_score(all_y, -all_ent)
    beh = max(beh, 1 - beh)

    ceiling = abs(r_val) < 0.05 and beh < 0.65 and fisher_auroc > 0.60
    verdict = "CEILING_REPLICATED" if ceiling else "DOMAIN_DEPENDENT"

    print(f"{ts()} [{name}] r(J_know,entropy)={r_val:.4f} (p={p_val:.3f})", flush=True)
    print(f"{ts()} [{name}] Fisher={fisher_auroc:.4f}  Behavioral={beh:.4f}  Gap={fisher_auroc-beh:+.4f}", flush=True)
    print(f"{ts()} [{name}] Verdict: {verdict}", flush=True)

    return {
        "dataset": name, "n_pos": n_p, "n_neg": n_n, "scanned": scanned,
        "label_pos": lp, "label_neg": ln,
        "r_j_know_entropy": float(r_val), "r_pvalue": float(p_val),
        "fisher_auroc": float(fisher_auroc), "behavioral_auroc": float(beh),
        "gap": float(fisher_auroc - beh),
        "ceiling_replicated": bool(ceiling), "verdict": verdict, "status": "complete"
    }

# ── Run ────────────────────────────────────────────────────────────────────────
results = {}
results["mmlu"]     = run_mmlu()
results["hotpotqa"] = run_hotpotqa()

# ── Summary ────────────────────────────────────────────────────────────────────
print(f"\n{ts()} {'='*55}", flush=True)
print(f"{ts()} SUMMARY", flush=True)
replicated = [k for k, v in results.items() if v.get("verdict") == "CEILING_REPLICATED"]
global_verdict = (
    "FULL_REPLICATION"    if len(replicated) == 2 else
    "PARTIAL_REPLICATION" if len(replicated) == 1 else
    "REPLICATION_FAILED"
)
for ds, res in results.items():
    r = res.get("r_j_know_entropy", float("nan"))
    f = res.get("fisher_auroc",     float("nan"))
    b = res.get("behavioral_auroc", float("nan"))
    v = res.get("verdict", res.get("status", "?"))
    try:    print(f"{ts()}   {ds}: r={r:.4f}  Fisher={f:.4f}  Behav={b:.4f}  → {v}", flush=True)
    except: print(f"{ts()}   {ds}: {v}", flush=True)

print(f"{ts()} Global: {global_verdict}", flush=True)
print(f"{ts()} Reference TriviaQA: r=0.0039  Fisher=0.989  Behav=0.51  n=800", flush=True)

results["global_verdict"] = global_verdict
results["triviaqa_reference"] = {"r": 0.0039, "fisher": 0.989, "behavioral": 0.51, "n": 800}
results["elapsed_s"] = int(time.time() - T0)

out = "/kaggle/working/cross_dataset_replication_v1_results.json"
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"{ts()} Saved: {out}", flush=True)
print(json.dumps(results, indent=2), flush=True)
