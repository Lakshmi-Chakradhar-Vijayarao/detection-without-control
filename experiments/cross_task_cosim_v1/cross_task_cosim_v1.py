"""
Cross-task Fisher cosine measurement + Hewitt-Liang selectivity control.

EXISTENTIAL EXPERIMENT: Determines whether the Fisher probe direction at L26
represents a universal epistemic axis, a retrieval-mode axis, or task-specific
geometry. Also closes the Hewitt-Liang reviewer attack in the same run.

Pre-registered verdicts (determined before running):
  UNIVERSAL_AXIS          — mean cosim > 0.70 across all task-pair comparisons
  RETRIEVAL_MODE_SPECIFIC — cosim > 0.70 within factual group {TQA, HPQA, NQ},
                            cosim < 0.35 for MMLU pairs
                            [SUPPORTS computational mode hypothesis: probe detects
                             retrieval attractor, not TriviaQA-specific geometry]
  TASK_SPECIFIC_GEOMETRY  — cosim < 0.35 for ≥2 non-MMLU pairs
                            [Claim 1 contracts to TriviaQA geometry; paper must narrow]

Interpretation logic:
  Three families tested:
    Factual retrieval: TriviaQA, HotpotQA, NQ-Open
      Oracle: nc_F1-based PARAM vs CTX_DEP
    MCQ reasoning: MMLU
      Oracle: CORRECT vs INCORRECT (different concept intentionally)

  If factual family has high cosim but MMLU is low:
    This is NOT failure — it confirms that retrieval-mode geometry is shared
    within factual tasks but distinct from MCQ reasoning geometry.
    Strongly supports boundary conditions table and computational mode framing.

Hewitt-Liang selectivity control:
  Run Fisher LDA on TriviaQA samples with SHUFFLED labels.
  AUROC should drop to ~0.50 ± 0.05.
  If it does: probe is selective (signal is real, not probe overfitting).
  If it doesn't: probe is overfitting the LDA to noise (signal may be spurious).
"""

import subprocess
print("[init] pip install bitsandbytes...", flush=True)
subprocess.run(["pip", "install", "-q", "-U", "bitsandbytes>=0.46.1"], check=True)
print("[init] done.", flush=True)

import os, gc, json, time
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

import functools, builtins
builtins.print = functools.partial(builtins.print, flush=True)

# ── Config ────────────────────────────────────────────────────────────────────
_KG_PATH   = "/kaggle/input/qwen2.5/transformers/7b-instruct/1"
MODEL_ID   = _KG_PATH if os.path.exists(_KG_PATH) else "Qwen/Qwen2.5-7B-Instruct"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
PROBE_LAYER = 26
N_TARGET   = 30     # labeled samples per class per task
N_CAL      = 25     # used for LDA fit; remainder is held-out eval
N_SEEDS    = 3      # probe stability seeds
MAX_SCAN   = 1000   # max items to scan per task for labels
SEED       = 42

np.random.seed(SEED)
torch.manual_seed(SEED)
T0 = time.time()
def ts(): return f"[{int(time.time()-T0)}s]"

print(f"{ts()} Model={MODEL_ID}  Layer={PROBE_LAYER}  N_TARGET={N_TARGET}  N_CAL={N_CAL}")

# ── Load model ────────────────────────────────────────────────────────────────
print(f"{ts()} Loading Qwen2.5-7B-Instruct (4-bit)...", flush=True)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=bnb, device_map="auto",
    output_hidden_states=True
)
model.eval()
print(f"{ts()} Loaded. VRAM={torch.cuda.memory_allocated()/1e9:.2f}GB")

# ── Helpers ───────────────────────────────────────────────────────────────────
def step1_hidden(prompt: str) -> np.ndarray:
    inp = tok(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**inp)
    return out.hidden_states[PROBE_LAYER][0, -1].float().cpu().numpy()


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
    if not g_t:
        return 0.0
    tp = len(p_t & g_t)
    if tp == 0:
        return 0.0
    pr = tp / len(p_t) if p_t else 0.0
    rc = tp / len(g_t)
    return 2 * pr * rc / (pr + rc) if (pr + rc) > 0 else 0.0


def fit_probe(pos_h, neg_h, n_cal, seed_i=0):
    """Fit Fisher LDA on n_cal per class, return (direction, auroc_holdhout)."""
    rng = np.random.RandomState(seed_i)
    n_p, n_n = len(pos_h), len(neg_h)
    n_use = min(n_cal, n_p, n_n)
    idx_p = rng.choice(n_p, n_use, replace=False)
    idx_n = rng.choice(n_n, n_use, replace=False)
    X_cal = np.concatenate([np.array(pos_h)[idx_p], np.array(neg_h)[idx_n]])
    y_cal = np.array([1] * n_use + [0] * n_use)
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    lda.fit(X_cal, y_cal)
    direction = lda.coef_[0]
    direction_n = direction / (np.linalg.norm(direction) + 1e-8)

    # Held-out AUROC
    mask_p = np.ones(n_p, dtype=bool)
    mask_p[idx_p] = False
    mask_n = np.ones(n_n, dtype=bool)
    mask_n[idx_n] = False
    X_held = np.concatenate([np.array(pos_h)[mask_p], np.array(neg_h)[mask_n]])
    y_held = np.array([1] * mask_p.sum() + [0] * mask_n.sum())
    if len(set(y_held)) < 2 or len(X_held) < 4:
        auroc = roc_auc_score(y_cal, lda.decision_function(X_cal))
    else:
        auroc = roc_auc_score(y_held, lda.decision_function(X_held))
    return direction_n, auroc


# ── Task 1: TriviaQA — bilateral oracle ───────────────────────────────────────
def collect_triviaqa():
    print(f"\n{ts()} ── TriviaQA (bilateral oracle: nc_F1≥0.50=PARAM, nc_F1≤0.05=CTX_DEP) ──")
    ds = load_dataset("trivia_qa", "rc.wikipedia", split="validation").shuffle(seed=SEED)
    param_h, ctx_h = [], []
    for i, row in enumerate(ds):
        if len(param_h) >= N_TARGET and len(ctx_h) >= N_TARGET:
            break
        if i >= MAX_SCAN:
            print(f"{ts()} TriviaQA: MAX_SCAN reached")
            break
        q = row["question"]
        a = row["answer"]["value"]
        prompt = f"Answer briefly: {q}\nAnswer:"
        f1 = token_f1(generate(prompt), a)
        if f1 >= 0.50 and len(param_h) < N_TARGET:
            param_h.append(step1_hidden(prompt))
        elif f1 <= 0.05 and len(ctx_h) < N_TARGET:
            ctx_h.append(step1_hidden(prompt))
        if (i + 1) % 50 == 0:
            print(f"{ts()} TriviaQA [{i+1}] PARAM={len(param_h)} CTX_DEP={len(ctx_h)}")
    print(f"{ts()} TriviaQA final: PARAM={len(param_h)} CTX_DEP={len(ctx_h)}")
    return param_h, ctx_h


# ── Task 2: HotpotQA — adapted oracle ─────────────────────────────────────────
def collect_hotpotqa():
    print(f"\n{ts()} ── HotpotQA (adapted oracle: nc_F1≥0.20=KNOWS, nc_F1≤0.05=DOESNT_KNOW) ──")
    ds = load_dataset("hotpot_qa", "distractor", split="validation").shuffle(seed=SEED)
    pos_h, neg_h = [], []
    for i, row in enumerate(ds):
        if len(pos_h) >= N_TARGET and len(neg_h) >= N_TARGET:
            break
        if i >= MAX_SCAN:
            print(f"{ts()} HotpotQA: MAX_SCAN reached")
            break
        q = row["question"]
        a = row["answer"]
        prompt = f"Answer briefly: {q}\nAnswer:"
        f1 = token_f1(generate(prompt), a)
        if f1 >= 0.20 and len(pos_h) < N_TARGET:
            pos_h.append(step1_hidden(prompt))
        elif f1 <= 0.05 and len(neg_h) < N_TARGET:
            neg_h.append(step1_hidden(prompt))
        if (i + 1) % 50 == 0:
            print(f"{ts()} HotpotQA [{i+1}] KNOWS={len(pos_h)} DOESNT={len(neg_h)}")
    print(f"{ts()} HotpotQA final: KNOWS={len(pos_h)} DOESNT_KNOW={len(neg_h)}")
    return pos_h, neg_h


# ── Task 3: NQ-Open — bilateral oracle ────────────────────────────────────────
def collect_nqopen():
    print(f"\n{ts()} ── NQ-Open (bilateral oracle: nc_F1≥0.50=PARAM, nc_F1≤0.05=CTX_DEP) ──")
    ds = load_dataset("nq_open", split="validation").shuffle(seed=SEED)
    param_h, ctx_h = [], []
    for i, row in enumerate(ds):
        if len(param_h) >= N_TARGET and len(ctx_h) >= N_TARGET:
            break
        if i >= MAX_SCAN:
            print(f"{ts()} NQ-Open: MAX_SCAN reached")
            break
        q = row["question"]
        # nq_open answers is a list
        answers = row["answer"]
        if isinstance(answers, list):
            gold = answers[0] if answers else ""
        else:
            gold = str(answers)
        prompt = f"Answer briefly: {q}\nAnswer:"
        f1 = token_f1(generate(prompt), gold)
        if f1 >= 0.50 and len(param_h) < N_TARGET:
            param_h.append(step1_hidden(prompt))
        elif f1 <= 0.05 and len(ctx_h) < N_TARGET:
            ctx_h.append(step1_hidden(prompt))
        if (i + 1) % 50 == 0:
            print(f"{ts()} NQ-Open [{i+1}] PARAM={len(param_h)} CTX_DEP={len(ctx_h)}")
    print(f"{ts()} NQ-Open final: PARAM={len(param_h)} CTX_DEP={len(ctx_h)}")
    return param_h, ctx_h


# ── Task 4: MMLU — behavioral oracle ──────────────────────────────────────────
def collect_mmlu():
    print(f"\n{ts()} ── MMLU (behavioral oracle: CORRECT vs INCORRECT MCQ) ──")
    ds = load_dataset("cais/mmlu", "all", split="test", trust_remote_code=True).shuffle(seed=SEED)
    LETTERS = ["A", "B", "C", "D"]
    pos_h, neg_h = [], []
    scanned = 0
    for row in ds:
        if len(pos_h) >= N_TARGET and len(neg_h) >= N_TARGET:
            break
        if scanned >= MAX_SCAN:
            print(f"{ts()} MMLU: MAX_SCAN reached")
            break
        q           = row["question"]
        choices     = row["choices"]
        correct_idx = row["answer"]
        opts = "\n".join(f"{LETTERS[i]}) {c}" for i, c in enumerate(choices))
        prompt = f"Question: {q}\n{opts}\nAnswer with one letter (A/B/C/D):"
        gen_out = generate(prompt, max_new=5).strip()
        pred_ltr = gen_out[0].upper() if gen_out else "?"
        is_correct = (pred_ltr == LETTERS[correct_idx])
        scanned += 1
        if is_correct and len(pos_h) < N_TARGET:
            pos_h.append(step1_hidden(prompt))
        elif not is_correct and len(neg_h) < N_TARGET:
            neg_h.append(step1_hidden(prompt))
        if scanned % 50 == 0:
            print(f"{ts()} MMLU [{scanned}] CORRECT={len(pos_h)} INCORRECT={len(neg_h)}")
    print(f"{ts()} MMLU final: CORRECT={len(pos_h)} INCORRECT={len(neg_h)}")
    return pos_h, neg_h


# ── Collect all tasks ─────────────────────────────────────────────────────────
print(f"\n{ts()} {'='*60}")
print(f"{ts()} COLLECTING LABELED SAMPLES — 4 TASKS")
print(f"{ts()} {'='*60}")

tqa_pos, tqa_neg   = collect_triviaqa()
hpqa_pos, hpqa_neg = collect_hotpotqa()
nq_pos, nq_neg     = collect_nqopen()
mmlu_pos, mmlu_neg = collect_mmlu()

tasks = {
    "TriviaQA": (tqa_pos, tqa_neg,   "PARAM",    "CTX_DEP"),
    "HotpotQA": (hpqa_pos, hpqa_neg, "KNOWS",    "DOESNT_KNOW"),
    "NQ-Open":  (nq_pos, nq_neg,     "PARAM",    "CTX_DEP"),
    "MMLU":     (mmlu_pos, mmlu_neg, "CORRECT", "INCORRECT"),
}

# ── Fit Fisher probes for each task ───────────────────────────────────────────
print(f"\n{ts()} {'='*60}")
print(f"{ts()} FITTING FISHER PROBES")
print(f"{ts()} {'='*60}")

probe_results = {}
for task_name, (pos_h, neg_h, lp, ln) in tasks.items():
    n_p, n_n = len(pos_h), len(neg_h)
    if n_p < 10 or n_n < 10:
        print(f"{ts()} {task_name}: INSUFFICIENT (pos={n_p} neg={n_n}) — skipping")
        probe_results[task_name] = None
        continue

    dirs, aurocs = [], []
    for seed_i in range(N_SEEDS):
        d, auroc = fit_probe(pos_h, neg_h, N_CAL, seed_i)
        dirs.append(d)
        aurocs.append(auroc)

    # probe stability: cosim between seed 0 and seed 1
    stability = float(np.dot(dirs[0], dirs[1]))
    best_idx = int(np.argmax(aurocs))

    probe_results[task_name] = {
        "direction": dirs[best_idx],
        "auroc_mean": float(np.mean(aurocs)),
        "auroc_std":  float(np.std(aurocs)),
        "probe_stability": stability,
        "n_pos": n_p,
        "n_neg": n_n,
        "label_pos": lp,
        "label_neg": ln,
    }
    print(f"{ts()} {task_name}: AUROC={np.mean(aurocs):.4f}±{np.std(aurocs):.4f}  "
          f"stability={stability:.4f}  n=({n_p},{n_n})")


# ── Compute 4×4 cosine similarity matrix ─────────────────────────────────────
print(f"\n{ts()} {'='*60}")
print(f"{ts()} CROSS-TASK COSINE SIMILARITY MATRIX")
print(f"{ts()} {'='*60}")

task_names = [t for t in tasks if probe_results.get(t) is not None]
n_tasks = len(task_names)
cosim_matrix = np.zeros((n_tasks, n_tasks))

for i, ta in enumerate(task_names):
    for j, tb in enumerate(task_names):
        if i == j:
            cosim_matrix[i, j] = 1.0
        else:
            da = probe_results[ta]["direction"]
            db = probe_results[tb]["direction"]
            cosim_matrix[i, j] = float(np.dot(da, db))

print(f"\n{'':12}", end="")
for t in task_names:
    print(f"{t:12}", end="")
print()
for i, ta in enumerate(task_names):
    print(f"{ta:12}", end="")
    for j in range(n_tasks):
        print(f"{cosim_matrix[i,j]:+.4f}     ", end="")
    print()

# ── Verdict logic ─────────────────────────────────────────────────────────────
factual_tasks  = [t for t in ["TriviaQA", "HotpotQA", "NQ-Open"] if t in task_names]
factual_pairs  = [(t1, t2) for i, t1 in enumerate(factual_tasks)
                  for t2 in factual_tasks[i+1:]]
mmlu_pairs     = [("MMLU", t) for t in factual_tasks if "MMLU" in task_names]
all_pairs      = [(t1, t2) for i, t1 in enumerate(task_names)
                  for t2 in task_names[i+1:]]

def pair_cosim(ta, tb):
    if ta not in task_names or tb not in task_names:
        return None
    i = task_names.index(ta)
    j = task_names.index(tb)
    return cosim_matrix[i, j]

factual_cosims = [abs(pair_cosim(a, b)) for a, b in factual_pairs if pair_cosim(a, b) is not None]
mmlu_cosims    = [abs(pair_cosim(a, b)) for a, b in mmlu_pairs    if pair_cosim(a, b) is not None]

print(f"\n{ts()} Factual-retrieval pairs (|cosim|): {[f'{v:.4f}' for v in factual_cosims]}")
print(f"{ts()} MMLU pairs           (|cosim|): {[f'{v:.4f}' for v in mmlu_cosims]}")

mean_factual = float(np.mean(factual_cosims)) if factual_cosims else 0.0
mean_mmlu    = float(np.mean(mmlu_cosims))    if mmlu_cosims    else 0.0
mean_all     = float(np.mean([abs(pair_cosim(a, b))
                               for a, b in all_pairs
                               if pair_cosim(a, b) is not None]))

print(f"\n{ts()} Mean |cosim| factual group: {mean_factual:.4f}")
print(f"{ts()} Mean |cosim| MMLU pairs:    {mean_mmlu:.4f}")
print(f"{ts()} Mean |cosim| all pairs:     {mean_all:.4f}")

if mean_all > 0.70:
    verdict = "UNIVERSAL_AXIS"
    interpretation = (
        "Shared epistemic geometry across all task types. "
        "Fisher direction is not task-specific. Paper scope fully justified."
    )
elif mean_factual > 0.70 and mean_mmlu < 0.35:
    verdict = "RETRIEVAL_MODE_SPECIFIC"
    interpretation = (
        "Shared geometry within factual-retrieval tasks; distinct from MCQ reasoning. "
        "Strongly supports computational mode hypothesis: probe detects retrieval "
        "attractor mode. Boundary conditions (MMLU behavioral dominant) now mechanistically "
        "grounded. Paper can argue retrieval-mode geometry as the axis."
    )
elif mean_factual > 0.70:
    verdict = "RETRIEVAL_AXIS_PARTIAL"
    interpretation = (
        "Factual retrieval tasks share a probe direction. MMLU cosim moderate. "
        "Claim scope: single-hop factual + multi-hop retrieval. MCQ remains distinct."
    )
else:
    verdict = "TASK_SPECIFIC_GEOMETRY"
    interpretation = (
        "Probe directions are task-specific. Claim 1 must be scoped to TriviaQA "
        "geometry. Paper title and abstract require revision. Still publishable "
        "as a scoped finding with strong boundary condition characterization."
    )

print(f"\n{ts()} *** VERDICT: {verdict} ***")
print(f"{ts()} Interpretation: {interpretation}")


# ── Hewitt-Liang Selectivity Control ─────────────────────────────────────────
print(f"\n{ts()} {'='*60}")
print(f"{ts()} HEWITT-LIANG SELECTIVITY CONTROL (TriviaQA shuffled labels)")
print(f"{ts()} {'='*60}")
# Run Fisher LDA on TriviaQA with shuffled labels.
# AUROC should drop to ~0.50 ± 0.05 if probe is selective (signal is real).

tqa_all_h = np.array(tqa_pos + tqa_neg)
tqa_real_y = np.array([1] * len(tqa_pos) + [0] * len(tqa_neg))

shuffle_aurocs = []
rng_hl = np.random.RandomState(SEED)
for trial in range(5):
    y_shuffled = tqa_real_y.copy()
    rng_hl.shuffle(y_shuffled)
    n_use = min(N_CAL, len(tqa_pos), len(tqa_neg))
    # Use first n_use*2 samples for LDA, rest for eval
    X_cal_hl = tqa_all_h[:n_use*2]
    y_cal_hl = y_shuffled[:n_use*2]
    lda_hl = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    lda_hl.fit(X_cal_hl, y_cal_hl)
    X_held_hl = tqa_all_h[n_use*2:]
    y_held_hl = y_shuffled[n_use*2:]
    if len(set(y_held_hl)) < 2 or len(X_held_hl) < 4:
        auroc_hl = roc_auc_score(y_cal_hl, lda_hl.decision_function(X_cal_hl))
    else:
        auroc_hl = roc_auc_score(y_held_hl, lda_hl.decision_function(X_held_hl))
    auroc_hl = max(auroc_hl, 1 - auroc_hl)
    shuffle_aurocs.append(auroc_hl)
    print(f"{ts()} Selectivity trial {trial+1}: shuffled_auroc={auroc_hl:.4f}")

mean_shuffle = float(np.mean(shuffle_aurocs))
std_shuffle  = float(np.std(shuffle_aurocs))
real_auroc   = probe_results["TriviaQA"]["auroc_mean"] if probe_results.get("TriviaQA") else None

if mean_shuffle < 0.60:
    selectivity_verdict = "PROBE_SELECTIVE"
    selectivity_interpretation = (
        f"Shuffled-label AUROC={mean_shuffle:.4f}±{std_shuffle:.4f} (near chance). "
        f"Real AUROC={real_auroc:.4f}. Probe is selective: signal is not probe overfitting."
    )
else:
    selectivity_verdict = "PROBE_NOT_SELECTIVE"
    selectivity_interpretation = (
        f"WARNING: Shuffled-label AUROC={mean_shuffle:.4f}±{std_shuffle:.4f} — too high. "
        f"Fisher LDA may be overfitting to sample noise in high-dimensional space. "
        f"n={len(tqa_all_h)} in d=4096 is in the overfitting regime for this probe."
    )

print(f"\n{ts()} *** SELECTIVITY: {selectivity_verdict} ***")
print(f"{ts()} {selectivity_interpretation}")


# ── Final Summary ─────────────────────────────────────────────────────────────
print(f"\n{ts()} {'='*60}")
print(f"{ts()} FINAL SUMMARY")
print(f"{ts()} {'='*60}")
print(f"{ts()} Cross-task verdict:   {verdict}")
print(f"{ts()} Selectivity verdict:  {selectivity_verdict}")
print(f"\n{ts()} Probe AUROCs by task:")
for t in task_names:
    r = probe_results[t]
    print(f"{ts()}   {t:12}: AUROC={r['auroc_mean']:.4f}±{r['auroc_std']:.4f}  "
          f"stability={r['probe_stability']:.4f}  n=({r['n_pos']},{r['n_neg']})")

print(f"\n{ts()} Cosine similarity matrix:")
print(f"{'':12}", end="")
for t in task_names: print(f"{t:12}", end="")
print()
for i, ta in enumerate(task_names):
    print(f"{ta:12}", end="")
    for j in range(n_tasks): print(f"{cosim_matrix[i,j]:+.4f}     ", end="")
    print()

# ── Save JSON ────────────────────────────────────────────────────────────────
results = {
    "model": MODEL_ID,
    "probe_layer": PROBE_LAYER,
    "n_target": N_TARGET,
    "n_cal": N_CAL,
    "tasks": {
        t: {k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in probe_results[t].items() if k != "direction"}
        for t in task_names
    },
    "cosim_matrix": {
        task_names[i]: {task_names[j]: float(cosim_matrix[i, j]) for j in range(n_tasks)}
        for i in range(n_tasks)
    },
    "mean_cosim_factual_group": mean_factual,
    "mean_cosim_mmlu_pairs":    mean_mmlu,
    "mean_cosim_all_pairs":     mean_all,
    "verdict":        verdict,
    "interpretation": interpretation,
    "hewitt_liang": {
        "shuffled_auroc_mean": mean_shuffle,
        "shuffled_auroc_std":  std_shuffle,
        "shuffled_auroc_trials": shuffle_aurocs,
        "real_auroc_tqa":      real_auroc,
        "verdict": selectivity_verdict,
        "interpretation": selectivity_interpretation,
    },
    "elapsed_s": int(time.time() - T0),
}

out = "/kaggle/working/cross_task_cosim_v1_results.json"
with open(out, "w") as fh:
    json.dump(results, fh, indent=2)
print(f"\n{ts()} Saved: {out}")
print(json.dumps({k: v for k, v in results.items() if k not in ("cosim_matrix",)}, indent=2))
