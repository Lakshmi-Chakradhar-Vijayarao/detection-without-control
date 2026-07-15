#!/usr/bin/env python3
"""
exp_q14_difficulty_control.py — BILATERAL ORACLE DIFFICULTY CONTROL
====================================================================
SCIENTIFIC QUESTION (Q14, Tier 1 — MCE for Bilateral Oracle):
  Does the Fisher discriminant separate PARAM from CTX_DEP because PARAM items
  are intrinsically easier questions, or because they represent a genuinely different
  epistemic state?

ATTACK:
  "PARAM items are easy (the model already knows the answer). CTX_DEP items are
  hard (the model can't answer). Fisher is measuring question difficulty, not
  epistemic routing."

DESIGN:
  Run bilateral oracle with difficulty annotation on all labeled items.

  Difficulty proxy: with-context F1 (context_f1) — how easily ANY model can answer
  the question when given the passage. This is independent of whether the Qwen model
  has parametric knowledge of the answer.

  Collection (2 passes per item):
    1. No-context pass → text_nc, hs_nc, no_context_f1 (for labeling)
    2. With-context pass → context_f1 (for difficulty annotation, run on ALL labeled items)

  Analysis:
    A) Descriptive: context_f1 distributions for PARAM vs CTX_DEP
       - If distributions overlap substantially: difficulty skeptic attack is already weak
       - If PARAM has higher context_f1: matching is needed to rule out confound

    B) Difficulty-matched probe:
       - Bin items into Easy (context_f1 ≥ 0.80) and Medium (0.50 ≤ context_f1 < 0.80)
       - Within each tier: match N_tier PARAM and CTX_DEP items
       - Fit Fisher+PCA64 probe on matched set → AUROC per tier

    C) Within-PARAM internal control:
       - Easy PARAM (no_context_f1 ≥ 0.90) vs Hard PARAM (no_context_f1 ∈ [0.50, 0.70))
       - If probe separates within-PARAM by difficulty → artifact
       - If probe cannot separate (AUROC ≈ 0.50): probe is discriminating label, not difficulty

    D) Full unmatched probe (replicates C001)

VERDICT CRITERIA (pre-registered):
  H-EPISTEMIC:
    - AUROC(Easy tier) ≥ 0.65  AND
    - AUROC(Medium tier) ≥ 0.60  AND
    - AUROC(within-PARAM) ≤ 0.60  (no within-class difficulty signal)
  H-DIFFICULTY:
    - AUROC(matched) < 0.60  OR
    - AUROC drops > 0.15 from unmatched to matched  OR
    - AUROC(within-PARAM) ≥ 0.65  (difficulty signal exists within the probe)

MODEL: Qwen/Qwen2.5-1.5B-Instruct (same as C001)
LAYER: 26, Step 1
DATASET: TriviaQA rc.wikipedia (pool=12000 for sufficient matched items)
N_TARGET: 200/class (standard bilateral oracle)

GPU: T4 (~3-4h — double generation cost: with-context pass on all PARAM items)
"""

from __future__ import annotations
import gc, json, os, random, time
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score

# ── Config ─────────────────────────────────────────────────────────────────────
SEED          = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}", flush=True)
if DEVICE == "cpu":
    raise RuntimeError("GPU required. Exiting.")

MODEL_ID      = "Qwen/Qwen2.5-1.5B-Instruct"
LAYER_IDX     = 26
PCA_DIM       = 64
N_TARGET      = 200
TRAIN_FRAC    = 0.75
POOL_SIZE     = 12_000
N_BOOTSTRAP   = 1000
N_SHUFFLED    = 3
MAX_GEN       = 60
MAX_CTX       = 800
PARAM_MIN_F1  = 0.50
CTX_MAX_F1    = 0.05
CTX_MIN_CTX   = 0.50
N_TIER_MIN    = 30   # minimum per class per tier to run the probe
OUTPUT_FILE   = "/kaggle/working/exp_q14_difficulty_control_results.json"


# ── Token F1 ───────────────────────────────────────────────────────────────────
def token_f1(pred: str, gold_list: list[str]) -> float:
    pred_tokens = set(pred.lower().split())
    if not pred_tokens:
        return 0.0
    best = 0.0
    for gold in gold_list:
        gold_tokens = set(gold.lower().split())
        if not gold_tokens:
            continue
        common = len(pred_tokens & gold_tokens)
        if common == 0:
            continue
        p = common / len(pred_tokens)
        r = common / len(gold_tokens)
        f1 = 2 * p * r / (p + r)
        best = max(best, f1)
    return best


# ── HF token ───────────────────────────────────────────────────────────────────
def _get_hf_token():
    t = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if t:
        return t
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        return None


# ── Dataset ────────────────────────────────────────────────────────────────────
def load_triviaqa():
    print("Loading TriviaQA...", flush=True)
    ds = load_dataset("trivia_qa", "rc.wikipedia", split="validation", trust_remote_code=True)
    items = []
    for ex in ds:
        q = ex["question"]
        answers = ex["answer"]["aliases"] if ex["answer"]["aliases"] else [ex["answer"]["value"]]
        # Extract passage
        passage = ""
        if "search_results" in ex:
            for sr in ex.get("search_results", {}).get("search_context", []):
                if sr:
                    passage = sr
                    break
        if not passage:
            pages = ex.get("entity_pages", {}).get("wiki_context", [])
            if pages:
                passage = pages[0]
        items.append({"question": q, "answers": answers, "passage": passage})
    random.shuffle(items)
    print(f"  Loaded {len(items)} TriviaQA items", flush=True)
    return items


# ── Prompts ────────────────────────────────────────────────────────────────────
def fmt_nc(q: str, tok) -> str:
    msgs = [{"role": "user", "content": f"Answer this question in a few words: {q}"}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def fmt_wc(q: str, passage: str, tok) -> str:
    ctx = passage[:MAX_CTX]
    content = f"Answer this question using the provided context.\nContext: {ctx}\nQuestion: {q}"
    msgs = [{"role": "user", "content": content}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


# ── Model loading ──────────────────────────────────────────────────────────────
def load_model(hf_token):
    print(f"Loading {MODEL_ID}...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto", token=hf_token
    )
    model.eval()
    print(f"  Loaded. {sum(p.numel() for p in model.parameters())/1e9:.2f}B params", flush=True)
    return model, tok


# ── Step-1 hidden state extraction ────────────────────────────────────────────
@torch.no_grad()
def generate_and_extract(model, tok, prompt: str) -> tuple[str, np.ndarray | None]:
    """Returns (generated_text, step1_hidden_state_at_LAYER_IDX)."""
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    if inputs["input_ids"].shape[1] > 1900:
        return "", None
    out = model.generate(
        **inputs,
        max_new_tokens=MAX_GEN,
        do_sample=False,
        output_hidden_states=True,
        return_dict_in_generate=True,
        pad_token_id=tok.eos_token_id,
    )
    hs = None
    if out.hidden_states and len(out.hidden_states) > 1:
        step1_layers = out.hidden_states[1]
        if LAYER_IDX < len(step1_layers):
            hs = step1_layers[LAYER_IDX][0, -1, :].float().cpu().numpy()
    gen_ids = out.sequences[0][inputs["input_ids"].shape[1]:]
    text = tok.decode(gen_ids, skip_special_tokens=True).strip()
    return text, hs


# ── Collection pass ────────────────────────────────────────────────────────────
def collect_labeled_items(model, tok, items):
    """
    Run bilateral oracle with difficulty annotation.
    For every labeled item, record both no_context_f1 and context_f1.
    Returns: list of dicts with keys: label, hs, no_context_f1, context_f1
    """
    labeled = []
    n_param, n_ctxdep = 0, 0
    scanned = 0
    t0 = time.time()

    print(f"\nRunning bilateral oracle with difficulty annotation (pool={POOL_SIZE})...", flush=True)

    for item in items:
        if n_param >= N_TARGET and n_ctxdep >= N_TARGET:
            break
        if scanned >= POOL_SIZE:
            break
        scanned += 1

        q = item["question"]
        answers = item["answers"]
        passage = item.get("passage", "")

        # — No-context pass —
        prompt_nc = fmt_nc(q, tok)
        text_nc, hs_nc = generate_and_extract(model, tok, prompt_nc)
        if hs_nc is None:
            continue
        f1_nc = token_f1(text_nc, answers)

        # — PARAM candidate —
        if f1_nc >= PARAM_MIN_F1 and n_param < N_TARGET:
            # Run with-context pass for difficulty annotation
            context_f1 = float("nan")
            if passage:
                prompt_wc = fmt_wc(q, passage, tok)
                text_wc, _ = generate_and_extract(model, tok, prompt_wc)
                context_f1 = token_f1(text_wc, answers)
            else:
                context_f1 = f1_nc  # if no passage, use no_context_f1 as proxy
            labeled.append({
                "label": "PARAM",
                "hs": hs_nc,
                "no_context_f1": float(f1_nc),
                "context_f1": float(context_f1),
            })
            n_param += 1
            if scanned % 100 == 0 or n_param % 50 == 0:
                print(f"  scanned={scanned} PARAM={n_param} CTX_DEP={n_ctxdep} ({time.time()-t0:.0f}s)", flush=True)
            continue

        # — CTX_DEP candidate —
        if f1_nc <= CTX_MAX_F1 and passage and n_ctxdep < N_TARGET:
            prompt_wc = fmt_wc(q, passage, tok)
            text_wc, _ = generate_and_extract(model, tok, prompt_wc)
            f1_wc = token_f1(text_wc, answers)
            if f1_wc >= CTX_MIN_CTX:
                labeled.append({
                    "label": "CTX_DEP",
                    "hs": hs_nc,
                    "no_context_f1": float(f1_nc),
                    "context_f1": float(f1_wc),
                })
                n_ctxdep += 1
                if scanned % 100 == 0 or n_ctxdep % 50 == 0:
                    print(f"  scanned={scanned} PARAM={n_param} CTX_DEP={n_ctxdep} ({time.time()-t0:.0f}s)", flush=True)

    print(f"Done. PARAM={n_param}, CTX_DEP={n_ctxdep}, scanned={scanned}", flush=True)
    return labeled


# ── Fisher+PCA64 probe ─────────────────────────────────────────────────────────
def run_probe(X_pos: np.ndarray, X_neg: np.ndarray, label: str = "") -> dict:
    """Fit Fisher+PCA64 on 75% train, evaluate on 25% test. Bootstrap CI, shuffled control."""
    n = min(len(X_pos), len(X_neg))
    if n < N_TIER_MIN:
        return {"auroc": float("nan"), "n": n, "note": f"insufficient ({n} < {N_TIER_MIN})"}

    rng = np.random.default_rng(SEED)
    idx_pos = rng.permutation(len(X_pos))[:n]
    idx_neg = rng.permutation(len(X_neg))[:n]
    X_pos_s = X_pos[idx_pos]
    X_neg_s = X_neg[idx_neg]

    n_train = int(n * TRAIN_FRAC)
    X_train = np.vstack([X_pos_s[:n_train], X_neg_s[:n_train]])
    y_train = np.array([1]*n_train + [0]*n_train)
    X_test  = np.vstack([X_pos_s[n_train:], X_neg_s[n_train:]])
    y_test  = np.array([1]*(n - n_train) + [0]*(n - n_train))

    pca = PCA(n_components=PCA_DIM, random_state=SEED)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca  = pca.transform(X_test)

    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    lda.fit(X_train_pca, y_train)
    scores = lda.decision_function(X_test_pca)
    auroc_point = float(roc_auc_score(y_test, scores))

    # Bootstrap CI
    boot = []
    X_all_pca = np.vstack([X_train_pca, X_test_pca])
    y_all = np.concatenate([y_train, y_test])
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, len(y_all), len(y_all))
        if len(np.unique(y_all[idx])) < 2:
            continue
        pca_b = PCA(n_components=PCA_DIM, random_state=SEED)
        X_b = pca_b.fit_transform(X_all_pca[idx])
        lda_b = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        lda_b.fit(X_b, y_all[idx])
        scores_b = lda_b.decision_function(X_all_pca)
        boot.append(float(roc_auc_score(y_all, scores_b)))
    ci_low, ci_high = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

    # Shuffled controls
    shuf_aurocs = []
    for seed_s in range(N_SHUFFLED):
        rng_s = np.random.default_rng(SEED + seed_s + 1)
        y_shuf = y_train.copy()
        rng_s.shuffle(y_shuf)
        pca_s = PCA(n_components=PCA_DIM, random_state=SEED)
        Xt_s = pca_s.fit_transform(X_train)
        Xe_s = pca_s.transform(X_test)
        lda_s = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        lda_s.fit(Xt_s, y_shuf)
        sc_s = lda_s.decision_function(Xe_s)
        shuf_aurocs.append(float(roc_auc_score(y_test, sc_s)))

    result = {
        "label": label,
        "n_per_class": n,
        "auroc": round(auroc_point, 4),
        "ci_95_low": round(ci_low, 4),
        "ci_95_high": round(ci_high, 4),
        "shuffled_auroc_mean": round(float(np.mean(shuf_aurocs)), 4),
        "shuffled_auroc_max": round(float(np.max(shuf_aurocs)), 4),
        "clean": auroc_point > 0.55 and float(np.max(shuf_aurocs)) < 0.60,
    }
    print(f"  [{label}] AUROC={auroc_point:.4f} CI=[{ci_low:.4f},{ci_high:.4f}]  "
          f"shuffled_max={float(np.max(shuf_aurocs)):.4f}  n={n}", flush=True)
    return result


# ── Difficulty-matched analysis ────────────────────────────────────────────────
def difficulty_matched_analysis(labeled: list[dict]) -> dict:
    """Run probes on difficulty-matched subsets."""
    param_items  = [x for x in labeled if x["label"] == "PARAM"]
    ctxdep_items = [x for x in labeled if x["label"] == "CTX_DEP"]

    # ─ Descriptive stats ─
    param_cf1  = [x["context_f1"] for x in param_items  if not np.isnan(x["context_f1"])]
    ctxdep_cf1 = [x["context_f1"] for x in ctxdep_items if not np.isnan(x["context_f1"])]
    param_nf1  = [x["no_context_f1"] for x in param_items]
    ctxdep_nf1 = [x["no_context_f1"] for x in ctxdep_items]

    descriptive = {
        "param":  {
            "n": len(param_items),
            "context_f1_mean": float(np.mean(param_cf1))  if param_cf1  else float("nan"),
            "context_f1_std":  float(np.std(param_cf1))   if param_cf1  else float("nan"),
            "context_f1_median": float(np.median(param_cf1)) if param_cf1 else float("nan"),
            "no_context_f1_mean": float(np.mean(param_nf1)),
        },
        "ctxdep": {
            "n": len(ctxdep_items),
            "context_f1_mean": float(np.mean(ctxdep_cf1)) if ctxdep_cf1 else float("nan"),
            "context_f1_std":  float(np.std(ctxdep_cf1))  if ctxdep_cf1 else float("nan"),
            "context_f1_median": float(np.median(ctxdep_cf1)) if ctxdep_cf1 else float("nan"),
            "no_context_f1_mean": float(np.mean(ctxdep_nf1)),
        },
    }
    mean_diff = descriptive["param"]["context_f1_mean"] - descriptive["ctxdep"]["context_f1_mean"]
    descriptive["context_f1_mean_diff_PARAM_minus_CTXDEP"] = round(float(mean_diff), 4)
    print(f"\nDescriptive: PARAM context_f1={descriptive['param']['context_f1_mean']:.3f}  "
          f"CTX_DEP context_f1={descriptive['ctxdep']['context_f1_mean']:.3f}  "
          f"diff={mean_diff:.3f}", flush=True)

    # ─ Tier definitions ─
    tier_results = {}

    def items_in_tier(items_list, low, high):
        return [x for x in items_list if not np.isnan(x["context_f1"]) and low <= x["context_f1"] < high]

    tiers = [
        ("Easy",   0.80, 1.01),
        ("Medium", 0.50, 0.80),
    ]

    for tier_name, low, high in tiers:
        param_t  = items_in_tier(param_items,  low, high)
        ctxdep_t = items_in_tier(ctxdep_items, low, high)
        n_use    = min(len(param_t), len(ctxdep_t))
        print(f"\n  Tier '{tier_name}' (context_f1 ∈ [{low},{high})): "
              f"PARAM={len(param_t)}, CTX_DEP={len(ctxdep_t)}, using={n_use}", flush=True)

        if n_use < N_TIER_MIN:
            tier_results[tier_name] = {
                "n_param": len(param_t),
                "n_ctxdep": len(ctxdep_t),
                "auroc": float("nan"),
                "note": f"insufficient (min {N_TIER_MIN} required per class)",
            }
            continue

        rng = np.random.default_rng(SEED)
        p_sel = rng.permutation(len(param_t))[:n_use]
        c_sel = rng.permutation(len(ctxdep_t))[:n_use]

        X_param  = np.stack([param_t[i]["hs"]  for i in p_sel])
        X_ctxdep = np.stack([ctxdep_t[i]["hs"] for i in c_sel])

        probe_r = run_probe(X_param, X_ctxdep, label=f"matched_{tier_name}")
        tier_results[tier_name] = {
            "n_param": len(param_t),
            "n_ctxdep": len(ctxdep_t),
            **probe_r,
        }

    return descriptive, tier_results


# ── Within-PARAM difficulty test ───────────────────────────────────────────────
def within_param_difficulty_test(labeled: list[dict]) -> dict:
    """
    Test if the Fisher probe separates easy PARAM from hard PARAM.
    These have the same label but different difficulty.
    If AUROC ≈ 0.50: probe is not capturing difficulty signal.
    If AUROC >> 0.50: probe IS capturing difficulty, not just label.
    """
    param_items = [x for x in labeled if x["label"] == "PARAM"]
    easy_param = [x for x in param_items if x["no_context_f1"] >= 0.90]
    hard_param = [x for x in param_items if x["no_context_f1"] < 0.70 and x["no_context_f1"] >= 0.50]

    print(f"\nWithin-PARAM difficulty test: easy(≥0.90)={len(easy_param)}, "
          f"hard([0.50,0.70))={len(hard_param)}", flush=True)

    if min(len(easy_param), len(hard_param)) < N_TIER_MIN:
        return {
            "n_easy": len(easy_param),
            "n_hard": len(hard_param),
            "auroc": float("nan"),
            "note": "insufficient items",
        }

    X_easy = np.stack([x["hs"] for x in easy_param])
    X_hard = np.stack([x["hs"] for x in hard_param])
    probe_r = run_probe(X_easy, X_hard, label="within_PARAM_easy_vs_hard")
    return {
        "n_easy": len(easy_param),
        "n_hard": len(hard_param),
        "interpretation": (
            "ARTIFACT: probe separates difficulty within PARAM — difficulty confound exists"
            if probe_r.get("auroc", 0) >= 0.65
            else "CLEAN: probe does not separate easy vs hard PARAM — not a difficulty detector"
        ),
        **probe_r,
    }


# ── Verdict ────────────────────────────────────────────────────────────────────
def classify_verdict(full_probe, tier_results, within_param) -> str:
    full_auroc  = full_probe.get("auroc", 0.0)
    easy_auroc  = tier_results.get("Easy",   {}).get("auroc", float("nan"))
    med_auroc   = tier_results.get("Medium", {}).get("auroc", float("nan"))
    wp_auroc    = within_param.get("auroc", float("nan"))

    matched_aurocs = [a for a in [easy_auroc, med_auroc] if not np.isnan(a)]

    if not matched_aurocs:
        return "INSUFFICIENT_DATA"

    min_matched = min(matched_aurocs)
    drop = full_auroc - min_matched if not np.isnan(min_matched) else float("nan")

    # H-EPISTEMIC: all tiers ≥ 0.65, within-PARAM ≤ 0.60, drop ≤ 0.15
    h_ep = (
        min_matched >= 0.65
        and (np.isnan(wp_auroc) or wp_auroc <= 0.60)
        and (np.isnan(drop) or drop <= 0.15)
    )
    # H-DIFFICULTY: any tier < 0.60 OR drop > 0.15 OR within-PARAM ≥ 0.65
    h_diff = (
        min_matched < 0.60
        or (not np.isnan(drop) and drop > 0.15)
        or (not np.isnan(wp_auroc) and wp_auroc >= 0.65)
    )

    if h_ep:
        return "H_EPISTEMIC_SUPPORTED"
    if h_diff:
        return "H_DIFFICULTY_SUPPORTED"
    return "INCONCLUSIVE"


# ── Intermediate save ──────────────────────────────────────────────────────────
def _save(data):
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(data, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70, flush=True)
    print("EXP_Q14 — DIFFICULTY CONTROL EXPERIMENT", flush=True)
    print("Q14: Does Fisher discriminate epistemic routing or question difficulty?", flush=True)
    print("=" * 70, flush=True)

    hf_token = _get_hf_token()
    items    = load_triviaqa()
    model, tok = load_model(hf_token)

    # — Phase 1: Collection —
    labeled = collect_labeled_items(model, tok, items)

    n_param  = sum(1 for x in labeled if x["label"] == "PARAM")
    n_ctxdep = sum(1 for x in labeled if x["label"] == "CTX_DEP")
    print(f"\nCollection complete: {n_param} PARAM, {n_ctxdep} CTX_DEP", flush=True)

    # Intermediate save after collection
    _save({"phase": "collection_complete", "n_param": n_param, "n_ctxdep": n_ctxdep,
           "items_count": len(labeled)})

    if min(n_param, n_ctxdep) < N_TIER_MIN * 2:
        print("ERROR: Insufficient labeled items. Exiting.", flush=True)
        return

    # Save hidden states to disk so Phase 2+ can be recovered if it crashes
    X_param  = np.stack([x["hs"] for x in labeled if x["label"] == "PARAM"])
    X_ctxdep = np.stack([x["hs"] for x in labeled if x["label"] == "CTX_DEP"])
    np.save("/kaggle/working/q14_X_param.npy",  X_param)
    np.save("/kaggle/working/q14_X_ctxdep.npy", X_ctxdep)
    print(f"  Hidden states saved: X_param={X_param.shape} X_ctxdep={X_ctxdep.shape}", flush=True)

    # — Phase 2: Full unmatched probe (replicates C001) —
    print("\n" + "-" * 50, flush=True)
    print("Phase 2: Full unmatched probe (C001 replication)...", flush=True)
    full_probe = run_probe(X_param, X_ctxdep, label="full_unmatched")

    # — Phase 3: Difficulty-matched analysis —
    print("\n" + "-" * 50, flush=True)
    print("Phase 3: Difficulty-matched probes by context_f1 tier...", flush=True)
    descriptive, tier_results = difficulty_matched_analysis(labeled)

    # — Phase 4: Within-PARAM difficulty test —
    print("\n" + "-" * 50, flush=True)
    print("Phase 4: Within-PARAM difficulty test...", flush=True)
    within_param = within_param_difficulty_test(labeled)

    # — Phase 5: Verdict —
    verdict = classify_verdict(full_probe, tier_results, within_param)

    final = {
        "experiment": "EXP_Q14_DIFFICULTY_CONTROL",
        "date": time.strftime("%Y-%m-%d"),
        "model": MODEL_ID,
        "layer": LAYER_IDX,
        "n_labeled": len(labeled),
        "n_param": n_param,
        "n_ctxdep": n_ctxdep,
        "full_unmatched_probe": full_probe,
        "descriptive_statistics": descriptive,
        "difficulty_matched_probes": tier_results,
        "within_param_difficulty_test": within_param,
        "verdict": verdict,
        "verdict_criteria": {
            "H_EPISTEMIC": "all tiers AUROC ≥ 0.65 AND within-PARAM AUROC ≤ 0.60 AND drop ≤ 0.15",
            "H_DIFFICULTY": "any tier AUROC < 0.60 OR drop > 0.15 OR within-PARAM ≥ 0.65",
        },
    }

    _save(final)

    # — Summary —
    print("\n" + "=" * 70, flush=True)
    print("RESULTS SUMMARY", flush=True)
    print("=" * 70, flush=True)
    print(f"  Full unmatched AUROC: {full_probe.get('auroc', 'nan'):.4f}  "
          f"(replicates C001={0.7312})", flush=True)
    print(f"\n  Difficulty descriptive:", flush=True)
    print(f"    PARAM context_f1:   mean={descriptive['param']['context_f1_mean']:.3f}  "
          f"std={descriptive['param']['context_f1_std']:.3f}", flush=True)
    print(f"    CTX_DEP context_f1: mean={descriptive['ctxdep']['context_f1_mean']:.3f}  "
          f"std={descriptive['ctxdep']['context_f1_std']:.3f}", flush=True)
    print(f"    mean diff (PARAM - CTX_DEP): "
          f"{descriptive['context_f1_mean_diff_PARAM_minus_CTXDEP']:.3f}", flush=True)
    print(f"\n  Difficulty-matched probes:", flush=True)
    for tier, tr in tier_results.items():
        auroc_str = f"{tr['auroc']:.4f}" if not np.isnan(tr.get("auroc", float("nan"))) else "N/A"
        print(f"    {tier}: AUROC={auroc_str}  n={tr.get('n_param',0)}/{tr.get('n_ctxdep',0)}", flush=True)
    print(f"\n  Within-PARAM test AUROC: "
          f"{within_param.get('auroc', float('nan')):.4f}  "
          f"n_easy={within_param.get('n_easy',0)} n_hard={within_param.get('n_hard',0)}", flush=True)
    print(f"    → {within_param.get('interpretation', 'N/A')}", flush=True)
    print(f"\n  VERDICT: {verdict}", flush=True)
    print(f"  Output: {OUTPUT_FILE}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
