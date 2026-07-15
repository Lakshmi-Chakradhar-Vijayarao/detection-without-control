#!/usr/bin/env python3
"""
exp_a_law3_llama.py — LAW 3 CROSS-FAMILY REPLICATION (LLAMA BACKBONE)
=======================================================================
SCIENTIFIC QUESTION (Q4, Tier 2 — Laws):
  Does Law 3 replicate on the Llama backbone?
  - L1 (bilateral oracle, TriviaQA): INVERTED_U or different pattern?
  - L2 (entropy-matched CO labeling): MONOTONE_RISE or different pattern?

PRE-REGISTRATION: science/preregistration/exp_a_law3_llama.md
COMPETING THEORIES: docs/COMPETING_THEORIES.md

VERDICT CRITERIA (locked before running — from pre-registration):
  L1 INVERTED_U:     INSTRUCT > BASE by > 0.05 AND REASONING < INSTRUCT by > 0.03
  L1 MONOTONE_RISE:  BASE < INSTRUCT < REASONING, each step > 0.03
  L1 FLAT:           max − min < 0.05

STAGES:
  Stage 1 BASE:      meta-llama/Llama-3.2-3B         (pure pretrain, 28 layers)
  Stage 2 INSTRUCT:  meta-llama/Llama-3.2-3B-Instruct (SFT + RLHF, 28 layers)
  Stage 3 REASONING: deepseek-ai/DeepSeek-R1-Distill-Llama-8B (reasoning-distilled, 32 layers)
    ⚠ SIZE CONFOUND: Stage 3 is 8B vs 3B for Stages 1-2. Directional comparison is valid;
      absolute comparison is confounded. Layer index adjusted proportionally (L30/32).

GPU: T4 — Stages 1+2 in session A (~6h), Stage 3 in session B (~5h, 4-bit quantization)
"""

from __future__ import annotations
import subprocess
subprocess.run(["pip", "install", "-q", "bitsandbytes>=0.46.1"], check=False)

import gc, json, os, random, time
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score

# ── Config ────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
if DEVICE == "cpu":
    raise RuntimeError("GPU required for this experiment. Exiting.")

LAYER_IDX_DEFAULT  = 26      # for 28-layer models (Llama-3.2-3B)
PCA_DIM            = 64
N_TARGET           = 200     # per class, matched across all stages
TRAIN_FRAC         = 0.75
POOL_SIZE          = 10_000
N_BOOTSTRAP        = 1000
N_SHUFFLED_SEEDS   = 3
MAX_GEN            = 60
MAX_CTX            = 800
PARAM_MIN_F1       = 0.50
CTX_MAX_F1         = 0.05
CTX_MIN_CTX        = 0.50
THETA_CONF_L2      = 0.50    # entropy ceiling for CO labeling (L2); 0.15 was too tight (CC=0 on Llama-Instruct)
N_TARGET_L2        = 100
OUTPUT_FILE        = "/kaggle/working/exp_a_law3_llama_results.json"

STAGES = [
    {
        "name":       "BASE",
        "model_id":   "meta-llama/Llama-3.2-3B",
        "n_layers":   28,
        "layer_idx":  26,         # L26/28
        "is_instruct": False,
        "use_4bit":   False,
    },
    {
        "name":       "INSTRUCT",
        "model_id":   "meta-llama/Llama-3.2-3B-Instruct",
        "n_layers":   28,
        "layer_idx":  26,         # L26/28
        "is_instruct": True,
        "use_4bit":   False,
    },
    {
        "name":       "REASONING",
        "model_id":   "meta-llama/Llama-3.1-8B-Instruct",
        "n_layers":   32,
        "layer_idx":  30,         # proportional: round(26/28 * 32) = 30
        "is_instruct": True,
        "use_4bit":   True,       # 8B requires 4-bit on T4
        "size_confound_note": "8B vs 3B for INSTRUCT — directional comparison only. DeepSeek-R1-Distill-Llama-8B excluded: CoT <think> tokens fill MAX_GEN=60 budget, bilateral oracle yields PARAM=0 (same protocol failure as BASE/C026).",
    },
]

# Run only specified stages (set to ["BASE", "INSTRUCT"] for session A, ["REASONING"] for session B)
# BASE excluded: pure base Llama does not follow Q/A format reliably (C026 — protocol requires
# instruction-following capability). Bilateral oracle yield on base Llama ≈ 0% PARAM.
# INSTRUCT vs REASONING comparison fully discriminates H-A / H-B / H-C.
STAGES_TO_RUN = os.environ.get("STAGES_TO_RUN", "INSTRUCT,REASONING").split(",")


# ── HF token ─────────────────────────────────────────────────────────────────
def _get_hf_token():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        return None


# ── Dataset ───────────────────────────────────────────────────────────────────
def load_triviaqa():
    print("Loading TriviaQA...")
    ds = load_dataset("trivia_qa", "rc.wikipedia", split="validation", trust_remote_code=True)
    items = []
    for ex in ds:
        q = ex["question"]
        answers = ex["answer"]["aliases"] if ex["answer"]["aliases"] else [ex["answer"]["value"]]
        # Extract passage for CTX_DEP labeling
        passage = ""
        if ex.get("entity_pages", {}).get("wiki_context"):
            passage = ex["entity_pages"]["wiki_context"][0]
        if not passage:
            for sr in ex.get("search_results", {}).get("search_context", []):
                if sr:
                    passage = sr
                    break
        items.append({"question": q, "answers": answers, "passage": passage})
    random.shuffle(items)
    print(f"  Loaded {len(items)} TriviaQA items")
    return items


# ── F1 scoring ────────────────────────────────────────────────────────────────
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


# ── Prompt formatting ─────────────────────────────────────────────────────────
def format_nocontext(question: str, is_instruct: bool, tokenizer) -> str:
    if is_instruct:
        msgs = [{"role": "user", "content": f"Answer this question in a few words: {question}"}]
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return f"Q: {question}\nA:"


def format_withcontext(question: str, passage: str, is_instruct: bool, tokenizer) -> str:
    ctx = passage[:MAX_CTX] if len(passage) > MAX_CTX else passage
    if is_instruct:
        content = f"Answer this question using the provided context.\nContext: {ctx}\nQuestion: {question}"
        msgs = [{"role": "user", "content": content}]
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return f"Context: {ctx}\nQ: {question}\nA:"


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model(stage: dict, hf_token: str):
    model_id = stage["model_id"]
    use_4bit = stage.get("use_4bit", False)
    print(f"\nLoading {model_id} (4-bit={use_4bit})...")

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_4bit:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_cfg,
            device_map="auto",
            token=hf_token,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            token=hf_token,
        )
    model.eval()
    print(f"  Model loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")
    return model, tokenizer


# ── Generation + hidden state extraction ─────────────────────────────────────
@torch.no_grad()
def generate_and_extract(model, tokenizer, prompt: str, layer_idx: int) -> tuple[str, np.ndarray | None, float]:
    """Returns (generated_text, hidden_state_at_step1, output_entropy)."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    if inputs["input_ids"].shape[1] > 1800:
        return "", None, float("nan")

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_GEN,
            do_sample=False,
            output_hidden_states=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Extract hidden state at step 1 (first new token), specified layer
    hs = None
    if out.hidden_states and len(out.hidden_states) > 1:
        step1_layers = out.hidden_states[1]  # step index 1
        if layer_idx < len(step1_layers):
            hs_tensor = step1_layers[layer_idx][0, -1, :]  # [hidden_dim]
            hs = hs_tensor.float().cpu().numpy()

    # Output entropy from logits at step 1
    entropy = float("nan")
    if out.scores and len(out.scores) > 0:
        logits = out.scores[0][0]  # [vocab_size]
        probs = torch.softmax(logits.float(), dim=-1)
        log_probs = torch.log(probs + 1e-10)
        entropy = float(-torch.sum(probs * log_probs).item())

    # Decode only new tokens
    gen_ids = out.sequences[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return text, hs, entropy


# ── Bilateral oracle labeling ─────────────────────────────────────────────────
def run_bilateral_oracle(model, tokenizer, items, stage, n_target):
    """Run bilateral oracle on items, return labeled hidden states."""
    layer_idx = stage["layer_idx"]
    is_instruct = stage["is_instruct"]
    print(f"\n  Running bilateral oracle (layer={layer_idx}, n_target={n_target}/class)...")

    param_hs, ctxdep_hs = [], []
    scanned = 0
    t0 = time.time()

    for item in items:
        if len(param_hs) >= n_target and len(ctxdep_hs) >= n_target:
            break
        if scanned >= POOL_SIZE:
            break
        scanned += 1

        q = item["question"]
        answers = item["answers"]
        passage = item.get("passage", "")

        # No-context pass
        prompt_nc = format_nocontext(q, is_instruct, tokenizer)
        text_nc, hs_nc, _ = generate_and_extract(model, tokenizer, prompt_nc, layer_idx)
        f1_nc = token_f1(text_nc, answers)

        if f1_nc >= PARAM_MIN_F1 and len(param_hs) < n_target and hs_nc is not None:
            param_hs.append(hs_nc)
            continue

        if f1_nc <= CTX_MAX_F1 and passage:
            prompt_wc = format_withcontext(q, passage, is_instruct, tokenizer)
            text_wc, _, _ = generate_and_extract(model, tokenizer, prompt_wc, layer_idx)
            f1_wc = token_f1(text_wc, answers)
            if f1_wc >= CTX_MIN_CTX and len(ctxdep_hs) < n_target and hs_nc is not None:
                ctxdep_hs.append(hs_nc)

        if scanned % 200 == 0:
            elapsed = time.time() - t0
            print(f"    scanned={scanned}  PARAM={len(param_hs)}  CTX_DEP={len(ctxdep_hs)}  ({elapsed:.0f}s)")

    print(f"  Done. PARAM={len(param_hs)}, CTX_DEP={len(ctxdep_hs)}, scanned={scanned}")
    return np.array(param_hs), np.array(ctxdep_hs)


# ── CO labeling for L2 ────────────────────────────────────────────────────────
def run_co_labeling(model, tokenizer, items, stage, n_target):
    """Correct vs Wrong labeling with entropy ≤ theta_conf (entropy-matched)."""
    layer_idx = stage["layer_idx"]
    is_instruct = stage["is_instruct"]
    print(f"\n  Running CO labeling for L2 (theta_conf={THETA_CONF_L2}, n_target={n_target}/class)...")

    cc_hs, cw_hs = [], []
    scanned = 0
    t0 = time.time()

    for item in items:
        if len(cc_hs) >= n_target and len(cw_hs) >= n_target:
            break
        if scanned >= POOL_SIZE:
            break
        scanned += 1

        q = item["question"]
        answers = item["answers"]
        prompt = format_nocontext(q, is_instruct, tokenizer)
        text, hs, entropy = generate_and_extract(model, tokenizer, prompt, layer_idx)

        if hs is None or np.isnan(entropy):
            continue
        if entropy > THETA_CONF_L2:
            continue  # not in confident zone

        f1 = token_f1(text, answers)
        if f1 >= PARAM_MIN_F1 and len(cc_hs) < n_target:
            cc_hs.append(hs)
        elif f1 <= CTX_MAX_F1 and len(cw_hs) < n_target:
            cw_hs.append(hs)

        if scanned % 200 == 0:
            print(f"    scanned={scanned}  CC={len(cc_hs)}  CW={len(cw_hs)}  ({time.time()-t0:.0f}s)")

    print(f"  Done. CC={len(cc_hs)}, CW={len(cw_hs)}, scanned={scanned}")
    return np.array(cc_hs) if cc_hs else np.zeros((0, 1)), np.array(cw_hs) if cw_hs else np.zeros((0, 1))


# ── Fisher+PCA64 probe ────────────────────────────────────────────────────────
def fit_probe(X_pos, X_neg, seed=SEED):
    """Fit Fisher+PCA64 probe. Returns AUROC on held-out test set."""
    n = min(len(X_pos), len(X_neg))
    if n < 20:
        return float("nan"), 0, 0

    rng = np.random.default_rng(seed)
    idx_pos = rng.permutation(len(X_pos))[:n]
    idx_neg = rng.permutation(len(X_neg))[:n]
    X_pos_s, X_neg_s = X_pos[idx_pos], X_neg[idx_neg]

    n_train = int(n * TRAIN_FRAC)
    X_train = np.vstack([X_pos_s[:n_train], X_neg_s[:n_train]])
    y_train = np.array([1] * n_train + [0] * n_train)
    X_test  = np.vstack([X_pos_s[n_train:], X_neg_s[n_train:]])
    y_test  = np.array([1] * (n - n_train) + [0] * (n - n_train))

    pca = PCA(n_components=min(PCA_DIM, X_train.shape[1] - 1), random_state=seed)
    X_train_p = pca.fit_transform(X_train)
    X_test_p  = pca.transform(X_test)

    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    lda.fit(X_train_p, y_train)
    scores = lda.decision_function(X_test_p)
    auroc = roc_auc_score(y_test, scores)
    return auroc, n_train, len(y_test)


def bootstrap_auroc(X_pos, X_neg, n_boot=N_BOOTSTRAP, seed=SEED):
    rng = np.random.default_rng(seed)
    n = min(len(X_pos), len(X_neg))
    if n < 20:
        return float("nan"), float("nan"), float("nan")

    boot_aurocs = []
    for i in range(n_boot):
        idx_p = rng.integers(0, len(X_pos), n)
        idx_n = rng.integers(0, len(X_neg), n)
        a, _, _ = fit_probe(X_pos[idx_p], X_neg[idx_n], seed=int(rng.integers(0, 2**31)))
        if not np.isnan(a):
            boot_aurocs.append(a)

    if len(boot_aurocs) < 10:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(boot_aurocs)), float(np.percentile(boot_aurocs, 2.5)), float(np.percentile(boot_aurocs, 97.5))


def shuffled_auroc(X_pos, X_neg, n_seeds=N_SHUFFLED_SEEDS):
    rng = np.random.default_rng(SEED)
    aurocs = []
    for _ in range(n_seeds):
        X_all = np.vstack([X_pos, X_neg])
        y_all = np.array([1] * len(X_pos) + [0] * len(X_neg))
        rng.shuffle(y_all)
        n = min(len(X_pos), len(X_neg))
        a, _, _ = fit_probe(X_all[y_all == 1][:n], X_all[y_all == 0][:n], seed=int(rng.integers(0, 2**31)))
        if not np.isnan(a):
            aurocs.append(a)
    return float(np.mean(aurocs)) if aurocs else float("nan")


# ── Verdict classification ────────────────────────────────────────────────────
def classify_l1_verdict(stage_results: dict) -> str:
    """Apply pre-registered verdict criteria to L1 AUROCs.
    Handles both 3-stage (BASE+INSTRUCT+REASONING) and 2-stage (INSTRUCT+REASONING) cases.
    BASE excluded when bilateral oracle yield ≈ 0% (C026).
    """
    has_base = "BASE" in stage_results
    has_instruct = "INSTRUCT" in stage_results
    has_reasoning = "REASONING" in stage_results

    if not has_instruct or not has_reasoning:
        return "INCOMPLETE"

    instruct  = stage_results["INSTRUCT"]["L1"].get("auroc", float("nan"))
    reasoning = stage_results["REASONING"]["L1"].get("auroc", float("nan"))
    if np.isnan(instruct) or np.isnan(reasoning):
        return "INCOMPLETE"

    if has_base:
        base = stage_results["BASE"]["L1"].get("auroc", float("nan"))
        if np.isnan(base):
            return "INCOMPLETE"
        rng = max(base, instruct, reasoning) - min(base, instruct, reasoning)
        if rng < 0.05:
            return "FLAT"
        if (instruct - base > 0.05) and (instruct - reasoning > 0.03):
            return "INVERTED_U"
        if (instruct > base + 0.03) and (reasoning > instruct + 0.03):
            return "MONOTONE_RISE"
        return "MIXED"
    else:
        # 2-stage: INSTRUCT vs REASONING only (BASE excluded due to C026)
        delta = instruct - reasoning
        if abs(delta) < 0.05:
            return "FLAT_2STAGE"
        if delta > 0.03:
            return "INSTRUCT_PEAK"   # INSTRUCT > REASONING — consistent with INVERTED_U
        if delta < -0.03:
            return "MONOTONE_RISE"   # REASONING > INSTRUCT
        return "MIXED"


def classify_l2_verdict(stage_results: dict) -> str:
    has_base = "BASE" in stage_results
    has_instruct = "INSTRUCT" in stage_results
    has_reasoning = "REASONING" in stage_results

    if not has_instruct or not has_reasoning:
        return "INCOMPLETE"

    instruct  = stage_results["INSTRUCT"]["L2"].get("auroc", float("nan"))
    reasoning = stage_results["REASONING"]["L2"].get("auroc", float("nan"))
    if np.isnan(instruct) or np.isnan(reasoning):
        return "INCOMPLETE_L2"

    if has_base:
        base = stage_results["BASE"]["L2"].get("auroc", float("nan"))
        if np.isnan(base):
            return "INCOMPLETE"
        rng = max(base, instruct, reasoning) - min(base, instruct, reasoning)
        if rng < 0.05:
            return "FLAT"
        if reasoning > instruct + 0.10 and abs(instruct - base) < 0.05:
            return "REASONING_JUMP"
        if (instruct > base + 0.03) and (reasoning > instruct + 0.03):
            return "MONOTONE_RISE"
        return "MIXED"
    else:
        delta = reasoning - instruct
        if abs(delta) < 0.05:
            return "FLAT_2STAGE"
        if delta > 0.10:
            return "REASONING_JUMP"
        if delta > 0.03:
            return "MONOTONE_RISE"
        return "MIXED"


def theory_implications(l1_verdict: str, l2_verdict: str) -> dict:
    matrix = {
        "INVERTED_U":    {"H-A": "SUPPORTS",      "H-B": "FALSIFIES_L1", "H-C": "FALSIFIES"},
        "INSTRUCT_PEAK": {"H-A": "SUPPORTS",      "H-B": "WEAKENS",      "H-C": "FALSIFIES"},  # 2-stage proxy for INVERTED_U
        "MONOTONE_RISE": {"H-A": "WEAKENS_L1",    "H-B": "SUPPORTS",     "H-C": "FALSIFIES"},
        "FLAT":          {"H-A": "FALSIFIES",      "H-B": "FALSIFIES",    "H-C": "SUPPORTS"},
        "FLAT_2STAGE":   {"H-A": "FALSIFIES",      "H-B": "FALSIFIES",    "H-C": "SUPPORTS"},
        "MIXED":         {"H-A": "AMBIGUOUS",      "H-B": "AMBIGUOUS",    "H-C": "AMBIGUOUS"},
        "INCOMPLETE":    {"H-A": "UNKNOWN",        "H-B": "UNKNOWN",      "H-C": "UNKNOWN"},
    }
    l2_matrix = {
        "MONOTONE_RISE": {"H-A": "CONSISTENT", "H-B": "CONSISTENT", "H-C": "WEAKENS"},
        "REASONING_JUMP": {"H-A": "SUPPORTS_RL_AMP", "H-B": "CONSISTENT", "H-C": "WEAKENS"},
        "FLAT": {"H-A": "INCONSISTENT", "H-B": "INCONSISTENT", "H-C": "CONSISTENT"},
        "MIXED": {"H-A": "AMBIGUOUS", "H-B": "AMBIGUOUS", "H-C": "AMBIGUOUS"},
        "INCOMPLETE": {"H-A": "UNKNOWN", "H-B": "UNKNOWN", "H-C": "UNKNOWN"},
    }
    return {
        "L1_verdict": l1_verdict,
        "L1_theory": matrix.get(l1_verdict, {}),
        "L2_verdict": l2_verdict,
        "L2_theory": l2_matrix.get(l2_verdict, {}),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 70)
    print("EXP_A_LAW3_LLAMA — Law 3 Cross-Family Replication")
    print("=" * 70)
    print(f"Stages to run: {STAGES_TO_RUN}")

    hf_token = _get_hf_token()
    items = load_triviaqa()

    # Load intermediate results if this is a continuation session
    all_results = {}
    if Path(OUTPUT_FILE).exists():
        with open(OUTPUT_FILE) as f:
            saved = json.load(f)
            all_results = saved.get("stage_results", {})
        print(f"\nLoaded {len(all_results)} previously completed stages from {OUTPUT_FILE}")

    for stage in STAGES:
        name = stage["name"]
        if name not in STAGES_TO_RUN:
            print(f"\nSkipping stage {name} (not in STAGES_TO_RUN)")
            continue
        if name in all_results:
            print(f"\nSkipping stage {name} (already completed)")
            continue

        print(f"\n{'─' * 50}")
        print(f"STAGE: {name}  ({stage['model_id']})")
        print(f"{'─' * 50}")

        model, tokenizer = load_model(stage, hf_token)
        stage_result = {
            "model": stage["model_id"],
            "layer_used": stage["layer_idx"],
            "n_layers_total": stage["n_layers"],
        }
        if "size_confound_note" in stage:
            stage_result["size_confound_note"] = stage["size_confound_note"]

        # ── L1: bilateral oracle ─────────────────────────────────────────────
        param_hs, ctxdep_hs = run_bilateral_oracle(model, tokenizer, items, stage, N_TARGET)

        if len(param_hs) >= 20 and len(ctxdep_hs) >= 20:
            auroc_point, n_train, n_test = fit_probe(param_hs, ctxdep_hs)
            auroc_mean, ci_low, ci_high  = bootstrap_auroc(param_hs, ctxdep_hs)
            auroc_shuf = shuffled_auroc(param_hs, ctxdep_hs)
            stage_result["L1"] = {
                "auroc": round(float(auroc_point), 4),
                "bootstrap_mean": round(float(auroc_mean), 4),
                "ci_95_low": round(float(ci_low), 4),
                "ci_95_high": round(float(ci_high), 4),
                "shuffled_auroc": round(float(auroc_shuf), 4),
                "n_param": len(param_hs),
                "n_ctxdep": len(ctxdep_hs),
                "n_train": n_train,
                "n_test": n_test,
                "signal_valid": float(auroc_point) - float(auroc_shuf) > 0.15,
            }
        else:
            stage_result["L1"] = {"auroc": float("nan"), "n_param": len(param_hs), "n_ctxdep": len(ctxdep_hs)}
            print(f"  ⚠ Insufficient L1 samples: PARAM={len(param_hs)}, CTX_DEP={len(ctxdep_hs)}")

        # Save L1 results immediately — L2 may timeout for slow models (REASONING ~6.5h)
        all_results[name] = stage_result
        _save(all_results)
        print(f"  Saved L1 results for {name} to {OUTPUT_FILE}")

        # ── L2: CO labeling ──────────────────────────────────────────────────
        cc_hs, cw_hs = run_co_labeling(model, tokenizer, items, stage, N_TARGET_L2)

        if len(cc_hs) >= 20 and len(cw_hs) >= 20:
            auroc_point_l2, _, _      = fit_probe(cc_hs, cw_hs)
            auroc_mean_l2, ci_l_l2, ci_h_l2 = bootstrap_auroc(cc_hs, cw_hs)
            auroc_shuf_l2 = shuffled_auroc(cc_hs, cw_hs)
            stage_result["L2"] = {
                "auroc": round(float(auroc_point_l2), 4),
                "bootstrap_mean": round(float(auroc_mean_l2), 4),
                "ci_95_low": round(float(ci_l_l2), 4),
                "ci_95_high": round(float(ci_h_l2), 4),
                "shuffled_auroc": round(float(auroc_shuf_l2), 4),
                "n_cc": len(cc_hs),
                "n_cw": len(cw_hs),
                "theta_conf": THETA_CONF_L2,
            }
        else:
            stage_result["L2"] = {"auroc": float("nan"), "n_cc": len(cc_hs), "n_cw": len(cw_hs)}
            print(f"  ⚠ Insufficient L2 samples: CC={len(cc_hs)}, CW={len(cw_hs)}")

        all_results[name] = stage_result

        # Free memory before next stage
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Save intermediate
        _save(all_results)
        print(f"\n  Saved intermediate results to {OUTPUT_FILE}")

    # ── Final classification ──────────────────────────────────────────────────
    l1_verdict = classify_l1_verdict(all_results)
    l2_verdict = classify_l2_verdict(all_results)
    implications = theory_implications(l1_verdict, l2_verdict)

    c042_promoted = l1_verdict == "INVERTED_U"
    c043_promoted = l2_verdict in ("MONOTONE_RISE", "REASONING_JUMP")

    final = {
        "experiment": "EXP_A_LAW3_LLAMA",
        "date": time.strftime("%Y-%m-%d"),
        "stage_results": all_results,
        "L1_verdict": l1_verdict,
        "L2_verdict": l2_verdict,
        "theory_implications": implications,
        "law3_c042_promoted": c042_promoted,
        "law3_c043_promoted": c043_promoted,
    }

    _save(final)
    _print_summary(final)


def _save(data):
    with open(OUTPUT_FILE, "w") as f:
        json.dump(data, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)


def _print_summary(final):
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    for stage_name, sr in final.get("stage_results", {}).items():
        l1 = sr.get("L1", {})
        l2 = sr.get("L2", {})
        print(f"\n  {stage_name} ({sr.get('model', 'unknown')}):")
        print(f"    L1 AUROC: {l1.get('auroc', 'nan'):.4f}  (shuffled={l1.get('shuffled_auroc', 'nan'):.4f})  "
              f"N={l1.get('n_param', 0)}/{l1.get('n_ctxdep', 0)}")
        print(f"    L2 AUROC: {l2.get('auroc', 'nan'):.4f}  (shuffled={l2.get('shuffled_auroc', 'nan'):.4f})  "
              f"N={l2.get('n_cc', 0)}/{l2.get('n_cw', 0)}")

    print(f"\n  L1 VERDICT:  {final.get('L1_verdict', 'INCOMPLETE')}")
    print(f"  L2 VERDICT:  {final.get('L2_verdict', 'INCOMPLETE')}")
    print(f"\n  Theory implications (L1):")
    for th, imp in final.get("theory_implications", {}).get("L1_theory", {}).items():
        print(f"    {th}: {imp}")
    print(f"\n  C042 (INVERTED_U) promoted to CONFIRMED: {final.get('law3_c042_promoted', False)}")
    print(f"  C043 (MONOTONE_RISE) promoted to CONFIRMED: {final.get('law3_c043_promoted', False)}")
    print(f"\n  Output: {OUTPUT_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()
