"""
experiments/reasoning_model_extension/reasoning_model_extension.py

TASK 2.2 — BILATERAL ORACLE ON REASONING MODELS
================================================

Central question: Does Fisher+PCA64 AUROC ≥ 0.82 (from C001) hold for
DeepSeek-R1-Distill-Qwen-1.5B (a reasoning model with <think>...</think> chains)?

Three extraction points tested:
  A. Step-1 (pre-think): HS captured at first generated token — same protocol as C3-v3
  B. Pre-answer: HS at the token immediately before first answer token (after </think>)
  C. Think-end: HS at the </think> close token

Hypothesis:
  - If AUROC_A ≈ 0.84 → reasoning architecture doesn't disrupt step-1 signal
  - If AUROC_B > AUROC_A → model organizes epistemic state DURING thinking
  - If AUROC_B ≈ AUROC_A → commitment is pre-reasoning (budget is wasted on decoration)

Comparison baseline: C3-v3 Qwen2.5-1.5B-Instruct AUROC = 0.841

Model: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
  — Same backbone as Qwen2.5-1.5B, <3 GB at float16, T4-compatible
  — Native thinking-chain generation
Dataset: TriviaQA rc.wikipedia (streaming)
Output: reasoning_model_extension_results.json
"""

from __future__ import annotations

import json
import os
import re
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from datasets import load_dataset
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID     = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
LAYER_IDX    = 26          # same as C3-v3
N_TARGET     = 80          # per class (reasoning models are slow)
POOL_SIZE    = 3000
PARAM_MIN    = 0.50
CTX_MIN_NC   = 0.05
CTX_MIN_CTX  = 0.50
MAX_NEW_PRE  = 2           # step-1 extraction
MAX_NEW_FULL = 512         # full think-chain generation
N_BOOTSTRAP  = 500
RESULTS_FILE = "reasoning_model_extension_results.json"

THINK_OPEN  = "<think>"
THINK_CLOSE = "</think>"

# ── Helpers ───────────────────────────────────────────────────────────────────
def token_f1(pred: str, gold: str) -> float:
    p_tok = set(pred.lower().split())
    g_tok = set(gold.lower().split())
    if not p_tok or not g_tok:
        return 0.0
    prec = len(p_tok & g_tok) / len(p_tok)
    rec  = len(p_tok & g_tok) / len(g_tok)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)

def best_f1(pred: str, answers: list[str]) -> float:
    return max(token_f1(pred, a) for a in answers) if answers else 0.0


def fmt_nocontext(q: str) -> str:
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{q}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

def fmt_withcontext(q: str, ctx: str) -> str:
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\nBackground: {ctx}\n\n{q}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

def extract_answer_from_generation(text: str) -> str:
    """Strip <think>...</think> block; return remaining text."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model():
    print(f"Loading {MODEL_ID}...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).to(DEVICE).eval()
    return tok, mdl


# ── Hidden state extraction ───────────────────────────────────────────────────
@dataclass
class ExtractionResult:
    hs_step1:     Optional[np.ndarray]   # step-1 (pre-think)
    hs_think_end: Optional[np.ndarray]   # HS at </think> token
    hs_pre_ans:   Optional[np.ndarray]   # HS just before answer token
    generation:   str                    # full decoded output
    answer:       str                    # extracted answer


def extract_hs(tok, mdl, prompt: str) -> ExtractionResult:
    """
    Single-pass extraction: generate up to MAX_NEW_FULL tokens, capture HS
    at three points.
    """
    captured: dict[str, Optional[np.ndarray]] = {
        "step1": None, "think_end": None, "pre_ans": None
    }
    step_ctr = [0]
    think_close_id = tok.encode(THINK_CLOSE, add_special_tokens=False)
    think_close_id = think_close_id[-1] if think_close_id else None

    layer = mdl.model.layers[LAYER_IDX]

    def hook_fn(module, inp_t, out):
        hs = out[0] if isinstance(out, tuple) else out
        if hs.shape[1] != 1:   # prompt pass — skip
            return
        s = step_ctr[0]
        if s == 0:
            captured["step1"] = hs[0, 0, :].detach().float().cpu().numpy()
        step_ctr[0] += 1

    handle = layer.register_forward_hook(hook_fn)
    inputs = tok(prompt, return_tensors="pt").to(DEVICE)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = mdl.generate(
            **inputs,
            max_new_tokens=MAX_NEW_FULL,
            do_sample=False,
            temperature=1.0,
            use_cache=True,
            output_hidden_states=False,
            pad_token_id=tok.pad_token_id,
        )

    handle.remove()

    # Decode full generation
    gen_ids  = out[0, prompt_len:].tolist()
    gen_text = tok.decode(gen_ids, skip_special_tokens=False)

    # Find </think> position in generated tokens for HS at think_end and pre_ans
    # We need another pass to extract those — use the prompt+gen prefix approach
    if think_close_id is not None and think_close_id in gen_ids:
        think_close_pos = gen_ids.index(think_close_id)
        # Re-run with prefix up to think_close_pos to get think_end HS
        prefix_ids = out[0, : prompt_len + think_close_pos + 1].unsqueeze(0)
        hs_at_think_end = _get_last_gen_hs(mdl, layer, prefix_ids)
        captured["think_end"] = hs_at_think_end

        # Pre-answer: token after </think>
        if think_close_pos + 1 < len(gen_ids):
            prefix_ids_2 = out[0, : prompt_len + think_close_pos + 2].unsqueeze(0)
            hs_pre = _get_last_gen_hs(mdl, layer, prefix_ids_2)
            captured["pre_ans"] = hs_pre

    answer = extract_answer_from_generation(gen_text)

    return ExtractionResult(
        hs_step1=captured["step1"],
        hs_think_end=captured["think_end"],
        hs_pre_ans=captured["pre_ans"],
        generation=gen_text,
        answer=answer,
    )


def _get_last_gen_hs(mdl, layer, prefix_ids: torch.Tensor) -> Optional[np.ndarray]:
    """Feed prefix through model and capture HS at last position."""
    captured = [None]

    def hook_fn(module, inp_t, out):
        hs = out[0] if isinstance(out, tuple) else out
        captured[0] = hs[0, -1, :].detach().float().cpu().numpy()

    handle = layer.register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            mdl(input_ids=prefix_ids)
    finally:
        handle.remove()
    return captured[0]


# ── Oracle labeling ───────────────────────────────────────────────────────────
def build_oracle_pool(tok, mdl, dataset_iter):
    param_items, ctx_dep_items = [], []
    skip_count = 0
    pool_seen  = 0

    for ex in dataset_iter:
        if pool_seen >= POOL_SIZE:
            break
        pool_seen += 1

        q = ex["question"]
        answers = ex["answer"]["aliases"] if "aliases" in ex["answer"] else [ex["answer"]["value"]]

        # Context: take first entity page paragraph
        ctx_parts = ex.get("entity_pages", {}).get("wiki_context", [])
        ctx = ctx_parts[0][:800] if ctx_parts else ""

        if not ctx:
            skip_count += 1
            continue

        # --- Nocontext pass (full generation to get F1) ---
        nc_prompt = fmt_nocontext(q)
        nc_res = extract_hs(tok, mdl, nc_prompt)
        nc_f1  = best_f1(nc_res.answer, answers)

        # --- Withcontext pass ---
        wc_prompt = fmt_withcontext(q, ctx)
        wc_res = extract_hs(tok, mdl, wc_prompt)
        wc_f1  = best_f1(wc_res.answer, answers)

        if nc_f1 >= PARAM_MIN:
            if nc_res.hs_step1 is not None:
                param_items.append({
                    "hs_step1":     nc_res.hs_step1,
                    "hs_think_end": nc_res.hs_think_end,
                    "hs_pre_ans":   nc_res.hs_pre_ans,
                    "nc_f1": nc_f1, "wc_f1": wc_f1,
                })
        elif nc_f1 <= CTX_MIN_NC and wc_f1 >= CTX_MIN_CTX:
            if nc_res.hs_step1 is not None:
                ctx_dep_items.append({
                    "hs_step1":     nc_res.hs_step1,
                    "hs_think_end": nc_res.hs_think_end,
                    "hs_pre_ans":   nc_res.hs_pre_ans,
                    "nc_f1": nc_f1, "wc_f1": wc_f1,
                })
        else:
            skip_count += 1

        n_p = len(param_items); n_c = len(ctx_dep_items)
        if n_p % 10 == 0 or n_c % 10 == 0:
            print(f"  pool={pool_seen} PARAM={n_p} CTX_DEP={n_c} skip={skip_count}")

        if n_p >= N_TARGET and n_c >= N_TARGET:
            break

    return param_items[:N_TARGET], ctx_dep_items[:N_TARGET]


# ── Probe + AUROC ─────────────────────────────────────────────────────────────
def fit_probe_auroc(X: np.ndarray, y: np.ndarray, n_boot: int = N_BOOTSTRAP):
    """Fisher+PCA64, 5-fold CV, bootstrap CI."""
    n_comp = min(64, X.shape[1], X.shape[0] - 2)
    pca = PCA(n_components=n_comp, random_state=42)
    X_r = pca.fit_transform(X)

    skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for tr, va in skf.split(X_r, y):
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        lda.fit(X_r[tr], y[tr])
        proba = lda.predict_proba(X_r[va])[:, 1]
        scores.append(roc_auc_score(y[va], proba))
    mean_auroc = float(np.mean(scores))

    # Bootstrap CI on full dataset
    full_lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    full_lda.fit(X_r, y)
    proba_full = full_lda.predict_proba(X_r)[:, 1]
    boot = []
    rng  = np.random.default_rng(42)
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        boot.append(roc_auc_score(y[idx], proba_full[idx]))
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])

    # Shuffled null
    y_shuf = rng.permutation(y)
    shuf_scores = []
    for tr, va in skf.split(X_r, y_shuf):
        lda_shuf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        lda_shuf.fit(X_r[tr], y_shuf[tr])
        p = lda_shuf.predict_proba(X_r[va])[:, 1]
        shuf_scores.append(roc_auc_score(y_shuf[va], p))
    shuffled = float(np.mean(shuf_scores))

    return {
        "auroc":    mean_auroc,
        "ci_lo":    float(ci_lo),
        "ci_hi":    float(ci_hi),
        "shuffled": shuffled,
        "n":        int(len(y)),
    }


def evaluate_extraction_point(param_items, ctx_dep_items, hs_key: str):
    """Run probe for one extraction point (step1 / think_end / pre_ans)."""
    # Filter out items where hs is None
    p_hs = [it[hs_key] for it in param_items   if it[hs_key] is not None]
    c_hs = [it[hs_key] for it in ctx_dep_items if it[hs_key] is not None]
    n = min(len(p_hs), len(c_hs))
    if n < 20:
        return {"auroc": None, "note": f"insufficient data n={n}"}
    X = np.vstack(p_hs[:n] + c_hs[:n])
    y = np.array([1] * n + [0] * n)
    return fit_probe_auroc(X, y)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    tok, mdl = load_model()

    print("Loading TriviaQA...")
    ds = load_dataset(
        "trivia_qa", "rc.wikipedia",
        split="validation",
        streaming=True,
        trust_remote_code=True,
    )

    print(f"Building oracle pool (target {N_TARGET}/class)...")
    param_items, ctx_dep_items = build_oracle_pool(tok, mdl, iter(ds))

    n_p = len(param_items); n_c = len(ctx_dep_items)
    print(f"Pool complete: PARAM={n_p} CTX_DEP={n_c}")

    results = {
        "model":     MODEL_ID,
        "layer_idx": LAYER_IDX,
        "n_param":   n_p,
        "n_ctx_dep": n_c,
        "baseline_c3v3_qwen": 0.841,
    }

    # ── Probe at each extraction point ────────────────────────────────────────
    for key, label in [
        ("hs_step1",     "step_1_pre_think"),
        ("hs_think_end", "think_end"),
        ("hs_pre_ans",   "pre_answer"),
    ]:
        print(f"\nFitting probe: {label}...")
        res = evaluate_extraction_point(param_items, ctx_dep_items, key)
        results[label] = res
        if res.get("auroc") is not None:
            print(
                f"  AUROC={res['auroc']:.4f} "
                f"[{res['ci_lo']:.3f}-{res['ci_hi']:.3f}] "
                f"shuffled={res['shuffled']:.4f}"
            )
        else:
            print(f"  Skipped: {res.get('note')}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    step1_auroc    = results["step_1_pre_think"].get("auroc")
    pre_ans_auroc  = results["pre_answer"].get("auroc")
    baseline       = results["baseline_c3v3_qwen"]

    if step1_auroc is None:
        verdict = "INSUFFICIENT_DATA"
    elif step1_auroc >= baseline - 0.05:
        # Signal holds in reasoning model
        if pre_ans_auroc is not None and pre_ans_auroc > step1_auroc + 0.04:
            verdict = "HOLDS_WITH_IMPROVEMENT"  # reasoning adds info
        else:
            verdict = "HOLDS"                   # reasoning irrelevant to signal
    else:
        verdict = "DEGRADED"                    # reasoning model disrupts signal

    results["verdict"] = verdict
    results["interpretation"] = {
        "HOLDS":              "Step-1 signal transfers to reasoning model. Commitment precedes thinking.",
        "HOLDS_WITH_IMPROVEMENT": "Signal improves post-think. Reasoning reorganizes epistemic geometry.",
        "DEGRADED":           "Reasoning model disrupts step-1 signal. Protocol needs adaptation.",
        "INSUFFICIENT_DATA":  "Not enough labeled items to assess.",
    }.get(verdict, "")

    print(f"\n{'='*50}")
    print(f"VERDICT: {verdict}")
    print(f"Step-1 AUROC: {step1_auroc:.4f} vs baseline {baseline}")
    print(f"{'='*50}")

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results → {RESULTS_FILE}")


if __name__ == "__main__":
    main()
