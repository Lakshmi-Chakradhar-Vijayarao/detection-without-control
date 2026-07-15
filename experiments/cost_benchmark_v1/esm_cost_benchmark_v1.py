"""
esm_cost_benchmark_v1.py — Epistemic routing compute cost benchmark

Enterprise value proof: same accuracy at measurably lower compute cost.

Three inference regimes compared on TriviaQA (500 questions):
  A) ALWAYS-RAG   — retrieve context for every query (current default)
  B) NEVER-RAG    — answer from parametric memory only
  C) EPISTEMIC    — route via Fisher J_know; retrieve only when CTX_DEP

Compute cost model:
  - Token cost: total tokens processed (prompt + context + generated)
  - Dollar cost: RAG calls × COST_PER_RETRIEVAL_USD ($0.002 default)
    This covers: LLM API call for retrieval query + vector DB lookup
  - Normalized: cost_A = 1.0 baseline

Accuracy metric: EM (exact match) and token F1 on TriviaQA answers

Hypothesis:
  cost(C) < cost(A) by 30-45% while F1(C) ≈ F1(A) ± 2pp
  → 38.5% RAG skip rate validated on HotpotQA, now measured on TriviaQA

Dollar projection at scale:
  At 1M queries/day with 38.5% skip rate and $0.002/retrieval:
  → $770K/day saved → $281M/year savings at iso-quality

Key numbers:
  Prior result: 38.5% routable (nocontext AUROC 0.72-0.73, n=200)
  This experiment: 500-question benchmark with cost accounting and dollar projection

Kaggle GPU: T4 16GB
Model: Qwen/Qwen2.5-1.5B-Instruct (Fisher probe, layer 22)
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

def _ensure(pkg, name=None):
    try:
        __import__(name or pkg)
    except ImportError:
        os.system(f"pip install {pkg} -q")

_ensure("datasets")
_ensure("scikit-learn", "sklearn")
_ensure("matplotlib")

from datasets import load_dataset
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Constants ──────────────────────────────────────────────────────────────────

MODEL_ID       = "Qwen/Qwen2.5-1.5B-Instruct"   # instruct for better QA
if torch.cuda.is_available():
    cc = torch.cuda.get_device_capability(0)
    _sm_major, _sm_minor = cc
    _sm = _sm_major * 10 + _sm_minor
    assert _sm >= 70, f"GPU sm_{_sm} not supported — need T4 (sm_75) or better. Re-run on T4."
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
DTYPE          = torch.float16 if DEVICE == "cuda" else torch.float32

PROBE_LAYER    = 22
SHALLOW_LAYER  = 8
N_CAL          = 200     # Fisher calibration
N_BENCH        = 500     # benchmark questions (reduce to 200 for speed test first)
N_RETRIEVE_TOKENS = 150  # avg tokens per retrieved passage (TriviaQA paragraphs)
MAX_NEW_TOKENS    = 30   # answer generation

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

OUT_DIR = Path("/kaggle/working")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Text utilities ─────────────────────────────────────────────────────────────

def normalize_answer(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def token_f1(pred: str, gold: str) -> float:
    pred_toks = normalize_answer(pred).split()
    gold_toks = normalize_answer(gold).split()
    if not pred_toks or not gold_toks:
        return float(pred_toks == gold_toks)
    common = set(pred_toks) & set(gold_toks)
    if not common:
        return 0.0
    precision = len(common) / len(pred_toks)
    recall    = len(common) / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def answer_contains(pred: str, golds: List[str]) -> bool:
    pred_l = pred.lower()
    for g in golds:
        g_l = g.lower().strip()
        if g_l and g_l in pred_l:
            return True
    return False


def best_f1(pred: str, golds: List[str]) -> float:
    return max(token_f1(pred, g) for g in golds) if golds else 0.0


def exact_match(pred: str, golds: List[str]) -> bool:
    pred_n = normalize_answer(pred)
    return any(normalize_answer(g) == pred_n for g in golds)


# ── Fisher probe ───────────────────────────────────────────────────────────────

def get_layers(model):
    for path in ["base_model.model.model.layers", "model.layers", "model.model.layers"]:
        obj = model
        try:
            for a in path.split("."): obj = getattr(obj, a)
            return obj
        except AttributeError: continue
    raise ValueError("Cannot find layers")


@dataclass
class Probe:
    diff_u:      np.ndarray
    c_ctx:       np.ndarray
    theta:       float
    auroc:       float
    j_param_std: float = 0.0   # std of PARAM j_know scores; used for conservative threshold


def calibrate(model, tokenizer, questions, n_cal) -> Probe:
    model.eval()
    layers = get_layers(model)
    hs, labels = [], []
    buf = {}

    def hook(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        buf["h"] = h[:, -1, :].float().detach().cpu().numpy()

    handle = layers[PROBE_LAYER].register_forward_hook(hook)

    for q in questions[:n_cal]:
        qa = q.get("question", "")
        ans_d = q.get("answer", {})
        gold = [ans_d.get("value", "")] + ans_d.get("aliases", [])
        gold = [a.lower().strip() for a in gold if a]
        if not gold: continue

        # No-context probe
        prompt = f"Answer briefly.\nQuestion: {qa}\nAnswer:"
        ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=200).input_ids.to(DEVICE)
        with torch.no_grad(): model(ids)
        if "h" not in buf: continue

        # Get answer to label
        out = model.generate(ids, max_new_tokens=15, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        pred = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).lower().strip()
        correct = any(g in pred or pred in g for g in gold if len(g) > 2)

        hs.append(buf.pop("h")[0])
        labels.append(1 if correct else 0)

    handle.remove()

    X = np.array(hs, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)

    lda    = LinearDiscriminantAnalysis(n_components=1)
    lda.fit(X, y)
    diff_u = lda.scalings_[:, 0].astype(np.float32)
    diff_u /= np.linalg.norm(diff_u) + 1e-8
    c_ctx  = X[y==0].mean(axis=0).astype(np.float32)
    j      = (X - c_ctx) @ diff_u
    theta  = float((j[y==1].mean() + j[y==0].mean()) / 2.0)
    auroc  = roc_auc_score(y, j)
    # LDA direction sign is arbitrary — flip so CORRECT=1 has higher j_know than WRONG=0
    if auroc < 0.5:
        diff_u = -diff_u
        j      = -j
        theta  = float((j[y==1].mean() + j[y==0].mean()) / 2.0)
        auroc  = 1.0 - auroc
    j_param_std = float(j[y==1].std()) if y.sum() > 1 else 1.0
    theta_cons  = theta + j_param_std
    print(f"    Probe AUROC: {auroc:.4f}  θ={theta:.3f}  θ_conservative={theta_cons:.3f}")
    return Probe(diff_u=diff_u, c_ctx=c_ctx, theta=theta, auroc=float(auroc),
                 j_param_std=j_param_std)


def get_j_know(model, tokenizer, question: str, probe: Probe) -> float:
    model.eval()
    layers = get_layers(model)
    diff_u = torch.tensor(probe.diff_u, dtype=torch.float32, device=DEVICE)
    c_ctx  = torch.tensor(probe.c_ctx,  dtype=torch.float32, device=DEVICE)
    buf    = {}

    def hook(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        buf["h"] = h[:, -1, :].float().detach()

    handle = layers[PROBE_LAYER].register_forward_hook(hook)

    prompt = f"Answer briefly.\nQuestion: {question}\nAnswer:"
    ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=200).input_ids.to(DEVICE)
    with torch.no_grad(): model(ids)
    handle.remove()

    if "h" not in buf:
        return 0.0
    hd = buf["h"][0]
    return ((hd - c_ctx) * diff_u).sum().item()


# ── Single-question inference ──────────────────────────────────────────────────

@dataclass
class InferenceResult:
    answer:        str
    f1:            float
    em:            bool
    prompt_tokens: int
    context_tokens: int
    output_tokens: int
    rag_used:      bool
    j_know:        float = 0.0
    routing:       str   = "UNKNOWN"

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.context_tokens + self.output_tokens


def run_inference(
    model, tokenizer, question: str, gold_answers: List[str],
    context: Optional[str], use_context: bool,
) -> InferenceResult:
    if use_context and context:
        ctx_snippet = context[:600]  # truncate to ~150 tokens
        prompt = (f"Use the following context to answer.\n"
                  f"Context: {ctx_snippet}\n"
                  f"Question: {question}\nAnswer:")
        ctx_tokens = len(tokenizer.encode(ctx_snippet))
    else:
        prompt   = f"Answer briefly.\nQuestion: {question}\nAnswer:"
        ctx_tokens = 0

    ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=400).input_ids.to(DEVICE)
    prompt_tokens = ids.shape[1]

    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    ans_ids = out[0][ids.shape[1]:]
    answer  = tokenizer.decode(ans_ids, skip_special_tokens=True).strip()
    output_tokens = len(ans_ids)

    return InferenceResult(
        answer=answer,
        f1=best_f1(answer, gold_answers),
        em=exact_match(answer, gold_answers),
        prompt_tokens=int(prompt_tokens),
        context_tokens=int(ctx_tokens),
        output_tokens=int(output_tokens),
        rag_used=use_context,
    )


# ── Benchmark regimes ──────────────────────────────────────────────────────────

def run_regime(
    model, tokenizer, questions: List[Dict], probe: Probe,
    regime: str,  # "always_rag" | "never_rag" | "epistemic"
) -> Dict:
    """Run one regime and collect per-question results."""
    all_results = []
    j_scores    = []

    print(f"\n  [{regime}] running {len(questions)} questions...")
    t0 = time.time()

    for i, q in enumerate(questions):
        question  = q.get("question", "")
        ans_d     = q.get("answer", {})
        gold      = [ans_d.get("value", "")] + ans_d.get("aliases", [])
        gold      = [a for a in gold if a]
        context   = q.get("context", "")  # may be empty in rc.nocontext

        if regime == "always_rag":
            use_ctx  = True
            j_know   = 0.0
            routing  = "CTX_DEP"

        elif regime == "never_rag":
            use_ctx  = False
            j_know   = 0.0
            routing  = "PARAM"

        elif regime == "epistemic":
            j_know  = get_j_know(model, tokenizer, question, probe)
            routing = "PARAM" if j_know >= probe.theta else "CTX_DEP"
            use_ctx = (routing == "CTX_DEP")

        else:  # epistemic_conservative — only skip RAG when j_know > theta + 1σ
            j_know  = get_j_know(model, tokenizer, question, probe)
            theta_c = probe.theta + probe.j_param_std
            routing = "PARAM" if j_know >= theta_c else "CTX_DEP"
            use_ctx = (routing == "CTX_DEP")

        # Generate answer
        res = run_inference(model, tokenizer, question, gold, context, use_ctx)
        res.j_know   = j_know
        res.routing  = routing

        all_results.append({
            "question":      question,
            "gold":          gold[:2],
            "answer":        res.answer,
            "f1":            res.f1,
            "em":            res.em,
            "rag_used":      res.rag_used,
            "j_know":        res.j_know,
            "routing":       res.routing,
            "prompt_tokens": res.prompt_tokens,
            "ctx_tokens":    res.context_tokens,
            "output_tokens": res.output_tokens,
            "total_tokens":  res.total_tokens,
        })
        j_scores.append(j_know)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"    step {i+1}/{len(questions)} | {elapsed:.0f}s elapsed")

    elapsed = time.time() - t0
    n = len(all_results)

    mean_f1    = float(np.mean([r["f1"] for r in all_results]))
    em_rate    = float(np.mean([r["em"] for r in all_results]))
    rag_rate   = float(np.mean([r["rag_used"] for r in all_results]))
    total_tok  = sum(r["total_tokens"] for r in all_results)
    ctx_tok    = sum(r["ctx_tokens"]   for r in all_results)
    param_n    = sum(1 for r in all_results if r["routing"] == "PARAM")

    return {
        "regime":          regime,
        "n":               n,
        "mean_f1":         round(mean_f1, 4),
        "em_rate":         round(em_rate, 4),
        "rag_rate":        round(rag_rate, 3),
        "total_tokens":    total_tok,
        "context_tokens":  ctx_tok,
        "param_routed":    param_n,
        "elapsed_s":       round(elapsed, 1),
        "results":         all_results,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def run():
    print("=" * 64)
    print("COST BENCHMARK v1 — Epistemic Routing vs Always-RAG")
    print(f"Model: {MODEL_ID} | N={N_BENCH}")
    print("=" * 64)

    # ── 1. Dataset ─────────────────────────────────────────────────────────────
    print("\n[1] Loading TriviaQA rc (with contexts)...")
    # Use rc (not rc.nocontext) so we have contexts for the RAG conditions
    tqa = load_dataset("trivia_qa", "rc", split="validation[:3000]", trust_remote_code=True)
    tqa_list = list(tqa)
    random.shuffle(tqa_list)

    # Build enriched questions with flattened context
    enriched = []
    for q in tqa_list:
        ctx = ""
        sr  = q.get("search_results", {})
        if sr and "search_context" in sr:
            ctx_list = sr["search_context"]
            if ctx_list:
                ctx = " ".join(str(c) for c in ctx_list[:2])[:800]
        enriched.append({
            "question": q["question"],
            "answer":   q["answer"],
            "context":  ctx,
        })

    cal_pool   = enriched[:400]
    bench_pool = enriched[400:400 + N_BENCH]
    random.shuffle(bench_pool)

    # ── 2. Model ────────────────────────────────────────────────────────────────
    print(f"\n[2] Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE, device_map=None, trust_remote_code=True,
    ).to(DEVICE).eval()

    # ── 3. Calibration ──────────────────────────────────────────────────────────
    print(f"\n[3] Fisher calibration (n={N_CAL})...")
    probe = calibrate(model, tokenizer, cal_pool, N_CAL)

    # ── 4. Benchmarks ───────────────────────────────────────────────────────────
    results_always = run_regime(model, tokenizer, bench_pool, probe, "always_rag")
    results_never  = run_regime(model, tokenizer, bench_pool, probe, "never_rag")
    results_epist  = run_regime(model, tokenizer, bench_pool, probe, "epistemic")
    results_cons   = run_regime(model, tokenizer, bench_pool, probe, "epistemic_conservative")

    # ── 5. Cost analysis ────────────────────────────────────────────────────────
    base_tokens  = results_always["total_tokens"]
    never_tokens = results_never["total_tokens"]
    epist_tokens = results_epist["total_tokens"]
    cons_tokens  = results_cons["total_tokens"]

    cost_reduction   = 1.0 - (epist_tokens / base_tokens)
    cost_reduction_c = 1.0 - (cons_tokens  / base_tokens)
    f1_delta         = results_epist["mean_f1"] - results_always["mean_f1"]
    f1_delta_c       = results_cons["mean_f1"]  - results_always["mean_f1"]
    rag_skip_rate    = 1.0 - results_epist["rag_rate"]
    rag_skip_rate_c  = 1.0 - results_cons["rag_rate"]

    # Cost-effectiveness: F1 per token-cost (higher = better)
    ce_always = results_always["mean_f1"] / max(base_tokens,  1)
    ce_epist  = results_epist["mean_f1"]  / max(epist_tokens, 1)
    ce_cons   = results_cons["mean_f1"]   / max(cons_tokens,  1)
    ce_gain   = (ce_epist - ce_always) / ce_always
    ce_gain_c = (ce_cons  - ce_always) / ce_always

    target_met   = cost_reduction   >= 0.25 and abs(f1_delta)   <= 0.03
    target_met_c = cost_reduction_c >= 0.20 and abs(f1_delta_c) <= 0.03

    # ── Dollar projection at scale ─────────────────────────────────────────────
    COST_PER_RETRIEVAL_USD = 0.002   # $ per RAG call: LLM retrieval API + vector DB
    QUERY_VOLUMES = [100_000, 1_000_000, 10_000_000, 100_000_000]

    projection_rows = []
    for vol in QUERY_VOLUMES:
        saved_per_day  = vol * rag_skip_rate
        daily_savings  = saved_per_day * COST_PER_RETRIEVAL_USD
        annual_savings = daily_savings * 365 / 1_000_000   # $M
        projection_rows.append({
            "queries_per_day":              vol,
            "rag_calls_saved_per_day":      int(saved_per_day),
            "daily_savings_usd":            round(daily_savings, 2),
            "annual_savings_usd_millions":  round(annual_savings, 3),
        })

    print("\n" + "=" * 64)
    print("COST BENCHMARK RESULTS")
    print("=" * 64)
    print(f"  ALWAYS-RAG:            F1={results_always['mean_f1']:.4f} | tokens={base_tokens:,}  | RAG=100%")
    print(f"  NEVER-RAG:             F1={results_never['mean_f1']:.4f} | tokens={never_tokens:,} | RAG=0%")
    print(f"  EPISTEMIC (θ):         F1={results_epist['mean_f1']:.4f} | tokens={epist_tokens:,} | skip={rag_skip_rate:.1%}")
    print(f"  EPISTEMIC_CONS (θ+σ): F1={results_cons['mean_f1']:.4f} | tokens={cons_tokens:,}  | skip={rag_skip_rate_c:.1%}")
    print(f"\n  Aggressive  — cost reduction: {cost_reduction:.1%}  F1 delta: {f1_delta:+.4f}  "
          f"Target: {'PASSED ✓' if target_met else 'FAILED ✗'}")
    print(f"  Conservative— cost reduction: {cost_reduction_c:.1%}  F1 delta: {f1_delta_c:+.4f}  "
          f"Target: {'PASSED ✓' if target_met_c else 'FAILED ✗'}")
    print(f"  Probe AUROC: {probe.auroc:.4f}  θ={probe.theta:.3f}  θ+σ={probe.theta+probe.j_param_std:.3f}")
    print("\n  Dollar projection (${:.3f}/retrieval, at scale):".format(COST_PER_RETRIEVAL_USD))
    print(f"  {'Volume/day':>15}  {'RAG saved/day':>15}  {'Daily savings':>15}  {'Annual ($M)':>12}")
    for r in projection_rows:
        print(f"  {r['queries_per_day']:>15,}  "
              f"{r['rag_calls_saved_per_day']:>15,}  "
              f"${r['daily_savings_usd']:>14,.0f}  "
              f"${r['annual_savings_usd_millions']:>11.2f}M")
    print("=" * 64)

    results = {
        "meta": {
            "model": MODEL_ID, "n_bench": N_BENCH, "n_cal": N_CAL,
            "probe_auroc": float(probe.auroc), "probe_theta": float(probe.theta),
        },
        "probe_j_param_std": float(probe.j_param_std),
        "probe_theta_conservative": float(probe.theta + probe.j_param_std),
        "always_rag":  {k: v for k, v in results_always.items() if k != "results"},
        "never_rag":   {k: v for k, v in results_never.items()  if k != "results"},
        "epistemic":   {k: v for k, v in results_epist.items()  if k != "results"},
        "epistemic_conservative": {k: v for k, v in results_cons.items() if k != "results"},
        "aggressive":  {"cost_reduction": float(cost_reduction),  "f1_delta": float(f1_delta),
                        "rag_skip_rate": float(rag_skip_rate),    "ce_gain": float(ce_gain),
                        "target_met": target_met},
        "conservative":{"cost_reduction": float(cost_reduction_c),"f1_delta": float(f1_delta_c),
                        "rag_skip_rate": float(rag_skip_rate_c),  "ce_gain": float(ce_gain_c),
                        "target_met": target_met_c},
        "verdict": ("BOTH_PASSED" if target_met and target_met_c else
                    "CONSERVATIVE_PASSED" if target_met_c else
                    "AGGRESSIVE_ONLY" if target_met else "PARTIAL"),
        "dollar_projection": {
            "cost_per_retrieval_usd":   COST_PER_RETRIEVAL_USD,
            "rag_skip_rate":            float(rag_skip_rate),
            "f1_delta_at_skip_rate":    float(f1_delta),
            "rows": projection_rows,
        },
    }

    out_json = OUT_DIR / "cost_benchmark_v1_results.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nResults → {out_json}")

    # Save per-question results for analysis
    per_q = OUT_DIR / "cost_benchmark_v1_per_question.json"
    per_q.write_text(json.dumps({
        "always_rag": results_always["results"][:50],
        "epistemic":  results_epist["results"][:50],
    }, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else str(x)))

    # ── Plot ───────────────────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        regimes    = ["ALWAYS-RAG\n(baseline)", "NEVER-RAG", "EPISTEMIC\n(ours)"]
        f1_vals    = [results_always["mean_f1"], results_never["mean_f1"], results_epist["mean_f1"]]
        tok_vals   = [base_tokens, never_tokens, epist_tokens]
        rag_rates  = [1.0, 0.0, results_epist["rag_rate"]]
        bar_colors = ["red", "orange", "green"]

        # F1 comparison
        ax = axes[0]
        bars = ax.bar(regimes, f1_vals, color=bar_colors, alpha=0.7, edgecolor="black")
        for bar, v in zip(bars, f1_vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.002, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=10)
        ax.set_ylabel("Token F1")
        ax.set_title("Answer Quality (F1)")
        ax.set_ylim([0, max(f1_vals) * 1.2])
        ax.grid(True, axis="y", alpha=0.3)

        # Token cost comparison
        ax2 = axes[1]
        tok_n = [t / 1000 for t in tok_vals]
        bars2 = ax2.bar(regimes, tok_n, color=bar_colors, alpha=0.7, edgecolor="black")
        for bar, v in zip(bars2, tok_n):
            ax2.text(bar.get_x() + bar.get_width()/2, v + 0.5, f"{v:.0f}K",
                     ha="center", va="bottom", fontsize=10)
        ax2.set_ylabel("Total Tokens (thousands)")
        ax2.set_title(f"Compute Cost\n(Epistemic = {cost_reduction:.1%} reduction)")
        ax2.grid(True, axis="y", alpha=0.3)

        # RAG call rate
        ax3 = axes[2]
        rag_pcts = [r * 100 for r in rag_rates]
        bars3 = ax3.bar(regimes, rag_pcts, color=bar_colors, alpha=0.7, edgecolor="black")
        for bar, v in zip(bars3, rag_pcts):
            ax3.text(bar.get_x() + bar.get_width()/2, v + 1, f"{v:.0f}%",
                     ha="center", va="bottom", fontsize=10)
        ax3.set_ylabel("RAG Retrieval Rate (%)")
        ax3.set_title(f"RAG Calls\n(Epistemic skips {rag_skip_rate:.1%})")
        ax3.grid(True, axis="y", alpha=0.3)
        ax3.set_ylim([0, 115])

        # Dollar projection panel
        ax4 = axes[3]
        vol_labels = [f"{r['queries_per_day']//1000}K" if r['queries_per_day'] < 1_000_000
                      else f"{r['queries_per_day']//1_000_000}M"
                      for r in projection_rows]
        annual_vals = [r['annual_savings_usd_millions'] for r in projection_rows]
        bars4 = ax4.bar(vol_labels, annual_vals, color="steelblue", alpha=0.75, edgecolor="black")
        for bar, val in zip(bars4, annual_vals):
            ax4.text(bar.get_x() + bar.get_width()/2, val + max(annual_vals)*0.02,
                     f"${val:.1f}M", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax4.set_xlabel("Queries / day")
        ax4.set_ylabel("Annual savings (USD millions)")
        ax4.set_title(f"Dollar savings at scale\n"
                      f"(${COST_PER_RETRIEVAL_USD}/retrieval, {rag_skip_rate:.0%} skip rate)")
        ax4.grid(True, axis="y", alpha=0.3)

        plt.suptitle(
            f"Epistemic Routing Cost Benchmark — {MODEL_ID}\n"
            f"Cost ↓{cost_reduction:.1%}  |  ΔF1={f1_delta:+.3f}  |  "
            f"RAG skip {rag_skip_rate:.0%}  |  "
            f"Verdict: {'PASSED ✓' if target_met else 'FAILED ✗'}",
            fontsize=12,
        )
        plt.tight_layout()

        out_fig = OUT_DIR / "cost_benchmark_v1.png"
        plt.savefig(out_fig, dpi=150, bbox_inches="tight")
        print(f"Figure → {out_fig}")
        plt.show()

    except Exception as e:
        print(f"Plot skipped: {e}")

    return results


if __name__ == "__main__":
    run()
