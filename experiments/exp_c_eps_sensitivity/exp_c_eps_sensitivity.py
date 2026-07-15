#!/usr/bin/env python3
"""
exp_c_eps_sensitivity.py — COMMITMENT FRACTION ε_C SENSITIVITY ANALYSIS
========================================================================
SCIENTIFIC QUESTION (Q11, Tier 3 — Mechanism):
  Is commit_pct ≥ 70% robust across ε_C threshold values, or is it an artifact
  of the specific threshold chosen?

BACKGROUND:
  Law 2 reports commit_pct ≥ 70% for three reasoning models (C022, C045).
  commit_pct = (T - t*) / T where t* is the generation step where conditional
  entropy of the answer distribution falls below ε_C.

  The specific ε_C is not reported with sensitivity bounds. Qwen3's result
  (commit_pct = 99.8%) is consistent with any ε_C above the minimum think-block
  entropy — it may not be robust to threshold changes.

DESIGN:
  Models: the three reasoning models from C022/C045
    - DeepSeek-R1-Distill-Qwen-1.5B    (C022: commit_pct=75.8%)
    - DeepSeek-R1-Distill-Llama-8B     (C045: commit_pct=82.9%)
    - Qwen/QwQ-32B-Preview or Qwen3-1.7B (C045: commit_pct=99.8%)
      [Use Qwen3-1.7B if 32B too large for T4]

  ε_C values: [0.05, 0.10, 0.15, 0.20, 0.25]

  For each model × ε_C:
    - Generate N=100 think-block responses on TriviaQA
    - Extract hidden states at each generation step (steps 1-20 of think block)
    - Compute conditional entropy from logits at each step
    - Determine commit point t*(ε_C): first step where entropy ≤ ε_C
    - Compute commit_pct = (T - t*) / T
    - Report mean ± std over N=100 items

VERDICT CRITERIA (pre-registered):
  ROBUST:    std(commit_pct over ε_C range) < 0.10 for each model
             AND commit_pct ≥ 0.70 holds for all models at all ε_C values
  SENSITIVE: std(commit_pct over ε_C range) > 0.15 for any model
             OR commit_pct < 0.70 for any model at ε_C ≤ 0.10
  MONOTONE:  commit_pct increases as ε_C increases (expected — report slope)

GPU: T4 (~2-3h — three models, step-by-step generation)

CLAIM IMPACT:
  Updates C022, C045 with sensitivity bounds.
  ROBUST → Law 2 confidence increases; add "robust across ε_C ∈ [0.05, 0.25]"
  SENSITIVE → Law 2 requires ε_C specification; add ε_C to the law statement
"""

from __future__ import annotations
import subprocess
subprocess.run(["pip", "install", "-q", "bitsandbytes>=0.46.1"], check=False)

import gc, json, os, random, time
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ── Config ────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
if DEVICE == "cpu":
    raise RuntimeError("GPU required. Exiting.")

EPS_C_VALUES    = [0.05, 0.10, 0.15, 0.20, 0.25]
N_ITEMS         = 100
MAX_THINK_STEPS = 30   # number of think-block steps to analyze
MAX_NEW_TOKENS  = 200  # enough to generate a complete think block
LAYER_IDX       = 26   # for hidden state extraction (not primary analysis target here)
OUTPUT_FILE     = "/kaggle/working/exp_c_eps_sensitivity_results.json"

MODELS = [
    {
        "name":     "R1_QWEN",
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "use_4bit": False,
        "expected_commit_pct": 0.758,  # from C022
        "c022_eps_c": 0.15,            # assumed ε_C used in C022 (verify in original kernel)
    },
    {
        "name":     "R1_LLAMA",
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "use_4bit": True,
        "expected_commit_pct": 0.829,  # from C045
    },
    {
        "name":     "QWEN3",
        "model_id": "Qwen/Qwen3-1.7B",  # smaller Qwen3 for T4 budget; use QwQ-32B if A100
        "use_4bit": False,
        "expected_commit_pct": 0.998,  # from C045
    },
]


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
def load_triviaqa_questions(n=N_ITEMS * 3):
    print(f"Loading TriviaQA (n={n})...")
    ds = load_dataset("trivia_qa", "rc.wikipedia", split="validation", trust_remote_code=True)
    items = []
    for ex in ds:
        items.append({
            "question": ex["question"],
            "answers": ex["answer"]["aliases"] if ex["answer"]["aliases"] else [ex["answer"]["value"]],
        })
    random.shuffle(items)
    return items[:n]


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model(model_cfg, hf_token):
    model_id = model_cfg["model_id"]
    use_4bit = model_cfg.get("use_4bit", False)
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
    return model, tokenizer


# ── Step-by-step entropy extraction ──────────────────────────────────────────
@torch.no_grad()
def extract_step_entropies(model, tokenizer, prompt: str, max_steps: int) -> list[float]:
    """
    Generate up to max_new_tokens tokens, collecting output entropy at each step.
    Returns list of entropies (length = actual steps generated).
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    if inputs["input_ids"].shape[1] > 1600:
        return []

    entropies = []
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))

    with torch.no_grad():
        for step in range(min(max_steps, MAX_NEW_TOKENS)):
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
            logits = out.logits[0, -1, :].float()  # [vocab_size]
            probs = torch.softmax(logits, dim=-1)
            log_probs = torch.log(probs + 1e-10)
            entropy = float(-torch.sum(probs * log_probs).item())
            entropies.append(entropy)

            # Greedy next token
            next_token = logits.argmax(dim=-1, keepdim=True).unsqueeze(0)  # [1,1]
            input_ids = torch.cat([input_ids, next_token], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones(1, 1, device=model.device)], dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    return entropies


def compute_commit_pct(entropies: list[float], eps_c: float) -> float:
    """
    Commit point t* = first step where entropy ≤ ε_C.
    commit_pct = (T - t*) / T where T = total steps.
    Returns 0.0 if no commit point found.
    """
    T = len(entropies)
    if T == 0:
        return float("nan")

    for t, e in enumerate(entropies):
        if e <= eps_c:
            return float(T - t) / float(T)

    return 0.0  # never committed below threshold


# ── Run one model ─────────────────────────────────────────────────────────────
def run_model(model, tokenizer, model_cfg, items):
    print(f"\n  Running sensitivity sweep for {model_cfg['name']}...")
    model_name = model_cfg["model_id"]
    results_by_eps = {}

    # Collect entropy trajectories for N items
    print(f"  Collecting entropy trajectories for {N_ITEMS} items...")
    trajectories = []
    t0 = time.time()

    for i, item in enumerate(items[:N_ITEMS]):
        q = item["question"]
        msgs = [{"role": "user", "content": f"Think carefully then answer: {q}"}]
        try:
            prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt = f"Think carefully then answer: {q}\n"

        entropies = extract_step_entropies(model, tokenizer, prompt, MAX_THINK_STEPS)
        if entropies:
            trajectories.append(entropies)

        if (i + 1) % 25 == 0:
            print(f"    {i+1}/{N_ITEMS}  ({time.time()-t0:.0f}s)")

    print(f"  Collected {len(trajectories)} valid trajectories")
    if not trajectories:
        return {"error": "No valid trajectories", "model": model_name}

    # Compute commit_pct for each ε_C
    print(f"\n  ε_C sweep: {EPS_C_VALUES}")
    commit_pcts_by_eps = {}
    for eps_c in EPS_C_VALUES:
        commit_pcts = [compute_commit_pct(traj, eps_c) for traj in trajectories]
        valid = [c for c in commit_pcts if not np.isnan(c)]
        commit_pcts_by_eps[str(eps_c)] = {
            "mean": round(float(np.mean(valid)), 4) if valid else float("nan"),
            "std":  round(float(np.std(valid)), 4) if valid else float("nan"),
            "pct_committed": round(float(np.mean([c > 0 for c in valid])), 4) if valid else float("nan"),
            "n_valid": len(valid),
        }
        print(f"    ε_C={eps_c}: commit_pct={np.mean(valid):.4f} ± {np.std(valid):.4f}  (n={len(valid)})")

    # Summary stats
    means = [commit_pcts_by_eps[str(e)]["mean"] for e in EPS_C_VALUES if not np.isnan(commit_pcts_by_eps[str(e)]["mean"])]
    stds_over_eps = float(np.std(means)) if means else float("nan")
    min_mean = float(np.min(means)) if means else float("nan")

    # Mean entropy by step (for trajectory visualization)
    max_len = max(len(t) for t in trajectories)
    entropy_by_step = []
    for step in range(min(MAX_THINK_STEPS, max_len)):
        step_entropies = [t[step] for t in trajectories if step < len(t)]
        entropy_by_step.append({
            "step": step,
            "mean_entropy": round(float(np.mean(step_entropies)), 4),
            "std_entropy": round(float(np.std(step_entropies)), 4),
            "n": len(step_entropies),
        })

    return {
        "model": model_name,
        "n_trajectories": len(trajectories),
        "commit_pcts_by_eps_c": commit_pcts_by_eps,
        "std_over_eps_c_range": round(float(stds_over_eps), 4),
        "min_mean_commit_pct": round(float(min_mean), 4),
        "expected_commit_pct": model_cfg.get("expected_commit_pct"),
        "entropy_by_step": entropy_by_step[:20],  # first 20 steps
    }


# ── Verdict ───────────────────────────────────────────────────────────────────
def classify_verdict(model_results: dict) -> str:
    if "error" in model_results:
        return "ERROR"

    std = model_results.get("std_over_eps_c_range", float("nan"))
    min_mean = model_results.get("min_mean_commit_pct", float("nan"))

    if np.isnan(std) or np.isnan(min_mean):
        return "INCOMPLETE"
    if std > 0.15 or min_mean < 0.70:
        return "SENSITIVE"
    if std < 0.10 and min_mean >= 0.70:
        return "ROBUST"
    return f"BORDERLINE (std={std:.3f}, min={min_mean:.3f})"


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 70)
    print("EXP_C_EPS_SENSITIVITY — ε_C Sensitivity Analysis for Law 2")
    print("=" * 70)
    print(f"ε_C values: {EPS_C_VALUES}")
    print(f"N items per model: {N_ITEMS}")

    hf_token = _get_hf_token()
    items = load_triviaqa_questions()

    all_results = {}
    for model_cfg in MODELS:
        name = model_cfg["name"]
        print(f"\n{'─' * 50}")
        print(f"MODEL: {name}")
        print(f"{'─' * 50}")

        model, tokenizer = load_model(model_cfg, hf_token)
        result = run_model(model, tokenizer, model_cfg, items)
        all_results[name] = result

        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Save after each model
        _save(all_results)
        print(f"  Saved intermediate to {OUTPUT_FILE}")

    # ── Classify verdicts ────────────────────────────────────────────────────
    verdicts = {name: classify_verdict(r) for name, r in all_results.items()}

    # Overall verdict
    if all(v == "ROBUST" for v in verdicts.values() if v not in ("ERROR", "INCOMPLETE")):
        overall = "ROBUST"
        law2_update = "Law 2 robust across ε_C ∈ [0.05, 0.25]; no threshold specification required"
    elif any(v == "SENSITIVE" for v in verdicts.values()):
        overall = "SENSITIVE"
        law2_update = "Law 2 requires ε_C specification; commit_pct below 70% at low ε_C for one or more models"
    else:
        overall = "MIXED"
        law2_update = "Mixed — some models robust, some borderline; report per-model ε_C sensitivity bounds"

    final = {
        "experiment": "EXP_C_EPS_SENSITIVITY",
        "date": time.strftime("%Y-%m-%d"),
        "eps_c_values": EPS_C_VALUES,
        "model_results": all_results,
        "verdicts_by_model": verdicts,
        "overall_verdict": overall,
        "law2_update": law2_update,
        "c022_update": verdicts.get("R1_QWEN", "UNKNOWN"),
        "c045_update": verdicts.get("R1_LLAMA", "UNKNOWN"),
    }
    _save(final)
    _print_summary(final)


def _np_default(x):
    if isinstance(x, np.integer): return int(x)
    if isinstance(x, (np.floating, np.float32, np.float64)): return float(x)
    if isinstance(x, np.bool_): return bool(x)
    if isinstance(x, np.ndarray): return x.tolist()
    raise TypeError(f"Object of type {type(x).__name__} is not JSON serializable")

def _save(data):
    with open(OUTPUT_FILE, "w") as f:
        json.dump(data, f, indent=2, default=_np_default)


def _print_summary(final):
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    for name, r in final.get("model_results", {}).items():
        if "error" in r:
            print(f"\n  {name}: ERROR — {r['error']}")
            continue
        print(f"\n  {name}:")
        for eps_str, stats in r.get("commit_pcts_by_eps_c", {}).items():
            print(f"    ε_C={eps_str}: commit_pct={stats.get('mean','nan'):.4f} ± {stats.get('std','nan'):.4f}")
        print(f"    std over ε_C range: {r.get('std_over_eps_c_range','nan'):.4f}")
        print(f"    min mean commit_pct: {r.get('min_mean_commit_pct','nan'):.4f}")
        verdict = final["verdicts_by_model"].get(name, "UNKNOWN")
        print(f"    → VERDICT: {verdict}")

    print(f"\n  OVERALL VERDICT: {final.get('overall_verdict')}")
    print(f"  Law 2 update: {final.get('law2_update')}")
    print(f"\n  Output: {OUTPUT_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()
