# Pre-Registration: EXP_A_LAW3_LLAMA
**Question addressed:** Q4 — Does Law 3 (INVERTED_U at L1, MONOTONE_RISE at L2) hold across model families?  
**Date registered:** 2026-07-12  
**Status:** PENDING — not yet run  
**Kernel:** `experiments/exp_a_law3_llama/exp_a_law3_llama.py`  
**Competing theories discriminated:** H-A (Compression) vs H-B (Supervision) vs H-C (Architecture)  
See: `docs/COMPETING_THEORIES.md`

---

## Why This Experiment

C042 (L1 INVERTED_U) and C043 (L2 MONOTONE_RISE) are SUPPORTED on Qwen backbone only (N=200/class, matched).
A single-backbone finding cannot be called a Law. Replication on Llama backbone determines:
- Whether H-A (Compression) explains both families
- Whether H-B (Supervision) explains both families  
- Whether H-C (Architecture) is falsified for Llama as well

This is the highest-priority remaining experiment. Its outcome determines whether C042/C043 are
promoted to CONFIRMED or scoped to "Qwen-specific observation."

---

## Protocol

**Models (three training stages on Llama backbone):**

| Stage | Model | Notes |
|---|---|---|
| BASE | `meta-llama/Llama-3.2-3B` | Pure pretraining, no instruction tuning |
| INSTRUCT | `meta-llama/Llama-3.2-3B-Instruct` | SFT + RLHF |
| REASONING | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | ⚠ Size confound: 8B vs 3B |

**Size confound note:** Stages 1-2 are 3B; Stage 3 is 8B. Absolute AUROC comparison between Stage 3
and Stages 1-2 is confounded by model size. Directional comparison (does REASONING differ from INSTRUCT
in the direction predicted by H-A vs H-B?) remains informative. If budget allows, run Stage 3 also on
`meta-llama/Llama-3.2-3B-Instruct` as matched-size proxy for reasoning capability at 3B scale.

**Task:** TriviaQA bilateral oracle (same as C001-C004 primary experiments)

**L1 design:**
- Bilateral oracle: PARAM (nocontext_F1 ≥ 0.50), CTX_DEP (nocontext_F1 ≤ 0.05 AND withcontext_F1 ≥ 0.50)
- N = 200/class (matched across all stages — this is critical for valid comparison)
- Pool = 10,000 items
- Fisher+PCA64 at Layer 26, Step 1
- Entropy baseline at same extraction point
- Shuffled control (3 seeds)
- Bootstrap CI (n=1000)

**L2 design:**
- CO labeling: correct (F1 ≥ 0.50, entropy ≤ θ_conf) vs wrong (F1 ≤ 0.10, entropy ≤ θ_conf)
- θ_conf = 0.15 (entropy-matched to exclude uncertain generations)
- N = 100/class (or maximum available if pool exhausted)
- Same Fisher+PCA64 estimator and controls

**Layer index:** 26 for all three stages (Llama-3.2-3B has 28 layers — L26 is layer index 26 of 28).
For DeepSeek-R1-Distill-Llama-8B (32 layers), use proportional: round(26/28 * 32) = 30. Document both.

---

## Pre-Registered Verdict Criteria

These criteria are locked before the experiment runs. Results will be classified by applying them
mechanically to the output AUROC values. No post-hoc adjustment.

### L1 Verdicts (from H-A/H-B/H-C predictions):

| Verdict | Condition | Theory implication |
|---|---|---|
| `INVERTED_U` | INSTRUCT > BASE by > 0.05 AND REASONING < INSTRUCT by > 0.03 | Supports H-A; falsifies H-B (at L1); falsifies H-C |
| `MONOTONE_RISE` | BASE < INSTRUCT < REASONING, each step > 0.03 | Supports H-B; weakens H-A (at L1); falsifies H-C |
| `FLAT` | max − min < 0.05 across all three stages | Supports H-C; falsifies H-A and H-B |
| `MIXED` | Partial rise/fall not matching above patterns | Ambiguous — report raw numbers, update H-A/H-B/H-C notes |

**If size confound is severe:** apply INVERTED_U/MONOTONE_RISE criteria only to Stage 1 vs Stage 2 (matched 3B).
Stage 3 directional comparison is exploratory only.

### L2 Verdicts:

| Verdict | Condition | Theory implication |
|---|---|---|
| `MONOTONE_RISE` | BASE < INSTRUCT < REASONING, each step > 0.03 | Consistent with H-A and H-B |
| `FLAT` | max − min < 0.05 | Inconsistent with H-A and H-B; would require new explanation |
| `REASONING_JUMP` | REASONING >> INSTRUCT by > 0.10, BASE ≈ INSTRUCT | RL amplification — consistent with H-A |

### Law 3 promotion criteria:

- **C042 (L1 INVERTED_U) promoted to CONFIRMED:** L1 verdict = INVERTED_U on Llama backbone
- **C043 (L2 MONOTONE_RISE) promoted to CONFIRMED:** L2 verdict = MONOTONE_RISE on Llama backbone
- **C042/C043 scoped to "Qwen-specific":** Both verdicts diverge from Qwen pattern

---

## What to Report

Regardless of verdict, report:
1. Raw AUROC ± 95% CI for each stage × task (L1 and L2) 
2. Shuffled control AUROC for each (must be < real AUROC by > 0.15 for valid signal)
3. N achieved per class per stage (may be < 200 for base model due to low CTX_DEP yield)
4. The verdict label applied from the criteria above
5. Theory implications from COMPETING_THEORIES.md discrimination matrix
6. Layer index used for each model and justification

---

## Expected Runtime and GPU Requirements

- Stage 1 (BASE 3B): ~3h on T4 (bilateral oracle scanning 10k items + extraction)
- Stage 2 (INSTRUCT 3B): ~3h on T4
- Stage 3 (REASONING 8B, if loaded in 4-bit): ~5h on T4
- Total: ~11h — may require two T4 sessions (stages 1-2 in session A, stage 3 in session B)
- Save intermediate outputs after each stage to `/kaggle/working/` to prevent loss on timeout

---

## Outputs

```
exp_a_law3_llama_results.json:
  {
    "stage_BASE": {
      "model": "meta-llama/Llama-3.2-3B",
      "L1": {"auroc": float, "ci_low": float, "ci_high": float, "n_param": int, "n_ctxdep": int},
      "L2": {"auroc": float, "ci_low": float, "ci_high": float, "n_cc": int, "n_cw": int},
      "shuffled_L1": float,
      "layer_used": int
    },
    "stage_INSTRUCT": {...},
    "stage_REASONING": {...},
    "L1_verdict": str,
    "L2_verdict": str,
    "law3_c042_promoted": bool,
    "law3_c043_promoted": bool,
    "theory_implications": {...}
  }
```

---

*Registered: 2026-07-12 by Lakshmi-Chakradhar Vijayarao*  
*See COMPETING_THEORIES.md for full theory formalization.*  
*See OPEN_QUESTIONS.md Q4 for the scientific question context.*
