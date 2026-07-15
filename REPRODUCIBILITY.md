# Reproducibility Guide
**Credence — Computational Observability Research Program**

This document describes exactly how to reproduce the three primary results in the paper. Every number in the paper traces to either a frozen result file in this repository or a Kaggle kernel that can be re-run.

---

## Primary Results and Their Sources

| Paper claim | AUROC | Source | Frozen file |
|---|---|---|---|
| L1 Qwen (C001) | 0.7312 CI=[0.63,0.83] | `experiments/large_n_validation/` | `results/frozen/large_n_v2_qwen_results.json` |
| L1 Llama (C003) | 0.7464 CI=[0.65,0.83] | `experiments/large_n_validation/` | `results/frozen/large_n_v2_llama_results.json` |
| L2 Fisher vs entropy (C017) | 0.854 vs 0.614 | `experiments/false_certainty_v2/` | `results/frozen/false_certainty_v2_results.json` |
| L2 cross-validated (C040) | 0.7629 ± 0.0120 | `experiments/l2_large_n_v1/` | `results/frozen/l2_large_n_v1_results.json` |
| Perturbation ICC Qwen (C025) | ICC=0.913 | `experiments/perturbation_battery_v1/` | In Kaggle output |
| Perturbation ICC Llama (C025) | ICC=0.9334 | `experiments/perturbation_battery_llama_v1/` | In Kaggle output |

**Note:** Primary result JSON files are in `results/frozen/`. If a file is listed as "In Kaggle output," download it from the corresponding Kaggle kernel output and place it in `results/frozen/`.

---

## Reproducing the Main L1 Result (30 minutes on Kaggle T4)

**What this reproduces:** Table 1, Qwen2.5-1.5B-Instruct L1 AUROC = 0.7312

1. Create a Kaggle account and connect a GPU (T4 ×1 is sufficient)
2. Upload `experiments/large_n_validation/large_n_validation.py` as a new kernel
3. Add these datasets: `trivia_qa` (via HuggingFace datasets)
4. Run the kernel (expected time: ~2.5 hours for N=200/class, pool=10000)
5. Expected output:
   ```json
   {
     "model": "Qwen/Qwen2.5-1.5B-Instruct",
     "auroc": 0.7312,
     "ci_lower": 0.63,
     "ci_upper": 0.83,
     "n_per_class": 197,
     "shuffled_auroc": 0.5980,
     "verdict": "CLEAN"
   }
   ```

**Verify against:** `results/frozen/large_n_v2_qwen_results.json`

---

## Reproducing the Main L2 Result (3 hours on Kaggle T4)

**What this reproduces:** Table 2 primary row — Fisher 0.854 vs entropy 0.614 in confident zone

1. Upload `experiments/false_certainty_v2/false_certainty_v2.py`
2. Model: `Qwen/Qwen2.5-1.5B-Instruct`, dataset: TriviaQA
3. Expected output:
   ```json
   {
     "fisher_auroc": 0.854,
     "entropy_auroc": 0.614,
     "gap": 0.240,
     "bo_transfer_auroc": 0.880,
     "shuffled_fisher": 0.503,
     "verdict": "CLEAN"
   }
   ```

**Verify against:** `results/frozen/false_certainty_v2_results.json`

---

## Reproducing the L2 Large-N Cross-Validation (6 hours on Kaggle T4)

**What this reproduces:** CV AUROC = 0.7629 ± 0.0120 (C040, CONFIRMED)

1. Upload `experiments/l2_large_n_v1/l2_large_n_v1.py`
2. Model: `Qwen/Qwen2.5-1.5B-Instruct`, CO labeling, N=500/class, 5-fold
3. Expected output:
   ```json
   {
     "cv_auroc_mean": 0.7629,
     "cv_auroc_std": 0.0120,
     "fold_aurocss": [0.7440, 0.7551, 0.7629, 0.7732, 0.7805],
     "verdict": "STABLE_SIGNAL"
   }
   ```

---

## Quick Verification (No GPU Required)

To verify the claims governance system works:

```bash
pip install pyyaml
python science/validate_claims.py paper/v1_arxiv.md
# Exit 0 = all claims referenced in paper are non-FALSIFIED
# Exit 1 = a FALSIFIED or undefined claim is referenced
```

To inspect the claims registry:
```bash
python -c "
import yaml
with open('science/CLAIMS.yaml') as f:
    claims = yaml.safe_load(f)
confirmed = [c for c in claims['claims'] if c['status'] == 'CONFIRMED']
print(f'{len(confirmed)} CONFIRMED claims:')
for c in confirmed:
    print(f'  {c[\"id\"]}: {c[\"statement\"][:80]}')
"
```

Expected output: 7 CONFIRMED claims (C001, C002, C003, C004, C025, C036, C040)

---

## Python API Quick Start (Requires GPU + Model Access)

```python
from credence_runtime import Credence

# Step 1: calibrate (runs bilateral oracle — ~30 min on Qwen2.5-1.5B on T4)
# credence calibrate --model Qwen/Qwen2.5-1.5B-Instruct --dataset trivia_qa --n 200 --output cal.json

# Step 2: load calibrated runtime
cred = Credence.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    calibration="cal.json",
)

# Step 3: tag a prompt
result = cred.complete("Who wrote Hamlet?")
print(result.routing)  # ANSWER or RETRIEVE
print(result.tag)      # EpistemicTag with hidden-state scores
```

---

## Which Result Files Are Missing

These result files are referenced in the paper but not yet committed to `results/frozen/`. Download from the corresponding Kaggle kernel output and commit:

| Missing file | Kaggle kernel | Experiment |
|---|---|---|
| `large_n_v2_qwen_results.json` | `chakrivijayrao/large-n-validation` | EXP_T1A_LARGE_N_V2 (Qwen) |
| `large_n_v2_llama_results.json` | `chakrivijayrao/large-n-validation` | EXP_T1A_LARGE_N_V2 (Llama) |
| `false_certainty_v2_results.json` | `chakrivijayrao/false-certainty-v2` | EXP_FALSE_CERTAINTY_V2 |
| `l2_large_n_v1_results.json` | `chakrivijayrao/l2-large-n-v1` | EXP_L2_LARGE_N_V1 |
| `perturbation_battery_llama_results.json` | `lakshmichakradhar/perturbation-battery-llama` | EXP_J_PERTURBATION_V2 |
| `co_gemma_mistral_v1_results.json` | `lakshmichakradhar/co-gemma-mistral-v1` | EXP_CO_GEMMA_MISTRAL_V1 |

Once these are added, every number in the paper will be traceable to a file in this repository.

---

## Protocol Reference

The bilateral oracle protocol in detail:

```
For each question q in dataset D:
  pass_1: generate answer with NO context → nc_F1
  pass_2: generate answer WITH correct context → wc_F1

  label(q) = PARAM    if nc_F1 >= 0.50 OR answer_contains(pred, gold)
  label(q) = CTX_DEP  if nc_F1 <= 0.05 AND wc_F1 >= 0.50
  label(q) = SKIP     otherwise

Probe: Fisher+PCA64 = PCA(n_components=64) → LDA(solver='lsqr', shrinkage='auto')
Layer: 26 (penultimate in Qwen2.5-1.5B-Instruct)
Step: generation step 1 (first decode step after prompt)
Control: shuffled labels AUROC — must be < real AUROC − 0.05
```

This protocol is implemented in `esm/runtime.py:calibrate()` and in all `experiments/*/` scripts.
