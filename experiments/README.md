# Kaggle T4 Experiment Scripts

Three scripts for the GPU-required experiments. Run in order.

## 1. Calibrate (run first)

**File:** `kaggle_calibration_llama3b.py`  
**Runtime:** ~20 min on T4, ~6GB VRAM  
**Model:** `Qwen/Qwen2.5-3B-Instruct` (default, no HF token) or `meta-llama/Llama-3.2-3B-Instruct` (set `HF_TOKEN` secret)

Runs bilateral oracle on TriviaQA. Outputs:
- `/kaggle/working/llama3b_cal.json` — CalibrationState JSON
- `/kaggle/working/calibration_log.json` — per-sample oracle scores

## 2. Adversarial Battery

**File:** `kaggle_adversarial_battery.py`  
**Runtime:** ~35 min on T4  
**Requires:** Step 1 checkpoint attached as Kaggle dataset

Five attacks × 50 questions. Key verdict: `GEOMETRY_IS_EPISTEMIC`
- paraphrase / lexical / retrieval_poison → routing stable (<15% change)
- negation / false_premise → appropriately shifts (>20% change)

Outputs: `/kaggle/working/adversarial_results.json`, `adversarial_summary.json`

## 3. Cross-Arch Validation

**File:** `kaggle_cross_arch.py`  
**Runtime:** ~60 min on T4 (phi + qwen)  
**Models:** Phi-3.5-mini + Qwen2.5-3B (default), optionally Mistral-7B

Bilateral oracle on each architecture. Key verdict: `SIGNAL_IS_ARCHITECTURAL`

Expected results:
- Phi-3.5-mini (dense MHA): STRONG (≥0.90)
- Qwen2.5-3B (GQA): STRONG (≥0.92 per prior results)
- Mistral-7B (SWA+GQA): WEAK (documented, expected)

Outputs: `/kaggle/working/cross_arch_results.json`, `cross_arch_summary.json`

## Setup

```
# Kaggle notebook setup — paste at the top of any script
import os
os.environ["MODEL_ID"]  = "Qwen/Qwen2.5-3B-Instruct"
os.environ["N_TARGET"]  = "80"
os.environ["MAX_SCAN"]  = "2000"
# For adversarial battery, after saving calibration as dataset:
os.environ["CAL_PATH"]  = "/kaggle/input/my-calibration/llama3b_cal.json"
```

Upload each script as a Kaggle notebook (Python script mode). Attach the calibration output from Step 1 as a dataset for Step 2.
