# REPRODUCE.md — Claim-to-Experiment Mapping

**Detection Without Control: Confabulation Geometry Is Dissociable from Causal Answer Generation in Language Models**

Every CONFIRMED and SUPPORTED claim from `science/CLAIMS.yaml` is mapped below to the exact script, Kaggle kernel, frozen result file (where available), and key numerical result. FALSIFIED, EXPLORATORY, and INCONCLUSIVE claims are omitted.

Run `python science/validate_claims.py` to verify the claims registry is self-consistent before any paper revision.

---

## How to Reproduce

All experiments run on Kaggle T4 GPUs unless stated otherwise. Scripts live in `experiments/<dir>/`. Frozen JSON outputs live in `results/frozen/`. Where a frozen file does not yet exist, download from the Kaggle kernel output and place it there.

**Common dependencies:** `transformers>=4.40`, `torch>=2.2`, `scikit-learn>=1.4`, `datasets>=2.18`, `bitsandbytes>=0.43` (for 4-bit models), `numpy`, `scipy`.

---

## CONFIRMED Claims (7)

These claims have ≥2 architectures + clean controls + N≥128/class. They may be cited in the paper without qualification beyond their stated scope.

---

### C001 — CONFIRMED
**Fisher+PCA64 (L26, step-1) achieves AUROC ≥ 0.70 for bilateral oracle PARAM/CTX_DEP on TriviaQA across two architectures.**

Primary estimates (large-N, shuffled, clean): Qwen2.5-1.5B-Instruct = 0.7312, Llama-3.2-3B-Instruct = 0.7464. Earlier estimates (0.841/0.846) had ordering bias and are superseded.

| Experiment | Script | Kaggle Kernel | Results File | Key Result |
|---|---|---|---|---|
| EXP_LARGE_N_V2 (Qwen) | `experiments/large_n_validation/large_n_validation.py` | `chakrivijayarao/large-n-validation` | `results/frozen/large_n_v2_qwen_results.json` | AUROC=0.7312 CI=[0.63,0.83] shuffled=0.598 CLEAN N=197 |
| EXP_LARGE_N_V2 (Llama) | `experiments/large_n_validation/large_n_validation.py` | `chakrivijayarao/large-n-validation` | `results/frozen/large_n_v2_llama_results.json` | AUROC=0.7464 CI=[0.65,0.83] shuffled=0.502 CLEAN N=200 |
| EXP_T1A_LARGE_N_V1 (Qwen) | `experiments/large_n_validation/large_n_validation.py` | — | `results/frozen/large_n_validation_results_v1.json` | AUROC=0.6566 CI=[0.52,0.79] N=121 (lower bound; pool too small) |
| EXP_C3V3 (superseded) | `experiments/nonlinear_probe_v3/nonlinear_probe_v3.py` | `chakrivijayarao/nonlinear-probe-recovery-c3-v3-fixed-probes` | `results/frozen/c3v3_results.json` | Qwen=0.841 shuffled=0.617 WARN; Llama=0.846 shuffled=0.427 CLEAN — both superseded |

---

### C002 — CONFIRMED
**The bilateral oracle signal is linearly organized at layer 26. No nonlinear probe improves AUROC above linear Fisher+PCA64 by more than 0.05.**

PCA128 is marginally better; PCA256 collapses (ill-conditioned); full-space Fisher is best at large N. Delta nonlinear−linear ≤ 0.019 across both architectures.

| Experiment | Script | Kaggle Kernel | Results File | Key Result |
|---|---|---|---|---|
| EXP_C3V3 | `experiments/nonlinear_probe_v3/nonlinear_probe_v3.py` | `chakrivijayarao/nonlinear-probe-recovery-c3-v3-fixed-probes` | `results/frozen/c3v3_results.json` | Qwen Δ=−0.019, Llama Δ=−0.005 (NO_RECOVERY) |
| EXP_B_FULLSPACE_FISHER_V1 | `experiments/exp_b_fullspace_fisher/exp_b_fullspace_fisher.py` | — | `results/frozen/exp_b_fullspace_fisher_results.json` | N=135: PCA64=0.747, full=0.818, MLP=0.652. PCA_BOTTLENECK cosim=0.547 |
| EXP_B_FULLSPACE_FISHER_V2 | `experiments/exp_b_fullspace_fisher/exp_b_fullspace_fisher.py` | — | `results/frozen/exp_b_fullspace_fisher_results.json` | N=175: PCA64=0.741, full=0.753, MLP=0.721. PCA_VALIDATED cosim=0.457 |

---

### C003 — CONFIRMED
**Fisher+PCA64 bilateral oracle AUROC replicated across Qwen2.5-1.5B-Instruct and Llama-3.2-3B-Instruct (statistically indistinguishable despite 2× hidden dimension difference).**

| Experiment | Script | Kaggle Kernel | Results File | Key Result |
|---|---|---|---|---|
| EXP_LARGE_N_V2 | `experiments/large_n_validation/large_n_validation.py` | `chakrivijayarao/large-n-validation` | `results/frozen/large_n_v2_qwen_results.json` / `large_n_v2_llama_results.json` | Qwen=0.7312 vs Llama=0.7464, Δ=0.015, CIs fully overlap |

---

### C004 — CONFIRMED
**The bilateral oracle two-pass protocol (PARAM: nc_F1≥0.50; CTX_DEP: nc_F1≤0.05 AND wc_F1≥0.50) produces well-separated hidden-state distributions.**

| Experiment | Script | Kaggle Kernel | Results File | Key Result |
|---|---|---|---|---|
| EXP_C3V3 | `experiments/nonlinear_probe_v3/nonlinear_probe_v3.py` | `chakrivijayarao/nonlinear-probe-recovery-c3-v3-fixed-probes` | `results/frozen/c3v3_results.json` | Protocol yields AUROC 0.841–0.846 across two architectures |
| EXP_2X2V1 | — | — | — | HotpotQA AUROC=0.9429 (j_score OOD) |

---

### C025 — CONFIRMED
**Fisher+PCA64 decision scores at L26 step-1 are invariant under REPHRASE / LOWERCASE / APPEND / TYPO surface perturbations. ICC≥0.91 on both architectures.**

| Experiment | Script | Kaggle Kernel | Results File | Key Result |
|---|---|---|---|---|
| EXP_J_PERTURBATION_BATTERY_V1 (Qwen) | `experiments/perturbation_battery_v1/perturbation_battery_v1.py` | `chakrivijayarao/perturbation-battery-v7` (final: v8) | — | ICC=0.913, ratio=10.5:1, N=160 items×4 variants, ROBUST |
| EXP_J_PERTURBATION_BATTERY_V2 (Llama) | `experiments/perturbation_battery_v1/perturbation_battery_v1.py` | — | — | ICC=0.9334, ratio=14.0:1. REPHRASE corr=0.910, TYPO corr=0.922, ROBUST |

---

### C036 — CONFIRMED
**CO labeling (entropy ≤ θ_conf, no window) recovers L2 confabulation detection for Gemma-2-2B-IT and Mistral-7B-Instruct-v0.3, where entropy-matched framing failed (T2_L2). Fisher gaps ≥ 0.20 on both architectures.**

| Experiment | Script | Kaggle Kernel | Results File | Key Result |
|---|---|---|---|---|
| EXP_CO_GEMMA_MISTRAL_V1 | `experiments/co_gemma_mistral_v1/co_gemma_mistral_v1.py` | — | — | Gemma: Fisher=0.8368 CI=[0.749,0.910] gap=0.206 CLEAN; Mistral: Fisher=0.8580 CI=[0.777,0.927] gap=0.211 CLEAN |

---

### C040 — CONFIRMED
**Large-N cross-validated L2 AUROC (CO-style, Qwen, N=500/class, 5-fold) = 0.7629 ± 0.0120. Stable: fold range [0.744, 0.781]. All folds shuffled CLEAN.**

Resolves the item-subset variance concern in C034: prior estimates 0.670–0.885 were pool-section heterogeneity, not probe instability.

| Experiment | Script | Kaggle Kernel | Results File | Key Result |
|---|---|---|---|---|
| EXP_L2_LARGE_N_V1 | `experiments/l2_large_n_v1/l2_large_n_v1.py` | — | `results/frozen/l2_large_n_v1_results.json` | CV_Fisher=0.7629, std=0.0120, CV_Gap=0.163, all folds CLEAN |

---

## SUPPORTED Claims

These claims have valid evidence from one architecture or N<128/class. They are citable but require an explicit needs_replication note.

---

### C005 — SUPPORTED
**Centroid-direction residual-stream patching at L4–L26 step-1 produces no causal improvement in CTX_DEP F1 (specific Δ≤0.000 at all layers). Extended by C024 to the CC/CW domain.**

SCOPE: centroid-direction mean patching only. Head-level / SAE / circuit-level patching untested.

| Experiment | Script | Kaggle Kernel | Results File | Key Result |
|---|---|---|---|---|
| EXP_P1V3 (single layer) | `experiments/activation_patching_p1/` | `chakrivijayarao/activation-patching-bilateral-p1` | — | Alpha=2.0 Δ=+0.0054 == shuffled Δ=+0.0054; specific_delta=0.0000 EPIPHENOMENAL |
| EXP_P1V5 (layer sweep L4–L24) | `experiments/activation_patching_p1/` | `chakrivijayarao/activation-patching-bilateral-p1` | — | ALL patch_Δ=0.0000 across L4/L8/L12/L16/L20/L22/L24 |
| EXP_H_CC_CW_PATCHING_V1 (CC/CW domain) | `experiments/cc_cw_patching_v1/cc_cw_patching_v1.py` | `chakrivijayarao/cc-cw-patching-v1` | — | max Δ_F1=+0.0004 across λ∈{0,0.25,0.5,1,2,4} EPIPHENOMENAL |

---

### C006 — SUPPORTED
**Layer 26 step-1 bilateral oracle AUROC is substantially higher than prefill (mean step-1=0.785 vs prefill=0.567 across 4 models). Within generation, AUROC rises monotonically for EOS-filtered verbose items.**

| Experiment | Script | Kaggle Kernel | Results File | Key Result |
|---|---|---|---|---|
| EXP_ESM_V33 | — | — | `results/frozen/esm_table2_results.json` | Mean step-1=0.785 vs prefill=0.567 across 4 models |
| EXP_T1D_STEP_INDEX_V3 | `experiments/step_index_auroc/step_index_auroc.py` | `chakrivijayarao/step-index-auroc-profile-task-1-1` | `results/frozen/step_index_results_v3.json` | MONOTONE_RISE: step-0=0.639 → step-2=0.762 → step-10=0.906 (EOS-filtered n=44/class) |

---

### C007 — SUPPORTED
**Fisher trajectory AUROC (28 J-scores as sequence features, LDA) = 0.9947, substantially above single-point Fisher+PCA64.**

| Experiment | Script | Kaggle Kernel | Results File | Key Result |
|---|---|---|---|---|
| EXP_TRAJECTORY_LDA | — | — | `results/frozen/esm_trajectory_results.json` | trajectory_auroc=0.9947 vs single-point L26=0.8464 |

---

### C015 — SUPPORTED
**Bilateral oracle geometry is BIMODAL: STRONG_PARAM (+1.316) and STRONG_CTX_DEP (−1.316) form distinct poles. WEAK/BORDERLINE items are geometrically indistinguishable (KS p=0.120).**

| Experiment | Script | Kaggle Kernel | Results File | Key Result |
|---|---|---|---|---|
| EXP_T1C_BORDERLINE | `experiments/borderline_geometry/borderline_geometry.py` | `chakrivijayarao/borderline-population-geometry-task-1-2` | `results/frozen/borderline_geometry_results_v4.json` | STRONG_P=+1.316, STRONG_C=−1.316, train AUROC=0.9631, N=60/group |

---

### C016 — SUPPORTED
**Output token entropy at step-1 achieves AUROC 0.9043 (Qwen) and 0.874 (Llama) for bilateral oracle L1 classification — equal to or exceeding Fisher+PCA64. PARAM≈1.1–1.2 nats; CTX_DEP≈2.4–3.3 nats (2–3× gap).**

| Experiment | Script | Kaggle Kernel | Results File | Key Result |
|---|---|---|---|---|
| EXP_ENTROPY_BASELINE (Qwen) | `experiments/entropy_baseline/entropy_baseline.py` | — | — | Entropy-raw AUROC=0.9043 N=121/class; PARAM=1.187 nats, CTX_DEP=2.399 nats |
| EXP_ENTROPY_BASELINE (Llama) | `experiments/entropy_baseline/entropy_baseline.py` | — | — | Entropy-raw AUROC=0.874 N=150/class; PARAM=1.114 nats, CTX_DEP=3.329 nats |

---

### C017 — SUPPORTED
**Fisher+PCA64 (L26, step-1) distinguishes CONFIDENT_CORRECT from CONFIDENT_WRONG in the entropy-matched confident zone. Fisher-entropy gap: Qwen gap=0.240 (AUROC=0.854), Llama gap=0.365 (AUROC=0.818).**

Also see baseline comparison: Fisher=0.779 exceeds best behavioral baseline (self-consistency B2=0.613) by +0.166 on entropy-matched CC/CW.

| Experiment | Script | Kaggle Kernel | Results File | Key Result |
|---|---|---|---|---|
| EXP_A_FALSE_CERTAINTY_V2 (Qwen) | `experiments/false_certainty_v2/false_certainty_v2.py` | `chakrivijayarao/false-certainty-detection-v2-exp-a-pivot` | `results/frozen/false_certainty_v2_results.json` | Fisher=0.8544, Entropy=0.6144, gap=0.240, θ_conf=1.1043, N=100/class |
| EXP_FALSE_CERTAINTY_LLAMA_V1 (Llama) | `experiments/false_certainty_llama_v1/false_certainty_llama_v1.py` | `chakrivijayarao/false-certainty-llama-v2` | — | Fisher=0.8175, Entropy=0.453, gap=0.365, θ_conf=1.20, N=80/class |
| EXP_1_BASELINE_COMPARISON_V1 | `experiments/baseline_comparison_v1/baseline_comparison_v1.py` | — | — | Fisher=0.779, B1_verbalized=0.605, B2_self_consistency=0.613, B3_top1=0.384, B_entropy=0.483 |

---

### C018 — SUPPORTED
**The bilateral oracle probe (PARAM vs CTX_DEP) transfers to CC/CW confabulation detection: Qwen BO_Transfer=0.880, Llama BO_Transfer=0.768. Both architectures ≥ 0.70.**

| Experiment | Script | Kaggle Kernel | Results File | Key Result |
|---|---|---|---|---|
| EXP_A_FALSE_CERTAINTY_V2 (Qwen) | `experiments/false_certainty_v2/false_certainty_v2.py` | `chakrivijayarao/false-certainty-detection-v2-exp-a-pivot` | `results/frozen/false_certainty_v2_results.json` | BO_Transfer=0.880 > direct=0.854 (Δ=+0.026) |
| EXP_FALSE_CERTAINTY_LLAMA_V1 (Llama) | `experiments/false_certainty_llama_v1/false_certainty_llama_v1.py` | `chakrivijayarao/false-certainty-llama-v2` | — | BO_Transfer=0.768 (n_test=20/class, wide CI) |

---

### C019 — SUPPORTED
**15-step entropy trajectory (KV-cached) distinguishes CC from CW with LR AUROC=0.730. Inversion at steps 2–3. CW items show entropy burst (0.75→1.57); CC items flat.**

| Experiment | Script | Kaggle Kernel | Results File | Key Result |
|---|---|---|---|---|
| EXP_D_ENTROPY_TRAJECTORY_V1 (v2 script) | `experiments/entropy_trajectory_v1/entropy_trajectory_v1.py` | `chakrivijayarao/entropy-trajectory-v1` | — | trajectory_auroc=0.730, max_per_step=0.617 at step-4, inversion at step-0 (AUROC=0.331) |

---

### C020 — SUPPORTED
**Architecture family predicts bilateral oracle Fisher AUROC better than parameter count. Qwen2.5-1.5B Fisher=0.845 vs Llama-3.2-3B Fisher=0.608 (gap=0.238) despite 2× fewer parameters.**

| Experiment | Script | Kaggle Kernel | Results File | Key Result |
|---|---|---|---|---|
| EXP_E_SCALE_OBS_V1 | `experiments/scale_obs_v1/scale_obs_v1.py` | `chakrivijayarao/scale-obs-v1` | — | Qwen1.5B Fisher=0.845, Llama3B Fisher=0.608, Spearman ρ=−0.50 (FLAT_BOTH). FAMILY_DIVERGENCE gap=0.238 |

---

### C021 — SUPPORTED
**The bilateral oracle protocol has a parametric capability Goldilocks zone for BASE models on TriviaQA (~1B–2B params). The upper ceiling does NOT apply to instruction-tuned models (7B-Instruct viable: C029).**

| Experiment | Script | Kaggle Kernel | Results File | Key Result |
|---|---|---|---|---|
| EXP_E_SCALE_OBS_V1 | `experiments/scale_obs_v1/scale_obs_v1.py` | `chakrivijayarao/scale-obs-v1` | — | Qwen0.5B: PARAM=0/3000 (floor); Qwen3B: CTX_DEP=9/3000 (ceiling); 1.5B–3B viable |
| EXP_SCALE_EXTENSION_V1 | `experiments/scale_extension_v1/scale_extension_v1.py` | `chakrivijayarao/scale-extension-v1` | — | Qwen7B-Instruct: PARAM=50, CTX_DEP=50 from 486 items — ceiling falsified for instruct |

---

### C022 — SUPPORTED
**Early commitment observed across reasoning-distilled models: R1-Qwen commit_pct=75.8% (z=49.77), R1-Llama commit_pct=82.9% (z=679.73). Teacher confound resolved by C045 (Qwen3 commit_pct=99.8%).**

| Experiment | Script | Kaggle Kernel | Results File | Key Result |
|---|---|---|---|---|
| EXP_B_REASONING_GEOMETRY_V1 (R1-Qwen) | `experiments/reasoning_geometry_v1/reasoning_geometry_v1.py` | `chakrivijayarao/reasoning-geometry-v1` | — | commit_pct=75.8%, z=49.77, tokens_saved=75.9%, F1_delta=+0.008 |
| EXP_C_REASONING_GEOMETRY_LLAMA_V1 (R1-Llama) | `experiments/reasoning_geometry_llama_v1/reasoning_geometry_llama_v1.py` | `chakrivijayarao/reasoning-geometry-llama-v1` | — | commit_pct=82.9%, z=679.73, Cal AUROC L28=1.000 (N=10 saturation) |

---

### C024 — SUPPORTED
**Centroid-direction patching of CC/CW Fisher geometry at L26 produces no causal F1 improvement at any patch magnitude. Max Δ_F1=+0.0004. EPIPHENOMENAL. Extends C005 to the confabulation domain (4× larger gap).**

| Experiment | Script | Kaggle Kernel | Results File | Key Result |
|---|---|---|---|---|
| EXP_H_CC_CW_PATCHING_V1 | `experiments/cc_cw_patching_v1/cc_cw_patching_v1.py` | `chakrivijayarao/cc-cw-patching-v1` | — | λ∈{0,0.25,0.5,1,2,4}: Δ_F1∈{0,0,−0.0006,+0.0004,+0.0004,+0.0001}. Kill criterion triggered. |

---

### C026 — SUPPORTED
**The bilateral oracle requires a minimum instruction-following capability. Pythia-1.4b (pure base LM) yields CTX_DEP=0 at all checkpoints. Qwen2.5-1.5B-Base passes (CTX_DEP=50 from 232 items). This is a capability threshold, not a universal base-LM failure.**

| Experiment | Script | Kaggle Kernel | Results File | Key Result |
|---|---|---|---|---|
| EXP_K_PYTHIA_LARGE_N_V4 | `experiments/pythia_sweep_large_n/pythia_sweep_large_n.py` | `chakrivijayarao/pythia-sweep-large-n-v4` | — | CTX_DEP=0 at step16k/33k/66k/143k after 7950+ items each. BILATERAL_ORACLE_INAPPLICABLE |
| EXP_L_STAGE_SWEEP_V1 (Qwen base) | `experiments/exp_l_stage_sweep_v1/exp_l_stage_sweep_v1.py` | `chakrivijayarao/exp-l-stage-sweep-v1` | — | CTX_DEP=50 from 232 items. ORACLE_APPLICABLE. Fisher AUROC=0.888 but shuffled=0.876 (near-random gap) |

---

### C027 — SUPPORTED
**Post-answer entropy burst pattern (CW spike then collapse) observed in both base Qwen/TriviaQA (EXP-D) and reasoning-distilled Qwen/GSM8K (EXP-G). Trajectory AUROC=0.842 (GSM8K).**

| Experiment | Script | Kaggle Kernel | Results File | Key Result |
|---|---|---|---|---|
| EXP_G_REASONING_ENTROPY_TRAJ_V1 | `experiments/reasoning_entropy_traj_v1/reasoning_entropy_traj_v1.py` | `chakrivijayarao/reasoning-entropy-traj-v1` | — | trajectory_auroc=0.8424, BURST pattern, peak step=4 AUROC=0.693. CW step-1 entropy=1.322, step-4=0.127 |

---

### C028 — SUPPORTED
**Truncating reasoning chain at Fisher commit point achieves 87.4% reduction in thinking tokens at mean quality cost of +0.006 F1. Commit detected in 99.5% of items.**

| Experiment | Script | Kaggle Kernel | Results File | Key Result |
|---|---|---|---|---|
| EXP_I_EARLY_EXIT_CAUSAL_V1 | `experiments/early_exit_causal_v1/early_exit_causal_v1.py` | `chakrivijayarao/early-exit-causal-v1` | — | N=200, mean_Δf1=+0.0059 SD=0.047, p=0.08 two-tailed, helped=7.54%, hurt=4.02%, mean_commit_pct=87.4% |

---

### C029 — SUPPORTED
**Fisher+PCA64 AUROC improves with scale for Qwen2.5-Instruct: 1.5B→7B AUROC 0.73→0.840 (Δ=+0.11). Entropy near-perfect at 7B (0.965). Goldilocks ceiling falsified for instruct models.**

| Experiment | Script | Kaggle Kernel | Results File | Key Result |
|---|---|---|---|---|
| EXP_SCALE_EXTENSION_V1 | `experiments/scale_extension_v1/scale_extension_v1.py` | `chakrivijayarao/scale-extension-v1` | — | Fisher=0.840 shuffled=0.627, Entropy=0.965, N=50/class, 4-bit NF4, VRAM=15.3 GB |

---

### C031 — SUPPORTED
**Gemma-2-2B-IT L1 bilateral oracle AUROC=0.7528 CI=[0.652,0.848] (CLEAN). Third independent architecture. Architecture spread: Qwen=0.731, Llama=0.746, Gemma=0.753 (range=0.022).**

L2 NOT SUPPORTED on Gemma with entropy-matched framing (theta_conf=0.067; recovered by CO labeling — see C036).

| Experiment | Script | Kaggle Kernel | Results File | Key Result |
|---|---|---|---|---|
| EXP_GEMMA_BILATERAL_V1 | `experiments/gemma_bilateral_v1/gemma_bilateral_v1.py` | `chakrivijayarao/gemma-bilateral-v1` | — | L1: AUROC=0.7528 CI=[0.652,0.848] shuffled=0.530 CLEAN, N=200/class, L24, scanned=2342 |

---

### C032 — SUPPORTED
**CO labeling (PARAM vs any wrong answer) achieves AUROC=0.885, outperforming bilateral oracle PARAM/CTX_DEP (AUROC=0.806) by Δ=+0.079. CTX_DEP items are geometrically closer to PARAM than confabulated items.**

| Experiment | Script | Kaggle Kernel | Results File | Key Result |
|---|---|---|---|---|
| EXP_0_BO_ABLATION_V1 | `experiments/bo_ablation_v1/bo_ablation_v1.py` | — | — | CO_AUROC=0.885 CI=[0.810,0.947] vs BO_AUROC=0.806 CI=[0.712,0.887], Δ=+0.079, CO_BETTER |

---

### C033 — SUPPORTED
**Fisher+PCA64 (AUROC=0.845) exceeds all behavioral baselines on entropy-matched CC/CW: B1_verbalized=0.605, B2_self_consistency=0.613, B3_top1_prob=0.384, B_entropy=0.483. Gap vs best = +0.232.**

Note: B3_top1 is BELOW CHANCE (0.384) — confabulation produces more peaked wrong-token distributions.

| Experiment | Script | Kaggle Kernel | Results File | Key Result |
|---|---|---|---|---|
| EXP_1_BASELINE_COMPARISON_V1 | `experiments/baseline_comparison_v1/baseline_comparison_v1.py` | — | — | Fisher=0.845 CI=[0.740,0.939], gap_vs_B2=0.232, N=100/class entropy window [0.462,1.062] |

---

### C034 — SUPPORTED
**CO-L2 (no entropy matching) and BO-L2 (entropy-matched) give equivalent Fisher gaps at L2 on the same pool: CO gap=0.109, BO gap=0.124, diff=−0.016 (EQUIVALENT, |diff|<0.05). Entropy matching is not required.**

VARIANCE CONCERN: Fisher L2 AUROC shows high item-subset variance (0.670–0.854 across pool sections); C040 resolves this with large-N CV.

| Experiment | Script | Kaggle Kernel | Results File | Key Result |
|---|---|---|---|---|
| EXP_CO_L2_V1 | `experiments/co_l2_v1/co_l2_v1.py` | — | — | CO: Fisher=0.670 CI=[0.570,0.762], gap=0.109; BO: Fisher=0.670 CI=[0.516,0.826], gap=0.124 |

---

### C035 — SUPPORTED
**Mistral-7B-Instruct-v0.2 L1 AUROC=0.778 CI=[0.692,0.863] (CLEAN). Highest of 4 architectures [0.731–0.778]. L2 entropy-matched framing fails (theta_conf=0.122; recovered by CO labeling — see C036).**

| Experiment | Script | Kaggle Kernel | Results File | Key Result |
|---|---|---|---|---|
| EXP_MISTRAL_BILATERAL_V1 | `experiments/mistral_bilateral_v1/mistral_bilateral_v1.py` | — | — | L1: AUROC=0.778 CI=[0.692,0.863] shuffled=0.552 CLEAN N=200 4-bit NF4, scanned=3795 |

---

### C037 — SUPPORTED
**Fisher+PCA64 detects a geometrical commitment point at ~13% of think block. Causal truncation at commit point: mean F1 delta=+0.004 (full − truncated), token reduction=87%, commit detected in 99.4% of items.**

See also C028 (same experiment, focus on token efficiency rather than causal framing).

| Experiment | Script | Kaggle Kernel | Results File | Key Result |
|---|---|---|---|---|
| EXP_I_EARLY_EXIT_CAUSAL_V1 | `experiments/early_exit_causal_v1/early_exit_causal_v1.py` | `chakrivijayarao/early-exit-causal-v1` | — | N=200, commit_pct=87.4%, n_committed=199, mean_Δf1=+0.0059, CI=[−0.0007,+0.0125] |

---

### C038 — SUPPORTED
**J_know (Fisher score trajectory during reasoning on MATH) does not correlate with oracle process reward at the same step. Pearson=0.053, Spearman=0.047. Fisher is an entry-point predictor, not a step-level PRM.**

| Experiment | Script | Kaggle Kernel | Results File | Key Result |
|---|---|---|---|---|
| EXP_PRM_CORRELATION_V1 | `experiments/prm_correlation_v1/prm_correlation_v1.py` | — | — | N=80 problems, Pearson=0.053, Spearman=0.047, early_J_vs_final=−0.008 (p=0.945). PRM_SIGNAL_WEAK |

---

### C039 — SUPPORTED
**Fisher+PCA64 at L25–L26 step-1 predicts MATH-500 correctness before any reasoning begins. Corrected N=100: AUROC=0.856 CI=[0.729,0.957] shuffled=0.574 CLEAN. Architecture-independent with C048.**

| Experiment | Script | Kaggle Kernel | Results File | Key Result |
|---|---|---|---|---|
| EXP_MATH_AUROC_V2 (corrected) | `experiments/math_auroc_v2/math_auroc_v2.py` | — | — | N=100/class (scanned 235), model_accuracy=0.426. L25=L26=0.856 CI=[0.729,0.957] |
| EXP_PRM_CORRELATION_V1 (original, superseded) | `experiments/prm_correlation_v1/prm_correlation_v1.py` | — | — | N=30/class, L25=L26=0.911 (upward-biased small-N estimate; superseded) |

---

### C041 — SUPPORTED
**Bilateral oracle probe is format-scoped: TriviaQA→HotpotQA transfer AUROC=0.657 (efficiency=84.5%); TriviaQA→MMLU-STEM transfer AUROC=0.529 (≈random). Multi-hop QA shares geometry; multiple-choice does not.**

| Experiment | Script | Kaggle Kernel | Results File | Key Result |
|---|---|---|---|---|
| EXP_OOD_GENERALIZATION_V2 | `experiments/ood_generalization_v2/ood_generalization_v2.py` | — | — | HotpotQA transfer=0.657 within=0.777 efficiency=84.5%; MMLU transfer=0.529 within=0.949. OOD_PARTIAL |

---

### C042 — SUPPORTED
**INVERTED_U on L1 across training stages (Qwen backbone, matched N=200): BASE=0.740 → INSTRUCT=0.803 → REASONING=0.725. SFT improves L1; reasoning distillation on math CoT reduces L1 below BASE.**

Cross-family directional support on Llama: INSTRUCT(3B)=0.708 > REASONING(8B)=0.657 (size-confounded; directional only).

| Experiment | Script | Kaggle Kernel | Results File | Key Result |
|---|---|---|---|---|
| EXP_L_STAGE_SWEEP_V2 (Qwen) | `experiments/exp_l_stage_sweep_v2/exp_l_stage_sweep_v2.py` | — | — | L1: BASE=0.740, INSTRUCT=0.803, REASONING=0.725. INVERTED_U. N=200/class matched. |
| EXP_A_LAW3_LLAMA_V7 (Llama directional) | `experiments/exp_a_law3_llama/exp_a_law3_llama.py` | — | — | Llama INSTRUCT(3B) L1=0.708 N=112 > REASONING(8B) L1=0.657 N=73 (size confound: directional only) |

---

### C043 — SUPPORTED
**MONOTONE_RISE on L2 across training stages (Qwen, matched N=200): L2 Fisher gap = BASE(0.064) → INSTRUCT(0.096) → REASONING(0.148). Reasoning distillation produces strongest confabulation separation (+131% vs BASE).**

Entropy L2 baseline stable (0.577–0.611); improvement is pure Fisher signal growth.

| Experiment | Script | Kaggle Kernel | Results File | Key Result |
|---|---|---|---|---|
| EXP_L_STAGE_SWEEP_V2 | `experiments/exp_l_stage_sweep_v2/exp_l_stage_sweep_v2.py` | — | — | L2 gap: BASE=0.064, INSTRUCT=0.096, REASONING=0.148. Entropy stable 0.577–0.611. MONOTONE_RISE |

---

### C044 — SUPPORTED
**Phi-3.5-Mini-Instruct (3.8B) L1 AUROC=0.846 CI=[0.758,0.921] (CLEAN) — highest of 5 architectures [0.731–0.846]. L2 SUPPORTED: Fisher=0.712 Entropy=0.590 gap=0.122 BO_Transfer=0.792.**

Architecture spread with Phi: Qwen(0.731), Llama(0.746), Gemma(0.753), Mistral(0.778), Phi(0.846).

| Experiment | Script | Kaggle Kernel | Results File | Key Result |
|---|---|---|---|---|
| EXP_PHI_BILATERAL_V1 | `experiments/phi_bilateral_v1/phi_bilateral_v1.py` | — | — | L1: AUROC=0.846 CI=[0.758,0.921] shuffled=0.499 CLEAN, N=200, L30, scanned=2268. L2: Fisher=0.712 gap=0.122 SUPPORTED |

---

### C045 — SUPPORTED
**Qwen3-1.7B (non-R1, Qwen-native reasoning) commit_pct=99.8% (z=1.3×10¹⁵). More extreme than R1-distilled models (75.8%/82.9%). Resolves teacher confound in C022. Law 2 is teacher-independent.**

| Experiment | Script | Kaggle Kernel | Results File | Key Result |
|---|---|---|---|---|
| EXP_TEACHER_INDEPENDENCE_V1 | `experiments/teacher_independence_v1/teacher_independence_v1.py` | — | — | N=100/100 committed, commit_pct=99.8%, z=1.3e15, mean_think_len=766, cal_AUROC=0.927 |

---

### C046 — SUPPORTED
**Law 1 generalizes to NQ-Open: AUROC=0.679 CI=[0.557,0.800] (CLEAN, N=100/class). Pre-registered threshold (≥0.70) missed. Verdict: LAW1_WEAK. Kill criterion (< 0.65) not triggered.**

| Experiment | Script | Kaggle Kernel | Results File | Key Result |
|---|---|---|---|---|
| EXP_NQ_BLIND_PREDICTION_V1 | `experiments/nq_blind_prediction_v1/nq_blind_prediction_v1.py` | `chakrivijayarao/nq-blind-prediction-v5` | — | AUROC=0.6792 CI=[0.557,0.800] shuffled=0.571 CLEAN N=100 scanned=815. LAW1_WEAK |

---

### C048 — SUPPORTED
**Law 4 replicates on Llama backbone: DeepSeek-R1-Distill-Llama-8B L29 step-1 predicts MATH correctness. AUROC=0.846 CI=[0.721,0.947] shuffled=0.463 CLEAN. Δ vs Qwen (C039)=−0.010 (within CI).**

Verdict LAW4_WEAKENED: pre-registered threshold was ≥0.85; 0.846 misses by 0.004. Kill criterion (< 0.70) not triggered.

| Experiment | Script | Kaggle Kernel | Results File | Key Result |
|---|---|---|---|---|
| MATH_AUROC_LLAMA_V1 | `experiments/math_auroc_llama_v1/math_auroc_llama_v1.py` | `chakrivijayarao/math-auroc-llama-law4-replication-v1` | — | N=100/class scanned=213, L29=0.846 CI=[0.721,0.947] shuffled=0.463, L30=0.811 |

---

### C049 — SUPPORTED
**Commitment fraction ε_C-insensitive across all 3 architectures (R1-Qwen, R1-Llama, Qwen3). std(commit_pct over ε_C∈[0.05,0.25]) < 0.001 for every model. The COMMITTED_EARLY finding (C022, C045) is not threshold-artifact.**

| Experiment | Script | Kaggle Kernel | Results File | Key Result |
|---|---|---|---|---|
| EXP_C_EPS_SENSITIVITY | `experiments/exp_c_eps_sensitivity/exp_c_eps_sensitivity.py` | `chakrivijayarao/exp-c-commitment-fraction-eps-sensitivity-v1` | — | R1_QWEN std=0.0002, R1_LLAMA std=0.0002, QWEN3 std=0.0. All pct_committed=1.0. ROBUST |

---

### C050 — SUPPORTED
**Fisher geometry does not sharply terminate at the bilateral oracle boundary. SOFT_CTX_DEP items (nc_F1∈[0.05,0.20)) show AUROC=0.696 CI=[0.491,0.876] (CLEAN). Signal degrades gradually (HARD=0.853 → SOFT=0.696, Δ=−0.157).**

| Experiment | Script | Kaggle Kernel | Results File | Key Result |
|---|---|---|---|---|
| EXP_CONTINUOUS_ORACLE_V1 | `experiments/continuous_oracle_v1/continuous_oracle_v1.py` | `chakrivijayarao/continuous-oracle-v4-excluded-zone-rc-pca-fix` | — | SOFT: AUROC=0.696 CI=[0.491,0.876] shuffled=0.521 CLEAN N=43; HARD: AUROC=0.853 CI=[0.780,0.915] N=150. Threshold MET. |

---

## Frozen Results Files

| File | Exists | Covers |
|---|---|---|
| `results/frozen/large_n_validation_results_v1.json` | YES | EXP_T1A_LARGE_N_V1 (Qwen only, N=121) |
| `results/frozen/large_n_v2_qwen_results.json` | download needed | EXP_LARGE_N_V2 Qwen — primary C001 estimate |
| `results/frozen/large_n_v2_llama_results.json` | download needed | EXP_LARGE_N_V2 Llama — primary C001/C003 estimate |
| `results/frozen/c3v3_results.json` | download needed | EXP_C3V3 (superseded calibration-phase estimates) |
| `results/frozen/step_index_results_v3.json` | YES | EXP_T1D_STEP_INDEX_V3 (C006 MONOTONE_RISE) |
| `results/frozen/borderline_geometry_results_v4.json` | YES | EXP_T1C_BORDERLINE (C015 BIMODAL) |
| `results/frozen/false_certainty_v2_results.json` | download needed | EXP_A_FALSE_CERTAINTY_V2 (C017, C018) |
| `results/frozen/l2_large_n_v1_results.json` | download needed | EXP_L2_LARGE_N_V1 (C040 CONFIRMED) |
| `results/frozen/exp_b_fullspace_fisher_results.json` | YES | EXP_B_FULLSPACE_FISHER_V1/V2 (C002) |
| `results/frozen/esm_trajectory_results.json` | YES | EXP_TRAJECTORY_LDA (C007) |
| `results/frozen/esm_table2_results.json` | YES | EXP_ESM_V33 (C006 step-1 vs prefill) |

All other experiments store results in Kaggle kernel outputs. Download and place in `results/frozen/` to make every number locally traceable.

---

## FALSIFIED Claims Reference

These claims must NOT be cited as findings. They are documented for transparency.

| Claim | Falsified by | What replaced it |
|---|---|---|
| C008 (Fisher ⊥ Entropy, r≈0) | EXP_STRUCTURAL_INDEP_V2, EXP_ENTROPY_BASELINE | C016 (entropy AUROC=0.87–0.90); C017 (Fisher essential at L2 where entropy is controlled) |
| C012 (RLHF attenuation Δ=−0.036) | EXP_C3V3 | Instruct AUROC=0.841 > Base=0.714 (opposite sign) |
| C013 (Llama AUROC≈0.629) | EXP_C3V3 | Corrected Llama AUROC=0.846 (Fisher LDA was degenerate at small N) |
| C014 (nonlinear recovery > 0.05) | EXP_C3V3 | C002 (NO_RECOVERY; delta=−0.019/−0.005) |
