# Experiments

Every script runs on Kaggle (T4 GPU, 15.3 GB VRAM) and is self-contained.
For the claim→script→result mapping see [../REPRODUCE.md](../REPRODUCE.md).

Status key: ✅ CONFIRMED · 🔵 SUPPORTED · 🔶 EXPLORATORY · ❌ FALSIFIED · ⬜ INCONCLUSIVE

---

## Core L2 — Confabulation Detection (CC/CW)

| Directory | Status | What it does |
|-----------|--------|--------------|
| `false_certainty_v2` | ✅ C004 | Main result: Fisher+PCA64 AUROC=0.854 on Qwen2.5-1.5B entropy-matched CC/CW |
| `false_certainty_llama_v1` | ✅ C025 | Llama-3.2-3B replication: AUROC=0.818 |
| `baseline_comparison_v1` | ✅ C017 | Fisher vs self-consistency (N=5), verbalized uncertainty, logit baselines |
| `l2_large_n_v1` | ✅ C036 | 5-fold CV N=500/class: AUROC=0.763±0.012 (stable large-N estimate) |
| `co_l2_v1` | ✅ C036 | CO labeling variant on Qwen L2 |
| `co_gemma_mistral_v1` | ✅ C036 | CO labeling on Gemma + Mistral for cross-arch validation |
| `llama_l2_co_v1` | ⬜ C047 | Llama plain CO: AUROC=0.613 ≈ shuffled=0.580 (INCONCLUSIVE) |
| `false_certainty_v1` | 🔵 | Earlier false certainty run; superseded by v2 |
| `large_n_validation` | 🔵 | Intermediate large-N validation before l2_large_n_v1 |
| `entropy_baseline` | ✅ C004 | Single-pass entropy on same entropy-matched pool: AUROC=0.483 |

## Core L2 — Causal Intervention (EXP-H)

| Directory | Status | What it does |
|-----------|--------|--------------|
| `cc_cw_patching_v1` | ✅ C024 | Centroid-direction patching λ=0.1–2.0 all layers: max ΔF1=+0.0004 (EPIPHENOMENAL) |
| `head_patching_v1` | 🔵 | Attention-head level patching (exploratory extension of EXP-H) |
| `targeted_head_patching` | 🔵 | Targeted patching on top attribution heads |
| `head_attribution` | 🔵 | Attribution of Fisher direction to individual attention heads |
| `activation_patching_p1` | ❌ C013 | Earlier patching: EPIPHENOMENAL confirmed |

## Core L1 — Knowledge Source Routing (PARAM/CTX_DEP)

| Directory | Status | What it does |
|-----------|--------|--------------|
| `gemma_bilateral_v1` | ✅ C040 | Gemma bilateral oracle: L26 AUROC=0.846 |
| `phi_bilateral_v1` | ✅ C040 | Phi-3.5-mini bilateral oracle: L1 AUROC=0.856 (5th architecture) |
| `mistral_bilateral_v1` | ✅ C040 | Mistral-7B-Instruct bilateral oracle |
| `gemma_geometry_v1` | 🔵 | Gemma geometry sweep across layers |
| `gemma_l2_v1` | 🔵 | Gemma L2 confabulation detection |
| `exp_a_law3_llama` | ✅ C025 | Llama L26 step-1: AUROC=0.818, ICC=0.9334 |
| `exp_b_fullspace_fisher` | 🔵 | Full-space Fisher (no PCA) ablation |
| `exp_c_eps_sensitivity` | 🔵 C049 | ε-sensitivity robustness check: ROBUST |
| `structural_independence_v2` | ❌ C008 | j_score ≈ entropy — original structural independence claim falsified |

## Commitment Timing (L3)

| Directory | Status | What it does |
|-----------|--------|--------------|
| `reasoning_geometry_v1` | 🔵 C022 | Qwen: commit%=82.9%, Cal AUROC=1.000 at L28 (COMMITTED_EARLY) |
| `reasoning_geometry_llama_v1` | 🔵 C022 | Llama: commit%=75.8%, tokens_saved=75.9% |
| `step_index_auroc` | 🔵 | AUROC vs generation step: monotone rise 0.639→0.906 |
| `commit_quality_v1` | 🔵 | Quality of commitment point detection |
| `continuous_oracle_v1` | 🔵 C050 | Continuous oracle in excluded zone: AUROC=0.696, gradient 0.85→0.70 |
| `answer_jump_v2` | 🔵 | Answer-jump signal at commitment moment |
| `early_exit_causal_v1` | 🔵 | Early exit inference from commitment timing |
| `entropy_trajectory_v1` | 🔵 C019 | Entropy trajectory: inversion at step 4; CC bursts, CW flat |

## Scale and Architecture Generalization

| Directory | Status | What it does |
|-----------|--------|--------------|
| `scale_obs_v1` | 🔵 C020/C021 | Scale sweep: Goldilocks zone 0.5B–3B |
| `scale_extension_v1` | 🔵 | Extended scale experiment |
| `pythia_sweep` | 🔵 | Pythia checkpoints: floor ≥0.67 all 8 checkpoints |
| `pythia_sweep_large_n` | 🔵 | Pythia large-N replication |
| `obs_2d_surface` | 🔵 | 2D observability surface: L16 dominates, step-2 > step-1 |
| `math_auroc_v2` | 🔵 | Math domain AUROC on Qwen |
| `math_auroc_llama_v1` | 🔵 C048 | Math domain on Llama: AUROC=0.8462 (LAW4_WEAKENED) |
| `mamba_architecture` | 🔶 | Mamba (SSM) architecture transfer |
| `moe_architecture` | 🔶 | Mixture-of-Experts architecture |
| `rwkv_architecture` | 🔶 | RWKV architecture |
| `reasoning_model_extension` | 🔶 | R1-style reasoning model extension |

## NQ-Open Replication

| Directory | Status | What it does |
|-----------|--------|--------------|
| `nq_open_replication_v1` | 🔵 | NQ-Open: AUROC=0.854 on different QA dataset |
| `nq_blind_prediction_v1` | 🔵 C046 | NQ blind prediction: AUROC=0.6792 (LAW1_WEAK) |

## OOD Generalization

| Directory | Status | What it does |
|-----------|--------|--------------|
| `ood_generalization_v1` | 🔵 | OOD generalization across question types |
| `ood_generalization_v2` | 🔵 | OOD v2 with better held-out split |
| `cross_dataset_replication_v1` | 🔵 | Cross-dataset replication |
| `gqa_cosim_v1` | 🔵 | GQA family cosim: GQA_FAMILY confirmed |
| `cross_task_cosim_v1` | ❌ C014 | Cross-task geometry: TASK_SPECIFIC (cosims 0.004–0.035) |
| `bo_ablation_v1` | 🔵 | Bilateral oracle ablation (one-pass vs two-pass) |

## Probe Variants

| Directory | Status | What it does |
|-----------|--------|--------------|
| `nonlinear_probe_v1` | 🔵 | MLP probe vs Fisher LDA |
| `nonlinear_probe_v2` | 🔵 | Nonlinear probe v2 |
| `nonlinear_probe_v3` | 🔵 | Nonlinear probe v3 final |
| `teacher_independence_v1` | 🔵 C045 | Probe is teacher-independent |
| `student_classifier_v1` | 🔶 | Student classifier transfer |
| `seakr_comparison_v1` | 🔵 | SeaKR competitor comparison |
| `seakr_comparison_v2` | 🔵 | SeaKR v2 matched conditions |
| `prm_correlation_v1` | 🔵 | PRM correlation |
| `perturbation_battery_v1` | 🔵 | Probe robustness under perturbation |
| `counterfactual_divergence_v1` | 🔵 | Counterfactual divergence |

## RLHF Geometry

| Directory | Status | What it does |
|-----------|--------|--------------|
| `rl_regime_collapse_v4` | 🔵 | RL regime collapse: Qwen z=7.29 |
| `rl_regime_collapse_v5` | 🔵 | RL regime v5 |
| `rl_regime_collapse_v6` | 🔵 | RL regime v6 |
| `rlhf_attenuation_mistral_v1` | 🔵 | RLHF attenuation — Mistral |
| `rlhf_attenuation_olmo_v1` | 🔵 | RLHF attenuation — OLMo |
| `rlhf_attenuation_universal` | 🔵 | RLHF attenuation — universal |
| `rlhf_attenuation_yi_v1` | 🔵 | RLHF attenuation — Yi |
| `regime2_control_v1` | 🔵 | Regime 2 control |
| `exp_l_stage_sweep_v1` | 🔵 | Training stage sweep v1 |
| `exp_l_stage_sweep_v2` | 🔵 | Training stage sweep v2 |
| `reasoning_entropy_traj_v1` | 🔵 | Reasoning entropy trajectory |

## Difficulty Confound (Q14 — unresolved at L1)

| Directory | Status | What it does |
|-----------|--------|--------------|
| `exp_q14_difficulty_control` | 🔶 | Q14 difficulty confound control |
| `exp_q14_tier_targeted_v1` | 🔶 | Tier-targeted Q14 |
| `borderline_geometry` | 🔵 | Borderline geometry: BIMODAL, strong poles ±1.316 |

## Exploratory / Null Results

| Directory | Status | What it does |
|-----------|--------|--------------|
| `sae_integration` | 🔶 | SAE feature-level patching (untested, scope §6.6) |
| `cost_benchmark_v1` | 🔶 | Probe inference cost |
| `jvelocity_training_v1` | ❌ C012 | J-velocity: FALSIFIED — j_score ≈ entropy (r=0.0039) |
