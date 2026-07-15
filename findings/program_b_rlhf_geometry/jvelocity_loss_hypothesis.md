# Trainable Epistemic Transparency

**Research design document — long-horizon frontier experiment**

---

## 1. Motivation

The current approach to epistemic monitoring treats the model as a fixed black box and probes its residual stream at inference time. This works well — Fisher LDA on the final layer achieves AUROC 0.989 on Qwen2.5-7B-Instruct for separating PARAM from CTX_DEP knowledge. But there is a systematic vulnerability: **instruction tuning erodes the signal**.

Across the Llama and Qwen model families, the RLHF attenuation result shows a consistent drop of $\Delta = -0.036$ AUROC when comparing base models to their instruction-tuned counterparts. This is not noise — it is a structural effect of RLHF training, which optimizes for output behavior (human preference) without any objective over the geometry of the residual stream. The Fisher direction — a subspace of $\mathbb{R}^d$ encoding knowledge-source routing — is collateral damage.

This raises a more fundamental question than "how do we monitor better": **can we train models whose internal epistemic state is intentionally transparent and stable across fine-tuning?**

Trainable transparency is the bet that the answer is yes. If it holds, epistemic transparency becomes a training paradigm rather than a monitoring layer. The project shifts from "an inference-time add-on for opaque models" to "part of the training stack."

---

## 2. Hypothesis

A model trained with an epistemic auxiliary loss during SFT will produce:

1. **Stronger Fisher AUROC** — greater separation of PARAM vs. CTX_DEP in the residual stream geometry, measured by Fisher LDA at the optimal layer.
2. **Lower RLHF attenuation** — when the SFT+EML checkpoint is subsequently instruction-tuned, the drop in AUROC satisfies $\Delta < 0.010$ (vs. the $\Delta = 0.036$ baseline across unmodified instruction tuning).
3. **More faithful J-velocity** — the correlation between trajectory variance $\text{Var}(J_{L_1 \ldots L_K})$ and generation correctness is preserved or strengthened post-SFT.

The auxiliary loss targets the hidden-state geometry directly at generation step 1, where the model's epistemic routing decision is empirically established to occur (the "commitment moment" at L18 for Llama, L26 universal for Fisher LDA).

---

## 3. Proposed Auxiliary Loss: Epistemic Margin Loss (EML)

### 3.1 Definition

Let $h_1 \in \mathbb{R}^d$ be the residual stream at generation step 1 (the first generated token), at the calibration layer $L^*$ (take $L^* = 26$ for universality, or $L^* = 18$ for Llama-specific zero-shot strength). Let $\text{diff}_u \in \mathbb{R}^d$ be the Fisher LDA direction computed on a held-out calibration split — the unit vector pointing from the CTX_DEP centroid to the PARAM centroid in hidden-state space. Let $j = h_1 \cdot \text{diff}_u$ be the scalar Fisher projection.

The Epistemic Margin Loss is:

$$\mathcal{L}_{\text{EML}} = \frac{1}{N} \sum_{i=1}^{N} \max\!\left(0,\; m - y_i \cdot j_i\right)$$

where:
- $y_i \in \{+1, -1\}$: epistemic label ($+1$ for PARAM, $-1$ for CTX_DEP)
- $j_i = h_{1,i} \cdot \text{diff}_u$: Fisher projection for example $i$
- $m > 0$: margin hyperparameter (ablate over $\{0.5, 1.0, 2.0\}$)

This is a standard hinge loss in the Fisher direction. For PARAM examples it pushes $j$ above $+m$; for CTX_DEP examples it pushes $j$ below $-m$.

### 3.2 Total training objective

$$\mathcal{L} = \mathcal{L}_{\text{CE}} + \alpha \cdot \mathcal{L}_{\text{EML}}$$

where $\mathcal{L}_{\text{CE}}$ is the standard cross-entropy SFT loss over the answer tokens, and $\alpha$ controls the relative weight of the epistemic objective.

**Hyperparameter grid:**

| Parameter | Default | Ablation range |
|---|---|---|
| $\alpha$ (loss weight) | 0.05 | {0.01, 0.05, 0.1, 0.5} |
| $m$ (margin) | 1.0 | {0.5, 1.0, 2.0} |
| $L^*$ (target layer) | 26 | {18, 26} |

### 3.3 Critical constraint: frozen Fisher direction

$\text{diff}_u$ is computed on a calibration split **before** SFT begins and is frozen for the duration of training. It must not be updated or re-derived mid-training. Rationale: if $\text{diff}_u$ were updated online, the loss would become a moving target and could be minimized trivially by rotating the Fisher direction rather than by separating the hidden states. The calibration split is disjoint from the training split.

---

## 4. Comparison to J-Velocity Training

The earlier `jvelocity_training_v1.py` experiment defined a loss over the **trajectory** of Fisher projections across layers:

$$\mathcal{L}_{J\text{-vel}} = -\text{Var}(J_{L_1}, J_{L_2}, \ldots, J_{L_K})$$

That loss encouraged the model to produce hidden-state trajectories with high layer-to-layer variance, on the theory that high J-velocity correlates with epistemic commitment.

EML differs in two important ways:

1. **Target**: EML targets the absolute position of $h_1$ in hidden-state space (push PARAM above $+m$, CTX_DEP below $-m$). J-velocity targets the dynamics across layers. These are complementary signals from the signal decomposition results: family A ($j_\text{score} + j_\text{velocity}$, AUROC 0.74) and family B (entropy + margin, AUROC 0.73) are nearly independent. EML strengthens family A geometry; J-velocity strengthens family A dynamics.

2. **Signal source**: EML operates on a single layer at step 1. J-velocity requires a multi-layer forward pass to compute the trajectory. EML is cheaper to implement and easier to ablate in a Kaggle T4 environment.

The two losses can in principle be combined, but the experiment below isolates EML first to establish its independent effect.

---

## 5. Experiment Design

### 5.1 Infrastructure

- **Base model:** Qwen2.5-1.5B (fits on a T4 GPU with 4-bit quantization; tractable for SFT)
- **Training data:** 1,000 Q&A pairs from TriviaQA, with bilateral oracle labels (PARAM / CTX_DEP). Split: 800 train / 100 calibration (for $\text{diff}_u$) / 100 held-out test.
- **SFT format:** prompt = question, target = short answer. Labels from the bilateral oracle (same labeling logic used throughout the experiment series).
- **Compute budget:** single T4 (16 GB), estimated 2–4 hours per condition.

### 5.2 Conditions

| Condition | Training | Post-processing |
|---|---|---|
| A — SFT baseline | $\mathcal{L}_{\text{CE}}$ only | — |
| B — SFT + EML | $\mathcal{L}_{\text{CE}} + 0.05 \cdot \mathcal{L}_{\text{EML}}$ | — |
| C — SFT baseline + RLHF-style | $\mathcal{L}_{\text{CE}}$, then preference tuning | DPO on 200 preference pairs |
| D — SFT + EML + RLHF-style | $\mathcal{L}_{\text{CE}} + 0.05 \cdot \mathcal{L}_{\text{EML}}$, then DPO | DPO on same 200 preference pairs |

The primary comparison of interest is C vs. D: does EML during SFT reduce the AUROC attenuation caused by subsequent instruction tuning?

### 5.3 Measurements

At each checkpoint (after SFT, after DPO), measure:
1. **Fisher AUROC** — LDA probe at layer $L^*$ on the held-out 100 examples.
2. **Answer F1** — exact match / token F1 on held-out TriviaQA to verify the EML loss is not degrading generation quality.
3. **J-velocity correlation** — $\text{corr}(\text{Var}(J_{L_{14} \ldots L_{26}}),\; \text{correct})$ on held-out set.
4. **RLHF attenuation $\Delta$** — AUROC(condition B) $-$ AUROC(condition D) after DPO.

### 5.4 Success criterion

The experiment is considered a positive result if, after DPO:

$$\Delta_{\text{AUROC}}^{\text{EML}} < 0.010 \quad \text{vs.} \quad \Delta_{\text{AUROC}}^{\text{baseline}} = 0.036$$

A secondary success criterion: Answer F1 does not drop by more than 2 percentage points relative to the SFT baseline, confirming the auxiliary loss is not purchasing epistemic transparency at the cost of task performance.

---

## 6. Why This Matters

If the experiment is a positive result, the implications are layered:

**For the Epistemic Runtime project:** the monitoring infrastructure becomes more reliable as models are scaled — the signal it depends on is preserved by design rather than eroded by fine-tuning. The probe direction $\text{diff}_u$ stays valid across model versions.

**For AI oversight broadly:** a model trained with EML is, in a precise geometric sense, more transparent about its knowledge routing. An auditor with access to the residual stream can always determine whether the model is answering from parametric memory or context. This is a form of architectural honesty that output-only monitoring cannot provide.

**For the training stack:** EML is lightweight — a single hinge loss term at one generation step, one layer. It does not require new architectures, new data collection methods, or new inference infrastructure. It plugs into any SFT pipeline. If it generalizes across model families (Llama, Qwen, Phi), it becomes a candidate for inclusion in standard instruction-tuning recipes.

**For competitive positioning:** the compression theorem establishes that output-space monitoring has a structural ceiling. Trainable transparency is the only known path to dissolving that ceiling — not by better output monitoring, but by making the model's internal geometry reliably expressive. This is a defensible moat.

---

## 7. Risk Assessment

| Risk | Severity | Mitigation |
|---|---|---|
| Fisher direction drifts post-EML: $\text{diff}_u$ no longer separates PARAM/CTX_DEP | High | Re-derive $\text{diff}_u$ after SFT+EML and re-evaluate; if drift is large, the loss is self-defeating |
| EML conflicts with CE loss: F1 drops significantly | Medium | Ablate $\alpha$; start at 0.01; monitor F1 at each checkpoint |
| 1.5B is too small to show the effect | Medium | If negative, rerun on Qwen2.5-3B or Llama-3.2-3B before concluding |
| DPO synthetic pairs are not representative | Low | Use the same preference-pair construction as in the RLHF attenuation study |
| T4 memory pressure during SFT | Low | Use LoRA rank 8 + 4-bit base; EML adds only a dot product at one layer |

---

## 8. Implementation Note

This experiment builds on `jvelocity_training_v1.py` but the signal is different. That file constructs trajectory-level losses from multi-layer hidden states. EML requires:

1. A hook at layer $L^*$ capturing $h_1$ (step-1 hidden state only — not the full trajectory).
2. The frozen $\text{diff}_u$ vector loaded from the calibration run.
3. A single dot product $j = h_1 \cdot \text{diff}_u$ followed by the hinge loss.

The calibration step (computing $\text{diff}_u$ from the 100-example calibration split before training begins) should be factored into a separate `calibrate_fisher.py` utility that saves $\text{diff}_u$ as a `.pt` file. The SFT loop then loads this file at startup and registers it as a non-trainable buffer.

The full implementation maps cleanly to a Kaggle kernel: calibration cell, SFT loop cell, DPO cell, evaluation cell. Estimated total runtime on T4: 3–5 hours.

---

*This document describes the next frontier experiment after the EIL series. Prerequisite: ECS v5 results confirmed (LR AUROC 0.9196, J-score 0.8605). Status: design complete, implementation pending.*
