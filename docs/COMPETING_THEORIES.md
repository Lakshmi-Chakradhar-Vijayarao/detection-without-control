# Competing Theories of Computational Observability
**Status:** LIVING DOCUMENT — updated when experiments eliminate theories  
**Version:** 1.0  
**Date:** 2026-07-12  
**Rule:** Theories are never deleted. They are falsified (with evidence and claim ID), merged, or scoped.

---

## Purpose

This document formalizes the competing theoretical explanations for Computational Observability.
Before running any experiment designed to test a candidate Law, a written prediction from each
theory must be added here. This satisfies the predictive science gate from PROGRAM_CHARTER.md:

> *"Before running an experiment to test a law, the theory must first generate a written prediction
> for an unseen system. Confirmed findings not preceded by written predictions are downgraded from
> laws to correlations."*

---

## The Central Empirical Puzzle

Why is O non-trivially non-zero across instruction-tuned transformer decoders?

Four facts that any complete theory must explain:

1. **O ≥ 0.70 across five transformer families at L1** (Law 1, C001-C004, C025, C031, C035, C044)
2. **Fisher essential beyond entropy at L2** (C017: gap = +0.240; C040: CV AUROC = 0.763 ± 0.012)
3. **INVERTED_U at L1, MONOTONE_RISE at L2 during training** (C042/C043 — Qwen backbone only)
4. **Commit_pct ≥ 75% in reasoning models** (C022, C045 — three models, two lineages)

No theory has been eliminated. The program enters Era 2 with all three main theories consistent
with the existing data.

---

## Theory H-A — Compression (Information Bottleneck)

### Verbal Statement

Optimization compresses computation into low-dimensional structure because reusable internal
representations are more parameter-efficient. PARAM knowledge is organized into a stable retrieval
subspace during training; CTX_DEP queries that fail parametric retrieval produce a different
activation pattern. The Fisher discriminant reads this compression boundary.

**Key claim:** O measures the quality of epistemically-specialized compression in the residual
stream at late layers.

### Differential Equation

Let O(t) be observability at training step t, L(t) the training loss, O_max the architectural
maximum, β the compression rate, and R(t) the reasoning-amplification signal from RL training.

For L1 (bilateral oracle, TriviaQA):
```
dO_L1/dt = α · (O_max − O_L1) · (−dL/dt)   [organization phase: falls with loss]
          − γ · O_L1 · (1 − L(t)/L_final)   [compression phase: degrades near convergence]
```
This produces an **INVERTED_U**: O_L1 first rises as the model organizes parametric knowledge into a separable subspace, then falls as over-compression removes epistemic nuance at late training.

For L2 (confabulation in confident zone):
```
dO_L2/dt = β · O_L1(t) · R(t)
```
L2 rises **monotonically** because RL training explicitly reinforces self-consistency, which amplifies the compression boundary even as L1 softens.

### Predictions for Upcoming Experiments

**Experiment A (Law 3 cross-family, Llama backbone):**
- L1: INVERTED_U — INSTRUCT AUROC > BASE AUROC by > 0.05; REASONING AUROC < INSTRUCT AUROC by > 0.03
- L2: MONOTONE_RISE — BASE < INSTRUCT < REASONING, each step > 0.03
- Numerical bound: Llama INSTRUCT L1 AUROC ≥ 0.73 (matches Law 1 floor)

**Experiment B (full-space Fisher ablation):**
- H-A makes no strong prediction on PCA bottleneck — compression does not require specific linear projection
- But: if the signal is IB-compressed, it should be concentrated in top principal components → H-linear more likely

**Experiment E (differential diagnosis):**
- Fisher should track *retrieval quality* rather than RLHF assertiveness pressure
- In the orthogonalized condition: high RLHF pressure + low retrieval quality → CTX_DEP prediction

### What Would Falsify H-A

- FLAT pattern at L1 on Llama (|max - min| < 0.05) — architecture doesn't produce INVERTED_U
- MONOTONE_RISE at L1 on both Llama and a third family — compression predicts peak then fall, not monotone
- O existing in SSMs (Mamba, RWKV) at matched extraction points — if architecture is not residual stream, compression alone can't explain the measurement protocol's success

---

## Theory H-B — Supervised Routing

### Verbal Statement

Observability emerges because supervised training objectives explicitly reward the ability to
distinguish parametric from contextual knowledge sources. SFT data mixes factual and
context-reading tasks; the model develops a routing pathway that separates these sources.
Fisher reads the activation strength of the "parametric retrieval attempt" pathway.

**Key claim:** O measures the strength of a trained routing signal — partially orthogonal to,
and partially in tension with, the output assertiveness trained by RLHF.

### Differential Equation

Let T(t) be training task diversity at time t, R_SFT(t) the density of factual SFT examples,
and R_RL(t) the density of RL reward signal at time t.

For L1:
```
dO_L1/dt = α · T(t) · R_SFT(t) · (1 − O_L1)
```
This produces **MONOTONE_RISE** at L1: more supervised routing examples → stronger routing signal.

For L2:
```
dO_L2/dt = β · O_L1 + γ · R_RL(t) · (1 − O_L2)
```
L2 also rises monotonically, with a jump at the RL stage when routing is reinforced.

### Predictions for Upcoming Experiments

**Experiment A (Law 3 cross-family, Llama backbone):**
- L1: MONOTONE_RISE — BASE < INSTRUCT < REASONING, each step > 0.03
- L2: MONOTONE_RISE with larger step at REASONING stage (RL reinforcement jump)
- If Qwen shows INVERTED_U but Llama shows MONOTONE_RISE at L1: H-B explains Llama but H-A explains Qwen — pattern is training-recipe-specific, not universal

**Experiment E (differential diagnosis):**
- Fisher should track *RLHF assertiveness pressure*: in the orthogonalized condition, high RLHF pressure items → PARAM prediction (model is "asserting" parametric knowledge)
- This is the distinguishing prediction from H-A

### What Would Falsify H-B

- L1 INVERTED_U on both Qwen (confirmed) and Llama — monotone supervision predicts MONOTONE_RISE, not INVERTED_U
- O existing in unsupervised or self-supervised models without routing-relevant SFT data
- Fisher tracking retrieval quality rather than RLHF assertiveness in Experiment E

---

## Theory H-C — Architecture

### Verbal Statement

O is a side effect of residual stream geometry in transformer architectures. Skip connections
allow early-layer epistemic signals to propagate unmodified to late layers without modification
by subsequent MLP layers. The bilateral oracle Fisher probe at L26 reads this accumulated signal.
The signal exists because of the architecture, not because of training.

**Key claim:** O is an architectural prior, not a trained property. Training merely clarifies the
signal; it does not create it.

### Differential Equation

```
dO/dt ≈ 0
O(t) ≈ O_arch   [set by architectural parameters: depth, embedding dim, skip density]
```

O is constant across training stages — or varies only because training sharpens the bilateral
oracle's label quality, not because the underlying geometry changes.

### Predictions for Upcoming Experiments

**Experiment A (Law 3 cross-family, Llama backbone):**
- L1: FLAT — |AUROC_max − AUROC_min| < 0.05 across BASE/INSTRUCT/REASONING stages
- L2: FLAT or marginally rising (RLHF sharpens oracle labels, not geometry)
- If FLAT: H-C is strongly supported for Llama. If Qwen showed INVERTED_U and Llama shows FLAT, H-C explains Llama specifically (perhaps Llama's residual stream geometry is more stable).

**Non-transformer architecture test (future Q3/Q8):**
- H-C's clearest prediction: O would NOT be measurable in Mamba or RWKV using the same bilateral oracle + layer-specific probe protocol
- Skip connections are required for the accumulated signal to persist to L26

### What Would Falsify H-C

- Any systematic training-stage variation > 0.05 AUROC range on any architecture — INVERTED_U or MONOTONE_RISE both falsify FLAT
- O in SSMs at matched architectural extraction points — if non-residual models show O, architecture alone is insufficient
- O increasing monotonically with training-data scale in controlled experiments

---

## Discrimination Matrix

| Experiment | H-A (Compression) | H-B (Supervision) | H-C (Architecture) |
|---|---|---|---|
| **Exp A: Llama L1 INVERTED_U** | Supports ✓ | Falsifies ✗ | Falsifies ✗ |
| **Exp A: Llama L1 MONOTONE_RISE** | Weakens (L1) | Supports ✓ | Falsifies ✗ |
| **Exp A: Llama L1 FLAT** | Falsifies ✗ | Falsifies ✗ | Supports ✓ |
| **Exp A: Llama L2 MONOTONE_RISE** | Supports ✓ | Supports ✓ | Weakens |
| **Exp B: Full-space > PCA64 by > 0.05** | Ambiguous | Ambiguous | Ambiguous |
| **Exp B: Full-space ≈ PCA64** | Supports (IB concentration) | Ambiguous | Ambiguous |
| **Exp E: Fisher tracks retrieval quality** | Supports ✓ | Weakens | Ambiguous |
| **Exp E: Fisher tracks RLHF pressure** | Weakens | Supports ✓ | Ambiguous |
| **Non-transformer O exists (Q3/Q8)** | Supports ✓ | Supports ✓ | Falsifies ✗ |
| **Non-transformer O absent (Q3/Q8)** | Weakens | Weakens | Supports ✓ |

**Critical experiment:** Experiment A is the most informative near-term discriminator.
- INVERTED_U on Llama → eliminates H-B and H-C
- FLAT on Llama → eliminates H-A and H-B
- MONOTONE_RISE on Llama → eliminates H-C, partially weakens H-A

No single experiment eliminates all three theories. The program needs Experiment A (Q4) + 
Experiment E (Q6) + non-transformer test (Q3) to fully discriminate.

---

## Current State of Evidence (2026-07-12)

| Theory | Status | Primary Supporting Evidence | Primary Weakness |
|---|---|---|---|
| H-A (Compression) | **Best supported** | C042 INVERTED_U at L1 (Qwen) | Single backbone; L2 also consistent with H-B |
| H-B (Supervision) | Not falsified | C043 MONOTONE_RISE at L2 consistent | L1 INVERTED_U (Qwen) requires special pleading |
| H-C (Architecture) | Weakest | No direct evidence against | C042/C043 training-stage variation directly inconsistent with FLAT |

H-C is the weakest: any systematic training-stage pattern falsifies FLAT. C042/C043 (single backbone, SUPPORTED) already weakens H-C but does not falsify it because Qwen might be atypical.

**The decisive experiment is Experiment A.** If Llama also shows INVERTED_U at L1: H-A is strongly supported across two independent families, H-B and H-C are eliminated as explanations for the L1 pattern.

---

## The RLHF-Decoupling Story — Prediction, Not Mechanism

The RLHF-decoupling explanation for the Fisher-entropy gap is frequently discussed as if it were a confirmed mechanism. It is not. It is a testable prediction.

**Locked framing (2026-07-12):**

> *One possible explanation is that alignment training reshapes output confidence more strongly than intermediate representations. This predicts that increasingly aligned models should exhibit larger hidden-state/output divergence at L2.*

This is **Theory H-F (Alignment Hiding)** — a candidate theory that would explain C017/C033 mechanistically. It is not yet formalized as a differential equation because the distinguishing experiment has not run.

**H-F prediction:** In a controlled sweep Base → SFT → DPO → RLHF → reasoning-distilled on the same backbone, the Fisher-entropy gap at L2 should increase monotonically. Each alignment stage adds more output-confidence calibration pressure while leaving the residual stream geometry intact.

**What would falsify H-F:**
- Fisher-entropy gap does not increase monotonically across alignment stages
- Reasoning distillation reduces the gap (rather than amplifying it)
- Base models show the same gap as RLHF models at matched capability level

**Write rule:** The paper must NOT say "RLHF covers up uncertainty." It must say: "One possible explanation is that alignment training reshapes output confidence more strongly than intermediate representations. This predicts increasingly aligned models should exhibit larger hidden-state/output divergence." That is scientifically stronger because it is explicitly predictive and falsifiable.

H-F will be formally added to the discrimination matrix once the experiment is pre-registered.

---

## Theories Not Yet Formalized

The following alternative explanations are acknowledged but not yet given differential equations.
They should be formalized before their distinguishing experiments are run.

- **H-D (Predictive Coding):** Residual stream at L26 encodes prediction error between expected and actual information retrieval. O measures the magnitude of this error, not the routing decision per se.
- **H-E (Superposition):** O is not a clean linear signal but a superposition of many atomic features (as per mechanistic interpretability). The Fisher discriminant identifies a mixed direction that happens to separate PARAM/CTX_DEP because those features are differentially active.

H-D and H-E require SAE analysis (Q6, Experiment F) before they can be formalized here.

---

*See [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md) Q1 for the hypothesis structure.*  
*See [PROGRAM_CHARTER.md](PROGRAM_CHARTER.md) for the predictive science gate protocol.*  
*Pre-registration documents for each experiment: `science/preregistration/`*
