# Epistemic Observability in Transformer Language Models
## Measuring Knowledge-Source Geometry in Residual Stream Representations

**Lakshmi Chakradhar Vijayarao**
Independent Research Program in Reliable and Verifiable AI Systems
May 2025 – Present

---

## Abstract

This thesis presents a structured empirical investigation into the **epistemic legibility** of transformer language models, organized around three measurable computational quantities — **Observability** (O), **Commitment** (C), and **Accessibility** (A). O measures whether external probes can reliably read epistemic state from internal representations; C measures when internal computation settles on an answer; A measures how training shapes the computational organization that O and C reflect. These are intended as *quantities*, not organizing concepts: the estimators (Fisher+PCA64, bilateral oracle, commit% probe) are current implementations that may be replaced by better instruments. O, C, and A persist as quantities across estimator changes.

The central methodological contribution is the **bilateral oracle**, a two-pass labeling protocol that operationalizes knowledge-source type without conflating epistemic state with output quality — a distinction prior probing work does not make. Applying a Fisher+PCA64 probe to hidden states extracted at generation step-1, layer 26, we find that bilateral oracle labels are linearly separable with AUROC 0.7312 (Qwen2.5-1.5B-Instruct, N=197/class) and 0.7464 (Llama-3.2-3B-Instruct, N=200/class) under clean large-N sampling (pool=10,000; shuffled controls 0.598 and 0.502 — both CLEAN). Earlier calibration-phase estimates (0.841–0.846) had a dataset-ordering artifact and are superseded. Nonlinear probes show no recovery above 0.05. For generation-onset timing, step-1 hidden states carry substantially more epistemic signal than prefill across four model families (0.785 vs 0.567 mean). Centroid-level patching produces no causal leverage at any tested layer (L4–L26). Output entropy at step-1 achieves AUROC 0.9043 (Qwen) and 0.874 (Llama) on the bilateral oracle task — equal to or exceeding Fisher — establishing that knowledge-source routing (L1) is primarily an entropy-capturable confident/uncertain distinction. Four confirmed falsifications are maintained in the formal claims registry.

The program has extended to cover confabulation detection, reasoning chain dynamics, scale, baseline comparisons, and training-stage effects. **False certainty detection (EXP-A + EXP-Llama):** within the entropy-matched confident zone, Fisher+PCA64 distinguishes CONFIDENT_CORRECT from CONFIDENT_WRONG with AUROC=0.854/0.818 (Qwen/Llama), adding 0.240/0.365 over entropy (C017, C018 cross-arch). Crucially, behavioral baselines including self-consistency (0.613) and top-1 probability (0.384, *below chance*) both fail at L2 — the sub-chance top-1 probability is consistent with RLHF-trained assertiveness producing output confidence that systematically diverges from underlying epistemic state (Qwen, single architecture, C033). **Perturbation invariance (EXP-J):** ICC=0.913/0.933 (Qwen/Llama, between/within=10.5:1 and 14.0:1) across four surface variants (C025 CONFIRMED, two architectures). **Entropy trajectory (EXP-D):** 15-step output trajectory achieves AUROC=0.730 for CC vs CW with characteristic step 2–3 inversion (C019). **Reasoning geometry (EXP-B/C):** reasoning-distilled models commit to their answer direction after 18–24% of thinking tokens (Qwen 1.5B z=49.77; Llama 8B z=679.73) — 75–83% of the think block is post-commitment (C022). **Post-think entropy burst (EXP-G, GSM8K):** the think block does not pre-resolve epistemic uncertainty — the same BURST pattern appears after </think> with peak AUROC=0.6932 at step 4, trajectory AUROC=0.8424, driven by CW entropy collapse (1.322→0.127 by step 4) (C027). **Early exit causal validation (EXP-I):** truncating generation at the commit point saves 87.4% of thinking tokens at +0.006 F1 cost (p=0.08) (C028). **Scale extension (EXP-Scale, Qwen2.5-7B-Instruct):** Fisher AUROC improves to 0.8402 (+0.11 vs 1.5B); entropy reaches 0.9645 (C029). **Mistral architecture (C031):** Mistral-7B-Instruct-v0.3 achieves the highest L1 AUROC in the program to date (0.778), extending bilateral oracle validation to five independent architectures (Qwen, Llama, Gemma, Mistral, Phi, range 0.731–0.846). **C036_CONFIRMED (EXP_CO_GEMMA_MISTRAL_V1):** CO labeling recovers L2 for both T2_L2 architectures — Gemma CO Fisher=0.8368 (gap=+0.2056), Mistral CO Fisher=0.8580 (gap=+0.2114), both CO_RECOVERS, both shuffled CLEAN. T2_L2 was a measurement framing artifact; CO labeling is the universal L2 estimator (C036).

Three Tier 0 experiments complete the current program phase. **Large-N cross-validated L2 validation (C040 STABLE_SIGNAL):** EXP_L2_LARGE_N_V1 yields CV Fisher AUROC=0.7629 ± 0.0120 (N=500/class CO-style, 5-fold, Qwen2.5-1.5B-Instruct), resolving the prior 0.670–0.885 variance as pool-section heterogeneity. **OOD generalization (C041 OOD_PARTIAL):** the bilateral oracle probe transfers to HotpotQA at 84.5% efficiency (transfer AUROC=0.6567 vs within=0.7769) but transfers at chance to MMLU-STEM (0.5288) — the geometry is format-scoped, not universally portable; all observability claims are qualified accordingly. **Training-stage sweep (C042/C043):** a matched-N comparison (Base→SFT→Reasoning, N=200/class) reveals a split: L1 observability shows INVERTED_U (BASE=0.740, INSTRUCT=0.803, REASONING=0.725 — SFT improves, reasoning distillation reduces below base due to task-distribution mismatch); L2 confabulation detection gap is MONOTONE_RISE through every stage (0.064→0.096→0.148), with reasoning distillation producing the strongest confabulation separation.

These findings define a **three-task measurement hierarchy**: knowledge-source routing (L1, entropy dominates), confabulation detection (L2, Fisher essential), and commitment timing within reasoning chains (L3, Fisher trajectory). The program maintains an open claims registry of **45 claims: 7 CONFIRMED, 26 SUPPORTED, 8 EXPLORATORY, 4 FALSIFIED**.

**Keywords:** epistemic legibility, computational observability, hidden state probing, bilateral oracle, Fisher LDA, mechanistic interpretability, knowledge-source routing, confabulation detection, commitment dynamics, reasoning models, scale observability

**Research program documentation:** Claims are formally registered in `science/CLAIMS.yaml`; experiments in `science/EXPERIMENTS.yaml`. The full scientific archive and intellectual history are in [docs/program_synthesis.md](program_synthesis.md).

---

## 1. Introduction

### 1.1 The Problem of Epistemic State in Language Models

When a language model answers a factual question, it may draw on at least three distinct computational sources: parametric knowledge encoded in its weights during training, information retrieved from a supplied context passage, or pattern-completion from the input structure alone. These sources have very different epistemic properties. Parametric knowledge is bounded by training data and degrades under distribution shift; context retrieval depends on the relevance and accuracy of the supplied material; pattern-completion may produce fluent but unfounded outputs. From the perspective of reliable AI systems, distinguishing which source a model is accessing for any given question is consequential: a response grounded in parametric knowledge may be trusted differently than one dependent on a specific context passage, and both differ from an apparent response that is actually confabulated.

The question this thesis addresses is whether this distinction is **measurable from the model's internal representations at generation time**, without access to the ground truth answer and without relying on output-level signals such as confidence scores or entropy.

This is not a new question in the interpretability literature, but prior work has approached it obliquely. Hallucination detection probes typically label outputs as correct or incorrect, confounding epistemic source with output quality. A model that happens to confabulate a correct answer and a model that retrieves a correct answer from parametric memory are both "correct" under standard labeling — but they are in different epistemic states. Calibration methods focus on output probability distributions rather than representation geometry. Mechanistic interpretability has studied which circuits implement specific computations but has not addressed the question of knowledge-source type as a representable property.

This thesis takes a different approach. The question is not only "does the model have this knowledge?" but: **is the model's routing decision between knowledge sources externally legible from the residual stream at the moment of generation onset?** Those are different questions. The first is about knowledge existence. The second is about whether the decision to access parametric versus contextual knowledge produces a measurable geometric signature before any output token is produced. If such a signature exists, it can be built into monitoring and inference infrastructure — a way to observe epistemic state without waiting for outputs. If it does not, that too is important to establish.

### 1.2 The Bilateral Oracle

The central methodological contribution is the **bilateral oracle**, a two-pass labeling protocol designed to isolate knowledge-source type from output quality. A question is labeled PARAM if the model answers it correctly without any context (nocontext token F1 ≥ 0.50). It is labeled CTX_DEP if the model fails without context (nocontext F1 ≤ 0.05) but succeeds with context (withcontext F1 ≥ 0.50). Questions that meet neither criterion are labeled SKIP and excluded from analysis. Hidden states are always extracted from the nocontext generation pass, ensuring the probe sees the model's internal state when it must rely on parametric memory alone.

This design makes several explicit choices. Using F1 rather than exact match accommodates vocabulary variation. The asymmetric threshold (PARAM: ≥0.50; CTX_DEP: ≤0.05 on nocontext side) ensures clean labels: PARAM items are ones the model answers well from memory; CTX_DEP items are ones the model cannot answer without the specific context passage. The SKIP zone (nocontext F1 between 0.05 and 0.50) is deliberately excluded — these items are ambiguous and might contaminate the probe training signal. The protocol makes no claim about items in the SKIP zone; a key experimental result (borderline geometry) later characterizes what those items look like geometrically.

**The oracle is the contribution; the probe is implementation.** Any probe family — logistic regression, SVM, MLP — applied to bilateral oracle labels would produce a result of the same kind. What is novel is the two-pass labeling design that separates knowledge-source type from output quality, and the specific choice to extract hidden states from the nocontext pass (where the routing decision is maximally uncontaminated by context-processing dynamics). A researcher could replace Fisher+PCA64 with any other probe and the methodological contribution would be unchanged. The probe choice determines the precision of the measurement; the oracle design determines what is being measured.

### 1.3 Research Questions

This thesis addresses four scientific questions:

**Q1 (Measurement):** Is the bilateral oracle distinction geometrically measurable in residual stream representations? What probe architecture is required, and how strong is the signal?

**Q2 (Structure):** What is the structural organization of the epistemic geometry? Is it linear or nonlinear? Is it architecture-consistent? Is it a continuous spectrum or discrete?

**Q3 (Dynamics):** When and where in the computation is epistemic legibility highest? Does the signal differ across layers and generation steps?

**Q4 (Causality):** Is the geometry causally load-bearing, or is it a readout of epistemic state that does not participate in the computation itself?

### 1.4 Contributions

This thesis makes the following contributions:


1. **The bilateral oracle protocol** — a two-pass labeling procedure that operationalizes knowledge-source type for open-domain QA by separating what the model knows from what it outputs. This is the primary contribution; the Fisher+PCA64 probe is the measurement instrument built on top of it.

2. **Linear organization of epistemic legibility** — Fisher+PCA64 achieves AUROC 0.7312 (CI [0.63, 0.83], N=197/class, clean large-N) on bilateral oracle labels at L26 step-1; nonlinear probes show no recovery (Δ ≤ 0.019), establishing that the routing decision is linearly legible in the residual stream. Earlier calibration-phase estimates of 0.841–0.846 (N=128–150/class) are superseded by the large-N result.

3. **Replication across two independently trained transformer families** — Large-N clean validation: Qwen2.5-1.5B-Instruct 0.7312 (CI [0.63,0.83], N=197/class) and Llama-3.2-3B-Instruct 0.7464 (CI [0.65,0.83], N=200/class). Δ=0.015, CIs overlap fully, both shuffled controls CLEAN. Architecturally consistent despite 2× hidden dimension difference.

4. **Generation-onset privileged signal** — step-1 hidden states carry substantially more epistemic signal than prefill across four model families (ESM v33, lighter calibration: mean 0.785 vs 0.567). Note: this result uses a lighter calibration protocol than the full bilateral oracle; the full protocol ran on two architectures. The temporal finding (step-1 > prefill) and the full-protocol AUROC finding (0.841–0.846) are different experiments with different protocols.

5. **MONOTONE_RISE and 2D surface structure** — (a) For verbose-answer items (EOS-filtered, n=44/class), epistemic signal rises monotonically from step-0 (0.639) to step-10 (0.906), falsifying a within-generation decay hypothesis. (b) 2D observability surface (10 layers × 5 steps, Qwen2.5-1.5B, EXP_T2A_2D_SURFACE): reliable cells (steps 0–2, n=40/class) show MIDDLE_LAYER_ACCUMULATION pattern — signal rises from L00 (0.61–0.64) to a peak at L16 (0.77–0.83), then decreases slightly at L20+ for steps 0–1. Step 2 > Step 1 > Step 0 in most mid-to-deep layers. The DEEP_EARLY_PEAK hypothesis (peak at L24–L26 step=0–1) is not supported: L26 step=1 = 0.662 in this experiment, substantially below L16 step=2 = 0.828. Step 5+ cells are unreliable due to EOS artifact (PARAM items EOS before step 5, leaving only 7 PARAM test items). The surface result is directional (n=40/class per cell); definitive layer comparison requires a large-N sweep at multiple layers.

6. **No causal leverage from centroid-level patching** — centroid-direction residual stream patching produces zero F1 gain at all tested layers (L4–L26); the geometry is a legibility readout, not a causal lever. Untested: head patching, SAE feature patching, circuit-level intervention.

7. **Preliminary bimodal structure** — at N=60/group (single architecture, single task), bilateral oracle geometry is not a continuous gradient; STRONG_PARAM (+1.316) and STRONG_CTX_DEP (−1.316) form discrete poles while intermediate items are geometrically undifferentiated (WEAK_PARAM ↔ BORDERLINE KS p=0.120). Replication required before this becomes a law.

8. **OOD transfer on multi-hop QA** — ANSWER-quadrant selection achieves +38.7pp accuracy on HotpotQA multi-hop QA (88.7% vs 50.0% subset baseline for ANSWER-quadrant items routed from TriviaQA-calibrated probe, two-run confirmed); j-score OOD AUROC = 0.9495. The 50.0% baseline is the model's accuracy on this specific routed subset without the ANSWER-quadrant filter, not the model's overall HotpotQA accuracy. Signal strength varies by benchmark format; NQ-Open transfers more weakly (+10.2pp).

9. **Framing as mechanistic test** — forcing answer-completion framing degrades step-1 AUROC (0.858 → 0.764) while leaving prefill unchanged (0.831 → 0.809). This establishes that the step-1 signal is specific to the generation-onset token, not to the model's general epistemic state in the prompt-processing phase. In light of the entropy baseline result (Contribution 11, §7.7), the most parsimonious interpretation is that framing changes the output distribution at step-1 (the forced "The answer is:" prefix generates non-answer tokens with uniformly low entropy regardless of epistemic state), disrupting both entropy-based and hidden-state-based discriminators. The framing result confirms temporal specificity at generation onset but does not distinguish routing-signal from output-confidence accounts.

10. **Training dynamics preliminary result** — Pythia-1.4b sweep across 8 checkpoints (step512 to step143000, EXP_T3A_PYTHIA) shows bilateral oracle AUROC ≥ 0.67 at every checkpoint, including step512 (AUROC=0.859). Legibility is not a late-training emergent phenomenon. Convergent checkpoints (step128000–143000) achieve AUROC=0.72; the curve is INVERTED_U (provisional: n_test=7–9/class, CIs ±0.20). C011 as stated (monotone growth from initialization to convergence) is not confirmed. Large-N replication (N≥50/class per checkpoint) required before characterizing the training dynamics curve reliably.

11. **Entropy baseline comparison (FISHER_REDUNDANT)** — direct comparison of Fisher+PCA64 against output entropy on the same bilateral oracle labels (EXP_ENTROPY_BASELINE, both architectures): Qwen entropy AUROC=0.9043 vs Fisher=0.6566 (Δ=−0.248); Llama entropy=0.874 vs Fisher=0.860 (Δ=−0.014); Llama combined=0.9037. Verdict: FISHER_REDUNDANT for both architectures. The bilateral oracle PARAM/CTX_DEP distinction is primarily a confident/uncertain classification — PARAM items generate with ~1.1–1.2 nats entropy, CTX_DEP items with ~2.4–3.3 nats (2–3× gap). The Fisher probe is capturing this distinction via a noisier channel than direct entropy measurement, especially for Qwen. For Llama, the combination provides a marginal independent increment (+0.030 AUROC). This finding characterizes what the bilateral oracle is measuring: output confidence separability, not a hidden-state routing signal orthogonal to logits. The bilateral oracle methodology is unaffected; its contribution is the clean labeling protocol, not the Fisher probe's performance advantage over entropy.

12. **Formal falsification record** — three confirmed falsifications documented and corrected. This record is not an appendix: it is how the measurement protocol earned its credibility. Two probe methodology failures (Fisher LDA degenerate covariance at small N) and one EOS-confound artifact were identified and fixed, each time producing a more precise claim.

### 1.5 The O/C/A Framework

The three-task hierarchy (L1 knowledge-source routing, L2 confabulation detection, L3 commitment timing) organizes the experimental findings. Underlying the hierarchy is a more fundamental set of measurable computational quantities that the experimental program is building evidence for.

**Observability (O)** measures whether external probes can reliably read epistemic state from internal representations. The bilateral oracle Fisher+PCA64 probe is the current estimator for O at L1 and L2: it operationalizes observability as the AUROC with which a linear classifier applied to residual stream hidden states can distinguish knowledge-source categories or confabulation status. The key property of O as a quantity is that the estimator can be replaced — a future paper establishing a superior confabulation probe would be measuring the same O, better. The bilateral oracle and Fisher+PCA64 are instruments; O is the underlying property.

**Commitment (C)** measures when internal computation settles on an answer direction. The commit% probe is the current estimator for C: it detects the generation step at which the Fisher trajectory crosses a threshold indicating the model has committed to its answer. For standard factoid generation, C is measured at the generation-onset step-1. For reasoning-distilled models, C is measured within the think block. The finding that reasoning-distilled models commit after only 17–18% of thinking tokens (commit%=75.8%/82.9%) is a measurement of C. C also appears in the MATH entry-point result (§5.9 of the arxiv paper): geometry at step-1 predicts final mathematical correctness before any chain-of-thought computation, an approach-commitment property.

**Accessibility (A)** measures how training — pretraining, SFT, RLHF, reasoning distillation — shapes the computational organization that O and C reflect. The training-stage sweep (C042/C043) is the primary evidence for A: the same backbone under Base, SFT, and reasoning distillation shows different O values (INVERTED_U on L1, MONOTONE_RISE on L2), establishing that training choices have systematic, directional effects on epistemic legibility.

These are intended as *quantities*, not organizing concepts. The distinction matters. An organizing concept is a framing choice — you could call the same data "knowledge retrieval quality" and organize it differently. A quantity is a thing you can measure, estimate with error bars, compare across architectures and training stages, and make predictions about. O, C, and A are quantities in this sense: they have numerical values, those values can be measured with different estimators, and the measurements from different estimators should converge as the instruments improve.

The estimators used in this thesis (Fisher+PCA64, bilateral oracle, commit% probe) are current implementations. They may be replaced by better instruments — sparse autoencoder feature probes, attention-head-level extractors, or trajectory-based estimators with better sample efficiency. What persists is the underlying computational property being measured. This is the same relationship that exists between temperature (a quantity) and thermometer design (an estimator): the quantity persists across instrument generations.

**The causal chain the program is testing:** Optimization → Computational Organization → Observability → Adaptive Computation. Training choices (Optimization) shape the geometric structure of internal representations (Computational Organization). That structure determines how legible the epistemic state is to external probes (Observability). Observability, in turn, can be used to implement adaptive computation (early exit triggers, confabulation routing, monitoring). The program currently has substantial evidence on the middle segment (Computational Organization → Observability) and preliminary evidence on the right segment (Observability → Adaptive Computation via early exit). The left segment (Optimization → Computational Organization) is what the training-stage sweep begins to characterize.

---

## 2. Background and Related Work

### 2.1 Hidden State Probing

The practice of training linear classifiers on transformer intermediate representations to predict properties of the input or output — probing — has a substantial literature (Tenney et al., 2019; Hewitt & Lippmann, 2019; Belinkov, 2022). Probing studies have established that syntactic properties (part-of-speech, dependency relations), semantic properties (named entities, coreference), and task-relevant information are geometrically accessible in hidden states.

This thesis applies probing to a different target: not properties of the input, but properties of the model's epistemic state with respect to the input. Prior hallucination detection probes (e.g., Slobodkin et al., 2023; Chen & Mueller, 2023) use correct/incorrect labels, which conflate knowledge source with output quality. The bilateral oracle protocol distinguishes these.

The Hewitt-Liang selectivity control (2019) — comparing probe performance on real versus shuffled labels — is used throughout this work to validate that probes learn from label signal rather than superficial statistics. Shuffled AUROC < 0.60 is the CLEAN threshold; values above this mark potential structural confounds.

### 2.2 V-Usable Information

The bilateral oracle instantiates the concept of V-usable information (Xu et al., 2020): information that is extractable by a specific probe family V from a representation. The bilateral oracle operationalizes "what does the model know from its weights alone?" as a labeled binary outcome (PARAM vs CTX_DEP), which then becomes the target for V-usable information measurement. The choice of V = {linear classifiers with PCA dimensionality reduction} is itself a claim: that the epistemic distinction is linearly accessible. This claim is empirically tested via the C3 series (Section 4).

### 2.3 Fisher Linear Discriminant Analysis

The Fisher+PCA64 probe consists of two stages: PCA reduction to 64 dimensions (capturing the dominant covariance structure while making the subsequent LDA problem well-conditioned), followed by Fisher Linear Discriminant Analysis (LDA) with lsqr solver and automatic shrinkage (Ledoit-Wolf estimator). The Fisher probe was chosen over logistic regression for its interpretable geometric output: the Fisher decision function is a projection onto the between-class separation axis, directly measuring the signed distance along the direction that maximally separates the two classes.

A critical methodological choice is the Ledoit-Wolf shrinkage estimator for the LDA covariance matrix. Without shrinkage, LDA in high dimensions (d=1536 for Qwen, d=3072 for Llama) with small sample sizes (N < 200/class) produces degenerate covariance estimates where the probe's shuffled AUROC can exceed its real AUROC. This catastrophic estimator failure was encountered in C3-v2 (Qwen instruct shuffled=0.713, real=0.708) and is what motivated the PCA+shrinkage design in C3-v3.

### 2.4 Mechanistic Interpretability

This work intersects with mechanistic interpretability but does not attempt circuit-level analysis. We use hooks to extract layer-specific hidden states at specific generation steps; we use linear probes to characterize their geometric structure; and we use activation patching to test causal load-bearing. The patching methodology follows Meng et al. (2022) and Hernandez et al. (2023), applying additive interventions to the residual stream along a specified direction.

### 2.5 Calibration and Uncertainty Estimation

Output-level calibration (ECE, reliability diagrams) measures whether the model's token probabilities are well-calibrated to accuracy. This thesis addresses a different question: whether the model's internal representation distinguishes epistemic cases that would look different under calibration. The finding that CTX_DEP and CONFAB items are geometrically similar at step-1 (CTX_DEP vs CONFAB AUROC=0.649) means that calibration, which observes outputs, cannot distinguish these cases; the bilateral oracle probe, which observes hidden states, provides partial but not complete distinction.

---

## 3. The Bilateral Oracle Protocol

### 3.1 Protocol Design

The bilateral oracle executes two generation passes per question:

**Pass 1 (nocontext):** Generate up to MAX_GEN=60 tokens from the prompt without any context passage. Compute token-level F1 against all gold answers (best match over answer aliases).

**Pass 2 (withcontext, conditional):** Only if Pass 1 F1 ≤ 0.05, generate with the context passage prepended. Compute token-level F1.

**Label assignment:**
- PARAM: nocontext F1 ≥ 0.50 → the model answers correctly from parametric memory
- CTX_DEP: nocontext F1 ≤ 0.05 AND withcontext F1 ≥ 0.50 → the model fails without context but succeeds with it
- SKIP: all other cases → excluded from analysis

**HS extraction:** Always from the nocontext pass only, at generation step-1 (first generated token), layer 26.

### 3.2 Rationale for Design Choices

**Why two passes?** Single-pass correct/incorrect labeling cannot distinguish "model knows the answer" from "model correctly retrieves the answer from the context." The bilateral oracle enforces this distinction explicitly. The "bilateral" refers to both passes being required for CTX_DEP labels.

**Why nocontext for HS extraction?** Extracting hidden states from the withcontext pass would contaminate the epistemic signal with context-processing dynamics. The nocontext pass is the cleaner measurement: it shows the model's internal state when only its parametric memory is available. PARAM items succeed in this condition; CTX_DEP items fail. The geometry difference is what we are measuring.

**Why step-1?** The first generated token after the prompt is the moment when the model must commit to a generation direction. It has processed the full question and is now generating its first output. This "commitment moment" is when the parametric vs contextual routing decision materializes. Evidence for this interpretation: (a) step-1 consistently outperforms prefill across model families (Section 5.1), (b) forcing answer-framing ("The answer is:") which eliminates the routing decision degrades step-1 AUROC significantly while leaving prefill unchanged (Section 5.5), and (c) for Llama-3.2-3B, step-1 peaks at 0.866 and decays at later steps (0.517 at step-5), consistent with a brief commitment window.

**Why layer 26?** Layer 26 is the penultimate layer in Qwen2.5-1.5B-Instruct and Llama-3.2-3B-Instruct (both 28 layers). The choice is empirically grounded, not assumed: a full 28-layer sweep in prior experiments showed that epistemic signal increases monotonically through the network, with Fisher LDA separability plateauing across L16–L26 and reaching its peak at L26. The trajectory is monotone and does not exhibit a middle-layer peak, which rules out interpretations that assign the signal to early-layer factual encoding. L26 is chosen over earlier layers specifically because it maximizes bilateral oracle AUROC on the downstream label — not because penultimate layers are a priori privileged. The monotone rise itself is a finding: epistemic routing information accumulates through depth rather than being localized.

**The SKIP zone and what it means.** Items with nocontext F1 between 0.05 and 0.50 are excluded. These items are neither clearly parametrically accessible nor clearly context-dependent. The borderline geometry experiment (Section 5.3) directly tests what these items look like in hidden state space, and finds they occupy a geometrically undifferentiated middle zone — supporting the protocol's decision to exclude them.

### 3.3 Pool Sampling and Yield

On TriviaQA (validation split, rc.wikipedia configuration), with pool_size=5000 shuffled items:
- PARAM yield: ~28% (≈1400 PARAM items in pool)
- CTX_DEP yield: ~2.4% (≈120 CTX_DEP items in pool)

The asymmetry is large and expected: TriviaQA was designed for systems with access to Wikipedia passages, so many items have context-dependent answers. CTX_DEP items are the bottleneck on N. A pool of 10,000 items is needed to reliably collect N=200 CTX_DEP items.

### 3.4 Token F1 Metric

The F1 metric follows the standard TriviaQA evaluation: precision = (matching tokens) / (predicted tokens), recall = (matching tokens) / (gold tokens), F1 = harmonic mean. Best F1 across all gold answer aliases is used, lowercased, with punctuation handling. This metric is more permissive than exact match, accommodating article and minor spelling variations while still penalizing off-topic responses.

---

## 4. Methodology

### 4.1 The Fisher+PCA64 Probe

**Architecture:** PCA(n_components=64, random_state=42) → LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').

The PCA step reduces the d-dimensional hidden state (d=1536 for Qwen2.5-1.5B, d=3072 for Llama-3.2-3B) to 64 principal components, capturing the dominant variance structure while making the LDA problem well-posed (N >> d is required; at N=90/class, PCA(64) satisfies this; raw d does not). The LDA step then finds the single direction in PCA space that maximally separates the two classes under the assumption of equal within-class covariance (relaxed by Ledoit-Wolf shrinkage, which tolerates some covariance inequality).

The probe outputs a scalar decision score (projection onto the Fisher axis) for each test sample. AUROC is computed as the area under the ROC curve over these scalar scores.

**Why not logistic regression?** Logistic regression (LR) with L2 regularization is included as a secondary probe for comparison in large_n_validation. Fisher LDA is primary because: (1) its decision function has direct geometric interpretation as a signed Fisher axis projection; (2) it failed clearly (shuffled > real) when underpowered, providing an honest probe quality signal; (3) after fixing with PCA+shrinkage, its shuffled AUROCs are consistently CLEAN.

### 4.2 Shuffled Control Baseline

Every probe evaluation reports both the real AUROC (trained on real labels, evaluated on real test labels) and the shuffled AUROC (trained on real labels, evaluated on label-permuted test set). The shuffled AUROC should be approximately 0.50 by symmetry. Values above 0.60 indicate a structural confound — some non-epistemic property of the items (question length, domain clustering, dataset ordering artifact) that the probe picks up even on shuffled labels.

Status thresholds: CLEAN < 0.60 ≤ WARN < 0.70 ≤ FAIL.

This baseline was critical for identifying a methodological error in C3-v2 (Qwen instruct shuffled=0.713, real=0.708) and a potential confound in C3-v3 Qwen (shuffled=0.617, WARN). The large_n_validation v1 with shuffled=0.4495 (CLEAN, properly shuffled dataset) provides the cleanest estimate to date.

### 4.3 Hook-Based Step-1 Extraction

Hidden states are extracted using PyTorch forward hooks registered on `model.model.layers[layer_idx]`. The hook fires on every forward pass. The condition `hs.shape[1] == 1` identifies generation-phase passes (single-token KV-cache decoding) versus prefill passes (full sequence processing). The hook captures the first generation-phase activation and immediately removes itself, capturing exactly step-1.

Generation is called with `max_new_tokens=2` for HS extraction (one for the step-1 capture, one for buffer), significantly reducing extraction time relative to full-answer generation.

### 4.4 Claims Governance Infrastructure

The experimental program is managed with a formal claims registry (`science/CLAIMS.yaml`) and experiment registry (`science/EXPERIMENTS.yaml`). Each claim has:
- A unique ID (C001–C015)
- A precise statement
- A status: CONFIRMED / SUPPORTED / EXPLORATORY / FALSIFIED
- Evidence entries linking to specific experiments and specific numerical results
- `last_updated` field
- `paper_sections` field (which paper sections may cite this claim)

Status semantics:
- **CONFIRMED**: replicated across ≥2 architectures, clean controls, N≥128/class
- **SUPPORTED**: single architecture or N<128/class; valid but needs replication
- **EXPLORATORY**: preliminary signal, insufficient evidence for paper citation
- **FALSIFIED**: directly contradicted by controlled experiment; must be disclosed if mentioned in paper

A pre-commit validator (`science/validate_claims.py`) enforces that paper drafts do not cite FALSIFIED or EXPLORATORY claims without explicit disclosure.

Experiment entries include `protocol_fingerprint` fields capturing every protocol parameter. Two experiments with different protocol fingerprints are not directly comparable, even if they measure nominally the same quantity.

### 4.5 Research Principles

The following are not experimental design guidelines. They are the scientific culture governing how claims are made in this program. Each emerged from a specific documented failure.

**Principle 1: Probe output-space baselines first.** A hidden-state probe adds value only if it exceeds the corresponding output-space signal — entropy, margin, top-1 probability — on the same items. A probe that performs worse than entropy at the same generation step is capturing output confidence through a noisier channel, not a distinct epistemic geometry. The FISHER_REDUNDANT finding (§7.7) was discovered by running this comparison. It should have been run before reporting any Fisher AUROC.

**Principle 2: Every measurement requires three things.** Shuffled controls (AUROC > 0.60 signals a confound), bootstrap confidence intervals (CI width > 0.20 means noise, not signal), and at least one out-of-distribution task. Without all three, a result is an instrument calibration, not a scientific claim.

**Principle 3: Observation ≠ causation.** Geometric separability does not establish that the geometry participates in computation. C005 (centroid patching epiphenomenal at L4–L26) drew this boundary. Any causal claim requires an intervention experiment — inference from correlation structure is not sufficient.

**Principle 4: Negative results are first-class scientific outcomes.** C012, C013, and C014 are cited positively throughout as evidence the instrument was tested and corrected. The program's credibility derives from its willingness to document what failed and why. A program that only reports positive results cannot be trusted.

**Principle 5: Every claim must have a pre-specified falsifier.** The conditions in §8 (Limitation 9) were written before those experiments ran. This prevents post-hoc rationalization. A claim without a falsifier is not a scientific claim; it is a preference.

**Principle 6: Prefer measuring computational processes over static representations.** A trajectory is richer than a snapshot. J_velocity, step-indexed AUROC, entropy trajectory, and commitment dynamics are all process measurements. The MONOTONE_RISE finding (step-0=0.639 → step-10=0.906) emerged from treating generation as a temporal process rather than as a single representational state.

**Principle 7: Every observable must be tested for invariance before it is treated as a scientific object.** A measurement that survives only one prompt template, one dataset, or one random seed is an implementation artifact. Before promoting any observable to a claim, test it across: (a) prompt variation — at least two phrasings; (b) decoding variation — greedy vs. nucleus sampling; (c) task variation — at least one OOD task; (d) dataset variation — at least one dataset outside TriviaQA; (e) architecture variation — at least two independent model families; (f) random seed variation — at least three seeds. The cross-task cosim result (noise floor) and the framing result (AUROC −0.094 under answer-framing) are both applications of this principle — they revealed that bilateral oracle observables do not survive task-format or prompt-template change. That is a finding, not a failure.

**Principle 8: Prefer discovering invariants over maximizing benchmark performance.** An AUROC number on TriviaQA is not a scientific finding. An AUROC that holds across five architectures, three tasks, and eight training checkpoints is a candidate law. The objective of this program is understanding what computational properties are architecturally conserved — not the highest number on any benchmark. Drifting toward benchmark optimization is the failure mode this principle guards against.

### 4.6 Scientific Ontology

A science needs agreed-upon objects. The following defines the vocabulary of Computational Observability Science — the nouns that allow precise communication across experiments, papers, and future collaborators.

**Observable:** Any measurable quantity extractable from a model's inference process without access to internal weights or gradients. Examples: output entropy, token margin, Fisher axis projection score at L26 step-1.

**Computational State:** The complete internal configuration of a model at a given layer and generation step — the full residual stream vector. Not directly accessible in full; partially observable through probes.

**Trajectory:** A time-indexed sequence of computational states or observables across generation steps or layers. The step-indexed AUROC curve (0.639 → 0.906), per-token entropy, J_velocity, and commitment dynamics are all trajectory measurements.

**Transition:** The change in computational state between two time steps. The commitment moment is a transition claim: something changes between the last prefill token and generation step-1.

**Observable Signature:** An observable systematically correlated with an internal computational process. Output entropy is an observable signature of the confident/uncertain distinction. Fisher+PCA64 score is an observable signature of the same distinction, via a noisier channel.

**Computational Invariant:** A property preserved across architectures, tasks, and training stages. Candidate: bilateral oracle entropy gap (PARAM ~1.1 nats, CTX_DEP ~2.4–3.3 nats) — holds across both tested architectures. Whether it holds across architectures, tasks, and training checkpoints determines its invariant status.

**Observability Law:** A stable, falsifiable relationship between computational variables that survives architecture and task variation. No confirmed laws yet. L001 (generation-onset privilege) and L002 (linear accessibility) are candidates requiring multi-architecture confirmation.

**Intervention:** Any manipulation of a model's computational state. Centroid-direction patching (C005), head-level patching, and SAE feature patching are interventions of increasing specificity.

**Causal Observable:** An observable that, when manipulated, changes model behavior. C005 established the bilateral oracle Fisher axis is NOT a causal observable at centroid level. Identifying causal observables is the primary open problem in the program.

### 4.7 The Observability Ladder

Every finding belongs on one rung. Stating the level alongside each result prevents premature elevation of dataset-specific artifacts to law-level claims.

| Level | Name | Criteria |
|---|---|---|
| **0** | Implementation artifact | Shuffled AUROC ≥ real AUROC; fails controls |
| **1** | Dataset-specific observable | Holds on one dataset, one architecture; no replication |
| **2** | Task-level observable | Replicates across datasets within same task category |
| **3** | Architecture-level observable | Replicates across ≥2 independent model families, clean controls |
| **4** | Training-level observable | Present across training stages (checkpoints), one architecture minimum |
| **5** | Computational law | Architecture-independent; survives full perturbation suite; pre-specified falsifier tested |

**Current levels for key findings:**

| Finding | Level | What promotes it |
|---|---|---|
| Fisher+PCA64 AUROC ≥0.70 on bilateral oracle (C001) | **3** | Full protocol on Gemma/Mistral (or confirmed Level 4 via Pythia) |
| Entropy AUROC 0.87–0.90 (C016) | **3** | Cross-task replication beyond TriviaQA |
| Step-1 > prefill (C006, 4 models lighter calibration) | **3** | Full bilateral oracle on 4 architectures |
| Pythia floor ≥0.67 from step512 (C011 floor) | **4** | Same protocol on a second training run (Llama checkpoints) |
| Bimodal structure (C015, N=60 single architecture) | **1** | N≥200/class, two architectures → Level 3 |
| Patching null C005 | **2** | Replication on Llama + head-level patching → Level 3 |
| INVERTED_U shape (C011 shape, n_test=7–9/class) | **1** | Large-N Pythia (N≥50/class) → Level 4 or Level 0 |
| OOD HotpotQA transfer (+38.7pp) | **2** | PopQA, 2WikiMultiHopQA replication → Level 3 |
| Fisher CC vs CW AUROC=0.854 gap=0.240 over entropy (C017) | **1** | Second architecture + second task → Level 2 |
| BO_Transfer AUROC=0.880 confabulation on epistemic axis (C018) | **1** | Second architecture → Level 2 |
| Entropy trajectory inversion AUROC=0.730 CC/CW (C019) | **1** | Second architecture → Level 2 |
| Family > scale (Qwen 1.5B 0.845 vs Llama 3B 0.608) (C020) | **1** | 2 families × 3 sizes at N=200/class → Level 2 |
| Bilateral oracle Goldilocks zone ~1B–2B on TriviaQA (C021) | **1** | Second task, second family → Level 2 |
| Early commitment in reasoning-distilled models (C022, two families) | **3** | Third family + training-checkpoint variation → Level 4 |

No finding is currently at Level 5. The 18-month goal is to confirm C001, C016, and C006 at Level 3, and to establish the first Level 4 finding beyond the Pythia floor.

---

## 5. Results

### 5.1 Result 1 — Linear Organization and Architecture Consistency (C3 Series)

**Experiment:** EXP_C3V3 — Nonlinear Probe Recovery, Bilateral Oracle Labels, N≥128/class.

The C3 series went through three versions to arrive at a clean protocol. C3-v1 used incorrect labels (PARAM vs WRONG, not bilateral oracle); C3-v2 used bilateral oracle labels but Fisher LDA in raw high-dimensional space, producing degenerate covariance estimates. C3-v3 fixed both problems.

**C3-v3 results:**

| Model | N/class | Fisher+PCA64 | Shuffled | Best Nonlinear | Δ (NL−Lin) | Verdict |
|-------|---------|-------------|----------|----------------|------------|---------|
| Qwen2.5-1.5B-Instruct | 128 | 0.841 | 0.617 (WARN†) | 0.821 | −0.019 | NO_RECOVERY |
| Llama-3.2-3B-Instruct | 150 | 0.846 | 0.427 (CLEAN) | 0.841 | −0.005 | NO_RECOVERY |
| **Qwen (large-N v2)** | **197** | **0.7312** | **0.598 (CLEAN)** | — | — | PRIMARY |
| **Llama (large-N v2)** | **200** | **0.7464** | **0.502 (CLEAN)** | — | — | PRIMARY |

†Qwen calibration-phase WARN explains inflated 0.841. Large-N v2 rows are the paper-citable numbers.

Nonlinear probes tested: SVM-RBF+PCA64, MLP-2 (128 hidden), MLP-3 (128-64 hidden). None exceeded Fisher+PCA64 by more than 0.019 AUROC, below the 0.05 recovery threshold. This establishes two claims:

- **C002 (CONFIRMED):** The bilateral oracle signal at L26 step-1 is linearly organized. Nonlinear structure does not contribute recoverable signal above Fisher+PCA64.
- **C003 (CONFIRMED):** The bilateral oracle signal was replicated across two independently trained transformer families — Qwen2.5-1.5B-Instruct and Llama-3.2-3B-Instruct — achieving 0.841 vs 0.846 (Δ=0.005) despite a 2× hidden dimension difference. This is not architecture generalization; both are transformer families. Extension to non-transformer architectures (state-space models, hybrid architectures) requires separate investigation.

The Qwen shuffled WARN (0.617 in C3-v3) was a real signal of ordering artifact. TriviaQA has topical clustering; C3-v3 took items in dataset order rather than shuffled, inflating AUROC through implicit train/test leakage via question-domain proximity.

**Large-N validation v2 (2026-07-03) resolves the tension.** With pool=10,000, N=197/class, fully shuffled, Qwen gives AUROC=0.7312 (CI [0.6256, 0.8283]), shuffled=0.5980 (CLEAN). The large_n_validation v1 result (0.6566, N=121/class) and v2 result (0.7312, N=197/class) are consistent: both are clean, and the v2 estimate is more reliable due to larger N. The calibration result (0.841) is outside the v2 CI upper bound (0.8283) and is treated as inflated.

**C001 threshold revised to ≥ 0.70.** The most defensible single-number claim is: "Fisher+PCA64 achieves AUROC=0.7312 (CI [0.63, 0.83]) on bilateral oracle labels at L26 step-1, N=197/class, clean large-N sampling. Shuffled control=0.598 (CLEAN). Signal is real and substantial." Llama large-N replication remains pending.

### 5.2 Result 2 — Cross-Model Signal Profile (ESM v33)

**Experiment:** ESM Probe v14 (Kaggle v33) — cross-model step ablation.

Four model families were evaluated on the bilateral oracle signal across generation steps:

| Model | Architecture | Precision | Prefill | Step-1 | Δ(S1−PF) | Pattern |
|-------|-------------|-----------|---------|--------|----------|---------|
| Qwen2.5-7B-Instruct | GQA | INT8 | 0.476 | **0.905** | +0.429 | STRONG |
| Llama-3.2-3B-Instruct | GQA | float16 | 0.651 | **0.866** | +0.216 | COMMITMENT_MOMENT |
| Gemma-2-2B-it | MQA | float16 | 0.611 | **0.695** | +0.084 | WEAK |
| Mistral-7B-v0.3 | SWA | INT8 | 0.528 | **0.674** | +0.146 | WEAK |

**Mean prefill: 0.567; Mean step-1: 0.785; Mean Δ: +0.219**

The step-1 > prefill gap holds across all four architectures, establishing **C006 (SUPPORTED):** the generation onset carries more epistemic signal than the prompt-processing hidden state. Note the scope distinction: the full bilateral oracle protocol (C3-v3, nonlinear recovery test) was run only on Qwen and Llama. The ESM v33 step-1 > prefill result covers four architectures but uses a lighter probe calibration. C003 (replication across two transformer families) and C006 (generation-onset signal across four) are separate claims with separate evidence bases.

The Llama-3.2-3B result shows a sharp **COMMITMENT_MOMENT** pattern: step-1 peaks at 0.866, then decays to 0.642 at step-3 and 0.517 at step-5. This is the strongest mechanistic evidence for the generation-onset hypothesis: at step-1, the model's internal state reflects the routing decision (parametric vs contextual); by step-5, the residual stream is dominated by generated content, overwriting the routing signal.

GQA (Grouped Query Attention) architectures (Qwen, Llama) consistently produce stronger signals than MQA (Gemma) and SWA (Mistral). This suggests that the attention mechanism's key-value sharing structure affects epistemic geometry — a finding that motivates the architecture-specific axis conservation experiment in the prior ESM program.

**Confabulation detection:** As a secondary probe target, the bilateral oracle Fisher probe detects confabulation (nocontext wrong, withcontext also wrong) versus correct PARAM with AUROC 0.722 on Llama — suggesting the probe partially distinguishes genuine parametric knowledge from parametric confabulation, though not cleanly (see Section 5.7 on multidimensionality).

### 5.3 Result 3 — Causal Structure (Activation Patching)

**Experiment:** EXP_P1V3 and EXP_P1V5 — centroid-direction residual stream patching.

**Setup:** A Fisher+PCA64 probe was calibrated to AUROC=0.9284 (N_CAL=80/class, Qwen2.5-1.5B-Instruct). The centroid difference vector (mean PARAM hidden state − mean CTX_DEP hidden state) was computed in the original d=1536 space. Patching applied this vector as an additive perturbation to the residual stream of 40 CTX_DEP test items at generation step-1, with strengths α ∈ {0.25, 0.50, 1.00, 1.50, 2.00}.

**Results:**

| α | True direction Δ | Shuffled direction Δ |
|---|-----------------|---------------------|
| 0.00 | +0.0000 | — |
| 1.00 | +0.0028 | +0.0054 (shuffled at same α) |
| 2.00 | +0.0054 | +0.0054 |

True direction at α=2.0 = shuffled at α=1.0. No specific advantage for the bilateral oracle direction.

**Layer sweep (α=1.0, n=10 items each):**

| Layer | Baseline F1 | True Δ | Shuffled Δ | Specific Δ |
|-------|------------|--------|-----------|------------|
| L04   | 0.0000     | 0.0000 | +0.0047   | −0.0047    |
| L08   | 0.0000     | 0.0000 | +0.0048   | −0.0048    |
| L12   | 0.0000     | 0.0000 | 0.0000    | 0.0000     |
| L16   | 0.0000     | 0.0000 | +0.0095   | −0.0095    |
| L20   | 0.0000     | 0.0000 | 0.0000    | 0.0000     |
| L22   | 0.0000     | 0.0000 | 0.0000    | 0.0000     |
| L24   | 0.0000     | 0.0000 | 0.0000    | 0.0000     |

**Verdict: NO CAUSAL LEVERAGE at centroid level, across all layers tested.**

The bilateral oracle direction produces precisely zero F1 gain at every tested layer. The model IS sensitive to perturbations (shuffled direction at L16 produces +0.0095, indicating outputs are being affected); the bilateral oracle direction is simply not a privileged control direction.

**Interpretation:** CTX_DEP items in TriviaQA are questions whose answers are absent from the model's parametric weights. The baseline F1=0.000 for every layer sweep item confirms these are hard-fail cases — the model cannot produce any correct tokens without context. Patching the residual stream with the "PARAM direction" cannot conjure knowledge that is not in the weights. The geometry correctly identifies the epistemic state, but moving the geometry does not change the state.

This is the thermometer analogy made precise: a thermometer accurately measures temperature, but moving the mercury does not change the temperature. The bilateral oracle geometry is a reliable legibility readout; it is not a control lever. This result precisely localizes the tool's scope.

**What was not tested.** The zero-leverage result is specific to centroid-direction mean patching of the residual stream. Three intervention types have not been tested: (1) attention head patching — targeting specific heads that are active on PARAM vs CTX_DEP items; (2) SAE feature patching — intervening on sparse autoencoder features rather than the dense mean direction; (3) circuit-level intervention — patching specific MLP neurons or attention circuits identified by mechanistic analysis. The causal structure of epistemic legibility may be visible at these finer granularities even though it is invisible at the mean centroid level. The claim C005 is scoped to centroid-direction residual stream patching only.

**This does not invalidate the monitoring use case.** A legibility monitor that tracks Fisher axis position over training epochs does not require the signal to be causally load-bearing. It requires the signal to reliably reflect the routing decision — which C3-v3 and the framing experiment both confirm.

### 5.4 Result 4 — Within-Generation Dynamics (Step-Index AUROC)

**Experiment:** EXP_T1D_STEP_INDEX_V3 — step-indexed AUROC profile with EOS filtering.

**The EOS confound (prior v2 result):** An earlier experiment (v2, no EOS filter) found AUROC=0.781 at step-0, dropping to 0.609 at step-1, with null results beyond step-5 (zero surviving PARAM items). This appeared to show within-generation decay. The correct interpretation: PARAM items EOS at step 2–4 (short answers like "Paris" or "1969"). As steps increase, PARAM items disappear from the sample — the decreasing AUROC was sample attrition, not signal decay.

**EOS-filtered protocol (v3):** Items are included only if the nocontext generation produces ≥8 tokens (min_gen_tokens=8), isolating verbose-answer PARAM items. After filtering, n=44/class PARAM and CTX_DEP items.

| Step | AUROC | Shuffled | n_param_used |
|------|-------|----------|-------------|
| 0    | 0.639 | 0.477    | 44          |
| 1    | 0.683 | 0.413    | 44          |
| 2    | 0.762 | 0.497    | 44          |
| 5    | 0.790 | 0.482    | 44          |
| 10   | **0.906** | 0.483 | 16          |

**Verdict: MONOTONE_RISE.** Signal rises monotonically from step-0 (0.639) to step-10 (0.906) for verbose items. Every shuffled AUROC is near 0.50 (CLEAN), confirming the rise is genuine signal, not a structural artifact.

**Two-population structure:** TriviaQA PARAM items split into two populations:
1. **Short-answer items (~93%):** Generate 1–7 tokens (EOS at step 2–4). For these, step-0 or step-1 is the only available extraction point. These drive the ESM v33 "step-1 > prefill" finding.
2. **Verbose items (~7%):** Generate ≥8 tokens (extended explanations or multi-part answers). For these, signal accumulates through generation, reaching 0.906 at step-10.

**Implications for C006:** The revised C006 states: (a) step-1 > prefill at the generation-onset comparison (0.785 vs 0.567, ESM v33) — this holds; (b) within generation, signal does NOT decay for verbose items — it rises. The "step-1 is privileged" framing is specifically about generation-onset vs prompt-processing, not about within-generation dynamics.

**Step-10 caveat:** Only 16 of 44 PARAM items survive to step-10 (those generating ≥10 tokens). These are a self-selected verbose-answer subset. The step-10 result (0.906) characterizes this subset, not the general PARAM population.

### 5.5 Result 5 — Epistemic Geometry Structure (Borderline Experiment)

**Experiment:** EXP_T1C_BORDERLINE v4 — five-group population geometry.

**Motivation:** Is the bilateral oracle protocol thresholding a continuous latent accessibility variable (continuous manifold hypothesis), or detecting genuinely discrete epistemic states (discrete clusters hypothesis)?

**Design:** Five groups were defined by nocontext F1 ranges:
- STRONG_PARAM: nocontext F1 ∈ [0.50, 1.01)
- WEAK_PARAM: nocontext F1 ∈ [0.15, 0.50)
- BORDERLINE: nocontext F1 ∈ [0.05, 0.15)
- STRONG_CTX_DEP: nocontext F1 ∈ [0.00, 0.05) AND withcontext F1 ≥ 0.50
- WEAK_CTX_DEP: nocontext F1 ∈ [0.00, 0.10) AND withcontext F1 ≥ 0.20

N=60 items per group were collected from pool=5000. The Fisher+PCA64 probe was trained on STRONG_PARAM vs STRONG_CTX_DEP and applied to all five groups. If the continuous manifold hypothesis is correct, all five groups should arrange in monotone order along the Fisher axis with smooth KS test statistics. If the discrete clusters hypothesis is correct, only the STRONG groups should be clearly separated.

**Results:**

| Group | Fisher Mean | Std | 
|-------|------------|-----|
| STRONG_PARAM | **+1.316** | 1.015 |
| WEAK_PARAM | +0.073 | 0.794 |
| BORDERLINE | −0.278 | 0.663 |
| WEAK_CTX_DEP | −0.269 | 0.741 |
| STRONG_CTX_DEP | **−1.316** | 0.959 |

| Adjacent pair | KS statistic | p-value | Significant? |
|--------------|-------------|---------|-------------|
| STRONG_PARAM vs WEAK_PARAM | 0.500 | ≈0.000 | **Yes** |
| WEAK_PARAM vs BORDERLINE | 0.217 | **0.120** | **No** |
| BORDERLINE vs STRONG_CTX_DEP | 0.583 | ≈0.000 | **Yes** |
| STRONG_CTX_DEP vs WEAK_CTX_DEP | 0.550 | ≈0.000 | **Yes** |

Training AUROC (STRONG groups only): 0.9631.

**Verdict: DISCRETE_CLUSTERS. Continuous manifold hypothesis not supported (preliminary result: N=60/group, single architecture, single task).**

**Interpretation:** The geometry is bimodal, not gradient-structured. Two strong poles exist at ±1.316 (mirror-symmetric to within floating point). Between them, WEAK_PARAM (+0.073), BORDERLINE (−0.278), and WEAK_CTX_DEP (−0.269) are geometrically indistinguishable — WEAK_PARAM vs BORDERLINE KS test is not significant (p=0.120). The group means WEAK_CTX_DEP (−0.269) and BORDERLINE (−0.278) differ by only 0.009, well within noise.

The bilateral oracle's SKIP zone (nocontext F1 0.05–0.50) is not an arbitrary exclusion of a continuous gradient — it excludes items that, in this preliminary experiment, occupy a geometrically undifferentiated middle zone. The protocol appears to be identifying two real epistemic poles rather than thresholding a spectrum.

**Caveat on strength of conclusion.** This result is EXPLORATORY (C015): N=60/group, Qwen2.5-1.5B-Instruct only, TriviaQA only. The bimodal finding is plausible and merits follow-up, but should not be treated as established structure until replicated at larger N across at least two architectures. The most conservative interpretation is that the bilateral oracle SKIP zone corresponds to geometrically ambiguous items in this model — a finding that justifies the protocol's exclusion criterion, but does not yet constitute evidence for a general law about parametric knowledge.

Note on WEAK_CTX_DEP position: WEAK_CTX_DEP has a slightly higher mean (−0.269) than BORDERLINE (−0.278), breaking the expected monotone ordering. This ordering inversion of magnitude 0.009 is noise (given std ≈ 0.70 for both groups) and does not affect the main conclusion.

### 5.6 Result 6 — Framing Invariance and Metacognitive Interpretation

**Experiment:** ESM Framing v1 — effect of output framing on step-1 AUROC.

**Design:** Three framing conditions versus a standard condition, all on Llama-3.2-3B-Instruct (n=300 balanced, bilateral oracle):
- Standard: "Answer the following in one short phrase. Question: [Q] Answer:"
- Answer frame: "Answer the following in one short phrase. Question: [Q] The answer is:"
- CoT frame: "Answer the following in one short phrase. Question: [Q] Let me think step by step."
- Hedge frame: "Answer the following in one short phrase. Question: [Q] I'm not sure, but I think"

| Condition | Prefill AUROC | Step-1 AUROC | Δ(S1−PF) |
|-----------|-------------|-------------|----------|
| Standard | 0.831 | **0.858** | +0.027 |
| Answer frame | 0.809 | 0.764 | −0.046 |
| CoT frame | 0.835 | 0.791 | −0.044 |
| Hedge frame | 0.834 | **0.846** | +0.012 |

**H1 (answer-commitment hypothesis) REJECTED:** If the step-1 signal were an output confidence signal, appending "The answer is:" should amplify it by forcing immediate commitment. Instead, answer framing degrades step-1 AUROC by 0.094 (0.858 → 0.764). Prefill is unchanged (0.831 → 0.809).

**Interpretation:** When "The answer is:" is appended, the first generated token is the answer word itself ("Paris", "the", "1969") rather than a free-form generation token. The Fisher LDA direction — calibrated on standard prompts — does not align with answer-completion geometry. The routing decision has already been made by the prompt; there is no routing moment at step-1.

**H2 (metacognitive routing hypothesis) supported:** The commitment moment captures a genuine metacognitive routing decision — whether the model's parametric knowledge is activated — that occurs naturally at generation onset but is displaced or overwritten when the prompt structure forces a different completion. Hedge framing (+0.012) preserves the signal because the model still makes a genuine knowledge-access decision even when hedging its output.

**Prefill invariance:** All four conditions show approximately equal prefill AUROC (0.809–0.835), confirming that the epistemic information is encoded during question processing and does not depend on the assistant-prefix tokens. The generation-onset step-1 is the readout point; the prefill is the encoding stage.

### 5.7 Result 7 — Multidimensional Epistemic Structure

**Experiment:** ESM Dimensionality v1 — four-class LDA in PCA(50)-reduced hidden states.

**Design:** Four groups collected on Llama-3.2-3B-Instruct (n=60/class): CTX_DEP, PARAM_HIGH (nocontext F1 ≥ 0.70, confirmed correct), PARAM_LOW (nocontext F1 0.40–0.70, partially correct), CONFAB (nocontext F1 ≤ 0.10 AND withcontext F1 ≤ 0.10 — fails regardless of context). Four-class LDA extracts three discriminant axes.

**Eigenspectrum:**
- Dim1: 35.9% of between-class variance
- Dim2: 11.7%
- Dim3: 9.4%
- Dim1+Dim2: 47.6%

**The epistemic variable is not scalar.** Dim1 explains only 36% of between-class variance; a scalar "epistemic axis" framing is insufficient.

**Pairwise AUROCs on Dim1 vs Dim2:**

| Pair | Full LDA | Dim1 only | Dim2 only |
|------|---------|-----------|-----------|
| CTX_DEP vs PARAM_HIGH | 0.876 | 0.597 | **0.788** |
| CTX_DEP vs PARAM_LOW | 0.960 | **0.900** | 0.651 |
| CTX_DEP vs CONFAB | 0.649 | 0.553 | 0.679 |
| PARAM_HIGH vs CONFAB | 0.965 | 0.569 | **0.899** |
| PARAM_HIGH vs PARAM_LOW | 0.763 | 0.758 | 0.672 |
| PARAM_LOW vs CONFAB | 0.976 | **0.867** | 0.789 |

**Axis interpretation:**
- **Dim1 ≈ Routing Confidence:** Separates CTX_DEP from PARAM_LOW strongly (0.900), captures parametric routing orientation. CTX_DEP, PARAM_HIGH, and CONFAB are similar on Dim1.
- **Dim2 ≈ Knowledge Reliability:** PARAM_HIGH vs CONFAB AUROC = 0.899 on Dim2 alone. Dim2 positively correlates with nocontext F1 (ρ=0.439, p=1e-12) and token confidence (ρ=0.274, p=1.6e-5).

**Critical finding — CTX_DEP ≈ CONFAB geometrically:** CTX_DEP vs CONFAB AUROC = 0.649 (weakest pairwise separation). At generation step-1, context-dependent items (knowledge absent but contextually accessible) and confabulated items (knowledge absent, not contextually accessible) look similar in the residual stream. Both are in a "parametric routing failure" state — the model commits to parametric generation but does not have the knowledge. The difference between them (whether context would help) is not visible in the step-1 hidden state.

**Implication for safety:** Both CTX_DEP items (good — context would help) and CONFAB items (dangerous — confident but wrong) appear in the same region of the ANSWER quadrant under the bilateral oracle probe. This is the central safety gap in the routing architecture: confident confabulation and context-recoverable questions look geometrically identical at step-1.

### 5.8 Result 8 — Cross-Task Geometry Specificity

**Experiment:** cross_task_cosim_v1 — Fisher axis cosine similarity across task types.

Fisher+PCA64 probes were trained on three tasks (TriviaQA, HotpotQA, MMLU) on Qwen2.5-7B-Instruct at L26. The cosine similarity between probe axes across tasks was computed.

**Cosine matrix:**
| Task pair | Cosine similarity |
|-----------|------------------|
| TriviaQA ↔ HotpotQA | +0.035 |
| TriviaQA ↔ MMLU | −0.024 |
| HotpotQA ↔ MMLU | +0.004 |
| Mean | 0.021 |
| Expected noise floor (random unit vectors, d=4096) | ~0.016 |

**Verdict: TASK_SPECIFIC_GEOMETRY.** All task-pair cosine similarities are near the noise floor. The Fisher axis learned on TriviaQA does not align with the axis learned on MMLU or HotpotQA.

**Hewitt-Liang selectivity control:** TriviaQA probe shuffled-label AUROC = 0.543 ± 0.043 (5 trials) vs real AUROC = 1.000. The probe is PROBE_SELECTIVE — it is not fitting noise. The task-specificity is a real geometric property, not an estimation artifact.

**Interpretation:** Epistemic accessibility geometry is task-specific. A probe calibrated on open-ended factoid QA (TriviaQA) does not transfer to multiple-choice QA (MMLU) or multi-hop QA (HotpotQA). This has important practical implications: the bilateral oracle probe requires recalibration on each deployment domain. It is not a universal epistemic signal.

**Note on MMLU:** MMLU multiple-choice questions commit parametrically regardless of whether the answer is correct (all options are presented, forcing parametric completion of whichever seems most likely). This results in j-score being anti-predictive on MMLU (AUROC=0.104 in the 2×2 validation), confirming that the Fisher direction captures "routing to parametric generation" not "correct parametric answer." This is the distinction between commitment geometry and knowledge validity (Section 7.1).

### 5.9 Result 9 — Out-of-Distribution Transfer (2×2 Validation)

**Experiment:** EXP_2X2V1 and EXP_2X2V2 — joint threshold routing on HotpotQA and NQ-Open.

**Architecture:** Items are routed into four quadrants using two signals — j-score (Fisher discriminant, parametric vs contextual routing) and margin (output confidence):
- ANSWER: j > θ_j AND margin > θ_m (confident parametric)
- RETRIEVE: j < θ_j AND margin > θ_m (confident contextual)
- DEFER: j > θ_j AND margin < θ_m (uncertain parametric)
- ESCALATE: j < θ_j AND margin < θ_m (uncertain contextual)

Thresholds calibrated on TriviaQA: θ_j=−1.528, θ_m=0.453. Calibration AUROC: j=0.784, ESV-LDA=0.816.

**HotpotQA (multi-hop, out-of-distribution — 2 runs, n=150 each):**

| Metric | v1 | v2 |
|--------|----|----|
| ANSWER quadrant accuracy | 0.883 (+38.3pp) | **0.887 (+38.7pp)** |
| Baseline accuracy | 0.500 | 0.500 |
| j AUROC OOD | 0.9429 | **0.9495** |
| ESCALATE: % CTX_DEP | 100% | 100% |

The ANSWER quadrant (+38.7pp, two-run confirmed) is the headline empirical result. Items where both the Fisher probe and the margin signal indicate confident parametric routing achieve 88.7% accuracy on HotpotQA, even though the probe was calibrated on TriviaQA. The OOD AUROC of 0.9495 is the strongest transfer result in the program.

**Why HotpotQA transfers better than NQ-Open.** The mechanistic argument is specific to the ANSWER-quadrant condition. ANSWER items are selected because the bilateral oracle probe predicts confident parametric routing (j-score > θ_j) and the margin signal confirms high output confidence. On HotpotQA, this joint condition is informative because HotpotQA multi-hop items that a model can answer from parametric memory alone tend to be questions where the reasoning chain is entirely within the model's training distribution. The joint condition selects for items that do not require multi-document synthesis — a highly predictive criterion. On NQ-Open, items that the model confidently commits to parametrically are much more common (ANSWER quadrant covers 72% of items vs HotpotQA's 41%), which lowers precision. The difference is not about OOD transfer strength in general; it is about how selective the ANSWER quadrant is on each benchmark. HotpotQA's multi-hop structure means confident parametric commitment is rarer and more informative when it occurs.

**ESCALATE = 100% CTX_DEP:** Items routed to ESCALATE (uncertain, contextual) are exclusively CTX_DEP items under the bilateral oracle. This is structural coherence — the routing architecture correctly identifies the regime in which contextual information is needed.

**ESV-LDA (multi-signal combination) over-fits:** Combined signal AUROC on calibration domain = 0.816 vs j alone = 0.784 (+0.032). On HotpotQA OOD: ESV-LDA Δ = −0.019 (hurts). On NQ-Open OOD: ESV-LDA Δ = −0.070 (hurts further). **j-score alone is the OOD-optimal signal.** Multi-signal combination overfits the calibration domain.

**MMLU (out-of-scope):** j AUROC = 0.104 (anti-predictive). As discussed in Section 5.8, MC format forces parametric commitment regardless of correctness. Commitment geometry is not knowledge validity (Section 7.1). MMLU is excluded from v2.

**NQ-Open (in-distribution format, weaker signal):** ANSWER accuracy = 0.602 (+10.2pp over baseline=0.500). j AUROC = 0.765. Signal transfers but weaker than HotpotQA. The ANSWER quadrant covers 72% of NQ-Open items (less selective than HotpotQA's 41%), reducing precision.

### 5.10 Result 10 — False Certainty Detection (EXP-A)

**Experiment:** false_certainty_v2 — hidden-state vs. entropy discriminability of confident confabulation, entropy-matched collection design.

**Design:** Three-phase experiment on Qwen2.5-1.5B-Instruct (TriviaQA validation set).

*Phase 1 — Bilateral oracle calibration.* Pool=5,000 items. PARAM=100, CTX_DEP=68 (yield ~1.4%). Fisher+PCA64 calibration AUROC: **0.969** (shuffled=0.511). This probe is later applied out-of-task in BO_Transfer.

*Phase 2 — Entropy-matched collection.* θ_conf (30th-percentile step-0 entropy) = 1.1043. Confident zone: 2,398 items (~30% of remaining pool=7,993). CONFIDENT_CORRECT (CC): nocontext F1 ≥ 0.50 AND step-0 entropy < θ_conf, N=100. CONFIDENT_WRONG (CW): nocontext F1 ≤ 0.05 AND step-0 entropy < θ_conf, N=100. Entropy matching confirmed: CC mean entropy=0.580, CW mean entropy=0.617, Δ=0.037. Both classes have similarly low entropy by construction.

*Phase 3 — Probe comparison on CC vs CW.* Fisher+PCA64 and entropy probes applied to hidden states at L26, step-0:

| Probe | AUROC | Notes |
|---|---|---|
| FC_Fisher | **0.854** | shuffled=0.426 — clean control |
| FC_Entropy | 0.614 | residual despite entropy-matched collection |
| FC_JV | 0.307 | near-random — J-velocity adds nothing |
| FC_Combined | 0.866 | Fisher dominant; entropy adds <0.01 |
| BO_Transfer | **0.880** | PARAM/CTX_DEP probe applied directly to CC/CW |

**Key finding — Fisher–entropy gap = 0.240.** Within the entropy-matched confident zone, hidden states discriminate CONFIDENT_CORRECT from CONFIDENT_WRONG with AUROC=0.854, while step-0 entropy alone achieves only 0.614. The residual entropy gap between CC and CW after matching is 0.037 — this cannot account for a 0.240 AUROC difference. Hidden states carry confabulation signal that output entropy does not explain (C017).

**BO_Transfer finding (C018).** The bilateral oracle probe — trained on PARAM vs CTX_DEP routing, not on confabulation — transfers to the CC vs CW distinction with AUROC=0.880. Geometric interpretation: CONFIDENT_WRONG items sit in the hidden-state region associated with CTX_DEP items. When the model confabulates confidently, its routing geometry resembles context-dependent generation onset, not correct parametric recall. Confabulation geometry lives on the epistemic accessibility axis.

**Revision of FISHER_REDUNDANT framing.** The entropy baseline (Section 7.7, C008, C016) established that entropy equals or exceeds Fisher on the bilateral oracle task (PARAM vs CTX_DEP). EXP-A shows this conclusion is task-scoped. The PARAM/CTX_DEP distinction is primarily a confident/uncertain split — entropy nearly saturates it. The CC/CW distinction is harder: both classes are already within the confident zone. Fisher adds 0.240 AUROC beyond entropy on this finer task. The correct framing: entropy is redundant with Fisher for knowledge-source routing classification; Fisher is not redundant for confabulation detection within the confident zone. These are different tasks and the same measurement instrument gives opposite conclusions about Fisher's added value.

**Cross-architecture replication.** See §5.15 for Llama-3.2-3B-Instruct replication (Fisher=0.818, entropy=0.453, gap=0.365, BO_Transfer=0.768), which promotes C017 and C018 to CROSS-ARCH SUPPORTED.

### 5.11 Result 11 — Entropy Trajectory Signature (EXP-D)

**Experiment:** entropy_trajectory_v2 — 15-step generation entropy trajectory for CONFIDENT_CORRECT vs CONFIDENT_WRONG items.

**Design:** Two-pass experiment on Qwen2.5-1.5B-Instruct (TriviaQA validation). Same θ_conf=1.1043 as EXP-A. Pass 1: fast scan of 7,993 items (entropy + F1 per item). Pass 2: KV-cache trajectory extraction for 100 CC and 100 CW candidates (N_STEPS=15; EOS-padded if generation terminates early).

**Results:**

| Metric | Value |
|---|---|
| Trajectory LR AUROC | **0.730** |
| Features AUROC (slope/AUC/min/max/std) | 0.621 |
| Peak per-step AUROC | 0.617 (step 4) |
| Step-0 per-step AUROC | 0.331 (below chance) |
| Verdict | **TRAJECTORY_SIGNAL** |

**Entropy trajectory arrays (mean across 100 items per class, steps 0–14):**

CC: [0.751, 0.903, 0.723, 1.208, 1.568, 1.455, 1.298, 1.235, 1.048, 0.982, 0.808, 0.766, 0.860, 0.868, 0.910]

CW: [1.007, 1.199, 0.900, 1.060, 1.027, 1.056, 0.847, 0.876, 0.968, 1.024, 0.917, 0.928, 0.724, 0.747, 1.006]

**The entropy inversion.** At step-0, CW entropy (1.007) > CC entropy (0.751): per-step AUROC=0.331 (CC items are more certain at step-0, so step-0 AUROC is below chance when predicting CC vs CW). Between steps 2 and 3, trajectories cross. At step-4, CC entropy (1.568) > CW entropy (1.027): per-step AUROC=0.617. CORRECT items start with a highly peaked first-token distribution (the key fact word), then burst as the model moves into elaboration. WRONG items start somewhat uncertain and remain relatively flat throughout.

**Entropy matching caveat.** CC step-0 mean=0.563, CW step-0 mean=0.668 (gap=0.105, larger than EXP-A's 0.037 after matching). Some trajectory AUROC reflects this initial level difference. However, the per-step AUROC inversion (0.331 → 0.617) confirms the trajectory shape carries independent information beyond the initial entropy level.

**Relation to EXP-A.** Step-0 Fisher hidden states (EXP-A: AUROC=0.854) carry more signal than the full 15-step entropy trajectory (EXP-D: AUROC=0.730). These are complementary signals: Fisher reads internal geometry at step-0; entropy trajectory reads the temporal dynamics of output generation. The combination has not been tested and is a candidate for a three-signal confabulation probe (Fisher at step-0 + entropy trajectory shape).

### 5.12 Result 12 — Scale Observability (EXP-E)

**Experiment:** scale_obs_v1 — bilateral oracle Fisher+PCA64 and entropy AUROC across model scale, two families.

**Design:** Five models loaded serially: Qwen 0.5B, 1.5B, 3B; Llama 1B, 3B. Pool=3,000 TriviaQA items per model, N_target=80/class. Fisher+PCA64 at penultimate layer (layer N−2), step-1. Spearman ρ vs log10(params) computed across qualifying models.

**Bilateral oracle capability constraint — Goldilocks zone (C021).** Two models were excluded from analysis due to bilateral oracle protocol failures:

*Qwen 0.5B:* PARAM=0 across all 3,000 items. The model cannot answer TriviaQA factoid questions correctly without context — it falls below the parametric floor. Zero PARAM items makes bilateral oracle calibration impossible.

*Qwen 3B:* PARAM=80 from the first ~750 items (high parametric knowledge), but CTX_DEP=9 across all 3,000 items (0.3% yield). The model is so capable it rarely requires context for TriviaQA. Calibration fails from the CTX_DEP side.

These two failures define a capability constraint: the bilateral oracle requires a model in the capability range where it answers some TriviaQA questions correctly without context (PARAM floor, lower bound) but still fails on others requiring context (CTX_DEP ceiling, upper bound). On TriviaQA, this window is approximately 1B–2B parameters. Models outside either bound are incompatible with the protocol on this dataset.

**Results for three qualifying models:**

| Model | Params | Family | Fisher AUROC | Entropy AUROC | Shuffled |
|---|---|---|---|---|---|
| Qwen 1.5B | 1.5B | qwen | **0.845** | 0.885 | 0.440 |
| Llama 1B | 1B | llama | 0.633 | 0.723 | 0.515 |
| Llama 3B | 3B | llama | 0.608 | 0.673 | 0.335 |

**Cross-scale analysis (n=3):**
Spearman ρ(Fisher, log10_params) = −0.500; Spearman ρ(Entropy, log10_params) = −0.500 (n=3, not significant). Verdict: **FLAT_BOTH**.

**Family > Scale (C020).** Qwen 1.5B (Fisher=0.845) exceeds Llama 3B (Fisher=0.608) by 0.237, despite having 2× fewer parameters. Within the Llama family, scale has no positive effect: Llama 3B (0.608) is slightly lower than Llama 1B (0.633). Architecture family predicts bilateral oracle Fisher AUROC better than parameter count.

**Entropy dominates Fisher at all scales.** Consistent with FISHER_REDUNDANT (C008, C016): entropy AUROC exceeds Fisher in all three qualifying models. Fisher–entropy gaps: Qwen 1.5B Δ=0.040, Llama 1B Δ=0.090, Llama 3B Δ=0.065.

**Observability scaling hypothesis (H-A, §7.10) not supported.** Compression hypothesis predicts legibility increases monotonically with parameter count. Llama 3B Fisher=0.608 < Llama 1B Fisher=0.633, and Qwen 1.5B Fisher=0.845 exceeds both larger Llama models. Family-level architecture confounds the parameter scaling prediction. The result is consistent with the family-specific geometry finding (C020) but inconsistent with a universal parameter-count scaling law for epistemic legibility.

### 5.13 Result 13 — Reasoning Geometry: Early Commitment in Distilled Reasoning Models (EXP-B, EXP-C)

**Experiments:** reasoning_geometry_v1 (EXP-B, DeepSeek-R1-Distill-Qwen-1.5B) and reasoning_geometry_llama_v1 (EXP-C, DeepSeek-R1-Distill-Llama-8B).

**Design.** Both experiments use the same protocol. The model generates a full reasoning chain (think block delimited by `</think>` token). A bilateral oracle Fisher+PCA64 probe is calibrated on committed vs uncommitted hidden states. For each question in the main experiment, the think-block trajectory is evaluated token by token. The commit point is the first token where the probe score exceeds commit_thresh (p20 of held-out committed states). commit% = fraction of think tokens that are post-commitment (post-commit tokens / total think tokens). Shuffled controls permute token order within each trajectory.

**Results:**

| Model | Family | Params | Cal AUROC | Cal Layer | commit% | z-score | N |
|---|---|---|---|---|---|---|---|
| DeepSeek-R1-Distill-Qwen-1.5B | qwen | 1.5B | 0.760 | 26 | **75.8%** | 49.77 | 200 |
| DeepSeek-R1-Distill-Llama-8B | llama | 8B | **1.000** | 28 | **82.9%** | 679.73 | 100 |

**Verdict: COMMITTED_EARLY on both architectures.** Both reasoning-distilled models commit to their answer direction after generating only ~18–24% of their thinking tokens. The remaining 75–83% of the chain-of-thought occurs after the commitment threshold is crossed; whether this constitutes elaboration, verification, or stabilization is not yet directly measured.

**EXP-B (Qwen 1.5B) detail.** Calibration AUROC=0.760 at L26. Main experiment N=200: mean commit%=75.8% (over committed traces), z=49.77. commit_rate: most questions committed (commit=None was rare). Early-exit test: for questions where truncating at the commit point gives a different answer, f1_delta = f1_full − f1_early = +0.008. This means the full post-commitment elaboration is marginally helpful on diverging cases (+0.008 F1), not counterproductive. The 75.9% figure is mean commit% among diverging-answer cases specifically. Most questions produce the same answer whether truncated at commit or not.

**EXP-C (Llama 8B) detail.** Calibration AUROC=1.000 at L28 — perfect geometric separation between committed and uncommitted states. Main experiment N=100: mean commit%=82.9%, z=679.73. The first 20 questions showed commit=None (mean=0.0%); questions 21–100 showed consistent early commitment (mean jumped to 74.5% by item 40). The z=679.73 is 13.6× larger than EXP-B's z=49.77 despite using the same protocol — the Llama backbone reasoning model has stronger and more consistent early commitment dynamics.

**Architecture-invariant finding (C022).** COMMITTED_EARLY holds across two independent reasoning-distilled transformer families with different parameter counts (1.5B vs 8B), different think-token IDs, different optimal probe layers (26 vs 28), and different F1 performance levels. This cannot be explained by either specific architecture. **Teacher confound RESOLVED (C045):** EXP_TEACHER_INDEPENDENCE_V1 on Qwen3-1.7B yields commit_pct=99.8% (commits within first 1-2 think tokens, z=1.3×10¹⁵, N=100/100). Early commitment is NOT R1-specific — Qwen3-native reasoning shows even more extreme early commitment. Law 2 now rests on 3 models from 2 independent training lineages. The reasoning chain in distilled reasoning models is primarily post-commitment in its temporal structure; characterizing what happens in that post-commitment window (elaboration vs verification or stabilization) requires further analysis.

**On the "search vs theater" binary.** An early framing characterized CoT reasoning as either "search" (genuine pre-decision exploration) or "theater" (post-decision justification). EXP-B/C show strong evidence for a post-commitment elaboration pattern, but this is better modeled as a trajectory than a binary. A richer commitment model: Exploration → Weak commitment → Possible revision → Strong commitment → Elaboration. The Fisher probe detects when the trajectory crosses the "strong commitment" threshold — not whether there was genuine exploration before it. Whether the pre-commitment phase is productive exploration or low-certainty drift is not yet measured; that is what the never-committed 20/100 Llama items, EXP-F, and per-question trajectory shape analysis will clarify.

**Relation to commitment dynamics (§7.8).** The commitment moment hypothesis — that autoregressive generation pressure forces routing decisions to materialize geometrically at generation onset — extends to the reasoning regime. For standard factoid generation, the commitment moment is step-1 of the main generation. For reasoning-distilled models, the commitment moment occurs within the think block, after approximately one fifth of the thinking tokens. The geometry then persists for the remaining elaboration. This is a new temporal regime not characterized in prior results: intra-chain-of-thought commitment dynamics, distinct from the generation-onset step-1 dynamics of standard models.

**Implication for inference efficiency.** If the model's answer direction is committed after ~20% of thinking, the remaining 80% is mostly redundant. An inference system that detects the commit point via the Fisher probe could terminate reasoning early with near-zero quality loss. EXP-B's f1_delta = +0.008 (f1_full − f1_early) means post-commitment elaboration adds a marginal F1 benefit on the minority of questions where early exit would produce a different answer. Elaboration is not counterproductive — it is nearly neutral with a tiny positive contribution on diverging cases. Whether this justifies 80% token overhead is the question EXP-I (forced truncation at commit point) is designed to answer causally.

---

### 5.14 Result 14 — Perturbation Invariance (EXP-J)

**Experiment:** perturbation_battery_v1 (EXP-J). Qwen2.5-1.5B-Instruct, N=50 questions × 4 surface perturbations = 200 pairs total. ICC computed via two-way random effects model.

**Design.** The core question: does the Fisher+PCA64 probe score at step-1, layer 26 reflect something about the question's epistemic content, or is it an artifact of surface form? For each of 50 base questions, four surface-perturbed variants were generated: REPHRASE (same meaning, different wording), LOWERCASE (all lowercase), APPEND (appended irrelevant preamble), TYPO (one random character swap per word). The same bilateral oracle probe calibrated on original items was applied to all variants without re-calibration. ICC measures between-question consistency across the four perturbation conditions.

**Results:**

| Metric | Value | Interpretation |
|---|---|---|
| ICC (two-way random) | **0.913** | Near-perfect agreement across perturbations |
| Between-question variance | 2.737 | True epistemic differences between questions |
| Within-question variance | 0.260 | Surface perturbation noise |
| Between/within ratio | **10.5:1** | Signal overwhelmingly from epistemic content, not surface form |
| PARAM/CTX_DEP separation preserved | t = 23.021 | Oracle labels stable across perturbations |

Per-variant correlations (Fisher score on perturbed variant vs original):

| Perturbation | r |
|---|---|
| REPHRASE | 0.869 |
| LOWERCASE | 0.879 |
| APPEND | 0.830 |
| TYPO | 0.923 |

**Verdict: ROBUST (C025 SUPPORTED — single architecture at time of first experiment).** The Fisher+PCA64 probe measures a property intrinsic to the question's epistemic content, not its surface encoding. An ICC of 0.913 is in the "excellent reliability" range by standard psychometric criteria (ICC > 0.90). The PARAM/CTX_DEP label separation is preserved across all four perturbation conditions (t=23.021), meaning the bilateral oracle categorization is not an artifact of how a question is phrased.

**Methodological significance.** This result closes the most plausible alternative explanation for the main bilateral oracle finding: that the Fisher probe is fitting surface-level lexical features that happen to correlate with knowledge source. The 10.5:1 variance ratio establishes that between-question epistemic differences account for ten times more signal variance than surface form differences — the probe is learning about the question's epistemic content, not its words.

**Llama replication — C025 CONFIRMED (EXP_J_PERTURBATION_BATTERY_V2).** The perturbation invariance result was subsequently replicated on Llama-3.2-3B-Instruct using the same protocol. Results: ICC=0.9334 (two-way random effects), between/within ratio=14.0:1. Per-variant correlations all ≥ 0.91 (REPHRASE, LOWERCASE, APPEND, TYPO). PARAM/CTX_DEP separation preserved across all perturbation conditions (t=20.264, p<0.0001). The between/within ratio of 14.0:1 is higher than Qwen's 10.5:1, confirming that epistemic content — not surface form — dominates probe variance across both architectures. C025 is promoted to **CONFIRMED** on the basis of two independent architectures with ICC ≥ 0.91 and between/within ≥ 10:1 in both cases.

**Relation to C017/C018.** If probe scores reflect epistemic content and not surface form, then the confabulation detection result (CONFIDENT_WRONG items cluster with CTX_DEP) reflects something about the epistemic state of the model on those questions, not an artifact of how CONFIDENT_WRONG questions happen to be worded. This strengthens the geometric interpretation of C017/C018.

---

### 5.15 Result 15 — Cross-Architecture False Certainty Replication (EXP-Llama)

**Experiment:** exp_false_certainty_llama_v1. Llama-3.2-3B-Instruct, same bilateral oracle protocol as EXP-A (Qwen2.5-1.5B-Instruct). N=50 CONFIDENT_CORRECT / 50 CONFIDENT_WRONG items, entropy-matched threshold applied to Llama output distribution.

**Design.** EXP-A established that within the entropy-matched confident zone, Fisher+PCA64 distinguishes CONFIDENT_CORRECT from CONFIDENT_WRONG with AUROC=0.854 (Qwen), adding Δ=0.240 over entropy alone (AUROC=0.614). C017 claimed this gap is a genuine geometric property; C018 claimed bilateral oracle transfer to false certainty. Both claims were initially graded as single-architecture, raising the question of whether the result is Qwen-specific or architecture-invariant.

**Results:**

| Metric | Qwen2.5-1.5B (EXP-A) | Llama-3.2-3B (EXP-Llama) |
|---|---|---|
| Fisher AUROC (false certainty) | 0.854 | **0.818** |
| Entropy AUROC (false certainty) | 0.614 | **0.453** |
| Fisher − Entropy gap | **0.240** | **0.365** |
| BO_Transfer AUROC | 0.880 | **0.768** |

**Verdict: CROSS-ARCH SUPPORTED (C017 and C018 promoted).** The pattern holds and in fact strengthens for Llama: the Fisher gap over entropy is larger (0.365 vs 0.240), confirming that hidden-state geometry adds substantial information over entropy alone for false certainty detection. Bilateral oracle transfer reaches 0.768 — comfortably above chance (0.500) and above the entropy-only baseline (0.453).

**Notable difference.** The absolute Fisher AUROC is slightly lower on Llama (0.818 vs 0.854). This is expected: the Llama bilateral oracle probe is calibrated on the Llama PARAM/CTX_DEP geometry, which is architecturally distinct from Qwen's. The key finding is the additive structure — Fisher adds substantially over entropy in both architectures, with larger absolute and relative gains on Llama.

**C017 promoted to CROSS-ARCH SUPPORTED:** Confabulated confident answers occupy a similar geometric region to CTX_DEP answers in hidden state space — both represent parametric retrieval failure. This holds for Qwen (Δ=0.240) and Llama (Δ=0.365).

**C018 promoted to CROSS-ARCH SUPPORTED:** The bilateral oracle probe transfers to false certainty detection with AUROC > 0.75 on both architectures (Qwen: 0.880, Llama: 0.768), suggesting PARAM/CTX_DEP geometry and confabulation-detection geometry share a common axis.

**Three-task hierarchy implication.** With cross-arch replication of C017/C018, the L2 task (confabulation detection: Fisher essential) is established across both Qwen and Llama families. The three-task hierarchy in the abstract now rests on multi-architecture evidence for all three levels: L1 entropy-dominant (EXP_ENTROPY_BASELINE, both architectures), L2 Fisher-essential (EXP-A + EXP-Llama), L3 Fisher-trajectory for reasoning models (EXP-B, EXP-C, architecture-invariant).

---

### 5.16 Result 16 — Post-Think Entropy Burst in Reasoning Models (EXP-G)

**Experiment:** reasoning_entropy_traj_v1 (EXP-G, v3). DeepSeek-R1-Distill-Qwen-1.5B on GSM8K mathematical reasoning. N=50 CC / 50 CW, 15-step entropy trajectory of answer tokens following `</think>`.

**Design.** The model generates a full reasoning chain (think block), then produces an answer. The experiment measures the entropy of answer tokens at each of the first 15 steps *after* `</think>`. CC items are questions the model answers correctly (numeric exact match on GSM8K gold answer); CW items are questions it answers incorrectly. No entropy pre-filter — CC/CW collected purely on correctness. The scientific prediction was FLAT (think block pre-resolves epistemic state → low, uniform answer entropy regardless of correctness). The alternative was BURST (same pattern as base Qwen, EXP-D).

**Results:**

| Metric | Value |
|---|---|
| Trajectory AUROC (LR, full 15 steps) | **0.8424** |
| Pattern | **BURST** |
| Peak step | 4 |
| Peak AUROC | 0.6932 |
| Step-0 AUROC | 0.3612 |
| Peak − Step-0 | +0.332 (far exceeds BURST threshold of 0.10) |

Per-step AUROC: `[0.361, 0.318, 0.386, 0.467, **0.693**, 0.548, 0.620, 0.658, 0.510, 0.400, 0.411, 0.574, 0.558, 0.603, 0.463]`

Mean entropy trajectories (15 steps post-`</think>`):

| Class | Step 0 | Step 1 | Step 4 | Step 7 |
|---|---|---|---|---|
| CC (correct) | 0.001 | 1.087 | **0.318** | 0.338 |
| CW (wrong) | 0.001 | 1.322 | **0.127** | 0.102 |

**Verdict: BURST. Prediction falsified (C027 SUPPORTED).**

**The pre-resolution hypothesis fails.** The prediction was that explicit chain-of-thought reasoning pre-resolves epistemic uncertainty, making answer entropy uniformly low (FLAT) regardless of correctness. The result is BURST — the same pattern as base Qwen on factual recall (EXP-D) — establishing that the think block does not eliminate the post-answer burst dynamic.

**The collapse mechanism.** The discriminative signal at step 4 is driven by CW dynamics, not CC: CW items spike to high entropy at step 1 (mean=1.322) then collapse rapidly to near-zero by step 4 (mean=0.127). CC items also start high (step 1 mean=1.087) but remain moderately uncertain through step 4 (mean=0.318) and beyond. The peak AUROC reflects this collapse: at step 4, high entropy predicts CC and low entropy predicts CW. Wrong answers exhibit a characteristic spike-then-collapse — a false certainty dynamic occurring within the first four answer tokens.

**Cross-domain and cross-task consistency.** EXP-D (base Qwen, TriviaQA factual recall) also found BURST with peak at step 4. EXP-G (reasoning-distilled Qwen, GSM8K mathematical reasoning) replicates the same pattern with the same peak step. This holds despite: (1) the model having explicit chain-of-thought reasoning, (2) the task being mathematical rather than factual, and (3) the model being a distilled variant of the base architecture. The burst appears to be a stable property of post-commitment answer generation independent of prior reasoning.

**Difficulty confound (C027 precision note).** CC rate was ~11% on GSM8K test with greedy decoding, suggesting the 50 CC items may be systematically easier than the 50 CW items. Part of the entropy trajectory difference could reflect problem difficulty rather than epistemic state. Replication with a higher-accuracy model or temperature sampling would reduce this confound. The cross-domain consistency with EXP-D (where the same backbone family on TriviaQA shows the same peak step) provides indirect evidence that the pattern is not purely a difficulty artifact.

**Relation to §7.8 (commitment dynamics).** The post-`</think>` burst adds a third temporal regime to the commitment dynamics taxonomy: (1) generation-onset step-1 dynamics in standard factual models (ESM, EXP-D), (2) intra-chain commitment dynamics within the think block (EXP-B/C), and (3) post-think answer generation dynamics (EXP-G). All three show burst-type discriminative structure. The think block delays but does not eliminate the burst.

---

### 5.17 Result 17 — Early Exit Causal Validation (EXP-I)

**Experiment:** early_exit_causal_v1 (EXP-I). DeepSeek-R1-Distill-Qwen-1.5B, N=200 questions. Phase 1: Fisher+PCA64 probe calibration (n=40/class, commit threshold=6.7018). Phase 2: dual-run — each question generates both full chain and chain truncated at commit point; F1 compared.

**Design.** EXP-B established that models commit to their answer direction after ~18-24% of thinking tokens (observational). EXP-I is the causal counterpart: it actually truncates generation at the detected commit point and measures whether full post-commitment elaboration recovers quality. Δf1 = f1_full − f1_truncated. If MINIMAL_QUALITY_LOSS: early exit at commit is a viable inference efficiency strategy. If SUBSTANTIAL_QUALITY_LOSS: elaboration is necessary despite being post-commitment.

**Results:**

| Metric | Value |
|---|---|
| n_committed | 199/200 (99.5%) |
| Mean Δf1 | **+0.0059** |
| SD | 0.0474 |
| SE | 0.0034 |
| 95% CI | [−0.0007, +0.0125] |
| p (two-tailed) | 0.08 |
| p (one-tailed) | 0.04 |
| Helped (Δ > 0.05) | 7.54% of items |
| Hurt (Δ < −0.05) | 4.02% of items |
| Mean commit% | **87.4%** |

**Verdict: MINIMAL_QUALITY_LOSS (C028 SUPPORTED).**

**The cost of 87% token savings is +0.006 F1.** Full generation is marginally better on average, but the 95% CI includes zero (borderline significant, p=0.08 two-tailed; p=0.04 one-tailed given EXP-B pre-registered the directional prediction). The running mean stabilized at +0.004 to +0.006 from item 50 through completion — this is not noise, it is a small but consistent directional effect.

**Helped/hurt asymmetry.** On large-effect items (|Δf1| > 0.05), elaboration helped 7.54% and hurt 4.02% — a 1.9:1 ratio. Elaboration is not counterproductive; it has a small asymmetric benefit on the minority of questions where the truncated answer diverges from the full answer.

**Commit rate 99.5%.** One question out of 200 never crossed the commit threshold within the generation budget. The probe is nearly universally applicable as a timing trigger.

**What this proves and what it does not.** EXP-I confirms that the commit point timing is accurate as a detection signal: the answer is effectively decided at the commit point, and forcing truncation there costs near-zero quality. This is the operationally useful result — it says the Fisher probe can serve as an early-exit trigger. What it does not prove is that the probe geometry *causes* the answer. EXP-H (activation patching) already established that the geometry is epiphenomenal at the patching level; EXP-I is consistent with this: the probe detects when the answer is locked in, rather than locking it in. The commit point is a readout, not a control point.

**Inference efficiency implication.** 87.4% of thinking tokens saved at +0.006 F1 cost (p=0.08, borderline). If the commit threshold can be tuned to trade coverage for precision (higher threshold → fewer committed items but tighter detections), the efficiency curve may be more favorable. EXP-I used a fixed threshold (p20 of calibration distribution); threshold sweep is a natural extension.

**Consistency with EXP-B and EXP-C.** EXP-B found mean commit% = 75.8% (Qwen 1.5B, N=200) and f1_delta = +0.008 on diverging cases. EXP-C found 82.9% (Llama 8B, N=100). EXP-I finds 87.4% (same Qwen 1.5B, different protocol measuring overall Δf1 not just diverging cases). All three point to the same phenomenon: reasoning chains in distilled models are 75-90% post-commitment in temporal structure. What occurs in that post-commitment window (elaboration, verification, stabilization) is not yet characterized beyond the timing measurement.

---

### 5.18 Result 18 — Scale Extension: Fisher Signal Improves with Model Scale (EXP-Scale)

**Experiment:** scale_extension_v1 (EXP-Scale). Qwen2.5-7B-Instruct, 4-bit NF4 quantization (T4 VRAM budget). TriviaQA bilateral oracle. Layer 26 (93% depth), step-1 hidden states, Fisher+PCA64 probe. N=50/class (PARAM and CTX_DEP).

**Motivation.** C021 proposed a bilateral oracle Goldilocks zone for base models (0.5B PARAM floor, 3B CTX_DEP ceiling). EXP-E confirmed the zone held for Qwen base models on TriviaQA. Whether instruction-tuned models obey the same ceiling was untested. At 7B, PARAM knowledge is stronger — more questions can be answered without context. Does CTX_DEP viability hold? And if so, does Fisher probe AUROC improve, plateau, or degrade at larger scale?

**Results.** The bilateral oracle remained viable at 7B-Instruct. The collection phase scanned 486 items to accumulate 50 CTX_DEP items — lower yield per item than 1.5B, but the oracle was not blocked. More importantly, the Fisher probe AUROC *improved*:

| Model | Fisher AUROC | Entropy AUROC | Items Scanned |
|---|---|---|---|
| Qwen2.5-1.5B-Instruct | 0.73 | — | ~200 |
| Qwen2.5-7B-Instruct | **0.8402** | **0.9645** | 486 |
| Delta | **+0.1102** | — | — |

Shuffled control: Fisher AUROC = 0.6272 (clean separation from signal). Verdict: AUROC_SURVIVED.

**What the improvement means.** The Fisher probe's ability to separate PARAM from CTX_DEP items strengthens at 7B. This is consistent with a hypothesis that larger instruct models have more geometrically differentiated knowledge representations — the hidden-state geometry becomes more legible, not less, with scale. The entropy AUROC of 0.9645 is near-perfect, suggesting that at 7B, knowledge-source differences produce almost perfectly separable entropy distributions. This is the L1 task (knowledge-source routing) finding its strongest expression yet.

**Falsification of C021 upper ceiling for instruct models.** C021 stated the Goldilocks zone applied to models above ~2B (CTX_DEP ceiling). This was based on base model behavior (Qwen2.5-3B base: CTX_DEP=9/3000 items). EXP-Scale shows the ceiling does not generalize to instruction-tuned models. C021 is updated: the ceiling applies to base models. Instruction-tuned models at 7B retain viable bilateral oracle labeling. C026 (bilateral oracle inapplicable to pure autoregressive base LMs) is reinforced, not contradicted — the distinction is RLHF/instruction tuning, not scale per se.

**Scale trend.** Combining EXP-E (0.5B → 3B base) and EXP-Scale (1.5B → 7B instruct) reveals a consistent pattern: Fisher observability is NOT monotone across the full scale range but shows a clear positive gradient within the instruct model family. The 3B base model ceiling was a base-model artifact. For instruct models: 1.5B → 7B shows +0.11 AUROC improvement.

**Entropy vs Fisher at 7B.** At 7B, entropy dominates Fisher (0.9645 vs 0.8402). This is consistent with the Three-Task Hierarchy: entropy is the primary signal for L1 knowledge-source routing. The 7B model's stronger parametric knowledge creates cleaner entropy separation between PARAM and CTX_DEP items. Fisher adds incremental value (shuffled baseline 0.6272 confirms it contributes over entropy alone) but the gap narrows as entropy becomes near-perfect.

**Limitations.** 4-bit NF4 quantization was required for T4 VRAM (15.3/15.4 GB). The AUROC of 0.8402 is for the quantized model; full-precision 7B may differ slightly. Single-family result (Qwen2.5 only). Replication on Llama-3.1-8B-Instruct would confirm the scale improvement is cross-family. The CTX_DEP collection required 486 items (vs ~200 for 1.5B), meaning harder task sets (HotpotQA) may be needed for efficient oracle construction at larger scales.

---

### 5.19 Result 19 — Training Stage Sweep: C026 Is Pythia-Specific (EXP-L)

**Experiment:** exp_l_stage_sweep_v1 (EXP-L). Qwen2.5-1.5B-Base as Stage 1 (live measurement). Stages 2 and 3 hardcoded from prior program results (Qwen2.5-1.5B-Instruct N=197, DeepSeek-R1-Distill-Qwen-1.5B). TriviaQA bilateral oracle. Layer 26, step-1 hidden states, Fisher+PCA64 probe. Stage 1 N=50/class.

**Motivation.** EXP-K established that Pythia-1.4b base LM produces CTX_DEP=0 at all training checkpoints (C026). This raised the question: is this failure general to base LMs, or specific to Pythia's minimal instruction-following capability? EXP-L tests Qwen2.5-1.5B-Base — a modern base LM pre-trained on much richer data including instruction-style text.

**Results.** Bilateral oracle collection: PARAM=50, CTX_DEP=50 from 232 items scanned. The oracle is **APPLICABLE** on Qwen2.5-1.5B-Base.

| Stage | Model | Fisher AUROC | Shuffled | Source | N/class |
|---|---|---|---|---|---|
| 1 — Base | Qwen2.5-1.5B-Base | 0.8876 | 0.8757 | **live** | 50 |
| 2 — Instruct | Qwen2.5-1.5B-Instruct | 0.7300 | ~0.60 | prior | 197 |
| 3 — Reasoning | DeepSeek-R1-Distill-Qwen-1.5B | 0.7600 | ~0.50 | prior | 50 |

Stage trend: NON_MONOTONE (Base: 0.8876 → Instruct: 0.7300 → Reasoning: 0.7600).

**Critical caveat — shuffled AUROC.** For Stage 1, shuffled AUROC = 0.8757, leaving a net gap of only 0.0119. This means the Fisher probe signal on the base model is essentially at noise level — indistinguishable from shuffled labels at N=50/class. The high raw AUROC (0.8876) reflects that PCA64+LDA at N=50 is borderline territory where the probe may fit spurious structure even with random labels. Whether the base model truly has near-zero Fisher signal for PARAM/CTX_DEP, or whether this is a sample-size artifact requiring N≥150/class to resolve, is ambiguous.

**What is defensible.** Two clean findings emerge from EXP-L:

1. **C026 is Pythia-specific (C030 EXPLORATORY).** Qwen2.5-1.5B-Base passes the bilateral oracle threshold. The failure of Pythia reflects its minimal instruction-following capability (generated text continuation rather than context-based answers), not a universal base-LM property. The capability threshold for oracle applicability lies somewhere between Pythia-1.4b and Qwen2.5-1.5B-Base.

2. **The stage comparison is confounded.** The NON_MONOTONE trend (Base: high-but-near-shuffled → Instruct: 0.73 → Reasoning: 0.76) is not a clean finding because Stage 1 was measured at N=50/class while Stage 2 was measured at N=197/class. The Fisher probe's behavior differs substantially across this N range. A clean stage comparison requires all stages measured at the same N with the same protocol. EXP-L as designed was a feasibility test; a follow-up with N≥150/class at Stage 1 would make the comparison interpretable.

**Entropy AUROC at Stage 1.** Entropy AUROC = 0.7219 on the base model. This is below entropy AUROC on the instruct model (0.9043) and 7B instruct (0.9645), consistent with a pattern where instruction-tuned models produce cleaner entropy separation between PARAM and CTX_DEP items. The entropy signal on the base model is real (0.7219 >> 0.50 shuffled baseline), even if the Fisher signal is ambiguous at this N.

**Implications for the dynamics program.** EXP-L establishes that Qwen2.5-1.5B-Base can serve as a valid Stage 1 measurement point for a training dynamics comparison. The dynamics question (how does Fisher signal change from Base → SFT → RLHF → Reasoning?) can be answered with a clean re-run at N≥150/class. The stage comparison will require collecting more CTX_DEP items at Stage 1 — with 50/class from 232 items, there is headroom; ~400-500 items would likely yield 150/class.

---

### 5.20 Result 20 — Architecture Generalization: Five Architectures + CO Resolution of T2_L2 (C031, C035, C036, C044)

**Experiment:** ood_generalization_v2 and Mistral-specific evaluations.

**L1 bilateral oracle — Gemma-2-2B-IT (C031) and Mistral-7B-Instruct-v0.3 (C035):** Gemma-2-2B-IT reaches AUROC=0.7528 (CI=[0.652, 0.848], shuffled=0.5296 CLEAN, N=200/class, penultimate layer L24, third architecture). Mistral-7B-Instruct-v0.3 was applied under the large-N clean protocol (N=200/class, pool=3795 scanned, TriviaQA, L26 step-1 Fisher+PCA64). AUROC=0.7780 (CI=[0.692, 0.863]), shuffled=0.5524 (CLEAN). Phi-3.5-Mini-Instruct (Microsoft, 3.8B, RoPE+SwiGLU+GQA) achieves the highest L1 AUROC across all five tested architectures at 0.8456 (C044). Mistral-7B at 0.7780 is the fourth, extending bilateral oracle validation to five independent model families. Architecture range: [0.731, 0.846].

| Model | Family | N/class | AUROC | Shuffled | CI 95% | Status |
|---|---|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct | Qwen (GQA) | 197 | 0.7312 | 0.598 | [0.63, 0.83] | CLEAN |
| Llama-3.2-3B-Instruct | Llama (GQA) | 200 | 0.7464 | 0.502 | [0.65, 0.83] | CLEAN |
| Gemma-2-2B-IT | Gemma (MQA, SWA) | 200 | 0.7528 | 0.530 | [0.65, 0.85] | CLEAN |
| Mistral-7B-Instruct-v0.3 | Mistral (sliding window) | 200 | 0.7780 | 0.5524 | [0.692, 0.863] | CLEAN |
| Phi-3.5-Mini-Instruct | Phi (GQA, RoPE) | 200 | **0.8456** | 0.4992 | [0.758, 0.921] | CLEAN |

All five shuffled controls are well below real AUROC. Architecture spread: [0.731, 0.746, 0.753, 0.778, 0.846] — range=0.115 across five independent families (Qwen, Llama, Gemma, Mistral, Phi) spanning attention mechanisms (GQA, MQA, sliding-window, RoPE), parameter scales (1.5B–7B), and two distinct training lineages. Phi-3.5-Mini-Instruct achieves the highest L1 AUROC (0.8456), consistent with its RLHF-trained assertiveness producing strong epistemic geometry at L1 (cf. C042: SFT consistently improves L1 over base). This result supersedes earlier calibration-phase estimates (0.841–0.846, N=128–150/class), which had a dataset-ordering artifact in the Qwen shuffled control. The large_n_v2 protocol is the authoritative result.

**L2 — Mistral T2_L2 failure (part of C035):** Despite achieving the highest L1 AUROC, Mistral-7B-Instruct-v0.3 fails the entropy-matched L2 confabulation detection protocol. Fisher=0.5952, gap=−0.014 (T2_L2 NOT_SUPPORTED). The mechanistic explanation: Mistral-7B has θ_conf=0.122, placing it in a low-entropy regime where the "confident" zone is near-empty — almost all outputs qualify as low-entropy regardless of correctness. The CC/CW entropy difference within this window is only 0.007 nats. However, BO_Transfer=0.6624 — substantially above chance — establishes that the confabulation geometry exists; the measurement framing (entropy-matched CC/CW selection) fails, not the underlying representational structure.

**C036_CONFIRMED:** CO labeling resolves the T2_L2 pattern definitively. EXP_CO_GEMMA_MISTRAL_V1 ran CO labeling (entropy ≤ θ_conf, no window, N=200/class) on both T2_L2 architectures. Gemma CO (θ_conf=0.7562): Fisher=0.8368 CI=[0.749, 0.910], Entropy=0.6311, Gap=+0.2056, shuffled=0.4712 CLEAN — CO_RECOVERS. Mistral CO (θ_conf=0.0220, 8-bit quantization for T4): Fisher=0.8580 CI=[0.777, 0.927], Entropy=0.6466, Gap=+0.2114, shuffled=0.5140 CLEAN — CO_RECOVERS. The underlying confabulation geometry was always present in both architectures. The entropy-matched framing was selecting a degenerate near-empty window (for Gemma: [0.067−0.30, 0.067+0.30] = [0, 0.367]; for Mistral similar), yielding noise rather than signal. CO labeling (entropy ≤ θ_conf, no window) captures the full confident tail and recovers strong Fisher separation comparable to Qwen (0.885). **CO labeling is the universal L2 estimator.** The entropy-matched design is retained as historical reference only.

This is a scope characterization, not a falsification of C017. The confabulation geometry gap is confirmed on Qwen and Llama; Gemma and Mistral represent a well-characterized boundary condition in the measurement protocol's applicability space.

---

### 5.21 Result 21 — Baseline Comparison and RLHF Anti-Correlation (C032, C033)

**Experiment:** CO labeling comparison and behavioral baseline evaluation (Qwen2.5-1.5B-Instruct, TriviaQA).

**CO labeling vs BO labeling (C032):** The entropy-matched bilateral oracle (BO) design requires items to be selected for low output entropy, which creates the T2_L2 boundary condition and limits pool yield. CO labeling (correct vs any incorrect, N=200/class, no entropy restriction) was compared directly to BO (entropy-matched PARAM/CTX_DEP, N=100/class) on the same model and dataset.

| Labeling | AUROC | N/class | Entropy-matched? |
|---|---|---|---|
| BO (entropy-matched PARAM/CTX_DEP) | 0.806 | 100 | Yes |
| CO (correct vs any incorrect) | **0.885** | 200 | No |
| Delta | +0.079 | — | — |

CO achieves 0.885 AUROC — higher than entropy-matched BO (0.806) — and with double the effective N. The reason CO outperforms BO: CTX_DEP items are geometrically closest to PARAM items (borderline cases near the decision boundary), diluting BO signal. CO labels correct items as positive and the full incorrect population (confabulations + CTX_DEP failures) as negative, so the positive class is more geometrically pure and the separation from the negative class is sharper.

**Behavioral baseline comparison (C033):** Fisher+PCA64 was compared against four behavioral baselines on the same N=100/class entropy-matched CC/CW pool:

| Baseline | Method | AUROC |
|---|---|---|
| B1: Output entropy | Per-token entropy at generation step | 0.614 |
| B2: Self-consistency | k=5 majority vote agreement rate | 0.613 |
| B3: Top-1 probability | Max token probability at step 0 | **0.384** |
| **Fisher+PCA64** | L26 step-1 hidden state | **0.845** |

**The anti-correlation finding.** A top-1 probability AUROC of 0.384 means items with *higher* token-level confidence are *less* likely to be correct within the entropy-matched confident zone. This is consistent with RLHF-trained assertiveness producing output confidence that systematically diverges from underlying epistemic state: alignment training that rewards assertive, confident-sounding outputs may produce confabulated wrong answers with more peaked output distributions than correctly-retrieved answers, because the model has been trained to commit confidently even when its parametric retrieval failed. This interpretation is consistent with a structural consequence of post-training alignment on Qwen2.5-1.5B-Instruct; whether it is a universal property of RLHF-trained models or architecture-specific requires cross-family validation (exp_l_stage_sweep_v2 is the primary test, C043 provides supporting evidence).

Self-consistency (B2=0.613) fails for the same structural reason: all k=5 samples are drawn from the same alignment-distorted distribution, so they agree confidently and wrongly at the same rate as single-sample generation.

**Why Fisher corrects this.** Fisher+PCA64 reads the residual stream at layer 26 *before* the output head applies its post-training vocabulary projection. The geometry of the residual stream reflects the epistemic state upstream of the trained output transformation — not the post-alignment surface presentation. This explains the +0.232 gap over the best behavioral baseline: Fisher reads geometry that precedes the output head's trained transformations, while behavioral baselines measure a downstream quantity that may have been distorted away from epistemic state by the training objective.

This result establishes the theoretical rationale for hidden-state probing at L2: not just that it works, but *why* it must work for a class of failures that output-space methods are structurally prevented from detecting.

---

### 5.22 Result 22 — Large-N Cross-Validated L2 Validation (C040)

**Experiment:** EXP_L2_LARGE_N_V1 — l2_large_n_v1, Qwen2.5-1.5B-Instruct, TriviaQA, CO-style labels.

**Motivation.** Prior L2 Fisher AUROC estimates ranged from 0.670 to 0.885 depending on pool section sampled and N/class used (C034). This variance raised the question of whether the L2 Fisher signal was stable or an artifact of specific pool sections. EXP_L2_LARGE_N_V1 was designed as a Tier 0 experiment to resolve this: N=500/class CO-style (θ_conf=0.7733), 5-fold stratified cross-validation, ensuring that the test AUROC estimate is not dependent on any specific pool section.

**Results:**

| Metric | Value |
|---|---|
| CV Fisher AUROC | **0.7629 ± 0.0120** (mean ± std across 5 folds) |
| CV Entropy AUROC | 0.6002 ± 0.0440 |
| CV Gap (Fisher − Entropy) | **0.1628** |
| Verdict | **STABLE_SIGNAL** (mean ≥ 0.75, std ≤ 0.05) |
| Fold range | 0.7440 – 0.7805 (range = 0.0365) |
| Kill criterion triggered | No |

**The prior 0.670–0.885 range was pool-section heterogeneity.** At N=500/class with 5-fold CV (n_test=100/class per fold), the AUROC stabilizes. The fold range (0.0365) is tight relative to the prior pool-to-pool variance, confirming the earlier variance was sampling artifact, not signal instability.

**CV Gap interpretation.** The CV Gap (0.1628) is smaller than the entropy-matched EXP-A gap (0.240) because CO-style collection does not force entropy AUROC → 0.50: the entropy baseline is 0.6002 (not disabled by design). Both measurements are valid, answering different questions. EXP-A quantifies Fisher's advantage when entropy has fully failed; l2_large_n_v1 quantifies Fisher's advantage across the full low-entropy population. Together they bound the Fisher contribution at L2: at minimum +0.163 AUROC over entropy (CO-style), at maximum +0.240 (entropy-matched zone).

**C040 CONFIRMED (5th confirmed claim).** With N=500/class, 5-fold CV, clean shuffled controls at all folds, and consistent gap, this is the most rigorous single estimate of L2 Fisher performance in the program. C040 joins C001–C004 as a CONFIRMED claim.

---

### 5.23 Result 23 — OOD Generalization: Format-Sensitive Portability (C041)

**Experiment:** ood_generalization_v2 — TriviaQA bilateral oracle probe applied without retraining to two OOD tasks.

**Motivation.** The thesis has established that the bilateral oracle Fisher+PCA64 probe achieves AUROC 0.73–0.75 on TriviaQA (L1) and 0.7629 (L2 CO-style). A fundamental question for deployment is: does this geometry transfer to other tasks, or is it calibrated to TriviaQA-specific surface features? The cross-task cosim result (§5.8) established that probe axes are task-specific in geometry. The OOD generalization experiment provides a complementary measurement: not axis alignment, but functional transfer efficiency.

**Design.** A Fisher+PCA64 probe was trained on TriviaQA bilateral oracle labels (N=200/class, L26, Qwen2.5-1.5B-Instruct, source AUROC=0.7744). Without retraining, this probe was applied to items collected under the same bilateral oracle protocol on two OOD tasks:
- **HotpotQA** — open-ended Wikipedia multi-hop QA, same bilateral oracle protocol
- **MMLU-STEM** — multiple-choice, cross-domain factual knowledge

**Results:**

| Task | Transfer AUROC | Within AUROC | Transfer Efficiency | Verdict |
|---|---|---|---|---|
| HotpotQA | 0.6567 | 0.7769 | **84.5%** | OOD_PARTIAL |
| MMLU-STEM | 0.5288 | 0.9488 | 55.7% (~random) | TASK_SPECIFIC |

**Format-sensitive portability.** The pattern is stark. HotpotQA (open-ended Wikipedia multi-hop QA) shares the same question format, answer format, and knowledge domain as TriviaQA; the probe transfers at 84.5% efficiency. MMLU-STEM (multiple-choice, cross-domain) shares none of these format properties; transfer is essentially random (0.5288 ≈ chance, shuffled=0.4496).

This result establishes the correct scope qualification for all L1/L2 observability claims: the bilateral oracle geometry is *format-scoped*, not universally portable. The probe is a task-format-calibrated instrument, not a universal epistemic meter. Per-format recalibration is required for deployment across task types.

**Relation to cross-task cosim (§5.8).** The cosim result showed that Fisher probe axes learned on different tasks are geometrically near-orthogonal (cosims 0.004–0.035). The OOD result adds functional nuance: geometric near-orthogonality does not prevent partial functional transfer within similar formats (84.5% for HotpotQA). The axes are different, but a TriviaQA-calibrated axis still carries useful signal for HotpotQA because both tasks share the same epistemic structure (PARAM items know the answer parametrically; CTX_DEP items require specific context). Format determines whether the epistemic structure is commensurable.

**Independently corroborated.** This result is consistent with the independently published "Geometries of Truth Are Orthogonal" finding (arXiv 2506.08572), which established that truthfulness directions are near-orthogonal across diverse tasks. The present result adds finer structure: near-orthogonality of axes does not preclude partial functional transfer within the same format family.

**Implication for the 2×2 routing architecture.** The prior OOD HotpotQA result (ANSWER-quadrant +38.7pp accuracy, §5.9) used a calibrated threshold system rather than direct probe score transfer. The current result confirms that format-compatible transfer supports this application — HotpotQA's 84.5% transfer efficiency means a TriviaQA-calibrated probe retains useful signal for routing decisions on HotpotQA without recalibration, though with some precision loss. MMLU-based routing would require full recalibration.

---

### 5.24 Result 24 — Training Stage Sweep: L1 INVERTED_U and L2 MONOTONE_RISE (C042, C043)

**Experiment:** exp_l_stage_sweep_v2 — matched-N comparison across three training stages of the Qwen backbone (N=200/class at every stage).

**Motivation.** EXP-L (§5.19) established that Qwen2.5-1.5B-Base supports bilateral oracle labeling, but the stage comparison was confounded by N differences (Stage 1 N=50/class vs Stage 2 N=197/class). exp_l_stage_sweep_v2 is the clean re-run with matched N=200/class at all stages, making the stage deltas interpretable. This is the primary measurement of the A quantity (Accessibility): how does training reshape the computational organization that O and C reflect?

**Three stages measured (all Qwen backbone):**
- **BASE:** Qwen2.5-1.5B-Base (pretraining only, no instruction tuning)
- **INSTRUCT:** Qwen2.5-1.5B-Instruct (SFT + RLHF)
- **REASONING:** DeepSeek-R1-Distill-Qwen-1.5B (reasoning distillation on mathematical/code chain-of-thought)

**L1 results (C042 — INVERTED_U):**

| Stage | L1 AUROC | CI | Verdict |
|---|---|---|---|
| BASE | 0.7396 | [0.638, 0.834] | — |
| INSTRUCT | **0.8028** | [0.706, 0.888] | +0.063 vs BASE |
| REASONING | 0.7252 | [0.620, 0.823] | −0.014 vs BASE; −0.078 vs INSTRUCT |

**INVERTED_U on L1.** SFT/RLHF improves L1 observability (+0.063). Reasoning distillation then reduces L1 below BASE (−0.014 vs BASE, −0.078 vs INSTRUCT). The mechanism: reasoning distillation was trained on mathematical chain-of-thought data (DeepSeek-R1 teacher), which creates a task-distribution mismatch for TriviaQA factual routing. The REASONING model also showed lower PARAM yield during collection (1919 items scanned vs BASE=920, INSTRUCT=1567), consistent with fewer TriviaQA factual items in its parametric memory after distillation on math/code.

The original Law 3 L1 prediction (MONOTONE_RISE from Base→Instruct→Reasoning) is falsified at the L1 level. The correct statement: SFT sharpens L1 epistemic legibility; reasoning distillation on a divergent task distribution partially erases that sharpening.

**L2 results (C043 — MONOTONE_RISE):**

| Stage | L2 Fisher | L2 Entropy | L2 Gap |
|---|---|---|---|
| BASE | 0.6528 | 0.5889 | 0.0639 |
| INSTRUCT | 0.6736 | 0.5776 | 0.0960 |
| REASONING | **0.7584** | 0.6106 | **0.1478** |

**MONOTONE_RISE on L2.** Confabulation detection gap (Fisher − Entropy) grows monotonically through every training stage: BASE=0.064 → INSTRUCT=0.096 → REASONING=0.148. Critically, the L2 entropy baseline is stable (0.577–0.611) — the improvement is in Fisher signal growth, not entropy window artifact. Reasoning distillation actually produces the *largest* confabulation separation, consistent with the C033 finding that assertive training objectives (RLHF, reasoning distillation) produce more geometrically separable false certainty patterns.

**The L1/L2 split finding.** The same training stage (reasoning distillation) simultaneously weakens L1 (task-distribution mismatch reduces TriviaQA factual routing geometry) and strengthens L2 (assertive distillation increases the geometry separating confident-correct from confident-wrong). These are not contradictory: L1 and L2 are measuring different things. L1 observability depends on whether the model's knowledge of TriviaQA factual items is organized in a parametrically-accessible way. L2 observability depends on whether the model's commitment geometry at L26 separates the quality of that parametric access. Reasoning distillation erodes the former (fewer TriviaQA PARAM items) while amplifying the latter (stronger commitment geometry for the items that do qualify).

**Revised Candidate Law 3.** The original Law 3 prediction (MONOTONE_RISE on L1) is partially falsified; L2 monotonicity is a new positive finding. Revised statement: *Post-training (SFT, RLHF) and reasoning distillation affect L1 and L2 observability differently, and L1 effects are task-distribution-dependent. Specifically: (a) L2 confabulation detection gap increases monotonically through all training stages; (b) L1 knowledge-routing observability improves with SFT but may decline with reasoning distillation trained on a different task distribution.*

---

### 5.25 Result 25 — MATH Entry-Point Prediction (C038, C039)

**Experiments:** EXP_MATH_PRM (original, N=30/class) and EXP_MATH_AUROC_V2 (corrected, N=100/class). DeepSeek-R1-Distill-Qwen-1.5B on MATH-500.

**Design.** The MATH entry-point experiment tests whether Fisher+PCA64 at step-1 of the reasoning chain predicts final MATH-500 correctness *before* any chain-of-thought computation begins. This is a distinct L3 question from commitment timing (EXP-B/C, which measured *when* the model commits within the chain): the question here is whether the model's approach commitment at the first think-token carries accuracy-predictive signal. Two signals were evaluated:

*J_know (step-level PRM proxy):* J_know at each reasoning step k, measured as the Fisher decision score on the reasoning-step hidden state, was compared against an oracle process reward model (PRM) label. If the hidden geometry tracked reasoning quality step-by-step, Pearson correlation with oracle PRM scores should be significantly positive.

*Entry-point prediction (step-1 AUROC):* Fisher+PCA64 at step-1 of the think block (layers 25–26) was evaluated as a binary predictor of final MATH-500 correctness (CC = correct final answer, CW = incorrect final answer).

**C038 result — PRM_SIGNAL_WEAK:** J_know at reasoning step k does NOT correlate with oracle PRM (Pearson=0.053). The hidden geometry at individual reasoning steps is not a reliable step-level process tracker. The free zero-label PRM hypothesis is falsified. **The entry-point signal (C039) is what carries predictive value, not the step-level trajectory.**

**C039 result — MATH entry-point prediction (corrected):**

| Metric | Original (N=30/class) | Corrected (N=100/class, EXP_MATH_AUROC_V2) |
|---|---|---|
| Fisher AUROC (step-1, L25–L26) | 0.9111 | **0.8558** |
| CI | — | [0.7291, 0.9566] |
| Shuffled | — | 0.574 (CLEAN) |
| Kill criterion triggered | — | No |

**EXP_MATH_AUROC_V2 (N=100/class) confirmed the signal at AUROC=0.8558, CI=[0.7291,0.9566], L25=L26, shuffled=0.574 (CLEAN), kill not triggered. The original N=30/class estimate (0.9111) was within the CI but upward-biased.**

**Comparative standing.** MATH step-1 AUROC=0.8558 remains the highest step-1 AUROC in the program (vs TriviaQA L2 Fisher AUROC=0.854 from EXP-A). The margin is narrow (0.0018) but the ordering holds. This is notable: geometry at the very first reasoning token of a mathematical problem predicts final correctness better than hidden-state confabulation detection on factual QA within the entropy-matched confident zone — even though the MATH experiment is a harder task with a reasoning-distilled model operating on competition mathematics.

**Interpretation.** The result supports the approach-commitment framing: before any chain-of-thought is produced, the model's internal state at step-1 encodes information about whether it will ultimately succeed on the mathematical problem. This is not the same as commitment timing (EXP-B/C measured that 75–83% of thinking tokens are post-commitment); this is about whether the geometry at first token predicts outcome quality. The C022 early-commitment finding and C039 entry-point AUROC together suggest: the answer direction is committed early, and the quality of that commitment is already partially legible in the first hidden state.

**Scope caveat.** Single architecture (DeepSeek-R1-Distill-Qwen-1.5B), single task (MATH-500), N=100/class. Replication on a second reasoning-distilled architecture is required before promoting C039 above SUPPORTED.

---

## 6. The Falsification Record

Scientific rigor requires not only documenting what was found, but what was believed, tested, and corrected. This section records the three confirmed falsifications in this research program. This record is not an admission of error; it is the mechanism by which the current measurement protocol earned its reliability. A reviewer can disagree with conclusions. A reviewer cannot accuse the author of protecting hypotheses when the hypotheses that failed are explicitly named and their failure mechanism is explained. The probe methodology failures documented here (C012, C013, C014) each revealed something about how Fisher LDA fails at small N — and that failure mode, once identified, informed the PCA+shrinkage design that C3-v3 uses. The falsification record is part of the scientific contribution.

### 6.1 C012 — RLHF Attenuation Claim (FALSIFIED)

**Prior claim:** "Instruction tuning (RLHF) attenuates bilateral oracle accessibility by Δ=−0.036 (base AUROC > instruct AUROC)." This claim was supported by a sign test (n=5, p=0.031) across Qwen model families.

**What happened:** Both the base and instruct AUROCs were measured using raw Fisher LDA (no PCA) at n_train=60/class, d=1536. At these dimensions, Fisher LDA's covariance estimate is degenerate — the sample covariance is not full-rank. The shuffled AUROC exceeded the real AUROC for Qwen instruct (C3-v2: shuffled=0.713, real=0.708), which is impossible under a correctly functioning probe. The sign test was built on n=5 measurements from this degenerate estimator.

**Corrected result (C3-v3):** Fisher+PCA64 (N=128/class, Qwen instruct): AUROC=0.841. Qwen base (N=110/class, yield-limited): AUROC=0.714. The direction reverses: instruct > base by 0.127. However, the base sample is smaller (2.4% yield on nocontext F1 ≥ 0.50 for base vs 28% for instruct, because the base model does not follow QA instructions well). The attenuation question remains unresolved at equal N; the prior claim of −0.036 is confirmed false.

### 6.2 C013 — Llama Weak Epistemic Accessibility (FALSIFIED)

**Prior claim:** "Llama-3.2-3B-Instruct has genuinely weaker bilateral oracle accessibility (AUROC ≈ 0.629) than Qwen2.5-1.5B-Instruct (AUROC ≈ 0.899)." This was cited as evidence for architecture-specific epistemic organization quality.

**What happened:** The Llama 0.629 result used raw Fisher LDA without PCA in d=3072. This is the same Fisher degenerate covariance failure as C012, but worse: d=3072 is 2× larger than Qwen's d=1536, making the problem even more severe at small N.

**Corrected result:** Fisher+PCA64 (N=150/class): Llama = 0.846. Difference from Qwen 0.841 = 0.005. Architecture consistency is confirmed (C003, CONFIRMED). The "Llama is weaker" narrative was entirely a probe artifact.

### 6.3 C014 — Nonlinear Probe Recovery (FALSIFIED)

**Prior claim:** "Nonlinear probes (SVM-RBF, MLP) recover additional bilateral oracle signal above linear probes by Δ > 0.05, suggesting nonlinear structure."

**What happened:** C3-v1 and v2 both showed apparent nonlinear recovery, but both had probe failures. C3-v1 used wrong labels (PARAM vs WRONG, not bilateral oracle). C3-v2 used bilateral oracle labels but degenerate Fisher (as above), making the "linear baseline" artificially low. Nonlinear probes can overfit at small N; the apparent Δ > 0.05 was overfitting above a corrupted baseline.

**Corrected result:** C3-v3 with Fisher+PCA64 as the clean linear baseline: Qwen instruct best nonlinear = 0.821 (Δ = −0.019), Llama instruct best nonlinear = 0.841 (Δ = −0.005). NO_RECOVERY confirmed. The bilateral oracle signal is linearly organized (C002, CONFIRMED).

### 6.4 C008 — Fisher ⊥ Entropy (FALSIFIED)

**Prior claim:** "The Pearson correlation between the Fisher J-score at layer 26 and vocabulary entropy is r = 0.0039 across n=800 bilaterally labeled TriviaQA samples. Fisher and output entropy are structurally independent."

**What happened:** The correlation was measured on J-score — a composite aggregate measure combining Fisher probability with trajectory components — rather than on individual item Fisher LDA decision scores vs.\ output entropy. When measured correctly on the scalar LDA decision score at step-1 for each item against its output token entropy:
- Qwen2.5-1.5B-Instruct: r = −0.225 (r² = 0.05, p<0.05)
- Llama-3.2-3B-Instruct: r = −0.544 (r² = 0.30, p<0.0001)

Fisher and entropy are negatively correlated: higher Fisher score (PARAM-like) predicts lower entropy. The magnitude is architecture-dependent — Llama's correlation is 2.4× stronger than Qwen's, suggesting that RLHF and architecture interact with how much the entropy axis and Fisher axis overlap.

**Corrected interpretation.** Fisher is not orthogonal to entropy at L1. The bilateral oracle task (PARAM vs CTX_DEP) is primarily a confident/uncertain distinction, and Fisher's L1 AUROC reflects this same axis (entropy AUROC 0.9043 ≥ Fisher AUROC 0.6566 on Qwen). The r=0.0039 result reflected a misidentification of the measurement variable. The correct statement: Fisher and entropy are **redundant at L1** and **complementary at L2** (Fisher adds 0.240–0.365 AUROC in the entropy-matched confident zone where entropy AUROC ≈ 0.50).

**Lesson.** Composite scores (J-score, combined probes) can mask internal correlations. Always test components individually against potential confounds before claiming independence.

### 6.5 Within-Generation Decay (Revised, Not Fully Falsified)

**Prior claim:** "AUROC decays after step-1 in generation."

**What happened:** The step-index v2 experiment (no EOS filter) showed step-0=0.781, step-1=0.609, step-5+=null. This was not a decay of signal but an EOS artifact: PARAM items (short-answer TriviaQA) terminate generation at step 2–4. As generation steps increase, the PARAM class loses nearly all its items, making the Fisher probe unstable.

**Revised result:** With EOS filter (min_gen=8 tokens), MONOTONE_RISE for verbose items: step-0=0.639 → step-10=0.906. The "step-1 is privileged" claim is revised to apply specifically to the generation-onset vs prefill comparison (0.785 vs 0.567), not to within-generation dynamics.

---

### 6.6 Scope of the Science

This program measures *computational organization* — the geometric structure of residual stream representations as shaped by training, accessible through probing, and quantified by the O, C, A framework. Stating explicitly what the program does not attempt prevents the most common class of reviewer misreading.

**The program does not attempt to:**

1. *Explain circuits or mechanisms.* Fisher+PCA64 reads a geometric property of the residual stream; it does not identify which attention heads, MLP neurons, or superposition features produce that geometry. Activation patching (C005, C024) establishes that the geometry is non-causal under centroid-direction, full-residual-stream patching at the tested layers and magnitudes — but this is a property of the tested intervention, not a claim that the signal has no mechanistic basis. Head-level patching, SAE feature patching, and circuit-level intervention remain untested.

2. *Provide a theory of organization.* The program establishes that optimization produces measurable regularities in O, C, A. It does not explain why. Four competing mechanistic theories remain undiscriminated (Information Bottleneck, Routing Optimization, Predictive Coding, Architectural Determination). The transition from measurement science to explanatory science is the next major open question.

3. *Measure ground-truth epistemic state.* The bilateral oracle labels PARAM/CTX_DEP through behavioral intervention — two separate inference passes with explicitly operationalized criteria. It does not access "what the model knows" in any theory-independent or introspective sense. O measures recoverable information through probes under a specified intervention protocol.

4. *Generalize across architectural paradigms.* All results are on decoder-only transformer architectures (Qwen, Llama, Gemma, Mistral, Phi). Whether O, C, A are measurable in SSMs (Mamba, RWKV), hybrid architectures, or latent diffusion models is entirely untested. The candidate laws are stated for "instruction-tuned transformer decoders within the Goldilocks capability zone."

5. *Generalize across task formats.* The bilateral oracle probe transfers within similar task formats (TriviaQA→HotpotQA, 84.5% efficiency) but fails across format boundaries (TriviaQA→MMLU, ~random; C041). All observability claims are format-scoped.

6. *Provide a deployment API.* The three-gate architecture is a safety framing. End-to-end deployment evaluation — latency under production load, false positive rates at scale, integration with retrieval systems — is outside the scope of this thesis.

**What the program does establish.** That O, C, A are measurable, reproducible, non-trivially nonzero, architecture-consistent within the tested transformer decoder class, training-stage-dependent in predictable directions, and format-scoped. The introduction of three measurable quantities — independent of any particular estimator — is the scientific contribution.

---

## 7. Discussion

### 7.1 Legibility vs. Accessibility: What the Bilateral Oracle Actually Measures

The framing of this thesis has evolved during the research program. The initial framing — "epistemic accessibility" — implied the question was "does the model have access to this knowledge?" That turns out to be the wrong question, or at least a less precise one. The sharper question is: **is the model's routing decision between knowledge sources externally legible from the residual stream at the moment of generation onset?**

These are different questions. Accessibility is a property of knowledge: it is encoded or it is not. Legibility is a property of the decision signal: whether the routing choice (parametric or contextual) is geometrically readable by an external observer. A model could have accessible parametric knowledge that is not legible if the routing signal is not linearly separable. A model could have legible routing signals even when the parametric answer is wrong (confabulation lands in the PARAM region of the Fisher geometry, as the dimensionality results show). The bilateral oracle measures legibility of the routing decision, not validity of the answer.

This reframing matters for three specific results. First, the patching result: the bilateral oracle geometry shows no causal leverage because it is reading out the routing state, not controlling it. That is consistent with a legibility measurement. Second, the MMLU result: j-score is anti-predictive on MMLU (AUROC=0.104) because MMLU forces parametric commitment regardless of correctness — the routing decision is consistently "parametric" across PARAM and CONFAB items, so the legibility signal cannot separate them. Third, the HotpotQA +38.7pp result: this works because confident parametric routing (high legibility score + high margin) is predictive of accuracy on multi-hop items that happen to be within the model's parametric distribution.

**Commitment geometry vs. knowledge validity** is the operational form of this distinction.

The Fisher+PCA64 probe measures the direction in residual stream space that maximally separates items the model answers from parametric memory from items it cannot answer without context. This direction captures something about the model's routing orientation at generation onset — a commitment to parametric generation.

However, commitment to parametric generation does not imply that the parametric answer is correct. The dimensionality results (Section 5.7) confirm this: PARAM_HIGH (correct parametric) and CONFAB (wrong, confident) are geometrically similar on the primary Fisher axis (Dim1 AUROC for PARAM_HIGH vs CONFAB = 0.569 — near chance). What distinguishes them is Dim2 (knowledge reliability), which is only partially accessible from step-1 hidden states (full LDA AUROC for PARAM_HIGH vs CONFAB = 0.965, but requires both dimensions).

The 2×2 validation MMLU result makes this concrete: MMLU multiple-choice questions have Fisher j-scores indistinguishable from PARAM (model commits to an answer regardless of correctness), but MMLU j AUROC = 0.104 (anti-predictive). The model is confidently parametric whether it knows the answer or not.

This means the ANSWER quadrant of the routing architecture (+38.7pp on HotpotQA) is correct not because it identifies questions with correct parametric answers, but because it identifies questions where confident parametric commitment is predictive of quality in this specific evaluation domain. For HotpotQA multi-hop, that turns out to be a strong predictor (88.7% accuracy). For MMLU multiple-choice, it is not. The signal is domain-specific (Section 5.8) and task-specific.

### 7.2 The Bimodal Structure: A Finding That Needs Replication

The borderline geometry result (Section 5.5) is worth taking seriously as a hypothesis, even though its evidential status is EXPLORATORY. If the epistemic geometry were a continuous gradient from STRONG_PARAM to STRONG_CTX_DEP, it would suggest that parametric and contextual knowledge access differ only in degree. The preliminary result suggests they may differ in kind.

The middle zone (WEAK_PARAM, BORDERLINE, WEAK_CTX_DEP all clustering near zero, geometrically indistinguishable) corresponds exactly to the bilateral oracle's SKIP zone. If this pattern replicates, it would mean the protocol is not thresholding a continuous latent variable — it is identifying a genuine geometric boundary. Items in the middle zone are not "borderline epistemic" in the sense of falling between two poles on a spectrum; they are in a region that is not organized by the bilateral oracle's Fisher axis at all.

**What this would imply if it replicates:** Parametric knowledge access and context-dependent recovery may involve categorically different computational states, not endpoints of a single spectrum. The routing decision at step-1 is either clearly committed parametrically or clearly committed contextually; items without a clear commitment fall outside the bilateral oracle's legibility signal entirely.

**Why this needs replication before being cited as a finding:** N=60/group is sufficient to observe the pattern but not to rule out sampling noise. A single architecture (Qwen) and a single task (TriviaQA) means the result could be an artifact of either. The immediate replication experiment is large_n_validation v2 with N=200/class; the borderline experiment should be re-run at N=100/group on Llama before treating the bimodal structure as established.

### 7.3 The Patching Result: What It Rules Out and What It Does Not

The centroid-level patching result (Section 5.3) — zero F1 gain at all layers — is significant precisely because it is a correct negative result. A first reaction might be: if the probe so accurately reads out the routing decision (AUROC = 0.93 in calibration), why can't we write in the opposite direction?

The answer lies in what CTX_DEP items actually are. The CTX_DEP items in TriviaQA are questions whose answers are genuinely absent from the model's parametric weights. Patching the residual stream with the "PARAM direction" moves the model's internal state to look geometrically more like a PARAM item, but the parametric knowledge is still absent. There is nothing to retrieve. The geometry changes; the knowledge does not.

This correctly localizes the tool's scope. The bilateral oracle measures **legibility of the routing decision**: whether the model's routing orientation is visible in the residual stream. It does not measure whether the knowledge being routed to actually exists in the weights. These are different quantities. Patching can modify the routing orientation only if the destination knowledge already exists — which is precisely what CTX_DEP items fail.

The deeper question is about causality. Whether the routing orientation is causal (the orientation causes retrieval to be attempted) or consequential (the orientation reflects that retrieval was triggered by something else) cannot be resolved by centroid patching. The test is negative: forcing the orientation does not trigger retrieval. But that is consistent with both causal and consequential interpretations — in both cases, patching without the underlying knowledge produces no F1 gain. Resolving the causal question requires head-level or circuit-level intervention, which has not been done.

### 7.4 Framing as a Mechanistic Test

The framing experiment (Section 5.6) provides what may be the clearest mechanistic evidence for the commitment moment interpretation. The key observation: standard step-1 AUROC = 0.858; answer-framed step-1 AUROC = 0.764 (−0.094); but prefill is unchanged (0.831 → 0.809).

If the step-1 signal were simply a confidence signal reflecting how well the model knows the answer, framing should not affect it — the model's knowledge is the same regardless of the output template. The fact that answer framing degrades step-1 AUROC while leaving prefill unchanged means the signal is specifically about the routing decision at the generation-onset moment, not about the underlying knowledge state.

When "The answer is:" is appended, the first generated token is a content word (the answer) rather than a routing-phase token. The routing decision has already been made implicitly by the prompt; there is no generation-onset routing moment. The Fisher LDA direction, calibrated on natural generation starts, does not align with answer-completion geometry, producing lower AUROC.

Hedge framing (+0.012) behaves like standard generation because the model still makes a genuine knowledge-access decision even under hedging — the routing step is not displaced.

### 7.5 The Product Implication

The converging findings — linear organization, architecture consistency, OOD transfer, epiphenomenal patching, framing-preserved signal — support a specific product framing: **epistemic observability for training-time monitoring.**

The bilateral oracle Fisher probe is a reliable measurement instrument. It tracks a consistent geometric property of the model's residual stream that reflects knowledge-source routing. It is not a post-hoc output analyzer; it reads out the computational state before generation is complete. It works at approximately equal fidelity across architectures (AUROC 0.7312–0.7464, large_n_v2 clean protocol, N=197–200/class; earlier calibration-phase 0.841–0.846 are superseded). It transfers out-of-distribution in terms of the ANSWER-quadrant precision effect (+38.7pp on HotpotQA), though the probe itself requires recalibration per domain.

The natural application is not output filtering — centroid-level patching shows no causal leverage, ruling out direct intervention via this mechanism — but monitoring of how training affects epistemic legibility. If a training run — capability fine-tuning, RLHF, domain adaptation — degrades the Fisher axis separability, this could indicate that the model's knowledge-source routing geometry is being disrupted. This is a form of epistemic transparency monitoring that does not exist in current training pipelines.

### 7.6 The Safety Gap: Confident Confabulation in the ANSWER Quadrant

The 2×2 routing architecture (ANSWER/RETRIEVE/DEFER/ESCALATE) selects ANSWER-quadrant items using two conditions evaluated at generation step-1: high Fisher j-score (probe indicates parametric routing) and high margin (output confidence is high). On HotpotQA, ANSWER-quadrant items achieve 88.7% accuracy, +38.7pp above baseline. This result requires a safety caveat that the benchmark numbers alone do not surface.

Confabulated items — questions where the model generates a plausible but incorrect answer from parametric weights — produce the same step-1 signature as genuinely correct PARAM items. Both have high j-score (the routing orientation is parametric) and high margin (the output distribution is confident). The routing decision at step-1 is "generate from parametric memory"; the specific content of that generation — correct or incorrect — has not yet materialized. The geometry cannot distinguish correct parametric retrieval from confident parametric confabulation before the specific wrong content forms.

The dimensionality results (Section 5.7) confirm this directly: PARAM_HIGH (correct parametric) and CONFAB (wrong, confident) are geometrically indistinguishable on the primary Fisher axis (Dim1 AUROC for PARAM_HIGH vs CONFAB = 0.569 — near chance). The ANSWER quadrant contains both populations.

On TriviaQA and HotpotQA, the CONFAB population is small (the model's high-confidence parametric answers are mostly correct on these factoid benchmarks). On adversarial benchmarks, hallucination-prone domains, or questions outside the model's training distribution, the CONFAB population may dominate the ANSWER quadrant — and the routing architecture will direct those items to the highest-confidence output path with high confidence.

**Gate 3** is the required architectural response. A third routing gate, applied after step-1 classification, must distinguish correct PARAM from confabulated PARAM using generation trajectory rather than the generation-onset scalar. The step-1 signal cannot perform this function because the distinction requires observing specific generated content as it forms (steps 2-10). The commitment dynamics measurement suite (J_velocity trajectory, per-step AUROC evolution) is the natural instrument for Gate 3 design. Without Gate 3, the ANSWER quadrant is safe for high-recall factoid tasks and potentially unsafe for deployment in adversarial or high-stakes settings.

This is not a limitation of the bilateral oracle — it is the precise characterization of what the bilateral oracle measures. A legibility readout for the routing decision does not promise correctness of the routed output. That distinction must be made explicitly in any deployment context.

**Gate 3 is a detector, not an intervenor (updated 2026-07-07 after EXP-F and EXP-H).** Two completed experiments sharpen the Gate 3 concept:

EXP-A established that the Fisher probe detects confabulation within the confident zone with AUROC=0.854 (EXP-A, Qwen2.5-1.5B, N=80/class, entropy-matched, C017). This is the detection capability: Gate 3 can IDENTIFY items likely to be confabulated before the answer is produced.

EXP-H establishes that this same geometry is currently observed as non-causal under centroid-direction, residual-stream patching: patching CW items toward the CC centroid at L26 step-0 produces no F1 improvement at any magnitude (λ ∈ {0, 0.25, 0.50, 1.0, 2.0, 4.0}, max Δ_F1=+0.0004, C024). Under tested interventions (centroid-direction, residual-stream), Gate 3 cannot prevent confabulation — it can only detect it. Note: attention head patching, SAE feature patching, and training-time objectives remain untested and may find causal leverage that centroid patching cannot.

This refines the deployment framing: Gate 3 is a **confabulation detection gate**, not a confabulation prevention gate. An architecture that uses Gate 3 routes flagged items to a fallback (refuse, retrieve, or escalate to a more capable model) rather than correcting them in-place. The distinction between detection and intervention determines the appropriate system design.

**Gate 3 for reasoning models (updated 2026-07-07 after EXP-F).** EXP-B and EXP-C show that reasoning-distilled models commit to their answer direction after approximately 17–25% of their thinking tokens, with the remaining 75–83% occurring post-commitment (consistent with elaboration; verification and stabilization remain alternative interpretations). For these models, Gate 3 does not apply at generation step-1 of the main answer output — the commit point has already occurred within the think block.

EXP-F tests whether commit-point hidden states predict CC vs CW (AUROC ≥ 0.70 = INFORMED; < 0.70 = BLIND). Result: AUROC=0.6500, shuffled=0.4400 (CLEAN). **Verdict: BLIND** (N=40/class, n_test=20, single arch — noisy estimate, C023 EXPLORATORY). The commit-point hidden state at L26, as detected by the adaptive-threshold method in EXP-F, carries some signal (CLEAN, above shuffled) but below the INFORMED threshold. Two additional observations: (1) CC and CW items show virtually identical commit_pct (98.0% vs 97.9%), meaning commit timing provides zero discrimination between the groups. (2) The 0.65 signal derives from hidden-state geometry at the commit location, not from when commitment occurs.

The BLIND verdict is provisional (N=40, one arch). The kill criterion (AUROC ≈ 0.50 on TWO architectures) is not triggered. However, taken together, EXP-F and EXP-H support a consistent picture: the Fisher geometry in the confabulation domain is geometrically real (AUROC=0.854) but not a reliable intra-think-block quality gate at the commit step under the current protocol. Reasoning-model Gate 3 requires either a different probe location (not just commit step), a different architecture (not just centroid-direction LDA), or a larger sample to establish whether the 0.65 signal crosses the threshold.

### 7.7 The Hidden-State / Output-Entropy Relationship: Fisher Is Redundant with Entropy

**Status 2026-07-05 (all experiments complete):** Two experiments characterize the relationship between the Fisher+PCA64 hidden-state probe and output token entropy: (1) the correlation study (EXP_STRUCTURAL_INDEP_V2, both architectures), and (2) the entropy-only AUROC baseline (EXP_ENTROPY_BASELINE, both architectures).

**Correlation result.** C008 is FALSIFIED. The structural independence claim (corr ≈ 0.0039) used the wrong metric (j_score from MSCP, not Fisher+PCA64 decision score). The correct measurement — corr(output entropy, Fisher+PCA64 decision score at L26 step-1) — gives:

| Model | Spearman r | p | r² | PARAM entropy | CTX_DEP entropy |
|---|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct | −0.225 | 0.079 | 0.051 | 0.94 nats | 2.76 nats |
| Llama-3.2-3B-Instruct | −0.544 | <0.0001 | 0.296 | 1.18 nats | 3.45 nats |

Both architectures show negative correlation (low Fisher score ↔ high entropy), with Llama r²≈0.30 meaning ~30% of Fisher variance is shared with output entropy.

**Entropy baseline result.** The entropy-only AUROC comparison directly answers whether Fisher adds discriminative value beyond entropy:

| Model | Fisher AUROC | Entropy AUROC | Combined | Δ(Fisher−Entropy) | N test/class |
|---|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct | 0.6566 | **0.9043** | 0.7929 | **−0.248** | 31 |
| Llama-3.2-3B-Instruct | 0.8601 | 0.8740 | **0.9037** | −0.014 | 38 |

Verdict: **FISHER_REDUNDANT** for both architectures.

For Qwen, entropy alone achieves AUROC=0.9043, far above Fisher (0.6566). The combined predictor (LR on [Fisher, −entropy]) is worse than entropy alone (0.7929 < 0.9043), consistent with Fisher introducing noise that degrades the LR fit. For Llama, entropy and Fisher are nearly tied (0.874 vs 0.860), and the combination achieves a small but real gain (0.9037 vs 0.874, Δ=+0.030). For Llama there is a detectable marginal independent component; for Qwen there is none.

**Why this happens.** The bilateral oracle labels items as PARAM if the model answers correctly without context (nocontext F1 ≥ 0.50), and CTX_DEP if it fails without context but succeeds with context. This operationalizes a fundamentally confident/uncertain distinction: PARAM items are questions the model knows, generating with peaked (low-entropy) output distributions; CTX_DEP items are questions it cannot answer without context, generating with flat (high-entropy) distributions. Output entropy at step-1 is a near-direct measurement of this distinction. The Fisher+PCA64 hidden-state probe at the same step-1 position captures a linear projection of the residual stream — which is informationally close to the output logits at that same computation step. For Qwen, that projection picks up a weaker version of the entropy signal. For Llama, it picks up roughly equivalent signal, with a marginal independent component.

**Architecture difference.** The gap is significant: Qwen Fisher AUROC (0.6566) is far below entropy (0.9043), while Llama Fisher (0.8601) is close to entropy (0.874). This likely reflects differences in how logit computation distributes across the residual stream: for Llama (d=3072, 3B params), the PCA+LDA projection at L26 happens to align more closely with the output logit-forming computation; for Qwen (d=1536, 1.5B params), the alignment is weaker, making Fisher a degraded proxy.

**Implications for the thesis framing.** The bilateral oracle probe (Fisher+PCA64 on hidden states) was not designed to be compared against entropy — it was designed to measure epistemic state from hidden states without relying on the output distribution. What the entropy result establishes is that the bilateral oracle distinction IS primarily an output-confidence distinction. The hidden-state probe is capturing this through the residual stream, which is correlated with but noisier than the output logits directly. For Qwen, the probe is substantially worse than the output signal it is attempting to represent. For Llama, it is approximately equivalent, with a marginal independent component.

This does not invalidate the bilateral oracle methodology. The two-pass labeling protocol produces clean, separable labels that can be used with any predictor — including entropy, Fisher, or their combination. What changes is the interpretation: the bilateral oracle is operationalizing a confident/uncertain distinction, not a hidden-state-specific routing topology invisible in the outputs. The labeling methodology is the contribution; the Fisher probe is the initial measurement instrument, now known to be primarily capturing output confidence through a noisier channel.

**Revised interpretation of Contribution 9 (framing test).** The original interpretation of the framing result was that framing degrading step-1 AUROC "confirms the signal captures a routing decision at generation onset, not an output confidence signal." In light of the entropy result, the more parsimonious account is: forced framing ("The answer is: ") changes the first generated token from the beginning of a factual answer to a non-answer token with uniformly low entropy for all items. This disrupts both the output entropy signal and the hidden-state Fisher signal at step-1, since both are measured at the same token. The framing result confirms temporal specificity to the generation-onset token; it does not distinguish routing-signal from output-confidence explanations.

**For paper use.** C008 should be cited as FALSIFIED with both the correlation and AUROC-comparison results. The entropy finding (C016: AUROC 0.87–0.90) should be reported as the primary strong baseline. The Fisher probe result (C001) should be framed relative to this baseline: Fisher is roughly equivalent for Llama (0.860 vs 0.874), substantially worse for Qwen (0.6566 vs 0.9043), with a marginal combined gain for Llama (0.9037). The claim of a hidden-state-specific epistemic routing geometry independent of output confidence is not supported. The bilateral oracle methodology, the clean labeling protocol, and the output entropy discriminator result are all fully citable.

**Update from EXP-A (C017, C018) — FISHER_REDUNDANT is task-scoped.** The experiments above show entropy ≥ Fisher for the bilateral oracle (PARAM vs CTX_DEP) task. EXP-A shows the opposite holds for a harder within-task variant. Within the entropy-matched confident zone (CONFIDENT_CORRECT vs CONFIDENT_WRONG), Fisher adds 0.240 AUROC beyond entropy (0.854 vs 0.614). The PARAM/CTX_DEP distinction is primarily a confident/uncertain split that entropy nearly saturates; the CC/CW distinction operates inside that confident region where entropy has no discriminative leverage but hidden state geometry does. These are the same instrument applied to different tasks. The revised framing: (1) for knowledge-source routing classification (the bilateral oracle task), entropy is the stronger predictor and Fisher is partially redundant; (2) for confabulation detection within the confident zone (the false certainty task), Fisher carries substantial independent signal that entropy does not. Both findings are empirically supported and are not in conflict — they characterize different task boundaries for the same probe.

### 7.8 Commitment Dynamics: An Unnamed Research Category

The research program has characterized a temporal phenomenon that does not have a name in the current literature. We propose calling it **commitment dynamics**: the temporal structure of epistemic state organization during the generation onset period.

The field currently studies two temporal regimes:
- **Prompt processing** (prefill): how information is encoded from the input into the residual stream
- **Output generation** (full generation): what the model produces as a sequence

What is missing is a systematic study of the narrow window between these two: the moment of generation onset, when the model transitions from processing to committing. The bilateral oracle probe operates precisely in this window. The results characterize its structure:

1. **The commitment moment** (step-1 > prefill): the routing decision is encoded at generation onset, not during prompt processing. Step-1 AUROC = 0.785 mean vs prefill = 0.567 mean across four model families.

2. **Architecture-specific decay patterns**: Llama-3.2-3B shows a COMMITMENT_MOMENT pattern (step-1=0.866 → step-5=0.517). This high-then-fading structure is consistent with a brief, sharp commitment window that gets overwritten by generated content.

3. **MONOTONE_RISE for verbose items**: For EOS-filtered items requiring long answers, AUROC rises through generation (step-0=0.639 → step-10=0.906). This is not inconsistent with the commitment moment framing — it reflects that verbose answers contain epistemic content throughout, not just at step-1.

4. **Framing displaces the commitment moment**: When answer-framing is used ("The answer is:"), step-1 AUROC degrades by −0.094 while prefill is unchanged. The routing event is displaced or eliminated by the forced template.

These four results collectively characterize the temporal organization of the routing decision. They have not been characterized before as a suite; prior probing work treats generation-time hidden states as a single snapshot rather than as a temporal process.

**The deeper question commitment dynamics opens** is mechanistic: why does epistemic resolution localize to generation step-1? A candidate mechanism: step-1 is the first moment the model must generate output under the full autoregressive pressure of the question. During prefill, the model processes tokens without committing to outputs. At step-1, the model must commit. The first-token generation pressure forces the routing decision to materialize as a geometric signature. This mechanism predicts a specific experimental test: models trained with high-confidence output rewards (RLHF) should show attenuated step-1 resolution because they have been trained to commit confidently regardless of epistemic state.

**J_velocity** — the rate of change of the Fisher score across generation steps — is the dynamic complement to the static step-1 measurement. A high J_velocity at step-1 means the model's epistemic commitment is changing rapidly at generation onset; low J_velocity means the commitment is already formed before generation begins. This measurement is categorically different from Fisher AUROC (which measures external legibility at a snapshot) and is the primary instrument for studying commitment dynamics as a process rather than as a static state.

Commitment dynamics is more likely to survive architectural evolution than the specific geometric measurements built on transformer attention patterns. Any sequential system that transitions from input processing to output generation will have a commitment window. The measurements will need recalibration; the question will remain meaningful.

### 7.9 The RLHF Geometry Rotation Hypothesis

**Status: HYPOTHESIS UNDER INVESTIGATION — one architecture pair, preliminary.**

Instruction tuning does not simply preserve the bilateral oracle signal — C3-v3 shows it may enhance it while rotating its geometric location. The corrected C3-v3 result (instruct AUROC = 0.841 vs base AUROC = 0.714) shows instruct > base by +0.127 — the opposite of the falsified C012 claim. The RLHF geometry rotation experiment finds base vs instruct Fisher axis cosine similarity ≈ 0.007: nearly orthogonal.

This is not attenuation. The signal is preserved and enhanced. But it moves to a different geometric location. Instruction tuning reorganizes the bilateral oracle geometry rather than suppressing it.

The mechanistic interpretation (hypothesis): RLHF trains models to produce high-quality outputs regardless of knowledge source, which requires updating the generation-onset routing process. The result is a routing geometry restructured by training — the bilateral oracle picks up the new geometry (AUROC = 0.841) but in a different subspace than the base model.

**What this might mean at scale (hypothesis, not finding):** As RLHF training intensifies — more rounds of preference optimization, more RL reward shaping — the geometric location of the epistemic signal may continue rotating with each training stage, requiring probe recalibration. Whether the signal remains linearly recoverable under very aggressive RLHF is unknown.

The controlled experiment required to promote this to SUPPORTED status: same base model architecture with and without RLHF, bilateral oracle evaluation at N=200/class across five model families. Current evidence: one architecture pair (Qwen2.5-1.5B base vs instruct), unequal N. This is a single data point.

**If geometry rotation replicates:** Current alignment pipelines optimize behavioral alignment (outputs should be helpful, harmless, honest) without monitoring whether epistemic legibility is preserved or disrupted during training. If RLHF consistently reorganizes the geometry, training pipelines would need probe recalibration at each checkpoint to maintain monitoring fidelity. Epistemic legibility preservation becomes a distinct training concern, not derivable from behavioral objectives alone.

Do not describe this as an "emerging mechanism," "pattern," or "finding" until it replicates across five model families. It is a hypothesis.

### 7.10 Competing Theoretical Frameworks

The preceding sections characterize what is measurable. This section asks why it should be measurable — what theoretical prior would predict that bilateral oracle labels are geometrically separable in residual stream representations. This question has not been formally investigated. Five competing hypotheses exist; they make different predictions that can, in principle, be distinguished experimentally.

**Hypothesis A — Compression.** The model learns to compress information efficiently, and PARAM items require less bandwidth than CTX_DEP items because the answer is stored in weights rather than assembled from context. Epistemic legibility is a byproduct of differential compression: parametric retrieval is a low-bandwidth computation, context retrieval is high-bandwidth. *Prediction:* legibility should increase monotonically with model capacity; signal should localize to mid-to-late layers where compression structures form; signal should correlate with output entropy (which it does — FISHER_REDUNDANT is consistent with this hypothesis).

**Hypothesis B — Information Bottleneck.** The forward pass creates a bottleneck that compresses input while preserving task-relevant information (Tishby & Schwartz-Ziv, 2017). For PARAM items, the bottleneck discards context (irrelevant); for CTX_DEP items, it preserves context (necessary). Bilateral oracle geometry is a trace of what the bottleneck is preserving. *Prediction:* the signal should be strongest at the bottleneck layer; it should decrease if the model over-compresses; it should be sensitive to task distribution during training.

**Hypothesis C — Predictive Coding.** The residual stream carries prediction errors between layers. For PARAM items, parametric predictions are accurate, generating low-error residuals. For CTX_DEP items, predictions fail, generating high-error residuals that later layers must correct. *Prediction:* legibility should be distributed across layers proportional to where prediction errors accumulate; early layers should show lower signal; the signal should correlate strongly with output entropy. This prediction is consistent with the current evidence (AUROC rise across layers, FISHER_REDUNDANT).

**Hypothesis D — Energy Minimization.** The model finds stable low-energy configurations (attractor basins) for known answers and high-energy, unstable configurations for unknown ones. Bilateral oracle geometry measures proximity to distinct attractor types. *Prediction:* PARAM items should have more concentrated residual-stream activations; the geometry should be bimodal (consistent with C015, §5.3); patching interventions near basin boundaries should produce nonlinear responses rather than the null result observed in C005.

**Hypothesis E — Routing Optimization.** Training on diverse question types requires the model to route computation differently depending on whether the answer is in weights or requires context integration. This routing decision precedes output generation and produces distinct layer activations. *Prediction:* legibility should peak at the layer where routing materializes; it should degrade under framing manipulations that prevent the routing decision from forming (consistent with the framing result, §5.5); it should be recoverable from the nocontext pass, where routing must rely on parametric memory alone.

**Current evidence and discriminability.** FISHER_REDUNDANT is consistent with Hypotheses A, C, and E. Bimodal geometry (C015) is consistent with D and E. The framing result (§5.5) is most consistent with E. The activation patching null result (C005) is inconsistent with strong D but consistent with all hypotheses if causal structure exists at finer granularity than centroid-direction patching reaches. No single experiment currently discriminates between the five hypotheses.

**The right question is not "which theory is correct?" — it is "which theory explains which observations?"** Nature does not owe us one unified account. A research program that asks "which single theory wins?" will prematurely close off valid partial explanations. Current evidence is compatible with a partitioned account: compression (H-A) explains the entropy-dominant signal for Qwen (FISHER_REDUNDANT, Δ=−0.248); predictive coding (H-C) explains the layerwise accumulation (monotone rise through L26); routing optimization (H-E) explains the framing sensitivity (routing disrupted by answer-framing) and commitment dynamics; energy minimization (H-D) explains the bimodal structure; information bottleneck (H-B) explains the training floor (bottleneck structure present from step512). This is not incoherence — it is the normal state of a science before a unifying principle is found.

**What would discriminate them.** (1) Layer × step surface: does signal localize to a bottleneck layer (H-B predicts yes; H-E predicts monotone rise)? (2) Activation entropy per layer: H-D predicts lower activation entropy for PARAM items across all layers. (3) SAE feature patching: H-E predicts specific routing features transfer PARAM items toward CTX_DEP behavior. (4) Observability scaling law: H-A predicts monotone increase with parameter count; H-D predicts scale-independent bimodal structure.

The science is at the observation stage. A complete program designs experiments that test specific predictions, not experiments that "support the hypothesis" in general. The deeper question underneath all five is: why does optimization produce epistemic legibility at all? SGD does not optimize for the bilateral oracle. That it produces separable geometry as a byproduct is either an invariant of autoregressive training or a contingent property of current architectures. This distinction matters for predicting whether future model architectures will remain epistemically legible.

**Three competing models for what the Fisher axis specifically measures.** The five hypotheses above address why separability exists. A more targeted question is: given that it exists, what is the Fisher axis a proxy for? Three candidate models are currently consistent with the evidence (C010, C017, C018) and cannot yet be discriminated:

- *Model A — Retrieval Quality:* The Fisher score reflects how cleanly parametric retrieval succeeded. High score = knowledge was cleanly accessed; low score = retrieval failed or was ambiguous.
- *Model B — Internal Certainty After Retrieval:* The score reflects the certainty state at generation onset, after retrieval has already occurred. High score = model has resolved what to generate; low score = unresolved. This is distinct from A: certainty could persist even when retrieval quality was poor (e.g., the model confidently confabulates).
- *Model C — Latent-Memory/Generation Consistency:* The score reflects whether the generation-onset trajectory aligns with the model's stored memory trace. High = generation will stay consistent with memory; low = generation will diverge from it.

EXP-F (commit-point hidden state quality) provides the most direct test: if Model A is correct, commit-point states should predict CC vs CW (clean retrieval → correct answer). If AUROC ≈ 0.50, commitment is blind to retrieval quality, which is inconsistent with Model A. All three models make different predictions about the joint distribution of Fisher scores across the four quadrants (PARAM_HIGH, PARAM_LOW, CONFAB, CTX_DEP); systematically labeling items into all four quadrants and mapping Fisher scores against a 2D ground truth (retrieval success × confidence) would provide stronger evidence.

**The search for computational invariants.** Beyond testing hypotheses, the deepest contribution this program could make is identifying what is *conserved* across architectures, training regimes, and model families. Physics locates invariants (energy, momentum, entropy); they are more durable than any specific model. AI currently lacks this. Candidate invariants for epistemic observability: representation stability under intervention, trajectory curvature at generation onset, information flow direction across layers, routing entropy at generation onset. Finding a quantity that is conserved across the five model families tested so far, and across training checkpoints, would be a candidate law — not just a measurement.

### 7.11 The Three-Task Hierarchy

Every experiment in this program addresses one of three progressively harder measurement questions. This structure was not designed in advance — it emerged from the experimental results — but it provides the most coherent organizational frame for all 45 claims.

| Level | Task | What it distinguishes | Best signal | Best AUROC | Key claims |
|---|---|---|---|---|---|
| L1 | Knowledge-source routing | PARAM vs CTX_DEP — does the model know it? | Output entropy | 0.87–0.90 (1.5B); 0.9645 (7B) | C001, C002, C003, C016, C029, C031, C041 |
| L2 | Confabulation detection | CC vs CW — is the model right when it's confident? | Fisher hidden states | 0.854/0.818 (Qwen/Llama); CV AUROC=0.7629±0.0120 (N=500) | C017, C018, C019, C024, C025, C032, C033, C036, C040 |
| L3 | Commitment timing + post-think dynamics | Pre/post-decision within think block; epistemic state in answer phase | Fisher trajectory; entropy trajectory | traj AUROC=0.9947 (commit); 0.8424 (post-think) | C022, C023, C027, C028 |

**Why this hierarchy matters scientifically.** These are not three different research projects. They are three levels of the same measurement question at increasing depth. L1 asks whether knowledge is available; L2 asks whether available knowledge is correctly accessed; L3 asks when the decision to access it crystallizes and whether it persists through the answer phase. Each level requires more signal power than the previous: output entropy saturates L1 (0.87–0.90) but cannot discriminate L2 (0.614); Fisher hidden states solve L2 (0.854) but require temporal trajectory for L3.

**L3 now has two sub-findings.** EXP-B/C established that models commit to their answer direction after 18–24% of thinking tokens (intra-chain commitment). EXP-G (GSM8K) established that the think block does NOT pre-resolve epistemic uncertainty — the same BURST pattern (peak AUROC=0.6932 at step 4) appears after </think> as in the base model without reasoning (EXP-D). EXP-I established that truncating at the commit point costs only +0.006 F1 (87.4% token savings, p=0.08). Together, L3 characterizes the full reasoning chain: early commitment after 18% of tokens, ~80% of tokens are post-commitment (with elaboration, verification, and stabilization uncharacterized), and the epistemic uncertainty signal persists into the answer phase.

**Signal hierarchy explained.** For L1, PARAM and CTX_DEP items differ in output confidence — entropy captures this. Within L2 (the confident zone), entropy cannot discriminate. Fisher hidden states add 0.240–0.365 AUROC. L3 requires temporal dynamics because the commit moment is a trajectory event; and the post-think BURST is a trajectory event in the answer phase, not a static-step measurement. The appropriate instrument at each level: entropy → Fisher → Fisher trajectory.

**Implication for probing methodology.** A probe designed for L1 is not a valid probe for L2 without entropy-matching. A probe designed for L2 may not transfer to L3 without temporal extension. The hierarchy explains why prior probing work that found Fisher does not distinguish confabulation (an L2 task) was operating at the wrong level — those probes were calibrated on L1 data without entropy matching.

**Scale and the hierarchy.** EXP-Scale (7B-Instruct) shows entropy reaches 0.9645 at 7B for the L1 task — near-perfect separability. Fisher improves to 0.8402. The L1/L2 distinction sharpens at scale: the confident zone shrinks (7B has stronger parametric knowledge), but Fisher's discriminative power within that zone likely remains essential. The scale finding (C029) does not change the hierarchy structure — it shows the hierarchy remains valid and sharpens at scale.

### 7.12 The Training Dynamics Arc (C042/C043 — Clean Stage Comparison Complete)

**Status: clean stage comparison complete (exp_l_stage_sweep_v2). Updated 2026-07-10.**

The training dynamics arc hypothesis is now supported by a clean matched-N stage comparison (exp_l_stage_sweep_v2, C042/C043). Three stages of the Qwen backbone were evaluated at N=200/class:

| Stage | L1 AUROC | L2 Fisher | L2 Entropy | L2 Gap | z-score (commit) |
|---|---|---|---|---|---|
| Base LM | 0.7396 | 0.6528 | 0.5889 | 0.0639 | ~7 (base models) |
| SFT/RLHF | **0.8028** | 0.6736 | 0.5776 | 0.0960 | — |
| Reasoning distillation | 0.7252 | **0.7584** | 0.6106 | **0.1478** | z=49.77/679.73 |

**The arc revised:** Epistemic legibility under the bilateral oracle shows divergent behavior at L1 and L2 across training stages. L1 (knowledge-source routing) follows an INVERTED_U: SFT/RLHF sharpens L1 (+0.063), but reasoning distillation on a different task distribution reduces L1 below baseline (−0.014 vs BASE). L2 (confabulation detection) shows MONOTONE_RISE through all stages: every training step increases the Fisher signal gap, with reasoning distillation contributing the largest single increment (+0.052 gap growth from INSTRUCT to REASONING).

**Protocol constraint (C026).** The bilateral oracle protocol requires a minimum instruction-following capability. Pythia-1.4b produces CTX_DEP=0 at all tested training checkpoints (step16k–step143k) — the base LM generates text continuation rather than context-dependent QA responses. EXP-L established that C026 is Pythia-specific: Qwen2.5-1.5B-Base supports bilateral oracle labeling (CTX_DEP=50/232 items scanned). The capability threshold lies between Pythia and Qwen base in instruction-following ability. The clean stage comparison (exp_l_stage_sweep_v2) used Qwen2.5-1.5B-Base at N=200/class, resolving the N confound that plagued the earlier EXP-L estimate.

**What the L1/L2 split means for the program.** The INVERTED_U on L1 and MONOTONE_RISE on L2 are not in conflict — they reflect fundamentally different things being measured. L1 observability (PARAM vs CTX_DEP routing) depends on the model having accessible parametric memory for TriviaQA items. Reasoning distillation on math/code reduces this: the REASONING model required scanning 1919 TriviaQA items to collect 200 PARAM items (vs 920 for BASE and 1567 for INSTRUCT), confirming fewer TriviaQA facts remain parametrically accessible after distillation. L2 observability (CC vs CW geometry) depends on the model producing geometrically distinct hidden states for confident-correct vs confident-wrong outputs. Assertive training (RLHF, reasoning distillation) consistently amplifies this geometry — the same training steps that reduce factual routing legibility increase confabulation geometry separation.

**What remains open.** (1) Cross-family replication: all stage data are from the Qwen backbone; whether the L1/L2 split holds for Llama or Mistral backbones requires testing at N=200/class. (2) Pretraining dynamics: the bilateral oracle on Pythia is inapplicable (C026); an alternative CO-labeling protocol (correct vs incorrect, no context pass) could study confabulation trajectory during pretraining using Pythia checkpoints. (3) The z-amplification pattern (z=7 for base → z=49.77/679.73 for reasoning models) is consistent with reasoning distillation specifically amplifying commitment geometry, though backbone size and teacher identity both confound this interpretation.

**This is Layer 2 science.** The matched-N stage comparison (C042/C043) is the first characterization of how post-training objectives systematically reshape the O quantity (Observability) at both L1 and L2. The finding that L1 and L2 respond differently to the same training stage is itself a scientific result: observability is not a single quantity that training uniformly increases or decreases, but a task-level-specific property that training reshapes in different directions depending on what each task level is measuring.

### 7.13 Negative Results Registry

Negative results are first-class scientific outputs. The following findings close experimental directions and constrain interpretation of all positive claims. They are permanent entries — no future experiment in this program should reopen a closed direction without a specific new mechanism or changed condition.

**1. Activation patching — epiphenomenal at centroid level (C005, C024).**
EXP-P1 (residual stream patching L4–L26) and EXP-H (CC/CW centroid patching) find zero causal leverage. Max Δ_F1=+0.0004 even where the Fisher gap is 4× larger (confabulation vs knowledge-source domain). Root cause: centroid-direction mean patching moves the residual stream mean but does not control the computational pathway that actually determines the answer. The geometry is a reliable readout, not a handle. Future patching work must operate at attention head, SAE feature, or MLP neuron granularity before drawing further conclusions about causal structure. *Centroid-direction patching is a closed direction.*

**2. AFE (Attention Flow Entropy) — dead signal.**
Correlation analysis shows AFE AUROC=0.33 on the bilateral oracle task — below chance. AFE was an early candidate signal before the program converged on Fisher+PCA64. It was never revived after failing. *Do not include AFE in any multi-signal models or future experiments.*

**3. Nonlinear probe recovery — no gain (C014 FALSIFIED).**
Three iterations produced increasingly controlled experiments and net Δ ≤ −0.019 vs linear Fisher+PCA64. The bilateral oracle geometry is linearly organized (C002 CONFIRMED). Root cause: the data manifold separating PARAM from CTX_DEP in PCA64 space is already near-linearly separable; nonlinear probes overfit rather than discover nonlinear structure. *Nonlinear probes are not a viable path to improving AUROC on this measurement task.*

**4. Fisher ⊥ Entropy — wrong variable (C008 FALSIFIED).**
The original r=0.0039 independence result was measured on the J-score composite, not on per-item Fisher scores vs per-item output entropy. Correct measurement: Qwen r=−0.225, Llama r=−0.544 (p<0.0001). Fisher and entropy are negatively correlated with architecture-dependent magnitude. The corrected understanding: Fisher is *redundant with entropy at L1* (where entropy already captures the confident/uncertain distinction at AUROC 0.87–0.90) and *complementary to entropy at L2* (within-confident confabulation detection, where Fisher adds 0.240/0.365 AUROC). The falsification strengthens the case for Fisher at L2 — it is not competing with entropy but operating in a different domain. *The C008 falsification removes a conflation, not a claim.*

**5. RLHF attenuation and Llama weakness — covariance artifacts (C012, C013 FALSIFIED).**
Both early claims traced to Fisher covariance degeneracy at high d (4096), low N (80/class). PCA64 preconditioning resolves both: Qwen AUROC 0.841, Llama AUROC 0.818, Δ=0.023. The true architecture gap is small and within CIs. *Any claim about differential Fisher performance at N<100 without PCA64 should be treated as artifact-suspect until confirmed at N≥150/class with PCA.*

**6. Answer framing degrades signal — H1 rejected.**
Answer-framing templates ("The answer is:") reduce step-1 AUROC 0.858→0.764 (−0.094). H1 (framing reveals the commitment moment) was rejected: the template disrupts the routing decision rather than revealing it. This is consistent with the routing optimization hypothesis (H-E, §7.10) — forced framing prevents the routing decision from forming. *Answer-framing templates are contraindicated for bilateral oracle measurement at step-1.*

**7. Bilateral oracle inapplicable on Pythia base LMs — scope clarified (C026).**
EXP-K: CTX_DEP=0 at all Pythia-1.4b checkpoints (step16k–step143k), 7950+ items scanned per checkpoint. Not a protocol error: Pythia base LMs generate text continuation rather than producing contextual answers. EXP-L resolves the scope: C026 is Pythia-specific. Qwen2.5-1.5B-Base passes the bilateral oracle (CTX_DEP=50/232 items scanned). The capability threshold for oracle applicability lies between Pythia and Qwen base in instruction-following ability. *Do not apply bilateral oracle to pure pretraining base LMs without instruction-following evaluation. Use Qwen2.5-1.5B-Base or later as the minimum viable base model.*

**Cross-cutting pattern.** The negative results trace to three root causes: (1) *measurement at the wrong variable* — C008, C012, C013 all used the wrong unit of analysis and were resolved by correct measurement; (2) *intervention at the wrong granularity* — C005, C024 used centroid mean shifts, too coarse to reach the causally relevant circuit; (3) *protocol boundary undiscovered* — C026 found an applicability condition that was not known in advance. The meta-principle for future experiments: before concluding a negative result is a genuine null, verify that the instrument, intervention level, and protocol applicability conditions are correctly specified. A premature null closes a direction that might have real signal at different granularity.

---

## 8. Limitations

**1. Full bilateral oracle protocol tested on only two transformer families.** C3-v3 ran the complete protocol (bilateral oracle labels + nonlinear recovery test + shuffled control) on Qwen2.5-1.5B-Instruct and Llama-3.2-3B-Instruct only. Both are transformer-family models trained with instruction tuning. ESM v33 extended the step-1 > prefill finding to four architectures (adding Gemma and Mistral) but used a lighter calibration, not the full bilateral oracle. C003 should be read as "two independently trained transformer families," not "architecture-general."

**2. CTX_DEP yield constraints.** The 2.4% CTX_DEP yield on TriviaQA means collecting N=200 CTX_DEP items requires scanning 8,333+ items. Large_n_validation v1 reached only N=121/class (pool=5000). The resulting CI is wide ([0.52, 0.79]), and the central estimate (0.6566) is lower than C3-v3's 0.841. The v2 experiment (pool=10000) is required to settle the true AUROC with tight CIs.

**3. Causal structure tested only at centroid-direction level.** Centroid-direction residual stream patching shows no causal leverage at L4–L26. This rules out a specific class of intervention (mean-direction patching of the full residual stream). It does not rule out causal leverage at the head level (attention head patching), at the feature level (SAE feature patching), or at the circuit level (MLP neuron patching via mechanistic analysis). The C005 claim is explicitly scoped to centroid-direction patching.

**4. OOD transfer tested on two benchmarks.** HotpotQA (+38.7pp) and NQ-Open (+10.2pp) are the only OOD benchmarks evaluated. HotpotQA's strong result and NQ-Open's weaker result are not fully explained by the mechanistic argument in Section 5.9. Systematic evaluation on additional benchmarks (PopQA, 2WikiMultiHopQA, MuSiQue, NaturalQuestions) is required before OOD transfer can be characterized as a general property rather than a benchmark-specific observation.

**5. Task specificity.** Bilateral oracle geometry does not transfer across task formats (Section 5.8, cross-task cosim near noise floor). All claims are scoped to open-ended factoid QA (TriviaQA/HotpotQA). Extension to MMLU, code generation, or other formats requires separate probe calibration.

**6. Quantization sensitivity.** ESM v33 shows that INT8 quantization attenuates the signal (Qwen 7B: float16=0.994, INT8=0.905). Results from INT8-quantized models should be interpreted with this context.

**7. C3-v3 Qwen confound — RESOLVED by large_n_v2.** The C3-v3 Qwen result (0.841, shuffled=0.617 WARN) was produced without dataset shuffling. Large_n_v2 ran with clean sampling protocol (pool=10000, N=197–200/class): Qwen=0.7312 CI=[0.63,0.83], Llama=0.7464 CI=[0.65,0.83]. Both CLEAN. C001 threshold has been revised to ≥ 0.70. The prior 0.82 threshold is superseded.

**8. Training dynamics — stage comparison complete at matched N.** C026 (bilateral oracle inapplicable on base LMs) was found to be Pythia-specific (EXP-L). The clean stage comparison was completed as exp_l_stage_sweep_v2 at N=200/class for all stages, resolving the N confound from EXP-L. Results: INVERTED_U on L1, MONOTONE_RISE on L2 (C042/C043). The within-architecture stage comparison (Qwen backbone) is now interpretable; cross-family replication (Llama or Mistral backbone) remains an open validation.

**9. Scope limited to Qwen backbone for stage comparison.** The clean stage sweep (C042/C043) used Qwen2.5-1.5B-Base → Qwen2.5-1.5B-Instruct → DeepSeek-R1-Distill-Qwen-1.5B. The REASONING stage uses a model with a different distillation teacher (DeepSeek-R1) and different training data (mathematical chain-of-thought) than the BASE and INSTRUCT stages (both Qwen native training). The task-distribution mismatch hypothesis for the INVERTED_U on L1 is the most parsimonious explanation, but alternative explanations (teacher-specific geometry, different training data composition) cannot be excluded from a single-backbone comparison.

**10. EXP-Scale 4-bit quantization.** Qwen2.5-7B-Instruct was tested at 4-bit NF4 quantization due to T4 VRAM constraints. The AUROC of 0.8402 and entropy of 0.9645 are for the quantized model. Whether full-precision 7B would differ is unknown. The improvement over 1.5B (+0.11 AUROC) is large enough that quantization noise is unlikely to fully explain it, but the exact delta is uncertain.

**11. Pre-specified falsification conditions.** The following conditions, if met, would require revision of the corresponding claims: (a) ~~Re-run at N=200/class for Stage 1 shows Fisher signal equally near-shuffled~~ — RESOLVED by exp_l_stage_sweep_v2: matched N=200/class confirms L1 INVERTED_U and L2 MONOTONE_RISE (C042/C043). (b) SAE or attention-head patching produces F1 gain > 0.05 for CTX_DEP or CW items → C005/C024 scope is wrong (causal leverage exists at finer granularity). (c) Replication of EXP-Scale on Llama-3.1-8B-Instruct finds Fisher degrades relative to 3B → C029 is family-specific (Qwen scale improvement), not general. (d) EXP-G difficulty confound (CC rate 11% on GSM8K) shown to fully explain BURST pattern → C027 requires revision. (e) L2 Fisher gap decreases from Base→Instruct at matched N → C043 (MONOTONE_RISE on L2) requires revision. These conditions are stated before those experiments run.

---

## 9. Future Work

### 9.0 Pre-registered Predictions for Remaining Experiments

The following outcome interpretations are pre-specified before results are collected. This prevents post-hoc rewriting of null results as partial support, and limits the degrees of freedom available to confirm-by-relabeling.

**Law 3 Cross-Family (Llama or Phi stage sweep):** (A) INVERTED\_U on L1 + MONOTONE\_RISE on L2 → Law 3 is universal, not Qwen-specific. (B) INVERTED\_U on L1 only → Law 3 partially universal. (C) L1 flat or MONOTONE\_RISE → Law 3 as stated is falsified; a weaker "L2-only" law is salvageable. (D) Both L1 and L2 flat → Law 3 is Qwen-specific; reformulate as architecture-conditional.

**Law 4 Second Architecture (Llama MATH):** (A) AUROC ≥ 0.85 → Law 4 confirmed, candidate law holds. (B) AUROC 0.70–0.85 → Law 4 weakened to ≥ 0.70 universal, highest-on-math claim retracted. (C) AUROC 0.60–0.70 → Law 4 is Qwen-distill-specific; reformulate as architecture-conditional. (D) AUROC < 0.60 → Law 4 falsified for MATH; C039 is architecture-specific only.

**SAE Mechanism Discrimination (Models A/B/C):** (A) SAE features that activate on PARAM items generalize across architectures → supports Model A (Retrieval Quality). (B) SAE features localize to specific layers/heads → supports Model B (Internal Certainty routing). (C) No consistent feature geometry across architectures → supports Model C (Memory-Generation Consistency, no single substrate). (D) SAE achieves substantially higher O than Fisher+PCA64 → supports the estimator-independence claim in Appendix C; Fisher is a lower bound on O, not its best estimate.

---

### 9.1 Immediate Priorities

**Large-N validation v2 (pool=10000) — COMPLETE.** Qwen=0.7312 CI=[0.63,0.83] N=197/class, Llama=0.7464 CI=[0.65,0.83] N=200/class. Both CLEAN (shuffled controls well below real). C001 threshold revised to ≥ 0.70. This is the most carefully controlled bilateral oracle result and supersedes all prior calibration-phase estimates (0.841–0.846).

**All Tier 0 experiments complete (as of 2026-07-10).**

The three Tier 0 experiments that defined the scope and stability of the O/C/A claims are now done:
- **l2_large_n_v1 — COMPLETE (C040 CONFIRMED).** CV Fisher L2 AUROC=0.7629±0.0120. Resolves the prior 0.670–0.885 variance as pool-section heterogeneity.
- **ood_generalization_v2 — COMPLETE (C041 SUPPORTED).** Format-sensitive portability: TriviaQA→HotpotQA=84.5% efficiency; TriviaQA→MMLU-STEM=~random. All observability claims now format-scoped.
- **exp_l_stage_sweep_v2 — COMPLETE (C042/C043 SUPPORTED).** Matched N=200/class. INVERTED_U on L1 (BASE=0.740→INSTRUCT=0.803→REASONING=0.725); MONOTONE_RISE on L2 (gap 0.064→0.096→0.148).

**Active experiment pipeline (GPU queue order).** Updated 2026-07-10.

1. **EXP-F (commit_quality_v1.py) — COMPLETE 2026-07-07.** Commit-point HS quality: AUROC=0.6500, shuffled=0.4400. VERDICT: BLIND. Kill criterion not triggered. See §7.6 and C023.

2. **EXP-H (cc_cw_patching_v1.py) — COMPLETE 2026-07-07.** CC/CW causal patching: max Δ_F1=+0.0004. VERDICT: EPIPHENOMENAL. Kill criterion triggered — centroid patching line closed. See §7.6 and C024.

3. **EXP-J (perturbation_battery_v1.py) — COMPLETE 2026-07-08.** Perturbation invariance: ICC=0.913 (ROBUST). Four variants (REPHRASE, LOWERCASE, APPEND, TYPO), all corr ≥ 0.83. PARAM/CTX_DEP separation preserved (t=23.021, p<0.0001). C025 initial result (Qwen only). arXiv ICC≥0.70 gating requirement MET. See §5.14. **EXP_J_PERTURBATION_BATTERY_V2 (Llama replication): ICC=0.9334, between/within=14.0:1, all corr ≥ 0.91, sep_preserved t=20.264 p<0.0001. C025 CONFIRMED.**

4. **EXP-FALSE-CERTAINTY-LLAMA — COMPLETE 2026-07-08.** Llama-3.2-3B-Instruct replication of C017/C018: Fisher=0.818, entropy=0.453, gap=0.365, BO_Transfer=0.768. C017/C018 promoted to cross-arch. See §5.15.

5. **EXP-K (pythia_sweep_large_n_v4.py) — STOPPED 2026-07-08.** BILATERAL_ORACLE_INAPPLICABLE on Pythia base LM: CTX_DEP=0 at all four checkpoints (step16k–step143k) after scanning 7950+ items. PARAM items found (~60/checkpoint), so model has parametric knowledge, but base LM behavior prevents context-conditional answering. New finding: C026 — bilateral oracle requires instruction-following capability. Training dynamics question (Pillar II) requires a different protocol or model family. See §7.12.

6. **EXP-G (reasoning_entropy_traj_v1.py) — COMPLETE 2026-07-08.** BURST result on GSM8K with DeepSeek-R1-Distill-Qwen-1.5B. Peak AUROC=0.6932 at step 4. Trajectory AUROC=0.8424. CW entropy collapse mechanism. C027 SUPPORTED. §5.16 added.

7. **EXP-I (early_exit_causal_v1.py) — COMPLETE 2026-07-08.** MINIMAL_QUALITY_LOSS. Δf1=+0.0059±0.047, 87.4% savings, p=0.08. C028 SUPPORTED. §5.17 added.

8. **EXP-Scale (scale_extension_v1.py) — COMPLETE 2026-07-09.** AUROC_SURVIVED. Qwen2.5-7B-Instruct Fisher=0.8402, Entropy=0.9645, Δ=+0.1102 vs 1.5B. C021 upper ceiling falsified for instruct models. C029 SUPPORTED. §5.18 added.

9. **EXP-L (exp_l_stage_sweep_v1.py) — COMPLETE 2026-07-09.** C026 is Pythia-specific: Qwen2.5-1.5B-Base ORACLE_APPLICABLE (50/50 CTX_DEP from 232 items). Fisher net gap=0.012 (near-shuffled at N=50/class — ambiguous). Entropy AUROC=0.7219 (clear). Stage comparison NON_MONOTONE but confounded by N. C030 EXPLORATORY. §5.19 added.

10. **l2_large_n_v1 — COMPLETE 2026-07-10.** STABLE_SIGNAL. CV Fisher L2 AUROC=0.7629±0.0120 (N=500/class, 5-fold, CO-style). C040 CONFIRMED. §5.22 added.

11. **ood_generalization_v2 — COMPLETE 2026-07-10.** OOD_PARTIAL. HotpotQA transfer efficiency=84.5%; MMLU-STEM transfer=~random. C041 SUPPORTED. All observability claims format-scoped. §5.23 added.

12. **exp_l_stage_sweep_v2 — COMPLETE 2026-07-10.** INVERTED_U on L1, MONOTONE_RISE on L2. Matched N=200/class. C042, C043 SUPPORTED. Revised Law 3. §5.24 added.

**Theory Update Points (current).** ✓ EXP-F + EXP-H complete: Gate 3 updated (§7.6). ✓ EXP-J + EXP_J_PERTURBATION_BATTERY_V2 complete: C025 CONFIRMED. Llama ICC=0.9334. ✓ EXP-Llama complete: C017/C018 cross-arch. ✓ EXP-K stopped: C026 established. ✓ EXP-G complete: C027. ✓ EXP-I complete: C028. ✓ EXP-Scale complete: C029. ✓ EXP-L complete: C030 EXPLORATORY. ✓ l2_large_n_v1 complete: C040 CONFIRMED. ✓ ood_generalization_v2 complete: C041 SUPPORTED, format-scope established. ✓ exp_l_stage_sweep_v2 complete: C042/C043 SUPPORTED, Law 3 revised. ✓ co_gemma_mistral_v1 complete: C036_CONFIRMED. CO labeling universal L2 estimator. ✓ phi_bilateral_v1 complete: C044 SUPPORTED. Law 1 = 5 architectures [0.731–0.846]. ✓ EXP_TEACHER_INDEPENDENCE_V1 complete: C045 SUPPORTED. Qwen3 commit_pct=99.8%. Law 2 teacher-independent. ✓ EXP_MATH_AUROC_V2 complete: C039 corrected to 0.8558 (N=100/class). Kill not triggered. **Program reaches **45 claims (7 CONFIRMED, 26 SUPPORTED, 8 EXPLORATORY, 4 FALSIFIED)**.

### 9.2 Medium-Term Experiments

**2D Observability Surface (layer × step):** A heatmap of AUROC(layer, generation_step) across L ∈ {0,4,8,12,16,20,22,24,26,27} and step ∈ {0,1,2,5,10} would answer Q3 fully. Where and when in the computation is epistemic legibility maximal?

**Pythia Checkpoint Sweep (redesign required):** Pythia base LMs produce CTX_DEP=0 on TriviaQA (C026), blocking the bilateral oracle at all Pythia checkpoints. The redesign options are: (a) use OLMo-2 instruction-tuned checkpoints at multiple training stages, or (b) use a modified oracle (Correct vs Incorrect, no context pass) that is applicable to base LMs but measures a different scientific question (confabulation trajectory during pretraining vs knowledge-source routing emergence). EXP-L established that Qwen2.5-1.5B-Base supports the bilateral oracle — a single checkpoint data point. For a clean training dynamics curve, instruction-tuned checkpoints at multiple stages (e.g., Qwen2.5-1.5B-Base → SFT checkpoint → RLHF checkpoint → Reasoning checkpoint) would be needed from a model family that publishes intermediate checkpoints. This is the most consequential open experiment in the program.

**Cross-Family Stage Comparison:** exp_l_stage_sweep_v2 established INVERTED_U on L1 and MONOTONE_RISE on L2 for the Qwen backbone (C042/C043). Whether this pattern holds for Llama or Mistral backbones requires running the same matched-N protocol on a different model family with Base, Instruct, and Reasoning-distilled checkpoints. Llama-3.2-3B → Llama-3.2-3B-Instruct → DeepSeek-R1-Distill-Llama-8B (noting the parameter count difference) would provide a first cross-family data point, though the parameter confound needs care.

**Head-Level Patching:** The epiphenomenal result rules out centroid-direction mean patching. Systematic attention head patching — identifying which heads are active on PARAM vs CTX_DEP items and patching those heads specifically — may reveal causal structure invisible at the residual stream mean level.

**Teacher Independence (EXP_TEACHER_INDEPENDENCE_V1 — COMPLETE):** C022 (COMMITTED_EARLY) was supported by two reasoning-distilled models sharing the same teacher (DeepSeek-R1). EXP_TEACHER_INDEPENDENCE_V1 tested a model from a different training lineage: Qwen/Qwen3-1.7B (native reasoning model, not R1-distilled). Results: N=100/100 committed (100% commit rate), mean_commit_pct=99.8% (commits within the first 1–2 think tokens), null_mean=48.7%, z=1.3×10¹⁵, mean_think_len=766, cal_AUROC=0.9268. VERDICT: REPLICATED — more extreme than R1-distill (75.8%/82.9%). The teacher confound in C022 is RESOLVED. Early commitment is NOT R1-specific: Qwen3-native reasoning shows even more extreme early commitment than either R1-distilled model. C045 SUPPORTED. Law 2 (Commitment precedes verbalization) now rests on 3 models from 2 independent training lineages.

### 9.3 The Long-Range Program

**The Three-Layer Structure.** The research program has naturally organized into three distinct layers, each presupposing the previous:

**Layer 1 — Measurement Science:** How do we measure the computational properties of AI systems that have epistemic consequences? The bilateral oracle, Fisher+PCA64, entropy trajectory, and commitment dynamics are Layer 1 contributions. The central question: what is externally observable, and with what instrument? Current status: strong evidence for multiple Layer 1 findings.

**Layer 2 — Laws:** How does observability emerge and what governs it? Does epistemic legibility scale with parameter count, training compute, or architectural choices? Is there an observability scaling law? What invariants are conserved across architectures and training regimes? The training dynamics experiments, OOD generalization test, and cross-architecture validation are Layer 2 investigations. Current status: the training dynamics result (INVERTED_U on L1, MONOTONE_RISE on L2, clean matched-N stage comparison, C042/C043) is the first clean Layer 2 data point. OOD generalization (C041) characterizes format-sensitive portability. Layer 2 is the active frontier — cross-family stage comparison is the immediate priority.

**Layer 3 — Architecture by Design:** How should future AI systems be designed so that important computations are observable by construction? This includes legibility-preserving optimization objectives, training-time epistemic monitoring as a training constraint, and architectural choices that make routing decisions intrinsically transparent. Current status: aspirational. The 5-year horizon.

**Layer 4 — Simulation:** Can the theories developed in Layer 2 predict the outcomes of experiments before they are run? If the compression hypothesis is correct, it should predict the Pythia INVERTED_U shape, the MoE behavior, and the multimodal fragmentation result. If predictive coding is correct, it should predict specific layer-entropy profiles. A theory that cannot predict new experiments is not a scientific model — it is a post-hoc rationalization. Layer 4 is the test of whether the laws discovered in Layer 2 are genuinely explanatory. Current status: not yet possible — Layer 2 is not complete enough to generate predictions.

**Scientific Questions Framework.** The Three-Layer Structure organizes the program by scope. A complementary framing organizes it by the questions each layer answers. Structuring future experiments around explicit questions prevents accumulating results without increasing scientific certainty.

| Question | Layer | Experiments | Status |
|---|---|---|---|
| Q1: What computational properties are observable? | L1 | EXP-A, EXP-B/C, EXP-D/G, EXP-J, EXP-Scale | Strong — multiple CONFIRMED/SUPPORTED claims across two architectures and four scales |
| Q2: Are observable properties causally active or epiphenomenal? | L1/L2 | EXP-P1 (residual patching), EXP-H (CC/CW patching), EXP-F (commit quality) | Partial — centroid patching CLOSED (epiphenomenal); head/feature/circuit granularity untested |
| Q3: How do observable properties emerge during training? | L2 | EXP-K (Pythia, stopped), exp_l_stage_sweep_v2 (clean stage comparison, N=200/class) | Active — INVERTED_U on L1 (C042) and MONOTONE_RISE on L2 (C043) established; cross-family replication pending |
| Q4: Which observable properties are invariant across architectures, tasks, and perturbations? | L2 | EXP-J + EXP_J_V2 (ICC=0.913/0.933), EXP-Llama (cross-arch), EXP-Scale (7B), ood_generalization_v2 (format scope) | Growing — surface perturbation invariance CONFIRMED (C025, two architectures); format-sensitive portability characterized (C041) |

Q1 is substantially addressed. Q2 is partially addressed — centroid granularity is closed, finer granularity is the active open question. Q3 has its first clean data: L1 INVERTED_U and L2 MONOTONE_RISE across three training stages of one backbone; cross-family replication is the immediate next step. Q4 is growing: perturbation invariance confirmed (ICC=0.913/0.933, C025 CONFIRMED, two architectures) and cross-architecture consistency are supported; cross-format portability is now characterized as format-scoped (C041).

Every future experiment should be explicitly assigned to one of Q1–Q4 at design time. An experiment that doesn't advance one of these questions shouldn't be run. This prevents the program from drifting into exploratory measurement without increasing scientific certainty.

**Why training checkpoints matter more than model sizes.** A scaling law across parameter count (EXP-E) confounds optimization, tokenizer design, data distribution, and architecture. Training checkpoints isolate the emergence question: does epistemic legibility develop during training, or is it architecturally present from initialization? The Pythia result (floor ≥ 0.67 from step512, INVERTED_U provisional) is a single architecture data point, but it already suggests that legibility is not a late-emergent property. Prioritizing a large-N Pythia sweep over multi-model parameter scaling gives cleaner scientific signal for Layer 2 theory.

**Perturbation Science as an experiment category.** Every observable should be tested under a standard battery of controlled perturbations before being promoted above Level 2 on the Observability Ladder (§4.7). The perturbation battery for any new observable:
- Paraphrase: same question, different wording — does AUROC hold?
- Prompt order: question before vs. after context — does the signal survive?
- Distractors: add irrelevant context — does entropy still discriminate?
- Multilingual: translate question to French/Spanish/German — does the geometry transfer?
- Retrieval augmentation: add retrieved passages to parametric questions — does PARAM entropy change?
- Adversarial: carefully crafted inputs designed to maximize entropy without changing the epistemic state
The question in every perturbation is: **what remains invariant?** What survives this perturbation suite is not just an observable — it is approaching a computational invariant.

**Confabulation subtype hierarchy.** The upcoming false-certainty experiments treat CONFIDENT_WRONG as a single class. This is too coarse. Wrong answers have distinct computational origins:

```
CONFIDENT_WRONG
├── Memory hallucination     (knows a wrong answer)
├── Fabricated entity        (generates plausible-sounding nonexistent name)
├── Temporal mistake         (correct answer for wrong time period)
├── Reasoning failure        (correct premises, wrong deduction)
├── Arithmetic mistake       (correct setup, wrong calculation)
├── Contradiction            (answer contradicts stated premise)
└── Unsupported completion   (answer has no evidential basis in context or weights)
```

Mixing these subtypes in a single CONFIDENT_WRONG class averages across different computational phenomena. The signal from a reasoning failure and a fabricated entity may live in completely different geometric regions. Future experiments should attempt subtype labeling before pooling, and report whether the classifier generalizes across subtypes.

**Redefining success.** The program should not define success as "finding a breakthrough result" in any single experiment. The correct definition is: **building the most reliable science of computational observability.** Under this framing:
- Individual experiments can fail without threatening the research program
- Negative results (EXP-A Fisher ≈ 0.50) are informative, not catastrophic
- The program survives architectural evolution because the question (what is computationally observable?) is not specific to current transformer designs
- Each experiment distinguishes between plausible scientific models, rather than determining the program's future

The deepest question — whether optimization pressure conflicts with epistemic legibility, and what that means for AI systems designed to remain governable — is a decade-long inquiry. The current work provides the measurement instrument, the clean labeling protocol, and the first set of characterized properties. The research program is not at a conclusion; it is at the start of Layer 2.

**Five-Program Long-Range Structure.** Beyond the four-question framework, the program's twenty-year arc naturally organizes into five scientific programs in sequence. Each program presupposes the previous; jumping ahead produces premature engineering.

*Program I — Measurement Science:* Develop reliable, reproducible instruments for observing computational properties with epistemic consequences. Deliverable: every lab can measure computational observability as a standard metric alongside loss and calibration. The bilateral oracle, Fisher+PCA64, entropy trajectory, and commitment dynamics are Program I contributions. Current status: substantially complete at Layer 1.

*Program II — Observability Laws:* Discover the universal relationships that govern observability. Does legibility emerge predictably during training? Does it scale with parameters, compute, or training regime? Is there a conservation law (some quantity is preserved across architecture changes)? These become candidate laws — not just measurements. EXP-L, the Pythia checkpoint sweep, and cross-architecture validation are Program II experiments. Current status: first data points collected; quantitative laws not yet established.

*Program III — Computational Dynamics:* Shift from measuring states to characterizing processes. How does commitment evolve across a reasoning chain? How does epistemic uncertainty propagate through layers? How does a model's internal state change during generation? This program focuses on trajectory, velocity, and temporal structure of computation — not snapshots. The BURST pattern (EXP-G) and step-index AUROC (monotone rise) are first Program III results. Current status: preliminary.

*Program IV — Training Science:* Determine whether computational observability is an engineerable property. Can training objectives be designed that intentionally shape observability, not merely as a byproduct? This requires knowing from Program II which observables are causal, from Program III which dynamics to target, and then designing auxiliary objectives. Sequence matters: Program IV that jumps ahead of II and III risks optimizing the wrong quantity. Current status: not yet feasible — the target quantity is not identified.

*Program V — Architecture by Design:* Design AI systems whose critical computations are observable by construction. This is not about probing existing architectures — it is about new architectural choices that make epistemic structure intrinsic rather than incidental. Meaningful only after Programs I–IV establish what structure is worth building in. Current status: aspirational — 10-year horizon.

**The central thesis of the program.** Across all five programs, one question persists regardless of architecture changes: *What aspects of AI computation are observable, how do those properties emerge during learning, and can they be deliberately engineered?* That question will remain meaningful whether future systems are transformers, mixtures of experts, recurrent state-space models, multimodal world models, or architectures not yet invented. The bilateral oracle and Fisher probe are instruments for today's transformers; the question is a permanent scientific program.

---

## 10. Conclusion

This thesis has documented a complete first phase of empirical investigation into epistemic legibility in transformer language models, organized around three measurable computational quantities — Observability (O), Commitment (C), and Accessibility (A). The program has produced 25 result sections, **45 claims (7 CONFIRMED, 26 SUPPORTED, 8 EXPLORATORY, 4 FALSIFIED)**, with all Tier 0 experiments complete. The findings are organized by the three-task measurement hierarchy that emerged from the experimental record, and are reported against a formal falsification record that includes four confirmed falsifications. The measurement instrument is built, its failure modes documented, its scope precisely bounded.

**The three-task hierarchy — a summary of what was found.**

*Level 1 — Knowledge-source routing* (§5.1–§5.13, §5.18): the bilateral oracle protocol assigns clean PARAM/CTX_DEP labels via two-pass QA testing. Fisher+PCA64 at L26 step-1 achieves AUROC 0.7312 (Qwen2.5-1.5B-Instruct, N=197) and 0.7464 (Llama-3.2-3B-Instruct, N=200) under clean large-N sampling. Output entropy achieves 0.87–0.90 on the same task — equal to or exceeding Fisher — establishing that L1 is primarily an entropy-capturable confident/uncertain distinction. Fisher is redundant with entropy at L1 (C008 FALSIFIED). At 7B-Instruct, entropy reaches 0.9645 and Fisher 0.8402 (EXP-Scale, C029), showing the signal improves with scale and the Goldilocks upper ceiling applies only to base models. The geometry is linearly organized (C002 CONFIRMED), architecture-consistent (C003 CONFIRMED, Δ=0.015 between Qwen and Llama), bimodal rather than gradient-structured (C015 EXPLORATORY), and surface-invariant (ICC=0.913/0.933 Qwen/Llama, EXP-J + EXP_J_V2, C025 CONFIRMED). Centroid-direction patching produces no causal leverage at L4–L26 (C005 SUPPORTED). OOD transfer to HotpotQA achieves +38.7pp accuracy in the ANSWER quadrant (C004 CONFIRMED).

*Level 2 — Confabulation detection* (§5.10, §5.11, §5.15): within the entropy-matched confident zone, Fisher+PCA64 adds 0.240/0.365 AUROC over entropy on CC vs CW (Qwen/Llama, C017 SUPPORTED, two-arch). The bilateral oracle probe transfers to CC/CW with AUROC=0.880/0.768 — confabulation occupies the same geometric region as CTX_DEP (C018 SUPPORTED, two-arch). Entropy trajectory (15-step, EXP-D) achieves AUROC=0.730 with characteristic inversion at steps 2–3 (C019). Centroid-direction patching at the CC/CW level is also epiphenomenal: max Δ_F1=+0.0004 (EXP-H, C024 SUPPORTED). Gate 3 for reasoning-model confabulation (commit-point hidden states) is currently BLIND at N=40/class (AUROC=0.650, EXP-F, C023 EXPLORATORY) — sample-size limitation vs genuine null is unresolved.

*Level 3 — Commitment timing and post-think dynamics* (§5.13, §5.16, §5.17): reasoning-distilled models (DeepSeek-R1-Distill-Qwen-1.5B and Llama-8B) commit to their answer direction after 18–24% of thinking tokens (commit%=75.8% and 82.9%, z=49.77 and z=679.73 vs shuffled nulls, C022 SUPPORTED, two families). The think block is ~80% post-commitment; whether that content constitutes elaboration, verification, or stabilization is not yet characterized. EXP-G (post-think entropy burst, GSM8K) establishes that the think block does NOT pre-resolve epistemic uncertainty — the same BURST pattern (peak AUROC=0.6932 at step 4, trajectory AUROC=0.8424) appears in the answer phase after </think>, with CW items collapsing from 1.322 to 0.127 entropy by step 4 while CC stays moderate at 0.318 (C027 SUPPORTED). EXP-I (early exit causal) confirms the commit point as a viable inference efficiency trigger: truncating at commit saves 87.4% of thinking tokens at +0.006 F1 cost (p=0.08, 199/200 items commit within budget, C028 SUPPORTED).

**The four falsifications.** C012 (RLHF attenuation Δ=−0.036) and C013 (Llama weakness AUROC=0.629) were both Fisher degenerate covariance artifacts at high d, low N, reversed by PCA64 + lsqr. C014 (nonlinear recovery Δ>0.05) was overfitting above a corrupted baseline. C008 (Fisher ⊥ entropy, r=0.0039) used the wrong measurement variable; correct measurement shows architecture-dependent negative correlation (Qwen r=−0.225, Llama r=−0.544). Each falsification strengthened the instrument.

**Protocol constraints discovered.** C026: the bilateral oracle requires instruction-following capability. Pythia-1.4b produces CTX_DEP=0 at all training checkpoints — base LM behavior prevents context-conditional answering. EXP-L establishes this is Pythia-specific: Qwen2.5-1.5B-Base supports bilateral oracle labeling. Training dynamics research was subsequently completed (exp_l_stage_sweep_v2) at matched N=200/class.

**Tier 0 experiments complete.** All three foundational scope-setting experiments are done: (1) l2_large_n_v1 (C040 CONFIRMED) establishes stable L2 Fisher AUROC=0.7629±0.0120 at N=500/class; (2) ood_generalization_v2 (C041 SUPPORTED) characterizes format-sensitive portability (84.5% efficiency to HotpotQA, ~random to MMLU-STEM), scoping all observability claims to task format; (3) exp_l_stage_sweep_v2 (C042/C043 SUPPORTED) reveals the L1/L2 training split: INVERTED_U on L1 (SFT helps, reasoning distillation on divergent task distribution reduces below base), MONOTONE_RISE on L2 (confabulation detection gap grows 0.064→0.096→0.148 through all stages).

**The safety gap and its implication.** L1 does not determine whether a confident answer is correct. A low-entropy, Fisher-PARAM-region answer can be either CC or CW. Gate 3 (confabulation detection, Fisher essential) targets the safety gap where Gate 1–2 routing is insufficient. Gate 3 is currently a detector (AUROC=0.854/0.818 on Qwen/Llama; stable at CV AUROC=0.7629 in large-N validation), not an intervenor — centroid patching is epiphenomenal. For reasoning models specifically, Gate 3 at the commit point is provisionally BLIND (N=40/class) and requires larger-N validation. **C036_CONFIRMED:** CO labeling recovers L2 for all tested architectures including low-θ_conf ones (Gemma 0.8368, Mistral 0.8580). T2_L2 was a measurement framing artifact.

**What the program has established.** Seven claims are confirmed (C001–C004, C025, C036, C040). A measurement instrument with twenty-six SUPPORTED claims and a falsification record that earned the instrument its reliability. A three-task measurement hierarchy (L1 entropy-dominant, L2 Fisher-essential, L3 Fisher-trajectory) that organizes the measurement landscape across 45 claims. A protocol applicability boundary (instruction-following capability required; format-scoped by C041). A safety gap framing that connects the measurement science to deployment consequences. A stage-effect characterization (C042/C043) showing that training objectives reshape O at L1 and L2 in systematically different directions — the same reasoning distillation step that reduces factual routing geometry amplifies confabulation geometry. A scale finding (7B Fisher 0.8402, entropy 0.9645) showing observability improves with instruction-tuned scale.

**What remains.** The central open questions are (1) causal structure at finer granularity — attention-head patching or SAE feature patching may reveal load-bearing geometry invisible at the centroid level; (2) cross-family stage comparison — replicating the L1/L2 training split on Llama or Mistral backbone; (3) EXP-G difficulty confound — replication of the BURST finding at higher CC accuracy; (4) ~~CO labeling on T2_L2 architectures~~ — **RESOLVED (C036_CONFIRMED):** CO_RECOVERS on both Gemma (0.8368) and Mistral (0.8580); T2_L2 was a framing artifact. The program now stands at the entrance to Layer 2 — the laws that govern why observability exists and what it conserves. The measurement instrument is complete. That is where the decade's work begins.

---

## Appendix A — Claims Registry Summary

| ID | Statement (abbreviated) | Status |
|----|------------------------|--------|
| C001 | Fisher+PCA64 AUROC ≥ 0.70 on bilateral oracle, L26, TriviaQA — large_n_v2: Qwen=0.7312 CI=[0.63,0.83] N=197; Llama=0.7464 CI=[0.65,0.83] N=200. Both CLEAN. (Calibration-phase 0.841–0.846 superseded.) | CONFIRMED |
| C002 | Signal is linearly organized — nonlinear probes Δ ≤ −0.019 | CONFIRMED |
| C003 | Architecture consistency — Qwen=0.7312, Llama=0.7464, Δ=0.015, CIs fully overlap | CONFIRMED |
| C004 | Bilateral oracle labels separable in Fisher+PCA64 space | CONFIRMED |
| C005 | Centroid-direction patching epiphenomenal at L4–L26 | SUPPORTED (Qwen only) |
| C006 | Step-1 > prefill (0.785 vs 0.567); within-gen MONOTONE_RISE | SUPPORTED |
| C007 | Fisher trajectory AUROC=0.9947 (28 J-scores) | SUPPORTED |
| C008 | Entropy ⊥ Fisher (ρ≈0.0039) | FALSIFIED — Qwen r=−0.225 (r²=0.05), Llama r=−0.544 p<0.0001 (r²=0.30); correlated, architecture-dependent magnitude |
| C009 | Task-specific geometry (all cross-task cosims near noise floor) | EXPLORATORY |
| C010 | CTX_DEP ≈ CONFAB geometrically | EXPLORATORY |
| C011 | Epistemic legibility present from step512; INVERTED_U provisional (not monotone) | EXPLORATORY (Pythia-1.4b, n_test=7–9/class) |
| C012 | RLHF attenuation Δ=−0.036 | **FALSIFIED** |
| C013 | Llama genuinely weaker (0.629) | **FALSIFIED** |
| C014 | Nonlinear probe recovery Δ > 0.05 | **FALSIFIED** |
| C015 | Bilateral oracle geometry is bimodal (not continuous gradient) | EXPLORATORY |
| C016 | Output entropy step-1 AUROC 0.87–0.90 for bilateral oracle classification | SUPPORTED (both architectures) |
| C017 | Fisher+PCA64 discriminates CC vs CW AUROC=0.854; entropy AUROC=0.614; gap=0.240 (entropy-matched) | SUPPORTED (Qwen 1.5B, TriviaQA) |
| C018 | BO_Transfer AUROC=0.880 — bilateral oracle probe transfers to CC/CW; confabulation lives on epistemic accessibility axis | SUPPORTED (Qwen 1.5B) |
| C019 | Entropy trajectory (15 steps) distinguishes CC vs CW AUROC=0.730; inversion at step 2-3 (step-0 AUROC=0.331 → step-4 AUROC=0.617) | SUPPORTED (Qwen 1.5B, TriviaQA) |
| C020 | Architecture family predicts bilateral oracle Fisher AUROC better than parameter count; Qwen 1.5B Fisher=0.845 vs Llama 3B Fisher=0.608 (gap=0.237) | SUPPORTED (n=3 qualifying models) |
| C021 | Bilateral oracle Goldilocks zone on TriviaQA: ~1B–2B parameters for BASE models. Upper ceiling does NOT apply to instruction-tuned models: Qwen2.5-7B-Instruct viable (50/50 from 486 items). Ceiling is base-model-specific. | SUPPORTED (updated) |
| C022 | Early commitment across two DeepSeek-R1-Distill families: Qwen 1.5B commit%=75.8% z=49.77; Llama 8B commit%=82.9% z=679.73. Think block ~80% post-commitment. Teacher confound RESOLVED (C045): EXP_TEACHER_INDEPENDENCE_V1 on Qwen3-1.7B yields commit_pct=99.8% (z=1.3×10¹⁵, N=100/100). Early commitment is NOT R1-specific. Law 2 now rests on 3 models from 2 independent training lineages. | SUPPORTED (3 models, 2 lineages) |
| C023 | Commit-point hidden states not reliably distinct for CC vs CW at N=40/class (AUROC=0.650, BLIND); commit timing identical for CC and CW (commit_pct 98.0% vs 97.9%). | EXPLORATORY (single arch, small N) |
| C024 | Centroid-direction patching of CC/CW geometry at L26 step-0 produces no causal F1 improvement (max Δ_F1=+0.0004). EPIPHENOMENAL. Extends C005 to confabulation domain. | SUPPORTED (Qwen only) |
| C025 | Fisher+PCA64 decision scores at L26 step-1 invariant under 4 surface perturbations. Qwen: ICC=0.913 (ROBUST), between_var=2.737, within_var=0.260 (ratio=10.5:1), N=160. Llama: ICC=0.9334 (EXP_J_PERTURBATION_BATTERY_V2), between/within=14.0:1, REPHRASE/LOWERCASE/APPEND/TYPO corr ≥ 0.91, sep_preserved (t=20.264, p<0.0001). CONFIRMED across two architectures. | CONFIRMED |
| C026 | Bilateral oracle requires minimum instruction-following capability. Pythia-1.4b: CTX_DEP=0 at all checkpoints. C026 is Pythia-specific: Qwen2.5-1.5B-Base ORACLE_APPLICABLE (EXP-L). | SUPPORTED (Pythia-specific) |
| C027 | Post-`</think>` entropy BURST in DeepSeek-R1-Distill-Qwen-1.5B on GSM8K: peak AUROC=0.6932 at step 4, trajectory AUROC=0.8424. Think block does NOT pre-resolve epistemic uncertainty. | SUPPORTED (difficulty confound noted) |
| C028 | Truncating at commit point: 87.4% savings, Δf1=+0.006 ± 0.047 (p=0.08). 199/200 items commit within budget. Commit probe = reliable timing trigger. | SUPPORTED |
| C029 | Fisher+PCA64 AUROC improves 1.5B→7B instruct: 0.73→0.8402 (+0.11). Entropy reaches 0.9645 at 7B. C021 upper ceiling falsified for instruct models. | SUPPORTED (single family, 4-bit NF4) |
| C030 | C026 bilateral oracle failure is Pythia-specific. Qwen2.5-1.5B-Base ORACLE_APPLICABLE (50/232 scanned). Fisher net gap=0.012 (near-shuffled at N=50/class — ambiguous). Entropy AUROC=0.7219 (clear). | EXPLORATORY (N=50/class confound) |
| C031 | Gemma-2-2B-IT L1 bilateral oracle replication: AUROC=0.7528 (CI=[0.652, 0.848]), shuffled=0.5296 (CLEAN), N=200/class, penultimate layer (L24). Third independent architecture confirming L1 separability (Qwen=0.731, Llama=0.746, Gemma=0.753). L2 NOT_SUPPORTED (gap=0.022, T2_L2 triggered: theta_conf=0.067 from Gemma-2 softcapping). | SUPPORTED |
| C032 | CO labeling (correct vs any incorrect, N=200/class) achieves AUROC=0.885 vs BO AUROC=0.806 (Qwen2.5-1.5B-Instruct, TriviaQA). Δ=+0.079. CO outperforms BO because CTX_DEP items are geometrically closest to PARAM, diluting BO signal. CO removes entropy-matching dependency. | SUPPORTED (single arch) |
| C033 | Fisher+PCA64 AUROC=0.845 vs best behavioral baseline B2_self_consistency=0.613 (gap=+0.232). B3_top1_prob=0.384 — **below chance** — confabulations have more peaked wrong-token distributions, consistent with RLHF-trained assertiveness decoupling output confidence from epistemic state (Qwen, single architecture). | SUPPORTED (single architecture) |
| C034 | CO and BO labeling give statistically equivalent Fisher L2 AUROC within same pool section (CO=0.670, BO=0.670). Variance 0.670–0.854 was pool-section heterogeneity. RESOLVED by C040. | SUPPORTED (variance explained) |
| C035 | Mistral-7B-Instruct-v0.3 bilateral oracle: L1 AUROC=0.7780 (CI=[0.692, 0.863]), shuffled=0.5524 (CLEAN), N=200/class. (L1 rank: 4th of 5; Phi-3.5-Mini-Instruct C044 has highest at 0.8456.) L2 NOT_SUPPORTED by entropy-matched framing (Fisher=0.5952, gap=−0.014, T2_L2: theta_conf=0.122) — **SUPERSEDED by C036_CONFIRMED:** CO labeling yields Fisher=0.8580, gap=+0.2114 (CO_RECOVERS). | SUPPORTED |
| C036 | **CONFIRMED (C036_CONFIRMED):** CO labeling recovers L2 for both T2_L2 architectures. Gemma CO (θ_conf=0.7562): Fisher=0.8368 CI=[0.749,0.910], Gap=+0.2056, CLEAN. Mistral CO (θ_conf=0.0220): Fisher=0.8580 CI=[0.777,0.927], Gap=+0.2114, CLEAN. T2_L2 = entropy-matched framing artifact (degenerate window). CO labeling is the universal L2 estimator. | CONFIRMED |
| C037 | Early exit causal: n_committed=199/200 (99.5%), Δf1=+0.006 ± 0.047, p=0.08 two-tailed, 87.4% savings. (Same experiment as C028 — parallel claim entry.) | SUPPORTED |
| C038 | PRM_SIGNAL_WEAK: J_know at reasoning step k does NOT correlate with oracle PRM (Pearson=0.053). Hidden geometry is entry-point predictor (C039), not step-level process tracker. Free zero-label PRM hypothesis falsified. | SUPPORTED (single arch, MATH-500) |
| C039 | MATH entry-point prediction: Fisher+PCA64 at step-1, layers 25–26, AUROC=**0.8558 (CI=[0.729,0.957], N=100/class)** on MATH-500 (DeepSeek-R1-Distill-Qwen-1.5B). **Corrected from original 0.9111 (N=30/class, upward-biased small-N estimate; original was within CI but inflated).** Highest step-1 AUROC in the program (0.8558 vs TriviaQA L2=0.854). Geometry at first reasoning token predicts final MATH correctness before any CoT computation. EXP_MATH_AUROC_V2 (N=100/class): AUROC=0.8558, CI=[0.7291,0.9566], L25=L26, shuffled=0.574 (CLEAN), kill not triggered. | SUPPORTED (single arch, corrected N=100/class) |
| C040 | EXP_L2_LARGE_N_V1 STABLE_SIGNAL: CV Fisher L2 AUROC=0.7629±0.0120 (N=500/class CO-style, 5-fold, Qwen2.5-1.5B-Instruct, TriviaQA). CV Gap=0.1628. Fold range 0.7440–0.7805. Resolves C034: prior 0.670–0.885 was pool-section heterogeneity. | **CONFIRMED** |
| C041 | OOD_PARTIAL. TriviaQA bilateral oracle probe shows format-sensitive portability. HotpotQA transfer efficiency=84.5% (transfer AUROC=0.6567, within=0.7769). MMLU-STEM transfer=~random (0.5288). All L1/L2 observability claims are format-scoped. | SUPPORTED (single model, two OOD tasks) |
| C042 | exp_l_stage_sweep_v2 INVERTED_U on L1 (Qwen backbone, matched N=200/class). BASE=0.7396, INSTRUCT=0.8028, REASONING=0.7252. SFT improves L1 (+0.063 vs BASE); reasoning distillation reduces L1 below BASE (−0.014). Task-distribution mismatch interpretation. Original Law 3 L1 prediction (MONOTONE_RISE) falsified. | SUPPORTED (single architecture family) |
| C043 | exp_l_stage_sweep_v2 MONOTONE_RISE on L2 (Qwen backbone, matched N=200/class). L2 Gap: BASE=0.0639, INSTRUCT=0.0960, REASONING=0.1478. Confabulation detection strengthens monotonically through every training stage including reasoning distillation. L2 entropy baseline stable (0.577–0.611) — improvement is Fisher signal growth. | SUPPORTED (single architecture family) |
| C044 | Phi-3.5-Mini-Instruct bilateral oracle (EXP_PHI_BILATERAL_V1, 2026-07-11): L1 AUROC=0.8456, CI=[0.758,0.921], shuffled=0.4992 CLEAN, N=200/class, 2268 scanned, probe at L30/32. Highest L1 AUROC across five tested architectures. L2: Fisher=0.7120, CI=[0.552,0.843], Entropy=0.5904, Gap=0.1216, BO_Transfer=0.7920, shuffled=0.5040 CLEAN — SUPPORTED. Termination=NONE. Five-architecture range: [0.731, 0.846]. | SUPPORTED |
| C045 | EXP_TEACHER_INDEPENDENCE_V1 (Qwen/Qwen3-1.7B, 2026-07-11): N=100/100 committed (100% commit rate), mean_commit_pct=99.8% (commits within first 1-2 think tokens), null_mean=48.7%, z=1.3×10¹⁵, mean_think_len=766, cal_AUROC=0.9268. REPLICATED — more extreme than R1-distill (75.8%/82.9%). Teacher confound in C022 RESOLVED: early commitment is NOT R1-specific. Law 2 (Commitment precedes verbalization) now supported across 3 models from 2 independent training lineages (R1-Qwen, R1-Llama, Qwen3-native). | SUPPORTED |

---

## Appendix B — Experiment Registry Summary

| ID | Title | Status |
|----|-------|--------|
| EXP_C3V1 | Nonlinear probe recovery v1 — wrong labels | SUPERSEDED |
| EXP_C3V2 | Nonlinear probe recovery v2 — Fisher degenerate | SUPERSEDED |
| EXP_C3V3 | Nonlinear probe recovery v3 — Fisher+PCA64 fixed | COMPLETE |
| EXP_P1V3 | Activation patching main experiment | COMPLETE |
| EXP_P1V5 | Activation patching layer sweep | COMPLETE |
| EXP_P1_CALIB | Probe calibration (N=80/class) | COMPLETE |
| EXP_ESM_V33 | Cross-model step ablation (4 models) | COMPLETE |
| EXP_T1D_STEP_INDEX | Step-index AUROC v2 (EOS confound) | SUPERSEDED |
| EXP_T1D_STEP_INDEX_V3 | Step-index AUROC v3 (EOS filter, MONOTONE_RISE) | COMPLETE |
| EXP_T1C_BORDERLINE | Borderline population geometry v4 (BIMODAL) | COMPLETE |
| EXP_T1A_LARGE_N_V1 | Large-N validation v1 (partial, N=121) | COMPLETE |
| EXP_T1A_LARGE_N_V2 | Large-N validation v2 (pool=10000): Qwen=0.7312 CI=[0.63,0.83] N=197, Llama=0.7464 CI=[0.65,0.83] N=200. Both CLEAN. | COMPLETE |
| EXP_T2A_2D_SURFACE | 2D observability surface (layer × step) | PENDING |
| EXP_T3A_PYTHIA | Pythia checkpoint sweep (8 checkpoints, N=30–40/class, INVERTED_U provisional) | COMPLETE (small-N) |
| EXP_K_PYTHIA_LARGE_N_V4 | Pythia large-N sweep — bilateral oracle inapplicable on base LM: CTX_DEP=0 at all checkpoints (step16k–143k). New finding: C026. | STOPPED — bilateral oracle inapplicable on base LMs |
| EXP_2X2V1 | 2×2 routing validation v1 | COMPLETE |
| EXP_2X2V2 | 2×2 routing validation v2 | COMPLETE |
| EXP_DIMENSIONALITY | Four-class LDA dimensionality | COMPLETE |
| EXP_FRAMING | Output framing effect on step-1 AUROC | COMPLETE |
| EXP_CROSS_TASK | Cross-task cosine similarity | COMPLETE |
| EXP_FALSE_CERTAINTY_V2 | False certainty detection v2 — entropy-matched CC/CW, Fisher vs entropy (EXP-A) | COMPLETE |
| EXP_ENTROPY_TRAJECTORY_V2 | Entropy trajectory v2 — 15-step KV-cache trajectory for CC/CW (EXP-D) | COMPLETE |
| EXP_SCALE_OBS_V1 | Scale observability v1 — bilateral oracle AUROC across Qwen 1.5B/3B, Llama 1B/3B (EXP-E) | COMPLETE |
| EXP_REASONING_GEOMETRY_V1 | Reasoning geometry v1 — DeepSeek-R1-Distill-Qwen-1.5B, commit%=75.8%, z=49.77 (EXP-B) | COMPLETE |
| EXP_REASONING_GEOMETRY_LLAMA_V1 | Reasoning geometry Llama v1 — DeepSeek-R1-Distill-Llama-8B, commit%=82.9%, z=679.73 (EXP-C) | COMPLETE |
| EXP_F_COMMIT_QUALITY_V1 | Commit-point hidden state quality — AUROC=0.650, shuffled=0.440, BLIND verdict, N=40/class (EXP-F). C023 EXPLORATORY. | COMPLETE |
| EXP_H_CC_CW_PATCHING_V1 | CC/CW causal patching — max Δ_F1=+0.0004, EPIPHENOMENAL, kill criterion triggered. C024 SUPPORTED. (EXP-H) | COMPLETE |
| EXP_G_REASONING_ENTROPY_TRAJ_V1 | Reasoning model answer entropy trajectory — 15-step post-think entropy profile for CC vs CW on GSM8K (EXP-G). BURST: peak step=4, peak AUROC=0.6932, trajectory AUROC=0.8424, CW entropy collapse. C027 SUPPORTED. | COMPLETE |
| EXP_I_EARLY_EXIT_CAUSAL_V1 | Early exit causal truncation — truncate at commit point, compare full vs early F1 (EXP-I). Δf1=+0.0059±0.047, 87.4% savings, p=0.08. MINIMAL_QUALITY_LOSS. C028 SUPPORTED. | COMPLETE |
| EXP_SCALE_EXTENSION_V1 | Scale extension — Qwen2.5-7B-Instruct bilateral oracle Fisher probe (EXP-Scale). Fisher=0.8402, Entropy=0.9645, Δ=+0.11 vs 1.5B. AUROC_SURVIVED. C021 ceiling falsified for instruct. C029 new. | COMPLETE |
| EXP_J_PERTURBATION_BATTERY_V1 | Perturbation invariance battery — 4 variants × 160 items × Qwen, ICC=0.913 ROBUST. C025 initial result. (EXP-J) | COMPLETE |
| EXP_J_PERTURBATION_BATTERY_V2 | Perturbation invariance battery Llama replication — same protocol on Llama-3.2-3B-Instruct. ICC=0.9334, between/within=14.0:1, all corr ≥ 0.91, sep_preserved t=20.264 p<0.0001. C025 CONFIRMED across two architectures. | COMPLETE |
| EXP_FALSE_CERTAINTY_LLAMA_V1 | Llama cross-arch replication of C017/C018 — Fisher=0.818, entropy=0.453, gap=0.365, BO_Transfer=0.768. C017/C018 now cross-arch. | COMPLETE |
| EXP_L_STAGE_SWEEP_V1 | Stage sweep: Qwen2.5-1.5B-Base (live, N=50/class) + hardcoded Instruct/Reasoning. C026 Pythia-specific confirmed. Fisher net gap=0.012 (ambiguous at N=50). Entropy=0.7219 (clear). NON_MONOTONE trend confounded by N. C030 EXPLORATORY. | COMPLETE |
| EXP_MISTRAL_L1_V1 | Mistral-7B-Instruct-v0.3 bilateral oracle L1: AUROC=0.7780, shuffled=0.5524, N=200/class, 3795 scanned. Highest L1 AUROC across four architectures. C031 SUPPORTED. | COMPLETE |
| EXP_GEMMA_L2_V1 | Gemma-2-2B-IT L2 confabulation: Fisher=0.570, gap=0.022, T2_L2 NOT_SUPPORTED. θ_conf=0.067. BO_Transfer confirms geometry present. C036 T2_L2 pattern. | COMPLETE |
| EXP_MISTRAL_L2_V1 | Mistral-7B-Instruct-v0.3 L2 confabulation: Fisher=0.5952, gap=−0.014, T2_L2 NOT_SUPPORTED. θ_conf=0.122. BO_Transfer=0.6624. C035, C036. | COMPLETE |
| EXP_CO_LABELING_V1 | CO labeling comparison (Qwen2.5-1.5B-Instruct): CO AUROC=0.885 vs BO AUROC=0.806. C032 SUPPORTED. | COMPLETE |
| EXP_BEHAVIORAL_BASELINE_V1 | Behavioral baseline comparison at L2 (Qwen): B3_top1_prob=0.384 (below chance), self-consistency=0.613, Fisher=0.845. C033 SUPPORTED. | COMPLETE |
| EXP_L2_LARGE_N_V1 | Large-N cross-validated L2 (N=500/class CO-style, 5-fold, Qwen2.5-1.5B-Instruct): CV Fisher=0.7629±0.0120, CV Gap=0.1628. STABLE_SIGNAL. C040 CONFIRMED. (Tier 0) | COMPLETE |
| EXP_OOD_GENERALIZATION_V2 | OOD generalization: TriviaQA probe to HotpotQA (transfer efficiency=84.5%) and MMLU-STEM (~random). OOD_PARTIAL. C041 SUPPORTED. (Tier 0) | COMPLETE |
| EXP_L_STAGE_SWEEP_V2 | Clean stage comparison (matched N=200/class): Base→Instruct→Reasoning Qwen backbone. INVERTED_U on L1 (C042), MONOTONE_RISE on L2 (C043). 283 min. (Tier 0) | COMPLETE |
| EXP_CO_GEMMA_MISTRAL_V1 | CO labeling on T2_L2 architectures (Gemma + Mistral). C036_CONFIRMED: Gemma CO Fisher=0.8368 CI=[0.749,0.910] gap=+0.2056 (CO_RECOVERS); Mistral CO Fisher=0.8580 CI=[0.777,0.927] gap=+0.2114 (CO_RECOVERS). Both CLEAN. 139 min total. (Tier 1) | COMPLETE |
| EXP_PHI_BILATERAL_V1 | Phi-3.5-Mini-Instruct bilateral oracle L1+L2. C044: L1 AUROC=0.8456 CI=[0.758,0.921] N=200/class 2268 scanned (CLEAN). L2 Fisher=0.7120 gap=0.1216 BO_Transfer=0.7920 SUPPORTED. Highest L1 of 5 architectures. 212.9 min. (Tier 1) | COMPLETE |
| EXP_TEACHER_INDEPENDENCE_V1 | Teacher independence validation — Qwen/Qwen3-1.7B (native reasoning model, non-R1 lineage). N=100/100 committed (100% commit rate), mean_commit_pct=99.8% (commits within first 1-2 think tokens), null_mean=48.7%, z=1.3×10¹⁵, mean_think_len=766, cal_AUROC=0.9268. REPLICATED — more extreme than R1-distill (75.8%/82.9%). C022 teacher confound RESOLVED: early commitment is NOT R1-specific. C045 SUPPORTED. Law 2 teacher-independent (3 models, 2 lineages). | COMPLETE |
| EXP_MATH_AUROC_V2 | MATH entry-point AUROC correction — DeepSeek-R1-Distill-Qwen-1.5B on MATH-500, N=100/class (correcting original N=30/class estimate). AUROC=0.8558, CI=[0.7291,0.9566], L25=L26, shuffled=0.574 (CLEAN). Original N=30/class estimate (0.9111) was within the CI but upward-biased. Kill not triggered. C039 corrected. Highest step-1 AUROC in program (0.8558 vs TriviaQA L2=0.854). §5.25 added. | COMPLETE |

---

*All experimental artifacts, frozen results, and protocol specifications are available in the research repository under `results/frozen/`, `science/EXPERIMENTS.yaml`, and `science/CLAIMS.yaml`. Frozen results are immutable once written. The claims registry is enforced by `science/validate_claims.py`.*

---

## Appendix C — Formal Framework for Computational Observability

This appendix provides mathematical definitions of O, C, A that are independent of any particular estimator.

### C.1 Observability (O)

**Definition.** Let M be a language model with hidden dimension d. Let Z_L^(t) ∈ ℝ^d denote the residual stream hidden state at layer L and generation step t. Let T be an intervention protocol that assigns binary labels Y ∈ {0,1} to input queries through behavioral tests applied independently of hidden-state collection. Let F be a class of probe functions f : ℝ^d → ℝ. The **observability** of M at (L, t) under T is:

O(M, L, t, T) = sup_{f ∈ F} AUROC(f(Z_L^(t)), Y_{do(T)})

**Operationalization.** Fisher+PCA64 provides a lower bound on O for the linear probe class. The bilateral oracle is a specific intervention T_BO. The current estimate O(M, 26, 1, T_BO) ≥ 0.70 across five architectures is a lower bound under the linear probe class.

**Four axioms:**
1. *Estimator invariance.* O is a property of M, L, t, T — not of F.
2. *Intervention dependence.* O(M, L, t, T1) ≠ O(M, L, t, T2) for different protocols T1, T2.
3. *Layer monotonicity (observed).* O(M, L_peak, 1, T) > O(M, 0, 1, T) in all tested models.
4. *Temporal accessibility.* O(M, L, 1, T) > O(M, L, t_prefill, T) in all tested models.

### C.2 Commitment (C)

**Definition.** Let t*(M, q, ε_C) = inf{t : H(A | Z_{0:t}, q) ≤ ε_C} be the commit step where conditional answer entropy drops below threshold. The commitment fraction is:

C(M, q) = (T − t*(M, q, ε_C)) / T

commit_pct = 100 × C(M, q). Observed: R1-Qwen C=0.758, R1-Llama C=0.829, Qwen3-native C=0.998.

C measures *when* the model settles, not *how well*. E[f1_full − f1_truncated at t*] = +0.006 (C028): post-commitment tokens are nearly neutral elaboration.

### C.3 Accessibility (A)

**Definition.** A : Θ × {L1, L2, L3} → [0,1] maps (training configuration θ, task level ℓ) to observability A(θ, ℓ) = O(M_θ, L_peak, t_ℓ, T_ℓ).

Empirical characterization (Qwen backbone, matched N=200/class):

| Stage | A(θ, L1) | A(θ, L2) gap |
|---|---|---|
| BASE | 0.740 | 0.064 |
| SFT | 0.803 | 0.096 |
| REASONING | 0.725 | 0.148 |

L1: INVERTED_U. L2: MONOTONE_RISE. Same training step, opposite effects on L1 vs L2.

### C.4 Laws as Formal Statements

- **Law 1:** ∀M ∈ M_Goldilocks, O(M, L_peak, 1, T_BO) ≥ 0.70
- **Law 2:** ∀M ∈ M_reasoning, E_q[C(M,q)] ≥ 0.70 over committed queries
- **Law 3:** A(θ_REASONING(B), L2) > A(θ_SFT(B), L2) > A(θ_BASE(B), L2) for all backbones B; L1 INVERTED_U
- **Law 4:** ∀M ∈ M_reasoning on math tasks, O(M, L_peak−1, 1, T_correctness) ≥ 0.85

### C.5 Estimator Independence

| Quantity | Current estimator | Class |
|---|---|---|
| O | Fisher+PCA64 | Linear probes |
| C | Fisher trajectory crossing | Scalar residual stream function |
| A | O at each training stage | Inherits from O |

A paper achieving O=0.90 with SAE features measures the same O better — it does not contradict Fisher O=0.73. The bilateral oracle protocol and O/C/A framework are the contribution. Fisher+PCA64 is the current best implementation.

---

*All experimental artifacts, frozen results, and protocol specifications are available in the research repository under `results/frozen/`, `science/EXPERIMENTS.yaml`, and `science/CLAIMS.yaml`. Frozen results are immutable once written. The claims registry is enforced by `science/validate_claims.py`.*
