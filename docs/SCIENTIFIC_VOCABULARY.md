# Scientific Vocabulary — Computational Observability Research
**Version:** 1.0  
**Status:** CANONICAL — terms defined here are used consistently across all documents  
**Date:** 2026-07-07

These definitions are fixed. When a term below is used in a paper, thesis, experiment file, or claim, it carries exactly this meaning. Do not redefine terms in local contexts.

---

## Core Object

**Computational Observability**  
The degree to which the internal computational state of an intelligent system can be reliably measured from its activations using lightweight, externally-applied procedures. The primary scientific object of this program. Applies to any computational system — not specific to transformers, language models, or any particular architecture.

**Epistemic Legibility**  
The specific case of computational observability applied to knowledge-state representation. A model is epistemically legible to the degree that its internal activations encode reliable information about knowledge availability and knowledge validity. Subset of computational observability; not synonymous with it.

---

## Measurement Protocol

**Bilateral Oracle**  
A two-pass labeling procedure that creates epistemically clean datasets. 
- Pass 1 (no-context): model generates an answer without any provided context. No-context F1 (nc_F1) measures parametric knowledge availability.
- Pass 2 (with-context): model generates an answer with the correct context provided. With-context F1 (wc_F1) measures retrieval-augmented performance.
- Together: items are classified as PARAM, CTX_DEP, or SKIP based on thresholds.

The bilateral oracle is an estimator-agnostic methodology. It does not depend on Fisher probes, LDA, or any specific measurement technique.

**Protocol Fingerprint**  
The minimal set of parameters that uniquely identifies an experiment's measurement procedure. Two experiments with different protocol fingerprints are not directly comparable, even if they measure nominally the same quantity. Includes: label type, oracle thresholds, extraction layer, extraction step, estimator, dataset, N, pool size, shuffled control flag.

**Observability Surface**  
The 2D function O(layer, step) describing probe AUROC across layers and generation steps for a given task. Current peak: L26, step-1 for standard models on TriviaQA. Step-index measurements show MONOTONE_RISE for EOS-filtered items (step 0→10: 0.639→0.906).

**Shuffled Control**  
A mandatory paired control where probe labels are randomly shuffled before fitting. Reports shuffled AUROC alongside real AUROC. Clean threshold: real AUROC > shuffled + 0.05. WARN threshold: gap < 0.05 (possible estimator pathology or label noise). **Every AUROC claim must report its shuffled control.**

---

## Classification Labels

**PARAM**  
Items where the model correctly answers from parametric memory, without provided context.  
Operational definition: nc_F1 ≥ 0.50 OR answer_contains(pred, golds) is True.  
Represents: parametric knowledge available and retrievable.

**CTX_DEP (Context-Dependent)**  
Items where the model cannot answer without context but can with context.  
Operational definition: nc_F1 ≤ 0.05 AND wc_F1 ≥ 0.50.  
Represents: knowledge absent from parametric memory; requires retrieval.

**SKIP**  
Items that satisfy neither PARAM nor CTX_DEP criteria. Excluded from analysis.

**CC (Correctly-Confident)**  
Items within the entropy-matched confident zone where the model answers correctly.  
Used for Task-L2 (knowledge validity) experiments.  
Defined within a confidence window (THETA_CONF ± margin on output entropy).

**CW (Wrongly-Confident / Confabulation)**  
Items within the entropy-matched confident zone where the model answers incorrectly.  
These are confabulations: confident generation of factually incorrect content.  
Defined within the same confidence window as CC (entropy-matched).

---

## Task Levels

**Three-Task Hierarchy**  
A decomposition of epistemic measurement into three distinct levels, each requiring different measurement procedures:

| Level | Task | Signal Required | Best Current AUROC |
|---|---|---|---|
| Task-L1 | Knowledge Availability (PARAM vs CTX_DEP) | Output entropy | 0.87–0.90 |
| Task-L2 | Knowledge Validity (CC vs CW, entropy-matched) | Hidden-state geometry (Fisher+PCA64) | 0.854 |
| Task-L3 | Commitment Timing (within think block) | Fisher trajectory | 0.760–1.000★ |

★ Cal AUROC 1.000 = N=10/class small-N saturation. Not a quality claim.

**Task-L1 (Knowledge Availability)**  
The question: does the model have the relevant knowledge in parametric memory? Operationalized by the bilateral oracle PARAM/CTX_DEP distinction. Output entropy alone achieves AUROC=0.87–0.90 at this level; Fisher is redundant here.

**Task-L2 (Knowledge Validity)**  
The question: when the model is confident, is it right? Operationalized as CC vs CW within an entropy-matched window. Fisher+PCA64 is essential at this level — entropy alone achieves only 0.614; the gap is 0.240. This is the level where hidden-state geometry carries independent information beyond the output distribution.

**Task-L3 (Commitment Timing)**  
The question: when does the model commit to an answer direction during generation? Operationalized by Fisher trajectory over generation steps. Requires continuous monitoring across steps, not a single step extraction.

---

## Generation Dynamics

**Commitment**  
The state at which a model's output distribution has stabilized on a specific answer direction. Operationalized as the first generation step at which the Fisher probe crosses a calibrated threshold (THETA_COMMIT).

**Commitment Moment**  
The specific token position (step index within think block or answer generation) at which commitment occurs.

**Commit_pct**  
A continuous metric: 100% × (think_len − commit_step) / think_len.  
Represents the fraction of the think block occurring AFTER commitment.  
Example: commit_pct=82.9% means 82.9% of think tokens come after commitment — the model committed at ~17% of its think budget.  
Computed as mean over committed traces only. Never-committed traces are excluded from mean_commit_pct and recorded separately as commit_rate.

**Commit_rate**  
The fraction of questions for which a commitment is detected. Distinct from commit_pct.  
Example: commit_rate=0.80 means 80/100 questions had a detectable commit; the 20 never-committed questions are excluded from commit_pct statistics.

**Computational Trajectory**  
The sequence of internal computational states across generation steps. Includes hidden state evolution, entropy evolution, and commitment dynamics. Used in trajectory-based measurements (EXP-D, Task-L3).

**Commitment Geometry**  
The multi-dimensional structure of internal activations at the commit point that encodes commitment state. Characterized by Fisher+PCA64 LDA at L26. Architecture-specific (CKA~0.18 across families — probe does not transfer between Qwen and Llama).

---

## Training Dynamics

**Observability Emergence**  
The process by which computational observability changes during pretraining. Current provisional finding (C011, EXP-T3A / EXP-K): floor ≥ 0.67 from step 512 in Pythia-1.4b, with INVERTED_U shape (peak mid-training, partial decay at convergence). Not yet confirmed at large N.

**Commitment Signal Amplification**  
The observed pattern of z-score increase from base to reasoning-distilled models: base z≈7 → reasoning-distilled Qwen z≈50 → reasoning-distilled Llama z≈679. Pattern is measured; explanation (training objective, distillation, teacher, backbone, token budget) is not yet determined.

**Training Stage**  
One of: Base (pretraining only), SFT (instruction following), RLHF (preference optimization), Reasoning (reasoning trace distillation). EXP-L measures observability across all four stages on a single backbone.

---

## Laws and Theory

**Observability Law**  
A quantitative relationship between observability metrics and training quantities (compute, loss, scale, training stage) that makes specific numerical predictions about unseen systems. Current candidates: INVERTED_U law (provisional, EXP-K), Task-L2 gap law (Fisher vs entropy, gap=0.240 at L2). Neither has been confirmed as a law; both are currently patterns or supported claims.

**Competing Theory**  
A candidate mechanistic explanation for an observed observability pattern. Current competing theories for observability emergence:
- **Information Bottleneck**: observability peaks at maximum compression (mid-training), decays as model overfits
- **Routing Optimization**: observability grows monotonically as the model learns to specialize computation
- **Predictive Coding**: observability reflects prediction error signals, peaks when prediction error is structured
- **Architectural Determination**: observability is set by architecture, not training — FLAT across checkpoints

Each theory makes specific numerical predictions about the EXP-K checkpoint curve. Never collapse competing theories to one without a discriminating experiment.

**Invariant**  
A measurement that remains stable across prompt variations (paraphrase, reordering, distractor, multilingual, adversarial), architectures, tasks, and training stages. EXP-J tests prompt invariance. Architecture invariance requires ≥3 distinct families. Task invariance requires ≥2 task domains.

---

## Estimators and Probes

**Estimator**  
A computational procedure for measuring an observability quantity from internal activations. Examples:
- Fisher+PCA64 LDA (current primary — validated, estimator-pathology-free)
- Raw Fisher LDA (RETIRED — degenerate covariance at small N; caused C012/C013/C014 failures)
- Logistic regression + PCA
- Sparse autoencoder (SAE) features
- Contrastive probes

**The science is not committed to any particular estimator.** Fisher is today's validated choice. When a better estimator is validated, it supersedes Fisher. Claims built on Fisher that survive re-measurement with the new estimator are strengthened. Claims that do not survive require re-evaluation.

**Probe**  
A lightweight classifier trained on internal activations to predict a behavioral outcome. A probe uses an estimator. The bilateral oracle provides probe training labels.

**Estimator Pathology**  
Failure mode where raw Fisher LDA at small N produces degenerate or inverted AUROC results. Cause: Fisher (LDA) requires estimating a (d × d) covariance matrix; at small N with large d, the estimate is unreliable. Resolution: PCA dimension reduction to n_components=64 before Fisher. All pre-PCA Fisher results (C012, C013, C014) were FALSIFIED due to estimator pathology.

---

## Confabulation (precise definition)

**Confabulation**  
Confident generation of factually incorrect content, operationally defined as CW items: wrong answers within the entropy-matched confident zone. Confabulation is:
- NOT hallucination (hallucination is broader — includes uncertain wrong answers)
- NOT uncertainty (confabulation is specifically confident-wrong, not unsure)
- NOT sycophancy (the model was not influenced by user input)
- NOT a model "lying" (there is no evidence of intent)

Use "confabulation" consistently for CW items. Use "hallucination" only when citing prior work that uses that term.

---

## Infrastructure Terms

**J-score (J_know)**  
A scalar legibility signal combining Fisher direction, entropy, and margin. Used for operational routing decisions. Not the same as Fisher AUROC. Fisher AUROC measures probe separability quality; J-score is a per-item inference-time signal.

**Observatory Specification**  
The standardized set of required procedures for making a valid computational observability measurement. Analogous to POSIX for operating systems: a specification, not a paper. See `OBSERVATORY_SPECIFICATION.md`.

**Promotion Ladder**  
The evidence ladder for claim confidence:
- L0: Single experiment
- L1: Replicated on same architecture/task
- L2: Replicated on different task, same architecture
- L3: Replicated on different architecture, same task
- L4: Training-general (across training stages)
- L5: Candidate Law (cross-architecture, cross-task, cross-training, with predictions)

**Negative Result Registry**  
The record of experiments that falsified hypotheses. Format: Observation → Reason → Evidence → Replacement hypothesis. Currently includes: raw Fisher null (estimator pathology), centroid patching null (C005/C024, both PARAM/CTX_DEP and CC/CW domains).
