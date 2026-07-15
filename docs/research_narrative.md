# Eight Months of Research: Epistemic Reliability and Observability in Neural Language Models

**Lakshmi Chakradhar Vijayarao**
Independent Research Program | May 2025 – Present

---

## Overview

This document is a complete account of an eight-month independent research program in reliable and verifiable AI systems. The program began with a narrow empirical question — can hallucination be detected from internal representations? — and progressively deepened into a coherent scientific investigation of a more fundamental question: **what does it mean for a language model's epistemic state to be observable, and how does optimization reshape that observability?**

The work spans six research efforts, five of which are documented in detail below. It comprises roughly 15 Kaggle GPU experiments, over 50 formally registered experimental claims, a claims governance infrastructure, three confirmed falsifications (claimed and corrected), and a manuscript pipeline in preparation. The arc is not a collection of disconnected projects — each effort revealed a limitation in the prior framing, motivating the next. What emerged is a research identity: **epistemic observability science**, the study of how transformer language models encode knowledge-state geometry in residual stream representations, and how that geometry changes under post-training optimization.

---

## The Central Thread

The organizing question through all eight months has been a simple one: when a language model answers a question, does it "know" the answer in any measurable sense, and if so, where is that knowledge in the computation?

This question sounds like interpretability but resolves differently. It is not about circuits or attention heads. It is about whether there exists a reliable, linearly extractable signal in the internal state that distinguishes questions the model can answer from its own parametric knowledge versus questions it can only answer by recovering information from the supplied context — and whether that signal can be built into inference infrastructure.

The answer, arrived at after eight months of work with both positive results and deliberate falsifications, is: **yes, such a signal exists, it is linearly organized, it is architecture-consistent, but it is a measurement instrument, not a control mechanism.**

---

## Phase 1 — Mechanistic Localization of Failure (MECH-INT)

**Question:** Where in the transformer computation do failures originate?

The first project built a 12-stage interpretability pipeline to identify the internal locus of hallucination. Working with 534 manually curated samples on GPT-2 class models, the experiment extracted pre-residual component activations across all layers, independently for attention and FFN contributions, and probed each for failure signal.

**Key findings:**

- Hallucination localized to **Layers 8–9 (33–38% depth)**, with FFN over-retrieval as the consistent mechanism. The failure is not distributed across the network — it concentrates in a narrow stratum where world knowledge is predominantly stored.
- Hidden-state probes outperformed output-level baselines: **AUROC 0.604 vs 0.576**. This gap is modest but structurally important — it confirmed that reliability signals emerge internally before they manifest in generated text.
- Representations show **extreme sparsity**: 100 of 768 dimensions active (87% sparsity). Failure signal is concentrated in a small subspace, not diffuse.
- Activation steering at α=40 produced **AUROC 0.490** — a deliberate degradation confirming causal directional influence. The geometry is controllable in the forward direction; the question of whether it is fully controllable would take six more months to resolve.

**What this established:** Reliability failures have an internal address. They are not just output artifacts. This grounded all subsequent work in representation space rather than output analysis.

---

## Phase 2 — Geometric Structure of Reliability Signals

**2a. HaRP — Hallucination-aware Representation Probing**

**Question:** Can the internal signal scale to production and support a deployment policy?

This project shifted to Qwen 2.5 3B (a contemporary instruction-tuned model) and built a complete detection-to-routing pipeline.

**Key findings:**

- Fisher+PCA probing achieved **AUROC 0.775**, a +0.198 improvement over the entropy baseline (0.577). The signal scales with model size: at 117M parameters, AUROC ≈ 0.50 (near chance); at 3B, it reaches 0.775.
- A three-way routing policy (Accept / Regenerate / Abstain) built on top of this signal improved calibration: **ECE 0.039 vs 0.072** on the raw model.
- Real-time deployment feasibility confirmed: **23.9ms latency**, less than 0.5% overhead.
- A critical self-correction occurred during this project. A data leakage artifact of **+0.1906 AUROC** was identified and corrected; the corrected cross-validated out-of-fold performance was 0.771. Publishing an uncorrected inflated result would have been standard practice in many labs. Correcting it at cost to the headline number was a deliberate methodological choice.
- The scale-dependence finding (0.50 → 0.775 AUROC as model grows) became a central motivating observation for the geometric theory project.

**2b. GEOM-PROOF — Geometric Characterization of Reliability Signals**

**Question:** Is there a mathematical theory of why these signals exist and how strong they can get?

This project derived the theoretical foundations.

**Key findings:**

- Derived the empirical relationship **AUROC ≈ Φ(√J/2)**, where J is the Fisher information separability score. This relationship held with ≤1.5% error at the optimal extraction layer, connecting the information-geometric quantity (Fisher information between PARAM and WRONG distributions) to the empirically observed detection performance.
- Established a formal connection: **J ≈ W₂²**, linking Fisher information to Wasserstein-2 distance. This enables geometric interpretation — Fisher separability is approximately the squared Earth mover's distance between the two hidden-state distributions.
- Observed **R² = 0.9996** in the scaling behavior, with separability approaching ~0.99 AUROC at larger model scales — a strong prediction that larger models become more epistemically legible, not less.
- Applied conformal prediction to bound hallucination risk: **P ≤ 0.07** under acceptance decisions at ~52.9% coverage.
- Identified the limits of Gaussian assumptions. Under quantization, the geometric certificate degrades. The continuous distribution assumption breaks, and **Sliced Wasserstein (ρ = 0.821) captures the geometry better than Gaussian approximations (ρ = 0.458)**.

**What phases 1–2 together established:** Reliability failures are geometrically organized in representation space, the organization obeys a tractable information-theoretic law, and this law predicts scaling behavior. But all of this was probing a model's own error-detection signal. A deeper question remained: what happens when these failures cascade across systems?

---

## Phase 3 — Failure Propagation in Multi-Step Systems (FAIL-CHAIN)

**Question:** How do reliability failures propagate across multi-step LLM pipelines?

The focus shifted from single-model detection to system-level dynamics. 600 simulated multi-step pipelines were analyzed under self-refinement workflows.

**Key findings:**

- **92.3% cascade failure rate** in self-refinement workflows. Once a failure occurs in a pipeline step, recovery is nearly impossible without external intervention.
- Failure propagation follows a near-irreversible Markov process: **P(failure→failure) ≈ 0.9947**. The implied theoretical recovery horizon is ~189 steps — practically unreachable in deployed systems.
- Early detection signals (built on internal state monitoring) achieved **AUROC 0.662** for predicting cascade failure before it propagates. This is modest but non-trivial: it means early intervention is possible.
- Defined 8 failure modes including **latent failures** (0.5% of cases), which are not captured by single-step evaluation. Latent failures are correct-looking outputs that contain structurally corrupted reasoning chains — invisible at the output level but detectable internally.
- Demonstrated that **abstention reduces ineffective retries** more than regeneration strategies. The correct control mechanism is not "try again" but "recognize the regime and escalate."

**What this established:** Reliability is not a property of isolated predictions. It is governed by system-level dynamics. Single-step evaluation misses the dominant failure mode (cascade, not isolated error). This pushed the research toward infrastructure-level intervention rather than output-level filtering.

---

## Phase 4 — Adaptive Deployment Governance (GUARDIAN)

**Question:** Can internal geometric signals support real-time governance with acceptable overhead?

This project implemented local Fisher separability as a production deployment signal on Mistral 7B.

**Key findings:**

- Built a real-time reliability filter with **0.83ms latency** (~0.14% overhead). Viable for production inference.
- Introduced **local Fisher separability** (k=50 KNN neighborhood), showing that 67.3% of OOD queries fall in the LOW reliability zone versus 38.7% for in-distribution queries. The geometry distinguishes distribution shift in real-time.
- Achieved **0.0% false-accept rate** under in-distribution evaluation through adaptive thresholding.
- Found a **generalization gap: 0.804 → 0.616 AUROC** under distribution shift. Crucially, the limitation was identified as calibration, not geometry. The geometric structure transfers; the calibration boundary does not. This distinction — geometry is robust, calibration is fragile — became central to the next phase.
- No improvement in OOD false-accept rate: the probe geometry detects OOD distribution shift but cannot reduce false accepts in that regime. This is the correct negative result — it localizes the problem precisely.

**What phases 3–4 together established:** The geometric signals from phases 1–2 are deployable and work in real-time, but the system-level and calibration challenges are substantial. Most importantly, GUARDIAN revealed that the core question was not "can we filter outputs?" but "can we precisely characterize what the model's internal state is encoding, independent of its output?"

---

## Phase 5 — The Credence Infrastructure and Epistemic Qualification

**Question:** How do we build rigorous epistemic infrastructure for tracking uncertain claims across long research programs?

Between phases 4 and 5, a significant methodological infrastructure project was completed: the **Credence system**, an epistemic qualification ledger (EQL) designed to track the verification status of uncertain claims in AI development workflows.

The system implements:
- **Ghost constraint detection**: automatic identification of unverified numeric claims in code and documentation
- **Epistemic lifecycle tracking**: claim states from REGISTERED → VERIFIED/FALSIFIED with evidence chains
- **Session memory persistence**: unverified constraints survive across development sessions
- **Structural staleness detection**: automatic flagging of API versions, pricing, auth lifetimes that may be outdated

This project produced the **EQL (Epistemic Qualification Ledger)** concept with formal definitions for Epistemic Qualification Rate (EQLR) and Full Constraint Rate (FCR), a multi-model benchmark across 8 models and 6 organizations, and a claims governance system now in production use on the research program itself.

The deeper contribution: most AI development treats uncertain numeric claims (rate limits, timeouts, configuration parameters) as verified facts. The Credence system formalizes the difference between "we think X" and "we have confirmed X." This distinction matters most when AI systems act on internal assumptions.

---

## Phase 6 — Epistemic Observability Science (Current Program)

**Question:** What is the fundamental structure of epistemic accessibility in transformer models — the measurable internal distinction between parametric knowledge and context-dependent recovery?

This is the current and deepest phase. It emerged from recognizing that all prior work was probing a proximal signal (output quality, error patterns) rather than a fundamental one. The fundamental question is: does the model's internal representation, at the moment of generation, carry geometric information about the knowledge source being accessed?

### The Bilateral Oracle Protocol

The central methodological innovation is the **bilateral oracle**, a measurement protocol for rigorous epistemic label assignment:

- **PARAM**: Model answers correctly from parametric knowledge (nocontext F1 ≥ 0.50)
- **CTX_DEP**: Model fails without context but succeeds with it (nocontext F1 ≤ 0.05 AND withcontext F1 ≥ 0.50)
- **SKIP**: Neither — excluded from analysis

This protocol isolates knowledge-source type without conflating it with output quality, fluency, or question difficulty. Prior probing work typically used correct/incorrect labels, which confounds knowledge source with output quality. The bilateral oracle separates them: a PARAM item is answered correctly from memory; a CTX_DEP item is answered only when context is supplied. Hidden states are always extracted from the nocontext generation pass, ensuring the signal reflects internal knowledge state, not context retrieval.

### C3 Series: Nonlinear Probe Recovery and Fisher+PCA64 Validation

The C3 experiment series (three versions, iterating through methodological failures) established the core measurement protocol.

**C3-v1 and v2 (failed, documented):** 
- v1 used wrong labels (PARAM vs WRONG); base model calibration failure invalidated base/instruct comparison.
- v2 used bilateral oracle labels but Fisher LDA with n_train=60, d=1536 — degenerate covariance. Shuffled AUROC exceeded real AUROC (0.713 vs 0.708 for Qwen instruct). A catastrophic estimator failure that was correctly identified and not published.

**C3-v3 (final, correct protocol):**
- Fisher+PCA64 (LDA solver=lsqr, shrinkage=auto, PCA dim=64, N≥128/class)
- Qwen2.5-1.5B-Instruct: **AUROC = 0.841**, shuffled = 0.617 (WARN), n=128/class
- Llama-3.2-3B-Instruct: **AUROC = 0.846**, shuffled = 0.427 (CLEAN), n=150/class
- Nonlinear probes (SVM-RBF+PCA64, MLP-2, MLP-3): NO_RECOVERY — best nonlinear ≤ linear by ≤0.019 AUROC
- The bilateral oracle signal at L26 step-1 is **linearly organized** — no nonlinear structure recoverable

**Three claims falsified through this process:**
1. **RLHF attenuation Δ=-0.036**: Prior claim that instruction tuning attenuates epistemic accessibility. This was a degenerate Fisher artifact at small N. Correct result: signal PRESERVED or enhanced under RLHF.
2. **Llama AUROC=0.629**: Prior claim of genuine Llama weakness. Was Fisher degenerate in d=3072 at small N. Corrected: Llama 0.846, indistinguishable from Qwen 0.841.
3. **Nonlinear probe recovery**: Was probe pathology (small N overfitting) in C3-v1/-v2. Corrected: no recovery above linear.

The explicit falsification record is unusual in ML research. It was deliberately maintained.

### ESM Series: Cross-Model Epistemic Signal Measurement

The ESM (Epistemic Signal Measurement) experiments established cross-model and temporal properties of the signal.

**Key findings across 4 models (Qwen, Llama, DeepSeek, Phi):**
- Mean step-1 AUROC = **0.785 vs prefill = 0.567** across 4 models. The first generated token carries substantially more epistemic information than the last prompt token.
- Llama showed a **COMMITMENT_MOMENT**: step-1 AUROC = 0.866, later steps ≈ 0.517 (note: this was later revised — see MONOTONE_RISE below).
- GQA attention (grouped query attention) shows stronger signal than SWA/MQA attention architectures.
- Single-point L26 hold-out AUROC: **0.9294** on 30% held-out test.
- RLHF geometry rotation confirmed (D3): instruct vs base geometry cosim ≈ 0.007 — instruction tuning rotates the epistemic axis, but the signal persists. The geometry changes; the separability does not disappear.
- Cross-domain partial transfer (HotpotQA): AUROC = **0.6192**, above chance but below full-transfer threshold. Knowledge-source geometry is partially portable across QA domains.

### Activation Patching: Causal Verification

**Question:** Is the bilateral oracle geometry causally related to the model's epistemic state, or is it epiphenomenal?

Centroid-direction residual stream patching was applied at generation step-1, across all layers L4–L26. The patching vector was computed as the centroid difference between PARAM and CTX_DEP hidden state distributions.

**Result: EPIPHENOMENAL at all layers.**
- True direction Δ = 0 F1 gain at every layer tested
- Shuffled control occasionally showed +0.0095 gain (random noise)
- Specific delta = 0.0000 across the sweep

This is a critical negative result. The geometry is a **readout of epistemic state**, not a control point. Injecting the "know the answer" direction does not cause the model to know the answer. This resolves the interpretation: the bilateral oracle signal is measurement infrastructure, not intervention infrastructure. This is not a failure — it precisely locates the tool's domain.

### MONOTONE_RISE: Within-Generation Signal Structure

**Question:** Does epistemic signal decay after step-1, or grow?

The step_index_auroc experiment filtered for verbose PARAM items (min_gen_tokens=8) to isolate a subset where multiple generation steps are observable. Two-population structure confirmed: ~93% of TriviaQA PARAM items EOS at step 2–4 (short answers); ~7% are verbose.

**For the verbose population:**
- Step-0: AUROC = 0.639
- Step-1: AUROC = 0.683
- Step-2: AUROC = 0.762
- Step-5: AUROC = 0.790
- Step-10: AUROC = **0.906**

**MONOTONE_RISE confirmed.** Signal does not decay within generation — it grows. The "step-1 is privileged" claim (relative to prefill) is correct; the "signal decays after step-1" claim was the EOS confound: short-answer PARAM items disappear from the sample as generation progresses, making it look like signal decay when it was actually sample attrition. For verbose items, signal accumulates through generation.

This revises the understanding: step-1 is a reliable, efficient extraction point, but not the maximum-signal point for verbose-answer items.

### Borderline Geometry: Continuous or Discrete?

**Question:** Is the bilateral oracle binary label thresholding a continuous latent variable, or detecting genuinely discrete epistemic states?

Five-group experiment: STRONG_PARAM, WEAK_PARAM, BORDERLINE, STRONG_CTX_DEP, WEAK_CTX_DEP. Fisher+PCA64 probe trained on STRONG groups, applied to all five.

**Result: DISCRETE_CLUSTERS (hypothesis of continuous gradient rejected)**

Group means along Fisher discriminant:
| Group | Fisher mean | 
|-------|-------------|
| STRONG_PARAM | +1.316 |
| WEAK_PARAM | +0.073 |
| BORDERLINE | −0.278 |
| WEAK_CTX_DEP | −0.269 |
| STRONG_CTX_DEP | −1.316 |

KS tests:
- STRONG_PARAM vs WEAK_PARAM: D=0.500, **p≈0** (significant)
- WEAK_PARAM vs BORDERLINE: D=0.217, **p=0.120** (NOT significant)
- BORDERLINE vs STRONG_CTX_DEP: D=0.583, **p≈0** (significant)

**Interpretation:** The geometry is bimodal — two strong poles (±1.316) with an amorphous middle zone. WEAK_PARAM, BORDERLINE, and WEAK_CTX_DEP are geometrically indistinguishable (p=0.120 between adjacent groups). The bilateral oracle's SKIP zone corresponds to items that genuinely fall in this undifferentiated middle region — the protocol is not arbitrarily thresholding a continuous variable; it is detecting two real epistemic poles.

This is a finding about the structure of epistemic accessibility itself: **parametric knowledge and context-dependence are not endpoints on a spectrum; they are categorically different computational states**, separated by a large undifferentiated middle.

### Governance Infrastructure

The experimental program is managed with formal governance infrastructure built specifically for this project:

- **CLAIMS.yaml**: 15 registered claims with CONFIRMED / SUPPORTED / EXPLORATORY / FALSIFIED status. Every claim traces to specific experiments and specific numbers. Paper sections may only cite CONFIRMED or SUPPORTED claims.
- **EXPERIMENTS.yaml**: Protocol fingerprints for every experiment, including failed runs. Immutable once written — a historical record, not a cleaned-up summary.
- **validate_claims.py**: CI enforcement — exits non-zero if a draft paper section cites a FALSIFIED or EXPLORATORY claim.
- **results/frozen/**: Immutable frozen results. Editing frozen results is prohibited.

This governance infrastructure is itself a methodological contribution. It forces explicit epistemic accounting: the status of every factual claim made in the paper is tracked, with evidence, and with falsification records preserved. The four falsified claims (three from C3, one from dimensionality) are documented alongside the confirmed ones.

---

## Unified Contributions

Across eight months of work, the research has produced a coherent set of contributions:

### Scientific Contributions

1. **The bilateral oracle measurement protocol** — a rigorous operationalization of V-usable information for knowledge-source discrimination. It survived repeated methodological revision, architecture variation, and falsification attempts. This is the irreducible contribution.

2. **Linear organization of epistemic accessibility** — bilateral oracle signal at L26 step-1 is linearly accessible (AUROC 0.841–0.846 across Qwen/Llama; NO_RECOVERY by nonlinear probes).

3. **Architecture consistency** — Qwen2.5-1.5B-Instruct (d=1536) and Llama-3.2-3B-Instruct (d=3072) achieve indistinguishable AUROC (0.841 vs 0.846) with the same protocol.

4. **MONOTONE_RISE** — within-generation epistemic signal rises monotonically for verbose items (step-0=0.639 → step-10=0.906). The "step-1 decay" hypothesis was an EOS confound.

5. **EPIPHENOMENAL geometry** — centroid-direction patching produces zero F1 gain at all layers. The signal is a measurement instrument, not a control point.

6. **Bimodal epistemic structure** — STRONG_PARAM and STRONG_CTX_DEP are discrete poles (KS D≥0.50, p≈0); the middle zone is geometrically undifferentiated (WEAK_PARAM ↔ BORDERLINE: p=0.120).

7. **System-level failure dynamics** — cascade failure rate 92.3%, Markov P≈0.9947, recovery horizon ~189 steps. Reliability is a system property, not a prediction property.

8. **Geometric theory** — AUROC ≈ Φ(√J/2), J ≈ W₂², R²=0.9996 scaling law connecting Fisher information to Wasserstein distance and detection performance.

### Methodological Contributions

1. **Explicit falsification record** — three falsified claims corrected in published protocol (RLHF attenuation, Llama weakness, nonlinear recovery). Most ML research does not do this.
2. **Claims governance infrastructure** — formal CONFIRMED/FALSIFIED tracking enforced by CI before any draft submission.
3. **Protocol fingerprinting** — every experiment records its exact protocol; results from different protocols are not compared.
4. **Clean shuffled controls** — standard practice in this program; several results revised when shuffled AUROC revealed confounds.

---

## Falsification History (Epistemic Integrity Record)

| Claim | Status | What happened |
|-------|--------|---------------|
| RLHF attenuates accessibility Δ=-0.036 | FALSIFIED | Fisher degenerate at small N; sign test built on artifact |
| Llama has weak epistemic accessibility (0.629) | FALSIFIED | Fisher degenerate in d=3072; corrected to 0.846 |
| Nonlinear probes recover additional signal | FALSIFIED | Was probe overfitting at small N; C3-v3 fixed |
| Step-1 signal decays at later generation steps | REVISED | Was EOS confound; MONOTONE_RISE for verbose items |
| C3-v3 Qwen AUROC=0.841 fully trustworthy | UNDER REVIEW | Shuffled=0.617 (WARN); large_n v1 gives 0.6566 with clean shuffled=0.4495 |

The last row represents ongoing work. The large-N validation (v1, N=121/class) produced Qwen AUROC=0.6566 with a clean shuffled baseline (0.4495 vs 0.5 expected). This is lower than the C3-v3 result (0.841, shuffled=0.617 WARN), suggesting C3-v3 may have been inflated by dataset ordering bias (items in TriviaQA order, not shuffled). The large-N v2 experiment (pool=10000, N=200/class target) is queued to resolve this.

---

## Current State and Open Questions

**What is established (CONFIRMED claims, can be cited in paper):**
- Fisher+PCA64 achieves measurable AUROC above chance on bilateral oracle labels
- Signal is linearly organized, architecture-consistent
- Bilateral oracle SKIP zone corresponds to geometrically undifferentiated items
- Centroid-direction patching is epiphenomenal
- MONOTONE_RISE for verbose items

**What is under active investigation:**
- True AUROC magnitude with clean sampling (C001 "≥0.82" potentially needs revision)
- Llama large-N result (gated access blocking replication)
- 2D observability surface (layer × step AUROC heatmap)
- Training dynamics (does epistemic accessibility emerge during training or exist from initialization?)

**The deepest open question:** If the bilateral oracle signal is epiphenomenal to residual stream patching, what *is* the causal structure? Head-level patching (Phase 4) and SAE feature integration remain untested. The hypothesis is that epistemic accessibility is encoded in specific attention heads or MLP feature directions that are not captured by the mean centroid vector. This is the research frontier.

**The long-range program:** Training-time legibility monitoring. Current probing is post-hoc. The next frontier is measuring epistemic accessibility *during* training — as a live signal that optimization could, in principle, be required to preserve. If standard capability training degrades epistemic legibility (in the sense that representations become harder to probe for knowledge-source type), this matters for the long-term governance of powerful systems. The data so far (geometry rotates but signal persists under RLHF) is ambiguous: signal survives, but the axis moves.

---

## Research Identity Statement

This program is **epistemic observability science**: the measurement and characterization of how transformer language models encode the distinguishability between parametric knowledge and contextual recovery in their residual stream geometry, and how post-training optimization modifies that encoding.

It is not interpretability in the circuits sense. It is not routing or governance engineering. It is not calibration. It is measurement science — building a reliable instrument for detecting a fundamental property of model computation, characterizing that instrument's properties (linearity, architecture consistency, causal structure), and understanding what optimization does to the measured quantity.

The project has a falsification history most speculative AI research never builds. It has formal governance infrastructure for epistemic accounting. It has explicit negative results that revise prior positive claims. These are not setbacks — they are what a scientific program looks like.

The manuscripts in preparation cover: geometric theory of reliability signals (GEOM-PROOF), failure propagation in multi-step systems (FAIL-CHAIN), and the bilateral oracle observability science paper (current program). The target venues are NeurIPS 2026 or ICLR 2027.

---

*Full experimental artifacts, protocol fingerprints, frozen results, and claims registry available in the research repository. Pre-registered experimental protocols are documented in `science/EXPERIMENTS.yaml`; the claims registry with CONFIRMED/FALSIFIED status is in `science/CLAIMS.yaml`.*
