# Open Scientific Questions
**Status:** PERMANENT — this document is the laboratory agenda  
**Updated:** 2026-07-12  
**Rule:** Questions are never removed. They are answered (with evidence and claim ID), superseded, or scoped out. New questions are added when experiments reveal them.

---

## Era 2 Orientation — Invariants of Computational Organization

The program has graduated from "Can we measure O?" (answered: yes, Law 1 confirmed) to the harder question:

> **What computational structures remain invariant across architecture, scale, training, alignment, and modality?**

These invariants are the true scientific objects. O, C, and A are measurable proxies for such invariants — not the invariants themselves. Every Tier 1 question should be read as a question about invariants, not about the existence of a measurement.

**On causality:** the activation patching null result (C005, C024) rules out *representational causality at the centroid level*. It does not rule out (a) **training causality** — does training with observability objectives reduce confabulation?; (b) **routing causality** — does Fisher-based inference-time routing change output quality?; or (c) **architectural causality** — do design choices that raise O produce less confabulation? Do not let "epiphenomenal under centroid patching" collapse into "no causal relevance." These are different claims about different mechanism families. See `docs/COMPETING_THEORIES.md` for the formal discrimination framework.

---

These are not TODOs. They are the scientific questions that the program exists to answer.  
Every experiment should trace to at least one question here.  
Every new experiment proposal should state which question it addresses and which competing interpretation it would eliminate.

---

## Tier 1 — Foundational (determines whether the framework survives)

### Q1: Why should optimization produce observable computation?
Not: *does it produce observable computation?* (Answer: yes, Law 1 confirmed.)  
But: *what property of the optimization objective or architecture makes O systematically non-zero?*

Three competing hypotheses:
- **H-A (Compression):** Optimization compresses computation into low-dimensional structure because reusable representations are more efficient. Predicts: O scales with task diversity in training data.
- **H-B (Supervision):** Observability emerges because supervised objectives reward reusable internal representations. Predicts: O differs systematically between SFT and RL-aligned models in ways that track objective differences.
- **H-C (Architecture):** Observability is a side effect of residual stream geometry in transformer architectures. Predicts: O would not be measurable in non-residual architectures (SSMs, Mamba) using the same protocol.

Current status: H-A, H-B, H-C are all consistent with the data. **Nothing in the current program discriminates between them.** This is the deepest open question.

What resolves it: A/B test of observation (1) across non-transformer architectures, (2) against models trained on different objective types, or (3) against models at different capability levels with controlled architecture.

---

### Q2: What is the minimal computational object required for O?
The bilateral oracle requires instruction-following capability (C026). But instruction-following is a coarse requirement. What is the actual computational property?

- Is it the formation of a stable hidden representation of the query before generation?
- Is it the development of an internal separation between "things I know" and "things I don't know"?
- Is it something about the attention pattern structure that only emerges after sufficient training?

Current status: Unknown. EXP-K (Pythia sweep) established that base LMs without instruction tuning produce CTX_DEP=0, but this is a protocol limitation not a mechanistic answer.

What resolves it: Systematic sweep across instruction-tuned checkpoints at multiple SFT data sizes to find the minimum data-at-instruction-tuning that produces non-zero O.

---

### Q3: Is Commitment fundamental or architecture-specific?
Law 2 (commit_pct ≥ 0.70) holds across three models from two independent training lineages (R1-Qwen, R1-Llama, Qwen3-native). But all three are transformer reasoning models.

- Would C exist in a state-space model trained to do chain-of-thought?
- Would C exist in a multimodal model doing visual reasoning?
- Is commit_pct driven by the hidden state geometry (which we measure) or by the attention patterns (which we don't)?

Current status: C is robust within the tested class. Architecture-generality is unconfirmed.

What resolves it: Running the commit detection protocol on Mamba-3B-Instruct or any non-transformer instruction-tuned model. Also: SAE analysis of what feature directions shift at the commit moment.

---

## Tier 2 — Laws (determines whether the candidate laws are universal)

### Q4: Does Law 3 (INVERTED_U on L1, MONOTONE_RISE on L2) hold across model families?
Currently: single backbone (Qwen), matched N=200/class, C042/C043 SUPPORTED.

The question is whether this is a property of optimization in general, or a property of the Qwen training recipe specifically.

Pre-registered interpretations (before running cross-family sweep):
- Both patterns replicate on Llama backbone → Law 3 is universal across decoder-only transformers
- L1 replicates, L2 does not → Law 3 is partially universal; L2 pattern may be Qwen-specific
- Neither replicates → Law 3 as stated is architecture-conditional, not a law

This is the highest-priority remaining experiment.

---

### Q5: Does Law 4 (O ≥ 0.85 on mathematical reasoning step-1) hold across architectures?
Currently: C039 SUPPORTED at AUROC=0.8558, single architecture (DeepSeek-R1-Distill-Qwen-1.5B), N=100/class.

Pre-registered interpretations (before running second-arch replication):
- AUROC ≥ 0.85 on Llama variant → Law 4 confirmed as architecture-consistent
- AUROC 0.70–0.85 → Law 4 weakened to ≥ 0.70 universal, "highest-on-math" claim retracted
- AUROC 0.60–0.70 → Law 4 is Qwen-distill-specific; reformulate as architecture-conditional
- AUROC < 0.60 → Law 4 falsified; C039 is architecture-specific only

---

## Tier 3 — Mechanism (determines depth of scientific understanding)

### Q6: What computational object generates O at L2?
Three candidate models remain uneliminated:

- **Model A (Retrieval Quality):** O measures the quality of the parametric retrieval attempt. Fisher separates questions where the retrieval attempt is strong from questions where it fails mid-generation.
- **Model B (Internal Certainty Routing):** O measures an internal signal about whether the model "trusts" its own generation. The routing decision is made before output.
- **Model C (Memory-Generation Consistency):** O measures consistency between what memory encodes and what generation produces. The gap appears because RLHF trains the output to be confident regardless of internal memory state.

Current status: All three are consistent with C017, C033, C036, C040. Fisher gap = +0.240 over entropy is consistent with any model where internal state contains information that the output distribution loses.

What resolves it: SAE feature analysis — which specific feature directions in the residual stream at L26 step-1 are most predictive of the PARAM/CTX_DEP distinction? If those features correspond to identifiable semantic content (factual recall, uncertainty markers, source attribution), that discriminates between models.

**Note on causality mechanism families:** Experiment E (differential diagnosis — orthogonalize retrieval quality from RLHF assertiveness pressure) is the experiment that discriminates Models A, B, C. Design a condition where retrieval quality and RLHF pressure are orthogonal: (a) high RLHF pressure + low retrieval quality, (b) low RLHF pressure + strong retrieval. If Fisher tracks retrieval quality independently of RLHF pressure: Models A or C. If Fisher tracks RLHF pressure: Model B. Centroid patching (C005, C024) cannot answer this — it only rules out writable representational causality at the centroid level.

---

### Q7: Why does the geometry rotate under RLHF rather than disappear?
C033 (behavioral baselines gap=+0.232 over Fisher) is consistent with RLHF training assertiveness that decouples output confidence from internal epistemic state. But the mechanism is unknown.

- Does RLHF training modify the output projection (the compression step), leaving the deep residual stream geometry intact?
- Or does RLHF training modify the residual stream geometry itself, producing a new geometry that happens to remain legible under the bilateral oracle?

These have different implications. If the first: the deep geometry is RLHF-invariant, and Fisher reads "below the alignment horizon." If the second: RLHF reshapes epistemic geometry but preserves separability, which is a stronger and more surprising result.

Current status: C033 is single-architecture. Llama replication would establish generality. Neither mechanism is distinguishable from behavior alone.

**Locked framing (2026-07-12):** The RLHF-decoupling story is a *prediction*, not a confirmed mechanism. The prediction is: *"Increasingly aligned models should exhibit larger hidden-state/output divergence at L2."* This predicts that a controlled comparison of Base → SFT → DPO → RLHF → reasoning-distilled stages on the same backbone will show monotonically increasing Fisher-entropy gap. This is a pre-registered prediction. Do not state RLHF as the confirmed mechanism until this controlled sweep is run.

---

### Q10: Is PCA64 compression a bottleneck on measurable O, or is the signal genuinely linear?

The NO_RECOVERY result (C002, CONFIRMED) found that nonlinear probes achieve no higher AUROC than Fisher+PCA64. But PCA64 was applied *before* probing. PCA discards approximately 96% of variance in a 1536-dimensional Qwen representation. If epistemic routing is encoded in superposition across low-variance directions (as SAE research suggests is common), PCA compression destroys the signal before the probe is trained.

C002 proves: "no nonlinear recovery within 64 PCA dimensions." It does not prove: "no nonlinear recovery in full representation space." These are different claims.

Competing hypotheses:
- **H-linear:** The PARAM/CTX_DEP signal is genuinely linear in the full representation space. Fisher+PCA64 is a near-optimal estimator. PCA64 is not a bottleneck.
- **H-bottleneck:** PCA64 discards causally relevant structure. Full-space Fisher or nonlinear probes applied without PCA compression achieve substantially higher AUROC.

Pre-registered interpretations (Experiment B — full-space Fisher ablation):
- Full-space AUROC > PCA64 AUROC by > 0.05 → H-bottleneck supported; switch estimator
- Full-space AUROC within 0.02 of PCA64 AUROC → H-linear supported; C002 fully confirmed

Claims updated: C002 (NO_RECOVERY — will be updated with scope "within PCA64 projection" or expanded to full space).

---

### Q11: Is commit_pct ≥ 70% robust across ε_C threshold values?

Law 2 (commit_pct ≥ 70%) is reported for three reasoning models. But commit_pct is defined using a threshold ε_C. The specific ε_C value is not reported with sensitivity bounds. Qwen3's result (commit_pct = 99.8%) is consistent with any ε_C above the minimum think-block entropy — it may not be a robust result.

Competing hypotheses:
- **H-robust:** commit_pct ≥ 70% holds for each model across ε_C ∈ {0.05, 0.10, 0.15, 0.20, 0.25}. Law 2 is threshold-independent.
- **H-sensitive:** commit_pct falls below 70% for one or more models at lower ε_C values. Law 2 requires ε_C specification.

Pre-registered interpretations (Experiment C — ε_C sensitivity sweep):
- std(commit_pct) < 0.10 across ε_C range for all models → H-robust supported; C022/C045 strengthened
- commit_pct < 0.70 for any model at ε_C ≤ 0.10 → H-sensitive supported; Law 2 must be stated with ε_C specification

Claims updated: C022 (commit_pct ≥ 70%), C045 (teacher-independent commit_pct).

---

### Q12: Does the bilateral oracle Fisher discriminant measure the same object as the Marks–Tegmark truth probe?

Marks and Tegmark (2023) found that truth-value has a linear probe signature in transformer hidden states. The bilateral oracle Fisher discriminant separates PARAM from CTX_DEP (routing) and CC from CW (confabulation). These are related but potentially distinct:

- PARAM/CTX_DEP is about *source routing*, not truth-value (a PARAM item is answered correctly from parametric memory; CTX_DEP is not answered without context — correctness is not the separator)
- CO labeling (CC vs CW) is closer to Marks–Tegmark's truth framing

Competing hypotheses:
- **H-same:** Fisher discriminant is reading the same direction as Marks–Tegmark truth probe. The bilateral oracle is a more principled procedure that arrives at the same signal.
- **H-distinct:** Fisher discriminant is orthogonal to (or partially rotated from) the truth-value direction. These frameworks are measuring different aspects of internal representation.

Pre-registered interpretations (Experiment D — Marks-Tegmark comparison on same items):
- Discriminant direction cosine similarity > 0.70 AND AUROC difference < 0.05 → H-same supported
- Discriminant direction cosine similarity < 0.30 → H-distinct supported; bilateral oracle is measuring a distinct epistemic signal

Claims updated: None existing — this is a new scope-defining comparison.

---

## Tier 4 — Scope Expansion (determines future relevance)

### Q8: Does O exist in non-knowledge domains?
Everything measured so far is about knowledge retrieval (TriviaQA, MATH) or confabulation detection. These are narrow cognitive tasks. 

Does O exist for:
- **Planning** — does the hidden state at planning step-1 predict whether a multi-step plan will succeed?
- **Code generation** — does the hidden state before first token predict whether the generated code will compile/run?
- **Tool use / agent tasks** — does the hidden state when deciding which tool to call predict whether that tool call will succeed?
- **Multimodal reasoning** — does a vision-language model show O for visual questions?

This is a 5-year question, not a thesis question. But it should be explicitly registered as the scope boundary the program needs to cross eventually.

---

### Q9: Can O be directly improved through training?
JVelocityLoss (`esm/training.py`) is designed to preserve or amplify epistemic geometry during fine-tuning. Does it work?

Current status: Theoretical — implemented but not empirically validated. C038 (PRM correlation negative result) suggests that step-level J_know is not predictive of process reward, which constrains one application.

This question is important: if O can be trained into a model, then every model could be made more observable. That changes observability from a measurement science into a design science.

---

### Q13: How does the hidden state move through latent space during reasoning?

The program currently measures O, C, A as static snapshots at specific extraction points (step-1 of generation, commit point, final token). But internal computation is a trajectory, not a snapshot.

The missing dimension is **computational dynamics** — how the representation moves through latent space from input encoding to committed output.

Competing hypotheses:
- **H-trajectory-divergent:** Correct and incorrect trajectories diverge early (before or at the commit point). The commitment corresponds to a geometric phase transition — a sharp directional change in the residual stream. After commitment, trajectories become more stable (lower velocity in representation space).
- **H-trajectory-parallel:** Trajectories for correct and incorrect items are parallel throughout; the separation is static, not dynamic. The commit point does not correspond to a geometric phase transition.

Questions within this experiment:
- Do correct and incorrect generation trajectories diverge before, at, or after the commit point?
- Does commitment correspond to a sharp velocity change in the residual stream (phase transition)?
- Are there characteristic trajectory shapes (e.g., search-then-convergence for CTX_DEP items)?
- Do trajectories become more stable (lower cosine distance between consecutive tokens) after commitment?

Why this matters: If H-trajectory-divergent is confirmed, the program moves from *static measurement* to *computational dynamics*. The commit point becomes a geometric event, not just a token-count statistic. This is the experiment that opens the most transformative long-term research direction.

What resolves it: Extract residual stream at every generation token for a set of CC/CW × PARAM/CTX_DEP items. Compute trajectory velocity (cosine distance between consecutive token representations). Identify the velocity minimum/change-point. Test whether it coincides with the commit point defined by entropy.

Tier: 3 (Mechanism) — also bridges into Pillar II (Dynamics). MCE: single architecture, N=50/class, 10-token trajectory windows.

---

### Q14: Is the bilateral oracle signal measuring epistemic routing or question difficulty?

**This is the single strongest alternative explanation for the L1 result and must be answered before the paper is fully credible to hostile reviewers.**

The bilateral oracle labels PARAM items (model can answer without context) and CTX_DEP items (model cannot). A skeptic can always say:

> "PARAM items are easy and CTX_DEP items are hard. Fisher is just measuring question difficulty."

This alternative explanation is plausible, never directly tested, and would completely reframe the significance of the bilateral oracle if true.

Competing hypotheses:
- **H-difficulty:** Fisher discriminates PARAM/CTX_DEP because PARAM items are uniformly easier. Match for difficulty and the signal disappears.
- **H-epistemic:** Fisher discriminates PARAM/CTX_DEP at matched difficulty levels. The signal reflects knowledge-source routing, not question difficulty.

Experiment design (pre-registered before running):
1. Sample items from a QA dataset with human-rated difficulty scores (e.g., TriviaQA with IRT calibration).
2. Create difficulty-matched PARAM/CTX_DEP pairs: PARAM items at difficulty tier k matched with CTX_DEP items at the same difficulty tier k.
3. Run bilateral oracle Fisher probe on the matched set.
4. If AUROC remains ≥ 0.65 at matched difficulty: H-epistemic supported.
5. If AUROC drops below 0.60 at matched difficulty: H-difficulty supported; bilateral oracle measures difficulty, not routing.

Why this matters: This is the Achilles heel. It does not threaten the program if it confirms H-epistemic — it strengthens the claim. It only threatens the program if H-difficulty is confirmed, in which case the contribution is a good difficulty estimator, not an epistemic router. Either outcome is scientific progress.

**This is a Tier-1 experiment by the MCE principle.** The MCE for the bilateral oracle is exactly this experiment. Everything else is secondary until it runs.

Tier: 1 (Foundational — determines whether the bilateral oracle contribution survives its strongest attack).

---

### Q15: At what capability level does computational observability become meaningful?

The program currently knows: O exists across instruction-tuned transformer models. C026 establishes that instruction-following capability is required. But instruction-following is a coarse requirement.

The real question is: **Is there a capability threshold — a phase transition — below which O is near-zero and above which it becomes reliably non-zero?**

This question is much more interesting than "does O work on another architecture?" because it connects O to the emergence of internal epistemic organization as a function of capability.

Competing hypotheses:
- **H-threshold:** O shows a phase transition at a specific capability level. Below the threshold, models have no internal epistemic separation; above it, O ≥ 0.70 reliably. This predicts a step-function in O vs. capability.
- **H-continuous:** O grows gradually with capability. There is no threshold — just increasing observability with increasing scale and training quality.
- **H-flat:** O is driven by architecture, not capability. Once the architecture is transformer-based and residual, O appears immediately. Training improves it marginally.

Why this matters: If H-threshold is confirmed, this provides the most important practical implication of the program — a capability screening tool. Models below the threshold are not internally observable; alignment methods that depend on probing their internal states would fail.

What resolves it: Pythia model family sweep at matched task and extraction protocol. Run bilateral oracle at every checkpoint (Pythia-70M through Pythia-12B, multiple training stages). Plot O vs. capability level. Look for phase transition.

Current evidence: EXP-K (Pythia sweep) has partial data. C011 (monotone growth NOT confirmed). Provisional result: O ≥ 0.67 floor at all 8 checkpoints — suggesting no capability floor was found in the Pythia range, but the sweep may not have gone far enough to find the threshold.

Tier: 4 (Scope) — but with implications for Tier 1 if a threshold is found.

---

## Format for New Questions

When adding a question, include:
- Why this matters (what breaks if we don't know the answer)
- What the competing hypotheses are
- What the pre-registered interpretations would be for each experimental outcome
- Which tier it belongs to (Foundational / Laws / Mechanism / Scope)
- Which existing claims it would update if answered
