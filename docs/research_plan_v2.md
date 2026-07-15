# Research Plan v2 — Computational Observability Program

**Status:** LOCKED for execution. Do not revise strategy during experiment runs.
**Date locked:** July 2026
**Revision:** v2.1 (founding sentence updated July 9 2026 — adaptive computation framing)

---

## The Founding Sentence

> This program builds the measurement science required to make adaptive computation principled rather than heuristic — by establishing when, where, and under what conditions the internal computational state of an AI system can be reliably read, distinguished, and eventually used to allocate computation.

Everything in this document derives from that sentence. If a proposed action does not advance it, do not take the action.

**Why this sentence replaces the prior one.** The prior sentence ("conditions under which internal AI computations become externally measurable") framed observability as the end goal. The revised sentence clarifies the destination: observability is the *mechanism* by which adaptive computation becomes principled. Routing, reasoning budget, retrieval, verification, and safety checks are all currently built on heuristics because no reliable computational signal exists. This program builds that signal. Every experiment either (a) establishes that a signal is real and generalizes, (b) characterizes when it breaks down, or (c) tests whether it can drive a decision. The paper(s) are the measurement layer. The systems applications — early exit, Gate 3 routing, training diagnostics — are the adaptive computation layer built on top.

---

## STOP List (Frozen — No Exceptions)

These four actions are prohibited for the duration of the current experiment queue:

1. **Do not add new claims to CLAIMS.yaml.** Existing 30 claims are sufficient. Adding claims before replication is complete dilutes the paper.
2. **Do not design new experiments outside the queue below.** The queue is complete. New experiments after ICLR submission only.
3. **Do not revise the thesis architecture.** The three-task hierarchy is locked. Revision is post-ICLR.
4. **Do not begin training intervention experiments** until at least one head-level patching result exists.

---

## Risk Zero (Read First)

Before executing the experiment queue, acknowledge this risk explicitly.

**Risk Zero:** What if the bilateral oracle still primarily measures confidence, just under stricter sampling? Entropy matches this explanation for L1. The question is whether L2 (the Fisher gap at CC/CW) also resolves to confidence.

**Explicit discriminator:** Run both Fisher and entropy on CC/CW items at identical entropy thresholds. If entropy within the matched band still accounts for the Fisher gap, then bilateral oracle = confidence (stricter sampling). If Fisher adds after entropy-matching — and this holds across multiple architectures — then bilateral oracle captures something entropy does not.

This discriminator is already embedded in EXP-A (Qwen: Fisher gap=0.240 after entropy-matching). The architecture replication experiments ARE this discriminator across architectures. If Fisher gap collapses to near-zero in 3+ architectures, the paper changes to "rigorous confidence measurement, not epistemic geometry."

**Write both papers now, in outline, so narrative drift does not occur if the experiments fail.**

---

## The Two Papers

### Paper A (Fisher gap generalizes to ≥4 architectures)

**Title:** "Confabulation Geometry: A Hidden-State Signature Distinguishes Confident-Wrong from Confident-Correct Outputs Across Transformer Architectures"

**Central argument:** Within the entropy-matched confident zone, Fisher+bilateral oracle detects confabulation with AUROC 0.85+ across five architectures. This signal is not capturable by output entropy alone. The bilateral oracle is the required instrument.

**Key result:** Fisher gap ≥0.20 AUROC in ≥4 architectures, all entropy-matched.

**Narrative:** Problem → why entropy fails → bilateral oracle design → Fisher gap → five-architecture replication → causal caveat → applications.

---

### Paper B (Fisher gap fails to generalize — 1-2 architectures only)

**Title:** "A Reproducible Protocol for Measuring Epistemic Source Type in Language Models: The Bilateral Oracle"

**Central argument:** The bilateral oracle provides the first architecture-tested protocol for separating knowledge-source type from output quality. Fisher+PCA64 achieves AUROC 0.73/0.75 at L1 and 0.854/0.818 at L2 on two transformer families. Architecture generality is an open question.

**Key result:** Bilateral oracle validated at N≥200/class, two architectures, clean shuffled controls. Fisher gap is real but architecture-specific.

**Narrative:** Measurement problem → protocol design → validation → two-architecture results → Fisher gap in two families → scope limitations explicit → future architecture tests.

---

### Paper C (Bilateral oracle itself questioned)

Triggered if: bilateral oracle ablation (Experiment 0) shows that correct/incorrect labels achieve similar or higher AUROC than bilateral oracle labels on same dataset.

**Paper C pivot:** "The bilateral oracle does not add over output-quality labels on the tasks tested" — scope contracts to N=2 architectures with honest characterization.

---

## Computational Property Framework

Do not organize results by architecture. Organize by computational property.

Each property is a column. Each architecture is a row. The paper is the populated table.

| Property | Qwen2.5-1.5B | Llama3.2-3B | Gemma-2-2B | Mistral-7B | Phi-3.5 |
|---|---|---|---|---|---|
| Knowledge routing legibility (L1 entropy) | 0.73 | 0.75 | TBD | TBD | TBD |
| Confabulation geometry (L2 Fisher gap) | 0.240 | 0.365 | TBD | TBD | TBD |
| Commitment mass (L3 commit%) | 75.8% | 82.9%* | — | — | — |
| Perturbation invariance (ICC) | 0.913 | — | — | — | — |

*R1-distilled confound. Teacher independence pending.

**The paper's result section is not "here are our five architectures." It is "here are the three properties we measure, and here is how each property behaves across architectures."**

---

## Causal Evidence Ladder

Every experiment occupies exactly one level of this ladder. The ladder level determines what conclusion it justifies.

| Level | Name | What the experiment establishes |
|---|---|---|
| O1 | Observation | A geometric signature exists (can be measured) |
| O2 | Replication | Signature replicates across independent datasets/architectures |
| I1 | Correlation | Signature predicts behavioral outcome |
| I2 | Intervention (coarse) | Mean-direction patching of signature changes behavior |
| I3 | Intervention (fine) | Head/feature-level patching changes behavior |
| M1 | Mechanism | Specific circuit implements the computation |
| M2 | Mechanism + training | Circuit is shaped by training in predictable ways |
| E1 | Engineering | Optimizing signature during training changes downstream behavior |

**Current program position:** O2 on L1 (two architectures), I1 on L2 (predictive with Fisher gap), O1 on I2 (null at I2 means I2 not yet established). 

**Each experiment below is labeled with its ladder level.**

---

## Experiment Queue (Ranked by Expected Information Gain)

Experiments are ordered by EIG — the expected amount by which the result changes the scientific picture or paper narrative — not by publication convenience.

---

### Experiment 0: Bilateral Oracle Ablation (No GPU — Existing Data)

**EIG rank:** #1 (if this fails, the primary contribution is questioned before any GPU spend)

**Ladder level:** O1 → O2 (tests whether bilateral oracle adds over output-quality labels)

**Design:** Take the existing Qwen2.5-1.5B and Llama3.2-3B datasets. Label the same items two ways: (A) bilateral oracle labels (PARAM/CTX_DEP), (B) correctness labels (Correct/Incorrect). Train Fisher+PCA64 probes on both. Compare AUROC.

**Hypothesis:** Bilateral oracle AUROC > correctness AUROC on same items, because the bilateral oracle operationalizes a distinct construct (knowledge-source type) rather than output quality.

**Kill criterion:** If correctness labels achieve AUROC within 0.02 of bilateral oracle labels, bilateral oracle's protocol advantage over simpler labels is not established on these architectures.

**Decision tree:**
- Bilateral oracle AUROC > correctness by >0.05: Paper A/B proceeds as planned. Bilateral oracle is justified.
- Bilateral oracle AUROC within ±0.02 of correctness: Paper C pivot. Bilateral oracle = convenient relabeling, not novel protocol.
- Bilateral oracle AUROC < correctness by >0.05: Redesign required. Consider why.

**GPU hours:** 0. Existing data only.

---

### Experiment 1: Baseline Comparison Analysis (No GPU — Existing Data)

**EIG rank:** #2 (if baselines beat Fisher, the paper narrative changes fundamentally)

**Ladder level:** O2 (establishes Fisher adds over output-space baselines in L2)

**Design:** On existing datasets, compute: verbalized uncertainty ("I'm not sure..."), self-consistency (majority vote over N=5 samples), calibration score (top-1 probability). Compare each to Fisher+bilateral oracle on L2 task (CC/CW detection).

**Hypothesis:** Fisher+bilateral oracle exceeds all output-space baselines on the CC/CW task within the entropy-matched zone.

**Kill criterion:** If any output-space baseline achieves AUROC within 0.05 of Fisher at L2, Fisher's contribution is not established.

**Decision tree:**
- Fisher exceeds all baselines by >0.05: Paper A/B narrative intact — "hidden states carry signal that outputs don't."
- One baseline within 0.05 of Fisher: Paper characterizes Fisher as marginally additive; lead with bilateral oracle protocol as contribution.
- All baselines ≥ Fisher: Paper B only. Fisher is not a contribution beyond the bilateral oracle itself.

**GPU hours:** 0. Existing data only.

---

### Experiment 2: Gemma-2-2B-IT (Third Architecture)

**EIG rank:** #3 (first full out-of-family replication; highest architectural EIG remaining)

**Ladder level:** O2 (architecture replication of L1 and L2)

**Design:** Bilateral oracle protocol at N≥200/class, TriviaQA + HotpotQA. Extract Fisher+PCA64 at L{d-2} (penultimate layer). Measure L1 AUROC and L2 Fisher gap.

**Success criterion:** L1 AUROC ≥0.70, L2 Fisher gap ≥0.15, shuffled controls confirm validity.

**Kill criterion:** L1 AUROC <0.65 on both datasets, OR L2 gap <0.05. If triggered: characterize as architecture-specific for now, proceed with Paper B narrative.

**Decision tree:**
- Both L1 ≥0.70 AND L2 gap ≥0.15: Three architectures confirmed. Architecture generalization claim strengthened. Continue to Mistral.
- L1 ≥0.70 but L2 gap <0.15: L1 generalizes; L2 is architecture-specific or requires larger N. Note and continue.
- L1 <0.65: L1 does not trivially generalize. Investigate whether bilateral oracle protocol requires adaptation.

**GPU hours:** 8–10 hrs T4.

---

### Experiment 3: Mistral-7B-Instruct (Fourth Architecture)

**EIG rank:** #4 (second out-of-family replication; increasing confidence in generality)

**Ladder level:** O2 (architecture replication, larger model scale)

**Design:** Same as Gemma. Also test whether Fisher gap scales with model size (compare to Qwen2.5-7B L2 result if available).

**Kill criterion:** Same as Gemma. If both Gemma and Mistral fail L2, L2 is Qwen/Llama-specific.

**Decision tree:**
- L2 gap ≥0.15: Four architectures at L2. Paper A is becoming viable.
- L2 gap <0.15 (after Gemma also failed): L2 claim contracts to "two GQA-family architectures."
- L2 gap ≥0.15 but Gemma failed: L2 gap is not monotone with architecture; investigate what distinguishes Mistral from Gemma.

**GPU hours:** 12–14 hrs T4.

---

### Experiment 4: Attention Head Patching — Qwen2.5-1.5B (Causal Intervention)

**EIG rank:** #5 (highest remaining EIG for scientific impact; transforms observational → mechanistic)

**Ladder level:** I3 (fine-grained intervention, head-level)

**Design:** Identify attention heads at L20–L26 that contribute most to the PARAM/CTX_DEP LDA direction (using attention attribution). Ablate top-K heads (K=1,3,5,10) during inference. Measure: (a) LDA AUROC change, (b) behavioral routing change (does the model produce PARAM answers on CTX_DEP items or vice versa?).

**Hypothesis:** A small number of attention heads at L26 implement the routing decision. Ablating them reduces behavioral routing performance.

**Kill criterion:** Ablating top-10 heads produces <5% change in behavioral routing. If triggered: geometry is not localized to any head set at this granularity. Add as second null result alongside centroid patching. The observational signature exists without identified mechanism.

**Decision tree:**
- Top-K ablation changes routing behavior (>10%): FIRST CAUSAL RESULT. Claims C005/C024 require revision. Paper changes from "we characterized the geometry" to "we identified a causally relevant circuit." This is the most important possible outcome.
- Ablation changes LDA AUROC but not routing behavior: Geometry disrupted but not causal for behavior. Interesting but not a causal claim.
- No effect at any granularity: Extended null at I3. Confirm: "geometry is observational at centroid and head levels; causality unestablished." Design I4 (SAE feature patching, Year 2).

**GPU hours:** 6 hrs T4.

---

### Experiment 5: Teacher Independence — Qwen3-1.7B (C022 Replication)

**EIG rank:** #6 (closes the R1-lineage confound on C022; promotes to CONFIRMED if successful)

**Ladder level:** O2 (replication outside training data confound)

**Design:** Apply Fisher trajectory analysis to Qwen3-1.7B (own reasoning training, not R1-distilled) with thinking enabled. Measure commit% at same threshold used in EXP-B/C.

**Hypothesis:** commit% ≥60% in a non-R1-distilled reasoning model, confirming that early commitment is a property of reasoning training generally, not R1-distilled training specifically.

**Kill criterion:** commit% <50%, which suggests the R1 lineage was the operative variable.

**Decision tree:**
- commit% ≥60%: C022 upgrades toward CONFIRMED. "Early commitment is a general property of reasoning-trained models." L3 claim strengthens.
- commit% 50–60%: Ambiguous. Need a second non-R1 reasoning model.
- commit% <50%: C022 remains SUPPORTED (R1-lineage only). L3 claim contracts to "R1-distilled model families."

**GPU hours:** 6–8 hrs T4.

---

### Experiment 6: Phi-3.5-Mini-Instruct (Fifth Architecture)

**EIG rank:** #7 (diminishing returns; confirms or bounds generality across five families)

**Ladder level:** O2 (architecture replication — fifth family)

**Design:** Same as Gemma.

**Kill criterion:** Same. By this point, three or four architectures already tested. Phi failure is informative about scope boundaries, not fatal to the paper.

**GPU hours:** 8 hrs T4.

---

## Program Termination Rules

These are pre-committed. If triggered, do not negotiate. Execute.

**T1: Claim contraction**
If fewer than 3/5 architectures replicate L2 Fisher gap ≥0.15:
→ L2 claim contracts to "two transformer families (GQA-architecture, ~1.5–3B parameters)"
→ Paper B narrative. Bilateral oracle is the contribution.

**T2: L1 claim contraction**
If fewer than 3/5 architectures achieve L1 bilateral oracle AUROC ≥0.65:
→ L1 claim contracts to "architecture-specific instrument requiring per-family calibration"
→ Paper B narrative, more heavily caveated.

**T3: Protocol pivot**
If 0/5 architectures achieve L2 Fisher gap ≥0.10:
→ Fisher+bilateral oracle does not generalize beyond Qwen/Llama
→ The bilateral oracle is a protocol contribution only
→ Paper C. No claim about geometric signatures.

**T4: Full program pivot**
If bilateral oracle ablation (Experiment 0) shows correctness labels achieve AUROC within 0.02 of bilateral oracle labels:
→ The bilateral oracle's distinguishing property is not established
→ The paper becomes a calibration paper, not an epistemic geometry paper
→ Pause queue. Redesign before proceeding.

**T5: Do not run Experiments 4–6 if T3 has triggered.**
There is no reason to seek causal evidence for a phenomenon that did not generalize.

---

## Scientific Dependency Graph

Every claim depends on specific experiments. Every experiment answers a specific question.

```
CLAIMS.yaml
    │
    ├── L1 Claims (C009, C010, C016, C019, C020, C021)
    │   │
    │   ├── Bilateral oracle validation ──► EXP-0 (ablation) ──► "protocol adds over correctness labels"
    │   ├── Large-N AUROC (0.731/0.746) ──► DONE ──► C009 SUPPORTED
    │   ├── Entropy baseline (FISHER_REDUNDANT) ──► DONE ──► C016 SUPPORTED
    │   └── Architecture replication ──► EXP-2 (Gemma) ──► EXP-3 (Mistral) ──► EXP-6 (Phi)
    │
    ├── L2 Claims (C017, C018, C023)
    │   │
    │   ├── False certainty L2 (Qwen 0.854, Llama 0.818) ──► DONE ──► C017/C018 SUPPORTED
    │   ├── Baseline comparison ──► EXP-1 (no-GPU) ──► "Fisher adds over output baselines"
    │   └── Architecture replication ──► EXP-2 ──► EXP-3 ──► EXP-6
    │
    ├── L3 Claims (C022)
    │   │
    │   ├── EXP-B/C (R1-distilled commit%) ──► DONE ──► C022 SUPPORTED
    │   └── Teacher independence ──► EXP-5 (Qwen3) ──► C022 toward CONFIRMED
    │
    ├── Causal Claims (C005, C024)
    │   │
    │   ├── Centroid patching null ──► DONE ──► C005/C024 CONFIRMED
    │   └── Head patching ──► EXP-4 ──► either promotes I3 level or extends null
    │
    └── Protocol Claims (C026, C028)
        │
        ├── Bilateral oracle inapplicable to base LMs ──► DONE ──► C026 CONFIRMED
        └── Perturbation invariance (ICC=0.913) ──► DONE ──► C028 SUPPORTED

PAPER
    │
    ├── Abstract ──► depends on L1 + L2 architecture replication results
    ├── Section 3 (Protocol) ──► depends on EXP-0 (ablation confirms protocol value)
    ├── Section 4 (L1) ──► depends on ≥3 architecture replications
    ├── Section 5 (L2) ──► depends on ≥3 architecture replications + baseline comparison
    ├── Section 6 (Causal) ──► depends on EXP-4 result (patching)
    └── Section 7 (Limitations) ──► depends on termination rules + protocol scope
```

---

## Year 2 Roadmap (Post-ICLR Submission)

Revised per change request: training checkpoints before SAEs.

**Phase 1 (Q1 Year 2): Training Emergence**

The first Year 2 priority is clean training stage comparison (replication of EXP-L at matched N). This has higher EIG than SAE work because:
- Training dynamics could produce observability laws (quantitative)
- Laws generalize across architectures
- SAEs are currently architecture-specific

Design: Qwen2.5-1.5B-Base vs Qwen2.5-1.5B-Instruct at N=200/class (matched). Measure L1 AUROC and L2 Fisher gap at each stage. If OLMo-2 checkpoints available: 4-checkpoint sweep at matched N.

**Phase 2 (Q2 Year 2): Observability Scaling Curve**

Measure bilateral oracle Fisher AUROC as a function of parameter count: Qwen2.5-0.5B, 1.5B, 3B, 7B, 14B if feasible. Fit power law. If the curve has predictable shape, this is the program's first law.

**Phase 3 (Q3 Year 2): SAE Feature Patching**

After training emergence and scaling curve are characterized, begin mechanistic work at feature level using SAEs trained on Qwen2.5-1.5B. This is high-EIG for causal evidence but requires understanding the training dynamics first.

**Why this order:** Training laws are broader than circuit-specific mechanisms. If training dynamics produce laws that SAEs cannot explain, the laws are the higher-value contribution. If SAEs reveal circuits that explain training dynamics, the circuit work is prioritized for Year 3.

---

## Program Invariants (Read When Uncertain)

These principles do not change based on experimental outcomes. They are the scientific culture of this program.

1. **Separate measurement from interpretation.** An AUROC number is a measurement. "The model knows what it knows" is an interpretation. Never let interpretation appear in a results section without explicit labeling.

2. **Every claim requires a falsification path specified before the experiment runs.** Write the falsification condition in EXPERIMENTS.yaml before the experiment starts. Claims without falsifiers are hypotheses, not science.

3. **Every new architecture is a test of scope, not a confirmation.** Gemma, Mistral, and Phi are not validation — they are scope expansion attempts. Each one can narrow the claim. A claim that survives five architectures is not confirmed; it is well-scoped.

4. **Negative results are first-class scientific outputs.** FISHER_REDUNDANT, C005, C012/C013/C014, C026 — these are the program's most precisely informative findings. They constrain what is possible. They get sections in the paper, not footnotes.

5. **Scope of the claim = scope of the evidence.** "Across transformer families" requires >2 architectures. "Architecture-invariant" requires different-class architectures (transformer, SSM, recurrent). Never use scope language that outruns the evidence.

6. **The bilateral oracle is a scientific instrument, not a product.** Do not optimize it for deployment latency at the expense of measurement validity. The instrument's job is to produce clean labels. Speed is a Year 3 engineering problem.

7. **Theory must be updated before the next experiment design.** After every experiment, update the relevant sections in thesis.md and CLAIMS.yaml before running the next experiment. Theory lag is the program's most expensive failure mode.

8. **Priority = expected information gain.** When choosing between two experiments that can run in the same time, run the one whose result changes the scientific picture more. Experiment 0 (ablation) runs before Gemma because its result is more consequential.

---

## Quantitative Success Metrics

The program succeeds if ALL FOUR of the following are true at ICLR 2027 submission:

**M1: Reproducibility.** Another lab can re-run the bilateral oracle protocol unchanged on any standard QA dataset using the open-source tool, achieving AUROC within ±0.05 of the published result. This requires: open-source code, documented protocol, reproducible dataset sampling.

**M2: Scope characterization.** The paper explicitly characterizes the scope of every major claim: which architectures, which model sizes, which task types, which training regimes. No claim is stated more generally than the evidence supports.

**M3: Durable instrument.** At least one measurement protocol or empirical regularity established by this program is cited by ≥3 papers outside this program within 18 months of publication. The bilateral oracle and its L1/L2 hierarchy are the candidates.

**M4: Falsifiable claims.** Every CONFIRMED or SUPPORTED claim in CLAIMS.yaml at submission time has a pre-specified falsification condition. No claim without a falsifier appears in the paper.

The program does not succeed if:
- Only one architecture is tested
- Fisher gap claims are not entropy-controlled
- Bilateral oracle protocol is not released
- Paper leads with Fisher AUROC numbers without stating the entropy-matched design

---

## Governance Loop (Run After Every Experiment)

After every experiment completes, in this order:

1. Update EXPERIMENTS.yaml (result, outcome, ladder level achieved)
2. Update CLAIMS.yaml (any claim status changes: EXPLORATORY→SUPPORTED, SUPPORTED→CONFIRMED, or FALSIFIED)
3. Update thesis.md §6/§7 if any numbers changed
4. Check termination rules: has any rule T1–T5 been triggered?
5. Decide: proceed with next experiment, or pivot to Paper B/C narrative?

Do not run the next experiment until this loop completes for the current one. The loop is the scientific record.

---

*End of Research Plan v2. Version 1.0. July 2026.*

*This document will not be revised until the experiment queue is complete. If experimental results force a decision not covered by the decision trees above, consult the Program Invariants before taking any action.*
