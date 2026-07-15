# Findings

Three research programs. See `RESEARCH_PROGRAM.md` at the repo root for the full
strategic context, roadmap, and experiment queue.

---

## Program A — Structural Independence (Claim 1)

**Core claim**: Hidden-state epistemic geometry is structurally orthogonal to output-space
uncertainty. The information about knowledge state exists in geometric structure that behavioral
signals cannot recover. Text ceiling gap = 0.30.

| Key number | Value |
|---|---|
| r(J_know, entropy) | +0.0039, 95% CI [−0.066, 0.073], n=800 |
| Fisher AUROC | 0.989 (TriviaQA, Qwen2.5-7B, L26) |
| SBERT text ceiling | 0.570 |
| Cross-task cosim | 0.004–0.035 (noise floor ~0.016, task-specific geometry) |
| Probe selectivity | shuffled 0.543 vs real 1.000 (Hewitt-Liang PROBE_SELECTIVE) |

**Files:**
```
program_a_output_ceiling/
  independence_result.md    ← r=0.0039, full evidence
  theorem.md                ← formal statement + replication requirements
```

---

## Program B — RLHF Attenuation (Claim 2)

**Core claim**: RLHF alignment compresses epistemic accessibility in a measurable, systematic
way. The compression follows baseline AUROC (not architecture label). Alignment is a
transparency tax. RLHF does not destroy the information — it closes the observable window.

| Key number | Value |
|---|---|
| Sign test (5 non-ceiling pairs, all Δ<0) | p = 0.031 |
| Strong baseline Δ | −0.036 |
| Weak baseline Δ | −0.183 (near-destroyed) |
| Commitment trajectory collapse | r = 0.985 → 0.374 (base→instruct) |
| RLHF masking Δ cross-family | 0.0359 |

**Open**: Pythia/OLMo checkpoint sweep (accessibility as a function of RLHF steps, not just
base vs instruct) is the key experiment for Paper 2.

**Files:**
```
program_b_rlhf_geometry/
  rlhf_attenuation.md             ← sign test + Δ values across 6 families
  jvelocity_collapse_finding.md   ← J_velocity r=0.985→0.374 PRIMARY FINDING (added 2026-06-16)
  jvelocity_loss_hypothesis.md    ← training objective to preserve geometry (separate use case)
  theorem.md                      ← formal statement + open questions
```

---

## Program C — Reasoning Regimes (Claim 3)

**Core claim**: Post-training regime determines commitment geometry. RL reasoning models
exhibit deferred commitment — J_know stays low through the CoT chain and jumps at the
`</think>` boundary. This is an EPISTEMIC_TRANSITION, not a mode-change confound.

| Key number | Value |
|---|---|
| answer_jump (Qwen backbone) | 5.54 ± 5.10, n=17, t=4.48, p=0.0002 |
| Probe direction stability | 1.0000 (cosim, 3 seeds) |
| Control (CoT vs generic) | +26.2 vs −11.7 (opposite sign → confound ruled out) |
| answer_jump_v2 (Llama backbone) | REGIME_2_PARTIAL — 0.101 (positive, below threshold); n=20/20 all positive; backbone-stratified magnitude |

**Files:**
```
program_c_reasoning_regimes/
  reasoning_regimes.md    ← full experimental record
  regime_taxonomy.md      ← descriptor definitions + classification procedure
  theorem.md              ← formal statement + falsifiability criteria
```

---

## Boundary Finding — Gemma 3 Accessibility Fragmentation

**NOT YET IN PAPER — add to §4 before submission.**

Gemma 3 (multimodal pre-trained) failed to exhibit unimodal epistemic geometry at L26.
This is the first observed instance of accessibility fragmentation under multimodal
pre-training. The information may still be present internally; the single-stream
unimodal accessibility surface has reorganized.

This should be framed as a boundary condition finding, not a probe failure.

---

## Status

| Program | Core result | Paper 1? | Key open |
|---|---|---|---|
| A | r=0.0039, AUROC 0.989 | Yes — Claim 1 | Nemotron-H (architecture generalization) |
| B | Δ attenuation, p=0.031 sign test | Yes — Claim 2 | Pythia checkpoint decay curve |
| C | answer_jump 5.54, p=0.0002 | Yes — Claim 3 | answer_jump_v2 (Llama backbone) |
| Gemma 3 | Accessibility fragmentation | Pending add | Multimodal accessibility program |
