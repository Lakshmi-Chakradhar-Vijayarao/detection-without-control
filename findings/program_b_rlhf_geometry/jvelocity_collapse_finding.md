# Finding 2b: J_velocity Collapse — RLHF Trains Away Epistemic Monitoring

## Status: PRIMARY FINDING — one architecture pair measured, replication required

---

## One-line result

```
r(J_velocity, correctness) = 0.985  →  0.374   [Qwen2.5-1.5B, base → instruct]
```

RLHF training trains away the model's internal generation-time epistemic monitoring.
The base model dynamically tracks whether its generation is epistemically grounded.
After alignment training, this tracking collapses.

---

## What J_velocity measures

J_velocity = rate of change of J_know during generation. Specifically:
`Var(J_L14..L18..L24..L26)` across the trajectory — the layer-to-layer momentum of the
epistemic projection during the forward pass.

A high J_velocity means the model's epistemic state is changing rapidly during generation —
it is actively resolving something. A low J_velocity means the epistemic state is static
(either committed from the start, or uniformly flat).

**Critical distinction**: J_velocity is NOT the same as AUROC attenuation (Finding 2).
- AUROC attenuation = does the probe separate PARAM from CTX_DEP at step-1? (static measurement)
- J_velocity collapse = does the model's epistemic state actively evolve during generation? (dynamic measurement)

These measure different phenomena. The AUROC can stay high (the signal exists) while
J_velocity collapses (the model stops tracking it dynamically).

---

## Results

| Model | r(J_velocity, correctness) | Interpretation |
|---|---|---|
| Qwen2.5-1.5B **base** | **0.985** | Nearly perfect dynamic epistemic monitoring |
| Qwen2.5-1.5B **instruct** | **0.374** | Monitoring collapsed; generation is uniformly confident |

The base model's trajectory variance (how much J_know changes during a generation)
correlates with correctness at r=0.985. The model "knows" when it's on track.

After RLHF, this collapses to r=0.374. The model generates confidently regardless
of whether it should.

---

## Mechanistic interpretation

**Base model**: The model is performing active epistemic monitoring during generation.
As it generates, J_know updates — rising for answers the model "knows," falling for
answers where it should be uncertain. The trajectory variance is the signal of this
ongoing monitoring process. r=0.985 with correctness means this monitoring is nearly
perfectly predictive.

**Instruct model (after RLHF)**: Human preference optimization trains against expressing
uncertainty in output. The behavioral signal for correctness (tone, hedging, verbosity)
becomes more uniform and confident-sounding. This training propagates backwards into
the generation dynamics: J_velocity is suppressed because the model no longer needs to
actively track its epistemic state — it will generate confidently regardless.

**The implication**: RLHF doesn't just attenuate the static step-1 probe signal (Finding 2).
It eliminates the dynamic tracking behavior that the base model uses during generation to
monitor whether it is epistemically grounded. The model stops "checking itself."

---

## Why this is potentially the most important finding in the program

Every adaptive inference system (CALM, SkipDecode, TARG, SeaKR) assumes that output signals
can detect when a model is uncertain. The J_velocity collapse shows that RLHF systematically
trains away the **internal process that would generate such signals**. The model doesn't
just hide its uncertainty — it loses the internal mechanism that produced accurate uncertainty
tracking in the first place.

This is a direct, specific, measurable mechanism for confabulation: RLHF trains models to
generate confidently because confident outputs receive higher human preference scores.
The side effect is that the internal process that monitored correctness during generation
is weakened proportionally.

---

## Current limitations

**CRITICAL**: This result is measured on ONE model pair (Qwen2.5-1.5B base vs instruct).
One data point is not sufficient to claim a general phenomenon.

What is needed before this can be elevated to a primary claim:
1. **Replication on 4+ architecture families** — Llama-3.2, Mistral-7B, Phi-3.5, Gemma-2
2. **Confirmation the effect size is consistent** — is Δ = -0.611 (0.985 → 0.374) consistent or varies by architecture?
3. **Checkpoint sweep** — does J_velocity correlate with RLHF training steps? (Pythia OLMo checkpoints)
4. **Causal test** — does suppressing J_velocity (by activation patching) cause the model to confabulate more?

---

## Relationship to other findings

| Finding | What it measures | Relationship to J_velocity |
|---|---|---|
| AUROC attenuation (Finding 2) | Static step-1 probe AUROC | J_velocity measures the dynamic version of the same degradation |
| Step-1 spike (E3) | Commitment onset at step-1 | J_velocity explains why step-1 is special: maximum epistemic flux |
| Answer_jump (Claim 3) | Deferred commitment in RL reasoning | J_velocity is the per-step version; answer_jump is the aggregate event |
| Intelligence-observability tradeoff | Framing | J_velocity collapse is the first mechanistic evidence for the tradeoff |

---

## Required next experiment

**J_velocity multi-architecture test** (Paper 2 lead experiment):

Protocol:
- 5 model pairs: Qwen2.5-1.5B, Llama-3.2-3B, Mistral-7B, Phi-3.5, Gemma-2-2B (base + instruct each)
- Measure r(J_velocity, correctness) for EACH model at BOTH training stages
- Use bilateral oracle labels (same as main study)
- Compare Δ r across architecture families
- Report: is J_velocity collapse (a) general, (b) architecture-stratified, or (c) Qwen-specific?

If general → strongest mechanistic support for RLHF-induced epistemic monitoring collapse.
If Qwen-specific → document as Qwen-family finding, investigate why Qwen's base model has
such strong dynamic monitoring (hypothesis: Qwen pre-training includes more step-by-step
reasoning data than other architectures).

---

## Files

- `rlhf_attenuation.md` — J_velocity data appears as "additional evidence" (Mechanism A section)
- `theorem.md` — J_velocity appears as "supplementary" in the evidence table
- `jvelocity_loss_hypothesis.md` — training objective to PRESERVE J_velocity (separate use case)
- Memory: `research_gaps_critical.md` → GAP 3 explanation
