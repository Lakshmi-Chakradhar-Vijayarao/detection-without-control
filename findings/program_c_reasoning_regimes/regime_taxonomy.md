# Reasoning Regime Taxonomy

## Formal definitions

Three distinct reasoning geometries have been observed empirically across model families.
Classification is based on six trajectory descriptors measured during generation.

---

## Descriptor definitions

| Descriptor | Formula | Interpretation |
|---|---|---|
| `exploration_intensity` | mean(std(J_know[0:T])) across traces | Variance of epistemic state during generation |
| `crossing_count` | mean(count(J > θ) per trace) | Discrete commitment events above threshold |
| `convergence_speed` | mean(var(J[0:T/2]) − var(J[T/2:T])) | +ve = converging; −ve = diverging |
| `answer_jump` | mean(J_answer_onset) − mean(J_CoT) | Deferred commitment at answer token (think models only) |
| `plateau_duration` | tokens with |ΔJ| < 0.01 after last crossing | Commitment stability post-event |
| `cw_j_gap` | mean(J_correct) − mean(J_wrong) | Epistemic differentiation between correct and wrong traces |

---

## Regime 1: Dynamic Convergence

**Exemplar**: Qwen2.5-7B-Instruct (SFT + RLHF)

**Definition**: The model transitions from exploratory to committed state during
generation. J_know crosses the commitment threshold mid-generation and stabilizes.

**Signature**:
- commit_rate > 0.5, z_score > 5
- crossing_count > 10 (per trace mean)
- convergence_speed > 0 (variance decreasing second half)
- answer_jump ≈ N/A (no </think> phase)
- cw_j_gap > 0.2 (correct answers show higher mean J_know)

**Empirical values (v4)**:
```
commit_rate  = 0.675    z = 7.29
j_know_mean  = +0.261   j_know_max = +3.74
cw_j_gap     = +0.304   traj_var   = 0.396
```

**Interpretation**: The model genuinely searches during generation. The commitment
moment is detectable and temporally localized (~40% through generation on average).
This is consistent with generation-time epistemic computation.

**Implication**: Early-exit optimization is applicable. When J_know crosses θ and
stabilizes, subsequent tokens add narration quality, not epistemic content.

---

## Regime 2: Answer-Deferred Commitment

**Exemplar**: DeepSeek-R1-Distill-Qwen-7B (RL reasoning)

**Definition**: The model maintains persistent latent exploration throughout CoT.
J_know stays far below the commitment threshold during the reasoning chain.
Commitment, if it occurs, is deferred to the answer-onset token (after </think>).

**Signature**:
- commit_rate = 0.0 (no mid-generation crossings)
- j_know_mean << 0 (sustained exploration state; v4: −8.3)
- j_know_max < 0.1 (never approaches PARAM region during CoT)
- traj_variance high (1.8+ in v4; active trajectory movement)
- answer_jump >> 0 (v5 hypothesis: commitment spike at </think>)
- probe_auroc high at calibration (0.876 in v4): epistemic structure exists

**Empirical values (v4)**:
```
commit_rate  = 0.000    z = 0.00
j_know_mean  = −8.307   j_know_max = +0.042
cw_j_gap     = −0.092   traj_var   = 1.876
probe_auroc  = 0.876    (calibration AUROC — signal exists)
```

**Interpretation**: RL training (process reward optimization) creates sustained
deliberation during CoT. The model maintains competing latent hypotheses rather
than converging. The commit-at-answer hypothesis (v5) tests whether J_know spikes
at </think> — if confirmed, this is the mechanistic signature of Regime 2.

**Implication**: Generation-time commitment detection is inapplicable. Monitoring
point must shift to answer onset or prefill. The reasoning chain is not epistemic
narration of a committed state — it is active latent search.

---

## Regime 3: Calibration-to-Generation Distribution Gap

**Exemplar**: Gemma-2-2B-IT (MQA architecture)

**Definition**: The Fisher probe achieves good calibration AUROC (signal exists at
step-1), but generation-time hidden states occupy a different activation regime
from calibration-time states. J_know is persistently negative during generation
even on correct answers.

**Signature**:
- probe_auroc > 0.75 at calibration (signal exists)
- j_know_mean << 0 during generation (v17: −1.149)
- j_know_max < mu_CTX_DEP (never reaches either calibration class)
- commit_rate = 0.0
- Root cause: architectural (MQA attention pattern alters generation-time hidden state
  distribution relative to step-1 prefill)

**Empirical values (Gemma v17)**:
```
cal_auroc    = 0.830    commit_rate = 0.000
j_know_mean  = −1.149   j_know_max  = +1.819
mu_PARAM     = +25.36   mu_CTX      = +1.71
```

**Interpretation**: The epistemic signal is present at step-1 but MQA's attention
mechanism causes generation-time activations to drift out of the calibrated
subspace. This is an architectural effect, not a training-regime effect.

**Implication**: Standard Fisher probe at fixed layer is insufficient for MQA.
Architectural adaptation needed (potentially cross-layer aggregation or different
extraction point).

---

## Regime classification procedure

Given a new model, run the following sequence:

1. **Calibrate**: Fisher LDA AUROC at gen-step-1. If < 0.65, signal absent — stop.
2. **Trajectory scan** (n=60): collect all 6 descriptors.
3. **Classify**:
   - `commit_rate > 0.4 AND cw_j_gap > 0.15` → **Regime 1**
   - `commit_rate < 0.05 AND traj_variance > 1.0` → **Regime 2 candidate** → run v5 answer-onset test
   - `calibration AUROC > 0.7 AND j_know_mean < −1.0 AND j_know_max < mu_CTX_DEP` → **Regime 3**
   - None of the above → **Unclassified** (needs deeper investigation)

---

## Open questions (v5 targets)

1. **Regime 2 confirmation**: Does answer_jump >> 0 for DeepSeek? (v5 primary test)
2. **Causal attribution**: Does the base model (no RL) show Regime 1 on the same backbone?
   If yes, RL training alone is sufficient to induce Regime 2 on the same architecture.
3. **Regime 3 fix**: Does cross-layer aggregation (mean of L18-L26) recover the signal for MQA models?
4. **Regime 1 preservation**: Can JVelocityLoss training objective maintain Regime 1 geometry during RLHF fine-tuning?

---

## Experiment record

| Experiment | Status | Key finding |
|---|---|---|
| ESM main (Qwen2.5-1.5B) | COMPLETE | Regime 1 confirmed, commit_rate=0.54 |
| reasoning_geometry_llama_v1 (21 versions) | COMPLETE | Prior Regime 2 claim for Llama-8B — reinterpreted in v4 |
| gemma_geometry_v17 | COMPLETE | Regime 3 confirmed for MQA |
| rl_regime_collapse_v4 | COMPLETE | Qwen=Regime 1 (z=7.29), DeepSeek=Regime 2 candidate |
| rl_regime_collapse_v5 | PENDING | Causal chain + answer_jump test |
