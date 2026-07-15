# Program C: Reasoning Regimes — Core Theorem

## Theorem statement

**Distinct Epistemic Commitment Architectures (empirical, FULLY CONFIRMED)**

Different post-training regimes produce fundamentally distinct epistemic commitment
architectures, measurable as trajectory descriptors of J_know during generation.
The causal chain (base → instruct → RL) and Regime 2 identity are fully confirmed.

**Regime 0 (Base / no alignment)**:
  commit_rate ≈ 0.4–0.5, z_score < 2 — weak, inconsistent commitment geometry.
  Post-training is necessary to organize latent epistemic structure into coherent patterns.

**Regime 1 (SFT + RLHF)**:
  commit_rate > 0.85, z_score > 10 — strong mid-generation commitment.
  Low exploration_intensity. Commitment event temporally localized mid-CoT.
  cw_j_gap > 0 — correct answers show higher J_know throughout generation.

**Regime 2 (RL reasoning / process-reward) — CONFIRMED (v6, 2026-06-14)**:
  Behavioral signature: HIGH exploration during CoT + SUDDEN commitment at answer onset.
  crossing_count = 198 (vs 71 for Regime 1) — model holds competing hypotheses throughout.
  answer_jump = 5.54 (n=17/17) — J_know rises sharply at </think> boundary.
  probe_dir_stability = 1.0000 — geometry is clean and stable at N_CAL=50.
  Monitoring implication: epistemic state is NOT readable from CoT tokens. 
  The </think> boundary is the only reliable extraction point for RL-trained models.

**Regime 3 (MQA architecture)**:
  cal_auroc > 0.75, generation J_know falls outside calibrated range.
  Architectural distribution gap, not training-regime effect.

---

## Causal chain + answer-onset evidence (v5+v6, COMPLETE)

### Causal chain (v5 — same backbone, three regimes)

| Model | Training | commit_rate | z_score | j_know_mean | exploration_intensity |
|---|---|---|---|---|---|
| Qwen2.5-7B base | None | 0.467 | 1.94 | 0.169 | 0.248 |
| Qwen2.5-7B-Instruct | SFT+RLHF | 0.950 | 23.51 | 0.489 | 0.289 |
| DeepSeek-R1-Distill-Qwen-7B | RL | 1.000 | 369.18 | 1.838 | 0.716 |

Architecture: identical Qwen2.5-7B. Training is the only variable. z_score and
exploration_intensity both increase monotonically with training intensity.

### Answer-onset test (v6 — Regime 2 confirmation)

| Metric | DeepSeek-R1-Distill-Qwen-7B | Threshold |
|---|---|---|
| probe_dir_stability | **1.0000** | — |
| n_think_completed | 17/20 | — |
| n_answer_jumps | 17/17 | min 5 |
| **answer_jump** | **5.5402** | > 1.0 |
| crossing_count (during CoT) | 198.15 | — |
| cw_j_gap | 2.863 | — |

Verdict: **REGIME_2_CONFIRMED**. RL model commits at answer onset after sustained exploration.
This was not detectable in v5 (n=3 completions with 300-token budget).

---

## Central falsifiability criteria — ALL CONFIRMED

**1. Causal chain** (CONFIRMED v5):
- Monotone z_score (1.94→23.51→369.18): training determines epistemic geometry ✓
- base=Regime 0, instruct=Regime 1, RL≠Regime 0 ✓

**2. Regime 2 answer_jump test** (CONFIRMED v6):
- answer_jump = 5.54 > 1.0 ✓
- n=17/17 (100% of completed think blocks show the jump) ✓
- Regime 2 = commitment DEFERRED to answer onset, not absent

**3. Probe stability test** (CONFIRMED v6):
- N_CAL=50 per class: probe_dir_stability = 1.0000 (cosim across 3 seeds) ✓
- v4/v5 discrepancy resolved: was N_CAL too small, not a geometry problem ✓

---

## Why this matters

Most reasoning research assumes CoT unfolds progressively toward a conclusion.
If Regime 2 is confirmed (commit at answer onset): RL-trained reasoning models maintain
competing latent hypotheses throughout CoT and commit only at answer onset. This changes:

- **PRM design**: PRMs trained on CoT traces may reward narration quality, not epistemic
  accuracy. The epistemic state at answer onset (not the CoT trajectory) is what matters.
- **CoT faithfulness**: For RL-trained models, faithfulness should be evaluated at answer
  onset, not averaged over the full chain.
- **Alignment monitoring**: Monitoring J_know during CoT is insufficient for Regime 2
  models. Monitoring must shift to the answer-onset token.

If Regime 2 is NOT confirmed (globally non-committed): the finding is equally important —
RL-trained models lose the epistemic commit signal entirely, making them harder to monitor.

---

## Files

- `reasoning_regimes.md` — full experimental record with v5 update
- `regime_taxonomy.md` — formal descriptor definitions
- `experiments/rl_regime_collapse_v4/` — v4 results (bilateral calibration)
- `experiments/rl_regime_collapse_v5/` — v5 results (causal chain)
- `experiments/rl_regime_collapse_v6/` — v6 (planned: N_CAL=50, MAX_GEN=1500)
- `results/master_results.json` → `reasoning_regimes` entry
