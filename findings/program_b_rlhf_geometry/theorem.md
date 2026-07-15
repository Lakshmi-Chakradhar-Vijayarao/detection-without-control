# Program B: RLHF Geometry — Core Theorem

## Theorem statement

**RLHF Geometry — Baseline-Stratified (empirical, three mechanisms)**

RLHF alignment affects epistemic transparency through three distinct mechanisms
determined by **baseline epistemic geometry strength** (measured as probe AUROC),
not directly by attention architecture type.

**Mechanism A — Attenuation (strong baseline)**:
  RLHF scales down the Fisher LDA discriminant direction without rotating it.
  cosim(base_probe, instruct_probe) ≈ 0.80–0.85 (strong, same direction).
  Δ ≈ −0.036 (uniform across Qwen2.5-1.5B and Llama-3.2-3B).
  Condition: base_auroc > 0.65.
  Models observed: Qwen2.5-1.5B (GQA), Llama-3.2-3B (GQA).

**Mechanism C — Rotation without collapse (medium baseline)**:
  RLHF rotates the probe direction near-orthogonally, but the new direction retains
  most epistemic signal.
  cosim ≈ 0.04–0.08 (near-orthogonal). Small signal loss (Δ ≈ −0.02 to −0.05).
  Condition: base_auroc 0.55–0.62.
  Models observed: Mistral-7B-v0.1 (SWA, cosim=0.082, Δ=−0.050),
                   Yi-6B (GQA, cosim=0.037, Δ=−0.020).
  Note: Yi-6B is nominally GQA but has medium baseline — it follows the medium-baseline
  pattern, not the strong-GQA attenuation pattern.

**Mechanism B — Rotation with collapse (weak baseline)**:
  RLHF rotates the probe direction near-orthogonally AND collapses the signal severely.
  cosim ≈ 0.05 (near-orthogonal). Large signal loss (Δ ≈ −0.18).
  Condition: base_auroc < 0.55.
  Models observed: Gemma-2-2B (MQA, cosim=0.054, Δ=−0.183).

**Revised unified prediction (replaces architecture-based taxonomy)**:
  Primary predictor: baseline AUROC (not architecture label).
  base_auroc > 0.65  → Mechanism A: attenuation
  base_auroc 0.55–0.62 → Mechanism C: rotation w/o collapse
  base_auroc < 0.55  → Mechanism B: rotation + collapse

  Architecture label correlates with baseline strength (GQA tends to produce stronger
  baselines than MQA) but is not deterministic. Yi-6B demonstrates this: nominally GQA
  but low baseline (0.580) → rotation pattern, not attenuation.

---

## Evidence table (FINAL — 5 models, 5 orgs, 3 mechanisms)

| Family | Architecture | Base AUROC | Δ | cosim | Mechanism |
|---|---|---|---|---|---|
| Qwen2.5-1.5B | GQA | 0.899 | **−0.036** | ~0.85 (est.) | A: Attenuation |
| Llama-3.2-3B | GQA | 0.665 | **−0.036** | ~0.80 (est.) | A: Attenuation |
| Gemma-2-2B | MQA | 0.513 | −0.183 | **0.054** | B: Rotation + collapse |
| Mistral-7B-v0.1 | SWA | 0.590 | **−0.050** | **0.082** | C: Rotation w/o collapse |
| Yi-6B (01-ai) | GQA | 0.580 | **−0.020** | **0.037** | C: Rotation w/o collapse |

**Critical result**: Yi-6B (GQA) follows Mechanism C, not Mechanism A.
This falsifies the architecture-based prediction and confirms baseline AUROC is the correct predictor.

---

## J_velocity collapse (supplementary)

The commitment trajectory also collapses under RLHF (measured on Qwen2.5-1.5B):

| Model | J_velocity correlation with correctness |
|---|---|
| Qwen2.5-1.5B base | r = 0.985 |
| Qwen2.5-1.5B instruct | r = 0.374 |

RLHF trains uniform confident generation regardless of epistemic state. This
collapses the trajectory from strongly predictive to near-flat, separately from
the static step-1 AUROC effect.

---

## Implications for monitoring

Sorted by baseline AUROC:

- **Strong baseline (>0.65)** — Qwen2.5, Llama-3.2: monitorable after RLHF with base-model probe.
  Signal is weaker but axis is stable. Same probe transfers directly.

- **Medium baseline (0.55–0.62)** — Mistral-7B, Yi-6B: monitorable after RLHF, BUT probe
  must be recalibrated on instruct-model data. Base-model probe will underestimate signal
  because the direction has rotated near-orthogonally.

- **Weak baseline (<0.55)** — Gemma-2-2B: near-unmonitorable with base-model probe.
  Signal collapses to below-chance. Recalibration on instruct data required; may still be weak.

**Architecture label is insufficient to determine monitoring strategy.** Use baseline AUROC.
Yi-6B is nominally GQA but requires instruct-probe recalibration, same as Mistral-7B.

Epistemic geometry strength (baseline AUROC) is a concrete engineering parameter for
deciding probe deployment strategy before and after RLHF alignment.

---

## Open questions

1. **Direct cosim for Qwen/Llama (Mechanism A)** — currently estimated ~0.80–0.85.
   Measuring directly would confirm attenuation interpretation at the strong-baseline end.

2. **Why is Yi-6B baseline low despite being GQA?**
   Hypothesis: bilingual (Chinese/English) training domain → weaker geometry on TriviaQA English.
   Distinguishes architecture effect from training-data effect on baseline geometry.

3. **JVelocityLoss** — auxiliary loss to preserve GQA epistemic geometry during SFT

---

## Files

- `rlhf_attenuation.md` — full experimental record with three mechanisms
- `jvelocity_loss_hypothesis.md` — training objective for geometry preservation
- `esm/training.py` — JVelocityLoss implementation
- `experiments/rlhf_attenuation_yi_v1/` — pending (third GQA family)
- `results/master_results.json` → `rlhf_attenuation` entry
