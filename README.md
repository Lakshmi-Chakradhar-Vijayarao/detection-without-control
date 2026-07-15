# Detection Without Control

**The hidden-state direction that best separates confident-correct from confident-wrong answers is not the direction that causally controls which answer the model generates.**

This repository accompanies the paper:

> **Detection Without Control: Confabulation Geometry Is Dissociable from Causal Answer Generation in Language Models**  
> Lakshmi Chakradhar Vijayarao — July 2026

---

## What this paper shows

A Fisher+PCA64 linear probe at layer 26 detects confident-wrong answers with AUROC = 0.854 (gap = 0.240 over entropy, gap = 0.166 over self-consistency with N=5 samples) in an entropy-matched zone where output statistics are blind by construction.

Centroid-patching along the Fisher LDA class-mean axis produces max ΔF1 = +0.0004 across all tested layers and patch magnitudes (EXP-H) — indistinguishable from zero.

The two axes are empirically dissociable: one reads whether an answer is correct; the other writes which answer gets generated. This is consistent with Cox et al. (2026), who found causal control via a *different* direction (the committed-answer direction). The present work establishes that the Fisher detection direction is not that direction.

---

## Repository structure

```
paper/          LaTeX source for the paper (main.tex, figures/)
science/        CLAIMS.yaml — full claims registry (C001–C050)
                EXPERIMENTS.yaml — experiment record
experiments/    Kaggle scripts for all reported experiments
```

---

## Key results at a glance

| Task | Signal | AUROC | Baseline | Gap |
|------|--------|-------|----------|-----|
| L2 confabulation detection (Qwen, entropy-matched) | Fisher+PCA64 | 0.854 | Entropy 0.614 | +0.240 |
| L2 confabulation detection (Llama, entropy-matched) | Fisher+PCA64 | 0.818 | Entropy 0.453 | +0.365 |
| L2 large-N cross-validated (N=500/class, 5-fold) | Fisher+PCA64 | 0.763 ± 0.012 | — | C040 confirmed |
| L1 knowledge-source routing (5 architectures) | Fisher+PCA64 | 0.731–0.846 | Entropy 0.87–0.90 | Fisher < entropy at L1 |
| Causal patching (EXP-H) | Centroid patch | ΔF1 = +0.0004 | — | Not significant |

---

## Honest limitations

- EXP-H tests one intervention family (centroid patching on residual stream). Attention-head patching, SAE feature patching, and circuit-level interventions are untested.
- The difficulty confound (Q14) is unresolved at L1: PARAM items and easy items share low entropy; the experiment to separate these (Q14v4, TriviaQA train split) has not yet been run.
- All L2 experiments use models ≤ 8B parameters. Whether the Fisher advantage persists at larger scale is unknown.
- Llama plain CO labeling at L2 is INCONCLUSIVE (C047: AUROC = 0.613, shuffled = 0.580).

---

## Reproducibility

All experiments run on Kaggle (T4 GPU, 15.3 GB VRAM). Scripts in `experiments/` are self-contained. Each script records: random seed, question source, sampling method, pool size, and N/class.

The claims registry (`science/CLAIMS.yaml`) tracks every claim with status (CONFIRMED / SUPPORTED / EXPLORATORY / FALSIFIED), evidence, and kill criteria. Four claims are FALSIFIED and documented in §7 of the paper.
