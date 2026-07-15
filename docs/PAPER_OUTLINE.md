# Paper Outline — ArXiv + ICLR 2027 Submission

**Title (current):** Epistemic Legibility in Transformer Language Models: A Three-Task Measurement Hierarchy  
**Short title:** Three-Task Epistemic Hierarchy  
**Author:** Lakshmi-Chakradhar Vijayarao  
**Affiliation:** Khoury College of Computer Sciences, Northeastern University  
**Target venue:** ICLR 2027 (primary)  
**ArXiv preprint target:** 2026-Q4 (after EXP-F/H/K complete + EXP-J)  
**Abstract deadline:** ~2026-09-26  
**Full submission:** ~2026-10-03  
**Last updated:** 2026-07-07

**Draft:** see [paper/v1_arxiv.md](../paper/v1_arxiv.md) — complete working draft with all current numbers.

---

## ⚠️ Stale Numbers Archive

The following numbers appeared in prior outlines and are **FALSIFIED or SUPERSEDED**. Do not use.

| Old number | What it was | Why wrong | Replacement |
|---|---|---|---|
| AUROC = 0.989 | Fisher primary result | Calibration-phase artifact (N=50–80, ordering bug) | 0.7312 / 0.7464 (large_n_v2) |
| r = 0.0039 | Fisher ⊥ entropy correlation | Used J-score not Fisher+PCA64 decision score | Qwen r=−0.225; Llama r=−0.544 p<0.0001 (C008 FALSIFIED) |
| RLHF Δ = −0.036 | RLHF attenuation | Degenerate Fisher LDA at small N | C012 FALSIFIED — instruct > base (+0.127) |
| Llama AUROC = 0.629 | Llama weakness | Same degenerate estimator | C013 FALSIFIED — Llama = 0.846, same as Qwen |
| Behavioral AUROC = 0.510 | Entropy baseline | Used wrong comparison | Entropy AUROC = 0.87–0.90 (C016 SUPPORTED) |

---

## Current Core Thesis (2026-07-07)

Transformer language models contain epistemic information in their residual stream that is partially but not fully accessible from their output distributions. We characterize this accessibility precisely through a **three-task measurement hierarchy**:

- **Task-L1 (knowledge-source routing):** Output entropy is sufficient — Fisher is redundant at this level. Entropy achieves AUROC 0.87–0.90 for PARAM vs CTX_DEP classification.
- **Task-L2 (confabulation detection):** Fisher is essential at this level. Within the entropy-matched confident zone, Fisher adds 0.240 AUROC beyond entropy (0.854 vs 0.614, C017).
- **Task-L3 (commitment timing in reasoning chains):** Fisher trajectory is required. Reasoning-distilled models commit to their answer direction after 17–18% of think tokens; the rest is elaboration (C022).

The hierarchy's key contribution: the appropriate measurement signal changes at each level, and the epistemic difficulty increases at each level. Entropy handles Task-L1; only Fisher handles Task-L2 and Task-L3.

---

## Current Numbers (authoritative — do not change without re-running)

```
# Task-L1: Knowledge-source routing
Fisher AUROC (Qwen, large_n_v2)    = 0.7312  CI=[0.63,0.83]  N=197/class
Fisher AUROC (Llama, large_n_v2)   = 0.7464  CI=[0.65,0.83]  N=200/class
Entropy AUROC (Qwen)               = 0.9043  (entropy > Fisher at L1 for Qwen)
Entropy AUROC (Llama)              = 0.874   (entropy ≈ Fisher at L1 for Llama)
Combined AUROC (Llama)             = 0.9037  (marginal Fisher independent component)
Cross-arch AUROC difference        = 0.015   (CIs overlap fully; both CLEAN)

# Task-L2: Confabulation detection
Fisher+PCA64 CC vs CW AUROC        = 0.854   (Qwen, EXP-A, N=100/class, θ_conf=1.1043)
Entropy AUROC (entropy-matched)    = 0.614
Gap (Fisher over entropy)          = 0.240   ← key L2 finding
BO_Transfer AUROC                  = 0.880   (bilateral oracle probe → CC/CW, C018)
Entropy trajectory AUROC           = 0.730   (15-step, EXP-D, inversion at step 2–3, C019)

# Task-L3: Commitment timing
Qwen 1.5B commit_pct (mean)        = 75.8%   z=49.77   commit_rate=80/100
Llama 8B commit_pct (mean)         = 82.9%   z=679.73  commit_rate=80/100
Cal AUROC (Qwen L26)               = 0.760   (EXP-B)
Cal AUROC (Llama L28)              = 1.000★  (EXP-C, N=10/class — small-N saturation)

# Supplementary
Cross-arch step-1 mean             = 0.785   (ESM v33, 4 model families, lighter calibration)
Cross-arch prefill mean            = 0.567   (same)
Qwen-Llama family gap (EXP-E)      = 0.237   (Qwen 0.845 vs Llama 0.608 at same param scale)
Fisher ⊥ entropy correlation       = FALSIFIED — Qwen r=−0.225, Llama r=−0.544
```

---

## Paper Structure (v2, reflecting current science)

### 1. Introduction (~1.5 pages)

**Opening:** The epistemic routing decision — *can I answer from my weights, or do I need context?* — happens in the residual stream before output tokens are decoded. Output monitoring cannot access it after the fact.

**The three-task problem:** Not all epistemic distinctions are equally observable, and different signals are required at different levels. This paper characterizes exactly three levels and what each requires.

**Contributions:**
1. Bilateral oracle protocol — clean PARAM/CTX_DEP labeling (§2)
2. Task-L1 result: entropy achieves 0.87–0.90, Fisher redundant (§3)
3. Task-L2 result: Fisher essential, gap = 0.240 within confident zone (§4)
4. Task-L3 result: commitment timing architecture-invariant, 75–83% elaboration (§5)
5. Safety implication: Gate 3 (L2/L3 Fisher) required before confident routing is safe (§6)

---

### 2. Methods (~0.75 pages)

**2.1 Bilateral Oracle Protocol**
- Two-pass labeling: PARAM (nc_F1 ≥ 0.50), CTX_DEP (nc_F1 ≤ 0.05 AND wc_F1 ≥ 0.50)
- Items not meeting either criterion excluded
- Hidden states always from nocontext pass

**2.2 Probe Architecture**
- PCA(n_components=64) → LDA(solver='lsqr', shrinkage='auto')
- Layer 26, step-1 (generation step 0 = prefill, step 1 = first decode step)
- Shuffled control mandatory; bootstrap CI at N ≥ 100

**2.3 Models**
- Task-L1/L2: Qwen2.5-1.5B-Instruct, Llama-3.2-3B-Instruct
- Task-L3: DeepSeek-R1-Distill-Qwen-1.5B, DeepSeek-R1-Distill-Llama-8B

---

### 3. Task-L1: Knowledge-Source Routing (~1 page)

**Main result (Table 1):**

| Model | N/class | Fisher AUROC | Shuffled | Entropy AUROC | Fisher vs Entropy |
|---|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct | 197 | 0.7312 [0.63,0.83] | 0.598 CLEAN | 0.9043 | Fisher ≪ Entropy |
| Llama-3.2-3B-Instruct | 200 | 0.7464 [0.65,0.83] | 0.502 CLEAN | 0.874 | Fisher ≈ Entropy |

**Honest framing:** Fisher at L1 is redundant with entropy. This is the correct characterization. The bilateral oracle's contribution is the labeling protocol, not the probe.

**The Goldilocks zone (EXP-E):** Protocol is scale-constrained: Qwen 0.5B PARAM floor, Qwen 3B CTX_DEP ceiling. Valid range ~1B–2B on TriviaQA (C021).

---

### 4. Task-L2: Confabulation Detection (~1.25 pages)

**The entropy-matched setup:** Items collected at θ_conf=1.1043. Both CC and CW have low entropy — entropy discrimination controlled out by design. This isolates the hidden-state dimension.

**Main result:**
- Fisher+PCA64 AUROC = 0.854 (C017)
- Entropy AUROC = 0.614 (residual after matching)
- Gap = 0.240 — **Fisher is genuinely essential at L2**

**Transfer result:**
- BO_Transfer AUROC = 0.880 (C018) — bilateral oracle probe transfers to CC/CW
- Transfer > direct: CONFAB items occupy same geometric region as CTX_DEP items

**Entropy trajectory (EXP-D):**
- Trajectory LR AUROC = 0.730 (C019)
- Inversion pattern: step-0 AUROC = 0.331 (below chance) → step-4 = 0.617
- CW burst (entropy rising 0.75→1.57); CC flat

**Competing interpretations:** Three models remain viable (see §5.3). No premature selection.

---

### 5. Task-L3: Commitment Timing in Reasoning Chains (~1 page)

**The commit point:** Fisher trajectory within think block detects transition from exploration to committed direction. commit_pct = fraction of think block *after* commit point.

**Main result (Table 2 — Architecture Invariance):**

| Model | Family | commit_pct (mean) | z-score | Verdict |
|---|---|---|---|---|
| DeepSeek-R1-Distill-Qwen-1.5B | Qwen | 75.8% | 49.77 | COMMITTED_EARLY |
| DeepSeek-R1-Distill-Llama-8B | Llama | 82.9% | 679.73 | COMMITTED_EARLY |

**Interpretation:** ~80% of think block is post-commitment elaboration. The think block is not search — it is primarily justification. Architecture-invariant across two independent reasoning families.

**f1_delta:** mean(f1_full − f1_early) = +0.008 for diverging traces. Positive = full run marginally better. Post-commitment elaboration provides marginal F1 benefit on diverging cases.

**Z-score amplification:** five competing explanations registered (EXP-L will partially resolve). Do not attribute to single cause.

---

### 6. The Three-Task Hierarchy and Safety Implications (~0.75 pages)

**6.1 Why the hierarchy matters:**
- L1: Entropy sufficient — any output-monitoring system can do this
- L2: Fisher required — entropy was controlled out; this is where Fisher earns its keep
- L3: Fisher trajectory required — temporal dimension, single-point insufficient

**6.2 The safety gap:**
- Gate 1–2 (knowledge-source routing) ≠ Gate 3 (confabulation detection)
- A confident wrong answer passes Gates 1–2 (low entropy, Fisher near PARAM centroid)
- Gate 3 requires Fisher within the confident zone → prevents confident-wrong from passing
- For reasoning models: Gate 3 = commit-point Fisher check within think block (EXP-F pending)

---

### 7. Falsification Record (~0.5 pages)

Dedicate explicit space to all four falsified claims:
- C008: Fisher ⊥ entropy (FALSIFIED — r = −0.225 to −0.544)
- C012: RLHF attenuation Δ = −0.036 (FALSIFIED — estimator pathology)
- C013: Llama weakness AUROC = 0.629 (FALSIFIED — same pathology)
- C014: Nonlinear probe recovery Δ > 0.05 (FALSIFIED — contaminated baseline)

Frame: "The falsification record is not an appendix; it is the mechanism by which the measurement instrument earned its reliability."

---

### 8. Related Work (~0.75 pages)

**Output monitoring:** Kuhn et al. 2023 (semantic entropy), Kadavath et al. 2022 (P(IK)), Guo et al. 2017 (calibration) — all operate post-logits. L2 confabulation result is directly relevant: entropy fails within the confident zone where these methods also fail.

**Probing:** Alain & Bengio 2016, Tenney et al. 2019, Hewitt & Liang 2019 — bilateral oracle adds clean two-pass labeling protocol.

**Reasoning analysis:** Wei et al. 2022 (CoT), Kojima et al. 2022 — L3 commitment timing provides first hidden-state evidence about think-block structure.

**Concurrent work:**
- "LLM Reasoning as Trajectories" (arXiv:2604.05655, 2026): convergent discovery, ROC-AUC 0.87 (our Fisher trajectory = 0.9947)
- "Tell-Tale Norm" (arXiv:2606.06188, 2026): ℓ₂ norm signals reasoning dynamics — scalar vs our directional approach
- "Geometries of Truth Are Orthogonal Across Tasks" (Apple, arXiv:2506.08572, 2026): consistent with our cross-task findings (C009)

---

### 9. Conclusion (~0.25 pages)

The three-task hierarchy precisely locates where Fisher is essential (Task-L2: gap=0.240 over entropy in the confident zone) and where it is redundant (Task-L1: entropy equals or exceeds Fisher). The bilateral oracle protocol provides the clean labeling methodology that makes this comparison possible. The commitment timing result (Task-L3) opens a new measurement category: the temporal organization of epistemic state within reasoning chains.

---

## Pre-Submission Requirements

### EXP dependencies (must complete before arXiv)
- [ ] EXP-F complete (commit-point Fisher → CC/CW — fills §5 inference gap)
- [ ] EXP-H complete (causality check — either confirms epiphenomenal or shows leverage)
- [ ] EXP-J complete (perturbation battery — required for any generalizability claim)
- [ ] C017/C018 replicated on Llama (promotes L1→L2, critical for Task-L2 section)

### EXP dependencies (nice to have before ICLR)
- [ ] EXP-K complete (Pythia large-N — strengthens §3.2 scale discussion)
- [ ] EXP-G complete (reasoning entropy trajectory — enriches §5)
- [ ] EXP-I complete (early exit F1 — enables concrete efficiency claim in §6)

### Writing requirements
- [ ] All figures generated (Table 1, Table 2, entropy trajectory plot, commit trajectory example)
- [ ] References.bib complete (all concurrent papers above)
- [ ] Validate numbers against science/CLAIMS.yaml via `python science/validate_claims.py`
- [ ] Abstract ≤ 500 words
- [ ] Main body ≤ 8 pages (ICLR format) + unlimited references
- [ ] Appendix: Claims registry (Appendix A), Experiment registry pointer (Appendix B)

---

## HISTORICAL ARCHIVE — Prior Outline Content

*Preserved below for reference. Numbers and claims are STALE. Do not use.*

<details>
<summary>Prior outline (2026-06-15, stale numbers — click to expand)</summary>

Prior core thesis (now superseded by C008 falsification and large_n_v2):
> "Transformer language models contain epistemic information in their residual stream that
> is structurally orthogonal to their output distribution... r=0.0039... AUROC 0.989..."

Prior primary model: Qwen2.5-7B-Instruct (now superseded — primary is 1.5B/3B under large_n_v2 protocol).

Prior §4 (RLHF Geometric Attenuation): C012 and C013 are FALSIFIED. The RLHF attenuation framing is withdrawn. Do not re-add without re-running corrected experiments at N≥128/class.

Prior §6 (Routing Application): 0.9834 routing AUROC, 32.9% cost reduction — these were from earlier calibration-phase results with different models and may not hold under large_n_v2 protocol. Not included in current paper framing.

</details>
