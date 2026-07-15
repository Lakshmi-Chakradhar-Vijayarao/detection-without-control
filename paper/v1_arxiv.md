# A Measurement Protocol for Computational Observability in Language Models: A Three-Task Hierarchy

**Lakshmi Chakradhar Vijayarao**  
Khoury College of Computer Sciences, Northeastern University  
vijayarao.l@northeastern.edu

*Preprint — July 2026*

---

## Abstract

We introduce three measurable computational quantities — *Observability* (O), *Commitment* (C), and *Accessibility* (A) — together with experimental protocols and estimators for quantifying them across learning systems. O measures whether external probes can reliably read epistemic state from internal representations; C measures when internal computation settles on an answer; A measures how training shapes the computational organization that O and C reflect. The central methodological contribution is the **bilateral oracle protocol**, a two-pass labeling design that separates parametric knowledge (PARAM: answerable from weights alone) from context-dependent items (CTX_DEP: requiring external context) by behavioral intervention, without conflating epistemic state with output quality. Applying a Fisher+PCA64 probe at layer 26, step-1, we establish three levels of measurement. **L1 (knowledge-source routing):** AUROC ∈ [0.731, 0.846] across five independent architectures (Qwen, Llama, Gemma, Mistral, Phi, 1.5B–7B); output entropy dominates (0.87–0.90), Fisher is redundant at L1. **L2 (confabulation detection):** Fisher adds 0.240–0.365 AUROC over entropy baselines on entropy-matched CONFIDENT_CORRECT vs CONFIDENT_WRONG items (Qwen AUROC=0.854, Llama=0.818); behavioral baselines including self-consistency (0.613) and top-1 probability (0.384, *below chance*) fail; the sub-chance top-1 probability is consistent with RLHF-trained assertiveness producing output confidence that systematically diverges from underlying epistemic state in the confident zone (Qwen, single architecture). A perturbation battery yields ICC=0.913/0.933 (Qwen/Llama, between/within=10.5:1 and 14.0:1 respectively), confirmed across two independent architectures (C025). A T2_L2 boundary condition is characterized: entropy-matched L2 fails for architectures with θ_conf < 0.15 (Gemma, Mistral); CO labeling removes this dependency and establishes a stable cross-validated L2 estimate of 0.7629 ± 0.0120 (N=500/class, 5-fold CV, gap=0.163 over entropy). **L3 (commitment timing):** Reasoning-distilled models commit to their answer direction after 17–18% of thinking tokens (commit%=75.8%/82.9%, z=49.77/679.73); truncating at the commit point saves 87.4% of compute at +0.006 F1 cost (p=0.08); activation patching at the commit point is epiphenomenal (max Δf1=+0.0004) — geometry is a routing signal, not a control register. A teacher-independence replication on Qwen3-1.7B (Qwen-native reasoning, not R1-distilled) yields commit_pct=99.8% (commits within first ~2 think tokens, z=1.3×10¹⁵, N=100/100), establishing that early commitment is not specific to DeepSeek-R1 training. The entire think block is post-commitment elaboration (C045). A fourth measurement regime is established: at MATH step-1 before any reasoning begins, residual stream geometry predicts final mathematical correctness at AUROC=0.8558 (CI=[0.729,0.957], N=100/class; original small-N estimate was 0.9111) (layers 25–26, entry-point predictor). J_know at intermediate reasoning steps does not correlate with oracle PRM (r=0.053), confirming that hidden geometry captures approach-commitment, not computational progress. Four candidate scientific laws are proposed. An OOD generalization experiment establishes format-sensitive portability: the TriviaQA probe transfers to HotpotQA at 84.5% efficiency but transfers at chance to MMLU-STEM — all observability claims are format-scoped. A training-stage sweep (Base→SFT→Reasoning, matched N) reveals a split: L1 observability shows INVERTED_U (SFT +0.063, reasoning distillation −0.078 on a different task distribution), while L2 confabulation detection gap is MONOTONE_RISE through every stage (0.064→0.096→0.148). An open claims registry maintains 45 claims (7 CONFIRMED, 26 SUPPORTED, 8 EXPLORATORY, 4 FALSIFIED) with pre-specified falsification conditions for all pending experiments.

**Keywords:** confabulation detection, epistemic legibility, bilateral oracle, Fisher LDA, computational observability, commitment dynamics, knowledge-source routing, reasoning models

---

## 1. Introduction

The standard approach to detecting model uncertainty monitors output distributions: entropy, calibration, self-consistency, verbalized confidence. This works when the model *signals* uncertainty. It fails silently in the one case that matters most for safety: when the model is *confidently wrong*.

A model answering incorrectly with high certainty and a model answering correctly with high certainty are indistinguishable in output space — by construction. Both produce low entropy. Both pass entropy-based routing. Both will verbalize confidence around 80–90%. This is the confabulation problem, and it cannot be solved by monitoring outputs.

This paper demonstrates that hidden states at the generation onset encode geometry that discriminates CONFIDENT_CORRECT from CONFIDENT_WRONG items — with AUROC=0.854 on Qwen2.5-1.5B-Instruct (gap=0.240 over the entropy baseline of 0.614) and AUROC=0.818 on Llama-3.2-3B-Instruct (gap=0.365 over entropy baseline of 0.453). The entropy baselines are not coincidentally low — they are by design: items were selected *for* low output entropy, making this a test of exactly the case where output monitoring has already failed. Fisher+PCA64 applied to residual stream states before the first output token is decoded sees what the output distribution cannot.

**The bilateral oracle as measurement foundation.** The core methodological contribution is not the Fisher probe — it is the **bilateral oracle protocol**, a two-pass labeling design that separates knowledge-source from correctness by construction. The protocol tests each question twice: once without context (to verify the model *cannot* answer from weights alone) and once with context (to verify the model *can* answer when context is provided). This produces clean PARAM/CTX_DEP labels that exclude the ambiguous middle — items where the model might be using context even without it, or where both passes fail. Without this labeling discipline, probing work conflates epistemic state with output quality, and the confabulation result cannot be isolated.

**An unexpected discovery at L1.** When we apply this protocol to knowledge-source routing — classifying PARAM vs CTX_DEP items — output entropy achieves 0.87–0.90 AUROC, dominating Fisher+PCA64 (0.73–0.75). Knowledge-source routing is primarily a confident/uncertain distinction. This is a genuine empirical finding, not a failure: it characterizes what L1 actually measures, and it makes the confabulation finding structurally necessary. Fisher is not essential for a distinction that entropy can already make. Fisher is essential for the distinction that entropy cannot make.

**The underlying research program.** The three-task hierarchy is the first layer of a broader measurement science for three *computational quantities*. *Observability* (O): can external probes reliably read epistemic state from internal representations? *Commitment* (C): when does internal computation settle on an answer, and can that moment be detected and used? *Accessibility* (A): how does training — pretraining, SFT, RLHF, reasoning distillation — shape the computational organization that O and C measure? These are intended as *quantities*, not organizing concepts. The estimators (Fisher+PCA64, bilateral oracle, commit% probe) are current implementations that may be replaced by better instruments. O, C, and A persist as quantities across estimator changes — a future paper establishing a superior confabulation probe would be measuring the same O, better. This paper reports substantial evidence on O (L1–L2, §3–4) and C (L3, §5), and preliminary evidence on A (§6.3). The underlying causal chain the program tests: Optimization → Computational Organization → Observability → Adaptive Computation.

**Organization.** §2 describes the bilateral oracle protocol and probe design. §3 presents L2 results (confabulation detection, Fisher essential; T2_L2 pattern; CO labeling). §4 presents L1 results (knowledge-source routing, five architectures; Fisher vs. behavioral baseline comparison). §5 presents L3 results (commitment timing, causal early exit, MATH entry-point, PRM negative result). §6 discusses the three-task hierarchy, candidate scientific laws, and what remains unknown. §7 documents the falsification record. §8 describes competing theories. §9 describes pending experiments and kill criteria. §10 discusses related work.

**What we find at three levels of epistemic difficulty:**

**Level 2 (L2) — Confabulation detection** (primary result): Fisher+PCA64 achieves AUROC=0.854 (Qwen2.5-1.5B, gap=0.240) and AUROC=0.818 (Llama-3.2-3B, gap=0.365) for CONFIDENT_CORRECT vs CONFIDENT_WRONG within the entropy-matched confident zone. The bilateral oracle probe transfers to CC/CW with AUROC=0.880/0.768 without retraining — both CTX_DEP and CONFAB items share the same geometric region, suggesting both represent parametric retrieval failure. A perturbation battery (ICC=0.913) establishes that Fisher scores track the underlying epistemic property, not surface question form.

**Level 1 (L1) — Knowledge-source routing**: Fisher+PCA64 achieves AUROC=0.73–0.75 on bilateral oracle labels under large-N clean sampling. Output entropy achieves 0.87–0.90 on the same task. The bilateral oracle protocol (not the Fisher probe) is the contribution at this level: clean knowledge-source labels enable L2 by construction.

**Level 3 (L3) — Commitment timing in reasoning chains**: Reasoning-distilled models commit to their answer direction after processing 17–18% of their think-block tokens. The remaining 75–83% is post-commitment elaboration. A causal early-exit experiment (EXP-I) establishes that truncating at the commit point saves 87.4% of compute at +0.006 F1 cost (p=0.08).

---

## 2. Methods

### 2.1 Bilateral Oracle Protocol

The bilateral oracle assigns knowledge-source labels through two separate inference passes:

- **PARAM label:** The model can answer without context. Operationalized as: nocontext F1 ≥ 0.50 (i.e., correct answer retrievable from parametric memory alone).
- **CTX_DEP label:** The model requires context. Operationalized as: nocontext F1 ≤ 0.05 AND withcontext F1 ≥ 0.50 (i.e., cannot answer without context but can answer with it).

Items not meeting either criterion are excluded. Hidden states are always collected from the nocontext pass (the parametric condition), ensuring that the probe learns about knowledge-source geometry, not context-processing geometry.

**Why two passes matter.** A single-pass design that labels items by whether they appear to use context conflates: (a) items the model genuinely cannot answer from weights, (b) items the model can answer but chooses not to without context, and (c) items where both are possible. The two-pass design operationalizes the distinction at the behavioral level, without making claims about internal mechanisms.

### 2.2 Probe Architecture

All probes follow the same pipeline:

```
PCA(n_components=64) → LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
```

Applied to hidden states at: layer L=26, generation step s=1 (immediately before the first output token is decoded). Layer 26 is selected from prior layer sweeps showing peak AUROC across 28 layers. Step-1 is selected from generation-onset experiments showing step-1 > prefill (0.785 vs 0.567 mean AUROC across four model families in ESM v33).

PCA dimensionality reduction to 64 components is applied before LDA to address Fisher LDA's known failure mode at high dimension/low sample ratio. Without PCA, Fisher LDA produces degenerate covariance estimates at d=1536–3072, inflating or deflating AUROC nonlinearly (documented in §7, C012/C013 falsifications).

### 2.3 Evaluation Protocol

- **Shuffled control:** Labels are randomly permuted while keeping the probe architecture and data unchanged. Any AUROC above shuffled baseline indicates signal; shuffled AUROC ≈ 0.50 indicates a clean null.
- **Bootstrap CI:** 95% confidence intervals are computed by bootstrap resampling (n=1000).
- **Protocol fingerprint:** Each major experiment records: random seed, question source, sampling method, pool size, and N/class, enabling exact replication.

### 2.4 Models

- **Bilateral oracle (L1/L2):** Qwen2.5-1.5B-Instruct, Llama-3.2-3B-Instruct (TriviaQA, large-N validation v2)
- **Confabulation detection (L2):** Qwen2.5-1.5B-Instruct (TriviaQA, EXP-A false_certainty_v2)
- **Reasoning geometry (L3):** DeepSeek-R1-Distill-Qwen-1.5B (EXP-B), DeepSeek-R1-Distill-Llama-8B (EXP-C)
- **Scale comparison (supplementary):** Qwen2.5-0.5B, Qwen2.5-3B, Llama-3.2-1B (EXP-E)

All models loaded with `torch_dtype=torch.float16`, `device_map=None` (explicit `.to(device)`), `trust_remote_code=True`. The `device_map=None` constraint is required to avoid EvictionCache device-split failures in hidden-state collection experiments.

### 2.5 Invariance Under Surface Perturbations (EXP-J)

To verify that Fisher+PCA64 scores reflect the underlying epistemic state rather than surface question form, we apply a **perturbation battery** to the calibrated probe. For each labeled item, we generate 4 surface variants:

- **REPHRASE:** Swap question opener ("What is X?" → "Can you name X?", "Who is X?" → "Name the person: X")
- **LOWERCASE:** Entire question lowercased
- **APPEND:** Neutral suffix appended ("Please answer briefly.")
- **TYPO:** Drop the 3rd character of the longest word

We extract L26 step-1 hidden states and compute the probe's scalar decision score for all 5 versions (original + 4 variants) per item. We then compute the Intraclass Correlation Coefficient (ICC):

$$\text{ICC} = \frac{\sigma^2_\text{between}}{\sigma^2_\text{between} + \sigma^2_\text{within}}$$

where $\sigma^2_\text{between}$ is the variance of original scores across items and $\sigma^2_\text{within}$ is the mean per-item variance across the 5 score versions. ICC ≥ 0.70 = ROBUST (sufficient for generalizability claims); ICC < 0.50 = FRAGILE (kill criterion: all L1–L3 findings require major qualification).

**Result (EXP-J, N=160, 80 PARAM + 80 CTX_DEP, Qwen2.5-1.5B-Instruct):**

| Metric | Value |
|---|---|
| ICC | **0.913** |
| Verdict | **ROBUST** (threshold: ≥ 0.70) |
| Kill triggered | False |
| between_var | 2.737 |
| mean_within_var | 0.260 |
| Variance ratio (between/within) | 10.5:1 |
| Cal AUROC (Phase 1 probe) | 0.710 (shuffled=0.550) |
| Separation preserved under variants | True (t=23.021, p<0.0001) |
| PARAM variant mean | +0.938 |
| CTX_DEP variant mean | −1.066 |

Per-variant correlation with original score:

| Perturbation | Score–original corr | Mean |Δ| |
|---|---|---|
| REPHRASE | 0.869 | 0.453 |
| LOWERCASE | 0.879 | 0.460 |
| APPEND | 0.830 | 0.579 |
| TYPO | 0.923 | 0.339 |

All four variants exceed r=0.83. TYPO (character-level noise) produces the least score shift; APPEND (irrelevant clause appended) produces the most, yet remains strongly correlated. The between-item variance is 10.5× the within-item variance. Question surface form does not detectably shift the Fisher score; the score tracks the underlying knowledge-source property, not the exact token sequence.

**Replication on Llama-3.2-3B-Instruct (perturbation_battery_v1):**

| Metric | Value |
|---|---|
| ICC | **0.9334** |
| Verdict | **ROBUST** (threshold: ≥ 0.70) |
| between_var | 2.253 |
| mean_within_var | 0.161 |
| Variance ratio (between/within) | 14.0:1 |
| Separation preserved under variants | True (t=20.264, p<0.0001) |
| REPHRASE/LOWERCASE/APPEND/TYPO correlations | ≥ 0.91 |

ICC improves from 0.913 (Qwen) to 0.9334 (Llama), with a tighter within-item variance (0.161 vs 0.260) and higher between/within ratio (14.0:1 vs 10.5:1). Both architectures exceed the ROBUST threshold. PARAM/CTX_DEP separation is preserved under all four perturbation types on Llama (t=20.264, p<0.0001). **C025 CONFIRMED — two independent architectures.**

---

## 3. Level 2: Confabulation Detection

### 3.1 The L2 Problem

Confabulation detection isolates the hardest case in output monitoring: CONFIDENT_CORRECT (CC) versus CONFIDENT_WRONG (CW) items, where both classes have been selected for low output token entropy (θ_conf=1.1043 nats for Qwen). By construction, entropy cannot discriminate these two classes. The question is whether hidden states at the residual stream encode a geometry that output distributions cannot expose.

### 3.2 Fisher Is Essential at L2

**Entropy-matched collection protocol.** CC items: generated answer matches gold, output entropy ≤ θ_conf. CW items: generated answer does not match gold, output entropy ≤ θ_conf. By matching on entropy, the task is designed so that entropy cannot distinguish the two classes.

**Results (EXP-A, false_certainty_v2, N=100/class, Qwen2.5-1.5B-Instruct, TriviaQA):**

| Signal | AUROC | Shuffled | Control | Notes |
|---|---|---|---|---|
| Output entropy | 0.614 | ~0.50 | (matched by design) | Residual Δ=0.037 nats between CC/CW despite entropy matching |
| Fisher+PCA64 (L26, step-1) | **0.854** | 0.4256 | CLEAN | Gap = 0.240 over entropy |

After entropy matching, the CC/CW entropy difference is 0.037 nats (CW slightly higher). Fisher adds 0.240 AUROC above this residual entropy signal. This is the result where Fisher is genuinely essential, not just equivalent to entropy.

**Llama replication (EXP-A-Llama, false_certainty_llama_v2, N=80/class, Llama-3.2-3B-Instruct, TriviaQA):**

| Signal | AUROC | Shuffled | Notes |
|---|---|---|---|
| Output entropy | 0.453 | ~0.50 | Window precisely matched (CC mean=1.188 nats, CW mean=1.201 nats, Δ=0.013) |
| Fisher+PCA64 (L26, step-1) | **0.818** | 0.543 | CLEAN |
| Gap (Fisher − Entropy) | **0.365** | — | Larger than Qwen (0.240) due to better entropy match |
| BO_Transfer | 0.768 | — | Bilateral oracle (Phase 1 AUROC=0.874) transfers to CC/CW |

The gap is *larger* on Llama (0.365 vs 0.240) because the entropy window is more precisely matched (entropy AUROC=0.453 ≈ 0.50). Both C017 (Fisher AUROC ≥ 0.70, gap ≥ 0.10) and C018 (BO_Transfer ≥ 0.70) are SUPPORTED on Llama-3.2-3B-Instruct, promoting these claims from single-architecture to **two-architecture**.

**Gemma-2-2B-IT (T2_L2 scope boundary — EXP_GEMMA_L2_V1, N=100/class):**

| Signal | AUROC | Shuffled | Notes |
|---|---|---|---|
| Output entropy | 0.547 | ~0.50 | θ_conf=0.067 (extremely low; final_logit_softcapping collapses entropy toward 0) |
| Fisher+PCA64 (L24, step-1) | 0.570 | 0.424 | CI=[0.401, 0.734] spans 0.50 |
| Gap (Fisher − Entropy) | 0.022 | — | **Below 0.05 threshold — T2_L2 NOT SUPPORTED** |

Gemma-2-2B-IT triggers the pre-specified T2_L2 termination rule (gap < 0.05): the L2 confabulation geometry does not replicate on this architecture at current N. The mechanistic interpretation is that Gemma-2's `final_logit_softcapping` (tanh(x/30)×30) collapses the entropy distribution toward zero (θ_conf=0.067 vs Qwen≈0.5, Llama≈1.2), making the "confident" zone poorly conditioned — nearly all outputs are low-entropy, and the CC/CW entropy difference is only 0.007 nats (0.114 vs 0.121). Crucially, the L1 bilateral oracle *does* replicate on Gemma (AUROC=0.753, §4.1 table), so the hidden-state geometry at L1 is present but the specific entropy-regime separation required for L2 does not hold.

**Mistral-7B-Instruct-v0.3 (T2_L2 — second instance, C035):**

| Signal | AUROC | Shuffled | Notes |
|---|---|---|---|
| Output entropy | — | — | θ_conf=0.122; Mistral is a low-entropy confident model |
| Fisher+PCA64 (L26, step-1) | 0.5952 | — | — |
| Gap (Fisher − Entropy) | −0.014 | — | **T2_L2 NOT SUPPORTED** |
| BO_Transfer | 0.6624 | — | Geometry present; entropy-matched framing fails |

Mistral-7B-Instruct achieves the highest L1 AUROC in the program (0.778, §4.1) but triggers the T2_L2 termination rule. Like Gemma, its θ_conf=0.122 places it in the low-entropy regime where the entropy-matched CC/CW window is degenerate. Despite the L2 framing failure, BO_Transfer=0.662 — substantially above chance — establishes that the underlying geometry exists and separates the relevant classes; the measurement framing (entropy-matched CC/CW selection) is the failure point, not the geometry. The 7B scale difference between Mistral and Qwen/Llama confirms that θ_conf is not simply a function of parameter count: Mistral-7B is *more* low-entropy than Qwen-1.5B despite being 4.7× larger.

**T2_L2 Pattern (C036 — formalized across two architectures):**

The two T2_L2 failures now define a clear boundary condition: entropy-matched L2 confabulation detection fails when θ_conf < 0.15. The affected architectures are Gemma-2 (θ_conf=0.067, caused by `final_logit_softcapping`) and Mistral-7B (θ_conf=0.122, caused by low-entropy output distributions in the 7B parameter regime under Mistral training). Both show BO_Transfer > 0.5, confirming that the confabulation geometry exists and the failure is in the measurement framing, not in the underlying representational structure. Three competing explanations for T2_L2 remain: (a) the entropy window is too narrow to provide sufficient CC/CW contrast; (b) low-θ_conf models have qualitatively different confabulation dynamics (less distinction between confident-correct and confident-wrong); (c) the Fisher probe requires re-calibration on low-entropy distributions. CO labeling (§3.5 below) provides one path to bypass the entropy-matching dependency entirely.

This is a scope contraction, not a falsification of C017: the confabulation geometry gap is confirmed on Qwen and Llama; Gemma and Mistral represent a well-characterized boundary condition.

### 3.3 Bilateral Oracle Transfer (C018)

The bilateral oracle probe (trained on PARAM vs CTX_DEP) transfers directly to the CC/CW task with AUROC=0.880 on Qwen and AUROC=0.768 on Llama — both above the 0.70 threshold, and both trained entirely on the PARAM/CTX_DEP distinction with no CC/CW labels. This transfer result is the most theoretically striking finding in the program: the geometry that separates parametric from context-dependent items also separates correct-confident from wrong-confident items across two independent architectures. Both CTX_DEP and CONFAB items appear to occupy the same geometric region — both represent failure of parametric retrieval (either the model never had the knowledge, or it had it but retrieved it wrong).

Three competing interpretations are currently consistent with this evidence (maintained per competing-models discipline — no premature selection):

- **Model A (Retrieval Quality):** Fisher scores how well parametric retrieval succeeded. CTX_DEP = retrieval unavailable; CONFAB = retrieval partially failed.
- **Model B (Internal Certainty):** Fisher reflects the model's post-retrieval certainty state. Both CTX_DEP and CONFAB = low certainty, for different reasons.
- **Model C (Memory-Generation Consistency):** Fisher reflects alignment between what the model generated and its memory trace. Both CTX_DEP and CONFAB = low alignment.

EXP-F (commit-point states → CC/CW prediction) returned BLIND at N=40/class: AUROC=0.650, commit timing identical for CC and CW (98.0% vs 97.9% commit_pct). This result did not discriminate among Models A, B, C at current N — a larger-N follow-up (N≥100/class, second architecture) is required. SAE feature patching would be most discriminating for Model B vs C.

### 3.4 CO Labeling: Removing the Entropy-Matching Dependency (C032)

The entropy-matched L2 design (§3.1) controls for output entropy by construction, ensuring that Fisher adds value specifically in the zone where entropy has failed. A cost of this design is that it requires a large pool to find sufficient CC/CW items within the entropy window, and it creates the T2_L2 boundary condition for low-θ_conf architectures. CO (Correct-vs-Incorrect) labeling removes this dependency: instead of entropy-matched CONFIDENT_CORRECT vs CONFIDENT_WRONG, CO labels items as simply correct (any confidence) vs. incorrect (any confidence), using N=200/class with no entropy restriction.

**Results (Qwen2.5-1.5B-Instruct, TriviaQA, N=200/class):**

| Labeling | AUROC | N/class | Entropy-matched? |
|---|---|---|---|
| BO (entropy-matched PARAM vs CTX_DEP) | 0.806 | 100 | Yes |
| CO (correct vs. any incorrect) | **0.885** | 200 | No |
| Delta | +0.079 | — | — |

CO achieves 0.885 AUROC — higher than the entropy-matched BO (0.806) — and does so with double the effective N. The reason CO outperforms BO is that CTX_DEP items are geometrically the *closest* to PARAM items: they represent borderline cases where the model is near the decision boundary. By excluding CTX_DEP items (which BO does by including only PARAM and CTX_DEP as two classes), BO actually includes these low-Fisher-separation items in its training. CO labels correct items as positive and confabulations + CTX_DEP failures as negative, so the positive class is more geometrically pure and the negative class is more extreme.

**C036_CONFIRMED: CO labeling recovers L2 for T2_L2 architectures (EXP_CO_GEMMA_MISTRAL_V1).** The CO-on-Gemma/Mistral experiment has resolved the T2_L2 question definitively. Gemma CO labeling (θ_conf=0.7562, N=200/class): Fisher AUROC=0.8368 CI=[0.749, 0.910], Entropy=0.6311, Gap=+0.2056, shuffled=0.4712 (CLEAN) — CO_RECOVERS. Mistral-7B CO labeling (θ_conf=0.0220, N=200/class): Fisher AUROC=0.8580 CI=[0.777, 0.927], Entropy=0.6466, Gap=+0.2114, shuffled=0.5140 (CLEAN) — CO_RECOVERS. Both results are clean and both match Qwen CO performance (0.885). T2_L2 was entirely a measurement framing artifact: the entropy-matched window [θ_conf ± 0.30] degenerates to near-empty for low-θ_conf architectures, selecting noise rather than signal. The confabulation geometry is present and robust in all tested architectures when measured with CO labeling. **CO labeling is now the primary L2 estimator across all architectures, regardless of θ_conf.** The entropy-matched design is retained as a historical reference; CO is the canonical forward protocol.

**Authoritative L2 estimate (C040 — l2_large_n_v1 COMPLETE).** The C034 variance concern is now resolved. EXP_L2_LARGE_N_V1 (N=500/class CO-style, θ_conf=0.7733, 5-fold stratified CV, Qwen2.5-1.5B-Instruct, TriviaQA) yields:

| Metric | Value |
|---|---|
| CV Fisher AUROC | **0.7629 ± 0.0120** (mean ± std across 5 folds) |
| CV Entropy AUROC | 0.6002 ± 0.0440 |
| CV Gap | **0.1628** |
| Verdict | **STABLE_SIGNAL** (mean ≥ 0.75, std ≤ 0.05) |
| Fold range | 0.7440 – 0.7805 (range = 0.0365) |

The prior 0.670–0.885 range was pool-section heterogeneity plus small n_test sampling variance. At N=500/class with 5-fold CV (n_test=100/class per fold), the AUROC stabilizes. The CV Gap (0.1628) is smaller than the entropy-matched EXP-A gap (0.240) because CO-style collection does not force entropy AUROC → 0.50: the entropy baseline is 0.6002 (not disabled by design). Both are valid measurements of Fisher's contribution at L2, answering different questions: EXP-A quantifies Fisher's advantage when entropy has fully failed; l2_large_n quantifies Fisher's advantage across the full low-entropy population.

### 3.5 Entropy Trajectory at L2 (EXP-D)

A 15-step output entropy trajectory (KV-cached, generation steps 0–14) achieves AUROC=0.730 for CC vs CW (C019). The trajectory has a characteristic inversion: step-0 AUROC=0.331 (below chance), rising to step-4 AUROC=0.617 and continuing upward. CW items show entropy bursts (0.75→1.57) while CC items stay flat. This trajectory signal is complementary to, not redundant with, the Fisher step-1 result: Fisher uses hidden states at one moment; the trajectory captures temporal dynamics over 15 generation steps.

---

## 4. Level 1: Knowledge-Source Routing

### 4.1 Main Result

Under large-N clean sampling (pool=10,000, TriviaQA), the Fisher+PCA64 probe achieves:

| Model | Family | N/class | AUROC | Shuffled | CI 95% | Status |
|---|---|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct | Qwen (GQA) | 197 | 0.7312 | 0.598 | [0.63, 0.83] | CLEAN |
| Llama-3.2-3B-Instruct | Llama (GQA) | 200 | 0.7464 | 0.502 | [0.65, 0.83] | CLEAN |
| Gemma-2-2B-IT | Gemma (MQA, SWA) | 200 | 0.7528 | 0.530 | [0.65, 0.85] | CLEAN |
| Mistral-7B-Instruct-v0.3 | Mistral (sliding window) | 200 | 0.7780 | 0.5524 | [0.692, 0.863] | CLEAN |
| Phi-3.5-Mini-Instruct | Phi (GQA, RoPE) | 200 | **0.8456** | 0.4992 | [0.758, 0.921] | CLEAN |

All five shuffled controls are well below real AUROC. Architecture spread: [0.731, 0.746, 0.753, 0.778, 0.846] — range=0.115 across five independent families (Qwen, Llama, Gemma, Mistral, Phi) spanning attention mechanisms (GQA, MQA, sliding-window, RoPE), parameter scales (1.5B–7B), and two distinct training lineages (Qwen-family, Phi-family). Phi-3.5-Mini-Instruct achieves the highest L1 AUROC (0.8456), consistent with its RLHF-trained assertiveness producing strong epistemic geometry at L1 (cf. C042: SFT consistently improves L1 over base). This result supersedes earlier calibration-phase estimates (0.841–0.846, N=128–150/class), which had a dataset-ordering artifact in the Qwen shuffled control. The large_n_v2 protocol is the authoritative result.

### 4.2 The Entropy Baseline

A critical comparison: what does output entropy alone achieve on this task?

| Model | Entropy AUROC | Fisher AUROC | Combined | Fisher independent? |
|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct | **0.9043** | 0.6566 | ~0.90 | No — entropy dominates |
| Llama-3.2-3B-Instruct | 0.874 | 0.8601 | **0.9037** | Marginal — Fisher adds ~0.03 |

For Qwen, output entropy (0.9043) substantially exceeds Fisher+PCA64 (0.6566): Fisher is redundant. For Llama, entropy (0.874) approximately equals Fisher (0.860); their combination reaches 0.9037, suggesting a marginal independent hidden-state component.

**What this means.** Knowledge-source routing is primarily a confident/uncertain distinction. PARAM items tend to produce lower output entropy (the model is confident in its parametric answer); CTX_DEP items tend to produce higher entropy (the model is uncertain without context). The Fisher probe partially captures this same confident/uncertain axis from hidden states. This is **not** the independence claim originally hypothesized (r=0.0039, now falsified — see §7). The bilateral oracle protocol remains valuable as a clean labeling methodology; the Fisher probe is not independently essential at L1.

### 4.3 Architecture Comparison at L1

All five architectures achieve AUROC ≥ 0.73 with clean shuffled controls (C001, C003, C020, C044). The range across five families (0.731–0.846) is 0.115 — moderate given the parameter range (1.5B–7B) and architectural diversity. Phi-3.5-Mini-Instruct achieves the highest L1 AUROC (0.8456), consistent with its RLHF-trained assertiveness producing strong epistemic geometry at L1 (cf. C042: SFT consistently improves L1 over base). The scale observability experiment (EXP-E) established that this holds within a parametric capability Goldilocks zone: models too small (Qwen 0.5B: PARAM=0/3000 items qualifying) or too large for the TriviaQA setup (Qwen 3B: CTX_DEP=9/3000 qualifying) fall outside the zone where bilateral oracle labeling works. The protocol is architecture-agnostic in design but task/scale-constrained in practice.

### 4.4 Fisher vs. Behavioral Baselines at L2 (C033)

At L1, output entropy dominates Fisher (§4.2). At L2 (entropy-matched confabulation detection), the comparison inverts. We test Fisher+PCA64 against four behavioral baselines on the same N=100/class entropy-matched CC/CW pool (Qwen2.5-1.5B-Instruct, TriviaQA):

| Baseline | Method | AUROC | Notes |
|---|---|---|---|
| B1: Output entropy | Per-token entropy at generation step | 0.614 | By design ≈ chance (entropy-matched) |
| B2: Self-consistency | k=5 majority vote agreement rate | 0.613 | Multi-sample behavioral proxy |
| B3: Top-1 probability | Max token probability at step 0 | **0.384** | **Below chance — anti-correlated** |
| **Fisher+PCA64** | L26 step-1 hidden state | **0.845** | **+0.232 over best behavioral** |

**The anti-correlation finding (B3).** A top-1 probability AUROC of 0.384 means that items with *higher* token-level confidence are *less* likely to be correct within the entropy-matched confident zone. This is consistent with a structural consequence of post-training alignment on the tested model family (Qwen2.5-1.5B-Instruct): alignment training that rewards assertive, confident-sounding outputs may produce confabulated wrong answers with *more* peaked output distributions than correctly-retrieved answers, because the model has been trained to commit confidently even when its parametric retrieval failed. Output confidence (top-1 probability) would then have been progressively decoupled from epistemic validity by the training objective. Whether this anti-correlation is a universal property of RLHF-trained models or an artifact of specific training configurations requires validation across model families (exp_l_stage_sweep_v2).

**Why Fisher corrects this.** Fisher+PCA64 reads the residual stream at layer 26 *before* the output head applies its post-training vocabulary projection. The geometry of the residual stream reflects the epistemic state upstream of the trained output transformation — not the post-alignment surface presentation. This explains the +0.232 gap: Fisher reads geometry that precedes the output head's trained transformations; behavioral baselines measure a downstream quantity that may have been distorted away from epistemic state by the training objective. Self-consistency (B2=0.613) fails for the same reason: all k=5 samples are drawn from the same alignment-distorted distribution, so they agree confidently and wrongly.

This result establishes the theoretical rationale for hidden-state probing at L2: not just that it works, but *why* it must work for a class of failures that output-space methods are structurally prevented from detecting.

### 4.5 Scope: Format-Sensitive Portability (C041)

All L1 and L2 results reported in §3–4 were obtained on TriviaQA (open-ended Wikipedia factual recall). The OOD generalization experiment (ood_generalization_v2) characterizes the portability of the bilateral oracle probe across task formats. A Fisher+PCA64 probe trained on TriviaQA bilateral oracle labels (N=200/class, L26, Qwen2.5-1.5B-Instruct, source AUROC=0.7744) was applied without retraining to two OOD tasks:

- **HotpotQA** (open-ended Wikipedia multi-hop QA, same bilateral oracle protocol): transfer AUROC=0.6567, within AUROC=0.7769, transfer efficiency=84.5% — OOD_PARTIAL.
- **MMLU-STEM** (multiple-choice, cross-domain factual knowledge): transfer AUROC=0.5288, within AUROC=0.9488, transfer efficiency=55.7% — TASK_SPECIFIC (essentially random transfer).

The pattern characterizes the scope of portability: geometry transfers within similar task formats (same labeling protocol, same question style, same knowledge domain) but not across format boundaries. This is consistent with the independently published finding that truthfulness directions are orthogonal across diverse tasks (arXiv 2506.08572); the present result adds that within-format transfer is partial but non-trivial (84.5% efficiency). The bilateral oracle is therefore correctly understood as a task-format-calibrated instrument — not a universal epistemic meter — and per-format recalibration is required for deployment across task types.

---

## 5. Level 3: Commitment Timing in Reasoning Chains

### 5.1 The L3 Problem

Reasoning-distilled models generate an explicit think block before producing an answer. What is the internal structure of this think block? Prior work assumes it represents search — the model exploring alternatives before committing to one. The commitment timing experiments test whether this assumption holds.

### 5.2 The Commit Point

We define a commit point operationally: the generation step at which the Fisher trajectory (applied to hidden states during think-block token generation) crosses a threshold indicating the model has committed to an answer direction. Let commit_step be the step index; think_len be the total length of the think block in tokens.

**commit_pct** = 100.0 × (think_len − commit_step) / think_len

This is a continuous metric: what fraction of the think block occurs *after* the commit point. Higher commit_pct = earlier commitment (more of the block is post-commitment elaboration).

### 5.3 Results

**EXP-B (DeepSeek-R1-Distill-Qwen-1.5B, TriviaQA, N=100):**

| Metric | Value |
|---|---|
| Commit rate | 80/100 questions committed |
| Mean commit_pct (committed only) | 75.8% |
| z-score vs shuffled null | 49.77 |
| Cal AUROC at L26 step-1 | 0.760 |

**EXP-C (DeepSeek-R1-Distill-Llama-8B, TriviaQA, N=100):**

| Metric | Value |
|---|---|
| Commit rate | 80/100 questions committed |
| Mean commit_pct (committed only) | 82.9% |
| z-score vs shuffled null | 679.73 |
| Cal AUROC at L28 step-1 | 1.000★ |

★ Cal AUROC=1.000 reflects N=10/class calibration set — likely small-N saturation. Not a quality claim.

**Interpretation.** Across both families, ~80% of the think block occurs after the model has committed to its answer direction. The think block is primarily post-commitment elaboration, not pre-decision search. This result holds across both tested reasoning-distilled families and is statistically robust (z >> shuffled null in both cases, C022). **Teacher-independence (C045 — RESOLVED):** Both models are DeepSeek-R1-Distill variants sharing the same teacher (DeepSeek-R1). EXP_TEACHER_INDEPENDENCE_V1 tested Qwen3-1.7B (Qwen-native reasoning, not R1-distilled): N=100/100 committed (100% commit rate), mean_commit_pct=99.8% (commits within first ~2 think tokens), null_mean=48.7%, z=1.3×10¹⁵, cal_AUROC=0.9268. The result is more extreme than both R1-distill models (75.8%/82.9%). Early commitment is not specific to DeepSeek-R1 training — it holds in a non-R1 Qwen-native reasoning model. The entire think block is post-commitment elaboration (C045).

### 5.4 The Z-Score Amplification Pattern

Base models show commitment signals with z ≈ 7 (Qwen2.5-1.5B-Instruct, EXP from earlier experiments). Reasoning-distilled models show z = 49.77 (EXP-B) and z = 679.73 (EXP-C). The 13× difference between Qwen and Llama reasoning models (z=49 vs z=680) may reflect backbone architecture, distillation method, teacher identity, reasoning token budget, or some combination.

**Five competing explanations** (no premature selection):

| Explanation | Prediction | Test |
|---|---|---|
| Training objective | z amplifies only at reasoning distillation stage | EXP-L: Base→SFT→RLHF→Reasoning on single backbone |
| Distillation method | KL compression sharpens commitment geometry | EXP-L + second distillation method |
| Teacher identity | DeepSeek-R1 teacher transfers specific patterns | Reasoning distillation from different teacher |
| Backbone architecture | Llama 8B has more expressive geometry than Qwen 1.5B | EXP-B vs EXP-C partly consistent — but size confounds |
| Reasoning token budget | Longer think blocks → higher z by statistical artifact | Token count normalization within EXP-B/C |

EXP-L (training stages on single backbone) will resolve some but not all of these.

### 5.5 The Commitment Trajectory Model

The binary theater/search framing (thinking = one of two modes) is too coarse. The evidence is consistent with a continuous commitment trajectory:

**Exploration → Weak commitment → Revision → Strong commitment → Elaboration**

The Fisher trajectory detects the transition from Exploration to Strong commitment. Whether the Revision stage is real — whether the model sometimes changes its committed answer direction — is testable but not yet tested.

### 5.6 Post-Think Entropy Burst (EXP-G)

Does explicit chain-of-thought reasoning pre-resolve epistemic uncertainty, producing flat post-answer entropy? We test this by measuring the entropy trajectory of answer tokens immediately following `</think>` for a reasoning-distilled model on GSM8K mathematical reasoning (N=50/class, DeepSeek-R1-Distill-Qwen-1.5B).

**Prediction:** FLAT — think block pre-resolves uncertainty, answer entropy low and uniform regardless of correctness. **Result: BURST** — prediction falsified (C027).

| Metric | Value |
|---|---|
| Trajectory AUROC (LR, 15 steps) | **0.8424** |
| Pattern | **BURST** |
| Peak step | 4 |
| Peak per-step AUROC | 0.6932 |
| Step-0 AUROC | 0.3612 |

Mean entropy trajectories (steps 0–14 post-`</think>`):

| Class | Step 0 | Step 1 | Step 4 |
|---|---|---|---|
| CC (correct) | 0.001 | 1.087 | 0.318 |
| CW (wrong) | 0.001 | 1.322 | **0.127** |

The discriminative signal at step 4 is driven by a **collapse in CW items**: wrong answers spike to high entropy at step 1 (1.322) then collapse to near-zero by step 4 (0.127), while correct answers remain moderately uncertain (0.318). This is the same BURST pattern with the same peak step as EXP-D (base Qwen on TriviaQA). The think block delays but does not eliminate the burst, establishing a third temporal regime: post-think answer generation dynamics, distinct from generation-onset dynamics (EXP-D) and intra-chain commitment dynamics (EXP-B/C).

### 5.7 Early Exit Causal Validation (EXP-I)

EXP-B established that ~80% of thinking tokens occur post-commitment (observational). EXP-I converts this to a causal test: for each of N=200 questions, generation is run twice — once to completion and once truncated at the detected commit point. Δf1 = f1_full − f1_truncated.

| Metric | Value |
|---|---|
| n_committed | 199/200 (99.5%) |
| Mean Δf1 | **+0.0059 ± 0.0474 SD** |
| SE | 0.0034 |
| 95% CI | [−0.0007, +0.0125] |
| p (two-tailed) | 0.08 |
| p (one-tailed, pre-registered directional) | 0.04 |
| Helped (Δ > 0.05) | 7.54% |
| Hurt (Δ < −0.05) | 4.02% |
| Mean commit% | **87.4%** |

**Verdict: MINIMAL_QUALITY_LOSS (C028).** Truncating at commit saves 87% of thinking tokens at +0.006 F1 cost (borderline significant; running mean stable at +0.004 to +0.006 from item 50 through 200). The 1.9:1 helped/hurt ratio on large-effect items confirms elaboration is directionally beneficial but not essential. The commit probe works as a reliable timing trigger: 99.5% of items commit within budget.

Together with EXP-H (centroid patching epiphenomenal), EXP-I establishes the correct framing: the Fisher probe detects *when* the answer is decided, not *what* decides it. The commit point is a timing readout, not a causal control point. This distinction matters for deployment: the probe can trigger early exit (saving compute) but cannot be used to correct confabulation by patching hidden states toward a "better" commit direction.

### 5.8 EXP-F: Commit-Point Hidden States for Confabulation Detection

EXP-F tested whether Fisher+PCA64 applied specifically at the commit-point step (rather than step-1 of the answer) could distinguish CC from CW items in reasoning-distilled models. The commit point is the theoretically motivated location: if confabulation is detectable anywhere in the think block, the moment the model commits to an answer direction is where geometric separation should be largest.

**Results (DeepSeek-R1-Distill-Qwen-1.5B, TriviaQA, N=40/class):**

| Metric | Value |
|---|---|
| Fisher+PCA64 AUROC at commit step | 0.6500 |
| Shuffled control | 0.4400 (CLEAN) |
| Verdict | **BLIND** (threshold=0.70 INFORMED; CI wide at N=40/class) |
| Mean commit_pct — CC items | 98.0% |
| Mean commit_pct — CW items | 97.9% |

Commit timing is indistinguishable between CC and CW items: both classes commit after processing approximately 98% of their think block. The Fisher score at the commit step carries no reliable confabulation signal at current N (AUROC=0.65, provisional, noisy at N=40/class).

**Conclusion.** Commit-point hidden states are not a reliable Gate 3 signal at the current scale (N=40/class, single architecture). The result is BLIND, not REFUTED — whether this is a sample-size limitation or a genuine null requires validation at larger N. Gate 3 for reasoning models cannot yet be instantiated at the commit point; further experiments are required before deployment framing can be finalized.

### 5.9 Entry-Point Prediction on Mathematical Reasoning (C039)

The PRM correlation experiment (§5.10) included a calibration phase: can Fisher+PCA64 at step-1 predict the *final* correctness of a MATH problem before the reasoning chain begins? The calibration phase used DeepSeek-R1-Distill-Qwen-1.5B on HuggingFaceH4/MATH-500 (N=30/class correct vs. incorrect).

**Results (layer sweep, step-1 hidden states, MATH-500):**

| Layer | AUROC | Notes |
|---|---|---|
| L24 | 0.8978 | — |
| L25 | **0.8558 (CI=[0.729,0.957], N=100/class; original small-N estimate was 0.9111)** | Peak |
| L26 | **0.8558** | Peak (tied) |
| L27 | 0.9100 | — |

AUROC=0.8558 (CI=[0.729,0.957], N=100/class; original small-N estimate was 0.9111) at layers 25–26 is the highest step-1 AUROC observed in this research program across any task. It still exceeds the L2 confabulation AUROC on the same model family (0.854 on Qwen-1.5B, TriviaQA) and the commitment AUROC (0.760–1.000 on reasoning chains).

**What this establishes.** At the *first reasoning token* — before any chain-of-thought computation has occurred — the residual stream geometry at layers 25–26 already predicts whether the model will correctly solve a MATH problem (AUROC=0.8558, CI=[0.729,0.957], N=100/class; original small-N estimate was 0.9111). This is an *entry-point predictor*, not a step-tracker: the geometry at the moment the model begins thinking about a problem encodes the epistemic state it will arrive at after generating hundreds of tokens of reasoning.

This result has a specific interpretation within the three-task hierarchy: it establishes a fourth measurement regime beyond L1/L2/L3 — **approach-commitment geometry**. Before the model processes any reasoning chain, the residual stream geometry captures the model's *approach commitment*: the mathematical strategy, schematic structure, or knowledge retrieval pattern the model will use to attempt the problem. This geometry predicts final correctness because approach-commitment determines outcome. The geometry at step-1 does not change as reasoning proceeds (see §5.10 below); it is not tracking computational progress through the chain.

### 5.10 Process Reward Model Correlation — Negative Result (C038)

If step-1 geometry predicts final MATH correctness (§5.9), does hidden-state geometry at *intermediate reasoning steps* track the oracle process reward — the probability that the remaining reasoning will produce a correct answer? If so, hidden states would provide a free, zero-label process reward model (PRM) for reasoning supervision.

**Method.** For each question, we run the reasoning model to step k (k=1,2,...,K_i for K_i reasoning steps), extract Fisher J-score from the intermediate hidden state, and compute oracle PRM at step k by running M=1 greedy completion from step k and measuring F1 against the gold answer. We then compute: (a) Pearson/Spearman correlation between J-score and oracle PRM across all (question, step) pairs; (b) Pearson r between early J-score (step-1) and final correctness.

**Results (DeepSeek-R1-Distill-Qwen-1.5B, MATH-500, N=50 questions, 292.9 min):**

| Metric | Value | Interpretation |
|---|---|---|
| Pearson r(J_know, oracle_PRM) | 0.053 | Negligible correlation |
| Spearman ρ(J_know, oracle_PRM) | 0.047 | Negligible |
| Pearson r(early J_know, final correctness) | −0.008 | Negligible (p=0.945) |
| Verdict | **PRM_SIGNAL_WEAK** | Kill criterion not triggered |

**Interpretation.** The Fisher probe at intermediate reasoning steps does not correlate with the oracle process reward. The geometry does not track step-by-step computational progress through the reasoning chain. This is consistent with the commitment architecture: step-1 geometry predicts final answer correctness (AUROC=0.8558, N=100/class corrected; original small-N estimate 0.9111) precisely because it captures approach-commitment, not because it tracks where the computation is within the chain. Once the model has committed to an approach at step-1, intermediate-step geometry does not update to reflect whether the approach is succeeding.

This result is a genuine *negative finding*, not a kill criterion trigger: it distinguishes what hidden geometry measures (approach-commitment, epistemic state at problem onset) from what it does not measure (step-level process reward, computational progress). The free-PRM hypothesis is falsified; the entry-point predictor result stands.

---

## 6. The Three-Task Hierarchy: What It Means

### 6.1 Why a Hierarchy?

The three tasks are not interchangeable. Each level isolates a strictly harder epistemic distinction:

- L1 separates confident from uncertain items. Entropy ≈ Fisher. Well-studied in calibration literature.
- L2 separates wrong-confident from right-confident items, within the low-entropy zone. Fisher essential. Less studied.
- L3 separates early-commit from late-commit moments within a single item's think block. Fisher trajectory required. No prior characterization.

The appropriate signal changes at each level. Reporting only L1 results invites the objection "output entropy already does this." Reporting all three levels makes the Fisher probe's unique contribution precisely locatable: it is essential at L2 and L3, not at L1.

### 6.2 The Safety Gap

L1 does not determine whether a confident answer is correct. A confident answer — low entropy, Fisher near PARAM centroid — can be either CC or CW. The routing architecture Gates 1–2 (knowledge-source routing) are necessary but not sufficient for safe deployment. Gate 3 (confabulation detection within the confident zone) requires Fisher and is the prerequisite for adversarial robustness.

**Gate 3 is a detector, not an intervenor (under tested interventions).** EXP-H (centroid-direction patching at λ=0.1–2.0 across all probed layers) finds max Δ_F1=+0.0004 — indistinguishable from zero (EPIPHENOMENAL). This extends the C005 patching null to the confabulation domain and establishes that the CC/CW geometry is currently observed as non-causal under centroid-direction, residual-stream patching at the tested layers and magnitudes. The geometry detects the CC/CW distinction with AUROC=0.854, but the same geometric axis does not correct confabulation via centroid patching. We note that alternative intervention families (attention head patching, SAE feature patching, training-time objectives) remain untested and may find causal leverage. Deployment architecture should treat Gate 3 as a **routing signal**: when the Fisher score falls in the CW region, route the item to a fallback action (refuse, retrieve, or escalate to human review).

**Gate 3 for reasoning models.** Commit-point hidden states (EXP-F) give AUROC=0.65 — below the 0.70 INFORMED threshold (BLIND verdict, provisional N=40/class, single architecture). Commit timing is identical for CC and CW items (commit_pct 98.0% vs 97.9%), confirming that the model commits at the same point in its think block regardless of whether it will produce a correct or incorrect answer. Commit-point hidden states at L26 are not a reliable Gate 3 signal at current N. Gate 3 for reasoning models remains an open experimental question requiring validation at larger N and across architectures before deployment recommendations can be made.

### 6.3 The Three-Layer Research Program

The three-task hierarchy organizes **what** is measurable. The underlying research program asks **why** it is measurable:

- **Layer 1 (Measurement):** What can be read from residual streams, and with what instrument? Current status: substantial results at L1–L3.
- **Layer 2 (Laws):** When does observability emerge? What governs its magnitude? Current status: z-amplification pattern established (z×13 from base to reasoning-distilled). EXP-K (Pythia checkpoint sweep) was stopped after discovering a protocol applicability constraint (C026): pure autoregressive base LMs produce CTX_DEP=0 regardless of training stage, because they generate text continuation rather than following QA instructions. The INVERTED_U hypothesis (C011, provisional) cannot be tested with the current bilateral oracle on Pythia base models. Training dynamics require a model family with instruction-tuned checkpoints at multiple stages (e.g., OLMo-2) or a modified oracle designed for base LM evaluation. EXP-L is being redesigned accordingly.
- **Layer 3 (Design):** How should future AI systems be built to remain observable by construction? Current status: aspirational. 5-year horizon.

### 6.4 What Remains Unknown

Four scientific questions organize the open program:

| Q | Question | Status |
|---|---|---|
| Q1 | What internal properties can be read from residual stream? | Active — L1/L2/L3 hierarchy established |
| Q2 | Is what we observe causal? | Single null result (C005, centroid patching); L2/L3 causal test pending (EXP-H) |
| Q3 | How does observability emerge? When and why? | Provisional — INVERTED_U + z-amplification | 
| Q4 | What is invariant? Which properties survive prompt/language/architecture variation? | **ANSWERED (EXP-J):** ICC=0.913 (ROBUST). Fisher+PCA64 score stable under 4 surface perturbation types (between/within variance ratio=10.5:1). PARAM/CTX_DEP separation preserved under all variants (t=23.021, p<0.0001). Single architecture (Qwen2.5-1.5B-Instruct); language/architecture invariance pending. |

### 6.5 Candidate Scientific Laws

The experiments in this paper collectively produce a set of recurring quantitative patterns that appear across architectures, tasks, and experimental conditions. We state them as *candidate laws* — empirical regularities that have survived multiple tests and are now making predictions for future experiments. We do not call them laws; we call them candidates because (a) the experiments are still accumulating, (b) the governing mechanism is not yet identified, and (c) architectural generalization to non-transformer systems is untested. A candidate law becomes a law when it has: survived an out-of-distribution test, been given a mechanistic account, and made a novel successful prediction.

**Candidate Law 1 — Observability is measurable across transformer decoders (O)**

*Statement:* A bilateral oracle probe (Fisher+PCA64 at layer L_peak, step-1) achieves AUROC ≥ 0.70 for knowledge-source classification (PARAM vs CTX_DEP) on any instruction-tuned transformer decoder that falls within the Goldilocks capability zone for the evaluation task.

*Current evidence:* AUROC ∈ [0.731, 0.846] across five independent architectures (Qwen, Llama, Gemma, Mistral, Phi) at parameter scales 1.5B–7B on TriviaQA. ICC=0.913 under surface perturbations. Shuffled controls ∈ [0.499, 0.598] in all cases. Phi-3.5-Mini-Instruct sets the new high (0.8456, C044).

*Result (Phi-3.5-Mini-Instruct, C044):* AUROC=0.8456, CI=[0.758, 0.921], shuffled=0.4992 (CLEAN). L2 also SUPPORTED (Fisher=0.712, gap=0.122, BO_Transfer=0.792). Prediction confirmed — Law 1 now rests on 5 architectures.

*Falsification condition:* Any instruction-tuned transformer decoder within the Goldilocks zone achieving AUROC < 0.65 with clean shuffled control.

---

**Candidate Law 2 — Commitment precedes verbalization in reasoning systems (C)**

*Statement:* In reasoning-distilled language models, the hidden-state geometry commits to an answer direction after processing ≤ 20% of thinking tokens. The remaining ≥ 80% of the think block is post-commitment elaboration that contributes negligible quality: truncating at the commit point changes final answer F1 by < 0.02.

*Current evidence:* commit_pct ∈ [75.8%, 82.9%] across two DeepSeek-R1-Distill families (z=49.77, z=679.73 vs shuffled null). EXP-I: truncating at 87.4% savings costs +0.006 F1 (p=0.08). Replicated by independent concurrent work (arXiv 2605.06723: 17–31 token pre-verbalization commitment). **Teacher-independence confirmed (C045):** Qwen3-1.7B (non-R1, Qwen-native reasoning) commit_pct=99.8%, z=1.3×10¹⁵, N=100/100 — more extreme than both R1-distill models. Law 2 is teacher-independent.

*Prediction for next test (teacher-independent reasoning model):* **CONFIRMED (C045).** Qwen3-1.7B commit_pct=99.8% >> 70% threshold; teacher confound resolved.

*Falsification condition:* A reasoning model with commit_pct < 50% (commitment evenly distributed through chain) OR a model where truncation at the Fisher-detected commit point causes Δf1 < −0.05.

---

**Candidate Law 3 — Training stage effects on observability are task-level-dependent (A)**

*Statement:* Post-training (SFT, RLHF) and reasoning distillation affect L1 and L2 observability differently, and effects on L1 are task-distribution-dependent. Specifically: (a) L2 confabulation detection gap increases monotonically through all training stages; (b) L1 knowledge-routing observability improves with SFT but declines with reasoning distillation trained on a different task distribution.

*Result (exp_l_stage_sweep_v2, matched N=200/class):*

| Stage | L1 AUROC | L2 Fisher | L2 Entropy | L2 Gap |
|---|---|---|---|---|
| BASE (Qwen2.5-1.5B) | 0.7396 | 0.6528 | 0.5889 | 0.0639 |
| INSTRUCT (Qwen2.5-1.5B-Instruct) | 0.8028 | 0.6736 | 0.5776 | 0.0960 |
| REASONING (DeepSeek-R1-Distill-Qwen-1.5B) | 0.7252 | 0.7584 | 0.6106 | 0.1478 |

*Verdict:* **INVERTED_U on L1** — SFT improves L1 (+0.063 vs BASE); reasoning distillation reduces L1 to below BASE (−0.014 vs BASE, −0.078 vs INSTRUCT). Explanation: reasoning distillation on mathematical chain-of-thought creates task-distribution mismatch for TriviaQA factual routing. **MONOTONE_RISE on L2** — confabulation detection gap grows through every stage (C042/C043). Original Law 3 prediction (MONOTONE_RISE on L1) is partially falsified; L2 monotonicity is a new positive finding.

*Current evidence:* C042 (INVERTED_U on L1), C043 (MONOTONE_RISE on L2). C033: B3_top1_prob=0.384 (below chance), consistent with RLHF-induced output confidence decoupling (single architecture). C029: Fisher AUROC improves 0.731→0.840 at 7B-Instruct. Literature: Neutral Mask (arXiv 2606.09735), Tracing Geometry (arXiv 2509.23024).

*Falsification condition:* L2 Fisher gap decreases from Base→Instruct at matched N. (L1 falsification already observed; L2 monotonicity is the remaining testable component.)

---

**Candidate Law 4 — Approach-commitment geometry predicts outcome before computation begins (C)**

*Statement:* At the first reasoning token (step-1), before any chain-of-thought computation has occurred, the residual stream geometry at layers L_peak−1 to L_peak predicts final task correctness with AUROC ≥ 0.85 for mathematical reasoning tasks.

*Current evidence:* C039: MATH-500, DeepSeek-R1-Distill-Qwen-1.5B, layer 25–26, AUROC=0.8558 (CI=[0.729,0.957], N=100/class; original small-N estimate was 0.9111). C038 (negative): intermediate-step geometry does not track oracle PRM (r=0.053), confirming that this is an entry-point predictor, not a process tracker.

*Prediction for next test (second reasoning model on MATH, or second mathematical task):* AUROC ≥ 0.85 at L_peak, step-1.

*Falsification condition:* A reasoning model or task where step-1 AUROC < 0.75, confirmed with clean shuffled control.

---

**Invariants that underlie the candidate laws:**
The four laws share a common structural pattern: in each case, the hidden-state geometry at an early moment (step-1, onset of problem processing, start of reasoning chain) contains information about an epistemic outcome that will only be revealed much later in generation or in final evaluation. This temporal compression — the future is readable from the present in the geometry before the output reflects it — is the central empirical regularity of the program. It exists at L1 (geometry at step-1 separates knowledge sources), L2 (geometry at step-1 separates correct from confabulated confident answers), L3 (geometry at reasoning step-1 separates early-commit from late-commit), and the entry-point regime (geometry at reasoning step-1 predicts MATH final correctness at AUROC=0.8558, N=100/class corrected). Whether this temporal compression is a universal property of trained neural networks or a transformer-specific artifact is the primary open scientific question.

---

### 6.6 Scope of the Science

This program measures *computational organization* — the geometric structure of residual stream representations as shaped by training, accessible through probing, and quantified by the O, C, A framework. Stating explicitly what the program does not attempt prevents the most common class of reviewer misreading.

**The program does not attempt to:**

1. *Explain circuits or mechanisms.* Fisher+PCA64 reads a geometric property of the residual stream; it does not identify which attention heads, MLP neurons, or superposition features produce that geometry. Activation patching (C005, C024) establishes that the geometry is non-causal under centroid-direction, full-residual-stream patching at the tested layers and magnitudes — but this is a property of the tested intervention, not a claim that the signal has no mechanistic basis. Head-level patching, SAE feature patching, and circuit-level intervention remain untested.

2. *Provide a theory of organization.* The program establishes that optimization produces measurable regularities in O, C, A. It does not explain why. Four competing mechanistic theories remain undiscriminated (§8.1: Information Bottleneck, Routing Optimization, Predictive Coding, Architectural Determination). The transition from *measurement science* to *explanatory science* is the next major open question. It is not attempted here.

3. *Measure ground-truth epistemic state.* The bilateral oracle labels PARAM/CTX_DEP through behavioral intervention — two separate inference passes with explicitly operationalized criteria (§2.1). It does not access "what the model knows" in any theory-independent or introspective sense. O measures recoverable information through probes under a specified intervention protocol; it is an external measurement of a behavioral distinction, not a direct readout of internal epistemic state.

4. *Generalize across architectural paradigms.* All results are on decoder-only transformer architectures (Qwen, Llama, Gemma, Mistral, Phi). Whether O, C, A are measurable in SSMs (Mamba, RWKV), hybrid architectures, or latent diffusion models is entirely untested. The candidate laws are explicitly stated for "instruction-tuned transformer decoders within the Goldilocks capability zone."

5. *Generalize across task formats.* The bilateral oracle probe transfers within similar task formats (TriviaQA→HotpotQA, 84.5% efficiency) but fails across format boundaries (TriviaQA→MMLU, ~random; C041). All observability claims are format-scoped. Cross-format deployment requires per-format recalibration.

6. *Provide a deployment API.* The three-gate architecture (§6.2) is a safety framing derived from the measurement results. End-to-end deployment evaluation — latency under production load, false positive rates at scale, integration with retrieval systems, adversarial robustness — is outside the scope of this paper.

**What the program does establish.** That O, C, A are measurable, reproducible, non-trivially nonzero, architecture-consistent within the tested transformer decoder class, training-stage-dependent in predictable directions, and format-scoped. The introduction of three measurable quantities — independent of any particular estimator for them — is the scientific contribution. A future paper that measures O with a superior estimator (SAE features, contrastive probes, circuit-level readout) and finds the same architectural and training-stage regularities would strengthen the program, not compete with it.

---

## 7. Falsification Record

The reliability of the instrument depends on the falsification record being as prominent as the positive results.

### 7.1 C012 — RLHF Attenuation (FALSIFIED)

**Original claim:** RLHF-aligned instruct models show AUROC Δ=−0.036 lower than base models, measured by Fisher LDA.

**Mechanism of failure:** Fisher LDA with `solver='svd'` at N=60/class, d=1536 produces degenerate covariance estimates. The degenerate estimator inflated instruct-model AUROC and deflated base-model AUROC in a way that appeared as RLHF attenuation. After switching to `solver='lsqr'` with `shrinkage='auto'` + PCA64, instruct AUROC exceeded base AUROC (+0.127). The direction was reversed.

**Lesson:** Always check Fisher LDA's degenerate covariance mode (reported as near-zero eigenvalues) before interpreting LDA results. N/d ratio < 0.05 is dangerous territory without PCA.

### 7.2 C013 — Llama Weakness (FALSIFIED)

**Original claim:** Llama-3.2-3B-Instruct AUROC ≈ 0.629, suggesting architecture-specific weakness.

**Mechanism of failure:** Same degenerate covariance pathology, amplified for Llama (d=3072 vs Qwen d=1536). The larger hidden dimension amplified the degenerate estimator's bias. After PCA64 + lsqr: Llama=0.846, equal to Qwen.

### 7.3 C014 — Nonlinear Probe Recovery (FALSIFIED)

**Original claim:** Neural network probes recover Δ > 0.05 above the Fisher+PCA64 baseline.

**Mechanism of failure:** Contaminated baseline from C3-v1/v2 labeling issues. With correct PARAM-vs-WRONG exclusion and clean labels, nonlinear probes show no recovery. Linear geometry is sufficient.

### 7.4 C008 — Fisher ⊥ Entropy (FALSIFIED)

**Original claim:** r(Fisher_score, output_entropy) ≈ 0.0039, Fisher and entropy are structurally independent.

**Mechanism of failure:** The correlation was measured on J-score (a composite measure), not on Fisher AUROC. When measured correctly on individual item Fisher LDA scores vs output entropy:
- Qwen: r=−0.225 (r²=0.05, p<0.05)
- Llama: r=−0.544 (r²=0.30, p<0.0001)

Fisher and entropy are correlated, with architecture-dependent magnitude. The independence claim is falsified. The correct statement: for the knowledge-source routing task (L1), Fisher is redundant with entropy; for confabulation detection (L2, entropy-matched), Fisher adds substantial predictive value.

---

## 8. Competing Theories and Discriminating Experiments

### 8.1 Competing Theoretical Frameworks

The observed patterns in this paper are consistent with multiple mechanistic theories. We list them explicitly alongside what each predicts, and which current evidence distinguishes among them. We do not select one theory as correct until a discriminating experiment forces a choice.

| Theory | Core Claim | Prediction for EXP-K Checkpoint Curve | Prediction for CC/CW gap |
|---|---|---|---|
| **Information Bottleneck** | Observability peaks at maximum compression (mid-training), decays at convergence | INVERTED_U: peak at step 16k–33k | Large gap — model has compressed out spurious correlations |
| **Routing Optimization** | Observability grows as model learns to specialize computation by knowledge type | MONOTONE_RISE: asymptote by step 100k | Moderate gap — routing specializes by knowledge availability, not validity |
| **Predictive Coding** | Observability reflects structured prediction error signals | PHASE_TRANSITION: rapid rise at step 1k–8k | Gap depends on prediction error structure for CC vs CW |
| **Architectural Determination** | Observability is fixed by attention architecture, not training | FLAT: variance < 0.02 across all checkpoints | Gap is architecture-specific, training-invariant |

**Current evidence status:** EXP-K (Pythia checkpoint sweep) was stopped — the bilateral oracle protocol is inapplicable to pure autoregressive base LMs (C026: CTX_DEP=0 at all Pythia checkpoints). EXP-L (Qwen2.5-1.5B stage sweep) completed with EXPLORATORY results (C030): NON_MONOTONE trajectory confounded by N differences across stages; exp_l_stage_sweep_v2 at matched N=200/class is the required clean test. The INVERTED_U (provisional, Pythia small-N) is most consistent with Information Bottleneck. The CC/CW gap (0.240) being larger than PARAM/CTX_DEP gap (0.060) is consistent with knowledge validity being a more compressed feature than knowledge availability — but all four theories remain live.

**Candidate Retrieval-Quality Hypothesis (interpretive, not a finding):** The Fisher axis may be measuring "parametric retrieval quality" — how well the retrieval of stored knowledge succeeded — rather than "knowledge source type." C010 (CTX_DEP ≈ CONFAB geometrically on Dim1), C018 (bilateral oracle probe transfers to CC/CW with AUROC=0.880), and C017 are all consistent with this interpretation. However, at least three competing models remain undiscriminated (retrieval quality, post-retrieval certainty, memory-generation consistency). This interpretation should appear in Discussion only, not in Results.

---

## 9. Pending Experiments and Kill Criteria

The following experiments are scripted and ready to run. Kill criteria are pre-registered: if a kill criterion triggers, the corresponding claim or framework is revised before continuing.

| Exp | Status | Question | Kill Criterion | Result |
|---|---|---|---|---|
| EXP-F | **COMPLETE** | Fisher at commit point → CC/CW? | AUROC ≈ 0.50 | BLIND: AUROC=0.650, N=40/class; commit_pct identical for CC/CW (98.0% vs 97.9%). Sample-size limitation vs genuine null undetermined. |
| EXP-H | **COMPLETE** | Patching CC centroid improves F1? | Null at all λ | EPIPHENOMENAL: max Δ_F1=+0.0004. Kill triggered. Geometry predicts CC/CW but patching cannot correct. |
| EXP-J | **COMPLETE** | Signal survives 4 surface perturbations? | ICC < 0.50 | **ROBUST: ICC=0.913.** between/within=10.5:1. PARAM/CTX_DEP separation preserved (t=23.021). Kill not triggered. |
| Llama replication | **COMPLETE** | C017/C018 replicate on Llama-3.2-3B? | Fisher AUROC < 0.70 | **SUPPORTED:** Fisher=0.818, gap=0.365, BO_Transfer=0.768. C017/C018 promoted to two-arch. |
| EXP-G | **COMPLETE** | Post-`</think>` entropy → FLAT or BURST? | (observational) | **BURST** (prediction falsified): peak AUROC=0.6932 at step 4, traj AUROC=0.8424. CW collapse (1.322→0.127). Same peak step as base Qwen on TriviaQA. |
| EXP-I | **COMPLETE** | Early exit F1 cost at commit point? | mean Δf1 < −0.05 | **MINIMAL_QUALITY_LOSS:** Δf1=+0.006 ± 0.047, p=0.08. 87.4% token savings. Helped 7.5%, hurt 4.0%. Kill not triggered. |
| EXP-K | **STOPPED** | Pythia large-N INVERTED_U? | FLAT | **BILATERAL_ORACLE_INAPPLICABLE (C026):** CTX_DEP=0 at all Pythia checkpoints. Base LMs do not follow QA instructions; two-pass oracle requires instruction-following capability. EXP-L redesigned for Qwen base. |
| EXP-L | **COMPLETE** | Does C026 extend to Qwen2.5-1.5B-Base? Stage comparison Base→Instruct→Reasoning? | CTX_DEP=0 on Qwen base | **C026 PYTHIA-SPECIFIC (C030 EXPLORATORY):** Qwen2.5-1.5B-Base oracle APPLICABLE (CTX_DEP=50 from 232 scanned). Fisher AUROC=0.8876 but shuffled=0.8757 (net gap=0.012 — near-random at N=50/class). Entropy AUROC=0.7219 (clear). Stage comparison NON_MONOTONE (Base: 0.88→Instruct: 0.73→Reasoning: 0.76) but confounded by N differences across stages. |
| EXP-Scale | **COMPLETE** | Does Fisher signal survive above 3B Goldilocks ceiling? (7B-Instruct) | AUROC < 0.65 | **AUROC_SURVIVED (C029):** Fisher=0.8402, shuffled=0.6272, Entropy=0.9645, Δ=+0.11 vs 1.5B. 4-bit NF4, N=50/class from 486 scanned. C021 upper ceiling falsified for instruct models. |
| l2_large_n_v1 | **COMPLETE** | Stable cross-validated Fisher L2 AUROC (Tier 0). C034 variance resolved. | Fisher CV AUROC mean < 0.65 (not triggered) | **STABLE_SIGNAL (C040):** CV Fisher=0.7629±0.0120, CV Gap=0.1628, CV Entropy=0.6002±0.0440. N=500/class CO-style, θ_conf=0.7733, 5-fold CV, 71.5 min. Prior 0.670–0.885 range was pool-section heterogeneity. Kill NOT triggered. |
| OOD generalization (ood_generalization_v2) | **COMPLETE** | Does bilateral oracle probe generalize to MMLU, HotpotQA without retraining? (Tier 0 — defines scope of all observability claims) | AUROC drop > 25pp OOD (TASK_SPECIFIC) | **OOD_PARTIAL (C041):** Source AUROC=0.7744. MMLU transfer=0.5288 (within=0.9488) — TASK_SPECIFIC. HotpotQA transfer=0.6567 (within=0.7769, gap=0.120) — OOD_PARTIAL. Format-sensitive portability: same-format QA (TriviaQA→HotpotQA) achieves 84.5% transfer efficiency; cross-format (TriviaQA→MMLU) is essentially random. Kill criterion NOT triggered (claim becomes format-scoped, not revoked). |
| exp_l_stage_sweep_v2 | **COMPLETE** | Clean Base→SFT→Reasoning Qwen comparison at matched N=200/class. (Tier 0 — fills A quantity, tests Law 3 monotonicity prediction) | Fisher AUROC decreases Base→Instruct (Law 3 falsified) | **INVERTED_U on L1; MONOTONE_RISE on L2 (C042/C043).** L1: BASE=0.740, INSTRUCT=0.803, REASONING=0.725 — INVERTED_U, SFT helps, reasoning distillation reduces L1 below BASE (task-distribution mismatch). L2 Gap: 0.064→0.096→0.148 — MONOTONE_RISE across all stages. Kill criterion NOT triggered (L1 Law 3 portion revised; L2 monotonicity confirmed). 283 min elapsed. |
| phi_bilateral_v1 | **COMPLETE** | 5th architecture bilateral oracle — Phi-3.5-Mini (Tier 1 — tests Law 1 generalizability) | AUROC < 0.65 with clean shuffled control | **L1 AUROC=0.8456, CI=[0.758,0.921], shuffled=0.4992 CLEAN. Highest of 5 architectures. L2 SUPPORTED: Fisher=0.7120, gap=0.1216, BO_Transfer=0.7920. NO_TERMINATION. 212.9 min. C044 SUPPORTED.** |
| CO-on-Gemma/Mistral | **COMPLETE** | CO labeling on T2_L2 architectures (Tier 1 — resolves C036: framing failure vs. geometry absence) | Fisher gap < 0.05 on CO labels (T2_L2 = geometry absence, not framing failure) | **C036_CONFIRMED.** Gemma CO: Fisher=0.8368 CI=[0.749,0.910], gap=+0.2056, CLEAN (CO_RECOVERS). Mistral CO (8-bit): Fisher=0.8580 CI=[0.777,0.927], gap=+0.2114, CLEAN (CO_RECOVERS). T2_L2 = framing artifact. CO labeling is universal L2 estimator. 139 min total. |
| EXP_TEACHER_INDEPENDENCE_V1 | **COMPLETE** | Law 2 teacher confound (Qwen3-1.7B, non-R1) | commit% < 30% or z < 2 | **REPLICATED_AMPLIFIED.** commit_pct=99.8%, z=1.3×10¹⁵, N=100/100. More extreme than R1-distill models (75.8%/82.9%). Law 2 teacher-independent. C045. |
| perturbation_battery_v1 | **COMPLETE** | Multi-architecture ICC under surface form changes — C025 cross-architecture generalization | ICC < 0.50 (kill not triggered) | **C025 CONFIRMED (two architectures):** EXP-J Qwen ICC=0.913 (ratio=10.5:1) + perturbation_battery_llama_v1 Llama ICC=0.9334 (ratio=14.0:1). Both ROBUST. PARAM/CTX_DEP separation preserved under all 4 perturbation types in both architectures (t≥20, p<0.0001). C025 upgraded SUPPORTED→CONFIRMED. |
| EXP_MATH_AUROC_V2 | **COMPLETE** | C039 corrected N=100/class MATH entry-point AUROC (corrects small-N 0.9111 estimate) | AUROC < 0.75 (would require revising C039 and Law 4) | COMPLETE: AUROC=0.8558 CI=[0.729,0.957] N=100/class L25=L26 shuffled CLEAN kill=False. Corrects C039 small-N estimate. |
| sae_integration | **PENDING** | SAE/MLP features vs Fisher+PCA64 at L2 (Tier 3 — mechanism, not thesis-critical) | (observational) | *Not yet run.* Would discriminate Retrieval-Quality vs Internal-Certainty competing models (§8.1). |

### 9.1 Pre-registered Predictions for Remaining Experiments

The following predictions are frozen *before* results exist. Interpretations are pre-specified for all plausible outcome categories; they will not be revised after results are obtained. This practice converts the experiments from confirmatory to genuinely predictive.

---

**Experiment: Law 3 cross-family stage sweep** (Llama or Phi backbone, matched N=200/class, Base→SFT→Reasoning stages)

*Pre-registered prediction:* L1 observability will follow INVERTED_U (SFT improves, reasoning distillation reduces), and L2 gap will follow MONOTONE_RISE, replicating C042/C043 on the Qwen backbone.

| Outcome | Interpretation |
|---|---|
| A: Both INVERTED_U on L1 and MONOTONE_RISE on L2 replicate | Law 3 upgrades from single-family to cross-family. C042/C043 become CONFIRMED. The split between L1 task-sensitivity and L2 monotonicity is a robust property of the training pipeline, not a Qwen artifact. |
| B: L1 INVERTED_U replicates but L2 MONOTONE_RISE does not | L2 monotonicity is family-specific. The universal claim must be restricted to "L2 gap does not decrease through training." The mechanism for the Qwen L2 growth is backbone-dependent. |
| C: Neither replicates (e.g., L1 MONOTONE_RISE or L2 INVERTED_U) | Law 3 is model-family-specific. The training-stage effects on observability are not a universal property but depend on the specific curriculum, architecture, or distillation method. Major revision required. |
| D: L1 replicates but L2 gap is near zero on new backbone | New backbone does not support L2 confabulation detection at the base/SFT stages. Would suggest L2 geometry requires sufficient confabulation surface area, which varies with model assertiveness. |

*Kill criterion:* L2 gap decreases from Base→Instruct on the new backbone at matched N (Outcome B or D variant).

---

**Experiment: Law 4 second-architecture test** (DeepSeek-R1-Distill-Llama-8B on MATH-500, step-1 AUROC)

*Pre-registered prediction:* Fisher+PCA64 at step-1, penultimate layer, will achieve AUROC ≥ 0.80 for correct vs. incorrect MATH-500 problems before any reasoning chain begins.

| Outcome | Interpretation |
|---|---|
| A: AUROC ≥ 0.85 (meets Law 4 threshold) | Law 4 is architecture-consistent for reasoning-distilled models. Approach-commitment geometry at step-1 generalizes beyond Qwen backbone. Both architectures confirm: hidden state before reasoning predicts outcome after reasoning. |
| B: AUROC ∈ [0.75, 0.85) | Law 4 threshold (≥ 0.85) requires revision to ≥ 0.75. The entry-point predictor holds cross-architecture but at lower magnitude. Llama's geometry may be less tightly organized at the approach-commitment step. |
| C: AUROC ∈ [0.65, 0.75) | Entry-point prediction is architecture-dependent. Qwen's strong result (0.8558) reflects Qwen-family geometry, not a universal property. Law 4 scope must be restricted to Qwen-family or restated as EXPLORATORY. |
| D: AUROC < 0.65 with clean shuffled control | Law 4 FALSIFIED for Llama. The approach-commitment geometry at MATH step-1 is model-specific. Requires investigating whether Llama begins reasoning differently (no sharp geometry at token-1) or whether layer_idx needs retuning. |

*Kill criterion:* AUROC < 0.70 with clean shuffled control AND replicated in a second run — revise Law 4 scope.

---

**Experiment: SAE mechanism discrimination** (SAE features at L26 step-1 vs Fisher+PCA64 on CC/CW task)

*Pre-registered prediction:* SAE features will achieve AUROC ≥ 0.70 on the CC/CW task and will identify a small number (≤ 50) of features with high individual AUROCs — identifying the mechanistic correlate of the Fisher geometry.

| Outcome | Interpretation |
|---|---|
| A: SAE features achieve higher AUROC than Fisher AND identify ≤ 50 key features | Mechanism partially localized. Model A (Retrieval Quality) or B (Internal Certainty) moves from viable to preferred depending on feature semantics. The Fisher geometry is a compressed readout of these features. |
| B: SAE features achieve equivalent AUROC to Fisher but distributed across many features | The confabulation geometry is distributed, not localized. No single feature or small feature set captures the full Fisher signal. This is evidence against circuit-level localization and toward holographic organization. |
| C: SAE features achieve lower AUROC than Fisher | Fisher+PCA64 extracts information that SAE features (as implemented) do not recover. Could reflect SAE training quality, feature completeness, or that the relevant geometry is in the residual stream superposition space not well covered by current SAE dictionaries. Does not falsify the Fisher result. |
| D: SAE features identify features that semantically map to "retrieval success" | Strong support for Model A (Retrieval Quality) over Models B and C. The Fisher axis would be interpretable as a retrieval success readout. Would convert the measurement program from "geometry exists" to "geometry represents retrieval quality." |

*Kill criterion:* none pre-specified — this is a mechanism experiment, not a threshold test. Any outcome is informative.

---

## 10. Related Work

**Output monitoring approaches.** Temperature calibration (Guo et al., 2017) and Platt scaling operate post-logits, cannot access pre-compression structure. Semantic entropy (Kuhn et al., 2023) generates multiple samples to assess consistency — a different operational definition of uncertainty. P(IK) self-knowledge (Kadavath et al., 2022) elicits verbal confidence estimates, subject to RLHF-induced calibration shifts. None of these approaches access the bilateral oracle distinction: confident/uncertain conflates knowledge-source with output quality.

**Probing approaches.** Probing classifiers (Alain & Bengio, 2016; Tenney et al., 2019) test whether specific information is decodable from intermediate representations. Linear representation hypothesis (Elhage et al., 2023) establishes that many features are linearly encoded. Inference-Time Intervention (Li et al., 2023) demonstrates directional control of truthfulness. The bilateral oracle contribution is the labeling protocol, not the probe architecture: two-pass labeling separates knowledge-source from correctness in a way that prior probing work does not systematically apply.

**Uncertainty quantification.** Factual self-awareness (UKP-Lab, 2025) tests whether models distinguish known from unknown facts; their approach uses verbal elicitation. The confabulation detection result (L2) is complementary: we probe hidden states in the entropy-matched confident zone, where verbal elicitation has already failed (both CC and CW items have been selected for low output entropy, meaning the model does not verbally signal uncertainty).

**Reasoning chain analysis.** Prior work on chain-of-thought reasoning (Wei et al., 2022; Kojima et al., 2022) treats the think block as deliberation. The commitment timing result (L3) provides the first hidden-state evidence that models commit to an answer direction after processing ~17–18% of their think tokens — with post-commitment tokens representing 75–83% of the think budget. The causal experiment (EXP-I) establishes that truncating at the commit point costs +0.006 F1 (p=0.08) — nearly neutral quality loss for 87.4% token savings. Post-commitment tokens retain marginal value (helped 7.5% of items vs hurt 4.0%), but the think block is primarily post-commitment elaboration. A post-think entropy trajectory experiment (EXP-G, GSM8K) further establishes that the think block does not pre-resolve epistemic uncertainty: the BURST pattern (CW spike-then-collapse at step 4) persists identically in reasoning-distilled models after explicit CoT, mirroring results on base Qwen without reasoning. Together these results characterize the full reasoning chain dynamics: intra-chain early commitment (EXP-B/C), post-commitment elaboration nearly neutral to quality (EXP-I), and epistemic uncertainty persisting into the answer phase despite prior reasoning (EXP-G).

---

## 11. Conclusion

We have established a three-task measurement hierarchy for epistemic legibility in transformer language models. At L1 (knowledge-source routing), output entropy achieves 0.87–0.90 AUROC and Fisher+PCA64 is largely redundant. At L2 (confabulation detection), Fisher is essential: it adds 0.240–0.365 AUROC over entropy across two architectures, and the bilateral oracle probe transfers to CC/CW detection with AUROC=0.880/0.768 (Qwen/Llama). At L3 (commitment timing), reasoning-distilled models commit after 17–18% of think tokens; truncating at the commit point saves 87.4% of compute at +0.006 F1 cost (p=0.08). A post-think entropy experiment (EXP-G) further establishes that explicit reasoning does not pre-resolve epistemic uncertainty — the same BURST pattern (peak at step 4) persists after the think block.

The hierarchy has a precise structural implication: the appropriate signal changes at each level (entropy → Fisher → Fisher trajectory), and difficulty increases at each level. A deployment safety architecture must implement all three gates: routing (Gate 1–2) is necessary but not sufficient; confabulation detection (Gate 3) requires Fisher and targets the safety gap where confident wrong answers are indistinguishable by entropy alone. Beyond early exit applications, commitment timing (C) may generalize as a computational primitive: if tool invocation, verification decisions, and planning transitions occur preferentially post-commitment, the commit point may serve as a natural synchronization boundary for model-external actions. That hypothesis is untested in this paper but follows structurally from the commitment results.

A protocol applicability constraint was discovered: the bilateral oracle requires instruction-following capability and is inapplicable to pure autoregressive base LMs (C026 — CTX_DEP=0 at all Pythia checkpoints tested). Training dynamics experiments (EXP-L) require instruction-tuned model families at multiple stages.

Seven claims are confirmed (C001–C004, C025, C036, C040), twenty-six are supported, eight are exploratory, and four are falsified — all maintained in an open claims registry (C001–C045). The falsification record is not an appendix; it is the mechanism by which the measurement instrument earned its reliability.

All three Tier 0 experiments are now complete. l2_large_n_v1 (C040: STABLE_SIGNAL, CV AUROC=0.7629 ± 0.0120) resolved the L2 variance question. The OOD generalization test (C041: OOD_PARTIAL) established format-sensitive portability: the bilateral oracle probe transfers within similar task formats (TriviaQA→HotpotQA at 84.5% efficiency) but not across format boundaries (TriviaQA→MMLU at ~random), consistent with the independently published "Geometries of Truth Are Orthogonal" finding (arXiv 2506.08572). exp_l_stage_sweep_v2 (C042/C043) revealed a split in Law 3: L1 observability shows INVERTED_U across training stages (SFT improves, reasoning distillation on a different task distribution reduces L1 below base — task-distribution-dependent); L2 confabulation detection gap is MONOTONE_RISE through all stages (0.064→0.096→0.148), with reasoning distillation producing the largest confabulation separation. Together these results characterize the A quantity: training stage effects on observability are task-level-dependent — the same distillation step that strengthens confabulation detection weakens factual routing geometry when the distillation task distribution diverges from the measurement task. The program has produced 45 claims (7 CONFIRMED, 26 SUPPORTED, 8 EXPLORATORY, 4 FALSIFIED), maintained in an open claims registry (Appendix A) with pre-specified falsification conditions for all pending experiments. The most recent results (C036_CONFIRMED, C025_CONFIRMED, C044) complete the first-phase architecture generalization program: CO-labeled L2 confabulation geometry holds in all five tested architectures, Law 1 (L1 AUROC ≥ 0.70) is confirmed across the full tested family, and perturbation invariance (ICC ≥ 0.91) is confirmed across two independent architectures.

---

## Appendix A — Claims Registry

| ID | Claim (abbreviated) | Status |
|----|---------------------|--------|
| C001 | Fisher+PCA64 AUROC ≥ 0.70 on bilateral oracle, L26, TriviaQA — large_n_v2: Qwen=0.7312 [0.63,0.83] N=197; Llama=0.7464 [0.65,0.83] N=200. Both CLEAN. | CONFIRMED |
| C002 | Signal is linearly organized — nonlinear probes Δ ≤ −0.019 | CONFIRMED |
| C003 | Architecture consistency — Qwen=0.7312 vs Llama=0.7464, Δ=0.015, CIs overlap, both CLEAN | CONFIRMED |
| C004 | Bilateral oracle labels separable in Fisher+PCA64 space | CONFIRMED |
| C005 | Centroid-direction patching epiphenomenal at L4–L26 (Qwen only) | SUPPORTED |
| C006 | Step-1 > prefill (0.785 vs 0.567 mean, 4 model families, ESM v33 lighter calibration) | SUPPORTED |
| C007 | Fisher trajectory AUROC=0.9947 (28 J-scores) | SUPPORTED |
| C008 | ~~Fisher ⊥ entropy (r≈0.0039)~~ | **FALSIFIED** — Qwen r=−0.225, Llama r=−0.544 p<0.0001 |
| C009 | Task-specific geometry (cross-task cosims near noise floor) | EXPLORATORY |
| C010 | CTX_DEP ≈ CONFAB geometrically | EXPLORATORY |
| C011 | Legibility present from early training; INVERTED_U provisional. Capability floor discovered: step512–step2000 Pythia checkpoints produce 0% bilateral oracle yield — measurement requires a model with baseline factual capability | EXPLORATORY (small-N; capability floor being characterized) |
| C012 | RLHF attenuation Δ=−0.036 | **FALSIFIED** |
| C013 | Llama weakness (AUROC ≈ 0.629) | **FALSIFIED** |
| C014 | Nonlinear probe recovery Δ > 0.05 | **FALSIFIED** |
| C015 | Bimodal geometry — STRONG poles ±1.316, WEAK/BORDERLINE indistinguishable | EXPLORATORY |
| C016 | Output entropy step-1 AUROC 0.87–0.90 for bilateral oracle classification | SUPPORTED |
| C017 | Fisher+PCA64 AUROC=0.854/0.818 for CC vs CW (entropy-matched); entropy=0.614/0.453; gap=0.240/0.365 on Qwen/Llama | SUPPORTED (two architectures) |
| C018 | BO_Transfer AUROC=0.880/0.768 — bilateral oracle probe transfers to CC/CW on Qwen/Llama | SUPPORTED (two architectures) |
| C019 | Entropy trajectory 15 steps AUROC=0.730; inversion at step 2–3 | SUPPORTED (Qwen only) |
| C020 | Architecture family predicts bilateral oracle AUROC better than parameter count | SUPPORTED (n=3 qualifying) |
| C021 | Goldilocks zone: ~1B–2B parameters for TriviaQA bilateral oracle | SUPPORTED |
| C022 | Early commitment across two DeepSeek-R1-Distill families: commit%=75.8%/82.9%, z=49.77/679.73. Teacher-independence confirmed by C045 (Qwen3-1.7B, non-R1, commit_pct=99.8%, z=1.3×10¹⁵). | SUPPORTED (two families; teacher-independence confirmed — see C045) |
| C023 | Commit-point hidden states at L26 not reliably distinct for CC vs CW at N=40/class (AUROC=0.65, BLIND); commit timing identical for CC and CW (commit_pct 98.0% vs 97.9%) | EXPLORATORY (single architecture, small-N) |
| C024 | Centroid-direction patching epiphenomenal at CC/CW domain: max Δ_F1=+0.0004 across λ=0.1–2.0 and all tested layers. Extends C005 null to L2 domain. | SUPPORTED |
| C025 | Fisher+PCA64 decision scores at L26 step-1 are invariant under 4 surface question perturbations. **Qwen2.5-1.5B-Instruct (EXP-J):** ICC=0.913 (ROBUST, threshold=0.70). between_var=2.737, mean_within_var=0.260 (ratio=10.5:1). PARAM/CTX_DEP separation preserved under all variants (t=23.021, p<0.0001, N=160). **Llama-3.2-3B-Instruct (perturbation_battery_v1):** ICC=0.9334 (ROBUST). between_var=2.253, mean_within_var=0.161 (ratio=14.0:1). REPHRASE/LOWERCASE/APPEND/TYPO correlations ≥0.91. sep_preserved (t=20.264, p<0.0001). Both architectures: ROBUST. | CONFIRMED (two architectures) |
| C026 | The bilateral oracle protocol requires a minimum instruction-following capability. Pure autoregressive base LMs (Pythia-1.4b) produce CTX_DEP=0 across all tested training checkpoints (step16k–step143k) despite PARAM~60 items/checkpoint. Base models generate text continuation, not context-dependent QA responses. Protocol requires instruction-tuned (SFT/RLHF) models. | SUPPORTED |
| C027 | Post-`</think>` answer entropy BURST pattern (peak AUROC=0.6932 at step 4, trajectory AUROC=0.8424) observed in DeepSeek-R1-Distill-Qwen-1.5B on GSM8K (N=50/class). Same BURST pattern and same peak step as base Qwen on TriviaQA (EXP-D). Think block does not pre-resolve epistemic uncertainty. CW items collapse from high entropy (step 1: 1.322) to near-zero (step 4: 0.127); CC items stay moderate (step 4: 0.318). | SUPPORTED (difficulty confound: CC rate 11% on GSM8K test with greedy decoding) |
| C028 | Truncating reasoning chain at commit point: 87.4% token savings, mean Δf1=+0.006 ± 0.047 SD (N=199/200 committed; p=0.08 two-tailed, p=0.04 one-tailed pre-registered). Helped 7.54% of items (Δ>0.05); hurt 4.02% (Δ<−0.05). Commit probe works as timing trigger on 99.5% of items. Elaboration is nearly neutral with slight asymmetric directional benefit. | SUPPORTED |
| C029 | Fisher+PCA64 AUROC improves 1.5B→7B instruct (0.73→0.8402, +0.11). Entropy AUROC reaches 0.9645 at 7B-Instruct. Bilateral oracle viable at 7B (50/50/486 scanned). C021 upper ceiling falsified for instruct models — base-model-specific behavior. | SUPPORTED (single family, 4-bit NF4 quant) |
| C030 | C026 bilateral oracle failure is Pythia-specific. Qwen2.5-1.5B-Base: oracle APPLICABLE (CTX_DEP=50/232 scanned). Fisher net gap=0.012 (near-shuffled at N=50/class — ambiguous). Entropy=0.7219 (clear). Stage comparison NON_MONOTONE but N-confounded. | EXPLORATORY |
| C031 | Mistral-7B-Instruct-v0.3 L1 bilateral oracle: AUROC=0.7780, shuffled=0.5524, CLEAN, CI=[0.692, 0.863], N=200/class, scanned=3795. (Superseded by Phi-3.5-Mini-Instruct C044 as highest L1 AUROC.) | SUPPORTED |
| C032 | CO labeling (correct vs. any incorrect, N=200/class, no entropy matching) achieves AUROC=0.885 on Qwen2.5-1.5B-Instruct, vs. BO (entropy-matched PARAM/CTX_DEP, N=100/class) AUROC=0.806. Delta=+0.079. CO outperforms BO because CTX_DEP items are geometrically closest to PARAM (borderline cases), diluting BO signal. CO labeling removes entropy-matching dependency entirely. | SUPPORTED (single architecture) |
| C033 | Fisher+PCA64 AUROC=0.845 vs. best behavioral baseline B2_self_consistency=0.613, gap=+0.232 on entropy-matched L2 (Qwen2.5-1.5B-Instruct). Critical finding: B3_top1_prob=0.384 — **below chance** — confabulations have MORE peaked wrong-token distributions, consistent with RLHF-trained assertiveness decoupling output confidence from epistemic state. Fisher reads geometry upstream of the output head's trained projection. Mechanism unconfirmed; exp_l_stage_sweep_v2 is the primary test. | SUPPORTED (single architecture) |
| C034 | CO and BO labeling give statistically equivalent Fisher L2 AUROC within the same pool section (CO=0.670, BO=0.670, diff=−0.016). Variance 0.670–0.854 at n_test=50 was pool-section heterogeneity. **RESOLVED by C040.** | SUPPORTED (variance explained) |
| C035 | Mistral-7B-Instruct-v0.3 L2: Fisher=0.5952, gap=−0.014, T2_L2 NOT_SUPPORTED. theta_conf=0.122 (low-entropy model). BO_Transfer=0.6624 — geometry exists; entropy-matched framing fails. T2_L2 triggers for Mistral for same reason as Gemma: low theta_conf degenerates entropy-matched CC/CW window. | SUPPORTED |
| C036 | **CONFIRMED (C036_CONFIRMED):** CO labeling recovers L2 for both T2_L2 architectures. Gemma CO (θ_conf=0.7562): Fisher=0.8368 CI=[0.749,0.910], Gap=+0.2056, CLEAN (CO_RECOVERS). Mistral CO (θ_conf=0.0220, 8-bit): Fisher=0.8580 CI=[0.777,0.927], Gap=+0.2114, CLEAN (CO_RECOVERS). T2_L2 = entropy-matched framing artifact (degenerate window for low-θ_conf). CO labeling is the universal L2 estimator across all 4+ tested architectures. | CONFIRMED |
| C037 | Early exit causal validation: n_committed=199/200 (99.5%), mean Δf1=+0.0059±0.0474 SD, p=0.08 two-tailed, p=0.04 one-tailed (pre-registered directional). Mean commit%=87.4%. Helped 7.54%, hurt 4.02%. Commit probe functions as reliable timing trigger; truncation is nearly quality-neutral. (See C028 — same experiment.) | SUPPORTED |
| C038 | PRM_SIGNAL_WEAK: J_know at reasoning step k does NOT correlate with oracle PRM (Pearson=0.053, Spearman=0.047). Early J_know vs final correctness: r=−0.008, p=0.945. Hidden geometry is an entry-point predictor (C039), not a step-level process tracker. Free zero-label PRM hypothesis falsified. | SUPPORTED (single architecture, MATH-500) |
| C039 | MATH entry-point prediction: Fisher+PCA64 at step-1, layers 25–26, AUROC=0.8558 (corrected, N=100/class; original N=30 estimate was 0.9111, within CI) on MATH-500 (DeepSeek-R1-Distill-Qwen-1.5B). CI=[0.729,0.957]. Layer sweep (original N=30/class small-N): L24=0.8978, L25=0.9111, L26=0.9111, L27=0.9100. Corrected N=100/class estimate: AUROC=0.8558 at L25–L26. Highest step-1 AUROC across any task in the program. Geometry at first reasoning token predicts final MATH correctness before any CoT computation. Approach-commitment geometry, not process tracking. | SUPPORTED (single architecture, N=100/class corrected) |
| C040 | EXP_L2_LARGE_N_V1 STABLE_SIGNAL: CV Fisher L2 AUROC = 0.7629 ± 0.0120, CV Entropy = 0.6002 ± 0.0440, CV Gap = 0.1628. N=500/class CO-style (θ_conf=0.7733), 5-fold stratified CV, Qwen2.5-1.5B-Instruct, TriviaQA, 71.5 min. Fold range: 0.7440–0.7805. Resolves C034: prior 0.670–0.885 variance was pool-section heterogeneity + small n_test artifact. The stable CO L2 Fisher AUROC is 0.7629. Gap (0.1628) is lower than entropy-matched EXP-A (0.240) because CO-style entropy baseline is 0.60 (not forced to 0.50). Both measurements are valid: EXP-A measures Fisher's advantage when entropy has fully failed; l2_large_n measures Fisher's advantage across the full low-entropy population. | CONFIRMED |
| C041 | OOD_PARTIAL. TriviaQA bilateral oracle Fisher+PCA64 probe (N=200/class source, L26, Qwen2.5-1.5B-Instruct) shows format-sensitive portability. HotpotQA transfer AUROC=0.6567 (shuffled=0.5477, within=0.7769, gap=0.120, transfer efficiency=84.5%) — OOD_PARTIAL. MMLU-STEM transfer AUROC=0.5288 (shuffled=0.4496, within=0.9488) — essentially random, TASK_SPECIFIC. Verdict: the bilateral oracle geometry generalizes within similar task formats (open-ended Wikipedia factual QA: TriviaQA→HotpotQA) but not across format boundaries (factual recall→multiple-choice STEM). Independently corroborates the "Geometries of Truth Are Orthogonal" finding (arXiv 2506.08572) while providing finer-grained structure: within-format transfer is partial but non-trivial (84.5% efficiency). All L1/L2 observability claims are qualified as format-scoped; per-format recalibration is required for cross-format deployment. | SUPPORTED (single model, two OOD tasks) |
| C042 | exp_l_stage_sweep_v2 INVERTED_U on L1. Qwen family, matched N=200/class. BASE=0.7396 [0.638,0.834], INSTRUCT=0.8028 [0.706,0.888], REASONING=0.7252 [0.620,0.823]. SFT improves L1 observability (+0.063 vs BASE); reasoning distillation (DeepSeek-R1-Distill-Qwen-1.5B, trained on mathematical CoT) reduces L1 below BASE (−0.014). INVERTED_U verdict. Interpretation: task-distribution mismatch between reasoning distillation (MATH/code) and TriviaQA (factual recall) reduces parametric knowledge observability. REASONING model also showed lower PARAM yield (scanned=1919 vs BASE=920, INSTRUCT=1567), consistent with fewer TriviaQA facts in parametric memory. Original Law 3 L1 prediction (MONOTONE_RISE) is FALSIFIED. | SUPPORTED (single architecture family) |
| C043 | exp_l_stage_sweep_v2 MONOTONE_RISE on L2. L2 Fisher AUROC: BASE=0.6528, INSTRUCT=0.6736, REASONING=0.7584. L2 Gap: BASE=0.0639, INSTRUCT=0.0960, REASONING=0.1478. Confabulation detection signature strengthens monotonically through every training stage including reasoning distillation. L2 entropy baseline stable (0.577–0.611) — the improvement is Fisher signal growth, not entropy window artifact. Consistent with reasoning distillation increasing the assertiveness of incorrect outputs (stronger false certainty pattern), making CONFIDENT_WRONG items more geometrically separable. Extends C033 (single architecture, Instruct) to a full training-stage trajectory. | SUPPORTED (single architecture family) |
| C044 | Phi-3.5-Mini-Instruct bilateral oracle (EXP_PHI_BILATERAL_V1): L1 AUROC=0.8456, CI=[0.758,0.921], shuffled=0.4992 CLEAN, N=200/class, 2268 scanned, probe at L30/32. Highest L1 AUROC across all five tested architectures. L2: Fisher=0.7120, CI=[0.552,0.843], Entropy=0.5904, Gap=0.1216, BO_Transfer=0.7920, shuffled=0.5040 CLEAN — SUPPORTED. Termination=NONE (no T2_L2, consistent with θ_conf > 0.15). 5th architecture confirming Law 1. Architecture range now [0.731, 0.846] across Qwen(0.731), Llama(0.746), Gemma(0.753), Mistral(0.778), Phi(0.846). | SUPPORTED |
| C045 | EXP_TEACHER_INDEPENDENCE_V1: Qwen3-1.7B (Qwen-native reasoning, not R1-distilled). commit_rate=100/100, mean_commit_pct=99.8% (commits within first ~2 think tokens), null_mean=48.7%, z=1.3×10¹⁵, mean_think_len=766, cal_AUROC=0.9268. More extreme than R1-distill models (75.8%/82.9%). Teacher confound in C022 RESOLVED: early commitment is not specific to DeepSeek-R1 training. Law 2 teacher-independent. The entire think block is post-commitment elaboration in a non-R1 model. | SUPPORTED |

---

## Appendix B — Experimental Record

Full experiment registry with results, protocol fingerprints, and reproducibility notes maintained in `science/EXPERIMENTS.yaml` in the research repository.

---

## Appendix C — Formal Framework for Computational Observability

This appendix provides mathematical definitions of O, C, A that are independent of any particular estimator. The definitions establish these as scientific quantities — properties of a system that any sufficiently capable instrument should measure consistently — rather than measurements tied to Fisher+PCA64 or the bilateral oracle protocol.

### C.1 Observability (O)

**Definition.** Let M be a language model with hidden dimension d. Let $Z_L^{(t)} \in \mathbb{R}^d$ denote the residual stream hidden state at layer $L \in \{0,\ldots,N_L\}$ and generation step $t \in \{0,1,\ldots\}$. Let T be an *intervention protocol* that assigns binary labels $Y \in \{0,1\}$ to input queries through behavioral tests applied independently of hidden-state collection (a do() operator separating labeling from measurement). Let $\mathcal{F}$ be a class of probe functions $f : \mathbb{R}^d \to \mathbb{R}$. The **observability** of M at (L, t) under intervention T is:

$$O(M, L, t, T) = \sup_{f \in \mathcal{F}} \text{AUROC}(f(Z_L^{(t)}), Y_{do(T)})$$

where AUROC is the area under the receiver operating characteristic curve, and the supremum is taken over the probe class $\mathcal{F}$.

**Operationalization.** Fisher+PCA64 (PCA to 64 components → LDA with lsqr solver and shrinkage=auto) provides a lower bound on O for the linear probe class ($\mathcal{F}$ = linear functions). The bilateral oracle (§2.1) is a specific intervention $T_{BO}$: PARAM label iff nocontext F1 ≥ 0.50; CTX_DEP label iff nocontext F1 ≤ 0.05 AND withcontext F1 ≥ 0.50. The do() separation is operationalized by using separate inference passes: one to assign labels, one to collect hidden states. The current estimate $O(M, 26, 1, T_{BO}) \geq 0.70$ across five architectures is a lower bound on the true O under the linear probe class.

**Four properties (axioms for the quantity, not the estimator):**

1. *Estimator invariance.* O is a property of M, L, t, T — not of $\mathcal{F}$. A result that changes substantially when $\mathcal{F}$ changes (e.g., linear vs. nonlinear probe, Fisher vs. SAE features) is measuring estimator sensitivity, not O. The bilateral oracle L1 result is confirmed estimator-invariant within the linear class (C002: nonlinear probes Δ ≤ −0.019).

2. *Intervention dependence.* $O(M, L, t, T_1) \neq O(M, L, t, T_2)$ in general for different protocols $T_1, T_2$. The PARAM/CTX_DEP bilateral oracle ($T_{BO}$) and the CC/CW confabulation protocol ($T_{CC/CW}$) measure different epistemic distinctions at the same (M, L, t). Results under one protocol are not transferable to the other without the BO_Transfer experiment (§3.3, C018). OOD failures (C041) are intervention-scope violations: the probe was calibrated on one task distribution and applied to a different one.

3. *Layer monotonicity (observed).* In all tested models, $O(M, L_{peak}, 1, T) > O(M, 0, 1, T)$ — later layers carry more separable geometry than the embedding layer. Whether this is a necessary consequence of representation learning objectives or an architectural artifact is an open question.

4. *Temporal accessibility.* $O(M, L, 1, T) > O(M, L, t_{prefill}, T)$ for generation onset (step-1) vs. the final prompt token. Knowledge-source geometry is more accessible during generation than during prompt processing (ESM v33: step-1 mean=0.785 vs. prefill=0.567 across four models).

**Relationship to mutual information.** AUROC is a distribution-free measure of separability. For any fixed probe f, $\text{AUROC}(f(Z), Y)$ lower-bounds $I(Z; Y) / H(Y)$ under mild assumptions, making O a practical proxy for mutual information between the hidden state and the intervention-defined label.

---

### C.2 Commitment (C)

**Definition.** Let q be a query, let M be a reasoning model generating a think block of T tokens. Let $Z_0, Z_1, \ldots, Z_T$ denote the residual stream hidden states at each think token, and let A denote the final answer distribution. The **commit step** is:

$$t^*(M, q, \varepsilon_C) = \inf \{ t \in \{0,\ldots,T\} : H(A \mid Z_{0:t}, q) \leq \varepsilon_C \}$$

where H is conditional entropy and $\varepsilon_C$ is a pre-specified threshold. The **commitment fraction** is:

$$C(M, q) = \frac{T - t^*(M, q, \varepsilon_C)}{T}$$

This is the fraction of the think block occurring after the commit step. Operationally, commit\_pct = 100 × C(M, q), and the Fisher trajectory serves as a proxy for the entropy decrease that defines t*.

**Properties.** C ∈ [0, 1]. C = 1 means commitment at the first think token (entire block is post-commitment elaboration); C = 0 means commitment at the final think token. Observed: R1-Qwen C = 0.758; R1-Llama C = 0.829; Qwen3-native C = 0.998. All substantially above C = 0.5 (the null of uniformly distributed commitment).

C is defined over hidden states, not output text. Two models with different think-block lengths and surface content can have the same C if they commit at the same relative position in the trajectory. EXP-I (§5.7) establishes that $\mathbb{E}[f1_{full} - f1_{truncated\,at\,t^*}] = +0.006$, confirming that post-commitment tokens are nearly neutral elaboration — C is a timing property, orthogonal to correctness.

---

### C.3 Accessibility (A)

**Definition.** Let Θ denote a space of training configurations (BASE, SFT, RLHF, REASONING_DISTILLATION, identified by training stage and curriculum). **Accessibility** is the function:

$$A : \Theta \times \{\text{L1}, \text{L2}, \text{L3}\} \to [0, 1]$$

mapping each (training configuration θ, task level ℓ) pair to the observability of the model trained at θ:

$$A(\theta, \ell) = O(M_\theta, L_{peak}, t_\ell, T_\ell)$$

where $M_\theta$ is the model trained at configuration θ, and $(L_{peak}, t_\ell, T_\ell)$ are the standard measurement settings for task level ℓ.

**Empirical characterization (Qwen backbone, matched N=200/class):**

| Stage θ | A(θ, L1) | A(θ, L2) gap | Pattern |
|---|---|---|---|
| BASE | 0.740 | 0.064 | baseline |
| SFT | 0.803 | 0.096 | L1↑, L2↑ |
| REASONING | 0.725 | 0.148 | L1↓, L2↑↑ |

The divergence — A(θ, L1) INVERTED_U while A(θ, L2) MONOTONE_RISE — is the central Law 3 finding. The same training step simultaneously decreases L1 and increases L2 accessibility, confirming they measure distinct computational properties.

A does not measure model performance (accuracy, calibration, F1). A reasoning model can have higher task accuracy while A(θ, L1) decreases (C042). A is specifically about geometric organization of the residual stream as shaped by training.

**Theory of organization (open question).** The program establishes *that* A has a specific structure. It does not explain *why* training produces this structure. The four competing theories in §8.1 (Information Bottleneck, Routing Optimization, Predictive Coding, Architectural Determination) each predict a different functional form for A(θ, ℓ). Discriminating among them is the primary open scientific question for the program's second phase.

---

### C.4 The Four Candidate Laws as Formal Statements

**Law 1:** $\forall M \in \mathcal{M}_{Goldilocks}$, $O(M, L_{peak}, 1, T_{BO}) \geq 0.70$.

**Law 2:** $\forall M \in \mathcal{M}_{reasoning}$, $\mathbb{E}_q[C(M, q)] \geq 0.70$ over committed queries q.

**Law 3:** $\forall$ backbone B, $A(\theta_{REASONING}(B), L2) > A(\theta_{SFT}(B), L2) > A(\theta_{BASE}(B), L2)$; with L1 INVERTED_U: $A(\theta_{SFT}(B), L1) > A(\theta_{BASE}(B), L1) > A(\theta_{REASONING}(B), L1)$ for same-format factual routing tasks.

**Law 4:** $\forall M \in \mathcal{M}_{reasoning}$ on mathematical tasks, $O(M, L_{peak}-1, 1, T_{correctness}) \geq 0.85$.

Each law is a universally quantified claim over a specified model class. Falsification requires: a model from the specified class, clean shuffled control, and result below the stated threshold. The formal statement makes the falsification condition precise and unambiguous.

---

### C.5 Estimator Independence

O, C, A are estimator-independent quantities. The current estimators:

| Quantity | Current estimator | Estimator class |
|---|---|---|
| O | Fisher+PCA64 (LDA, lsqr, shrinkage=auto) | Linear probes |
| C | Fisher trajectory threshold crossing | Scalar function of residual stream sequence |
| A | O measured at each training stage | Inherits from O estimator |

A superior estimator achieving higher AUROC for the same (M, L, t, T) gives a better lower bound on O — strengthening the same claim. A future paper establishing that SAE features achieve $O(M, 26, 1, T_{BO}) = 0.90$ measures the same O with a better instrument; it does not contradict the Fisher result of 0.73. The bilateral oracle protocol and the O, C, A framework are the program's contribution. Fisher+PCA64 is the current best implementation.

---

*This preprint presents work in progress. All claims are formally registered with pre-specified falsification conditions. The claims registry, experiment registry, and raw result files are maintained in the associated research repository.*
