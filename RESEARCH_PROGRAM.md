# Research Program — Computational Observability Science

**Single source of truth. Updated 2026-07-10. LOCKED.**  
*If this document conflicts with another file, this one is correct.*

---

## THE CENTRAL HYPOTHESIS CHAIN

> **Optimization → Computational Organization → Observability → Adaptive Computation**

Learning systems that train on data develop organized internal representations of their computational state. That organization is externally measurable. The reliability of measurement depends on: the experimental protocol (bilateral oracle), the training stage (A), and the architecture. Measurement enables adaptive computation: systems that can be observed can be gated, routed, and stopped efficiently.

This chain survives architectural evolution. Every sufficiently advanced learning system will develop internal state organization. The question of whether that organization remains accessible from the outside becomes more important, not less, as systems become more capable and more opaque.

---

## THE THREE MEASURABLE QUANTITIES

**O — Observability**: Can external probes reliably read epistemic state from internal representations?  
*Measured by:* bilateral oracle protocol, Fisher+PCA64 at L_peak step-1, AUROC on L1/L2 tasks.

**C — Commitment**: When does internal computation settle on an answer, and can that moment be detected and used?  
*Measured by:* Fisher trajectory commit detection, commit%, entry-point prediction AUROC.

**A — Accessibility**: How does training — pretraining, SFT, RLHF, reasoning distillation, scale, architecture — shape the computational organization that O and C measure?  
*Measured by:* training stage comparison (exp_l), checkpoint sweep, RLHF anti-correlation, architecture spread.

---

## THE BILATERAL ORACLE PROTOCOL (LOCKED)

The core methodological contribution. General form:

> Controlled behavioral intervention → two-pass test → clean epistemic labels

Specific instantiation: PARAM (nocontext F1 ≥ 0.50) vs CTX_DEP (nocontext F1 ≤ 0.05 AND withcontext F1 ≥ 0.50). Items not meeting either criterion excluded. Hidden states always from nocontext pass.

This is an experimental design pattern, not a probe architecture. It separates epistemic state from output quality by construction. It can be instantiated with any estimator (Fisher, entropy, SAE features, attention patterns). The protocol, not the estimator, is the lasting contribution.

---

## THE THREE-TASK HIERARCHY

**L1 — Knowledge-source routing**: PARAM vs CTX_DEP. Signal: output entropy sufficient (0.87–0.90 AUROC). Fisher redundant. Bilateral oracle as labeling protocol is the contribution, not Fisher.

**L2 — Confabulation detection**: CONFIDENT_CORRECT vs CONFIDENT_WRONG, entropy-matched. Signal: Fisher essential (+0.240–0.365 AUROC over entropy). RLHF anti-correlation explains why behavioral baselines fail (B3_top1_prob=0.384, below chance). CO labeling (AUROC=0.885) removes entropy-matching dependency.

**L3 — Commitment timing**: commit%=75.8–82.9% across reasoning architectures. Truncation at commit costs +0.006 F1 (87.4% token savings). Entry-point predictor: MATH step-1 AUROC=0.9111 before any CoT begins. Activation patching epiphenomenal (Δf1=+0.0004) — geometry is a routing signal, not a control register.

---

## FOUR CANDIDATE SCIENTIFIC LAWS

**Law 1 — L1 Observability**: Bilateral oracle Fisher+PCA64 achieves AUROC ≥ 0.70 on any instruction-tuned transformer in the Goldilocks capability zone.  
*Evidence*: [0.731, 0.778] across 4 architectures (Qwen, Llama, Gemma, Mistral). ICC=0.913.  
*Next test*: Phi-3.5-Mini (prediction: ≥ 0.70).

**Law 2 — Commitment precedes verbalization**: In reasoning-distilled models, commit% ≥ 70%; truncation at Fisher-detected commit costs Δf1 < 0.02.  
*Evidence*: commit%=75.8%/82.9%, Δf1=+0.006.  
*Next test*: Teacher-independent reasoning model (prediction: commit% ≥ 70%).

**Law 3 — RLHF severs pathways without removing geometry**: Fisher AUROC increases or holds post-RLHF on hidden states; behavioral confidence calibration degrades.  
*Evidence*: B3_top1_prob=0.384 (anti-correlated), C029 (7B AUROC=0.840).  
*Next test*: exp_l stage sweep (prediction: Fisher Base ≤ Instruct ≤ Reasoning).

**Law 4 — Approach-commitment geometry predicts outcome before computation begins**: Step-1 geometry at L_peak predicts mathematical reasoning correctness at AUROC ≥ 0.85.  
*Evidence*: C039 MATH-500 AUROC=0.9111 at layers 25–26.  
*Next test*: Second reasoning model or second mathematical task.

---

## EXPERIMENT QUEUE (FROZEN)

### Tier 0 — Program-defining (run in order, one at a time)
1. **l2_large_n_v1** (running): N=500/class, 5-fold CV — fixes C034 variance, authoritative L2 AUROC
2. **OOD generalization test**: train on TriviaQA, test on MMLU/HotpotQA/GSM8K — defines O scope
3. **exp_l_stage_sweep_v2**: Base→SFT→Reasoning Qwen, matched N=200/class — fills A quantity, tests Law 3

### Tier 1 — Architecture coverage
4. **phi_bilateral_v1**: 5th architecture, Law 1 test
5. **CO-on-Gemma + Mistral**: Tests T2_L2 as framing failure vs. geometry absence

### Tier 2 — Robustness
6. **perturbation_battery_v1**: ICC under surface form changes

### Tier 3 — Mechanism
7. **sae_integration**: SAE/MLP features vs Fisher+PCA64 — not required for thesis

---

## PAPER STRUCTURE (LOCKED)

**Paper title**: "A Measurement Protocol for Computational Observability in Language Models: A Three-Task Hierarchy"  
**File**: `paper/v1_arxiv.md`  
**Target**: arXiv preprint → ICLR 2027

**Sections**:
1. Introduction (O/C/A framework, bilateral oracle, hypothesis chain)
2. Methods (bilateral oracle, Fisher+PCA64, evaluation protocol)
3. L2: Confabulation detection (Fisher essential; T2_L2; CO labeling)
4. L1: Knowledge-source routing (4 architectures; Fisher vs. behavioral baselines)
5. L3: Commitment timing (EXP-B/C/I; MATH entry-point C039; PRM negative C038)
6. Three-task hierarchy, candidate laws, open questions
7. Falsification record
8. Competing theories
9. Pending experiments and kill criteria
10. Related work
11. Conclusion
Appendix A: Claims registry (C001–C039, 39 total)
Appendix B: Experimental record

---

## WHAT IS COMPLETE TODAY (2026-07-10)

**Results complete**: All of §3 (L2), §4 (L1), §5 (L3 except teacher-independence), §6 (except exp_l), §7 (falsification), §8 (competing theories), Appendix A through C039.

**Results pending**: l2_large_n (authoritative L2 AUROC), OOD scope, exp_l (A quantity), phi (5th arch), CO-on-Gemma/Mistral (T2_L2 resolution).

**Writable now**: Introduction (complete), §3 L1 section, §5 all L3 sections, §6 partial (Laws 1/2/4 evidenced), all falsification, related work, conclusion.

---

## WHAT THIS IS NOT

- Not hallucination detection (we measure internal epistemic state, not output correctness)
- Not hidden-state probing (bilateral oracle is the contribution, not Fisher)
- Not interpretability (we do not explain what neurons or circuits do)
- Not an application (no deployed system — measurement framework that enables applications)

---

*Full experimental record: `science/EXPERIMENTS.yaml` | Claims: `science/CLAIMS.yaml` | Paper: `paper/v1_arxiv.md`*
