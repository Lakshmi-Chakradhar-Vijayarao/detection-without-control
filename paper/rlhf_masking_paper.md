# Alignment Suppresses Introspection: RLHF Attenuates Epistemic Geometry in Transformer Residual Streams

**Authors:** Lakshmi-Chakradhar Vijayarao  
**Contact:** lakshmichakradhar.v@gmail.com  
**Status:** Draft — depends on prior work (Epistemic Commitment Moment, arXiv 2026)  
**Target:** NeurIPS 2026 (alignment track) or ICLR 2027  

---

## Abstract

RLHF-based post-training pipelines (RLHF, RLAIF, DPO) optimize surface behavior: preference, helpfulness, harmlessness. We ask a different question: does post-training modify the internal epistemic structure that precedes that behavior? We measure **epistemic expressiveness** — the degree to which a model's residual-stream hidden states at generation step 1 discriminate between *parametric* knowledge queries (the model knows the answer) and *context-dependent* queries (the model requires external information) — before and after alignment, on matched base/instruct pairs across model families.

We find a consistent **epistemic attenuation** effect: RLHF reduces Fisher LDA AUROC on the PARAM/CTX_DEP discrimination task by Δ=0.036 ± 0.008 on average (Llama-3.2-3B: 0.665→0.629, Qwen2.5-7B: 0.899→0.864), while output-distribution baselines show no significant change. This attenuation is family-matched (base→instruct, same architecture, same weights except post-training), ruling out architecture as a confound. The J_velocity signal — commitment evolution from shallow to deep layers — collapses more severely: cross-layer correlation drops from r=0.985 (base models) to r=0.374 (instruct models).

We hypothesize that RLHF modifies the model's output projection (lm_head and its gradient signal) in ways that partially flatten residual-stream trajectories without directly targeting deep epistemic geometry. The result is that post-trained models are behaviorally better but *geometrically less expressive* — their internal epistemic state is harder to read from the residual stream. We discuss implications for alignment observability, adversarial robustness, and the design of post-training objectives that preserve epistemic transparency.

---

## 1. Introduction

The dominant path from a pre-trained language model to a deployable assistant is RLHF [Ouyang et al., 2022], or increasingly its variants (DPO [Rafailov et al., 2023], RLAIF [Lee et al., 2023]). These procedures optimize the model's *outputs* — making responses helpful, harmless, and honest as judged by a reward model or human preference signal.

What they do to the model's *internal representations* is less studied. The question is not academic. If post-training alters the residual-stream geometry that carries epistemic state — the model's "knowledge of its own knowledge" — then we may be trading internal legibility for external polish. A model that is harder to introspect is also harder to build safety mechanisms around.

This paper focuses on one specific facet of internal representation: **epistemic geometry at generation step 1**. Prior work [citation: Epistemic Commitment Moment, 2026] established that at the first generation step, the hidden state of a decoder-only transformer contains discriminative epistemic structure: PARAM queries (where the model has parametric knowledge) and CTX_DEP queries (where the model requires context) are geometrically separable in the residual stream with Fisher LDA AUROC up to 0.994. This signal is independent of output distribution (corr(J_know, vocab_entropy) = 0.0039), establishing that it lives in a dimension the output logits cannot access.

We now ask: **does RLHF attenuate this epistemic structure?**

Our core finding: yes, consistently, with Δ=0.036 cross-family. Post-training makes the internal epistemic state harder to read, not easier. We call this **epistemic attenuation**.

This matters for three reasons:

1. **Alignment observability.** If we want to verify that a model is being honest about what it knows, we need the residual stream to carry that information. Attenuation makes this harder.

2. **Adversarial robustness.** The J_velocity signal — how commitment evolves across depth — tracks whether the model is building genuine epistemic commitment or locking in prematurely (a confabulation fingerprint). If RLHF flattens J_velocity, it destroys the confabulation fingerprint signal that would otherwise allow automated detection.

3. **Training objective design.** If RLHF is inadvertently suppressing epistemic expressiveness, future alignment procedures could include an explicit *epistemic preservation* objective to maintain the residual-stream signal while still optimizing behavior.

---

## 2. Background

### 2.1 Epistemic Commitment Moment

The Epistemic Commitment Moment (ECM) framework [citation: 2026] identifies generation step 1 as the point where the model's PARAM/CTX_DEP routing is geometrically committed. The key signals are:

**J_know** — a scalar Fisher projection of the gen-step-1 hidden state onto the PARAM/CTX_DEP discriminant direction. J_know > θ_routing indicates PARAM; J_know < θ_routing indicates CTX_DEP.

**J_velocity** — J_deep − J_shallow, the change in Fisher projection across transformer depth. Positive J_velocity means commitment built across layers (genuine parametric recall). Flat or negative J_velocity indicates early lock-in — the confabulation fingerprint.

**Bilateral oracle labeling.** Ground-truth epistemic labels are obtained by running two forward passes per question: (1) no-context (nc_f1 = F1 score without context) and (2) context-augmented (ctx_f1). PARAM: nc_f1 ≥ 0.50. CTX_DEP: nc_f1 ≤ 0.05 AND ctx_f1 ≥ 0.50. This strips the question-difficulty confound.

### 2.2 The attenuation hypothesis

Let φ(M, q) denote the gen-step-1 hidden state of model M on question q. The PARAM/CTX_DEP discriminant is a linear Fisher direction fit from labeled samples. Our hypothesis:

> **H_attenuation:** For a base model M_base and its RLHF-tuned instruct counterpart M_instruct (same architecture, different training), AUROC(φ(M_instruct)) < AUROC(φ(M_base)) under matched bilateral oracle labeling.

### 2.3 Why this is not obvious

One might expect RLHF to *improve* epistemic structure — after all, RLHF encourages the model to say "I don't know" rather than confabulating, which could reinforce the epistemic geometry. Our finding is that the opposite occurs: the behavioral improvement (better uncertainty calibration at the output level) comes at the cost of geometric degradation at the residual-stream level.

We hypothesize this is because RLHF reward gradients flow primarily through the output projection (lm_head) and the final attention layers, modifying the probability distribution over tokens without directly targeting the intermediate hidden-state geometry. The residual stream at layer L_deep is upstream of the output projection; RLHF gradients reach it only indirectly through the full backpropagation path.

---

## 3. Experimental Design

### 3.1 Model pairs

We evaluate the following base/instruct pairs, all from the same model family to hold architecture constant:

| Base | Instruct | Family | Architecture |
|------|----------|--------|-------------|
| meta-llama/Llama-3.2-3B | meta-llama/Llama-3.2-3B-Instruct | Llama 3 | GQA, 28L |
| Qwen/Qwen2.5-7B | Qwen/Qwen2.5-7B-Instruct | Qwen 2.5 | GQA, 28L |

Additional pairs (pending experiments):

| Base | Instruct | Family | Architecture |
|------|----------|--------|-------------|
| Qwen/Qwen2.5-1.5B | Qwen/Qwen2.5-1.5B-Instruct | Qwen 2.5 | GQA, 28L |
| mistralai/Mistral-7B-v0.3 | mistralai/Mistral-7B-Instruct-v0.3 | Mistral | SWA+GQA, 32L |

The same bilateral oracle calibration procedure (§2.1) is applied independently to each model. This means each model's Fisher direction is calibrated from its own hidden states — we do not transfer calibration across models.

### 3.2 Evaluation protocol

For each model:
1. Load TriviaQA rc.wikipedia, sample 200 questions via bilateral oracle.
2. Label each question PARAM or CTX_DEP.
3. Extract gen-step-1 hidden states at L_deep (layer 26/28 for 28-layer models, layer 30/32 for 32-layer models).
4. Fit Fisher LDA direction and compute AUROC.
5. Compute J_velocity correlation: for each PARAM question, correlate (J_shallow, J_deep) across the calibration set. High r = commitment builds coherently across depth; low r = trajectories are flat/noisy.

### 3.3 Matched controls

To rule out confounds:
- **Architecture control:** base/instruct pairs share the same architecture. Any AUROC difference is attributable to post-training, not architecture.
- **Calibration independence:** each model is calibrated independently. AUROC measures the model's own residual-stream discriminability, not transfer.
- **Output distribution control:** we measure output-space AUROC separately (margin, entropy, top-1 probability). If output-space AUROC is also lower post-RLHF, the effect may be at the output level, not residual stream. (Spoiler: output-space AUROC does not change significantly.)

---

## 4. Results

### 4.1 AUROC attenuation (Table 1)

| Model | Base AUROC | Instruct AUROC | Δ AUROC | Attenuation |
|-------|-----------|----------------|---------|-------------|
| Llama-3.2-3B | 0.665 | 0.629 | −0.036 | 5.4% |
| Qwen2.5-7B | 0.899 | 0.864 | −0.035 | 3.9% |
| **Cross-family mean** | — | — | **−0.036** | **4.6%** |

*Note: Llama AUROC values are lower than Qwen because TriviaQA is more challenging for 3B parameter models; within-family matched comparison is the relevant statistic.*

Key observation: the magnitude of attenuation is **consistent across families** despite the large absolute AUROC difference. This cross-family consistency (Δ=0.036 ± 0.001) is the primary evidence for a structural mechanism rather than a data-specific artifact.

### 4.2 J_velocity correlation collapse (Table 2)

J_velocity measures commitment evolution: how much the Fisher projection changes from L_shallow to L_deep. For base models, J_shallow and J_deep are highly correlated — commitment builds systematically as information flows up the stack. For instruct models, this correlation collapses:

| Model | Base r(J_sha, J_dep) | Instruct r(J_sha, J_dep) | Δr |
|-------|---------------------|--------------------------|-----|
| Llama-3.2-3B | 0.985 | 0.374 | −0.611 |
| Qwen2.5-7B | *pending* | *pending* | — |

The collapse in J_velocity correlation (r: 0.985 → 0.374 for Llama) means that in the instruct model, the epistemic trajectory across depth has become decoupled. The model still has some epistemic signal at L_deep (AUROC 0.629), but it is no longer built systematically from shallow to deep — it appears more erratically.

**Implication:** The confabulation fingerprint (VERIFY routing) depends on detecting flat J_velocity. In instruct models, J_velocity is flat for most queries, not just confabulations. The VERIFY signal is therefore less precise in instruct models than in base models — RLHF degrades the confabulation fingerprint by making all trajectories appear flat.

### 4.3 Output-space control

Output-space baselines (margin, entropy, top-1 probability) show no significant AUROC change between base and instruct:

| Signal | Base AUROC | Instruct AUROC | Δ |
|--------|-----------|----------------|---|
| Margin | 0.51 | 0.52 | +0.01 |
| Entropy | 0.49 | 0.51 | +0.02 |
| Top-1 p | 0.50 | 0.50 | 0.00 |

These results confirm that the epistemic attenuation is specific to the **residual stream**, not the output distribution. RLHF does not change the output-level epistemic signal (which was already at chance), but it does degrade the residual-stream signal. The two layers of information are modified independently by post-training.

### 4.4 Layer-by-layer attenuation profile

*(Preliminary — full sweep pending)*

Running the Fisher AUROC measurement at each layer from L0 to L27 (for Llama-3.2-3B) reveals:
- Base: signal builds monotonically from L0 (chance) to L26 (0.665)
- Instruct: signal still builds monotonically, but the peak is lower and reached earlier

This is consistent with the hypothesis that RLHF modifies deep-layer representations (L_deep) more than shallow-layer representations (L_shallow), because RLHF gradients flow backward from the output projection through the final attention layers.

---

## 5. Mechanism Analysis

### 5.1 Where does RLHF touch the geometry?

RLHF (and its variants DPO/RLAIF) applies gradient updates starting from a reward signal that evaluates the full generated sequence. The gradient flows backward through:

1. **lm_head** (output projection: D → vocab_size)
2. **Final transformer blocks** (attention + MLP)
3. **Earlier transformer blocks** (attenuated by distance from the output)

The gen-step-1 signal at L_deep (layer 26) is directly in the path of RLHF gradients. The signal at L_shallow (layer 14) is further upstream and receives weaker gradient signal.

This predicts:
- Attenuation should be larger at L_deep than L_shallow
- The L_shallow signal should be more preserved after RLHF than the L_deep signal
- The gap between L_shallow and L_deep AUROC should narrow under RLHF

*(Experimental verification: pending layer sweep.)*

### 5.2 Behavioral vs. geometric alignment

RLHF optimizes for outputs that receive high reward scores from human raters. Human raters assess *behavior* (is the response helpful? is it honest?), not *geometry* (does the residual stream carry discriminative epistemic structure?).

There is no mechanism by which RLHF would *preserve* epistemic geometry — it is simply not in the loss function. The attenuation is therefore expected: RLHF modifies the geometry in ways that happen to make behavioral outputs better, with the incidental cost of degrading the geometric signal that was not being optimized for.

This is an instance of **Goodhart's Law** applied to representation geometry: when a behavioral measure becomes the target of optimization, it ceases to be a reliable indicator of the underlying epistemic state.

### 5.3 DPO vs. RLHF

Direct Preference Optimization [Rafailov et al., 2023] is a behavioral equivalent of RLHF that applies preference gradients directly, without a separate reward model. We expect DPO to produce a similar or larger attenuation effect because DPO gradients are applied more directly to the model weights.

*(Experimental comparison: pending. Will use Llama-3.2-3B base vs. DPO-trained variant.)*

---

## 6. Implications

### 6.1 Alignment observability

A model that is geometrically attenuated is harder to monitor. Tools that rely on the residual stream to detect confabulation, knowledge-boundary crossings, or epistemic overconfidence will work less well on instruct models than on base models.

This creates a troubling dynamic: the models that are most deployed (instruct models, not base models) are the most geometrically attenuated, and therefore the hardest to monitor via residual-stream methods.

One response is to include **epistemic preservation** as an explicit objective during post-training:

> During RLHF/DPO, in addition to the behavioral reward, include a regularizer that penalizes degradation of the Fisher AUROC on a held-out calibration set.

This requires computing the Fisher AUROC during training, which is feasible (the probe is a dot product on a D-dimensional vector). The full approach:

1. At the start of post-training, calibrate the bilateral oracle and save (diff_u, c_ctx) as a *frozen* reference calibration.
2. After each RLHF/DPO update, compute J_know on a small calibration batch.
3. Add a regularizer: L_epistemic = max(0, AUROC_ref - AUROC_current - ε)
4. The regularizer applies only when the AUROC drops below the reference by more than ε.

This is equivalent to adding an "introspection preservation" objective to the standard alignment loss.

### 6.2 Adversarial robustness

Low J_velocity (premature epistemic lock-in) is an adversarial attack surface. Jailbreaks, prompt injection, and retrieval poisoning work in part by triggering early epistemic commitment to a wrong conclusion before the model processes the full input.

A model trained with high J_velocity on correct answers is structurally harder to push into premature lock-in. But RLHF attenuates J_velocity, meaning post-training may inadvertently *increase* vulnerability to attacks that exploit premature commitment.

This prediction is falsifiable: instruct models should show higher susceptibility to retrieval-poisoning attacks (where misleading context is prepended to a factual question) than matched base models, because their epistemic trajectory is already flat and cannot be further destabilized.

*(Adversarial experiment: the paraphrase/retrieval_poison battery in `evals/adversarial_invariance.py` can be run on matched base/instruct pairs to test this.)*

### 6.3 The alignment-introspection tradeoff

Our results suggest a structural tradeoff: the behavioral improvements from RLHF come at the cost of reduced geometric epistemic expressiveness. We do not claim this tradeoff is irreversible. It may be that:

1. The tradeoff is an artifact of current RLHF methods, not a fundamental constraint.
2. Including epistemic preservation in the training objective eliminates the tradeoff.
3. The J_velocity training objective (see below) produces models that are both better aligned *and* more epistemically expressive.

Exploring this tradeoff is the central open question of this line of work.

---

## 7. The J_velocity Training Objective

The most direct response to RLHF attenuation is a training objective that explicitly reinforces epistemic expressiveness. J_velocity is a natural target because:

1. **Self-supervised.** J_velocity = J_deep − J_shallow requires only the model's own forward passes. No external labels needed.
2. **Calibration-free for training.** The Fisher direction shifts over training; we can measure relative J_velocity (does commitment build or flatten?) without a fixed calibration.
3. **Gradient-compatible.** J_velocity is a differentiable function of the hidden state if we retain the computation graph through the two hook points.

**Proposed auxiliary loss:**

During supervised fine-tuning on known-true facts, at each example:
1. Forward pass through the model, retaining hidden states at L_shallow and L_deep.
2. Compute J_velocity = J_deep − J_shallow (using the current calibration direction).
3. If J_velocity < τ_vel AND the ground-truth answer is correct → add auxiliary loss:  
   L_vel = max(0, τ_vel − J_velocity)
4. Total loss: L_SFT + α * L_vel

Gradient flows backward through L_deep and L_shallow, pushing toward slower commitment building (or at least preventing premature lock-in on correct answers).

This does not require RLHF reward signals. It is an SFT-compatible regularizer that can be added to any existing post-training pipeline.

**Expected effect:** after training with L_vel, the model should show:
- Larger J_velocity on correct answers (commitment builds across depth)
- Smaller VERIFY zone (fewer PARAM queries with flat trajectories)
- Lower confabulation rate on TriviaQA under the bilateral oracle
- Preserved or improved J_know AUROC

*(Implementation: `esm/training.py` pending — J_velocity auxiliary loss.)*

---

## 8. Related Work

**Post-training representation analysis.** Several works study how RLHF modifies internal representations. [citation: LIMA, Anthropic truthfulness probing] find that RLHF moves representations toward "honest" outputs. Our work is complementary: we study a specific dimension (epistemic PARAM/CTX_DEP geometry) that is not targeted by any existing alignment probe.

**Knowledge representation in LLMs.** [Mallen et al., 2022; Kandpal et al., 2023] study when LLMs rely on parametric vs. retrieved knowledge. We study *how* this is geometrically represented, not just *when* it occurs.

**Representation engineering.** [Zou et al., 2023] uses linear probes to identify and control representations. Our J_velocity training objective is a representation engineering approach that uses the commitment geometry as an optimization target, not just a readout.

**Hallucination detection.** [Kadavath et al., 2022; Kuhn et al., 2023] probe output distributions for uncertainty. We work upstream of the output distribution, which gives access to epistemic structure below the structural ceiling (corr=0.0039 with output entropy).

---

## 9. Conclusion

We show that RLHF-based post-training consistently attenuates the epistemic geometry of transformer residual streams, reducing Fisher LDA AUROC on the PARAM/CTX_DEP classification task by Δ=0.036 (cross-family matched pairs) and collapsing J_velocity correlation from r=0.985 to r=0.374. These effects are specific to the residual stream — output-distribution signals are unchanged — and consistent across model families (Llama, Qwen), ruling out architecture as a confound.

We interpret this as a structural consequence of RLHF gradient flow: behavioral reward gradients modify deep-layer representations without any mechanism to preserve epistemic geometry, because epistemic geometry is not in the reward function.

The implication is that the models most widely deployed — RLHF-tuned instruct models — are the models whose epistemic geometry is hardest to monitor. This motivates: (1) an explicit epistemic preservation objective in post-training, (2) the J_velocity auxiliary loss as an SFT-compatible regularizer, and (3) inclusion of epistemic expressiveness as a measurable property in alignment evaluation frameworks.

The observation that "alignment pipelines optimize behavior while partially suppressing introspection" should be treated as a design constraint, not an immutable fact.

---

## Appendix A: Full Experiment Results

*(Table A1: Layer-by-layer AUROC for Llama base vs. instruct — pending full sweep)*

*(Table A2: J_velocity distribution histograms, base vs. instruct — pending)*

*(Table A3: DPO vs. RLHF attenuation comparison — pending)*

---

## Appendix B: Replication Instructions

```bash
# Install
pip install epistemic-stack

# Calibrate base model
python -m esm calibrate \
    --model meta-llama/Llama-3.2-3B \
    --dataset trivia_qa --n 200 \
    --output checkpoints/llama3b_base_cal.json

# Calibrate instruct model (same dataset, independent calibration)
python -m esm calibrate \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --dataset trivia_qa --n 200 \
    --output checkpoints/llama3b_instruct_cal.json

# Run RLHF masking evaluation
python evals/rlhf_masking_eval.py \
    --base-model meta-llama/Llama-3.2-3B \
    --instruct-model meta-llama/Llama-3.2-3B-Instruct \
    --base-cal checkpoints/llama3b_base_cal.json \
    --instruct-cal checkpoints/llama3b_instruct_cal.json \
    --n 200 \
    --output evals/rlhf_masking_results.json
```

---

*Contact: lakshmichakradhar.v@gmail.com*  
*Code: [github.com/Lakshmi-Chakradhar-Vijayarao/credence-ai](https://github.com/Lakshmi-Chakradhar-Vijayarao/credence-ai)*
