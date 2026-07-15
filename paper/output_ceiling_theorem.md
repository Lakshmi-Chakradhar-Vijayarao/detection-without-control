# Output Compression Theorem

**Formal statement and empirical validation**

---

## 1. Setup

Let $M$ be an autoregressive language model with residual stream $h \in \mathbb{R}^d$ at the final layer before the unembedding head. The output distribution is produced by an affine-then-softmax map:

$$p = \text{softmax}(W_U \cdot h + b_U), \quad W_U \in \mathbb{R}^{|V| \times d}, \quad p \in \Delta^{|V|}$$

where $\Delta^{|V|}$ is the $|V|$-simplex over the vocabulary. Let $c \in \{0, 1\}$ be a binary epistemic label assigned to the generation context — for example, $c = 1$ if the answer is drawn from parametric memory (PARAM) and $c = 0$ if the answer depends on context provided in the prompt (CTX_DEP). Let $g: \Delta^{|V|} \to \mathbb{R}$ be any measurable function of the output distribution — entropy, top-1 margin, self-consistency score, a trained verifier, or any composite thereof.

---

## 2. Theorem: Output Compression Inequality

**Theorem.** For any binary epistemic label $c$ and any function $g: \Delta^{|V|} \to \mathbb{R}$:

$$I(h;\, c) \;\geq\; I(p;\, c) \;\geq\; I(g(p);\, c)$$

where $I(\,\cdot\,;\,\cdot\,)$ denotes mutual information.

**Proof sketch.** The three quantities form a Markov chain:

$$c \;\longleftrightarrow\; h \;\xrightarrow{W_U,\,\text{softmax}}\; p \;\xrightarrow{g}\; g(p)$$

Each arrow is a deterministic function applied to the preceding variable. By the **data processing inequality** (DPI): for any deterministic function $f$, $I(X; Y) \geq I(f(X); Y)$. Since $p = \text{softmax}(W_U h + b_U)$ is a deterministic function of $h$, we have $I(h; c) \geq I(p; c)$. Since $g(p)$ is a deterministic function of $p$, we have $I(p; c) \geq I(g(p); c)$. The chain of inequalities follows. $\blacksquare$

Note: the inequalities are equalities if and only if the respective maps are sufficient statistics for $c$. In general, $W_U$ discards dimensions of $h$ that are orthogonal to the vocabulary directions it has learned — including directions that encode epistemic state but are not needed for next-token prediction.

---

## 3. Corollary: AUROC Ceiling

**Corollary.** For any function $g: \Delta^{|V|} \to \mathbb{R}$ and any linear probe $w \in \mathbb{R}^d$:

$$\text{AUROC}(w^\top h,\; c) \;\geq\; \text{AUROC}(g(p),\; c)$$

where the left-hand side is maximized over all $w$ (i.e., taken at the optimal Fisher LDA direction).

**Proof.** AUROC is a monotone functional of mutual information under the Gaussian linear discriminant model (Hanley-McNeil; also follows directly from the Neyman-Pearson lemma). Since $I(h; c) \geq I(g(p); c)$, the best achievable AUROC from $h$ upper-bounds the best achievable AUROC from any function of $p$. The optimal linear probe on $h$ realizes the mutual information available in the linear projection direction. $\blacksquare$

This corollary establishes a **structural AUROC ceiling** for all output-monitoring systems: no matter how sophisticated the function $g$, its discriminative power for epistemic label $c$ cannot exceed that of the optimal probe on the residual stream.

---

## 4. Empirical Magnitude

The gap between the theoretical ceiling and the best achievable output-space monitor was measured on **Qwen2.5-7B-Instruct**, $n = 800$ generation episodes drawn from TriviaQA, with bilateral oracle labels (PARAM / CTX_DEP).

| Signal family | Method | AUROC |
|---|---|---|
| Residual stream $h$ (ceiling) | Fisher LDA on $h$ at layer 26 | **0.989** |
| Output distribution $p$ | Entropy $H(p)$ | 0.51 |
| Output distribution $p$ | Top-1 margin $p_1 - p_2$ | ~0.51 |
| Output distribution $p$ | Self-consistency (sample majority) | ~0.51 |
| **Empirical gap** | | **0.479 AUROC units** |

The near-orthogonality of the two signal families is confirmed by:

$$\text{corr}(\text{Fisher projection on } h,\; H(p)) = 0.0039$$

A correlation of $0.0039$ across $n = 800$ samples (well below any threshold for linear dependence) indicates that the Fisher direction and output entropy are measuring almost entirely different properties of the generation. This is not a calibration failure — it is a structural consequence of the compression.

---

## 5. Why the Gap Is Not Trivially Small

The output projection $W_U$ is trained by cross-entropy loss over next-token predictions. Its optimization objective is:

$$\mathcal{L}_{\text{CE}} = -\mathbb{E}_{(x, y)}\left[\log p(y \mid x)\right]$$

This objective is entirely agnostic to the epistemic label $c$. There is no gradient signal that would cause $W_U$ to preserve $I(h; c)$ across the projection. Specifically:

- $W_U$ projects from $\mathbb{R}^d$ to $\mathbb{R}^{|V|}$ and then normalizes. Any component of $h$ that lies in the null space of $W_U$ — or in a direction that is washed out by the softmax temperature — is discarded.
- The residual stream $h$ at the final layer encodes the full computation history of the forward pass: attention patterns, key-value interactions, positional information, and intermediate representations of retrieved vs. parametric facts. The output $p$ encodes only the marginal next-token distribution.
- Empirically, the Fisher LDA direction (the linear combination of $h$ that best separates PARAM from CTX_DEP) has correlation $0.0039$ with entropy — confirming it lives in a subspace of $\mathbb{R}^d$ that $W_U$ does not amplify.

The gap is large (0.479 AUROC units) because epistemic state is encoded in the geometry of $h$, not in the shape of $p$.

---

## 6. Implication for AI Reliability Systems

The theorem applies uniformly to every output-space reliability method in current use:

- **Entropy thresholding** — a function of $p$, bounded by the ceiling
- **Self-consistency** — a function of multiple draws from $p$, still bounded (each draw is a function of $p$)
- **Verbalized confidence** — the model's own output token sequence is itself a function of $p$; bounded
- **Chain-of-thought heuristics** — the CoT token sequence is an output; bounded
- **Verifier / reward models** — trained on outputs; bounded by the output-space ceiling unless they access hidden states directly

This is a **structural** bound, not a performance gap that better calibration, more training data, or larger models can close while using only outputs. The only way to exceed the ceiling is to monitor $h$ directly — i.e., to build infrastructure that exposes the residual stream at inference time.

---

## 7. Open Question: Trainable Transparency

The theorem establishes the ceiling but does not fix its height relative to the residual stream. In principle, a model could be trained so that $I(p; c)$ is substantially larger — i.e., so that epistemic state is deliberately encoded in the output distribution rather than hidden in the residual stream geometry.

**Open question:** Under what training conditions does $I(p; c) \to I(h; c)$? Specifically:
- Does adding an epistemic auxiliary loss during SFT (targeting $c$ from $p$) push the Fisher direction into the range of $W_U$?
- Does instruction tuning (RLHF) systematically reduce $I(h; c)$, and if so, can an auxiliary loss counteract this attenuation?
- Is there a training-time intervention that makes the output distribution a sufficient statistic for $c$ — collapsing the gap to zero?

Answering this question would determine whether epistemic transparency is a permanent architectural feature of residual-stream monitoring, or whether it can be shifted into the output layer through training — making lightweight output-space tools eventually competitive with hidden-state probes.

---

*Empirical results from the EIL/MSCP experiment series (2026). Fisher LDA results: layer sweep on Qwen2.5-7B-Instruct, n=800, TriviaQA, bilateral oracle labels. See `evals/` for raw data.*
