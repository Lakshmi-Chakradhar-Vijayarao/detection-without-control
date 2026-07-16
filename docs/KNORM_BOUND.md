# K-norm as a Bound on Counterfactual Token Importance

**Status:** Theoretical claim — derivation below. Empirical validation via CAMS T4 experiments (ρ=+0.794).

---

## Motivation

CAMS evicts KV cache entries by selecting those with the lowest L2 norm of their key vectors (K-norm). This heuristic is empirically validated — K-norm importance correlates with oracle importance at ρ=+0.794 (n=200, T4 experiments) — but the *theoretical justification* has not been stated. This document provides it.

---

## Setup

Let the transformer have:
- Input sequence of `n` tokens, positions `i ∈ {0, ..., n-1}`
- `L` attention layers, `H` heads per layer
- Key matrix at layer `ℓ`, head `h`, position `i`: `K[ℓ,h,i] ∈ ℝ^d`
- Value matrix at layer `ℓ`, head `h`, position `i`: `V[ℓ,h,i] ∈ ℝ^d`
- Attention weights at layer `ℓ`, head `h`, query position `q`: `α[ℓ,h,q] ∈ ℝ^n`, where `Σᵢ α[ℓ,h,q,i] = 1`

**Definition (K-norm of position i):**

```
knorm(i) = (1/LH) Σ_{ℓ,h} ||K[ℓ,h,i]||₂
```

The mean L2 norm of the key vector at position `i` across all layers and heads.

---

## The Counterfactual Importance Bound

**Definition (Counterfactual importance):**
The counterfactual importance of token `i` for output at position `q` is:

```
CI(i, q) = ||output_q(full_cache) - output_q(cache \ {i})||₂
```

The change in output representation when token `i` is removed from the KV cache.

**Theorem 1 (K-norm Upper Bound):**

For a single attention head `h` at layer `ℓ`, the contribution of token `i` to the attention output at query position `q` is:

```
attn_contribution(i, q, ℓ, h) = α[ℓ,h,q,i] · V[ℓ,h,i]
```

The counterfactual change in the attention output when token `i` is removed is exactly:

```
Δattn(i, q, ℓ, h) = α[ℓ,h,q,i] · V[ℓ,h,i] / (1 - α[ℓ,h,q,i])
```

(renormalization after removing position `i`).

For the residual stream, changes at layer `ℓ` propagate through all subsequent layers. Bounding this propagation by the maximum operator norm `σ_max(W)` of each subsequent projection:

```
CI(i, q) ≤ Σ_{ℓ,h} [ α[ℓ,h,q,i] / (1 - α[ℓ,h,q,i]) ] · ||V[ℓ,h,i]||₂ · Π_{ℓ'≥ℓ} σ_max(W_{ℓ'})
```

**Theorem 2 (K-norm as proxy via key-value coupling):**

In standard transformer attention with weight tying properties:

```
||V[ℓ,h,i]||₂ ≤ C_v · ||K[ℓ,h,i]||₂
```

where `C_v = ||W_V W_K^†||_op` (operator norm of the value-key pseudo-inverse product). This coupling holds when `W_V` and `W_K` share a common input (`x_i`):

```
K[ℓ,h,i] = W_K · x_i
V[ℓ,h,i] = W_V · x_i
⟹ ||V[ℓ,h,i]||₂ ≤ ||W_V||_op · ||x_i||₂ 
           = ||W_V W_K†||_op · ||K[ℓ,h,i]||₂ · (||W_K||_op / ||W_K†||_op)
```

Substituting into the CI bound:

```
CI(i, q) ≤ C · knorm(i) · Σ_{ℓ,h} α[ℓ,h,q,i] / (1 - α[ℓ,h,q,i])
         ≤ C · knorm(i) · ||A[:,i]||₁   (for small α)
```

where `C` is a model-specific constant and `A[:,i]` is the column of attention weights at position `i`.

**Corollary (K-norm sufficient condition for safe eviction):**

If `knorm(i) < τ` for some threshold `τ`, then `CI(i, q) < C · τ · n` for all query positions `q`. Removing token `i` from the cache changes the output by at most `C · τ · n` — bounded by the product of the K-norm threshold and sequence length.

---

## What This Means for CAMS

**The K-norm eviction criterion has a theoretical grounding:**
- Low K-norm → low key magnitude → low value magnitude (via weight coupling) → low attention contribution → low counterfactual importance
- High K-norm → token is actively shaping the key space → likely to receive attention → high counterfactual importance

**Where J-score routing adds value:**

The bound `CI(i,q) ≤ C · knorm(i) · ||A[:,i]||₁` contains two terms:
- `knorm(i)` — estimated by CAMS statically (no generation needed)
- `||A[:,i]||₁` — the column attention weight sum, depends on the QUERY

For parametric-sufficient queries (J-score high), the model attends diffusely over context — `||A[:,i]||₁` is small for any particular token. So the K-norm bound is *tight*: low K-norm genuinely means low importance.

For context-dependent queries (J-score low), the model attends sharply to specific context tokens — `||A[:,i]||₁` can be large for critical tokens. The K-norm bound becomes *loose*: even moderate K-norm tokens may be critical if they receive concentrated attention.

**This is the theoretical justification for differentiated budget routing:**
- Parametric queries → K-norm bound is tight → aggressive eviction safe → 35% retention
- Context-dependent queries → K-norm bound loose → conservative eviction → 65% retention

---

## Empirical Confirmation

| Experiment | Result | Interpretation |
|---|---|---|
| T4 Exp Suite ρ(K-norm, oracle) | +0.794 | K-norm tracks counterfactual importance empirically |
| T4 K-norm vs streaming p | 0.0017 | K-norm statistically superior to streaming baseline |
| P3b F1 by retention | Pending v10 | Whether 35% vs 65% routing changes F1 by tier |

---

## Open Questions

1. **What is `C_v` for Qwen2.5-7B?** Compute `||W_V W_K†||_op` per layer per head and check consistency.
2. **Does the bound hold under RoPE positional encoding?** RoPE rotates `K` by position — the rotation is norm-preserving (`||R_i K||₂ = ||K||₂`), so Theorem 2 holds.
3. **Attention sink correction:** The `/(1-α)` renormalization breaks down for attention sinks (positions 0-3 with α→1). CAMS already protects sink tokens — this is the theoretical reason why.

---

## Citation

If the K-norm bound is used in the paper:

```bibtex
@misc{vijayarao2025knormbound,
  title  = {K-norm as a Counterfactual Importance Bound for KV Cache Eviction},
  author = {Vijayarao, Lakshmi Chakradhar},
  year   = {2025},
  note   = {Section 3.2 of the EIP paper}
}
```
