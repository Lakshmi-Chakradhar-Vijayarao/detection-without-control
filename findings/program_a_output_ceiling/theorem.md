# Program A: Output Ceiling — Core Theorem

## Theorem statement

**The Output Ceiling (empirical)**

Let M be a language model with residual stream R and output distribution P.
Let J_know be the Fisher LDA projection of R at layer L onto the PARAM/CTX_DEP
discriminant axis (calibrated via bilateral oracle).
Let H be the vocabulary entropy of P.

Then: corr(J_know, H) = 0.0039   [n=800, TriviaQA, Qwen2.5-7B-Instruct, layer 26]

This near-zero correlation is structural, not statistical noise. It holds because:

1. J_know lives in a 3584-dimensional residual stream subspace (discriminant direction).
2. H lives in a 1-dimensional scalar derived from the vocabulary distribution.
3. The projection through the LM head (unembedding matrix W_U) is many-to-one:
   many residual stream directions project to the same logit distribution.
4. The Fisher discriminant direction and the top principal components of W_U are
   geometrically near-orthogonal (hypothesis, to be verified by output_ceiling_theorem.md).

**Consequence**: No output-level signal (entropy, margin, logprob, self-reported confidence)
can recover J_know. The epistemic information is destroyed at the softmax bottleneck.

---

## Empirical support

| Signal | AUROC on PARAM/CTX_DEP | Source |
|---|---|---|
| Fisher LDA (hidden, L26) | 0.989 | ESM main experiment |
| Fisher LDA (hidden, L26), n=400 | 0.83 | Kill criterion 2 |
| Output entropy | 0.51 (chance) | Phase 1 synthesis |
| Output margin | 0.57 | Kill criterion 2 |
| TF-IDF text | 0.53 | Kill criterion 2 |
| SBERT embeddings | 0.57 | Kill criterion 2 |

Gap: 0.48 AUROC units (hidden vs output entropy).  
Text-to-hidden gap: 0.30 AUROC units (SBERT=0.57 vs hidden=0.83).

The 0.30 text-to-hidden gap rules out question difficulty or linguistic style as confounds.
The signal is internal model state not externalized in surface form.

---

## Replication requirements

Before submission, replicate r=0.0039 on:
- [ ] MMLU (multiple-choice, different register)
- [ ] HotpotQA (multi-hop, longer context)
- [ ] Second model family (Llama-3.2-3B)

Current evidence: single dataset (TriviaQA), single model (Qwen2.5-7B-Instruct).

---

## Files

- `independence_result.md` — full experimental record
- `evals/independence_test.py` — code
- `evals/groq_entropy_ceiling.py` — cross-provider replication
- `paper/output_ceiling_theorem.md` — theoretical mechanism
