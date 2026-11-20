# Erratum (20 Nov 2026)

These statements in `paper/main.tex` are **withdrawn**. The README's status note (1 Nov 2026) already says so; this file states it next to the paper.

1. **The commit-point causal validation (C028, C037; EXP-I), "90% token savings, a causal result".**
   - **Why:** the commit probe was trained to separate early chain steps from late ones, so it tracks position in the reasoning chain, not commitment to an answer.
   - **Withdrawn:** every claim built on it, including "+0.012 F1 at 90% token savings", "C028 confirmed" and "the paper's most directly deployable result".
2. **The Fisher trajectory AUROC 0.9947** (`science/CLAIMS.yaml`, the "28 J-scores as sequence features" entry). It has no supporting script and is not supported.
3. **CAMS eviction, presented as validated** (`docs/KNORM_BOUND.md`). An audit (7 Nov 2026) found that KV eviction never took effect in the experiment code, so these results are withdrawn.

Until the paper is revised, read it with these sections struck out.
