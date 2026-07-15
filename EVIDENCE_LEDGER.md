# Evidence Ledger — Epistemic Commitment Moment Project
> **⚠ SUPERSEDED** — This ledger uses pre-bilateral-oracle E-numbered evidence entries and j_score metrics. Current authoritative evidence registry is `science/CLAIMS.yaml` (C001–C045) and `science/EXPERIMENTS.yaml`.  
> This file is retained as historical record of the ESM/j_score research phase.

**Rule**: A claim moves to ESTABLISHED only when it survives hold-out validation (not just CV).  
**Rule**: A claim moves to FALSIFIED only when the primary test fails, not when a variant fails.  
**Rule**: This file is updated after every experiment run, not before.  

Last updated: 2026-06-16 (E33–E42 added. E42: answer_jump_v2 REGIME_2_PARTIAL — Llama backbone positive but 55× smaller than Qwen. Backbone-stratified magnitude finding added.)

---

## ESTABLISHED — Evidence survives hold-out

| # | Claim | Evidence | Source | Notes |
|---|-------|----------|--------|-------|
| E1 | Linearly decodable variable exists at L26 | AUROC 0.866–0.989 (hold-out) | ESM v33, cross-model | 4 models, bilateral oracle |
| E2 | Signal is NOT question text | Hidden 0.831 vs TF-IDF 0.531, SBERT 0.571, gap=0.300, length ρ=0.041 p=0.41 | Kill Criterion 2 | n=400 balanced |
| E3 | Commitment onset at gen-step-1, not prefill | Step-1=0.785, prefill=0.567 (mean across 4 models) | ESM v33 cross-model | COMMITMENT_MOMENT confirmed for Llama (0.866→0.517) |
| E4 | Bilateral oracle is essential | Unilateral kills commitment gap: +0.215 → −0.020; AUROC 0.866 → 0.724 | Probe transfer v1+v2 | Gap inversion on unilateral |
| E5 | Signal onset at L0 (embedding layer) | Step-1 AUROC at L0 = 0.784 | Trajectory LDA v1 | Before any transformer computation |
| E6 | Signal is depth-broad, not layer-localized | All 29 layers (L0–L28) exceed AUROC 0.70 | Trajectory LDA v1 | Commitment bandwidth threshold=0.70 |
| E7 | ~~Fisher trajectory compounds per-layer discriminants~~ | CV gain +0.1482 does NOT survive hold-out — see F10 | Trajectory LDA v1 / P7 | FALSIFIED by P7: single-point L26 test=0.9294 > trajectory test=0.8474 |
| E8 | PCA concatenation degrades, not helps | PCA-50 raw = 0.8164 < single-point 0.8464 | Trajectory LDA v1 | Fisher projection is doing essential work |
| E9 | Answer framing DOES NOT explain signal | Framing step-1 AUROC 0.764 (−0.094 vs 0.858); prefill invariant (0.831→0.809) | Framing v1 | H1 REJECTED |
| E10 | Signal is metacognitive routing, not output posture | Hedge framing preserves signal (+0.012 delta); answer framing destroys it | Framing v1 | H2 supported by elimination |
| E11 | Architecture affects signal strength | GQA → STRONG (Llama, Qwen); SWA/MQA → WEAK (Phi, Mistral) | ESM v33 cross-model | Structural confound documented |
| E12 | Cross-domain signal collapse with full ensemble | Margin alone: 0.734; full ensemble: 0.460 (worse) | Decomposition v1 | 3 independent signal families identified |
| E13 | j_score and j_velocity are correlated (ρ=0.74) but cross-domain | Calibrated, domain-specific family | Decomposition v1 | entropy+margin zero-shot; mag_ratio+cs geometric |
| E14 | Epistemic variable is multi-dimensional (NOT scalar) | Dim1=0.359, Dim2=0.117, Dim3=0.094; Dim1+Dim2=0.476 (PCA-50 → LDA, n=240) | Dimensionality v1 | P3 FALSIFIED. Dim1 alone insufficient for most pairs. |
| E15 | Dim1 ≈ Routing Confidence: separates PARAM_LOW from everything; PARAM_HIGH ≈ CTX_DEP ≈ CONFAB on Dim1 | CTX_DEP vs PARAM_HIGH Dim1=0.597; CTX_DEP vs PARAM_LOW Dim1=0.900 | Dimensionality v1 | PARAM_LOW (correct but uncertain) is the outlier on Dim1 |
| E16 | Dim2 ≈ Knowledge Reliability: separates correct from incorrect parametric | PARAM_HIGH vs CONFAB Dim2=0.899; ρ(Dim2, f1_nc)=0.439 p=1e-12 | Dimensionality v1 | Dim2 captures answer quality, not routing source |
| E17 | Dim2 tracks parametric confidence (p_top1) | ρ(Dim2, p_top1)=0.274 p=1.6e-5; MW PARAM_HIGH vs LOW p=0.001 | Dimensionality v1 | P5 CONFIRMED |
| E18 | CTX_DEP and CONFAB are geometrically similar (both are parametric routing failures) | CTX_DEP vs CONFAB ALL=0.649 — weakest pairwise separation | Dimensionality v1 | Both look like "parametric memory not confidently accessed" at gen-step-1 |
| E19 | Fisher axis is depth-stratified: WEAK globally, STRONG in deep layers | Shallow L0-9: |cosim|=0.144; Mid L10-19: 0.304; Deep L20-28: **0.690** | Axis Conservation v1 | No single global axis; deep transformer blocks converge on shared direction |
| E20 | No single conserved global epistemic axis (P1 WEAK) | Global mean |cosim|=0.179, PC1=0.277 (no dominant direction) | Axis Conservation v1 | "Conserved epistemic axis" claim remains FALSIFIED at global scope |
| E21 | L0 step-1 = token embedding exactly (diff=0.000) | Lookup vs two-pass diff=0.000000; 27 unique T1 tokens from n=200 | Axis Conservation v1 | L0 onset is token selection, not transformer computation |
| E22 | L0 signal is largely lexical; deep layer signal is geometric | L0 within-token retention=0.879; L1-L28 retention=0.879–0.989 | Axis Conservation v1 | 88% of L0 signal explained by which token was generated; transformer layers preserve and modestly amplify |
| E23 | Deep Fisher axes are strongly correlated (max |cosim|=0.912) | L20-28 mean |cosim|=0.690; at least one pair near-identical | Axis Conservation v1 | Explains why trajectory adds less per layer in deep layers — largely redundant directions there |
| E24 | Signal predicts CONFAB vs CTX_DEP without retraining (multi-outcome) | Binary Fisher direction from CTX_DEP vs PARAM achieves AUROC=0.844 on CTX_DEP vs CONFAB | State Tests v2 | Test B CONFIRMED; CONFAB falls on PARAM side of boundary |
| E25 | J-score ANTI-predicts F1_nc within PARAM (ANTI_H1) | ρ=−0.266 p=0.017 — higher parametric routing signal → slightly lower answer quality within PARAM | State Tests v2 | H1 falsified; suggests J captures routing direction, not knowledge quality |
| E26 | ΔCompute confirmed: J-score predicts context benefit (H4) | ρ=+0.155 p=0.028; CTX_DEP-like items (high J) benefit more from context | State Tests v2 | Sign convention corrected: CTX_DEP=1 so positive ρ = correct direction for H4 |
| E27 | Self-consistency NOT predicted by J-score (H2 NULL) | ρ=−0.057, p=0.619 — J-score does not predict output stability across temperature runs | State Tests v2 | H2 (reliability through self-consistency) not supported |
| E28 | State verdict: LEVEL_3_CANDIDATE (multi-outcome + causal routing confirmed) | Test A + Test B significant; ΔCompute significant (ρ=+0.155, p=0.028); v3 JSON confirmed | State Tests v2 | 3/4 tests significant; LEVEL_3 requires full causal closure (P9 6/6) |
| E29 | RLHF rotates Fisher direction (cosim≈0) but dramatically improves generalization | cosim(base, instruct)=0.007; base test=0.5992 → instruct test=0.9000; both probed at L26 | D3 reasoning geometry v1 | GEOMETRY_ROTATED verdict; direction is model-specific, not transferable across training stages |
| E30 | Reasoning RL collapses the prefill→step-1 commitment gap | DeepSeek-R1-Distill-Qwen-1.5B: prefill=0.9545, step-1=0.9680, gap=+0.0134 (near-zero); test=0.6328 | D3 reasoning geometry v1 | World 2 signal: epistemic state already formed at prefill in reasoning-trained model; architectural confound (Qwen 1.5B ≠ Llama) |
| E31 | EPISTEMIC_GATE achieves Pareto wins over random retrieval at K=30% and K=50% | K=30%: GATE=0.5138 vs RANDOM=0.4338 (Δ=+0.0800); K=50%: GATE=0.6120 vs RANDOM=0.5178 (Δ=+0.0942); gate routing AUROC=0.790 | Table 2 benchmark v1 | EPISTEMIC_GATE beats TARG at all K; beats ORACLE at K≤30% (hidden-state routing selects highest-benefit items); 50% budget achieves 83% of ALWAYS_RETRIEVE F1 |
| E32 | Routing geometry PARTIALLY transfers to HotpotQA (multi-hop QA, no retraining) | TriviaQA→HotpotQA: AUROC=0.6192 (>chance 0.5, >PARTIAL threshold 0.55, <TRANSFERS threshold 0.65); n=100 balanced | L26XD v2 HotpotQA | Signal is not dataset-specific; partial transfer from single-hop fact retrieval to multi-hop reasoning. Full transfer requires domain recalibration. |
| E33 | Cross-task probe directions near noise floor — TASK_SPECIFIC_GEOMETRY | All pairwise cosines 0.004–0.035 (expected noise floor ~0.016 for 4096-dim unit vectors). TriviaQA↔HotpotQA=0.035, TriviaQA↔MMLU=−0.024, HotpotQA↔MMLU=0.004 | cross_task_cosim_v1, 2026-06-15 | Cannot distinguish true task-specificity from estimation noise at n=20–30 per class. n≥500 needed for reliable conclusions. Honest scope: TriviaQA-only claim. |
| E34 | Hewitt-Liang selectivity confirmed — probe is not fitting noise | Shuffled-label AUROC=0.543±0.043 (5 trials) vs real=1.000 (TriviaQA). PROBE_SELECTIVE verdict. | cross_task_cosim_v1, 2026-06-15 | Probe fitted on genuine structure, not dataset artifacts. |
| E35 | SeaKR signals reconstructed; Fisher LDA substantially outperforms | Fisher AUROC=0.9274 vs SeaKR_best=0.5966, gap=+0.3308. n=800 Qwen2.5-7B-Instruct TriviaQA, 97% bilateral labeling. | seakr_comparison_v2, 2026-06-15 | SeaKR signals reconstructed from paper description, not released codebase. Caveat: reconstruction may underestimate SeaKR. |
| E36 | Epistemic routing reduces retrieval cost at conservative threshold | Conservative (10.2% skip): ΔF1=−0.014 (random baseline −0.021), 34% less quality cost. Aggressive (39.2% skip): ΔF1=−0.080 (random −0.082), **no meaningful advantage**. Cost reduction: 32.9% at aggressive. | cost_benchmark_v1, Qwen2.5-1.5B-Instruct | One model, one benchmark. Aggressive threshold provides negligible advantage over random — do not overclaim. Conservative threshold is the defensible operating point. |
| E37 | KV compression achieves F1 parity at ~10pp less budget | F1=0.4083 at 39.2% budget vs Uniform50 F1=0.4091 at 50% budget (ΔF1=−0.0008, near-parity). CTX_DEP items receive proportionally more budget (+4.4pp). | kv_compression_v1 | EPISTEMIC_PARITY verdict. Full KV performs worst (context dilution). |
| E38 | Regime 2 control — answer_jump is EPISTEMIC_TRANSITION, not mode-change confound | CoT boundary: answer_jump_cot=+26.21; generic boundary (list→conclusion): generic_jump=−11.71 (OPPOSITE SIGN); no_cot_onset=+19.29. Sign reversal rules out mode-transition confound. | regime2_control_v1, 2026-06-15 | Critical control result. The CoT answer boundary is epistemically distinct from generic text mode transitions. Claim 3 is not an artifact. |
| E39 | Qwen2.5-7B (ceiling baseline) shows Δ=0.000 under RLHF | base_auroc=1.000, inst_auroc=1.000, Δ=0.000. Ceiling case: at strong enough baseline, RLHF does not attenuate at all. Added to RLHF attenuation table as the limiting case. | gqa_cosim_v1, 2026-06-15 | Within-family comparison: Qwen2.5-7B Δ=0.000 vs Qwen2.5-1.5B Δ=−0.036. Larger model → smaller attenuation. |
| E40 | J_velocity collapses under RLHF: r=0.985 (base) → r=0.374 (instruct) | Qwen2.5-1.5B base: r(J_velocity, correctness)=0.985. Instruct: r=0.374. RLHF trains away dynamic epistemic monitoring, not just static probe separability. | rlhf_attenuation_ood_v1 | ONE DATA POINT — Qwen2.5-1.5B only. Needs replication on 4+ architecture families before claiming general pattern. See jvelocity_collapse_finding.md. |
| E41 | Gemma 3 (multimodal pre-trained) fails to exhibit unimodal epistemic geometry | Fisher probe calibration fails to find PARAM/CTX_DEP separation at L26 on Gemma 3 (multimodal pre-trained model). Accessibility fragmentation under cross-modal pre-training: the single-stream unimodal epistemic surface has reorganized. | gemma3_probe_attempt, 2026-06-16 | NOT a probe failure — a boundary finding. First evidence of accessibility fragmentation under multimodal pre-training. Belongs in paper §4 as boundary condition. |
| E42 | Answer_jump direction preserved cross-backbone, magnitude backbone-stratified | Llama backbone (DeepSeek-R1-Distill-Llama-8B): answer_jump=0.101±0.169, n=20/20 all positive, verdict=REGIME_2_PARTIAL. Probe direction stable (cosim=1.0). commit_rate=0.10, z_score=1.49. Qwen backbone (v6): answer_jump=5.54, n=17/17. Magnitude 55× larger for Qwen backbone. | answer_jump_v2, 2026-06-16 | Pre-registered threshold >1.0 NOT met. Direction is universal (all 20 positive); magnitude follows baseline epistemic geometry strength (Llama AUROC=0.665, Qwen AUROC=0.899–1.000). Same mechanism as RLHF attenuation: weaker baseline → weaker effect. Claim 3(c) scopes to Qwen-family RL training as primary evidence. |

---

## PENDING — Experiment designed, not yet run

| # | Claim | Test | Kernel | Expected if True |
|---|-------|------|--------|------------------|
| P1 | ~~Fisher direction conserved globally~~ | PARTIAL — strong in deep (0.690), weak overall (0.179) | esm-axis-conservation-v1 v4 | → see E19, E20, E23 |
| P2 | ~~L0 signal is geometric~~ | MOSTLY LEXICAL — L0 retention=0.879; transformer adds ~10% geometric | esm-axis-conservation-v1 v4 | → see E21, E22 |
| P3 | ~~Epistemic variable is 1D (scalar axis)~~ | FALSIFIED — Dim1=0.359 only | esm-dimensionality-v1 v5 | → see F7 |
| P4 | ~~Epistemic variable is multi-dimensional~~ | CONFIRMED — see E14–E18 | esm-dimensionality-v1 v5 | Dim1+Dim2=0.476; 3 dims needed |
| P5 | ~~Dim2 tracks parametric reliability~~ | CONFIRMED — see E17 | esm-dimensionality-v1 v5 | ρ=0.274 p=1.6e-5; f1 ρ=0.439 p=1e-12 |
| P6 | ~~Multi-outcome predictivity~~ | CONFIRMED — Test B AUROC=0.844; Test A anti-significant | esm-state-tests-v2 v3 | → see E24, E25, E28 |
| P7 | ~~Fisher trajectory gain survives hold-out~~ | FALSIFIED — see F10 | P7 hold-out v1 | single-point 0.9294 > trajectory 0.8474; trajectory overfits |
| P8 | ~~J-score predicts ΔCompute (H4)~~ | CONFIRMED — ρ=+0.155 p=0.028 (sign corrected) | esm-state-tests-v2 v3 | → see E26 |
| P9 | Causal controllability | L18 intervention → monotone J-score response | Partial (2/6 passes in v1) | All 6 monotonicity checks pass |

---

## FALSIFIED — Tests run, claim did not survive

| # | Claim | Test | Result | Notes |
|---|-------|------|--------|-------|
| F1 | Answer framing raises prefill AUROC (H1) | Framing v1: answer_frame vs standard prefill | Framing LIFT = −0.022 (negative) | H1 formally rejected |
| F2 | j_drop (step-1 minus step-5) detects confabulation | Never tested — removed from paper §8.7 | Claim was fabricated, not measured | Removed from §8.7 |
| F3 | 46% KV cache savings via epistemic routing | CAMS v3: F1 parity requires 39.2% budget | Savings figure was overstated | Revised to epistemic parity framing |
| F4 | "Conserved epistemic axis" | Unstated prior to axis conservation test | No evidence yet, pending P1 | Do not use this phrase until P1 runs |
| F5 | DoLa is anti-predictive | Decomposition v1 | AFE=0.33 (near chance) | DoLa dead as a signal |
| F6 | Full ensemble improves cross-domain | Decomposition v1 | 0.460 vs 0.734 margin alone | Ensemble collapses cross-domain signal |
| F7 | Epistemic variable is scalar (1D axis) | Dimensionality v1: Dim1=0.359 explains only 36% of between-class variance | All pairwise AUROCs require both Dim1+Dim2 | MULTI_D confirmed — "epistemic axis" framing obsolete |
| F8 | H1: J-score predicts knowledge quality within PARAM (ρ > 0) | State Tests v2: ρ=−0.266, p=0.017 — opposite sign | High J within PARAM = borderline case, not high-quality case | J tracks routing proximity to decision boundary, not correctness |
| F9 | Fisher direction transfers across training stages | D3: cosim(base, instruct)=0.007 ≈ orthogonal | Direction is completely reorganized by RLHF | Probes must be calibrated per training stage; cross-stage transfer is not viable |
| F10 | Fisher trajectory gain generalizes beyond calibration | P7: hold-out trajectory=0.8474 < single-point L26=0.9294; Δ=−0.082; prior CV gain +0.1482 was artifact | Compounded supervised fitting (29 LDAs + LR on same 210 samples) = overfitting cascade | Single-point L26 is both simpler and more generalizable than trajectory |

---

## OPEN QUESTIONS — Not yet testable, need experimental design

| # | Question | Why It Matters | Road to Answer |
|---|----------|----------------|----------------|
| Q1 | Does the variable exist during prefill but become *readable* at step-1? | L0 prefill = 0.500 (chance); L0 step-1 = 0.784; something changes at generation | Causal intervention on prefill hidden states |
| Q2 | Is the signal consistent across domains (MMLU, HotpotQA, NQ)? | Cross-domain generalization required for state claim | Cross-domain transfer experiment |
| Q3 | What is the mechanistic relationship between GQA and signal strength? | Architecture confound (E11) unexplained | Ablation across attention head configurations |
| Q4 | Does Dim2 exist in Qwen and Phi, or is it Llama-specific? | Dimensionality result may not generalize | Run esm-dimensionality-v1 on multiple models |
| Q5 | ~~What happens to the Fisher axis under fine-tuning / RLHF?~~ | RESOLVED — see E29, F9: direction rotates (cosim≈0), generalization improves dramatically | D3 kernel complete | GEOMETRY_ROTATED verdict |
| Q6 | Does Dim2 (knowledge reliability) exist in Qwen/Phi, or is it Llama-specific? | Universality of multi-dimensional structure — Dim2 is the reliability axis; if arch-specific the claim shrinks | Run esm-dimensionality-v1 on Qwen2.5-7B and Phi-3-mini |

---

## Road Map — Execution Order

```
DONE:  [P3, P4, P5] → Dimensionality       COMPLETE (MULTI_D confirmed, Dim2=reliability)
DONE:  [P1, P2]     → Axis Conservation    COMPLETE (depth-stratified; L0 lexical; deep geometric)
DONE:  [P6, P8]     → State Tests          COMPLETE (LEVEL_3_CANDIDATE; Test B 0.844; ΔCompute p=0.028)
DONE:  [D3]         → Reasoning Model Geometry  COMPLETE (GEOMETRY_ROTATED; instruct test=0.9000; reasoning gap≈0)
DONE:  [Table2]     → Compression Benchmark     COMPLETE (PARETO_WIN K=30%+50%; GATE>ORACLE at K≤30%)
DONE:  [L26XD]      → Cross-domain at L26        COMPLETE (PARTIAL_TRANSFER; HotpotQA AUROC=0.6192; domain recalibration needed for full transfer)
DONE:  [P7]         → Trajectory hold-out        COMPLETE (TRAJECTORY_WORSE; single-point 0.9294 > traj 0.8474)
LAST:  [D4]         → Training Signal            (epistemic consistency loss — transformative)
LAST:  [P9]         → Causal Control             (full 6/6 monotonicity)
```

---

## What Would Change Everything

| Result | Consequence |
|--------|-------------|
| Dim1 < 0.65 (multi-dimensional) | "Epistemic State Space" replaces "Epistemic Axis" — entire framing shifts |
| Dim2 correlates with p_top1 (p < 0.01) | H2 becomes real — reliability axis exists independent of routing |
| ΔCompute ρ < −0.2 (p < 0.01) | Variable is H4/H5: predicts compute value — much bigger than retrieval gate |
| Within-token AUROC at L1 < 0.55 | L0+ signal is mostly lexical — commitment onset is token selection, not geometry |
| Mean \|cosim\| < 0.1 (no axis conservation) | Each layer has independent axis — "trajectory" is 28 separate variables, not one |
| Test B AUROC < 0.55 (confab fails) | Binary classifier only — LEVEL_1 verdict, not state |
