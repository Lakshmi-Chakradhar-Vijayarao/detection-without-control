# Research Atlas — Computational Observability Program
**Status:** LIVING DOCUMENT — updated when the research landscape shifts  
**Last updated:** 2026-07-12  
**Purpose:** Map where this program sits in the broader research ecosystem. Build by scientific question, not by keyword.

---

## Program Position

```
                     THEORY
                        │
           Computational Observability
                        │
         ┌──────────────┼──────────────┐
         │              │              │
   Observability   Commitment    Accessibility
         │              │              │
         └──────────────┼──────────────┘
                        │
              Measurement Science
                        │
            Mechanistic AI Research
                        │
         Alignment • RAG • Reasoning
```

The program is **not inside one community**. It sits at the junction of five.

That is both the opportunity (nobody else is asking this exact question) and the risk (no single community owns it — every reviewer brings a different frame).

---

## The One Sentence

> **The purpose of this work is not to introduce another probing method. It is to establish experimental methodology for studying internal computation independently of any specific estimator.**

Every paper, experiment, and claim should be legible through this frame. If something does not connect to experimental methodology for studying internal computation, it is secondary.

---

## The Five Research Communities

### How to read this map

For each community, the column structure is:

| Paper | Main Question Answered | Overlap with Thesis | Where Thesis Differs | Relation | Relevant Experiment |
|---|---|---|---|---|---|

**Relation types:**
- **SUPPORTS** — provides foundation or evidence for thesis claims
- **CHALLENGES** — directly attacks a thesis claim (must be addressed)
- **COMPLEMENTS** — answers a different question, used as context
- **BACKGROUND** — establishes vocabulary/tools the thesis uses

---

## Community 1 — Mechanistic Interpretability

**Central question this community asks:** *What is represented inside a neural network, and how?*

**Why it matters for the thesis:** The bilateral oracle + Fisher probe are tools from this community's toolkit. The mechanistic interpretability community establishes the conceptual vocabulary (residual stream, features, superposition) that makes the measurement design sensible.

**Key researchers:** Chris Olah, Elhage, Trenton Bricken, Adly Templeton, Neel Nanda, Samuel Marks, Joshua Batson (Anthropic Transformer Circuits team)

### Core Papers

| Paper | Main Question | Overlap | Difference | Relation | Experiment |
|---|---|---|---|---|---|
| Elhage et al. 2022 — *Toy Models of Superposition* | Why are individual neurons polysemantic? | Establishes why PCA is needed (superposition) | We measure at the class level (PARAM vs CTX_DEP), not feature level | BACKGROUND | Supports PCA rationale in bilateral oracle |
| Bricken et al. 2023 — *Towards Monosemanticity* | Can SAEs recover interpretable features? | SAE is next estimator after Fisher (see Q6) | We use a discriminative estimator; SAE is generative | COMPLEMENTS | Q6 — SAE feature analysis |
| Cunningham et al. ICLR 2024 — *Sparse Autoencoders Find Highly Interpretable Features* | Do SAEs scale beyond toy models? | Establishes SAE feasibility for residual stream | Residual stream vs. MLP-only | COMPLEMENTS | Q6 |
| Templeton et al. 2024 — *Scaling Monosemanticity* | Do SAE features work at production scale? | Confirms SAEs viable on frontier models (Claude 3) | Different model family | COMPLEMENTS | Future estimator replacement |
| Gurnee et al. 2026 — *Verbalizable Representations Form a Global Workspace* | Which representations does the model actively reason with? | Most adjacent Anthropic work: also uses probing + intervention | J-lens identifies *which* concepts are active; bilateral oracle identifies *what epistemic class* applies | COMPLEMENTS | Q12 (same/distinct direction comparison) |
| Marks & Tegmark 2023 — *The Geometry of Truth* | Is truth linearly represented in LLMs? | Direct comparison for L2 CO labeling; Fisher vs. mass-mean probe | We separate *routing* (PARAM/CTX_DEP) from *correctness* (CC/CW); they only do correctness | SUPPORTS | Q12 — Marks-Tegmark comparison |
| Zou et al. 2023 — *Representation Engineering* | Can arbitrary concepts be read and steered from hidden states? | RepE provides the top-down framing our measurement is grounded in | RepE is concept-agnostic; bilateral oracle is epistemically specific | COMPLEMENTS | Q14 difficulty control |
| Li et al. 2023 — *Inference-Time Intervention (ITI)* | Can activation steering elicit truthful outputs? | Prior work establishing "truthful directions" exist in activations | ITI elicits truthfulness; bilateral oracle *measures* epistemic routing | SUPPORTS | Validates premise of L1 result |
| Meng et al. 2022 — *ROME* | Where are factual associations stored causally? | ROME provides the causal tracing precedent for C005/C024 null results | ROME finds causal MLP modules; we find non-causal Fisher directions | SUPPORTS C005/C024 | Null patching (C005, C024) |
| Hase et al. 2023 — *Does Localization Inform Editing?* | Does causal localization predict which layers to edit? | Directly relevant: shows localization ≠ editing success | We don't edit; we probe — but same decoupling principle applies | SUPPORTS C005/C024 | Strengthens causal null interpretation |
| Park et al. ICML 2024 — *Linear Representation Hypothesis* | Is linear encoding formally provable? | Provides rigorous grounding for why Fisher works | More abstract; we are empirical | BACKGROUND | Validates probe design |
| Jiang et al. ICML 2024 — *Origins of Linear Representations* | Why does linearity emerge from training? | Explains why bilateral oracle sees linear structure | Theoretical; we are measurement-first | BACKGROUND | Supports the estimator choice |

---

## Community 2 — Truth / Hallucination Geometry

**Central question this community asks:** *Can hidden states predict whether a model is confabulating?*

**Why it matters for the thesis:** L2 (confabulation detection) operates in this community. Every paper here is either a direct competitor or a baseline for the Fisher-entropy gap claim.

**Key researchers:** Collin Burns (CCS), Samuel Marks (Geometry of Truth), Yarin Gal (semantic entropy), Kenneth Li (ITI), OATML Oxford group

### Core Papers

| Paper | Main Question | Overlap | Difference | Relation | Experiment |
|---|---|---|---|---|---|
| Burns et al. ICLR 2023 — *CCS* | Can truth be found in activations without labels? | CCS is the unsupervised version of what bilateral oracle does with supervision | CCS uses logical consistency; bilateral oracle uses behavioral intervention | SUPPORTS | L2 result should note CCS as prior approach |
| Marks & Tegmark 2023 — *Geometry of Truth* | Is truth linearly organized and task-transferable? | Q12: are we measuring the same direction? | They find "truth direction"; we find "epistemic routing direction" — may be distinct | CHALLENGES (Q12) | Q12 comparison experiment |
| Azizian et al. 2025 — *Geometries of Truth Are Orthogonal Across Tasks* | Do truth directions generalize across tasks? | DIRECTLY CHALLENGES Law 1 generalization claim | Their finding: truth directions are task-specific (disjoint feature support). Our finding: AUROC ≥ 0.70 across tasks. Possible resolution: routing is more conserved than truth-direction geometry | CHALLENGES | Q14 — difficulty control + task transfer |
| Li et al. NeurIPS 2023 — *ITI* | Can activation steering improve TruthfulQA? | Foundation for understanding "truthful directions" in activations | We probe, don't steer (except EXP-I) | SUPPORTS | EXP-I causal validation |
| Chen et al. ICLR 2024 — *INSIDE / EigenScore* | Do covariance eigenvalues of hidden states predict hallucination? | Also reads geometric signal from hidden states for confabulation | Different estimator (Gram matrix eigenvalues vs. Fisher LDA) | COMPLEMENTS | L2 comparison |
| Kossen et al. 2024 — *Semantic Entropy Probes* | Can a single-generation probe approximate semantic entropy? | DIRECT BASELINE for L2. The thesis's Fisher gap (+0.240) is a claim of beating this. | Their probe is trained on hidden states for semantic entropy; our probe is trained for PARAM/CTX_DEP | CHALLENGES | L2 result — Fisher vs. entropy probe comparison |
| Su et al. ACL 2024 — *MIND* | Can unsupervised internal-state detection work at inference time? | Also uses internal states without sampling | Unsupervised vs. bilateral oracle supervision | COMPLEMENTS | L2 |
| Zhang et al. ACL 2025 — *ICR Probe* | Do cross-layer dynamics improve hallucination detection over single-layer probes? | Cross-layer dynamics vs. our single extraction point (L26, step-1) | We measure at a fixed extraction point; ICR integrates across layers | COMPLEMENTS | Q13 trajectory analysis |
| Zhang et al. ACL 2025 — *PRISM* | Can prompt design improve cross-domain probe generalization? | Cross-domain generalization of hidden-state probes | We don't currently test cross-domain generalization | COMPLEMENTS | Future scope |
| [Do LLMs Know?] 2024 — arXiv:2510.09033 | Do probes detect knowledge recall rather than truthfulness? | STRONGEST METHODOLOGICAL CHALLENGE. Must be addressed explicitly. | Our bilateral oracle assigns labels via intervention, not self-report. Partial defense, but Q14 (difficulty) remains the full answer. | **CHALLENGES** | Q14 difficulty control |
| Dunning-Kruger ACL 2026 — *GHOST* | Can geometric hidden-state observation detect confabulation while decoupling from output? | Almost identical L2 task; may have higher AUROC than our Fisher | Different estimator (geometry-based); may be simultaneous work | **CHALLENGES / COMPLEMENTS** | Must compare to L2 result |

---

## Community 3 — Reasoning / Commitment

**Central question this community asks:** *When is the answer actually decided during a chain-of-thought trace?*

**Why it matters for the thesis:** L3 (commitment timing) operates here. Multiple 2026 papers independently confirmed the same findings. Frame these as convergent evidence.

**Key researchers:** Gabriele Sarti, Long Zhang, Kyle Cox, Adrià Garriga-Alonso, Owain Evans, Neel Nanda, Arthur Conmy

### Core Papers

| Paper | Main Question | Overlap | Difference | Relation | Experiment |
|---|---|---|---|---|---|
| Scalena et al. 2026 — *Beyond the Commitment Boundary* (arXiv:2606.13603) | When does the answer probability stabilize in a CoT trace? | DIRECT PARALLEL to commit_pct (L3). Same finding independently. | They use early-exit probing at each step; we use entropy-based commit detection | **CONVERGENT** | L3 result — cite as independent confirmation |
| Zhang et al. 2026 — *When Does an LM Commit?* (arXiv:2605.06723) | Can a formal theory of pre-verbalization commitment be built? | Formal theory of commitment; provides mathematical framework the thesis lacks | They use log-odds projection (finite answer set); we use entropy threshold | **COMPLEMENTS** | Law 2 — provides theoretical formalism |
| Cox et al. 2026 — *Decoding Answers Before CoT* (arXiv:2603.01437) | Are committed-answer directions causal (not just diagnostic)? | DIRECTLY RELEVANT to C005/C024 null patching causality discussion | They find the committed-answer direction IS causal (>50% flip). C005/C024 patched Fisher centroid directions. These may be different directions. | **CHALLENGES C005/C024 interpretation** | Must clarify: centroid Fisher ≠ committed-answer direction |
| Yuan et al. 2026 — *Hidden Error Awareness* (arXiv:2605.09502) | Can hidden states detect reasoning errors, and can you fix them? | DIRECT PARALLEL to C005/C024. Same finding: 0.95 AUROC signal, but interventions all fail. | They try steering, self-correction, best-of-N; all fail. We try centroid patching; fails. | **SUPPORTS C005/C024** | Cite as independent replication of epiphenomenal result |
| Boppana et al. 2026 — *Reasoning Theater* (arXiv:2603.05488) | Is visible CoT post-hoc rationalization? | Consistent with high commit_pct interpretation | They show performative CoT on largest models; we show it on 1.5B–8B | **SUPPORTS L3** | L3 result |
| Wang 2026 — *LLM Reasoning Is Latent* (arXiv:2604.15726) | Should latent state be the primary object of study for LLM reasoning? | Programmatic argument for our L3 measurement approach | Theoretical/position paper; we provide empirical evidence | **SUPPORTS** | L3 framing |
| [Value Axis] 2026 — arXiv:2606.17056 | Is there a linear direction encoding "on the right track"? | Closely related to L2 CO labeling (correct vs. wrong trajectory) | They find the direction; we find the AUROC (same object, different measurement) | **COMPLEMENTS** | Q13 trajectory analysis |
| [Point of No Return] 2026 — arXiv:2605.17113 | Can counterfactual intervention localize the deceptive commitment step? | Causal analysis of commitment — most directly related to EXP-I | They study deceptive reasoning; we study honest commitment timing | **COMPLEMENTS** | EXP-I causal validation |
| Turpin et al. 2023 — *Unfaithfulness in CoT* | Does CoT actually reflect the model's reasoning? | Related to post-hoc rationalization interpretation of high commit_pct | They show behavioral unfaithfulness; we show internal commitment timing | **SUPPORTS** | Law 2 framing |
| Lanham et al. 2023 — *Faithfulness in CoT* | Can faithfulness of CoT be measured? | Defines faithfulness metrics the thesis should reference for commit_pct framing | They define faithfulness from output; we define commitment from hidden states | **COMPLEMENTS** | Law 2 |

---

## Community 4 — Knowledge Source

**Central question this community asks:** *Where did this answer come from — parametric memory or retrieved context?*

**Why it matters for the thesis:** This is the exact question L1 (bilateral oracle, PARAM/CTX_DEP) answers. The closest prior work lives here.

**Key researchers:** Tighidet et al., Zhao et al. (NeurIPS 2025), Gottesman & Geva (KEEN), Zijun Yao (SeaKR), Ingeol Baek (Probing-RAG)

### Core Papers

| Paper | Main Question | Overlap | Difference | Relation | Experiment |
|---|---|---|---|---|---|
| Tighidet et al. EMNLP 2024 — *Probing LMs on Their Knowledge Source* (arXiv:2410.05817) | Can probes distinguish PK vs. CK from activations? | **CLOSEST PRIOR WORK.** Same scientific question as L1. Must be addressed explicitly in related work. | They use adversarial prompts (injection); bilateral oracle uses context-withholding. Their labels may be confounded by prompt injection artifacts. | **CHALLENGES** | L1 result — write comparison paragraph |
| [Knowledge Attribution] 2025 — arXiv:2602.22787 | Can lightweight probes attribute LLM answers to PK vs. CK? | Very close labeling protocol; self-supervised; 0.96 F1 | Their labels come from asking the model which source it used (self-report); bilateral oracle uses behavioral intervention (external ground truth) | **CHALLENGES** | L1 result — self-report vs. intervention |
| Zhao et al. NeurIPS 2025 — *PK/CK Reconciliation* | How do transformers internally arbitrate between PK and CK? | Mechanistic question behind the bilateral oracle signal | They find distinct attention head sets for PK/CK; we find Fisher-separable residual stream geometry | **COMPLEMENTS** | Q6 mechanism analysis |
| Gottesman & Geva EMNLP 2024 — *KEEN* | Can per-entity knowledge be estimated without generating tokens? | Same goal (know before generation); probes entity tokens | They probe entity token representations; we probe step-1 after query | **COMPLEMENTS** | L1 |
| Gekhman et al. 2025 — *Inside-Out* | Do LLMs internally encode more knowledge than they surface? | Motivates probing over behavioral measurement for L1/L2 | They find ~40% more knowledge in hidden states; we quantify the AUROC signal | **SUPPORTS** | Foundation for why L2 matters |
| [Do LLMs Know?] 2024 — arXiv:2510.09033 | Do probes detect recall patterns rather than truthfulness? | Direct methodological challenge to L1 and L2 | Our bilateral oracle assigns labels via intervention; partial defense but difficulty control (Q14) needed | **CHALLENGES** | Q14 |
| Yang et al. ACL 2025 — *SeaKR* | Can Gram matrix geometry of hidden states gate retrieval? | Direct application; already compared in CLAIM_SHEET (Fisher AUROC 0.93 vs. SeaKR 0.60) | Different estimator (Gram matrix); different task framing | **SUPPORTS** | Comparison already in CLAIM_SHEET |
| Baek et al. NAACL 2025 — *Probing-RAG* | Can a lightweight feed-forward prober gate retrieval? | 57.5% retrieval skip rate — closest routing application | They train a full classifier; bilateral oracle Fisher is simpler and cleaner | **COMPLEMENTS** | Application layer |
| [Self-Routing RAG] 2025 — arXiv:2504.01018 | Can k-NN over hidden states drive source routing? | k-NN over hidden states as routing policy | Different geometry-based approach | **COMPLEMENTS** | Application layer |

---

## Community 5 — Training Dynamics

**Central question this community asks:** *Why do these representations emerge during training, and when?*

**Why it matters for the thesis:** This is where Law 3 (Accessibility, INVERTED_U at L1, MONOTONE_RISE at L2) lives. The most important experiments for program expansion live here.

**Key researchers:** Max Müller-Eberstein (Subspace Chronicles), Ahmad Dawar Hakimi (Time Course MechInterp), Naftali Tishby (IB), Alethea Power (Grokking)

### Core Papers

| Paper | Main Question | Overlap | Difference | Relation | Experiment |
|---|---|---|---|---|---|
| Müller-Eberstein et al. EMNLP 2023 — *Subspace Chronicles* (arXiv:2310.16484) | How do linguistic subspaces emerge, shift, and interact during training? | **MOST DIRECT PRECEDENT FOR LAW 3.** Tracks probing AUROC across 2M training steps. | They track linguistic subspaces (syntax, semantics); we track epistemic routing signal. Same methodology, different signal. | **SUPPORTS** | Exp A (Law 3 Llama) — cite as methodological precedent |
| Hakimi et al. ACL 2025 — *Time Course MechInterp* (arXiv:2506.03434) | How do attention head and FFN roles evolve across 40 training checkpoints? | Tracks component roles across OLMo-7B checkpoints — directly relevant to INVERTED_U/MONOTONE_RISE | They find answer-specific attention heads have high turnover; FFNs stable and refining. This partially explains H-A (compression) and H-B (supervision). | **SUPPORTS Law 3** | Exp A interpretation |
| [Crosscoding Through Time] 2025 — arXiv:2509.05291 | How do SAE features emerge and consolidate during pretraining? | SAE feature evolution — analogous to what we want to know about Fisher | SAE-level granularity vs. our AUROC-level measurement | **COMPLEMENTS** | Future: EXP-L + SAE tracking |
| Power et al. 2022 — *Grokking* | Why does generalization appear suddenly after apparent overfitting? | INVERTED_U at L1 is potentially a grokking-like phenomenon (rise then fall) | Grokking is about generalization; we study observability | **SUPPORTS H-A** | Law 3 theoretical interpretation |
| Tishby & Zaslavsky 2015 — *Information Bottleneck Principle* | Does deep learning compress representations via information bottleneck? | H-A (Compression) is built on IB theory | IB is the theory; Law 3 is the empirical observation | **SUPPORTS H-A** | Competing theories framework |
| Nanda et al. 2023 — *Progress Measures for Grokking* | What are the mechanistic stages of grokking? | Phase transitions in training — directly relevant to INVERTED_U shape | They study toy arithmetic; we study epistemic routing | **SUPPORTS H-A** | Law 3 interpretation |

---

## Community 6 — Measurement Science

**Central question this community asks:** *What does it mean to measure something rigorously in ML, and why does it matter?*

**Why it matters for the thesis:** This is the meta-scientific community that gives the program its identity. Most papers in ML ignore this question. The program's deepest contribution lives here.

**Key researchers:** Finale Doshi-Velez, Been Kim, Zachary Lipton, Belinkov (probing methodology)

### Core Papers

| Paper | Main Question | Overlap | Difference | Relation | Experiment |
|---|---|---|---|---|---|
| Doshi-Velez & Kim 2017 — *Towards a Rigorous Science of Interpretable ML* (arXiv:1702.08608) | What makes an interpretability claim rigorous? | Defines the measurement standard the bilateral oracle is designed to meet | They define the standard; we provide an instance of meeting it | **SUPPORTS** | Frame the whole paper against this standard |
| Lipton 2016 — *The Mythos of Model Interpretability* (arXiv:1606.03490) | Are interpretability claims systematically vague? | Critiques weak interpretability claims — exactly what the bilateral oracle is designed to avoid | Critique paper; we provide a response | **SUPPORTS** | Introduction framing |
| Belinkov 2022 — *Probing Classifiers: Promises, Shortcomings, Advances* | What are the failure modes of probing methodology? | Comprehensive audit of probing — the bilateral oracle addresses three of the listed failure modes | Survey; we apply the lessons | **BACKGROUND** | Related work |
| Hewitt & Manning 2019 — *Structural Probe for Finding Syntax* | How should a probe be designed to find structure vs. noise? | Selectivity test — our shuffled control is the equivalent | They define structural probing; we apply analogous discipline | **SUPPORTS** | Statistical controls |

---

## The Three Mandatory Papers

These three papers are not just citations. They require explicit engagement in the related work section. A reviewer who knows these papers will evaluate the thesis against them.

---

### 1. Tighidet et al. 2024 — Probing LMs on Their Knowledge Source (arXiv:2410.05817)

**Why mandatory:** This is the closest prior work. A reviewer at EMNLP or ACL will immediately compare the bilateral oracle to this paper. The comparison must be explicit.

**What they do:** Train classifiers on hidden activations to distinguish parametric knowledge (PK) from contextual knowledge (CK), using prompts designed to contradict the model's parametric knowledge (adversarial injection).

**What we do differently:**
- Labels assigned via context-withholding (two-pass intervention), not prompt injection
- Prompt injection can create artifacts: the model may detect the injection pattern, not the epistemic routing
- Bilateral oracle tests each item twice independently; their approach is single-pass
- Fisher+PCA64 is the probe architecture, chosen for its interpretable geometry

**Draft paragraph for related work:**
> The closest prior work on knowledge-source probing is Tighidet et al.\ \citeyear{tighidet2024}, who train classifiers on internal activations to distinguish parametric knowledge (PK) from contextual knowledge (CK) by designing prompts that contradict the model's parametric memory. The bilateral oracle differs in three respects: (1) labels are assigned by independent behavioral testing under context-withholding rather than by adversarial prompt injection, avoiding the confound that the model may detect the injection style rather than the epistemic state; (2) bilateral agreement (a question must pass both the no-context and with-context filter) provides cleaner labels than single-pass evaluation; (3) the labeling protocol is estimator-agnostic — the Fisher probe is downstream of the labeling and could be replaced without changing the protocol's validity.

---

### 2. "Do LLMs Really Know What They Don't Know?" 2024 (arXiv:2510.09033)

**Why mandatory:** This is the strongest methodological challenge to the program. It argues that probes trained on hidden states learn to detect *knowledge recall patterns* (does the model recognize this item?) rather than *truthfulness* (is this answer actually correct?). If true, the bilateral oracle is a good recall detector, not an epistemic router.

**The defense:**
- Bilateral oracle labels are assigned by *behavioral intervention* (withholding context), not by asking the model to report its own knowledge
- The PARAM/CTX_DEP distinction is defined operationally — what the model *needs* — not by what the model *believes* about its knowledge
- This is a partial defense: the difficulty control experiment (Q14) is the full answer. A probe that detects recall patterns would degrade at matched difficulty; a probe that detects epistemic routing would not.

**Draft paragraph for related work:**
> A key methodological challenge is raised by [Do LLMs Know?], who argue that probes trained on hidden states learn knowledge recall patterns rather than truthfulness: the probe detects ``does the model remember this?'' rather than ``is this correct?'' The bilateral oracle partially addresses this through its intervention-based labeling: \textsc{param} and \textsc{ctx\_dep} labels are assigned by behavioral testing under context-withholding, not by self-report. Nevertheless, the distinction between recall and routing remains an open empirical question. A fully satisfactory answer requires a difficulty-controlled experiment (Q14, pending): if Fisher discriminates \textsc{param}/\textsc{ctx\_dep} at matched item difficulty, the signal is epistemic; if it does not, it may reflect difficulty. The difficulty control experiment is the Minimum Convincing Experiment for the bilateral oracle's core claim.

---

### 3. Cox et al. 2026 — Decoding Answers Before CoT (arXiv:2603.01437)

**Why mandatory:** This paper directly changes the causality discussion. Our C005/C024 null results interpret the geometry as "diagnostic, not causal." Cox et al. show that a *specific* activation direction is causal — activation steering flips answers in >50% of cases. A reviewer will immediately flag the apparent contradiction.

**The resolution:**
- C005/C024 patched *class-mean Fisher directions* (centroid of PARAM vs. CTX_DEP)
- Cox et al. found the *committed-answer direction* (the direction that encodes the specific answer the model will produce)
- These are different directions. Fisher LDA maximizes the ratio of between-class to within-class variance; the committed-answer direction encodes the specific answer label
- The null patching result establishes: "the class-mean Fisher direction is not the causal control vector." It does not establish: "no causal direction exists."

**Draft paragraph for related work:**
> Our causal null results (C005, C024) used class-mean centroid patching along the Fisher LDA direction and found no significant quality change, suggesting the geometry is diagnostic but not causally controlling via this specific perturbation. Cox et al.\ \citeyear{cox2026decoding} subsequently demonstrated that a \emph{specific} activation direction---the committed-answer direction, extracted from residual stream activations before chain-of-thought begins---achieves $\sim$0.90 AUC for predicting the final answer, and that steering along this direction flips answers in $>$50\% of cases. These results are consistent: the Fisher LDA class-mean direction and the committed-answer direction are not required to be the same. Fisher LDA extracts the direction maximally separating \textsc{param} from \textsc{ctx\_dep} class means; Cox et al.\ extract the direction encoding the specific output label. The null patching result establishes that the class-mean direction is not the causal control vector for knowledge-source routing; it does not establish that no causal direction exists.

---

## Key Researcher Map

### Anthropic (Transformer Circuits / Interpretability)
- **Chris Olah** — circuits framework, superposition, features
- **Nelson Elhage** — toy models, MLP circuits, SoLU
- **Trenton Bricken, Adly Templeton, Joshua Batson** — sparse autoencoders, monosemanticity
- **Samuel Marks** — Geometry of Truth, truth directions
- **Wes Gurnee** — J-space / verbalizable representations (Global Workspace paper)
- **Nora Belrose** — probing and intervention methodology
- **Jack Lindsey** — Global Workspace paper

### DeepMind / Google
- **Been Kim** — rigorous interpretability methodology (TCAV, Towards Rigorous Science)
- **Owain Evans** — faithfulness in CoT, reasoning models
- **David Krueger** — alignment and interpretability

### Academic (recurring in this program's bibliography)
- **Yarin Gal (OATML Oxford)** — semantic entropy, semantic entropy probes
- **Jacob Steinhardt** — CCS, latent knowledge
- **David Bau** — ROME, causal tracing
- **Yonatan Belinkov** — probing survey, probing methodology
- **Kiho Park, Victor Veitch** — linear representation hypothesis formalization
- **Max Müller-Eberstein** — Subspace Chronicles (training dynamics)
- **Ahmad Dawar Hakimi** — Time Course MechInterp

### Authors of the Three Mandatory Papers
- **Zineddine Tighidet** — Probing LMs on Knowledge Source (EMNLP 2024)
- **Kyle Cox, Adrià Garriga-Alonso** — Decoding Answers Before CoT (2026)
- **[Authors of Do LLMs Know?]** — Internal States Reflect Knowledge Recall (2024)

---

## Experiment-to-Literature Mapping

This table maps every pending experiment in the program to the papers it addresses, extends, or must respond to.

| Experiment | Addresses | Must Respond To | Extends |
|---|---|---|---|
| **Exp A — Law 3 Llama** (PENDING) | Q4 — Law 3 cross-family | Subspace Chronicles (same methodology) | Müller-Eberstein 2023 |
| **Exp B — Full-space Fisher** (PENDING) | Q10 — PCA bottleneck | Cunningham 2024 (SAE vs. linear) | Geometry of categorical concepts |
| **Exp C — ε_C sensitivity** (PENDING) | Q11 — commit_pct robustness | Beyond Commitment Boundary (arXiv:2606.13603) | Formal commitment theory |
| **Q14 — Difficulty Control** (DESIGN PENDING) | Bilateral oracle core claim | Tighidet 2024, Do LLMs Know? 2024 | SeaKR, Probing-RAG |
| **Q13 — Trajectory Analysis** (FUTURE) | How representation moves through latent space | Value Axis (arXiv:2606.17056), ICR Probe | Time Course MechInterp |
| **EXP-L — Continuous Checkpoint Sweep** (FUTURE) | Q4 — early dynamics predict late O | Subspace Chronicles, Grokking | Time Course MechInterp |
| **Q12 — Marks-Tegmark Comparison** (FUTURE) | Is billing oracle measuring same direction as truth probe? | Marks & Tegmark 2023, Geometries Orthogonal | Geometry of Truth literature |
| **Q6 — SAE Mechanism** (FUTURE) | What features generate L2 signal? | Bricken 2023, Cunningham 2024 | Templeton 2024 (scaling) |

---

## What the Program Is Not

A reviewer from each community will frame the thesis differently. The program needs to explicitly say what it is NOT:

- **Not J-space**: J-space asks "what concepts is the model reasoning with?" The bilateral oracle asks "what epistemic class does this item belong to?" Different object of measurement.
- **Not Geometry of Truth**: GoT decodes truth-value from representations. The bilateral oracle assigns epistemic routing labels via intervention, not from representation geometry alone.
- **Not SeaKR / Probing-RAG**: These are retrieval-routing applications. The bilateral oracle is the measurement protocol that makes those applications principled.
- **Not just another probing paper**: The contribution is the experimental methodology (bilateral oracle, three-task hierarchy, claims registry) — not the AUROC values.

---

*See [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md) for the scientific question agenda.*  
*See [COMPETING_THEORIES.md](COMPETING_THEORIES.md) for the formal theory discrimination framework.*  
*See [PROGRAM_CHARTER.md](PROGRAM_CHARTER.md) for the program mission and governance.*
