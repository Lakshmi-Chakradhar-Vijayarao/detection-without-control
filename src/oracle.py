"""
Bilateral oracle labeling: PARAM / CTX_DEP / CO.

Three labeling protocols used across the thesis experiments.

1. Bilateral Oracle (L1 — knowledge source routing)
   ─────────────────────────────────────────────────
   Two-pass behavioral test on each question:

   Pass A (no-context):  model answers without any retrieved passage.
   Pass B (with-context): model answers with the gold passage prepended.

   Labels:
     PARAM    — nocontext_f1 ≥ 0.50  (model knows from parameters)
     CTX_DEP  — nocontext_f1 ≤ 0.05 AND withcontext_f1 ≥ 0.50
                (model needs context; the right context fixes it)
     EXCLUDED — everything else (ambiguous; dropped before training)

2. Correctness Oracle (CO, L2 — confabulation detection)
   ──────────────────────────────────────────────────────
   Single-pass labeling using output entropy + F1:

     CC (confident-correct) — entropy ≤ θ_conf AND f1 ≥ 0.50
     CW (confident-wrong)   — entropy ≤ θ_conf AND f1 = 0.0

   θ_conf is set so that the CC/CW pool is entropy-matched via excluded
   zone RC (see EntropyMatcher). Items in the RC zone are dropped.

3. Plain CO (L2 alternative without entropy matching)
   ────────────────────────────────────────────────────
   CC — f1 ≥ 0.50
   CW — f1 = 0.0
   No entropy gate. Simpler but output statistics can explain the signal.
   Llama plain CO is INCONCLUSIVE (C047, AUROC=0.613 ≈ shuffled=0.580).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


# ────────────────────────────────────────────────────────────────────────────
# Data containers
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class OracleItem:
    question_id: str
    question: str
    nocontext_f1: float
    withcontext_f1: float
    label: str          # "PARAM" | "CTX_DEP" | "EXCLUDED"


@dataclass
class COItem:
    question_id: str
    question: str
    output_entropy: float
    f1: float
    label: str          # "CC" | "CW" | "RC" (excluded zone)


# ────────────────────────────────────────────────────────────────────────────
# Bilateral Oracle — L1 knowledge-source routing
# ────────────────────────────────────────────────────────────────────────────

class BilateralOracle:
    """
    Two-pass behavioral labeler for PARAM vs CTX_DEP classification.

    Thresholds match the paper:
        param_threshold   = 0.50   (nocontext F1 ≥ this → PARAM)
        ctx_dep_no_ctx    = 0.05   (nocontext F1 ≤ this, required for CTX_DEP)
        ctx_dep_with_ctx  = 0.50   (withcontext F1 ≥ this, required for CTX_DEP)
    """

    def __init__(
        self,
        param_threshold: float = 0.50,
        ctx_dep_no_ctx: float = 0.05,
        ctx_dep_with_ctx: float = 0.50,
    ) -> None:
        self.param_threshold = param_threshold
        self.ctx_dep_no_ctx = ctx_dep_no_ctx
        self.ctx_dep_with_ctx = ctx_dep_with_ctx

    def label(
        self,
        question_id: str,
        question: str,
        nocontext_f1: float,
        withcontext_f1: float,
    ) -> OracleItem:
        if nocontext_f1 >= self.param_threshold:
            label = "PARAM"
        elif (
            nocontext_f1 <= self.ctx_dep_no_ctx
            and withcontext_f1 >= self.ctx_dep_with_ctx
        ):
            label = "CTX_DEP"
        else:
            label = "EXCLUDED"
        return OracleItem(question_id, question, nocontext_f1, withcontext_f1, label)

    def label_batch(
        self,
        question_ids: Sequence[str],
        questions: Sequence[str],
        nocontext_f1s: Sequence[float],
        withcontext_f1s: Sequence[float],
    ) -> list[OracleItem]:
        return [
            self.label(qid, q, nf, wf)
            for qid, q, nf, wf in zip(question_ids, questions, nocontext_f1s, withcontext_f1s)
        ]

    def split(
        self, items: list[OracleItem]
    ) -> tuple[list[OracleItem], list[OracleItem]]:
        """Return (param_items, ctx_dep_items) — EXCLUDED items are dropped."""
        param = [i for i in items if i.label == "PARAM"]
        ctx_dep = [i for i in items if i.label == "CTX_DEP"]
        return param, ctx_dep


# ────────────────────────────────────────────────────────────────────────────
# Correctness Oracle — L2 confabulation detection
# ────────────────────────────────────────────────────────────────────────────

def label_co(
    question_ids: Sequence[str],
    questions: Sequence[str],
    output_entropies: Sequence[float],
    f1_scores: Sequence[float],
    theta_conf: float,
    theta_low: float | None = None,
) -> list[COItem]:
    """
    Label items as CC, CW, or RC (excluded zone).

    Parameters
    ----------
    theta_conf : upper entropy threshold — items with entropy > theta_conf
                 are not confidently committed and are excluded entirely.
    theta_low  : lower bound of the RC zone. Items with entropy in
                 (theta_low, theta_conf] are excluded (RC zone). If None,
                 no lower bound is applied (plain CO without entropy matching).

    CC: entropy ≤ theta_conf AND f1 ≥ 0.50
    CW: entropy ≤ theta_conf AND f1 = 0.0
    RC: entropy in (theta_low, theta_conf] — excluded to prevent entropy
        from explaining the probe signal (see EntropyMatcher)
    """
    items = []
    for qid, q, ent, f1 in zip(question_ids, questions, output_entropies, f1_scores):
        if ent > theta_conf:
            label = "RC"
        elif theta_low is not None and ent > theta_low:
            label = "RC"
        elif f1 >= 0.50:
            label = "CC"
        elif f1 == 0.0:
            label = "CW"
        else:
            label = "RC"
        items.append(COItem(qid, q, ent, f1, label))
    return items


def co_to_arrays(
    items: list[COItem],
    hidden_states: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert labeled COItems + hidden state dict into (X, y) arrays for FisherProbe.

    Parameters
    ----------
    items         : list from label_co(), RC items are automatically excluded.
    hidden_states : mapping from question_id → hidden state vector.

    Returns
    -------
    X : (n_items, hidden_dim) — CC and CW items only, entropy-matched.
    y : (n_items,) — 1=CC, 0=CW.
    """
    X, y = [], []
    for item in items:
        if item.label == "RC":
            continue
        if item.question_id not in hidden_states:
            continue
        X.append(hidden_states[item.question_id])
        y.append(1 if item.label == "CC" else 0)
    return np.array(X), np.array(y)
