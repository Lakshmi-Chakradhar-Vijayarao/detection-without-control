"""
EntropyMatcher — excluded zone RC construction for entropy-matched CC/CW pools.

The core threat to validity: if CC items systematically have lower entropy than
CW items, any entropy-sensitive classifier would distinguish them without
learning anything about the hidden-state geometry. The excluded zone RC removes
this confound by construction.

How it works
────────────
1. Compute output entropy for every item in the pool.
2. Find theta_conf: the entropy threshold below which items are "committed."
3. Find theta_low: set so that the CC and CW distributions within
   [0, theta_low] are statistically indistinguishable (KS test p > 0.05).
4. Items in (theta_low, theta_conf] are excluded (the RC zone).
5. The surviving items are entropy-matched: a classifier that only reads
   entropy cannot exceed chance on this pool.

This is why the Fisher+PCA64 probe AUROC=0.854 is not explained by entropy:
the single-pass entropy baseline on the same pool gives AUROC=0.483 (§4.3).

Entropy definition
──────────────────
H(x) = -∑_v p_v log p_v  over the top-K token logits at generation step 1.
Normalised to [0,1] by dividing by log(K).
"""

from __future__ import annotations

import numpy as np
from scipy import stats


class EntropyMatcher:
    """
    Construct the excluded zone RC and return entropy-matched CC/CW pools.

    Parameters
    ----------
    theta_conf      : float, default 0.5
        Upper entropy gate. Items above this are "uncertain" and excluded
        from both CC and CW pools before matching.
    ks_alpha        : float, default 0.05
        Target KS-test p-value for entropy distributions of CC vs CW.
        theta_low is raised until KS p > ks_alpha.
    max_iterations  : int, default 100
        Safety cap on the binary search for theta_low.
    """

    def __init__(
        self,
        theta_conf: float = 0.5,
        ks_alpha: float = 0.05,
        max_iterations: int = 100,
    ) -> None:
        self.theta_conf = theta_conf
        self.ks_alpha = ks_alpha
        self.max_iterations = max_iterations
        self.theta_low_: float | None = None

    def fit(
        self,
        cc_entropies: np.ndarray,
        cw_entropies: np.ndarray,
    ) -> "EntropyMatcher":
        """
        Find theta_low via binary search such that KS(CC_below, CW_below) is
        not significant at ks_alpha.

        Parameters
        ----------
        cc_entropies : entropies of items labeled CC (entropy ≤ theta_conf, f1 ≥ 0.5).
        cw_entropies : entropies of items labeled CW (entropy ≤ theta_conf, f1 = 0.0).
        """
        lo, hi = 0.0, self.theta_conf
        theta_low = 0.0

        for _ in range(self.max_iterations):
            mid = (lo + hi) / 2.0
            cc_below = cc_entropies[cc_entropies <= mid]
            cw_below = cw_entropies[cw_entropies <= mid]

            if len(cc_below) < 5 or len(cw_below) < 5:
                lo = mid
                continue

            ks_stat, p_val = stats.ks_2samp(cc_below, cw_below)

            if p_val > self.ks_alpha:
                theta_low = mid
                hi = mid
            else:
                lo = mid

            if hi - lo < 1e-4:
                break

        self.theta_low_ = theta_low
        return self

    def mask(self, entropies: np.ndarray) -> np.ndarray:
        """
        Return boolean mask — True for items that survive the RC exclusion.

        Items with entropy ≤ theta_low_ are kept (entropy-matched zone).
        Items in (theta_low_, theta_conf] are the RC zone and are excluded.
        Items above theta_conf were already excluded before this step.
        """
        if self.theta_low_ is None:
            raise RuntimeError("EntropyMatcher not fitted. Call fit() first.")
        return entropies <= self.theta_low_

    @property
    def rc_bounds(self) -> tuple[float, float]:
        """(theta_low, theta_conf) — the bounds of the excluded zone RC."""
        if self.theta_low_ is None:
            raise RuntimeError("EntropyMatcher not fitted.")
        return self.theta_low_, self.theta_conf


def token_entropy(logits: np.ndarray, top_k: int = 50, normalize: bool = True) -> float:
    """
    Compute output entropy from top-K token logits at generation step 1.

    Parameters
    ----------
    logits    : (vocab_size,) raw logits before softmax.
    top_k     : number of top tokens to include.
    normalize : if True, divide by log(top_k) so result ∈ [0, 1].

    Returns
    -------
    float — entropy value.
    """
    top_k = min(top_k, len(logits))
    top_logits = np.partition(logits, -top_k)[-top_k:]
    top_logits -= top_logits.max()
    probs = np.exp(top_logits)
    probs /= probs.sum()
    probs = np.clip(probs, 1e-12, None)
    h = -np.sum(probs * np.log(probs))
    if normalize:
        h /= np.log(top_k)
    return float(h)
