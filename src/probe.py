"""
FisherProbe — PCA(n=64) → LDA(lsqr, shrinkage=auto) confabulation detector.

Spine finding: the direction this probe learns (Fisher LDA class-mean axis)
separates CC from CW answers at AUROC=0.854, but patching along it produces
max ΔF1=+0.0004 (EXP-H) — indistinguishable from zero. Detection and causal
control are dissociable.

Validated on:
    Qwen2.5-1.5B-Instruct   L26 step-1   AUROC=0.854  (entropy-matched, N=100/class)
    Llama-3.2-3B-Instruct   L26 step-1   AUROC=0.818  (entropy-matched, N=100/class)
    5-fold CV (N=500/class)              AUROC=0.763±0.012

Usage:
    probe = FisherProbe(layer=26, step=1)
    probe.fit(X_train, y_train)
    auroc = probe.auroc(X_test, y_test)
    direction = probe.direction_in_hidden_space()
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score


class FisherProbe:
    """
    Two-stage linear probe: PCA dimensionality reduction → Fisher LDA.

    Labels
    ------
    1 = CC  (confident-correct)  entropy ≤ θ_conf AND F1 ≥ 0.5
    0 = CW  (confident-wrong)    entropy ≤ θ_conf AND F1 = 0.0
    Items in the excluded zone RC (θ_low < entropy ≤ θ_conf) are removed
    before training so output statistics cannot explain probe performance.
    See EntropyMatcher for construction.
    """

    def __init__(
        self,
        n_pca_components: int = 64,
        layer: int = 26,
        step: int = 1,
    ) -> None:
        self.n_pca_components = n_pca_components
        self.layer = layer
        self.step = step
        self.pca = PCA(n_components=n_pca_components)
        self.lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        self._fitted = False

    # ------------------------------------------------------------------
    # Fit / predict
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FisherProbe":
        """
        Fit PCA then LDA on entropy-matched hidden states.

        Parameters
        ----------
        X : (n_items, hidden_dim) hidden states at self.layer, self.step.
        y : (n_items,) binary labels — 1=CC, 0=CW.
            Pass only entropy-matched items (use EntropyMatcher first).
        """
        n_components = min(self.n_pca_components, X.shape[0] - 1, X.shape[1])
        if n_components != self.n_pca_components:
            self.pca = PCA(n_components=n_components)
        X_pca = self.pca.fit_transform(X)
        self.lda.fit(X_pca, y)
        self._fitted = True
        return self

    def decision_scores(self, X: np.ndarray) -> np.ndarray:
        """Raw LDA decision scores — higher means more CC-like."""
        self._check_fitted()
        return self.lda.decision_function(self.pca.transform(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Hard binary predictions (1=CC, 0=CW)."""
        self._check_fitted()
        return self.lda.predict(self.pca.transform(X))

    def auroc(self, X: np.ndarray, y: np.ndarray) -> float:
        """AUROC on held-out entropy-matched items."""
        return float(roc_auc_score(y, self.decision_scores(X)))

    # ------------------------------------------------------------------
    # Geometry — the detection axis
    # ------------------------------------------------------------------

    @property
    def direction(self) -> np.ndarray:
        """
        Fisher LDA class-mean direction in PCA space, shape (n_pca_components,).

        EXP-H result: centroid-patching along this direction across all layers
        and patch magnitudes λ∈[0.1, 2.0] produces max ΔF1=+0.0004.
        The detection axis ≠ the causal control axis.
        """
        self._check_fitted()
        return self.lda.coef_[0]

    def direction_in_hidden_space(self) -> np.ndarray:
        """
        Fisher direction projected back into the original hidden-state space,
        shape (hidden_dim,). Useful for patching experiments or visualisation.
        """
        self._check_fitted()
        return self.pca.components_.T @ self.lda.coef_[0]

    def class_means(self) -> tuple[np.ndarray, np.ndarray]:
        """
        (mean_CW, mean_CC) in the original hidden-state space.
        Used to construct the centroid-patching vector for replication of EXP-H.
        """
        self._check_fitted()
        mu_pca = self.lda.means_  # shape (2, n_pca_components), order follows lda.classes_
        classes = list(self.lda.classes_)
        idx_cw = classes.index(0)
        idx_cc = classes.index(1)
        mu_cw = self.pca.components_.T @ mu_pca[idx_cw]
        mu_cc = self.pca.components_.T @ mu_pca[idx_cc]
        return mu_cw, mu_cc

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save probe weights to a .npz file."""
        self._check_fitted()
        np.savez(
            path,
            pca_components=self.pca.components_,
            pca_mean=self.pca.mean_,
            pca_explained_variance=self.pca.explained_variance_,
            lda_coef=self.lda.coef_,
            lda_intercept=self.lda.intercept_,
            lda_means=self.lda.means_,
            lda_classes=self.lda.classes_,
            layer=self.layer,
            step=self.step,
            n_pca_components=self.n_pca_components,
        )

    @classmethod
    def load(cls, path: str) -> "FisherProbe":
        """Load a saved probe from a .npz file."""
        d = np.load(path, allow_pickle=False)
        probe = cls(
            n_pca_components=int(d["n_pca_components"]),
            layer=int(d["layer"]),
            step=int(d["step"]),
        )
        probe.pca = PCA(n_components=int(d["n_pca_components"]))
        probe.pca.components_ = d["pca_components"]
        probe.pca.mean_ = d["pca_mean"]
        probe.pca.explained_variance_ = d["pca_explained_variance"]
        probe.lda.coef_ = d["lda_coef"]
        probe.lda.intercept_ = d["lda_intercept"]
        probe.lda.means_ = d["lda_means"]
        probe.lda.classes_ = d["lda_classes"]
        probe._fitted = True
        return probe

    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("FisherProbe is not fitted. Call fit() first.")

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "unfitted"
        return (
            f"FisherProbe(layer={self.layer}, step={self.step}, "
            f"n_pca={self.n_pca_components}, {status})"
        )
