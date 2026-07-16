"""
Detection Without Control — core probe implementation.

Paper: "Detection Without Control: Confabulation Geometry Is Dissociable
from Causal Answer Generation in Language Models"

Modules:
    probe           — FisherProbe: PCA(64) → LDA classifier
    oracle          — BilateralOracle: PARAM/CTX_DEP/CO behavioral labeling
    entropy_match   — EntropyMatcher: excluded-zone RC construction
"""

from .probe import FisherProbe
from .oracle import BilateralOracle, label_co
from .entropy_match import EntropyMatcher

__all__ = ["FisherProbe", "BilateralOracle", "label_co", "EntropyMatcher"]
__version__ = "0.1.0"
