"""Result fusion strategies."""

from rag.retrieve.fusion.rrf import RRFFusion
from rag.retrieve.fusion.weighted import WeightedFusion

__all__ = ["RRFFusion", "WeightedFusion"]
