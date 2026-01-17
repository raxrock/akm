"""Vector search module."""

from akm.vector.search.hybrid import (
    DocumentStats,
    HybridSearchConfig,
    HybridSearcher,
    compute_bm25_score,
    compute_term_overlap_score,
    hybrid_search,
    linear_combination,
    reciprocal_rank_fusion,
)

__all__ = [
    # Config
    "HybridSearchConfig",
    "DocumentStats",
    # Searcher
    "HybridSearcher",
    # Scoring functions
    "compute_bm25_score",
    "compute_term_overlap_score",
    # Fusion functions
    "reciprocal_rank_fusion",
    "linear_combination",
    # Convenience function
    "hybrid_search",
]
