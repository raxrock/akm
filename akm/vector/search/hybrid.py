"""Hybrid search combining semantic and keyword-based search."""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from akm.core.models import SearchResult

logger = logging.getLogger(__name__)


@dataclass
class HybridSearchConfig:
    """Configuration for hybrid search."""

    # Weight for semantic vs keyword scores
    semantic_weight: float = 0.7
    keyword_weight: float = 0.3

    # Keyword search parameters
    use_bm25: bool = True
    bm25_k1: float = 1.2  # Term frequency saturation parameter
    bm25_b: float = 0.75  # Length normalization parameter

    # Result merging
    reciprocal_rank_constant: int = 60  # For RRF fusion
    min_keyword_score: float = 0.0
    boost_exact_matches: bool = True
    exact_match_boost: float = 0.2

    # Preprocessing
    remove_stopwords: bool = True
    lowercase: bool = True
    stem_terms: bool = False


@dataclass
class DocumentStats:
    """Statistics for BM25 scoring."""

    total_docs: int = 0
    avg_doc_length: float = 0.0
    term_doc_frequencies: Dict[str, int] = field(default_factory=dict)

    def update(self, documents: List[str]) -> None:
        """Update statistics with new documents."""
        self.total_docs = len(documents)
        total_length = 0
        term_docs: Dict[str, Set[int]] = {}

        for idx, doc in enumerate(documents):
            terms = _tokenize(doc)
            total_length += len(terms)

            for term in set(terms):
                if term not in term_docs:
                    term_docs[term] = set()
                term_docs[term].add(idx)

        self.avg_doc_length = total_length / max(self.total_docs, 1)
        self.term_doc_frequencies = {term: len(docs) for term, docs in term_docs.items()}


# Default English stopwords
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "he", "in", "is", "it", "its", "of", "on", "or", "that",
    "the", "to", "was", "were", "will", "with", "this", "but", "they",
    "have", "had", "what", "when", "where", "who", "which", "why", "how",
    "all", "each", "every", "both", "few", "more", "most", "other",
    "some", "such", "no", "not", "only", "same", "so", "than", "too",
    "very", "can", "just", "should", "now"
}


def _tokenize(text: str, remove_stopwords: bool = True, lowercase: bool = True) -> List[str]:
    """
    Tokenize text into terms.

    Args:
        text: Input text
        remove_stopwords: Whether to remove stopwords
        lowercase: Whether to lowercase terms

    Returns:
        List of tokens
    """
    if lowercase:
        text = text.lower()

    # Simple word tokenization
    terms = re.findall(r'\b\w+\b', text)

    if remove_stopwords:
        terms = [t for t in terms if t not in STOPWORDS]

    return terms


def compute_bm25_score(
    query_terms: List[str],
    document: str,
    doc_stats: DocumentStats,
    config: HybridSearchConfig,
) -> float:
    """
    Compute BM25 score for a document given a query.

    BM25 formula:
    score = sum(IDF(term) * (term_freq * (k1 + 1)) / (term_freq + k1 * (1 - b + b * doc_len / avg_doc_len)))

    Args:
        query_terms: Tokenized query terms
        document: Document text
        doc_stats: Document statistics
        config: Hybrid search configuration

    Returns:
        BM25 score
    """
    doc_terms = _tokenize(document, config.remove_stopwords, config.lowercase)
    doc_length = len(doc_terms)
    term_freqs = Counter(doc_terms)

    score = 0.0
    k1 = config.bm25_k1
    b = config.bm25_b
    avg_dl = max(doc_stats.avg_doc_length, 1.0)
    N = max(doc_stats.total_docs, 1)

    for term in query_terms:
        if term not in term_freqs:
            continue

        tf = term_freqs[term]
        df = doc_stats.term_doc_frequencies.get(term, 0)

        # IDF with smoothing
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)

        # BM25 term score
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * doc_length / avg_dl)
        term_score = idf * (numerator / denominator)

        score += term_score

    return score


def compute_term_overlap_score(
    query_terms: List[str],
    document: str,
    config: HybridSearchConfig,
) -> float:
    """
    Compute simple term overlap score.

    Args:
        query_terms: Tokenized query terms
        document: Document text
        config: Hybrid search configuration

    Returns:
        Overlap score (0-1)
    """
    doc_terms = set(_tokenize(document, config.remove_stopwords, config.lowercase))
    query_terms_set = set(query_terms)

    if not query_terms_set:
        return 0.0

    overlap = query_terms_set & doc_terms
    score = len(overlap) / len(query_terms_set)

    # Boost for exact phrase matches
    if config.boost_exact_matches:
        query_text = " ".join(query_terms).lower()
        if query_text in document.lower():
            score = min(1.0, score + config.exact_match_boost)

    return score


def reciprocal_rank_fusion(
    semantic_results: List[SearchResult],
    keyword_results: List[SearchResult],
    k: int = 60,
) -> List[SearchResult]:
    """
    Combine results using Reciprocal Rank Fusion (RRF).

    RRF score = sum(1 / (k + rank_i)) for each ranking list

    Args:
        semantic_results: Results from semantic search
        keyword_results: Results from keyword search
        k: Constant for RRF formula (default 60)

    Returns:
        Fused and re-ranked results
    """
    scores: Dict[str, float] = {}
    result_map: Dict[str, SearchResult] = {}

    # Process semantic results
    for rank, result in enumerate(semantic_results):
        doc_id = result.document_id
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
        if doc_id not in result_map:
            result_map[doc_id] = result

    # Process keyword results
    for rank, result in enumerate(keyword_results):
        doc_id = result.document_id
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
        if doc_id not in result_map:
            result_map[doc_id] = result

    # Sort by RRF score
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    # Build final results with updated scores
    fused_results = []
    for doc_id in sorted_ids:
        result = result_map[doc_id]
        # Create new result with fused score
        fused_results.append(
            SearchResult(
                document_id=result.document_id,
                chunk_id=result.chunk_id,
                content=result.content,
                score=scores[doc_id],
                metadata={**result.metadata, "rrf_score": scores[doc_id]},
                entity_matches=result.entity_matches,
            )
        )

    return fused_results


def linear_combination(
    semantic_results: List[SearchResult],
    keyword_scores: Dict[str, float],
    config: HybridSearchConfig,
) -> List[SearchResult]:
    """
    Combine semantic and keyword scores using linear combination.

    Args:
        semantic_results: Results from semantic search
        keyword_scores: Keyword scores by document ID
        config: Hybrid search configuration

    Returns:
        Results with combined scores
    """
    combined_results = []

    for result in semantic_results:
        doc_id = result.document_id
        semantic_score = result.score
        keyword_score = keyword_scores.get(doc_id, 0.0)

        # Linear combination
        combined_score = (
            config.semantic_weight * semantic_score +
            config.keyword_weight * keyword_score
        )

        combined_results.append(
            SearchResult(
                document_id=result.document_id,
                chunk_id=result.chunk_id,
                content=result.content,
                score=combined_score,
                metadata={
                    **result.metadata,
                    "semantic_score": semantic_score,
                    "keyword_score": keyword_score,
                },
                entity_matches=result.entity_matches,
            )
        )

    # Sort by combined score
    combined_results.sort(key=lambda x: x.score, reverse=True)
    return combined_results


class HybridSearcher:
    """
    Hybrid searcher combining semantic and keyword search.

    This class orchestrates hybrid search by:
    1. Performing semantic search using embeddings
    2. Computing keyword scores (BM25 or term overlap)
    3. Fusing results using RRF or linear combination
    """

    def __init__(
        self,
        config: Optional[HybridSearchConfig] = None,
    ) -> None:
        """
        Initialize the hybrid searcher.

        Args:
            config: Hybrid search configuration
        """
        self.config = config or HybridSearchConfig()
        self._doc_stats = DocumentStats()
        self._documents: Dict[str, str] = {}  # doc_id -> content

    def index_documents(self, documents: Dict[str, str]) -> None:
        """
        Index documents for keyword search.

        Args:
            documents: Dictionary mapping document IDs to content
        """
        self._documents = documents
        self._doc_stats.update(list(documents.values()))
        logger.info(f"Indexed {len(documents)} documents for hybrid search")

    def add_document(self, doc_id: str, content: str) -> None:
        """
        Add a single document to the index.

        Args:
            doc_id: Document ID
            content: Document content
        """
        self._documents[doc_id] = content
        # Rebuild stats (could be optimized for incremental updates)
        self._doc_stats.update(list(self._documents.values()))

    def remove_document(self, doc_id: str) -> bool:
        """
        Remove a document from the index.

        Args:
            doc_id: Document ID

        Returns:
            True if removed, False if not found
        """
        if doc_id in self._documents:
            del self._documents[doc_id]
            self._doc_stats.update(list(self._documents.values()))
            return True
        return False

    def compute_keyword_scores(
        self,
        query: str,
        doc_ids: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Compute keyword scores for documents.

        Args:
            query: Search query
            doc_ids: Optional list of document IDs to score

        Returns:
            Dictionary mapping document IDs to keyword scores
        """
        query_terms = _tokenize(query, self.config.remove_stopwords, self.config.lowercase)

        if not query_terms:
            return {}

        target_docs = doc_ids or list(self._documents.keys())
        scores = {}

        for doc_id in target_docs:
            if doc_id not in self._documents:
                continue

            content = self._documents[doc_id]

            if self.config.use_bm25:
                score = compute_bm25_score(
                    query_terms,
                    content,
                    self._doc_stats,
                    self.config,
                )
            else:
                score = compute_term_overlap_score(
                    query_terms,
                    content,
                    self.config,
                )

            if score >= self.config.min_keyword_score:
                scores[doc_id] = score

        # Normalize scores to 0-1 range
        if scores:
            max_score = max(scores.values())
            if max_score > 0:
                scores = {k: v / max_score for k, v in scores.items()}

        return scores

    def search(
        self,
        query: str,
        semantic_results: List[SearchResult],
        top_k: int = 10,
        use_rrf: bool = True,
    ) -> List[SearchResult]:
        """
        Perform hybrid search.

        Args:
            query: Search query
            semantic_results: Results from semantic search
            top_k: Number of results to return
            use_rrf: Whether to use RRF fusion (vs linear combination)

        Returns:
            Fused search results
        """
        # Get document IDs from semantic results
        doc_ids = [r.document_id for r in semantic_results]

        # Index content from semantic results if not already indexed
        for result in semantic_results:
            if result.document_id not in self._documents:
                self._documents[result.document_id] = result.content

        # Update stats if we added new documents
        if self._doc_stats.total_docs != len(self._documents):
            self._doc_stats.update(list(self._documents.values()))

        # Compute keyword scores
        keyword_scores = self.compute_keyword_scores(query, doc_ids)

        if use_rrf:
            # Create keyword results for RRF
            keyword_results = []
            sorted_by_keyword = sorted(
                [(doc_id, score) for doc_id, score in keyword_scores.items()],
                key=lambda x: x[1],
                reverse=True,
            )

            for doc_id, score in sorted_by_keyword:
                # Find matching semantic result
                matching = next((r for r in semantic_results if r.document_id == doc_id), None)
                if matching:
                    keyword_results.append(
                        SearchResult(
                            document_id=doc_id,
                            chunk_id=matching.chunk_id,
                            content=matching.content,
                            score=score,
                            metadata=matching.metadata,
                            entity_matches=matching.entity_matches,
                        )
                    )

            fused = reciprocal_rank_fusion(
                semantic_results,
                keyword_results,
                k=self.config.reciprocal_rank_constant,
            )
        else:
            fused = linear_combination(
                semantic_results,
                keyword_scores,
                self.config,
            )

        return fused[:top_k]

    def get_stats(self) -> Dict[str, Any]:
        """Get searcher statistics."""
        return {
            "indexed_documents": len(self._documents),
            "total_docs": self._doc_stats.total_docs,
            "avg_doc_length": self._doc_stats.avg_doc_length,
            "unique_terms": len(self._doc_stats.term_doc_frequencies),
            "config": {
                "semantic_weight": self.config.semantic_weight,
                "keyword_weight": self.config.keyword_weight,
                "use_bm25": self.config.use_bm25,
            },
        }


# Convenience function for one-shot hybrid search
def hybrid_search(
    query: str,
    semantic_results: List[SearchResult],
    top_k: int = 10,
    alpha: float = 0.7,
    use_bm25: bool = True,
) -> List[SearchResult]:
    """
    Perform hybrid search with default configuration.

    Args:
        query: Search query
        semantic_results: Results from semantic search
        top_k: Number of results to return
        alpha: Weight for semantic score (1-alpha for keyword)
        use_bm25: Whether to use BM25 for keyword scoring

    Returns:
        Fused search results
    """
    config = HybridSearchConfig(
        semantic_weight=alpha,
        keyword_weight=1 - alpha,
        use_bm25=use_bm25,
    )

    searcher = HybridSearcher(config)
    return searcher.search(query, semantic_results, top_k)
