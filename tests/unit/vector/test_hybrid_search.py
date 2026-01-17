"""Tests for hybrid search functionality."""

import pytest
from typing import List

from akm.core.models import SearchResult
from akm.vector.search.hybrid import (
    HybridSearchConfig,
    HybridSearcher,
    DocumentStats,
    compute_bm25_score,
    compute_term_overlap_score,
    reciprocal_rank_fusion,
    linear_combination,
    hybrid_search,
    _tokenize,
)


@pytest.fixture
def sample_documents() -> dict:
    """Create sample documents for testing."""
    return {
        "doc1": "The user service handles authentication and user management.",
        "doc2": "The order service processes orders and manages inventory.",
        "doc3": "The authentication module provides JWT token management.",
        "doc4": "Database connections are managed by the connection pool.",
        "doc5": "User authentication requires valid credentials and tokens.",
    }


@pytest.fixture
def sample_search_results() -> List[SearchResult]:
    """Create sample search results."""
    return [
        SearchResult(
            document_id="doc1",
            content="The user service handles authentication.",
            score=0.9,
            metadata={},
        ),
        SearchResult(
            document_id="doc3",
            content="The authentication module provides JWT.",
            score=0.85,
            metadata={},
        ),
        SearchResult(
            document_id="doc5",
            content="User authentication requires credentials.",
            score=0.8,
            metadata={},
        ),
    ]


class TestTokenization:
    """Tests for text tokenization."""

    def test_basic_tokenization(self) -> None:
        """Test basic tokenization."""
        tokens = _tokenize("Hello world, this is a test!")
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens

    def test_stopword_removal(self) -> None:
        """Test stopword removal."""
        tokens = _tokenize("This is a test", remove_stopwords=True)
        assert "test" in tokens
        assert "this" not in tokens
        assert "is" not in tokens
        assert "a" not in tokens

    def test_keep_stopwords(self) -> None:
        """Test keeping stopwords."""
        tokens = _tokenize("This is a test", remove_stopwords=False)
        assert "this" in tokens
        assert "is" in tokens

    def test_lowercase(self) -> None:
        """Test lowercasing."""
        tokens = _tokenize("Hello WORLD", lowercase=True)
        assert "hello" in tokens
        assert "world" in tokens
        assert "Hello" not in tokens


class TestDocumentStats:
    """Tests for document statistics."""

    def test_update_stats(self, sample_documents: dict) -> None:
        """Test updating document statistics."""
        stats = DocumentStats()
        stats.update(list(sample_documents.values()))

        assert stats.total_docs == 5
        assert stats.avg_doc_length > 0
        assert len(stats.term_doc_frequencies) > 0

    def test_term_frequencies(self, sample_documents: dict) -> None:
        """Test term document frequencies."""
        stats = DocumentStats()
        stats.update(list(sample_documents.values()))

        # "authentication" appears in multiple documents
        assert stats.term_doc_frequencies.get("authentication", 0) >= 2

        # "user" appears in multiple documents
        assert stats.term_doc_frequencies.get("user", 0) >= 2


class TestBM25Score:
    """Tests for BM25 scoring."""

    def test_basic_scoring(self, sample_documents: dict) -> None:
        """Test basic BM25 scoring."""
        stats = DocumentStats()
        stats.update(list(sample_documents.values()))
        config = HybridSearchConfig()

        query_terms = _tokenize("user authentication")
        score = compute_bm25_score(
            query_terms,
            sample_documents["doc1"],
            stats,
            config,
        )

        assert score > 0

    def test_relevant_doc_scores_higher(self, sample_documents: dict) -> None:
        """Test that relevant documents score higher."""
        stats = DocumentStats()
        stats.update(list(sample_documents.values()))
        config = HybridSearchConfig()

        query_terms = _tokenize("user authentication")

        # Doc with both terms should score higher
        score1 = compute_bm25_score(query_terms, sample_documents["doc1"], stats, config)
        # Doc without query terms should score lower
        score4 = compute_bm25_score(query_terms, sample_documents["doc4"], stats, config)

        assert score1 > score4

    def test_zero_score_no_match(self, sample_documents: dict) -> None:
        """Test zero score when no terms match."""
        stats = DocumentStats()
        stats.update(list(sample_documents.values()))
        config = HybridSearchConfig()

        query_terms = _tokenize("completely unrelated xyz")
        score = compute_bm25_score(
            query_terms,
            sample_documents["doc1"],
            stats,
            config,
        )

        assert score == 0


class TestTermOverlapScore:
    """Tests for term overlap scoring."""

    def test_full_overlap(self) -> None:
        """Test full term overlap."""
        config = HybridSearchConfig()
        query_terms = ["user", "service"]
        document = "The user service handles requests."

        score = compute_term_overlap_score(query_terms, document, config)
        assert score == 1.0  # All query terms present

    def test_partial_overlap(self) -> None:
        """Test partial term overlap."""
        config = HybridSearchConfig()
        query_terms = ["user", "authentication", "unknown"]
        document = "The user service handles authentication."

        score = compute_term_overlap_score(query_terms, document, config)
        assert 0 < score < 1.0  # Some but not all terms

    def test_no_overlap(self) -> None:
        """Test no term overlap."""
        config = HybridSearchConfig()
        query_terms = ["completely", "different", "terms"]
        document = "The user service handles requests."

        score = compute_term_overlap_score(query_terms, document, config)
        assert score == 0


class TestReciprocalRankFusion:
    """Tests for RRF result fusion."""

    def test_basic_fusion(self, sample_search_results: List[SearchResult]) -> None:
        """Test basic RRF fusion."""
        semantic_results = sample_search_results
        keyword_results = list(reversed(sample_search_results))  # Different order

        fused = reciprocal_rank_fusion(semantic_results, keyword_results)

        assert len(fused) == len(sample_search_results)
        # All original document IDs should be present
        fused_ids = {r.document_id for r in fused}
        original_ids = {r.document_id for r in sample_search_results}
        assert fused_ids == original_ids

    def test_fusion_scores(self, sample_search_results: List[SearchResult]) -> None:
        """Test that fused results have valid scores."""
        fused = reciprocal_rank_fusion(sample_search_results, sample_search_results)

        for result in fused:
            assert result.score > 0
            assert "rrf_score" in result.metadata


class TestLinearCombination:
    """Tests for linear combination fusion."""

    def test_basic_combination(self, sample_search_results: List[SearchResult]) -> None:
        """Test basic linear combination."""
        keyword_scores = {
            "doc1": 0.8,
            "doc3": 0.7,
            "doc5": 0.6,
        }
        config = HybridSearchConfig(semantic_weight=0.5, keyword_weight=0.5)

        combined = linear_combination(sample_search_results, keyword_scores, config)

        assert len(combined) == len(sample_search_results)

    def test_weights_applied(self, sample_search_results: List[SearchResult]) -> None:
        """Test that weights are correctly applied."""
        keyword_scores = {"doc1": 0.5, "doc3": 0.5, "doc5": 0.5}
        config = HybridSearchConfig(semantic_weight=0.7, keyword_weight=0.3)

        combined = linear_combination(sample_search_results, keyword_scores, config)

        # First result should have combined score
        result = combined[0]
        assert "semantic_score" in result.metadata
        assert "keyword_score" in result.metadata


class TestHybridSearcher:
    """Tests for HybridSearcher class."""

    def test_index_documents(self, sample_documents: dict) -> None:
        """Test document indexing."""
        searcher = HybridSearcher()
        searcher.index_documents(sample_documents)

        stats = searcher.get_stats()
        assert stats["indexed_documents"] == 5

    def test_compute_keyword_scores(self, sample_documents: dict) -> None:
        """Test keyword score computation."""
        searcher = HybridSearcher()
        searcher.index_documents(sample_documents)

        scores = searcher.compute_keyword_scores("user authentication")

        assert len(scores) > 0
        # Documents with query terms should have scores
        assert "doc1" in scores or "doc5" in scores

    def test_search(
        self,
        sample_documents: dict,
        sample_search_results: List[SearchResult],
    ) -> None:
        """Test hybrid search."""
        searcher = HybridSearcher()
        searcher.index_documents(sample_documents)

        results = searcher.search(
            "user authentication",
            sample_search_results,
            top_k=3,
        )

        assert len(results) <= 3
        assert all(isinstance(r, SearchResult) for r in results)

    def test_add_remove_document(self, sample_documents: dict) -> None:
        """Test adding and removing documents."""
        searcher = HybridSearcher()
        searcher.index_documents(sample_documents)

        # Add new document
        searcher.add_document("doc6", "New document about testing.")
        assert searcher.get_stats()["indexed_documents"] == 6

        # Remove document
        searcher.remove_document("doc6")
        assert searcher.get_stats()["indexed_documents"] == 5


class TestHybridSearchConvenience:
    """Tests for the convenience hybrid_search function."""

    def test_hybrid_search_function(
        self,
        sample_search_results: List[SearchResult],
    ) -> None:
        """Test the convenience function."""
        results = hybrid_search(
            "user authentication",
            sample_search_results,
            top_k=3,
            alpha=0.7,
        )

        assert len(results) <= 3
        assert all(isinstance(r, SearchResult) for r in results)

    def test_different_alpha_values(
        self,
        sample_search_results: List[SearchResult],
    ) -> None:
        """Test with different alpha values."""
        # High semantic weight
        results_high = hybrid_search(
            "user authentication",
            sample_search_results,
            alpha=0.9,
        )

        # High keyword weight
        results_low = hybrid_search(
            "user authentication",
            sample_search_results,
            alpha=0.1,
        )

        # Both should return results
        assert len(results_high) > 0
        assert len(results_low) > 0
