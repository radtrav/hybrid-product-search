import math

import pytest

from reranker.services.feature_extractor import FeatureExtractor
from reranker.services.reranker import Reranker
from reranker.web.api.rerank.schema import Candidate


class TestTokenization:
    """Tests for tokenization functionality."""

    def test_tokenize_lowercase(self) -> None:
        """Test that tokenization converts to lowercase."""
        extractor = FeatureExtractor()
        result = extractor._tokenize("Hello World")
        assert result == ["hello", "world"]

    def test_tokenize_whitespace(self) -> None:
        """Test that tokenization splits on whitespace."""
        extractor = FeatureExtractor()
        result = extractor._tokenize("hello  world\ttab\nline")
        # split() without args splits on any whitespace and removes empty strings
        assert result == ["hello", "world", "tab", "line"]

    def test_tokenize_empty(self) -> None:
        """Test that empty string returns empty list."""
        extractor = FeatureExtractor()
        result = extractor._tokenize("")
        assert result == []


class TestTermFrequency:
    """Tests for term frequency calculation."""

    def test_compute_term_frequency_single(self) -> None:
        """Test term frequency for single term."""
        extractor = FeatureExtractor()
        result = extractor._compute_term_frequency(["hello"])
        assert result == {"hello": 1}

    def test_compute_term_frequency_repeated(self) -> None:
        """Test term frequency with repeated terms."""
        extractor = FeatureExtractor()
        result = extractor._compute_term_frequency(["hello", "world", "hello"])
        assert result == {"hello": 2, "world": 1}

    def test_compute_term_frequency_empty(self) -> None:
        """Test term frequency for empty list."""
        extractor = FeatureExtractor()
        result = extractor._compute_term_frequency([])
        assert result == {}


class TestIDF:
    """Tests for IDF (Inverse Document Frequency) calculation."""

    def test_compute_idf_all_documents(self) -> None:
        """Test IDF when term appears in all documents (should have low IDF)."""
        extractor = FeatureExtractor()

        candidates = [
            Candidate(
                id="1",
                title="common word",
                description="test",
                category="test",
                price=10.0,
                rating=4.0,
                num_reviews=100,
            ),
            Candidate(
                id="2",
                title="common word",
                description="test",
                category="test",
                price=20.0,
                rating=4.5,
                num_reviews=200,
            ),
            Candidate(
                id="3",
                title="common word",
                description="test",
                category="test",
                price=30.0,
                rating=5.0,
                num_reviews=300,
            ),
        ]

        query_terms = {"common"}
        result = extractor._compute_idf(query_terms, candidates)

        # When term is in all documents (df=3, N=3)
        # IDF = log((3 - 3 + 0.5) / (3 + 0.5) + 1) = log(0.5 / 3.5 + 1) ≈ log(1.14) ≈ 0.13
        assert "common" in result
        assert result["common"] < 0.2  # Should be low IDF

    def test_compute_idf_one_document(self) -> None:
        """Test IDF when term appears in only one document (should have high IDF)."""
        extractor = FeatureExtractor()

        candidates = [
            Candidate(
                id="1",
                title="rare word",
                description="test",
                category="test",
                price=10.0,
                rating=4.0,
                num_reviews=100,
            ),
            Candidate(
                id="2",
                title="common test",
                description="test",
                category="test",
                price=20.0,
                rating=4.5,
                num_reviews=200,
            ),
            Candidate(
                id="3",
                title="common test",
                description="test",
                category="test",
                price=30.0,
                rating=5.0,
                num_reviews=300,
            ),
        ]

        query_terms = {"rare"}
        result = extractor._compute_idf(query_terms, candidates)

        # When term is in 1 document (df=1, N=3)
        # IDF = log((3 - 1 + 0.5) / (1 + 0.5) + 1) = log(2.5 / 1.5 + 1) = log(2.667) ≈ 0.98
        assert "rare" in result
        assert result["rare"] > 0.8  # Should be high IDF

    def test_compute_idf_no_documents(self) -> None:
        """Test IDF when term appears in no documents."""
        extractor = FeatureExtractor()

        candidates = [
            Candidate(
                id="1",
                title="test one",
                description="test",
                category="test",
                price=10.0,
                rating=4.0,
                num_reviews=100,
            ),
            Candidate(
                id="2",
                title="test two",
                description="test",
                category="test",
                price=20.0,
                rating=4.5,
                num_reviews=200,
            ),
        ]

        query_terms = {"nonexistent"}
        result = extractor._compute_idf(query_terms, candidates)

        # When term is in 0 documents (df=0, N=2)
        # IDF = log((2 - 0 + 0.5) / (0 + 0.5) + 1) = log(2.5 / 0.5 + 1) = log(6) ≈ 1.79
        assert "nonexistent" in result
        expected_idf = math.log((2 - 0 + 0.5) / (0 + 0.5) + 1)
        assert abs(result["nonexistent"] - expected_idf) < 0.01


class TestBM25Scoring:
    """Tests for BM25 scoring functionality."""

    def test_bm25_exact_match(self) -> None:
        """Test that exact match scores higher than partial match."""
        extractor = FeatureExtractor()

        candidates = [
            Candidate(
                id="1",
                title="wireless headphones",
                description="high quality product",
                category="test",
                price=100.0,
                rating=4.5,
                num_reviews=100,
            ),
            Candidate(
                id="2",
                title="wireless speakers",
                description="high quality product",
                category="test",
                price=100.0,
                rating=4.5,
                num_reviews=100,
            ),
        ]

        query = "wireless headphones"

        score1 = extractor._compute_bm25(candidates[0], query, candidates)
        score2 = extractor._compute_bm25(candidates[1], query, candidates)

        # Exact match should score higher
        assert score1 > score2

    def test_bm25_rare_terms(self) -> None:
        """Test that rare terms are weighted more heavily than common terms."""
        extractor = FeatureExtractor()

        candidates = [
            Candidate(
                id="1",
                title="unique specialized product",
                description="common common common",
                category="test",
                price=100.0,
                rating=4.5,
                num_reviews=100,
            ),
            Candidate(
                id="2",
                title="common common common",
                description="common common common",
                category="test",
                price=100.0,
                rating=4.5,
                num_reviews=100,
            ),
            Candidate(
                id="3",
                title="common common common",
                description="common common common",
                category="test",
                price=100.0,
                rating=4.5,
                num_reviews=100,
            ),
        ]

        # Query with rare term should boost doc with rare term
        rare_score = extractor._compute_bm25(candidates[0], "specialized", candidates)
        common_score = extractor._compute_bm25(candidates[1], "common", candidates)

        # Rare term should have higher contribution due to higher IDF
        assert rare_score > common_score

    def test_bm25_length_normalization(self) -> None:
        """Test that shorter documents with term score higher than longer documents."""
        extractor = FeatureExtractor()

        candidates = [
            Candidate(
                id="1",
                title="headphones",  # Short doc with term
                description="great",
                category="test",
                price=100.0,
                rating=4.5,
                num_reviews=100,
            ),
            Candidate(
                id="2",
                title="headphones",  # Long doc with same term frequency
                description="great product with many features and lots of text to make it longer and longer",
                category="test",
                price=100.0,
                rating=4.5,
                num_reviews=100,
            ),
        ]

        query = "headphones"

        score_short = extractor._compute_bm25(candidates[0], query, candidates)
        score_long = extractor._compute_bm25(candidates[1], query, candidates)

        # Shorter document should score higher due to length normalization
        assert score_short > score_long

    def test_bm25_empty_query(self) -> None:
        """Test that empty query returns 0.0 score."""
        extractor = FeatureExtractor()

        candidate = Candidate(
            id="1",
            title="test product",
            description="test description",
            category="test",
            price=100.0,
            rating=4.5,
            num_reviews=100,
        )

        score = extractor._compute_bm25(candidate, "", [candidate])
        assert score == 0.0

    def test_bm25_empty_document(self) -> None:
        """Test that empty document returns 0.0 score."""
        extractor = FeatureExtractor()

        candidates = [
            Candidate(
                id="1",
                title="",
                description="",
                category="test",
                price=100.0,
                rating=4.5,
                num_reviews=100,
            ),
            Candidate(
                id="2",
                title="test",
                description="test",
                category="test",
                price=100.0,
                rating=4.5,
                num_reviews=100,
            ),
        ]

        score = extractor._compute_bm25(candidates[0], "query", candidates)
        assert score == 0.0

    def test_bm25_fallback(self) -> None:
        """Test fallback to simple overlap when all_candidates is None."""
        extractor = FeatureExtractor()

        candidate = Candidate(
            id="1",
            title="wireless headphones",
            description="premium quality",
            category="test",
            price=100.0,
            rating=4.5,
            num_reviews=100,
        )

        # Should fall back to _compute_text_match
        score = extractor._compute_bm25(candidate, "wireless headphones", None)

        # With simple overlap: query has 2 terms, both match → 2/2 = 1.0
        assert score == 1.0

    def test_bm25_fallback_single_candidate(self) -> None:
        """Test fallback to simple overlap when only one candidate."""
        extractor = FeatureExtractor()

        candidate = Candidate(
            id="1",
            title="wireless headphones",
            description="premium quality",
            category="test",
            price=100.0,
            rating=4.5,
            num_reviews=100,
        )

        # Should fall back when < 2 candidates
        score = extractor._compute_bm25(candidate, "wireless headphones", [candidate])

        # With simple overlap: query has 2 terms, both match → 2/2 = 1.0
        assert score == 1.0


class TestScoreNormalization:
    """Tests for score normalization in Reranker."""

    def test_normalize_scores_different(self) -> None:
        """Test normalization with different scores."""
        reranker = Reranker()
        scores = [1.0, 5.0, 3.0]
        normalized = reranker._normalize_scores(scores)

        # Min-max normalization: (x - min) / (max - min)
        # [1.0, 5.0, 3.0] → [(1-1)/4, (5-1)/4, (3-1)/4] = [0.0, 1.0, 0.5]
        assert normalized == pytest.approx([0.0, 1.0, 0.5])

    def test_normalize_scores_identical(self) -> None:
        """Test normalization when all scores are identical."""
        reranker = Reranker()
        scores = [3.0, 3.0, 3.0]
        normalized = reranker._normalize_scores(scores)

        # All identical → all 0.5
        assert normalized == [0.5, 0.5, 0.5]

    def test_normalize_scores_single(self) -> None:
        """Test normalization with single score."""
        reranker = Reranker()
        scores = [5.0]
        normalized = reranker._normalize_scores(scores)

        # Single score → 0.5
        assert normalized == [0.5]

    def test_normalize_scores_empty(self) -> None:
        """Test normalization with empty list."""
        reranker = Reranker()
        scores = []
        normalized = reranker._normalize_scores(scores)

        # Empty → empty
        assert normalized == []
