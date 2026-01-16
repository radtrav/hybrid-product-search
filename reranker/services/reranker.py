from reranker.services.feature_extractor import FeatureExtractor
from reranker.web.api.rerank.schema import Candidate


class Reranker:
    """Reranker service with weighted feature scoring."""

    def __init__(self, default_weights: dict[str, float] | None = None) -> None:
        """
        Initialize reranker with default weights.

        :param default_weights: Default feature weights.
        """
        self.feature_extractor = FeatureExtractor()
        self.default_weights = default_weights or {
            "text_match": 0.4,
            "price": 0.2,
            "rating": 0.2,
            "popularity": 0.2,
        }

    def rerank(
        self,
        candidates: list[Candidate],
        query: str,
        weights: dict[str, float] | None = None,
    ) -> list[Candidate]:
        """
        Rerank candidates using weighted feature scoring with BM25.

        :param candidates: List of candidates to rerank.
        :param query: User search query.
        :param weights: Optional custom weights (overrides defaults).
        :return: Sorted list of candidates with scores.
        """
        active_weights = weights or self.default_weights

        # Normalize weights to sum to 1.0
        total_weight = sum(active_weights.values())
        normalized_weights = {k: v / total_weight for k, v in active_weights.items()}

        # First pass: extract raw features (with raw BM25 scores)
        all_features = []
        for candidate in candidates:
            features = self.feature_extractor.extract_features(
                candidate, query, all_candidates=candidates
            )
            all_features.append(features)

        # Normalize BM25 scores
        bm25_scores = [f["text_match"] for f in all_features]
        normalized_bm25 = self._normalize_scores(bm25_scores)

        # Update features with normalized BM25
        for features, norm_bm25 in zip(all_features, normalized_bm25, strict=False):
            features["text_match"] = norm_bm25

        # Score each candidate with normalized features
        scored_candidates = []
        for candidate, features in zip(candidates, all_features, strict=False):
            score = sum(
                features.get(feature_name, 0.0) * weight
                for feature_name, weight in normalized_weights.items()
            )

            # Create new candidate with score
            candidate.score = round(score, 4)
            scored_candidates.append(candidate)

        # Sort by score descending
        scored_candidates.sort(key=lambda c: c.score or 0.0, reverse=True)

        return scored_candidates

    def _normalize_scores(self, scores: list[float]) -> list[float]:
        """
        Min-max normalize scores to [0, 1].

        :param scores: List of scores to normalize.
        :return: Normalized scores.
        """
        if not scores:
            return []

        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            return [0.5] * len(scores)

        return [(score - min_score) / (max_score - min_score) for score in scores]
