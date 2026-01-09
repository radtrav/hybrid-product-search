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
        Rerank candidates using weighted feature scoring.

        :param candidates: List of candidates to rerank.
        :param query: User search query.
        :param weights: Optional custom weights (overrides defaults).
        :return: Sorted list of candidates with scores.
        """
        active_weights = weights or self.default_weights

        # Normalize weights to sum to 1.0
        total_weight = sum(active_weights.values())
        normalized_weights = {k: v / total_weight for k, v in active_weights.items()}

        # Score each candidate
        scored_candidates = []
        for candidate in candidates:
            features = self.feature_extractor.extract_features(candidate, query)
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
