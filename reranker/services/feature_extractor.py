import math

from reranker.web.api.rerank.schema import Candidate


class FeatureExtractor:
    """Extract features from candidates for reranking."""

    def extract_features(self, candidate: Candidate, query: str) -> dict[str, float]:
        """
        Extract features for a single candidate.

        :param candidate: Candidate to extract features from.
        :param query: User query for text matching.
        :return: Dictionary of feature names to values.
        """
        return {
            "text_match": self._compute_text_match(candidate, query),
            "price": self._normalize_price(candidate.price),
            "rating": self._normalize_rating(candidate.rating),
            "popularity": self._normalize_popularity(candidate.num_reviews),
        }
    # TODO change to BM30
    def _compute_text_match(self, candidate: Candidate, query: str) -> float:
        """
        Compute text matching score (simple overlap).

        :param candidate: Candidate item.
        :param query: Search query.
        :return: Text match score between 0 and 1.
        """
        query_terms = set(query.lower().split())
        title_terms = set(candidate.title.lower().split())
        desc_terms = set(candidate.description.lower().split())

        all_terms = title_terms.union(desc_terms)

        if not query_terms or not all_terms:
            return 0.0

        overlap = len(query_terms.intersection(all_terms))
        return overlap / len(query_terms)

    def _normalize_price(self, price: float) -> float:
        """
        Normalize price (inverse: lower price = higher score).

        :param price: Product price.
        :return: Normalized price score between 0 and 1.
        """
        # Using log to handle wide price ranges
        # Lower price = higher score
        if price <= 0:
            return 1.0

        # Assume max reasonable price is 1000
        max_price = 1000.0
        normalized = 1.0 - (math.log(price + 1) / math.log(max_price + 1))
        return max(0.0, min(1.0, normalized))

    def _normalize_rating(self, rating: float) -> float:
        """
        Normalize rating to 0-1 scale.

        :param rating: Rating value (0-5).
        :return: Normalized rating between 0 and 1.
        """
        return rating / 5.0

    def _normalize_popularity(self, num_reviews: int) -> float:
        """
        Normalize number of reviews using log scale.

        :param num_reviews: Number of reviews.
        :return: Normalized popularity score between 0 and 1.
        """
        if num_reviews <= 0:
            return 0.0

        # Log scale for popularity
        max_reviews = 10000.0
        normalized = math.log(num_reviews + 1) / math.log(max_reviews + 1)
        return min(1.0, normalized)
