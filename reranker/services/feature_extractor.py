import math

from reranker.web.api.rerank.schema import Candidate


class FeatureExtractor:
    """Extract features from candidates for reranking."""

    def extract_features(
        self,
        candidate: Candidate,
        query: str,
        all_candidates: list[Candidate] | None = None,
    ) -> dict[str, float]:
        """
        Extract features for a single candidate.

        :param candidate: Candidate to extract features from.
        :param query: User query for text matching.
        :param all_candidates: All candidates for corpus-level statistics (optional).
        :return: Dictionary of feature names to values.
        """
        return {
            "text_match": self._compute_bm25(candidate, query, all_candidates),
            "price": self._normalize_price(candidate.price),
            "rating": self._normalize_rating(candidate.rating),
            "popularity": self._normalize_popularity(candidate.num_reviews),
        }

    def _tokenize(self, text: str) -> list[str]:
        """
        Tokenize text into lowercase terms.

        :param text: Text to tokenize.
        :return: List of lowercase terms.
        """
        return text.lower().split()

    def _compute_term_frequency(self, terms: list[str]) -> dict[str, int]:
        """
        Count term occurrences.

        :param terms: List of terms.
        :return: Dictionary of term to frequency.
        """
        freq = {}
        for term in terms:
            freq[term] = freq.get(term, 0) + 1
        return freq

    def _compute_idf(
        self,
        query_terms: set[str],
        all_candidates: list[Candidate],
    ) -> dict[str, float]:
        """
        Compute IDF for query terms across corpus.

        :param query_terms: Set of query terms.
        :param all_candidates: All candidates in the corpus.
        :return: Dictionary of term to IDF value.
        """
        n_candidates = len(all_candidates)
        doc_freq = dict.fromkeys(query_terms, 0)

        for candidate in all_candidates:
            doc_text = f"{candidate.title} {candidate.description}"
            doc_terms = set(self._tokenize(doc_text))
            for term in query_terms:
                if term in doc_terms:
                    doc_freq[term] += 1

        idf = {}
        for term, df in doc_freq.items():
            idf[term] = math.log((n_candidates - df + 0.5) / (df + 0.5) + 1)

        return idf

    def _compute_bm25(
        self,
        candidate: Candidate,
        query: str,
        all_candidates: list[Candidate] | None = None,
    ) -> float:
        """
        Compute BM25 score with k1=1.5, b=0.75.

        :param candidate: Candidate to score.
        :param query: Search query.
        :param all_candidates: All candidates for corpus statistics (optional).
        :return: BM25 score.
        """
        # Fallback if no corpus provided
        if all_candidates is None or len(all_candidates) < 2:
            return self._compute_text_match(candidate, query)

        k1 = 1.5
        b = 0.75

        # Tokenize query
        query_terms = self._tokenize(query)
        if not query_terms:
            return 0.0

        # Build corpus and compute avgdl
        corpus = []
        for cand in all_candidates:
            doc_text = f"{cand.title} {cand.description}"
            corpus.append(self._tokenize(doc_text))

        avgdl = sum(len(doc) for doc in corpus) / len(corpus)

        # Tokenize current document
        doc_text = f"{candidate.title} {candidate.description}"
        doc_terms = self._tokenize(doc_text)
        doc_len = len(doc_terms)

        if doc_len == 0:
            return 0.0

        # Compute term frequencies
        term_freq = self._compute_term_frequency(doc_terms)

        # Compute IDF
        idf = self._compute_idf(set(query_terms), all_candidates)

        # Apply BM25 formula
        score = 0.0
        for term in query_terms:
            if term in term_freq:
                tf = term_freq[term]
                idf_val = idf.get(term, 0.0)
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * (doc_len / avgdl))
                score += idf_val * (numerator / denominator)

        return score

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
