from abc import ABC, abstractmethod

from fastapi import Depends

from reranker.db.dao.product_dao import ProductDAO
from reranker.web.api.rerank.schema import Candidate


class BaseRetriever(ABC):
    """Abstract base class for retrievers."""

    @abstractmethod
    async def retrieve(self, candidate_ids: list[str]) -> list[Candidate]:
        """
        Retrieve candidates by their IDs.

        :param candidate_ids: List of candidate IDs to retrieve.
        :return: List of Candidate objects.
        """

    @abstractmethod
    async def retrieve_all(self) -> list[Candidate]:
        """
        Retrieve all available candidates.

        :return: List of all Candidate objects.
        """


class MockRetriever(BaseRetriever):
    """Mock retriever that fetches from SQLite database."""

    def __init__(self, product_dao: ProductDAO = Depends()) -> None:
        self.product_dao = product_dao

    async def retrieve(self, candidate_ids: list[str]) -> list[Candidate]:
        """
        Retrieve candidates from the database.

        :param candidate_ids: List of product IDs.
        :return: List of Candidate objects.
        """
        products = await self.product_dao.get_by_product_ids(candidate_ids)

        return [
            Candidate(
                id=product.product_id,
                title=product.title,
                description=product.description,
                category=product.category,
                price=product.price,
                rating=product.rating,
                num_reviews=product.num_reviews,
                score=None,
            )
            for product in products
        ]

    async def retrieve_all(self) -> list[Candidate]:
        """
        Retrieve all candidates from the database.

        :return: List of all Candidate objects.
        """
        products = await self.product_dao.get_all_products()

        return [
            Candidate(
                id=product.product_id,
                title=product.title,
                description=product.description,
                category=product.category,
                price=product.price,
                rating=product.rating,
                num_reviews=product.num_reviews,
                score=None,
            )
            for product in products
        ]
