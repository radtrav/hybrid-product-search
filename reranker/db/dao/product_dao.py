from typing import Any

from fastapi import Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from reranker.db.dependencies import get_db_session
from reranker.db.models.product_model import ProductModel


class ProductDAO:
    """Class for accessing product table."""

    def __init__(self, session: AsyncSession = Depends(get_db_session)) -> None:
        self.session = session

    async def create_product(
        self,
        product_id: str,
        title: str,
        description: str,
        category: str,
        price: float,
        rating: float,
        num_reviews: int,
    ) -> ProductModel:
        """
        Add single product to session.

        :param product_id: Unique product identifier.
        :param title: Product title.
        :param description: Product description.
        :param category: Product category.
        :param price: Product price.
        :param rating: Product rating (0-5).
        :param num_reviews: Number of reviews.
        :return: Created product model.
        """
        product = ProductModel(
            product_id=product_id,
            title=title,
            description=description,
            category=category,
            price=price,
            rating=rating,
            num_reviews=num_reviews,
        )
        self.session.add(product)
        await self.session.flush()
        return product

    async def get_all_products(self) -> list[ProductModel]:
        """
        Get all products from the database.

        :return: List of all products.
        """
        raw_products = await self.session.execute(select(ProductModel))
        return list(raw_products.scalars().fetchall())

    async def get_by_product_ids(self, product_ids: list[str]) -> list[ProductModel]:
        """
        Get products by their product IDs.

        :param product_ids: List of product IDs to fetch.
        :return: List of product models.
        """
        result = await self.session.execute(
            select(ProductModel).where(ProductModel.product_id.in_(product_ids)),
        )
        return list(result.scalars().fetchall())

    async def bulk_create_products(self, products: list[dict[str, Any]]) -> None:
        """
        Bulk insert products from list of dictionaries.

        :param products: List of product data dictionaries.
        """
        for product_data in products:
            await self.create_product(**product_data)
