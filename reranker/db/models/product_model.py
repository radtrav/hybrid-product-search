from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql.sqltypes import Float, Integer, String, Text

from reranker.db.base import Base


class ProductModel(Base):
    """Model for product data used in reranking."""

    __tablename__ = "products"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    product_id: Mapped[str] = mapped_column(
        String(length=100),
        unique=True,
        index=True,
    )
    title: Mapped[str] = mapped_column(String(length=500))
    description: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(length=100))
    price: Mapped[float] = mapped_column(Float)
    rating: Mapped[float] = mapped_column(Float)
    num_reviews: Mapped[int] = mapped_column(Integer)
