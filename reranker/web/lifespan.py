import json
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from reranker.db.dao.product_dao import ProductDAO
from reranker.db.meta import meta
from reranker.db.models import load_all_models
from reranker.settings import settings


def _setup_db(app: FastAPI) -> None:  # pragma: no cover
    """
    Creates connection to the database.

    This function creates SQLAlchemy engine instance,
    session_factory for creating sessions
    and stores them in the application's state property.

    :param app: fastAPI application.
    """
    engine = create_async_engine(str(settings.db_url), echo=settings.db_echo)
    session_factory = async_sessionmaker(
        engine,
        expire_on_commit=False,
    )
    app.state.db_engine = engine
    app.state.db_session_factory = session_factory


async def _create_tables() -> None:  # pragma: no cover
    """Populates tables in the database."""
    load_all_models()
    engine = create_async_engine(str(settings.db_url))
    async with engine.begin() as connection:
        await connection.run_sync(meta.create_all)
    await engine.dispose()


async def _load_mock_products() -> None:  # pragma: no cover
    """Load mock products into database from JSON file."""

    if not settings.mock_products_file.exists():
        return

    engine = create_async_engine(str(settings.db_url))
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async with session_factory() as session:
        dao = ProductDAO(session)

        # Check if products already exist
        existing = await dao.get_all_products()
        if existing:
            await engine.dispose()
            return

        # Load and insert products
        with Path.open(settings.mock_products_file) as f:
            products = json.load(f)

        await dao.bulk_create_products(products)
        await session.commit()

    await engine.dispose()


@asynccontextmanager
async def lifespan_setup(
    app: FastAPI,
) -> AsyncGenerator[None]:  # pragma: no cover
    """
    Actions to run on application startup.

    This function uses fastAPI app to store data
    in the state, such as db_engine.

    :param app: the fastAPI application.
    :return: function that actually performs actions.
    """

    app.middleware_stack = None
    _setup_db(app)
    await _create_tables()
    await _load_mock_products()
    app.middleware_stack = app.build_middleware_stack()

    yield
    await app.state.db_engine.dispose()
