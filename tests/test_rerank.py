import pytest
from fastapi import FastAPI
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession
from starlette import status

from reranker.db.dao.product_dao import ProductDAO
from reranker.services.feature_extractor import FeatureExtractor
from reranker.services.reranker import Reranker
from reranker.web.api.rerank.schema import Candidate


@pytest.fixture
async def sample_products(dbsession: AsyncSession) -> list[str]:
    """Create sample products in database for testing."""
    dao = ProductDAO(dbsession)

    products = [
        {
            "product_id": "test_prod_1",
            "title": "Wireless Headphones Premium",
            "description": "High quality wireless headphones with noise cancellation",
            "category": "Electronics",
            "price": 199.99,
            "rating": 4.5,
            "num_reviews": 1000,
        },
        {
            "product_id": "test_prod_2",
            "title": "Cheap Earbuds",
            "description": "Basic earbuds for everyday use",
            "category": "Electronics",
            "price": 15.99,
            "rating": 3.5,
            "num_reviews": 50,
        },
        {
            "product_id": "test_prod_3",
            "title": "Studio Headphones Professional",
            "description": "Professional studio headphones for audio production",
            "category": "Electronics",
            "price": 299.99,
            "rating": 4.8,
            "num_reviews": 500,
        },
    ]

    await dao.bulk_create_products(products)
    await dbsession.commit()

    return ["test_prod_1", "test_prod_2", "test_prod_3"]


async def test_rerank_endpoint(
    fastapi_app: FastAPI,
    client: AsyncClient,
    sample_products: list[str],
) -> None:
    """Test the rerank endpoint."""
    url = fastapi_app.url_path_for("rerank_candidates")

    request_data = {
        "query": "wireless headphones",
        "candidate_ids": sample_products,
    }

    response = await client.post(url, json=request_data)
    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert data["query"] == "wireless headphones"
    assert data["count"] == 3
    assert len(data["candidates"]) == 3

    # Check that all candidates have scores
    for candidate in data["candidates"]:
        assert "score" in candidate
        assert candidate["score"] is not None
        assert 0 <= candidate["score"] <= 1


async def test_rerank_with_custom_weights(
    fastapi_app: FastAPI,
    client: AsyncClient,
    sample_products: list[str],
) -> None:
    """Test reranking with custom weights."""
    url = fastapi_app.url_path_for("rerank_candidates")

    request_data = {
        "query": "headphones",
        "candidate_ids": sample_products,
        "weights": {
            "text_match": 0.1,
            "price": 0.7,  # Heavily weight price
            "rating": 0.1,
            "popularity": 0.1,
        },
    }

    response = await client.post(url, json=request_data)
    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    candidates = data["candidates"]

    # With price heavily weighted, cheaper items should rank higher
    # test_prod_2 (15.99) should rank higher than test_prod_3 (299.99)
    prices = [c["price"] for c in candidates]
    # First item should be cheaper than last
    assert prices[0] < prices[-1]


async def test_rerank_not_found(
    fastapi_app: FastAPI,
    client: AsyncClient,
) -> None:
    """Test reranking with non-existent product IDs."""
    url = fastapi_app.url_path_for("rerank_candidates")

    request_data = {
        "query": "test",
        "candidate_ids": ["nonexistent_1", "nonexistent_2"],
    }

    response = await client.post(url, json=request_data)
    assert response.status_code == status.HTTP_404_NOT_FOUND


def test_feature_extractor() -> None:
    """Test feature extraction."""
    extractor = FeatureExtractor()

    candidate = Candidate(
        id="test",
        title="Wireless Headphones",
        description="Premium quality wireless headphones",
        category="Electronics",
        price=99.99,
        rating=4.5,
        num_reviews=100,
    )

    features = extractor.extract_features(candidate, "wireless headphones")

    assert "text_match" in features
    assert "price" in features
    assert "rating" in features
    assert "popularity" in features

    # All features should be normalized to [0, 1]
    for value in features.values():
        assert 0 <= value <= 1

    # Text match should be high for exact match
    assert features["text_match"] > 0.5


def test_reranker_scoring() -> None:
    """Test reranker scoring logic."""
    reranker = Reranker()

    candidates = [
        Candidate(
            id="1",
            title="Good Match Expensive",
            description="Perfect match",
            category="Test",
            price=500.0,
            rating=5.0,
            num_reviews=1000,
        ),
        Candidate(
            id="2",
            title="Good Match Cheap",
            description="Perfect match",
            category="Test",
            price=10.0,
            rating=5.0,
            num_reviews=1000,
        ),
    ]

    reranked = reranker.rerank(candidates, "good match", weights=None)

    # Both should have scores
    assert all(c.score is not None for c in reranked)

    # Results should be sorted by score
    scores = [c.score for c in reranked if c.score is not None]
    assert scores == sorted(scores, reverse=True)
