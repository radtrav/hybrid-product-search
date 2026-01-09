from pydantic import BaseModel, ConfigDict, Field


class Candidate(BaseModel):
    """Candidate item for reranking."""

    id: str = Field(..., description="Unique identifier for the candidate")
    title: str = Field(..., description="Title of the item")
    description: str = Field(..., description="Description of the item")
    category: str = Field(..., description="Category of the item")
    price: float = Field(..., ge=0, description="Price of the item")
    rating: float = Field(..., ge=0, le=5, description="Rating (0-5)")
    num_reviews: int = Field(..., ge=0, description="Number of reviews")
    score: float | None = Field(default=None, description="Computed reranking score")


class RerankRequest(BaseModel):
    """Request for reranking candidates."""

    query: str = Field(..., min_length=1, description="Search query")
    candidate_ids: list[str] = Field(
        ...,
        min_length=1,
        description="List of candidate IDs to rerank",
    )
    weights: dict[str, float] | None = Field(
        None,
        description="Optional custom weights for features",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "wireless headphones",
                "candidate_ids": ["prod_1", "prod_2", "prod_3"],
                "weights": {"text_match": 0.5, "price": 0.2, "rating": 0.3},
            },
        },
    )


class RerankResponse(BaseModel):
    """Response containing reranked candidates."""

    query: str = Field(..., description="Original query")
    candidates: list[Candidate] = Field(
        ...,
        description="Reranked candidates with scores",
    )
    count: int = Field(..., description="Number of candidates returned")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "wireless headphones",
                "candidates": [
                    {
                        "id": "prod_1",
                        "title": "Premium Wireless Headphones",
                        "description": "High quality...",
                        "category": "Electronics",
                        "price": 199.99,
                        "rating": 4.5,
                        "num_reviews": 1024,
                        "score": 0.89,
                    },
                ],
                "count": 1,
            },
        },
    )


class SearchRequest(BaseModel):
    """Request for searching and getting best matches."""

    query: str = Field(..., min_length=1, description="Search query")
    top_k: int = Field(
        default=10,
        ge=1,
        le=20,
        description="Number of top results to return",
    )
    weights: dict[str, float] | None = Field(
        None,
        description="Optional custom weights for features",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "wireless headphones",
                "top_k": 3,
                "weights": {"text_match": 0.5, "price": 0.2, "rating": 0.3},
            },
        },
    )


class SearchResponse(BaseModel):
    """Response containing best matching candidates."""

    query: str = Field(..., description="Original query")
    results: list[Candidate] = Field(
        ...,
        description="Top matching candidates with scores",
    )
    count: int = Field(..., description="Number of results returned")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "wireless headphones",
                "results": [
                    {
                        "id": "prod_1",
                        "title": "Wireless Bluetooth Headphones",
                        "description": "Premium over-ear headphones...",
                        "category": "Electronics",
                        "price": 199.99,
                        "rating": 4.5,
                        "num_reviews": 1024,
                        "score": 0.89,
                    },
                ],
                "count": 1,
            },
        },
    )
