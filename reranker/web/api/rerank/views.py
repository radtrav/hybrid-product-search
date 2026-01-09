from fastapi import APIRouter, Depends, HTTPException, status

from reranker.services.reranker import Reranker
from reranker.services.retriever import MockRetriever
from reranker.settings import settings
from reranker.web.api.rerank.schema import RerankRequest, RerankResponse

router = APIRouter()


def get_reranker() -> Reranker:
    """
    Dependency to get reranker instance.

    :return: Reranker instance with default weights from settings.
    """
    return Reranker(default_weights=settings.rerank_default_weights)


@router.post("/", response_model=RerankResponse)
async def rerank_candidates(
    request: RerankRequest,
    retriever: MockRetriever = Depends(),
    reranker: Reranker = Depends(get_reranker),
) -> RerankResponse:
    """
    Rerank candidates based on query and features.

    :param request: Rerank request with query and candidate IDs.
    :param retriever: Retriever dependency for fetching candidates.
    :param reranker: Reranker dependency for scoring.
    :return: Reranked candidates with scores.
    """
    # Retrieve candidates
    candidates = await retriever.retrieve(request.candidate_ids)

    if not candidates:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No candidates found for IDs: {request.candidate_ids}",
        )

    # Rerank
    reranked = reranker.rerank(
        candidates=candidates,
        query=request.query,
        weights=request.weights,
    )

    return RerankResponse(
        query=request.query,
        candidates=reranked,
        count=len(reranked),
    )
