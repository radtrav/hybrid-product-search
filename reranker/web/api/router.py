from fastapi.routing import APIRouter

from reranker.web.api import docs, monitoring, rerank

api_router = APIRouter()
api_router.include_router(monitoring.router)
api_router.include_router(docs.router)
api_router.include_router(rerank.router, prefix="/rerank", tags=["rerank"])
