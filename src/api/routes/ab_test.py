from fastapi import APIRouter, Query
from src.retrieval.hybrid import hybrid_search, hybrid_search_with_expansion
from src.storage.elasticsearch_store import bm25_search
from src.embeddings.embedder import embed_single
import structlog
import hashlib

logger = structlog.get_logger()
router = APIRouter(tags=["ab_test"])


def assign_variant(query: str) -> str:
    """Deterministically assign a query to variant A or B based on hash."""
    hash_val = int(hashlib.md5(query.encode()).hexdigest(), 16)
    return "A" if hash_val % 2 == 0 else "B"


@router.get("/search/ab")
async def ab_search(
    q: str = Query(..., description="Search query"),
    k: int = Query(5, description="Number of results to return"),
):
    """
    A/B test endpoint:
    - Variant A: standard hybrid search (BM25 + dense + RRF)
    - Variant B: hybrid search with query expansion
    """
    variant = assign_variant(q)

    if variant == "A":
        results = await hybrid_search(q, k=k)
        strategy = "hybrid"
    else:
        results = await hybrid_search_with_expansion(q, k=k)
        strategy = "hybrid_expanded"

    logger.info(
        "ab_test_request",
        query=q,
        variant=variant,
        strategy=strategy,
        results_count=len(results),
    )

    return {
        "query": q,
        "variant": variant,
        "strategy": strategy,
        "total": len(results),
        "results": results,
    }