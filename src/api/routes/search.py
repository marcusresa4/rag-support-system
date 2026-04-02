from fastapi import APIRouter, Query
from src.retrieval.hybrid import hybrid_search, hybrid_search_with_expansion
from src.storage.elasticsearch_store import bm25_search
from src.storage.pgvector_store import dense_search
from src.embeddings.embedder import embed_single
import structlog
import redis
import json
import os

logger = structlog.get_logger()
router = APIRouter(tags=["search"])

redis_client = redis.Redis.from_url(
    os.getenv("REDIS_URL", "redis://localhost:6379"),
    decode_responses=True
)

CACHE_TTL = 3600  # 1 hour

@router.get("/search")
async def search(
    q: str = Query(..., description="Search query"),
    strategy: str = Query("hybrid", description="bm25 | dense | hybrid | hybrid_expanded"),
    k: int = Query(5, description="Number of results to return"),
    arxiv_id: str = Query(None, description="Filter by arxiv ID"),
):
    cache_key = f"search:{strategy}:{k}:{arxiv_id}:{q}"

    cached = redis_client.get(cache_key)
    if cached:
        logger.info("cache_hit", query=q, strategy=strategy)
        result = json.loads(cached)
        result["cached"] = True
        return result

    logger.info("cache_miss", query=q, strategy=strategy)

    if strategy == "bm25":
        results = await bm25_search(q, k=k, arxiv_id=arxiv_id)
    elif strategy == "dense":
        embedding = embed_single(q)
        results = await dense_search(embedding, k=k)
    elif strategy == "hybrid_expanded":
        results = await hybrid_search_with_expansion(q, k=k)
    else:
        results = await hybrid_search(q, k=k)

    response = {
        "query": q,
        "strategy": strategy,
        "total": len(results),
        "results": results,
        "cached": False,
    }

    redis_client.setex(cache_key, CACHE_TTL, json.dumps(response))

    return response