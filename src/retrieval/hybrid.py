# src/retrieval/hybrid.py
from src.storage.elasticsearch_store import bm25_search
from src.storage.pgvector_store import dense_search
from src.embeddings.embedder import embed_single


def reciprocal_rank_fusion(
    bm25_results: list[dict],
    dense_results: list[dict],
    k: int = 60,
) -> list[dict]:
    """
    RRF formula: score(d) = Σ 1 / (k + rank(d))
    k=60 is the standard constant from the original RRF paper.
    It dampens the impact of very high ranks.
    """
    scores: dict[str, float] = {}
    docs: dict[str, dict] = {}

    for rank, result in enumerate(bm25_results):
        doc_id = f"{result['arxiv_id']}_{result['chunk_index']}"
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
        docs[doc_id] = result

    for rank, result in enumerate(dense_results):
        doc_id = f"{result['arxiv_id']}_{result['chunk_index']}"
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
        docs[doc_id] = result

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return [
        {**docs[doc_id], "score": rrf_score, "strategy": "hybrid"}
        for doc_id, rrf_score in ranked
    ]


async def hybrid_search(query: str, k: int = 5) -> list[dict]:
    embedding = embed_single(query)

    bm25_results = await bm25_search(query, k=k)
    dense_results = await dense_search(embedding, k=k)

    return reciprocal_rank_fusion(bm25_results, dense_results)[:k]