import os
import anthropic
from dotenv import load_dotenv
from src.storage.elasticsearch_store import bm25_search
from src.storage.pgvector_store import dense_search
from src.embeddings.embedder import embed_single

load_dotenv()


def expand_query(query: str) -> list[str]:
    """Use LLM to generate alternative query formulations."""
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        messages=[
            {
                "role": "user",
                "content": f"""Generate 2 alternative search queries for the following question about scientific literature.
Return ONLY the queries, one per line, no numbering, no explanation.

Original query: {query}

Alternative queries:"""
            }
        ]
    )

    alternatives = message.content[0].text.strip().split("\n")
    alternatives = [q.strip() for q in alternatives if q.strip()]
    return [query] + alternatives[:2]


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
    """Hybrid search without query expansion."""
    embedding = embed_single(query)
    bm25_results = await bm25_search(query, k=k)
    dense_results = await dense_search(embedding, k=k)
    return reciprocal_rank_fusion(bm25_results, dense_results)[:k]


async def hybrid_search_with_expansion(query: str, k: int = 5) -> list[dict]:
    """Hybrid search with LLM query expansion."""
    queries = expand_query(query)

    all_bm25 = []
    all_dense = []

    for q in queries:
        embedding = embed_single(q)
        bm25_results = await bm25_search(q, k=k)
        dense_results = await dense_search(embedding, k=k)
        all_bm25.extend(bm25_results)
        all_dense.extend(dense_results)

    return reciprocal_rank_fusion(all_bm25, all_dense)[:k]