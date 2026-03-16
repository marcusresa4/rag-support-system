from elasticsearch import AsyncElasticsearch
import os

ES_HOST = os.getenv("ES_HOST", "http://localhost:9200")
INDEX_NAME = "arxiv_chunks"

es = AsyncElasticsearch(ES_HOST)

INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "arxiv_id":       {"type": "keyword"},
            "chunk_index":    {"type": "integer"},
            "content":        {"type": "text", "analyzer": "english"},
            "title":          {"type": "text", "analyzer": "english"},
            "authors":        {"type": "keyword"},
            "published_date": {"type": "date"},
            "chunk_strategy": {"type": "keyword"},
        }
    },
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
    }
}

async def ensure_index():
    exists = await es.indices.exists(index=INDEX_NAME)
    if not exists:
        await es.indices.create(index=INDEX_NAME, body=INDEX_MAPPING)

async def index_chunk(chunk: dict):
    doc = {k: v for k, v in chunk.items() if k != "embedding"}
    await es.index(index=INDEX_NAME, document=doc)

async def bulk_index_chunks(chunks: list[dict]):
    actions = []
    for chunk in chunks:
        doc = {k: v for k, v in chunk.items() if k != "embedding"}
        actions.append({"index": {"_index": INDEX_NAME}})
        actions.append(doc)
    if actions:
        await es.bulk(body=actions)