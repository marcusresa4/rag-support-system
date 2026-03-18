from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.ingestion.arxiv_fetcher import fetch_paper_by_id
from src.ingestion.chunker import fixed_size_chunks, sentence_chunks
from src.embeddings.embedder import embed_texts
from src.storage.pgvector_store import insert_chunks
from src.storage.elasticsearch_store import bulk_index_chunks
import structlog

logger = structlog.get_logger()
router = APIRouter(tags=["ingestion"])

class IngestRequest(BaseModel):
    arxiv_id: str
    strategy: str = "sentence"
    chunk_size: int = 512
    overlap: int = 51

@router.post("/ingest")
async def ingest_paper(request: IngestRequest):
    try:
        paper = fetch_paper_by_id(request.arxiv_id)

        text = f"{paper.title}\n\n{paper.abstract}"
        if request.strategy == "fixed":
            chunks = fixed_size_chunks(text, chunk_size=request.chunk_size, overlap=request.overlap)
        else:
            chunks = sentence_chunks(text, target_size=request.chunk_size)
            
        contents = [c.content for c in chunks]
        embeddings = embed_texts(contents)

        records = []
        for chunk, embedding in zip(chunks, embeddings):
            records.append({
                "arxiv_id": paper.arxiv_id,
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
                "title": paper.title,
                "authors": ", ".join(paper.authors[:3]),
                "published_date": paper.published_date.replace(tzinfo=None),
                "chunk_strategy": chunk.strategy,
                "embedding": embedding,
            })

        await insert_chunks(records)
        await bulk_index_chunks(records)

        logger.info("ingestion_complete", arxiv_id=request.arxiv_id, chunks=len(records))

        return {
            "status": "success",
            "arxiv_id": request.arxiv_id,
            "title": paper.title,
            "chunks_created": len(records),
            "strategy": request.strategy,
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("ingestion_failed", error=str(e), arxiv_id=request.arxiv_id)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")