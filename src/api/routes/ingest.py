from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(tags=["ingestion"])

class IngestRequest(BaseModel):
    arxiv_id: str

@router.post("/ingest")
async def ingest_paper(request: IngestRequest):
    return {"status": "queued", "arxiv_id": request.arxiv_id}