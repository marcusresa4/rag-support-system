from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from src.retrieval.hybrid import hybrid_search
from src.generation.generator import (
    generate_answer,
    inject_citations,
    check_hallucination,
    compute_confidence,
)
import structlog

logger = structlog.get_logger()
router = APIRouter(tags=["generation"])


class AskRequest(BaseModel):
    question: str
    k: int = 5
    strategy: str = "hybrid"
    stream: bool = False


@router.post("/ask")
async def ask(request: AskRequest):
    try:
        # 1. Retrieve relevant chunks
        chunks = await hybrid_search(request.question, k=request.k)
        if not chunks:
            raise HTTPException(status_code=404, detail="No relevant sources found.")

        # 2. Stream or full response
        if request.stream:
            from src.generation.generator import generate_answer_stream

            def stream_generator():
                for token in generate_answer_stream(request.question, chunks):
                    yield token

            return StreamingResponse(stream_generator(), media_type="text/plain")

        # 3. Full response with citations + guardrails
        answer = generate_answer(request.question, chunks)
        citations = inject_citations(answer, chunks)
        hallucination = check_hallucination(answer, chunks)
        confidence = compute_confidence(chunks, hallucination)

        logger.info(
            "ask_complete",
            question=request.question,
            confidence=confidence,
            hallucination_risk=hallucination["hallucination_risk"],
        )

        return {
            "question": request.question,
            "answer": citations["answer"],
            "cited_sources": citations["cited_sources"],
            "confidence": confidence,
            "hallucination_risk": hallucination["hallucination_risk"],
            "grounded_ratio": hallucination["grounded_ratio"],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("ask_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))