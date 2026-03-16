from fastapi import FastAPI
from contextlib import asynccontextmanager
from src.api.routes import health, ingest
from src.storage.pgvector_store import init_db
from src.storage.elasticsearch_store import ensure_index
import structlog

logger = structlog.get_logger()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — initializing database...")
    await init_db()
    await ensure_index()
    yield
    logger.info("Shutting down.")

app = FastAPI(
    title="RAG Support System",
    description="Academic literature search with hybrid retrieval",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(ingest.router, prefix="/api/v1")