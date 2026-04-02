from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
from src.api.routes import health, ingest, search, ask, ab_test
from src.storage.pgvector_store import init_db
from src.storage.elasticsearch_store import ensure_index
import structlog
import uuid

logger = structlog.get_logger()

def configure_logging():
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
    )

@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
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

@app.middleware("http")
async def add_trace_id(request: Request, call_next):
    trace_id = str(uuid.uuid4())[:8]
    structlog.contextvars.bind_contextvars(trace_id=trace_id)
    response = await call_next(request)
    structlog.contextvars.clear_contextvars()
    response.headers["X-Trace-ID"] = trace_id
    return response

app.include_router(health.router)
app.include_router(ingest.router, prefix="/api/v1")
app.include_router(search.router, prefix="/api/v1")
app.include_router(ask.router, prefix="/api/v1")
app.include_router(ab_test.router, prefix="/api/v1")