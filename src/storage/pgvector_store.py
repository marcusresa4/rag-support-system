from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, String, Text, DateTime, Integer
import os
from datetime import datetime

DATABASE_URL = (
    f"postgresql+asyncpg://{os.getenv('POSTGRES_USER', 'raguser')}:"
    f"{os.getenv('POSTGRES_PASSWORD', 'ragpassword')}@"
    f"{os.getenv('POSTGRES_HOST', 'localhost')}:"
    f"{os.getenv('POSTGRES_PORT', '5432')}/"
    f"{os.getenv('POSTGRES_DB', 'ragdb')}"
)

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

class Base(DeclarativeBase):
    pass

class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    arxiv_id = Column(String(50), index=True, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    title = Column(Text)
    authors = Column(Text)
    published_date = Column(DateTime)
    chunk_strategy = Column(String(50))
    embedding = Column(Vector(384))
    created_at = Column(DateTime, default=datetime.utcnow)

async def init_db():
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)

async def insert_chunks(chunks: list[dict]):
    async with AsyncSessionLocal() as session:
        for chunk in chunks:
            obj = DocumentChunk(**chunk)
            session.add(obj)
        await session.commit()

async def dense_search(query_embedding: list[float], k: int = 5) -> list[dict]:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text("""
                SELECT
                    arxiv_id,
                    chunk_index,
                    content,
                    title,
                    1 - (embedding <=> CAST(:embedding AS vector)) AS score
                FROM document_chunks
                ORDER BY embedding <=> CAST(:embedding AS vector)
                LIMIT :k
            """),
            {"embedding": str(query_embedding), "k": k}
        )
        rows = result.fetchall()

    return [
        {
            "arxiv_id": row.arxiv_id,
            "chunk_index": row.chunk_index,
            "content": row.content,
            "title": row.title,
            "score": float(row.score),
            "strategy": "dense",
        }
        for row in rows
    ]