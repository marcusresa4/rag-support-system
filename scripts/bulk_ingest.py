import asyncio
import httpx
import time

PAPERS = [
    {"arxiv_id": "1706.03762", "title": "Attention Is All You Need"},
    {"arxiv_id": "1810.04805", "title": "BERT"},
    {"arxiv_id": "2005.11401", "title": "RAG"},
    {"arxiv_id": "2203.02155", "title": "InstructGPT"},
]

BASE_URL = "http://localhost:8000/api/v1"
WAIT_SECONDS = 60


async def ingest_paper(client: httpx.AsyncClient, arxiv_id: str, title: str):
    print(f"Ingesting: {title} ({arxiv_id})...")
    response = await client.post(
        f"{BASE_URL}/ingest",
        json={"arxiv_id": arxiv_id, "strategy": "sentence", "chunk_size": 100},
        timeout=60,
    )
    if response.status_code == 200:
        data = response.json()
        print(f"  ✅ {data['title']} — {data['chunks_created']} chunks")
    else:
        print(f"  ❌ Failed: {response.json()['detail']}")


async def main():
    async with httpx.AsyncClient() as client:
        for i, paper in enumerate(PAPERS):
            await ingest_paper(client, paper["arxiv_id"], paper["title"])
            if i < len(PAPERS) - 1:
                print(f"  ⏳ Waiting {WAIT_SECONDS}s before next paper...")
                time.sleep(WAIT_SECONDS)

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())