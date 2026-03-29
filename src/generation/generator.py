# src/generation/generator.py
import os
import anthropic
from dotenv import load_dotenv
import re
from typing import Generator

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = """You are a scientific research assistant specializing in academic literature.
Answer questions using ONLY the provided context chunks.
If the answer is not in the context, say "I cannot find this information in the provided sources."
Always cite your sources using [Source N] notation where N is the source number."""


def build_prompt(query: str, chunks: list[dict]) -> str:
    context_parts = []
    for i, chunk in enumerate(chunks):
        context_parts.append(
            f"[Source {i+1}] {chunk['title']}\n{chunk['content']}"
        )
    context = "\n\n".join(context_parts)

    return f"""Context:
{context}

Question: {query}

Answer using the context above. Cite sources as [Source N]."""


def generate_answer(query: str, chunks: list[dict]) -> str:
    prompt = build_prompt(query, chunks)

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}]
    )

    return message.content[0].text

def inject_citations(answer: str, chunks: list[dict]) -> dict:
    """
    Finds all [Source N] references in the answer and maps them
    back to the actual paper metadata.
    """
    # Find all [Source N] references in the answer
    source_refs = re.findall(r'\[Source (\d+)\]', answer)
    source_refs = list(set(int(n) for n in source_refs))  # deduplicate

    cited_sources = []
    for n in sorted(source_refs):
        idx = n - 1  # [Source 1] → index 0
        if idx < len(chunks):
            cited_sources.append({
                "source_number": n,
                "arxiv_id": chunks[idx]["arxiv_id"],
                "title": chunks[idx]["title"],
                "chunk_index": chunks[idx]["chunk_index"],
            })

    return {
        "answer": answer,
        "cited_sources": cited_sources,
        "total_sources_used": len(cited_sources),
    }
    
def generate_answer_stream(query: str, chunks: list[dict]) -> Generator[str, None, None]:
    """Streams the answer token by token."""
    prompt = build_prompt(query, chunks)

    with client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}]
    ) as stream:
        for text in stream.text_stream:
            yield text
            
def check_hallucination(answer: str, chunks: list[dict]) -> dict:
    """
    Simple hallucination check — verifies the answer is grounded
    in the retrieved chunks using embedding similarity.
    """
    from src.embeddings.embedder import embed_texts
    import numpy as np

    # Split answer into sentences
    import nltk
    sentences = nltk.sent_tokenize(answer)
    if not sentences:
        return {"hallucination_risk": "low", "grounded_ratio": 1.0}

    # Embed all sentences and all chunks
    chunk_texts = [c["content"] for c in chunks]
    all_texts = sentences + chunk_texts
    all_embeddings = embed_texts(all_texts)

    sentence_embeddings = all_embeddings[:len(sentences)]
    chunk_embeddings = all_embeddings[len(sentences):]

    # For each sentence, find max similarity to any chunk
    grounded = 0
    for sent_emb in sentence_embeddings:
        max_sim = max(
            float(np.dot(sent_emb, chunk_emb))
            for chunk_emb in chunk_embeddings
        )
        if max_sim > 0.5:  # threshold — tune this in Phase 4
            grounded += 1

    grounded_ratio = grounded / len(sentences)
    risk = "low" if grounded_ratio > 0.7 else "medium" if grounded_ratio > 0.4 else "high"

    return {
        "hallucination_risk": risk,
        "grounded_ratio": round(grounded_ratio, 2),
        "sentences_checked": len(sentences),
        "sentences_grounded": grounded,
    }
    
def compute_confidence(chunks: list[dict], hallucination_result: dict) -> float:
    """
    Confidence score based on:
    - Average retrieval score of the chunks (normalized to 0-1)
    - Hallucination grounded ratio
    """
    if not chunks:
        return 0.0

    retrieval_scores = [c["score"] for c in chunks]
    avg_retrieval = sum(retrieval_scores) / len(retrieval_scores)
    max_retrieval = max(retrieval_scores)

    # Normalize relative to the best score in this result set
    normalized_retrieval = avg_retrieval / max_retrieval if max_retrieval > 0 else 0.0

    # Combine retrieval quality + grounded ratio (equal weight)
    grounded_ratio = hallucination_result.get("grounded_ratio", 0.5)
    confidence = (normalized_retrieval + grounded_ratio) / 2

    return round(confidence, 2)