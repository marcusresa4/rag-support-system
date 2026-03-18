import nltk
import numpy as np
from dataclasses import dataclass

@dataclass
class Chunk:
    content: str
    chunk_index: int
    strategy: str
    token_count: int

def _rough_token_count(text: str) -> int:
    return len(text.split())

def fixed_size_chunks(
    text: str,
    chunk_size: int = 512,
    overlap: int = 51,
) -> list[Chunk]:
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size})"
        )

    words = text.split()
    chunks = []
    start = 0
    idx = 0
    step = chunk_size - overlap

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(Chunk(
            content=" ".join(chunk_words),
            chunk_index=idx,
            strategy="fixed",
            token_count=len(chunk_words),
        ))
        start += step
        idx += 1

    return chunks


def sentence_chunks(
    text: str,
    target_size: int = 512,
    overlap_sentences: int = 1,
) -> list[Chunk]:
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current: list[str] = []
    current_tokens = 0
    idx = 0

    for sent in sentences:
        sent_tokens = _rough_token_count(sent)

        if current_tokens + sent_tokens > target_size and current:
            chunks.append(Chunk(
                content=" ".join(current),
                chunk_index=idx,
                strategy="sentence",
                token_count=current_tokens,
            ))
            idx += 1
            current = current[-overlap_sentences:] if overlap_sentences else []
            current_tokens = sum(_rough_token_count(s) for s in current)

        current.append(sent)
        current_tokens += sent_tokens

    if current:
        chunks.append(Chunk(
            content=" ".join(current),
            chunk_index=idx,
            strategy="sentence",
            token_count=current_tokens,
        ))

    return chunks

def semantic_chunks(
    text: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    breakpoint_percentile: int = 90,
) -> list[Chunk]:
    from sentence_transformers import SentenceTransformer  # lazy import

    sentences = nltk.sent_tokenize(text)
    if len(sentences) < 2:
        return [Chunk(content=text, chunk_index=0, strategy="semantic", token_count=_rough_token_count(text))]

    model = SentenceTransformer(model_name)
    embeddings = model.encode(sentences, show_progress_bar=False)

    similarities = []
    for i in range(len(embeddings) - 1):
        a, b = embeddings[i], embeddings[i + 1]
        sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        similarities.append(sim)

    threshold = np.percentile(similarities, 100 - breakpoint_percentile)
    breakpoints = {i + 1 for i, sim in enumerate(similarities) if sim < threshold}

    chunks = []
    current: list[str] = []
    idx = 0

    for i, sent in enumerate(sentences):
        if i in breakpoints and current:
            content = " ".join(current)
            chunks.append(Chunk(
                content=content,
                chunk_index=idx,
                strategy="semantic",
                token_count=_rough_token_count(content),
            ))
            idx += 1
            current = []
        current.append(sent)

    if current:
        content = " ".join(current)
        chunks.append(Chunk(
            content=content,
            chunk_index=idx,
            strategy="semantic",
            token_count=_rough_token_count(content),
        ))

    return chunks