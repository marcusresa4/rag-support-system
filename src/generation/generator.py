# src/generation/generator.py
import os
import anthropic
from dotenv import load_dotenv

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