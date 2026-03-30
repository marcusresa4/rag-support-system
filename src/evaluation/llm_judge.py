import os
import json
import anthropic
from dotenv import load_dotenv
import re

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

JUDGE_PROMPT = """You are an expert evaluator of RAG (Retrieval Augmented Generation) systems.
Evaluate the following answer based on the question and ground truth.

Question: {question}
Ground Truth: {ground_truth}
Generated Answer: {answer}
Retrieved Context: {context}

Score the answer on these criteria (1-5 scale):
1. Accuracy: Does the answer correctly reflect the ground truth?
2. Completeness: Does the answer cover all key points from the ground truth?
3. Citation Quality: Does the answer properly cite its sources?

Respond ONLY with a JSON object in this exact format:
{{
    "accuracy": <1-5>,
    "completeness": <1-5>,
    "citation_quality": <1-5>,
    "reasoning": "<one sentence explanation>"
}}"""


def llm_judge(
    question: str,
    answer: str,
    ground_truth: str,
    contexts: list[str],
) -> dict:
    context_text = "\n".join(contexts[:3])

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        messages=[
            {
                "role": "user",
                "content": JUDGE_PROMPT.format(
                    question=question,
                    ground_truth=ground_truth,
                    answer=answer,
                    context=context_text,
                )
            }
        ]
    )

    raw = message.content[0].text.strip()
    raw = re.sub(r"```json\s*|\s*```", "", raw).strip()
    scores = json.loads(raw)
    scores["average"] = round((scores["accuracy"] + scores["completeness"] + scores["citation_quality"]) / 3, 2)
    return scores