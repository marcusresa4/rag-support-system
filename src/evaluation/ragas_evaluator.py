import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()


def run_ragas(eval_samples: list[dict]) -> dict:
    # Configure RAGAS to use Claude instead of OpenAI
    llm = LangchainLLMWrapper(
        ChatAnthropic(
            model="claude-haiku-4-5-20251001",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )
    )

    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    )

    # Assign LLM and embeddings to each metric
    for metric in [faithfulness, answer_relevancy, context_recall, context_precision]:
        metric.llm = llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = embeddings

    data = {
        "question":     [s["question"]     for s in eval_samples],
        "answer":       [s["answer"]       for s in eval_samples],
        "contexts":     [s["contexts"]     for s in eval_samples],
        "ground_truth": [s["ground_truth"] for s in eval_samples],
    }

    dataset = Dataset.from_dict(data)

    results = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        ],
    )

    return {
        "faithfulness":      round(float(results["faithfulness"]), 3),
        "answer_relevancy":  round(float(results["answer_relevancy"]), 3),
        "context_recall":    round(float(results["context_recall"]), 3),
        "context_precision": round(float(results["context_precision"]), 3),
    }