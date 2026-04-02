import asyncio
import json
import math
import os
import sys
from datetime import datetime
import mlflow

sys.path.insert(0, os.path.abspath("."))

from dotenv import load_dotenv

load_dotenv()

from src.retrieval.hybrid import hybrid_search
from src.generation.generator import generate_answer, inject_citations
from src.evaluation.ragas_evaluator import run_ragas
from src.evaluation.llm_judge import llm_judge


def load_golden_dataset(path: str = "evals/golden_dataset.json") -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data["questions"]


async def evaluate_question(question_data: dict) -> dict:
    question = question_data["question"]
    ground_truth = question_data["ground_truth"]
    q_type = question_data["type"]

    chunks = await hybrid_search(question, k=5)

    if not chunks:
        return {
            "id": question_data["id"],
            "question": question,
            "type": q_type,
            "error": "no chunks retrieved",
        }

    answer = generate_answer(question, chunks)
    citations = inject_citations(answer, chunks)
    contexts = [c["content"] for c in chunks]

    judge_scores = llm_judge(
        question=question,
        answer=answer,
        ground_truth=ground_truth,
        contexts=contexts,
    )

    ragas_scores = run_ragas([{
        "question": question,
        "answer": answer,
        "contexts": contexts,
        "ground_truth": ground_truth,
    }])

    return {
        "id": question_data["id"],
        "question": question,
        "type": q_type,
        "answer": answer[:200],
        "cited_sources": len(citations["cited_sources"]),
        "judge": judge_scores,
        "ragas": ragas_scores,
    }


async def main():
    print("Loading golden dataset...")
    questions = load_golden_dataset()
    print(f"Evaluating {len(questions)} questions...\n")

    results = []
    for q in questions:
        print(f"  [{q['id']}] {q['question'][:60]}...")
        result = await evaluate_question(q)
        results.append(result)
        print(f"         judge avg: {result.get('judge', {}).get('average', 'N/A')} | "
              f"faithfulness: {result.get('ragas', {}).get('faithfulness', 'N/A')}")

    # Build report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    valid_judge = [r for r in results if "judge" in r]
    valid_faith = [r for r in results if "ragas" in r and not math.isnan(r["ragas"]["faithfulness"])]

    report = {
        "timestamp": timestamp,
        "total_questions": len(results),
        "results": results,
        "summary": {
            "avg_judge_score": round(
                sum(r["judge"]["average"] for r in valid_judge) / len(valid_judge), 2
            ),
            "avg_faithfulness": round(
                sum(r["ragas"]["faithfulness"] for r in valid_faith) / len(valid_faith), 2
            ),
        }
    }

    # Save JSON report
    json_path = f"evals/eval_report_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    # Save markdown report
    md_path = f"evals/eval_report_{timestamp}.md"
    with open(md_path, "w") as f:
        f.write("# Evaluation Report\n")
        f.write(f"**Date:** {timestamp}\n\n")
        f.write(f"**Questions evaluated:** {len(results)}\n\n")
        f.write("## Summary\n")
        f.write(f"- Average judge score: {report['summary']['avg_judge_score']}/5\n")
        f.write(f"- Average faithfulness: {report['summary']['avg_faithfulness']}\n\n")
        f.write("## Results\n\n")
        for r in results:
            f.write(f"### [{r['id']}] {r['question']}\n")
            f.write(f"**Type:** {r['type']}\n\n")
            f.write(f"**Answer:** {r.get('answer', 'N/A')}\n\n")
            if "judge" in r:
                f.write(f"**Judge scores:** accuracy={r['judge']['accuracy']} | "
                        f"completeness={r['judge']['completeness']} | "
                        f"citation={r['judge']['citation_quality']} | "
                        f"avg={r['judge']['average']}\n\n")
            if "ragas" in r:
                f.write(f"**RAGAS:** faithfulness={r['ragas']['faithfulness']} | "
                        f"relevancy={r['ragas']['answer_relevancy']}\n\n")
            f.write("---\n\n")

    print(f"\n✅ Report saved to {json_path}")
    print(f"✅ Report saved to {md_path}")
    print(f"\nSummary:")
    print(f"  Average judge score: {report['summary']['avg_judge_score']}/5")
    print(f"  Average faithfulness: {report['summary']['avg_faithfulness']}")

    # Quality gate
    print("\n🔍 Quality Gate Check:")
    failed = False

    if report["summary"]["avg_faithfulness"] < 0.8:
        print(f"  ❌ FAILED: avg faithfulness {report['summary']['avg_faithfulness']} < 0.8 threshold")
        failed = True
    else:
        print(f"  ✅ PASSED: avg faithfulness {report['summary']['avg_faithfulness']} >= 0.8")

    if report["summary"]["avg_judge_score"] < 3.0:
        print(f"  ❌ FAILED: avg judge score {report['summary']['avg_judge_score']} < 3.0 threshold")
        failed = True
    else:
        print(f"  ✅ PASSED: avg judge score {report['summary']['avg_judge_score']} >= 3.0")

    report["quality_gate"] = {
        "passed": not failed,
        "thresholds": {
            "faithfulness": 0.8,
            "judge_score": 3.0,
        }
    }

    if failed:
        print("\n⚠️  Quality gate FAILED — review results before deploying")
        sys.exit(1)
    else:
        print("\n✅ Quality gate PASSED")
        
        
    # Log to MLflow
    mlflow.set_experiment("rag-eval")

    with mlflow.start_run(run_name=f"eval_{timestamp}"):
        mlflow.log_metric("avg_judge_score", report["summary"]["avg_judge_score"])
        mlflow.log_metric("avg_faithfulness", report["summary"]["avg_faithfulness"])
        mlflow.log_metric("quality_gate_passed", int(not failed))
        mlflow.log_metric("total_questions", len(results))

        for r in results:
            if "judge" in r:
                mlflow.log_metric(f"{r['id']}_judge_avg", r["judge"]["average"])
            if "ragas" in r and not math.isnan(r["ragas"]["faithfulness"]):
                mlflow.log_metric(f"{r['id']}_faithfulness", r["ragas"]["faithfulness"])

        mlflow.log_param("golden_dataset_size", len(results))
        mlflow.log_param("retrieval_strategy", "hybrid")
        mlflow.log_param("llm_model", "claude-sonnet-4-6")
        mlflow.log_param("judge_model", "claude-haiku-4-5-20251001")
        mlflow.log_param("quality_gate_passed", not failed)

        mlflow.log_artifact(json_path)
        mlflow.log_artifact(md_path)

        print(f"\n📊 MLflow run logged — view at http://localhost:5000")


if __name__ == "__main__":
    asyncio.run(main())