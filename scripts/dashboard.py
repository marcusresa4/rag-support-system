import streamlit as st
import mlflow
import pandas as pd
import math

st.set_page_config(page_title="RAG Quality Dashboard", layout="wide")
st.title("📊 RAG Support System — Quality Dashboard")

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = mlflow.tracking.MlflowClient()

try:
    experiment = client.get_experiment_by_name("rag-eval")
    if not experiment:
        st.error("No 'rag-eval' experiment found. Run python scripts/run_eval.py first.")
        st.stop()

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
    )

    if not runs:
        st.warning("No runs found. Run python scripts/run_eval.py first.")
        st.stop()

except Exception as e:
    st.error(f"MLflow connection error: {e}")
    st.stop()

# Summary metrics
st.header("📈 Latest Run Summary")
latest = runs[0]
metrics = latest.data.metrics

col1, col2, col3, col4 = st.columns(4)
col1.metric("Avg Judge Score", f"{metrics.get('avg_judge_score', 0):.2f} / 5")
col2.metric("Avg Faithfulness", f"{metrics.get('avg_faithfulness', 0):.2f}")
col3.metric("Quality Gate", "✅ PASSED" if metrics.get('quality_gate_passed') == 1 else "❌ FAILED")
col4.metric("Questions Evaluated", int(metrics.get('total_questions', 0)))

st.divider()

# Metrics over time
st.header("📉 Metrics Over Time")

if len(runs) > 1:
    history = []
    for run in runs:
        history.append({
            "run": run.info.run_name,
            "avg_judge_score": run.data.metrics.get("avg_judge_score", None),
            "avg_faithfulness": run.data.metrics.get("avg_faithfulness", None),
        })

    df = pd.DataFrame(history).set_index("run")
    st.line_chart(df)
else:
    st.info("Run the eval at least twice to see trends over time.")

st.divider()

# Per-question breakdown
st.header("🔍 Per-Question Breakdown (Latest Run)")

question_ids = [f"q{str(i).zfill(3)}" for i in range(1, 11)]
rows = []
for qid in question_ids:
    judge = metrics.get(f"{qid}_judge_avg")
    faith = metrics.get(f"{qid}_faithfulness")
    if judge is not None:
        rows.append({
            "Question ID": qid,
            "Judge Score": judge,
            "Faithfulness": faith if faith and not math.isnan(faith) else "N/A",
            "Status": "✅" if judge >= 3.0 else "❌",
        })

df_questions = pd.DataFrame(rows)
st.dataframe(df_questions, use_container_width=True)

st.divider()

# Failed questions
st.header("⚠️ Failed Questions (Judge Score < 3.0)")
failed = [r for r in rows if isinstance(r["Judge Score"], float) and r["Judge Score"] < 3.0]
if failed:
    st.dataframe(pd.DataFrame(failed), use_container_width=True)
else:
    st.success("No failed questions in the latest run!")