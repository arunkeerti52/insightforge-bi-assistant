"""Model evaluation with LangChain's QAEvalChain + retrieval metrics.

Ground-truth answers are computed directly from the data with pandas, so the
evaluation measures whether the RAG assistant reports the *correct* figures.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

try:
    from langchain_classic.evaluation.qa import QAEvalChain
except ImportError:  # langchain < 1.0
    from langchain.evaluation.qa import QAEvalChain

from . import analytics as A
from .assistant import InsightForgeAssistant

RESULTS_PATH = Path(__file__).resolve().parent.parent / "evaluation" / "eval_results.csv"


def build_test_set(df: pd.DataFrame) -> list[dict]:
    k = A.kpis(df)
    prod = A.group_summary(df, "Product")
    reg = A.group_summary(df, "Region")
    yearly = A.sales_by_period(df, "Year")
    gender = A.group_summary(df, "Customer_Gender")
    ages = A.group_summary(df, "Age_Group")
    sat = df.groupby("Product")["Customer_Satisfaction"].mean()
    y_last = yearly.index[-1]
    tests = [
        ("What are the total sales across the whole dataset?", f"Total sales are {A.money(k['total_sales'])}.", "overview"),
        ("What is the median and standard deviation of sales?", f"Median sale is {A.money(k['median_sale'])} and standard deviation is {A.money(k['std_sale'])}.", "statistics"),
        ("Which product has the highest total sales?", f"{prod['Total_Sales'].idxmax()} has the highest total sales with {A.money(prod['Total_Sales'].max())}.", "product"),
        ("Which product has the lowest total sales?", f"{prod['Total_Sales'].idxmin()} has the lowest total sales with {A.money(prod['Total_Sales'].min())}.", "product"),
        ("Which region generates the most revenue?", f"The {reg['Total_Sales'].idxmax()} region generates the most revenue, {A.money(reg['Total_Sales'].max())}.", "region"),
        ("Which region has the lowest sales?", f"The {reg['Total_Sales'].idxmin()} region has the lowest sales, {A.money(reg['Total_Sales'].min())}.", "region"),
        (f"What were total sales in {y_last}?", f"Total sales in {y_last} were {A.money(yearly.loc[y_last, 'Total_Sales'])}.", "time"),
        ("What were the total sales for Widget A?", f"Widget A total sales were {A.money(prod.loc['Widget A', 'Total_Sales'])}.", "product"),
        ("What is the average sale value for Widget B?", f"The average sale for Widget B is {A.money(prod.loc['Widget B', 'Avg_Sale'])}.", "product"),
        ("What are total sales in the North region?", f"North region total sales are {A.money(reg.loc['North', 'Total_Sales'])}.", "region"),
        ("How much did female customers spend in total?", f"Female customers spent {A.money(gender.loc['Female', 'Total_Sales'])} in total.", "demographics"),
        ("Which age group generates the most revenue?", f"The {ages['Total_Sales'].idxmax()} age group generates the most revenue, {A.money(ages['Total_Sales'].max())}.", "demographics"),
        ("Which product has the highest customer satisfaction?", f"{sat.idxmax()} has the highest customer satisfaction at {sat.max():.2f}.", "satisfaction"),
        ("What was the year-over-year sales growth in the latest year?", f"Sales grew {yearly['Growth_%'].iloc[-1]:+.1f}% year-over-year in {y_last}.", "time"),
    ]
    return [{"question": q, "answer": a, "expected_topic": t} for q, a, t in tests]


def run_evaluation(df: pd.DataFrame, save: bool = True) -> tuple[pd.DataFrame, dict]:
    assistant = InsightForgeAssistant(df)
    examples = build_test_set(df)
    predictions, rows = [], []
    for ex in examples:
        t0 = time.time()
        turn = assistant.ask(ex["question"], use_memory=False)
        predictions.append({"result": turn.answer})
        retrieved_topics = {d.metadata.get("topic") for d in turn.sources}
        rows.append({"question": ex["question"], "reference": ex["answer"], "prediction": turn.answer,
                     "latency_s": round(time.time() - t0, 3), "n_sources": len(turn.sources),
                     "retrieval_hit": ex["expected_topic"] in retrieved_topics or "live" in retrieved_topics
                     or (ex["expected_topic"] == "statistics" and "overview" in retrieved_topics)})

    eval_chain = QAEvalChain.from_llm(assistant.llm)
    graded = eval_chain.evaluate(examples, predictions, question_key="question", answer_key="answer", prediction_key="result")
    for row, g in zip(rows, graded):
        text = (g.get("results") or g.get("text") or "").strip()
        row["grade"] = "CORRECT" if "INCORRECT" not in text.upper() and "CORRECT" in text.upper() else "INCORRECT"

    # Negative control: deliberately wrong answers must be graded INCORRECT, proving the judge discriminates.
    controls = [(examples[2], "Widget A has the highest total sales with $250,000."),
                (examples[5], "The North region has the lowest sales, $446,332."),
                (examples[6], f"Total sales in that year were $1,000,000.")]
    ctrl = eval_chain.evaluate([c[0] for c in controls], [{"result": c[1]} for c in controls],
                               question_key="question", answer_key="answer", prediction_key="result")
    caught = sum("INCORRECT" in (g.get("results") or g.get("text") or "").upper() for g in ctrl)

    res = pd.DataFrame(rows)
    summary = {
        "n_tests": len(res),
        "qa_accuracy_%": float(round(100 * (res["grade"] == "CORRECT").mean(), 1)),
        "retrieval_hit_rate_%": float(round(100 * res["retrieval_hit"].mean(), 1)),
        "avg_latency_s": float(round(res["latency_s"].mean(), 3)),
        "grader_negative_controls_caught": f"{caught}/{len(controls)}",
    }
    if save:
        RESULTS_PATH.parent.mkdir(exist_ok=True)
        res.to_csv(RESULTS_PATH, index=False)
        RESULTS_PATH.with_name("eval_summary.json").write_text(json.dumps(summary, indent=2))
    return res, summary


if __name__ == "__main__":
    from .data_loader import load_data

    results, summary = run_evaluation(load_data())
    print(results[["question", "grade", "retrieval_hit", "latency_s"]].to_string())
    print(summary)
