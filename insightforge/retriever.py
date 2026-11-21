"""Custom retriever: intent routing + live pandas statistics + FAISS semantic search."""
from __future__ import annotations

import re
from typing import Any, List

import pandas as pd
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from . import analytics as A

TOPIC_KEYWORDS = {
    "time": ["trend", "month", "monthly", "year", "yearly", "annual", "quarter", "season", "growth", "over time", "period", "20"],
    "product": ["product", "widget", "item", "best-selling", "best selling", "top selling"],
    "region": ["region", "north", "south", "east", "west", "regional", "geograph"],
    "product_region": ["region"],
    "demographics": ["customer", "age", "gender", "male", "female", "demograph", "segment", "men", "women"],
    "satisfaction": ["satisf", "happy", "rating", "nps"],
    "statistics": ["median", "standard deviation", "std", "variance", "correlat", "statistic", "skew", "distribution"],
    "overview": ["overall", "total", "summary", "overview", "kpi", "average"],
}


def detect_topics(q: str) -> list[str]:
    ql = q.lower()
    return [t for t, kws in TOPIC_KEYWORDS.items() if any(k in ql for k in kws)] or ["overview"]


def detect_filters(q: str, df: pd.DataFrame) -> dict:
    ql = q.lower()
    filters = {}
    for col in ("Product", "Region"):
        for val in df[col].unique():
            if re.search(rf"\b{re.escape(str(val).lower())}\b", ql):
                filters[col] = val
    years = re.findall(r"\b(20\d\d)\b", ql)
    if years and int(years[0]) in set(df["Year"]):
        filters["Year"] = years[0]
    if re.search(r"\b(female|women)\b", ql):
        filters["Customer_Gender"] = "Female"
    elif re.search(r"\b(male|men)\b", ql):
        filters["Customer_Gender"] = "Male"
    return filters


class BusinessStatsRetriever(BaseRetriever):
    """Combines three retrieval strategies:
    1. Live pandas computation for the exact filters in the question.
    2. Topic routing (metadata filter) over pre-computed statistic documents.
    3. FAISS semantic similarity search as a fallback / enrichment.
    """

    vectorstore: Any
    df: Any
    k: int = 6

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None) -> List[Document]:
        out: list[Document] = []
        filters = detect_filters(query, self.df)
        if filters:
            res = A.filtered_stats(self.df, filters)
            if res:
                label, k = res
                out.append(Document(page_content=(
                    f"Live computed statistics for {label}: total sales {A.money(k['total_sales'])}, {k['transactions']:,} transactions, "
                    f"average sale {A.money(k['avg_sale'])}, median sale {A.money(k['median_sale'])}, standard deviation {A.money(k['std_sale'])}, "
                    f"average satisfaction {k['avg_satisfaction']:.2f}."), metadata={"topic": "live", "entity": label}))

        topics = detect_topics(query)
        if "statistics" in topics and not filters:
            topics.append("overview")
        entities = {str(v).lower() for v in filters.values()}
        semantic = self.vectorstore.similarity_search(query, k=self.k * 2)
        all_docs = list(self.vectorstore.docstore._dict.values())
        routed = [d for d in all_docs if d.metadata.get("topic") in topics]
        # entity-specific docs first, then topic-level docs, then semantic hits
        routed.sort(key=lambda d: (not any(e in d.page_content.lower() for e in entities), d.metadata.get("entity") != "ranking"))
        for d in routed[: self.k] + semantic:
            if d.page_content not in {x.page_content for x in out}:
                out.append(d)
        return out[: self.k + 1]
