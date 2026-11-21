"""Knowledge base: turn pandas statistics into retrievable text documents + FAISS index."""
from __future__ import annotations

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from . import analytics as A
from .llm import get_embeddings


def _doc(text: str, topic: str, entity: str = "all") -> Document:
    return Document(page_content=text, metadata={"topic": topic, "entity": entity})


def build_documents(df: pd.DataFrame) -> list[Document]:
    docs: list[Document] = []
    k = A.kpis(df)
    docs.append(_doc(
        f"Overall business overview ({k['start']} to {k['end']}): total sales {A.money(k['total_sales'])} across "
        f"{k['transactions']:,} transactions. Average sale {A.money(k['avg_sale'])}, median sale {A.money(k['median_sale'])}, "
        f"standard deviation {A.money(k['std_sale'])}, minimum {A.money(k['min_sale'])}, maximum {A.money(k['max_sale'])}, "
        f"interquartile range {A.money(k['q1'])} to {A.money(k['q3'])}. Average customer satisfaction {k['avg_satisfaction']:.2f} out of 5. "
        f"Median customer age {k['median_age']:.0f}.", "overview"))

    # ---- Time periods -------------------------------------------------------
    yearly = A.sales_by_period(df, "Year")
    lines = [f"Year {y}: total sales {A.money(r.Total_Sales)}, {int(r.Transactions)} transactions, average sale {A.money(r.Avg_Sale)}, "
             f"median {A.money(r.Median_Sale)}, std dev {A.money(r.Std_Sale)}"
             + ("" if pd.isna(r.Growth_pct) else f", year-over-year growth {r.Growth_pct:+.1f}%")
             for y, r in yearly.rename(columns={"Growth_%": "Growth_pct"}).iterrows()]
    docs.append(_doc("Yearly sales performance (annual trend). " + "; ".join(lines) + ".", "time", "year"))
    for y, r in yearly.iterrows():
        docs.append(_doc(f"Sales performance in year {y}: total sales {A.money(r.Total_Sales)}, average sale {A.money(r.Avg_Sale)}, "
                         f"median sale {A.money(r.Median_Sale)}, standard deviation {A.money(r.Std_Sale)}, {int(r.Transactions)} transactions, "
                         f"average satisfaction {r.Avg_Satisfaction:.2f}.", "time", str(y)))

    quarterly = A.sales_by_period(df, "Quarter")
    docs.append(_doc("Quarterly sales performance. " + "; ".join(
        f"{q}: {A.money(r.Total_Sales)}" for q, r in quarterly.iterrows()) +
        f". Best quarter {quarterly['Total_Sales'].idxmax()} ({A.money(quarterly['Total_Sales'].max())}); "
        f"weakest quarter {quarterly['Total_Sales'].idxmin()} ({A.money(quarterly['Total_Sales'].min())}).", "time", "quarter"))

    monthly = A.sales_by_period(df, "Month")
    seasonal = df.groupby(df["Date"].dt.month_name(), sort=False)["Sales"].sum()
    docs.append(_doc(
        f"Monthly sales trend: best month {monthly['Total_Sales'].idxmax()} with {A.money(monthly['Total_Sales'].max())}, "
        f"lowest month {monthly['Total_Sales'].idxmin()} with {A.money(monthly['Total_Sales'].min())}. "
        f"Average monthly sales {A.money(monthly['Total_Sales'].mean())} (std dev {A.money(monthly['Total_Sales'].std())}). "
        f"Seasonality: the highest-selling calendar month overall is {seasonal.idxmax()} ({A.money(seasonal.max())}) and the "
        f"lowest is {seasonal.idxmin()} ({A.money(seasonal.min())}).", "time", "month"))

    # ---- Products & regions -------------------------------------------------
    for dim, topic in (("Product", "product"), ("Region", "region")):
        g = A.group_summary(df, dim).sort_values("Total_Sales", ascending=False)
        docs.append(_doc(
            f"{dim} ranking by total sales: " + ", ".join(f"{i} {A.money(r.Total_Sales)} ({r['Share_%']:.1f}% share)" for i, r in g.iterrows())
            + f". Highest {dim.lower()} is {g.index[0]}; lowest {dim.lower()} is {g.index[-1]}.", topic, "ranking"))
        for name, r in g.iterrows():
            docs.append(_doc(
                f"{dim} {name}: total sales {A.money(r.Total_Sales)} ({r['Share_%']:.1f}% of revenue), {int(r.Transactions)} transactions, "
                f"average sale {A.money(r.Avg_Sale)}, median sale {A.money(r.Median_Sale)}, standard deviation {A.money(r.Std_Sale)}, "
                f"average customer satisfaction {r.Avg_Satisfaction:.2f}.", topic, str(name)))
        yr = df.pivot_table(index=dim, columns="Year", values="Sales", aggfunc="sum")
        for name, row in yr.iterrows():
            docs.append(_doc(f"{dim} {name} sales by year: " + ", ".join(f"{c}: {A.money(v)}" for c, v in row.items()) + ".",
                             topic, f"{name}-yearly"))

    pr = df.pivot_table(index="Product", columns="Region", values="Sales", aggfunc="sum")
    for prod, row in pr.iterrows():
        docs.append(_doc(f"Product-region analysis for {prod}: " + ", ".join(f"{reg} region {A.money(v)}" for reg, v in row.items())
                         + f". {prod} sells best in the {row.idxmax()} region and lowest in the {row.idxmin()} region.", "product_region", prod))

    # ---- Customer segmentation ---------------------------------------------
    gender = A.group_summary(df, "Customer_Gender")
    docs.append(_doc("Customer segmentation by gender: " + "; ".join(
        f"{gname} customers total sales {A.money(r.Total_Sales)} ({r['Share_%']:.1f}%), average sale {A.money(r.Avg_Sale)}, "
        f"median {A.money(r.Median_Sale)}, average satisfaction {r.Avg_Satisfaction:.2f}" for gname, r in gender.iterrows()) + ".",
        "demographics", "gender"))
    ages = A.group_summary(df, "Age_Group")
    docs.append(_doc("Customer segmentation by age group: " + "; ".join(
        f"age {a} total sales {A.money(r.Total_Sales)} ({r['Share_%']:.1f}%), average sale {A.money(r.Avg_Sale)}, {int(r.Transactions)} customers, "
        f"satisfaction {r.Avg_Satisfaction:.2f}" for a, r in ages.iterrows())
        + f". Highest-spending age group (average sale) is {ages['Avg_Sale'].idxmax()}; largest revenue age group is {ages['Total_Sales'].idxmax()}.",
        "demographics", "age"))
    gp = df.pivot_table(index="Customer_Gender", columns="Product", values="Sales", aggfunc="sum")
    docs.append(_doc("Product preference by gender: " + "; ".join(
        f"{g} customers buy most {row.idxmax()} ({A.money(row.max())})" for g, row in gp.iterrows()) + ".", "demographics", "gender-product"))
    ap = df.pivot_table(index="Age_Group", columns="Product", values="Sales", aggfunc="sum")
    docs.append(_doc("Product preference by age group: " + "; ".join(
        f"age {a} prefers {row.idxmax()}" for a, row in ap.iterrows()) + ".", "demographics", "age-product"))

    sat = df.groupby("Product")["Customer_Satisfaction"].mean()
    c = A.correlations(df)
    docs.append(_doc(
        f"Customer satisfaction analysis: overall average {df['Customer_Satisfaction'].mean():.2f}, median {df['Customer_Satisfaction'].median():.2f}, "
        f"std dev {df['Customer_Satisfaction'].std():.2f}. Highest satisfaction product {sat.idxmax()} ({sat.max():.2f}); lowest satisfaction "
        f"product {sat.idxmin()} ({sat.min():.2f}). {(df['Customer_Satisfaction'] < 2.5).mean()*100:.1f}% of customers are dissatisfied (score below 2.5).",
        "satisfaction"))
    docs.append(_doc(
        f"Statistical measures and correlations: correlation between customer age and sales {c['age_vs_sales']:.3f}; between satisfaction "
        f"and sales {c['satisfaction_vs_sales']:.3f}; between age and satisfaction {c['age_vs_satisfaction']:.3f}. Sales skewness "
        f"{df['Sales'].skew():.2f}, kurtosis {df['Sales'].kurt():.2f}, coefficient of variation {df['Sales'].std()/df['Sales'].mean()*100:.1f}%.",
        "statistics"))
    return docs


def build_vectorstore(docs: list[Document]) -> FAISS:
    return FAISS.from_documents(docs, get_embeddings())
