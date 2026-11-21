"""Plotly visualisations for the dashboard and chat answers."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

PALETTE = ["#2563eb", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6"]
LAYOUT = dict(template="plotly_white", margin=dict(l=10, r=10, t=50, b=10), height=380,
              colorway=PALETTE, font=dict(family="Inter, sans-serif"))


def _style(fig, title):
    fig.update_layout(title=title, **LAYOUT)
    return fig


def sales_trend(df: pd.DataFrame):
    m = df.groupby("Month", as_index=False)["Sales"].sum()
    m["3-month rolling avg"] = m["Sales"].rolling(3, min_periods=1).mean()
    fig = go.Figure()
    fig.add_bar(x=m["Month"], y=m["Sales"], name="Monthly sales", marker_color="#93c5fd")
    fig.add_scatter(x=m["Month"], y=m["3-month rolling avg"], name="3-month rolling avg", line=dict(color="#1d4ed8", width=3))
    return _style(fig, "Sales trend over time (monthly)")


def yearly_quarterly(df: pd.DataFrame):
    q = df.groupby(["Year", df["Date"].dt.quarter.rename("Q")], as_index=False)["Sales"].sum()
    q["Quarter"] = "Q" + q["Q"].astype(str)
    fig = px.bar(q, x="Year", y="Sales", color="Quarter", barmode="group", text_auto=".3s")
    fig.update_xaxes(type="category")
    return _style(fig, "Quarterly sales by year")


def product_performance(df: pd.DataFrame):
    p = df.groupby("Product", as_index=False).agg(Total_Sales=("Sales", "sum"), Avg_Sale=("Sales", "mean"),
                                                   Satisfaction=("Customer_Satisfaction", "mean"))
    fig = px.bar(p.sort_values("Total_Sales", ascending=False), x="Product", y="Total_Sales", color="Product",
                 text_auto=".3s", hover_data={"Avg_Sale": ":.0f", "Satisfaction": ":.2f"})
    fig.update_layout(showlegend=False)
    return _style(fig, "Product performance - total sales")


def product_distribution(df: pd.DataFrame):
    fig = px.box(df, x="Product", y="Sales", color="Product", points=False)
    fig.update_layout(showlegend=False)
    return _style(fig, "Sale value distribution by product (median & spread)")


def product_trend(df: pd.DataFrame):
    q = df.groupby(["Quarter", "Product"], as_index=False)["Sales"].sum()
    return _style(px.line(q, x="Quarter", y="Sales", color="Product", markers=True), "Product sales by quarter")


def regional_analysis(df: pd.DataFrame):
    r = df.groupby("Region", as_index=False)["Sales"].sum()
    fig = px.pie(r, names="Region", values="Sales", hole=0.45)
    return _style(fig, "Revenue share by region")


def product_region_heatmap(df: pd.DataFrame):
    pr = df.pivot_table(index="Product", columns="Region", values="Sales", aggfunc="sum")
    fig = px.imshow(pr, text_auto=",.0f", color_continuous_scale="Blues", aspect="auto")
    return _style(fig, "Product x Region sales heatmap")


def age_group_sales(df: pd.DataFrame):
    a = df.groupby(["Age_Group", "Customer_Gender"], as_index=False)["Sales"].sum()
    fig = px.bar(a, x="Age_Group", y="Sales", color="Customer_Gender", barmode="group", text_auto=".3s")
    return _style(fig, "Customer segmentation - sales by age group & gender")


def gender_split(df: pd.DataFrame):
    g = df.groupby("Customer_Gender", as_index=False)["Sales"].sum()
    return _style(px.pie(g, names="Customer_Gender", values="Sales", hole=0.45), "Revenue by gender")


def satisfaction_by_product(df: pd.DataFrame):
    fig = px.violin(df, x="Product", y="Customer_Satisfaction", color="Product", box=True)
    fig.update_layout(showlegend=False)
    return _style(fig, "Customer satisfaction by product")


def age_vs_sales(df: pd.DataFrame):
    fig = px.scatter(df.sample(min(len(df), 800), random_state=1), x="Customer_Age", y="Sales", color="Product",
                     opacity=0.6, trendline=None)
    return _style(fig, "Customer age vs sale value")


TOPIC_CHART = {
    "time": sales_trend, "product": product_performance, "region": regional_analysis,
    "product_region": product_region_heatmap, "demographics": age_group_sales,
    "satisfaction": satisfaction_by_product, "statistics": product_distribution, "overview": sales_trend,
}


def chart_for_topics(df: pd.DataFrame, topics: list[str]):
    for t in topics:
        if t in TOPIC_CHART:
            return TOPIC_CHART[t](df)
    return None
