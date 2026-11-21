"""Advanced data summary with pandas: KPIs, time, product, region, demographics."""
from __future__ import annotations

import pandas as pd

AGG = {"Total_Sales": ("Sales", "sum"), "Avg_Sale": ("Sales", "mean"), "Median_Sale": ("Sales", "median"),
       "Std_Sale": ("Sales", "std"), "Transactions": ("Sales", "count"),
       "Avg_Satisfaction": ("Customer_Satisfaction", "mean")}


def money(x: float) -> str:
    return f"${x:,.0f}"


def kpis(df: pd.DataFrame) -> dict:
    s = df["Sales"]
    return {
        "total_sales": float(s.sum()), "transactions": int(s.count()), "avg_sale": float(s.mean()),
        "median_sale": float(s.median()), "std_sale": float(s.std()), "min_sale": float(s.min()),
        "max_sale": float(s.max()), "q1": float(s.quantile(.25)), "q3": float(s.quantile(.75)),
        "avg_satisfaction": float(df["Customer_Satisfaction"].mean()),
        "median_age": float(df["Customer_Age"].median()),
        "start": df["Date"].min().date().isoformat(), "end": df["Date"].max().date().isoformat(),
    }


def group_summary(df: pd.DataFrame, by: str | list[str]) -> pd.DataFrame:
    g = df.groupby(by, observed=True).agg(**AGG)
    g["Share_%"] = 100 * g["Total_Sales"] / df["Sales"].sum()
    return g.round(2)


def sales_by_period(df: pd.DataFrame, period: str = "Month") -> pd.DataFrame:
    out = df.groupby(period).agg(**AGG).round(2)
    out["Growth_%"] = (out["Total_Sales"].pct_change() * 100).round(2)
    return out


def correlations(df: pd.DataFrame) -> dict:
    return {
        "age_vs_sales": float(df["Customer_Age"].corr(df["Sales"])),
        "satisfaction_vs_sales": float(df["Customer_Satisfaction"].corr(df["Sales"])),
        "age_vs_satisfaction": float(df["Customer_Age"].corr(df["Customer_Satisfaction"])),
    }


def filtered_stats(df: pd.DataFrame, filters: dict) -> tuple[str, dict] | None:
    """Live pandas computation for any combination of filters."""
    sub = df
    for col, val in filters.items():
        sub = sub[sub[col].astype(str).str.lower() == str(val).lower()]
    if sub.empty:
        return None
    label = ", ".join(f"{k}={v}" for k, v in filters.items())
    return label, kpis(sub)
