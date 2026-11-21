"""Data preparation: load the sales dataset and add analysis-friendly columns."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "sales_data.csv"
REQUIRED = ["Date", "Product", "Region", "Sales", "Customer_Age", "Customer_Gender", "Customer_Satisfaction"]
AGE_BINS = [17, 25, 35, 45, 55, 70]
AGE_LABELS = ["18-25", "26-35", "36-45", "46-55", "56-69"]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Quarter"] = df["Date"].dt.to_period("Q").astype(str)
    df["Month"] = df["Date"].dt.to_period("M").astype(str)
    df["Month_Name"] = df["Date"].dt.month_name()
    df["Age_Group"] = pd.cut(df["Customer_Age"], bins=AGE_BINS, labels=AGE_LABELS).astype(str)
    return df.sort_values("Date").reset_index(drop=True)


def load_data(path: str | Path | None = None) -> pd.DataFrame:
    return prepare(pd.read_csv(path or DATA_PATH))
