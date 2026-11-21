"""Generate a realistic sample sales dataset for InsightForge.

The schema mirrors the capstone dataset (Date, Product, Region, Sales,
Customer_Age, Customer_Gender, Customer_Satisfaction). If you have the
official course dataset, simply drop it in as data/sales_data.csv - the
application works with any file that follows this schema.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N_ROWS = 2500

PRODUCTS = {"Widget A": 520, "Widget B": 610, "Widget C": 455, "Widget D": 700}
REGIONS = {"North": 1.06, "South": 0.94, "East": 1.02, "West": 0.98}


def generate(n_rows: int = N_ROWS) -> pd.DataFrame:
    start = pd.Timestamp("2023-01-01")
    days = (pd.Timestamp("2025-12-31") - start).days
    dates = start + pd.to_timedelta(np.sort(RNG.integers(0, days + 1, n_rows)), unit="D")

    products = RNG.choice(list(PRODUCTS), n_rows, p=[0.27, 0.25, 0.26, 0.22])
    regions = RNG.choice(list(REGIONS), n_rows, p=[0.26, 0.24, 0.27, 0.23])
    genders = RNG.choice(["Male", "Female"], n_rows, p=[0.51, 0.49])
    ages = RNG.integers(18, 70, n_rows)

    base = np.array([PRODUCTS[p] for p in products], dtype=float)
    region_factor = np.array([REGIONS[r] for r in regions])
    years_elapsed = (dates - start).days / 365.0
    trend = 1 + 0.06 * years_elapsed  # ~6% annual growth
    seasonality = 1 + 0.12 * np.cos(2 * np.pi * (dates.month - 11) / 12)  # Q4 peak
    age_factor = 1 + 0.004 * (ages - 40)  # older customers spend slightly more
    noise = RNG.normal(1, 0.28, n_rows)

    sales = np.clip(base * region_factor * trend * seasonality * age_factor * noise, 100, 1500).round(0)
    satisfaction = np.clip(
        RNG.normal(3.4, 0.8, n_rows)
        + np.where(products == "Widget D", 0.35, 0)
        - np.where(products == "Widget C", 0.25, 0)
        + np.where(regions == "North", 0.15, 0),
        1.0, 5.0,
    ).round(1)

    return pd.DataFrame({
        "Date": dates.strftime("%Y-%m-%d"),
        "Product": products,
        "Region": regions,
        "Sales": sales.astype(int),
        "Customer_Age": ages,
        "Customer_Gender": genders,
        "Customer_Satisfaction": satisfaction,
    })


if __name__ == "__main__":
    out = Path(__file__).parent / "sales_data.csv"
    generate().to_csv(out, index=False)
    print(f"Wrote {out}")
