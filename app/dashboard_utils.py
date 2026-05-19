from __future__ import annotations

import pandas as pd


def forecast_bias(actual: pd.Series, predicted: pd.Series) -> float:
    denominator = actual.sum()
    if denominator == 0:
        return 0.0
    return float((predicted.sum() - denominator) / denominator)


def safe_peak_week(df: pd.DataFrame, value_col: str) -> pd.Timestamp | None:
    if df.empty or value_col not in df.columns:
        return None
    weekly_totals = df.groupby("week_start")[value_col].sum()
    if weekly_totals.empty:
        return None
    return pd.Timestamp(weekly_totals.idxmax())


def wmape_status(wmape: float) -> str:
    if wmape >= 0.5:
        return "High uncertainty"
    if wmape >= 0.3:
        return "Watch"
    return "Stable"


def overload_status(value: float, threshold: float) -> str:
    if threshold <= 0:
        return "No threshold"
    ratio = value / threshold
    if ratio >= 1:
        return "Over capacity"
    if ratio >= 0.85:
        return "Near capacity"
    return "Within capacity"
