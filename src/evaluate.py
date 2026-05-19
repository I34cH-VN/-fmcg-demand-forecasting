from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def wmape(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    denominator = np.sum(np.abs(y_true_arr))
    if denominator == 0:
        return 0.0
    return float(np.sum(np.abs(y_true_arr - y_pred_arr)) / denominator)


def forecast_bias(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    denominator = np.sum(y_true_arr)
    if denominator == 0:
        return 0.0
    return float((np.sum(y_pred_arr) - denominator) / denominator)


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray, prefix: str | None = None) -> dict:
    metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "wmape": wmape(y_true, y_pred),
        "forecast_bias": forecast_bias(y_true, y_pred),
    }
    if prefix:
        return {f"{prefix}_{key}": value for key, value in metrics.items()}
    return metrics


def build_breakdown_metrics(
    predictions: pd.DataFrame,
    actual_col: str,
    predicted_col: str,
    group_cols: list[str],
) -> pd.DataFrame:
    rows = []
    for keys, group in predictions.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        metrics = regression_metrics(group[actual_col], group[predicted_col])
        row = dict(zip(group_cols, keys))
        row.update(metrics)
        row["actual_sum"] = float(group[actual_col].sum())
        row["predicted_sum"] = float(group[predicted_col].sum())
        row["rows"] = int(len(group))
        rows.append(row)
    return pd.DataFrame(rows)


def add_peak_season_label(frame: pd.DataFrame) -> pd.DataFrame:
    labelled = frame.copy()
    if "is_holiday_season" in labelled.columns:
        labelled["season_type"] = np.where(labelled["is_holiday_season"] == 1, "peak_season", "non_peak")
    else:
        labelled["season_type"] = np.where(labelled["week_start"].dt.month.isin([1, 2, 9, 12]), "peak_season", "non_peak")
    return labelled
