from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluate import forecast_bias, regression_metrics, wmape


def mae(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(true - pred)))


def rmse(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((true - pred) ** 2)))


__all__ = ["forecast_bias", "mae", "regression_metrics", "rmse", "wmape"]
