from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.features import ID_COLUMNS, get_feature_columns, make_model_frame
from src.train_model import predict_frame
from src.utils.config import ROOT_DIR, get_nested, resolve_project_path
from src.validation import validate_weekly_data


WEEKLY_COLUMNS = ["week_start", *ID_COLUMNS, "total_qty", "total_cbm"]


def _models_dir(config: dict[str, Any]) -> Path:
    return resolve_project_path(get_nested(config, ["paths", "models"], "models"), ROOT_DIR)


def _target_columns(config: dict[str, Any]) -> list[str]:
    return list(get_nested(config, ["columns", "target_columns"], ["total_qty"]))


def _normalize_ids(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    for column in ID_COLUMNS:
        normalized[column] = normalized[column].astype(str).str.strip().str.upper()
    return normalized


def _prepare_history(rows: list[dict[str, Any]]) -> pd.DataFrame:
    history = pd.DataFrame(rows)
    history = history[WEEKLY_COLUMNS].copy()
    history["week_start"] = pd.to_datetime(history["week_start"])
    history["total_qty"] = pd.to_numeric(history["total_qty"])
    history["total_cbm"] = pd.to_numeric(history["total_cbm"])
    history = _normalize_ids(history)
    validate_weekly_data(history)
    return history


def _prepare_forecast_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    forecast = pd.DataFrame(rows)
    forecast = forecast[["week_start", *ID_COLUMNS]].copy()
    forecast["week_start"] = pd.to_datetime(forecast["week_start"])
    forecast = _normalize_ids(forecast)
    if not (forecast["week_start"].dt.weekday == 0).all():
        raise ValueError("Forecast week_start must always be Monday.")
    duplicate_count = int(forecast.duplicated(subset=["week_start", *ID_COLUMNS]).sum())
    if duplicate_count:
        raise ValueError(f"Forecast request contains {duplicate_count} duplicate business-key rows.")
    forecast["total_qty"] = 0.0
    forecast["total_cbm"] = 0.0
    return forecast[WEEKLY_COLUMNS]


def _load_model(config: dict[str, Any], target_col: str):
    model_path = _models_dir(config) / f"lightgbm_{target_col}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


def _feature_columns(model: Any, model_frame: pd.DataFrame) -> list[str]:
    if hasattr(model, "feature_names_in_"):
        return [str(column) for column in model.feature_names_in_]
    return get_feature_columns(model_frame)


def predict_weekly_forecast(
    history_rows: list[dict[str, Any]],
    forecast_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    history = _prepare_history(history_rows)
    forecast = _prepare_forecast_rows(forecast_rows)
    combined = (
        pd.concat([history, forecast], ignore_index=True)
        .sort_values(ID_COLUMNS + ["week_start"])
        .reset_index(drop=True)
    )
    duplicate_count = int(combined.duplicated(subset=["week_start", *ID_COLUMNS]).sum())
    if duplicate_count:
        raise ValueError("Forecast rows overlap existing history rows.")

    horizon = int(get_nested(config, ["validation", "forecast_horizon_weeks"], 1))
    feature_config = config.get("features", {})
    result = forecast[["week_start", *ID_COLUMNS]].copy()

    for target_col in _target_columns(config):
        model = _load_model(config, target_col)
        model_frame = make_model_frame(
            combined,
            target_col=target_col,
            forecast_horizon_weeks=horizon,
            lag_weeks=feature_config.get("lag_weeks"),
            rolling_windows=feature_config.get("rolling_windows"),
        )
        inference_frame = forecast[["week_start", *ID_COLUMNS]].merge(
            model_frame,
            on=["week_start", *ID_COLUMNS],
            how="left",
        )
        feature_cols = _feature_columns(model, model_frame)
        if inference_frame.empty or inference_frame[feature_cols].isna().to_numpy().any():
            raise ValueError(
                f"Not enough weekly history to build inference features for {target_col}. "
                "Provide at least the configured lag and rolling-window history for each forecast row."
            )
        result[f"predicted_{target_col}"] = predict_frame(model, inference_frame, feature_cols)

    forecasts = result.copy()
    forecasts["week_start"] = forecasts["week_start"].dt.date.astype(str)
    return {
        "forecast_horizon_weeks": horizon,
        "forecasts": forecasts.to_dict(orient="records"),
    }
