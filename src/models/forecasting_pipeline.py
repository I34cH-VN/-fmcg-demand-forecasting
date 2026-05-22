from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.preprocess import build_weekly_series, clean_sales_data, load_raw_data
from src.train_model import train_for_target
from src.utils.config import ROOT_DIR, get_nested, resolve_project_path
from src.validation import validate_weekly_data


def _config_path(config: dict, key: str, fallback: str) -> Path:
    return resolve_project_path(get_nested(config, ["paths", key], fallback), ROOT_DIR)


def load_sales_data(config: dict) -> pd.DataFrame:
    input_path = get_nested(config, ["input", "path"])
    if input_path:
        path = resolve_project_path(input_path, ROOT_DIR)
        return pd.read_csv(path)
    raw_dir = _config_path(config, "raw_data", "data/raw")
    return load_raw_data(raw_dir)


def prepare_weekly_data(config: dict) -> pd.DataFrame:
    processed_dir = _config_path(config, "processed_data", "data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    raw = load_sales_data(config)
    cleaned = clean_sales_data(raw)
    weekly = build_weekly_series(cleaned)
    cleaned.to_csv(processed_dir / "sales_cleaned.csv", index=False)
    weekly.to_csv(processed_dir / "weekly_sales.csv", index=False)
    return weekly


def run_training_pipeline(config: dict) -> dict:
    processed_dir = _config_path(config, "processed_data", "data/processed")
    models_dir = _config_path(config, "models", "models")
    reports_dir = _config_path(config, "reports", "reports")
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    weekly_path = processed_dir / "weekly_sales.csv"
    if weekly_path.exists():
        weekly = pd.read_csv(weekly_path, parse_dates=["week_start"])
    else:
        weekly = prepare_weekly_data(config)
    validate_weekly_data(weekly)

    target_columns = get_nested(config, ["columns", "target_columns"], ["total_qty"])
    metrics = [train_for_target(weekly, target_col, config) for target_col in target_columns]
    metrics_path = reports_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    return {"metrics": metrics, "metrics_path": str(metrics_path)}
