from __future__ import annotations

from pathlib import Path
import json
import logging

import joblib
import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from evaluate import add_peak_season_label, build_breakdown_metrics, regression_metrics
from features import ID_COLUMNS, get_feature_columns, make_model_frame
from validation import validate_weekly_data


ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "config.yaml"

LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_path(config: dict, section: str) -> Path:
    return ROOT_DIR / config["paths"][section]


def build_pipeline(feature_cols: list[str], model_params: dict) -> Pipeline:
    categorical_cols = ID_COLUMNS
    numeric_cols = [col for col in feature_cols if col not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("numeric", "passthrough", numeric_cols),
        ]
    )
    model = LGBMRegressor(**model_params)
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def split_train_test(model_frame: pd.DataFrame, split_date: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = model_frame[model_frame["week_start"] < split_date].copy()
    test = model_frame[model_frame["week_start"] >= split_date].copy()
    if train.empty or test.empty:
        raise ValueError("Train/test split produced an empty dataset. Check the date range.")
    return train, test


def predict_frame(pipeline: Pipeline, frame: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    return np.clip(pipeline.predict(frame[feature_cols]), 0, None)


def walk_forward_validation(
    model_frame: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    split_date: pd.Timestamp,
    config: dict,
) -> pd.DataFrame:
    horizon = int(config["validation"]["forecast_horizon_weeks"])
    folds = int(config["validation"]["walk_forward_folds"])
    candidate_weeks = sorted(model_frame.loc[model_frame["week_start"] < split_date, "week_start"].unique())
    if len(candidate_weeks) <= horizon + 20:
        return pd.DataFrame()

    fold_cutoffs = candidate_weeks[-(folds + horizon) : -horizon]
    rows = []
    for cutoff in fold_cutoffs:
        cutoff = pd.Timestamp(cutoff)
        validation_end = cutoff + pd.Timedelta(weeks=horizon)
        train = model_frame[model_frame["week_start"] <= cutoff].copy()
        valid = model_frame[
            (model_frame["week_start"] > cutoff) & (model_frame["week_start"] <= validation_end)
        ].copy()
        if train.empty or valid.empty:
            continue

        pipeline = build_pipeline(feature_cols, config["model"])
        pipeline.fit(train[feature_cols], train[target_col])
        predictions = predict_frame(pipeline, valid, feature_cols)
        metrics = regression_metrics(valid[target_col], predictions)
        metrics.update(
            {
                "target": target_col,
                "cutoff_week": cutoff.date().isoformat(),
                "validation_end": validation_end.date().isoformat(),
                "validation_rows": int(len(valid)),
                "forecast_horizon_weeks": horizon,
            }
        )
        rows.append(metrics)
    return pd.DataFrame(rows)


def save_breakdowns(output: pd.DataFrame, target_col: str, reports_dir: Path) -> None:
    predicted_col = f"predicted_{target_col}"
    labelled = add_peak_season_label(output)
    breakdown_specs = {
        "brand": ["brand"],
        "category": ["category"],
        "whseid": ["whseid"],
        "season": ["season_type"],
    }
    for name, group_cols in breakdown_specs.items():
        breakdown = build_breakdown_metrics(labelled, target_col, predicted_col, group_cols)
        breakdown.to_csv(reports_dir / f"metrics_{target_col}_by_{name}.csv", index=False)


def train_for_target(weekly: pd.DataFrame, target_col: str, config: dict) -> dict:
    LOGGER.info("Training one-week-ahead LightGBM forecast for %s", target_col)
    model_frame = make_model_frame(weekly, target_col=target_col)
    split_date = pd.Timestamp(config["validation"]["test_start_date"])
    train, test = split_train_test(model_frame, split_date)
    feature_cols = get_feature_columns(model_frame)
    pipeline = build_pipeline(feature_cols, config["model"])

    LOGGER.info("Fitting model for %s with %s train rows and %s test rows", target_col, len(train), len(test))
    pipeline.fit(train[feature_cols], train[target_col])
    predictions = predict_frame(pipeline, test, feature_cols)
    metrics = regression_metrics(test[target_col], predictions)
    metrics.update(
        {
            "target": target_col,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "split_date": str(split_date.date()),
            "forecast_horizon": "one-week-ahead using historical lag features",
        }
    )

    output_columns = [
        "week_start",
        "category",
        "brand",
        "whseid",
        "is_holiday_season",
        "is_tet_season",
        target_col,
    ]
    output = test[output_columns].copy()
    output[f"predicted_{target_col}"] = predictions

    processed_dir = resolve_path(config, "processed_data")
    models_dir = resolve_path(config, "models")
    reports_dir = resolve_path(config, "reports")
    joblib.dump(pipeline, models_dir / f"lightgbm_{target_col}.pkl")
    model_frame.to_csv(processed_dir / f"model_frame_{target_col}.csv", index=False)
    output.to_csv(processed_dir / f"predictions_{target_col}.csv", index=False)
    save_breakdowns(output, target_col, reports_dir)

    walk_forward = walk_forward_validation(model_frame, target_col, feature_cols, split_date, config)
    if not walk_forward.empty:
        walk_forward.to_csv(reports_dir / f"walk_forward_{target_col}.csv", index=False)
    return metrics


def main() -> None:
    configure_logging()
    config = load_config()
    processed_dir = resolve_path(config, "processed_data")
    models_dir = resolve_path(config, "models")
    reports_dir = resolve_path(config, "reports")
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading weekly data from %s", processed_dir / "weekly_sales.csv")
    weekly = pd.read_csv(processed_dir / "weekly_sales.csv", parse_dates=["week_start"])
    validate_weekly_data(weekly, min_date_span_days=365)

    all_metrics = [
        train_for_target(weekly, target_col, config)
        for target_col in config["columns"]["target_columns"]
    ]

    with open(reports_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(all_metrics, file, indent=2)

    LOGGER.info("Saved models to %s", models_dir)
    LOGGER.info("Saved metrics to %s", reports_dir / "metrics.json")
    LOGGER.info("Metrics: %s", json.dumps(all_metrics, indent=2))


if __name__ == "__main__":
    main()
