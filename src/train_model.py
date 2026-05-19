from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from features import ID_COLUMNS, make_model_frame


ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"


def wmape(y_true: pd.Series, y_pred: np.ndarray) -> float:
    denominator = np.sum(np.abs(y_true))
    if denominator == 0:
        return 0.0
    return float(np.sum(np.abs(y_true - y_pred)) / denominator)


def train_for_target(weekly: pd.DataFrame, target_col: str) -> dict:
    model_frame = make_model_frame(weekly, target_col=target_col)
    split_date = pd.Timestamp("2025-01-01")

    train = model_frame[model_frame["week_start"] < split_date].copy()
    test = model_frame[model_frame["week_start"] >= split_date].copy()
    if train.empty or test.empty:
        raise ValueError("Train/test split produced an empty dataset. Check the date range.")

    feature_cols = [col for col in model_frame.columns if col not in ["week_start", "total_qty", "total_cbm"]]
    categorical_cols = ID_COLUMNS
    numeric_cols = [col for col in feature_cols if col not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("numeric", "passthrough", numeric_cols),
        ]
    )

    model = LGBMRegressor(
        objective="regression",
        n_estimators=450,
        learning_rate=0.04,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        verbosity=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    x_train = train[feature_cols]
    y_train = train[target_col]
    x_test = test[feature_cols]
    y_test = test[target_col]

    pipeline.fit(x_train, y_train)
    predictions = np.clip(pipeline.predict(x_test), 0, None)

    metrics = {
        "target": target_col,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "wmape": wmape(y_test, predictions),
        "split_date": str(split_date.date()),
    }

    output = test[["week_start", "category", "brand", "whseid", target_col]].copy()
    output[f"predicted_{target_col}"] = predictions

    joblib.dump(pipeline, MODELS_DIR / f"lightgbm_{target_col}.pkl")
    model_frame.to_csv(PROCESSED_DIR / f"model_frame_{target_col}.csv", index=False)
    output.to_csv(PROCESSED_DIR / f"predictions_{target_col}.csv", index=False)

    return metrics


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    weekly = pd.read_csv(PROCESSED_DIR / "weekly_sales.csv", parse_dates=["week_start"])
    all_metrics = [train_for_target(weekly, target_col) for target_col in ["total_qty", "total_cbm"]]

    with open(REPORTS_DIR / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(all_metrics, file, indent=2)

    print(json.dumps(all_metrics, indent=2))
    print(f"Saved models to {MODELS_DIR}")
    print(f"Saved metrics to {REPORTS_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
