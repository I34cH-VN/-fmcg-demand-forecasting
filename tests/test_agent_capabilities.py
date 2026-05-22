from pathlib import Path

import pandas as pd

from src.agent.forecasting_agent import ForecastingAgent
from src.data.quality import (
    QualityConfig,
    analyze_data_quality,
    check_train_test_leakage,
    check_week_start_monday,
)
from src.evaluation.metrics import forecast_bias, mae, rmse, wmape
from src.models import forecasting_pipeline
from src.reports.markdown_report import generate_markdown_report
from src.utils.config import load_config


def test_data_quality_detects_missing_negative_outlier_and_duplicates():
    df = pd.DataFrame(
        {
            "week_start": ["2023-01-02", "2023-01-02", "2023-01-16", "2023-01-30"],
            "category": ["DRY", "DRY", None, "DRY"],
            "brand": ["A", "A", "A", "A"],
            "whseid": ["WH1", "WH1", "WH1", "WH1"],
            "total_qty": [10, 10, -5, 1000],
        }
    )
    report = analyze_data_quality(
        df,
        QualityConfig(
            date_column="week_start",
            week_start_column="week_start",
            target_column="total_qty",
            group_columns=["category", "brand", "whseid"],
        ),
        split_date="2023-01-16",
    )
    checks = {item["check"]: item for item in report["checks"]}
    assert checks["missing_values"]["count"] == 1
    assert checks["duplicated_rows"]["count"] == 1
    assert checks["negative_values"]["count"] == 1
    assert checks["outliers"]["count"] >= 1


def test_week_start_monday_logic_flags_non_monday():
    df = pd.DataFrame({"week_start": ["2023-01-02", "2023-01-03"]})
    result = check_week_start_monday(df, "week_start")
    assert result["severity"] == "error"
    assert result["count"] == 1


def test_leakage_check_flags_current_period_target_features():
    df = pd.DataFrame({"week_start": pd.date_range("2023-01-02", periods=4, freq="W-MON")})
    result = check_train_test_leakage(
        df,
        date_column="week_start",
        split_date="2023-01-16",
        feature_columns=["total_qty", "total_qty_diff_1", "total_qty_lag_1"],
        target_column="total_qty",
    )
    assert result["severity"] == "warning"
    assert "total_qty" in result["details"][-1]
    assert "total_qty_lag_1" not in result["details"][-1]


def test_leakage_check_does_not_require_global_date_sort_for_time_split():
    df = pd.DataFrame({"week_start": ["2023-01-16", "2023-01-02", "2023-01-23"]})
    result = check_train_test_leakage(
        df,
        date_column="week_start",
        split_date="2023-01-16",
    )
    assert result["severity"] == "info"


def test_training_pipeline_rebuilds_weekly_data_when_input_path_is_configured(monkeypatch):
    calls = {"prepared": False}
    weekly = pd.DataFrame(
        {
            "week_start": pd.to_datetime(["2023-01-02", "2023-01-09"]),
            "category": ["DRY", "DRY"],
            "brand": ["A", "A"],
            "whseid": ["WH1", "WH1"],
            "total_qty": [10, 20],
            "total_cbm": [1.0, 2.0],
        }
    )

    def fake_prepare(config):
        calls["prepared"] = True
        return weekly

    monkeypatch.setattr(forecasting_pipeline, "prepare_weekly_data", fake_prepare)
    monkeypatch.setattr(forecasting_pipeline, "validate_weekly_data", lambda frame: None)
    monkeypatch.setattr(
        forecasting_pipeline,
        "train_for_target",
        lambda frame, target_col, config: {"target": target_col, "mae": 0, "rmse": 0, "wmape": 0, "forecast_bias": 0},
    )

    config = {
        "input": {"path": "data/sample/synthetic_sales.csv"},
        "paths": {"processed_data": "data/processed", "models": "models", "reports": "reports"},
        "columns": {"target_columns": ["total_qty"]},
    }
    result = forecasting_pipeline.run_training_pipeline(config)
    assert calls["prepared"] is True
    assert result["metrics"][0]["target"] == "total_qty"


def test_metric_functions_return_expected_values():
    y_true = pd.Series([100, 200, 300])
    y_pred = pd.Series([90, 210, 330])
    assert mae(y_true, y_pred) == 50 / 3
    assert round(rmse(y_true, y_pred), 4) == 19.1485
    assert wmape(y_true, y_pred) == 50 / 600
    assert forecast_bias(y_true, y_pred) == 30 / 600


def test_markdown_report_generation():
    output_path = Path("outputs/test_reports/forecast_report.md")
    generate_markdown_report(
        output_path=output_path,
        dataset_summary={"rows": 10, "columns": ["week_start", "total_qty"], "date_min": "2023-01-02", "date_max": "2023-03-06", "target_sum": 1000},
        quality_report={"summary": {"error_checks": 0, "warning_checks": 0}, "checks": []},
        metrics={"target": "total_qty", "mae": 1, "rmse": 2, "wmape": 0.1, "forecast_bias": -0.02},
        insights=["Demand is stable."],
        recommendations=["Monitor bias weekly."],
    )
    content = output_path.read_text(encoding="utf-8")
    assert "AI Data Analyst Forecasting Report" in content
    assert "| total_qty |" in content
    assert "Monitor bias weekly." in content


def test_config_loading_default_yaml():
    config = load_config("configs/default.yaml")
    assert config["columns"]["target"] == "total_qty"
    assert config["output"]["reports_path"] == "outputs/reports"


def test_default_sample_config_has_enough_history_for_model_frames():
    from src.features import make_model_frame
    from src.preprocess import build_weekly_series, clean_sales_data

    config = load_config("configs/default.yaml")
    raw = pd.read_csv(config["input"]["path"])
    weekly = build_weekly_series(clean_sales_data(raw))
    split = pd.Timestamp(config["validation"]["test_start_date"])
    for target in config["columns"]["target_columns"]:
        frame = make_model_frame(weekly, target)
        assert not frame[frame["week_start"] < split].empty
        assert not frame[frame["week_start"] >= split].empty


def test_forecasting_features_respect_horizon_safe_lags():
    from src.features import make_model_frame

    weekly = pd.DataFrame(
        {
            "week_start": pd.date_range("2024-01-01", periods=40, freq="W-MON"),
            "category": "DRY",
            "brand": "A",
            "whseid": "WH1",
            "total_qty": range(100, 140),
            "total_cbm": range(10, 50),
        }
    )
    configured_lags = [1, 2, 4, 8, 12]
    for horizon in [1, 2, 4]:
        frame = make_model_frame(
            weekly,
            target_col="total_qty",
            forecast_horizon_weeks=horizon,
            lag_weeks=configured_lags,
            rolling_windows=[4],
        )
        lag_columns = [
            column
            for column in frame.columns
            if column.startswith("total_qty_lag_") and "diff" not in column and "pct" not in column
        ]
        assert lag_columns
        assert all(int(column.rsplit("_", 1)[-1]) >= horizon for column in lag_columns)
        assert f"total_qty_lag_{horizon}" in frame.columns


def test_rolling_features_shift_by_forecast_horizon():
    from src.features import add_lag_features

    weekly = pd.DataFrame(
        {
            "week_start": pd.date_range("2024-01-01", periods=8, freq="W-MON"),
            "category": "DRY",
            "brand": "A",
            "whseid": "WH1",
            "total_qty": range(10, 18),
            "total_cbm": range(1, 9),
        }
    )
    featured = add_lag_features(
        weekly,
        target_col="total_qty",
        forecast_horizon_weeks=2,
        lag_weeks=[1, 2, 4],
        rolling_windows=[2],
    )
    expected = weekly["total_qty"].shift(2).rolling(2).mean()
    pd.testing.assert_series_equal(
        featured["total_qty_rolling_mean_2"].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


def test_agent_generates_dataset_summary_and_actions():
    config = load_config("configs/default.yaml")
    agent = ForecastingAgent(config)
    df = pd.DataFrame(
        {
            "ship_date": ["2023-01-02", "2023-01-09"],
            "category": ["DRY", "DRY"],
            "brand": ["A", "A"],
            "whseid": ["WH1", "WH1"],
            "total_qty": [10, 20],
        }
    )
    summary = agent.analyze_dataset(df)
    quality = agent.check_data_quality(df)
    actions = agent.suggest_next_actions(quality)
    assert summary["rows"] == 2
    assert actions
