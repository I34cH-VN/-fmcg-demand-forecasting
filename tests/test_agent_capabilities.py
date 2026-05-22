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
