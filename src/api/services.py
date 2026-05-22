from __future__ import annotations

import json
from pathlib import Path

from src.agent.forecasting_agent import ForecastingAgent
from src.cli import _load_analysis_frame
from src.models.forecasting_pipeline import run_training_pipeline
from src.utils.config import ROOT_DIR, get_nested, load_config, resolve_project_path


def run_analyze_workflow(config_path: str) -> dict:
    config = load_config(config_path)
    df = _load_analysis_frame(config)
    agent = ForecastingAgent(config)
    dataset_summary = agent.analyze_dataset(df)
    quality_report = agent.check_data_quality(df)
    output_dir = resolve_project_path(get_nested(config, ["output", "path"], "outputs"), ROOT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "data_quality_report.json"
    payload = {"dataset": dataset_summary, "quality": quality_report}
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"data_quality_json": payload, "report_path": str(output_path)}


def run_train_workflow(config_path: str) -> dict:
    config = load_config(config_path)
    result = run_training_pipeline(config)
    return {"metrics_json": result["metrics"], "report_path": result["metrics_path"]}


def run_report_workflow(config_path: str) -> dict:
    config = load_config(config_path)
    df = _load_analysis_frame(config)
    agent = ForecastingAgent(config)
    dataset_summary = agent.analyze_dataset(df)
    quality_report = agent.check_data_quality(df)
    reports_dir = resolve_project_path(get_nested(config, ["output", "reports_path"], "outputs/reports"), ROOT_DIR)
    metrics_path = resolve_project_path(get_nested(config, ["paths", "reports"], "reports"), ROOT_DIR) / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else []
    report_path = agent.generate_markdown_report(
        reports_dir / "forecast_report.md",
        dataset_summary,
        quality_report,
        metrics,
    )
    return {"data_quality_json": {"dataset": dataset_summary, "quality": quality_report}, "metrics_json": metrics, "report_path": str(report_path)}
