from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.agent.forecasting_agent import ForecastingAgent
from src.models.forecasting_pipeline import prepare_weekly_data, run_training_pipeline
from src.preprocess import build_weekly_series, clean_sales_data
from src.utils.config import ROOT_DIR, get_nested, load_config, resolve_project_path


def _load_analysis_frame(config: dict) -> pd.DataFrame:
    input_path = get_nested(config, ["input", "path"])
    if input_path:
        df = pd.read_csv(resolve_project_path(input_path, ROOT_DIR))
        if {"ACTUALSHIPDATE", "Total QTY", "Total CBM"}.issubset(df.columns):
            return build_weekly_series(clean_sales_data(df))
        return df
    processed_path = resolve_project_path(get_nested(config, ["paths", "processed_data"], "data/processed"), ROOT_DIR) / "weekly_sales.csv"
    if processed_path.exists():
        return pd.read_csv(processed_path, parse_dates=["week_start"])
    return prepare_weekly_data(config)


def analyze_command(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    df = _load_analysis_frame(config)
    agent = ForecastingAgent(config)
    dataset_summary = agent.analyze_dataset(df)
    quality_report = agent.check_data_quality(df)
    output_dir = resolve_project_path(get_nested(config, ["output", "path"], "outputs"), ROOT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "data_quality_report.json"
    output_path.write_text(json.dumps({"dataset": dataset_summary, "quality": quality_report}, indent=2), encoding="utf-8")
    print(f"Saved analysis report to {output_path}")


def train_command(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    result = run_training_pipeline(config)
    print(f"Saved metrics to {result['metrics_path']}")


def report_command(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    df = _load_analysis_frame(config)
    agent = ForecastingAgent(config)
    dataset_summary = agent.analyze_dataset(df)
    quality_report = agent.check_data_quality(df)
    reports_dir = resolve_project_path(get_nested(config, ["output", "reports_path"], "outputs/reports"), ROOT_DIR)
    metrics_path = resolve_project_path(get_nested(config, ["paths", "reports"], "reports"), ROOT_DIR) / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else []
    report_path = agent.generate_markdown_report(reports_dir / "forecast_report.md", dataset_summary, quality_report, metrics)
    print(f"Saved markdown report to {report_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI Data Analyst / Forecasting Agent CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, handler in {
        "analyze": analyze_command,
        "train": train_command,
        "report": report_command,
    }.items():
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--config", default="configs/default.yaml")
        subparser.set_defaults(func=handler)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
