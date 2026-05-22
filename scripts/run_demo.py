from __future__ import annotations

import json
from pathlib import Path
import sys
import warnings


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.agent.forecasting_agent import ForecastingAgent
from src.cli import _load_analysis_frame
from src.models.forecasting_pipeline import run_training_pipeline
from src.utils.config import load_config, resolve_project_path


CONFIG_PATH = ROOT_DIR / "configs" / "default.yaml"


def main() -> None:
    warnings.filterwarnings(
        "ignore",
        message="X does not have valid feature names.*",
        category=UserWarning,
    )
    config = load_config(CONFIG_PATH)
    agent = ForecastingAgent(config)

    df = _load_analysis_frame(config)
    dataset_summary = agent.analyze_dataset(df)
    quality_report = agent.check_data_quality(df)

    outputs_dir = resolve_project_path(config["output"]["path"], ROOT_DIR)
    reports_dir = resolve_project_path(config["output"]["reports_path"], ROOT_DIR)
    pipeline_reports_dir = resolve_project_path(config["paths"]["reports"], ROOT_DIR)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    quality_path = outputs_dir / "data_quality_report.json"
    quality_path.write_text(
        json.dumps({"dataset": dataset_summary, "quality": quality_report}, indent=2),
        encoding="utf-8",
    )

    training_result = run_training_pipeline(config)
    metrics_path = Path(training_result["metrics_path"])
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else []
    report_path = agent.generate_markdown_report(
        reports_dir / "forecast_report.md",
        dataset_summary,
        quality_report,
        metrics,
    )

    print("Demo completed successfully.")
    print(f"Data quality report: {quality_path.relative_to(ROOT_DIR)}")
    print(f"Metrics: {(pipeline_reports_dir / 'metrics.json').relative_to(ROOT_DIR)}")
    print(f"Forecast report: {report_path.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
