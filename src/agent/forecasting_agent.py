from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.agent.base_llm import BaseLLM
from src.agent.llm_client import ExternalLLM
from src.agent.mock_llm import MockLLM
from src.data.quality import QualityConfig, analyze_data_quality, quality_report_to_markdown
from src.reports.markdown_report import generate_markdown_report as build_markdown_report
from src.utils.config import get_nested


def build_default_llm() -> BaseLLM:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    import os

    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if provider == "mock":
        return MockLLM()
    if (provider and provider != "mock") or os.getenv("LLM_API_KEY"):
        return ExternalLLM()
    return MockLLM()


class ForecastingAgent:
    def __init__(self, config: dict[str, Any], llm: BaseLLM | None = None) -> None:
        self.config = config
        self.llm = llm or build_default_llm()

    def analyze_dataset(self, df: pd.DataFrame) -> dict:
        configured_date = get_nested(self.config, ["columns", "date"], "ship_date")
        date_column = configured_date if configured_date in df.columns else get_nested(self.config, ["columns", "week_start"], configured_date)
        target_column = get_nested(self.config, ["columns", "target"], "total_qty")
        return {
            "rows": int(len(df)),
            "columns": list(df.columns),
            "date_min": str(pd.to_datetime(df[date_column], errors="coerce").min()) if date_column in df.columns else None,
            "date_max": str(pd.to_datetime(df[date_column], errors="coerce").max()) if date_column in df.columns else None,
            "target_sum": float(pd.to_numeric(df[target_column], errors="coerce").sum()) if target_column in df.columns else None,
        }

    def check_data_quality(self, df: pd.DataFrame) -> dict:
        configured_date = get_nested(self.config, ["columns", "date"], "ship_date")
        date_column = configured_date if configured_date in df.columns else get_nested(self.config, ["columns", "week_start"], configured_date)
        config = QualityConfig(
            date_column=date_column,
            target_column=get_nested(self.config, ["columns", "target"], "total_qty"),
            group_columns=get_nested(self.config, ["columns", "id_columns"], ["category", "brand", "whseid"]),
            week_start_column=get_nested(self.config, ["columns", "week_start"], None),
        )
        return analyze_data_quality(
            df,
            config,
            split_date=get_nested(self.config, ["validation", "test_start_date"]),
        )

    def explain_forecast_results(self, metrics: list[dict] | dict) -> str:
        return self.llm.generate(f"Explain these forecast metrics for a business audience: {metrics}")

    def generate_business_insights(self, dataset_summary: dict, quality_report: dict, metrics: list[dict] | dict | None = None) -> list[str]:
        insights = [
            f"Dataset contains {dataset_summary.get('rows', 0):,} rows across {len(dataset_summary.get('columns', []))} columns.",
            f"Data quality produced {quality_report.get('summary', {}).get('warning_checks', 0)} warning checks and {quality_report.get('summary', {}).get('error_checks', 0)} error checks.",
        ]
        if metrics:
            metrics_list = metrics if isinstance(metrics, list) else [metrics]
            for item in metrics_list:
                target = item.get("target", "target")
                wmape = item.get("wmape")
                bias = item.get("forecast_bias")
                if wmape is not None:
                    insights.append(f"{target} WMAPE is {wmape:.2%}; bias is {bias:.2%}." if bias is not None else f"{target} WMAPE is {wmape:.2%}.")
        return insights

    def suggest_next_actions(
        self,
        quality_report: dict,
        metrics: list[dict] | dict | None = None,
        dataset_summary: dict | None = None,
    ) -> list[str]:
        actions = []
        if quality_report.get("summary", {}).get("error_checks", 0):
            actions.append("Fix error-level data quality checks before trusting forecast output.")
        if quality_report.get("summary", {}).get("warning_checks", 0):
            actions.append("Review warning-level quality checks and annotate accepted business exceptions.")
        if metrics:
            actions.append("Compare WMAPE and bias by brand/category/warehouse to identify unstable planning segments.")
        actions.append(
            self.llm.generate(
                "You are an FMCG demand forecasting analyst. "
                f"Dataset summary: {dataset_summary or {}}. "
                f"Data quality report: {quality_report}. "
                f"Metrics: {metrics or []}. "
                "Give 5 concrete next actions for operations and ML improvement. "
                "Do not ask follow-up questions."
            )
        )
        return actions

    def generate_markdown_report(
        self,
        output_path: str | Path,
        dataset_summary: dict,
        quality_report: dict,
        metrics: list[dict] | dict | None = None,
    ) -> Path:
        insights = self.generate_business_insights(dataset_summary, quality_report, metrics)
        recommendations = self.suggest_next_actions(quality_report, metrics, dataset_summary)
        return build_markdown_report(
            output_path=output_path,
            dataset_summary=dataset_summary,
            quality_report=quality_report,
            metrics=metrics or [],
            insights=insights,
            recommendations=recommendations,
            quality_markdown=quality_report_to_markdown(quality_report),
        )
