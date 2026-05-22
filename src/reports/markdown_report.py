from __future__ import annotations

from pathlib import Path
from typing import Any


def _format_metrics(metrics: list[dict[str, Any]] | dict[str, Any]) -> list[str]:
    metrics_list = metrics if isinstance(metrics, list) else [metrics]
    rows = []
    for item in metrics_list:
        if not item:
            continue
        target = item.get("target", "target")
        rows.append(
            "| {target} | {mae:.4f} | {rmse:.4f} | {wmape:.2%} | {bias:.2%} |".format(
                target=target,
                mae=float(item.get("mae", 0.0)),
                rmse=float(item.get("rmse", 0.0)),
                wmape=float(item.get("wmape", 0.0)),
                bias=float(item.get("forecast_bias", 0.0)),
            )
        )
    return rows


def generate_markdown_report(
    output_path: str | Path,
    dataset_summary: dict[str, Any],
    quality_report: dict[str, Any],
    metrics: list[dict[str, Any]] | dict[str, Any],
    insights: list[str],
    recommendations: list[str],
    quality_markdown: str | None = None,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metric_rows = _format_metrics(metrics)
    lines = [
        "# AI Data Analyst Forecasting Report",
        "",
        "## Dataset Summary",
        "",
        f"- Rows: {dataset_summary.get('rows', 0):,}",
        f"- Columns: {len(dataset_summary.get('columns', []))}",
        f"- Date range: {dataset_summary.get('date_min')} to {dataset_summary.get('date_max')}",
        f"- Target sum: {dataset_summary.get('target_sum')}",
        "",
        "## Data Quality Issues",
        "",
        f"- Error checks: {quality_report.get('summary', {}).get('error_checks', 0)}",
        f"- Warning checks: {quality_report.get('summary', {}).get('warning_checks', 0)}",
        "",
    ]
    for check in quality_report.get("checks", []):
        if check.get("severity") in {"error", "warning"}:
            lines.append(f"- {check['check']}: {check['message']}")
    if not any(check.get("severity") in {"error", "warning"} for check in quality_report.get("checks", [])):
        lines.append("- No error or warning checks found.")

    lines.extend(["", "## Model Metrics", ""])
    if metric_rows:
        lines.extend(["| Target | MAE | RMSE | WMAPE | Bias |", "| --- | ---: | ---: | ---: | ---: |"])
        lines.extend(metric_rows)
    else:
        lines.append("- Metrics are not available yet.")

    lines.extend(["", "## Top Insights", ""])
    lines.extend(f"- {item}" for item in insights)
    lines.extend(["", "## Business Recommendations", ""])
    lines.extend(f"- {item}" for item in recommendations)
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- Mock LLM output is deterministic unless OpenAI integration is configured.",
            "- Forecast quality depends on historical data coverage and stable demand patterns.",
            "- Promotion, price, stockout, and customer signals are not included by default.",
            "",
            "## Next Steps",
            "",
            "- Add direct multi-horizon forecasting when business use cases require longer planning windows.",
            "- Add model registry, drift monitoring, and scheduled retraining for production readiness.",
        ]
    )
    if quality_markdown:
        lines.extend(["", "## Full Quality Report", "", quality_markdown.strip()])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
