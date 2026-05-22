from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class QualityConfig:
    date_column: str
    target_column: str
    group_columns: list[str]
    week_start_column: str | None = None


def _issue(name: str, severity: str, count: int, message: str, sample: list[dict[str, Any]] | None = None) -> dict:
    return {
        "check": name,
        "severity": severity,
        "count": int(count),
        "message": message,
        "sample": sample or [],
    }


def check_missing_values(df: pd.DataFrame) -> dict:
    missing = df.isna().sum()
    details = {column: int(count) for column, count in missing.items() if count > 0}
    return {
        "check": "missing_values",
        "severity": "warning" if details else "info",
        "count": int(sum(details.values())),
        "message": "Missing values found." if details else "No missing values found.",
        "details": details,
    }


def check_duplicate_rows(df: pd.DataFrame, subset: list[str] | None = None) -> dict:
    duplicate_count = int(df.duplicated(subset=subset).sum())
    return _issue(
        "duplicated_rows",
        "warning" if duplicate_count else "info",
        duplicate_count,
        f"{duplicate_count} duplicate rows found." if duplicate_count else "No duplicate rows found.",
    )


def check_invalid_dates(df: pd.DataFrame, date_column: str) -> dict:
    if date_column not in df.columns:
        return _issue("invalid_dates", "error", len(df), f"Date column '{date_column}' is missing.")
    parsed = pd.to_datetime(df[date_column], errors="coerce")
    invalid = df[parsed.isna() & df[date_column].notna()]
    return _issue(
        "invalid_dates",
        "error" if len(invalid) else "info",
        len(invalid),
        f"{len(invalid)} invalid dates found in {date_column}." if len(invalid) else "No invalid dates found.",
        invalid.head(5).astype(str).to_dict(orient="records"),
    )


def check_negative_values(df: pd.DataFrame, numeric_columns: list[str]) -> dict:
    details = {}
    for column in numeric_columns:
        if column in df.columns:
            details[column] = int((pd.to_numeric(df[column], errors="coerce") < 0).sum())
    details = {column: count for column, count in details.items() if count > 0}
    return {
        "check": "negative_values",
        "severity": "error" if details else "info",
        "count": int(sum(details.values())),
        "message": "Negative sales/quantity values found." if details else "No negative values found.",
        "details": details,
    }


def check_outliers_iqr(df: pd.DataFrame, target_column: str) -> dict:
    if target_column not in df.columns:
        return _issue("outliers", "error", len(df), f"Target column '{target_column}' is missing.")
    values = pd.to_numeric(df[target_column], errors="coerce").dropna()
    if values.empty:
        return _issue("outliers", "warning", 0, f"No numeric values available in {target_column}.")
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return _issue("outliers", "info", 0, "No IQR outliers found.")
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = pd.to_numeric(df[target_column], errors="coerce").lt(lower) | pd.to_numeric(df[target_column], errors="coerce").gt(upper)
    count = int(mask.sum())
    return _issue(
        "outliers",
        "warning" if count else "info",
        count,
        f"{count} IQR outliers found in {target_column}." if count else "No IQR outliers found.",
        df.loc[mask].head(5).astype(str).to_dict(orient="records"),
    )


def check_date_gaps(
    df: pd.DataFrame,
    date_column: str,
    group_columns: list[str] | None = None,
    expected_freq: str = "W-MON",
) -> dict:
    if date_column not in df.columns:
        return _issue("date_gaps", "error", 0, f"Date column '{date_column}' is missing.")
    frame = df.copy()
    frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce")
    frame = frame.dropna(subset=[date_column])
    if frame.empty:
        return _issue("date_gaps", "warning", 0, "No valid dates available for gap detection.")

    groups = group_columns or []
    available_groups = [column for column in groups if column in frame.columns]
    rows = []
    iterator = frame.groupby(available_groups, dropna=False) if available_groups else [((), frame)]
    for keys, group in iterator:
        dates = pd.Series(group[date_column].drop_duplicates().sort_values())
        if len(dates) < 2:
            continue
        expected = pd.date_range(dates.min(), dates.max(), freq=expected_freq)
        missing_dates = sorted(set(expected) - set(dates))
        if missing_dates:
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = dict(zip(available_groups, keys))
            row["missing_periods"] = len(missing_dates)
            row["first_missing_date"] = missing_dates[0].date().isoformat()
            rows.append(row)
    count = int(sum(row["missing_periods"] for row in rows))
    return {
        "check": "date_gaps",
        "severity": "warning" if count else "info",
        "count": count,
        "message": f"{count} missing date periods found." if count else "No date gaps found.",
        "sample": rows[:5],
    }


def check_week_start_monday(df: pd.DataFrame, week_start_column: str) -> dict:
    if week_start_column not in df.columns:
        return _issue("week_start_monday", "warning", 0, f"Column '{week_start_column}' is not present.")
    dates = pd.to_datetime(df[week_start_column], errors="coerce")
    non_monday = int((dates.notna() & (dates.dt.weekday != 0)).sum())
    return _issue(
        "week_start_monday",
        "error" if non_monday else "info",
        non_monday,
        f"{non_monday} week_start values are not Monday." if non_monday else "All week_start values are Monday.",
    )


def check_train_test_leakage(
    df: pd.DataFrame,
    date_column: str,
    split_date: str | pd.Timestamp | None,
    feature_columns: list[str] | None = None,
    target_column: str | None = None,
) -> dict:
    risks = []
    if split_date is not None and date_column in df.columns:
        dates = pd.to_datetime(df[date_column], errors="coerce")
        split = pd.Timestamp(split_date)
        if not (dates < split).any() or not (dates >= split).any():
            risks.append("Time split creates an empty train or test side.")
    if target_column and feature_columns:
        target_tokens = {target_column.lower(), target_column.lower().replace("total_", "")}
        leakage_features = []
        for column in feature_columns:
            lowered = column.lower()
            if lowered == target_column.lower():
                leakage_features.append(column)
            if any(token in lowered for token in target_tokens) and not any(safe in lowered for safe in ["lag", "rolling", "history"]):
                leakage_features.append(column)
        if leakage_features:
            risks.append(f"Potential target-derived current-period features: {sorted(set(leakage_features))}.")
    return {
        "check": "train_test_leakage_risk",
        "severity": "warning" if risks else "info",
        "count": len(risks),
        "message": "Leakage risks found." if risks else "No obvious leakage risks found.",
        "details": risks,
    }


def analyze_data_quality(
    df: pd.DataFrame,
    config: QualityConfig,
    split_date: str | pd.Timestamp | None = None,
    feature_columns: list[str] | None = None,
) -> dict:
    date_column = config.week_start_column or config.date_column
    numeric_columns = [config.target_column]
    if "total_cbm" in df.columns and config.target_column != "total_cbm":
        numeric_columns.append("total_cbm")
    checks = [
        check_missing_values(df),
        check_duplicate_rows(df),
        check_invalid_dates(df, date_column),
        check_negative_values(df, numeric_columns),
        check_outliers_iqr(df, config.target_column),
        check_date_gaps(df, date_column, config.group_columns),
        check_train_test_leakage(df, date_column, split_date, feature_columns, config.target_column),
    ]
    if config.week_start_column:
        checks.append(check_week_start_monday(df, config.week_start_column))
    severities = [check["severity"] for check in checks]
    return {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "summary": {
            "error_checks": int(sum(severity == "error" for severity in severities)),
            "warning_checks": int(sum(severity == "warning" for severity in severities)),
            "info_checks": int(sum(severity == "info" for severity in severities)),
        },
        "checks": checks,
    }


def quality_report_to_markdown(report: dict) -> str:
    lines = [
        "# Data Quality Report",
        "",
        f"- Rows: {report.get('rows', 0):,}",
        f"- Columns: {len(report.get('columns', []))}",
        f"- Error checks: {report.get('summary', {}).get('error_checks', 0)}",
        f"- Warning checks: {report.get('summary', {}).get('warning_checks', 0)}",
        "",
        "## Checks",
        "",
    ]
    for check in report.get("checks", []):
        lines.append(f"- **{check['check']}** [{check['severity']}]: {check['message']}")
    return "\n".join(lines) + "\n"
