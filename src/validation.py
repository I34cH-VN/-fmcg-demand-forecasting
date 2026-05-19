from __future__ import annotations

import logging
from collections.abc import Iterable

import pandas as pd


LOGGER = logging.getLogger(__name__)

RAW_REQUIRED_COLUMNS = [
    "ACTUALSHIPDATE",
    "CATEGORY",
    "WHSEID",
    "BRAND",
    "Total QTY",
    "Total CBM",
]
CLEAN_REQUIRED_COLUMNS = ["ship_date", "category", "whseid", "brand", "total_qty", "total_cbm"]
WEEKLY_REQUIRED_COLUMNS = ["week_start", "category", "brand", "whseid", "total_qty", "total_cbm"]


def _missing_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> list[str]:
    return [column for column in required_columns if column not in df.columns]


def validate_required_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> None:
    missing = _missing_columns(df, required_columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def validate_raw_data(df: pd.DataFrame, required_columns: Iterable[str] = RAW_REQUIRED_COLUMNS) -> None:
    validate_required_columns(df, required_columns)
    parsed_dates = pd.to_datetime(df["ACTUALSHIPDATE"], errors="coerce")
    invalid_dates = int(parsed_dates.isna().sum())
    if invalid_dates:
        LOGGER.warning("Raw data contains %s unparseable ACTUALSHIPDATE values", invalid_dates)
    LOGGER.info("Raw data validation completed")


def validate_clean_data(df: pd.DataFrame, required_columns: Iterable[str] = CLEAN_REQUIRED_COLUMNS) -> None:
    validate_required_columns(df, required_columns)
    critical = ["ship_date", "category", "whseid", "brand"]
    missing = df[critical].isna().sum()
    if missing.any():
        raise ValueError(f"Clean data still has missing critical values: {missing.to_dict()}")
    if df[["category", "whseid", "brand"]].isin(["nan", "NaN", "NAN", ""]).any().any():
        raise ValueError("Clean data contains stringified missing categorical values.")
    if (df["total_qty"] < 0).any() or (df["total_cbm"] < 0).any():
        raise ValueError("Clean data contains negative total_qty or total_cbm.")
    duplicate_key = ["ship_date", "category", "whseid", "brand", "total_qty", "total_cbm"]
    duplicate_count = int(df.duplicated(subset=duplicate_key).sum())
    if duplicate_count:
        raise ValueError(f"Clean data contains {duplicate_count} duplicate business-key rows.")
    LOGGER.info("Clean data validation completed")


def validate_weekly_data(df: pd.DataFrame, min_date_span_days: int = 0) -> None:
    validate_required_columns(df, WEEKLY_REQUIRED_COLUMNS)
    if df.empty:
        raise ValueError("Weekly data is empty.")
    week_start = pd.to_datetime(df["week_start"], errors="coerce")
    if week_start.isna().any():
        raise ValueError("Weekly data contains unparseable week_start values.")
    if not (week_start.dt.weekday == 0).all():
        raise ValueError("week_start must always be Monday.")
    if (df["total_qty"] < 0).any() or (df["total_cbm"] < 0).any():
        raise ValueError("Weekly data contains negative total_qty or total_cbm.")
    duplicate_key = ["week_start", "category", "brand", "whseid"]
    duplicate_count = int(df.duplicated(subset=duplicate_key).sum())
    if duplicate_count:
        raise ValueError(f"Weekly data contains {duplicate_count} duplicate business-key rows.")
    date_span_days = (week_start.max() - week_start.min()).days
    if min_date_span_days and date_span_days < min_date_span_days:
        raise ValueError(
            f"Weekly data covers {date_span_days} days, less than required {min_date_span_days} days."
        )
    if df[["category", "brand", "whseid"]].isna().any().any():
        raise ValueError("Weekly data contains missing category, brand, or whseid.")
    LOGGER.info("Weekly data validation completed")
