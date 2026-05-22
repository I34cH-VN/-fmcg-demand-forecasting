from __future__ import annotations

import logging

import numpy as np
import pandas as pd

try:
    from src.holidays import add_vietnam_holiday_features
    from src.validation import validate_weekly_data
except ImportError:
    from holidays import add_vietnam_holiday_features
    from validation import validate_weekly_data


LOGGER = logging.getLogger(__name__)

ID_COLUMNS = ["category", "brand", "whseid"]
TARGET_COLUMNS = ["total_qty", "total_cbm"]
DEFAULT_LAG_WEEKS = [1, 2, 4, 8, 12]
DEFAULT_ROLLING_WINDOWS = [4, 8, 12]


def add_calendar_features(df: pd.DataFrame, date_col: str = "week_start") -> pd.DataFrame:
    featured = df.copy()
    date = pd.to_datetime(featured[date_col])

    featured["year"] = date.dt.year
    featured["month"] = date.dt.month
    featured["quarter"] = date.dt.quarter
    featured["weekofyear"] = date.dt.isocalendar().week.astype(int)
    featured["is_month_start"] = date.dt.is_month_start.astype(int)
    featured["is_month_end"] = date.dt.is_month_end.astype(int)
    featured["is_year_end"] = featured["month"].isin([11, 12]).astype(int)

    return add_vietnam_holiday_features(featured, date_col=date_col)


def get_safe_lag_weeks(lag_weeks: list[int] | None = None, forecast_horizon_weeks: int = 1) -> list[int]:
    horizon = max(int(forecast_horizon_weeks), 1)
    configured_lags = lag_weeks or DEFAULT_LAG_WEEKS
    safe_lags = sorted({int(lag) for lag in configured_lags if int(lag) >= horizon})
    if horizon not in safe_lags:
        safe_lags.insert(0, horizon)
    if len(safe_lags) < 2:
        safe_lags.append(horizon + 1)
    return sorted(set(safe_lags))


def add_lag_features(
    df: pd.DataFrame,
    target_col: str,
    forecast_horizon_weeks: int = 1,
    lag_weeks: list[int] | None = None,
    rolling_windows: list[int] | None = None,
) -> pd.DataFrame:
    featured = df.sort_values(ID_COLUMNS + ["week_start"]).copy()
    group = featured.groupby(ID_COLUMNS, observed=True)[target_col]
    horizon = max(int(forecast_horizon_weeks), 1)
    safe_lags = get_safe_lag_weeks(lag_weeks, horizon)
    windows = rolling_windows or DEFAULT_ROLLING_WINDOWS

    for lag in safe_lags:
        featured[f"{target_col}_lag_{lag}"] = group.shift(lag)

    for window in windows:
        shifted = group.shift(horizon)
        featured[f"{target_col}_rolling_mean_{window}"] = (
            shifted.groupby([featured[col] for col in ID_COLUMNS], observed=True)
            .rolling(int(window))
            .mean()
            .reset_index(level=list(range(len(ID_COLUMNS))), drop=True)
        )

    # Leakage guard:
    # Current-period diff/pct_change would use y_t and lets the model reconstruct
    # the target from lag_1 + diff_1. These features are shifted fully into the
    # past, so each row only sees values available at least horizon weeks
    # before the forecast week.
    primary_lag, secondary_lag = safe_lags[:2]
    lag_primary = featured[f"{target_col}_lag_{primary_lag}"]
    lag_secondary = featured[f"{target_col}_lag_{secondary_lag}"]
    featured[f"{target_col}_lag_diff_{primary_lag}"] = lag_primary - lag_secondary
    featured[f"{target_col}_lag_pct_change_{primary_lag}"] = (
        (lag_primary - lag_secondary) / lag_secondary.replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)

    return featured


def make_model_frame(
    df: pd.DataFrame,
    target_col: str = "total_qty",
    forecast_horizon_weeks: int = 1,
    lag_weeks: list[int] | None = None,
    rolling_windows: list[int] | None = None,
) -> pd.DataFrame:
    LOGGER.info("Creating model frame for target=%s", target_col)
    validate_weekly_data(df)
    featured = add_calendar_features(df)
    featured = add_lag_features(
        featured,
        target_col,
        forecast_horizon_weeks=forecast_horizon_weeks,
        lag_weeks=lag_weeks,
        rolling_windows=rolling_windows,
    )
    before_drop = len(featured)
    featured = featured.dropna().reset_index(drop=True)
    LOGGER.info("Dropped %s rows without enough lag history", before_drop - len(featured))
    return featured


def get_feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {"week_start", "total_qty", "total_cbm"}
    return [column for column in frame.columns if column not in excluded]
