from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from holidays import add_vietnam_holiday_features
from validation import validate_weekly_data


LOGGER = logging.getLogger(__name__)

ID_COLUMNS = ["category", "brand", "whseid"]
TARGET_COLUMNS = ["total_qty", "total_cbm"]


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


def add_lag_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    featured = df.sort_values(ID_COLUMNS + ["week_start"]).copy()
    group = featured.groupby(ID_COLUMNS, observed=True)[target_col]

    for lag in [1, 2, 4, 8, 12]:
        featured[f"{target_col}_lag_{lag}"] = group.shift(lag)

    for window in [4, 8, 12]:
        shifted = group.shift(1)
        featured[f"{target_col}_rolling_mean_{window}"] = (
            shifted.groupby([featured[col] for col in ID_COLUMNS], observed=True)
            .rolling(window)
            .mean()
            .reset_index(level=list(range(len(ID_COLUMNS))), drop=True)
        )

    # Leakage guard:
    # Current-period diff/pct_change would use y_t and lets the model reconstruct
    # the target from lag_1 + diff_1. These features are shifted fully into the
    # past, so each row only sees values available before the forecast week.
    lag_1 = featured[f"{target_col}_lag_1"]
    lag_2 = featured[f"{target_col}_lag_2"]
    featured[f"{target_col}_lag_diff_1"] = lag_1 - lag_2
    featured[f"{target_col}_lag_pct_change_1"] = (
        (lag_1 - lag_2) / lag_2.replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)

    return featured


def make_model_frame(df: pd.DataFrame, target_col: str = "total_qty") -> pd.DataFrame:
    LOGGER.info("Creating model frame for target=%s", target_col)
    validate_weekly_data(df)
    featured = add_calendar_features(df)
    featured = add_lag_features(featured, target_col)
    before_drop = len(featured)
    featured = featured.dropna().reset_index(drop=True)
    LOGGER.info("Dropped %s rows without enough lag history", before_drop - len(featured))
    return featured


def get_feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {"week_start", "total_qty", "total_cbm"}
    return [column for column in frame.columns if column not in excluded]
