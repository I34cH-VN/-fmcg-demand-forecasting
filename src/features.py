from __future__ import annotations

import pandas as pd


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

    # A practical proxy for Vietnamese FMCG Tet demand buildup.
    featured["is_tet_season"] = featured["month"].isin([1, 2]).astype(int)
    featured["is_year_end"] = featured["month"].isin([11, 12]).astype(int)

    return featured


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

    featured[f"{target_col}_diff_1"] = group.diff(1)
    featured[f"{target_col}_pct_change_1"] = group.pct_change(1).replace([float("inf"), -float("inf")], 0)
    return featured


def make_model_frame(df: pd.DataFrame, target_col: str = "total_qty") -> pd.DataFrame:
    featured = add_calendar_features(df)
    featured = add_lag_features(featured, target_col)
    featured = featured.dropna().reset_index(drop=True)
    return featured


def get_feature_columns(frame: pd.DataFrame, target_col: str) -> list[str]:
    excluded = {"week_start", "total_qty", "total_cbm"}
    return [column for column in frame.columns if column not in excluded or column == target_col]
