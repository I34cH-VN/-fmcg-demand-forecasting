from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Holiday:
    name: str
    date: str
    pre_window_days: int


# Dates are Gregorian dates for Vietnam holidays relevant to the 2023-2025
# source data. Lunar holidays vary by year and should be maintained as config
# or calendar data in a real production pipeline.
VIETNAM_HOLIDAYS = [
    Holiday("new_year", "2023-01-01", 14),
    Holiday("tet", "2023-01-22", 28),
    Holiday("hung_kings", "2023-04-29", 14),
    Holiday("reunification_day", "2023-04-30", 14),
    Holiday("labor_day", "2023-05-01", 14),
    Holiday("mid_autumn", "2023-09-29", 21),
    Holiday("national_day", "2023-09-02", 14),
    Holiday("christmas", "2023-12-25", 21),
    Holiday("new_year", "2024-01-01", 14),
    Holiday("tet", "2024-02-10", 28),
    Holiday("hung_kings", "2024-04-18", 14),
    Holiday("reunification_day", "2024-04-30", 14),
    Holiday("labor_day", "2024-05-01", 14),
    Holiday("mid_autumn", "2024-09-17", 21),
    Holiday("national_day", "2024-09-02", 14),
    Holiday("christmas", "2024-12-25", 21),
    Holiday("new_year", "2025-01-01", 14),
    Holiday("tet", "2025-01-29", 28),
    Holiday("hung_kings", "2025-04-07", 14),
    Holiday("reunification_day", "2025-04-30", 14),
    Holiday("labor_day", "2025-05-01", 14),
    Holiday("mid_autumn", "2025-10-06", 21),
    Holiday("national_day", "2025-09-02", 14),
    Holiday("christmas", "2025-12-25", 21),
]


def add_vietnam_holiday_features(df: pd.DataFrame, date_col: str = "week_start") -> pd.DataFrame:
    featured = df.copy()
    date = pd.to_datetime(featured[date_col]).dt.normalize()
    featured["is_tet_season"] = 0
    featured["is_mid_autumn_season"] = 0
    featured["is_national_day"] = 0
    featured["is_holiday_season"] = 0
    nearest_distances = []

    holiday_dates = [pd.Timestamp(holiday.date) for holiday in VIETNAM_HOLIDAYS]
    for current_date in date:
        nearest_distances.append(min(abs((holiday_date - current_date).days) for holiday_date in holiday_dates))

    for holiday in VIETNAM_HOLIDAYS:
        holiday_date = pd.Timestamp(holiday.date)
        days_to_holiday = (holiday_date - date).dt.days
        in_pre_window = days_to_holiday.between(0, holiday.pre_window_days)
        in_event_week = days_to_holiday.between(-6, 0)
        in_season = in_pre_window | in_event_week

        if holiday.name == "tet":
            featured.loc[in_season, "is_tet_season"] = 1
        if holiday.name == "mid_autumn":
            featured.loc[in_season, "is_mid_autumn_season"] = 1
        if holiday.name == "national_day":
            featured.loc[in_season, "is_national_day"] = 1
        featured.loc[in_season, "is_holiday_season"] = 1

    featured["days_to_nearest_holiday"] = np.array(nearest_distances, dtype=int)
    return featured
