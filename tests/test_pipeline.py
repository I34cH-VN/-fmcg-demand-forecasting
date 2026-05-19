from pathlib import Path
import sys

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))
sys.path.insert(0, str(ROOT_DIR / "app"))

from dashboard_utils import safe_peak_week
from evaluate import wmape
from features import make_model_frame
from preprocess import build_weekly_series, clean_sales_data


def test_week_start_is_monday():
    raw = pd.DataFrame(
        {
            "ACTUALSHIPDATE": ["2025-01-01", "2025-01-05"],
            "CATEGORY": ["dry", "dry"],
            "WHSEID": ["bkd1", "bkd1"],
            "BRAND": ["afc", "afc"],
            "Total QTY": [10, 20],
            "Total CBM": [1.0, 2.0],
        }
    )
    weekly = build_weekly_series(clean_sales_data(raw))
    assert (weekly["week_start"].dt.weekday == 0).all()


def test_lag_features_do_not_reconstruct_current_target():
    weeks = pd.date_range("2024-01-01", periods=20, freq="W-MON")
    weekly = pd.DataFrame(
        {
            "week_start": weeks,
            "category": "DRY",
            "brand": "AFC",
            "whseid": "BKD1",
            "total_qty": range(100, 120),
            "total_cbm": range(10, 30),
        }
    )
    frame = make_model_frame(weekly, "total_qty")
    assert "total_qty_diff_1" not in frame.columns
    assert "total_qty_pct_change_1" not in frame.columns
    assert "total_qty_lag_diff_1" in frame.columns
    reconstructed = frame["total_qty_lag_1"] + frame["total_qty_lag_diff_1"]
    assert not reconstructed.equals(frame["total_qty"])


def test_missing_categorical_is_not_stringified_nan():
    raw = pd.DataFrame(
        {
            "ACTUALSHIPDATE": ["2025-01-01", "2025-01-02"],
            "CATEGORY": ["dry", None],
            "WHSEID": ["bkd1", "bkd1"],
            "BRAND": ["afc", "afc"],
            "Total QTY": [10, 20],
            "Total CBM": [1.0, 2.0],
        }
    )
    cleaned = clean_sales_data(raw)
    assert "NAN" not in set(cleaned["category"])
    assert len(cleaned) == 1


def test_duplicate_business_key_removed():
    raw = pd.DataFrame(
        {
            "ACTUALSHIPDATE": ["2025-01-01", "2025-01-01"],
            "CATEGORY": ["dry", "dry"],
            "WHSEID": ["bkd1", "bkd1"],
            "BRAND": ["afc", "afc"],
            "Total QTY": [10, 10],
            "Total CBM": [1.0, 1.0],
            "source_file": ["a.csv", "b.csv"],
        }
    )
    cleaned = clean_sales_data(raw)
    assert len(cleaned) == 1


def test_wmape_zero_denominator_returns_zero():
    assert wmape(pd.Series([0, 0]), pd.Series([1, 2])) == 0.0


def test_safe_peak_week_empty_dataframe():
    assert safe_peak_week(pd.DataFrame(), "predicted_total_qty") is None
