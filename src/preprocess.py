from pathlib import Path
import logging

import pandas as pd

try:
    from src.validation import (
        CLEAN_REQUIRED_COLUMNS,
        RAW_REQUIRED_COLUMNS,
        validate_clean_data,
        validate_raw_data,
        validate_weekly_data,
    )
except ImportError:
    from validation import (
        CLEAN_REQUIRED_COLUMNS,
        RAW_REQUIRED_COLUMNS,
        validate_clean_data,
        validate_raw_data,
        validate_weekly_data,
    )


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def load_raw_data(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    LOGGER.info("Loading raw CSV files from %s", raw_dir)
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    frames = []
    for file in files:
        frame = pd.read_csv(file)
        frame["source_file"] = file.name
        frames.append(frame)
        LOGGER.info("Loaded %s rows from %s", len(frame), file.name)

    raw = pd.concat(frames, ignore_index=True)
    validate_raw_data(raw, required_columns=RAW_REQUIRED_COLUMNS)
    LOGGER.info("Loaded %s raw rows from %s files", len(raw), len(files))
    return raw


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized.columns = (
        normalized.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )
    return normalized.rename(columns={"actualshipdate": "ship_date"})


def clean_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    LOGGER.info("Cleaning raw sales data")
    cleaned = _normalize_columns(df)

    missing_before = cleaned[["ship_date", "category", "whseid", "brand"]].isna().sum()
    LOGGER.info("Missing critical fields before cleaning: %s", missing_before.to_dict())

    cleaned["ship_date"] = pd.to_datetime(cleaned["ship_date"], errors="coerce")
    cleaned["total_qty"] = pd.to_numeric(cleaned["total_qty"], errors="coerce")
    cleaned["total_cbm"] = pd.to_numeric(cleaned["total_cbm"], errors="coerce")

    critical_columns = ["ship_date", "category", "whseid", "brand", "total_qty", "total_cbm"]
    before_drop = len(cleaned)
    cleaned = cleaned.dropna(subset=critical_columns).copy()
    LOGGER.info("Dropped %s rows with missing critical values", before_drop - len(cleaned))

    for column in ["category", "whseid", "brand"]:
        cleaned[column] = cleaned[column].str.strip().str.upper()
        cleaned.loc[cleaned[column] == "", column] = pd.NA

    before_empty_drop = len(cleaned)
    cleaned = cleaned.dropna(subset=["category", "whseid", "brand"]).copy()
    LOGGER.info("Dropped %s rows with blank categorical values", before_empty_drop - len(cleaned))

    before_non_negative = len(cleaned)
    cleaned = cleaned[(cleaned["total_qty"] >= 0) & (cleaned["total_cbm"] >= 0)].copy()
    LOGGER.info("Dropped %s rows with negative quantity or CBM", before_non_negative - len(cleaned))

    business_key = ["ship_date", "category", "whseid", "brand", "total_qty", "total_cbm"]
    duplicate_count = int(cleaned.duplicated(subset=business_key).sum())
    cleaned = cleaned.drop_duplicates(subset=business_key).copy()
    LOGGER.info("Dropped %s duplicate rows using business key %s", duplicate_count, business_key)

    missing_after = cleaned[["ship_date", "category", "whseid", "brand"]].isna().sum()
    LOGGER.info("Missing critical fields after cleaning: %s", missing_after.to_dict())

    cleaned = cleaned.sort_values(["ship_date", "category", "brand", "whseid"]).reset_index(drop=True)
    validate_clean_data(cleaned, required_columns=CLEAN_REQUIRED_COLUMNS)
    return cleaned


def build_weekly_series(df: pd.DataFrame) -> pd.DataFrame:
    LOGGER.info("Aggregating cleaned sales data to Monday-start weekly series")
    weekly = df.copy()
    weekly["week_start"] = (
        weekly["ship_date"] - pd.to_timedelta(weekly["ship_date"].dt.weekday, unit="D")
    ).dt.normalize()
    if not (weekly["week_start"].dt.weekday == 0).all():
        raise ValueError("week_start must always be Monday.")

    weekly = (
        weekly.groupby(["week_start", "category", "brand", "whseid"], as_index=False)
        .agg(total_qty=("total_qty", "sum"), total_cbm=("total_cbm", "sum"))
        .sort_values(["category", "brand", "whseid", "week_start"])
        .reset_index(drop=True)
    )

    validate_weekly_data(weekly)
    LOGGER.info("Built %s weekly rows", len(weekly))
    return weekly


def main() -> None:
    configure_logging()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_raw_data()
    cleaned = clean_sales_data(raw)
    weekly = build_weekly_series(cleaned)
    validate_weekly_data(weekly, min_date_span_days=365)

    cleaned.to_csv(PROCESSED_DIR / "sales_cleaned.csv", index=False)
    weekly.to_csv(PROCESSED_DIR / "weekly_sales.csv", index=False)

    LOGGER.info("Cleaned rows: %s", f"{len(cleaned):,}")
    LOGGER.info("Weekly rows: %s", f"{len(weekly):,}")
    LOGGER.info("Date range: %s to %s", cleaned["ship_date"].min().date(), cleaned["ship_date"].max().date())
    LOGGER.info("Saved cleaned and weekly files to %s", PROCESSED_DIR)


if __name__ == "__main__":
    main()
