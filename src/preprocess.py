from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"


def load_raw_data(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    frames = []
    for file in files:
        frame = pd.read_csv(file)
        frame["source_file"] = file.name
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def clean_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = (
        cleaned.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    rename_map = {
        "actualshipdate": "ship_date",
        "total_qty": "total_qty",
        "total_cbm": "total_cbm",
    }
    cleaned = cleaned.rename(columns=rename_map)

    cleaned["ship_date"] = pd.to_datetime(cleaned["ship_date"], errors="coerce")
    cleaned["category"] = cleaned["category"].astype(str).str.strip().str.upper()
    cleaned["whseid"] = cleaned["whseid"].astype(str).str.strip().str.upper()
    cleaned["brand"] = cleaned["brand"].astype(str).str.strip().str.upper()
    cleaned["total_qty"] = pd.to_numeric(cleaned["total_qty"], errors="coerce")
    cleaned["total_cbm"] = pd.to_numeric(cleaned["total_cbm"], errors="coerce")

    cleaned = cleaned.dropna(subset=["ship_date", "category", "whseid", "brand", "total_qty", "total_cbm"])
    cleaned = cleaned[cleaned["total_qty"] >= 0]
    cleaned = cleaned[cleaned["total_cbm"] >= 0]
    cleaned = cleaned.drop_duplicates()
    cleaned = cleaned.sort_values(["ship_date", "category", "brand", "whseid"]).reset_index(drop=True)

    return cleaned


def build_weekly_series(df: pd.DataFrame) -> pd.DataFrame:
    weekly = df.copy()
    weekly["week_start"] = weekly["ship_date"].dt.to_period("W-MON").dt.start_time

    weekly = (
        weekly.groupby(["week_start", "category", "brand", "whseid"], as_index=False)
        .agg(total_qty=("total_qty", "sum"), total_cbm=("total_cbm", "sum"))
        .sort_values(["category", "brand", "whseid", "week_start"])
        .reset_index(drop=True)
    )

    return weekly


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_raw_data()
    cleaned = clean_sales_data(raw)
    weekly = build_weekly_series(cleaned)

    cleaned.to_csv(PROCESSED_DIR / "sales_cleaned.csv", index=False)
    weekly.to_csv(PROCESSED_DIR / "weekly_sales.csv", index=False)

    print(f"Cleaned rows: {len(cleaned):,}")
    print(f"Weekly rows: {len(weekly):,}")
    print(f"Date range: {cleaned['ship_date'].min().date()} to {cleaned['ship_date'].max().date()}")
    print(f"Saved: {PROCESSED_DIR / 'sales_cleaned.csv'}")
    print(f"Saved: {PROCESSED_DIR / 'weekly_sales.csv'}")


if __name__ == "__main__":
    main()
