from pathlib import Path

import pandas as pd
import plotly.express as px


ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
FIGURES_DIR = ROOT_DIR / "reports" / "figures"


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    sales = pd.read_csv(PROCESSED_DIR / "sales_cleaned.csv", parse_dates=["ship_date"])
    weekly = pd.read_csv(PROCESSED_DIR / "weekly_sales.csv", parse_dates=["week_start"])
    predictions = pd.read_csv(PROCESSED_DIR / "predictions_total_qty.csv", parse_dates=["week_start"])

    monthly = (
        sales.assign(month=sales["ship_date"].dt.to_period("M").dt.to_timestamp())
        .groupby("month", as_index=False)
        .agg(total_qty=("total_qty", "sum"), total_cbm=("total_cbm", "sum"))
    )
    fig = px.line(monthly, x="month", y="total_qty", title="Monthly Total Quantity")
    fig.write_html(FIGURES_DIR / "monthly_total_qty.html")

    top_brand = weekly.groupby("brand", as_index=False)["total_qty"].sum().sort_values("total_qty", ascending=False)
    fig = px.bar(top_brand, x="brand", y="total_qty", title="Total Quantity by Brand")
    fig.write_html(FIGURES_DIR / "total_qty_by_brand.html")

    actual_pred = (
        predictions.groupby("week_start", as_index=False)
        .agg(actual=("total_qty", "sum"), predicted=("predicted_total_qty", "sum"))
        .melt(id_vars="week_start", value_vars=["actual", "predicted"], var_name="series", value_name="qty")
    )
    fig = px.line(actual_pred, x="week_start", y="qty", color="series", title="Actual vs Predicted Weekly Quantity")
    fig.write_html(FIGURES_DIR / "actual_vs_predicted_qty.html")

    print(f"Saved figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
