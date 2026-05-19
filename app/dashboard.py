from pathlib import Path
import json

import pandas as pd
import plotly.express as px
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
REPORTS_DIR = ROOT_DIR / "reports"


st.set_page_config(
    page_title="FMCG Demand Forecasting",
    layout="wide",
)


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict]]:
    sales = pd.read_csv(PROCESSED_DIR / "sales_cleaned.csv", parse_dates=["ship_date"])
    weekly = pd.read_csv(PROCESSED_DIR / "weekly_sales.csv", parse_dates=["week_start"])
    qty_pred = pd.read_csv(PROCESSED_DIR / "predictions_total_qty.csv", parse_dates=["week_start"])
    cbm_pred = pd.read_csv(PROCESSED_DIR / "predictions_total_cbm.csv", parse_dates=["week_start"])

    prediction = qty_pred.merge(
        cbm_pred[["week_start", "category", "brand", "whseid", "total_cbm", "predicted_total_cbm"]],
        on=["week_start", "category", "brand", "whseid"],
        how="left",
    )

    metrics_path = REPORTS_DIR / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else []
    return sales, weekly, prediction, metrics


def format_number(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:,.0f}"


sales, weekly, prediction, metrics = load_data()

st.title("FMCG Demand Forecasting Dashboard")
st.caption("Weekly demand forecasting by category, brand, and warehouse using LightGBM.")

with st.sidebar:
    st.header("Filters")
    selected_categories = st.multiselect(
        "Category",
        sorted(weekly["category"].unique()),
        default=sorted(weekly["category"].unique()),
    )
    selected_brands = st.multiselect(
        "Brand",
        sorted(weekly["brand"].unique()),
        default=sorted(weekly["brand"].unique()),
    )
    selected_whse = st.multiselect(
        "Warehouse",
        sorted(weekly["whseid"].unique()),
        default=sorted(weekly["whseid"].unique()),
    )

filtered_weekly = weekly[
    weekly["category"].isin(selected_categories)
    & weekly["brand"].isin(selected_brands)
    & weekly["whseid"].isin(selected_whse)
].copy()

filtered_prediction = prediction[
    prediction["category"].isin(selected_categories)
    & prediction["brand"].isin(selected_brands)
    & prediction["whseid"].isin(selected_whse)
].copy()

total_qty = filtered_weekly["total_qty"].sum()
total_cbm = filtered_weekly["total_cbm"].sum()
peak_week = filtered_weekly.groupby("week_start")["total_qty"].sum().idxmax()
brand_count = filtered_weekly["brand"].nunique()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total quantity", format_number(total_qty))
col2.metric("Total CBM", format_number(total_cbm))
col3.metric("Brands", brand_count)
col4.metric("Peak week", peak_week.date().isoformat())

tab_overview, tab_forecast, tab_metrics, tab_data = st.tabs(
    ["Overview", "Forecast", "Model Metrics", "Data"]
)

with tab_overview:
    monthly = (
        filtered_weekly.assign(month=filtered_weekly["week_start"].dt.to_period("M").dt.to_timestamp())
        .groupby("month", as_index=False)
        .agg(total_qty=("total_qty", "sum"), total_cbm=("total_cbm", "sum"))
    )

    left, right = st.columns(2)
    with left:
        fig = px.line(monthly, x="month", y="total_qty", markers=True, title="Monthly Quantity Trend")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        by_brand = (
            filtered_weekly.groupby("brand", as_index=False)["total_qty"]
            .sum()
            .sort_values("total_qty", ascending=False)
        )
        fig = px.bar(by_brand, x="brand", y="total_qty", title="Quantity by Brand")
        st.plotly_chart(fig, use_container_width=True)

    category_month = (
        filtered_weekly.assign(month=filtered_weekly["week_start"].dt.month)
        .groupby(["category", "month"], as_index=False)["total_qty"]
        .sum()
    )
    fig = px.density_heatmap(
        category_month,
        x="month",
        y="category",
        z="total_qty",
        histfunc="sum",
        title="Seasonality by Category and Month",
        color_continuous_scale="Teal",
    )
    st.plotly_chart(fig, use_container_width=True)

with tab_forecast:
    if filtered_prediction.empty:
        st.info("No forecast rows match the selected filters.")
    else:
        forecast_summary = (
            filtered_prediction.groupby("week_start", as_index=False)
            .agg(
                actual_qty=("total_qty", "sum"),
                predicted_qty=("predicted_total_qty", "sum"),
                actual_cbm=("total_cbm", "sum"),
                predicted_cbm=("predicted_total_cbm", "sum"),
            )
        )

        qty_long = forecast_summary.melt(
            id_vars="week_start",
            value_vars=["actual_qty", "predicted_qty"],
            var_name="series",
            value_name="quantity",
        )
        fig = px.line(
            qty_long,
            x="week_start",
            y="quantity",
            color="series",
            markers=True,
            title="Actual vs Predicted Weekly Quantity",
        )
        st.plotly_chart(fig, use_container_width=True)

        cbm_long = forecast_summary.melt(
            id_vars="week_start",
            value_vars=["actual_cbm", "predicted_cbm"],
            var_name="series",
            value_name="cbm",
        )
        fig = px.line(
            cbm_long,
            x="week_start",
            y="cbm",
            color="series",
            markers=True,
            title="Actual vs Predicted Weekly CBM",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            filtered_prediction.sort_values(["week_start", "category", "brand"]),
            use_container_width=True,
            hide_index=True,
        )

with tab_metrics:
    if not metrics:
        st.warning("Run `python src/train_model.py` to generate model metrics.")
    else:
        metrics_df = pd.DataFrame(metrics)
        metrics_df["wmape_percent"] = metrics_df["wmape"] * 100
        st.dataframe(
            metrics_df[["target", "train_rows", "test_rows", "mae", "rmse", "wmape_percent", "split_date"]],
            use_container_width=True,
            hide_index=True,
        )

        fig = px.bar(metrics_df, x="target", y="wmape_percent", title="WMAPE by Forecast Target")
        st.plotly_chart(fig, use_container_width=True)

with tab_data:
    st.subheader("Weekly Aggregated Data")
    st.dataframe(filtered_weekly.sort_values("week_start", ascending=False), use_container_width=True, hide_index=True)
