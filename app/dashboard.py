from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard_utils import forecast_bias, overload_status, safe_peak_week, wmape_status


ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
REPORTS_DIR = ROOT_DIR / "reports"


st.set_page_config(
    page_title="FMCG Demand Planning",
    layout="wide",
)


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict], dict[str, pd.DataFrame]]:
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
    breakdowns = {
        "brand": _read_optional_csv(REPORTS_DIR / "metrics_total_qty_by_brand.csv"),
        "category": _read_optional_csv(REPORTS_DIR / "metrics_total_qty_by_category.csv"),
        "whseid": _read_optional_csv(REPORTS_DIR / "metrics_total_cbm_by_whseid.csv"),
    }
    return sales, weekly, prediction, metrics, breakdowns


def _read_optional_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def format_number(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:,.0f}"


def render_alert_card(title: str, value: str, status: str, detail: str) -> None:
    palette = {
        "Over capacity": ("#fff1f2", "#be123c"),
        "Near capacity": ("#fffbeb", "#b45309"),
        "High uncertainty": ("#fff1f2", "#be123c"),
        "Watch": ("#fffbeb", "#b45309"),
        "Stable": ("#ecfdf5", "#047857"),
        "Within capacity": ("#ecfdf5", "#047857"),
        "No threshold": ("#f8fafc", "#475569"),
    }
    background, color = palette.get(status, ("#f8fafc", "#475569"))
    st.markdown(
        f"""
        <div style="background:{background}; border-left:5px solid {color}; padding:14px 16px; border-radius:6px;">
            <div style="font-size:0.82rem; color:#475569; font-weight:600;">{title}</div>
            <div style="font-size:1.25rem; color:#0f172a; font-weight:700; margin-top:4px;">{value}</div>
            <div style="font-size:0.9rem; color:{color}; font-weight:700; margin-top:6px;">{status}</div>
            <div style="font-size:0.82rem; color:#475569; margin-top:4px;">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def apply_filters(
    weekly: pd.DataFrame,
    prediction: pd.DataFrame,
    categories: list[str],
    brands: list[str],
    warehouses: list[str],
    date_range: tuple[pd.Timestamp, pd.Timestamp],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    start_date, end_date = date_range
    weekly_mask = (
        weekly["category"].isin(categories)
        & weekly["brand"].isin(brands)
        & weekly["whseid"].isin(warehouses)
        & weekly["week_start"].between(start_date, end_date)
    )
    prediction_mask = (
        prediction["category"].isin(categories)
        & prediction["brand"].isin(brands)
        & prediction["whseid"].isin(warehouses)
        & prediction["week_start"].between(start_date, end_date)
    )
    return weekly[weekly_mask].copy(), prediction[prediction_mask].copy()


sales, weekly, prediction, metrics, breakdowns = load_data()

st.title("FMCG Demand Forecasting & Warehouse Volume Planning")
st.caption("Industry-style portfolio dashboard for weekly one-week-ahead demand and CBM planning.")

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
    cbm_capacity = st.number_input(
        "Weekly CBM capacity alert",
        min_value=0.0,
        value=25000.0,
        step=1000.0,
        help="Simple planning threshold used only for portfolio alerting.",
    )
    min_week = weekly["week_start"].min().date()
    max_week = weekly["week_start"].max().date()
    selected_dates = st.date_input(
        "Week range",
        value=(min_week, max_week),
        min_value=min_week,
        max_value=max_week,
    )

if len(selected_dates) != 2:
    st.warning("Please select a start and end date.")
    st.stop()

date_range = (pd.Timestamp(selected_dates[0]), pd.Timestamp(selected_dates[1]))
filtered_weekly, filtered_prediction = apply_filters(
    weekly,
    prediction,
    selected_categories,
    selected_brands,
    selected_whse,
    date_range,
)

if filtered_weekly.empty and filtered_prediction.empty:
    st.warning("No data available for selected filters.")
    st.stop()

forecast_qty = filtered_prediction["predicted_total_qty"].sum() if not filtered_prediction.empty else np.nan
forecast_cbm = filtered_prediction["predicted_total_cbm"].sum() if not filtered_prediction.empty else np.nan
peak_week = safe_peak_week(filtered_prediction, "predicted_total_qty")
bias = (
    forecast_bias(filtered_prediction["total_qty"], filtered_prediction["predicted_total_qty"])
    if not filtered_prediction.empty
    else np.nan
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Forecast QTY", format_number(forecast_qty))
col2.metric("Total Forecast CBM", format_number(forecast_cbm))
col3.metric("Peak Week", peak_week.date().isoformat() if peak_week is not None else "N/A")
col4.metric("Forecast Bias", f"{bias:.2%}" if not pd.isna(bias) else "N/A")

if not filtered_prediction.empty:
    weekly_forecast_cbm = filtered_prediction.groupby("week_start")["predicted_total_cbm"].sum()
    peak_cbm = float(weekly_forecast_cbm.max()) if not weekly_forecast_cbm.empty else 0.0
    peak_cbm_week = pd.Timestamp(weekly_forecast_cbm.idxmax()) if not weekly_forecast_cbm.empty else None
else:
    peak_cbm = 0.0
    peak_cbm_week = None

metrics_df_for_alerts = pd.DataFrame(metrics) if metrics else pd.DataFrame()
qty_wmape = (
    float(metrics_df_for_alerts.loc[metrics_df_for_alerts["target"] == "total_qty", "wmape"].iloc[0])
    if not metrics_df_for_alerts.empty and (metrics_df_for_alerts["target"] == "total_qty").any()
    else np.nan
)
brand_breakdown_for_alerts = breakdowns["brand"].copy()
high_uncertainty_segments = (
    brand_breakdown_for_alerts[brand_breakdown_for_alerts["wmape"] >= 0.5].sort_values("wmape", ascending=False)
    if not brand_breakdown_for_alerts.empty and "wmape" in brand_breakdown_for_alerts.columns
    else pd.DataFrame()
)

alert_cols = st.columns(3)
with alert_cols[0]:
    render_alert_card(
        "CBM Peak Alert",
        f"{format_number(peak_cbm)} CBM",
        overload_status(peak_cbm, float(cbm_capacity)),
        f"Peak week: {peak_cbm_week.date().isoformat() if peak_cbm_week is not None else 'N/A'}",
    )
with alert_cols[1]:
    render_alert_card(
        "WMAPE Warning",
        f"{qty_wmape:.1%}" if not pd.isna(qty_wmape) else "N/A",
        wmape_status(qty_wmape) if not pd.isna(qty_wmape) else "No threshold",
        "Quantity model accuracy on 2025 holdout.",
    )
with alert_cols[2]:
    render_alert_card(
        "High Uncertainty Segments",
        f"{len(high_uncertainty_segments)} brand(s)",
        "High uncertainty" if len(high_uncertainty_segments) else "Stable",
        "Brands with WMAPE >= 50%.",
    )

tab_plan, tab_accuracy, tab_overview, tab_data = st.tabs(
    ["Planning", "Accuracy", "Overview", "Data"]
)

with tab_plan:
    if filtered_prediction.empty:
        st.warning("No forecast rows match the selected filters.")
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
        st.plotly_chart(
            px.line(qty_long, x="week_start", y="quantity", color="series", markers=True, title="Actual vs Forecast QTY"),
            use_container_width=True,
        )

        cbm_by_whse = (
            filtered_prediction.groupby("whseid", as_index=False)["predicted_total_cbm"]
            .sum()
            .sort_values("predicted_total_cbm", ascending=False)
        )
        cbm_by_whse["capacity_status"] = cbm_by_whse["predicted_total_cbm"].apply(
            lambda value: overload_status(float(value), float(cbm_capacity))
        )
        st.plotly_chart(
            px.bar(
                cbm_by_whse,
                x="whseid",
                y="predicted_total_cbm",
                color="capacity_status",
                color_discrete_map={
                    "Over capacity": "#be123c",
                    "Near capacity": "#f59e0b",
                    "Within capacity": "#047857",
                    "No threshold": "#64748b",
                },
                title="Forecast CBM by Warehouse",
            ),
            use_container_width=True,
        )

        top_brand = (
            filtered_prediction.groupby("brand", as_index=False)["predicted_total_qty"]
            .sum()
            .sort_values("predicted_total_qty", ascending=False)
            .head(1)
        )
        top_whse = cbm_by_whse.head(1)
        peak_demand_week = safe_peak_week(filtered_prediction, "predicted_total_qty")
        insight_cols = st.columns(3)
        insight_cols[0].info(
            f"Highest forecast brand: {top_brand.iloc[0]['brand']} ({format_number(top_brand.iloc[0]['predicted_total_qty'])} QTY)"
        )
        insight_cols[1].info(
            f"Highest CBM warehouse: {top_whse.iloc[0]['whseid']} ({format_number(top_whse.iloc[0]['predicted_total_cbm'])} CBM)"
        )
        insight_cols[2].info(
            f"Peak demand risk week: {peak_demand_week.date().isoformat() if peak_demand_week is not None else 'N/A'}"
        )

        weekly_plan = forecast_summary.rename(
            columns={
                "week_start": "week_start",
                "predicted_qty": "forecast_qty",
                "predicted_cbm": "forecast_cbm",
            }
        )
        st.subheader("Weekly Forecast Plan")
        st.dataframe(weekly_plan, use_container_width=True, hide_index=True)

        if not high_uncertainty_segments.empty:
            st.subheader("High Uncertainty Segments")
            risk_table = high_uncertainty_segments.copy()
            risk_table["wmape_percent"] = risk_table["wmape"] * 100
            risk_table["forecast_bias_percent"] = risk_table["forecast_bias"] * 100
            st.dataframe(
                risk_table[
                    ["brand", "rows", "actual_sum", "predicted_sum", "wmape_percent", "forecast_bias_percent"]
                ],
                use_container_width=True,
                hide_index=True,
            )

with tab_accuracy:
    if metrics:
        metrics_df = pd.DataFrame(metrics)
        metrics_df["wmape_percent"] = metrics_df["wmape"] * 100
        metrics_df["forecast_bias_percent"] = metrics_df["forecast_bias"] * 100
        st.dataframe(
            metrics_df[
                [
                    "target",
                    "train_rows",
                    "test_rows",
                    "mae",
                    "rmse",
                    "wmape_percent",
                    "forecast_bias_percent",
                    "split_date",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning("Run `python src/train_model.py` to generate model metrics.")

    brand_breakdown = breakdowns["brand"]
    if not brand_breakdown.empty:
        brand_breakdown["wmape_percent"] = brand_breakdown["wmape"] * 100
        st.plotly_chart(
            px.bar(
                brand_breakdown.sort_values("wmape_percent", ascending=False),
                x="brand",
                y="wmape_percent",
                title="WMAPE by Brand",
            ),
            use_container_width=True,
        )

with tab_overview:
    if filtered_weekly.empty:
        st.warning("No historical data available for selected filters.")
    else:
        monthly = (
            filtered_weekly.assign(month=filtered_weekly["week_start"].dt.to_period("M").dt.to_timestamp())
            .groupby("month", as_index=False)
            .agg(total_qty=("total_qty", "sum"), total_cbm=("total_cbm", "sum"))
        )
        left, right = st.columns(2)
        with left:
            st.plotly_chart(
                px.line(monthly, x="month", y="total_qty", markers=True, title="Monthly Quantity Trend"),
                use_container_width=True,
            )
        with right:
            by_brand = (
                filtered_weekly.groupby("brand", as_index=False)["total_qty"]
                .sum()
                .sort_values("total_qty", ascending=False)
            )
            st.plotly_chart(
                px.bar(by_brand, x="brand", y="total_qty", title="Historical Quantity by Brand"),
                use_container_width=True,
            )

with tab_data:
    st.subheader("Forecast Rows")
    if filtered_prediction.empty:
        st.warning("No forecast data available for selected filters.")
    else:
        st.dataframe(
            filtered_prediction.sort_values(["week_start", "category", "brand"]),
            use_container_width=True,
            hide_index=True,
        )
    st.subheader("Weekly Historical Data")
    if not filtered_weekly.empty:
        st.dataframe(filtered_weekly.sort_values("week_start", ascending=False), use_container_width=True, hide_index=True)
