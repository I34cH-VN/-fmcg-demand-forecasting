# FMCG Demand Forecasting

This project builds an end-to-end demand forecasting workflow for three years of FMCG shipment data from a manufacturing context. It cleans raw sales files, aggregates demand weekly, engineers time-series features, trains LightGBM models, evaluates forecast accuracy, and provides a Streamlit dashboard for business exploration.

## Business Goal

Forecast weekly sales demand and warehouse volume so planners can understand:

- Which categories and brands drive shipment volume.
- How demand changes by month and season.
- How well a LightGBM model predicts future quantity and CBM.
- Which weeks may require higher warehouse or transportation capacity.

## Dataset

Raw CSV files are stored locally in `data/raw/` and contain:

- `ACTUALSHIPDATE`: shipment date.
- `CATEGORY`: product category.
- `WHSEID`: warehouse identifier.
- `BRAND`: product brand.
- `Total QTY`: shipped quantity.
- `Total CBM`: shipment volume.
- `Week`, `Day`: calendar fields from the source data.

## Project Structure

```text
demand_forecast/
├── app/
│   └── dashboard.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
├── reports/
│   └── figures/
├── src/
│   ├── features.py
│   ├── make_figures.py
│   ├── preprocess.py
│   └── train_model.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Methodology

1. Combine yearly CSV files into one dataset.
2. Clean dates, numeric fields, categories, brands, and warehouse IDs.
3. Aggregate daily shipment rows into weekly demand by `category + brand + whseid`.
4. Create calendar, Tet-season, lag, rolling mean, difference, and percentage-change features.
5. Train LightGBM regression models for:
   - `total_qty`
   - `total_cbm`
6. Evaluate on a time-based split:
   - Train: 2023-2024
   - Test: 2025
7. Visualize historical demand, actual vs predicted demand, and model metrics in Streamlit.

## How to Run

Place the source CSV files in `data/raw/`:

```text
data/raw/data_2023.csv
data/raw/data_2024.csv
data/raw/data_2025.csv
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run preprocessing:

```powershell
python src/preprocess.py
```

Train LightGBM models:

```powershell
python src/train_model.py
```

Generate report figures:

```powershell
python src/make_figures.py
```

Start the dashboard:

```powershell
streamlit run app/dashboard.py
```

## Outputs

Generated files:

- `data/processed/sales_cleaned.csv`
- `data/processed/weekly_sales.csv`
- `data/processed/predictions_total_qty.csv`
- `data/processed/predictions_total_cbm.csv`
- `models/lightgbm_total_qty.pkl`
- `models/lightgbm_total_cbm.pkl`
- `reports/metrics.json`
- `reports/figures/*.html`

## Portfolio Summary

Built an end-to-end FMCG demand forecasting project using three years of sales shipment data. The workflow includes data cleaning, exploratory analysis, time-series feature engineering, LightGBM forecasting, model evaluation, and an interactive Streamlit dashboard for quantity and warehouse volume planning.
