# Notebooks

Use this folder for optional exploratory notebooks.

- `01_eda_synthetic_demo.ipynb`: public-safe EDA using synthetic sample data.

The reproducible production-style pipeline is implemented in `src/`.

Private business data is not committed to GitHub. If you want to run EDA on real shipment data, place CSV files in `data/raw/`, run `python src/preprocess.py`, and create local notebooks that stay untracked.
