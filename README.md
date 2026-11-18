# F1 Tire Sim

Exploratory analysis and machine-learning models for Formula 1 tyre status, built in a Google Colab notebook on a public Kaggle lap dataset. The notebook ends with a small Gradio demo.

> Personal project (solo). Not affiliated with Formula 1, the FIA or any team; the team name used in the demo title is for illustration only.

## What the notebook does

`tire_sim.ipynb` runs top to bottom in these sections:

1. **Data preparation:** downloads the dataset with `kagglehub` and loads `f1_all.parquet` (69,230 laps × 30 columns, three seasons from 2021: race results, tyre compound, tyre life, stint and weather).
2. **Target:** creates `tire_status` with a fixed rule:
   - `optimal`: tyre life ≤ 15 laps on a fresh tyre
   - `marginal`: tyre life ≤ 40 laps
   - `critical`: otherwise
3. **Cleaning:** normalises column names, fixes types, handles missing values, standardises compound names and flags outliers (IQR).
4. **EDA:**
   - univariate: histograms, count plots, skewness and kurtosis
   - bivariate: correlation heatmap, scatter, box plots, bar plots
   - multivariate: 3D Plotly scatter, VIF, PCA
5. **Preprocessing:** one-hot encoding, standard scaling, IQR capping, SMOTE class balancing, 80/20 stratified split.
6. **Models:**
   - Fuzzy C-Means clustering (`scikit-fuzzy`)
   - Random Forest classifier
   - XGBoost classifier
   - Random Forest regressor for tyre life
7. **Demo:** a Gradio interface that takes lap and weather inputs and returns a status and a tyre-life figure.

## Data

The dataset is **not included** in this repository. See [`data/README.md`](data/README.md) for how to download it and where to check its terms of use.

## Setup

The notebook was developed in Google Colab (Python 3.12). To run it locally:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install jupyter
jupyter notebook tire_sim.ipynb
```

Or upload `tire_sim.ipynb` to Google Colab and run it there. The notebook installs `scikit-fuzzy` and `gradio` itself with `!pip`.

## How to run

Run the cells in order. The first cell downloads the dataset (Kaggle access may be required, see `data/README.md`). The last cell starts the Gradio demo with a temporary public link (`share=True`).

## Results saved in the notebook

These numbers are the outputs saved in the notebook from its last run. Read them together with the limitations below.

| Task | Metric | Value |
|---|---|---|
| Fuzzy C-Means (3 clusters) | Silhouette / Davies-Bouldin / FPC | 0.1798 / 1.8247 / 0.3864 |
| Random Forest classifier | Accuracy / weighted F1 / ROC-AUC (OvR) | 0.9974 / 0.9974 / 0.9988 |
| XGBoost classifier | Accuracy | 0.9975 |
| Random Forest regressor (tyre life) | MAE / RMSE / R² | 2.82 / 4.48 / 0.7976 |

## Known limitations

- **The classification task is close to trivial.** `tire_status` is computed from `TyreLife` and `FreshTyre`, and both are also model inputs, so the near-perfect accuracy mostly reflects the rule rather than predictive skill.
- **Pre-split leakage:**
  - SMOTE, scaling and outlier capping are applied before the train/test split, so synthetic and transformed rows reach the test set.
  - In the regression, the features include `pca1`/`pca2`, which are computed from columns that include tyre life (the target).
- **The notebook does not run top to bottom as saved.** The clustering cell uses a variable `X_cluster` that no cell defines; its saved output comes from an earlier session.
- **The Gradio demo is untested in this repository.** It passes the classification feature columns to the regression model, and it labels the regression output as remaining laps although the model predicts tyre life (laps already run on the tyre).

## Status

Prototype / learning project. The limitations above are known and not yet addressed.
