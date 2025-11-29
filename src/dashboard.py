"""
dashboard.py
============

Interactive Streamlit dashboard for bibliometric analysis visualization.

This module provides a comprehensive web-based interface for analyzing
and visualizing author bibliometric data collected from Scopus.

Features
--------
1. Author Selection
   - Dropdown menu to select from collected author data
   - Automatic loading of publications and h-index timeline

2. Statistical Overview
   - Key metrics: Publications, Citations, H-Index, G-Index, M-Index, I10
   - Career-at-a-glance summary cards

3. H-Index Evolution
   - Interactive time series plot
   - Power-law forecast overlay
   - Growth trend visualization

4. Slope Analysis
   - Global h-index growth rate
   - Local rolling window slopes (3-year default)
   - Identification of career acceleration/deceleration

5. Citations by Year
   - Bar chart of annual citation counts
   - Temporal impact patterns

6. Publications Table
   - Sortable list of all publications
   - Citation counts, DOI links, source journals

7. Model Comparison & Prediction
   - Multiple regression models (Linear, Polynomial, Spline, Random Forest)
   - Growth models (Exponential saturation, Power-law, Hirsch)
   - Train/test split with adjustable cutoff year
   - Model accuracy metrics (RMSE, MAE)

8. Shape Parameter Analysis
   - Model fit quality (RMSE comparison)
   - Parameter values for all models
   - Temporal evolution of parameters
   - Stability analysis

9. Novel Metrics
   - HGC (Hirsch Growth Coefficient)
   - CV(a) (Growth stability)
   - SaH (Stability-adjusted H-growth)

Usage
-----
Run from command line:
    $ streamlit run dashboard.py

Then open browser at: http://localhost:8501

Data Requirements
-----------------
Expects directory structure:
    info/
    └── {author_id}/
        ├── {Author_Name}.csv
        ├── {Author_Name}.json
        ├── {Author_Name}_h_index_timeline.json
        └── cited_by_data/
            └── {eid}_citations.json

These files are created by running cli.py (options 1-3).

Dependencies
------------
- streamlit: Web dashboard framework
- plotly: Interactive plotting
- pandas, numpy: Data manipulation
- scikit-learn: Machine learning models
- scipy: Scientific computing

Authors: Diogo Abreu, João Machado, Pedro Lopes
Date: 11/2025
Version: 1.0

See Also
--------
cli.py : Data collection interface
shape_metrics.py : Growth model fitting
prediction_models.py : Forecasting functions
metrics.py : Bibliometric indices
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import os
import json
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import from reorganized modules
from metrics import g_index, m_index, i10_index, compute_h_index_slopes
from shape_metrics import (
    fit_shape_parameters, 
    compute_shape_evolution, 
    compute_hirsch_metrics,
    _hirsch_func
)
from prediction_models import fit_power_law, predict_power_law_on_years


# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================
st.set_page_config(page_title="Scopus Research Dashboard", layout="wide")

st.title("📊 Scopus Research Dashboard")
st.write("Interactive visualization of Scopus API collected data.")

# ============================================================
# AUTHOR SELECTION
# ============================================================
info_dir = "info"
author_folders = [f for f in os.listdir(info_dir) if os.path.isdir(os.path.join(info_dir, f))]

csv_paths = []
for folder in author_folders:
    folder_path = os.path.join(info_dir, folder)
    for f in os.listdir(folder_path):
        if f.endswith(".csv"):
            csv_paths.append(os.path.join(folder_path, f))

if not csv_paths:
    st.warning("No CSV files found in author folders in /info.")
    st.stop()

selected_path = st.selectbox("Select author file:", csv_paths)
if not selected_path:
    st.error("No file selected.")
    st.stop()
    
selected_path = str(selected_path)
selected_file = os.path.basename(selected_path)
author_folder = os.path.dirname(selected_path)
safe_author_name = os.path.splitext(selected_file)[0]
author_id = os.path.basename(author_folder)

# Predictions directory
predictions_dir = "Predictions"
os.makedirs(predictions_dir, exist_ok=True)
predictions_output_file = os.path.join(predictions_dir, f"{author_id}_h_index_predictions.csv")

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv(selected_path)
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

# Load h-index timeline
h_index_timeline_path = os.path.join(author_folder, f"{safe_author_name}_h_index_timeline.json")
if not os.path.exists(h_index_timeline_path):
    st.error("H-Index timeline file not found for this author.")
    st.stop()
    
with open(h_index_timeline_path, 'r', encoding='utf-8') as f:
    h_timeline = json.load(f)
    
h_by_year = pd.DataFrame({"Year": list(h_timeline.keys()), "H-Index": list(h_timeline.values())})
h_by_year["Year"] = pd.to_numeric(h_by_year["Year"])

# ============================================================
# COMPUTE METRICS
# ============================================================
# H-index slopes
slopes_info = compute_h_index_slopes(h_by_year, window=3)
global_slope = slopes_info["global_slope"]
local_slopes = slopes_info["local_slopes"]

# Shape parameters
shape_params = fit_shape_parameters(
    h_by_year["Year"].values,
    h_by_year["H-Index"].values
)

# Basic statistics
total_pubs = len(df)
total_citations = df["Cited by"].fillna(0).astype(int).sum()
h_index = h_by_year["H-Index"].iloc[-1] if not h_by_year.empty else 0

citations = df["Cited by"].dropna().astype(int).tolist()
years_list = df["Year"].dropna().astype(int).tolist()

g_val = g_index(citations)
i10_val = i10_index(citations)

first_year = min(years_list)
last_year = max(years_list)
m_val = m_index(h_index, first_year, last_year)

# ============================================================
# GENERAL STATISTICS
# ============================================================
st.header("📈 General Statistics")
col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Publications", total_pubs)
col2.metric("Citations", total_citations)
col3.metric("H-Index", h_index)
col4.metric("G-Index", g_val)
col5.metric("M-Index", f"{m_val:.2f}")
col6.metric("I10-Index", i10_val)

# ============================================================
# H-INDEX EVOLUTION
# ============================================================
st.header("📉 H-Index Evolution")

# Power-law forecast
future_x, pred_power, params_power = fit_power_law(
    h_by_year["Year"].to_numpy(),
    h_by_year["H-Index"].to_numpy()
)

fig_h = px.line(h_by_year, x="Year", y="H-Index", markers=True, title="H-Index over the years")
fig_h.add_scatter(
    x=future_x,
    y=pred_power,
    mode="lines+markers",
    name="Power-Law Forecast",
    line=dict(color="green", dash="dash")
)
st.plotly_chart(fig_h, use_container_width=True)

# ============================================================
# SLOPE ANALYSIS
# ============================================================
st.subheader("🔍 H-Index Trajectory Slope")
st.write(f"Approximate global slope (dh/dt): **{global_slope:.3f} h/year**")

if not local_slopes.empty:
    fig_slope = px.line(
        local_slopes,
        x="Year_center",
        y="Slope",
        markers=True,
        title="Local slope (3-year rolling window)"
    )
    st.plotly_chart(fig_slope, use_container_width=True)
else:
    st.info("Insufficient data to calculate local slopes.")

# ============================================================
# CITATIONS BY YEAR
# ============================================================
st.header("🔢 Citations per Year")
citations_by_year = df.groupby("Year")["Cited by"].sum().reset_index()
fig_cit = px.bar(citations_by_year, x="Year", y="Cited by", title="Total Citations per Year")
st.plotly_chart(fig_cit, use_container_width=True)

# ============================================================
# PUBLICATIONS TABLE
# ============================================================
st.header("📚 Publications List")
st.dataframe(df[["Title", "Year", "Cited by", "Source title", "DOI"]].sort_values(by="Cited by", ascending=False))

# ============================================================
# PREDICTIONS WITH DIFFERENT MODELS
# ============================================================
st.header("📉 H-Index Evolution and Predictions")

min_year = int(h_by_year["Year"].min())
max_year = int(h_by_year["Year"].max()) - 1
default_cutoff = int((min_year + max_year) / 2)

cutoff_year = st.slider(
    "Select training cutoff year (models train on data up to this year, predict after)",
    min_year, max_year, default_cutoff
)

# Split data
train = h_by_year[h_by_year["Year"] <= cutoff_year].copy()
test = h_by_year[h_by_year["Year"] > cutoff_year].copy()

years_train = train["Year"].to_numpy().reshape(-1, 1)
h_train_values = train["H-Index"].to_numpy()
years_test = test["Year"].to_numpy().reshape(-1, 1)
test_years_flat = test["Year"].to_numpy()

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(years_train, h_train_values)
linear_predictions = linear_model.predict(years_test) if len(years_test) else np.array([])
linear_predictions = np.maximum(linear_predictions, 0)

# Spline Regression
try:
    spline = UnivariateSpline(train["Year"], h_train_values, s=0)
    spline_predictions = spline(test_years_flat) if len(test_years_flat) else np.array([])
except Exception:
    spline_predictions = np.array([np.nan] * len(test_years_flat))
spline_predictions = np.maximum(spline_predictions, 0)

# Polynomial Regression
poly_features = PolynomialFeatures(degree=2)
X_poly_train = poly_features.fit_transform(years_train)
poly_model = LinearRegression()
poly_model.fit(X_poly_train, h_train_values)
X_poly_test = poly_features.transform(years_test)
poly_predictions = poly_model.predict(X_poly_test) if len(years_test) else np.array([])
poly_predictions = np.maximum(poly_predictions, 0)

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(years_train, h_train_values)
rf_predictions = rf_model.predict(years_test) if len(years_test) else np.array([])
rf_predictions = np.maximum(rf_predictions, 0)

# Exponential saturation model
def exp_model(t, a, b, c):
    t = np.asarray(t, dtype=float)
    x = -b * t
    x = np.clip(x, -700, 700)
    return a * (1.0 - np.exp(x)) + c

try:
    popt_exp, _ = curve_fit(
        exp_model,
        train["Year"],
        h_train_values,
        p0=(1.0, 0.05, 0.0),
        bounds=([0.0, 0.0, 0.0], [np.inf, 1.0, np.inf]),
        maxfev=5000,
    )
    exp_predictions = exp_model(test_years_flat, *popt_exp)
    exp_predictions = np.maximum(exp_predictions, 0)
except Exception:
    exp_predictions = np.array([np.nan] * len(test_years_flat))

# Power-law model
power_predictions = predict_power_law_on_years(
    years_train,
    h_train_values,
    test_years_flat
)

# Hirsch model
try:
    popt_h, _ = curve_fit(_hirsch_func, train["Year"], h_train_values, maxfev=5000)
    hirsch_predictions = _hirsch_func(test_years_flat, *popt_h)
    hirsch_predictions = np.maximum(hirsch_predictions, 0)
except Exception:
    hirsch_predictions = np.array([np.nan] * len(test_years_flat))

# Combine for plot
plot_data = pd.concat([
    h_by_year.assign(Type="Actual"),
    pd.DataFrame({"Year": test_years_flat, "H-Index": linear_predictions, "Type": "Linear Prediction"}),
    pd.DataFrame({"Year": test_years_flat, "H-Index": spline_predictions, "Type": "Spline Prediction"}),
    pd.DataFrame({"Year": test_years_flat, "H-Index": poly_predictions, "Type": "Polynomial Prediction"}),
    pd.DataFrame({"Year": test_years_flat, "H-Index": rf_predictions, "Type": "Random Forest Prediction"}),
    pd.DataFrame({"Year": test_years_flat, "H-Index": exp_predictions, "Type": "Exponential Prediction"}),
    pd.DataFrame({"Year": test_years_flat, "H-Index": power_predictions, "Type": "Power-law Prediction"}),
    pd.DataFrame({"Year": test_years_flat, "H-Index": hirsch_predictions, "Type": "Hirsch Prediction"}),
])

color_map = {
    "Actual": "#2b7bba",
    "Linear Prediction": "#333333",
    "Spline Prediction": "#d76e6e",
    "Polynomial Prediction": "#c70039",
    "Random Forest Prediction": "#009688",
    "Exponential Prediction": "#e67e22",
    "Power-law Prediction": "#16a085",
    "Hirsch Prediction": "#8e44ad"
}

fig_pred = px.line(
    plot_data,
    x="Year",
    y="H-Index",
    color="Type",
    markers=True,
    title=f"H-Index Prediction (Training up to {cutoff_year}, Prediction after)",
    color_discrete_map=color_map
)
st.plotly_chart(fig_pred, use_container_width=True)

# ============================================================
# SHAPE PARAMETERS
# ============================================================
st.subheader("🔍 Shape parameters (models fitted to h-index)")

rows = []
lin = shape_params.get("linear", {})
if lin.get("success"):
    rows.append({
        "Model": "Linear",
        "Param 1": f"slope = {lin['slope']:.3f}",
        "Param 2": f"intercept = {lin['intercept']:.3f}",
        "RMSE": f"{lin['rmse']:.3f}",
    })

hir = shape_params.get("hirsch", {})
if hir.get("success"):
    rows.append({
        "Model": "Hirsch (a√t + b)",
        "Param 1": f"a = {hir['a']:.3f}",
        "Param 2": f"b = {hir['b']:.3f}",
        "RMSE": f"{hir['rmse']:.3f}",
    })

pw = shape_params.get("power", {})
if pw.get("success"):
    rows.append({
        "Model": "Power-law (a·t^b)",
        "Param 1": f"a = {pw['a']:.3f}",
        "Param 2": f"b = {pw['b']:.3f}",
        "RMSE": f"{pw['rmse']:.3f}",
    })

exp_sh = shape_params.get("exponential", {})
if exp_sh.get("success"):
    rows.append({
        "Model": "Exponential (a·(1-e^{-b t})+c)",
        "Param 1": f"a = {exp_sh['a']:.3f}, b = {exp_sh['b']:.3f}",
        "Param 2": f"c = {exp_sh['c']:.3f}",
        "RMSE": f"{exp_sh['rmse']:.3f}",
    })

if rows:
    st.table(pd.DataFrame(rows))
else:
    st.info("Could not fit any model (insufficient data).")

# ============================================================
# PARAMETER EVOLUTION
# ============================================================
shape_evol = compute_shape_evolution(h_by_year)

st.header("🔍 Stability of shape parameters throughout career")

if shape_evol is None or shape_evol.empty:
    st.info("Insufficient data to analyze parameter evolution.")
else:
    st.subheader("Year-by-year parameters")
    st.dataframe(shape_evol.round(3))

    # Hirsch 'a' parameter evolution
    if shape_evol["hirsch_a"].notna().sum() > 0:
        fig_hir_a = px.line(
            shape_evol,
            x="Year",
            y="hirsch_a",
            markers=True,
            title="Evolution of parameter a (Hirsch: h(t) = a√t + b)"
        )
        st.plotly_chart(fig_hir_a, use_container_width=True)

    # Power-law 'b' exponent evolution
    if shape_evol["power_b"].notna().sum() > 0:
        fig_pow_b = px.line(
            shape_evol,
            x="Year",
            y="power_b",
            markers=True,
            title="Evolution of exponent b (Power-law: h(t) = a·t^b)"
        )
        st.plotly_chart(fig_pow_b, use_container_width=True)

    # RMSE evolution
    rmse_cols = ["lin_rmse", "hirsch_rmse", "power_rmse", "exp_rmse"]
    if any(c in shape_evol.columns for c in rmse_cols):
        rmse_long = shape_evol.melt(
            id_vars="Year",
            value_vars=rmse_cols,
            var_name="Model",
            value_name="RMSE"
        )
        rmse_long["Model"] = rmse_long["Model"].map({
            "lin_rmse": "Linear",
            "hirsch_rmse": "Hirsch",
            "power_rmse": "Power-law",
            "exp_rmse": "Exponential"
        })
        fig_rmse = px.line(
            rmse_long.dropna(),
            x="Year",
            y="RMSE",
            color="Model",
            markers=True,
            title="Fit quality (RMSE) over the years"
        )
        st.plotly_chart(fig_rmse, use_container_width=True)

# ============================================================
# HIRSCH METRICS
# ============================================================
hgc, cv_a, sah = compute_hirsch_metrics(shape_params, shape_evol)

st.subheader("🆕 Novel Hirsch-based Metric")

cols = st.columns(3)
if hgc is not None:
    cols[0].metric("HGC (coef. a)", f"{hgc:.3f}")
if cv_a is not None:
    cols[1].metric("CV(a) (stability)", f"{cv_a:.3f}")
if sah is not None:
    cols[2].metric("SaH = a·(1−CV(a))", f"{sah:.3f}")

st.caption(
    "HGC is coefficient 'a' from the Hirsch model h(t)=a√t+b (h-index growth rate). "
    "SaH combines this growth with temporal stability of 'a': the higher, "
    "the stronger and more stable the author's impact throughout their career."
)

# ============================================================
# MODEL ACCURACY
# ============================================================
if len(test) > 0:
    st.header("🎯 Model Accuracy in Holdout (Test) Years")
    metrics = {}
    for label, preds in [("Linear", linear_predictions),
                         ("Spline", spline_predictions),
                         ("Polynomial", poly_predictions),
                         ("RandomForest", rf_predictions),
                         ("Exponential", exp_predictions),
                         ("Power-law", power_predictions),
                         ("Hirsch", hirsch_predictions)]:
        y_true = test["H-Index"].values
        mask = ~np.isnan(preds)
        if mask.any():
            metrics[label] = {
                "RMSE": mean_squared_error(y_true[mask], preds[mask]) ** 0.5,
                "MAE": mean_absolute_error(y_true[mask], preds[mask])
            }
    st.dataframe(pd.DataFrame(metrics).T)

# ============================================================
# SAVE PREDICTIONS
# ============================================================
prediction_df = pd.DataFrame({
    "Year": test_years_flat,
    "Linear": linear_predictions,
    "Spline": spline_predictions,
    "Polynomial": poly_predictions,
    "RandomForest": rf_predictions
})

if os.path.exists(predictions_output_file):
    old_predictions = pd.read_csv(predictions_output_file)
    prediction_df = prediction_df[~prediction_df['Year'].isin(old_predictions['Year'])]
    if not prediction_df.empty:
        updated = pd.concat([old_predictions, prediction_df], ignore_index=True)
        updated.to_csv(predictions_output_file, index=False)
else:
    prediction_df.to_csv(predictions_output_file, index=False)