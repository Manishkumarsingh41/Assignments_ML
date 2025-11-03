# File: mini_project_partA.py
import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

try:
    import statsmodels.api as sm
except Exception:
    sm = None

warnings.filterwarnings("ignore")

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

RESULTS_DIR = "results"
CLEANED_DIR = os.path.join(RESULTS_DIR, "cleaned_data")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
SUMMARY_DIR = os.path.join(RESULTS_DIR, "summary_reports")
for d in (RESULTS_DIR, CLEANED_DIR, PLOTS_DIR, SUMMARY_DIR):
    os.makedirs(d, exist_ok=True)

CSV_FILE = "flights.csv"

def try_download_from_kaggle():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files("usdot/flight-delays", path=".", unzip=True, quiet=False)
        return True
    except Exception:
        return False

if not os.path.exists(CSV_FILE):
    try_download_from_kaggle()

if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(f"Required file '{CSV_FILE}' not found.")

df = pd.read_csv(CSV_FILE, low_memory=False)
original_shape = df.shape

col_map_candidates = {
    "Year": ["Year", "year"],
    "Month": ["Month", "month"],
    "Day": ["DayofMonth", "Day", "day"],
    "Carrier": ["UniqueCarrier", "Carrier", "OP_UNIQUE_CARRIER", "Reporting_Airline"],
    "FlightNum": ["FlightNum", "FlightNumber", "Flight"],
    "Origin": ["Origin", "ORIGIN", "origin"],
    "Dest": ["Dest", "DEST", "dest"],
    "DepDelay": ["DepDelay", "DEP_DELAY", "DepDelayMinutes"],
    "ArrDelay": ["ArrDelay", "ARR_DELAY", "ArrDelayMinutes"],
    "Distance": ["Distance", "DISTANCE"],
    "Cancelled": ["Cancelled", "CANCELLED"],
}

found_cols = {}
for canonical, candidates in col_map_candidates.items():
    for c in candidates:
        if c in df.columns:
            found_cols[canonical] = c
            break

use_cols = list(found_cols.values())
working = df[use_cols].copy()
inv_map = {v: k for k, v in found_cols.items()}
working.rename(columns=inv_map, inplace=True)

for col in ("ArrDelay", "DepDelay", "Distance", "Month", "Day", "Year"):
    if col in working.columns:
        working[col] = pd.to_numeric(working[col], errors="coerce")

if "Cancelled" in working.columns:
    working["Cancelled"] = pd.to_numeric(working["Cancelled"], errors="coerce").fillna(0).astype(int)
else:
    working["Cancelled"] = 0

working["Delay"] = working.get("ArrDelay").fillna(working.get("DepDelay"))
working = working[working["Delay"].notna()]

delay_df = working[working["Cancelled"] == 0].copy()
delay_df = delay_df[delay_df["Delay"].abs() < 360].copy()
delay_df.drop_duplicates(inplace=True)

# Descriptive stats

desc_delay = delay_df["Delay"].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])

carrier_stats = pd.DataFrame()
if "Carrier" in delay_df.columns:
    carrier_stats = (
        delay_df.groupby("Carrier")["Delay"]
        .agg(["count", "mean", "median", "std", "min", "max"]) 
        .sort_values("count", ascending=False)
    )

# Plots

def save_fig(fig, fname):
    path = os.path.join(PLOTS_DIR, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path

fig = plt.figure()
sns.histplot(delay_df["Delay"], bins=100, kde=True)
plt.title("Delay distribution (minutes)")
plt.xlabel("Delay (minutes)")
plt.xlim(-60, 300)
hist_path = save_fig(fig, "delay_distribution.png")

carrier_box_path = None
if not carrier_stats.empty:
    top_carriers = carrier_stats.head(8).index.tolist()
    subset = delay_df[delay_df["Carrier"].isin(top_carriers)]
    fig = plt.figure(figsize=(12, 6))
    sns.boxplot(x="Carrier", y="Delay", data=subset, order=top_carriers)
    plt.ylim(-60, 300)
    carrier_box_path = save_fig(fig, "airline_comparison_boxplot.png")

# Correlation heatmap
numeric_cols = [c for c in ("Delay", "Distance", "Month", "Day") if c in delay_df.columns]
corr = delay_df[numeric_cols].corr()
fig = plt.figure()
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
heatmap_path = save_fig(fig, "correlation_heatmap.png")

# T-tests

tt_test_summary = {}
if "Carrier" in delay_df.columns and len(carrier_stats) >= 2:
    carriers_ordered = carrier_stats.sort_values("count", ascending=False).index.tolist()
    c1, c2 = carriers_ordered[0], carriers_ordered[1]
    d1 = delay_df[delay_df["Carrier"] == c1]["Delay"].dropna()
    d2 = delay_df[delay_df["Carrier"] == c2]["Delay"].dropna()
    max_n = 5000
    if len(d1) > max_n:
        d1 = d1.sample(max_n, random_state=1)
    if len(d2) > max_n:
        d2 = d2.sample(max_n, random_state=1)
    t_stat, p_two = stats.ttest_ind(d1, d2, equal_var=False, nan_policy="omit")
    if t_stat < 0:
        p_one = p_two / 2
    else:
        p_one = 1 - p_two / 2
    tt_test_summary = {"carrier_a": c1, "carrier_b": c2, "t_statistic": float(t_stat), "p_two": float(p_two), "p_one": float(p_one)}

# Regression: simple and multiple
regression_info = {}
if "Distance" in delay_df.columns:
    X = delay_df[["Distance"]].values.reshape(-1, 1)
    y = delay_df["Delay"].values
    sample_idx = np.random.RandomState(0).choice(np.arange(len(X)), size=min(20000, len(X)), replace=False)
    Xs = X[sample_idx]
    ys = y[sample_idx]
    X_train, X_test, y_train, y_test = train_test_split(Xs, ys, test_size=0.2, random_state=42)
    lr = LinearRegression().fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    regression_info['simple'] = {"coef_distance": float(lr.coef_[0]), "intercept": float(lr.intercept_), "mse": float(mean_squared_error(y_test, y_pred)), "r2": float(r2_score(y_test, y_pred))}
    fig = plt.figure()
    plt.scatter(X_test[:, 0], y_test, alpha=0.2, s=10)
    x_line = np.linspace(X_test.min(), X_test.max(), 100)
    y_line = lr.intercept_ + lr.coef_[0] * x_line
    plt.plot(x_line, y_line, color="red")
    reg_simple_path = save_fig(fig, "regression_fit_distance.png")

if "Carrier" in delay_df.columns:
    sample = delay_df.sample(n=min(20000, len(delay_df)), random_state=2)
    X_multi = pd.DataFrame()
    if "Distance" in sample.columns:
        X_multi['Distance'] = sample['Distance'].values
    if 'Month' in sample.columns:
        X_multi['Month'] = sample['Month'].values
    top_n = 8
    top_carriers = sample['Carrier'].value_counts().nlargest(top_n).index.tolist()
    sample['Carrier_top'] = sample['Carrier'].where(sample['Carrier'].isin(top_carriers), 'OTHER')
    dummies = pd.get_dummies(sample['Carrier_top'], prefix='Carrier', drop_first=True)
    X_multi = pd.concat([X_multi, dummies], axis=1)
    y_multi = sample['Delay'].values
    X_tr, X_te, y_tr, y_te = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)
    lr_multi = LinearRegression().fit(X_tr, y_tr)
    y_pred_m = lr_multi.predict(X_te)
    regression_info['multiple'] = {"features": X_multi.columns.tolist(), "mse": float(mean_squared_error(y_te, y_pred_m)), "r2": float(r2_score(y_te, y_pred_m))}

# Save cleaned data and summary
cleaned_path = os.path.join(CLEANED_DIR, "clean_flights.csv")
delay_df.to_csv(cleaned_path, index=False)

summary_lines = []
summary_lines.append("Airline On-Time Performance - Part A")
summary_lines.append(f"Generated: {datetime.utcnow().isoformat()} UTC")
summary_lines.append(f"Original shape: {original_shape}")
summary_lines.append(f"Rows after cleaning: {delay_df.shape}")
summary_lines.append("")
summary_lines.append("Delay descriptive statistics:")
summary_lines.append(desc_delay.to_string())
summary_lines.append("")
summary_lines.append("Carrier stats (top 10):")
if not carrier_stats.empty:
    summary_lines.append(carrier_stats.head(10).to_string())
summary_lines.append("")
summary_lines.append("T-test summary:")
summary_lines.append(str(tt_test_summary))
summary_lines.append("")
summary_lines.append("Regression summary:")
summary_lines.append(str(regression_info))
summary_text = "\n".join(summary_lines)
summary_path = os.path.join(SUMMARY_DIR, "airline_summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write(summary_text)

print("Saved cleaned data, plots, and summary.")