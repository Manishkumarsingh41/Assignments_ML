# mini_project_partA.py
# Purpose: Analyze airline on-time performance using the 2019 US DOT flight delays dataset.
# The script loads data, cleans it, performs EDA, runs hypothesis tests, fits regressions,
# creates plots, and saves results under results/.

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

# Reduce noisy warnings
warnings.filterwarnings("ignore")

# Plotting style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# Output directories
RESULTS_DIR = "results"
CLEANED_DIR = os.path.join(RESULTS_DIR, "cleaned_data")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
SUMMARY_DIR = os.path.join(RESULTS_DIR, "summary_reports")
for d in (RESULTS_DIR, CLEANED_DIR, PLOTS_DIR, SUMMARY_DIR):
    os.makedirs(d, exist_ok=True)

# Expected CSV
CSV_FILE = "flights.csv"

def try_download_from_kaggle():
    """
    Attempt to download the dataset using Kaggle CLI if credentials are available.
    Requires ~/.kaggle/kaggle.json to be configured.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files("usdot/flight-delays", path=".", unzip=True, quiet=False)
        return True
    except Exception:
        return False

# Try to download if file not present
if not os.path.exists(CSV_FILE):
    try_download_from_kaggle()

if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(
        f"Required file '{CSV_FILE}' not found. Place flights.csv in the working directory or provide Kaggle credentials."
    )

# Load data
df = pd.read_csv(CSV_FILE, low_memory=False)
original_shape = df.shape
print(f"Loaded {CSV_FILE} with shape: {original_shape}")

# Map possible column names to canonical names used in the script
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

# Reduce to found columns and rename to canonical names
use_cols = list(found_cols.values())
working = df[use_cols].copy()
inv_map = {v: k for k, v in found_cols.items()}
working.rename(columns=inv_map, inplace=True)

# Convert to numeric where appropriate
for col in ("ArrDelay", "DepDelay", "Distance", "Month", "Day", "Year"):
    if col in working.columns:
        working[col] = pd.to_numeric(working[col], errors="coerce")

# Normalise Cancelled; default to 0 if missing
if "Cancelled" in working.columns:
    working["Cancelled"] = pd.to_numeric(working["Cancelled"], errors="coerce").fillna(0).astype(int)
else:
    working["Cancelled"] = 0

# Create Delay = ArrDelay if present, else DepDelay
working["Delay"] = working.get("ArrDelay").fillna(working.get("DepDelay"))
working = working[working["Delay"].notna()]

# Focus on non-cancelled flights and remove extreme outliers (> 6 hours)
delay_df = working[working["Cancelled"] == 0].copy()
delay_df = delay_df[delay_df["Delay"].abs() < 360].copy()
delay_df.drop_duplicates(inplace=True)

# Descriptive stats
desc_delay = delay_df["Delay"].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
print("Delay summary:")
print(desc_delay.to_string())

# Carrier-level stats (if present)
carrier_stats = pd.DataFrame()
if "Carrier" in delay_df.columns:
    carrier_stats = (
        delay_df.groupby("Carrier")["Delay"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .sort_values("count", ascending=False)
    )
    print("Top carriers (by flight count):")
    print(carrier_stats.head(10).to_string())

# Helper to save figures
def save_fig(fig, fname):
    path = os.path.join(PLOTS_DIR, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path

# Plot: delay distribution
fig = plt.figure()
sns.histplot(delay_df["Delay"], bins=100, kde=True)
plt.title("Delay distribution (minutes)")
plt.xlabel("Delay (minutes)")
plt.xlim(-60, 300)
hist_path = save_fig(fig, "delay_distribution.png")

# Plot: boxplot by carrier (top carriers)
carrier_box_path = None
if not carrier_stats.empty:
    top_carriers = carrier_stats.head(8).index.tolist()
    subset = delay_df[delay_df["Carrier"].isin(top_carriers)]
    fig = plt.figure(figsize=(12, 6))
    sns.boxplot(x="Carrier", y="Delay", data=subset, order=top_carriers)
    plt.ylim(-60, 300)
    carrier_box_path = save_fig(fig, "airline_comparison_boxplot.png")

# Correlation heatmap for numeric features
numeric_cols = [c for c in ("Delay", "Distance", "Month", "Day") if c in delay_df.columns]
corr = delay_df[numeric_cols].corr()
fig = plt.figure()
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
heatmap_path = save_fig(fig, "correlation_heatmap.png")

# Hypothesis testing: Welch's two-sample t-test between the top two carriers
tt_test_summary = {}
if "Carrier" in delay_df.columns and len(carrier_stats) >= 2:
    carriers_ordered = carrier_stats.sort_values("count", ascending=False).index.tolist()
    c1, c2 = carriers_ordered[0], carriers_ordered[1]
    d1 = delay_df[delay_df["Carrier"] == c1]["Delay"].dropna()
    d2 = delay_df[delay_df["Carrier"] == c2]["Delay"].dropna()

    # Sample for speed if necessary
    max_n = 5000
    if len(d1) > max_n:
        d1 = d1.sample(max_n, random_state=1)
    if len(d2) > max_n:
        d2 = d2.sample(max_n, random_state=1)

    t_stat, p_two = stats.ttest_ind(d1, d2, equal_var=False, nan_policy="omit")
    # One-tailed p (H1: mean_c1 < mean_c2)
    p_one = p_two / 2 if t_stat < 0 else 1 - p_two / 2
    tt_test_summary = {
        "carrier_a": c1,
        "carrier_b": c2,
        "t_statistic": float(t_stat),
        "p_two_tailed": float(p_two),
        "p_one_tailed_meanA_less": float(p_one),
    }
    print("T-test results:")
    print(tt_test_summary)

# Simple regression: Delay ~ Distance
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
    regression_info["simple"] = {
        "coef_distance": float(lr.coef_[0]),
        "intercept": float(lr.intercept_),
        "mse": float(mean_squared_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
    }
    fig = plt.figure()
    plt.scatter(X_test[:, 0], y_test, alpha=0.2, s=10)
    x_line = np.linspace(X_test.min(), X_test.max(), 100)
    y_line = lr.intercept_ + lr.coef_[0] * x_line
    plt.plot(x_line, y_line, color="red")
    reg_simple_path = save_fig(fig, "regression_fit_distance.png")

# Multiple regression: Distance + Month + Carrier (one-hot top carriers)
if "Carrier" in delay_df.columns:
    sample = delay_df.sample(n=min(20000, len(delay_df)), random_state=2)
    X_multi = pd.DataFrame()
    if "Distance" in sample.columns:
        X_multi["Distance"] = sample["Distance"].values
    if "Month" in sample.columns:
        X_multi["Month"] = sample["Month"].values
    top_n = 8
    top_carriers = sample["Carrier"].value_counts().nlargest(top_n).index.tolist()
    sample["Carrier_top"] = sample["Carrier"].where(sample["Carrier"].isin(top_carriers), "OTHER")
    dummies = pd.get_dummies(sample["Carrier_top"], prefix="Carrier", drop_first=True)
    X_multi = pd.concat([X_multi, dummies], axis=1)
    y_multi = sample["Delay"].values
    X_tr, X_te, y_tr, y_te = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)
    lr_multi = LinearRegression().fit(X_tr, y_tr)
    y_pred_m = lr_multi.predict(X_te)
    regression_info["multiple"] = {
        "features": X_multi.columns.tolist(),
        "mse": float(mean_squared_error(y_te, y_pred_m)),
        "r2": float(r2_score(y_te, y_pred_m)),
    }

# Save cleaned data and summary
cleaned_path = os.path.join(CLEANED_DIR, "clean_flights.csv")
delay_df.to_csv(cleaned_path, index=False)

summary_lines = [
    "Airline On-Time Performance - Part A",
    f"Generated: {datetime.utcnow().isoformat()} UTC",
    f"Original shape: {original_shape}",
    f"Rows after cleaning: {delay_df.shape}",
    "",
    "Delay descriptive statistics:",
    desc_delay.to_string(),
    "",
    "Carrier stats (top 10):",
]
if not carrier_stats.empty:
    summary_lines.append(carrier_stats.head(10).to_string())
summary_lines.extend(["", "T-test summary:", str(tt_test_summary), "", "Regression summary:", str(regression_info)])
summary_text = "\n".join(summary_lines)
summary_path = os.path.join(SUMMARY_DIR, "airline_summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write(summary_text)

print("Saved cleaned data, plots, and summary.")
