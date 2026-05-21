"""
============================================================
UCT Machine Learning Internship — Project 4
Prediction of Agriculture Crop Production in India
============================================================
Author  : UCT ML Intern
Dataset : data.gov.in  (2001–2014)
"""

# ─────────────────────────────────────────────
# 0. Imports & Setup
# ─────────────────────────────────────────────
import warnings, os
warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.model_selection  import train_test_split, cross_val_score, KFold
from sklearn.preprocessing    import LabelEncoder, StandardScaler
from sklearn.linear_model     import LinearRegression, Ridge, Lasso
from sklearn.ensemble         import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree             import DecisionTreeRegressor
from sklearn.metrics          import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline         import Pipeline
import joblib

BASE  = os.path.dirname(os.path.abspath(__file__))
DATA  = os.path.join(BASE, "data")
PLOTS = os.path.join(BASE, "plots")
MDL   = os.path.join(BASE, "models")
OUT   = os.path.join(BASE, "outputs")

sns.set_theme(style="whitegrid", palette="Set2", font_scale=1.1)
COLORS = sns.color_palette("Set2", 10)

# ─────────────────────────────────────────────
# 1. Data Loading & Initial Exploration
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 1 — DATA LOADING & EXPLORATION")
print("="*60)

# ── produce.csv (main time-series dataset 1993–2014) ──
produce_raw = pd.read_csv(os.path.join(DATA, "produce.csv"))
print(f"\nproduce.csv  :  {produce_raw.shape[0]} rows × {produce_raw.shape[1]} cols")
print(produce_raw.head(3).to_string())

# ── cost / yield by state ──
cost_df = pd.read_csv(os.path.join(DATA, "datafile (1).csv"))
cost_df.columns = cost_df.columns.str.strip()
print(f"\ndatafile(1)  :  {cost_df.shape[0]} rows × {cost_df.shape[1]} cols")
print(cost_df.head(3).to_string())

# ── production / area / yield index ──
index_df = pd.read_csv(os.path.join(DATA, "datafile (2).csv"))
index_df.columns = index_df.columns.str.strip()
print(f"\ndatafile(2)  :  {index_df.shape[0]} rows × {index_df.shape[1]} cols")
print(index_df.head(3).to_string())

# ── crop variety / zone ──
variety_df = pd.read_csv(os.path.join(DATA, "datafile (3).csv"))
variety_df.columns = variety_df.columns.str.strip()
print(f"\ndatafile(3)  :  {variety_df.shape[0]} rows × {variety_df.shape[1]} cols")
print(variety_df.head(3).to_string())

# ── price index ──
price_idx = pd.read_csv(os.path.join(DATA, "datafile.csv"))
print(f"\ndatafile.csv :  {price_idx.shape[0]} rows × {price_idx.shape[1]} cols")
print(price_idx.head(3).to_string())


# ─────────────────────────────────────────────
# 2. Data Cleaning & Pre-processing
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 2 — DATA CLEANING & PRE-PROCESSING")
print("="*60)

# ── 2A. Melt produce.csv into a long-format time-series ──
year_cols = [c for c in produce_raw.columns if c.strip().startswith("3-")]
produce   = produce_raw[["Particulars", "Frequency", "Unit"] + year_cols].copy()
produce   = produce.melt(
    id_vars=["Particulars", "Frequency", "Unit"],
    value_vars=year_cols,
    var_name="year_raw",
    value_name="value"
)
produce["year"] = produce["year_raw"].str.strip().str.replace("3-", "", regex=False).astype(int)
produce["value"] = pd.to_numeric(produce["value"], errors="coerce")
produce.dropna(subset=["value"], inplace=True)
print(f"\nproduce (long) shape: {produce.shape}")
print(produce.head())

# ── 2B. Clean cost/yield dataset ──
cost_df.columns = ["Crop", "State",
                   "Cost_CultivationA2FL_per_Ha",
                   "Cost_CultivationC2_per_Ha",
                   "Cost_Production_per_Quintal",
                   "Yield_Quintal_per_Ha"]
cost_df["Crop"]  = cost_df["Crop"].str.strip()
cost_df["State"] = cost_df["State"].str.strip()
for col in ["Cost_CultivationA2FL_per_Ha", "Cost_CultivationC2_per_Ha",
            "Cost_Production_per_Quintal", "Yield_Quintal_per_Ha"]:
    cost_df[col] = pd.to_numeric(cost_df[col], errors="coerce")
cost_df.dropna(inplace=True)
print(f"\ncost_df (clean) shape: {cost_df.shape}")

# ── 2C. Clean variety / zone ──
variety_df = variety_df[["Crop", "Variety", "Season/ duration in days", "Recommended Zone"]]
variety_df.columns = ["Crop", "Variety", "Season_Days", "Recommended_Zone"]
variety_df.dropna(inplace=True)
variety_df["Crop"] = variety_df["Crop"].str.strip()
print(f"\nvariety_df (clean) shape: {variety_df.shape}")

# ── 2D. Missing-value report ──
print("\nMissing values per dataset:")
for name, df in [("produce", produce), ("cost", cost_df), ("variety", variety_df)]:
    miss = df.isnull().sum().sum()
    print(f"  {name:10s} → {miss} missing values")


# ─────────────────────────────────────────────
# 3. Exploratory Data Analysis (EDA)
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 3 — EXPLORATORY DATA ANALYSIS")
print("="*60)

# ── Plot 1: Total Foodgrain Production 1993–2014 ──
food = produce[produce["Particulars"].str.contains("Foodgrains$", na=False, regex=True)].copy()
food = food.groupby("year")["value"].sum().reset_index()

fig, ax = plt.subplots(figsize=(11, 5))
ax.fill_between(food["year"], food["value"], alpha=0.25, color=COLORS[0])
ax.plot(food["year"], food["value"], marker="o", color=COLORS[0], linewidth=2.5, label="Foodgrain Production")
ax.set_title("Total Foodgrain Production in India  (1993–2014)", fontsize=15, fontweight="bold")
ax.set_xlabel("Year")
ax.set_ylabel("Production (Million Tonnes)")
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "01_foodgrain_production_trend.png"), dpi=150)
plt.close()
print("✔ Plot 1 saved: Foodgrain production trend")

# ── Plot 2: Major Crop Breakdown ──
major_keywords = ["Rice$", "Wheat Rabi$", "Maize$", "Sugarcane$", "Cotton$", "Pulses$"]
crop_labels    = ["Rice", "Wheat", "Maize", "Sugarcane", "Cotton", "Pulses"]

fig, ax = plt.subplots(figsize=(12, 6))
for kw, label, color in zip(major_keywords, crop_labels, COLORS):
    sub = produce[produce["Particulars"].str.contains(kw, na=False, regex=True)]
    sub = sub.groupby("year")["value"].sum().reset_index()
    if not sub.empty:
        ax.plot(sub["year"], sub["value"], marker="o", linewidth=2,
                label=label, color=color)
ax.set_title("Major Crop Production Trends  (1993–2014)", fontsize=15, fontweight="bold")
ax.set_xlabel("Year")
ax.set_ylabel("Production (Million Tonnes)")
ax.legend(loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "02_major_crops_trend.png"), dpi=150)
plt.close()
print("✔ Plot 2 saved: Major crop trends")

# ── Plot 3: Cost of Cultivation by Crop ──
cost_by_crop = cost_df.groupby("Crop")["Cost_CultivationC2_per_Ha"].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(cost_by_crop.index, cost_by_crop.values, color=COLORS[:len(cost_by_crop)])
ax.set_xlabel("Avg Cost of Cultivation (₹/Hectare, C2)")
ax.set_title("Average Cost of Cultivation by Crop", fontsize=14, fontweight="bold")
ax.bar_label(bars, fmt="₹%.0f", padding=3, fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "03_cost_by_crop.png"), dpi=150)
plt.close()
print("✔ Plot 3 saved: Cost of cultivation by crop")

# ── Plot 4: Yield by Crop ──
yield_by_crop = cost_df.groupby("Crop")["Yield_Quintal_per_Ha"].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(yield_by_crop.index, yield_by_crop.values, color=COLORS[:len(yield_by_crop)])
ax.set_xlabel("Avg Yield (Quintal/Hectare)")
ax.set_title("Average Crop Yield by Crop Type", fontsize=14, fontweight="bold")
ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "04_yield_by_crop.png"), dpi=150)
plt.close()
print("✔ Plot 4 saved: Yield by crop")

# ── Plot 5: Correlation heatmap ──
numeric_cost = cost_df.select_dtypes(include=np.number)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(numeric_cost.corr(), annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=.5, ax=ax)
ax.set_title("Correlation Heatmap — Cost & Yield Features", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "05_correlation_heatmap.png"), dpi=150)
plt.close()
print("✔ Plot 5 saved: Correlation heatmap")

# ── Plot 6: Yield Distribution ──
fig, ax = plt.subplots(figsize=(9, 5))
sns.histplot(cost_df["Yield_Quintal_per_Ha"], kde=True, color=COLORS[2], bins=20, ax=ax)
ax.set_title("Distribution of Crop Yield  (Quintal/Hectare)", fontsize=14, fontweight="bold")
ax.set_xlabel("Yield (Quintal/Hectare)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "06_yield_distribution.png"), dpi=150)
plt.close()
print("✔ Plot 6 saved: Yield distribution")

# ── Plot 7: Boxplot Yield by Crop ──
fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(data=cost_df, x="Crop", y="Yield_Quintal_per_Ha", palette="Set2", ax=ax)
ax.set_title("Yield Distribution by Crop Type", fontsize=14, fontweight="bold")
ax.set_xlabel("Crop")
ax.set_ylabel("Yield (Quintal/Ha)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "07_yield_boxplot_by_crop.png"), dpi=150)
plt.close()
print("✔ Plot 7 saved: Yield boxplot by crop")

# ── Plot 8: Scatter Cost vs Yield ──
fig, ax = plt.subplots(figsize=(9, 5))
for crop, grp in cost_df.groupby("Crop"):
    ax.scatter(grp["Cost_CultivationC2_per_Ha"], grp["Yield_Quintal_per_Ha"],
               label=crop, alpha=0.8, s=80)
ax.set_xlabel("Cost of Cultivation C2 (₹/Ha)")
ax.set_ylabel("Yield (Quintal/Ha)")
ax.set_title("Cost of Cultivation vs. Crop Yield", fontsize=14, fontweight="bold")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "08_cost_vs_yield_scatter.png"), dpi=150)
plt.close()
print("✔ Plot 8 saved: Cost vs Yield scatter")


# ─────────────────────────────────────────────
# 4. Feature Engineering
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 4 — FEATURE ENGINEERING")
print("="*60)

# ── 4A. Encode categorical features ──
le_crop  = LabelEncoder()
le_state = LabelEncoder()
model_df = cost_df.copy()
model_df["Crop_enc"]  = le_crop.fit_transform(model_df["Crop"])
model_df["State_enc"] = le_state.fit_transform(model_df["State"])

# ── 4B. Derived feature: profit proxy ──
model_df["Cost_per_Yield"] = (model_df["Cost_CultivationC2_per_Ha"]
                               / model_df["Yield_Quintal_per_Ha"])

print("\nEngineered feature preview:")
print(model_df[["Crop", "State", "Crop_enc", "State_enc",
                 "Cost_per_Yield", "Yield_Quintal_per_Ha"]].head(8).to_string())

# ── 4C. Define X / y ──
FEATURES = ["Crop_enc", "State_enc",
            "Cost_CultivationA2FL_per_Ha",
            "Cost_CultivationC2_per_Ha",
            "Cost_Production_per_Quintal",
            "Cost_per_Yield"]
TARGET   = "Yield_Quintal_per_Ha"

X = model_df[FEATURES].values
y = model_df[TARGET].values
print(f"\nFeature matrix X: {X.shape}   Target y: {y.shape}")


# ─────────────────────────────────────────────
# 5. Model Training & Evaluation
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 5 — MODEL TRAINING & EVALUATION")
print("="*60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {X_train.shape[0]} samples   |   Test: {X_test.shape[0]} samples")

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

MODELS = {
    "Linear Regression"         : LinearRegression(),
    "Ridge Regression"          : Ridge(alpha=1.0),
    "Lasso Regression"          : Lasso(alpha=0.1),
    "Decision Tree"             : DecisionTreeRegressor(max_depth=5, random_state=42),
    "Random Forest"             : RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting"         : GradientBoostingRegressor(n_estimators=100, random_state=42),
}

results = []
kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("\n{:<28} {:>7} {:>7} {:>7} {:>10}".format(
    "Model", "MAE", "RMSE", "R²", "CV-R²(5-fold)"))
print("-"*65)

best_model_name = None
best_r2         = -np.inf
best_model_obj  = None

for name, mdl in MODELS.items():
    use_scaled = name in ("Linear Regression", "Ridge Regression", "Lasso Regression")
    Xtr = X_train_s if use_scaled else X_train
    Xte = X_test_s  if use_scaled else X_test

    mdl.fit(Xtr, y_train)
    y_pred = mdl.predict(Xte)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    cv   = cross_val_score(mdl, Xtr, y_train, cv=kf, scoring="r2").mean()

    results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2, "CV_R2": cv})
    print(f"{name:<28} {mae:>7.3f} {rmse:>7.3f} {r2:>7.4f} {cv:>10.4f}")

    if r2 > best_r2:
        best_r2         = r2
        best_model_name = name
        best_model_obj  = mdl

results_df = pd.DataFrame(results).sort_values("R2", ascending=False)
results_df.to_csv(os.path.join(OUT, "model_comparison.csv"), index=False)
print(f"\n★  Best model: {best_model_name}  (R² = {best_r2:.4f})")


# ─────────────────────────────────────────────
# 6. Result Visualisations
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 6 — RESULT VISUALISATIONS")
print("="*60)

# ── Plot 9: Model Comparison Bar ──
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
metrics = ["MAE", "RMSE", "R2"]
titles  = ["MAE (lower = better)", "RMSE (lower = better)", "R² Score (higher = better)"]
for ax, metric, title in zip(axes, metrics, titles):
    bars = ax.barh(results_df["Model"], results_df[metric],
                   color=[COLORS[i % 8] for i in range(len(results_df))])
    ax.set_title(title, fontweight="bold")
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
plt.suptitle("Model Performance Comparison", fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "09_model_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()
print("✔ Plot 9 saved: Model comparison")

# ── Plot 10: Actual vs Predicted — best model ──
use_scaled = best_model_name in ("Linear Regression", "Ridge Regression", "Lasso Regression")
Xte_final  = X_test_s if use_scaled else X_test
y_pred_best = best_model_obj.predict(Xte_final)

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(y_test, y_pred_best, alpha=0.7, s=80, color=COLORS[0], edgecolors="k", linewidth=0.4)
lims = [min(y_test.min(), y_pred_best.min()) - 2,
        max(y_test.max(), y_pred_best.max()) + 2]
ax.plot(lims, lims, "r--", linewidth=2, label="Perfect Prediction")
ax.set_xlabel("Actual Yield (Quintal/Ha)")
ax.set_ylabel("Predicted Yield (Quintal/Ha)")
ax.set_title(f"Actual vs Predicted — {best_model_name}\nR² = {best_r2:.4f}",
             fontsize=13, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "10_actual_vs_predicted.png"), dpi=150)
plt.close()
print("✔ Plot 10 saved: Actual vs Predicted")

# ── Plot 11: Residuals ──
residuals = y_test - y_pred_best
fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(y_pred_best, residuals, alpha=0.7, color=COLORS[1], edgecolors="k", linewidth=0.4)
ax.axhline(0, color="red", linestyle="--", linewidth=2)
ax.set_xlabel("Predicted Yield")
ax.set_ylabel("Residuals")
ax.set_title("Residual Plot — Best Model", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "11_residuals.png"), dpi=150)
plt.close()
print("✔ Plot 11 saved: Residuals")

# ── Plot 12: Feature Importance (RF / GB) ──
if hasattr(best_model_obj, "feature_importances_"):
    fi = pd.Series(best_model_obj.feature_importances_, index=FEATURES).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(fi.index, fi.values, color=COLORS[:len(fi)])
    ax.set_title(f"Feature Importance — {best_model_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "12_feature_importance.png"), dpi=150)
    plt.close()
    print("✔ Plot 12 saved: Feature importance")

# ── Plot 13: Time-series forecast on produce data ──
# Build a simple year-based regression on Rice production
rice_ts = produce[produce["Particulars"].str.contains("Rice$", na=False, regex=True)]
rice_ts = rice_ts.groupby("year")["value"].sum().reset_index()

X_ts = rice_ts["year"].values.reshape(-1, 1)
y_ts = rice_ts["value"].values
lr_ts = LinearRegression().fit(X_ts, y_ts)
future_years = np.arange(rice_ts["year"].max() + 1, rice_ts["year"].max() + 6).reshape(-1, 1)
future_pred  = lr_ts.predict(future_years)

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(rice_ts["year"], rice_ts["value"], marker="o", linewidth=2.5,
        color=COLORS[0], label="Historical Rice Production")
ax.plot(future_years.flatten(), future_pred, marker="s", linestyle="--",
        linewidth=2, color=COLORS[2], label="Forecasted (Linear Trend)")
ax.fill_between(future_years.flatten(), future_pred * 0.93, future_pred * 1.07,
                alpha=0.2, color=COLORS[2], label="±7% Confidence Band")
ax.set_title("Rice Production Forecast  (India)", fontsize=14, fontweight="bold")
ax.set_xlabel("Year")
ax.set_ylabel("Production (Million Tonnes)")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "13_rice_production_forecast.png"), dpi=150)
plt.close()
print("✔ Plot 13 saved: Rice production forecast")


# ─────────────────────────────────────────────
# 7. Save Best Model
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 7 — SAVING BEST MODEL")
print("="*60)

joblib.dump(best_model_obj, os.path.join(MDL, "best_model.pkl"))
joblib.dump(scaler,         os.path.join(MDL, "scaler.pkl"))
joblib.dump(le_crop,        os.path.join(MDL, "le_crop.pkl"))
joblib.dump(le_state,       os.path.join(MDL, "le_state.pkl"))
print(f"✔ Model saved → models/best_model.pkl  ({best_model_name})")


# ─────────────────────────────────────────────
# 8. Prediction Demo
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 8 — PREDICTION DEMO")
print("="*60)

# Predict yield for: ARHAR crop, Andhra Pradesh
demo_samples = [
    {"Crop": "ARHAR",  "State": "Andhra Pradesh",
     "Cost_CultivationA2FL_per_Ha": 24000, "Cost_CultivationC2_per_Ha": 30000,
     "Cost_Production_per_Quintal": 3050},
    {"Crop": "COTTON", "State": "Gujarat",
     "Cost_CultivationA2FL_per_Ha": 45000, "Cost_CultivationC2_per_Ha": 62000,
     "Cost_Production_per_Quintal": 4800},
]

print("\nDemonstration Predictions:")
print("-"*70)
for sample in demo_samples:
    crop_enc  = le_crop.transform([sample["Crop"]])[0]
    state_enc = le_state.transform([sample["State"]])[0]
    cost_per_yield_proxy = sample["Cost_CultivationC2_per_Ha"] / 15.0   # rough proxy
    feat_vec  = np.array([[crop_enc, state_enc,
                           sample["Cost_CultivationA2FL_per_Ha"],
                           sample["Cost_CultivationC2_per_Ha"],
                           sample["Cost_Production_per_Quintal"],
                           cost_per_yield_proxy]])
    pred      = best_model_obj.predict(feat_vec)[0]
    print(f"  Crop: {sample['Crop']:<8}  |  State: {sample['State']:<20}  "
          f"→  Predicted Yield: {pred:.2f} Quintal/Ha")
print("-"*70)


# ─────────────────────────────────────────────
# 9. Summary Report
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 9 — SUMMARY STATISTICS EXPORT")
print("="*60)

summary = {
    "Dataset Rows"      : len(cost_df),
    "Dataset Features"  : len(FEATURES),
    "Train Samples"     : len(X_train),
    "Test Samples"      : len(X_test),
    "Best Model"        : best_model_name,
    "Best R2"           : round(best_r2, 4),
    "Total Plots"       : 13,
}

pd.DataFrame(summary.items(), columns=["Metric", "Value"])\
  .to_csv(os.path.join(OUT, "project_summary.csv"), index=False)

print("\n✔ Summary saved → outputs/project_summary.csv")
print("\nAll done!  Plots: plots/   Model: models/   Reports: outputs/")
print("="*60 + "\n")
