# ===============================================
# Stepwise (nested) regression for UCS models M1–M8
# Files:
#   - Biocementation_Data_Train_With_Transforms.xlsx
#   - Biocementation_Data_Test_With_Transforms.xlsx
# Author: Santosh Pokharel
# Date: October 2025
# Purpose: Fit multiple regression models with increasing complexity to predict UCS
# Generative AI assistance: ChatGPT-4, Acess granted by Arizona State University
# ===============================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from matplotlib.ticker import MaxNLocator


plt.rcParams["font.family"] = "Times New Roman"

# --------- USER CONFIG ---------
train_path = "Biocementation_Data_Train_With_Transforms.xlsx"
test_path  = "Biocementation_Data_Test_With_Transforms.xlsx"

y_col  = "Unconfined Compressive Strength (kPa)"  # response
# Predictor column names 
x_cc   = "Xcc"                # transformed carbonate variable
d50    = "d50 (mm)"
u_col  = "Urea Concentration (M)"
uc_col = "UC Ratio"

# Dummy columns for categorical variables:
# Sand Type: reference = SP; XS1=SM, XS2=SP-SM, XS3=GP
XS = ["XS1", "XS2", "XS3"]

# Sand Mineral: reference = Silica; XM=Calcareous
XM = ["XM"]

# Pathway: reference = MICP; XP=EICP
XP = ["XP"]

# Treatment Method: reference = INJECTION;
# XT1=IMMERSION, XT2=PERCOLATION, XT3=MIX-AND-COMPACT
XT = ["XT1", "XT2", "XT3"]

# Output folder for figures and CSV tables
out_dir = "stepwise_outputs_Power_UCS_log"   # will be created if missing
# --------------------------------

import os
os.makedirs(out_dir, exist_ok=True)

# ---------- Load data ----------cd 
df_tr = pd.read_excel(train_path)
df_te = pd.read_excel(test_path)

# Keep only rows with complete data for the largest model (fair comparison)
all_predictors = [x_cc, d50, u_col, uc_col] + XS + XM + XP + XT
needed_cols = [y_col] + all_predictors
df_tr = df_tr[needed_cols].dropna().copy()
df_te = df_te[needed_cols].dropna().copy()

# ---------- Utility functions ----------
def fit_ols(df, y, Xcols):
    """Fit OLS with intercept; return results object and design matrices."""
    X = sm.add_constant(df[Xcols], has_constant='add')
    yv = df[y].astype(float).values
    model = sm.OLS(yv, X.astype(float))
    res = model.fit()
    return res, X, yv

def metrics_from_results(res, y_true, X):
    """Compute core diagnostics on the training sample used to fit."""
    n = X.shape[0]
    p = X.shape[1]               # includes intercept
    sse = np.sum(res.resid**2)
    mse = sse / (n - p)          # residual variance (sigma^2 hat)
    rmse = np.sqrt(mse)
    return {
        "n_train": n,
        "p_params": p,
        "SSE": sse,
        "sigma2_hat": mse,
        "RMSE_train": rmse,
        "R2": res.rsquared,
        "Adj_R2": res.rsquared_adj,
        "AIC": res.aic,
        "BIC": res.bic
    }

def test_metrics(res, df_test, y, Xcols):
    """Compute test-set metrics using the model's fitted params."""
    X_te = sm.add_constant(df_test[Xcols], has_constant='add').astype(float)
    y_te = df_test[y].astype(float).values
    y_hat = res.predict(X_te)
    resid = y_te - y_hat
    rmse = np.sqrt(np.mean(resid**2))
    r2 = 1.0 - np.sum(resid**2) / np.sum((y_te - y_te.mean())**2)
    mae = np.mean(np.abs(resid))
    return {"RMSE_test": rmse, "R2_test": r2, "MAE_test": mae}, y_hat, resid

def residual_plot(y_true, y_hat, title, fname):
    """Residuals vs fitted with 0 line."""
    resid = y_true - y_hat
    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=200)
    ax.scatter(y_hat, resid, s=22, alpha=0.8, edgecolors="none")
    ax.axhline(0, linewidth=1.2)
    ax.set_xlabel("Fitted UCS (kPa)")
    ax.set_ylabel("Residual (kPa)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, fname))
    plt.close(fig)

# ---------- Model definitions (nested) ----------
MODELS = {
    "M1": [x_cc],
    "M2": [x_cc, d50],
    "M3": [x_cc, d50, u_col],
    "M4": [x_cc, d50, u_col, uc_col],
    "M5": [x_cc, d50, u_col, uc_col] + XS,
    "M6": [x_cc, d50, u_col, uc_col] + XS + XM,
    "M7": [x_cc, d50, u_col, uc_col] + XS + XM + XP,
    "M8": [x_cc, d50, u_col, uc_col] + XS + XM + XP + XT,
}

# Sanity check: ensure all referenced columns exist
for m, cols in MODELS.items():
    missing = [c for c in cols if c not in df_tr.columns]
    if missing:
        raise ValueError(f"{m} missing columns in TRAIN: {missing}")
    missing_te = [c for c in cols if c not in df_te.columns]
    if missing_te:
        raise ValueError(f"{m} missing columns in TEST: {missing_te}")

# ---------- Fit loop ----------
rows_summary = []
anova_rows = []
coef_tables = {}

prev_res = None
prev_name = None

for name, cols in MODELS.items():
    # Fit on training
    res, X_tr, y_tr = fit_ols(df_tr, y_col, cols)

    # Per-model diagnostics (train)
    base = {"Model": name, "Predictors": ", ".join(cols)}
    base.update(metrics_from_results(res, y_tr, X_tr))

    # Coefficient table (t, p)
    ct = pd.DataFrame({
        "coef": res.params,
        "std_err": res.bse,
        "t": res.tvalues,
        "p_value": res.pvalues
    })
    coef_tables[name] = ct
    ct.to_csv(os.path.join(out_dir, f"{name}_coeff_ttests.csv"))

    # Test-set metrics and residual plots
    test_mets, yhat_test, resid_test = test_metrics(res, df_te, y_col, cols)
    base.update(test_mets)

    # Residual plots
    yhat_train = res.fittedvalues
    residual_plot(y_tr, yhat_train,
                  f"{name} Residuals vs Fitted (TRAIN)",
                  f"{name}_resid_vs_fitted_TRAIN.png")
    residual_plot(df_te[y_col].values, yhat_test,
                  f"{name} Residuals vs Fitted (TEST)",
                  f"{name}_resid_vs_fitted_TEST.png")

    # Nested F-test vs previous model (only if nested and previous exists)
    if prev_res is not None:
        # statsmodels offers compare_f_test for nested models fit on same data
        F_stat, p_val, df_diff = res.compare_f_test(prev_res)
        anova_rows.append({
            "Model_prev": prev_name,
            "Model_curr": name,
            "df_diff": int(df_diff),
            "F": float(F_stat),
            "p_value": float(p_val)
        })
    else:
        anova_rows.append({
            "Model_prev": None, "Model_curr": name,
            "df_diff": np.nan, "F": np.nan, "p_value": np.nan
        })

    rows_summary.append(base)
    prev_res = res
    prev_name = name

# ---------- Save summaries ----------
summary_df = pd.DataFrame(rows_summary)
summary_df = summary_df[
    ["Model","Predictors","n_train","p_params",
     "R2","Adj_R2","AIC","BIC","SSE","sigma2_hat","RMSE_train",
     "R2_test","RMSE_test","MAE_test"]
]
summary_df.to_csv(os.path.join(out_dir, "model_summary_metrics.csv"), index=False)

anova_df = pd.DataFrame(anova_rows)
anova_df.to_csv(os.path.join(out_dir, "incremental_F_tests.csv"), index=False)

# Also write a single Excel workbook with all outputs
with pd.ExcelWriter(os.path.join(out_dir, "stepwise_results.xlsx")) as xl:
    summary_df.to_excel(xl, sheet_name="Model_Metrics", index=False)
    anova_df.to_excel(xl, sheet_name="Incremental_F", index=False)
    for k, v in coef_tables.items():
        v.to_excel(xl, sheet_name=f"{k}_Coefs", index=True)

print("Done. Files saved to:", os.path.abspath(out_dir))


# ============================================================
# ADDITIONAL ANALYSES: Predicted vs Measured plots (TEST),
# Model-level F-test significance (α=0.05),
# and sigma^2 (residual variance)
# ============================================================



model_sig_rows = []

for name, cols in MODELS.items():
    # Refit on training (to ensure res object)
    res, X_tr, y_tr = fit_ols(df_tr, y_col, cols)

    # Test dataset prediction
    X_te = sm.add_constant(df_te[cols], has_constant='add').astype(float)
    y_te = df_te[y_col].astype(float).values
    y_pred = res.predict(X_te)

    # 1️⃣ Plot Predicted vs Measured UCS (TEST)
    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=200)
    ax.scatter(y_te, y_pred, s=22, alpha=0.8)
    lims = [min(y_te.min(), y_pred.min()), max(y_te.max(), y_pred.max())]
    ax.plot(lims, lims, 'r--', linewidth=1.2, label='1:1 Line')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Measured UCS (kPa)")
    ax.set_ylabel("Predicted UCS (kPa)")
    ax.set_title(f"{name}: Predicted vs Measured UCS (TEST)")
    ax.legend()
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{name}_Predicted_vs_Measured_TEST.png"))
    plt.close(fig)

    # 2️⃣ Model F-test significance
    F_val = res.fvalue
    F_p = res.f_pvalue
    signif = "Significant (p<0.05)" if F_p < 0.05 else "Not Significant"

    # 3️⃣ Residual variance sigma^2
    n = X_tr.shape[0]
    p = X_tr.shape[1]
    sigma2 = np.sum(res.resid**2) / (n - p)

    model_sig_rows.append({
        "Model": name,
        "F_statistic": F_val,
        "p_value": F_p,
        "Significance_@5%": signif,
        "sigma2_hat": sigma2
    })

# Save the model-level F significance and sigma² to Excel
fsummary_df = pd.DataFrame(model_sig_rows)
fsummary_df.to_csv(os.path.join(out_dir, "model_Ftest_sigma2_summary.csv"), index=False)

# Append to the existing Excel workbook
with pd.ExcelWriter(os.path.join(out_dir, "stepwise_results.xlsx"), mode="a", engine="openpyxl", if_sheet_exists="replace") as xl:
    fsummary_df.to_excel(xl, sheet_name="Model_Ftest_Sigma2", index=False)

print("Additional plots and F-test significance results saved.")

# ============================================================
# FIXED EXTRA PLOT:
# UCS vs Carbonate Content (TRAIN & TEST) with M1 line + 95% CI
# NOTE: We reconstruct Carbonate Content (CCC) from Xcc using
#       CCC = ln(Xcc) / 0.0566 so we don't need the raw CCC column.
# ============================================================

# --- Refit M1 (UCS ~ Xcc) on the same training data used above ---
m1_res, X_tr_m1, y_tr_m1 = fit_ols(df_tr, y_col, [x_cc])

# --- Compute Carbonate Content (%) from Xcc for train and test ---
#     If you used a different transform than Xcc = exp(0.0566*CCC),
#     change 0.0566 accordingly.
_k = 0.0566
ccc_train = np.log(df_tr[x_cc].astype(float)) / _k
ccc_test  = np.log(df_te[x_cc].astype(float)) / _k

# --- Smooth grid over CCC spanning both datasets ---
ccc_min = float(np.nanmin(np.concatenate([ccc_train.values, ccc_test.values])))
ccc_max = float(np.nanmax(np.concatenate([ccc_train.values, ccc_test.values])))
ccc_grid = np.linspace(ccc_min, ccc_max, 200)

# Map CCC grid -> Xcc for prediction
xcc_grid = np.exp(_k * ccc_grid)

# Build exogenous design for predictions (const + Xcc)
exog_pred = sm.add_constant(pd.DataFrame({x_cc: xcc_grid}), has_constant='add')

# Mean prediction and 95% CI from statsmodels
pred_frame = m1_res.get_prediction(exog=exog_pred).summary_frame(alpha=0.05)
y_mean = pred_frame["mean"].values
ci_low = pred_frame["mean_ci_lower"].values
ci_high = pred_frame["mean_ci_upper"].values

# --- Scatter TRAIN & TEST on CCC vs UCS + overlay M1 with CI ---
fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
ax.scatter(ccc_train.values, df_tr[y_col].values, s=22, alpha=0.8, label="Train")
ax.scatter(ccc_test.values,  df_te[y_col].values,  s=22, alpha=0.8, label="Test")

ax.plot(ccc_grid, y_mean, linewidth=1.6, label="M1 fit (UCS ~ Xcc)")
ax.fill_between(ccc_grid, ci_low, ci_high, alpha=0.2, label="95% CI (mean)")

ax.set_xlabel("Carbonate Content (%)")
ax.set_ylabel("UCS (kPa)")
ax.set_title("UCS vs Carbonate Content with M1 Regression and 95% CI")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(out_dir, "UCS_vs_Carbonate_with_M1_CI.png"))
plt.close(fig)

# ============================================================
# UCS vs Carbonate Content with M1 Regression + 95% CI & PI
# ============================================================



# Refit M1
m1_res, X_tr_m1, y_tr_m1 = fit_ols(df_tr, y_col, [x_cc])

# Compute CCC from Xcc (using Xcc = exp(0.0566*CCC))
_k = 0.0566
ccc_train = np.log(df_tr[x_cc].astype(float)) / _k
ccc_test  = np.log(df_te[x_cc].astype(float)) / _k

ccc_grid = np.linspace(min(ccc_train.min(), ccc_test.min()),
                       max(ccc_train.max(), ccc_test.max()), 200)
xcc_grid = np.exp(_k * ccc_grid)
exog_pred = sm.add_constant(pd.DataFrame({x_cc: xcc_grid}), has_constant='add')

# Get both mean CI and prediction interval
pred_frame = m1_res.get_prediction(exog_pred).summary_frame(alpha=0.05)
y_mean = pred_frame["mean"]
ci_low, ci_high = pred_frame["mean_ci_lower"], pred_frame["mean_ci_upper"]
pi_low, pi_high = pred_frame["obs_ci_lower"], pred_frame["obs_ci_upper"]

# --- Plot ---
fig, ax = plt.subplots(figsize=(8, 6), dpi=600)
ax.scatter(ccc_train, df_tr[y_col], s=22, alpha=0.8, label="Train")
ax.scatter(ccc_test, df_te[y_col], s=22, alpha=0.8, label="Test")

# Regression line
ax.plot(ccc_grid, y_mean, 'b-', linewidth=1.8, label="M1 fit (UCS ~ Xcc)")
# 95% CI for mean
ax.fill_between(ccc_grid, ci_low, ci_high, color='blue', alpha=0.2, label="95% CI (mean)")
# 95% Prediction Interval
ax.fill_between(ccc_grid, pi_low, pi_high, color='green', alpha=0.15, label="95% PI (individual)")

ax.set_xlabel("Carbonate Content (%)")
ax.set_ylabel("UCS (kPa)")
ax.set_xlim(0, ccc_grid.max())
ax.set_ylim(10, 10000)  # min axis zero, max UCS 1000

ax.set_yscale("log", base=10)

ax.set_title("UCS vs Carbonate Content with M1 Regression, 95% CI & PI")
ax.legend()
ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
fig.tight_layout()
fig.savefig(os.path.join(out_dir, "UCS_vs_Carbonate_with_M1_CI_PI_min_axis_zero_max_UCS_10000_log.png"))
plt.close(fig)
