# ---------------------------------------------
# Bivariate outlier detection with Author–Date legend (outside)
# Now also saves a cleaned Excel file (outliers removed)
# ---------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import chi2

plt.rcParams["font.family"] = "Times New Roman"

# ========= USER INPUTS =========
excel_path = "Biocementation_Data.xlsx"
x_col = "Carbonate Content (%)"
y_col = "Unconfined Compressive Strength (kPa)"
author_col = "First Author"
date_col = "Date of Publication"
conf_level = 0.99  # confidence level (95% or 99%)
save_outliers_csv = "bivariate_outliers.csv"
save_cleaned_excel = "Biocementation_Data_Cleaned_NoOutliers.xlsx"
save_fig_path = "bivariate_confidence_ellipse.png"
point_alpha = 0.85  # scatter transparency
# ===============================

# ---------- 1) Load & clean ----------
df_raw = pd.read_excel(excel_path)

# Coerce to numeric
df = df_raw.copy()
df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
df[y_col] = pd.to_numeric(df[y_col], errors="coerce")

# Build Author–Date labels
def _to_year(val):
    try:
        return pd.to_datetime(val).year
    except Exception:
        return str(val)

df[author_col] = df[author_col].astype(str).str.strip()
df[date_col] = df[date_col].astype(str).str.strip()
years = df[date_col].apply(_to_year)
df["Author-Date"] = df[author_col] + " (" + years.astype(str) + ")"

df = df.dropna(subset=[x_col, y_col, "Author-Date"]).reset_index(drop=True)

X = df[[x_col, y_col]].to_numpy()
n = X.shape[0]
if n < 3:
    raise ValueError("Not enough valid rows (need ≥3 after cleaning).")

# ---------- 2) Mean, covariance ----------
mu = X.mean(axis=0)
S = np.cov(X, rowvar=False)

# Regularize covariance if near-singular
eps = 1e-9
S_reg = S + eps * np.eye(2)
S_inv = np.linalg.inv(S_reg)

# ---------- 3) Mahalanobis distances ----------
diff = X - mu
D2 = np.einsum("ij,jk,ik->i", diff, S_inv, diff)

# χ² threshold
threshold = chi2.ppf(conf_level, df=2)

# Outliers
is_outlier = D2 > threshold
n_out = int(is_outlier.sum())

# ---------- 4) Save outliers and cleaned dataset ----------
outliers_df = df.loc[is_outlier, [x_col, y_col, "Author-Date", author_col, date_col]].copy()
outliers_df["Mahalanobis^2"] = D2[is_outlier]
outliers_df.to_csv(save_outliers_csv, index=False)

# Save cleaned dataset (outliers removed)
df_cleaned = df.loc[~is_outlier].copy()
df_cleaned.to_excel(save_cleaned_excel, index=False)

# ---------- 5) Confidence ellipse ----------
eigvals, eigvecs = np.linalg.eigh(S_reg)
radii = np.sqrt(eigvals * threshold)

theta = np.linspace(0, 2*np.pi, 400)
circle = np.vstack([np.cos(theta), np.sin(theta)])
ellipse = (eigvecs @ (radii[:, None] * circle)).T
ellipse[:, 0] += mu[0]
ellipse[:, 1] += mu[1]

# ---------- 6) Plot ----------
fig, ax = plt.subplots(figsize=(7.6, 6.4), dpi=150)

# Unique Author–Date groups
author_dates = sorted(df["Author-Date"].unique())  # alphabetically sorted

# Marker & color maps
markers = ["o", "s", "^", "D", "v", "P", "*", "X", "<", ">", "h", "H", "8", "p"]
cmap = plt.cm.get_cmap("tab20", len(author_dates))
color_map = {ad: cmap(i) for i, ad in enumerate(author_dates)}
marker_map = {ad: markers[i % len(markers)] for i, ad in enumerate(author_dates)}

# Plot inliers
for ad, sub in df.loc[~is_outlier].groupby("Author-Date"):
    ax.scatter(sub[x_col], sub[y_col],
               s=28, alpha=point_alpha,
               label=ad,
               marker=marker_map[ad],
               facecolor=color_map[ad],
               edgecolor="none")

# Plot outliers (same color/marker, black edge)
for ad, sub in df.loc[is_outlier].groupby("Author-Date"):
    ax.scatter(sub[x_col], sub[y_col],
               s=55, alpha=1.0,
               marker=marker_map[ad],
               facecolor=color_map[ad],
               edgecolor="k", linewidth=0.7)

# Ellipse & mean
ax.plot(ellipse[:, 0], ellipse[:, 1], linewidth=2.0, color="k")
ax.scatter(mu[0], mu[1], s=64, marker="+", linewidth=2, color="k")

# Labels & grid
ax.set_xlabel(x_col)
ax.set_ylabel(y_col)
ax.grid(True, alpha=0.3)
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)

# ---------- 7) Legends ----------
handles = [Line2D([0], [0], marker=marker_map[ad], linestyle="",
                  markerfacecolor=color_map[ad], markeredgecolor="none", markersize=7)
           for ad in author_dates]
labels = author_dates

fig.subplots_adjust(right=0.80)

leg1 = ax.legend(
    handles, labels,
    title="Author–Date",
    loc="upper left",
    bbox_to_anchor=(1.01, 1.00),
    frameon=True,
    fontsize=8.5,
    title_fontsize=9.5,
    ncol=2,
    markerscale=1.1,
    handlelength=1.0,
    handletextpad=0.4,
    borderaxespad=0.0,
)

leg2 = ax.legend(
    [Line2D([0],[0], color="k", lw=2),
     Line2D([0],[0], marker="+", color="k", lw=0, markersize=9)],
    [f"{int(conf_level*100)}% CI ellipse", "Mean"],
    loc="upper right",
    frameon=True,
    fontsize=9,
)
ax.add_artist(leg1)

# ---------- 8) Save & show ----------
fig.tight_layout()
fig.savefig(save_fig_path, bbox_inches="tight")
plt.show()

# ---------- 9) Summary ----------
total = len(df_raw)
clean_n = len(df)
print("==== Bivariate Outlier Analysis ====")
print(f"Confidence level               : {conf_level:.2%}")
print(f"Total rows in file             : {total}")
print(f"Rows used after cleaning       : {clean_n}")
print(f"Outliers detected              : {n_out}")
print(f"Chi-square threshold (df=2)    : {threshold:.3f}")
print(f"Saved outliers to              : {save_outliers_csv}")
print(f"Saved cleaned dataset to       : {save_cleaned_excel}")
print(f"Saved figure to                : {save_fig_path}")
