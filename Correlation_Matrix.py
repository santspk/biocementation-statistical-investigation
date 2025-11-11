import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

# -----------------------------
# 1) Correlation matrix
# -----------------------------
# Example: load your data (replace with your path or existing DataFrame)
df = pd.read_excel("Biocementation_Data_Analysis_Cleaned.xlsx")
df["Urea Concentration (M)"] = pd.to_numeric(
    df["Urea Concentration (M)"], errors="coerce"
)
df.dropna(subset=["Urea Concentration (M)"], inplace=True)
# Specify the variables you want to profile (order = plotting order)
# Mapping descriptive names to symbolic names
mapping = {
    "Carbonate Content (%)": "X1",
    "Unconfined Compressive Strength (kPa)": "X2",
    "d50 (mm)": "X3",
    "Urea Concentration (M)": "X4",
    "UC Ratio": "X5"
}

# Rename columns in a copy for plotting
df_plot = df.rename(columns=mapping)

cols = ["X1", "X2", "X3", "X4", "X5"]


# Choose correlation type: "pearson" (linear), "spearman" (rank), or "kendall"
corr_method = "pearson"

# Compute the correlation matrix
# corr = df[cols].corr(method=corr_method)
# print(corr.round(3))

# -----------------------------
# 2) Scatterplot-matrix with correlations
# -----------------------------
def scatter_matrix_with_corr(df, cols, corr_method="spearman",
                             bins=12, figsize=(8, 8), s=8, alpha=0.6,
                             number_fmt="{:.3f}"):
    """
    df         : pandas DataFrame containing the columns in `cols`
    cols       : list of column names to include (order matters)
    corr_method: 'pearson', 'spearman', or 'kendall'
    bins       : histogram bins for diagonal cells
    figsize    : overall figure size
    s, alpha   : scatter size and transparency
    number_fmt : formatting for correlation coefficients
    """
    data = df[cols].dropna()
    k = len(cols)
    corr = data.corr(method=corr_method)

    fig, axes = plt.subplots(k, k, figsize=figsize, squeeze=False)
    plt.subplots_adjust(wspace=0.15, hspace=0.15)

    # Pre-compute axis limits per variable so rows/cols line up
    lims = {}
    for c in cols:
        x = data[c].to_numpy()
        # small padding around min/max
        pad = 0.03 * (np.nanmax(x) - np.nanmin(x) + 1e-12)
        lims[c] = (np.nanmin(x) - pad, np.nanmax(x) + pad)

    for i in range(k):
        for j in range(k):
            ax = axes[i, j]
            xi, xj = cols[i], cols[j]

            # Diagonal: histograms
            if i == j:
                ax.hist(data[xi].to_numpy(), bins=bins, edgecolor="black")
                ax.set_xlim(lims[xi])
                # Clean look
                ax.spines[["top", "right"]].set_visible(False)

            # Lower triangle: scatter
            elif i > j:
                ax.scatter(data[xj], data[xi], s=s, alpha=alpha)
                ax.set_xlim(lims[xj]); ax.set_ylim(lims[xi])
                ax.spines[["top", "right"]].set_visible(False)

            # Upper triangle: correlation text only
            else:
                ax.axis("off")
                r = corr.loc[xi, xj]
                txt = number_fmt.format(r)
                ax.text(0.5, 0.5, txt,
                        ha="center", va="center", fontsize=12)

            # Tick labels only on the outer edge
            if i < k - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(xj)
            if j > 0:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(xi)

    fig.suptitle(f"Scatterplot Matrix with {corr_method.title()} Correlations", y=0.98)
    return fig, axes, corr

# -----------------------------
# Example usage
# -----------------------------
fig, axes, corr = scatter_matrix_with_corr(df_plot, cols,
                                           corr_method="spearman",
                                           bins=12, figsize=(8,8),
                                           s=8, alpha=0.6,
                                           number_fmt="{:.3f}")
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig("scatter_matrix_with_corr_spearman.png", dpi=600)
print(corr.round(3))
plt.show()
