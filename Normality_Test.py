# ===============================================
# Normality panels (Q–Q + Histogram), no top secondary axis
# ===============================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import FuncFormatter
from scipy import stats

# ---------- A. Appearance ----------
plt.rcParams["font.family"]    = "Times New Roman"
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10

# ---------- B. Load & column mapping ----------
df = pd.read_excel("Biocementation_Data_Analysis_Cleaned.xlsx")
df["Urea Concentration (M)"] = pd.to_numeric(df["Urea Concentration (M)"], errors="coerce")
df.dropna(subset=["Urea Concentration (M)"], inplace=True)

mapping = {
    "Carbonate Content (%)": "X1",
    "Unconfined Compressive Strength (kPa)": "X2",
    "d50 (mm)": "X3",
    "Urea Concentration (M)": "X4",
    "UC Ratio": "X5",
}
df_plot = df.rename(columns=mapping)
pretty = {v: k for k, v in mapping.items()}
cols   = ["X1", "X2", "X3", "X4", "X5"]

# ---------- C. Helpers ----------
def freedman_diaconis_bins(x, max_bins=40):
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return 5
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    h = 2 * iqr * (len(x) ** (-1/3))
    if h <= 0:
        return min(10, max_bins)
    bins = int(np.ceil((x.max() - x.min()) / h))
    return max(5, min(bins, max_bins))

def thousands(v, pos):  # integer with separators
    return f"{v:,.0f}"

def add_stats_text(ax, text, loc="tl"):
    # loc: 'tl' (top-left) or 'br' (bottom-right)
    if loc == "br":
        ax.text(0.98, 0.02, text, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9)
    else:
        ax.text(0.02, 0.98, text, transform=ax.transAxes,
                ha="left", va="top", fontsize=9)

# ---------- D. Main plotting routine ----------
def qq_hist_panels(
    df, cols, pretty_names=None, log10_cols=None,
    dpi=600, save_prefix="normality_",
    combined_name="normality_panels_all.png",
    stats_loc="tl"   # 'tl' or 'br'
):
    pretty = pretty_names or {c: c for c in cols}
    log10_cols = set() if log10_cols is None else set(log10_cols)

    n = len(cols)
    fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(12, 2.8*n), constrained_layout=True)
    if n == 1:
        axes = np.array([axes])

    for i, c in enumerate(cols):
        x = pd.to_numeric(df[c], errors="coerce").dropna().values

        label_suffix = ""
        if c in log10_cols:
            x = x[x > 0]
            x = np.log10(x)
            label_suffix = " (log10)"

        n_i = len(x)
        mu  = np.mean(x) if n_i else np.nan
        sd  = np.std(x, ddof=1) if n_i > 1 else np.nan
        skew = stats.skew(x, bias=False) if n_i > 2 else np.nan
        try:
            W, p = stats.shapiro(x) if n_i >= 3 else (np.nan, np.nan)
        except Exception:
            W, p = (np.nan, np.nan)
        stats_text = f"n={n_i}   μ={mu:.3g}   σ={sd:.3g}   skew={skew:.2f}\nShapiro p={p:.2g}"

        # ---- Q–Q ----
        axq = axes[i, 0]
        (osm, osr), (slope, intercept, _) = stats.probplot(x, dist="norm")
        axq.scatter(osm, osr, s=14, alpha=0.75)
        axq.plot(osm, slope*osm + intercept, lw=1.3)
        axq.set_xlabel("Theoretical Quantiles (Normal)")
        axq.set_ylabel("Observed Data")
        # axq.set_title(f"{pretty.get(c, c)}{label_suffix} — Q–Q")
        axq.grid(True, alpha=0.15)
        axq.spines[["top", "right"]].set_visible(False)
        add_stats_text(axq, stats_text, loc=stats_loc)

        # ---- Histogram + Normal fit ----
        axh = axes[i, 1]
        nb = freedman_diaconis_bins(x, max_bins=40)
        axh.hist(x, bins=nb, density=True, edgecolor="black", alpha=0.85)
        if n_i > 1 and np.isfinite(sd) and sd > 0:
            xx = np.linspace(np.min(x), np.max(x), 400)
            axh.plot(xx, stats.norm.pdf(xx, loc=mu, scale=sd), lw=1.6)
        axh.set_xlabel(pretty.get(c, c) + label_suffix)
        axh.set_ylabel("Density")
        # axh.set_title(f"{pretty.get(c, c)}{label_suffix} — Histogram")
        axh.grid(True, axis="y", alpha=0.15)
        axh.spines[["top", "right"]].set_visible(False)

        # ---- Tick formatting rules ----
        is_ucs = ("UCS" in pretty.get(c, c)) or ("Compressive Strength" in pretty.get(c, c))
        is_log = (c in log10_cols)

        if is_ucs and is_log:
            # UCS plotted in log10 units -> decimal ticks and NO secondary axis
            axh.xaxis.set_major_locator(mtick.MultipleLocator(0.5))           # 2.0, 2.5, 3.0, …
            axh.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
        else:
            # Linear UCS => thousands; non-UCS keep default
            if is_ucs:
                axq.yaxis.set_major_formatter(FuncFormatter(thousands))
                axh.xaxis.set_major_formatter(FuncFormatter(thousands))

        # ---- Save per-variable panel ----
        f_single, a_single = plt.subplots(1, 2, figsize=(12, 2.8), constrained_layout=True)
        # Q–Q single
        a_single[0].scatter(osm, osr, s=14, alpha=0.75)
        a_single[0].plot(osm, slope*osm + intercept, lw=1.3)
        a_single[0].set_xlabel("Theoretical Quantiles (Normal)")
        a_single[0].set_ylabel("Observed Data")
        # a_single[0].set_title(f"{pretty.get(c, c)}{label_suffix} — Q–Q")
        a_single[0].grid(True, alpha=0.15)
        a_single[0].spines[["top", "right"]].set_visible(False)
        add_stats_text(a_single[0], stats_text, loc=stats_loc)

        # Hist single
        a_single[1].hist(x, bins=nb, density=True, edgecolor="black", alpha=0.85)
        if n_i > 1 and np.isfinite(sd) and sd > 0:
            a_single[1].plot(xx, stats.norm.pdf(xx, loc=mu, scale=sd), lw=1.6)
        a_single[1].set_xlabel(pretty.get(c, c) + label_suffix)
        a_single[1].set_ylabel("Density")
        # a_single[1].set_title(f"{pretty.get(c, c)}{label_suffix} — Histogram")
        a_single[1].grid(True, axis="y", alpha=0.15)
        a_single[1].spines[["top", "right"]].set_visible(False)

        # Mirror tick rules for single panel
        if is_ucs and is_log:
            a_single[1].xaxis.set_major_locator(mtick.MultipleLocator(0.5))
            a_single[1].xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
        else:
            if is_ucs:
                a_single[0].yaxis.set_major_formatter(FuncFormatter(thousands))
                a_single[1].xaxis.set_major_formatter(FuncFormatter(thousands))

        f_single.savefig(f"{save_prefix}{c}.png", dpi=dpi)
        plt.close(f_single)

    fig.savefig(combined_name, dpi=dpi)
    plt.show()
    return fig

# ---------- E. Run ----------
# Log-transform strongly right-skewed variables (UCS typical)
log10_vars = {"X2"}    # add "X1" if Carbonate Content is very skewed
stats_location = "tl"  # 'tl' or 'br'

qq_hist_panels(
    df_plot,
    cols=cols,
    pretty_names=pretty,
    log10_cols=log10_vars,
    dpi=600,
    save_prefix="normality_",
    combined_name="normality_panels_all.png",
    stats_loc=stats_location
)
