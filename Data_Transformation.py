# ---------------------------------------------
# Transformations:
# 1) X_CC conversion using Deng et al. Power Model
# 2) Indicator coding with specified reference categories
# ---------------------------------------------
import numpy as np
import pandas as pd
mode = ["Test", "Train"]  # "Test" or "Train"

for dset in mode:# ========= USER INPUTS =========
    excel_path = f"Biocementation_Data_{dset}.xlsx"
    save_path  = f"Biocementation_Data_{dset}_logUCS_logCCC_With_Transforms.xlsx"

    ucs_col = "Unconfined Compressive Strength (kPa)"
    cc_col      = "Carbonate Content (%)"
    soil_col    = "Soil Type (USCS)"
    mineral_col = "Sand Mineral"
    path_col    = "Pathway"
    treat_col   = "Treatment Method"
    # ===============================

    # ---------- 1) Load ----------
    df = pd.read_excel(excel_path)

    # ---------- 2) Helper: normalize category text ----------
    def norm(s):
        if pd.isna(s):
            return np.nan
        s = str(s).strip()
        # Standardize multiple spaces and casing; keep hyphens as-is
        s = " ".join(s.split())        # collapse internal spaces
        return s.upper()

    # Create normalized shadow columns just for matching (not saved)
    _soil    = df[soil_col].map(norm)
    _mineral = df[mineral_col].map(norm)
    _path    = df[path_col].map(norm)
    _treat   = df[treat_col].map(norm)

    # ---------- 3) Numeric transform ----------
    # df["Xcc"] = np.exp(0.13 * pd.to_numeric(df[cc_col], errors="coerce"))
    df["Xcc"] = np.log10(pd.to_numeric(df[cc_col], errors="coerce"))
    df["logUCS"] = np.log10(pd.to_numeric(df[ucs_col], errors="coerce"))
    # ---------- 4) Indicator coding (reference shown in comments) ----------
    df["lnUCS"] = np.log(df[ucs_col])
    # Soil Type: reference = SP
    # New variables: XS1(SM), XS2(SP-SM), XS3(GP)
    df["XS1"] = (_soil == "SM").astype("Int64")
    df["XS2"] = (_soil == "SP-SM").astype("Int64")
    df["XS3"] = (_soil == "GP").astype("Int64")
    # Note: when Soil Type == "SP" (reference), all XS* = 0.
    # Any other/unrecognized category will yield all XS* = 0 as well.

    # Sand Mineral: reference = SILICA
    # New variable: XM(CALCAREOUS = 1)
    df["XM"] = (_mineral == "CALCAREOUS").astype("Int64")
    # If SILICA (reference) -> XM = 0.

    # Pathway: reference = MICP
    # New variable: XP(EICP = 1)
    df["XP"] = (_path == "EICP").astype("Int64")

    # Treatment Method: reference = INJECTION
    # New variables: XT1(IMMERSION), XT2(PERCOLATION), XT3(MIX-AND-COMPACT)
    df["XT1"] = (_treat == "IMMERSION").astype("Int64")
    df["XT2"] = (_treat == "PERCOLATION").astype("Int64")
    df["XT3"] = (_treat == "MIX-COMPACT").astype("Int64")

    # ---------- 5) Optional: quick sanity report ----------
    def counts(series, title):
        c = series.map(norm).value_counts(dropna=False).sort_index()
        print(f"\n{title} (normalized) counts:")
        print(c)

    counts(df[soil_col],    "Soil Type")
    counts(df[mineral_col], "Sand Mineral")
    counts(df[path_col],    "Pathway")
    counts(df[treat_col],   "Treatment Method")

    print("\nReference categories encoded implicitly as zeros:")
    print(" - Soil Type reference: SP (XS1=XS2=XS3=0)")
    print(" - Sand Mineral reference: SILICA (XM=0)")
    print(" - Pathway reference: MICP (XP=0)")
    print(" - Treatment Method reference: INJECTION (XT1=XT2=XT3=0)")

    # ---------- 6) Save ----------
    df.to_excel(save_path, index=False)
    print(f"\nSaved transformed dataset to: {save_path}")
