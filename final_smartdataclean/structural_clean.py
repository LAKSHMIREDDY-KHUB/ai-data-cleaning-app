import pandas as pd
import numpy as np

def structural_auto_clean(df, missing_threshold=40, drop_ids=False):

    # -------------------------------------------------
    # 🔥 Step 1: Remove duplicate columns (critical fix)
    # -------------------------------------------------
    df = df.loc[:, ~df.columns.duplicated()].copy()

    df_cleaned = df.copy()
    total_rows = len(df_cleaned)

    auto_removed = []
    suggested_ids = []

    # Safety check
    if total_rows == 0:
        return df_cleaned, auto_removed, suggested_ids

    # -------------------------------------------------
    # Iterate safely over COPY of column names
    # -------------------------------------------------
    for col in list(df.columns):

        # Extra safety: skip if column already dropped
        if col not in df_cleaned.columns:
            continue

        col_data = df_cleaned[col]

        # 🔥 Ensure we always work with Series
        if isinstance(col_data, pd.DataFrame):
            col_data = col_data.iloc[:, 0]

        # ----------------------------
        # Missing %
        # ----------------------------
        missing_count = col_data.isna().sum()
        missing_percent = (missing_count / total_rows) * 100

        # ----------------------------
        # Unique count
        # ----------------------------
        unique_count = col_data.nunique(dropna=True)
        uniqueness_ratio = unique_count / total_rows

        # ----------------------------
        # 1️⃣ Drop constant columns
        # ----------------------------
        if unique_count <= 1:
            df_cleaned.drop(columns=[col], inplace=True)
            auto_removed.append(f"{col} (Constant)")
            continue

        # ----------------------------
        # 2️⃣ Drop high missing columns
        # ----------------------------
        if missing_percent > missing_threshold:
            df_cleaned.drop(columns=[col], inplace=True)
            auto_removed.append(f"{col} (> {missing_threshold}% Missing)")
            continue

        # ----------------------------
        # 3️⃣ Detect potential ID column
        # ----------------------------
        if uniqueness_ratio > 0.95:
            if drop_ids:
                df_cleaned.drop(columns=[col], inplace=True)
                auto_removed.append(f"{col} (ID Column)")
            else:
                suggested_ids.append(col)

    return df_cleaned, auto_removed, suggested_ids