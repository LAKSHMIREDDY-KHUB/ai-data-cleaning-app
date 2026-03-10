import pandas as pd
import numpy as np
# ==========================================================
# DATASET HEALTH SCORE 
# ==========================================================
def calculate_health_score(df):
    """
    Returns a simple dataset health score (0–100)
    based on missing values, duplicates, and outliers.
    """

    if df is None or df.empty:
        return 0.0

    score = 100
    total_rows = len(df)
    total_cols = len(df.columns)

    # ------------------------------------------
    # 1️⃣ Missing Values Penalty (Max -40)
    # ------------------------------------------
    total_cells = total_rows * total_cols
    missing_cells = df.isnull().sum().sum()

    if total_cells > 0:
        missing_pct = (missing_cells / total_cells) * 100
        score -= min(missing_pct * 0.6, 40)

    # ------------------------------------------
    # 2️⃣ Duplicate Rows Penalty (Max -20)
    # ------------------------------------------
    duplicate_rows = df.duplicated().sum()
    if total_rows > 0:
        duplicate_pct = (duplicate_rows / total_rows) * 100
        score -= min(duplicate_pct * 1.5, 20)

    # ------------------------------------------
    # 3️⃣ Outlier Penalty (Numeric Columns) (Max -30)
    # ------------------------------------------
    numeric_cols = df.select_dtypes(include=np.number)

    outlier_penalty = 0

    for col in numeric_cols.columns:
        series = numeric_cols[col].dropna()

        if len(series) < 10:
            continue

        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1

        if IQR == 0:
            continue

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = ((series < lower) | (series > upper)).sum()
        outlier_pct = (outliers / total_rows) * 100

        outlier_penalty += min(outlier_pct * 0.3, 10)

    score -= min(outlier_penalty, 30)

    # ------------------------------------------
    # Clamp Score
    # ------------------------------------------
    score = round(max(min(score, 100), 0), 2)

    return score


