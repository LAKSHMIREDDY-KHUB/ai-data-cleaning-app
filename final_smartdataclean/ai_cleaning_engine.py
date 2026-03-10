# ==========================================================
# SMARTPREP AI - AI CLEANING ENGINE
# ==========================================================

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest


def apply_ai_cleaning(df, ai_json):

    cleaned_df = df.copy()
    summary = []

    # ---------------------------------------------
    # 1️⃣ DROP COLUMNS
    # ---------------------------------------------
    drop_cols = ai_json.get("drop_columns", [])
    if drop_cols:
        cleaned_df.drop(
            columns=[c for c in drop_cols if c in cleaned_df.columns],
            inplace=True
        )
        summary.append(f"Dropped columns: {drop_cols}")

    # ---------------------------------------------
    # 2️⃣ REMOVE DUPLICATES
    # ---------------------------------------------
    if ai_json.get("remove_duplicates", False):
        before = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        after = len(cleaned_df)
        summary.append(f"Removed {before - after} duplicate rows")

    # ---------------------------------------------
    # 3️⃣ HANDLE MISSING VALUES
    # ---------------------------------------------
    missing_strategy = ai_json.get("missing_strategy", {})

    for col, strategy in missing_strategy.items():

        if col not in cleaned_df.columns:
            continue

        if strategy == "drop":
            cleaned_df.drop(columns=[col], inplace=True)
            summary.append(f"Dropped column {col} (AI decision)")
            continue

        if cleaned_df[col].dtype in ["int64", "float64"]:

            if strategy == "mean":
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())

            elif strategy == "median":
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())

            elif strategy == "knn":
                imputer = KNNImputer(n_neighbors=5)
                cleaned_df[[col]] = imputer.fit_transform(cleaned_df[[col]])

        else:
            # categorical
            if strategy == "mode":
                if not cleaned_df[col].mode().empty:
                    cleaned_df[col] = cleaned_df[col].fillna(
                        cleaned_df[col].mode()[0]
                    )

        summary.append(f"Applied {strategy} strategy on {col}")

    # ---------------------------------------------
    # 4️⃣ OUTLIER HANDLING
    # ---------------------------------------------
    outlier_strategy = ai_json.get("outlier_strategy", "none")
    numeric_cols = cleaned_df.select_dtypes(
        include=["int64", "float64"]
    ).columns

    if outlier_strategy == "remove" and len(numeric_cols) > 0:

        model = IsolationForest(
            contamination=0.05,
            random_state=42
        )

        preds = model.fit_predict(cleaned_df[numeric_cols])

        before = len(cleaned_df)

        # ✅ FIXED: correct boolean indexing
        cleaned_df = cleaned_df.loc[preds != -1]

        after = len(cleaned_df)
        summary.append(
            f"Removed {before - after} outliers using Isolation Forest"
        )

    elif outlier_strategy == "cap":

        for col in numeric_cols:
            Q1 = cleaned_df[col].quantile(0.25)
            Q3 = cleaned_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            cleaned_df[col] = np.where(
                cleaned_df[col] < lower,
                lower,
                np.where(
                    cleaned_df[col] > upper,
                    upper,
                    cleaned_df[col]
                )
            )

        summary.append("Capped outliers using IQR method")

    # ---------------------------------------------
    # FINAL SAFETY FILL (AI GUARANTEE)
    # ---------------------------------------------
    for col in cleaned_df.columns:
        if cleaned_df[col].isna().sum() > 0:
            if cleaned_df[col].dtype in ["int64", "float64"]:
                cleaned_df[col].fillna(
                    cleaned_df[col].median(),
                    inplace=True
                )
            else:
                if not cleaned_df[col].mode().empty:
                    cleaned_df[col].fillna(
                        cleaned_df[col].mode()[0],
                        inplace=True
                    )

    # ---------------------------------------------
    # 5️⃣ RETURN
    # ---------------------------------------------
    return cleaned_df, summary