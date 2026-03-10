import pandas as pd
import numpy as np

def auto_clean(df, numeric_strategy="median"):
    """
    RULE-BASED CLEANING ENGINE
    -------------------------
    1. Convert numeric-like object columns to numeric
    2. Handle missing values (numeric + categorical)
    3. Trim text columns
    4. Remove duplicates
    """

    cleaned_df = df.copy()
    cleaning_log = []

    # ----------------------------------
    # 1️⃣ FORCE NUMERIC CONVERSION
    # ----------------------------------
    for col in cleaned_df.columns:

        if cleaned_df[col].dtype == "object":

            # Try numeric conversion
            converted = pd.to_numeric(cleaned_df[col], errors="coerce")

            non_null_ratio = converted.notnull().mean()

            # If majority becomes numeric → convert
            if non_null_ratio > 0.6:
                cleaned_df[col] = converted
                cleaning_log.append(f"{col}: Converted to numeric")

    # ----------------------------------
    # 2️⃣ HANDLE MISSING VALUES
    # ----------------------------------
    for col in cleaned_df.columns:

        if pd.api.types.is_numeric_dtype(cleaned_df[col]):

            if cleaned_df[col].isnull().sum() > 0:

                if numeric_strategy == "median":
                    value = cleaned_df[col].median()
                else:
                    value = cleaned_df[col].mean()

                cleaned_df[col].fillna(value, inplace=True)
                cleaning_log.append(f"{col}: Missing filled with {numeric_strategy}")

        else:
            if cleaned_df[col].isnull().sum() > 0:
                mode = cleaned_df[col].mode()
                fill_value = mode[0] if not mode.empty else "Unknown"
                cleaned_df[col].fillna(fill_value, inplace=True)
                cleaning_log.append(f"{col}: Missing filled with mode")

    # ----------------------------------
    # 3️⃣ TEXT CLEANING
    # ----------------------------------
    for col in cleaned_df.select_dtypes(include="object").columns:
        cleaned_df[col] = cleaned_df[col].astype(str).str.strip()
        cleaning_log.append(f"{col}: Trimmed text")

    # ----------------------------------
    # 4️⃣ REMOVE DUPLICATES
    # ----------------------------------
    before = len(cleaned_df)
    cleaned_df.drop_duplicates(inplace=True)
    after = len(cleaned_df)

    if after < before:
        cleaning_log.append(f"Removed {before - after} duplicate rows")

    return cleaned_df

# ==========================================================
# IQR OUTLIER REMOVAL (SAFE)
# ==========================================================

def remove_outliers_iqr(df):
    """
    Removes outliers using IQR method (numeric columns only).
    """
    df = df.copy()

    for col in df.select_dtypes(include=np.number).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        if IQR == 0:
            continue

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df = df[(df[col] >= lower) & (df[col] <= upper)]

    df.reset_index(drop=True, inplace=True)
    return df