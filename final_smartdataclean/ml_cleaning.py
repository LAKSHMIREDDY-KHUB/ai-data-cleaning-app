from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import pandas as pd
import numpy as np


def ml_clean_pipeline(df):
    """
    ML-based missing value prediction
    - No row deletion
    - No outlier removal
    - Safe dtype handling
    - Production stable
    """

    if df is None or df.empty:
        return df

    cleaned_df = df.copy()
    encoders = {}

    # --------------------------------------------------
    # 1️⃣ Encode categorical columns safely
    # --------------------------------------------------
    for col in cleaned_df.columns:
        if cleaned_df[col].dtype == "object":
            le = LabelEncoder()
            cleaned_df[col] = le.fit_transform(
                cleaned_df[col].astype(str)
            )
            encoders[col] = le

    # --------------------------------------------------
    # 2️⃣ Predict missing values column-by-column
    # --------------------------------------------------
    for target_col in cleaned_df.columns:

        if cleaned_df[target_col].isna().sum() == 0:
            continue

        train_df = cleaned_df[cleaned_df[target_col].notna()]
        test_df = cleaned_df[cleaned_df[target_col].isna()]

        if train_df.empty or test_df.empty:
            continue

        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        X_test = test_df.drop(columns=[target_col])

        # Choose model
        if y_train.nunique() <= 10:
            model = RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
        else:
            model = RandomForestRegressor(
                n_estimators=100,
                random_state=42
            )

        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            cleaned_df.loc[test_df.index, target_col] = preds
        except:
            continue

    # --------------------------------------------------
    # 3️⃣ Decode categorical columns SAFELY
    # --------------------------------------------------
    for col, le in encoders.items():

        # Ensure numeric before rounding
        cleaned_df[col] = pd.to_numeric(
            cleaned_df[col],
            errors="coerce"
        )

        # Replace any remaining NaN
        if cleaned_df[col].isna().sum() > 0:
            cleaned_df[col].fillna(
                cleaned_df[col].mode()[0],
                inplace=True
            )

        # Clip to valid label range
        cleaned_df[col] = cleaned_df[col].round().astype(int)
        cleaned_df[col] = cleaned_df[col].clip(
            lower=0,
            upper=len(le.classes_) - 1
        )

        cleaned_df[col] = le.inverse_transform(cleaned_df[col])

    return cleaned_df