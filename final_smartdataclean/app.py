# ==========================================================
# SMARTPREP AI - ENHANCED STABLE VERSION
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import json
import matplotlib.pyplot as plt
import reportlab

# Core modules
from loader import load_file
from cleaner import auto_clean,remove_outliers_iqr
from ml_cleaning import ml_clean_pipeline
from visuals import missing_value_comparison
from health_score import calculate_health_score
from semantic_reasoner import semantic_cleaning_advisor
from ai_cleaning_engine import apply_ai_cleaning
from generate_pdf import generate_pdf_report
from structural_clean import structural_auto_clean


# AutoML
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, r2_score

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch


# ==========================================================
# STRUCTURAL AUTO CLEANING (COLUMN INTELLIGENCE)
# ==========================================================
if "ai_json" not in st.session_state:
    st.session_state.ai_json = None



import numpy as np
import pandas as pd

def profile_dataset(df):

    profile = {}
    column_profiles = {}

    total_rows = len(df)
    total_cols = len(df.columns)

    # Dataset-level metrics
    duplicate_percent = df.duplicated().sum() / total_rows * 100
    overall_missing_percent = df.isnull().sum().sum() / (total_rows * total_cols) * 100

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include="object").columns.tolist()

    profile["dataset_summary"] = {
        "total_rows": total_rows,
        "total_columns": total_cols,
        "duplicate_percent": round(duplicate_percent, 2),
        "overall_missing_percent": round(overall_missing_percent, 2),
        "numeric_columns": len(numeric_cols),
        "categorical_columns": len(categorical_cols)
    }

    # Column-level profiling
    for col in df.columns:

        col_data = df[col]
        missing_percent = col_data.isnull().mean() * 100
        unique_count = col_data.nunique()

        col_type = "categorical"

        if pd.api.types.is_numeric_dtype(col_data):
            col_type = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(col_data):
            col_type = "datetime"

        outlier_percent = 0
        skewness = 0

        if col_type == "numeric" and col_data.notnull().sum() > 0:

            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1

            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            outliers = col_data[(col_data < lower) | (col_data > upper)]
            outlier_percent = len(outliers) / total_rows * 100

            skewness = col_data.skew()

        high_cardinality = False
        if col_type == "categorical" and unique_count > 50:
            high_cardinality = True

        column_profiles[col] = {
            "type": col_type,
            "missing_percent": round(missing_percent, 2),
            "unique_values": unique_count,
            "outlier_percent": round(outlier_percent, 2),
            "skewness": round(skewness, 2),
            "high_cardinality": high_cardinality
        }

    profile["column_profiles"] = column_profiles

    return profile



import pandas as pd
import numpy as np

def calculate_column_health_scores(df):

    #remove duplicate columns safely
    df = df.loc[:, ~df.columns.duplicated()].copy()

    results = []
    total_rows = len(df)

    #Prevent division by zero
    if total_rows == 0:
        return pd.DataFrame()
    

    for col in df.columns:
        score = 100
        reasons = []
        col_data = df[col]

        # ----------------------------
        # 1️⃣ Missing values (max −40)
        # ----------------------------
        missing_count = int(col_data.isna().sum())
        missing_pct = float((missing_count / total_rows) * 100)

        if missing_pct > 0:
            penalty = min(missing_pct * 0.7, 40)  # softened penalty
            score -= penalty
            reasons.append(f"Missing {round(missing_pct,2)}%")

        # ----------------------------
        # 2️⃣ Constant column (very bad)
        # ----------------------------
        unique_non_null = col_data.nunique(dropna=True)

        if unique_non_null <= 1:
            score -= 40
            reasons.append("Constant column")

        # ----------------------------
        # 3️⃣ Low variance (NUMERIC ONLY)
        # ----------------------------
        if pd.api.types.is_numeric_dtype(col_data):

            unique_ratio = unique_non_null / total_rows

            if unique_ratio < 0.01 and unique_non_null > 1:
                score -= 15
                reasons.append("Low variance numeric")

        # ----------------------------
        # 4️⃣ High cardinality categorical
        # ----------------------------
        if col_data.dtype == "object":

            unique_count = col_data.nunique(dropna=True)

            if unique_count > 100:
                score -= 10
                reasons.append("High-cardinality categorical")

        # ----------------------------
        # 5️⃣ Outlier penalty (numeric)
        # ----------------------------
        if pd.api.types.is_numeric_dtype(col_data) and not pd.api.types.is_bool_dtype(col_data):

            numeric_series = pd.to_numeric(col_data, errors="coerce").dropna()

            if len(numeric_series) > 30:

                Q1 = numeric_series.quantile(0.25)
                Q3 = numeric_series.quantile(0.75)
                IQR = Q3 - Q1

                if IQR > 0:
                    outliers = (
                        (numeric_series < Q1 - 1.5 * IQR) |
                        (numeric_series > Q3 + 1.5 * IQR)
                    ).sum()

                    outlier_pct = (outliers / total_rows) * 100

                    if outlier_pct > 5:
                        score -= min(outlier_pct, 10)
                        reasons.append("Many outliers")

        # ----------------------------
        # Clamp score
        # ----------------------------
        score = max(round(score, 2), 0)

        results.append({
            "Column": col,
            "Missing Values": missing_count,
            "Missing %": round(missing_pct, 2),
            "Unique Values": unique_non_null,
            "Health Score": score,
            "Issues": ", ".join(reasons) if reasons else "Healthy"
        })

    return pd.DataFrame(results).sort_values("Health Score")


def go_back():
    if st.session_state.step > 1:
        st.session_state.step -= 1
        st.rerun()



# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(page_title="SMART DATA CLEANING", layout="wide")

st.title("🤖 SMART DATA CLEAN")
st.caption("AI-Powered End-to-End Data Preparation Assistant")


# ==========================================================
# SESSION STATE
# ==========================================================
if "step" not in st.session_state:
    st.session_state.step = 1

if "df" not in st.session_state:
    st.session_state.df = None

if "cleaned_df" not in st.session_state:
    st.session_state.cleaned_df = None

if "original_df" not in st.session_state:
    st.session_state.original_df = None

if "ai_text" not in st.session_state:
    st.session_state.ai_text = ""

if "comparison_results" not in st.session_state:
    st.session_state.comparison_results = None

if "best_strategy" not in st.session_state:
    st.session_state.best_strategy = None

# ==========================================================
# SIDEBAR WORKFLOW
# ==========================================================
st.sidebar.title("📌 Workflow")

steps = {
    1: "Upload Dataset",
    2: "EDA & Profiling",
    3: "Column Management",
    4: "AI Suggestions",
    5: "Cleaning Approaches",
    6: "Final Dashboard"
}

for num, name in steps.items():
    if st.session_state.step > num:
        st.sidebar.success(f"✅ {name}")
    elif st.session_state.step == num:
        st.sidebar.info(f"➡ {name}")
    else:
        st.sidebar.write(name)


# ==========================================================
# STEP 1 - UPLOAD
# ==========================================================
if st.session_state.step == 1:
    


    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"],key="dataset_uploader")

    if uploaded_file is not None:

        try:
            df = load_file(uploaded_file)

        
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            st.info("Please ensure the file is in CSV or Excel format and try again.")  
            st.stop()

        st.session_state.df = df
        st.session_state.cleaned_df = df.copy()  # Initialize cleaned_df with original
        st.session_state.original_df = df.copy()  # Store original for comparison  

        st.success("File/Dataset loaded successfully!") 
        st.session_state.step = 2
        st.rerun()

        profile = profile_dataset(st.session_state.df)

        st.subheader("🔎 Dataset Summary")
        st.write(profile["dataset_summary"])

        st.subheader("📊 Column Profiles")
        st.write(profile["column_profiles"])



def dataset_metadata_viewer(df):

    st.subheader("📌 Dataset Metadata")

    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", int(df.shape[0]))
    col2.metric("Columns", int(df.shape[1]))
    col3.metric(
        "Memory (MB)",
        round(df.memory_usage(deep=True).sum() / (1024 ** 2), 2)
    )

    # -------- COLUMN DATA TYPES --------
    st.markdown("### 🧬 Column Data Types")

    dtype_df = (
        df.dtypes
          .astype(str)                 # force string for Streamlit
          .reset_index()
    )

    dtype_df.columns = ["Column Name", "Data Type"]

    st.dataframe(dtype_df, use_container_width=True)
    
    

    # -------- ROW / COLUMN VIEWER --------
    with st.expander("🔍 Row & Column Viewer"):

        st.markdown("**Preview Rows**")
        start = st.number_input(
            "Start Row",
            min_value=0,
            max_value=max(len(df) - 1, 0),
            value=0
        )

        end = st.number_input(
            "End Row",
            min_value=start + 1,
            max_value=len(df),
            value=min(start + 10, len(df))
        )

        st.dataframe(df.iloc[start:end], use_container_width=True)

        st.markdown("**Preview Columns**")
        selected_cols = st.multiselect(
            "Select Columns",
            df.columns.tolist(),
            default=df.columns[: min(199, len(df))]
        )

        if selected_cols:
            st.dataframe(df[selected_cols].head(10), use_container_width=True)




# ==========================================================
# STEP 2 - EDA
# ==========================================================
if st.session_state.step == 2:
    st.button("⬅ Back", on_click=go_back)


    df = st.session_state.df
    dataset_metadata_viewer(df)

    st.subheader("📊 Dataset Overview")
    col1, col2 = st.columns(2)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])

    st.dataframe(df)
    st.subheader("📈 Summary Statistics")
    st.dataframe(df.describe(include="all"))

    st.write("Missing Values:", df.isnull().sum().sum())
    st.write("Duplicate Rows:", df.duplicated().sum())

    st.subheader("🧠 Dataset Health Score")
    health = calculate_health_score(df)
    st.metric("Dataset Health (Before Cleaning)", f"{health} / 100")

    if st.button("Next ➡ Column Management"):
        st.session_state.step = 3
        st.rerun()



# ==========================================================
# STEP 3 – COLUMN MANAGEMENT (NEW FEATURE)
# ==========================================================

if st.session_state.step == 3:
    st.button("⬅ Back", on_click=go_back)

    df = st.session_state.df

    st.subheader("🧠 Column Health Analysis")
    health_df = calculate_column_health_scores(df).sort_values("Health Score")
    st.dataframe(health_df)

    bad_columns = health_df[health_df["Health Score"] < 40]["Column"].tolist()

    if bad_columns:
        st.warning("⚠ Low Quality Columns (Score < 40)")
        st.write(bad_columns)

        if st.checkbox("🗑 Auto-delete low quality columns"):
            df.drop(columns=bad_columns, inplace=True)
            st.success("Low quality columns removed.")

    st.subheader("✏ Rename Column")
    col_to_rename = st.selectbox("Select column", df.columns)
    new_name = st.text_input("New name")

    if st.button("Rename") and new_name:
        df.rename(columns={col_to_rename: new_name}, inplace=True)

    st.subheader("🗑 Delete Columns Manually")
    cols_to_delete = st.multiselect("Select columns", df.columns)
    if st.button("Delete Selected"):
        df.drop(columns=cols_to_delete, inplace=True)

    if st.button("Next ➡ Cleaning"):
        st.session_state.step = 4
        st.rerun()


# ==========================================================
# STEP 4 - AI + STRUCTURAL INTELLIGENCE
# ==========================================================
if st.session_state.step == 4:

    st.button("⬅ Back", on_click=go_back)

    df = st.session_state.df

    # ------------------------------------------------------
    # STRUCTURAL RULE-BASED CLEANING
    # ------------------------------------------------------

    st.subheader("🧱 Structural Column Intelligence")

    # Run structural cleaning WITHOUT dropping IDs
    struct_df, auto_removed, suggested_ids = structural_auto_clean(
        df,
        drop_ids=False
    )

    # Show automatically removed structural columns (non-ID)
    if auto_removed:
        st.success("Automatically Removed Columns:")
        for col in auto_removed:
            st.write(f"• {col}")
    else:
        st.info("No structural issues detected.")

    # ------------------------------------------------------
    # ID COLUMN SUGGESTION (MANUAL APPROVAL ONLY)
    # ------------------------------------------------------

    if suggested_ids:

        st.warning("⚠ Potential ID Columns Detected")

        selected_ids = st.multiselect(
            "Select ID columns to remove:",
            suggested_ids
        )

        if st.button("Remove Selected ID Columns"):

            if selected_ids:
                struct_df = struct_df.drop(columns=selected_ids, errors="ignore")
                st.success("Selected ID columns removed.")
            else:
                st.info("No columns selected.")

    # Save cleaned dataframe after all logic
    st.session_state.cleaned_df = struct_df

    st.write("Preview After Structural Cleaning")
    st.dataframe(struct_df)

    # ------------------------------------------------------
    # AI DATASET UNDERSTANDING
    # ------------------------------------------------------
    st.subheader("🧠 AI Dataset Understanding")

    if st.button("Generate AI Recommendations"):

        with st.spinner("Analyzing dataset with AI..."):

            # ✅ SINGLE AI CALL (IMPORTANT)
            advice = semantic_cleaning_advisor(struct_df)

            # Store safely
            st.session_state.ai_text = advice.get("text_advice", "")
            st.session_state.ai_json = advice.get("json_advice", None)

    # ------------------------------------------------------
    # DISPLAY AI OUTPUTS
    # ------------------------------------------------------
    if st.session_state.ai_text:
        st.subheader("🧑‍🏫 AI Cleaning Explanation")
        st.markdown(st.session_state.ai_text)

    if st.session_state.ai_json:
        st.subheader("🤖 AI Cleaning Instructions (Used by System)")
        st.json(st.session_state.ai_json)

    # ------------------------------------------------------
    # NAVIGATION
    # ------------------------------------------------------
    if st.button("Next ➡ Cleaning"):
        st.session_state.step = 5
        st.rerun()


# ==========================================================
# AUTO-COMPARE TARGET RECOMMENDER
# ==========================================================
def recommend_target_column(df):
    scores = {}
    total_rows = len(df)

    for col in df.columns:
        score = 0
        series = df[col]

        if series.isna().mean() > 0.5:
            continue

        if pd.api.types.is_numeric_dtype(series):
            score += 40
        elif series.nunique() <= 10:
            score += 25

        missing_pct = series.isna().mean() * 100
        score -= min(missing_pct, 30)

        unique_ratio = series.nunique() / total_rows
        if unique_ratio > 0.9:
            score -= 40

        if series.dtype == "object" and series.nunique() > 50:
            score -= 30

        scores[col] = round(score, 2)

    if not scores:
        return None, {}

    return max(scores, key=scores.get), scores

# ==========================================================
# STRATEGY EVALUATION ENGINE
# ==========================================================
def evaluate_strategy(df, target_column):

    try:
        X = df.drop(columns=[target_column])
        y = df[target_column]

        X = pd.get_dummies(X, drop_first=True)

        if len(X) == 0:
            return 0

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        if y.nunique() <= 10:
            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            return accuracy_score(y_test, preds)

        else:
            model = RandomForestRegressor()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            return r2_score(y_test, preds)

    except:
        return 0



# ==========================================================
# STEP 5 - CLEANING
# ==========================================================

if st.session_state.step == 5:
    steps = {
        1: "Upload",
        2: "Profile",
        3: "Health Score",
        4: "AI Suggestions",
        5: "Cleaning",
        6: "Results"
    }

    st.markdown(
        " ➜ ".join(
            [f"**{steps[i]}**" if i == st.session_state.step else steps[i]
            for i in steps]
        )
    )
    st.markdown("---")
    st.button("⬅ Back", on_click=go_back)


    df = st.session_state.cleaned_df

    st.header("🧹 Cleaning Strategy Selection")

    # ======================================================
    # SMART CLEANING TYPE RECOMMENDER (LABELS)
    # ======================================================
    rows = df.shape[0]
    cols = df.shape[1]

    if rows < 300:
        recommended = "Rule-Based"
        reason = "Small dataset – fast, deterministic rules work best."
    elif rows < 5000:
        recommended = "ML-Based"
        reason = "Medium dataset – ML learns missing patterns effectively."
    else:
        recommended = "AI-Based"
        reason = "Large or complex dataset – AI handles semantic cleaning better."

    st.info(
        f"🤖 **System Recommendation:** `{recommended}`\n\n"
        f"📊 Dataset: {rows} rows × {cols} columns\n\n"
        f"🧠 Reason: {reason}"

    )
    st.success(f"⭐ Recommended Cleaning Strategy: **{recommended}**")

    
    method = st.radio(
        "Select Cleaning Strategy",
        ["Rule-Based", "ML-Based", "AI-Based", "Auto-Compare"],
        help="""
    Rule-Based: Fast, rule-driven cleaning (mean/median/mode)
    ML-Based: Predicts missing values using machine learning
    AI-Based: Uses LLM reasoning + dataset semantics
    Auto-Compare: Automatically selects best strategy
    """
    )



    # Select target column if Auto-Compare
    if method == "Auto-Compare":

        recommended_target, target_scores = recommend_target_column(df)

        st.subheader("🎯 Target Column Selection")

        if recommended_target:
            st.success(f"✅ Recommended Target Column: **{recommended_target}**")

        with st.expander("📊 Target Suitability Scores"):
            st.dataframe(
                pd.DataFrame.from_dict(
                    target_scores, orient="index", columns=["Score"]
                ).sort_values("Score", ascending=False)
            )

        st.session_state.target_column = st.selectbox(
            "Select Target Column (You can override)",
            df.columns,
            index=df.columns.get_loc(recommended_target)
            if recommended_target in df.columns else 0
        )
   

    # ======================================================
    # APPLY CLEANING BUTTON
    # ======================================================

    if st.button("Apply Cleaning"):

        cleaned_df = None

        # ----------------------------
        # RULE BASED
        # ----------------------------
        if method == "Rule-Based":
            with st.spinner("🧹 Applying Rule-Based Cleaning..."):
                cleaned_df = auto_clean(df, "median")

        # ----------------------------
        # ML BASED
        # ----------------------------
        elif method == "ML-Based":
            with st.spinner("🧠 Applying ML-Based Cleaning..."):
                cleaned_df = ml_clean_pipeline(df)

        # ----------------------------
        # AI BASED
        # ----------------------------
        elif method == "AI-Based":

            if st.session_state.ai_json is None:
                st.error("⚠ Please generate AI recommendations first.")
                st.stop()

            with st.spinner("🤖 Applying AI-Based Cleaning..."):
                cleaned_df, summary = apply_ai_cleaning(
                    df,
                    st.session_state.ai_json
                )

            st.subheader("🧠 AI Cleaning Summary")
            for item in summary:
                st.write("•", item)
            with st.expander("🤖 Why did the AI choose these actions?"):
                st.write(st.session_state.ai_json.get("explanation", ""))    

        # ----------------------------
        # AUTO COMPARE
        # ----------------------------
        elif method == "Auto-Compare":

            if "target_column" not in st.session_state:
                st.error("⚠ Please select target column.")
                st.stop()

            from sklearn.preprocessing import LabelEncoder
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score, r2_score
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

            def evaluate_strategy(df, target_column):

                try:
                    X = df.drop(columns=[target_column])
                    y = df[target_column]

                    X = pd.get_dummies(X, drop_first=True)

                    if len(X) == 0:
                        return 0

                    # Safe target handling
                    if y.dtype == "object":
                        le = LabelEncoder()
                        y = le.fit_transform(y.astype(str))
                        model = RandomForestClassifier()
                    else:
                        if y.nunique() <= 10:
                            model = RandomForestClassifier()
                        else:
                            model = RandomForestRegressor()

                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )

                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)

                    if isinstance(model, RandomForestClassifier):
                        return accuracy_score(y_test, preds)
                    else:
                        return r2_score(y_test, preds)

                except:
                    return 0

            target = st.session_state.target_column
            results = {}
            # Cache ML result
            if "ml_df" not in st.session_state:
                with st.spinner("Training ML cleaning model..."):
                    st.session_state.ml_df = ml_clean_pipeline(df)

            ml_df = st.session_state.ml_df

            # Cache AI result
            if "ai_df" not in st.session_state and st.session_state.ai_json:
                with st.spinner("Applying AI cleaning..."):
                    st.session_state.ai_df, _ = apply_ai_cleaning(
                        df, st.session_state.ai_json
                )

            ai_df = st.session_state.get("ai_df")


            # Rule-Based
            rb_df = auto_clean(df, "median")
            results["Rule-Based"] = evaluate_strategy(rb_df, target)

            # ML-Based
            results["ML-Based"] = evaluate_strategy(ml_df, target)

            # AI-Based
            results["AI-Based"] = evaluate_strategy(ai_df, target) if ai_df is not None else 0

            best_strategy = max(results, key=results.get)

            st.session_state.comparison_results = {
                "scores": results,
                "best": best_strategy,
                "datasets": {
                    "Rule-Based": rb_df,
                    "ML-Based": ml_df,
                    "AI-Based": ai_df
                }
            }

            st.session_state.best_strategy = best_strategy

        # ----------------------------
        # NORMAL CLEANING FLOW
        # ----------------------------
        if cleaned_df is not None:

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 📉 Before Cleaning")
                st.dataframe(df.head(20))

            with col2:
                st.markdown("### 📈 After Cleaning")
                st.dataframe(cleaned_df.head(20))
                st.session_state.step = 6

            before_missing = df.isna().sum().sum()
            after_missing = cleaned_df.isna().sum().sum()

            before_dupes = df.duplicated().sum()
            after_dupes = cleaned_df.duplicated().sum()

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Missing Values",
                before_missing,
                before_missing - after_missing
            )

            col2.metric(
                "Duplicates Removed",
                before_dupes,
                before_dupes - after_dupes
            )

            col3.metric(
                "Rows After Cleaning",
                len(cleaned_df)
            )
            # NOW move to next step
            st.session_state.cleaned_df = cleaned_df
            st.session_state.step = 6
            st.rerun()
            

    # ======================================================
    # SHOW AUTO-COMPARE RESULTS
    # ======================================================

    if st.session_state.get("comparison_results"):

        st.header("📊 Strategy Comparison Results")

        scores = st.session_state.comparison_results["scores"]
        best_strategy = st.session_state.best_strategy
        best_score = scores[best_strategy]

        # Performance cards
        col1, col2 = st.columns(2)
        col1.metric("🏆 Best Strategy", best_strategy)
        col2.metric("📈 Best Score", round(best_score, 4))

        # Bar chart
        fig, ax = plt.subplots()
        ax.bar(scores.keys(), scores.values())
        ax.set_ylabel("Model Score")
        ax.set_title("Cleaning Strategy Performance Comparison")
        st.pyplot(fig)

        # ==================================================
        # FEATURE IMPORTANCE
        # ==================================================

        from sklearn.preprocessing import LabelEncoder
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        st.subheader("🔍 Feature Importance (Best Strategy)")

        feature_importance_dict = {}

        best_df = st.session_state.comparison_results["datasets"][best_strategy]
        target = st.session_state.target_column

        try:
            X = best_df.drop(columns=[target])
            y = best_df[target]

            X = pd.get_dummies(X, drop_first=True)

            if len(X.columns) == 0:
                raise ValueError("No usable features found for importance analysis.")   

            if y.dtype == "object":
                le = LabelEncoder()
                y = le.fit_transform(y.astype(str))
                model = RandomForestClassifier()
            else:
                if y.nunique() <= 10:
                    model = RandomForestClassifier()
                else:
                    model = RandomForestRegressor()

            model.fit(X, y)

            importance_df = pd.DataFrame({
                "Feature": X.columns,
                "Importance": model.feature_importances_
            }).sort_values(by="Importance", ascending=False).head(10)

            fig2, ax2 = plt.subplots()
            ax2.barh(importance_df["Feature"], importance_df["Importance"])
            ax2.invert_yaxis()
            ax2.set_title("Top 10 Important Features")

            st.pyplot(fig2)

            feature_importance_dict = dict(
                zip(X.columns, model.feature_importances_)
            )

        except Exception as e:
            st.warning("Feature importance could not be computed.")
            feature_importance_dict = {}
        


        # Score breakdown
        for k, v in scores.items():
            st.write(f"**{k}** Score: {round(v, 4)}")

        st.success(f"🏆 Best Strategy: {best_strategy}")

        # Proceed button
        if st.button("🚀 Proceed With Best Strategy"):

            best_df = st.session_state.comparison_results["datasets"][best_strategy]
            st.session_state.cleaned_df = best_df
            st.session_state.step = 6
            st.rerun()

# ==========================================================
# STEP 6 - FINAL DASHBOARD
# ==========================================================
if st.session_state.step == 6:

    st.button("⬅ Back", on_click=go_back)

    # Safe check
    if "original_df" not in st.session_state or st.session_state.original_df is None:
        st.warning("Original dataset not found. Please upload a file.")
        st.stop()

    df = st.session_state.original_df
    cleaned_df = st.session_state.cleaned_df

    st.subheader("📊 Missing Value Comparison")

    col1, col2 = st.columns(2)
    col1.metric("Before", df.isnull().sum().sum())
    col2.metric("After", cleaned_df.isnull().sum().sum())

    # ------------------------------------------------------
    # HEALTH SCORE
    # ------------------------------------------------------
    st.subheader("🧠 Health Score Comparison")

    before_score = calculate_health_score(df)
    after_score = calculate_health_score(cleaned_df)

    col1, col2 = st.columns(2)
    col1.metric("Before Cleaning", before_score)
    col2.metric(
        "After Cleaning",
        after_score,
        delta=round(after_score - before_score, 2)
    )

    missing_value_comparison(df, cleaned_df)

    # ------------------------------------------------------
    # BEFORE vs AFTER DATASET VIEW
    # ------------------------------------------------------

    st.subheader("📋 Dataset Comparison (Before vs After)")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔴 Before Cleaning")
        st.dataframe(df.head(10), use_container_width=True)

    with col2:
        st.markdown("### 🟢 After Cleaning")
        st.dataframe(cleaned_df.head(10), use_container_width=True)

    # DATASET DIMENSION COMPARISON
    st.subheader("📏 Dataset Dimensions")
    col1, col2 = st.columns(2)
    col1.metric("Rows Before", df.shape[0])
    col2.metric("Rows After", cleaned_df.shape[0])

    col1, col2 = st.columns(2)
    col1.metric("Columns Before", df.shape[1])
    col2.metric("Columns After", cleaned_df.shape[1])

    # ------------------------------------------------------
    # DOWNLOAD CLEANED CSV
    # ------------------------------------------------------
    st.subheader("⬇ Download Cleaned Data")

    csv = cleaned_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇ Download Cleaned Dataset",
        csv,
        "cleaned_data.csv",
        "text/csv"
    )


    # ------------------------------------------------------
    # PDF REPORT DOWNLOAD
    # ------------------------------------------------------

    ai_text = st.session_state.get(
        "ai_text",
        "No AI recommendations available."
    )

    comparison_results = st.session_state.get("comparison_results") or {}
    strategy_scores = comparison_results.get("scores", {})

    best_strategy = st.session_state.get(
        "best_strategy",
        "Not Available"
    )

    feature_importance_dict = st.session_state.get(
        "feature_importance_dict",
        {}
    )

    cleaning_summary = (
        "Structural cleaning, missing value handling, "
        "and AI-based improvements were applied."
    )

    pdf_path = generate_pdf_report(
        df,
        cleaned_df,
        before_score,
        after_score,
        strategy_scores,
        best_strategy,
        feature_importance_dict,
        cleaning_summary,
        ai_text
    )

    with open(pdf_path, "rb") as f:
        st.download_button(
            label="📄 Download PDF Report",
            data=f,
            file_name="SmartDataClean_Report.pdf",
            mime="application/pdf"
        )

    # ------------------------------------------------------
    # RESTART
    # ------------------------------------------------------
    if st.button("🔄 Restart"):
        st.session_state.clear()
        st.session_state.step = 1
        st.session_state.df = None
        st.session_state.cleaned_df = None
        st.session_state.original_df = None
        st.session_state.ai_text = ""
        st.session_state.comparison_results = None
        st.session_state.best_strategy = None   
        st.session_state.pop("ml_df", None)
        st.session_state.pop("ai_df", None)
        st.session_state.pop("comparison_results", None)
        st.rerun()
