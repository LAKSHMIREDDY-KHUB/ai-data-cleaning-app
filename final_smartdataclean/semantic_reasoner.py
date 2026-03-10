# ==========================================================
# SMARTPREP AI - STRUCTURED SEMANTIC REASONER (TEXT + JSON)
# ==========================================================

import os
import json
from openai import OpenAI


# ----------------------------------------------------------
# Load API Key
# ----------------------------------------------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if OPENROUTER_API_KEY is None:
    raise ValueError("OPENROUTER_API_KEY not found. Please set environment variable.")


# ----------------------------------------------------------
# Initialize Client
# ----------------------------------------------------------
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)


# ----------------------------------------------------------
# AI Semantic Cleaning Advisor (TEXT + JSON)
# ----------------------------------------------------------
def semantic_cleaning_advisor(df):

    try:
        # ------------------------------
        # Dataset profiling
        # ------------------------------
        columns = list(df.columns)
        missing = df.isnull().sum().to_dict()
        dtypes = df.dtypes.astype(str).to_dict()
        shape = df.shape

        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        try:
            summary_stats = df.describe().round(2).to_dict()
        except:
            summary_stats = {}

        # ==================================================
        # 🧑‍🏫 TEXT PROMPT (Explainable AI)
        # ==================================================
        TEXT_PROMPT = f"""
You are a senior data scientist helping NON-TECHNICAL users.

Explain dataset cleaning recommendations in VERY SIMPLE language.

Dataset Information:
--------------------------------------------------
Shape: {shape}
Columns: {columns}
Data Types: {dtypes}
Missing Values: {missing}
Numeric Columns: {numeric_cols}
Categorical Columns: {categorical_cols}
Numeric Summary Stats: {summary_stats}
--------------------------------------------------

Provide response using this structure:

1️⃣ OVERALL DATASET HEALTH
2️⃣ MISSING VALUE HANDLING
3️⃣ NUMERIC DATA CLEANING
4️⃣ CATEGORICAL DATA CLEANING
5️⃣ STEP-BY-STEP CLEANING PLAN

Keep it beginner-friendly.
Avoid complex statistical language.
Do NOT return JSON.
"""

        # ==================================================
        # 🤖 JSON PROMPT (Executable AI)
        # ==================================================
        JSON_PROMPT = f"""
You are a senior data scientist.

Analyze the dataset and return ONLY valid JSON.
Do NOT write anything outside JSON.

Dataset Information:
--------------------------------------------------
Shape: {shape}
Columns: {columns}
Data Types: {dtypes}
Missing Values: {missing}
Numeric Columns: {numeric_cols}
Categorical Columns: {categorical_cols}
Numeric Summary Stats: {summary_stats}
--------------------------------------------------

Return JSON in this EXACT structure:

{{
  "missing_strategy": {{
    "column_name": "mean | median | mode | knn | drop"
  }},
  "outlier_strategy": "remove | cap | none",
  "remove_duplicates": true,
  "drop_columns": [],
  "confidence_score": 0.0,
  "explanation": "Simple explanation for user"
}}

Rules:
- Every column must appear inside missing_strategy.
- Keep explanation short.
- Return ONLY valid JSON.
"""

        # ==================================================
        # 🧑‍🏫 TEXT AI CALL
        # ==================================================
        text_response = client.chat.completions.create(
            model="meta-llama/llama-3-8b-instruct",
            messages=[{"role": "user", "content": TEXT_PROMPT}],
            temperature=0.4,
            max_tokens=900,
        )

        text_advice = text_response.choices[0].message.content.strip()

        # ==================================================
        # 🤖 JSON AI CALL
        # ==================================================
        json_response = client.chat.completions.create(
            model="meta-llama/llama-3-8b-instruct",
            messages=[{"role": "user", "content": JSON_PROMPT}],
            temperature=0.2,
            max_tokens=1200,
        )

        raw_json = json_response.choices[0].message.content.strip()
        raw_json = raw_json.replace("```json", "").replace("```", "")

        try:
            json_advice = json.loads(raw_json)
        except:
            json_advice = {
                "error": "AI did not return valid JSON",
                "raw_output": raw_json
            }

        # ==================================================
        # FINAL OUTPUT
        # ==================================================
        return {
            "text_advice": text_advice,
            "json_advice": json_advice
        }

    except Exception as e:
        return {"error": f"AI ERROR: {str(e)}"}
