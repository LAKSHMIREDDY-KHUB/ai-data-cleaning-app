import streamlit as st
import pandas as pd

def missing_value_comparison(before_df, after_df):
    comparison = pd.DataFrame({
        "Before Cleaning": before_df.isnull().sum(),
        "After Cleaning": after_df.isnull().sum()
    })
    st.bar_chart(comparison)
