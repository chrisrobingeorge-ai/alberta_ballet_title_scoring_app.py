import streamlit as st
from config.registry import load_feature_inventory

st.set_page_config(page_title="Feature Registry", layout="wide")
st.title("Feature Registry – Alberta Ballet Demand Model")

st.markdown("""
### What is this page?
The **Feature Registry** is a comprehensive catalog of all potential features (data fields) 
that could be used in the demand forecasting model. It serves as:

- 📋 **Documentation**: Lists every possible input variable with descriptions
- 🏷️ **Status Tracking**: Shows which features are `available`, `to be collected`, or `in development`
- 📦 **Data Source Reference**: Notes where each feature comes from

### About External Factors
Features marked `to be collected` or `in development` are **optional enhancements**. 
The current model works with available features from `history_city_sales.csv` and `baselines.csv`.

To add External Factors data:
1. Create a CSV file (e.g., `data/external_factors.csv`) with columns matching the Feature Names
2. Include a date/season column to join with historical sales
3. Update `data/loader.py` to merge this data with the main dataset
""")

df = load_feature_inventory()
theme = st.selectbox("Filter by Theme", ["All"] + sorted(df["Theme"].dropna().unique()))
if theme != "All":
    df = df[df["Theme"] == theme]

# Show status summary
status_counts = df["Status"].value_counts()
st.caption(f"**Status Summary**: {dict(status_counts)}")

st.dataframe(df, width='stretch', height=600)
