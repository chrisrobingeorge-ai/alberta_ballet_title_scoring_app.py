import streamlit as st
from config.registry import load_feature_inventory

st.set_page_config(page_title="Feature Registry", layout="wide")
st.title("Feature Registry – Alberta Ballet Demand Model")

df = load_feature_inventory()
theme = st.selectbox("Filter by Theme", ["All"] + sorted(df["Theme"].dropna().unique()))
if theme != "All":
    df = df[df["Theme"] == theme]
st.dataframe(df, use_container_width=True)
