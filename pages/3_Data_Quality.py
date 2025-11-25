import streamlit as st
from config.registry import load_feature_inventory
from data.loader import load_history_sales

st.set_page_config(page_title="Data Quality", layout="wide")
st.title("Data Quality – Registry vs Dataset")

df_reg = load_feature_inventory()
known = set(df_reg["Feature Name"].dropna().tolist())
df_raw = load_history_sales()

cols = set(df_raw.columns)
undefined_in_registry = sorted(cols - known)
unused_in_dataset = sorted(known - cols)

st.write("Columns present in dataset but not documented in registry:")
st.code("\n".join(undefined_in_registry) or "None")

st.write("Features documented in registry but not present in dataset:")
st.code("\n".join(unused_in_dataset) or "None")
