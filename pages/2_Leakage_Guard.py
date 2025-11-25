import streamlit as st
from config.registry import load_leakage_audit

st.set_page_config(page_title="Leakage Guard", layout="wide")
st.title("Leakage Guard – Allowed Features at Forecast Time")

audit = load_leakage_audit()
st.dataframe(audit, use_container_width=True)

st.subheader("Allowed Features Only")
allowed = audit[audit["Allowed at Forecast Time (Y/N)"].str.upper() == "Y"]["Feature Name"].tolist()
st.write(f"Count: {len(allowed)}")
st.code("\n".join(allowed))
