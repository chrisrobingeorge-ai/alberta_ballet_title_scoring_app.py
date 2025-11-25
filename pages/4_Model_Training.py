import streamlit as st
from ml.training import train_baseline_model

st.set_page_config(page_title="Model Training", layout="wide")
st.title("Model Training – Title Demand (Baseline)")

if st.button("Train Baseline Model"):
    with st.spinner("Training model..."):
        result = train_baseline_model()
    st.success("Model trained & saved.")
    st.json(result)
