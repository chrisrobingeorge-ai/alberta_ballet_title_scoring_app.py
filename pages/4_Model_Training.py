import streamlit as st
from ml.training import train_baseline_model

st.set_page_config(page_title="Model Training", layout="wide")
st.title("Model Training – Title Demand (Baseline)")

st.markdown("""
### What is this page?
The **Model Training** page allows you to train an ML model that predicts ticket demand.

**How it works:**
1. Loads historical sales data from `history_city_sales.csv`
2. Combines with baseline scores (familiarity, motivation) from `baselines.csv`
3. Trains a Random Forest regression model
4. Saves the model to `models/title_demand_rf.pkl`

**When to retrain:**
- After adding new historical sales data
- After modifying baseline scores
- When model performance degrades

The trained model is used on the **Title Scoring** page to generate forecasts.
""")

col1, col2 = st.columns([1, 2])
with col1:
    if st.button("🚀 Train Baseline Model", type="primary"):
        with st.spinner("Training model... This may take a moment."):
            result = train_baseline_model()
        st.success("✅ Model trained and saved successfully!")
        st.json(result)
        
with col2:
    st.info("""
    **Tip**: After training, go to the **Title Scoring** page 
    to score all your titles with the new model.
    """)
