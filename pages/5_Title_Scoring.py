import streamlit as st
import pandas as pd
from data.loader import load_history_sales
from data.features import derive_basic_features, apply_registry_renames
from ml.scoring import load_model, score_dataframe
from pathlib import Path
import joblib
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Title Scoring", layout="wide")
st.title("Title Scoring – Forecast")

st.caption(
    "This page displays all historical titles and allows you to score them using "
    "the trained ML model. Train a model first using the **Model Training** page."
)

# --- Accuracy Interpretation Guide ---
with st.expander("📊 Understanding the Accuracy Metrics (R² and MAE)"):
    st.markdown("""
    ### What do the accuracy metrics mean?
    
    When you score historical titles, the page displays accuracy metrics comparing the model's
    predictions (`forecast_single_tickets`) to actual sales (`total_single_tickets`).
    
    **⚠️ Important: These metrics are "retrodiction" accuracy, NOT predictive accuracy.**
    
    #### Why the R² might appear very high (e.g., 0.985):
    
    1. **In-sample evaluation**: The model is being evaluated on the same data (or a subset) 
       it was trained on. This typically produces optimistically high accuracy metrics.
    
    2. **Retrodiction vs. Prediction**: This is "retrodiction" — re-estimating past shows using 
       the same formula used for new shows. It checks whether the model's logic aligns with 
       historical patterns, but does NOT guarantee similar accuracy on truly new, unseen titles.
    
    3. **Small dataset size**: With a limited number of historical records, even a moderately good model
       can appear to fit very well.
    
    #### How to interpret the metrics:
    
    | Metric | What It Means | Interpretation |
    |--------|--------------|----------------|
    | **R²** | How much variance in ticket sales the model explains | High R² (>0.9) on historical data suggests the model captures past patterns well, but may be overfitting |
    | **MAE** | Average prediction error in tickets | More interpretable — e.g., MAE of 500 means predictions are off by ~500 tickets on average |
    
    #### For true predictive accuracy:
    
    - Use the **Model Training** page which reports metrics on a held-out test set (20% of data)
    - Use **time-aware cross-validation** via `scripts/backtest_timeaware.py`
    - Compare model predictions to actual sales for genuinely NEW titles not in training data
    
    #### What does the training-time R² represent?
    
    During model training (on the Model Training page), the R² is calculated on a 20% held-out 
    test set. This is a more realistic estimate of how the model performs on unseen data, though
    still from the same historical period. The R² shown here when scoring may be higher because 
    it includes all data.
    """)
    st.info(
        "**Bottom line**: High R² on this page is a good calibration check, but should not be "
        "interpreted as a guarantee of future prediction accuracy. Use MAE for practical planning."
    )

# --- Column Definitions (for reference) ---
# The data includes several ticket-related columns:
#
# FROM HISTORICAL DATA (history_city_sales.csv):
#   - single_tickets_calgary/edmonton: Actual single tickets sold per city
#   - total_single_tickets: Actual total single tickets sold (Calgary + Edmonton)
#   - yourmodel_single_tickets_calgary/edmonton: Historical model predictions per city
#   - yourmodel_total_single_tickets: Historical model's predicted total single tickets
#
# ML MODEL OUTPUT (from score_dataframe):
#   - forecast_single_tickets: ML model's prediction for single ticket sales
#
# ACCURACY INTERPRETATION:
#   Compare yourmodel_total_single_tickets vs total_single_tickets to assess
#   historical model accuracy. Compare forecast_single_tickets vs total_single_tickets
#   for the ML model's predictions.

df = apply_registry_renames(load_history_sales())
df = derive_basic_features(df)

# Show count of all titles
st.info(f"📊 Showing **{len(df)}** titles from historical data")

# Display ALL rows, not just head(20)
st.dataframe(df, use_container_width=True, height=600)

# Check if model exists
model_path = Path(__file__).parent.parent / "models" / "title_demand_rf.pkl"
if not model_path.exists():
    st.warning("No trained model found. Please train a model first using the Model Training page.")
else:
    if st.button("Score displayed rows"):
        try:
            model = load_model()
            # Get the feature names from the trained model's preprocessor
            # The model is a Pipeline with 'pre' (ColumnTransformer) step
            preprocessor = model.named_steps.get('pre')
            if preprocessor is not None:
                # Extract feature names from the ColumnTransformer
                num_features = preprocessor.transformers_[0][2] if len(preprocessor.transformers_) > 0 else []
                cat_features = preprocessor.transformers_[1][2] if len(preprocessor.transformers_) > 1 else []
                all_features = list(num_features) + list(cat_features)
                # Filter to only columns that exist in the current dataframe
                available_features = [f for f in all_features if f in df.columns]
                if available_features:
                    df_to_score = df[available_features]
                else:
                    # Fallback to numeric/categorical columns
                    df_to_score = df.select_dtypes(include=["number", "object"])
            else:
                df_to_score = df.select_dtypes(include=["number", "object"])
            
            preds = score_dataframe(df_to_score, model)
            
            # Build results DataFrame with show titles and key context columns
            result_df = pd.DataFrame({
                "show_title": df["show_title"] if "show_title" in df.columns else df.index,
                "forecast_single_tickets": preds.values,
            })
            
            # Add actual tickets for comparison if available
            if "total_single_tickets" in df.columns:
                result_df["actual_total_single_tickets"] = df["total_single_tickets"].values
            if "yourmodel_total_single_tickets" in df.columns:
                result_df["yourmodel_total_single_tickets"] = df["yourmodel_total_single_tickets"].values
            if "total_tickets_all" in df.columns:
                result_df["total_tickets_all"] = df["total_tickets_all"].values
            
            st.subheader("Scoring Results")
            st.dataframe(result_df, use_container_width=True)
            
            # Show accuracy metrics if actual data is available
            if "total_single_tickets" in df.columns:
                actual = df["total_single_tickets"].dropna()
                predicted = preds.loc[actual.index]
                if len(actual) > 0:
                    mae = mean_absolute_error(actual, predicted)
                    r2 = r2_score(actual, predicted)
                    
                    # Display metrics with clear context about what they mean
                    st.subheader("Accuracy Metrics (Retrodiction)")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            label="Mean Absolute Error (MAE)", 
                            value=f"{mae:,.0f} tickets",
                            help="Average prediction error in tickets. Lower is better."
                        )
                    with col2:
                        st.metric(
                            label="R² Score (In-Sample)", 
                            value=f"{r2:.3f}",
                            help="Variance explained. This is in-sample (retrodiction) — see expander above for interpretation."
                        )
                    
                    # Add context about the R² calculation
                    st.caption(
                        f"📊 **ML Model (forecast_single_tickets):** Scored on {len(actual)} historical titles. "
                        f"These metrics compare predictions to actual sales for titles the model may have seen during training."
                    )
                    
                    # Add warning if R² is very high
                    if r2 > 0.9:
                        st.warning(
                            "⚠️ **High R² Note**: An R² above 0.9 on historical data often indicates the model "
                            "is evaluating on data it was trained on (in-sample). This is useful as a calibration "
                            "check, but true predictive accuracy on new titles may be lower. "
                            "See the 'Understanding the Accuracy Metrics' expander above for details."
                        )
                    
                    # Compare with yourmodel if available
                    if "yourmodel_total_single_tickets" in df.columns:
                        ym = df.loc[actual.index, "yourmodel_total_single_tickets"].dropna()
                        if len(ym) > 0:
                            ym_actual = actual.loc[ym.index]
                            ym_mae = mean_absolute_error(ym_actual, ym)
                            ym_r2 = r2_score(ym_actual, ym)
                            st.caption(
                                f"**Historical Model (yourmodel_total_single_tickets):** MAE = {ym_mae:,.0f} tickets, R² = {ym_r2:.3f}"
                            )
        except Exception as e:
            st.error(f"Error scoring: {e}")
