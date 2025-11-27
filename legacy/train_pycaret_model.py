# legacy/train_pycaret_model.py
#
# ============================================================================
# ⚠️  DEPRECATED - DO NOT USE FOR PRODUCTION TRAINING  ⚠️
# ============================================================================
#
# This script has been moved to the legacy/ directory and is DEPRECATED.
# It is kept only for historical reference and should NOT be used for
# production model training.
#
# Use the safe training pipeline instead:
#   1. python scripts/build_modelling_dataset.py  # Creates safe feature set
#   2. python scripts/train_safe_model.py          # Trains with proper CV
#
# ============================================================================
# ⚠️  LEAKAGE WARNING - READ BEFORE USING  ⚠️
# ============================================================================
# This is a LEGACY QUICK-SCRIPT that may cause data leakage if used naively.
#
# PROBLEMS WITH THIS SCRIPT:
# 1. It uses current-run ticket columns directly as features AND target
# 2. This creates circular dependencies where the model learns to predict
#    from information that wouldn't be available at forecast time
# 3. The resulting model cannot generalize to new shows without history
#
# SAFE ALTERNATIVE:
# Use the new leak-free training pipeline instead:
#   1. python scripts/build_modelling_dataset.py  # Creates safe feature set
#   2. python scripts/train_safe_model.py          # Trains with proper CV
#
# If you must use this script, ensure you:
# 1. Generate features using scripts/build_modelling_dataset.py first
# 2. Pass that leak-free dataset to PyCaret instead of raw history
# 3. Verify no current-run ticket columns are in the feature set
# ============================================================================
#
# One-time script to train a PyCaret regression model from history_city_sales.csv
# and save it as "title_demand_model.pkl" for the Streamlit app.
# 
# Note: This app focuses on single ticket estimation only.

import pandas as pd
from pycaret.regression import setup, compare_models, save_model

# 1. Load your historical data
df = pd.read_csv("data/history_city_sales.csv", thousands=",")

# 2. Rename columns to remove spaces (PyCaret / LightGBM prefer this)
df.columns = [c.replace(" ", "_") for c in df.columns]

# Now your columns are:
# - Show_Title
# - Single_Tickets_-_Calgary
# - Single_Tickets_-_Edmonton

ticket_cols = [
    "Single_Tickets_-_Calgary",
    "Single_Tickets_-_Edmonton",
]

# 3. Make sure all ticket columns are numeric and treat blanks as 0
for col in ticket_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# 4. Build a Total_Tickets column as the target
df["Total_Tickets"] = df[ticket_cols].sum(axis=1)

print("Preview of training data:")
print(df[["Show_Title"] + ticket_cols + ["Total_Tickets"]].head())

# 5. Set up PyCaret
s = setup(
    data=df,
    target="Total_Tickets",
    session_id=42,
    normalize=True,
    feature_selection=True,
    remove_multicollinearity=True,
    ignore_features=["Show_Title"],  # don't use raw title text as a feature
)

# 6. Let PyCaret try different models and pick the best by MAE
best_model = compare_models(n_select=1, sort="MAE")

# 7. Save the best model
save_model(best_model, "title_demand_model")
print("\n✓ Model saved as 'title_demand_model.pkl' in this folder.")
print("   Add/commit this file next to streamlit_app.py so the app can load it.")


# ============================================================================
# SAFE USAGE EXAMPLE (Recommended)
# ============================================================================
# To train a leak-free model, use the following workflow instead:
#
# Step 1: Build a safe modelling dataset
# ```bash
# python scripts/build_modelling_dataset.py
# ```
#
# Step 2: Train with the safe training script
# ```bash
# python scripts/train_safe_model.py --tune --save-shap
# ```
#
# OR, if you must use PyCaret, load the leak-free dataset:
# ```python
# import pandas as pd
# from pycaret.regression import setup, compare_models, save_model
#
# # Load the leak-free dataset
# df = pd.read_csv("data/modelling_dataset.csv")
#
# # Use only forecast-time features (no current-run ticket columns!)
# feature_cols = ["wiki", "trends", "youtube", "spotify", "category", "gender",
#                 "prior_total_tickets", "years_since_last_run", "is_remount_recent"]
# target_col = "target_ticket_median"
#
# # Filter to rows with valid target
# df = df[df[target_col] > 0][feature_cols + [target_col]]
#
# # Setup PyCaret with safe features
# s = setup(data=df, target=target_col, session_id=42, silent=True)
# best = compare_models(n_select=1, sort="MAE")
# save_model(best, "models/title_demand_pycaret_safe")
# ```
# ============================================================================
