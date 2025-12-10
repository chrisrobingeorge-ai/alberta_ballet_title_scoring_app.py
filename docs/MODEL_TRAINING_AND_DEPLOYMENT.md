# Model Training and Deployment Guide

This guide explains how to train and deploy the ML model used by the Title Scoring Helper and main application.

## Quick Start

If you're getting a "No trained model file found" error, follow these steps:

```bash
# 1. Build the modelling dataset
python scripts/build_modelling_dataset.py

# 2. Train the model
python scripts/train_safe_model.py

# 3. Verify the model works
python -c "from ml.scoring import load_model; load_model('model_xgb_remount_postcovid'); print('✓ Model loaded successfully')"
```

## Understanding the Model

The trained model (`model_xgb_remount_postcovid.joblib`) is an XGBoost model wrapped in a scikit-learn Pipeline. This architecture provides:
- **Preprocessing automation**: Feature imputation, scaling, and one-hot encoding are bundled with the model
- **Feature consistency**: Ensures the same transformations are applied at training and inference time
- **Simplified deployment**: Single `.joblib` file contains both preprocessing and model
- **XGBoost regressor**: For robust ticket demand prediction

### Required Features

The model expects 35 features divided into:

**Baseline Signals (4):**
- `wiki`, `trends`, `youtube`, `spotify` - External demand signals

**Historical Features (7):**
- `prior_total_tickets`, `prior_run_count`, `ticket_median_prior`
- `years_since_last_run`, `is_remount_recent`, `is_remount_medium`, `run_count_prior`

**Date Features (14):**
- `month_of_opening`, `holiday_flag`, `opening_year`, `opening_month`, `opening_day_of_week`
- `opening_week_of_year`, `opening_quarter`, `opening_season`, `opening_is_winter`
- `opening_is_spring`, `opening_is_summer`, `opening_is_autumn`, `opening_is_holiday_season`
- `opening_is_weekend`, `run_duration_days`, `opening_date`

**Economic Features (6):**
- `consumer_confidence_prairies`, `energy_index`, `inflation_adjustment_factor`
- `city_median_household_income`, `aud__engagement_factor`, `res__arts_share_giving`

**Categorical Features (4):**
- `category`, `gender` - Production attributes
- `opening_season`, `opening_date` - Temporal categorical encodings (season name and date string)

### Feature Importance

Top features by importance (from trained model):
1. `prior_total_tickets` (72.8%) - Historical ticket sales
2. `prior_run_count` (18.8%) - Number of previous runs
3. `ticket_median_prior` (6.8%) - Median historical tickets
4. `spotify` (0.9%) - Spotify popularity
5. `youtube` (0.4%) - YouTube views

**Note:** When using `title_scoring_helper.py` with minimal features (only baseline signals), the model will default missing features to 0, resulting in conservative predictions based primarily on external signals rather than historical data.

## Model File Management

### Development
- Model files (`.joblib`, `.pkl`) are excluded from git via `.gitignore` by default
- This is standard practice as models can be large and are regenerated
- Developers should run the training script locally to generate the model

### Production/Deployment
- The primary production model (`model_xgb_remount_postcovid.joblib`) is included in the repository
- This exception is made in `.gitignore` to ensure the model is available on Streamlit Cloud and other deployment platforms
- The model file is small (111KB) and suitable for version control

### Updating the Model

When you need to update the production model:

```bash
# 1. Rebuild the dataset with latest data
python scripts/build_modelling_dataset.py

# 2. Retrain the model
python scripts/train_safe_model.py

# 3. Test the model
python -c "from ml.scoring import score_runs_for_planning; import pandas as pd; score_runs_for_planning(pd.DataFrame([{'wiki': 50, 'trends': 50, 'youtube': 50, 'spotify': 50, 'genre': 'classical', 'season': '2024-25'}]))"

# 4. Commit and push (model file is now tracked)
git add models/model_xgb_remount_postcovid.joblib
git commit -m "Update trained model with latest data"
git push
```

## Feature Handling in Production

The `ml/scoring.py` module includes automatic feature handling:

1. **Feature Extraction**: When the recipe file is missing, features are extracted directly from the trained model's `ColumnTransformer`
2. **Missing Features**: Any features not provided in the input DataFrame are automatically added with appropriate default values:
   - Binary features (`is_*`, `*_flag`): Default to 0
   - Categorical features (`category`, `gender`, `opening_season`, `opening_date`): Default to "missing"
   - Numeric features: Default to 0.0
3. **Schema Validation**: The system validates input features against the expected schema and warns about mismatches

This allows `title_scoring_helper.py` to work with minimal features (just baseline signals) while the model can still make predictions, albeit with higher uncertainty when historical data is missing.

## Troubleshooting

### "No trained model file found" Error

**Cause:** The model file doesn't exist in your environment.

**Solution:**
```bash
python scripts/build_modelling_dataset.py
python scripts/train_safe_model.py
```

### "X has 5 features, but ColumnTransformer is expecting 35 features" Error

**Cause:** Old version of `ml/scoring.py` that doesn't properly handle missing features.

**Solution:** Update to the latest version of `ml/scoring.py` which includes automatic feature extraction and default value handling.

### Low Prediction Values

**Cause:** When using `title_scoring_helper.py` with minimal features, the model has limited information and defaults to conservative predictions.

**Solution:** This is expected behavior. The model indicates uncertainty when historical data is missing. For better predictions, use the main `streamlit_app.py` with full historical data.

## See Also

- `scripts/train_safe_model.py` - Main training script
- `scripts/build_modelling_dataset.py` - Dataset preparation
- `ML_MODEL_DOCUMENTATION.md` - Detailed model documentation
- `TITLE_SCORING_HELPER_USAGE.md` - Title scoring helper usage guide
