# Model Validation Guide

## Overview

The Model Validation page allows you to compare your title scoring model's predictions against a PyCaret-trained machine learning model to assess performance and identify areas for improvement.

## Prerequisites

### 1. Install PyCaret (Optional)

PyCaret is only required if you want to use the Model Validation feature. It supports Python 3.9-3.12.

```bash
pip install pycaret>=3.3.0
```

### 2. Prepare Historical Data

You need historical data with:
- Actual outcomes (e.g., ticket sales numbers)
- Your model's predictions
- Features used for prediction

The data should be in CSV format (e.g., `data/history_city_sales.csv`).

## Creating a PyCaret Model

### Step 1: Prepare Your Data

```python
import pandas as pd
from pycaret.regression import setup, compare_models, save_model

# Load historical data
df = pd.read_csv('data/history_city_sales.csv')

# Example columns:
# - Title: Show title
# - Category: Show category
# - City: Calgary or Edmonton  
# - Total_Tickets: Actual tickets sold (target variable)
# - Familiarity: Your model's familiarity score
# - Motivation: Your model's motivation score
# - [other features...]
```

### Step 2: Set Up PyCaret

```python
# Initialize PyCaret with your target column
s = setup(
    data=df,
    target='Total_Tickets',  # Column with actual outcomes
    session_id=123,          # For reproducibility
    silent=True,             # Suppress verbose output
    verbose=False
)
```

### Step 3: Train and Select Best Model

```python
# PyCaret will automatically try multiple models
best_model = compare_models(
    n_select=1,              # Select top 1 model
    sort='MAE',              # Sort by Mean Absolute Error
    verbose=True
)

# View model performance
print(f"Best model: {type(best_model).__name__}")
```

### Step 4: Save the Model

```python
# Save with a descriptive name (no .pkl extension needed)
save_model(best_model, 'title_demand_model')

# This creates: title_demand_model.pkl
```

The model file will be saved in your current directory. Place it in the project root (same directory as `streamlit_app.py`).

## Using the Model Validation Page

### 1. Access the Feature

In the Streamlit app:
1. Navigate to the **Model Validation** page (sidebar)
2. The app will load your historical data automatically

### 2. Configure Comparison

- **Season**: Filter by season (optional)
- **City**: Filter by city (optional)
- **Actual outcome column**: Select the column with true values (e.g., `Total_Tickets`)
- **Your model prediction column**: Select your model's predictions
- **PyCaret saved model name**: Enter the name you used when saving (default: `title_demand_model`)

### 3. View Results

The page will display:
- **Performance Metrics**: MAE, RMSE, and R² for both models
- **Comparison Table**: Side-by-side predictions with error analysis
- **Better Model Indicator**: Shows which model performed better for each title

## Troubleshooting

### Error: "Could not find PyCaret model file"

**Cause**: The `.pkl` file doesn't exist or is in the wrong location.

**Solution**:
1. Ensure you've trained and saved a model (see "Creating a PyCaret Model" above)
2. Place the `.pkl` file in the project root directory
3. Enter the correct model name (without `.pkl` extension)

### Error: "PyCaret is required for this functionality"

**Cause**: PyCaret is not installed.

**Solution**:
```bash
pip install pycaret>=3.3.0
```

### Error: "Python Version Compatibility Notice"

**Cause**: Your Python version is outside PyCaret's supported range (3.9-3.12).

**Solution**:
- Use Python 3.12 or earlier
- Alternatively, skip the Model Validation feature (the main title scoring functionality doesn't require PyCaret)

### Error: Dependency conflicts

**Cause**: Version conflicts between PyCaret and other packages.

**Solution**:
1. Check `requirements.txt` for compatible versions
2. Install in a clean virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Best Practices

### 1. Data Quality
- Use a representative sample of historical data
- Include diverse title types and categories
- Ensure actual outcomes are accurately recorded

### 2. Model Training
- Use cross-validation for robust estimates
- Try multiple models (PyCaret's `compare_models()` does this automatically)
- Save models with descriptive names (e.g., `ballet_demand_2024_xgboost`)

### 3. Performance Evaluation
- Focus on MAE for interpretability (average error in tickets)
- Use R² to assess overall model fit (higher is better)
- Review individual predictions to identify patterns

### 4. Iteration
- Retrain models periodically as new data becomes available
- Compare different feature sets
- Document model versions and performance

## Example Workflow

```python
# Complete example: Train and save a PyCaret model

import pandas as pd
from pycaret.regression import setup, compare_models, save_model, pull

# 1. Load data
df = pd.read_csv('data/history_city_sales.csv')

# 2. Prepare target and features
# Note: This app focuses on single tickets only
df['Total_Tickets'] = df['Single Tickets - Calgary'] + df['Single Tickets - Edmonton']

# 3. Setup PyCaret
s = setup(
    data=df,
    target='Total_Tickets',
    session_id=42,
    normalize=True,           # Normalize features
    feature_selection=True,   # Select best features
    remove_multicollinearity=True,
    silent=True
)

# 4. Compare models
best_model = compare_models(n_select=1, sort='MAE')

# 5. View results
results = pull()
print("\nModel Performance:")
print(results)

# 6. Save model
save_model(best_model, 'title_demand_model')
print("\n✓ Model saved as 'title_demand_model.pkl'")
print("Place this file in the project root to use in the app.")
```

## Robust Training Pipeline (Recommended)

For production use, we recommend the new leak-free training pipeline instead of raw PyCaret:

### Quick Start

```bash
# Step 1: Build safe dataset (no current-run ticket columns as features)
python scripts/build_modelling_dataset.py

# Step 2: Train with XGBoost and cross-validation
python scripts/train_safe_model.py --tune

# Step 3: Run backtesting to validate
python scripts/backtest_timeaware.py

# Step 4 (Optional): Fit calibration
python scripts/calibrate_predictions.py fit --mode per_category
```

### Why Use the New Pipeline?

| Feature | PyCaret Script | New Pipeline |
|---------|---------------|--------------|
| Leakage Prevention | Manual | Automatic assertions |
| Cross-Validation | Basic | Time-aware / GroupKFold |
| Remount Features | None | years_since_last_run, is_remount |
| SHAP Explanations | No | Yes (--save-shap) |
| k-NN Fallback | No | Yes (for cold-start) |
| Calibration | No | Yes (global/per-category) |
| Reproducibility | Session seed | Full metadata saved |

### Leakage Prevention Checklist

Before training any model:

- [ ] Verify features are from `data/modelling_dataset.csv` (not raw history)
- [ ] Confirm no "Single Tickets" or "Total Tickets" columns
- [ ] Run `pytest tests/test_no_leakage_in_dataset.py` 
- [ ] Check that only prior-season aggregates are used (e.g., `prior_total_tickets`)

## Additional Resources

- [PyCaret Documentation](https://pycaret.gitbook.io/docs/)
- [PyCaret Regression Tutorial](https://pycaret.gitbook.io/docs/get-started/tutorials/regression)
- [Main App ML Documentation](ML_MODEL_DOCUMENTATION.md)
- [README - Robust Training Pipeline](README.md#robust-ml-training-pipeline-new)

## Support

For issues or questions:
1. Check this guide first
2. Review error messages in the app (they include helpful instructions)
3. Consult the PyCaret documentation
4. Check GitHub issues for similar problems
