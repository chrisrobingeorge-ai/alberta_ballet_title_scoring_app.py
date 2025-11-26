# 🎭 Alberta Ballet Title Scoring App

A Streamlit application for predicting ticket sales and planning ballet seasons using advanced machine learning models.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://blank-app-template.streamlit.app/)

## Features

- **Advanced ML Predictions**: Uses XGBoost and scikit-learn for accurate ticket projections
- **Multi-Factor Scoring**: Combines Wikipedia, Google Trends, YouTube, and Spotify metrics
- **Seasonality Adjustment**: Accounts for performance timing and repeat effects
- **City-Specific Analysis**: Separate projections for Calgary and Edmonton
- **Revenue Forecasting**: Estimates revenue by singles vs subscriptions
- **Marketing Budget Planning**: Recommends spend based on historical performance
- **Season Builder**: Interactive tool to plan full seasons with financial summaries
- **ML Feature Registry**: Config-driven feature inventory with leakage guardrails
- **Data Quality Dashboard**: Registry vs dataset validation
- **Robust Training Pipeline**: Leak-free training with backtesting and SHAP explanations
- **k-NN Cold-Start Fallback**: Similarity-based predictions for new titles
- **Prediction Calibration**: Adjustable calibration for model predictions

## How to Run

### 1. Install the requirements

```bash
pip install -r requirements.txt
```

### 2. Run the app

```bash
streamlit run streamlit_app.py
```

## Robust ML Training Pipeline (New)

The app includes a complete, leak-free training workflow:

### Step 1: Build Modelling Dataset

Creates a safe feature set with only forecast-time predictors:

```bash
python scripts/build_modelling_dataset.py
```

This produces:
- `data/modelling_dataset.csv` - Leak-free training data
- `diagnostics/modelling_dataset_report.json` - Data quality report

### Step 2: Train Model

Trains XGBoost with time-aware cross-validation:

```bash
python scripts/train_safe_model.py --tune --save-shap
```

Outputs:
- `models/model_xgb_remount_postcovid.joblib` - Trained pipeline
- `models/model_xgb_remount_postcovid.json` - Metadata (CV metrics, features)
- `results/feature_importances.csv` - Feature importance scores
- `results/shap/` - SHAP analysis outputs (if --save-shap)

### Step 3: Run Backtesting

Evaluate different prediction methods:

```bash
python scripts/backtest_timeaware.py
```

Outputs:
- `results/backtest_summary.json` - Method comparison (MAE, RMSE, R²)
- `results/backtest_comparison.csv` - Row-level predictions
- `results/plots/mae_by_method.png` - Visual comparison

### Step 4: Calibrate Predictions (Optional)

Fit linear calibration to adjust predictions:

```bash
python scripts/calibrate_predictions.py fit --mode per_category
```

### Configuration

Enable new features via `config.yaml`:

```yaml
# Calibration settings
calibration:
  enabled: false
  mode: "global"  # Options: global, per_category, by_remount_bin

# k-NN fallback for cold-start titles
knn:
  enabled: true
  k: 5

# Trained model path
model:
  path: "models/model_xgb_remount_postcovid.joblib"
```

## ML Feature Registry

The app uses a **config-driven approach** with CSV files as the single source of truth for ML feature metadata. This allows:

- **Feature Inventory**: Centralized documentation of all features (`config/ml_feature_inventory_alberta_ballet.csv`)
- **Leakage Prevention**: Audit trail of which features are allowed at forecast time (`config/ml_leakage_audit_alberta_ballet.csv`)
- **Data Source Mapping**: Links features to their source systems (`config/ml_data_sources_alberta_ballet.csv`)
- **Join Key Documentation**: Standardized keys for data integration (`config/ml_join_keys_alberta_ballet.csv`)

### Training a Model

From the **Model Training** page in the UI, or programmatically:

```python
from ml.training import train_baseline_model
result = train_baseline_model()
print(result)
```

## Machine Learning Models

The app uses advanced regression models to improve prediction accuracy:

- **XGBoost** for overall predictions (≥8 historical samples)
- **Gradient Boosting** for medium datasets (5-7 samples)
- **Ridge Regression** for category-specific models
- **k-NN Fallback** for cold-start titles without history
- Automatic fallback to simple linear regression for small datasets

Performance metrics are displayed in the UI including R², MAE, and cross-validated scores.

See [ML_MODEL_DOCUMENTATION.md](ML_MODEL_DOCUMENTATION.md) for detailed technical information.

## Data Files

The app uses several CSV files in the `data/` directory:

- `history_city_sales.csv` - Historical ticket sales by city
- `baselines.csv` - Familiarity and motivation scores for known titles
- `marketing_spend_per_ticket.csv` - Historical marketing spend data
- `past_runs.csv` - Performance dates for seasonality analysis
- `showtype_expense.csv` - Production expense by show type
- `segment_priors.csv` - Audience segment preferences
- `ticket_priors_raw.csv` - Historical ticket medians
- `title_id_map.csv` - Title canonicalization mapping
- `modelling_dataset.csv` - Generated leak-free dataset (created by scripts)

## Requirements

### Core Dependencies

- Python 3.11+
- Streamlit 1.37+
- pandas 2.0+
- numpy 1.21+
- scikit-learn 1.4+
- xgboost 2.0+
- matplotlib 3.0+
- joblib 1.3+

See `requirements.txt` for complete dependency list.

### Optional Dependencies

```bash
# For fuzzy title matching
pip install rapidfuzz

# For SHAP explanations (training scripts)
pip install shap

# For LightGBM alternative to XGBoost
pip install lightgbm

# For PyCaret model validation
pip install pycaret>=3.3.0  # requires Python ≤3.12
```

## Project Structure

```
.
├── streamlit_app.py           # Main application
├── title_scoring_helper.py    # Helper app for generating baselines
├── requirements.txt           # Python dependencies
├── config.yaml                # Configuration parameters
├── config/                    # ML registry CSVs
│   ├── registry.py            # Registry loader functions
│   └── ml_*.csv               # Feature/leakage/source metadata
├── data/                      # Data files
│   ├── loader.py              # Data loading utilities
│   ├── features.py            # Feature engineering
│   ├── leakage.py             # Leakage prevention
│   └── title_id_map.csv       # Title canonicalization
├── ml/                        # ML pipeline modules
│   ├── dataset.py             # Dataset builder
│   ├── training.py            # Model training
│   ├── scoring.py             # Model scoring
│   ├── knn_fallback.py        # k-NN cold-start predictions
│   └── predict_utils.py       # Streamlit prediction helpers
├── scripts/                   # Training & evaluation scripts
│   ├── build_modelling_dataset.py
│   ├── train_safe_model.py
│   ├── backtest_timeaware.py
│   └── calibrate_predictions.py
├── models/                    # Saved models & metadata
├── results/                   # Backtest results & plots
├── diagnostics/               # Dataset diagnostics
├── pages/                     # Streamlit multi-page app
├── tests/                     # Unit tests
├── utils/                     # Helper modules
│   ├── priors.py
│   └── canonicalize_titles.py # Title normalization
└── ML_MODEL_DOCUMENTATION.md  # Technical ML documentation
```

## Contributing

When making changes:

1. Maintain backward compatibility with existing data files
2. Preserve model performance metrics display in UI
3. Test with actual historical data in `data/` directory
4. Update ML_MODEL_DOCUMENTATION.md if changing models
5. Run security checks with CodeQL before committing
6. **Never use current-run ticket columns as predictors** - see [Leakage Prevention](#leakage-prevention)

### Leakage Prevention

The training pipeline includes safety checks to prevent data leakage:

```python
# These columns are FORBIDDEN as predictors:
# - Single Tickets - Calgary/Edmonton
# - Subscription Tickets - Calgary/Edmonton
# - Total Tickets / Total Single Tickets
# - YourModel_* columns

# These are ALLOWED (prior-season aggregates):
# - prior_total_tickets
# - ticket_median_prior
# - years_since_last_run
```

Run tests to verify no leakage:
```bash
pytest tests/test_no_leakage_in_dataset.py -v
```
