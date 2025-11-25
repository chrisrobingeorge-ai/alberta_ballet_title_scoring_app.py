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

## How to Run

### 1. Install the requirements

```bash
pip install -r requirements.txt
```

### 2. Run the app

```bash
streamlit run streamlit_app.py
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

## Requirements

### Core Dependencies

- Python 3.12+
- Streamlit 1.37+
- pandas 2.0-2.2 (compatible with PyCaret)
- numpy 1.21-1.27 (compatible with PyCaret)
- scikit-learn 1.4-1.5 (compatible with PyCaret)
- xgboost 2.0+
- matplotlib 3.0-3.8 (compatible with PyCaret)

See `requirements.txt` for complete dependency list.

### PyCaret Integration (Optional)

- **PyCaret** is optional and only needed for the Model Validation page
  - Install with: `pip install pycaret>=3.3.0` (supports Python 3.12)
  - Requires specific version constraints for pandas, numpy, matplotlib, and scikit-learn
  - The main title scoring features work without PyCaret
  - See [MODEL_VALIDATION_GUIDE.md](MODEL_VALIDATION_GUIDE.md) for detailed setup instructions

## Project Structure

```
.
├── streamlit_app.py           # Main application
├── title_scoring_helper.py    # Helper app for generating baselines
├── requirements.txt            # Python dependencies
├── config.yaml                 # Configuration parameters
├── config/                     # ML registry CSVs
│   ├── registry.py            # Registry loader functions
│   ├── ml_feature_inventory_alberta_ballet.csv
│   ├── ml_leakage_audit_alberta_ballet.csv
│   ├── ml_join_keys_alberta_ballet.csv
│   └── ml_data_sources_alberta_ballet.csv
├── data/                       # Data files
│   ├── loader.py              # Data loading utilities
│   ├── features.py            # Feature engineering
│   └── leakage.py             # Leakage prevention
├── ml/                         # ML pipeline modules
│   ├── dataset.py             # Dataset builder
│   ├── training.py            # Model training
│   └── scoring.py             # Model scoring
├── models/                     # Saved models directory
├── pages/                      # Streamlit multi-page app
│   ├── 1_Feature_Registry.py
│   ├── 2_Leakage_Guard.py
│   ├── 3_Data_Quality.py
│   ├── 4_Model_Training.py
│   └── 5_Title_Scoring.py
├── tests/                      # Unit tests
├── tools/                      # Utility scripts
├── utils/                      # Helper modules
└── ML_MODEL_DOCUMENTATION.md   # Technical ML documentation
```

## Contributing

When making changes:

1. Maintain backward compatibility with existing data files
2. Preserve model performance metrics display in UI
3. Test with actual historical data in `data/` directory
4. Update ML_MODEL_DOCUMENTATION.md if changing models
5. Run security checks with CodeQL before committing
