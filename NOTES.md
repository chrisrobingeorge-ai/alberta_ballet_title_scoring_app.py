# Statistical Implementation Notes

This document describes the statistical features and configuration in the title scoring application.

## Model Training Features (`ml/training.py`)

### Available Functions
- `load_ml_config()` - Load YAML configuration
- `apply_target_transform()` / `inverse_target_transform()` - Log1p transform support
- `get_hyperparam_grid()` - Get hyperparameter search space by model type
- `create_model_pipeline()` - Create sklearn pipeline with preprocessing
- `extract_feature_importances()` - Extract and aggregate feature importances
- `compute_subgroup_metrics()` - Compute MAE/RMSE/R² by subgroup
- `save_model_metadata()` - Save model versioning info
- `save_feature_importances()` - Export to JSON
- `save_evaluation_metrics()` - Save metrics to `metrics/` folder
- `train_with_cross_validation()` - Full CV training with metrics

### Model Training Capabilities
- **Time-based cross-validation**: Uses `TimeSeriesSplit` and custom `TimeSeriesCVSplitter` for walk-forward validation
- **Hyperparameter tuning**: `RandomizedSearchCV` with configurable search space
- **Ensemble models**: GradientBoostingRegressor as an alternative to Random Forest
- **Feature importance tracking**: Automatic extraction and export of feature importances
- **Model metadata versioning**: Saves training date, metrics, features, and hyperparameters to JSON

## Target Variable Configuration

**Configuration:**
```yaml
target:
  column: "total_single_tickets"
  use_log_transform: true
  normalize_by_shows: false
  normalize_by_capacity: false
```

- **Log1p transform**: Configurable via `target.use_log_transform` in config
- Automatic inverse transform for predictions to return to original scale
- Support for normalized targets (tickets_per_show, tickets_per_capacity) via config

## Cold Start KNN Fallback (`ml/knn_fallback.py`)

**Capabilities:**
- **Distance-weighted voting**: `weights='distance'` parameter for inverse-distance weighting
- **PCA preprocessing**: Optional PCA for dimensionality reduction (Mahalanobis-like distance)
- **Configurable parameters**: All parameters (k, recency_decay, etc.) tunable via YAML config

**Functions:**
- `load_knn_config()` - Load KNN settings from config file
- PCA transform during index building and prediction
- `use_pca` and `pca_components` parameters

**Configuration:**
```yaml
knn:
  k: 5
  metric: "cosine"
  weights: "distance"
  use_pca: false
  pca_components: 3
  recency_weight: 0.5
  recency_decay: 0.1
```

## Feature Validation + Schema Enforcement (`ml/scoring.py`)

**Capabilities:**
- **Schema validation**: Validates input DataFrame against training schema
- **Column drift detection**: Warns on missing, extra, or reordered columns
- **Custom warning class**: `SchemaValidationWarning` for filtering warnings

**Functions:**
- `load_training_schema()` - Load schema from model metadata
- `validate_input_schema()` - Validate DataFrame against schema
- `score_with_uncertainty()` - Score with prediction intervals

## Uncertainty + Explainability

**Capabilities:**
- **Prediction intervals**: Bootstrap-based uncertainty via tree variance (Random Forest)
- **Feature importance export**: Saved to `outputs/feature_importance.json`
- **Top N features**: Configurable number of top features to report

**Configuration:**
```yaml
explainability:
  export_importances: true
  importance_output_path: "outputs/feature_importance.json"
  top_n_features: 20

uncertainty:
  enable_intervals: true
  confidence_level: 0.9
  method: "quantile_forest"
```

## Evaluation Reporting

**Capabilities:**
- **Subgroup metrics**: MAE, RMSE, R² reported by genre, season, city, capacity bin
- **Metrics folder**: Results saved to `metrics/` as JSON and CSV
- **CV results**: Cross-validation fold-by-fold results saved to `metrics/cv_results.json`

**Output Files:**
- `metrics/evaluation_metrics.json` - Overall and subgroup metrics
- `metrics/evaluation_metrics_by_subgroup.csv` - CSV for analysis
- `metrics/cv_results.json` - Cross-validation results

## Model Versioning

**Capabilities:**
- **`model_metadata.json`**: Stores feature names, training date, metrics, hyperparameters
- **Config hash**: Tracks configuration version for reproducibility

**Output:**
```json
{
  "model_path": "models/title_demand_rf.pkl",
  "training_date": "2024-01-15T10:30:00",
  "metrics": {"mae": 1234.5, "r2": 0.78},
  "n_features": 15,
  "features": ["wiki", "trends", ...],
  "hyperparameters": {"n_estimators": 300, ...}
}
```

## Configuration File

The `configs/ml_config.yaml` contains all ML settings:
- Model type and hyperparameter search space
- Target variable transformation settings
- KNN fallback configuration
- Cross-validation settings
- Evaluation and explainability options
- Uncertainty quantification settings
- Model versioning options

## Usage

### Training with default config:
```bash
python -m ml.training
```

### Training with custom config:
```bash
python -m ml.training --config configs/my_config.yaml
```

### Cross-validation only:
```bash
python -m ml.training --cv-only
```

### Scoring with schema validation:
```python
from ml.scoring import score_dataframe, validate_input_schema

# Validate schema first
is_valid, warnings = validate_input_schema(df_features)

# Score with predictions
predictions = score_dataframe(df_features, validate_schema=True)

# Score with uncertainty intervals
from ml.scoring import score_with_uncertainty
results = score_with_uncertainty(df_features, confidence_level=0.9)
```

## Key Features

1. **No Future Data Leakage**: Strict chronological splits ensure training data always precedes test data
2. **Reproducibility**: Model metadata and config tracking enable exact reproduction
3. **Interpretability**: Feature importance export and subgroup metrics aid understanding
4. **Robustness**: Cross-validation provides more reliable performance estimates
5. **Configurability**: All key parameters exposed via YAML config
6. **Schema Safety**: Validation prevents silent failures from schema drift
