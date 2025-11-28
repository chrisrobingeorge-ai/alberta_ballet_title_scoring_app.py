# 🎭 Alberta Ballet Title Scoring App

A Streamlit application for predicting ticket sales and planning ballet seasons using advanced machine learning models.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://blank-app-template.streamlit.app/)

## Features

- **Advanced ML Predictions**: Uses XGBoost and scikit-learn for accurate ticket projections
- **Multi-Factor Scoring**: Combines Wikipedia, Google Trends, YouTube, and Spotify metrics
- **Seasonality Adjustment**: Accounts for performance timing and repeat effects
- **City-Specific Analysis**: Separate projections for Calgary and Edmonton
- **Revenue Forecasting**: Estimates single ticket revenue by city
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

- `history_city_sales.csv` - Historical ticket sales by city (includes actual and model predictions, and ticket priors)
- `baselines.csv` - Familiarity and motivation scores for all titles (includes both historical and reference titles)
- `marketing_spend_per_ticket.csv` - Historical marketing spend data
- `past_runs.csv` - Performance dates for seasonality analysis
- `showtype_expense.csv` - Production expense by show type
- `segment_priors.csv` - Audience segment preferences
- `title_id_map.csv` - Title canonicalization mapping
- `modelling_dataset.csv` - Generated leak-free dataset (created by scripts)

### Where Do the Titles Come From?

The application uses **114 unique ballet/performance titles** in `baselines.csv`, distinguished by the `source` column:

| Source | Titles | Description |
|--------|--------|-------------|
| `historical` | 67 | **Historical titles** - Alberta Ballet performances with actual ticket sales data. These are used for training ML models. |
| `external_reference` | 47 | **Reference titles** - Well-known ballets from other companies (Royal Ballet, ABT, etc.) without AB ticket history. Used for k-NN similarity matching and cold-start predictions. |

**How titles are loaded:**
- The main Streamlit app (`streamlit_app.py`) loads all titles from `baselines.csv` (114 titles) for scoring
- Use `load_all_baselines(include_reference=False)` to get only historical titles for ML training
- Historical sales data in `history_city_sales.csv` contains 42 title records with actual ticket sales

**📖 See [VARIABLE_REFERENCE.md](VARIABLE_REFERENCE.md) for detailed explanations of all ticket-related columns and how to add external factors.**

**📖 See [ADDING_BASELINE_TITLES.md](ADDING_BASELINE_TITLES.md) for how to add baseline titles to improve model accuracy.**

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
│   ├── validation.py          # Configuration validation
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
├── legacy/                    # ⚠️ DEPRECATED scripts (do not use for production)
│   └── build_city_priors.py   # Legacy city prior generator
├── integrations/              # API integrations
│   ├── ticketmaster.py        # Ticketmaster Discovery API client
│   ├── archtics.py            # Archtics Reporting API client
│   ├── normalizer.py          # Data normalization to target schema
│   └── csv_exporter.py        # CSV export with target column order
└── ML_MODEL_DOCUMENTATION.md  # Technical ML documentation
```

---

## Archtics + Ticketmaster Integration

Pull per-show/performance data from Ticketmaster and Archtics ticketing systems and export to a normalized CSV for analysis.

### Quick Start

1. **Set up credentials** - Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
# Edit .env with your credentials
```

2. **Run the CLI**:

```bash
# By show title
python scripts/pull_show_data.py --show_title "The Nutcracker" --season 2024-25

# By show ID
python scripts/pull_show_data.py --show_id nutcracker-2024 --city Calgary

# Dry run (no API calls)
python scripts/pull_show_data.py --show_title "Swan Lake" --dry-run
```

3. **Output**: Creates `data/<show_id>_archtics_ticketmaster.csv` with normalized data.

### Batch Mode: Pull Data from CSV

Automate pulling data for all shows in your historical dataset. **When run without arguments**, the script automatically uses `data/productions/history_city_sales.csv`:

```bash
# Pull all shows (uses history_city_sales.csv by default)
python scripts/pull_show_data.py

# Dry run to preview what would be processed
python scripts/pull_show_data.py --dry-run

# Specify a different CSV file
python scripts/pull_show_data.py --from_csv path/to/custom.csv

# With season filter
python scripts/pull_show_data.py --season 2024-25
```

**Batch Mode Features:**
- Reads all unique show titles from the CSV (`show_title` column)
- Automatically deduplicates titles (e.g., "Cinderella" appearing twice)
- Exports a normalized CSV for each show to the `data/` directory
- Prints a summary report showing successes and failures
- Logs and skips shows where API data is unavailable

**Example Output:**
```
============================================================
BATCH PROCESSING SUMMARY
============================================================
Total shows in CSV:     42
Unique shows processed: 39
Duplicates skipped:     3
------------------------------------------------------------
Successful:  35
Failed:      4

✓ SUCCESSFUL:
  - The Nutcracker: 10,000 tickets, 16 performances
  - Swan Lake: 8,500 tickets, 12 performances
  ...

✗ FAILED:
  - Unknown Show: No data retrieved from APIs
  ...
============================================================
```

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `TM_API_KEY` | Ticketmaster API key | For Ticketmaster data |
| `ARCHTICS_API_KEY` | Archtics API key | For Archtics data |
| `ARCHTICS_BASE_URL` | Archtics organization endpoint | For Archtics data |
| `ARCHTICS_CLIENT_ID` | Alternative Archtics auth | Optional |

**Important**: Never commit `.env` files. Use `.env.example` as a template.

### CLI Options

```
--show_title      Show title to search for (e.g., "The Nutcracker")
--show_id         Show identifier for output file
--from_csv        Path to CSV file for batch processing (expects 'show_title' column)
--season          Production season filter (e.g., "2024-25")
--city            Filter by city (Calgary or Edmonton)
--output          Custom output file path
--tm-only         Only fetch from Ticketmaster
--archtics-only   Only fetch from Archtics
--verbose         Enable debug logging
--dry-run         Show what would be done without API calls
```

### Output CSV Schema

The normalized CSV contains these columns in order:

| Column | Description |
|--------|-------------|
| `show_title` | Show name |
| `show_title_id` | Unique identifier |
| `production_season` | Season (e.g., "2024-25") |
| `city` | Primary city |
| `venue_name` | Venue name |
| `venue_capacity` | Venue capacity |
| `performance_count_city` | Performances in primary city |
| `performance_count_total` | Total performances |
| `single_tickets_calgary` | Single tickets sold in Calgary |
| `single_tickets_edmonton` | Single tickets sold in Edmonton |
| `subscription_tickets_calgary` | Subscription tickets in Calgary |
| `subscription_tickets_edmonton` | Subscription tickets in Edmonton |
| `total_single_tickets` | Total single tickets |
| `total_subscription_tickets` | Total subscription tickets |
| `total_tickets_all` | Grand total tickets |
| `avg_tickets_per_performance` | Average tickets per show |
| `load_factor` | Tickets / (capacity × performances) |
| `weeks_to_80pct_sold` | Time to 80% sold (if available) |
| `late_sales_share` | Share of late sales (if available) |
| `channel_mix_distribution` | Sales by channel (key:value pairs) |
| `group_sales_share` | Group sales percentage |
| `comp_ticket_share` | Comp ticket percentage |
| `refund_cancellation_rate` | Refund/cancellation rate |
| `pricing_tier_structure` | Price tiers (serialized) |
| `average_base_ticket_price` | Average ticket price |
| `opening_date` | First performance date |
| `closing_date` | Last performance date |
| `weekday_vs_weekend_mix` | Weekday/weekend distribution |

### Programmatic Usage

```python
from integrations import (
    TicketmasterClient,
    ArchticsClient,
    ShowDataNormalizer,
    export_show_csv,
)

# Initialize clients
tm_client = TicketmasterClient(api_key="your_key")
archtics_client = ArchticsClient(
    api_key="your_key",
    base_url="https://your-org.archtics.com/api"
)

# Fetch data
tm_events = tm_client.search_events(keyword="The Nutcracker", city="Calgary")
archtics_sales = archtics_client.get_sales_summary(event_id="12345")

# Normalize
normalizer = ShowDataNormalizer()
normalized = normalizer.normalize(
    show_title="The Nutcracker",
    show_id="nutcracker-2024",
    tm_events=tm_events,
    archtics_sales=archtics_sales,
    season="2024-25",
)

# Export
output_path = export_show_csv(normalized)
print(f"Saved to {output_path}")
```

### Troubleshooting

| Error | Solution |
|-------|----------|
| 401 Unauthorized | Check API key is correct and not expired |
| 403 Forbidden | Verify API key has required permissions |
| 429 Rate Limited | Wait and retry; the client handles automatic retries |
| Empty results | Verify show title/ID spelling; check season format |
| No credentials | Set environment variables or create `.env` file |

### Rate Limits

- **Ticketmaster Discovery API**: 5 requests/second, 5000/day (free tier)
- **Archtics**: Varies by contract

The API clients include built-in rate limiting and retry logic.

### Data Lineage

| Field | Ticketmaster Source | Archtics Source |
|-------|--------------------|-----------------| 
| Events/performances | Discovery API `/events` | `/events`, `/performances` |
| Venue info | Embedded in event | `/venues` endpoint |
| Ticket sales | Not available | `/sales` endpoint |
| Channel mix | Not available | `/sales/channels` |
| Price tiers | `priceRanges` in event | Venue manifest |

**Note**: Some fields require both data sources. Fields not available from APIs are documented in the output with null values.

### Deploying the Integration

The Archtics + Ticketmaster integration can be deployed in several ways depending on your use case:

#### Option 1: CLI Script (Recommended for Automation)

Run the CLI script directly for one-time or scheduled data pulls:

```bash
# One-time pull
python scripts/pull_show_data.py --show_title "The Nutcracker" --season 2024-25

# Scheduled pull via cron (Linux/macOS)
# Add to crontab: crontab -e
0 6 * * * cd /path/to/repo && python scripts/pull_show_data.py --show_title "The Nutcracker" --season 2024-25 >> /var/log/show_pull.log 2>&1

# Scheduled pull via Task Scheduler (Windows)
# Create a scheduled task pointing to the script
```

#### Option 2: Cloud Function / Lambda

Deploy as a serverless function for event-driven pulls. The implementation follows the same pattern as `scripts/pull_show_data.py`:

```python
# Example AWS Lambda handler - see scripts/pull_show_data.py for full implementation
import os
from integrations import TicketmasterClient, ArchticsClient, ShowDataNormalizer, export_show_csv

def lambda_handler(event, context):
    show_title = event.get("show_title", "The Nutcracker")
    season = event.get("season")
    
    # Initialize clients (API keys from environment variables)
    tm_client = TicketmasterClient(api_key=os.environ.get("TM_API_KEY"))
    archtics_client = ArchticsClient(
        api_key=os.environ.get("ARCHTICS_API_KEY"),
        base_url=os.environ.get("ARCHTICS_BASE_URL")
    )
    
    # Fetch and normalize - see scripts/pull_show_data.py for full logic
    tm_events = tm_client.search_events(keyword=show_title, state_code="AB")
    # ... additional API calls and normalization
    
    return {"statusCode": 200, "body": f"Processed {show_title}"}
```

#### Option 3: Integrate with Streamlit App

The data pulled by the CLI script lands in the `data/` directory as CSV files. The main Streamlit app (`streamlit_app.py`) automatically picks up these files for analysis:

1. **Pull data**: `python scripts/pull_show_data.py --show_title "Show Name"`
2. **Run Streamlit**: `streamlit run streamlit_app.py`
3. The app uses the normalized CSV for predictions and analysis

#### Which App to Run?

| Goal | Command |
|------|---------|
| Pull data from APIs (one-time) | `python scripts/pull_show_data.py --show_title "Show Name"` |
| Pull data from APIs (scheduled) | Set up cron/scheduler for `scripts/pull_show_data.py` |
| View predictions & analytics | `streamlit run streamlit_app.py` |
| Score new titles manually | `python title_scoring_helper.py` |

**Typical Workflow**:
1. Set up API credentials in `.env`
2. Run `scripts/pull_show_data.py` to fetch fresh data (manually or via scheduler)
3. Run `streamlit run streamlit_app.py` to view and analyze the data

---

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

---

## Bootstrapping the System (First Forecast)

This section explains how to set up the system from scratch and run your first forecast.

### Minimal Required Files

To get started, you need these data files in the `data/` directory:

| File | Purpose | Minimum Columns |
|------|---------|-----------------|
| `history_city_sales.csv` | Historical ticket sales | `Show Title`, `Single Tickets - Calgary`, `Single Tickets - Edmonton` |
| `baselines.csv` | Title signal scores | `title`, `wiki`, `trends`, `youtube`, `spotify`, `category`, `gender`, `source` |

Additionally, these config files should exist in `config/`:
- `ml_feature_inventory_alberta_ballet.csv`
- `ml_leakage_audit_alberta_ballet.csv`
- `ml_join_keys_alberta_ballet.csv`
- `ml_pipelines_alberta_ballet.csv`
- `ml_modelling_tasks_alberta_ballet.csv`
- `ml_data_sources_alberta_ballet.csv`

### Step-by-Step Bootstrap

1. **Prepare your data files** (see [Data Files](#data-files) section above)

2. **Build the modelling dataset** (creates leak-free features):
   ```bash
   python scripts/build_modelling_dataset.py
   ```

3. **Train the safe model** with cross-validation:
   ```bash
   python scripts/train_safe_model.py --tune
   ```

4. **Run backtests** to evaluate prediction methods:
   ```bash
   python scripts/backtest_timeaware.py
   ```

5. **Calibrate predictions** (optional, for fine-tuning):
   ```bash
   python scripts/calibrate_predictions.py fit --mode global
   ```

6. **Launch the app**:
   ```bash
   streamlit run streamlit_app.py
   ```

### Obtaining Initial Datasets

- **history_city_sales.csv**: Export from your ticketing system (Tessitura, Spektrix, etc.)
- **baselines.csv**: Use `title_scoring_helper.py` to generate signal scores for your titles
- **Config CSVs**: Template files are included in the repository

---

## Config YAML Reference

The `config.yaml` file controls app behavior. Here's a reference for each section:

### Segment Multipliers (`segment_mult`)

Adjusts familiarity/motivation scores by audience segment and category.

```yaml
segment_mult:
  General Population:
    gender: { female: 1.0, male: 1.0, co: 1.0, na: 1.0 }
    category: { family_classic: 1.0, classic_romance: 1.0, ... }
```

### Region Multipliers (`region_mult`)

Adjusts scores by geographic region.

```yaml
region_mult:
  Province: 1.00
  Calgary: 1.05
  Edmonton: 0.95
```

### City Splits (`city_splits`)

Controls Calgary/Edmonton allocation and subscriber shares.

| Key | Description | Default |
|-----|-------------|---------|
| `default_base_city_split` | Default Calgary/Edmonton ratio | 60/40 |
| `city_clip_range` | Min/max bounds for city share | [0.15, 0.85] |
| `default_subs_share` | Default subscriber percentage by city | YYC: 35%, YEG: 45% |

### Demand Settings (`demand`)

| Key | Description | Default |
|-----|-------------|---------|
| `postcovid_factor` | Post-COVID demand adjustment (0.85 = 15% haircut) | 0.85 |
| `ticket_blend_weight` | Balance between signals and historical tickets | 0.50 |

### Seasonality Settings (`seasonality`)

| Key | Description | Default |
|-----|-------------|---------|
| `k_shrink` | Shrinkage factor for low-sample months | 3.0 |
| `min_factor` | Minimum seasonality multiplier | 0.90 |
| `max_factor` | Maximum seasonality multiplier | 1.15 |
| `n_min` | Minimum samples to trust a month factor | 3 |

### Marketing Defaults (`marketing_defaults`)

Default marketing spend per single ticket by city when no historical data exists.

```yaml
marketing_defaults:
  default_marketing_spt_city:
    Calgary: 10.0
    Edmonton: 8.0
```

### Calibration Settings (`calibration`)

| Key | Description | Options |
|-----|-------------|---------|
| `enabled` | Apply calibration to predictions | true/false |
| `mode` | Calibration strategy | "global", "per_category", "by_remount_bin" |

### KNN Settings (`knn`)

| Key | Description | Default |
|-----|-------------|---------|
| `enabled` | Enable k-NN fallback for cold-start titles | true |
| `k` | Number of nearest neighbors | 5 |
| `metric` | Distance metric | "cosine" |

### Model Settings (`model`)

| Key | Description | Default |
|-----|-------------|---------|
| `path` | Path to trained model file | "models/model_xgb_remount_postcovid.joblib" |
| `use_for_cold_start` | Use ML model for new titles | true |
| `confidence_threshold` | R² threshold for fallback to KNN | 0.6 |

### Security Settings (`security`)

| Key | Description | Default |
|-----|-------------|---------|
| `hide_row_level_data` | Suppress raw sales rows in UI (show only aggregates) | false |
| `mask_sensitive_exports` | Mask specific values in reports | false |

---

## Security & Deployment

### API Key Handling

The app uses optional external APIs for fetching title signals:

| API | Environment Variable | Purpose |
|-----|---------------------|---------|
| YouTube Data API | `YOUTUBE_API_KEY` or `st.secrets["youtube_api_key"]` | Fetch video view counts |
| Spotify API | `SPOTIFY_CLIENT_ID`, `SPOTIFY_CLIENT_SECRET` | Track popularity scores |
| Google Trends | (No key required, uses pytrends) | Search interest data |

**Best Practices:**
- Store API keys in Streamlit secrets (`~/.streamlit/secrets.toml`) or environment variables
- Never commit API keys to source control
- Keys are optional - the app falls back to offline heuristics if unavailable

### Deployment Guidelines

⚠️ **This app contains sensitive business data and should not be deployed publicly.**

**Recommended Deployment:**
- Deploy behind a VPN or corporate network
- Use SSO/SAML authentication proxy (e.g., Okta, Azure AD)
- Enable Streamlit's built-in authentication if using Streamlit Cloud (Teams/Enterprise)

**Security Checklist:**
- [ ] API keys stored in secrets, not code
- [ ] App behind VPN or authentication proxy
- [ ] `hide_row_level_data: true` in production if needed
- [ ] HTTPS enabled for all connections
- [ ] Regular dependency updates (check for vulnerabilities)

**Not Recommended:**
- ❌ Public Streamlit Cloud deployment without auth
- ❌ Direct internet exposure without authentication
- ❌ Sharing URLs publicly

### Data Privacy

The app processes potentially sensitive data:
- Historical ticket sales by title and city
- Revenue figures
- Marketing spend data

When `hide_row_level_data: true` is set in `config.yaml`:
- Raw row-level sales data is hidden in the UI
- Only aggregate statistics are displayed
- Export files may still contain detailed data (controlled by `mask_sensitive_exports`)
