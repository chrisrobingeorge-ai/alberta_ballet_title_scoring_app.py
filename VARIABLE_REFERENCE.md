# Variable Reference Guide

This document explains the key ticket-related variables in the Alberta Ballet Title Scoring App and answers common questions about data integration.

---

## Table of Contents
1. [Ticket Column Definitions](#ticket-column-definitions)
2. [Accuracy Assessment](#accuracy-assessment)
3. [External Factors Integration](#external-factors-integration)
4. [Ready-to-Use Prompts](#ready-to-use-prompts)

---

## Ticket Column Definitions

### Data in `history_city_sales.csv`

The history file contains three categories of data:

#### 1. **Actual Box Office Data** (Ground Truth)

| Column | Source | Description |
|--------|--------|-------------|
| `single_tickets_calgary` | Box Office / Ticketmaster | Real single tickets sold in Calgary |
| `single_tickets_edmonton` | Box Office / Ticketmaster | Real single tickets sold in Edmonton |
| `total_single_tickets` | Calculated | Sum of single tickets across both cities (**ACTUAL**) |

#### 2. **External Model Predictions** (YourModel)

| Column | Source | Description |
|--------|--------|-------------|
| `yourmodel_single_tickets_calgary` | Prior forecasting system | **PREDICTED** singles for Calgary |
| `yourmodel_single_tickets_edmonton` | Prior forecasting system | **PREDICTED** singles for Edmonton |
| `yourmodel_total_single_tickets` | Prior forecasting system | **PREDICTED** total singles |

> **Note**: The "YourModel" columns appear to be predictions from an external/prior forecasting system—not from this Streamlit app. These can be used as a benchmark to compare against.

### App-Generated Forecasts

The app generates forecasts in two different places:

#### A. Main Title Scorer (`streamlit_app.py`)

When you click "Score Titles" on the main page:

| Column | Description |
|--------|-------------|
| `EstimatedTickets` | Predicted tickets from the ticket index calculation |
| `EstimatedTickets_Final` | Final predicted tickets (equal to EstimatedTickets - penalty factors removed) |
| `YYC_Singles` / `YEG_Singles` | City-level single ticket predictions |

#### B. ML Model Predictions

When using trained ML models (via `ml/scoring.py`):

| Column | Source | Description |
|--------|--------|-------------|
| `forecast_single_tickets` | ML model (`ml/scoring.py`) | **PREDICTED** single tickets from trained Random Forest/XGBoost model |

> **Key Difference**: `forecast_single_tickets` comes from a trained ML pipeline, while `EstimatedTickets_Final` uses the rule-based familiarity/motivation algorithm.

### How `EstimatedTickets_Final` is Calculated

The app's forecast is built in these steps:

1. **Signal Collection**: Gather online visibility signals (Wikipedia, Google Trends, YouTube, Spotify)
2. **Index Computation**: Convert signals to Familiarity & Motivation indices (normalized to benchmark = 100)
3. **Ticket Index**: Use regression on historical data to convert SignalOnly → TicketIndex
4. **Seasonality**: Apply category×month factor for the planned run date

> **Note (December 2024 Update)**: Remount decay and Post-COVID penalty factors have been **removed** per audit finding "Structural Pessimism". These compounding penalties caused up to 33% reduction in valid predictions. The base ML model now accounts for remount behavior through explicit features (`is_remount_recent`, `years_since_last_run`).

---

## Accuracy Assessment

### Which Column Should I Use?

| If you want... | Use this column | Notes |
|----------------|-----------------|-------|
| Actual past sales (singles) | `total_single_tickets` | Ground truth |
| Prior system's forecast | `yourmodel_total_single_tickets` | Benchmark comparison |
| ML pipeline forecast | `forecast_single_tickets` | From trained model (Title Scoring page) |
| Rule-based forecast | `EstimatedTickets_Final` | From main scorer (after all adjustments) |

### How to Compare Accuracy

To evaluate which forecast is better, compare predictions to actual results:

```python
# Example accuracy check
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("data/productions/history_city_sales.csv")

# Compare YourModel to Actual
mae_yourmodel = mean_absolute_error(df["total_single_tickets"], df["yourmodel_total_single_tickets"])
print(f"YourModel MAE: {mae_yourmodel:.0f} tickets")

# Compare App Forecast to Actual (after running Score Titles)
# The app displays this in the Model Validation page
```

### Interpreting Metrics

| Metric | What it Means | Good Value |
|--------|---------------|------------|
| **MAE** (Mean Absolute Error) | Average difference in tickets | Lower is better (< 1000 is good) |
| **RMSE** (Root Mean Squared Error) | Like MAE but penalizes big misses | Lower is better |
| **R²** (R-squared) | How much variance is explained | 0.4+ is decent, 0.6+ is good |

### Current Model Performance

Based on [ML_MODEL_DOCUMENTATION.md](ML_MODEL_DOCUMENTATION.md), the app's XGBoost model shows:
- R² Score: ~0.90 (on training data with cross-validation)
- MAE: ~4-9 index points
- Note: These are on the **TicketIndex** scale, not raw ticket counts

---

## External Factors Integration

### Current Status

**The app is NOT currently programmed to automatically pull external factor data.**

You will need to:
1. Create CSV files with your external data
2. Update `data/loader.py` to add loader functions
3. Update `scripts/build_modelling_dataset.py` to merge the data

### Suggested External Factor Files

Create these CSVs in the `data/` folder:

#### `external_economic.csv`
```csv
production_season,alberta_unemployment_rate,alberta_cpi_index,wti_oil_price_avg
2019-20,6.9,135.4,57.00
2020-21,11.4,137.0,39.00
2021-22,7.8,144.0,68.00
2022-23,5.8,157.2,78.50
2023-24,6.1,162.3,82.00
```

#### `external_weather.csv`
```csv
city,month,weather_severity_index
Calgary,January,3.2
Calgary,February,2.8
Edmonton,January,3.5
Edmonton,February,3.1
```

#### `external_competition.csv`
```csv
production_season,competing_events_count,major_event_flag
2023-24,12,0
2024-25,15,1
```

---

## Ready-to-Use Prompts

### Prompt 1: Add External Factors to the Model

Copy this prompt when you have created your external factor CSV files:

```
I've created CSV files for external factors in the data/ directory.
Please update the codebase to integrate these factors into the model:

1. Add loader functions in data/loader.py for each CSV file listed below
2. Update scripts/build_modelling_dataset.py to merge these factors
3. Add the new features to config/ml_feature_inventory_alberta_ballet.csv
4. Update config/ml_leakage_audit_alberta_ballet.csv to mark them as safe

My external factor files are:
- external_economic.csv with columns [production_season, alberta_unemployment_rate, alberta_cpi_index, wti_oil_price_avg] joined on production_season
- external_weather.csv with columns [city, month, weather_severity_index] joined on city + month_of_opening
- external_competition.csv with columns [production_season, competing_events_count, major_event_flag] joined on production_season

Please ensure the merge uses left joins to preserve all historical records.
```

### Prompt 2: Validate Model Accuracy

```
Please help me compare the accuracy of:
1. The "YourModel" predictions (yourmodel_total_single_tickets column)
2. The app's EstimatedTickets_Final predictions

Calculate MAE, RMSE, and R² for each against the actual total_single_tickets.
Show which model performs better and by how much.
```

### Prompt 3: Add a New External Factor File

```
I've created a new CSV file called [FILENAME.csv] in the data/ directory with columns:
- [column1]: [description]
- [column2]: [description]

The join key is [production_season / month_of_opening / show_title].

Please:
1. Add a loader function in data/loader.py
2. Update the dataset builder to merge this data
3. Add the feature to the ML feature registry
4. Re-train the model with: python scripts/train_safe_model.py --tune
```

---

## Summary

| Question | Answer |
|----------|--------|
| **What is `yourmodel_total_single_tickets`?** | Predictions from a prior/external forecasting system (not this app) |
| **What is `total_tickets_all`?** | Actual tickets (singles + subs) derived from real sales data |
| **What is `forecast_single_tickets`?** | ML model predictions from the Title Scoring page (trained pipeline) |
| **What is `EstimatedTickets_Final`?** | Rule-based forecast from main scorer (familiarity/motivation + adjustments) |
| **Are external factors auto-loaded?** | **No**—you need to create CSVs and update the code using the prompts above |
| **Which is most accurate?** | Compare MAE against `total_single_tickets` using the Title Scoring page |

---

*Last updated: November 2024*
