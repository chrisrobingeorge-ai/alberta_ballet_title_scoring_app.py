# Ticket Forecasting Model (Single Tickets)

## What it does
Forecasts single-ticket demand by **title × city × performance date**.
Primary output: a point forecast and three risk bounds (P10/P50/P90).

## What it does NOT do (for clarity)
- Does not forecast subscriptions or comps.
- Does not change UI layout; it only provides numbers the UI can display.

## Data it expects (now or later)
- Historical single-ticket sales at performance level (YYC/YEG).
- Marketing spend (daily by city).
- Baselines for new titles (wiki/trends/youtube/spotify).
- **Explicit date fields** for time-aware validation and feature engineering.
- Optional: economy & weather indicators.

**Note:** The pipeline now uses explicit date fields for time-aware validation and feature engineering. This improves forecast reliability and enables integration with external data sources.

## How the UI calls the model
`service/forecast.py -> predict(title, city, performance_dt)` returns:
{
  "point": int,
  "interval": {"p10": int, "p50": int, "p90": int},
  "drivers": [{"feature": str, "impact": float}]
}

## Validation approach (later)
Rolling backtests at -28, -14, -7 days to performance.
Metrics: MAE, sMAPE, and interval calibration.
