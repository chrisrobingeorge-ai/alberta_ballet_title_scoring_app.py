# Unified Data Schema (Simple Reference)

## Performance-level fields
- production_id (string)
- title (string)
- city (YYC or YEG)
- performance_dt (YYYY-MM-DD HH:MM local)
- historical_sales (int, per performance)

## Exogenous features (optional, can add later)
- marketing_spend_daily (float, by city)
- capacity_total (int), price_bands (text)
- baselines_wiki/trends/youtube/spotify (ints)
- economy_unemployment (float), oil_price (float)
- weather_temp (float), weather_precip (float)

Note: subscriptions go in a separate table to avoid mixing with single tickets.
