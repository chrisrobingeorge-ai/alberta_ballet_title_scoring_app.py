# Archive Directory

This directory contains CSV files that were previously in `data/economics/` but are not currently being used by the application.

## Archived Files

The following 13 CSV files were moved here after a comprehensive code review on 2025-12-17:

1. `author_index_economics_art_culture.csv`
2. `business_entrants.csv`
3. `business_exits.csv`
4. `calgary_creative_industries.csv`
5. `calgary_creative_industries_report.csv`
6. `capital_investments.csv`
7. `community_development_insights_gus.csv`
8. `economics_arts.csv`
9. `farm_cash_receipts.csv`
10. `funding_npo_performing_arts.csv`
11. `gdp_basics.csv`
12. `net_migration.csv`
13. `spotlight_all_waves_master_metrics.csv`

## Verification Process

A thorough scan of the entire repository was performed to identify which economics CSV files are actively referenced in:
- Python source code (`.py` files)
- Configuration files (`.yaml`, `.yml`, `.json` files)
- Documentation files

No active references to these files were found in the codebase.

## Files Still in Use

The following CSV files remain in `data/economics/` as they are actively used:
- `boc_annual_fx_rates.csv` - Referenced in pages/7_External_Data_Impact.py
- `boc_cpi_monthly.csv` - Referenced in data/loader.py and pages/7_External_Data_Impact.py
- `boc_legacy_annual_rates.csv` - Referenced in pages/7_External_Data_Impact.py
- `commodity_price_index.csv` - Referenced in data/loader.py and pages/7_External_Data_Impact.py
- `nanos_better_off.csv` - Referenced in data/loader.py and pages/7_External_Data_Impact.py
- `nanos_consumer_confidence.csv` - Referenced in data/loader.py and pages/7_External_Data_Impact.py
- `oil_price.csv` - Referenced in data/loader.py and pages/7_External_Data_Impact.py
- `stats_can_culture_sports_trade_domain.csv` - Referenced in pages/7_External_Data_Impact.py
- `stats_can_culture_sports_trade_gdp.csv` - Referenced in pages/7_External_Data_Impact.py
- `stats_can_culture_sports_trade_provincial.csv` - Referenced in pages/7_External_Data_Impact.py
- `travel_alberta_tourism_impact.csv` - Referenced in data/loader.py and pages/7_External_Data_Impact.py
- `unemployment_by_city.csv` - Referenced in data/loader.py and pages/7_External_Data_Impact.py

## Restoring Files

If you need to restore any of these files, simply move them back to `data/economics/` and ensure the application code references them appropriately.
