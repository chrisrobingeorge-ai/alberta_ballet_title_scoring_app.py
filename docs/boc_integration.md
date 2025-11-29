# Bank of Canada Valet API Integration

This document describes the integration with the Bank of Canada Valet API for fetching live economic data to supplement historical economic factors in the Title Scoring application.

## Overview

The Bank of Canada provides the [Valet Web API](https://www.bankofcanada.ca/valet/docs) for programmatic access to economic data. This integration uses the API to fetch current economic indicators that supplement the existing historical datasets.

**Important Design Principle**: The existing historical economic data (WCS oil prices, Alberta unemployment, etc.) remains fully intact. The BoC integration provides *supplemental* "live" or "latest" values for today's macro conditions. Historical analysis, backtests, and model training continue to rely on the existing historical datasets.

## Series-Based Data

### What is a Series?

A **series** is an individual time series of economic observations. Each series has:
- A unique identifier (e.g., `B114039` for the policy rate)
- A label and description
- Observations with date and value pairs

### Configured Series

The application fetches the following individual series (see `config/economic_boc.yaml`):

| Key | Series ID | Description |
|-----|-----------|-------------|
| `policy_rate` | B114039 | Bank of Canada Target Rate |
| `corra` | AVG.INTWO | Canadian Overnight Repo Rate Average |
| `gov_bond_2y` | BD.CDN.2YR.DQ.YLD | 2-year Government bond yield |
| `gov_bond_5y` | BD.CDN.5YR.DQ.YLD | 5-year Government bond yield |
| `gov_bond_10y` | BD.CDN.10YR.DQ.YLD | 10-year Government bond yield |
| `bcpi_total` | A.BCPI | Annual Bank of Canada Commodity Price Index |
| `bcpi_energy` | A.ENER | Annual BCPI Energy component |
| `bcpi_ex_energy` | A.BCNE | Annual BCPI Excluding Energy |
| `cpi_core` | ATOM_V41693242 | CPIX - Core CPI excluding volatile components |

### API Endpoint

Series data is fetched via:
```
GET https://www.bankofcanada.ca/valet/observations/{series_name}/json?recent=1
```

No API key is required.

## Group-Based Data (Macro Bundles)

### What is a Group?

A **group** is a bundle of related series that the Bank of Canada publishes together. Groups allow you to fetch observations for multiple related series in a single API call.

Each group has:
- A unique identifier (e.g., `FX_RATES_DAILY`, `BCPI_MONTHLY`)
- A label
- A link to the group endpoint
- A description

### How Groups Differ from Individual Series

| Aspect | Individual Series | Groups |
|--------|------------------|--------|
| Data returned | Single series observations | Multiple series observations |
| API call | One series per request | All series in group per request |
| Use case | Specific indicators | Related indicator bundles |
| Example | `A.BCPI` (total commodity index) | `BCPI_MONTHLY` (all BCPI components) |

### Local Groups List

The file `data/economics/boc_groups_list.json` contains a static list of all available BoC groups. This file is for **discovery and documentation only**—live observations come from the API endpoints.

Structure:
```json
{
  "terms": { "url": "https://www.bankofcanada.ca/terms/" },
  "groups": {
    "FX_RATES_DAILY": {
      "label": "Daily exchange rates",
      "link": "https://www.bankofcanada.ca/valet/groups/FX_RATES_DAILY",
      "description": "Daily average exchange rates..."
    },
    ...
  }
}
```

### Configured Groups

The application can optionally fetch the following groups for macro context (see `config/economic_boc.yaml`):

| Config Key | Group ID | Description |
|------------|----------|-------------|
| `exchange_rates_daily` | FX_RATES_DAILY | Daily average exchange rates |
| `exchange_rates_monthly` | FX_RATES_MONTHLY | Monthly average exchange rates |
| `bcpi_monthly` | BCPI_MONTHLY | Monthly Bank of Canada Commodity Price Index |
| `bcpi_annual` | BCPI_ANNUAL | Annual Bank of Canada Commodity Price Index |
| `ceer_daily` | CEER_DAILY | Daily Nominal Canadian Effective Exchange Rate |
| `ceer_monthly_nominal` | CEER_MONTHLY_NOMINAL | Monthly Nominal CEER |
| `ceer_monthly_real` | CEER_MONTHLY_REAL | Monthly Real CEER |

### API Endpoint for Groups

Group observations are fetched via:
```
GET https://www.bankofcanada.ca/valet/observations/group/{groupName}/json?recent=1
```

This returns the latest observations for all series within that group.

### How Group Data is Used

Currently, group data is used for **diagnostics and context only**, not for model logic:

1. **Macro Context Snapshot**: When computing economic sentiment, the app can optionally fetch group data to provide a broader view of current economic conditions.

2. **UI Display**: A collapsible panel in the External Data Impact page shows:
   - Key BCPI components from `BCPI_MONTHLY`
   - Effective exchange rates from `CEER_DAILY` or `CEER_MONTHLY_NOMINAL`
   - Top exchange rates (CAD/USD, CAD/EUR) from `FX_RATES_DAILY`

3. **Logging**: Group data values are logged for diagnostics when available.

### Configuration

Group-based context can be enabled/disabled via config:

```yaml
# In config/economic_boc.yaml
enable_boc_group_context: true  # Set to false to skip group calls
```

When disabled, the app skips all group API calls and continues using only series-based data.

### Error Handling

If group calls fail (timeout, API problem):
- The main scoring pipeline is **not** impacted
- A warning is logged
- Macro context is omitted from the response
- The existing economic scalar logic continues to work

## Caching

Both series and group data are cached in memory with a daily TTL:
- Values expire at midnight UTC or after 24 hours, whichever comes first
- This prevents excessive API calls during a single session
- Cache can be cleared programmatically via `clear_cache()`

## Future Enhancements

Potential future uses for group data:
- Feed specific group indicators into the economic sentiment scalar
- Create composite indices from multiple group series
- Time-series analysis of group trends

For now, groups remain a lightweight diagnostic feature.
