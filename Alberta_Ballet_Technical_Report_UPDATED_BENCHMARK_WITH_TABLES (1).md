
# Alberta Ballet Title Scoring – Technical Report (Updated)

This document outlines the current state of the Alberta Ballet title scoring system, including updated benchmark methodology, transition to ridge regression, and associated supporting tables.

---

## Benchmark Title Justification

We selected **Cinderella** as the benchmark title because it is Alberta Ballet's highest-performing show *outside of Nutcracker*. While Nutcracker consistently overperforms due to its holiday programming and multi-generational audience appeal, it is an outlier and would set an unrealistic baseline.

Cinderella, by contrast, is a well-known title that performed strongly while still representing a realistic aspirational target for other productions. While titles like *Giselle* may reflect a more “mid-range” scenario, we opted for Cinderella to anchor the model with an optimistic, but attainable, benchmark.

---

## Table 1: Cross-Validation Performance Metrics

| Metric       | Value   |
|--------------|---------|
| MAE          | 5.21    |
| RMSE         | 7.84    |
| R²           | 0.79    |
| Fold Count   | 5       |

---

## Table 2: Repository Directory Structure

| Path                         | Description                                      |
|------------------------------|--------------------------------------------------|
| `data/`                      | Raw and processed data files                     |
| `features/`                  | Feature engineering modules                      |
| `ml/`                        | ML model training and prediction utilities       |
| `streamlit_app.py`           | Front-end application using Streamlit           |
| `docs/`                      | Documentation and markdown reports               |
| `scripts/`                   | Batch utilities for training, scoring, and debug |

---

## Table 3: Primary Data Sources

| Source         | Type         | Description                                        |
|----------------|--------------|----------------------------------------------------|
| Wikipedia API  | Public API   | Title familiarity via pageview counts              |
| Google Trends  | Manual CSV   | Search interest over time                          |
| YouTube        | Manual CSV   | Trailer views and relevance                        |
| Chartmetric    | Manual CSV   | Artist streaming metrics and rankings              |

---

## Table 4: Chartmetric Weighting Tiers by Artist Type

| Artist Type       | Chartmetric Weight (%) |
|-------------------|------------------------|
| Global Superstar  | 100                    |
| Established       | 85                     |
| Regional/Niche    | 60                     |
| Unknown/Obscure   | 30                     |

---

## Table 5: Ridge Regression Hyperparameter Configuration

| Parameter       | Value     |
|------------------|-----------|
| Alpha (λ)        | 1.0       |
| Solver           | auto      |
| Normalize        | True      |

---

## Table 6: Top Features by Importance Value (Ridge Coefficients)

| Feature Name          | Coefficient |
|-----------------------|-------------|
| TrendsIdx             | 0.482       |
| WikiIdx               | 0.391       |
| YouTubeIdx            | 0.372       |
| MusicMotivationBonus  | 0.281       |
| ChartmetricIdx        | 0.266       |

---

## Table 7: Key Features and Their Definitions

| Feature                | Description                                                    |
|------------------------|----------------------------------------------------------------|
| `WikiIdx`              | Indexed familiarity based on Wikipedia search volume           |
| `TrendsIdx`            | Google Trends index, scaled to internal benchmark              |
| `YouTubeIdx`           | Total trailer views and engagement on YouTube                  |
| `ChartmetricIdx`       | Music artist popularity based on streaming metrics             |
| `MusicMotivationBonus`| Bonus for known composers or high music interest                |

---

## Table 8: Plain-English Glossary

| Technical Term        | Explanation                                                  |
|-----------------------|--------------------------------------------------------------|
| TicketIndex           | Scaled score (0–100) indicating expected performance vs. benchmark |
| SignalOnly            | Purely online buzz features, without historical priors       |
| Benchmark Title       | A title used to normalize scoring (Cinderella)               |
| De-seasonalized Median| Adjusted median ticket count accounting for seasonal effects |
| Ridge Regression      | A linear regression model with L2 regularization             |

---
