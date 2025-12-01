# - Learns YYC/YEG splits from history.csv (or uploaded CSV)
# - Single ticket estimation only
# - Removes arbitrary 60/40 split; uses title→category→default fallback
# - Small fixes: softmax bug, LA attach loop, duplicate imports, safer guards
# - Economic sentiment factor integration for market-aware ticket estimation
# - Title scoring helper functionality integrated (Wikipedia, Google Trends,
#   YouTube, Spotify) into main app

from __future__ import annotations

import io
import math
import re
import sys
import time
from datetime import datetime, timedelta, date
from typing import Dict, Optional, Tuple, List, Any

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from textwrap import dedent

from pytrends.request import TrendReq
from googleapiclient.discovery import build
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Economic / weather / live analytics
try:
    from data.loader import (
        get_economic_sentiment_factor,
        load_oil_prices,
        load_unemployment_rates,
        load_weather_calgary,
        load_weather_edmonton,
        get_weather_impact_factor,
        get_monthly_weather_summary,
        load_live_analytics_raw,
        get_live_analytics_category_factors,
        get_category_engagement_factor,
    )

    ECONOMIC_DATA_AVAILABLE = True
    WEATHER_DATA_AVAILABLE = True
    LIVE_ANALYTICS_AVAILABLE = True
except ImportError:
    ECONOMIC_DATA_AVAILABLE = False
    WEATHER_DATA_AVAILABLE = False
    LIVE_ANALYTICS_AVAILABLE = False

    def get_economic_sentiment_factor(*args, **kwargs):
        return 1.0

    def get_weather_impact_factor(*args, **kwargs):
        return 1.0

    def get_monthly_weather_summary(*args, **kwargs):
        return {}

    def get_category_engagement_factor(*args, **kwargs):
        return 1.0

    def get_live_analytics_category_factors(*args, **kwargs):
        return {}

    def load_weather_calgary(*args, **kwargs):
        return pd.DataFrame()

    def load_weather_edmonton(*args, **kwargs):
        return pd.DataFrame()

    def load_live_analytics_raw(*args, **kwargs):
        return pd.DataFrame()

# ML scoring (relaxed schema + intervals)
from ml.scoring import score_runs_for_planning

try:
    import yaml
except ImportError:
    yaml = None

try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge, LinearRegression
    from sklearn.model_selection import cross_val_score, GridSearchCV
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import xgboost as xgb

    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False


# =============================================================================
# GLOBAL CONFIG
# =============================================================================

SEGMENT_MULT: Dict[str, float] = {}
REGION_MULT: Dict[str, float] = {}
DEFAULT_BASE_CITY_SPLIT: Tuple[float, float] = (0.5, 0.5)
_CITY_CLIP_RANGE: Tuple[float, float] = (0.1, 0.9)
POSTCOVID_FACTOR: float = 1.0
TICKET_BLEND_WEIGHT: float = 0.7
DEFAULT_MARKETING_SPT_CITY: Dict[str, float] = {}

ML_CONFIG: Dict[str, Any] = {}
KNN_CONFIG: Dict[str, Any] = {}
CALIBRATION_CONFIG: Dict[str, Any] = {}


def load_config(path: str = "config.yaml") -> None:
    global SEGMENT_MULT, REGION_MULT
    global DEFAULT_BASE_CITY_SPLIT, _CITY_CLIP_RANGE
    global POSTCOVID_FACTOR, TICKET_BLEND_WEIGHT
    global DEFAULT_MARKETING_SPT_CITY
    global ML_CONFIG, KNN_CONFIG, CALIBRATION_CONFIG

    if yaml is None:
        return

    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        return

    if "segment_mult" in cfg:
        SEGMENT_MULT = cfg["segment_mult"]

    if "region_mult" in cfg:
        REGION_MULT = cfg["region_mult"]

    city_cfg = cfg.get("city_splits", {})
    DEFAULT_BASE_CITY_SPLIT = city_cfg.get("default_base_city_split", DEFAULT_BASE_CITY_SPLIT)
    _CITY_CLIP_RANGE = tuple(city_cfg.get("city_clip_range", _CITY_CLIP_RANGE))

    demand_cfg = cfg.get("demand", {})
    POSTCOVID_FACTOR = demand_cfg.get("postcovid_factor", POSTCOVID_FACTOR)
    TICKET_BLEND_WEIGHT = demand_cfg.get("ticket_blend_weight", TICKET_BLEND_WEIGHT)
