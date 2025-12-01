import io
import math
import re
import sys
import time
from datetime import datetime, timedelta, date
from typing import Dict, Optional, Tuple

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from textwrap import dedent

from reportlab.lib.pagesizes import LETTER, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

try:
    import yaml
except ImportError:
    yaml = None

try:
    import xgboost as xgb
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# ---------------------------------------------------------------------
# Config loading (preserved)
# ---------------------------------------------------------------------

def load_config(path: str = "config.yaml"):
    global SEGMENT_MULT, REGION_MULT
    global DEFAULT_BASE_CITY_SPLIT, _CITY_CLIP_RANGE
    global POSTCOVID_FACTOR, TICKET_BLEND_WEIGHT
    global K_SHRINK, MINF, MAXF, N_MIN
    global DEFAULT_MARKETING_SPT_CITY
    global ML_CONFIG

    if yaml is None:
        return

    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        cfg = {}

    SEGMENT_MULT = cfg.get("segment_multipliers", {})
    REGION_MULT = cfg.get("region_multipliers", {})

    DEFAULT_BASE_CITY_SPLIT = cfg.get("default_base_city_split", {"YYC": 0.5, "YEG": 0.5})
    _CITY_CLIP_RANGE = tuple(cfg.get("city_clip_range", (0.15, 0.85)))

    POSTCOVID_FACTOR = cfg.get("postcovid_factor", 0.8)
    TICKET_BLEND_WEIGHT = cfg.get("ticket_blend_weight", 0.5)

    seas_cfg = cfg.get("seasonality", {})
    K_SHRINK = seas_cfg.get("k_shrink", 3.0)
    MINF = seas_cfg.get("min_factor", 0.5)
    MAXF = seas_cfg.get("max_factor", 1.5)
    N_MIN = seas_cfg.get("n_min", 3)

    mkt_cfg = cfg.get("marketing_defaults", {})
    DEFAULT_MARKETING_SPT_CITY = mkt_cfg.get("default_marketing_spt_city", 1.0)

    ML_CONFIG = cfg.get("model", {"path": "model_xgb_remount_postcovid.json"})

SEGMENT_MULT = {}
REGION_MULT = {}
DEFAULT_BASE_CITY_SPLIT = {"YYC": 0.5, "YEG": 0.5}
_CITY_CLIP_RANGE = (0.15, 0.85)
POSTCOVID_FACTOR = 0.8
TICKET_BLEND_WEIGHT = 0.5
K_SHRINK = 3.0
MINF = 0.5
MAXF = 1.5
N_MIN = 3
DEFAULT_MARKETING_SPT_CITY = 1.0
ML_CONFIG = {"path": "model_xgb_remount_postcovid.json"}

# ---------------------------------------------------------------------
# Small format helpers (preserved)
# ---------------------------------------------------------------------

def _pct(v, places=0):
    try:
        return ("{0:." + str(places) + "%}").format(float(v))
    except Exception:
        return ""


def _num(v):
    try:
        return ("{0:,}").format(int(round(float(v))))
    except Exception:
        return ""


def _dec(v, places=3):
    try:
        return ("{0:." + str(places) + "f}").format(float(v))
    except Exception:
        return ""

# ---------------------------------------------------------------------
# Feature engineering stubs – here is where Stone Olafson, Econ, Live feed in
# ---------------------------------------------------------------------

FEATURE_COLUMNS = [
    "Familiarity",
    "Motivation",
    "CategoryIndex",
    "PrimarySegmentIndex",
    "SecondarySegmentIndex",
    "MonthIndex",
    "IsRemount",
    "PostCovidFactor",
    "EconomicSentiment",
    "UnemploymentRate",
    "OilPrice",
    "WeatherSeverity",
    "TrendScore",
    "YouTubeScore",
    "SpotifyScore",
    "WikipediaScore",
]


def build_feature_row(title_row: pd.Series) -> pd.Series:
    """Convert a title row into the feature space expected by the XGB model.

    This is the bridge from: Stone Olafson + Econ + Live analytics
    into the numerical vector the model was trained on.
    """
    out = {}

    out["Familiarity"] = float(title_row.get("Familiarity", 0.0) or 0.0)
    out["Motivation"] = float(title_row.get("Motivation", 0.0) or 0.0)
    out["CategoryIndex"] = float(title_row.get("CategoryIndex", 0.0) or 0.0)
    out["PrimarySegmentIndex"] = float(title_row.get("PrimarySegmentIndex", 0.0) or 0.0)
    out["SecondarySegmentIndex"] = float(title_row.get("SecondarySegmentIndex", 0.0) or 0.0)
    out["MonthIndex"] = float(title_row.get("MonthIndex", 0.0) or 0.0)
    out["IsRemount"] = float(title_row.get("IsRemount", 0.0) or 0.0)
    out["PostCovidFactor"] = float(title_row.get("PostCovidFactor", POSTCOVID_FACTOR))

    out["EconomicSentiment"] = float(title_row.get("EconomicSentiment", 0.0) or 0.0)
    out["UnemploymentRate"] = float(title_row.get("UnemploymentRate", 0.0) or 0.0)
    out["OilPrice"] = float(title_row.get("OilPrice", 0.0) or 0.0)
    out["WeatherSeverity"] = float(title_row.get("WeatherSeverity", 0.0) or 0.0)

    out["TrendScore"] = float(title_row.get("TrendScore", 0.0) or 0.0)
    out["YouTubeScore"] = float(title_row.get("YouTubeScore", 0.0) or 0.0)
    out["SpotifyScore"] = float(title_row.get("SpotifyScore", 0.0) or 0.0)
    out["WikipediaScore"] = float(title_row.get("WikipediaScore", 0.0) or 0.0)

    return pd.Series(out)

# ---------------------------------------------------------------------
# XGB model loading and prediction
# ---------------------------------------------------------------------

_XGB_MODEL = None


def load_xgb_model():
    global _XGB_MODEL
    if _XGB_MODEL is not None:
        return _XGB_MODEL

    model_path = ML_CONFIG.get("path", "model_xgb_remount_postcovid.json")
    booster = xgb.Booster()
    booster.load_model(model_path)
    _XGB_MODEL = booster
    return _XGB_MODEL


def predict_tickets_xgb(feature_df: pd.DataFrame) -> np.ndarray:
    if not ML_AVAILABLE:
        raise RuntimeError("xgboost is not installed in this environment")

    model = load_xgb_model()
    X = feature_df.reindex(columns=FEATURE_COLUMNS).astype(float)
    dmatrix = xgb.DMatrix(X.values, feature_names=list(X.columns))
    preds = model.predict(dmatrix)
    preds = np.maximum(preds, 0.0)
    return preds

# ---------------------------------------------------------------------
# Title scoring and season planning – lightweight reimplementation
# ---------------------------------------------------------------------


def score_titles(input_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in input_df.iterrows():
        feat = build_feature_row(row)
        rows.append(feat)
    feat_df = pd.DataFrame(rows)
    tickets = predict_tickets_xgb(feat_df)
    out = input_df.copy()
    out["EstimatedTickets_Final"] = tickets
    return out

# ---------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------

load_config()

st.set_page_config(page_title="Alberta Ballet Title Scoring", layout="wide")
st.title("Alberta Ballet Title Scoring App")

st.sidebar.header("Global Settings")
postcovid_factor_ui = st.sidebar.slider("Post-COVID Demand Factor", 0.4, 1.2, POSTCOVID_FACTOR, 0.05)
st.sidebar.caption("This feeds into the PostCovidFactor feature; it does not override the trained model but nudges context.")

st.sidebar.header("Model")
st.sidebar.caption("Using XGBoost remount post-COVID model as the demand engine.")

st.markdown("""
### 1. Upload or paste your titles

Provide a table of titles with at least Familiarity, Motivation, and category/segment info.
Any additional economic or live analytics fields you have will be used if present.
""")

upload = st.file_uploader("Upload CSV of titles", type=["csv"]) 
example = pd.DataFrame([
    {"Title": "Nutcracker", "Familiarity": 0.9, "Motivation": 0.85, "CategoryIndex": 1, "PrimarySegmentIndex": 1, "SecondarySegmentIndex": 0, "MonthIndex": 12, "IsRemount": 1},
    {"Title": "Mixed Bill", "Familiarity": 0.4, "Motivation": 0.6, "CategoryIndex": 2, "PrimarySegmentIndex": 2, "SecondarySegmentIndex": 1, "MonthIndex": 3, "IsRemount": 0},
])

input_df = None

if upload is not None:
    try:
        input_df = pd.read_csv(upload)
    except Exception as e:
        st.error("Could not read CSV: " + str(e))

st.expander("Or paste/edit a small table inline", expanded=input_df is None).dataframe(example, use_container_width=True)

run = st.button("Score Titles", type="primary")

if run:
    if input_df is None:
        input_df = example.copy()
    input_df["PostCovidFactor"] = postcovid_factor_ui
    try:
        results = score_titles(input_df)
        st.session_state["results"] = results
    except Exception as e:
        st.error("Error during scoring: " + str(e))

if st.session_state.get("results") is not None:
    res = st.session_state["results"]
    st.markdown("### 2. Results")
    st.dataframe(res.style.format({"EstimatedTickets_Final": "{:,.0f}"}), use_container_width=True)

    st.markdown("### 3. Quick charts")
    if "EstimatedTickets_Final" in res.columns and "Title" in res.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(res["Title"], res["EstimatedTickets_Final"])
        ax.set_ylabel("Estimated Tickets")
        ax.set_xticklabels(res["Title"], rotation=45, ha="right")
        st.pyplot(fig)

