# streamlit_app.py
# Alberta Ballet — Title Familiarity & Motivation Scorer (robust Wikipedia mode)
# v3: Option to disable Google Trends (for Streamlit Cloud rate-limits),
#     Wikipedia Search API to resolve best page, smarter normalization,
#     clear diagnostics, and PDF brief.

import os, math
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import requests

# Optional APIs
try:
    from googleapiclient.discovery import build  # YouTube Data API v3
except Exception:
    build = None

try:
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials
except Exception:
    spotipy = None
    SpotifyClientCredentials = None

# -------------------------
# CONFIG & WEIGHTS
# -------------------------
TRENDS_TIMEFRAME = "today 5-y"
WEIGHTS = {
    "search_trends": 0.35,   # Google Trends (can be disabled)
    "youtube": 0.25,         # needs API key
    "spotify": 0.10,         # needs API creds
    "wikipedia": 0.30,       # boosted so Wikipedia alone can carry if Trends disabled
    "safety_buffer": 0.00    # not needed with explicit weights
}

# -------------------------
# UTILITIES
# -------------------------
def normalize_series(values):
    if not values:
        return []
    vmin, vmax = min(values), max(values)
    if vmax == vmin:
        return [50.0 for _ in values]
    return [(v - vmin) * 100.0 / (vmax - vmin) for v in values]

def capped_mean(values):
    if not values:
        return 0.0
    arr = sorted(values)
    n = len(arr)
    k = max(1, int(0.05 * n))
    trimmed = arr[k: n-k] if n > 2*k else arr
    return float(sum(trimmed)) / len(trimmed)

# -------------------------
