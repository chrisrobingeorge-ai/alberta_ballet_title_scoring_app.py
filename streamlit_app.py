# Alberta Ballet — Title Familiarity & Motivation Scorer (v9 with segment-propensity outputs)
# - Hard-coded baselines (normalized to a user-selected benchmark = 100)
# - Add NEW titles; optional live fetch (Wikipedia/YouTube/Spotify) with outlier-safety
# - Segment + Region multipliers; charts + CSV
# - TicketIndex prediction for titles without history + EstimatedTickets
# - NEW: Segment propensity as an OUTPUT (primary/secondary, shares, tickets by segment)
# - NEW: Live Analytics overlays by program type

import math, time, re
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from textwrap import dedent

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

# Optional APIs used only when "Use Live Data" is ON
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

# Persist results across reruns so toggles/dropdowns don't wipe them
if "results" not in st.session_state:
    st.session_state["results"] = None  # {"df": ..., "benchmark": ..., "segment": ..., "region": ...}

# -------------------------
# PAGE / SPLASH
# -------------------------
st.set_page_config(page_title="Alberta Ballet — Title Familiarity & Motivation Scorer", layout="wide")
with st.spinner("🎭 Preparing Alberta Ballet Familiarity Baselines..."):
    time.sleep(1.0)

st.title("🎭 Alberta Ballet — Title Familiarity & Motivation Scorer (v9 — New Titles, Segment Outputs & LA overlays)")
st.caption("Hard-coded Alberta-wide baselines (normalized to your selected benchmark = 100). Add new titles; choose live fetch or offline estimate.")

# -------------------------
# METHODOLOGY & GLOSSARY SECTION
# -------------------------
with st.expander("📘 About This App — Methodology & Glossary"):
    st.markdown(dedent("""
    ### Purpose
    This tool estimates how familiar audiences are with a title and how motivated they are to attend, then blends those "online signal" estimates with ticket-informed scores. If a title has past ticket history, we use it; if it doesn’t, we predict a TicketIndex from similar shows so new titles get a fair, history-like adjustment. It also predicts which audience segments are most likely to respond, so segments become an output you can use for planning.

    ### What goes into the scores
    **Online signals (per title):** Wikipedia (awareness), Google Trends (search), YouTube (engagement, with outlier safety), Spotify (musical familiarity).  
    **Contextual multipliers:** by Audience Segment & Region.  
    **Ticket history & prediction:** convert medians to TicketIndex vs. benchmark; predict when missing.  
    **Segment propensity (output):** primary/secondary segments, shares, ticket splits.  
    **Live Analytics overlay:** attach behavior/channel/price/distance patterns by program type mapped from your categories.

    ### How the score is calculated
    1) Raw signals → Familiarity & Motivation  
       - Familiarity = 0.55·Wiki + 0.30·Trends + 0.15·Spotify  
       - Motivation = 0.45·YouTube + 0.25·Trends + 0.15·Spotify + 0.15·Wiki  
       - Apply segment and region multipliers.
    2) Normalize to your benchmark (= 100)  
    3) TicketIndex (history or predicted) — per-category model preferred, else overall  
    4) Composite = 50% Online Signals + 50% TicketIndex  
    5) Letter grade A/B/C/D/E

    ### Segment propensity (output)
    - Compute segment-specific online signal (normalized to the benchmark per segment)
    - Multiply by LiveAnalytics-informed segment priors for (region, category)
    - Normalize to shares; split EstimatedTickets accordingly

    **Note:** Use online signals to screen ideas, ticket history (or predicted TicketIndex) to ground expectations, and segment outputs+LA overlays to plan creative/price/channel tactics.
    """))

# -------------------------
# BASELINE DATA (subset for test run)
# -------------------------
BASELINES = {
    "Cinderella": {"wiki": 88, "trends": 80, "youtube": 82, "spotify": 80, "category": "family_classic", "gender": "female"},
    "Swan Lake": {"wiki": 95, "trends": 90, "youtube": 88, "spotify": 84, "category": "classic_romance", "gender": "female"},
    "Sleeping Beauty": {"wiki": 92, "trends": 85, "youtube": 78, "spotify": 74, "category": "classic_romance", "gender": "female"},
    "Hansel & Gretel": {"wiki": 78, "trends": 70, "youtube": 65, "spotify": 62, "category": "family_classic", "gender": "co"},
    "Don Quixote": {"wiki": 88, "trends": 75, "youtube": 72, "spotify": 68, "category": "classic_comedy", "gender": "male"},
    "Giselle": {"wiki": 82, "trends": 72, "youtube": 65, "spotify": 60, "category": "classic_romance", "gender": "female"},
    "La Sylphide": {"wiki": 75, "trends": 68, "youtube": 60, "spotify": 55, "category": "classic_romance", "gender": "female"},
    "Beauty and the Beast": {"wiki": 94, "trends": 97, "youtube": 92, "spotify": 90, "category": "family_classic", "gender": "female"},
    "Romeo and Juliet": {"wiki": 90, "trends": 82, "youtube": 79, "spotify": 77, "category": "romantic_tragedy", "gender": "co"},
    "The Merry Widow": {"wiki": 70, "trends": 60, "youtube": 55, "spotify": 50, "category": "romantic_comedy", "gender": "female"},
    "Peter Pan": {"wiki": 80, "trends": 78, "youtube": 85, "spotify": 82, "category": "family_classic", "gender": "male"},
    "Pinocchio": {"wiki": 72, "trends": 68, "youtube": 70, "spotify": 66, "category": "family_classic", "gender": "male"},
    "Grimm": {"wiki": 55, "trends": 52, "youtube": 50, "spotify": 45, "category": "contemporary", "gender": "na"},
    "Momix": {"wiki": 65, "trends": 60, "youtube": 68, "spotify": 60, "category": "contemporary", "gender": "na"},
    "Dangerous Liaisons": {"wiki": 66, "trends": 62, "youtube": 63, "spotify": 58, "category": "dramatic", "gender": "female"},
    "Frankenstein": {"wiki": 68, "trends": 63, "youtube": 66, "spotify": 62, "category": "dramatic", "gender": "male"},
    "Ballet Boyz": {"wiki": 45, "trends": 40, "youtube": 60, "spotify": 58, "category": "contemporary", "gender": "male"},
    "Contemporary Classical": {"wiki": 60, "trends": 55, "youtube": 58, "spotify": 50, "category": "contemporary", "gender": "na"},
    "Ballet BC": {"wiki": 50, "trends": 45, "youtube": 53, "spotify": 49, "category": "contemporary", "gender": "na"},
    "Complexions - Lenny Kravitz": {"wiki": 62, "trends": 58, "youtube": 66, "spotify": 60, "category": "contemporary", "gender": "na"},
    "Phi – David Bowie": {"wiki": 70, "trends": 65, "youtube": 72, "spotify": 75, "category": "pop_ip", "gender": "male"},
    "All of Us - Tragically Hip": {"wiki": 72, "trends": 68, "youtube": 78, "spotify": 80, "category": "pop_ip", "gender": "male"},
    "Dance Theatre of Harlem": {"wiki": 75, "trends": 68, "youtube": 74, "spotify": 68, "category": "pop_ip", "gender": "co"},
    "Shaping Sound": {"wiki": 68, "trends": 64, "youtube": 70, "spotify": 66, "category": "contemporary", "gender": "co"},
    "Taj Express": {"wiki": 66, "trends": 62, "youtube": 68, "spotify": 70, "category": "pop_ip", "gender": "male"},
    "Diavolo": {"wiki": 60, "trends": 58, "youtube": 66, "spotify": 64, "category": "contemporary", "gender": "co"},
    "Unleashed – Mixed Bill": {"wiki": 55, "trends": 50, "youtube": 60, "spotify": 52, "category": "contemporary", "gender": "co"},
    "Botero": {"wiki": 58, "trends": 54, "youtube": 62, "spotify": 57, "category": "contemporary", "gender": "male"},
    "Away We Go – Mixed Bill": {"wiki": 54, "trends": 52, "youtube": 58, "spotify": 50, "category": "contemporary", "gender": "co"},
    "Fiddle & the Drum – Joni Mitchell": {"wiki": 60, "trends": 55, "youtube": 62, "spotify": 66, "category": "pop_ip", "gender": "female"},
    "Midsummer Night’s Dream": {"wiki": 75, "trends": 70, "youtube": 70, "spotify": 68, "category": "classic_romance", "gender": "co"},
    "Dracula": {"wiki": 74, "trends": 65, "youtube": 70, "spotify": 65, "category": "romantic_tragedy", "gender": "male"},
}

# -------------------------
# SEGMENTS & REGIONS
# -------------------------
SEGMENT_MULT = {
    "General Population": {"female": 1.00, "male": 1.00, "co": 1.00, "na": 1.00,
                           "family_classic": 1.00, "classic_romance": 1.00, "romantic_tragedy": 1.00,
                           "classic_comedy": 1.00, "contemporary": 1.00, "pop_ip": 1.00, "dramatic": 1.00},
    "Core Classical (F35–64)": {"female": 1.12, "male": 0.95, "co": 1.05, "na": 1.00,
                                 "family_classic": 1.10, "classic_romance": 1.08, "romantic_tragedy": 1.05,
                                 "classic_comedy": 1.02, "contemporary": 0.90, "pop_ip": 1.00, "dramatic": 1.00},
    "Family (Parents w/ kids)": {"female": 1.10, "male": 0.92, "co": 1.06, "na": 1.00,
                                  "family_classic": 1.18, "classic_romance": 0.95, "romantic_tragedy": 0.85,
                                  "classic_comedy": 1.05, "contemporary": 0.82, "pop_ip": 1.20, "dramatic": 0.90},
    "Emerging Adults (18–34)": {"female": 1.02, "male": 1.02, "co": 1.00, "na": 1.00,
                                 "family_classic": 0.95, "classic_romance": 0.92, "romantic_tragedy": 0.90,
                                 "classic_comedy": 0.98, "contemporary": 1.25, "pop_ip": 1.15, "dramatic": 1.05},
}
REGION_MULT = {"Province": 1.00, "Calgary": 1.05, "Edmonton": 0.95}

# -------------------------
# LIKELY AUDIENCE MIX PRIORS (you can tune from Live Analytics)
# -------------------------
SEGMENT_KEYS_IN_ORDER = [
    "General Population",
    "Core Classical (F35–64)",
    "Family (Parents w/ kids)",
    "Emerging Adults (18–34)",
]

SEGMENT_PRIORS = {
    "Province": {
        "classic_romance":   {"General Population": 1.00, "Core Classical (F35–64)": 1.20, "Family (Parents w/ kids)": 0.95, "Emerging Adults (18–34)": 0.95},
        "family_classic":    {"General Population": 1.00, "Core Classical (F35–64)": 0.95, "Family (Parents w/ kids)": 1.20, "Emerging Adults (18–34)": 0.95},
        "contemporary":      {"General Population": 0.98, "Core Classical (F35–64)": 0.90, "Family (Parents w/ kids)": 0.92, "Emerging Adults (18–34)": 1.25},
        "pop_ip":            {"General Population": 1.05, "Core Classical (F35–64)": 0.95, "Family (Parents w/ kids)": 1.15, "Emerging Adults (18–34)": 1.10},
        "romantic_tragedy":  {"General Population": 1.00, "Core Classical (F35–64)": 1.10, "Family (Parents w/ kids)": 0.92, "Emerging Adults (18–34)": 0.98},
        "classic_comedy":    {"General Population": 1.02, "Core Classical (F35–64)": 1.00, "Family (Parents w/ kids)": 1.05, "Emerging Adults (18–34)": 0.98},
        "dramatic":          {"General Population": 1.05, "Core Classical (F35–64)": 1.05, "Family (Parents w/ kids)": 0.90, "Emerging Adults (18–34)": 0.98},
    },
    "Calgary": {
        "classic_romance":   {"General Population": 0.98, "Core Classical (F35–64)": 1.25, "Family (Parents w/ kids)": 0.92, "Emerging Adults (18–34)": 0.95},
        "family_classic":    {"General Population": 1.00, "Core Classical (F35–64)": 1.00, "Family (Parents w/ kids)": 1.18, "Emerging Adults (18–34)": 0.95},
        "contemporary":      {"General Population": 0.98, "Core Classical (F35–64)": 0.88, "Family (Parents w/ kids)": 0.92, "Emerging Adults (18–34)": 1.28},
        "pop_ip":            {"General Population": 1.05, "Core Classical (F35–64)": 0.95, "Family (Parents w/ kids)": 1.12, "Emerging Adults (18–34)": 1.10},
        "romantic_tragedy":  {"General Population": 0.98, "Core Classical (F35–64)": 1.15, "Family (Parents w/ kids)": 0.92, "Emerging Adults (18–34)": 0.98},
        "classic_comedy":    {"General Population": 1.02, "Core Classical (F35–64)": 1.02, "Family (Parents w/ kids)": 1.05, "Emerging Adults (18–34)": 0.98},
        "dramatic":          {"General Population": 1.05, "Core Classical (F35–64)": 1.08, "Family (Parents w/ kids)": 0.90, "Emerging Adults (18–34)": 0.98},
    },
    "Edmonton": {
        "classic_romance":   {"General Population": 1.02, "Core Classical (F35–64)": 1.15, "Family (Parents w/ kids)": 0.95, "Emerging Adults (18–34)": 0.98},
        "family_classic":    {"General Population": 1.00, "Core Classical (F35–64)": 0.98, "Family (Parents w/ kids)": 1.15, "Emerging Adults (18–34)": 0.95},
        "contemporary":      {"General Population": 1.00, "Core Classical (F35–64)": 0.92, "Family (Parents w/ kids)": 0.95, "Emerging Adults (18–34)": 1.22},
        "pop_ip":            {"General Population": 1.05, "Core Classical (F35–64)": 0.95, "Family (Parents w/ kids)": 1.10, "Emerging Adults (18–34)": 1.10},
        "romantic_tragedy":  {"General Population": 1.02, "Core Classical (F35–64)": 1.10, "Family (Parents w/ kids)": 0.92, "Emerging Adults (18–34)": 1.00},
        "classic_comedy":    {"General Population": 1.02, "Core Classical (F35–64)": 1.00, "Family (Parents w/ kids)": 1.05, "Emerging Adults (18–34)": 1.00},
        "dramatic":          {"General Population": 1.05, "Core Classical (F35–64)": 1.05, "Family (Parents w/ kids)": 0.92, "Emerging Adults (18–34)": 1.00},
    },
}

# Optional global knob: how strongly to apply priors (1.0 = exactly as above)
SEGMENT_PRIOR_STRENGTH = 1.0
def _prior_weights_for(region_key: str, category: str) -> dict:
    pri = SEGMENT_PRIORS.get(region_key, {}).get(category, {})
    if SEGMENT_PRIOR_STRENGTH == 1.0 or not pri:
        return pri or {k: 1.0 for k in SEGMENT_KEYS_IN_ORDER}
    tempered = {}
    p = float(SEGMENT_PRIOR_STRENGTH)
    for k in SEGMENT_KEYS_IN_ORDER:
        w = pri.get(k, 1.0)
        tempered[k] = (w ** p) if w > 0 else 1.0
    return tempered

def _softmax_like(d: dict[str, float], temperature: float = 1.0) -> dict[str, float]:
    """Turn positive weights into a normalized distribution (sums to 1)."""
    if not d: return {}
    vals = {k: max(1e-9, float(v)) for k, v in d.items()}
    logs = {k: math.log(v) / max(1e-6, temperature) for k, v in vals.items()}
    mx = max(logs.values())
    exps = {k: math.exp(v - mx) for k, v in logs.items()}
    Z = sum(exps.values())
    return {k: exps[k] / Z for k in exps}

def _infer_segment_mix_for(category: str, region_key: str, temperature: float = 1.0) -> dict[str, float]:
    pri = _prior_weights_for(region_key, category)
    if not pri:
        pri = {k: 1.0 for k in SEGMENT_KEYS_IN_ORDER}
    return _softmax_like(pri, temperature=temperature)

# --- TEMP: disable program-based mapping (we'll switch to category-based next) ---
CATEGORY_TO_PROGRAM = {}
LA_BY_PROGRAM = {}
# --- CATEGORY-BASED Live Analytics overlays (directly keyed by your app categories) ---
LA_BY_CATEGORY = {
    # pop_ip ≈ Pop Music Ballet
    "pop_ip": {
        "Presale": 12, "FirstDay": 7, "FirstWeek": 5, "WeekOf": 20,
        "Internet": 58, "Mobile": 41, "Phone": 1,
        "Tix_1_2": 73, "Tix_3_4": 22, "Tix_5_8": 5,
        "Premium": 4.5, "LT10mi": 71,
        "Price_Low": 18, "Price_Fair": 32, "Price_Good": 24, "Price_VeryGood": 16, "Price_Best": 10,
    },

    # classic_* and romantic_comedy ≈ Classic Ballet
    "classic_romance": {
        "Presale": 10, "FirstDay": 6, "FirstWeek": 4, "WeekOf": 20,
        "Internet": 55, "Mobile": 44, "Phone": 1,
        "Tix_1_2": 69, "Tix_3_4": 25, "Tix_5_8": 5,
        "Premium": 4.1, "LT10mi": 69,
        "Price_Low": 20, "Price_Fair": 29, "Price_Good": 24, "Price_VeryGood": 16, "Price_Best": 10,
    },
    "classic_comedy": {
        "Presale": 10, "FirstDay": 6, "FirstWeek": 4, "WeekOf": 20,
        "Internet": 55, "Mobile": 44, "Phone": 1,
        "Tix_1_2": 69, "Tix_3_4": 25, "Tix_5_8": 5,
        "Premium": 4.1, "LT10mi": 69,
        "Price_Low": 20, "Price_Fair": 29, "Price_Good": 24, "Price_VeryGood": 16, "Price_Best": 10,
    },
    "romantic_comedy": {
        "Presale": 10, "FirstDay": 6, "FirstWeek": 4, "WeekOf": 20,
        "Internet": 55, "Mobile": 44, "Phone": 1,
        "Tix_1_2": 69, "Tix_3_4": 25, "Tix_5_8": 5,
        "Premium": 4.1, "LT10mi": 69,
        "Price_Low": 20, "Price_Fair": 29, "Price_Good": 24, "Price_VeryGood": 16, "Price_Best": 10,
    },

    # contemporary ≈ Contemporary Mixed Bill
    "contemporary": {
        "Presale": 11, "FirstDay": 6, "FirstWeek": 5, "WeekOf": 21,
        "Internet": 59, "Mobile": 40, "Phone": 1,
        "Tix_1_2": 72, "Tix_3_4": 23, "Tix_5_8": 5,
        "Premium": 3.9, "LT10mi": 70,
        "Price_Low": 18, "Price_Fair": 32, "Price_Good": 25, "Price_VeryGood": 15, "Price_Best": 11,
    },

    # family_classic ≈ Family Ballet
    "family_classic": {
        "Presale": 10, "FirstDay": 6, "FirstWeek": 5, "WeekOf": 19,
        "Internet": 53, "Mobile": 45, "Phone": 1,
        "Tix_1_2": 67, "Tix_3_4": 27, "Tix_5_8": 6,
        "Premium": 4.0, "LT10mi": 66,
        "Price_Low": 21, "Price_Fair": 28, "Price_Good": 24, "Price_VeryGood": 17, "Price_Best": 10,
    },

    # romantic_tragedy ≈ CSNA (Classical Story Narrative - Adult)
    "romantic_tragedy": {
        "Presale": 11, "FirstDay": 6, "FirstWeek": 5, "WeekOf": 20,
        "Internet": 59, "Mobile": 39, "Phone": 2,
        "Tix_1_2": 72, "Tix_3_4": 22, "Tix_5_8": 5,
        "Premium": 4.2, "LT10mi": 71,
        "Price_Low": 18, "Price_Fair": 28, "Price_Good": 27, "Price_VeryGood": 16, "Price_Best": 11,
    },

    # dramatic ≈ Contemporary Narrative (you can switch to Cultural Narrative if preferred)
    "dramatic": {
        "Presale": 12, "FirstDay": 7, "FirstWeek": 5, "WeekOf": 20,
        "Internet": 54, "Mobile": 44, "Phone": 1,
        "Tix_1_2": 72, "Tix_3_4": 23, "Tix_5_8": 5,
        "Premium": 4.3, "LT10mi": 68,
        "Price_Low": 18, "Price_Fair": 31, "Price_Good": 24, "Price_VeryGood": 16, "Price_Best": 11,
    },
}

# --- Add age overlays for one category (test) ---
LA_BY_CATEGORY.setdefault("pop_ip", {}).update({
    "Age": 49.6,
    "Age_18_24": 8,
    "Age_25_34": 5,
    "Age_35_44": 26,
    "Age_45_54": 20,
    "Age_55_64": 26,
    "Age_65_plus": 15,
})

def _la_for_category(cat: str) -> dict:
    """Return the LA overlay dict for a given app category, or empty dict if unknown."""
    return LA_BY_CATEGORY.get(str(cat), {})

def _program_for_category(cat: str) -> Optional[str]:
    return None

def _add_live_analytics_overlays(df_in: pd.DataFrame) -> pd.DataFrame:
    """Join timing/channel/price overlays onto each title row USING LA_BY_CATEGORY directly."""
    df = df_in.copy()

    cols = [
        "LA_EarlyBuyerPct","LA_WeekOfPct",
        "LA_MobilePct","LA_InternetPct","LA_PhonePct",
        "LA_Tix12Pct","LA_Tix34Pct","LA_Tix58Pct",
        "LA_PremiumPct","LA_LocalLT10Pct",
        "LA_PriceHiPct","LA_PriceFlag",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan

    def _overlay_row_from_category(cat: str) -> dict:
        la = _la_for_category(cat)  # pulls from LA_BY_CATEGORY
        if not la:
            return {}

        early = (float(la.get("Presale", 0)) + float(la.get("FirstDay", 0)) + float(la.get("FirstWeek", 0))) / 100.0
        price_hi = (float(la.get("Price_VeryGood", 0)) + float(la.get("Price_Best", 0))) / 100.0

        return {
            "LA_EarlyBuyerPct": early,
            "LA_WeekOfPct": float(la.get("WeekOf", 0)) / 100.0,
            "LA_MobilePct": float(la.get("Mobile", 0)) / 100.0,
            "LA_InternetPct": float(la.get("Internet", 0)) / 100.0,
            "LA_PhonePct": float(la.get("Phone", 0)) / 100.0,
            "LA_Tix12Pct": float(la.get("Tix_1_2", 0)) / 100.0,
            "LA_Tix34Pct": float(la.get("Tix_3_4", 0)) / 100.0,
            "LA_Tix58Pct": float(la.get("Tix_5_8", 0)) / 100.0,
            "LA_PremiumPct": float(la.get("Premium", 0)) / 100.0,
            "LA_LocalLT10Pct": float(la.get("LT10mi", 0)) / 100.0,
            "LA_PriceHiPct": price_hi,
        }

    overlays = df["Category"].map(lambda c: _overlay_row_from_category(str(c)))
    overlays_df = pd.DataFrame(list(overlays)).reindex(df.index)

    # Update df with computed overlay columns
    for c in [c for c in cols if c in overlays_df.columns]:
        df[c] = overlays_df[c]

    # Human-readable price flag
    def _price_flag(p_hi: float) -> str:
        if pd.isna(p_hi): return "n/a"
        return "Elastic" if p_hi < 0.25 else ("Premium-tolerant" if p_hi > 0.30 else "Neutral")

    df["LA_PriceFlag"] = df["LA_PriceHiPct"].apply(_price_flag)
    return df

# -------------------------
# HEURISTICS for OFFLINE estimation of NEW titles
# -------------------------
def infer_gender_and_category(title: str) -> Tuple[str, str]:
    t = title.lower()
    gender = "na"
    female_keys = ["cinderella","sleeping","beauty and the beast","beauty","giselle","swan","widow","alice","juliet","sylphide"]
    male_keys = ["pinocchio","peter pan","don quixote","hunchback","hamlet","frankenstein","romeo","nijinsky"]
    if "romeo" in t and "juliet" in t: gender = "co"
    elif any(k in t for k in female_keys): gender = "female"
    elif any(k in t for k in male_keys): gender = "male"

    if any(k in t for k in ["wizard","peter pan","pinocchio","hansel","frozen","beauty","alice"]):
        cat = "family_classic"
    elif any(k in t for k in ["swan","sleeping","cinderella","giselle","sylphide"]):
        cat = "classic_romance"
    elif any(k in t for k in ["romeo","hunchback","notre dame","hamlet","frankenstein"]):
        cat = "romantic_tragedy"
    elif any(k in t for k in ["don quixote","merry widow"]):
        cat = "classic_comedy"
    elif any(k in t for k in ["contemporary","boyz","ballet boyz","momix","complexions","grimm","nijinsky","shadowland","deviate","phi"]):
        cat = "contemporary"
    elif any(k in t for k in ["taj","tango","harlem","tragically hip","l cohen","leonard cohen"]):
        cat = "pop_ip"
    else:
        cat = "dramatic"
    return gender, cat

def estimate_unknown_title(title: str) -> Dict[str, float | str]:
    gender, category = infer_gender_and_category(title)
    base_df = pd.DataFrame(BASELINES).T
    for k in ["wiki","trends","youtube","spotify"]:
        if k not in base_df.columns:
            base_df[k] = np.nan
    cats = {k: v["category"] for k, v in BASELINES.items()}
    tmp = base_df.copy()
    tmp["category"] = tmp.index.map(lambda x: cats.get(x, "dramatic"))
    cat_df = tmp[tmp["category"] == category]
    med = (cat_df[["wiki","trends","youtube","spotify"]].median()
           if not cat_df.empty else tmp[["wiki","trends","youtube","spotify"]].median())
    gender_adj = {"female": 1.05, "male": 0.98, "co": 1.03, "na": 1.00}[gender]
    est = {k: float(med[k] * gender_adj) for k in ["wiki","trends","youtube","spotify"]}
    for k in est: est[k] = float(max(30.0, min(160.0, est[k])))
    return {"wiki": est["wiki"], "trends": est["trends"], "youtube": est["youtube"], "spotify": est["spotify"],
            "gender": gender, "category": category}

# -------------------------
# OPTIONAL LIVE FETCHERS (only if toggled ON) — with safer YouTube mapping
# -------------------------
WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_PAGEVIEW = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
                 "en.wikipedia/all-access/user/{page}/daily/{start}/{end}")

BALLET_HINTS = ["ballet", "pas", "variation", "act", "scene", "solo", "adagio", "coda"]

def _title_tokens(s: str):
    return re.findall(r"[a-z0-9]+", s.lower())

def _looks_like_our_title(video_title: str, query_title: str) -> bool:
    vt = _title_tokens(video_title)
    qt = _title_tokens(query_title)
    if not vt or not qt:
        return False
    overlap = sum(1 for t in qt if t in vt)
    has_ballet_hint = any(h in vt for h in BALLET_HINTS)
    return (overlap >= max(1, len(qt) // 2)) and has_ballet_hint

def _yt_index_from_views(view_list):
    if not view_list:
        return 0.0
    v = float(np.median(view_list))
    idx = 50.0 + min(90.0, np.log1p(v) * 9.0)
    return float(idx)

def _winsorize_youtube_to_baseline(category: str, yt_value: float) -> float:
    try:
        base_df = pd.DataFrame(BASELINES).T
        base_df["category"] = [v.get("category", "dramatic") for v in BASELINES.values()]
        ref = base_df.loc[base_df["category"] == category, "youtube"].dropna()
        if ref.empty: ref = base_df["youtube"].dropna()
        if ref.empty: return yt_value
        lo = float(np.percentile(ref, 3))
        hi = float(np.percentile(ref, 97))
        return float(np.clip(yt_value, lo, hi))
    except Exception:
        return yt_value

def wiki_search_best_title(query: str) -> Optional[str]:
    try:
        params = {"action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": 5}
        r = requests.get(WIKI_API, params=params, timeout=10)
        if r.status_code != 200: return None
        items = r.json().get("query", {}).get("search", [])
        return items[0]["title"] if items else None
    except Exception:
        return None

def fetch_wikipedia_views_for_page(page_title: str) -> float:
    try:
        end = datetime.utcnow().strftime("%Y%m%d")
        start = (datetime.utcnow() - timedelta(days=365)).strftime("%Y%m%d")
        url = WIKI_PAGEVIEW.format(page=page_title.replace(" ", "_"), start=start, end=end)
        r = requests.get(url, timeout=10)
        if r.status_code != 200: return 0.0
        items = r.json().get("items", [])
        views = [it.get("views", 0) for it in items]
        return (sum(views) / 365.0) if views else 0.0
    except Exception:
        return 0.0

def fetch_live_for_unknown(title: str, yt_key: Optional[str], sp_id: Optional[str], sp_secret: Optional[str]) -> Dict[str, float | str]:
    w_title = wiki_search_best_title(title) or title
    wiki_raw = fetch_wikipedia_views_for_page(w_title)
    wiki_idx = 40.0 + min(110.0, (math.log1p(max(0.0, wiki_raw)) * 20.0))  # ~40..150
    trends_idx = 60.0 + (len(title) % 40)

    yt_idx = 0.0
    if yt_key and build is not None:
        try:
            yt = build("youtube", "v3", developerKey=yt_key)
            q = f"{title} ballet"
            search = yt.search().list(q=q, part="snippet", type="video", maxResults=30, relevanceLanguage="en").execute()
            items = search.get("items", []) or []
            filtered_ids = []
            for it in items:
                vid = it.get("id", {}).get("videoId")
                vtitle = it.get("snippet", {}).get("title", "") or ""
                if vid and _looks_like_our_title(vtitle, title):
                    filtered_ids.append(vid)
            if len(filtered_ids) < 5:
                filtered_ids = [it.get("id", {}).get("videoId") for it in items if it.get("id", {}).get("videoId")]
            views = []
            if filtered_ids:
                stats = yt.videos().list(part="statistics", id=",".join(filtered_ids[:60])).execute()
                for i in stats.get("items", []) or []:
                    vc = i.get("statistics", {}).get("viewCount")
                    if vc is not None:
                        try: views.append(int(vc))
                        except Exception: pass
            yt_idx = _yt_index_from_views(views) if views else 0.0
        except Exception:
            yt_idx = 0.0

    if yt_idx == 0.0:
        yt_idx = 55.0 + (len(title) * 1.2) % 45.0
    yt_idx = float(np.clip(yt_idx, 45.0, 140.0))

    sp_idx = 0.0
    if sp_id and sp_secret and spotipy is not None:
        try:
            auth = SpotifyClientCredentials(client_id=sp_id, client_secret=sp_secret)
            sp = spotipy.Spotify(auth_manager=auth)
            res = sp.search(q=title, type="track,album", limit=10)
            pops = [t.get("popularity", 0) for t in res.get("tracks", {}).get("items", [])]
            sp_idx = float(np.percentile(pops, 80)) if pops else 0.0
        except Exception:
            sp_idx = 0.0
    if sp_idx == 0.0:
        sp_idx = 50.0 + (len(title) * 1.7) % 40.0

    gender, category = infer_gender_and_category(title)
    yt_idx = _winsorize_youtube_to_baseline(category, yt_idx)

    return {"wiki": wiki_idx, "trends": trends_idx, "youtube": yt_idx, "spotify": sp_idx,
            "gender": gender, "category": category}

# -------------------------
# UI — API + OPTIONS + TITLES
# -------------------------
with st.expander("🔑 API Configuration (only used for NEW titles if enabled)"):
    yt_key = st.text_input("YouTube Data API v3 Key", type="password")
    sp_id = st.text_input("Spotify Client ID", type="password")
    sp_secret = st.text_input("Spotify Client Secret", type="password")
    use_live = st.checkbox("Use Live Data for Unknown Titles", value=False)

    st.markdown("""
**Helpful links (universal docs & dashboards):**
- **Spotify**: Web API docs · Getting started (create/manage keys in your Spotify Developer Dashboard).
- **YouTube**: YouTube Data API overview · API reference (create/manage keys in Google Cloud Console → APIs & Services → Credentials).
""")
    st.caption("Keys are created in your own accounts. These links are universal; your keys remain private to you.")

region = st.selectbox("Region", ["Province", "Calgary", "Edmonton"], index=0)
segment = st.selectbox("Audience Segment", list(SEGMENT_MULT.keys()), index=0)
# --- Seasonality control (global) ---
apply_seasonality = st.checkbox("Apply seasonality by month", value=False)
proposed_run_date = None
if apply_seasonality:
    _months = [
        ("January", 1), ("February", 2), ("March", 3), ("April", 4),
        ("May", 5), ("June", 6), ("July", 7), ("August", 8),
        ("September", 9), ("October", 10), ("November", 11), ("December", 12),
    ]
    _month_names = [m[0] for m in _months]
    sel = st.selectbox("Assumed run month (applies seasonality factor)", _month_names, index=2)  # default March
    month_idx = dict(_months)[sel]
    # Neutral mid-month date for factor lookups
    proposed_run_date = datetime(datetime.utcnow().year, month_idx, 15).date()

default_list = list(BASELINES.keys())[:50]
st.markdown("**Titles to score** (one per line). Add NEW titles freely:")
titles_input = st.text_area("Enter titles", value="\n".join(default_list), height=220)
titles = [t.strip() for t in titles_input.splitlines() if t.strip()]

# Fallback so we never end up with zero titles
if not titles:
    titles = list(BASELINES.keys())

# --- BENCHMARK SELECTOR (outside the function, above the Run button) ---
benchmark_title = st.selectbox(
    "Choose Benchmark Title for Normalization",
    options=list(BASELINES.keys()),
    index=0,
    key="benchmark_title"
)

# Fallback so we never end up with zero titles
if not titles:
    titles = list(BASELINES.keys())

# -------------------------
# SCORING
# -------------------------
def calc_scores(entry: Dict[str, float | str], seg_key: str, reg_key: str) -> Tuple[float,float]:
    gender = entry["gender"]; cat = entry["category"]
    fam = entry["wiki"] * 0.55 + entry["trends"] * 0.30 + entry["spotify"] * 0.15
    mot = entry["youtube"] * 0.45 + entry["trends"] * 0.25 + entry["spotify"] * 0.15 + entry["wiki"] * 0.15
    seg = SEGMENT_MULT[seg_key]
    fam *= seg.get(gender,1.0) * seg.get(cat,1.0)
    mot *= seg.get(gender,1.0) * seg.get(cat,1.0)
    fam *= REGION_MULT[reg_key]
    mot *= REGION_MULT[reg_key]
    return fam, mot

# --- Hard-coded ticket priors (lists -> we use median to be robust) ---
TICKET_PRIORS_RAW = {
    "Alice in Wonderland": [11216],
    "All of Us - Tragically Hip": [15488],
    "Away We Go – Mixed Bill": [4649],
    "Ballet BC": [4013],
    "Ballet Boyz": [7401],
    "BJM - Leonard Cohen": [7819],
    "Botero": [5460],
    "Cinderella": [16304],
    "Complexions - Lenny Kravitz": [7096],
    "Dance Theatre of Harlem": [7269],
    "Dangerous Liaisons": [6875],
    "deViate - Mixed Bill": [5144],
    "Diavolo": [10673],
    "Don Quixote": [5650],
    "Dona Peron": [5221],
    "Dracula": [11285],
    "Fiddle & the Drum – Joni Mitchell": [6024],
    "Frankenstein": [10470],
    "Giselle": [9111],
    "Grimm": [6362],
    "Handmaid's Tale": [6842],
    "Hansel & Gretel": [7290],
    "La Sylphide": [5221],
    "Midsummer Night’s Dream": [6587],
    "Momix": [8391],
    "Nijinsky": [4295],
    "Our Canada - Gordon Lightfoot": [10138],
    "Phi – David Bowie": [12336],
    "Shaping Sound": [10208],
    "Sleeping Beauty": [9596.5],
    "Swan Lake": [13157],
    "Taj Express": [10314],
    "Tango Fire": [7251],
    "Trockadero": [5476],
    "Unleashed – Mixed Bill": [5221],
    "Winter Gala": [1321],
    "Wizard of Oz": [8468],
}
def _median(xs):
    xs = sorted([float(x) for x in xs if x is not None])
    if not xs: return None
    n = len(xs); mid = n // 2
    return xs[mid] if n % 2 else (xs[mid-1] + xs[mid]) / 2.0

TICKET_BLEND_WEIGHT = 0.50  # 50% tickets, 50% familiarity/motivation composite

# =========================
# SEASONALITY (build table for Category×Month factors)
# =========================
from datetime import date

# 1) Your historical runs (Title, Start, End)
PAST_RUNS = [
    ("Alice in Wonderland","2017-03-16","2017-03-25"),
    ("All of Us - Tragically Hip","2018-05-02","2018-05-12"),
    ("Away We Go – Mixed Bill","2022-10-27","2022-11-05"),
    ("Ballet BC","2019-01-19","2019-01-23"),
    ("Ballet Boyz","2017-02-16","2017-02-25"),
    ("BJM - Leonard Cohen","2018-09-20","2018-09-22"),
    ("Botero","2023-05-05","2023-05-13"),
    ("Cinderella","2022-04-28","2022-05-14"),
    ("Complexions - Lenny Kravitz","2023-02-10","2023-02-18"),
    ("Dance Theatre of Harlem","2025-02-13","2025-02-22"),
    ("Dangerous Liaisons","2017-10-26","2017-11-04"),
    ("deViate - Mixed Bill","2019-02-15","2019-02-16"),
    ("Diavolo","2020-01-21","2020-01-22"),
    ("Don Quixote","2025-05-01","2025-05-10"),
    ("Dona Peron","2023-09-14","2023-09-23"),
    ("Dracula","2016-10-27","2016-11-05"),
    ("Fiddle and the Drum - Joni Mitchell","2019-05-01","2019-05-11"),
    ("Frankenstein","2019-10-23","2019-11-02"),
    ("Giselle","2023-03-09","2023-03-25"),
    ("Grimm","2024-10-17","2024-10-26"),
    ("Handmaid's Tale","2022-09-14","2022-09-24"),
    ("Hansel & Gretel","2024-03-07","2024-03-23"),
    ("La Sylphide","2024-09-12","2024-09-21"),
    ("Midsummer Night's Dream","2019-03-13","2019-03-16"),
    ("Momix","2018-02-15","2018-02-22"),
    ("Nijinsky","2025-10-16","2025-10-25"),
    ("Nutcracker","2023-12-06","2023-12-24"),
    ("Our Canada - Gordon Lightfoot","2017-05-04","2017-05-13"),
    ("Phi – David Bowie","2022-03-10","2022-03-19"),
    ("Shaping Sound","2018-01-19","2018-01-20"),
    ("Sleeping Beauty","2023-10-26","2023-11-04"),
    ("Swan Lake","2021-10-21","2021-11-07"),
    ("Taj Express","2019-09-12","2019-09-21"),
    ("Tango Fire","2017-09-21","2017-09-28"),
    ("Trockadero","2017-01-12","2017-01-14"),
    ("Unleashed – Mixed Bill","2020-02-13","2020-02-22"),
    ("Winter Gala","2025-01-18","2025-01-25"),
    ("Wizard of Oz","2025-03-13","2025-03-22"),
]

def _to_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()

def _mid_date(a: date, b: date) -> date:
    return a + (b - a) // 2

# 2) Build a runs DataFrame with MidMonth, Category, and TicketMedian
_runs_rows = []
for title, s, e in PAST_RUNS:
    d1, d2 = _to_date(s), _to_date(e)
    mid = _mid_date(d1, d2)
    # Category from BASELINES if known, else infer
    if title in BASELINES:
        cat = BASELINES[title].get("category", infer_gender_and_category(title)[1])
    else:
        cat = infer_gender_and_category(title)[1]
    # Ticket median from your priors (None if we don’t have it)
    aliases = {"Handmaid’s Tale": "Handmaid's Tale"}
    key = aliases.get(title.strip(), title.strip())
    tix_med = _median(TICKET_PRIORS_RAW.get(key, []))
    _runs_rows.append({
        "Title": title,
        "Category": cat,
        "Start": d1,
        "End": d2,
        "MidDate": mid,
        "Month": mid.month,            # 1..12
        "Year": mid.year,
        "TicketMedian": tix_med
    })

RUNS_DF = pd.DataFrame(_runs_rows)
# Fast lookup: title -> mid-run date (or None)
TITLE_TO_MIDDATE = {}
try:
    for _, r in RUNS_DF.iterrows():
        TITLE_TO_MIDDATE[str(r["Title"]).strip()] = r["MidDate"]
except Exception:
    TITLE_TO_MIDDATE = {}

# 3) Compute Category×Month seasonality factors with shrinkage + clipping
# Factor_raw(cat, m) = median(Tickets for cat in m) / median(Tickets for cat overall)
# Then shrink toward 1.0 by n/(n+k) and clip to [MINF, MAXF]

K_SHRINK = 3.0         # shrinkage strength (higher = stronger pull to 1.0 when n is small)
MINF, MAXF = 0.85, 1.25  # safety cap

_season_rows = []
for cat, g_cat in RUNS_DF.groupby("Category"):
    g_cat_hist = g_cat[g_cat["TicketMedian"].notna()].copy()
    if g_cat_hist.empty:
        continue

    cat_overall_med = g_cat_hist["TicketMedian"].median()
    for m, g_cm in g_cat_hist.groupby("Month"):
        vals = g_cm["TicketMedian"].dropna().values
        n = int(len(vals))
        if n == 0 or not cat_overall_med:
            continue

        m_med = float(np.median(vals))
        factor_raw = float(m_med / cat_overall_med)

        # --- Shrink toward 1.0 by sample size ---
        w = n / (n + K_SHRINK)
        factor_shrunk = 1.0 + w * (factor_raw - 1.0)

        # --- Clip extremes ---
        factor_final = float(np.clip(factor_shrunk, MINF, MAXF))

        _season_rows.append({
            "Category": cat,
            "Month": int(m),
            "Factor": factor_final,
            "FactorRaw": factor_raw,
            "N": n
        })

SEASONALITY_DF = (
    pd.DataFrame(_season_rows)
      .sort_values(["Category","Month"])
      .reset_index(drop=True)
)
SEASONALITY_TABLE = {
    (r["Category"], int(r["Month"])): float(r["Factor"])
    for _, r in SEASONALITY_DF.iterrows()
}

# 4) Helper to fetch factor (defaults to 1.0 when unknown)
def seasonality_factor(category: str, when: Optional[date]) -> float:
    """
    Returns multiplicative factor for a given category and date.
    Uses month-of-year. Falls back to 1.0 if we lack data.
    """
    if when is None:
        return 1.0
    key = (str(category), int(when.month))
    return float(SEASONALITY_TABLE.get(key, 1.0))

# --- Helpers for segment propensity (output) ---
def _signal_for_all_segments(entry: Dict[str, float | str], region_key: str) -> dict:
    seg_to_signal = {}
    for seg_key in SEGMENT_KEYS_IN_ORDER:
        fam_raw, mot_raw = calc_scores(entry, seg_key, region_key)
        seg_to_signal[seg_key] = (fam_raw, mot_raw)
    return seg_to_signal

def _normalize_signals_by_benchmark(seg_to_raw: dict, benchmark_entry: dict, region_key: str) -> dict:
    seg_to_indexed_signal = {}
    for seg_key, (fam_raw, mot_raw) in seg_to_raw.items():
        bench_fam_raw, bench_mot_raw = calc_scores(benchmark_entry, seg_key, region_key)
        bf = bench_fam_raw or 1.0
        bm = bench_mot_raw or 1.0
        fam_idx = (fam_raw / bf) * 100.0
        mot_idx = (mot_raw / bm) * 100.0
        seg_to_indexed_signal[seg_key] = (fam_idx + mot_idx) / 2.0
    return seg_to_indexed_signal

# -------------------------
# CORE: compute + store
# -------------------------
def compute_scores_and_store(
    titles,
    segment,
    region,
    use_live,
    yt_key,
    sp_id,
    sp_secret,
    benchmark_title,
    proposed_run_date=None,
):
    """
    - Uses seasonality Category×Month factors computed from PAST_RUNS.
    - De-seasonalizes history before model fit.
    - Re-seasonalizes outputs to the chosen future month (proposed_run_date) if seasonality UI is ON.
    """
    rows = []
    unknown_used_live, unknown_used_est = [], []

    # -------- 1) Base rows: build online-signal + raw calc --------
    for title in titles:
        if title in BASELINES:
            entry = BASELINES[title]; src = "Baseline"
        else:
            if use_live:
                entry = fetch_live_for_unknown(title, yt_key, sp_id, sp_secret)
                src = "Live"; unknown_used_live.append(title)
            else:
                entry = estimate_unknown_title(title)
                src = "Estimated"; unknown_used_est.append(title)

        fam_raw, mot_raw = calc_scores(entry, segment, region)
        rows.append({
            "Title": title, "Region": region, "Segment": segment,
            "Gender": entry["gender"], "Category": entry["category"],
            "FamiliarityRaw": fam_raw, "MotivationRaw": mot_raw,
            "WikiIdx": entry["wiki"], "TrendsIdx": entry["trends"],
            "YouTubeIdx": entry["youtube"], "SpotifyIdx": entry["spotify"],
            "Source": src
        })

    df = pd.DataFrame(rows)
    if df.empty:
        st.error("No titles to score — check the Titles box.")
        return

    # -------- 2) Normalize signals to benchmark --------
    bench_entry = BASELINES[benchmark_title]
    bench_fam_raw, bench_mot_raw = calc_scores(bench_entry, segment, region)
    bench_fam_raw = bench_fam_raw or 1.0
    bench_mot_raw = bench_mot_raw or 1.0

    df["Familiarity"] = (df["FamiliarityRaw"] / bench_fam_raw) * 100.0
    df["Motivation"]  = (df["MotivationRaw"]  / bench_mot_raw)  * 100.0
    st.caption(f"Scores normalized to benchmark: **{benchmark_title}**")

    # -------- 3) Ticket history → medians --------
    TICKET_MEDIANS = {k: _median(v) for k, v in TICKET_PRIORS_RAW.items()}

    # Helper: get the historical mid-run month for a title (if we have one)
    def _hist_month_for_title(title: str) -> Optional[int]:
        d = TITLE_TO_MIDDATE.get(title)
        # 'd' is built using datetime.date objects in RUNS_DF
        if isinstance(d, date):
            return d.month
        return None

    # Seasonality switch
    seasonality_on = proposed_run_date is not None

    # De-seasonalize benchmark
    bench_cat = BASELINES[benchmark_title]["category"]
    bench_hist_factor = seasonality_factor(bench_cat, TITLE_TO_MIDDATE.get(benchmark_title)) if seasonality_on else 1.0
    bench_med_hist = TICKET_MEDIANS.get(benchmark_title, None)
    bench_med_hist = float(bench_med_hist) if bench_med_hist else 1.0
    bench_med_deseason = bench_med_hist / (bench_hist_factor if bench_hist_factor else 1.0)

    # Helper: de-seasonalize a title’s median (if exists)
    aliases = {"Handmaid’s Tale": "Handmaid's Tale"}
    def _deseason_title_median(t: str, cat: str):
        key = aliases.get(t.strip(), t.strip())
        med = TICKET_MEDIANS.get(key)
        if med is None:
            return None, None, None  # no history
        hist_date = TITLE_TO_MIDDATE.get(t.strip())
        factor = seasonality_factor(cat, hist_date) if seasonality_on else 1.0
        return float(med), float(med) / (factor if factor else 1.0), factor

    # -------- 4) TicketIndex (de-seasonalized) vs de-seasonalized benchmark --------
    med_list, idx_list, hist_factor_list = [], [], []
    for _, r in df.iterrows():
        title = r["Title"]; cat = r["Category"]
        med, med_deseason, hist_factor = _deseason_title_median(title, cat)
        med_list.append(med)
        hist_factor_list.append(hist_factor if hist_factor is not None else np.nan)
        if med_deseason is None:
            idx_list.append(None)
        else:
            idx_list.append((med_deseason / (bench_med_deseason or 1.0)) * 100.0)

    df["TicketMedian"] = med_list
    df["TicketIndex_DeSeason"] = idx_list
    df["HistSeasonalityFactor"] = hist_factor_list

    # -------- 5) Signal-only and model fit (on de-seasonalized indices) --------
    df["SignalOnly"] = df[["Familiarity", "Motivation"]].mean(axis=1)
    df_known = df[pd.notna(df["TicketIndex_DeSeason"])].copy()

    def _fit_overall_and_by_category(df_known_in: pd.DataFrame):
        overall = None
        if len(df_known_in) >= 5:
            x = df_known_in["SignalOnly"].values
            y = df_known_in["TicketIndex_DeSeason"].values
            a, b = np.polyfit(x, y, 1)
            overall = (float(a), float(b))
        cat_coefs = {}
        for cat, g in df_known_in.groupby("Category"):
            if len(g) >= 3:
                xs = g["SignalOnly"].values
                ys = g["TicketIndex_DeSeason"].values
                a, b = np.polyfit(xs, ys, 1)
                cat_coefs[cat] = (float(a), float(b))
        return overall, cat_coefs

    overall_coef, cat_coefs = _fit_overall_and_by_category(df_known)

    # -------- 6) Impute de-seasonalized TicketIndex where history is missing --------
    def _predict_ticket_index_deseason(signal_only: float, category: str) -> tuple[float, str]:
        if category in cat_coefs:
            a, b = cat_coefs[category]; src = "Category model"
        elif overall_coef is not None:
            a, b = overall_coef; src = "Overall model"
        else:
            return np.nan, "Not enough data"
        pred = a * signal_only + b
        pred = float(np.clip(pred, 20.0, 180.0))
        return pred, src

    imputed_vals, imputed_srcs = [], []
    for _, r in df.iterrows():
        if pd.notna(r["TicketIndex_DeSeason"]):
            imputed_vals.append(r["TicketIndex_DeSeason"]); imputed_srcs.append("History")
        else:
            pred, src = _predict_ticket_index_deseason(r["SignalOnly"], r["Category"])
            imputed_vals.append(pred); imputed_srcs.append(src)

    df["TicketIndex_DeSeason_Used"] = imputed_vals
    df["TicketIndexSource"] = imputed_srcs

    # -------- 7) Re-seasonalize to FUTURE month (proposed_run_date) for the index --------
    def _future_factor(cat: str):
        return seasonality_factor(cat, proposed_run_date) if seasonality_on else 1.0

    df["FutureSeasonalityFactor"] = df["Category"].map(_future_factor).astype(float)
    df["EffectiveTicketIndex"] = df["TicketIndex_DeSeason_Used"] * df["FutureSeasonalityFactor"]

    # -------- 8) Composite: blend signals with (season-adjusted) TicketIndex --------
    tickets_component = np.where(
        df["TicketIndexSource"].eq("Not enough data"),
        df["SignalOnly"],
        df["EffectiveTicketIndex"]
    )
    TICKET_BLEND_WEIGHT = 0.50
    df["Composite"] = (1.0 - TICKET_BLEND_WEIGHT) * df["SignalOnly"] + TICKET_BLEND_WEIGHT * tickets_component

    # -------- 9) EstimatedTickets using season-adjusted benchmark for FUTURE month --------
    # ✅ benchmark stays de-seasonalized; only the title gets future factor
    bench_med_future = bench_med_deseason
    df["EstimatedTickets"] = ((df["EffectiveTicketIndex"] / 100.0) * (bench_med_future or 1.0)).round(0)

    # -------- 10) Live Analytics overlays --------
    df = _add_live_analytics_overlays(df)

    # -------- 11) Segment propensity & tickets by segment --------
    bench_entry_for_mix = BASELINES[benchmark_title]
    prim_list, sec_list = [], []
    mix_gp, mix_core, mix_family, mix_ea = [], [], [], []
    seg_gp_tix, seg_core_tix, seg_family_tix, seg_ea_tix = [], [], [], []

    for _, r in df.iterrows():
        entry_r = {
            "wiki": float(r["WikiIdx"]),
            "trends": float(r["TrendsIdx"]),
            "youtube": float(r["YouTubeIdx"]),
            "spotify": float(r["SpotifyIdx"]),
            "gender": r["Gender"],
            "category": r["Category"],
        }

        seg_to_raw = _signal_for_all_segments(entry_r, region)
        seg_to_idx = _normalize_signals_by_benchmark(seg_to_raw, bench_entry_for_mix, region)

        pri = _prior_weights_for(region, r["Category"])
        combined = {k: max(1e-9, float(pri.get(k, 1.0)) * float(seg_to_idx.get(k, 0.0)))
                    for k in SEGMENT_KEYS_IN_ORDER}
        total = sum(combined.values()) or 1.0
        shares = {k: combined[k] / total for k in SEGMENT_KEYS_IN_ORDER}

        ordered = sorted(shares.items(), key=lambda kv: kv[1], reverse=True)
        primary = ordered[0][0]
        secondary = ordered[1][0] if len(ordered) > 1 else ""
        prim_list.append(primary); sec_list.append(secondary)

        mix_gp.append(shares["General Population"])
        mix_core.append(shares["Core Classical (F35–64)"])
        mix_family.append(shares["Family (Parents w/ kids)"])
        mix_ea.append(shares["Emerging Adults (18–34)"])

        est = float(r["EstimatedTickets"] or 0.0)
        seg_gp_tix.append(round(est * shares["General Population"]))
        seg_core_tix.append(round(est * shares["Core Classical (F35–64)"]))
        seg_family_tix.append(round(est * shares["Family (Parents w/ kids)"]))
        seg_ea_tix.append(round(est * shares["Emerging Adults (18–34)"]))

    # --- Seasonality meta for display/CSV ---
    df["SeasonalityApplied"] = bool(seasonality_on)
    df["SeasonalityMonthUsed"] = int(proposed_run_date.month) if seasonality_on else np.nan  # numeric (1–12) if ON, else NaN
    df["RunMonth"] = proposed_run_date.strftime("%B") if seasonality_on else ""              # friendly month name (e.g., "May")
    
    df["PredictedPrimarySegment"] = prim_list
    df["PredictedSecondarySegment"] = sec_list
    df["Mix_GP"] = mix_gp; df["Mix_Core"] = mix_core; df["Mix_Family"] = mix_family; df["Mix_EA"] = mix_ea
    df["Seg_GP_Tickets"] = seg_gp_tix; df["Seg_Core_Tickets"] = seg_core_tix
    df["Seg_Family_Tickets"] = seg_family_tix; df["Seg_EA_Tickets"] = seg_ea_tix

    # -------- 12) Transparency + stash --------
    df["SeasonalityApplied"] = bool(seasonality_on)
    df["SeasonalityMonthUsed"] = int(proposed_run_date.month) if seasonality_on else np.nan

    st.session_state["results"] = {
        "df": df,
        "benchmark": benchmark_title,
        "segment": segment,
        "region": region,
        "unknown_est": unknown_used_est,
        "unknown_live": unknown_used_live,
    }

import re
import pandas as pd

# =========================
# LIVE ANALYTICS: FULL TABLE (deep)
# =========================

LA_DEEP_BY_CATEGORY = {
    "pop_ip": {
        "DA_Customers": 15861,
        "DA_CustomersWithLiveEvents": 13813,
        "DA_CustomersWithDemo_CAN": 14995,
        "CES_Gender_Male_pct": 44,
        "CES_Gender_Female_pct": 56,
        "CES_Age_Mean": 49.6,
        "CES_Age_18_24_pct": 8,
        "CES_Age_25_34_pct": 5,
        "CES_Age_35_44_pct": 26,
        "CES_Age_45_54_pct": 20,
        "CES_Age_55_64_pct": 26,
        "CES_Age_65_plus_pct": 15,
        "CEP_NeverPurchased_pct": 25.9,
        "CEP_1Event_pct": 43.2,
        "CEP_2_3Events_pct": 22.0,
        "CEP_4_5Events_pct": 5.7,
        "CEP_6_10Events_pct": 2.8,
        "CEP_11plusEvents_pct": 0.3,
        "CESpend_le_250_pct": 64.7,
        "CESpend_251_500_pct": 19.8,
        "CESpend_501_1000_pct": 11.2,
        "CESpend_1001_2000_pct": 3.6,
        "CESpend_2001plus_pct": 0.6,
        "SOW_ClientEvents_pct": 26,
        "SOW_NonClientEvents_pct": 74,
        "LS_ClientEvents_usd": 268,
        "LS_NonClientEvents_usd": 2207,
        "TPE_ClientEvents": 2.3,
        "TPE_NonClientEvents": 2.6,
        "SPE_ClientEvents_usd": 187,
        "SPE_NonClientEvents_usd": 209,
        "ATP_ClientEvents_usd": 83,
        "ATP_NonClientEvents_usd": 84,
        "DIM_Generation_Millennials_pct": 24,
        "DIM_Generation_X_pct": 34,
        "DIM_Generation_Babyboomers_pct": 31,
        "DIM_Generation_Z_pct": 5,
        "DIM_Occ_Professionals_pct": 64,
        "DIM_Occ_SelfEmployed_pct": 0,
        "DIM_Occ_Retired_pct": 3,
        "DIM_Occ_Students_pct": 0,
        "DIM_Household_WorkingMoms_pct": 8,
        "DIM_Financial_Affluent_pct": 9,
        "DIM_Live_Major_Concerts_pct": 47,
        "DIM_Live_Major_Arts_pct": 76,
        "DIM_Live_Major_Sports_pct": 27,
        "DIM_Live_Major_Family_pct": 7,
        "DIM_Live_Major_MultiCategories_pct": 45,
        "DIM_Live_Freq_ActiveBuyers_pct": 60,
        "DIM_Live_Freq_RepeatBuyers_pct": 67,
        "DIM_Live_Timing_EarlyBuyers_pct": 37,
        "DIM_Live_Timing_LateBuyers_pct": 38,
        "DIM_Live_Product_PremiumBuyers_pct": 18,
        "DIM_Live_Product_AncillaryUpsellBuyers_pct": 2,
        "DIM_Live_Spend_HighSpenders_gt1k_pct": 5,
        "DIM_Live_Distance_Travelers_gt500mi_pct": 14,
    },
    "classic_romance": {
        "DA_Customers": 15294,
        "DA_CustomersWithLiveEvents": 11881,
        "DA_CustomersWithDemo_CAN": 12832,
        "CES_Gender_Male_pct": 47,
        "CES_Gender_Female_pct": 53,
        "CES_Age_Mean": 46.5,
        "CES_Age_18_24_pct": 14,
        "CES_Age_25_34_pct": 5,
        "CES_Age_35_44_pct": 23,
        "CES_Age_45_54_pct": 27,
        "CES_Age_55_64_pct": 19,
        "CES_Age_65_plus_pct": 13,
        "CEP_NeverPurchased_pct": 23.2,
        "CEP_1Event_pct": 39.1,
        "CEP_2_3Events_pct": 26.1,
        "CEP_4_5Events_pct": 7.3,
        "CEP_6_10Events_pct": 3.9,
        "CEP_11plusEvents_pct": 0.4,
        "CESpend_le_250_pct": 58.4,
        "CESpend_251_500_pct": 21.4,
        "CESpend_501_1000_pct": 13.8,
        "CESpend_1001_2000_pct": 5.4,
        "CESpend_2001plus_pct": 1.1,
        "SOW_ClientEvents_pct": 35,
        "SOW_NonClientEvents_pct": 65,
        "LS_ClientEvents_usd": 325,
        "LS_NonClientEvents_usd": 1832,
        "TPE_ClientEvents": 2.4,
        "TPE_NonClientEvents": 2.6,
        "SPE_ClientEvents_usd": 195,
        "SPE_NonClientEvents_usd": 212,
        "ATP_ClientEvents_usd": 82,
        "ATP_NonClientEvents_usd": 84,
        "DIM_Generation_Millennials_pct": 29,
        "DIM_Generation_X_pct": 37,
        "DIM_Generation_Babyboomers_pct": 26,
        "DIM_Generation_Z_pct": 5,
        "DIM_Occ_Professionals_pct": 83,
        "DIM_Occ_SelfEmployed_pct": 0,
        "DIM_Occ_Retired_pct": 0,
        "DIM_Occ_Students_pct": 0,
        "DIM_Household_WorkingMoms_pct": 13,
        "DIM_Financial_Affluent_pct": 9,
        "DIM_Live_Major_Concerts_pct": 38,
        "DIM_Live_Major_Arts_pct": 82,
        "DIM_Live_Major_Sports_pct": 25,
        "DIM_Live_Major_Family_pct": 8,
        "DIM_Live_Major_MultiCategories_pct": 42,
        "DIM_Live_Freq_ActiveBuyers_pct": 66,
        "DIM_Live_Freq_RepeatBuyers_pct": 66,
        "DIM_Live_Timing_EarlyBuyers_pct": 32,
        "DIM_Live_Timing_LateBuyers_pct": 38,
        "DIM_Live_Product_PremiumBuyers_pct": 18,
        "DIM_Live_Product_AncillaryUpsellBuyers_pct": 2,
        "DIM_Live_Spend_HighSpenders_gt1k_pct": 4,
        "DIM_Live_Distance_Travelers_gt500mi_pct": 12,
    },
    "classic_comedy": {
        "DA_Customers": 10402,
        "DA_CustomersWithLiveEvents": 8896,
        "DA_CustomersWithDemo_CAN": 9912,
        "CES_Gender_Male_pct": 46,
        "CES_Gender_Female_pct": 54,
        "CES_Age_Mean": 51.7,
        "CES_Age_18_24_pct": 6,
        "CES_Age_25_34_pct": 12,
        "CES_Age_35_44_pct": 12,
        "CES_Age_45_54_pct": 24,
        "CES_Age_55_64_pct": 25,
        "CES_Age_65_plus_pct": 22,
        "CEP_NeverPurchased_pct": 28.0,
        "CEP_1Event_pct": 38.2,
        "CEP_2_3Events_pct": 22.3,
        "CEP_4_5Events_pct": 7.1,
        "CEP_6_10Events_pct": 4.0,
        "CEP_11plusEvents_pct": 0.5,
        "CESpend_le_250_pct": 66.6,
        "CESpend_251_500_pct": 17.0,
        "CESpend_501_1000_pct": 11.1,
        "CESpend_1001_2000_pct": 4.5,
        "CESpend_2001plus_pct": 0.8,
        "SOW_ClientEvents_pct": 28,
        "SOW_NonClientEvents_pct": 72,
        "LS_ClientEvents_usd": 273,
        "LS_NonClientEvents_usd": 1916,
        "TPE_ClientEvents": 2.3,
        "TPE_NonClientEvents": 2.6,
        "SPE_ClientEvents_usd": 170,
        "SPE_NonClientEvents_usd": 206,
        "ATP_ClientEvents_usd": 74,
        "ATP_NonClientEvents_usd": 82,
        "DIM_Generation_Millennials_pct": 22,
        "DIM_Generation_X_pct": 36,
        "DIM_Generation_Babyboomers_pct": 31,
        "DIM_Generation_Z_pct": 6,
        "DIM_Occ_Professionals_pct": 55,
        "DIM_Occ_SelfEmployed_pct": 0,
        "DIM_Occ_Retired_pct": 5,
        "DIM_Occ_Students_pct": 5,
        "DIM_Household_WorkingMoms_pct": 3,
        "DIM_Financial_Affluent_pct": 9,
        "DIM_Live_Major_Concerts_pct": 43,
        "DIM_Live_Major_Arts_pct": 76,
        "DIM_Live_Major_Sports_pct": 26,
        "DIM_Live_Major_Family_pct": 7,
        "DIM_Live_Major_MultiCategories_pct": 43,
        "DIM_Live_Freq_ActiveBuyers_pct": 56,
        "DIM_Live_Freq_RepeatBuyers_pct": 67,
        "DIM_Live_Timing_EarlyBuyers_pct": 34,
        "DIM_Live_Timing_LateBuyers_pct": 39,
        "DIM_Live_Product_PremiumBuyers_pct": 19,
        "DIM_Live_Product_AncillaryUpsellBuyers_pct": 2,
        "DIM_Live_Spend_HighSpenders_gt1k_pct": 4,
        "DIM_Live_Distance_Travelers_gt500mi_pct": 13,
    },
    "contemporary": {
        "DA_Customers": 14987,
        "DA_CustomersWithLiveEvents": 13051,
        "DA_CustomersWithDemo_CAN": 13872,
        "CES_Gender_Male_pct": 56,
        "CES_Gender_Female_pct": 44,
        "CES_Age_Mean": 47.2,
        "CES_Age_18_24_pct": 10,
        "CES_Age_25_34_pct": 10,
        "CES_Age_35_44_pct": 27,
        "CES_Age_45_54_pct": 13,
        "CES_Age_55_64_pct": 31,
        "CES_Age_65_plus_pct": 9,
        "CEP_NeverPurchased_pct": 24.9,
        "CEP_1Event_pct": 43.1,
        "CEP_2_3Events_pct": 22.2,
        "CEP_4_5Events_pct": 6.2,
        "CEP_6_10Events_pct": 3.2,
        "CEP_11plusEvents_pct": 0.4,
        "CESpend_le_250_pct": 64.5,
        "CESpend_251_500_pct": 18.9,
        "CESpend_501_1000_pct": 11.4,
        "CESpend_1001_2000_pct": 4.5,
        "CESpend_2001plus_pct": 0.8,
        "SOW_ClientEvents_pct": 31,
        "SOW_NonClientEvents_pct": 69,
        "LS_ClientEvents_usd": 285,
        "LS_NonClientEvents_usd": 1875,
        "TPE_ClientEvents": 2.5,
        "TPE_NonClientEvents": 2.7,
        "SPE_ClientEvents_usd": 188,
        "SPE_NonClientEvents_usd": 213,
        "ATP_ClientEvents_usd": 77,
        "ATP_NonClientEvents_usd": 84,
        "DIM_Generation_Millennials_pct": 33,
        "DIM_Generation_X_pct": 35,
        "DIM_Generation_Babyboomers_pct": 27,
        "DIM_Generation_Z_pct": 2,
        "DIM_Occ_Professionals_pct": 62,
        "DIM_Occ_SelfEmployed_pct": 0,
        "DIM_Occ_Retired_pct": 0,
        "DIM_Occ_Students_pct": 4,
        "DIM_Household_WorkingMoms_pct": 8,
        "DIM_Financial_Affluent_pct": 8,
        "DIM_Live_Major_Concerts_pct": 40,
        "DIM_Live_Major_Arts_pct": 78,
        "DIM_Live_Major_Sports_pct": 28,
        "DIM_Live_Major_Family_pct": 10,
        "DIM_Live_Major_MultiCategories_pct": 44,
        "DIM_Live_Freq_ActiveBuyers_pct": 61,
        "DIM_Live_Freq_RepeatBuyers_pct": 66,
        "DIM_Live_Timing_EarlyBuyers_pct": 34,
        "DIM_Live_Timing_LateBuyers_pct": 36,
        "DIM_Live_Product_PremiumBuyers_pct": 18,
        "DIM_Live_Product_AncillaryUpsellBuyers_pct": 2,
        "DIM_Live_Spend_HighSpenders_gt1k_pct": 5,
        "DIM_Live_Distance_Travelers_gt500mi_pct": 12,
    },
    "family_classic": {
        "DA_Customers": 5800,
        "DA_CustomersWithLiveEvents": 4878,
        "DA_CustomersWithDemo_CAN": 5605,
        "CES_Gender_Male_pct": 42,
        "CES_Gender_Female_pct": 58,
        "CES_Age_Mean": 34.0,
        "CES_Age_18_24_pct": 36,
        "CES_Age_25_34_pct": 21,
        "CES_Age_35_44_pct": 21,
        "CES_Age_45_54_pct": 7,
        "CES_Age_55_64_pct": 14,
        "CES_Age_65_plus_pct": 0,
        "CEP_NeverPurchased_pct": 33.5,
        "CEP_1Event_pct": 33.0,
        "CEP_2_3Events_pct": 21.4,
        "CEP_4_5Events_pct": 6.7,
        "CEP_6_10Events_pct": 4.6,
        "CEP_11plusEvents_pct": 0.8,
        "CESpend_le_250_pct": 67.7,
        "CESpend_251_500_pct": 15.7,
        "CESpend_501_1000_pct": 10.7,
        "CESpend_1001_2000_pct": 4.9,
        "CESpend_2001plus_pct": 1.1,
        "SOW_ClientEvents_pct": 27,
        "SOW_NonClientEvents_pct": 73,
        "LS_ClientEvents_usd": 274,
        "LS_NonClientEvents_usd": 2041,
        "TPE_ClientEvents": 2.3,
        "TPE_NonClientEvents": 2.6,
        "SPE_ClientEvents_usd": 170,
        "SPE_NonClientEvents_usd": 209,
        "ATP_ClientEvents_usd": 75,
        "ATP_NonClientEvents_usd": 84,
        "DIM_Generation_Millennials_pct": 40,
        "DIM_Generation_X_pct": 20,
        "DIM_Generation_Babyboomers_pct": 16,
        "DIM_Generation_Z_pct": 20,
        "DIM_Occ_Professionals_pct": 44,
        "DIM_Occ_SelfEmployed_pct": 0,
        "DIM_Occ_Retired_pct": 0,
        "DIM_Occ_Students_pct": 11,
        "DIM_Household_WorkingMoms_pct": 0,
        "DIM_Financial_Affluent_pct": 10,
        "DIM_Live_Major_Concerts_pct": 42,
        "DIM_Live_Major_Arts_pct": 75,
        "DIM_Live_Major_Sports_pct": 26,
        "DIM_Live_Major_Family_pct": 7,
        "DIM_Live_Major_MultiCategories_pct": 40,
        "DIM_Live_Freq_ActiveBuyers_pct": 50,
        "DIM_Live_Freq_RepeatBuyers_pct": 65,
        "DIM_Live_Timing_EarlyBuyers_pct": 33,
        "DIM_Live_Timing_LateBuyers_pct": 37,
        "DIM_Live_Product_PremiumBuyers_pct": 19,
        "DIM_Live_Product_AncillaryUpsellBuyers_pct": 2,
        "DIM_Live_Spend_HighSpenders_gt1k_pct": 4,
        "DIM_Live_Distance_Travelers_gt500mi_pct": 12,
    },
    "romantic_tragedy": {
        "DA_Customers": 7781,
        "DA_CustomersWithLiveEvents": 6661,
        "DA_CustomersWithDemo_CAN": 7265,
        "CES_Gender_Male_pct": 0,
        "CES_Gender_Female_pct": 0,
        "CES_Age_Mean": 47.9,
        "CES_Age_18_24_pct": 14,
        "CES_Age_25_34_pct": 14,
        "CES_Age_35_44_pct": 23,
        "CES_Age_45_54_pct": 9,
        "CES_Age_55_64_pct": 15,
        "CES_Age_65_plus_pct": 26,
        "CEP_NeverPurchased_pct": 28.6,
        "CEP_1Event_pct": 40.1,
        "CEP_2_3Events_pct": 21.2,
        "CEP_4_5Events_pct": 5.8,
        "CEP_6_10Events_pct": 3.7,
        "CEP_11plusEvents_pct": 0.5,
        "CESpend_le_250_pct": 65.3,
        "CESpend_251_500_pct": 18.4,
        "CESpend_501_1000_pct": 11.1,
        "CESpend_1001_2000_pct": 4.2,
        "CESpend_2001plus_pct": 0.9,
        "SOW_ClientEvents_pct": 27,
        "SOW_NonClientEvents_pct": 73,
        "LS_ClientEvents_usd": 277,
        "LS_NonClientEvents_usd": 2247,
        "TPE_ClientEvents": 2.3,
        "TPE_NonClientEvents": 2.6,
        "SPE_ClientEvents_usd": 186,
        "SPE_NonClientEvents_usd": 213,
        "ATP_ClientEvents_usd": 81,
        "ATP_NonClientEvents_usd": 84,
        "DIM_Generation_Millennials_pct": 29,
        "DIM_Generation_X_pct": 28,
        "DIM_Generation_Babyboomers_pct": 29,
        "DIM_Generation_Z_pct": 9,
        "DIM_Occ_Professionals_pct": 49,
        "DIM_Occ_SelfEmployed_pct": 0,
        "DIM_Occ_Retired_pct": 4,
        "DIM_Occ_Students_pct": 15,
        "DIM_Household_WorkingMoms_pct": 8,
        "DIM_Financial_Affluent_pct": 10,
        "DIM_Live_Major_Concerts_pct": 43,
        "DIM_Live_Major_Arts_pct": 77,
        "DIM_Live_Major_Sports_pct": 26,
        "DIM_Live_Major_Family_pct": 8,
        "DIM_Live_Major_MultiCategories_pct": 43,
        "DIM_Live_Freq_ActiveBuyers_pct": 64,
        "DIM_Live_Freq_RepeatBuyers_pct": 67,
        "DIM_Live_Timing_EarlyBuyers_pct": 35,
        "DIM_Live_Timing_LateBuyers_pct": 38,
        "DIM_Live_Product_PremiumBuyers_pct": 18,
        "DIM_Live_Product_AncillaryUpsellBuyers_pct": 2,
        "DIM_Live_Spend_HighSpenders_gt1k_pct": 5,
        "DIM_Live_Distance_Travelers_gt500mi_pct": 14,
    },
}

# Canonical schema/order for export — each tuple: (column_name_in_csv, deep_key)
LA_DEEP_SCHEMA = [
    ("LA_DA_Customers", "DA_Customers"),
    ("LA_DA_CustomersWithLiveEvents", "DA_CustomersWithLiveEvents"),
    ("LA_DA_CustomersWithDemo_CAN", "DA_CustomersWithDemo_CAN"),
    ("LA_CES_Gender_Male_pct", "CES_Gender_Male_pct"),
    ("LA_CES_Gender_Female_pct", "CES_Gender_Female_pct"),
    ("LA_CES_Age_Mean", "CES_Age_Mean"),
    ("LA_CES_Age_18_24_pct", "CES_Age_18_24_pct"),
    ("LA_CES_Age_25_34_pct", "CES_Age_25_34_pct"),
    ("LA_CES_Age_35_44_pct", "CES_Age_35_44_pct"),
    ("LA_CES_Age_45_54_pct", "CES_Age_45_54_pct"),
    ("LA_CES_Age_55_64_pct", "CES_Age_55_64_pct"),
    ("LA_CES_Age_65_plus_pct", "CES_Age_65_plus_pct"),
    ("LA_CEP_NeverPurchased_pct", "CEP_NeverPurchased_pct"),
    ("LA_CEP_1Event_pct", "CEP_1Event_pct"),
    ("LA_CEP_2_3Events_pct", "CEP_2_3Events_pct"),
    ("LA_CEP_4_5Events_pct", "CEP_4_5Events_pct"),
    ("LA_CEP_6_10Events_pct", "CEP_6_10Events_pct"),
    ("LA_CEP_11plusEvents_pct", "CEP_11plusEvents_pct"),
    ("LA_CESpend_le_250_pct", "CESpend_le_250_pct"),
    ("LA_CESpend_251_500_pct", "CESpend_251_500_pct"),
    ("LA_CESpend_501_1000_pct", "CESpend_501_1000_pct"),
    ("LA_CESpend_1001_2000_pct", "CESpend_1001_2000_pct"),
    ("LA_CESpend_2001plus_pct", "CESpend_2001plus_pct"),
    ("LA_SOW_ClientEvents_pct", "SOW_ClientEvents_pct"),
    ("LA_SOW_NonClientEvents_pct", "SOW_NonClientEvents_pct"),
    ("LA_LS_ClientEvents_usd", "LS_ClientEvents_usd"),
    ("LA_LS_NonClientEvents_usd", "LS_NonClientEvents_usd"),
    ("LA_TPE_ClientEvents", "TPE_ClientEvents"),
    ("LA_TPE_NonClientEvents", "TPE_NonClientEvents"),
    ("LA_SPE_ClientEvents_usd", "SPE_ClientEvents_usd"),
    ("LA_SPE_NonClientEvents_usd", "SPE_NonClientEvents_usd"),
    ("LA_ATP_ClientEvents_usd", "ATP_ClientEvents_usd"),
    ("LA_ATP_NonClientEvents_usd", "ATP_NonClientEvents_usd"),
    ("LA_DIM_Generation_Millennials_pct", "DIM_Generation_Millennials_pct"),
    ("LA_DIM_Generation_X_pct", "DIM_Generation_X_pct"),
    ("LA_DIM_Generation_Babyboomers_pct", "DIM_Generation_Babyboomers_pct"),
    ("LA_DIM_Generation_Z_pct", "DIM_Generation_Z_pct"),
    ("LA_DIM_Occ_Professionals_pct", "DIM_Occ_Professionals_pct"),
    ("LA_DIM_Occ_SelfEmployed_pct", "DIM_Occ_SelfEmployed_pct"),
    ("LA_DIM_Occ_Retired_pct", "DIM_Occ_Retired_pct"),
    ("LA_DIM_Occ_Students_pct", "DIM_Occ_Students_pct"),
    ("LA_DIM_Household_WorkingMoms_pct", "DIM_Household_WorkingMoms_pct"),
    ("LA_DIM_Financial_Affluent_pct", "DIM_Financial_Affluent_pct"),
    ("LA_DIM_Live_Major_Concerts_pct", "DIM_Live_Major_Concerts_pct"),
    ("LA_DIM_Live_Major_Arts_pct", "DIM_Live_Major_Arts_pct"),
    ("LA_DIM_Live_Major_Sports_pct", "DIM_Live_Major_Sports_pct"),
    ("LA_DIM_Live_Major_Family_pct", "DIM_Live_Major_Family_pct"),
    ("LA_DIM_Live_Major_MultiCategories_pct", "DIM_Live_Major_MultiCategories_pct"),
    ("LA_DIM_Live_Freq_ActiveBuyers_pct", "DIM_Live_Freq_ActiveBuyers_pct"),
    ("LA_DIM_Live_Freq_RepeatBuyers_pct", "DIM_Live_Freq_RepeatBuyers_pct"),
    ("LA_DIM_Live_Timing_EarlyBuyers_pct", "DIM_Live_Timing_EarlyBuyers_pct"),
    ("LA_DIM_Live_Timing_LateBuyers_pct", "DIM_Live_Timing_LateBuyers_pct"),
    ("LA_DIM_Live_Product_PremiumBuyers_pct", "DIM_Live_Product_PremiumBuyers_pct"),
    ("LA_DIM_Live_Product_AncillaryUpsellBuyers_pct", "DIM_Live_Product_AncillaryUpsellBuyers_pct"),
    ("LA_DIM_Live_Spend_HighSpenders_gt1k_pct", "DIM_Live_Spend_HighSpenders_gt1k_pct"),
    ("LA_DIM_Live_Distance_Travelers_gt500mi_pct", "DIM_Live_Distance_Travelers_gt500mi_pct"),
]

def attach_la_report_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach deep LiveAnalytics columns (LA_DEEP_SCHEMA) by Category.
    Keeps any earlier LA_* columns you've already added (timing/channels/price).
    """
    df_out = df.copy()
    for col, key in LA_DEEP_SCHEMA:
        vals = []
        for _, r in df.iterrows():
            cat = str(r.get("Category"))
            deep = LA_DEEP_BY_CATEGORY.get(cat, {})
            v = deep.get(key, np.nan)
            vals.append(v)
        df_out[col] = vals
    return df_out

# -------------------------
# RENDER
# -------------------------
def render_results():
    import calendar

    R = st.session_state["results"]
    if not R:
        return

    df = R["df"].copy()
    benchmark_title = R["benchmark"]
    segment = R["segment"]
    region = R["region"]

    # Notices about data sources
    if R.get("unknown_est"):
        st.info("Estimated (offline) for new titles: " + ", ".join(R["unknown_est"]))
    if R.get("unknown_live"):
        st.success("Used LIVE data for new titles: " + ", ".join(R["unknown_live"]))

    # TicketIndex source status
    if "TicketIndexSource" in df.columns:
        src_counts = (
            df["TicketIndexSource"]
            .value_counts(dropna=False)
            .reindex(["History", "Category model", "Overall model", "Not enough data"])
            .fillna(0)
            .astype(int)
        )
        st.caption(
            f"TicketIndex source — History: {int(src_counts.get('History',0))} · "
            f"Category model: {int(src_counts.get('Category model',0))} · "
            f"Overall model: {int(src_counts.get('Overall model',0))} · "
            f"Not enough data: {int(src_counts.get('Not enough data',0))}"
        )

    # Seasonality status banner
    if "SeasonalityApplied" in df.columns and bool(df["SeasonalityApplied"].iloc[0]):
        run_month_num = int(df["SeasonalityMonthUsed"].iloc[0]) if not pd.isna(df["SeasonalityMonthUsed"].iloc[0]) else None
        run_month_name = calendar.month_name[run_month_num] if run_month_num and 1 <= run_month_num <= 12 else "n/a"
        st.caption(f"Seasonality: **ON** · Run month: **{run_month_name}**")
    else:
        st.caption("Seasonality: **OFF**")

    # Letter grades
    def _assign_score(v: float) -> str:
        if v >= 90: return "A"
        elif v >= 75: return "B"
        elif v >= 60: return "C"
        elif v >= 45: return "D"
        else: return "E"

    if "Score" not in df.columns:
        df["Score"] = df["Composite"].apply(_assign_score)

    # ---------- TABLE (now includes seasonality columns when present) ----------
    # Build the display dataframe in-place (TicketMedian -> TicketHistory; EffectiveTicketIndex -> TicketIndex used)
    df_show = df.rename(columns={
        "TicketMedian": "TicketHistory",
        "EffectiveTicketIndex": "TicketIndex used",
    }).copy()

    # Ensure RunMonth is always a nice month name in the table
    import calendar
    def _to_month_name(v):
        # Accept int/float month numbers, or pass through an existing string
        try:
            if pd.isna(v):
                return ""
        except Exception:
            pass
        if isinstance(v, (int, float)):
            m = int(v)
            return calendar.month_name[m] if 1 <= m <= 12 else ""
        if isinstance(v, str):
            return v
        return ""

    if "RunMonth" in df_show.columns:
        df_show["RunMonth"] = df_show["RunMonth"].apply(_to_month_name)
    elif "SeasonalityMonthUsed" in df_show.columns:
        df_show["RunMonth"] = df_show["SeasonalityMonthUsed"].apply(_to_month_name)

    # Preferred table columns (only shown if present)
    table_cols = [
        "Title","Region","Segment","Gender","Category",
        "WikiIdx","TrendsIdx","YouTubeIdx","SpotifyIdx",
        "Familiarity","Motivation",
        "TicketHistory",
        "TicketIndex used","TicketIndexSource",
        "RunMonth","FutureSeasonalityFactor","HistSeasonalityFactor",
        "Composite","Score","EstimatedTickets",
    ]

    present_cols = [c for c in table_cols if c in df_show.columns]

    st.subheader("🎟️ Estimated ticket sales (table view)")
    st.dataframe(
        df_show[present_cols]
            .sort_values(
                by=[
                    "EstimatedTickets" if "EstimatedTickets" in df_show.columns else "Composite",
                    "Composite", "Motivation", "Familiarity"
                ],
                ascending=[False, False, False, False]
            )
            .style
            .format({
                "WikiIdx": "{:.0f}", "TrendsIdx": "{:.0f}", "YouTubeIdx": "{:.0f}", "SpotifyIdx": "{:.0f}",
                "Familiarity": "{:.1f}", "Motivation": "{:.1f}",
                "Composite": "{:.1f}",
                "TicketIndex used": "{:.1f}",
                "EstimatedTickets": "{:,.0f}",
                "TicketHistory": "{:,.0f}",
                "FutureSeasonalityFactor": "{:.3f}",
                "HistSeasonalityFactor": "{:.3f}",
            })
            .map(
                lambda v: (
                    "color: green;" if v == "A" else
                    "color: darkgreen;" if v == "B" else
                    "color: orange;" if v == "C" else
                    "color: darkorange;" if v == "D" else
                    "color: red;" if v == "E" else ""
                ),
                subset=["Score"] if "Score" in df_show.columns else []
            ),
        use_container_width=True,
        hide_index=True
    )

    # Build the “full” export by deriving LA fields from category→program overlays
    df_full = attach_la_report_columns(df)

    # CSV with just the table columns (as displayed)
    st.download_button(
        "⬇️ Download Scores CSV (table columns only)",
        df_show[present_cols].to_csv(index=False).encode("utf-8"),
        "title_scores_table_view.csv",
        "text/csv"
    )

    # CSV with ALL columns (now includes the complete LA report fields + seasonality meta)
    st.download_button(
        "⬇️ Download Full CSV (includes all LA data)",
        df_full.to_csv(index=False).encode("utf-8"),
        "title_scores_full_with_LA.csv",
        "text/csv"
    )

    # ---------- OPTIONAL CHARTS ----------
    try:
        fig, ax = plt.subplots()
        ax.scatter(df["Familiarity"], df["Motivation"])
        for _, r in df.iterrows():
            ax.annotate(r["Title"], (r["Familiarity"], r["Motivation"]), fontsize=8)
        ax.axvline(df["Familiarity"].median(), linestyle="--")
        ax.axhline(df["Motivation"].median(), linestyle="--")
        ax.set_xlabel(f"Familiarity ({benchmark_title} = 100 index)")
        ax.set_ylabel(f"Motivation ({benchmark_title} = 100 index)")
        ax.set_title(f"Familiarity vs Motivation — {segment} / {region}")
        st.pyplot(fig)
    except Exception:
        pass

# ======= BUTTON HANDLER =======
run = st.button("Score Titles", type="primary")

if run:
    if not titles:
        st.warning("Add at least one title to score.")
    else:
        compute_scores_and_store(
            titles=titles,
            segment=segment,
            region=region,
            use_live=use_live,
            yt_key=yt_key,
            sp_id=sp_id,
            sp_secret=sp_secret,
            benchmark_title=st.session_state.get("benchmark_title", list(BASELINES.keys())[0]),
            proposed_run_date=proposed_run_date,
        )

# ======= ALWAYS RENDER LAST RESULTS IF AVAILABLE =======
if st.session_state["results"] is not None:
    render_results()
