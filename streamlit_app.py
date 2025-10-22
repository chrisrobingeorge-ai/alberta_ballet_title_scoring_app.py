# streamlit_app_v9_new_titles_fix.py
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
if "la_payload" not in st.session_state:
    st.session_state["la_payload"] = None  # will hold your LA report (df, dict, or raw text)

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
    "Complexions": {"wiki": 62, "trends": 58, "youtube": 66, "spotify": 60, "category": "contemporary", "gender": "na"},
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

# --- Live Analytics overlay mapping (program-type ↔ your categories) ---
CATEGORY_TO_PROGRAM = {
    "pop_ip":            "Pop Music Ballet",
    "classic_romance":   "Classic Ballet",
    "classic_comedy":    "Classic Ballet",
    "contemporary":      "Contemporary Mixed Bill",
    "family_classic":    "Family Ballet",
    "romantic_tragedy":  "CSNA",                  # Classical Story Narrative (Adult)
    "dramatic":          "Contemporary Narrative" # could also map some to "Cultural Narrative"
}

# Percent values from your Live Analytics table (per program type).
# All numbers are PERCENTAGES (we’ll divide by 100 below).
LA_BY_PROGRAM = {
    "Pop Music Ballet": {
        "Presale": 12, "FirstDay": 7, "FirstWeek": 5, "WeekOf": 20,
        "Internet": 58, "Mobile": 41, "Phone": 1,
        "Tix_1_2": 73, "Tix_3_4": 22, "Tix_5_8": 5,
        "Premium": 4.5, "LT10mi": 71,
        "Price_Low": 18, "Price_Fair": 32, "Price_Good": 24, "Price_VeryGood": 16, "Price_Best": 10,
    },
    "Classic Ballet": {
        "Presale": 10, "FirstDay": 6, "FirstWeek": 4, "WeekOf": 20,
        "Internet": 55, "Mobile": 44, "Phone": 1,
        "Tix_1_2": 69, "Tix_3_4": 25, "Tix_5_8": 5,
        "Premium": 4.1, "LT10mi": 69,
        "Price_Low": 20, "Price_Fair": 29, "Price_Good": 24, "Price_VeryGood": 16, "Price_Best": 10,
    },
    "Contemporary Mixed Bill": {
        "Presale": 11, "FirstDay": 6, "FirstWeek": 5, "WeekOf": 21,
        "Internet": 59, "Mobile": 40, "Phone": 1,
        "Tix_1_2": 72, "Tix_3_4": 23, "Tix_5_8": 5,
        "Premium": 3.9, "LT10mi": 70,
        "Price_Low": 18, "Price_Fair": 32, "Price_Good": 25, "Price_VeryGood": 15, "Price_Best": 11,
    },
    "Family Ballet": {
        "Presale": 10, "FirstDay": 6, "FirstWeek": 5, "WeekOf": 19,
        "Internet": 53, "Mobile": 45, "Phone": 1,
        "Tix_1_2": 67, "Tix_3_4": 27, "Tix_5_8": 6,
        "Premium": 4.0, "LT10mi": 66,
        "Price_Low": 21, "Price_Fair": 28, "Price_Good": 24, "Price_VeryGood": 17, "Price_Best": 10,
    },
    "CSNA": {
        "Presale": 11, "FirstDay": 6, "FirstWeek": 5, "WeekOf": 20,
        "Internet": 59, "Mobile": 39, "Phone": 2,
        "Tix_1_2": 72, "Tix_3_4": 22, "Tix_5_8": 5,
        "Premium": 4.2, "LT10mi": 71,
        "Price_Low": 18, "Price_Fair": 28, "Price_Good": 27, "Price_VeryGood": 16, "Price_Best": 11,
    },
    "Contemporary Narrative": {
        "Presale": 12, "FirstDay": 7, "FirstWeek": 5, "WeekOf": 20,
        "Internet": 54, "Mobile": 44, "Phone": 1,
        "Tix_1_2": 72, "Tix_3_4": 23, "Tix_5_8": 5,
        "Premium": 4.3, "LT10mi": 68,
        "Price_Low": 18, "Price_Fair": 31, "Price_Good": 24, "Price_VeryGood": 16, "Price_Best": 11,
    },
    "Cultural Narrative": {
        "Presale": 11, "FirstDay": 6, "FirstWeek": 5, "WeekOf": 21,
        "Internet": 61, "Mobile": 37, "Phone": 2,
        "Tix_1_2": 71, "Tix_3_4": 23, "Tix_5_8": 5,
        "Premium": 4.2, "LT10mi": 74,
        "Price_Low": 19, "Price_Fair": 28, "Price_Good": 26, "Price_VeryGood": 17, "Price_Best": 10,
    },
}

def _program_for_category(cat: str) -> Optional[str]:
    return CATEGORY_TO_PROGRAM.get(cat)

def _add_live_analytics_overlays(df_in: pd.DataFrame) -> pd.DataFrame:
    """Join LA behavior/price overlays onto each title row using Category→Program mapping."""
    df = df_in.copy()

    cols = [
        "LA_EarlyBuyerPct","LA_WeekOfPct",
        "LA_MobilePct","LA_InternetPct","LA_PhonePct",
        "LA_Tix12Pct","LA_Tix34Pct","LA_Tix58Pct",
        "LA_PremiumPct","LA_LocalLT10Pct",
        "LA_PriceHiPct","LA_PriceFlag"
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan

    def _overlay_row(cat: str) -> dict:
        prog = _program_for_category(cat)
        if not prog or prog not in LA_BY_PROGRAM:
            return {}
        la = LA_BY_PROGRAM[prog]
        early = (la["Presale"] + la["FirstDay"] + la["FirstWeek"]) / 100.0
        price_hi = (la["Price_VeryGood"] + la["Price_Best"]) / 100.0
        return {
            "LA_EarlyBuyerPct": early,
            "LA_WeekOfPct": la["WeekOf"] / 100.0,
            "LA_MobilePct": la["Mobile"] / 100.0,
            "LA_InternetPct": la["Internet"] / 100.0,
            "LA_PhonePct": la["Phone"] / 100.0,
            "LA_Tix12Pct": la["Tix_1_2"] / 100.0,
            "LA_Tix34Pct": la["Tix_3_4"] / 100.0,
            "LA_Tix58Pct": la["Tix_5_8"] / 100.0,
            "LA_PremiumPct": la["Premium"] / 100.0,
            "LA_LocalLT10Pct": la["LT10mi"] / 100.0,
            "LA_PriceHiPct": price_hi,
        }

    overlays = df["Category"].map(_overlay_row)
    overlays_df = pd.DataFrame(list(overlays)).reindex(df.index)
    df.update(overlays_df)

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

default_list = list(BASELINES.keys())[:50]
st.markdown("**Titles to score** (one per line). Add NEW titles freely:")
titles_input = st.text_area("Enter titles", value="\n".join(default_list), height=220)
titles = [t.strip() for t in titles_input.splitlines() if t.strip()]

with st.expander("📎 Live Analytics report (optional) — paste or upload"):
    up = st.file_uploader("Upload LA report (CSV or Excel)", type=["csv", "xlsx", "xls"])
    txt = st.text_area("…or paste a TSV/CSV-like block (Label<TAB>Value per line)", height=160)

    la_obj = None
    if up is not None:
        try:
            if up.name.lower().endswith(".csv"):
                la_obj = pd.read_csv(up)
            else:
                la_obj = pd.read_excel(up)
            st.success(f"Loaded LA report: {up.name} ({la_obj.shape[0]} rows × {la_obj.shape[1]} cols)")
        except Exception as e:
            st.error(f"Could not read file: {e}")

    if not la_obj and txt.strip():
        # Keep raw text; parser in attach_la_report_columns can handle TSV-ish blocks
        la_obj = txt
        st.info("Loaded LA report from pasted text.")

    if la_obj is not None:
        st.session_state["la_payload"] = la_obj

run = st.button("Score Titles", type="primary")

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
    "Away We Go - Mixed Bill": [4649],
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
    "Our Canada - Gordon Lightfoot": [10138],
    "Phi - David Bowie": [12336],
    "Shaping Sound": [10208],
    "Sleeping Beauty": [9596.5],
    "Swan Lake": [13157],
    "Taj Express": [10314],
    "Tango Fire": [7251],
    "Trockadero": [5476],
    "Unleashed - Mixed Bill": [5221],
    "Winter Gala": [1321],
    "Wizard of Oz": [8468],
}
def _median(xs):
    xs = sorted([float(x) for x in xs if x is not None])
    if not xs: return None
    n = len(xs); mid = n // 2
    return xs[mid] if n % 2 else (xs[mid-1] + xs[mid]) / 2.0

TICKET_BLEND_WEIGHT = 0.50  # 50% tickets, 50% familiarity/motivation composite

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
def compute_scores_and_store():
    rows = []
    unknown_used_live, unknown_used_est = [], []

    # 1) Build base rows from baselines or live/estimated
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

    # 2) Pick benchmark & normalize Familiarity/Motivation
    benchmark_title = st.selectbox(
        "Choose Benchmark Title for Normalization",
        options=list(BASELINES.keys()),
        index=0
    )
    bench_entry = BASELINES[benchmark_title]
    bench_fam_raw, bench_mot_raw = calc_scores(bench_entry, segment, region)
    bench_fam_raw = bench_fam_raw or 1.0
    bench_mot_raw = bench_mot_raw or 1.0

    df["Familiarity"] = (df["FamiliarityRaw"] / bench_fam_raw) * 100.0
    df["Motivation"]  = (df["MotivationRaw"]  / bench_mot_raw)  * 100.0
    st.caption(f"Scores normalized to benchmark: **{benchmark_title}**")

    # 3) Ticket medians & TicketIndex (history) relative to chosen benchmark
    TICKET_MEDIANS = {k: _median(v) for k, v in TICKET_PRIORS_RAW.items()}
    BENCHMARK_TICKET_MEDIAN = TICKET_MEDIANS.get(benchmark_title, None) or 1.0

    def ticket_index_for_title(title: str):
        aliases = {"Handmaid’s Tale": "Handmaid's Tale"}
        key = aliases.get(title.strip(), title.strip())
        med = TICKET_MEDIANS.get(key)
        if med: return float(med), float((med / BENCHMARK_TICKET_MEDIAN) * 100.0)
        return None, None

    medians, indices = [], []
    for t in df["Title"]:
        med, idx = ticket_index_for_title(t)
        medians.append(med); indices.append(idx)
    df["TicketMedian"] = medians
    df["TicketIndex"]  = indices

    # 4) Build SignalOnly and learn mapping TicketIndex ≈ a*SignalOnly + b
    df["SignalOnly"] = df[["Familiarity", "Motivation"]].mean(axis=1)
    df_known = df[df["TicketIndex"].notna()].copy()

    def _fit_overall_and_by_category(df_known_in: pd.DataFrame):
        overall = None
        if len(df_known_in) >= 5:
            x = df_known_in["SignalOnly"].values
            y = df_known_in["TicketIndex"].values
            a, b = np.polyfit(x, y, 1)
            overall = (float(a), float(b))
        cat_coefs = {}
        for cat, g in df_known_in.groupby("Category"):
            if len(g) >= 3:
                xs = g["SignalOnly"].values
                ys = g["TicketIndex"].values
                a, b = np.polyfit(xs, ys, 1)
                cat_coefs[cat] = (float(a), float(b))
        return overall, cat_coefs

    overall_coef, cat_coefs = _fit_overall_and_by_category(df_known)

    # 5) Impute TicketIndex for titles without history
    def _predict_ticket_index(signal_only: float, category: str) -> tuple[float, str]:
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
        if pd.notna(r["TicketIndex"]):
            imputed_vals.append(r["TicketIndex"]); imputed_srcs.append("History")
        else:
            pred, src = _predict_ticket_index(r["SignalOnly"], r["Category"])
            imputed_vals.append(pred); imputed_srcs.append(src)

    df["TicketIndexImputed"] = imputed_vals
    df["TicketIndexSource"]  = imputed_srcs

    # 6) Composite: blend signals with (history or imputed) TicketIndex
    tickets_component = np.where(
        df["TicketIndexSource"].eq("Not enough data"),
        df["SignalOnly"],
        df["TicketIndexImputed"]
    )
    df["Composite"] = (1.0 - TICKET_BLEND_WEIGHT) * df["SignalOnly"] + TICKET_BLEND_WEIGHT * tickets_component

    # 7) EstimatedTickets (index → tickets using benchmark's historical median)
    BENCHMARK_TICKET_MEDIAN_LOCAL = BENCHMARK_TICKET_MEDIAN or 1.0

    df["EffectiveTicketIndex"] = np.where(
        df["TicketIndex"].notna(), df["TicketIndex"],
        np.where(df["TicketIndexImputed"].notna(), df["TicketIndexImputed"], df["SignalOnly"])
    )

    def _est_src(row):
        if pd.notna(row.get("TicketMedian", np.nan)): return "History (actual median)"
        if pd.notna(row.get("TicketIndexImputed", np.nan)): return f'Predicted ({row.get("TicketIndexSource","model")})'
        return "Online-only (proxy: low confidence)"

    df["TicketEstimateSource"] = df.apply(_est_src, axis=1)
    df["EstimatedTickets"] = ((df["EffectiveTicketIndex"] / 100.0) * BENCHMARK_TICKET_MEDIAN_LOCAL).round(0)

    # --- Live Analytics overlays ---
    df = _add_live_analytics_overlays(df)

    # --- Segment propensity & ticket allocation (adds Mix_* and Seg_* columns) ---
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

    df["PredictedPrimarySegment"] = prim_list
    df["PredictedSecondarySegment"] = sec_list
    df["Mix_GP"] = mix_gp; df["Mix_Core"] = mix_core; df["Mix_Family"] = mix_family; df["Mix_EA"] = mix_ea
    df["Seg_GP_Tickets"] = seg_gp_tix; df["Seg_Core_Tickets"] = seg_core_tix
    df["Seg_Family_Tickets"] = seg_family_tix; df["Seg_EA_Tickets"] = seg_ea_tix

    # 8) Stash
    st.session_state["results"] = {
        "df": df,
        "benchmark": benchmark_title,
        "segment": segment,
        "region": region,
        "unknown_est": unknown_used_est,
        "unknown_live": unknown_used_live,
        # NEW: pass the LA report through the results dict so the exporter can use it
        "la_report": st.session_state.get("la_payload"),
    }

import re
import pandas as pd

# The LA labels you want in the CSV (from your list)
LA_FIELDS = [
    # --- DATA ATTRIBUTE ---
    "Customers","Customers with Live Events","Customers with Demo (US)","Customers with Demo (CAN)",
    # --- DEMOGRAPHICS (US) / Gender ---
    "Male","Female",
    # --- DEMOGRAPHICS (US) / Current Age ---
    "Age","18-24","25-34","35-44","45-54","55-64","65+",
    # --- Race/Ethnicity ---
    "Caucasian","Afr Am","Hispanic","Asian",
    # --- Education ---
    "High School","College Degree","Graduate Degree",
    # --- Occupation ---
    "Professional/Technical","Admin/Managerial","Clerical/White Collar","Crafts/Blue Collar","Sale/Services",
    "Self Employed","Retired","Home Maker","Student","Other",
    # --- Household Summary ---
    "Married","Presence of Children in Household","Presence of Children: Age ≤ 2","Presence of Children: Age 3-5",
    "Presence of Children: Age 6-10","Presence of Children: Age 11-15","Presence of Children: Age 16-17",
    "Working Woman in Household",
    # --- Household Income ---
    "Household Income","<$30K","$30-$49K","$50-$74K","$75-$99K","$100K-$149K","$150K-$199K","$200K-$249K","$250K+",
    # --- Net Worth ---
    "Net Worth","$0","$1K-$49K","$50K-$99K","$100K-$249K","$250K-$499K","$500K-$999K","$1M-$2M","$2M plus",
    # --- DII ---
    "DII Score","0","1-50","51-100","101-150","151-200","201-500","501+",
    # --- Financial Score ---
    "Financial Score","Bottom 25%","Bottom 26-35%","Bottom 36-50%","Top 51-60%","Top 61-75%","Top 76-90%","Top 91-100%",
    # --- Vehicle Type ---
    "Compact Car","Mid/Full Size Car","Luxury Car","Sports Car","SUV","SUV (Luxury)","Pickup/Truck","Van",
    # --- Lifestyle Interest ---
    "Theatre","Arts","Travel Domestic","Upscale Living","Music Grouping","Home Stereo","Music Player","Avid Music Listener",
    "Music Collector","Movie Collector","Sports Grouping","Football","Baseball","Basketball","Hockey","Racing","NASCAR",
    # --- PersonicX Clusters (many) ---
    "Summit Estates","Established Elite","Corporate Connected","Top Professionals","Active & Involved","Casual Comfort",
    "Active Lifestyles","Solid Surroundings","Busy Schedules","Careers & Travel","Schools & Shopping","On the Go",
    "Work & Play","Career Centered","Country Ways","Country Enthusiasts","Firmly Established","Climbing the Ladder",
    "Country Comfort","Carving Out Time","Children First","Comfortable Cornerstones","Good Neighbors","Career Building",
    "Clubs & Causes","Getting Established","Tenured Proprietors","Community Pillars","City Mixers","Out & About",
    "Mid-Americana","Metro Mix","Urban Diversity","Outward Bound","Working & Active","Persistent & Productive",
    "Firm Foundations","Occupational Mix","Setting Goals","Great Outdoors","Rural Adventure","Creative Variety",
    "Work & Causes","Open Houses","Offices & Entertainment","Rural & Active","Rural Parents","Farm & Home",
    "Home & Garden","Rural Community","Role Models","Stylish & Striving","Metro Strivers","Work & Outdoors",
    "Community Life","Metro Active","Collegiate Crowd","Outdoor Fervor","Mobile Mixers","Rural & Mobile","City Life",
    "Movies & Sports","Staying Home","Practical & Careful","Hobbies & Shopping","Helping Hands","First Steps",
    "Staying Healthy","Productive Havens","Favorably Frugal",
    # --- PersonicX Groups (many) ---
    "Starting Out","Taking Hold","Settling Down","Social Connectors","Busy Households","Working & Studying",
    "Career Oriented","Large Households","Comfortable Independence","Rural-Metro Mix","Affluent Households",
    "Comfortable Households","Working Households","Diverging Paths","Top Wealth ","Living Well","Bargain Hunters",
    "Thrifty & Active","Solid Prestige","Community Minded","Leisure Seekers",
    # --- DEMOGRAPHICS (Canada) / Income & Spend ---
    "Average Household Income","Disposable Income","Discretionary Income",
    # --- Recreation Expenditure ---
    "Total Recreation Expenditure","Entertainment","Live Arts","Live Sports",
    # --- Household Composition ---
    "Married Households","% of Household with No Children","% of Household with 1 Child",
    "% of Household with 2 Children","% of Household with 3+ Children",
    # --- Occupation (Canada) ---
    "Sales & Services","Bus Finance & Admin","Social Sci, Educ, & Govt","Trades & Transport","Management","Sciences",
    "Health","Art, Culture, Recreation","Agriculture","Manufacturer & Utilities",
    # --- Education (Canada) ---
    "No College Degree","College Degree",
    # --- Prizm5 Group (many) ---
    "Cosmopolitan Elite","Urbane Villagers","Arts & Affluence","Suburban Success","Asian Sophisticates","Kids & Careers",
    "Nouveaux Riches","Boomerang City","Satellite Burbs","Emptying Nests","Urban Digerati","Street Scenes","Asian Avenues",
    "Diversity Heights","Heritage Hubs","Pets & PCs","Exurban Wonderland","Management Material","Grey Pride",
    "South Asian Achievers","Beau Monde","Aging in Suburbia","Asian New Wave","Fresh Air Families","South Asian Society",
    "Second City Retirees","Diverse City","Metro Multiculturals","Silver Linings","La Vie est Belle","New World Symphony",
    "Mini Van & Vin Rouge","Heartland Retirees","Rooms with a View","Country Acres","Exurban Homesteaders","Trucks & Trades",
    "Grads & Pads","Our Time","Wide Open Spaces","Vieille École","Home Sweet Rows","Newcomers Rising","Jeunes et Actifs",
    "Jeunes d'Esprit","Villes Tranquilles","Traditional Town Living","Variété Suburbaine","Enclaves Multiethniques",
    "Suburban Scramble","Aging & Active","Striving Startups","Outdoor Originals","Serenity Springs","La Vie Bucolique",
    "Single City Jazz","First Nations Families","Rustic Roads","Locataires en Banlieues","Bons Vivants","Les Seniors",
    "Terre à Terre","Lunch at Tim's","Fête au Village","Young & Connected","Sunset Towers","Survivre en Ville","Low-Rise Renters",
    # --- Prizm Social Group (Urbanity) ---
    "Urban","Suburban","Exurban","Town","Rural",
    # --- Prizm Social Group (SES) ---
    "Exurban Elite","Exurban Middle-Aged","Exurban Francophone","Rural Midscale","Rural Downscale","Rural/Town Francophone",
    "Suburban Elite","Suburban Younger","Suburban Upscale Ethnic","Suburban Older","Suburban Francophone","Town Mix",
    "Urban Elite","Urban Upscale Ethnic","Urban Young","Urban Younger Ethnic","Urban Older","Urban Downscale","Urban Francophone",
    # --- Prizm LifeStage ---
    "Younger","Family","Mature",
    # --- Prizm LifeStage Group (many) ---
    "Young Diverse Families","Growing Families","Older Parents, Younger Kids","Families with Tweens","Midlife Families",
    "Midlife Quebec Families","Midlife Ethnic Families","Prosperous Parents","Older Families","Nearly Retired","Country Seniors",
    "Later Years","Singles Scene","Starter Nests",
    # --- LIVE EVENT SUMMARY / Transaction Summary ---
    "Number of Attractions","Number of Events","Number of Venues","Total Spend","Spend per Event","Tickets per Event",
    "Avg. Ticket Price","Ticket Price Pctile Ranking","Distance to Venue (Miles)","Distance to Local Venue (Miles)",
    # --- PERCENTANGE OF CUSTOMERS / Major Category ---
    "Concerts","Arts","Sports","Family","Misc",
    # --- Premium/Resale/Upsell ---
    "Platinum","Resale","Parking","Upsell/Ancil",
    # --- MAJOR CATEGORY STATISTICS / Ticket Price & Spend per Event ---
    # (We already included per-category above; reuse those labels for price/spend series)
    # --- Share of Wallet (per category) ---
    # (Reuses category labels)
    # --- PREFERENCES / Major Category ---
    # (Reuses category labels)
    # --- Ticket Quantity ---
    "1-2 Tickets","3-4 Tickets","5-8 Tickets","9+ Tickets",
    # --- Ticket Price Percentile Ranking ---
    "Low (Bottom 25%)","Fair (Bottom 26-49%)","Good (Top 50-75%)","Very Good (Top 75-89%)","Best (Top 90%)",
    # --- Purchase Timing ---
    "Presale","1st Day Onsale","First Week","In-Between","Week of Event",
    # --- Purchase Channel ---
    "Internet","Phone","Mobile","Box Office","Retail Outlet",
    # --- Payment Method ---
    "Visa","MC","AmEx","Discover","Other",
    # --- Travel Distance ---
    "<10 Miles","11-25 Miles","26-50 Miles","51-100 Miles","101-200 Miles","201+ Miles",
    # --- Event Location ---
    "US Events","Canada Events",
    # --- RFM ---
    "Last 6 Months","Last 12 Months","Last 18 Months","Last 24 Months","Inactive 24+ Months",
    "1 Event","2 Events","3 Events","4 Events","5+ Events",
    "≤ $100","$101-$250","$251-$500","$501-$1000","$1001+",
    "RFM Score","551-1000","451-550","351-450","0-351",
    # --- PROPENSITY SCORES (many) ---
    "Adult Contemporary","Alternative Rock","Country","EDM","Hard Rock/Metal","Jazz/Blues","Latin","Hip Hop","R&B/Soul",
    "Rock/Pop","Concert Festival","Other Concerts","Broadway Musical","Comedy","Classical/Opera","Theatre Plays","Ballet/Dance",
    "Arts Festival","Magic","Other Arts","Circus","Family Shows","Other Family","Boxing","MMA","Wrestling","Baseball","MLB",
    "College Baseball","Basketball","NBA","WNBA","College Basketball","Football","NFL","College Football","Hockey","NHL",
    "College Hockey","Soccer","MLS","Motorsport","Rodeo","Golf","PGA","Tennis","Other Sports",
    # --- CLIENT CUSTOMER SUMMARY / CURRENT RESIDENCE distances (labels reused above) ---
    # --- CLIENT EVENT SUMMARY / Age at Event ---
    "Age (at time of Event)","Age of Cohort (Retrospectively Computed)",
    # --- CLIENT EVENT SUMMARY / Travel Distance to Client Events (distances reused) ---
    "Distance to Client Events (Miles)","Client Events","Non-Client Events",
    "Client Events Purchased (Lifetime) - Never Purchased","Client Events Purchased (Lifetime) - 1 Event",
    "Client Events Purchased (Lifetime) - 2-3 Events","Client Events Purchased (Lifetime) - 4-5 Events",
    "Client Events Purchased (Lifetime) - 6-10 Events","Client Events Purchased (Lifetime) - 11+ Events",
    "Client Events Spend (Lifetime) - ≤ $250","Client Events Spend (Lifetime) - $251-$500",
    "Client Events Spend (Lifetime) - $501-$1000","Client Events Spend (Lifetime) - $1001-$2000",
    "Client Events Spend (Lifetime) - $2001+",
    "Tickets per Event - Client Events","Tickets per Event - Non-Client Events",
    "Spend per Event - Client Events","Spend per Event - Non-Client Events",
    "Average Ticket Price - Client Events","Average Ticket Price - Non-Client Events",
    # --- NON-CLIENT EVENT SUMMARY (per-category spend/avg ticket ‘All/Concerts/Arts/Sports/Family/Misc’) ---
    "Spend per Event - All","Average Ticket Price - All",
]

def _san(s: str) -> str:
    """Normalize a label to a safe column name."""
    s = s.strip()
    s = re.sub(r"\s*[:()]\s*", "_", s)      # colons/parentheses to underscores
    s = s.replace("&", "and").replace("+", "plus").replace("/", "_")
    s = re.sub(r"[^A-Za-z0-9_<>=\-]+", "_", s)  # non-alnum (keep <,>,= for ranges as-is)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def _series_from_any(la_any) -> pd.Series:
    """Turn various LA payloads into a flat Series of label->value."""
    if la_any is None:
        return pd.Series(dtype="object")
    # Case 1: dict-like {label: value}
    if isinstance(la_any, dict):
        return pd.Series(la_any, dtype="object")
    # Case 2: DataFrame with 2 columns (label, value)
    if isinstance(la_any, pd.DataFrame) and la_any.shape[1] >= 2:
        lab = la_any.columns[0]; val = la_any.columns[1]
        return pd.Series(la_any[val].values, index=la_any[lab].values, dtype="object")
    # Case 3: raw text/TSV blob (label \t value per line)
    if isinstance(la_any, str):
        rows = []
        for line in la_any.splitlines():
            if "\t" in line:
                k, v = line.split("\t", 1)
                rows.append((k.strip(), v.strip()))
        if rows:
            return pd.Series(dict(rows), dtype="object")
    return pd.Series(dtype="object")

def attach_la_report_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build LA_* columns for EVERY row, using the Category→Program mapping and LA_BY_PROGRAM.
    Only fields we can infer from your overlays are filled; others remain NaN.
    """
    df_full = df.copy()

    # Helper: map one program’s overlay to the big label space
    def _template_for_program(prog: Optional[str]) -> dict:
        la = LA_BY_PROGRAM.get(prog or "", {})
        if not la:
            return {}

        # Everything below is in PERCENT form in LA_BY_PROGRAM, so keep it as-is.
        # (Your CSV will show these as numbers; if you want decimals 0–1, divide by 100.0.)
        m: dict[str, float] = {}

        # --- Purchase Timing ---
        m["Presale"] = la.get("Presale")
        m["1st Day Onsale"] = la.get("FirstDay")
        m["First Week"] = la.get("FirstWeek")
        m["Week of Event"] = la.get("WeekOf")

        # --- Purchase Channel ---
        m["Internet"] = la.get("Internet")
        m["Mobile"] = la.get("Mobile")
        m["Phone"] = la.get("Phone")
        # We don’t have these; leave NaN:
        # "Box Office", "Retail Outlet"

        # --- Ticket Quantity ---
        m["1-2 Tickets"] = la.get("Tix_1_2")
        m["3-4 Tickets"] = la.get("Tix_3_4")
        m["5-8 Tickets"] = la.get("Tix_5_8")
        # If you want, you can compute residual into "9+ Tickets"; here we leave NaN.

        # --- Price Percentile Ranking ---
        m["Low (Bottom 25%)"]   = la.get("Price_Low")
        m["Fair (Bottom 26-49%)"] = la.get("Price_Fair")
        m["Good (Top 50-75%)"]  = la.get("Price_Good")
        m["Very Good (Top 75-89%)"] = la.get("Price_VeryGood")
        m["Best (Top 90%)"]     = la.get("Price_Best")

        # --- Travel Distance (we only know <10mi) ---
        m["<10 Miles"] = la.get("LT10mi")
        # Others ("11-25 Miles", ... "201+ Miles") stay NaN.

        # --- Premium/Resale/Upsell ---
        # We only have "Premium". The report asks for "Platinum", "Resale", "Parking", "Upsell/Ancil".
        # If you'd like "Premium" to populate "Platinum", uncomment the next line:
        # m["Platinum"] = la.get("Premium")

        return m

    # For every row, derive a dict of label → value from its Category’s program
    per_row_maps: list[dict] = []
    for _, r in df.iterrows():
        prog = _program_for_category(str(r["Category"]))
        per_row_maps.append(_template_for_program(prog))

    # Now materialize every requested LA field into df_full
    for label in LA_FIELDS:
        col = f"LA_{_san(label)}"
        vals = []
        for rowmap in per_row_maps:
            v = rowmap.get(label)
            vals.append(v if v is not None else np.nan)
        df_full[col] = vals

    return df_full

# -------------------------
# RENDER
# -------------------------
def render_results():
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

    # Letter grades
    def _assign_score(v: float) -> str:
        if v >= 90: return "A"
        elif v >= 75: return "B"
        elif v >= 60: return "C"
        elif v >= 45: return "D"
        else: return "E"

    if "Score" not in df.columns:
        df["Score"] = df["Composite"].apply(_assign_score)

    # ---------- TABLE (only your 17 columns) ----------
    # Build the display dataframe in-place (TicketMedian -> TicketHistory; EffectiveTicketIndex -> TicketIndex used)
    df_show = df.rename(columns={
        "TicketMedian": "TicketHistory",
        "EffectiveTicketIndex": "TicketIndex used",
    }).copy()

    table_cols = [
        "Title","Region","Segment","Gender","Category",
        "WikiIdx","TrendsIdx","YouTubeIdx","SpotifyIdx",
        "Familiarity","Motivation",
        "TicketHistory",
        "TicketIndex used","TicketIndexSource",
        "Composite","Score","EstimatedTickets",
    ]

    # Keep only columns that are present (avoids KeyErrors if something is missing)
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

    # CSV with just the table columns (your 17)
    st.download_button(
        "⬇️ Download Scores CSV (table columns only)",
        df_show[present_cols].to_csv(index=False).encode("utf-8"),
        "title_scores_table_view.csv",
        "text/csv"
    )
    
    # CSV with ALL columns (now includes the complete LA report fields)
    st.download_button(
        "⬇️ Download Full CSV (includes all LA data)",
        df_full.to_csv(index=False).encode("utf-8"),
        "title_scores_full_with_LA.csv",
        "text/csv"
    )

    # ---------- OPTIONAL CHARTS (unchanged) ----------
    try:
        fig, ax = plt.subplots()
        ax.scatter(df["Familiarity"], df["Motivation"])
        for _, r in df.iterrows():
            ax.annotate(r["Title"], (r["Familiarity"], r["Motivation"]), fontsize=8)
        ax.axvline(df["Familiarity"].median(), color="gray", linestyle="--")
        ax.axhline(df["Motivation"].median(), color="gray", linestyle="--")
        ax.set_xlabel(f"Familiarity ({benchmark_title} = 100 index)")
        ax.set_ylabel(f"Motivation ({benchmark_title} = 100 index)")
        ax.set_title(f"Familiarity vs Motivation — {segment} / {region}")
        st.pyplot(fig)
    except Exception:
        # keep UI resilient if any chart inputs are missing
        pass

# ======= BUTTON HANDLER =======
if run:
    if not titles:
        st.warning("Add at least one title to score.")
    else:
        compute_scores_and_store()

# ======= ALWAYS RENDER LAST RESULTS IF AVAILABLE =======
if st.session_state["results"] is not None:
    render_results()
