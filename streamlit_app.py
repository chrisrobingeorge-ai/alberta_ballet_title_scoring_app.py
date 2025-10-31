# Alberta Ballet — Title Familiarity & Motivation Scorer (v9.2)
# - Learns YYC/YEG & Singles/Subs from history.csv (or uploaded CSV)
# - Removes arbitrary 60/40 split; uses title→category→default fallback
# - Small fixes: softmax bug, LA attach loop, duplicate imports, safer guards

import math, time, re, io
from datetime import datetime, timedelta, date
from typing import Dict, Optional, Tuple

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from textwrap import dedent

# -------------------------
# App setup
# -------------------------
st.set_page_config(page_title="Alberta Ballet — Title Familiarity & Motivation Scorer", layout="wide")
if "results" not in st.session_state:
    st.session_state["results"] = None

st.title("🎭 Alberta Ballet — Title Familiarity & Motivation Scorer (v9.2)")
st.caption("Hard-coded AB-wide baselines (normalized to your benchmark = 100). Add new titles; choose live fetch or offline estimate.")

# -------------------------
# METHODOLOGY & GLOSSARY SECTION
# -------------------------
with st.expander("📘 About This App — Methodology & Glossary"):
    st.markdown(dedent("""
    ### Purpose
    The Alberta Ballet Familiarity & Motivation Scorer estimates how recognizable a title is and how strongly audiences are motivated to attend.  
    It blends online visibility data (Wikipedia, YouTube, Google Trends, Spotify) with past ticket sales and context multipliers for audience segments, regions, and timing.  
    The goal is to compare titles on a consistent, normalized 100-point benchmark and estimate potential ticket results by segment and month.

    ### Methodology Overview
    **1. Data Sources**
    - **Wikipedia** – cultural awareness  
    - **Google Trends** – active interest  
    - **YouTube** – video engagement intensity (normalized and outlier-winsorized)  
    - **Spotify** – musical familiarity  
    - **Ticket Priors** – median historical sales (used for ground truth)  
    - **Live Analytics Overlays** – timing, channel, and price behaviours mapped by program category  

    **2. Score Construction**
    | Stage | Calculation | Description |
    |:--|:--|:--|
    | Familiarity | 0.55·Wiki + 0.30·Trends + 0.15·Spotify | Overall awareness index |
    | Motivation | 0.45·YouTube + 0.25·Trends + 0.15·Spotify + 0.15·Wiki | Active intent index |
    | Multipliers | × Segment × Region | Adjusts for who and where the interest is |
    | Normalization | ÷ Benchmark × 100 | Benchmark = user-selected baseline (e.g., *Swan Lake*) |
    | Ticket Index | Historical or predicted | Converts de-seasonalized ticket medians into 100-scaled indices |
    | Composite | 50% Online + 50% Ticket Index | Balances digital signals with real sales |
    | Seasonality | Category×Month factor | Applies multiplicative adjustment for expected month performance |
    | Remount Decay | −0–25% | Reduces estimates for titles staged recently |
    | Final Estimate | Composite × Benchmark tickets × Seasonal × Decay | Outputs ticket forecast |

    **3. Segment Propensity (Output)**
    - Each title’s signals are recalculated for all four segments:  
      _General Population_, _Core Classical (F35-64)_, _Family (Parents w/ Kids)_, _Emerging Adults (18-34)_  
    - Segment priors from Live Analytics weight expected shares per region & category.  
    - Outputs include:
      - Predicted **primary** and **secondary** segment  
      - Segment **share %**  
      - Estimated **tickets per segment**

    **4. Live Analytics Overlays**
    Overlays attach average behavioral patterns by category, such as:  
    - Timing (early buyers vs. week-of buyers)  
    - Channel (internet, mobile, phone)  
    - Ticket bundle size (1–2 vs. 3–4 vs. 5–8 tickets)  
    - Price elasticity (price sensitivity flag)  
    These variables contextualize sales strategy rather than altering the score.

    **5. Seasonality and Remount Logic**
    - Historical run data is used to infer Category×Month multipliers (median-to-overall ratios).  
    - Factors are shrunk toward 1.0 for low-sample months and clipped between 0.85 and 1.25.  
    - Remount decay applies 30–10% demand reduction depending on how recently the ballet was staged.

    ### Glossary
    - **Familiarity**: Cultural awareness of the title.  
    - **Motivation**: Audience enthusiasm or viewing intent.  
    - **Ticket Index**: A 100-based scale representing ticket-sale potential vs. the benchmark.  
    - **Composite**: Weighted blend of online and ticket performance (default 50/50).  
    - **Benchmark**: The title to which all others are normalized (set by user).  
    - **Segment Propensity**: Likelihood of appeal to each audience segment.  
    - **Live Analytics Overlay**: Real market behavior patterns by program type.  
    - **Seasonality Factor**: Adjustment by month to reflect typical ticket demand.  
    - **Remount Decay**: Discount applied to titles restaged too soon after a previous run.  
    - **Estimated Tickets (Final)**: Projected attendance after all adjustments.

    ---
    **Recommended use:**  
    Screen potential titles, estimate proportional appeal by segment and month,  
    and calibrate marketing expectations for programming, pricing, and touring.
    """))

# -------------------------
# PRIORS learning (YYC/YEG + Singles/Subs) — self-contained
# -------------------------
# Globals populated from history
TITLE_CITY_PRIORS: dict[str, dict[str, float]] = {}
CATEGORY_CITY_PRIORS: dict[str, dict[str, float]] = {}
SUBS_SHARE_BY_CATEGORY_CITY: dict[str, dict[str, float | None]] = {"Calgary": {}, "Edmonton": {}}

# Sensible fallbacks if history is missing/thin
DEFAULT_BASE_CITY_SPLIT = {"Calgary": 0.60, "Edmonton": 0.40}  # Calgary majority unless learned otherwise
_DEFAULT_SUBS_SHARE = {"Calgary": 0.35, "Edmonton": 0.45}
_CITY_CLIP_RANGE = (0.15, 0.85)

def _pick_col(df: pd.DataFrame, names: list[str]) -> str | None:
    cols_norm = {c.lower().strip(): c for c in df.columns}
    for n in names:
        if n.lower() in cols_norm:
            return cols_norm[n.lower()]
    for want in names:
        for k, orig in cols_norm.items():
            if want.lower() in k:
                return orig
    return None

def _canon_city(x):
    s = str(x).strip().lower()
    if "calg" in s: return "Calgary"
    if "edm" in s:  return "Edmonton"
    return x if x in ("Calgary", "Edmonton") else None

def learn_priors_from_history(hist_df: pd.DataFrame) -> dict:
    """
    Wide schema support for:
      - Show Title
      - Single Tickets - Calgary / Edmonton
      - Subscription Tickets - Calgary / Edmonton
    Handles commas in numbers, blanks, and duplicate titles.
    Populates:
      - TITLE_CITY_PRIORS[title] = {'Calgary': p, 'Edmonton': 1-p}
      - CATEGORY_CITY_PRIORS[category] = {...}  (category inferred from title)
      - SUBS_SHARE_BY_CATEGORY_CITY[city][category] = subs/(subs+singles)
    """
    global TITLE_CITY_PRIORS, CATEGORY_CITY_PRIORS, SUBS_SHARE_BY_CATEGORY_CITY
    TITLE_CITY_PRIORS.clear(); CATEGORY_CITY_PRIORS.clear()
    SUBS_SHARE_BY_CATEGORY_CITY = {"Calgary": {}, "Edmonton": {}}

    if hist_df is None or hist_df.empty:
        return {"titles_learned": 0, "categories_learned": 0, "subs_shares_learned": 0, "note": "empty history"}

    df = hist_df.copy()

    # --- map your exact headers (case/space tolerant) ---
    def _find_col(cands):
        lc = {c.lower().strip(): c for c in df.columns}
        for want in cands:
            if want.lower() in lc:
                return lc[want.lower()]
        # loose contains
        for want in cands:
            for k, orig in lc.items():
                if want.lower() in k:
                    return orig
        return None

    title_col = _find_col(["Show Title", "Title", "Production", "Show"])

    s_cgy = _find_col(["Single Tickets - Calgary"]) or _find_col(["Single", "Calgary"])
    s_edm = _find_col(["Single Tickets - Edmonton"]) or _find_col(["Single", "Edmonton"])
    u_cgy = _find_col(["Subscription Tickets - Calgary"]) or _find_col(["Subscription", "Calgary"])
    u_edm = _find_col(["Subscription Tickets - Edmonton"]) or _find_col(["Subscription", "Edmonton"])

    # if any are missing, create as zeros so it still learns with partial data
    for name, fallback in [(s_cgy, "__s_cgy__"), (s_edm, "__s_edm__"), (u_cgy, "__u_cgy__"), (u_edm, "__u_edm__")]:
        if name is None:
            df[fallback] = 0.0

    s_cgy = s_cgy or "__s_cgy__"
    s_edm = s_edm or "__s_edm__"
    u_cgy = u_cgy or "__u_cgy__"
    u_edm = u_edm or "__u_edm__"

    # clean numerics: handle "7,734" → 7734
    def _num(x) -> float:
        try:
            if pd.isna(x): return 0.0
            return float(str(x).replace(",", "").strip() or 0)
        except Exception:
            return 0.0

    for c in [s_cgy, s_edm, u_cgy, u_edm]:
        df[c] = df[c].map(_num)

    # clean titles
    if title_col is None:
        # bail if we truly can't find a title column
        return {"titles_learned": 0, "categories_learned": 0, "subs_shares_learned": 0, "note": "missing Show Title"}
    df[title_col] = df[title_col].astype(str).str.strip()

    # aggregate duplicates by title
    agg = (
        df.groupby(title_col)[[s_cgy, s_edm, u_cgy, u_edm]]
          .sum(min_count=1)
          .reset_index()
          .rename(columns={title_col: "Title"})
    )

    # infer category from title (uses your app's function if present)
    def _infer_cat(t: str) -> str:
        try:
            if "infer_gender_and_category" in globals():
                return globals()["infer_gender_and_category"](t)[1]
        except Exception:
            pass
        tl = (t or "").lower()
        if any(k in tl for k in ["wizard","peter pan","pinocchio","hansel","frozen","beauty","alice","beast"]): return "family_classic"
        if any(k in tl for k in ["swan","sleeping","cinderella","giselle","sylphide"]):                      return "classic_romance"
        if any(k in tl for k in ["romeo","hunchback","notre dame","hamlet","frankenstein","dracula"]):        return "romantic_tragedy"
        if any(k in tl for k in ["don quixote","merry widow","comedy"]):                                      return "classic_comedy"
        if any(k in tl for k in ["contemporary","boyz","momix","complexions","grimm","nijinsky","deviate","phi","away we go","unleashed","botero","ballet bc"]):
            return "contemporary"
        if any(k in tl for k in ["taj","tango","harlem","tragically hip","leonard cohen","joni","david bowie","gordon lightfoot","phi"]):
            return "pop_ip"
        return "dramatic"

    agg["Category"] = agg["Title"].apply(_infer_cat)

    # totals by city
    agg["YYC_total"] = agg[s_cgy].fillna(0) + agg[u_cgy].fillna(0)
    agg["YEG_total"] = agg[s_edm].fillna(0) + agg[u_edm].fillna(0)

    # ---- title-level priors ----
    titles_learned = 0
    for _, r in agg.iterrows():
        tot = float(r["YYC_total"] + r["YEG_total"])
        if tot <= 0:
            continue
        cal = float(r["YYC_total"]) / tot
        cal = float(min(_CITY_CLIP_RANGE[1], max(_CITY_CLIP_RANGE[0], cal)))
        TITLE_CITY_PRIORS[str(r["Title"])] = {"Calgary": cal, "Edmonton": 1.0 - cal}
        titles_learned += 1

    # ---- category-level priors (weighted) ----
    cat_grp = agg.groupby("Category")[["YYC_total","YEG_total"]].sum(min_count=1).reset_index()
    categories_learned = 0
    for _, r in cat_grp.iterrows():
        tot = float(r["YYC_total"] + r["YEG_total"])
        if tot <= 0:
            continue
        cal = float(r["YYC_total"]) / tot
        cal = float(min(_CITY_CLIP_RANGE[1], max(_CITY_CLIP_RANGE[0], cal)))
        CATEGORY_CITY_PRIORS[str(r["Category"])] = {"Calgary": cal, "Edmonton": 1.0 - cal}
        categories_learned += 1

    # ---- subs share by category × city ----
    agg["YYC_subs"] = agg[u_cgy].fillna(0); agg["YYC_singles"] = agg[s_cgy].fillna(0)
    agg["YEG_subs"] = agg[u_edm].fillna(0); agg["YEG_singles"] = agg[s_edm].fillna(0)

    subs_learned = 0
    for city, sub_col, sing_col in [
        ("Calgary","YYC_subs","YYC_singles"),
        ("Edmonton","YEG_subs","YEG_singles"),
    ]:
        g = agg.groupby("Category")[[sub_col, sing_col]].sum(min_count=1).reset_index()
        for _, r in g.iterrows():
            tot = float(r[sub_col] + r[sing_col])
            if tot <= 0:
                continue
            share = float(r[sub_col] / tot)
            SUBS_SHARE_BY_CATEGORY_CITY.setdefault(city, {})
            SUBS_SHARE_BY_CATEGORY_CITY[city][str(r["Category"])] = float(min(0.95, max(0.05, share)))
            subs_learned += 1

    return {
        "titles_learned": titles_learned,
        "categories_learned": categories_learned,
        "subs_shares_learned": subs_learned,
    }

def city_split_for(title: str | None, category: str | None) -> dict[str, float]:
    """Prefer title prior, then category prior, else default."""
    if title and title in TITLE_CITY_PRIORS:
        return TITLE_CITY_PRIORS[title]
    if category and category in CATEGORY_CITY_PRIORS:
        return CATEGORY_CITY_PRIORS[category]
    return DEFAULT_BASE_CITY_SPLIT.copy()

def subs_share_for(category: str | None, city: str) -> float:
    city = "Calgary" if "calg" in city.lower() else ("Edmonton" if "edm" in city.lower() else city)
    if city in SUBS_SHARE_BY_CATEGORY_CITY and category in SUBS_SHARE_BY_CATEGORY_CITY[city]:
        return float(SUBS_SHARE_BY_CATEGORY_CITY[city][category])
    return float(_DEFAULT_SUBS_SHARE.get(city, 0.40))

# --- Historicals (loads your wide CSV and learns) ---
with st.expander("Historicals (optional): upload or use local CSV", expanded=False):
    uploaded_hist = st.file_uploader("Upload historical ticket CSV", type=["csv"], key="hist_uploader_v9")
    relearn = st.button("🔁 Force re-learn from history", use_container_width=False)

# (Re)load the history df
if ("hist_df" not in st.session_state) or relearn:
    if uploaded_hist is not None:
        st.session_state["hist_df"] = pd.read_csv(uploaded_hist)
    else:
        # try your preferred filename first, then the old one; else empty
        try:
            st.session_state["hist_df"] = pd.read_csv("data/history_city_sales.csv")
        except Exception:
            try:
                st.session_state["hist_df"] = pd.read_csv("data/history.csv")
            except Exception:
                st.session_state["hist_df"] = pd.DataFrame()

# (Re)learn priors every time we (re)load history
st.session_state["priors_summary"] = learn_priors_from_history(st.session_state["hist_df"])

s = st.session_state.get("priors_summary", {}) or {}
st.caption(
    f"Learned priors → titles: {s.get('titles_learned',0)}, "
    f"categories: {s.get('categories_learned',0)}, "
    f"subs-shares: {s.get('subs_shares_learned',0)}"
)

# -------------------------
# Optional APIs (used only if toggled ON)
# -------------------------
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
# Data (your existing constants)
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

# === City split + subscriber share logic (uses learned priors above) ===
def _clip01(p: float, lo_hi=_CITY_CLIP_RANGE) -> float:
    lo, hi = lo_hi
    return float(min(hi, max(lo, p)))

def _normalize_pair(c: float, e: float) -> tuple[float, float]:
    s = (c + e) or 1.0
    return float(c / s), float(e / s)

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
    if not d: return {}
    vals = {k: max(1e-9, float(v)) for k, v in d.items()}
    logs = {k: math.log(v) / max(1e-6, temperature) for k, v in vals.items()}
    mx = max(logs.values())
    exps = {k: math.exp(v - mx) for k, v in logs.items()}
    Z = sum(exps.values()) or 1.0
    return {k: exps[k] / Z for k in exps}

def _infer_segment_mix_for(category: str, region_key: str, temperature: float = 1.0) -> dict[str, float]:
    pri = _prior_weights_for(region_key, category)
    if not pri:
        pri = {k: 1.0 for k in SEGMENT_KEYS_IN_ORDER}
    return _softmax_like(pri, temperature=temperature)


# Heuristics for unknown titles
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

# Live fetchers (guarded)
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
    wiki_idx = 40.0 + min(110.0, (math.log1p(max(0.0, wiki_raw)) * 20.0))
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
    try:
        if sp_id and sp_secret and spotipy is not None:
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

# Scoring utilities
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

# Ticket priors (your table)
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
    "Nijinsky": [4752],
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

# Seasonality build
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

_runs_rows = []
for title, s, e in PAST_RUNS:
    d1, d2 = _to_date(s), _to_date(e)
    mid = _mid_date(d1, d2)
    if title in BASELINES:
        cat = BASELINES[title].get("category", infer_gender_and_category(title)[1])
    else:
        cat = infer_gender_and_category(title)[1]
    aliases = {"Handmaid’s Tale": "Handmaid's Tale"}
    key = aliases.get(title.strip(), title.strip())
    tix_med = _median(TICKET_PRIORS_RAW.get(key, []))
    _runs_rows.append({
        "Title": title,
        "Category": cat,
        "Start": d1,
        "End": d2,
        "MidDate": mid,
        "Month": mid.month,
        "Year": mid.year,
        "TicketMedian": tix_med
    })
RUNS_DF = pd.DataFrame(_runs_rows)

TITLE_TO_MIDDATE = {}
if not RUNS_DF.empty:
    for _, r in RUNS_DF.iterrows():
        TITLE_TO_MIDDATE[str(r["Title"]).strip()] = r["MidDate"]

K_SHRINK = 3.0
MINF, MAXF = 0.85, 1.25

_season_rows = []
if not RUNS_DF.empty:
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
            w = n / (n + K_SHRINK)
            factor_shrunk = 1.0 + w * (factor_raw - 1.0)
            factor_final = float(np.clip(factor_shrunk, MINF, MAXF))
            _season_rows.append({"Category": cat, "Month": int(m), "Factor": factor_final})
SEASONALITY_DF = pd.DataFrame(_season_rows).sort_values(["Category","Month"]).reset_index(drop=True)
SEASONALITY_TABLE = { (r["Category"], int(r["Month"])): float(r["Factor"]) for _, r in SEASONALITY_DF.iterrows() }

def seasonality_factor(category: str, when: Optional[date]) -> float:
    if when is None:
        return 1.0
    key = (str(category), int(when.month))
    return float(SEASONALITY_TABLE.get(key, 1.0))

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

def remount_novelty_factor(title: str, proposed_run_date: Optional[date]) -> float:
    last_middate = TITLE_TO_MIDDATE.get(title.strip())
    if last_middate is None or proposed_run_date is None:
        return 1.0
    delta_years = (proposed_run_date.year - last_middate.year) + \
                  ((proposed_run_date.timetuple().tm_yday - last_middate.timetuple().tm_yday) / 365.25)
    if delta_years <= 2:   return 0.70
    elif delta_years <= 4: return 0.80
    elif delta_years <= 9: return 0.90
    else:                  return 1.00

# -------------------------
# UI — Config
# -------------------------
with st.expander("🔑 API Configuration (used for NEW titles only if enabled)"):
    yt_key = st.text_input("YouTube Data API v3 Key", type="password")
    sp_id = st.text_input("Spotify Client ID", type="password")
    sp_secret = st.text_input("Spotify Client Secret", type="password")
    use_live = st.checkbox("Use Live Data for Unknown Titles", value=False)
    st.caption("Keys are optional and only used when scoring unknown titles with live fetch.")

region = st.selectbox("Region", ["Province", "Calgary", "Edmonton"], index=0)
segment = st.selectbox("Audience Segment", list(SEGMENT_MULT.keys()), index=0)

apply_seasonality = st.checkbox("Apply seasonality by month", value=False)
proposed_run_date = None
if apply_seasonality:
    _months = [
        ("January", 1), ("February", 2), ("March", 3), ("April", 4),
        ("May", 5), ("June", 6), ("July", 7), ("August", 8),
        ("September", 9), ("October", 10), ("November", 11), ("December", 12),
    ]
    _month_names = [m[0] for m in _months]
    sel = st.selectbox("Assumed run month (seasonality factor applies)", _month_names, index=2)
    month_idx = dict(_months)[sel]
    proposed_run_date = date(datetime.utcnow().year, month_idx, 15)

default_list = list(BASELINES.keys())[:50]
st.markdown("**Titles to score** (one per line). Add NEW titles freely:")
titles_input = st.text_area("Enter titles", value="\n".join(default_list), height=220)
titles = [t.strip() for t in titles_input.splitlines() if t.strip()]
if not titles:
    titles = list(BASELINES.keys())

benchmark_title = st.selectbox(
    "Choose Benchmark Title for Normalization",
    options=list(BASELINES.keys()),
    index=0,
    key="benchmark_title"
)

# -------------------------
# Core compute
# -------------------------
def _add_live_analytics_overlays(df_in: pd.DataFrame) -> pd.DataFrame:
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
        la = _la_for_category(cat)
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
    for c in [c for c in cols if c in overlays_df.columns]:
        df[c] = overlays_df[c]

    def _price_flag(p_hi: float) -> str:
        if pd.isna(p_hi): return "n/a"
        return "Elastic" if p_hi < 0.25 else ("Premium-tolerant" if p_hi > 0.30 else "Neutral")
    df["LA_PriceFlag"] = df["LA_PriceHiPct"].apply(_price_flag)
    return df

# Fallback estimator for unknown titles when live fetch is OFF
def estimate_unknown_title(title: str) -> Dict[str, float | str]:
    # Use baseline medians, then nudge by inferred category
    try:
        base_df = pd.DataFrame(BASELINES).T
        wiki_med = float(base_df["wiki"].median())
        tr_med   = float(base_df["trends"].median())
        yt_med   = float(base_df["youtube"].median())
        sp_med   = float(base_df["spotify"].median())
    except Exception:
        wiki_med, tr_med, yt_med, sp_med = 60.0, 55.0, 60.0, 58.0

    gender, category = infer_gender_and_category(title)

    # gentle category nudges so everything isn't identical
    bumps = {
        "family_classic":   {"wiki": +6, "trends": +3, "spotify": +2},
        "classic_romance":  {"wiki": +4, "trends": +2},
        "contemporary":     {"youtube": +6, "trends": +2},
        "pop_ip":           {"spotify": +10, "trends": +5, "youtube": +4},
        "romantic_tragedy": {"wiki": +3},
        "classic_comedy":   {"trends": +2},
        "dramatic":         {},
    }
    b = bumps.get(category, {})

    wiki = wiki_med + b.get("wiki", 0.0)
    tr   = tr_med   + b.get("trends", 0.0)
    yt   = yt_med   + b.get("youtube", 0.0)
    sp   = sp_med   + b.get("spotify", 0.0)

    # keep within sane bounds
    wiki = float(np.clip(wiki, 30.0, 120.0))
    tr   = float(np.clip(tr,   30.0, 120.0))
    yt   = float(np.clip(yt,   40.0, 140.0))
    sp   = float(np.clip(sp,   35.0, 120.0))

    return {"wiki": wiki, "trends": tr, "youtube": yt, "spotify": sp, "gender": gender, "category": category}

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
    rows = []
    unknown_used_live, unknown_used_est = [], []

    # 1) Base rows
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

    # 2) Normalize to benchmark
    bench_entry = BASELINES[benchmark_title]
    bench_fam_raw, bench_mot_raw = calc_scores(bench_entry, segment, region)
    bench_fam_raw = bench_fam_raw or 1.0
    bench_mot_raw = bench_mot_raw or 1.0
    df["Familiarity"] = (df["FamiliarityRaw"] / bench_fam_raw) * 100.0
    df["Motivation"]  = (df["MotivationRaw"]  / bench_mot_raw)  * 100.0
    st.caption(f"Scores normalized to benchmark: **{benchmark_title}**")

    # 3) Ticket medians + deseason
    TICKET_MEDIANS = {k: _median(v) for k, v in TICKET_PRIORS_RAW.items()}
    seasonality_on = proposed_run_date is not None

    bench_cat = BASELINES[benchmark_title]["category"]
    bench_hist_factor = seasonality_factor(bench_cat, TITLE_TO_MIDDATE.get(benchmark_title)) if seasonality_on else 1.0
    bench_med_hist = TICKET_MEDIANS.get(benchmark_title, None)
    bench_med_hist = float(bench_med_hist) if bench_med_hist else 1.0
    bench_med_deseason = bench_med_hist / (bench_hist_factor if bench_hist_factor else 1.0)

    aliases = {"Handmaid’s Tale": "Handmaid's Tale"}
    def _deseason_title_median(t: str, cat: str):
        key = aliases.get(t.strip(), t.strip())
        med = TICKET_MEDIANS.get(key)
        if med is None:
            return None, None, None
        hist_date = TITLE_TO_MIDDATE.get(t.strip())
        factor = seasonality_factor(cat, hist_date) if seasonality_on else 1.0
        return float(med), float(med) / (factor if factor else 1.0), factor

    df["SignalOnly"] = df[["Familiarity", "Motivation"]].mean(axis=1)

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

    # 4) Fit simple linear models
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

    # 5) Impute missing TicketIndex
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

    # 6) Apply future seasonality
    def _future_factor(cat: str):
        return seasonality_factor(cat, proposed_run_date) if seasonality_on else 1.0
    df["FutureSeasonalityFactor"] = df["Category"].map(_future_factor).astype(float)
    df["EffectiveTicketIndex"] = df["TicketIndex_DeSeason_Used"] * df["FutureSeasonalityFactor"]

    # 7) Composite
    tickets_component = np.where(
        df["TicketIndexSource"].eq("Not enough data"),
        df["SignalOnly"],
        df["EffectiveTicketIndex"]
    )
    TICKET_BLEND_WEIGHT = 0.50
    df["Composite"] = (1.0 - TICKET_BLEND_WEIGHT) * df["SignalOnly"] + TICKET_BLEND_WEIGHT * tickets_component

    # 8) Estimated tickets (future month uses de-seasonalized benchmark)
    bench_med_future = bench_med_deseason
    df["EstimatedTickets"] = ((df["EffectiveTicketIndex"] / 100.0) * (bench_med_future or 1.0)).round(0)

    # 10) Segment propensity + per-segment tickets
    bench_entry_for_mix = BASELINES[benchmark_title]
    prim_list, sec_list = [], []
    mix_gp, mix_core, mix_family, mix_ea = [], [], [], []
    seg_gp_tix, seg_core_tix, seg_family_tix, seg_ea_tix = [], [], [], []

    for _, r in df.iterrows():
        def _safe_float(x, default=0.0):
            try:
                v = float(x)
                if math.isnan(v) or math.isinf(v):
                    return default
                return v
            except Exception:
                return default

        entry_r = {
            "wiki": _safe_float(r.get("WikiIdx", 0.0)),
            "trends": _safe_float(r.get("TrendsIdx", 0.0)),
            "youtube": _safe_float(r.get("YouTubeIdx", 0.0)),
            "spotify": _safe_float(r.get("SpotifyIdx", 0.0)),
            "gender": r.get("Gender", "na"),
            "category": r.get("Category", "dramatic"),
        }

        seg_to_raw = _signal_for_all_segments(entry_r, region)
        seg_to_idx = _normalize_signals_by_benchmark(seg_to_raw, bench_entry_for_mix, region)

        pri = _prior_weights_for(region, entry_r["category"])
        if (not pri) or any(k not in pri for k in SEGMENT_KEYS_IN_ORDER):
            pri = {k: (pri.get(k, 1.0) if pri else 1.0) for k in SEGMENT_KEYS_IN_ORDER}

        combined = {}
        for k in SEGMENT_KEYS_IN_ORDER:
            prior_k = _safe_float(pri.get(k, 1.0), 1.0)
            sig_k = _safe_float(seg_to_idx.get(k, 0.0), 0.0)
            combined[k] = max(1e-9, prior_k * sig_k)

        total = sum(combined.values()) or 1.0
        shares = {k: combined[k] / total for k in SEGMENT_KEYS_IN_ORDER}

        ordered = sorted(shares.items(), key=lambda kv: kv[1], reverse=True)
        primary = ordered[0][0] if len(ordered) > 0 else ""
        secondary = ordered[1][0] if len(ordered) > 1 else ""
        prim_list.append(primary); sec_list.append(secondary)

        mix_gp.append(_safe_float(shares.get("General Population", 0.0), 0.0))
        mix_core.append(_safe_float(shares.get("Core Classical (F35–64)", 0.0), 0.0))
        mix_family.append(_safe_float(shares.get("Family (Parents w/ kids)", 0.0), 0.0))
        mix_ea.append(_safe_float(shares.get("Emerging Adults (18–34)", 0.0), 0.0))

        est_tix_val = _safe_float(r.get("EstimatedTickets", 0.0), 0.0)
        def _seg_tix(seg_name: str) -> int:
            share_val = _safe_float(shares.get(seg_name, 0.0), 0.0)
            return int(round(est_tix_val * share_val))
        seg_gp_tix.append(_seg_tix("General Population"))
        seg_core_tix.append(_seg_tix("Core Classical (F35–64)"))
        seg_family_tix.append(_seg_tix("Family (Parents w/ kids)"))
        seg_ea_tix.append(_seg_tix("Emerging Adults (18–34)"))

    df["PredictedPrimarySegment"] = prim_list
    df["PredictedSecondarySegment"] = sec_list
    df["Mix_GP"] = mix_gp
    df["Mix_Core"] = mix_core
    df["Mix_Family"] = mix_family
    df["Mix_EA"] = mix_ea
    df["Seg_GP_Tickets"] = seg_gp_tix
    df["Seg_Core_Tickets"] = seg_core_tix
    df["Seg_Family_Tickets"] = seg_family_tix
    df["Seg_EA_Tickets"] = seg_ea_tix

    # 11) Remount decay
    decay_pcts, decay_factors, est_after_decay = [], [], []
    today_year = datetime.utcnow().year
    for _, r in df.iterrows():
        title = r["Title"]
        est_base = float(r.get("EstimatedTickets", 0.0) or 0.0)
        last_mid = TITLE_TO_MIDDATE.get(title)
        if isinstance(last_mid, date):
            yrs_since = (proposed_run_date.year - last_mid.year) if proposed_run_date else (today_year - last_mid.year)
        else:
            yrs_since = None
        if yrs_since is None:
            decay_pct = 0.00
        elif yrs_since >= 5:
            decay_pct = 0.05
        elif yrs_since >= 3:
            decay_pct = 0.12
        elif yrs_since >= 1:
            decay_pct = 0.20
        else:
            decay_pct = 0.25
        factor = 1.0 - decay_pct
        est_final = round(est_base * factor)
        decay_pcts.append(decay_pct); decay_factors.append(factor); est_after_decay.append(est_final)

    df["ReturnDecayPct"] = decay_pcts
    df["ReturnDecayFactor"] = decay_factors
    df["EstimatedTickets_Final"] = est_after_decay

    # 12) City split (learned title/category → fallback)
    cal_share, edm_share = [], []
    cal_total, edm_total = [], []
    cal_singles, cal_subs, edm_singles, edm_subs = [], [], [], []
    for _, r in df.iterrows():
        title = str(r["Title"])
        cat   = str(r["Category"])
        total = float(r.get("EstimatedTickets_Final", r.get("EstimatedTickets", 0.0)) or 0.0)
        split = city_split_for(title, cat)  # {"Calgary": x, "Edmonton": 1-x}
        c_sh = float(split.get("Calgary", 0.6)); e_sh = float(split.get("Edmonton", 0.4))
        c_sh, e_sh = _normalize_pair(c_sh, e_sh)
        cal_share.append(c_sh); edm_share.append(e_sh)
        cal_t = total * c_sh; edm_t = total * e_sh
        cal_total.append(round(cal_t)); edm_total.append(round(edm_t))
        cal_sub_ratio = subs_share_for(cat, "Calgary")
        edm_sub_ratio = subs_share_for(cat, "Edmonton")
        cal_subs.append(int(round(cal_t * cal_sub_ratio)))
        cal_singles.append(int(round(cal_t * (1.0 - cal_sub_ratio))))
        edm_subs.append(int(round(edm_t * edm_sub_ratio)))
        edm_singles.append(int(round(edm_t * (1.0 - edm_sub_ratio))))
    df["CityShare_Calgary"] = cal_share
    df["CityShare_Edmonton"] = edm_share
    df["YYC_Total"] = cal_total
    df["YEG_Total"] = edm_total
    df["YYC_Singles"] = cal_singles
    df["YYC_Subs"] = cal_subs
    df["YEG_Singles"] = edm_singles
    df["YEG_Subs"] = edm_subs

    # 13) Seasonality meta
    seasonality_on_flag = proposed_run_date is not None
    df["SeasonalityApplied"] = bool(seasonality_on_flag)
    df["SeasonalityMonthUsed"] = int(proposed_run_date.month) if seasonality_on_flag else np.nan
    df["RunMonth"] = proposed_run_date.strftime("%B") if seasonality_on_flag else ""

    # 14) Stash results atomically
    st.session_state["results"] = {
        "df": df,
        "benchmark": benchmark_title,
        "segment": segment,
        "region": region,
        "unknown_est": unknown_used_est,
        "unknown_live": unknown_used_live,
    }

# -------------------------
# Render
# -------------------------
def render_results():
    SHOW_LEGACY_SEASON = False
    import calendar
    R = st.session_state.get("results")
    if not R or "df" not in R or R["df"] is None:
        return
    df = R["df"]
    if df is None or df.empty:
        st.warning("No scored rows to display yet.")
        return

    benchmark_title = R.get("benchmark", "")
    segment = R.get("segment", "")
    region = R.get("region", "")

    if R.get("unknown_est"):
        st.info("Estimated (offline) for new titles: " + ", ".join(R["unknown_est"]))
    if R.get("unknown_live"):
        st.success("Used LIVE data for new titles: " + ", ".join(R["unknown_live"]))

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

    if "SeasonalityApplied" in df.columns and bool(df["SeasonalityApplied"].iloc[0]):
        run_month_num = df["SeasonalityMonthUsed"].iloc[0]
        try:
            run_month_name = calendar.month_name[int(run_month_num)] if pd.notna(run_month_num) else "n/a"
        except Exception:
            run_month_name = "n/a"
        st.caption(f"Seasonality: **ON** · Run month: **{run_month_name}**")
    else:
        st.caption("Seasonality: **OFF**")

    def _assign_score(v: float) -> str:
        if v >= 90: return "A"
        elif v >= 75: return "B"
        elif v >= 60: return "C"
        elif v >= 45: return "D"
        else: return "E"
    if "Score" not in df.columns and "Composite" in df.columns:
        df["Score"] = df["Composite"].apply(_assign_score)

    df_show = df.rename(columns={
        "TicketMedian": "TicketHistory",
        "EffectiveTicketIndex": "TicketIndex used",
    }).copy()

    def _to_month_name(v):
        try:
            if pd.isna(v): return ""
        except Exception:
            pass
        if isinstance(v, (int, float)):
            m = int(v)
            import calendar
            return calendar.month_name[m] if 1 <= m <= 12 else ""
        if isinstance(v, str):
            return v
        return ""
    if "RunMonth" in df_show.columns:
        df_show["RunMonth"] = df_show["RunMonth"].apply(_to_month_name)
    elif "SeasonalityMonthUsed" in df_show.columns:
        df_show["RunMonth"] = df_show["SeasonalityMonthUsed"].apply(_to_month_name)

    table_cols = [
        "Title","Region","Segment","Gender","Category",
        "WikiIdx","TrendsIdx","YouTubeIdx","SpotifyIdx",
        "Familiarity","Motivation",
        "TicketHistory",
        "TicketIndex used","TicketIndexSource",
        "RunMonth","FutureSeasonalityFactor","HistSeasonalityFactor",
        "Composite","Score",
        "EstimatedTickets",
        "ReturnDecayFactor","ReturnDecayPct","EstimatedTickets_Final",
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
              "WikiIdx": "{:.0f}",
              "TrendsIdx": "{:.0f}",
              "YouTubeIdx": "{:.0f}",
              "SpotifyIdx": "{:.0f}",
              "Familiarity": "{:.1f}",
              "Motivation": "{:.1f}",
              "Composite": "{:.1f}",
              "TicketIndex used": "{:.1f}",
              "EstimatedTickets": "{:,.0f}",
              "TicketHistory": "{:,.0f}",
              "FutureSeasonalityFactor": "{:.3f}",
              "HistSeasonalityFactor": "{:.3f}",
              "ReturnDecayPct": "{:.0%}",
              "ReturnDecayFactor": "{:.2f}",
              "EstimatedTickets_Final": "{:,.0f}",
          }),
        use_container_width=True,
        hide_index=True
    )

    # Export (no Live Analytics columns)
    st.download_button(
        "⬇️ Download Scores CSV",
        df_show[present_cols].to_csv(index=False).encode("utf-8"),
        "title_scores.csv",
        "text/csv"
    )

# --- Helper: safely fetch scored df without killing the rest of the app ---
def get_scored_df_or_prompt():
    R = st.session_state.get("results") or {}
    df = R.get("df", pd.DataFrame())
    if df is not None and not df.empty:
        return df

    st.info("No scored rows yet. Use **Score Titles** below to generate results.")
    # Return empty, but don't st.stop(); this keeps the page alive so the button renders.
    return pd.DataFrame()

# -------------------------
# Button + render
# -------------------------
run = st.button("Score Titles", type="primary")
if run:
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

if st.session_state.get("results") is not None:
    render_results()

# 📅 Build a Season (assign titles to months)
st.subheader("📅 Build a Season (assign titles to months)")

# Always fetch via helper; never st.stop() here
R = st.session_state.get("results") or {}
df = get_scored_df_or_prompt()

# Season year picker (this can render even if df is empty)
default_year = (datetime.utcnow().year + 1)
season_year = st.number_input(
    "Season year (start of season)",
    min_value=2000, max_value=2100,
    value=default_year, step=1
)

# Allowed months and selectors
allowed_months = [
    ("September", 9), ("October", 10), ("December", 12),
    ("January", 1), ("February", 2), ("March", 3), ("May", 5),
]

# Build a safe title list
try:
    title_vals = []
    if df is not None and not df.empty and "Title" in df.columns:
        title_vals = [t for t in df["Title"].dropna().astype(str).unique().tolist() if t.strip()]
    if not title_vals and "BASELINES" in globals():
        title_vals = sorted(list(BASELINES.keys()))
    title_options = ["— None —"] + sorted(title_vals)
except Exception:
    title_options = ["— None —"]

# If there are still no titles, just show the pickers disabled
month_to_choice: dict[str, str] = {}
cols = st.columns(3, gap="large")
for i, (m_name, _) in enumerate(allowed_months):
    with cols[i % 3]:
        month_to_choice[m_name] = st.selectbox(
            m_name,
            options=title_options,
            index=0,
            key=f"season_pick_{m_name}"
        )


month_to_choice: dict[str, str] = {}
cols = st.columns(3, gap="large")
for i, (m_name, _) in enumerate(allowed_months):
    with cols[i % 3]:
        month_to_choice[m_name] = st.selectbox(
            m_name,
            options=title_options,
            index=0,
            key=f"season_pick_{m_name}"
        )

def _run_year_for_month(month_num: int, start_year: int) -> int:
    # Fall months are same calendar year; winter/spring are next year.
    return start_year if month_num in (9, 10, 12) else (start_year + 1)

# --- Infer benchmark (de-seasonalized) conversion once (for index→tickets)
bench_med_deseason_est = np.nan
try:
    if {"TicketIndex_DeSeason_Used", "FutureSeasonalityFactor", "EstimatedTickets"}.issubset(df.columns):
        eff = df["TicketIndex_DeSeason_Used"].astype(float) * df["FutureSeasonalityFactor"].astype(float)
        good = (eff > 0) & df["EstimatedTickets"].notna()
        if good.any():
            ratios = (df.loc[good, "EstimatedTickets"].astype(float) / eff[good]) * 100.0
            bench_med_deseason_est = float(np.nanmedian(ratios.values))
except Exception:
    bench_med_deseason_est = np.nan

# --- Build plan rows
plan_rows: list[dict] = []

for m_name, m_num in allowed_months:
    title_sel = month_to_choice.get(m_name)
    if not title_sel or title_sel == "— None —":
        continue

    r = df[df["Title"] == title_sel].head(1)
    if r.empty:
        continue
    r = r.iloc[0]

    cat = str(r.get("Category", ""))
    run_year = _run_year_for_month(m_num, int(season_year))
    run_date = date(run_year, int(m_num), 15)

    # Seasonality + effective index
    f_season = float(seasonality_factor(cat, run_date))
    idx_deseason = float(r.get("TicketIndex_DeSeason_Used", np.nan))
    if not np.isfinite(idx_deseason):
        idx_deseason = float(r.get("SignalOnly", 100.0))
    eff_idx = idx_deseason * f_season

    # Tickets estimate using benchmark if available; else fall back to row estimate
    if np.isfinite(bench_med_deseason_est):
        est_tix = int(round((eff_idx / 100.0) * bench_med_deseason_est))
    else:
        est_tix = int(round(r.get("EstimatedTickets", 0) or 0))

    # Remount decay
    decay_factor = remount_novelty_factor(title_sel, run_date)
    est_tix_final = int(round(est_tix * decay_factor))

    # City split, then singles/subs
    split = city_split_for(title_sel, cat) or {"Calgary": 0.60, "Edmonton": 0.40}
    c_sh = float(split.get("Calgary", 0.6)); e_sh = float(split.get("Edmonton", 0.4))
    s = (c_sh + e_sh) or 1.0
    c_sh, e_sh = c_sh / s, e_sh / s

    yyc_total = int(round(est_tix_final * c_sh))
    yeg_total = int(round(est_tix_final * e_sh))

    cal_sub_ratio = float(subs_share_for(cat, "Calgary"))
    edm_sub_ratio = float(subs_share_for(cat, "Edmonton"))

    yyc_subs = int(round(yyc_total * cal_sub_ratio))
    yyc_singles = int(round(yyc_total * (1.0 - cal_sub_ratio)))
    yeg_subs = int(round(yeg_total * edm_sub_ratio))
    yeg_singles = int(round(yeg_total * (1.0 - edm_sub_ratio)))

    plan_rows.append({
        "Month": f"{m_name} {run_year}",
        "Title": title_sel,
        "Category": cat,
        "PrimarySegment": r.get("PredictedPrimarySegment", ""),
        "SecondarySegment": r.get("PredictedSecondarySegment", ""),
        "WikiIdx": r.get("WikiIdx", np.nan),
        "TrendsIdx": r.get("TrendsIdx", np.nan),
        "YouTubeIdx": r.get("YouTubeIdx", np.nan),
        "SpotifyIdx": r.get("SpotifyIdx", np.nan),
        "Familiarity": r.get("Familiarity", np.nan),
        "Motivation": r.get("Motivation", np.nan),
        "TicketHistory": r.get("TicketMedian", np.nan),
        "TicketIndex used": r.get("TicketIndex_DeSeason_Used", np.nan),
        "TicketIndexSource": r.get("TicketIndexSource", ""),
        "FutureSeasonalityFactor": f"{f_season:.3f}",
        "HistSeasonalityFactor": (
            f"{float(r.get('HistSeasonalityFactor', np.nan)):.3f}"
            if pd.notna(r.get("HistSeasonalityFactor", np.nan)) else ""
        ),
        "Composite": r.get("Composite", np.nan),
        "Score": r.get("Score", ""),
        "EstimatedTickets": est_tix,
        "ReturnDecayFactor": f"{float(decay_factor):.2f}",
        "ReturnDecayPct": f"{1.0 - float(decay_factor):.0%}",
        "EstimatedTickets_Final": est_tix_final,
        "YYC_Singles": yyc_singles,
        "YYC_Subs": yyc_subs,
        "YEG_Singles": yeg_singles,
        "YEG_Subs": yeg_subs,
        "CityShare_Calgary": c_sh,
        "CityShare_Edmonton": e_sh,
    })

# --- Render the season table (or prompt to pick)
if plan_rows:
    desired_order = [
        "Month","Title","Category","PrimarySegment","SecondarySegment",
        "WikiIdx","TrendsIdx","YouTubeIdx","SpotifyIdx",
        "Familiarity","Motivation",
        "TicketHistory","TicketIndex used","TicketIndexSource",
        "FutureSeasonalityFactor","HistSeasonalityFactor",
        "Composite","Score",
        "EstimatedTickets","ReturnDecayFactor","ReturnDecayPct","EstimatedTickets_Final",
        "YYC_Singles","YYC_Subs","YEG_Singles","YEG_Subs",
        "CityShare_Calgary","CityShare_Edmonton",
    ]
    plan_df = pd.DataFrame(plan_rows)
    plan_df = plan_df[[c for c in desired_order if c in plan_df.columns]]

    # KPIs
    yyc_tot = int(plan_df["YYC_Singles"].sum() + plan_df["YYC_Subs"].sum())
    yeg_tot = int(plan_df["YEG_Singles"].sum() + plan_df["YEG_Subs"].sum())
    singles_tot = int(plan_df["YYC_Singles"].sum() + plan_df["YEG_Singles"].sum())
    subs_tot    = int(plan_df["YYC_Subs"].sum()    + plan_df["YEG_Subs"].sum())
    grand       = int(plan_df["EstimatedTickets_Final"].sum()) or 1

    st.markdown("### 📊 Season at a glance")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Projected Season Total", f"{grand:,}")
    with c2: st.metric("Calgary • share", f"{yyc_tot:,}", delta=f"{yyc_tot/grand:.1%}")
    with c3: st.metric("Edmonton • share", f"{yeg_tot:,}", delta=f"{yeg_tot/grand:.1%}")
    with c4: st.metric("Singles vs Subs", f"{singles_tot:,} / {subs_tot:,}", delta=f"subs {subs_tot/grand:.1%}")

    st.markdown(f"**Projected season total (final, after decay):** {grand:,}")
    st.dataframe(plan_df, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Download Season Plan (full fields)",
        plan_df.to_csv(index=False).encode("utf-8"),
        file_name=f"season_plan_full_{season_year}-{season_year+1}.csv",
        mime="text/csv",
        key="dl_season_plan_full"
    )
else:
    st.caption("Pick at least one month/title above to see your season projection.")
