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

# -------------------------
# App setup
# -------------------------
st.set_page_config(page_title="Alberta Ballet — Title Familiarity & Motivation Scorer", layout="wide")
if "results" not in st.session_state:
    st.session_state["results"] = None

st.title("🎭 Alberta Ballet — Title Familiarity & Motivation Scorer (v9.2)")
st.caption("Hard-coded AB-wide baselines (normalized to your benchmark = 100). Add new titles; choose live fetch or offline estimate.")

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
    """Expects rows with Title, Category, City, Singles, Subs (or Total/Type). Flexible and guarded."""
    global TITLE_CITY_PRIORS, CATEGORY_CITY_PRIORS, SUBS_SHARE_BY_CATEGORY_CITY
    TITLE_CITY_PRIORS.clear(); CATEGORY_CITY_PRIORS.clear()
    SUBS_SHARE_BY_CATEGORY_CITY = {"Calgary": {}, "Edmonton": {}}

    if hist_df is None or hist_df.empty:
        return {"titles_learned": 0, "categories_learned": 0, "subs_shares_learned": 0, "note": "empty history"}

    df = hist_df.copy()

    title_col   = _pick_col(df, ["title", "production", "show"])
    cat_col     = _pick_col(df, ["category", "segment", "genre"])
    city_col    = _pick_col(df, ["city"])
    singles_col = _pick_col(df, ["singles", "single_tickets", "single tickets"])
    subs_col    = _pick_col(df, ["subs", "subscriptions", "subscription_tickets", "sub tickets"])
    total_col   = _pick_col(df, ["total", "tickets", "total_tickets", "qty"])

    type_col = _pick_col(df, ["type"])         # if you have rows with Type in {Single, Sub}
    qty_col  = _pick_col(df, ["qty","quantity","tickets","count"])

    # If Type/Qty rows, pivot to wide
    if type_col and qty_col and not (singles_col and subs_col):
        need = [c for c in [title_col, cat_col, city_col, type_col, qty_col] if c]
        df = df[need].copy()
        df[city_col] = df[city_col].map(_canon_city)
        df = df[df[city_col].isin(["Calgary", "Edmonton"])]
        df[type_col] = df[type_col].str.lower().str.strip()
        df_piv = df.pivot_table(
            index=[c for c in [title_col, cat_col, city_col] if c],
            columns=type_col, values=qty_col, aggfunc="sum", fill_value=0
        ).reset_index()
        singles_col = "single" if "single" in df_piv.columns else ("singles" if "singles" in df_piv.columns else None)
        subs_col    = "sub"    if "sub"    in df_piv.columns else ("subs"    if "subs"    in df_piv.columns    else None)
        df = df_piv

    # Must have these three
    for c in (title_col, cat_col, city_col):
        if c is None:
            return {"titles_learned": 0, "categories_learned": 0, "subs_shares_learned": 0, "note": "missing title/category/city"}

    df = df[[title_col, cat_col, city_col] + [c for c in [singles_col, subs_col, total_col] if c]].copy()
    df[city_col] = df[city_col].map(_canon_city)
    df = df[df[city_col].isin(["Calgary", "Edmonton"])]

    # Build total if needed
    if total_col is None:
        df["_total"] = df[[c for c in [singles_col, subs_col] if c]].fillna(0).sum(axis=1)
        total_col = "_total"

    # Title-level city splits
    split_rows = df.groupby([title_col, city_col])[total_col].sum().reset_index()
    title_splits = split_rows.pivot(index=title_col, columns=city_col, values=total_col).fillna(0)
    titles_learned = 0
    for t, row in title_splits.iterrows():
        tot = float(row.get("Calgary", 0) + row.get("Edmonton", 0))
        if tot <= 0: continue
        cal = max(_CITY_CLIP_RANGE[0], min(_CITY_CLIP_RANGE[1], float(row.get("Calgary", 0)) / tot))
        TITLE_CITY_PRIORS[str(t)] = {"Calgary": cal, "Edmonton": 1.0 - cal}
        titles_learned += 1

    # Category-level city splits
    cat_rows = df.groupby([cat_col, city_col])[total_col].sum().reset_index()
    cat_splits = cat_rows.pivot(index=cat_col, columns=city_col, values=total_col).fillna(0)
    categories_learned = 0
    for c, row in cat_splits.iterrows():
        tot = float(row.get("Calgary", 0) + row.get("Edmonton", 0))
        if tot <= 0: continue
        cal = max(_CITY_CLIP_RANGE[0], min(_CITY_CLIP_RANGE[1], float(row.get("Calgary", 0)) / tot))
        CATEGORY_CITY_PRIORS[str(c)] = {"Calgary": cal, "Edmonton": 1.0 - cal}
        categories_learned += 1

    # Subs share by category × city
    subs_shares_learned = 0
    if subs_col or singles_col:
        df["_subs"]    = df[subs_col].fillna(0) if subs_col else 0
        df["_singles"] = df[singles_col].fillna(0) if singles_col else 0
        agg = df.groupby([cat_col, city_col])[["_subs", "_singles"]].sum().reset_index()
        for _, r in agg.iterrows():
            tot = float(r["_subs"] + r["_singles"])
            if tot <= 0: continue
            share = float(r["_subs"]) / tot
            city  = str(r[city_col]); catv = str(r[cat_col])
            SUBS_SHARE_BY_CATEGORY_CITY.setdefault(city, {})
            SUBS_SHARE_BY_CATEGORY_CITY[city][catv] = max(0.05, min(0.95, share))
            subs_shares_learned += 1

    return {
        "titles_learned": titles_learned,
        "categories_learned": categories_learned,
        "subs_shares_learned": subs_shares_learned,
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

# --- UI: upload or use local history, then learn priors ---
with st.expander("Historicals (optional): upload or use data/history.csv", expanded=False):
    uploaded_hist = st.file_uploader("Upload historical ticket CSV", type=["csv"], key="hist_uploader_v9")

if "hist_df" not in st.session_state:
    try:
        local_hist = pd.read_csv("data/history_city_sales.csv")
    except FileNotFoundError:
        local_hist = pd.DataFrame()
    st.session_state["hist_df"] = pd.read_csv(uploaded_hist) if uploaded_hist else local_hist
    if not st.session_state["hist_df"].empty:
    st.caption(f"Historicals loaded: {len(st.session_state['hist_df'])} rows from CSV")

elif uploaded_hist is not None:
    st.session_state["hist_df"] = pd.read_csv(uploaded_hist)

if "priors_summary" not in st.session_state or uploaded_hist is not None:
    s = learn_priors_from_history(st.session_state["hist_df"])
    st.session_state["priors_summary"] = s

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

# LA overlays by category
LA_BY_CATEGORY = {
    "pop_ip": {"Presale":12,"FirstDay":7,"FirstWeek":5,"WeekOf":20,"Internet":58,"Mobile":41,"Phone":1,"Tix_1_2":73,"Tix_3_4":22,"Tix_5_8":5,"Premium":4.5,"LT10mi":71,
               "Price_Low":18,"Price_Fair":32,"Price_Good":24,"Price_VeryGood":16,"Price_Best":10,
               "Age":49.6,"Age_18_24":8,"Age_25_34":5,"Age_35_44":26,"Age_45_54":20,"Age_55_64":26,"Age_65_plus":15},
    "classic_romance": {"Presale":10,"FirstDay":6,"FirstWeek":4,"WeekOf":20,"Internet":55,"Mobile":44,"Phone":1,"Tix_1_2":69,"Tix_3_4":25,"Tix_5_8":5,"Premium":4.1,"LT10mi":69,
                        "Price_Low":20,"Price_Fair":29,"Price_Good":24,"Price_VeryGood":16,"Price_Best":10},
    "classic_comedy": {"Presale":10,"FirstDay":6,"FirstWeek":4,"WeekOf":20,"Internet":55,"Mobile":44,"Phone":1,"Tix_1_2":69,"Tix_3_4":25,"Tix_5_8":5,"Premium":4.1,"LT10mi":69,
                       "Price_Low":20,"Price_Fair":29,"Price_Good":24,"Price_VeryGood":16,"Price_Best":10},
    "romantic_comedy": {"Presale":10,"FirstDay":6,"FirstWeek":4,"WeekOf":20,"Internet":55,"Mobile":44,"Phone":1,"Tix_1_2":69,"Tix_3_4":25,"Tix_5_8":5,"Premium":4.1,"LT10mi":69,
                        "Price_Low":20,"Price_Fair":29,"Price_Good":24,"Price_VeryGood":16,"Price_Best":10},
    "contemporary": {"Presale":11,"FirstDay":6,"FirstWeek":5,"WeekOf":21,"Internet":59,"Mobile":40,"Phone":1,"Tix_1_2":72,"Tix_3_4":23,"Tix_5_8":5,"Premium":3.9,"LT10mi":70,
                     "Price_Low":18,"Price_Fair":32,"Price_Good":25,"Price_VeryGood":15,"Price_Best":11},
    "family_classic": {"Presale":10,"FirstDay":6,"FirstWeek":5,"WeekOf":19,"Internet":53,"Mobile":45,"Phone":1,"Tix_1_2":67,"Tix_3_4":27,"Tix_5_8":6,"Premium":4.0,"LT10mi":66,
                       "Price_Low":21,"Price_Fair":28,"Price_Good":24,"Price_VeryGood":17,"Price_Best":10},
    "romantic_tragedy": {"Presale":11,"FirstDay":6,"FirstWeek":5,"WeekOf":20,"Internet":59,"Mobile":39,"Phone":2,"Tix_1_2":72,"Tix_3_4":22,"Tix_5_8":5,"Premium":4.2,"LT10mi":71,
                         "Price_Low":18,"Price_Fair":28,"Price_Good":27,"Price_VeryGood":16,"Price_Best":11},
    "dramatic": {"Presale":12,"FirstDay":7,"FirstWeek":5,"WeekOf":20,"Internet":54,"Mobile":44,"Phone":1,"Tix_1_2":72,"Tix_3_4":23,"Tix_5_8":5,"Premium":4.3,"LT10mi":68,
                 "Price_Low":18,"Price_Fair":31,"Price_Good":24,"Price_VeryGood":16,"Price_Best":11},
}

def _la_for_category(cat: str) -> dict:
    return LA_BY_CATEGORY.get(str(cat), {})

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

    # 9) Overlays
    df = _add_live_analytics_overlays(df)

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
# Deep LA report attachment
# -------------------------
LA_DEEP_BY_CATEGORY = {
    # (trim to what you need; keeping two for demo)
    "pop_ip": {
        "DA_Customers": 15861, "DA_CustomersWithLiveEvents": 13813, "DA_CustomersWithDemo_CAN": 14995,
        "CES_Gender_Male_pct": 44, "CES_Gender_Female_pct": 56, "CES_Age_Mean": 49.6,
        "CES_Age_18_24_pct": 8, "CES_Age_25_34_pct": 5, "CES_Age_35_44_pct": 26, "CES_Age_45_54_pct": 20,
        "CES_Age_55_64_pct": 26, "CES_Age_65_plus_pct": 15,
        "CEP_NeverPurchased_pct": 25.9, "CEP_1Event_pct": 43.2, "CEP_2_3Events_pct": 22.0,
        "CEP_4_5Events_pct": 5.7, "CEP_6_10Events_pct": 2.8, "CEP_11plusEvents_pct": 0.3,
        "CESpend_le_250_pct": 64.7, "CESpend_251_500_pct": 19.8, "CESpend_501_1000_pct": 11.2,
        "CESpend_1001_2000_pct": 3.6, "CESpend_2001plus_pct": 0.6,
        "SOW_ClientEvents_pct": 26, "SOW_NonClientEvents_pct": 74,
        "LS_ClientEvents_usd": 268, "LS_NonClientEvents_usd": 2207,
        "TPE_ClientEvents": 2.3, "TPE_NonClientEvents": 2.6,
        "SPE_ClientEvents_usd": 187, "SPE_NonClientEvents_usd": 209,
        "ATP_ClientEvents_usd": 83, "ATP_NonClientEvents_usd": 84,
        "DIM_Generation_Millennials_pct": 24, "DIM_Generation_X_pct": 34, "DIM_Generation_Babyboomers_pct": 31, "DIM_Generation_Z_pct": 5,
        "DIM_Occ_Professionals_pct": 64, "DIM_Occ_SelfEmployed_pct": 0, "DIM_Occ_Retired_pct": 3, "DIM_Occ_Students_pct": 0,
        "DIM_Household_WorkingMoms_pct": 8, "DIM_Financial_Affluent_pct": 9,
        "DIM_Live_Major_Concerts_pct": 47, "DIM_Live_Major_Arts_pct": 76, "DIM_Live_Major_Sports_pct": 27, "DIM_Live_Major_Family_pct": 7,
        "DIM_Live_Major_MultiCategories_pct": 45, "DIM_Live_Freq_ActiveBuyers_pct": 60, "DIM_Live_Freq_RepeatBuyers_pct": 67,
        "DIM_Live_Timing_EarlyBuyers_pct": 37, "DIM_Live_Timing_LateBuyers_pct": 38,
        "DIM_Live_Product_PremiumBuyers_pct": 18, "DIM_Live_Product_AncillaryUpsellBuyers_pct": 2,
        "DIM_Live_Spend_HighSpenders_gt1k_pct": 5, "DIM_Live_Distance_Travelers_gt500mi_pct": 14,
    },
    "classic_romance": {
        "DA_Customers": 15294, "DA_CustomersWithLiveEvents": 11881, "DA_CustomersWithDemo_CAN": 12832,
        "CES_Gender_Male_pct": 47, "CES_Gender_Female_pct": 53, "CES_Age_Mean": 46.5,
        "CES_Age_18_24_pct": 14, "CES_Age_25_34_pct": 5, "CES_Age_35_44_pct": 23, "CES_Age_45_54_pct": 27,
        "CES_Age_55_64_pct": 19, "CES_Age_65_plus_pct": 13,
        "CEP_NeverPurchased_pct": 23.2, "CEP_1Event_pct": 39.1, "CEP_2_3Events_pct": 26.1,
        "CEP_4_5Events_pct": 7.3, "CEP_6_10Events_pct": 3.9, "CEP_11plusEvents_pct": 0.4,
        "CESpend_le_250_pct": 58.4, "CESpend_251_500_pct": 21.4, "CESpend_501_1000_pct": 13.8,
        "CESpend_1001_2000_pct": 5.4, "CESpend_2001plus_pct": 1.1,
        "SOW_ClientEvents_pct": 35, "SOW_NonClientEvents_pct": 65,
        "LS_ClientEvents_usd": 325, "LS_NonClientEvents_usd": 1832,
        "TPE_ClientEvents": 2.4, "TPE_NonClientEvents": 2.6,
        "SPE_ClientEvents_usd": 195, "SPE_NonClientEvents_usd": 212,
        "ATP_ClientEvents_usd": 82, "ATP_NonClientEvents_usd": 84,
        "DIM_Generation_Millennials_pct": 29, "DIM_Generation_X_pct": 37, "DIM_Generation_Babyboomers_pct": 26, "DIM_Generation_Z_pct": 5,
        "DIM_Occ_Professionals_pct": 83, "DIM_Occ_SelfEmployed_pct": 0, "DIM_Occ_Retired_pct": 0, "DIM_Occ_Students_pct": 0,
        "DIM_Household_WorkingMoms_pct": 13, "DIM_Financial_Affluent_pct": 9,
        "DIM_Live_Major_Concerts_pct": 38, "DIM_Live_Major_Arts_pct": 82, "DIM_Live_Major_Sports_pct": 25, "DIM_Live_Major_Family_pct": 8,
        "DIM_Live_Major_MultiCategories_pct": 42, "DIM_Live_Freq_ActiveBuyers_pct": 66, "DIM_Live_Freq_RepeatBuyers_pct": 66,
        "DIM_Live_Timing_EarlyBuyers_pct": 32, "DIM_Live_Timing_LateBuyers_pct": 38,
        "DIM_Live_Product_PremiumBuyers_pct": 18, "DIM_Live_Product_AncillaryUpsellBuyers_pct": 2,
        "DIM_Live_Spend_HighSpenders_gt1k_pct": 4, "DIM_Live_Distance_Travelers_gt500mi_pct": 12,
    },
}

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
    df_out = df.copy()
    for col, key in LA_DEEP_SCHEMA:
        vals = []
        for _, r in df_out.iterrows():
            cat = str(r.get("Category"))
            deep = LA_DEEP_BY_CATEGORY.get(cat, {})
            v = deep.get(key, np.nan)
            vals.append(v)
        df_out[col] = vals
    return df_out

# -------------------------
# Render
# -------------------------
def render_results():
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

    # Exports
    df_full = attach_la_report_columns(df)
    st.download_button(
        "⬇️ Download Scores CSV (table columns only)",
        df_show[present_cols].to_csv(index=False).encode("utf-8"),
        "title_scores_table_view.csv",
        "text/csv"
    )
    st.download_button(
        "⬇️ Download Full CSV (includes LA deep data where available)",
        df_full.to_csv(index=False).encode("utf-8"),
        "title_scores_full_with_LA.csv",
        "text/csv"
    )

    # Build a Season
    st.subheader("📅 Build a Season (assign titles to months)")
    default_year = (datetime.utcnow().year + 1)
    season_year = st.number_input("Season year (start of season)", min_value=2000, max_value=2100, value=default_year, step=1)

    # Infer benchmark tickets
    bench_med_deseason_est = None
    try:
        eff = df["TicketIndex_DeSeason_Used"].astype(float) * df["FutureSeasonalityFactor"].astype(float)
        good = (eff > 0) & df["EstimatedTickets"].notna()
        if good.any():
            ratios = (df.loc[good, "EstimatedTickets"].astype(float) / eff[good]) * 100.0
            bench_med_deseason_est = float(np.nanmedian(ratios.values))
    except Exception:
        bench_med_deseason_est = None
    if bench_med_deseason_est is None or not np.isfinite(bench_med_deseason_est):
        st.warning("Couldn’t infer benchmark tickets for conversion. Season projections will show index-only where needed.")
        bench_med_deseason_est = None

    allowed_months = [("September", 9), ("October", 10), ("December", 12), ("January", 1), ("February", 2), ("March", 3), ("May", 5)]
    title_options = ["— None —"] + sorted(df["Title"].unique().tolist())
    month_to_choice = {}
    cols = st.columns(3, gap="large")
    for i, (m_name, _) in enumerate(allowed_months):
        with cols[i % 3]:
            month_to_choice[m_name] = st.selectbox(m_name, options=title_options, index=0, key=f"season_pick_{m_name}")

    def _run_year_for_month(month_num: int, start_year: int) -> int:
        return start_year if month_num in (9, 10, 12) else (start_year + 1)

    plan_rows = []
    for m_name, m_num in allowed_months:
        title_sel = month_to_choice.get(m_name)
        if not title_sel or title_sel == "— None —":
            continue
        r = df[df["Title"] == title_sel].head(1)
        if r.empty:
            continue
        r = r.iloc[0]
        cat = str(r.get("Category", "dramatic"))
        run_year = _run_year_for_month(m_num, int(season_year))
        run_date = date(run_year, int(m_num), 15)
        f_season = float(seasonality_factor(cat, run_date))
        idx_deseason = float(r.get("TicketIndex_DeSeason_Used", np.nan))
        if not np.isfinite(idx_deseason):
            idx_deseason = float(r.get("SignalOnly", 100.0))
        eff_idx = idx_deseason * f_season
        if bench_med_deseason_est is not None and np.isfinite(bench_med_deseason_est):
            est_tix = round((eff_idx / 100.0) * bench_med_deseason_est)
        else:
            est_tix = np.nan
        decay_factor = remount_novelty_factor(title_sel, run_date)
        est_tix_final = int(round((est_tix if np.isfinite(est_tix) else 0) * decay_factor))
        mix_gp = float(r.get("Mix_GP", 0.0))
        mix_core = float(r.get("Mix_Core", 0.0))
        mix_family = float(r.get("Mix_Family", 0.0))
        mix_ea = float(r.get("Mix_EA", 0.0))
        plan_rows.append({
            "MonthOrder": len(plan_rows),
            "Month": f"{m_name} {run_year}", "MonthName": m_name, "MonthNum": m_num, "RunYear": run_year,
            "Title": title_sel, "Category": cat,
            "PrimarySegment": r.get("PredictedPrimarySegment", ""), "SecondarySegment": r.get("PredictedSecondarySegment", ""),
            "SeasonalityFactor": f"{f_season:.3f}", "EffTicketIndex": f"{eff_idx:.1f}",
            "EstTickets_beforeDecay": (None if not np.isfinite(est_tix) else int(est_tix)),
            "ReturnDecayFactor": f"{float(decay_factor):.2f}", "EstimatedTickets_Final": est_tix_final,
            "Seg_GP_Tickets": int(round(est_tix_final * mix_gp)),
            "Seg_Core_Tickets": int(round(est_tix_final * mix_core)),
            "Seg_Family_Tickets": int(round(est_tix_final * mix_family)),
            "Seg_EA_Tickets": int(round(est_tix_final * mix_ea)),
        })

    if plan_rows:
        plan_df = pd.DataFrame(plan_rows).sort_values("MonthOrder")
        total_final = int(plan_df["EstimatedTickets_Final"].sum())
        st.markdown(f"**Projected season total (final, after decay):** {total_final:,}")
        st.dataframe(
            plan_df[
                ["Month","Title","Category","PrimarySegment","SecondarySegment",
                 "SeasonalityFactor","EffTicketIndex",
                 "EstTickets_beforeDecay","ReturnDecayFactor","EstimatedTickets_Final",
                 "Seg_GP_Tickets","Seg_Core_Tickets","Seg_Family_Tickets","Seg_EA_Tickets"]
            ],
            use_container_width=True, hide_index=True
        )
        st.download_button(
            "⬇️ Download Season Plan CSV",
            plan_df.drop(columns=["MonthOrder"]).to_csv(index=False).encode("utf-8"),
            file_name=f"season_plan_{season_year}-{season_year+1}.csv",
            mime="text/csv"
        )
    else:
        st.caption("Pick at least one month/title above to see your season projection.")

    # Calgary / Edmonton split quick view
    with st.expander("🏙️ Calgary / Edmonton split (singles vs subs)"):
        cols = ["Title","Category","EstimatedTickets_Final",
                "YYC_Total","YYC_Singles","YYC_Subs",
                "YEG_Total","YEG_Singles","YEG_Subs",
                "CityShare_Calgary","CityShare_Edmonton"]
        present = [c for c in cols if c in df.columns]
        st.dataframe(
            df[present].sort_values("EstimatedTickets_Final", ascending=False)
              .style.format({
                  "EstimatedTickets_Final": "{:,.0f}",
                  "YYC_Total": "{:,.0f}", "YYC_Singles": "{:,.0f}", "YYC_Subs": "{:,.0f}",
                  "YEG_Total": "{:,.0f}", "YEG_Singles": "{:,.0f}", "YEG_Subs": "{:,.0f}",
                  "CityShare_Calgary": "{:.1%}", "CityShare_Edmonton": "{:.1%}",
              }),
            use_container_width=True, hide_index=True
        )
        try:
            tot = {
                "YYC_Total": int(df["YYC_Total"].sum()),
                "YYC_Singles": int(df["YYC_Singles"].sum()),
                "YYC_Subs": int(df["YYC_Subs"].sum()),
                "YEG_Total": int(df["YEG_Total"].sum()),
                "YEG_Singles": int(df["YEG_Singles"].sum()),
                "YEG_Subs": int(df["YEG_Subs"].sum()),
                "EstimatedTickets_Final": int(df["EstimatedTickets_Final"].sum()),
            }
            st.caption(f"Totals — YYC: {tot['YYC_Total']:,} (Singles {tot['YYC_Singles']:,} / Subs {tot['YYC_Subs']:,}) · "
                       f"YEG: {tot['YEG_Total']:,} (Singles {tot['YEG_Singles']:,} / Subs {tot['YEG_Subs']:,}) · "
                       f"All-in: {tot['EstimatedTickets_Final']:,}")
        except Exception:
            pass

    # Scatter chart
    try:
        if "Familiarity" in df.columns and "Motivation" in df.columns:
            fig, ax = plt.subplots()
            ax.scatter(df["Familiarity"], df["Motivation"])
            for _, rr in df.iterrows():
                ax.annotate(rr["Title"], (rr["Familiarity"], rr["Motivation"]), fontsize=8)
            ax.axvline(df["Familiarity"].median(), linestyle="--")
            ax.axhline(df["Motivation"].median(), linestyle="--")
            ax.set_xlabel(f"Familiarity ({benchmark_title} = 100 index)")
            ax.set_ylabel(f"Motivation ({benchmark_title} = 100 index)")
            ax.set_title(f"Familiarity vs Motivation — {segment} / {region}")
            st.pyplot(fig)
    except Exception:
        pass

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
