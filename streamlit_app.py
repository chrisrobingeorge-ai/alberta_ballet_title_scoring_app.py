# streamlit_app_v9_new_titles_fix.py
# Alberta Ballet — Title Familiarity & Motivation Scorer (v9 with segment-propensity outputs)
# - Hard-coded baselines (normalized to a user-selected benchmark = 100)
# - Add NEW titles; optional live fetch (Wikipedia/YouTube/Spotify) with outlier-safety
# - Segment + Region multipliers; charts + CSV
# - TicketIndex prediction for titles without history + EstimatedTickets
# - NEW: Segment propensity as an OUTPUT (primary/secondary, shares, tickets by segment)

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

st.title("🎭 Alberta Ballet — Title Familiarity & Motivation Scorer (v9 — New Titles & Segment Outputs)")
st.caption("Hard-coded Alberta-wide baselines (normalized to your selected benchmark = 100). Add new titles; choose live fetch or offline estimate.")

# -------------------------
# METHODOLOGY & GLOSSARY SECTION
# -------------------------
with st.expander("📘 About This App — Methodology & Glossary"):
    st.markdown(dedent("""
    ### Purpose
    This tool estimates how familiar audiences are with a title and how motivated they are to attend, then blends those "online signal" estimates with ticket-informed scores. If a title has past ticket history, we use it; if it doesn’t, we predict a TicketIndex from similar shows so new titles get a fair, history-like adjustment. It also predicts which audience segments are most likely to respond, so segments become an output you can use for planning.

    ---

    ### What goes into the scores
    **Online signals (per title):**
    - Wikipedia page views → awareness
    - Google Trends (light heuristic) → search interest
    - YouTube activity → engagement potential (with outlier safety)
    - Spotify popularity → musical familiarity

    **Contextual multipliers (for signals):**
    - Audience Segment: weights by gender lead and category (e.g., Core Classical, Family, Emerging Adults)
    - Region: Province / Calgary / Edmonton adjustment

    **Ticket history & prediction:**
    - If a title has history, we convert its historical median tickets to a TicketIndex relative to the selected benchmark title's median (same 100-scale).
    - If a title has no history, we predict TicketIndex from the online signal via a learned linear fit from known titles (per-category when possible, otherwise overall). We show the source: History, Category model, or Overall model.

    ---

    ### How the score is calculated
    1) Raw signals → Familiarity & Motivation  
       - Familiarity = 0.55·Wiki + 0.30·Trends + 0.15·Spotify  
       - Motivation = 0.45·YouTube + 0.25·Trends + 0.15·Spotify + 0.15·Wiki  
       - Apply segment multipliers (by gender and category) and a region multiplier.
    2) Normalize to your benchmark (= 100)  
       You choose a benchmark title; all Familiarity/Motivation values are scaled so the benchmark equals 100 under the current segment and region.
    3) TicketIndex (history or predicted)  
       - History: median tickets ÷ benchmark median × 100  
       - No history: predicted from Online Signals (per-category model if ≥3 titles with history; else overall)
    4) Blend signals with (actual or predicted) tickets  
       Composite = 50% Online Signals + 50% TicketIndex
    5) Letter grade  
       A (≥90), B (≥75), C (≥60), D (≥45), E (<45)

    ---

    ### Segment propensity (output)
    For each title we also estimate which audience segments are most likely to attend:
    - Compute the online-signal score under each segment (using the same benchmark scaling).
    - Convert those into shares that sum to 100% (softmax-like).
    - Allocate EstimatedTickets across segments accordingly.
    You’ll see PredictedPrimarySegment, PredictedSecondarySegment, SegShare_* (%), and EstTix_* (counts) in the table.

    ---

    ### Glossary of Terms
    | Term | Definition |
    |------|------------|
    | Benchmark | The title you choose to set the 100 index. All other scores are scaled relative to this under the current segment/region. |
    | Familiarity | Online awareness signal from Wiki, Trends, Spotify (after multipliers), then normalized to the benchmark. |
    | Motivation | Online interest/engagement signal from YouTube, Trends, Spotify, Wiki (after multipliers), then normalized. |
    | Online Signals | Average of the normalized Familiarity and Motivation (what the title would score without ticket history). |
    | TicketIndex (history) | Historical median tickets ÷ benchmark median × 100 (actual ticket-based index). |
    | TicketIndex (predicted) | TicketIndex estimated from Online Signals via linear fit (per-category if possible, otherwise overall). |
    | Category model | Linear fit learned from titles in the same category (≥3 with history). Captures category-specific conversion from buzz to tickets. |
    | Overall model | Linear fit learned from all titles with history across categories. Used when a category lacks enough data. |
    | TicketIndex source | Whether we used History, the Category model, or the Overall model. |
    | Composite | 50% Online Signals + 50% TicketIndex (history or predicted). |
    | Segment propensity | Predicted primary/secondary segments + shares, based on signals under each segment (normalized to the benchmark). |
    | Normalization | Scaling that sets the benchmark title to 100 under the current segment/region. |

    **Note:** Use online signals to screen ideas, ticket history (or predicted TicketIndex) to ground expectations, and the segment outputs to plan creative and channel mix. This is a comparative tool, not an exact sales forecast.
    """))

# -------------------------
# BASELINE DATA (subset for test run)
# Values are indices relative to a general baseline (not capped).
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
# Region-level priors: category-by-segment weights (1.0 = neutral).
# Tweak these to match your Live Analytics report.
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
    # Return a dict of segment -> weight, adjusted by SEGMENT_PRIOR_STRENGTH.
    pri = SEGMENT_PRIORS.get(region_key, {}).get(category, {})
    if SEGMENT_PRIOR_STRENGTH == 1.0 or not pri:
        return pri or {k: 1.0 for k in SEGMENT_KEYS_IN_ORDER}
    # Temper the priors toward 1.0 using a power transform (e.g., 0.5 ~ sqrt, 0.0 ~ no effect).
    tempered = {}
    p = float(SEGMENT_PRIOR_STRENGTH)
    for k in SEGMENT_KEYS_IN_ORDER:
        w = pri.get(k, 1.0)
        tempered[k] = (w ** p) if w > 0 else 1.0
    return tempered

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
    v = float(np.median(view_list))  # robust vs outliers
    idx = 50.0 + min(90.0, np.log1p(v) * 9.0)
    return float(idx)

def _winsorize_youtube_to_baseline(category: str, yt_value: float) -> float:
    try:
        base_df = pd.DataFrame(BASELINES).T
        base_df["category"] = [v.get("category", "dramatic") for v in BASELINES.values()]
        ref = base_df.loc[base_df["category"] == category, "youtube"].dropna()
        if ref.empty:
            ref = base_df["youtube"].dropna()
        if ref.empty:
            return yt_value
        lo = float(np.percentile(ref, 3))
        hi = float(np.percentile(ref, 97))
        return float(np.clip(yt_value, lo, hi))
    except Exception:
        return yt_value

def wiki_search_best_title(query: str) -> Optional[str]:
    try:
        params = {"action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": 5}
        r = requests.get(WIKI_API, params=params, timeout=10)
        if r.status_code != 200:
            return None
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
        if r.status_code != 200:
            return 0.0
        items = r.json().get("items", [])
        views = [it.get("views", 0) for it in items]
        return (sum(views) / 365.0) if views else 0.0
    except Exception:
        return 0.0

def fetch_live_for_unknown(title: str, yt_key: Optional[str], sp_id: Optional[str], sp_secret: Optional[str]) -> Dict[str, float | str]:
    # Wikipedia index-ish transform
    w_title = wiki_search_best_title(title) or title
    wiki_raw = fetch_wikipedia_views_for_page(w_title)
    wiki_idx = 40.0 + min(110.0, (math.log1p(max(0.0, wiki_raw)) * 20.0))  # ~40..150

    # Trends heuristic
    trends_idx = 60.0 + (len(title) % 40)

    # YouTube (optional) — robust, ballet-focused
    yt_idx = 0.0
    if yt_key and build is not None:
        try:
            yt = build("youtube", "v3", developerKey=yt_key)
            q = f"{title} ballet"
            search = yt.search().list(
                q=q, part="snippet", type="video", maxResults=30, relevanceLanguage="en"
            ).execute()
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
                        try:
                            views.append(int(vc))
                        except Exception:
                            pass

            yt_idx = _yt_index_from_views(views) if views else 0.0
        except Exception:
            yt_idx = 0.0

    if yt_idx == 0.0:
        yt_idx = 55.0 + (len(title) * 1.2) % 45.0

    yt_idx = float(np.clip(yt_idx, 45.0, 140.0))

    # Spotify (optional)
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

    return {
        "wiki": wiki_idx,
        "trends": trends_idx,
        "youtube": yt_idx,
        "spotify": sp_idx,
        "gender": gender,
        "category": category
    }

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
- **Spotify**: [Web API docs](https://developer.spotify.com/documentation/web-api/) · [Getting started](https://developer.spotify.com/documentation/web-api/tutorials/getting-started) (create/manage keys in the Spotify Developer Dashboard).
- **YouTube**: [YouTube Data API overview](https://developers.google.com/youtube/v3) · [API reference](https://developers.google.com/youtube/v3/docs) (create/manage keys in Google Cloud Console → APIs & Services → Credentials).
""")
    st.caption("Keys are created in your own accounts. The links above are universal; your keys remain private to you.")

region = st.selectbox("Region", ["Province", "Calgary", "Edmonton"], index=0)
segment = st.selectbox("Audience Segment", list(SEGMENT_MULT.keys()), index=0)

default_list = list(BASELINES.keys())[:50]
st.markdown("**Titles to score** (one per line). Add NEW titles freely:")
titles_input = st.text_area("Enter titles", value="\n".join(default_list), height=220)
titles = [t.strip() for t in titles_input.splitlines() if t.strip()]

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
    n = len(xs)
    mid = n // 2
    return (xs[mid] if n % 2 else (xs[mid-1] + xs[mid]) / 2.0)

TICKET_BLEND_WEIGHT = 0.50  # 50% tickets, 50% familiarity/motivation composite

# --- Segment propensity helpers (output) ---
SEGMENT_KEYS_IN_ORDER = [
    "General Population",
    "Core Classical (F35–64)",
    "Family (Parents w/ kids)",
    "Emerging Adults (18–34)",
]

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

def _softmax_like(values: dict, temperature: float = 15.0) -> dict:
    arr = np.array(list(values.values()), dtype=float)
    exps = np.exp(arr / float(temperature))
    denom = exps.sum() if exps.sum() > 0 else 1.0
    shares = exps / denom
    return {k: float(v) for k, v in zip(values.keys(), shares)}

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
    st.caption(f"Scores normalized to benchmark: {benchmark_title}")

    # 3) Ticket medians & TicketIndex (history) relative to chosen benchmark
    TICKET_MEDIANS = {k: _median(v) for k, v in TICKET_PRIORS_RAW.items()}
    BENCHMARK_TICKET_MEDIAN = TICKET_MEDIANS.get(benchmark_title, None) or 1.0

    def ticket_index_for_title(title: str):
        aliases = {"Handmaid’s Tale": "Handmaid's Tale"}  # normalize common variant if it's ever entered
        key = aliases.get(title.strip(), title.strip())
        med = TICKET_MEDIANS.get(key)
        if med:
            return float(med), float((med / BENCHMARK_TICKET_MEDIAN) * 100.0)
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
        overall = None  # (a, b) if enough data, else None
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
            a, b = cat_coefs[category]
            src = "Category model"
        elif overall_coef is not None:
            a, b = overall_coef
            src = "Overall model"
        else:
            return np.nan, "Not enough data"
        pred = a * signal_only + b
        pred = float(np.clip(pred, 20.0, 180.0))
        return pred, src

    imputed_vals, imputed_srcs = [], []
    for _, r in df.iterrows():
        if pd.notna(r["TicketIndex"]):
            imputed_vals.append(r["TicketIndex"])
            imputed_srcs.append("History")
        else:
            pred, src = _predict_ticket_index(r["SignalOnly"], r["Category"])
            imputed_vals.append(pred)
            imputed_srcs.append(src)

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
    # Choose the TicketIndex to use for counts
    df["EffectiveTicketIndex"] = np.where(
        df["TicketIndex"].notna(), df["TicketIndex"],
        np.where(df["TicketIndexImputed"].notna(), df["TicketIndexImputed"], df["SignalOnly"])
    )
    def _est_src(row):
        if pd.notna(row.get("TicketMedian", np.nan)):
            return "History (actual median)"
        if pd.notna(row.get("TicketIndexImputed", np.nan)):
            return f'Predicted ({row.get("TicketIndexSource","model")})'
        return "Online-only (proxy: low confidence)"
    df["TicketEstimateSource"] = df.apply(_est_src, axis=1)
    df["EstimatedTickets"] = ((df["EffectiveTicketIndex"] / 100.0) * BENCHMARK_TICKET_MEDIAN_LOCAL).round(0)

    # 8) Segment propensity as OUTPUT (primary/secondary + shares + tickets)
    seg_primary = []
    seg_secondary = []
    seg_shares_cols = {k: [] for k in SEGMENT_KEYS_IN_ORDER}
    seg_tix_cols   = {k: [] for k in SEGMENT_KEYS_IN_ORDER}
    benchmark_entry = bench_entry  # use the same benchmark title's baseline for per-segment normalization

    for _, r in df.iterrows():
        # reconstruct an "entry" for this title
        entry_like = {
            "wiki":     float(r["WikiIdx"]),
            "trends":   float(r["TrendsIdx"]),
            "youtube":  float(r["YouTubeIdx"]),
            "spotify":  float(r["SpotifyIdx"]),
            "gender":   r["Gender"],
            "category": r["Category"],
        }
        seg_to_raw = _signal_for_all_segments(entry_like, r["Region"])
        seg_to_idx_signal = _normalize_signals_by_benchmark(seg_to_raw, benchmark_entry, r["Region"])
        shares = _softmax_like(seg_to_idx_signal, temperature=15.0)

        ordered = sorted(shares.items(), key=lambda kv: kv[1], reverse=True)
        seg_primary.append(ordered[0][0] if ordered else None)
        seg_secondary.append(ordered[1][0] if len(ordered) > 1 else None)

        est_tix = float(r.get("EstimatedTickets", np.nan)) if pd.notna(r.get("EstimatedTickets", np.nan)) else np.nan
        for seg_key in SEGMENT_KEYS_IN_ORDER:
            share = shares.get(seg_key, 0.0)
            seg_shares_cols[seg_key].append(share * 100.0)  # percent
            seg_tix_cols[seg_key].append(est_tix * share if pd.notna(est_tix) else np.nan)

    df["PredictedPrimarySegment"]   = seg_primary
    df["PredictedSecondarySegment"] = seg_secondary
    for seg_key in SEGMENT_KEYS_IN_ORDER:
        df[f"SegShare_{seg_key}"] = np.round(seg_shares_cols[seg_key], 0)
        df[f"EstTix_{seg_key}"]   = np.round(seg_tix_cols[seg_key], 0)
    df["SegmentMixNote"] = (
        df["PredictedPrimarySegment"].fillna("")
        + np.where(df["PredictedSecondarySegment"].notna(), " (secondary: " + df["PredictedSecondarySegment"] + ")", "")
    )

    # 9) Stash results
    st.session_state["results"] = {
        "df": df,
        "benchmark": benchmark_title,
        "segment": segment,
        "region": region,
        "unknown_est": unknown_used_est,
        "unknown_live": unknown_used_live,
    }

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

    # Notices
    if R["unknown_est"]:
        st.info("Estimated (offline) for new titles: " + ", ".join(R["unknown_est"]))
    if R["unknown_live"]:
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
        hist_count    = int(src_counts.get("History", 0))
        cat_count     = int(src_counts.get("Category model", 0))
        overall_count = int(src_counts.get("Overall model", 0))
        ned_count     = int(src_counts.get("Not enough data", 0))
        st.caption(
            f"TicketIndex source — History: {hist_count} · Category model: {cat_count} · "
            f"Overall model: {overall_count} · Not enough data: {ned_count}"
        )
        if ned_count > 0:
            st.info("Some titles fell back to online-only because there wasn't enough data to learn a reliable ticket mapping.")

    # Grades
    def _assign_score(v: float) -> str:
        if v >= 90: return "A"
        elif v >= 75: return "B"
        elif v >= 60: return "C"
        elif v >= 45: return "D"
        else: return "E"
    df["Score"] = df["Composite"].apply(_assign_score)

    # Helper to render full results table
    def _render_full_results_table(df_in: pd.DataFrame):
        df_show = df_in.rename(columns={"TicketMedian": "TicketHistory"}).copy()

        display_cols = [
            "Title", "Region", "Segment", "Gender", "Category",
            "WikiIdx", "TrendsIdx", "YouTubeIdx", "SpotifyIdx",
            "Familiarity", "Motivation",
            "TicketHistory",
            "EffectiveTicketIndex",  # will be renamed to 'TicketIndex used'
            "TicketIndexSource",
            "Composite", "Score",
            "EstimatedTickets",
            "PredictedPrimarySegment", "PredictedSecondarySegment",
            "SegShare_General Population", "SegShare_Core Classical (F35–64)",
            "SegShare_Family (Parents w/ kids)", "SegShare_Emerging Adults (18–34)",
            "EstTix_General Population", "EstTix_Core Classical (F35–64)",
            "EstTix_Family (Parents w/ kids)", "EstTix_Emerging Adults (18–34)",
            "SegmentMixNote",
            "Source",
        ]

        # Do not show raw TicketIndex and TicketIndexImputed columns (kept for CSV if you like)
        drop_if_present = ["TicketIndex", "TicketIndexImputed"]
        df_show = df_show.drop(columns=[c for c in drop_if_present if c in df_show.columns], errors="ignore")

        present = [c for c in display_cols if c in df_show.columns]

        st.dataframe(
            df_show[present]
              .sort_values(
                  by=[
                      "EstimatedTickets" if "EstimatedTickets" in df_show.columns else "Composite",
                      "Composite", "Motivation", "Familiarity"
                  ],
                  ascending=[False, False, False, False]
              )
              .rename(columns={"EffectiveTicketIndex": "TicketIndex used"})
              .style
                .format({
                    "WikiIdx": "{:.0f}", "TrendsIdx": "{:.0f}", "YouTubeIdx": "{:.0f}", "SpotifyIdx": "{:.0f}",
                    "Familiarity": "{:.1f}", "Motivation": "{:.1f}",
                    "Composite": "{:.1f}",
                    "TicketIndex used": "{:.1f}",
                    "EstimatedTickets": "{:,.0f}",
                    "TicketHistory": "{:,.0f}",
                    "SegShare_General Population": "{:.0f}%",
                    "SegShare_Core Classical (F35–64)": "{:.0f}%",
                    "SegShare_Family (Parents w/ kids)": "{:.0f}%",
                    "SegShare_Emerging Adults (18–34)": "{:.0f}%",
                    "EstTix_General Population": "{:,.0f}",
                    "EstTix_Core Classical (F35–64)": "{:,.0f}",
                    "EstTix_Family (Parents w/ kids)": "{:,.0f}",
                    "EstTix_Emerging Adults (18–34)": "{:,.0f}",
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

    # Header + full table
    st.subheader("🎟️ Estimated ticket sales and segment mix")
    known_mask = df["TicketHistory"].notna() if "TicketHistory" in df.columns else df["TicketMedian"].notna()
    _render_full_results_table(df)

    # Typical tickets by grade (based on shows with history)
    if "TicketMedian" in df.columns:
        known_mask = df["TicketMedian"].notna()
        if known_mask.any():
            grade_ticket_stats = (
                df.loc[known_mask]
                  .groupby("Score")["TicketMedian"]
                  .agg(
                      median="median",
                      p25=lambda s: np.percentile(s, 25),
                      p75=lambda s: np.percentile(s, 75),
                      n="count"
                  )
                  .reindex(["A","B","C","D","E"])
            )
            st.markdown("**Typical tickets by grade** (based on shows with history)")
            st.dataframe(
                grade_ticket_stats.rename(columns={
                    "median":"Median tickets",
                    "p25":"25th percentile",
                    "p75":"75th percentile",
                    "n":"# titles"
                }).style.format({
                    "Median tickets":"{:,.0f}",
                    "25th percentile":"{:,.0f}",
                    "75th percentile":"{:,.0f}",
                    "# titles":"{:,.0f}"
                }),
                use_container_width=True
            )

    # Charts
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

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Familiarity (Indexed)")
        st.bar_chart(df.set_index("Title")["Familiarity"])
    with col2:
        st.subheader("Motivation (Indexed)")
        st.bar_chart(df.set_index("Title")["Motivation"])

    st.download_button(
        "⬇️ Download CSV",
        df.to_csv(index=False).encode("utf-8"),
        "title_scores_v9_ticket_blend.csv",
        "text/csv"
    )

# ======= BUTTON HANDLER =======
if run:
    if not titles:
        st.warning("Add at least one title to score.")
    else:
        compute_scores_and_store()

# ======= ALWAYS RENDER LAST RESULTS IF AVAILABLE =======
if st.session_state["results"] is not None:
    render_results()
