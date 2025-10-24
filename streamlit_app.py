# Alberta Ballet — Title Familiarity & Motivation Scorer (v9 with segment-propensity outputs)
# - Hard-coded baselines (normalized to a user-selected benchmark = 100)
# - Add NEW titles; optional live fetch (Wikipedia/YouTube/Spotify) with outlier-safety
# - Segment + Region multipliers; charts + CSV
# - TicketIndex prediction for titles without history + EstimatedTickets
# - NEW: Segment propensity as an OUTPUT (primary/secondary, shares, tickets by segment)
# - NEW: Live Analytics overlays by program type

import calendar
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
@@ -93,50 +94,106 @@ BASELINES = {
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

PRODUCTION_SCHEDULE = [
    {"title": "Alice in Wonderland", "start_date": "2017-03-16", "end_date": "2017-03-25"},
    {"title": "All of Us - Tragically Hip", "start_date": "2018-05-02", "end_date": "2018-05-12"},
    {"title": "Away We Go - Mixed Bill", "start_date": "2022-10-27", "end_date": "2022-11-05"},
    {"title": "Ballet BC", "start_date": "2019-01-19", "end_date": "2019-01-23"},
    {"title": "Ballet Boyz", "start_date": "2017-02-16", "end_date": "2017-02-25"},
    {"title": "BJM - Leonard Cohen", "start_date": "2018-09-20", "end_date": "2018-09-22"},
    {"title": "Botero", "start_date": "2023-05-05", "end_date": "2023-05-13"},
    {"title": "Cinderella", "start_date": "2022-04-28", "end_date": "2022-05-14"},
    {"title": "Complexions - Lenny Kravitz", "start_date": "2023-02-10", "end_date": "2023-02-18"},
    {"title": "Dance Theatre of Harlem", "start_date": "2025-02-13", "end_date": "2025-02-22"},
    {"title": "Dangerous Liaisons", "start_date": "2017-10-26", "end_date": "2017-11-04"},
    {"title": "deViate - Mixed Bill", "start_date": "2019-02-15", "end_date": "2019-02-16"},
    {"title": "Diavolo", "start_date": "2020-01-21", "end_date": "2020-01-22"},
    {"title": "Don Quixote", "start_date": "2025-05-01", "end_date": "2025-05-10"},
    {"title": "Dona Peron", "start_date": "2023-09-14", "end_date": "2023-09-23"},
    {"title": "Dracula", "start_date": "2016-10-27", "end_date": "2016-11-05"},
    {"title": "Fiddle & the Drum – Joni Mitchell", "start_date": "2019-05-01", "end_date": "2019-05-11"},
    {"title": "Frankenstein", "start_date": "2019-10-23", "end_date": "2019-11-02"},
    {"title": "Giselle", "start_date": "2023-03-09", "end_date": "2023-03-25"},
    {"title": "Grimm", "start_date": "2024-10-17", "end_date": "2024-10-26"},
    {"title": "Handmaid's Tale", "start_date": "2022-09-14", "end_date": "2022-09-24"},
    {"title": "Hansel & Gretel", "start_date": "2024-03-07", "end_date": "2024-03-23"},
    {"title": "La Sylphide", "start_date": "2024-09-12", "end_date": "2024-09-21"},
    {"title": "Midsummer Night’s Dream", "start_date": "2019-03-13", "end_date": "2019-03-16"},
    {"title": "Momix", "start_date": "2018-02-15", "end_date": "2018-02-22"},
    {"title": "Nijinsky", "start_date": "2025-10-16", "end_date": "2025-10-25"},
    {"title": "Nutcracker", "start_date": "2023-12-06", "end_date": "2023-12-24"},
    {"title": "Our Canada - Gordon Lightfoot", "start_date": "2017-05-04", "end_date": "2017-05-13"},
    {"title": "Phi - David Bowie", "start_date": "2022-03-10", "end_date": "2022-03-19"},
    {"title": "Shaping Sound", "start_date": "2018-01-19", "end_date": "2018-01-20"},
    {"title": "Sleeping Beauty", "start_date": "2023-10-26", "end_date": "2023-11-04"},
    {"title": "Swan Lake", "start_date": "2021-10-21", "end_date": "2021-11-07"},
    {"title": "Taj Express", "start_date": "2019-09-12", "end_date": "2019-09-21"},
    {"title": "Tango Fire", "start_date": "2017-09-21", "end_date": "2017-09-28"},
    {"title": "Trockadero", "start_date": "2017-01-12", "end_date": "2017-01-14"},
    {"title": "Unleashed - Mixed Bill", "start_date": "2020-02-13", "end_date": "2020-02-22"},
    {"title": "Winter Gala", "start_date": "2025-01-18", "end_date": "2025-01-25"},
    {"title": "Wizard of Oz", "start_date": "2025-03-13", "end_date": "2025-03-22"},
]

SCHEDULE_LOOKUP = {}
for _item in PRODUCTION_SCHEDULE:
    start = pd.to_datetime(_item.get("start_date"), errors="coerce")
    if pd.isna(start):
        continue
    month = int(start.month)
    SCHEDULE_LOOKUP[_item["title"]] = {
        "month": month,
        "month_name": calendar.month_name[month],
        "start_date": start.date(),
        "end_date": pd.to_datetime(_item.get("end_date"), errors="coerce").date() if _item.get("end_date") else None,
    }

MONTH_NAME_LOOKUP = {i: calendar.month_name[i] for i in range(1, 13)}

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
@@ -371,50 +428,104 @@ def infer_gender_and_category(title: str) -> Tuple[str, str]:
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


def _compute_time_slot_factors(schedule: list[dict], ticket_medians: dict[str, Optional[float]], baselines: dict) -> tuple[dict, pd.DataFrame]:
    rows: list[dict] = []
    for item in schedule:
        title = item.get("title")
        if not title:
            continue
        med = ticket_medians.get(title)
        if med is None:
            continue
        start = pd.to_datetime(item.get("start_date"), errors="coerce")
        if pd.isna(start):
            continue
        month = int(start.month)
        base_meta = baselines.get(title, {})
        category = base_meta.get("category")
        if not category:
            _, category = infer_gender_and_category(title)
        rows.append({
            "title": title,
            "category": category,
            "month": month,
            "month_name": calendar.month_name[month],
            "ticket_median": float(med),
        })

    if not rows:
        return {}, pd.DataFrame(columns=["category", "month", "month_name", "SeasonalityFactor", "TicketsMedian", "SampleSize"])

    df = pd.DataFrame(rows)
    category_medians = df.groupby("category")["ticket_median"].median().to_dict()
    df["category_baseline"] = df["category"].map(category_medians)
    df["seasonality_factor"] = df.apply(
        lambda r: float(r["ticket_median"] / r["category_baseline"]) if r["category_baseline"] else 1.0,
        axis=1,
    )

    summary = (
        df.groupby(["category", "month", "month_name"], as_index=False)
        .agg(
            SeasonalityFactor=("seasonality_factor", "median"),
            TicketsMedian=("ticket_median", "median"),
            SampleSize=("title", "count"),
        )
    )

    factors: dict[str, dict[int, float]] = {}
    for _, row in summary.iterrows():
        cat = row["category"]
        month = int(row["month"])
        factors.setdefault(cat, {})[month] = float(row["SeasonalityFactor"])

    return factors, summary

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
@@ -510,50 +621,60 @@ def fetch_live_for_unknown(title: str, yt_key: Optional[str], sp_id: Optional[st
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

_month_names = [calendar.month_name[i] for i in range(1, 13)]
month_choice_options = ["Auto (use schedule / no override)"] + _month_names
default_month_label = st.selectbox(
    "Default month for NEW titles (seasonality adjustment)",
    options=month_choice_options,
    index=0,
)
MONTH_NAME_TO_NUMBER = {name: idx + 1 for idx, name in enumerate(_month_names)}
default_month_number = MONTH_NAME_TO_NUMBER.get(default_month_label)

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
@@ -590,126 +711,174 @@ TICKET_PRIORS_RAW = {
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

TICKET_MEDIANS = {k: _median(v) for k, v in TICKET_PRIORS_RAW.items()}

TIME_SLOT_FACTORS, TIME_SLOT_SUMMARY = _compute_time_slot_factors(PRODUCTION_SCHEDULE, TICKET_MEDIANS, BASELINES)

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
def compute_scores_and_store(
    titles,
    segment,
    region,
    use_live,
    yt_key,
    sp_id,
    sp_secret,
    benchmark_title,
    default_month_for_new_titles,
    default_month_label,
):
    rows = []
    unknown_used_live, unknown_used_est = [], []

    # 1) Build base rows
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

    time_slot_months, time_slot_labels, seasonality_factors = [], [], []
    seasonality_sources, seasonality_has_data = [], []
    for _, r in df.iterrows():
        title = r["Title"]
        category = r["Category"]
        sched = SCHEDULE_LOOKUP.get(title)
        if sched:
            month = sched["month"]
            label = sched["month_name"]
            source = f"Historical schedule ({sched['start_date']})"
        else:
            month = default_month_for_new_titles
            if month:
                label = MONTH_NAME_LOOKUP.get(month, "n/a")
                source = "User default month"
            else:
                label = "n/a"
                source = "No time slot provided"

        factor = 1.0
        has_data = False
        factor_lookup = TIME_SLOT_FACTORS.get(category, {}).get(month) if month else None
        if factor_lookup is not None:
            factor = float(factor_lookup)
            has_data = True
            source += " · factor from historical tickets"
        elif month:
            source += " · no historical sample for this category-month"
        else:
            source += " · seasonality inactive"

        time_slot_months.append(month)
        time_slot_labels.append(label)
        seasonality_factors.append(factor)
        seasonality_sources.append(source)
        seasonality_has_data.append(has_data and month is not None)

    df["TimeSlotMonthNumber"] = time_slot_months
    df["TimeSlot"] = time_slot_labels
    df["SeasonalityFactor"] = seasonality_factors
    df["SeasonalitySource"] = seasonality_sources
    df["SeasonalityHasData"] = seasonality_has_data

    # 2) Normalize using the PASSED benchmark_title
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
@@ -727,50 +896,71 @@ def compute_scores_and_store(
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

    adjusted_ticket_index = []
    seasonality_applied = []
    for val, src, factor, has_data in zip(
        df["TicketIndexImputed"],
        df["TicketIndexSource"],
        df["SeasonalityFactor"],
        df["SeasonalityHasData"],
    ):
        if src != "History" and pd.notna(val) and pd.notna(factor):
            adjusted_ticket_index.append(float(val) * float(factor))
            seasonality_applied.append(bool(has_data))
        else:
            adjusted_ticket_index.append(val)
            seasonality_applied.append(False)

    df["TicketIndexImputed"] = adjusted_ticket_index
    df["SeasonalityApplied"] = seasonality_applied
    df["SeasonalitySource"] = df["SeasonalitySource"] + df["SeasonalityApplied"].map(
        lambda applied: " · applied to prediction" if applied else " · not applied (history or no model)"
    )

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
@@ -810,50 +1000,52 @@ def compute_scores_and_store(
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
        "default_month_label": default_month_label,
        "default_month_number": default_month_for_new_titles,
    }

import re
import pandas as pd

# =========================
# LIVE ANALYTICS: FULL TABLE (deep)
# - Keys are your app categories (pop_ip, classic_romance, etc.)
# - Values are dicts of unique, namespaced metrics so columns never collide
# - Fill what you have; anything missing will remain NaN
# =========================

LA_DEEP_BY_CATEGORY = {
    # Use the six categories you provided numbers for.
    # If you later get romantic_comedy or dramatic, add them here the same way.

    "pop_ip": {
        # --- DATA ATTRIBUTE ---
        "DA_Customers": 15861,
        "DA_CustomersWithLiveEvents": 13813,
        "DA_CustomersWithDemo_CAN": 14995,

        # --- CLIENT EVENT SUMMARY / Gender ---
        "CES_Gender_Male_pct": 44,
        "CES_Gender_Female_pct": 56,
@@ -1326,167 +1518,204 @@ def attach_la_report_columns(df: pd.DataFrame) -> pd.DataFrame:

    # Fill in the deep columns (unique, no collisions)
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
    R = st.session_state["results"]
    if not R:
        return

    df = R["df"].copy()
    benchmark_title = R["benchmark"]
    segment = R["segment"]
    region = R["region"]
    default_month_label = R.get("default_month_label")

    # Notices about data sources
    if R.get("unknown_est"):
        st.info("Estimated (offline) for new titles: " + ", ".join(R["unknown_est"]))
    if R.get("unknown_live"):
        st.success("Used LIVE data for new titles: " + ", ".join(R["unknown_live"]))
    if default_month_label and default_month_label != "Auto (use schedule / no override)":
        st.caption(f"Default month override for new titles: **{default_month_label}**")

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
        "SeasonalitySource": "SeasonalityNotes",
    }).copy()

    table_cols = [
        "Title","Region","Segment","Gender","Category",
        "WikiIdx","TrendsIdx","YouTubeIdx","SpotifyIdx",
        "Familiarity","Motivation",
        "TicketHistory",
        "TicketIndex used","TicketIndexSource",
        "TimeSlot","SeasonalityFactor","SeasonalityNotes",
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
                "SeasonalityFactor": "{:.2f}",
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

    if not TIME_SLOT_SUMMARY.empty:
        with st.expander("📅 Category seasonality insights", expanded=False):
            st.markdown(
                "Seasonality factors &gt; 1.0 indicate months where that category outperformed its own ticket median; "
                "factors &lt; 1.0 underperformed."
            )
            cat_options = ["All categories"] + sorted(TIME_SLOT_SUMMARY["category"].unique())
            selected_cat = st.selectbox(
                "Filter by category",
                options=cat_options,
                index=0,
                key="seasonality_category_filter",
            )
            summary_df = TIME_SLOT_SUMMARY.copy()
            if selected_cat != "All categories":
                summary_df = summary_df[summary_df["category"] == selected_cat]
            summary_df = summary_df.sort_values(["category", "month"]).reset_index(drop=True)
            summary_display = summary_df.rename(columns={
                "category": "Category",
                "month": "Month #",
                "month_name": "Month",
                "SeasonalityFactor": "SeasonalityFactor",
                "TicketsMedian": "MedianTickets",
                "SampleSize": "SampleSize",
            }).copy()
            summary_display["SeasonalityFactor"] = summary_display["SeasonalityFactor"].round(2)
            summary_display["MedianTickets"] = summary_display["MedianTickets"].round(0)
            st.dataframe(summary_display, use_container_width=True, hide_index=True)

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
            default_month_for_new_titles=default_month_number,
            default_month_label=default_month_label,
        )

# ======= ALWAYS RENDER LAST RESULTS IF AVAILABLE =======
if st.session_state["results"] is not None:
    render_results()
