# streamlit_app_v89.py
# Alberta Ballet — Title Familiarity & Motivation Scorer (v8.9 Test Build)
# Simplified baseline version for validation
# - Static pseudo-randomized data replaces Wikipedia/GoogleTrends/YouTube/Spotify
# - Full UI, segment, and region logic retained
# - Use this build to confirm differentiable scoring & working charts before v9 baselines

import os, math
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# -------------------------
# CONFIG & WEIGHTS
# -------------------------
TRENDS_TIMEFRAME = "today 5-y"

WEIGHTS = {
    "fam_wiki": 0.55,
    "fam_trends": 0.30,
    "fam_spotify": 0.15,
    "mot_youtube": 0.45,
    "mot_trends": 0.25,
    "mot_spotify": 0.15,
    "mot_wiki": 0.15,
}

SEGMENT_RULES = {
    "General Population": {"fam": {}, "mot": {}},
    "Core Classical (Female 35–64)": {
        "fam": {"female_lead": +0.12, "romantic": +0.08, "classic_canon": +0.10,
                "male_lead": -0.04, "contemporary": -0.05, "tragic": -0.02},
        "mot": {"female_lead": +0.10, "romantic": +0.06, "classic_canon": +0.08,
                "male_lead": -0.03, "contemporary": -0.04, "spectacle": +0.02}
    },
    "Family (Parents w/ kids)": {
        "fam": {"female_lead": +0.15, "family_friendly": +0.20, "pop_ip": +0.10,
                "male_lead": -0.05, "tragic": -0.12},
        "mot": {"female_lead": +0.10, "family_friendly": +0.18, "spectacle": +0.06,
                "pop_ip": +0.10, "tragic": -0.15}
    },
    "Emerging Adults (18–34)": {
        "fam": {"contemporary": +0.18, "spectacle": +0.08, "pop_ip": +0.10,
                "classic_canon": -0.04},
        "mot": {"contemporary": +0.22, "spectacle": +0.10, "pop_ip": +0.10,
                "female_lead": +0.02, "male_lead": +0.02}
    },
}

ATTR_COLUMNS = [
    "female_lead", "male_lead", "family_friendly", "romantic", "tragic",
    "contemporary", "classic_canon", "spectacle", "pop_ip"
]

REGIONAL_MULTIPLIERS = {
    "Province-wide (default)": {"fam": 1.00, "mot": 1.00},
    "Calgary": {"fam": 1.05, "mot": 1.07},
    "Edmonton": {"fam": 0.95, "mot": 0.93},
}

# -------------------------
# STATIC FETCH FUNCTIONS (replace APIs)
# -------------------------
def fetch_google_trends_score(title, market):
    return float((len(title) * 3) % 100) + 10  # 10–100 range

def fetch_wikipedia_views(query):
    return (query, float((len(query) * 2.5) % 120) + 10)

def fetch_youtube_metrics(title, api_key):
    base = len(title)
    return (float(base % 10), float((base * 1.5) % 80) + 10)

def fetch_spotify_popularity(title, cid, secret):
    return float((len(title) * 2) % 100) + 5

# -------------------------
# UTILITIES
# -------------------------
def normalize_series(values):
    if not values: return []
    vmin, vmax = min(values), max(values)
    if vmax == vmin: return [50.0 for _ in values]
    return [(v - vmin) * 100.0 / (vmax - vmin) for v in values]

def percentile_normalize(vals):
    if not vals: return []
    s = pd.Series(vals, dtype="float64")
    return (s.rank(pct=True) * 100.0).tolist()

def infer_title_attributes(title: str):
    t = title.lower()
    attr = {k: False for k in ATTR_COLUMNS}
    female_titles = ["nutcracker","sleeping beauty","cinderella","beauty","alice","giselle","swan lake","merry widow","romeo"]
    male_titles = ["pinocchio","peter pan","don quixote","hunchback","notre dame","romeo"]
    family_list = ["nutcracker","cinderella","beauty","alice","frozen","peter pan","pinocchio","wizard","oz"]
    romantic_list = ["swan","cinderella","sleeping","romeo","merry","beauty","giselle"]
    tragic_list = ["swan","romeo","hunchback","notre dame","giselle"]
    classic_list = ["nutcracker","swan","sleeping","cinderella","romeo","don","giselle","merry"]
    spectacle_list = ["wizard","peter pan","pinocchio","frozen","notre","hunchback","don"]
    popip_list = ["frozen","beauty","wizard","bridgerton","harry","star","avengers","barbie"]

    if any(k in t for k in female_titles): attr["female_lead"] = True
    if any(k in t for k in male_titles): attr["male_lead"] = True
    if any(k in t for k in family_list): attr["family_friendly"] = True
    if any(k in t for k in romantic_list): attr["romantic"] = True
    if any(k in t for k in tragic_list): attr["tragic"] = True
    if any(k in t for k in classic_list): attr["classic_canon"] = True
    if any(k in t for k in spectacle_list): attr["spectacle"] = True
    if any(k in t for k in popip_list): attr["pop_ip"] = True
    if any(k in t for k in ["contemporary","composers","mixed","forsythe","balanchine","nijinsky","grimm"]):
        attr["contemporary"] = True
    if "romeo" in t:
        attr["female_lead"] = True
        attr["male_lead"] = True
    return attr

def segment_boosts(attrs, segment):
    rules = SEGMENT_RULES.get(segment, {})
    fam_delta = mot_delta = 0.0
    for k, v in rules.get("fam", {}).items():
        if attrs.get(k, False): fam_delta += v
    for k, v in rules.get("mot", {}).items():
        if attrs.get(k, False): mot_delta += v
    return max(-0.3, min(0.4, fam_delta)), max(-0.3, min(0.4, mot_delta))

# -------------------------
# SCORING
# -------------------------
def score_titles(titles, market, segment, region, attrs_df):
    regmult = REGIONAL_MULTIPLIERS[region]
    user_attrs = {str(r["Title"]).lower(): {c: bool(r.get(c, False)) for c in ATTR_COLUMNS} for _, r in attrs_df.iterrows()}

    rows = []
    for t in titles:
        w_title, w_val = fetch_wikipedia_views(t)
        trends_val = fetch_google_trends_score(t, market)
        yt_count, yt_views = fetch_youtube_metrics(t, None)
        sp_pop = fetch_spotify_popularity(t, None, None)

        wiki_fam = math.log1p(w_val)
        fam = WEIGHTS["fam_wiki"] * wiki_fam + WEIGHTS["fam_trends"] * trends_val + WEIGHTS["fam_spotify"] * sp_pop
        mot = WEIGHTS["mot_youtube"] * yt_views + WEIGHTS["mot_trends"] * trends_val + WEIGHTS["mot_spotify"] * sp_pop + WEIGHTS["mot_wiki"] * wiki_fam

        base_attrs = infer_title_attributes(t)
        over = user_attrs.get(t.lower(), {})
        base_attrs.update(over)
        fboost, mboost = segment_boosts(base_attrs, segment)

        fam_adj = fam * (1 + fboost) * regmult["fam"]
        mot_adj = mot * (1 + mboost) * regmult["mot"]

        rows.append({
            "Title": t, "WikiViews": w_val, "Trends": trends_val, "YTViews": yt_views, "Spotify": sp_pop,
            "Familiarity": round(fam, 1), "Motivation": round(mot, 1),
            "FamiliarityAdj": round(fam_adj, 1), "MotivationAdj": round(mot_adj, 1),
            "SegBoostF%": round(fboost * 100, 1), "SegBoostM%": round(mboost * 100, 1),
            "RegionFamMult": regmult["fam"], "RegionMotMult": regmult["mot"]
        })
    return pd.DataFrame(rows)

# -------------------------
# PLOTS
# -------------------------
def quadrant_plot(df, xcol, ycol, title):
    fig = plt.figure()
    x, y = df[xcol], df[ycol]
    plt.scatter(x, y)
    for _, r in df.iterrows():
        plt.annotate(r["Title"], (r[xcol], r[ycol]), fontsize=8, xytext=(3,3), textcoords="offset points")
    plt.axvline(np.median(x), linestyle="--")
    plt.axhline(np.median(y), linestyle="--")
    plt.title(title)
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    return fig

def bar_chart(df, col, title):
    fig = plt.figure()
    df2 = df.sort_values(by=col, ascending=True)
    plt.barh(df2["Title"], df2[col])
    plt.title(title)
    plt.xlabel(col)
    return fig

# -------------------------
# STREAMLIT UI
# -------------------------
st.set_page_config(page_title="Alberta Ballet — Title Familiarity & Motivation", layout="wide")
st.title("🎭 Alberta Ballet — Title Familiarity & Motivation Scorer (v8.9 Test Build)")

st.markdown("Testing static data sources — this version ensures scoring logic and charts function before full baselines.")

region = st.selectbox("Select Region", list(REGIONAL_MULTIPLIERS.keys()), index=0)
segment = st.selectbox("Select Audience Segment", list(SEGMENT_RULES.keys()), index=0)

default_titles = [
    "The Nutcracker","Sleeping Beauty","Cinderella","Pinocchio",
    "The Merry Widow","The Hunchback of Notre Dame","Frozen","Beauty and the Beast",
    "Alice in Wonderland","Peter Pan","Romeo and Juliet","Swan Lake","Don Quixote",
    "Contemporary Composers","Nijinsky","Notre Dame de Paris","Wizard of Oz","Grimm"
]
titles_input = st.text_area("Enter titles (one per line):", value="\n".join(default_titles), height=220)
titles = [t.strip() for t in titles_input.splitlines() if t.strip()]

# Attributes
st.subheader("Title Attributes (editable)")
attr_rows = []
for t in titles:
    inferred = infer_title_attributes(t)
    row = {"Title": t}
    row.update(inferred)
    attr_rows.append(row)
attrs_df = pd.DataFrame(attr_rows, columns=["Title"] + ATTR_COLUMNS)
attrs_df = st.data_editor(attrs_df, use_container_width=True, hide_index=True)

if st.button("Run Scoring", type="primary"):
    with st.spinner("Scoring titles…"):
        df = score_titles(titles, market="AB", segment=segment, region=region, attrs_df=attrs_df)
        st.success("Scoring complete.")
        st.dataframe(df, use_container_width=True)

        st.subheader("Quadrant")
        st.pyplot(quadrant_plot(df, "FamiliarityAdj", "MotivationAdj", "Familiarity vs Motivation (Adjusted)"))
        st.subheader("Familiarity Bar")
        st.pyplot(bar_chart(df, "FamiliarityAdj", "Adjusted Familiarity"))
        st.subheader("Motivation Bar")
        st.pyplot(bar_chart(df, "MotivationAdj", "Adjusted Motivation"))

        st.download_button("⬇️ Download CSV", df.to_csv(index=False).encode("utf-8"), "title_scores_test.csv", "text/csv")
