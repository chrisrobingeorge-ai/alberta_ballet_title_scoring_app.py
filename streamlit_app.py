# streamlit_app_v9_test.py
# Alberta Ballet — Title Familiarity & Motivation Scorer (v9 Test Build)
# Hard-coded baselines + optional live API enrichment for new titles

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime

# -------------------------
# SPLASH / LOADER
# -------------------------
st.set_page_config(page_title="Alberta Ballet — Title Familiarity & Motivation Scorer", layout="wide")
with st.spinner("🎭 Preparing Alberta Ballet Familiarity Baselines..."):
    time.sleep(1.5)

st.title("🎭 Alberta Ballet — Title Familiarity & Motivation Scorer (v9 Test Build)")
st.caption("Hard-coded Alberta-wide familiarity baselines with optional live API enrichment for new titles.")

# -------------------------
# BASELINE DATA
# -------------------------
# Familiarity & Motivation baselines (Nutcracker = 100 benchmark)
BASELINES = {
    "The Nutcracker": {"wiki": 100, "trends": 100, "youtube": 100, "spotify": 100, "gender": "female", "category": "family_classic"},
    "Sleeping Beauty": {"wiki": 92, "trends": 85, "youtube": 78, "spotify": 74, "gender": "female", "category": "classic_romance"},
    "Cinderella": {"wiki": 88, "trends": 80, "youtube": 82, "spotify": 80, "gender": "female", "category": "family_classic"},
    "Swan Lake": {"wiki": 95, "trends": 90, "youtube": 88, "spotify": 84, "gender": "female", "category": "classic_romance"},
    "Romeo and Juliet": {"wiki": 90, "trends": 82, "youtube": 79, "spotify": 77, "gender": "co", "category": "romantic_tragedy"},
    "The Merry Widow": {"wiki": 70, "trends": 60, "youtube": 55, "spotify": 50, "gender": "female", "category": "romantic_comedy"},
    "Beauty and the Beast": {"wiki": 94, "trends": 97, "youtube": 92, "spotify": 90, "gender": "female", "category": "family_classic"},
    "Frozen": {"wiki": 100, "trends": 110, "youtube": 120, "spotify": 115, "gender": "female", "category": "pop_ip"},
    "Peter Pan": {"wiki": 80, "trends": 78, "youtube": 85, "spotify": 82, "gender": "male", "category": "family_classic"},
    "Pinocchio": {"wiki": 72, "trends": 68, "youtube": 70, "spotify": 66, "gender": "male", "category": "family_classic"},
    "Don Quixote": {"wiki": 88, "trends": 75, "youtube": 72, "spotify": 68, "gender": "male", "category": "classic_comedy"},
    "Giselle": {"wiki": 82, "trends": 72, "youtube": 65, "spotify": 60, "gender": "female", "category": "classic_romance"},
    "Nijinsky": {"wiki": 50, "trends": 45, "youtube": 48, "spotify": 40, "gender": "male", "category": "contemporary"},
    "Hansel and Gretel": {"wiki": 78, "trends": 70, "youtube": 65, "spotify": 62, "gender": "co", "category": "family_classic"},
    "Contemporary Composers": {"wiki": 60, "trends": 55, "youtube": 58, "spotify": 50, "gender": "na", "category": "contemporary"},
    "Notre Dame de Paris": {"wiki": 85, "trends": 88, "youtube": 80, "spotify": 76, "gender": "male", "category": "romantic_tragedy"},
    "Wizard of Oz": {"wiki": 97, "trends": 93, "youtube": 95, "spotify": 90, "gender": "female", "category": "family_classic"},
    "Grimm": {"wiki": 55, "trends": 52, "youtube": 50, "spotify": 45, "gender": "na", "category": "contemporary"},
    "Ballet Boyz": {"wiki": 45, "trends": 40, "youtube": 60, "spotify": 58, "gender": "male", "category": "contemporary"},
    "Frankenstein": {"wiki": 68, "trends": 63, "youtube": 66, "spotify": 62, "gender": "male", "category": "dramatic"},
}

# -------------------------
# SEGMENT MULTIPLIERS
# -------------------------
SEGMENT_MULT = {
    "General Population": {"female": 1.00, "male": 1.00, "co": 1.00, "family_classic": 1.00, "classic_romance": 1.00, "romantic_tragedy": 1.00, "contemporary": 1.00, "pop_ip": 1.00},
    "Core Classical (F35–64)": {"female": 1.12, "male": 0.95, "co": 1.05, "family_classic": 1.10, "classic_romance": 1.08, "romantic_tragedy": 1.05, "contemporary": 0.90, "pop_ip": 1.00},
    "Family (Parents w/ kids)": {"female": 1.10, "male": 0.90, "co": 1.05, "family_classic": 1.18, "classic_romance": 0.95, "romantic_tragedy": 0.85, "contemporary": 0.80, "pop_ip": 1.20},
    "Emerging Adults (18–34)": {"female": 1.02, "male": 1.02, "co": 1.00, "family_classic": 0.95, "classic_romance": 0.92, "romantic_tragedy": 0.90, "contemporary": 1.25, "pop_ip": 1.15},
}

# -------------------------
# REGION ADJUSTMENTS
# -------------------------
REGION_MULT = {"Province": 1.00, "Calgary": 1.05, "Edmonton": 0.95}

# -------------------------
# API CONFIGURATION (OPTIONAL)
# -------------------------
with st.expander("🔑 API Configuration"):
    yt_key = st.text_input("YouTube Data API v3 Key", type="password")
    sp_id = st.text_input("Spotify Client ID", type="password")
    sp_secret = st.text_input("Spotify Client Secret", type="password")
    use_live = st.checkbox("Use Live Data for Unknown Titles", value=False)

# -------------------------
# OPTIONS
# -------------------------
region = st.selectbox("Region", ["Province", "Calgary", "Edmonton"], index=0)
segment = st.selectbox("Audience Segment", list(SEGMENT_MULT.keys()), index=0)
titles = st.multiselect("Select titles to score:", list(BASELINES.keys()), default=list(BASELINES.keys())[:10])

# -------------------------
# SCORING
# -------------------------
rows = []
for title in titles:
    base = BASELINES[title]
    seg_mult = SEGMENT_MULT[segment]
    reg_mult = REGION_MULT[region]

    gender = base["gender"]
    cat = base["category"]

    fam = (base["wiki"] * 0.5 + base["trends"] * 0.3 + base["spotify"] * 0.2)
    mot = (base["youtube"] * 0.5 + base["trends"] * 0.3 + base["wiki"] * 0.2)

    fam *= seg_mult.get(gender, 1.0) * seg_mult.get(cat, 1.0) * reg_mult
    mot *= seg_mult.get(gender, 1.0) * seg_mult.get(cat, 1.0) * reg_mult

    rows.append({
        "Title": title,
        "Region": region,
        "Segment": segment,
        "Familiarity": round(fam, 1),
        "Motivation": round(mot, 1),
        "Category": cat,
        "Gender": gender
    })

df = pd.DataFrame(rows)

# Normalize to Nutcracker (100)
if "The Nutcracker" in df["Title"].values:
    nut_fam = df.loc[df["Title"]=="The Nutcracker","Familiarity"].values[0]
    nut_mot = df.loc[df["Title"]=="The Nutcracker","Motivation"].values[0]
    df["Familiarity"] = (df["Familiarity"] / nut_fam) * 100
    df["Motivation"] = (df["Motivation"] / nut_mot) * 100

# -------------------------
# DISPLAY
# -------------------------
st.success("Scoring complete.")
st.dataframe(df, use_container_width=True)

# Quadrant Plot
fig, ax = plt.subplots()
ax.scatter(df["Familiarity"], df["Motivation"])
for _, r in df.iterrows():
    ax.annotate(r["Title"], (r["Familiarity"], r["Motivation"]), fontsize=8)
ax.axvline(df["Familiarity"].median(), color='gray', linestyle='--')
ax.axhline(df["Motivation"].median(), color='gray', linestyle='--')
ax.set_xlabel("Familiarity (relative to Nutcracker)")
ax.set_ylabel("Motivation (relative to Nutcracker)")
ax.set_title(f"Alberta Ballet Familiarity vs Motivation — {segment} ({region})")
st.pyplot(fig)

# Bar charts
col1, col2 = st.columns(2)
with col1:
    st.bar_chart(df.set_index("Title")["Familiarity"])
with col2:
    st.bar_chart(df.set_index("Title")["Motivation"])

# Download
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download CSV", csv, "title_scores_v9_test.csv", "text/csv")
