# - Learns YYC/YEG splits from history.csv (or uploaded CSV)
# - Single ticket estimation only
# - Removes arbitrary 60/40 split; uses title→category→default fallback
# - Small fixes: softmax bug, LA attach loop, duplicate imports, safer guards

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

# Advanced ML models for regression
try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge, LinearRegression
    from sklearn.model_selection import cross_val_score, GridSearchCV
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import xgboost as xgb
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

def load_config(path: str = "config.yaml"):
    global SEGMENT_MULT, REGION_MULT
    global DEFAULT_BASE_CITY_SPLIT, _CITY_CLIP_RANGE
    global POSTCOVID_FACTOR, TICKET_BLEND_WEIGHT
    global K_SHRINK, MINF, MAXF, N_MIN
    global DEFAULT_MARKETING_SPT_CITY
    # New robust forecasting settings
    global ML_CONFIG, KNN_CONFIG, CALIBRATION_CONFIG

    if yaml is None:
        # PyYAML not installed – just use hard-coded defaults
        return

    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        # No config file – keep defaults
        return

    # 1) Segment multipliers
    if "segment_mult" in cfg:
        SEGMENT_MULT = cfg["segment_mult"]

    # 2) Region multipliers
    if "region_mult" in cfg:
        REGION_MULT = cfg["region_mult"]

    # 3) City splits
    city_cfg = cfg.get("city_splits", {})
    DEFAULT_BASE_CITY_SPLIT = city_cfg.get("default_base_city_split", DEFAULT_BASE_CITY_SPLIT)
    _CITY_CLIP_RANGE = tuple(city_cfg.get("city_clip_range", _CITY_CLIP_RANGE))

    # 4) Demand knobs
    demand_cfg = cfg.get("demand", {})
    POSTCOVID_FACTOR = demand_cfg.get("postcovid_factor", POSTCOVID_FACTOR)
    TICKET_BLEND_WEIGHT = demand_cfg.get("ticket_blend_weight", TICKET_BLEND_WEIGHT)

    # 5) Seasonality knobs
    seas_cfg = cfg.get("seasonality", {})
    K_SHRINK = seas_cfg.get("k_shrink", K_SHRINK)
    MINF = seas_cfg.get("min_factor", MINF)
    MAXF = seas_cfg.get("max_factor", MAXF)
    N_MIN = seas_cfg.get("n_min", N_MIN)

    # 6) Marketing defaults
    mkt_cfg = cfg.get("marketing_defaults", {})
    DEFAULT_MARKETING_SPT_CITY = mkt_cfg.get("default_marketing_spt_city", DEFAULT_MARKETING_SPT_CITY)

    # 7) New robust forecasting settings (opt-in)
    ML_CONFIG = cfg.get("model", {"path": "models/model_xgb_remount_postcovid.joblib", "use_for_cold_start": True})
    KNN_CONFIG = cfg.get("knn", {"enabled": True, "k": 5})
    CALIBRATION_CONFIG = cfg.get("calibration", {"enabled": False, "mode": "global"})

# Default values for new config settings (used if config.yaml doesn't have them)
ML_CONFIG = {"path": "models/model_xgb_remount_postcovid.joblib", "use_for_cold_start": True}
KNN_CONFIG = {"enabled": True, "k": 5}
CALIBRATION_CONFIG = {"enabled": False, "mode": "global"}

def _pct(v, places=0):
    try:
        return f"{float(v):.{places}%}"
    except Exception:
        return "—"

def _num(v):
    try:
        return f"{int(round(float(v))):,}"
    except Exception:
        return "—"

def _dec(v, places=3):
    try:
        return f"{float(v):.{places}f}"
    except Exception:
        return "—"

def _short_month(m_full: str) -> str:
    # "September 2026" -> "Sep-26"
    import calendar
    try:
        name, year = str(m_full).split()
        m_num = list(calendar.month_name).index(name)
        return f"{calendar.month_abbr[m_num]}-{str(int(year))[-2:]}"
    except Exception:
        return str(m_full)

def _make_styles():
    ss = getSampleStyleSheet()
    styles = {
        "h1": ParagraphStyle("h1", parent=ss["Heading1"], fontSize=16, leading=20, spaceAfter=12),
        "h2": ParagraphStyle("h2", parent=ss["Heading2"], fontSize=13, leading=16, spaceAfter=8),
        "body": ParagraphStyle("body", parent=ss["BodyText"], fontSize=10.5, leading=14),
        "small": ParagraphStyle("small", parent=ss["BodyText"], fontSize=9, leading=12, textColor=colors.grey),
        "mono": ParagraphStyle("mono", parent=ss["BodyText"], fontName="Helvetica", fontSize=9.5, leading=13),
    }
    return styles

def _methodology_glossary_text() -> list:
    """
    Build the 'Methodology & Glossary' section for the PDF report.

    Returns a list of ReportLab Flowables (Paragraphs, Spacers).
    """
    styles = _make_styles()
    P = Paragraph
    SP = Spacer
    out = []

    # ─────────────────────────────────────────────────────────
    # 1. High-level description
    # ─────────────────────────────────────────────────────────
    out.append(P("How this forecast works", styles["h1"]))

    out.append(P(
        "This model is a structured, rule-based demand forecasting system that converts "
        "online visibility signals into ticket expectations. It uses four measurable "
        "sources of public interest—Wikipedia traffic, Google Trends, YouTube views, "
        "and Spotify popularity—to generate “Familiarity” and “Motivation” scores for "
        "each show. These scores are evaluated against a benchmark title and passed "
        "through calibrated relationships derived from Alberta Ballet’s real history: "
        "month-by-month seasonality, Calgary/Edmonton audience differences, "
        "and remount behaviour. The model then "
        "translates these calibrated interest scores into expected ticket totals, "
        "revenue, and recommended marketing spend for any title you evaluate, including "
        "new works.",
        styles["body"],
    ))
    out.append(SP(1, 6))

    out.append(P("How the model uses history (retrodiction, not circularity)", styles["h2"]))
    out.append(P(
        "The model uses historical sales to learn general patterns, but it never uses "
        "the past ticket total of a specific title to predict that same title’s future. "
        "Instead, it performs what economists call <b>retrodiction</b>—re-estimating "
        "past shows using the same formula it uses for new shows. Retrodiction is not "
        "a scientific measure of true forecast accuracy, but a calibration test: it "
        "shows whether the shape of the interest-to-ticket mapping makes sense, whether "
        "the model is too flat or too steep, and which titles the logic systematically "
        "over- or under-estimates. This is the same technique Broadway economics teams "
        "use when building benchmarked demand curves; the model checks its alignment "
        "with reality without simply copying past results.",
        styles["body"],
    ))
    out.append(SP(1, 10))

    # ─────────────────────────────────────────────────────────
    # 2. How to read the accuracy numbers
    # ─────────────────────────────────────────────────────────
    out.append(P("How to read the accuracy numbers", styles["h2"]))
    out.append(P(
        "When we test the model on past seasons, the dashboard shows three standard "
        "metrics: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R². "
        "MAE and RMSE are expressed in tickets and indicate, on average, how far the "
        "model’s estimates are from actual results (RMSE penalises large misses more "
        "heavily). R² is a 0–1 style measure of how much of the variation between "
        "stronger and weaker titles the model can explain. For example, an R² around "
        "0.40–0.45 means the model explains roughly 40–45% of the differences in "
        "historical ticket sales between titles. These figures describe how well the "
        "model back-fits history; they are best interpreted as a calibration check, "
        "not a guarantee that future seasons will match the same percentages exactly.",
        styles["body"],
    ))
    out.append(SP(1, 10))

    # ─────────────────────────────────────────────────────────
    # 3. Quick-read bullets (original content, kept)
    # ─────────────────────────────────────────────────────────
    out.append(P("How this forecast works (quick read)", styles["h2"]))

    bullets = [
        "We combine online visibility (Wikipedia, YouTube, Google) into two simple "
        "ideas: <b>Familiarity</b> (people know it) and <b>Motivation</b> "
        "(people engage with it).",
        "We anchor everything to a <b>benchmark title</b> so scores are on a shared "
        "0–100+ scale.",
        "We connect those scores to real ticket history to estimate a "
        "<b>Ticket Index</b> for each title.",
        "We adjust for the <b>month</b> you plan to run it (some months sell better), "
        "and for <b>recency</b> if it’s a quick remount.",
        "We split totals between <b>Calgary</b> and <b>Edmonton</b> using learned "
        "historical shares and then into <b>Singles</b>.",
        "We convert tickets into <b>revenue</b> using typical realized prices by "
        "city & product, and into recommended <b>marketing spend</b> using learned "
        "per-single-ticket marketing $ by title/category and city.",
    ]
    for b in bullets:
        out.append(P(f"• {b}", styles["body"]))
    out.append(SP(1, 10))

    # ─────────────────────────────────────────────────────────
    # 4. Plain-language glossary (original content, kept)
    # ─────────────────────────────────────────────────────────
    out.append(P("Plain-language glossary", styles["h2"]))

    glossary_items = [
        "<b>Familiarity</b>: how well-known the title is.",
        "<b>Motivation</b>: how keen people seem to be to watch it.",
        "<b>Ticket Index</b>: how that interest typically translates into tickets "
        "(vs the benchmark).",
        "<b>Seasonality</b>: some months sell better than others for a given type of show.",
        "<b>Remount</b>: recent repeats often sell a bit less; we reduce estimates accordingly.",
        "<b>YYC/YEG split</b>: we use your history to split totals between the two cities.",
        "<b>Revenue estimate</b>: Singles tickets in each city multiplied by typical "
        "realized prices from the last season.",
        "<b>Marketing spend per single</b>: historic median $ of paid media per sold single "
        "ticket, learned by title×city where possible, then category×city, then city-wide.",
        "<b>Marketing budget (YYC/YEG/Total)</b>: recommended paid-media spend = "
        "marketing $/single × forecast singles in each city.",
    ]
    for g in glossary_items:
        out.append(P(g, styles["body"]))

    return out

def _narrative_for_row(r: dict) -> str:
    title = r.get("Title",""); month = r.get("Month",""); cat = r.get("Category","")
    idx_used = r.get("TicketIndex used", None)
    f_season = r.get("FutureSeasonalityFactor", None)
    decay_pct = r.get("ReturnDecayPct", 0.0)

    yyc = (r.get("YYC_Singles",0) or 0)
    yeg = (r.get("YEG_Singles",0) or 0)
    c_share = r.get("CityShare_Calgary", None); e_share = r.get("CityShare_Edmonton", None)

    pri = r.get("PrimarySegment",""); sec = r.get("SecondarySegment","")

    # Revenue + marketing
    total_rev = r.get("Total_Revenue", 0) or 0
    total_mkt = r.get("Total_Mkt_Spend", 0) or 0
    singles_total = (
        (r.get("YYC_Singles", 0) or 0) +
        (r.get("YEG_Singles", 0) or 0)
    )
    try:
        mkt_per_single = (float(total_mkt) / float(singles_total)) if singles_total else 0.0
    except Exception:
        mkt_per_single = 0.0
		
    parts = []
    parts.append(f"<b>{month} — {title}</b> ({cat})")
    parts.append(f"Estimated demand comes from a combined interest score converted into a <b>Ticket Index</b> of {_dec(idx_used,1)}.")
    parts.append(f"For {month.split()[0]}, this category’s month factor is {_dec(f_season,3)} (months above 1.0 sell better).")
    if decay_pct and float(decay_pct) > 0:
        parts.append(f"We apply a small repeat reduction of {_pct(decay_pct)} due to recent performances.")
    if pri:
        parts.append(f"Likely audience skews to <b>{pri}</b>{' (then '+sec+')' if sec else ''}.")
    parts.append(
        f"We split sales using learned shares: Calgary {_pct(c_share,0)} / Edmonton {_pct(e_share,0)}, giving "
        f"<b>{_num(yyc)}</b> tickets in YYC and <b>{_num(yeg)}</b> in YEG."
    )
    if total_rev:
        parts.append(f"At typical realized prices this equates to about <b>${_num(total_rev)}</b> in total revenue.")
    if total_mkt:
        extra = f" (~${_dec(mkt_per_single,2)} in paid media per single)" if mkt_per_single else ""
        parts.append(f"Based on historic marketing spend per single, the recommended paid-media budget is about <b>${_num(total_mkt)}</b>{extra}.")
    return " ".join(parts)


def _build_month_narratives(plan_df: "pd.DataFrame") -> list:
    styles = _make_styles()
    blocks = [Paragraph("Season Rationale (by month)", styles["h1"])]
    for _, rr in plan_df.iterrows():
        blocks.append(Paragraph(_narrative_for_row(rr), styles["body"]))
        blocks.append(Spacer(1, 0.12*inch))
    blocks.append(Spacer(1, 0.2*inch))
    return blocks

def _make_season_table_wide(plan_df: pd.DataFrame) -> Table:
    """
    Backwards-compatible wrapper: for PDFs, use the same financial
    summary layout as the CSV so both stay in sync.
    """
    return _make_season_financial_summary_table_pdf(plan_df)


def build_season_financial_summary_table(plan_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a wide season summary with months as columns and rows:
      Show Title, Estimated Tickets, YYC/YEG Singles & Subs,
      Revenue, Marketing, Production, Net Income (Deficit),
    with all $ values formatted with dollar signs (table + CSV).
    """
    # Month order & labels
    month_order = ["September", "October", "January", "February", "March", "May"]
    label_map = {m: m.upper() for m in month_order}

    df = plan_df.copy()
    df["_mname"] = df["Month"].astype(str).str.split().str[0]
    df = df[df["_mname"].isin(month_order)].copy()
    if df.empty:
        return pd.DataFrame()

    order_map = {m: i for i, m in enumerate(month_order)}
    df["_order"] = df["_mname"].map(order_map)
    df = df.sort_values("_order")

    month_cols = [label_map[m] for m in df["_mname"].tolist()]

    index_labels = [
        "Show Title",
        "Estimated Tickets",
        "YYC Singles",
        "YEG Singles",
        "",
        "REVENUE",
        "YYC Singles Revenue",
        "YEG Singles Revenue",
        "Total Revenue",
        "",
        "EXPENSES",
        "Average YYC Marketing Spend",
        "Average YEG Marketing Spend",
        "YYC Marketing Spend",
        "YEG Marketing Spend",
        "Total Marketing Spend",
        "Total Production Expenses",
        "Total Expenses",
        "",
        "Net Income (Deficit)",
    ]

    out = pd.DataFrame(index=index_labels, columns=month_cols, dtype=object)

    for col_name, (_, r) in zip(month_cols, df.iterrows()):
        # Tickets
        est_tix = r.get("EstimatedTickets_Final", r.get("EstimatedTickets", np.nan))
        out.at["Show Title", col_name] = r.get("Title", "")
        out.at["Estimated Tickets", col_name] = int(round(est_tix)) if pd.notna(est_tix) else ""

        out.at["YYC Singles", col_name] = int(r.get("YYC_Singles", 0) or 0)
        out.at["YEG Singles", col_name] = int(r.get("YEG_Singles", 0) or 0)

        # Revenue (numeric for now; we format later)
        out.at["YYC Singles Revenue", col_name]       = float(r.get("YYC_Single_Revenue", 0) or 0)
        out.at["YEG Singles Revenue", col_name]       = float(r.get("YEG_Single_Revenue", 0) or 0)
        out.at["Total Revenue", col_name]             = float(r.get("Total_Revenue", 0) or 0)

        # Expenses (numeric for now)
        out.at["Average YYC Marketing Spend", col_name] = float(r.get("YYC_Mkt_SPT", 0) or 0)
        out.at["Average YEG Marketing Spend", col_name] = float(r.get("YEG_Mkt_SPT", 0) or 0)

        yyc_mkt = float(r.get("YYC_Mkt_Spend", 0) or 0)
        yeg_mkt = float(r.get("YEG_Mkt_Spend", 0) or 0)
        tot_mkt = float(r.get("Total_Mkt_Spend", 0) or (yyc_mkt + yeg_mkt))

        out.at["YYC Marketing Spend", col_name] = yyc_mkt
        out.at["YEG Marketing Spend", col_name] = yeg_mkt
        out.at["Total Marketing Spend", col_name] = tot_mkt

        prod = float(r.get("Prod_Expense", 0) or 0)
        out.at["Total Production Expenses", col_name] = prod

        total_exp = tot_mkt + prod
        out.at["Total Expenses", col_name] = total_exp

        net = float(r.get("Total_Revenue", 0) or 0) - total_exp
        out.at["Net Income (Deficit)", col_name] = net

    # --- Format all $ fields with dollar signs (for table + CSV) ---
    currency_rows = [
        "YYC Singles Revenue",
        "YEG Singles Revenue",
        "Total Revenue",
        "Average YYC Marketing Spend",
        "Average YEG Marketing Spend",
        "YYC Marketing Spend",
        "YEG Marketing Spend",
        "Total Marketing Spend",
        "Total Production Expenses",
        "Total Expenses",
        "Net Income (Deficit)",
    ]

    for row in currency_rows:
        if row not in out.index:
            continue
        formatted_vals = []
        for col in out.columns:
            v = out.at[row, col]
            if v == "" or pd.isna(v):
                formatted_vals.append("")
                continue
            try:
                fv = float(v)
            except Exception:
                formatted_vals.append(str(v))
                continue

            # Averages keep 2 decimals; everything else whole dollars
            if "Average" in row:
                formatted_vals.append(f"${fv:,.2f}")
            else:
                formatted_vals.append(f"${fv:,.0f}")
        out.loc[row] = formatted_vals

    return out


def _make_season_financial_summary_table_pdf(plan_df: pd.DataFrame) -> Table:
    """
    Convert the season financial summary DataFrame into a ReportLab Table
    for the PDF, using the same structure as the CSV.
    """
    summary_df = build_season_financial_summary_table(plan_df)
    if summary_df.empty:
        return Table([["No season data"]])

    # Header row
    rows = [["Metric"] + list(summary_df.columns)]

    # One row per index label from the CSV (including blank spacer rows)
    for idx, row in summary_df.iterrows():
        vals = []
        for col in summary_df.columns:
            v = row[col]
            if isinstance(v, str):
                vals.append(v)
            elif pd.isna(v):
                vals.append("")
            else:
                vals.append(str(v))
        rows.append([idx] + vals)

    table = Table(rows, repeatRows=1)
    table.setStyle(TableStyle([
        ("FONT", (0,0), (-1,-1), "Helvetica", 9),
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#f0f0f5")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.HexColor("#222222")),
        ("LINEABOVE", (0,0), (-1,0), 0.75, colors.black),
        ("LINEBELOW", (0,0), (-1,0), 0.75, colors.black),
        ("ALIGN", (1,1), (-1,-1), "CENTER"),
        ("ALIGN", (0,0), (0,-1), "LEFT"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.whitesmoke, colors.lightgrey]),
        ("GRID", (0,0), (-1,-1), 0.25, colors.HexColor("#aaaaaa")),
    ]))
    return table

def build_full_pdf_report(methodology_paragraphs: list,
                          plan_df: "pd.DataFrame",
                          season_year: int,
                          org_name: str = "Alberta Ballet") -> bytes:
    """
    Returns a PDF as bytes containing:
      1) Title page
      2) Season Rationale (per month/title)
      3) Methodology & Glossary
      4) Season Table (months as columns)
    """
    styles = _make_styles()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=landscape(LETTER),
        leftMargin=0.5*inch, rightMargin=0.5*inch,
        topMargin=0.5*inch, bottomMargin=0.5*inch,
        title=f"{org_name} — Season Report {season_year}"
    )
    story = []

    # Title
    story.append(Paragraph(f"{org_name} — Season Report ({season_year})", styles["h1"]))
    story.append(Paragraph(
        "Familiarity & Motivation • Ticket Index • Seasonality • Remount • City/Segment splits • Revenue & Marketing spend",
        styles["small"]
    ))
    story.append(Spacer(1, 0.25*inch))

    # (1) Season Rationale
    story.extend(_build_month_narratives(plan_df))
    story.append(PageBreak())

    # (2) Methodology & Glossary
    story.extend(methodology_paragraphs)
    story.append(PageBreak())

    # (3) Season Table (months as columns)
    story.append(Paragraph("Season Table (months as columns)", styles["h1"]))
    story.append(Paragraph(
        "Key metrics by month. Indices are benchmark-normalized; tickets include future seasonality and remount adjustments.",
        styles["small"]))
    story.append(Spacer(1, 0.15*inch))
    story.append(_make_season_financial_summary_table_pdf(plan_df))

    doc.build(story)
    return buf.getvalue()


# -------------------------
# App setup
# -------------------------
st.set_page_config(page_title="Alberta Ballet — Title Familiarity & Motivation Scorer", layout="wide")

# -------------------------
# Configuration Validation at Startup
# -------------------------
try:
    from config.validation import validate_config, validate_data_files
    
    config_ok, config_errors = validate_config()
    data_ok, data_errors = validate_data_files()
    
    if not config_ok or not data_ok:
        all_errors = config_errors + data_errors
        st.error("⚠️ **Configuration Validation Failed**")
        for err in all_errors:
            st.warning(f"• {err}")
        st.info(
            "Please fix the configuration issues above before using the app. "
            "See README.md for required file formats."
        )
except ImportError:
    # config.validation module not available - continue without validation
    pass
except Exception as e:
    st.warning(f"Configuration validation skipped: {e}")

if "results" not in st.session_state:
    st.session_state["results"] = None

st.title("Alberta Ballet — Title Scorer (v9.2)")
st.caption("Hard-coded AB-wide baselines (normalized to your benchmark = 100). Add new titles; choose live fetch or offline estimate.")

# -------------------------
# USER INSTRUCTIONS (Expander)
# -------------------------
with st.expander("👋 How to use this app (step-by-step)"):
    st.markdown(dedent("""
    ## Quick Build a Season — User Guide
    1. **Scroll down to _📅 Build a Season_**.
    2. **Select a title** from the drop-down for each month you want to schedule.
    3. **Review the Season table**:
       - Scroll to the bottom rows to see **EstimatedTickets_Final**, **Calgary vs Edmonton** totals, Singles, and other key metrics.
    4. **Export**:
       - Use **Download Season (wide) CSV** for spreadsheet analysis, or
       - **Download Full PDF Report** for a narrative + methodology + condensed season table.

    **What this tool does (in brief):**  
    Estimates ticket demand from title familiarity & motivation, links to sales history, then applies seasonality, remount decay, and learned city/subscriber splits.

    ### Full User Guide (5 steps)
    1. **(Optional) Load history:** In *Historicals*, upload your ticket history CSV or rely on `data/productions/history_city_sales.csv`.
    2. **Choose titles:** Add/modify the list in **Titles to score**. Unknown titles are estimated (or fetched live if you turn that on).
    3. **(Optional) Seasonality:** Toggle **Apply seasonality** and pick an assumed run month (affects indices & tickets).
    4. **Pick a benchmark:** Select the **Benchmark Title** to normalize indices (benchmark = 100).
    5. **Click _Score Titles_**, then use **📅 Build a Season** to assign titles to months.

    ### Interpreting results
    - **Familiarity / Motivation** → index vs benchmark (100).
    - **Ticket Index used** → signal→ticket linkage after seasonality.
    - **EstimatedTickets_Final** → tickets after remount decay.
    - **YYC / YEG** + **Singles / Subs** → learned splits with sensible fallbacks.

    ### Build a Season (assign titles to months)
    - Select one title for each month shown.
    - The app recomputes the **FutureSeasonalityFactor** for the chosen month/year, applies remount decay, and outputs city/sub splits.
    - Download **CSV (wide)** or the **Full PDF Report** (narrative + methodology + wide table).

    ### File outputs
    - **CSV (wide)**: rows = metrics, columns = months you picked.
    - **Full PDF Report**: 
      1) Season rationale by month  
      2) Methodology & glossary (plain-language)  
      3) Season table (months as columns; condensed metrics for readability)

    ### Tips & guardrails
    - If your history is thin, the model backs off to **category** or **overall** fits, then to **signals-only**.
    - Seasonality uses **Category×Month** medians with shrinkage & clipping to avoid overfitting.
    - You can safely change the benchmark at any time; all indices re-normalize instantly.

    ### Common issues
    - **Empty results:** Ensure you have at least one title in the text area and click **Score Titles**.
    - **History columns don’t match:** The learner tries multiple header variants (e.g., “Single Tickets - Calgary”). If needed, rename your headers or include the words *Single* and *Calgary/Edmonton* in them.
    - **PDF table too wide:** The PDF intentionally uses a condensed metric set. For full detail, export CSV.

    ### Data privacy
    - API keys (YouTube/Spotify) are optional and only used for unknown titles when **Use Live Data** is ON.
    """))

# -------------------------
# METHODOLOGY & GLOSSARY SECTION
# -------------------------
with st.expander("📘 About This App — Methodology & Glossary"):
    st.markdown(dedent("""
    ## Simple Terms
    This tool is basically a planning calculator. It looks at how well-known a show is (**Familiarity**) and how excited people seem to be about it (**Motivation**), then turns that into an estimate of how many tickets you might sell. To do that, it pulls clues from the internet (Wikipedia, Google, YouTube, Spotify) and combines them with what actually happened in your past seasons. It also remembers how your sales usually split between Calgary and Edmonton.

    On top of that, it adjusts for timing and repeats. Some months are just stronger than others, so the tool nudges each title up or down based on when you plan to run it. If you’re remounting something that ran recently, it assumes demand will be a bit lower than the first time and applies a sensible haircut. From there, it uses typical average ticket prices to turn ticket counts into rough revenue numbers for each city and for single tickets. If you provide marketing history, it also learns roughly how many dollars of paid media you’ve usually spent per single ticket and uses that to suggest marketing budgets by show and by city. The end result is one view that ties together demand, timing, cities, audience segments, revenue, and a ballpark paid-media ask for each title and for the season as a whole.

    Finally, if you give it marketing history, it learns roughly how many dollars of paid media you’ve usually spent per single ticket in Calgary and Edmonton for different kinds of shows. When you build a season, it uses those “dollars per single” figures to suggest marketing budgets city by city, and shows you the total season spend and spend per single. The end result: one view that ties together demand, timing, cities, audience segments, revenue, and a ballpark paid-media ask for each title and for the season as a whole.   
    ---
    
	### Purpose
    This tool estimates how recognizable a title is (**Familiarity**) and how strongly audiences are inclined to attend (**Motivation**) and then converts those indices into **ticket forecasts**. It blends online visibility signals with learned priors from your historical sales (including YYC/YEG split and Singles mix), applies **seasonality** and **remount decay**, and outputs Alberta-wide projections plus a season builder with **revenue** and **marketing spend** recommendations.

    ---
    ## Methods (end-to-end)

    ### 1) Inputs & signals
    **Online signals (index-scaled; higher = stronger)**  
    - **Wikipedia**: average daily pageviews over the past year for the best-match page.  
      Index formula: `40 + min(110, 20·ln(1+views/day))`.  
    - **Google Trends**: proxy for active interest (lightweight heuristic when offline).  
    - **YouTube**: engagement intensity from median view counts across relevant results.  
      Index formula: `50 + min(90, 9·ln(1+median_views))`, then **winsorized** by title **category** (3rd–97th pct).  
    - **Spotify**: 80th-percentile track popularity near the query (fallback heuristic if API not used).

    **History & context**  
    - **Ticket priors**: per-title median tickets (from your `TICKET_PRIORS_RAW`).  
    - **Past runs**: title → `(start, end)` dates to derive a mid-run month for seasonality learning and remount timing.  
    - **Marketing spend history (optional)**: `data/productions/marketing_spend_per_ticket.csv` with per single-ticket $ spend by city for past shows.
	
    > Unknown titles can be scored **two ways**:
    > - **Live** (optional keys): Wikipedia + YouTube + Spotify lookups, then category winsorizing.  
    > - **Estimated** (no keys): baseline medians nudged by inferred category/gender.

    ### 2) Feature construction
    **(a) Familiarity & Motivation (raw, pre-normalization)**  
    \[
    \\text{Familiarity} = 0.55\\cdot Wiki + 0.30\\cdot Trends + 0.15\\cdot Spotify
    \]  
    \[
    \\text{Motivation} = 0.45\\cdot YouTube + 0.25\\cdot Trends + 0.15\\cdot Spotify + 0.15\\cdot Wiki
    \]

    **(b) Segment & region multipliers (applied to raw scores)**  
    - Segment: `SEGMENT_MULT[segment][gender] × SEGMENT_MULT[segment][category]`  
    - Region: `REGION_MULT[region]`  
    (In this app, compute is fixed at **Alberta-wide + General Population** for the normalized indices. Segmented mixes are produced later.)

    **(c) Normalization to a benchmark**  
    For each segment, scores are **divided by the benchmark title’s raw value** and ×100.  
    - Output per title: **Familiarity (index)** and **Motivation (index)**; we also keep `SignalOnly = mean(Familiarity, Motivation)`.

    ### 3) Ticket index (linking signals to sales)
    1. **De-seasonalize** history: for any title with a known median, divide by its **historical month factor** (Category×Month) to get `TicketIndex_DeSeason` **relative to the benchmark**.  
    2. **Fit simple linear models** (y = a·x + b, clipped to [20, 180]):  
       - One **overall** model if ≥5 known points.  
       - **Per-category** models if a category has ≥3 known points.  
    3. **Impute** missing `TicketIndex_DeSeason` using per-category model → overall model → (if neither available) mark “Not enough data”.  
    4. **Apply future month**: multiply by **FutureSeasonalityFactor** (Category×Month at the proposed run month) to get `EffectiveTicketIndex`.

    ### 4) Composite index & tickets
    - **Composite** balances signals and ticket linkage:  
      \[
      \\text{Composite} = (1-w)\\cdot SignalOnly + w\\cdot EffectiveTicketIndex,\\quad w=0.50
      \]
    - **Benchmark tickets**: the app infers a de-seasonalized benchmark ticket figure to convert indices back into tickets.  
      \[
      \\text{EstimatedTickets} = (\\text{EffectiveTicketIndex}/100)\\times \\text{BenchmarkTickets}_{\\text{de-season}}
      \]

    ### 5) Remount decay (recency adjustment)
    Based on years since the last run’s mid-date (using either the **proposed month** or current year if not set), remounts are reduced stepwise so that very fresh repeats get the largest haircut and older titles are only nudged. A typical pattern is up to ~25% reduction for a repeat within a year, easing down to ~5% once the title hasn’t appeared for several seasons.  
    Result is **EstimatedTickets_Final**.

    ### 6) YYC/YEG split and Singles allocation
    **Learning from history (wide schemas tolerated):**  
    - **Title-level city share** (Calgary vs Edmonton) from Singles+Subs totals, **clipped** to [0.15, 0.85].  
    - **Category-level city share** as weighted fallback.  
    - **Subscriber share** learned per **Category×City**, **clipped** to [0.05, 0.95].  
    **Fallbacks when thin/missing:**  
    - City split default = **60% Calgary / 40% Edmonton**.  
    -  defaults = **35% YYC / 45% YEG** (category-agnostic).  
    Final outputs: YYC/YEG totals and Singles by city.

    ### 7) Segment mix & per-segment tickets
    For each title, we compute **segment-specific signals** (re-scored vs the benchmark), then combine them with **segment priors** (`SEGMENT_PRIORS[region][category]`) and a softmax-like normalization to derive **shares** across:  
    - **General Population**, **Core Classical (F35–64)**, **Family (Parents w/ kids)**, **Emerging Adults (18–34)**.  
    These shares are applied to **EstimatedTickets_Final** to get per-segment ticket estimates and the **primary/secondary segment**.

    ### 8) Revenue & marketing spend recommendations
    **Revenue estimates**  
    - The app uses fixed **average realized ticket prices** from the last season:  
      - YYC Singles: `YYC_SINGLE_AVG`  
      - YEG Singles: `YEG_SINGLE_AVG`  
    - For each season plan (in **📅 Build a Season**), it multiplies:
      - YYC Singles / Subs counts by their YYC averages  
      - YEG Singles / Subs counts by their YEG averages  
      - and then sums to get **YYC_Revenue**, **YEG_Revenue**, and **Total_Revenue**.

    **Marketing spend per single ticket (SPT)**
    - From `data/productions/marketing_spend_per_ticket.csv` we learn typical **$ of paid media per sold single ticket**:
      - **Title×City median** $/single where data exist (e.g., ** in YYC).
      - **Category×City median** $/single as a fallback (e.g., `classic_romance` in YEG).
      - **City-wide median** $/single as a final default.
    - The helper `marketing_spt_for(title, category, city)` picks the best available prior:
      1. Title×City  
      2. Category×City  
      3. City-wide default

    **Recommended marketing budgets in the season builder**  
    - For each picked show in **📅 Build a Season**, the app:
      1. Forecasts final tickets in YYC and YEG (after seasonality & remount).  
      2. Retrieves **SPT** for YYC and YEG via `marketing_spt_for`.  
      3. Multiplies singles × SPT to get:
         - `YYC_Mkt_Spend` = YYC singles × YYC_SPT
         - `YEG_Mkt_Spend` = YEG singles × YEG_SPT
         - `Total_Mkt_Spend` = YYC + YEG  
    - These appear in the **Season table**, **wide CSV**, and in the **Season at a glance** header as the projected season-level marketing spend and $/single.
	
    ---
    ## Seasonality model (Category × Month)
    - Built from `PAST_RUNS` + `TICKET_PRIORS_RAW`.  
    - Within each category we first remove global outliers using an IQR fence on ticket medians.  
    - We compute a **trimmed median** ticket level for the category (dropping one high/low value when there are ≥4 runs).  
    - For each month we use:
      - A trimmed median for that (Category, Month) if there are at least **N_MIN = 3** runs.  
      - A **pooled winter median** for Dec/Jan/Feb when individual winter months are sparse.  
      - Otherwise, the category’s overall level.  
    - Raw month factors are the ratio (month median / overall median).  
    - **Shrinkage**: \( w = \\frac{n}{n+K} \), with **K = 5**, pulling low-sample months toward 1.0.  
    - **Clipping**: factors constrained to **[0.90, 1.15]** to avoid extreme up/down months.  
    - **Historical factor**: used to de-seasonalize per-title medians (when available).  
    - **Future factor**: used when you pick an **assumed run month**.

    ---
    ## Interpreting outputs
    - **Familiarity vs Motivation**: plot quadrants reveal whether a title is known but “sleepy” (high Familiarity, low Motivation) vs buzzy but less known (reverse).  
    - **Composite**: best single index for ranking if you plan to blend signal and sales history.  
    - **EstimatedTickets_Final**: planning number after **future seasonality** and **remount decay**.  
    - **Revenue columns**: YYC/YEG/Singles revenue built from ticket counts × typical realized prices — useful for season-level financial framing.  
    - **Marketing columns**: `YYC_Mkt_SPT`, `YEG_Mkt_SPT`, city-level and total marketing spend — a guide to how much paid media is typically required to support the forecast.  
    - **Segment & city breakouts**: use for campaign design, pricing tests, and inventory planning.

    ---
    ## Assumptions & guardrails
    - Linear link between **SignalOnly** and **TicketIndex_DeSeason** (by category where possible).  
    - Outlier control: **YouTube** is winsorized within category (3rd–97th percentile).  
    - Seasonality is conservative: outliers trimmed, winter months pooled, and factors shrunk and clipped.  
    - Benchmark normalization cancels unit differences across segments/region.  
    - Revenue uses **fixed average prices** (not a pricing model): treat as directional unless refreshed regularly.  
    - Marketing SPT uses **medians** and ignores extreme per single-ticket spends (>200$) to avoid campaign one-offs dominating.
	- Heuristics are used when live APIs are off or data are thin (clearly labelled in the UI).

    ---
    ## Tunable constants (advanced)
    - **TICKET_BLEND_WEIGHT** = `0.50`  
    - **K_SHRINK** = `5.0`; **MINF/MAXF** = `0.90/1.15` (seasonality)  
    - **N_MIN** (per Category×Month before trusting a specific month) = `3`  
    - **DEFAULT_BASE_CITY_SPLIT** = `Calgary 0.60 / Edmonton 0.40`, **_CITY_CLIP_RANGE** = `[0.15, 0.85]`  
    Note: This app focuses on single ticket estimation only.  
    - **Prediction clip** for ticket index: `[20, 180]`  
    - **SEGMENT_PRIOR_STRENGTH** (exponent on priors): `1.0` (tempering off)  
    - **DEFAULT_MARKETING_SPT_CITY**: initial city-wide per single-ticket $ before learning from marketing history (`Calgary 10 / Edmonton 8`).
	
    ---
    ## Limitations
    - Sparse categories/months reduce model power; app falls back to overall fits or signals-only where needed.  
    - Google Trends and Spotify heuristics are proxies when live APIs are off—treat as directional.  
    - Title disambiguation (e.g., films vs ballets) is handled heuristically; review unknown-title results.  
    - Revenue and marketing outputs assume that **historic per-ticket prices and per single-ticket spend are roughly stable**; if your pricing or media strategy shifts significantly, those columns should be re-anchored.
	
    ---
    ## Glossary
    - **Benchmark**: reference title for normalization (index 100 by definition).  
    - **Familiarity (index)**: awareness proxy normalized to the benchmark.  
    - **Motivation (index)**: intent/engagement proxy normalized to the benchmark.  
    - **SignalOnly**: mean of Familiarity and Motivation indices.  
    - **TicketIndex_DeSeason**: ticket potential (vs benchmark=100) after removing historical month bias.  
    - **EffectiveTicketIndex**: de-season ticket index × **FutureSeasonalityFactor**.  
    - **Composite**: blend of SignalOnly and EffectiveTicketIndex (w=0.50).  
    - **FutureSeasonalityFactor**: Category×Month factor for the planned run month.  
    - **HistSeasonalityFactor**: Category×Month factor for the title’s last historical run month.  
    - **Remount Decay / Factor**: staged-recency reduction and its (1−decay) multiplier.  
    - **Primary/Secondary Segment**: segments with the highest modeled shares.  
    - **YYC/YEG Split**: learned Calgary/Edmonton allocation for the title or its category.  
    - **Singles Mix**: learned subscriber share by **Category×City** (fallbacks if thin).  
    - **EstimatedTickets / Final**: projected tickets before/after remount decay.  
    - **YYC_/YEG_ Revenue**: revenue by city = Singles tickets × typical realized prices.  
    - **Marketing SPT (YYC_Mkt_SPT / YEG_Mkt_SPT)**: typical $ of paid media per sold single ticket in each city.
    - **Marketing Spend (YYC_Mkt_Spend / YEG_Mkt_Spend / Total_Mkt_Spend)**: recommended campaign budget derived from SPT × forecast singles.

   ---
    **Recommendation:** Use **Composite** to rank programs, **EstimatedTickets_Final** for capacity planning, use the **revenue columns** for financial framing, and the **marketing columns** to benchmark and budget your paid media for each title.
    """))

# -------------------------
# PRIORS learning (YYC/YEG) — self-contained
# Note: This app focuses on single ticket estimation only.
# -------------------------
# Globals populated from history
TITLE_CITY_PRIORS: dict[str, dict[str, float]] = {}
CATEGORY_CITY_PRIORS: dict[str, dict[str, float]] = {}

# Sensible fallbacks if history is missing/thin
DEFAULT_BASE_CITY_SPLIT = {"Calgary": 0.60, "Edmonton": 0.40}  # Calgary majority unless learned otherwise
_CITY_CLIP_RANGE = (0.15, 0.85)
# --- Average realized ticket prices (last season) ---
YYC_SINGLE_AVG = 85.47
YEG_SINGLE_AVG = 92.12


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

# Heuristics for unknown titles
def infer_gender_and_category(title: str) -> Tuple[str, str]:
    t = title.lower()
    gender = "na"
    female_keys = ["cinderella","sleeping","beauty and the beast","beauty","giselle","swan","widow","alice","juliet","sylphide","carmina burana"]
    male_keys = ["pinocchio","peter pan","don quixote","hunchback","hamlet","frankenstein","romeo","nijinsky"]
    if "romeo" in t and "juliet" in t: gender = "co"
    elif any(k in t for k in female_keys): gender = "female"
    elif any(k in t for k in male_keys): gender = "male"

    if any(k in t for k in ["wizard","peter pan","pinocchio","hansel","frozen","beauty","alice","die fledermaus"]):
        cat = "family_classic"
    elif any(k in t for k in ["swan","sleeping","cinderella","giselle","sylphide"]):
        cat = "classic_romance"
    elif any(k in t for k in ["romeo","hunchback","notre dame","hamlet","frankenstein"]):
        cat = "romantic_tragedy"
    elif any(k in t for k in ["don quixote","merry widow"]):
        cat = "classic_comedy"
    elif any(k in t for k in ["contemporary","boyz","ballet boyz","momix","complexions","grimm","nijinsky","shadowland","deviate","phi","der wolf","rite of spring","beethoven"]):
        cat = "contemporary"
    elif any(k in t for k in ["taj","tango","harlem","tragically hip","l cohen","leonard cohen"]):
        cat = "pop_ip"
    else:
        cat = "dramatic"
    return gender, cat

def infer_show_type(title: str, category: str) -> str:
    """
    Map a title + category to one of:
      contemporary_show, guest_company, mixed_bill, classical_ballet, family_ballet, or 'unknown'.
    """
    t = title.lower().strip()
    c = (category or "").lower().strip()

    # Guest companies (touring / visiting)
    guest_keys = [
        "trockadero", "ballet bc", "shaping sound",
        "dance theatre of harlem", "complexions",
        "bjm", "momix", "ballet boyz", "diavolo",
    ]
    if any(k in t for k in guest_keys) or c == "touring_contemporary_company":
        return "guest_company"

    # Mixed bills
    if "mixed bill" in t or any(k in t for k in ["unleashed", "deviate"]) or c == "contemporary_mixed_bill":
        return "mixed_bill"

    # Explicit contemporary one-offs you listed
    contemp_title_keys = ["nijinsky", "der wolf", "rite of spring", "grimm"]
    if any(k in t for k in contemp_title_keys):
        return "contemporary_show"

    # Family ballets – storybook / kids-focused titles
    family_title_keys = [
        "wizard of oz", "hansel", "cinderella",
        "peter pan", "pinocchio", "once upon a time",
        "jungle book", "addams family",
    ]
    if any(k in t for k in family_title_keys) or c == "family_classic":
        return "family_ballet"

    # Classical full-lengths (the big warhorses)
    classical_title_keys = [
        "sleeping beauty", "romeo and juliet", "swan lake",
        "giselle", "don quixote", "la sylphide",
        "coppelia", "la bayadere", "la fille mal gardée", "die fledermaus"
    ]
    if any(k in t for k in classical_title_keys) or c in ("classic_romance", "classic_comedy", "romantic_tragedy"):
        return "classical_ballet"

    # Adult lit drama – treat like contemporary_show for budgeting
    if c == "adult_literary_drama":
        return "contemporary_show"

    # Remaining contemporary / pop / dramatic → treat as contemporary_show for budgeting
    if c in ("contemporary", "pop_ip", "dramatic"):
        return "contemporary_show"

    return "unknown"

PROD_EXPENSE_TITLE: dict[str, float] = {}
PROD_EXPENSE_SHOWTYPE: dict[str, float] = {}

def learn_production_expenses(path: str = "data/productions/showtype_expense.csv") -> None:
    """
    Expects a CSV with columns:
      fiscal_year, title, prod_expense
    Uses all years, takes medians per title and per inferred show_type.
    """
    global PROD_EXPENSE_TITLE, PROD_EXPENSE_SHOWTYPE

    try:
        df = pd.read_csv(path)
    except Exception:
        PROD_EXPENSE_TITLE = {}
        PROD_EXPENSE_SHOWTYPE = {}
        return

    # Clean
    colmap = {c.lower().strip(): c for c in df.columns}
    tcol = colmap.get("title", "title")
    ecol = colmap.get("prod_expense", "prod_expense")

    df[tcol] = df[tcol].astype(str).str.strip()

    def _num(x):
        try:
            if pd.isna(x):
                return None
            s = str(x).strip().replace(",", "")
            if not s:
                return None
            return float(s)
        except Exception:
            return None

    df[ecol] = df[ecol].map(_num)
    # drop blank/zero-ish rows
    df = df[df[ecol].notna() & (df[ecol] > 0)]

    # 1) per-title median
    PROD_EXPENSE_TITLE = (
        df.groupby(tcol)[ecol]
          .median()
          .to_dict()
    )

    # 2) per-show_type median as fallback
    def _show_type_for(t: str) -> str:
        return infer_show_type(t, infer_gender_and_category(t)[1])

    df["show_type"] = df[tcol].map(_show_type_for)

    PROD_EXPENSE_SHOWTYPE = (
        df.groupby("show_type")[ecol]
          .median()
          .to_dict()
    )

# --- Marketing spend priors (per single-ticket, by title × city and category × city) ---
MARKETING_SPT_TITLE_CITY: dict[str, dict[str, float]] = {}      # {"Cinderella": {"Calgary": 7.0, "Edmonton": 5.5}, ...}
MARKETING_SPT_CATEGORY_CITY: dict[str, dict[str, float]] = {}   # {"classic_romance": {"Calgary": 9.0, "Edmonton": 7.5}, ...}
DEFAULT_MARKETING_SPT_CITY: dict[str, float] = {"Calgary": 10.0, "Edmonton": 8.0}  # overwritten by learner


def learn_marketing_spt_from_history(mkt_df: pd.DataFrame) -> dict:
    """
    Expects columns:
      - Show Title
      - Calgary Show Date
      - Edmonton Show Date
      - Marketing Spend - Calgary   (per single-ticket $)
      - Marketing Spend - Edmonton  (per single-ticket $)
    Learns:
      - MARKETING_SPT_TITLE_CITY[title][city]  (median $/single)
      - MARKETING_SPT_CATEGORY_CITY[category][city]
      - DEFAULT_MARKETING_SPT_CITY[city]       (city-wide median)
    """
    global MARKETING_SPT_TITLE_CITY, MARKETING_SPT_CATEGORY_CITY, DEFAULT_MARKETING_SPT_CITY
    MARKETING_SPT_TITLE_CITY = {}
    MARKETING_SPT_CATEGORY_CITY = {}
    DEFAULT_MARKETING_SPT_CITY = {"Calgary": 10.0, "Edmonton": 8.0}

    if mkt_df is None or mkt_df.empty:
        return {"title_city": 0, "cat_city": 0, "note": "empty marketing history"}

    df = mkt_df.copy()

    def _find_col(cands):
        lc = {c.lower().strip(): c for c in df.columns}
        for want in cands:
            if want.lower() in lc:
                return lc[want.lower()]
        for want in cands:
            for k, orig in lc.items():
                if want.lower() in k:
                    return orig
        return None

    title_col = _find_col(["Show Title","Title","Production","Show"])
    cal_col   = _find_col(["Marketing Spend - Calgary","Calgary Spend"])
    edm_col   = _find_col(["Marketing Spend - Edmonton","Edmonton Spend"])

    if title_col is None or cal_col is None or edm_col is None:
        return {"title_city": 0, "cat_city": 0, "note": "missing columns"}

    df[title_col] = df[title_col].astype(str).str.strip()

    def _num(x):
        try:
            if pd.isna(x): return 0.0
            return float(str(x).replace(",","").strip() or 0)
        except Exception:
            return 0.0

    df[cal_col] = df[cal_col].map(_num)
    df[edm_col] = df[edm_col].map(_num)

    # Long form: one row per (Title, Category, City, SPT)
    rows = []
    for _, r in df.iterrows():
        title = str(r[title_col]).strip()
        if not title:
            continue
        cat = infer_gender_and_category(title)[1]

        cal_spt = _num(r[cal_col])
        edm_spt = _num(r[edm_col])

        if cal_spt > 0:
            rows.append((title, cat, "Calgary", cal_spt))
        if edm_spt > 0:
            rows.append((title, cat, "Edmonton", edm_spt))

    if not rows:
        return {"title_city": 0, "cat_city": 0, "note": "no positive spend rows"}

    long_df = pd.DataFrame(rows, columns=["Title","Category","City","SPT"])

    # Title × City medians
    title_city_count = 0
    for (title, city), g in long_df.groupby(["Title","City"]):
        val = float(np.median(g["SPT"].values))
        if val <= 0 or val > 200:
            continue
        MARKETING_SPT_TITLE_CITY.setdefault(title, {})[city] = val
        title_city_count += 1

    # Category × City medians
    cat_city_count = 0
    for (cat, city), g in long_df.groupby(["Category","City"]):
        val = float(np.median(g["SPT"].values))
        if val <= 0 or val > 200:
            continue
        MARKETING_SPT_CATEGORY_CITY.setdefault(cat, {})[city] = val
        cat_city_count += 1

    # City-wide defaults (medians over all titles)
    for city, g in long_df.groupby("City"):
        val = float(np.median(g["SPT"].values))
        if val > 0 and val < 200:
            DEFAULT_MARKETING_SPT_CITY[city] = val

    return {"title_city": title_city_count, "cat_city": cat_city_count}

CATEGORY_FALLBACK = {
    "adult_literary_drama": "dramatic",
    "contemporary_mixed_bill": "contemporary",
    "touring_contemporary_company": "contemporary",
}

def marketing_spt_for(title: str, category: str, city: str) -> float:
    city_norm = (city or "").lower()
    if "calg" in city_norm:
        city_key = "Calgary"
    elif "edm" in city_norm:
        city_key = "Edmonton"
    else:
        city_key = city or "Calgary"

    t = (title or "").strip()

    # 1) Title × City wins if present
    if t in MARKETING_SPT_TITLE_CITY and city_key in MARKETING_SPT_TITLE_CITY[t]:
        return MARKETING_SPT_TITLE_CITY[t][city_key]

    # 2) Category × City, with fallback mapping for new categories
    cat_key = (category or "").strip()
    if cat_key not in MARKETING_SPT_CATEGORY_CITY and cat_key in CATEGORY_FALLBACK:
        cat_key = CATEGORY_FALLBACK[cat_key]

    if cat_key in MARKETING_SPT_CATEGORY_CITY and city_key in MARKETING_SPT_CATEGORY_CITY[cat_key]:
        return MARKETING_SPT_CATEGORY_CITY[cat_key][city_key]

    # 3) City-wide default
    return DEFAULT_MARKETING_SPT_CITY.get(city_key, 8.0)


def learn_priors_from_history(hist_df: pd.DataFrame) -> dict:
    """
    Wide schema support for:
      - Show Title
      - Single Tickets - Calgary / Edmonton
    Handles commas in numbers, blanks, and duplicate titles.
    Populates:
      - TITLE_CITY_PRIORS[title] = {'Calgary': p, 'Edmonton': 1-p}
      - CATEGORY_CITY_PRIORS[category] = {...}  (category inferred from title)
    
    Note: This app focuses on single ticket estimation only.
    """
    global TITLE_CITY_PRIORS, CATEGORY_CITY_PRIORS
    TITLE_CITY_PRIORS.clear(); CATEGORY_CITY_PRIORS.clear()

    if hist_df is None or hist_df.empty:
        return {"titles_learned": 0, "categories_learned": 0, "note": "empty history"}

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

    # if any are missing, create as zeros so it still learns with partial data
    for name, fallback in [(s_cgy, "__s_cgy__"), (s_edm, "__s_edm__")]:
        if name is None:
            df[fallback] = 0.0

    s_cgy = s_cgy or "__s_cgy__"
    s_edm = s_edm or "__s_edm__"

    # clean numerics: handle "7,734" → 7734
    def _num(x) -> float:
        try:
            if pd.isna(x): return 0.0
            return float(str(x).replace(",", "").strip() or 0)
        except Exception:
            return 0.0

    for c in [s_cgy, s_edm]:
        df[c] = df[c].map(_num)

    # clean titles
    if title_col is None:
        # bail if we truly can't find a title column
        return {"titles_learned": 0, "categories_learned": 0, "note": "missing Show Title"}
    df[title_col] = df[title_col].astype(str).str.strip()

    # aggregate duplicates by title
    agg = (
        df.groupby(title_col)[[s_cgy, s_edm]]
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
        if any(k in tl for k in ["contemporary","boyz","momix","complexions","grimm","nijinsky","deviate","phi","away we go","unleashed","botero","ballet bc","der wolf","rite of spring","beethoven"]):
            return "contemporary"
        if any(k in tl for k in ["taj","tango","harlem","tragically hip","leonard cohen","joni","david bowie","gordon lightfoot","phi"]):
            return "pop_ip"
        return "dramatic"

    agg["Category"] = agg["Title"].apply(_infer_cat)

    # totals by city (single tickets only)
    agg["YYC_total"] = agg[s_cgy].fillna(0)
    agg["YEG_total"] = agg[s_edm].fillna(0)

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

    return {
        "titles_learned": titles_learned,
        "categories_learned": categories_learned,
    }

def city_split_for(title: str | None, category: str | None) -> dict[str, float]:
    """Prefer title prior, then category prior, else default."""
    if title and title in TITLE_CITY_PRIORS:
        return TITLE_CITY_PRIORS[title]
    if category and category in CATEGORY_CITY_PRIORS:
        return CATEGORY_CITY_PRIORS[category]
    return DEFAULT_BASE_CITY_SPLIT.copy()

def subs_share_for(category: str | None, city: str) -> float:
    """
    SINGLE-TICKET MODE:
    All tickets produced by the model are treated as single tickets.
    Subscription sales are handled through a different campaign and are
    not part of this forecasting tool.

    Therefore, the subscription share is always 0.
    """
    return 0.0

# --- Historicals (loads your wide CSV and learns) ---
with st.expander("Historicals (optional): upload or use local CSV", expanded=False):
    uploaded_hist = st.file_uploader("Upload historical ticket CSV", type=["csv"], key="hist_uploader_v9")
    relearn = st.button("🔁 Force re-learn from history", width='content')

# (Re)load the history df
if ("hist_df" not in st.session_state) or relearn:
    if uploaded_hist is not None:
        st.session_state["hist_df"] = pd.read_csv(uploaded_hist)
    else:
        # try your preferred filename first, then the old one; else empty
        try:
            st.session_state["hist_df"] = pd.read_csv("data/productions/history_city_sales.csv")
        except Exception:
            try:
                st.session_state["hist_df"] = pd.read_csv("data/productions/history.csv")
            except Exception:
                st.session_state["hist_df"] = pd.DataFrame()

# (Re)learn priors every time we (re)load history
st.session_state["priors_summary"] = learn_priors_from_history(st.session_state["hist_df"])

s = st.session_state.get("priors_summary", {}) or {}
st.caption(
    f"Learned priors → titles: {s.get('titles_learned',0)}, "
    f"categories: {s.get('categories_learned',0)}, "
    f"Note: Single tickets only"
)
# --- Marketing spend (fixed CSV from disk) ---
try:
    mkt_df = pd.read_csv("data/productions/marketing_spend_per_ticket.csv")
except Exception:
    mkt_df = pd.DataFrame()

mkt_summary = learn_marketing_spt_from_history(mkt_df)
st.caption(
    f"Marketing priors → title×city: {mkt_summary.get('title_city',0)}, "
    f"category×city: {mkt_summary.get('cat_city',0)}"
)
# --- Production expenses (per title / show type) ---
learn_production_expenses("data/productions/showtype_expense.csv")
# --- In-app production expense budgeting (optional overrides) ---
with st.expander("💰 Production expense budgeting (optional overrides)"):
    st.markdown(
        "Use these to set *budgeted* per-run production expenses by show type "
        "(and optionally by title). These override the history-based medians "
        "when forecasting future seasons."
    )

    # Initialize session defaults from learned history (once)
    if "budget_prod_expense_showtype" not in st.session_state:
        st.session_state["budget_prod_expense_showtype"] = PROD_EXPENSE_SHOWTYPE.copy()
    if "budget_prod_expense_title" not in st.session_state:
        st.session_state["budget_prod_expense_title"] = {}

    # --- Show-type level controls ---
    if not PROD_EXPENSE_SHOWTYPE:
        st.info("No production expense history loaded yet; overrides will apply once history is available.")
    else:
        st.subheader("By show type (category-level budgets)")
        new_showtype_budgets: dict[str, float] = {}

        for stype, hist_val in sorted(PROD_EXPENSE_SHOWTYPE.items()):
            default_val = float(st.session_state["budget_prod_expense_showtype"].get(stype, hist_val))
            budget_val = st.number_input(
                f"{stype}",
                min_value=0.0,
                value=default_val,
                step=10_000.0,
                format="%.0f",
                key=f"budget_stype_{stype}",
            )
            new_showtype_budgets[stype] = budget_val
            st.caption(f"Historical median for {stype}: ${hist_val:,.0f}")

        # save back to session
        st.session_state["budget_prod_expense_showtype"] = new_showtype_budgets

    # --- Optional: title-specific overrides ---
    st.subheader("Optional: per-title overrides")
    with st.form("prod_budget_title_form", clear_on_submit=True):
        title_for_override = st.text_input("Title (exact match to the season builder title)")
        title_budget = st.number_input(
            "Budgeted production expense for this title",
            min_value=0.0,
            step=10_000.0,
            format="%.0f",
        )
        submitted = st.form_submit_button("Add / update title override")
        if submitted and title_for_override.strip():
            st.session_state["budget_prod_expense_title"][title_for_override.strip()] = float(title_budget)
            st.success(f"Override set for '{title_for_override.strip()}': ${title_budget:,.0f}")

    # Show current title overrides
    if st.session_state["budget_prod_expense_title"]:
        st.markdown("**Current title-level overrides:**")
        for t, v in st.session_state["budget_prod_expense_title"].items():
            st.write(f"- {t}: ${v:,.0f}")

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
# Data (from CSV: baselines)
# -------------------------

BASELINES: dict[str, dict] = {}

def load_baselines(path: str = "data/productions/baselines.csv") -> None:
    """
    Load baseline familiarity/motivation inputs from CSV.

    Expected columns (case-insensitive):
        title, wiki, trends, youtube, spotify, category, gender
    """
    global BASELINES

    try:
        df = pd.read_csv(path)
    except Exception as e:
        # Hard fail (or soften this if you prefer)
        st.error(f"Could not load baselines CSV at '{path}': {e}")
        BASELINES = {}
        return

    # Normalize column names
    colmap = {c.lower(): c for c in df.columns}
    required = {"title", "wiki", "trends", "youtube", "spotify", "category", "gender"}
    missing = [c for c in required if c not in colmap]
    if missing:
        st.error(f"Baselines CSV is missing required columns: {', '.join(missing)}")
        BASELINES = {}
        return

    df[colmap["title"]] = df[colmap["title"]].astype(str).str.strip()

    baselines: dict[str, dict] = {}
    for _, r in df.iterrows():
        title = str(r[colmap["title"]]).strip()
        if not title:
            continue
        baselines[title] = {
            "wiki":     float(r[colmap["wiki"]]),
            "trends":   float(r[colmap["trends"]]),
            "youtube":  float(r[colmap["youtube"]]),
            "spotify":  float(r[colmap["spotify"]]),
            "category": str(r[colmap["category"]]),
            "gender":   str(r[colmap["gender"]]),
        }

    BASELINES = baselines

# Load immediately so everything below can use BASELINES
load_baselines()

SEGMENT_MULT = {
    "General Population": {
        "female": 1.00, "male": 1.00, "co": 1.00, "na": 1.00,
        "family_classic": 1.00,
        "classic_romance": 1.00,
        "romantic_tragedy": 1.00,
        "classic_comedy": 1.00,
        "contemporary": 1.00,
        "pop_ip": 1.00,
        "dramatic": 1.00,
        # new categories – neutral
        "adult_literary_drama": 1.00,          # like dramatic
        "contemporary_mixed_bill": 1.00,       # like contemporary
        "touring_contemporary_company": 1.00,  # like contemporary
    },
    "Core Classical (F35–64)": {
        "female": 1.12, "male": 0.95, "co": 1.05, "na": 1.00,
        "family_classic": 1.10,
        "classic_romance": 1.08,
        "romantic_tragedy": 1.05,
        "classic_comedy": 1.02,
        "contemporary": 0.90,
        "pop_ip": 1.00,
        "dramatic": 1.00,
        "adult_literary_drama": 1.00,      # like dramatic
        "contemporary_mixed_bill": 0.90,   # like contemporary
        "touring_contemporary_company": 0.90,
    },
    "Family (Parents w/ kids)": {
        "female": 1.10, "male": 0.92, "co": 1.06, "na": 1.00,
        "family_classic": 1.18,
        "classic_romance": 0.95,
        "romantic_tragedy": 0.85,
        "classic_comedy": 1.05,
        "contemporary": 0.82,
        "pop_ip": 1.20,
        "dramatic": 0.90,
        "adult_literary_drama": 0.90,      # like dramatic
        "contemporary_mixed_bill": 0.82,   # like contemporary
        "touring_contemporary_company": 0.82,
    },
    "Emerging Adults (18–34)": {
        "female": 1.02, "male": 1.02, "co": 1.00, "na": 1.00,
        "family_classic": 0.95,
        "classic_romance": 0.92,
        "romantic_tragedy": 0.90,
        "classic_comedy": 0.98,
        "contemporary": 1.25,
        "pop_ip": 1.15,
        "dramatic": 1.05,
        "adult_literary_drama": 1.05,          # like dramatic
        "contemporary_mixed_bill": 1.25,       # like contemporary
        "touring_contemporary_company": 1.25,
    },
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

# --- Segment priors by region × category × segment from CSV ---
# CSV: data/productions/segment_priors.csv
# Expected columns (case-insensitive):
#   region, category, segment, weight
#
# Example rows:
#   Province,classic_romance,General Population,1.00
#   Province,classic_romance,Core Classical (F35–64),1.20
#   Calgary,contemporary,Emerging Adults (18–34),1.28
#   ...

SEGMENT_PRIORS: dict[str, dict[str, dict[str, float]]] = {}

SEGMENT_PRIOR_STRENGTH = 1.0  # keep this; used in _prior_weights_for()

def load_segment_priors(path: str = "data/productions/segment_priors.csv") -> None:
    """
    Populates SEGMENT_PRIORS as:
      SEGMENT_PRIORS[region][category][segment] = weight
    """
    global SEGMENT_PRIORS

    try:
        df = pd.read_csv(path)
    except Exception as e:
        st.error(f"Could not load segment_priors CSV at '{path}': {e}")
        SEGMENT_PRIORS = {}
        return

    colmap = {c.lower().strip(): c for c in df.columns}
    r_col = colmap.get("region")
    c_col = colmap.get("category")
    s_col = colmap.get("segment")
    w_col = colmap.get("weight")

    if not (r_col and c_col and s_col and w_col):
        st.error("segment_priors.csv must have columns: 'region', 'category', 'segment', 'weight'")
        SEGMENT_PRIORS = {}
        return

    def _num(x):
        try:
            if pd.isna(x):
                return None
            s = str(x).strip()
            if not s:
                return None
            return float(s)
        except Exception:
            return None

    df[r_col] = df[r_col].astype(str).str.strip()
    df[c_col] = df[c_col].astype(str).str.strip()
    df[s_col] = df[s_col].astype(str).str.strip()
    df[w_col] = df[w_col].map(_num)

    priors: dict[str, dict[str, dict[str, float]]] = {}
    for _, r in df.iterrows():
        region = r[r_col]
        cat    = r[c_col]
        seg    = r[s_col]
        w      = r[w_col]

        if not region or not cat or not seg or w is None:
            continue
        if seg not in SEGMENT_KEYS_IN_ORDER:
            # Ignore unknown segment labels so typos don't break anything
            continue

        priors.setdefault(region, {}).setdefault(cat, {})[seg] = float(w)

    SEGMENT_PRIORS = priors

# Load priors once at startup
load_segment_priors()

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

# --- Post-COVID adjustment (hard-coded from audience research) ---
POSTCOVID_FACTOR = 0.85  # 15% haircut vs pre-COVID baseline

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

# --- Ticket priors (from CSV) ---
# CSV: data/productions/history_city_sales.csv
# Expected columns (case-insensitive):
#   show_title (or title), total single tickets (or tickets)
# One row per run: multiple rows per title are allowed; we keep a list per title.

TICKET_PRIORS_RAW: dict[str, list[float]] = {}

def _median(xs):
    xs = sorted([float(x) for x in xs if x is not None])
    if not xs:
        return None
    n = len(xs); mid = n // 2
    return xs[mid] if n % 2 else (xs[mid-1] + xs[mid]) / 2.0

def load_ticket_priors(path: str = "data/productions/history_city_sales.csv") -> None:
    global TICKET_PRIORS_RAW
    try:
        df = pd.read_csv(path, thousands=",")
    except Exception as e:
        st.error(f"Could not load ticket priors CSV at '{path}': {e}")
        TICKET_PRIORS_RAW = {}
        return

    # normalize columns
    colmap = {c.lower().strip().replace(" ", "_").replace("-", "_"): c for c in df.columns}
    # Support both 'show_title' (history_city_sales.csv) and 'title' (legacy)
    title_col = colmap.get("show_title") or colmap.get("title")
    # Support 'total_single_tickets' (history_city_sales.csv) and 'tickets' (legacy)
    tix_col = colmap.get("total_single_tickets") or colmap.get("tickets") or colmap.get("ticket_median")

    if not title_col or not tix_col:
        st.error("CSV must have title column ('show_title' or 'title') and tickets column ('Total Single Tickets', 'tickets', or 'ticket_median')")
        TICKET_PRIORS_RAW = {}
        return

    df[title_col] = df[title_col].astype(str).str.strip()

    # Convert ticket column to numeric (handles NaN and invalid values)
    df[tix_col] = pd.to_numeric(df[tix_col], errors="coerce")

    priors: dict[str, list[float]] = {}
    for title, g in df.groupby(title_col):
        vals = [v for v in g[tix_col].tolist() if pd.notna(v) and v > 0]
        if not vals:
            continue
        priors[str(title)] = vals

    TICKET_PRIORS_RAW = priors

# Load priors now so everything below can use them
load_ticket_priors()

# --- Past runs (for seasonality) from CSV ---
# CSV: data/productions/past_runs.csv
# Expected columns (case-insensitive):
#   title, start_date, end_date
# Dates in ISO format: YYYY-MM-DD

RUNS_DF: pd.DataFrame = pd.DataFrame()
TITLE_TO_MIDDATE: dict[str, date] = {}

def _to_date(s: str) -> date:
    return datetime.strptime(str(s), "%Y-%m-%d").date()

def _mid_date(a: date, b: date) -> date:
    return a + (b - a) // 2

def load_past_runs(path: str = "data/productions/past_runs.csv") -> None:
    """
    Build RUNS_DF with columns:
      Title, Category, Start, End, MidDate, Month, Year, TicketMedian
    and TITLE_TO_MIDDATE {title -> MidDate}.
    """
    global RUNS_DF, TITLE_TO_MIDDATE

    try:
        df = pd.read_csv(path)
    except Exception as e:
        st.error(f"Could not load past runs CSV at '{path}': {e}")
        RUNS_DF = pd.DataFrame()
        TITLE_TO_MIDDATE = {}
        return

    colmap = {c.lower().strip(): c for c in df.columns}
    t_col = colmap.get("title")
    s_col = colmap.get("start_date")
    e_col = colmap.get("end_date")

    if not t_col or not s_col or not e_col:
        st.error("past_runs.csv must have columns: 'title', 'start_date', 'end_date'")
        RUNS_DF = pd.DataFrame()
        TITLE_TO_MIDDATE = {}
        return

    df[t_col] = df[t_col].astype(str).str.strip()

    rows = []
    aliases = {"Handmaid’s Tale": "Handmaid's Tale"}

    for _, r in df.iterrows():
        title = str(r[t_col]).strip()
        if not title:
            continue

        try:
            d1 = _to_date(r[s_col])
            d2 = _to_date(r[e_col])
        except Exception:
            continue

        mid = _mid_date(d1, d2)

        # Category from BASELINES or heuristic
        if title in BASELINES:
            cat = BASELINES[title].get("category", infer_gender_and_category(title)[1])
        else:
            cat = infer_gender_and_category(title)[1]

        key = aliases.get(title, title)
        tix_med = _median(TICKET_PRIORS_RAW.get(key, []))

        rows.append({
            "Title": title,
            "Category": cat,
            "Start": d1,
            "End": d2,
            "MidDate": mid,
            "Month": mid.month,
            "Year": mid.year,
            "TicketMedian": tix_med,
        })

    RUNS_DF = pd.DataFrame(rows)
    TITLE_TO_MIDDATE = {}
    if not RUNS_DF.empty:
        for _, r in RUNS_DF.iterrows():
            TITLE_TO_MIDDATE[str(r["Title"]).strip()] = r["MidDate"]

# Load runs before building seasonality
load_past_runs()

# ========= Robust Seasonality (Category × Month) =========
# Replaces the block that builds RUNS_DF → SEASONALITY_DF / SEASONALITY_TABLE

# Tunables (safer defaults for sparse data)
K_SHRINK = 3.0          # stronger pull toward 1.00 when samples are small
MINF, MAXF = 0.90, 1.15 # tighter caps than 0.85/1.25
N_MIN = 3               # require at least 3 runs in a (category, month) before trusting a month-specific signal
WINTER_POOL = {12, 1, 2}
TICKET_BLEND_WEIGHT = 0.50  # weight for blending signals vs ticket history in composite index

load_config()

def _robust_category_frame(runs_df: pd.DataFrame, category: str) -> pd.DataFrame:
    g = runs_df[runs_df["Category"] == category].copy()
    g = g[g["TicketMedian"].notna()]
    if g.empty:
        return g

    # 1) Global-outlier removal within the category (IQR fence)
    vals = g["TicketMedian"].astype(float).values
    q1, q3 = np.percentile(vals, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    g = g[(g["TicketMedian"] >= lo) & (g["TicketMedian"] <= hi)].copy()
    return g

def _trimmed_median(x: list[float]) -> float:
    x = sorted([float(v) for v in x if v is not None])
    n = len(x)
    if n >= 4:
        x = x[1:-1]  # drop one low, one high
    if not x:
        return float("nan")
    mid = len(x) // 2
    return x[mid] if len(x) % 2 else (x[mid-1] + x[mid]) / 2.0

_season_rows = []

if not RUNS_DF.empty:
    # Build robust per-category frames
    cats = sorted(RUNS_DF["Category"].dropna().unique().tolist())
    for cat in cats:
        g_cat = _robust_category_frame(RUNS_DF, cat)
        if g_cat.empty:
            continue

        # Robust overall level for the category
        cat_overall_med = _trimmed_median(g_cat["TicketMedian"].tolist())
        if not np.isfinite(cat_overall_med) or cat_overall_med <= 0:
            continue

        # Precompute a pooled "winter" median for sparse months
        g_winter = g_cat[g_cat["Month"].isin(WINTER_POOL)]
        winter_med = _trimmed_median(g_winter["TicketMedian"].tolist()) if not g_winter.empty else np.nan
        winter_has_signal = np.isfinite(winter_med)

        # Compute month factors
        for m in range(1, 13):
            g_m = g_cat[g_cat["Month"] == m]
            n = int(len(g_m))
        
            if n >= 1:
                # use whatever data we have, but shrink hard when n is small
                m_med = _trimmed_median(g_m["TicketMedian"].tolist())
            elif m in WINTER_POOL and winter_has_signal:
                m_med = winter_med
            else:
                m_med = cat_overall_med
        
            if not np.isfinite(m_med) or m_med <= 0:
                continue
        
            raw = float(m_med / cat_overall_med)
            w = n / (n + K_SHRINK)  # with K_SHRINK = 5 this shrinks 1–2-run months a lot
            shrunk = 1.0 + w * (raw - 1.0)
            factor_final = float(np.clip(shrunk, MINF, MAXF))
        
            _season_rows.append({"Category": cat, "Month": m, "Factor": factor_final, "n": n})


            # Guard
            if not np.isfinite(m_med) or m_med <= 0:
                continue

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

# Fixed mode: AB-wide (Province) + General Population
SEGMENT_DEFAULT = "General Population"
REGION_DEFAULT = "Province"
segment = SEGMENT_DEFAULT
region  = REGION_DEFAULT

st.caption("Mode: **Alberta-wide** (Calgary/Edmonton split learned & applied later) • Audience: **General Population**")

# Post-COVID demand adjustment (global haircut, fixed)
postcovid_factor = POSTCOVID_FACTOR
st.caption(
    f"Post-COVID adjustment is hard-coded at ×{postcovid_factor:.2f} "
    "(e.g. 0.85 = 15% haircut vs pre-COVID baseline, based on audience research)."
)

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

default_list = list(BASELINES.keys())
st.markdown("**Titles to score** (one per line). Add NEW titles freely:")
titles_input = st.text_area("Enter titles", value="\n".join(default_list), height=220)
titles = [t.strip() for t in titles_input.splitlines() if t.strip()]
if not titles:
    titles = list(BASELINES.keys())

benchmark_options = list(BASELINES.keys())
default_benchmark_index = (
    benchmark_options.index("Cinderella")
    if "Cinderella" in benchmark_options
    else 0
)

benchmark_title = st.selectbox(
    "Choose Benchmark Title for Normalization",
    options=benchmark_options,
    index=default_benchmark_index,
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
        # new categories → reuse nearest neighbour
        "adult_literary_drama":      {"wiki": +3, "trends": +3},              # like dramatic-ish
        "contemporary_mixed_bill":   {"youtube": +4, "trends": +2},           # like contemporary-ish
        "touring_contemporary_company": {"youtube": +5, "trends": +3, "spotify": +2},  # slightly more pop-facing
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

def _train_ml_models(df_known_in: pd.DataFrame):
    """
    Train regression models (XGBoost/GradientBoosting for overall, Ridge for categories) 
    to predict TicketIndex_DeSeason from SignalOnly.
    Returns models and their performance metrics.
    """
    if not ML_AVAILABLE:
        return None, {}, {}, {}
    
    import warnings
    warnings.filterwarnings('ignore')
    
    overall_model = None
    overall_metrics = {}
    cat_models = {}
    cat_metrics = {}
    
    # Train overall model if enough data
    if len(df_known_in) >= 5:
        try:
            # Prepare training data
            X = df_known_in[['SignalOnly']].values
            y = df_known_in['TicketIndex_DeSeason'].values
            
            # Try XGBoost with cross-validation for robust performance
            if len(df_known_in) >= 8:
                # Use XGBoost for larger datasets
                model = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=1
                )
            else:
                # Use Gradient Boosting for smaller datasets
                model = GradientBoostingRegressor(
                    n_estimators=50,
                    max_depth=2,
                    learning_rate=0.1,
                    random_state=42
                )
            
            # Train the model
            model.fit(X, y)
            overall_model = model
            
            # Calculate cross-validated metrics
            cv_scores = cross_val_score(model, X, y, cv=min(3, len(df_known_in)), 
                                       scoring='neg_mean_absolute_error', n_jobs=1)
            mae_cv = -cv_scores.mean()
            
            # Calculate metrics on full dataset
            y_pred = model.predict(X)
            mae = mean_absolute_error(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            r2 = r2_score(y, y_pred)
            
            overall_metrics = {
                'MAE': float(mae),
                'MAE_CV': float(mae_cv),
                'RMSE': float(rmse),
                'R2': float(r2),
                'n_samples': len(df_known_in)
            }
        except Exception as e:
            # Silently fall back to None if training fails
            overall_model = None
    
    # Train per-category models if enough data per category
    for cat, g in df_known_in.groupby("Category"):
        if len(g) >= 3:
            try:
                X_cat = g[['SignalOnly']].values
                y_cat = g['TicketIndex_DeSeason'].values
                
                # Use Ridge regression for smaller per-category datasets
                if len(g) >= 5:
                    model = Ridge(alpha=1.0, random_state=42)
                else:
                    # Simple linear regression for very small datasets
                    model = LinearRegression()
                
                model.fit(X_cat, y_cat)
                cat_models[cat] = model
                
                # Calculate metrics
                y_pred = model.predict(X_cat)
                mae = mean_absolute_error(y_cat, y_pred)
                r2 = r2_score(y_cat, y_pred)
                
                cat_metrics[cat] = {
                    'MAE': float(mae),
                    'R2': float(r2),
                    'n_samples': len(g)
                }
            except Exception:
                # Silently skip categories with issues
                continue
    
    return overall_model, cat_models, overall_metrics, cat_metrics

def _predict_with_ml_model(model, signal_only: float) -> float:
    """Helper to make prediction with a scikit-learn or XGBoost model."""
    if model is None:
        return np.nan
    try:
        X = np.array([[signal_only]])
        pred = model.predict(X)
        return float(pred[0])
    except Exception:
        return np.nan

def _fit_overall_and_by_category(df_known_in: pd.DataFrame):
    """
    Fit regression models (XGBoost/GradientBoosting if ML available, otherwise simple linear).
    Returns model objects or coefficients depending on availability.
    """
    if ML_AVAILABLE and len(df_known_in) >= 3:
        # Use advanced ML models (XGBoost, GradientBoosting, Ridge)
        overall_model, cat_models, overall_metrics, cat_metrics = _train_ml_models(df_known_in)
        return ('ml', overall_model, cat_models, overall_metrics, cat_metrics)
    else:
        # Fallback to simple linear regression
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
        return ('linear', overall, cat_coefs, {}, {})

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
    postcovid_factor: float = 1.0,
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

    # Attach show type + production expense
    df["ShowType"] = df.apply(
        lambda r: infer_show_type(r["Title"], r["Category"]),
        axis=1,
    )

    def _prod_exp_for_row(r):
        t = str(r["Title"]).strip()
        stype = r["ShowType"]

        # 1) In-app overrides
        budget_by_title    = st.session_state.get("budget_prod_expense_title", {})
        budget_by_showtype = st.session_state.get("budget_prod_expense_showtype", {})

        if t in budget_by_title:
            return budget_by_title[t]
        if stype in budget_by_showtype:
            return budget_by_showtype[stype]

        # 2) History-based medians
        if t in PROD_EXPENSE_TITLE:
            return PROD_EXPENSE_TITLE[t]
        if stype in PROD_EXPENSE_SHOWTYPE:
            return PROD_EXPENSE_SHOWTYPE[stype]

        return np.nan

    df["Prod_Expense"] = df.apply(_prod_exp_for_row, axis=1)

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

    # 4) Fit regression models (XGBoost/GradientBoosting or simple linear)
    df_known = df[pd.notna(df["TicketIndex_DeSeason"])].copy()

    model_result = _fit_overall_and_by_category(df_known)
    model_type = model_result[0]
    
    if model_type == 'ml':
        _, overall_model, cat_models, overall_metrics, cat_metrics = model_result
        
        # Display model performance metrics
        if overall_metrics:
            model_name = "XGBoost" if overall_metrics.get('n_samples', 0) >= 8 else "GradientBoosting"
            st.caption(
                f"🤖 **{model_name} Model Performance** — "
                f"R²: {overall_metrics.get('R2', 0):.3f} | "
                f"MAE: {overall_metrics.get('MAE', 0):.1f} | "
                f"CV-MAE: {overall_metrics.get('MAE_CV', 0):.1f} | "
                f"RMSE: {overall_metrics.get('RMSE', 0):.1f} | "
                f"Samples: {overall_metrics.get('n_samples', 0)}"
            )
        if cat_metrics:
            cat_summary = ", ".join([
                f"{cat}: R²={m.get('R2', 0):.2f} (n={m.get('n_samples', 0)})" 
                for cat, m in cat_metrics.items()
            ])
            st.caption(f"📊 **Category Models (Ridge/Linear)** — {cat_summary}")
        
        # 5) Impute missing TicketIndex with ML models
        def _predict_ticket_index_deseason(signal_only: float, category: str) -> tuple[float, str]:
            # Try category model first
            if category in cat_models:
                pred = _predict_with_ml_model(cat_models[category], signal_only)
                if not np.isnan(pred):
                    pred = float(np.clip(pred, 20.0, 180.0))
                    return pred, "ML Category"
            
            # Try overall model
            if overall_model is not None:
                pred = _predict_with_ml_model(overall_model, signal_only)
                if not np.isnan(pred):
                    pred = float(np.clip(pred, 20.0, 180.0))
                    return pred, "ML Overall"
            
            return np.nan, "Not enough data"
    else:
        # Linear regression fallback
        _, overall_coef, cat_coefs, _, _ = model_result
        
        # 5) Impute missing TicketIndex with linear models
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

    # 11) Remount decay + post-COVID haircut
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
        est_after_remount = est_base * factor
        est_final = round(est_after_remount * POSTCOVID_FACTOR)

        decay_pcts.append(decay_pct)
        decay_factors.append(factor)
        est_after_decay.append(est_final)

    df["ReturnDecayPct"] = decay_pcts
    df["ReturnDecayFactor"] = decay_factors
    df["EstimatedTickets_Final"] = est_after_decay

    # 12) City split (learned title/category → fallback)
    cal_share, edm_share = [], []
    cal_total, edm_total = [], []
    cal_singles, edm_singles = [], []
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
        # All tickets are single tickets (no subscription split)
        cal_singles.append(int(round(cal_t)))
        edm_singles.append(int(round(edm_t)))
    df["CityShare_Calgary"] = cal_share
    df["CityShare_Edmonton"] = edm_share
    df["YYC_Total"] = cal_total
    df["YEG_Total"] = edm_total
    df["YYC_Singles"] = cal_singles
    df["YEG_Singles"] = edm_singles

    # 13) Revenue & marketing (title-level, for table view)
    yyc_single_rev = []
    yeg_single_rev = []
    yyc_rev = []
    yeg_rev = []
    total_rev = []
    yyc_spt = []
    yeg_spt = []
    yyc_mkt = []
    yeg_mkt = []
    total_mkt = []

    for _, r in df.iterrows():
        title = str(r.get("Title", ""))
        cat   = str(r.get("Category", ""))

        yyc_sing = float(r.get("YYC_Singles", 0.0) or 0.0)
        yeg_sing = float(r.get("YEG_Singles", 0.0) or 0.0)

        # Revenue by city (single tickets only)
        ysr   = yyc_sing * YYC_SINGLE_AVG
        esr   = yeg_sing * YEG_SINGLE_AVG

        yyc_single_rev.append(ysr)
        yeg_single_rev.append(esr)

        yyc_tot_r = ysr
        yeg_tot_r = esr
        tot_r     = yyc_tot_r + yeg_tot_r

        yyc_rev.append(yyc_tot_r)
        yeg_rev.append(yeg_tot_r)
        total_rev.append(tot_r)

        # Marketing $/ticket and spend
        spt_yyc = marketing_spt_for(title, cat, "Calgary")
        spt_yeg = marketing_spt_for(title, cat, "Edmonton")
        yyc_spt.append(spt_yyc)
        yeg_spt.append(spt_yeg)

        # Marketing spend is benchmarked on singles only
        yyc_m = yyc_sing * spt_yyc
        yeg_m = yeg_sing * spt_yeg
        yyc_mkt.append(yyc_m)
        yeg_mkt.append(yeg_m)
        total_mkt.append(yyc_m + yeg_m)

    df["YYC_Single_Revenue"] = yyc_single_rev
    df["YEG_Single_Revenue"] = yeg_single_rev
    df["YYC_Revenue"]        = yyc_rev
    df["YEG_Revenue"]        = yeg_rev
    df["Total_Revenue"]      = total_rev

    df["YYC_Mkt_SPT"]     = yyc_spt
    df["YEG_Mkt_SPT"]     = yeg_spt
    df["YYC_Mkt_Spend"]   = yyc_mkt
    df["YEG_Mkt_Spend"]   = yeg_mkt
    df["Total_Mkt_Spend"] = total_mkt

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
        "postcovid_factor": POSTCOVID_FACTOR,
    }

# -------------------------
# Render
# -------------------------
def render_results():
    R = st.session_state.get("results")
    if not R or "df" not in R or R["df"] is None:
        return

    df = R["df"]
    postcovid_factor = float(R.get("postcovid_factor", 1.0))

    import calendar

    if df is None or df.empty:
        st.warning("No scored rows to display yet.")
        return

    benchmark_title = R.get("benchmark", "")
    segment = R.get("segment", "")
    region = R.get("region", "")

    # Notifications about unknown titles
    if R.get("unknown_est"):
        st.info("Estimated (offline) for new titles: " + ", ".join(R["unknown_est"]))
    if R.get("unknown_live"):
        st.success("Used LIVE data for new titles: " + ", ".join(R["unknown_live"]))

    # Ticket index source mix
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

    # Seasonality status
    if "SeasonalityApplied" in df.columns and bool(df["SeasonalityApplied"].iloc[0]):
        run_month_num = df["SeasonalityMonthUsed"].iloc[0]
        try:
            run_month_name = calendar.month_name[int(run_month_num)] if pd.notna(run_month_num) else "n/a"
        except Exception:
            run_month_name = "n/a"
        st.caption(f"Seasonality: **ON** · Run month: **{run_month_name}**")
    else:
        st.caption("Seasonality: **OFF**")

    # Add letter score if missing
    def _assign_score(v: float) -> str:
        if v >= 90: return "A"
        elif v >= 75: return "B"
        elif v >= 60: return "C"
        elif v >= 45: return "D"
        else: return "E"

    if "Score" not in df.columns and "Composite" in df.columns:
        df["Score"] = df["Composite"].apply(_assign_score)

    # Friendly columns for the table
    df_show = df.rename(columns={
        "TicketMedian": "TicketHistory",
        "EffectiveTicketIndex": "TicketIndex used",
    }).copy()

    def _to_month_name(v):
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

    table_cols = [
        "Title","Gender","Category","ShowType","Prod_Expense",
        "PredictedPrimarySegment","PredictedSecondarySegment",

        "WikiIdx","TrendsIdx","YouTubeIdx","SpotifyIdx",
        "Familiarity","Motivation",
        "TicketHistory",
        "TicketIndex used","TicketIndexSource",
        "RunMonth","FutureSeasonalityFactor","HistSeasonalityFactor",
        "Composite","Score",
        "EstimatedTickets",
        "ReturnDecayFactor","ReturnDecayPct","EstimatedTickets_Final",
        "YYC_Singles","YEG_Singles",
        "CityShare_Calgary","CityShare_Edmonton",
        "YYC_Single_Revenue",
        "YEG_Single_Revenue",
        "YYC_Revenue","YEG_Revenue","Total_Revenue",
        "YYC_Mkt_SPT","YEG_Mkt_SPT",
        "YYC_Mkt_Spend","YEG_Mkt_Spend","Total_Mkt_Spend",
    ]
    present_cols = [c for c in table_cols if c in df_show.columns]

    # === Scores table ===
    st.subheader("🎟️ Estimated ticket sales (table view)")
    st.dataframe(
        df_show[present_cols]
        .sort_values(
            by=[
                "EstimatedTickets_Final" if "EstimatedTickets_Final" in df_show.columns else
                ("EstimatedTickets" if "EstimatedTickets" in df_show.columns else "Composite"),
                "Composite", "Motivation", "Familiarity",
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
            "TicketHistory": "{:,.0f}",
            "EstimatedTickets": "{:,.0f}",
            "EstimatedTickets_Final": "{:,.0f}",
            "FutureSeasonalityFactor": "{:.3f}",
            "HistSeasonalityFactor": "{:.3f}",
            "ReturnDecayPct": "{:.0%}",
            "ReturnDecayFactor": "{:.2f}",
            "YYC_Singles": "{:,.0f}",
            "YEG_Singles": "{:,.0f}",
            "CityShare_Calgary": "{:.0%}",
            "CityShare_Edmonton": "{:.0%}",
            "YYC_Single_Revenue": "${:,.0f}",
            "YEG_Single_Revenue": "${:,.0f}",
            "YYC_Revenue": "${:,.0f}",
            "YEG_Revenue": "${:,.0f}",
            "Total_Revenue": "${:,.0f}",
            "YYC_Mkt_SPT": "${:.2f}",
            "YEG_Mkt_SPT": "${:.2f}",
            "YYC_Mkt_Spend": "${:,.0f}",
            "YEG_Mkt_Spend": "${:,.0f}",
            "Total_Mkt_Spend": "${:,.0f}",
            "Prod_Expense": "${:,.0f}",
        }),
        width='stretch',
        hide_index=True
    )

    # === 📅 Build a Season (assign titles to months) ===
    st.subheader("📅 Build a Season (assign titles to months)")
    default_year = (datetime.utcnow().year + 1)
    season_year = st.number_input(
        "Season year (start of season)",
        min_value=2000, max_value=2100,
        value=default_year, step=1
    )

    # Infer benchmark tickets for conversion to attendance
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

    allowed_months = [
        ("September", 9), ("October", 10),
        ("January", 1), ("February", 2), ("March", 3), ("May", 5)
    ]
    title_options = ["— None —"] + sorted(df["Title"].unique().tolist())
    month_to_choice = {}
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
        return start_year if month_num in (9, 10, 12) else (start_year + 1)

    # --- Build a Season: collect rows with full detail ---
    plan_rows = []
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

        # seasonality + index for the chosen month
        f_season = float(seasonality_factor(cat, run_date))
        idx_deseason = float(r.get("TicketIndex_DeSeason_Used", np.nan))
        if not np.isfinite(idx_deseason):
            idx_deseason = float(r.get("SignalOnly", 100.0))
        eff_idx = idx_deseason * f_season

        # convert to tickets if we inferred a benchmark
        if bench_med_deseason_est is not None and np.isfinite(bench_med_deseason_est):
            est_tix = round((eff_idx / 100.0) * bench_med_deseason_est)
        else:
            est_tix = np.nan

        # remount decay + post-COVID haircut
        decay_factor = remount_novelty_factor(title_sel, run_date)
        est_tix_raw = (est_tix if np.isfinite(est_tix) else 0) * decay_factor
        est_tix_final = int(round(est_tix_raw * POSTCOVID_FACTOR))

        # --- City split (recompute for season-picked month) ---
        split = city_split_for(title_sel, cat)  # {"Calgary": p, "Edmonton": 1-p}
        c_sh = float(split.get("Calgary", 0.60))
        e_sh = float(split.get("Edmonton", 0.40))
        s = (c_sh + e_sh) or 1.0
        c_sh, e_sh = float(c_sh / s), float(e_sh / s)

        # Allocate totals to cities
        yyc_total = est_tix_final * c_sh
        yeg_total = est_tix_final * e_sh

        # All tickets are single tickets (no subscription split)
        yyc_singles = int(round(yyc_total))
        yeg_singles = int(round(yeg_total))

        # --- Recommended marketing spend (per city, based on $/ticket) ---
        spt_yyc = marketing_spt_for(title_sel, cat, "Calgary")
        spt_yeg = marketing_spt_for(title_sel, cat, "Edmonton")
        # Marketing spend is benchmarked on singles only
        yyc_mkt = float(yyc_singles or 0) * float(spt_yyc or 0)
        yeg_mkt = float(yeg_singles or 0) * float(spt_yeg or 0)
        total_mkt = float(yyc_mkt) + float(yeg_mkt)  

        # --- Revenue estimates by city (single tickets only) ---
        yyc_single_rev = yyc_singles * YYC_SINGLE_AVG
        yeg_single_rev = yeg_singles * YEG_SINGLE_AVG

        yyc_revenue = yyc_single_rev
        yeg_revenue = yeg_single_rev
        total_revenue = yyc_revenue + yeg_revenue

        # Show type + production expense for budgeting
        show_type = infer_show_type(title_sel, cat)

        # Same priority order as in _prod_exp_for_row
        budget_by_title    = st.session_state.get("budget_prod_expense_title", {})
        budget_by_showtype = st.session_state.get("budget_prod_expense_showtype", {})

        if title_sel in budget_by_title:
            prod_expense = float(budget_by_title[title_sel])
        elif show_type in budget_by_showtype:
            prod_expense = float(budget_by_showtype[show_type])
        elif title_sel in PROD_EXPENSE_TITLE:
            prod_expense = float(PROD_EXPENSE_TITLE[title_sel])
        elif show_type in PROD_EXPENSE_SHOWTYPE:
            prod_expense = float(PROD_EXPENSE_SHOWTYPE[show_type])
        else:
            prod_expense = float("nan")  # or 0.0 if you prefer

        # Net contribution after production + marketing
        net_contribution = float(total_revenue) - prod_expense - float(total_mkt)

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
            "TicketIndex used": eff_idx,
            "TicketIndexSource": r.get("TicketIndexSource", ""),
            "FutureSeasonalityFactor": f_season,
            "HistSeasonalityFactor": r.get("HistSeasonalityFactor", np.nan),
            "Composite": r.get("Composite", np.nan),
            "Score": r.get("Score", ""),
            "EstimatedTickets": (None if not np.isfinite(est_tix) else int(est_tix)),
            "ReturnDecayFactor": float(decay_factor),
            "ReturnDecayPct": float(1.0 - decay_factor),
            "EstimatedTickets_Final": int(est_tix_final),
            "YYC_Singles": int(yyc_singles),
            "YEG_Singles": int(yeg_singles),
            "CityShare_Calgary": float(c_sh),
            "CityShare_Edmonton": float(e_sh),

            # Revenue by city (single tickets only)
            "YYC_Single_Revenue": float(yyc_single_rev),
            "YEG_Single_Revenue": float(yeg_single_rev),

            # Totals
            "YYC_Revenue": float(yyc_revenue),
            "YEG_Revenue": float(yeg_revenue),
            "Total_Revenue": float(total_revenue),

            # Production expense + marketing + net
            "Prod_Expense": float(prod_expense),
            "YYC_Mkt_SPT":  float(spt_yyc),
            "YEG_Mkt_SPT":  float(spt_yeg),
            "YYC_Mkt_Spend": float(yyc_mkt),
            "YEG_Mkt_Spend": float(yeg_mkt),
            "Total_Mkt_Spend": float(total_mkt),
            "Net_Contribution": float(net_contribution),
        })

    # Guard + render
    if not plan_rows:
        st.caption("Pick at least one month/title above to see your season projection, charts, and scatter.")
        return

    # Keep full plan_df (do NOT trim away WikiIdx/Trends/etc.)
    plan_df = pd.DataFrame(plan_rows)

    # A view with a preferred column order for some displays
    desired_order = [
        "Month",
        "Title",
        "Category",
        "PrimarySegment",
        "SecondarySegment",
        "TicketIndex used",
        "FutureSeasonalityFactor",
        "ReturnDecayPct",
        "EstimatedTickets_Final",
        "YYC_Singles",
        "YEG_Singles",
        "CityShare_Calgary",
        "CityShare_Edmonton",
        "YYC_Single_Revenue",
        "YEG_Single_Revenue",
        "YYC_Revenue",
        "YEG_Revenue",
        "Total_Revenue",
        "YYC_Mkt_SPT",
        "YEG_Mkt_SPT",
        "YYC_Mkt_Spend",
        "YEG_Mkt_Spend",
        "Total_Mkt_Spend",
        "Prod_Expense",
        "Net_Contribution",
    ]
    present_plan_cols = [c for c in desired_order if c in plan_df.columns]
    plan_view = plan_df[present_plan_cols].copy()

    # --- Executive summary KPIs ---
    with st.container():
        st.markdown("### 📊 Season at a glance")
        c1, c2, c3, c4, c5, c6 = st.columns(6)

        yyc_tot = int(plan_df["YYC_Singles"].sum())
        yeg_tot = int(plan_df["YEG_Singles"].sum())
        singles_tot = int(plan_df["YYC_Singles"].sum() + plan_df["YEG_Singles"].sum())
        grand = int(plan_df["EstimatedTickets_Final"].sum()) or 1

        total_rev  = float(plan_df["Total_Revenue"].sum())
        yyc_rev    = float(plan_df["YYC_Revenue"].sum())
        yeg_rev    = float(plan_df["YEG_Revenue"].sum())
        total_mkt  = float(plan_df["Total_Mkt_Spend"].sum())
        total_single_tix = max(singles_tot, 1)
        total_prod = float(plan_df["Prod_Expense"].sum())
        net_season = float(plan_df["Net_Contribution"].sum())

        with c1:
            st.metric("Projected Season Tickets", f"{grand:,}")
        with c2:
            st.metric("Calgary • share", f"{yyc_tot:,}", delta=f"{yyc_tot/grand:.1%}")
        with c3:
            st.metric("Edmonton • share", f"{yeg_tot:,}", delta=f"{yeg_tot/grand:.1%}")
        with c4:
            st.metric(
                "Projected Marketing Spend",
                f"${total_mkt:,.0f}",
                delta=f"${(total_mkt / total_single_tix):.0f} per single"
            )
        with c5:
            st.metric("Season Production Expense", f"${total_prod:,.0f}")
        with c6:
            st.metric("Net Contribution (after prod + mkt)", f"${net_season:,.0f}")

        st.caption(
            f"Post-COVID adjustment applied: ×{postcovid_factor:.2f} "
            f"(e.g. 0.85 = 15% haircut vs pre-COVID baseline)."
        )

    # --- Tabs: Season table (wide) | City split | Rank | Scatter ---
    tab_table, tab_city, tab_rank, tab_scatter = st.tabs(
        ["Season table", "City Split by Month", "Rank by Composite", "Season Scatter"]
    )

    with tab_table:
        from pandas import IndexSlice as _S

        # UI month order to display
        month_name_order = ["September","October","January","February","March","May"]

        def _month_label(month_full: str) -> str:
            # "September 2026" -> "Sep-26"
            try:
                name, year = str(month_full).split()
                m_num = list(calendar.month_name).index(name)
                return f"{calendar.month_abbr[m_num]}-{str(int(year))[-2:]}"
            except Exception:
                return str(month_full)

        picked = (
            plan_df.copy()
            .assign(_mname=lambda d: d["Month"].str.split().str[0])
            .pipe(lambda d: d[d["_mname"].isin(month_name_order)])
        )
        picked["_order"] = picked["_mname"].apply(lambda n: month_name_order.index(n))
        picked = picked.sort_values("_order")

        # map month label -> row (full row with WikiIdx etc)
        month_to_row = { _month_label(r["Month"]): r for _, r in picked.iterrows() }
        month_cols = list(month_to_row.keys())

        metrics = [
            "Title","Category","PrimarySegment","SecondarySegment",
            "WikiIdx","TrendsIdx","YouTubeIdx","SpotifyIdx",
            "Familiarity","Motivation",
            "TicketHistory","TicketIndex used","TicketIndexSource",
            "FutureSeasonalityFactor","HistSeasonalityFactor",
            "Composite","Score",
            "EstimatedTickets","ReturnDecayFactor","ReturnDecayPct","EstimatedTickets_Final",
            "YYC_Singles","YEG_Singles",
            "CityShare_Calgary","CityShare_Edmonton",
            "YYC_Single_Revenue",
            "YEG_Single_Revenue",
            "YYC_Revenue","YEG_Revenue","Total_Revenue",
            "YYC_Mkt_SPT","YEG_Mkt_SPT",
            "YYC_Mkt_Spend","YEG_Mkt_Spend","Total_Mkt_Spend",
            "Prod_Expense","Net_Contribution",
        ]

        # assemble wide DF: rows = metrics, columns = month labels
        df_wide = pd.DataFrame(
            { col: [month_to_row[col].get(m, np.nan) for m in metrics] for col in month_cols },
            index=metrics
        )

        sty = df_wide.style

        # Integer counts / $ (tickets & money)
        int_like_rows = [
            "TicketHistory","EstimatedTickets","EstimatedTickets_Final",
            "YYC_Singles","YEG_Singles",
            "YYC_Single_Revenue",
            "YEG_Single_Revenue",
            "YYC_Revenue","YEG_Revenue","Total_Revenue",
            "YYC_Mkt_Spend","YEG_Mkt_Spend","Total_Mkt_Spend",
            "Prod_Expense","Net_Contribution",
        ]
        sty = sty.format("{:,.0f}", subset=_S[int_like_rows, :])

        # Indices / composites (one decimal)
        idx_rows = [
            "WikiIdx","TrendsIdx","YouTubeIdx","SpotifyIdx",
            "Familiarity","Motivation","Composite","TicketIndex used",
        ]
        sty = sty.format("{:.1f}", subset=_S[idx_rows, :])

        # Factors
        sty = sty.format("{:.3f}", subset=_S[["FutureSeasonalityFactor","HistSeasonalityFactor"], :])
        sty = sty.format("{:.2f}", subset=_S[["ReturnDecayFactor","YYC_Mkt_SPT","YEG_Mkt_SPT"], :])

        # Percentages
        sty = sty.format("{:.0%}", subset=_S[["CityShare_Calgary","CityShare_Edmonton","ReturnDecayPct"], :])

        st.markdown("#### 🗓️ Season table")
        st.dataframe(sty, width='stretch')

        # CSV download
        st.download_button(
            "⬇️ Download Season (wide) CSV",
            df_wide.reset_index().rename(columns={"index":"Metric"}).to_csv(index=False).encode("utf-8"),
            file_name=f"season_plan_wide_{season_year}.csv",
            mime="text/csv"
        )

        # Full PDF report download
        try:
            methodology_paragraphs = _methodology_glossary_text()
            pdf_bytes = build_full_pdf_report(
                methodology_paragraphs=methodology_paragraphs,
                plan_df=plan_df,
                season_year=int(season_year),
                org_name="Alberta Ballet"
            )
            st.download_button(
                "⬇️ Download Full PDF Report",
                data=pdf_bytes,
                file_name=f"alberta_ballet_season_report_{season_year}.pdf",
                mime="application/pdf",
                width='content'
            )
        except Exception as e:
            st.warning(f"PDF report unavailable: {e}")

    with tab_city:
        try:
            plot_df = (
                plan_df[["Month","YYC_Singles","YEG_Singles"]]
                .copy()
                .groupby("Month", as_index=False).sum()
            )
            fig, ax = plt.subplots()
            ax.bar(plot_df["Month"], plot_df["YYC_Singles"], label="YYC Singles")
            ax.bar(
                plot_df["Month"], plot_df["YEG_Singles"],
                bottom=plot_df["YYC_Singles"],
                label="YEG Singles"
            )
            ax.set_title("City split by month (stacked)")
            ax.set_xlabel("Month")
            ax.set_ylabel("Tickets (final)")
            ax.legend()
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig)
        except Exception as e:
            st.caption(f"City split chart unavailable: {e}")

    with tab_rank:
        rank_cols = ["Month","Title","Category","PrimarySegment","SecondarySegment","Composite","EstimatedTickets_Final"]
        rank_cols = [c for c in rank_cols if c in plan_df.columns]
        rank_df = plan_df[rank_cols].copy().sort_values(
            ["Composite","EstimatedTickets_Final"],
            ascending=[False, False]
        )
        st.dataframe(
            rank_df.style.format({"Composite":"{:.1f}","EstimatedTickets_Final":"{:,.0f}"}),
            width='stretch', hide_index=True
        )

    with tab_scatter:
        try:
            scat_df = plan_df[
                plan_df["Familiarity"].notna()
                & plan_df["Motivation"].notna()
                & pd.to_numeric(plan_df["EstimatedTickets_Final"], errors="coerce").notna()
            ].copy()

            if scat_df.empty:
                st.caption("Add at least one title to the season to see the scatter.")
            else:
                vals = scat_df["EstimatedTickets_Final"].astype(float).clip(lower=0)
                size = np.sqrt(vals.replace(0, 1.0)) * 3.0  # bubble ~ sqrt(tix)

                fig, ax = plt.subplots()
                ax.scatter(scat_df["Familiarity"], scat_df["Motivation"], s=size)

                for _, rr in scat_df.iterrows():
                    ax.annotate(
                        f"{rr['Month']}: {rr['Title']}",
                        (rr["Familiarity"], rr["Motivation"]),
                        fontsize=8, xytext=(3,3), textcoords="offset points"
                    )

                ax.axvline(scat_df["Familiarity"].median(), linestyle="--")
                ax.axhline(scat_df["Motivation"].median(), linestyle="--")
                ax.set_xlabel("Familiarity (benchmark = 100 index)")
                ax.set_ylabel("Motivation (benchmark = 100 index)")
                ax.set_title("Season scatter — Familiarity vs Motivation (bubble size = EstimatedTickets_Final)")
                st.pyplot(fig)
                st.caption("Bubble size is proportional to **EstimatedTickets_Final** (after remount decay & seasonality).")
        except Exception as e:
            st.caption(f"Season scatter unavailable: {e}")

    # === NEW: Season financial summary table + CSV download ===
    summary_df = build_season_financial_summary_table(plan_df)

    if not summary_df.empty:
        st.subheader("📊 Season Financial Summary (months as columns)")
        st.dataframe(summary_df, width='stretch')

        csv_bytes = summary_df.to_csv(index=True).encode("utf-8")
        st.download_button(
            "Download Season Summary CSV",
            data=csv_bytes,
            file_name=f"season_financial_summary_{season_year}.csv",
            mime="text/csv",
            width='content',
        )

# -------------------------
# Advanced ML Settings (opt-in toggles from config.yaml)
# -------------------------
# These features are loaded from config.yaml and can be toggled here
# Default behavior remains unchanged unless user enables these options
with st.sidebar.expander("⚙️ Advanced ML Settings", expanded=False):
    st.caption("Experimental features from robust forecasting pipeline")
    
    # KNN fallback toggle
    knn_enabled_ui = st.checkbox(
        "Enable k-NN fallback for cold-start",
        value=KNN_CONFIG.get("enabled", True),
        help="Use similarity matching for titles without ticket history"
    )
    
    # Calibration toggle
    calibration_enabled_ui = st.checkbox(
        "Apply prediction calibration",
        value=CALIBRATION_CONFIG.get("enabled", False),
        help="Adjust predictions using calibration parameters (if available)"
    )
    
    # Show model status
    try:
        from ml.predict_utils import is_ml_model_available, is_calibration_available
        if is_ml_model_available():
            st.success("✓ Trained ML model available", icon="🤖")
        else:
            st.info("Train a model with: `python scripts/train_safe_model.py`")
        if calibration_enabled_ui and is_calibration_available():
            st.success("✓ Calibration parameters loaded", icon="📊")
        elif calibration_enabled_ui:
            st.warning("Run: `python scripts/calibrate_predictions.py fit`")
    except ImportError:
        st.caption("ML utilities not loaded")
    
    st.caption(f"Config: knn.k={KNN_CONFIG.get('k', 5)}, calibration.mode={CALIBRATION_CONFIG.get('mode', 'global')}")

# Main app logic
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
        postcovid_factor=postcovid_factor,
    )

if st.session_state.get("results") is not None:
    render_results()
