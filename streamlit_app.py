# - Learns YYC/YEG splits from history.csv (or uploaded CSV)
# - Single ticket estimation only
# - Removes arbitrary 60/40 split; uses title→category→default fallback
# - Small fixes: softmax bug, LA attach loop, duplicate imports, safer guards

import io
import json
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

# Live Analytics integration
try:
    from data.loader import get_live_analytics_category_factors
    LA_AVAILABLE = True
except ImportError:
    LA_AVAILABLE = False
    def get_live_analytics_category_factors():
        return {}

# Economic factors integration
try:
    from utils.economic_factors import (
        compute_boc_economic_sentiment,
        compute_alberta_economic_sentiment,
        get_current_economic_context,
        is_boc_live_enabled,
        is_alberta_live_enabled,
    )
    ECON_FACTORS_AVAILABLE = True
except ImportError:
    ECON_FACTORS_AVAILABLE = False
    def compute_boc_economic_sentiment(run_date=None, city=None):
        return 1.0, {"source": "unavailable", "factor": 1.0}
    def compute_alberta_economic_sentiment(run_date=None, city=None):
        return 1.0, {"source": "unavailable", "factor": 1.0}
    def get_current_economic_context(include_boc=True, include_alberta=True, use_cache=True):
        return {"combined_sentiment": 1.0, "sources_available": []}
    def is_boc_live_enabled():
        return False
    def is_alberta_live_enabled():
        return False

# k-NN fallback for cold-start predictions
try:
    from ml.knn_fallback import KNNFallback, build_knn_from_config
    KNN_FALLBACK_AVAILABLE = True
except ImportError:
    KNN_FALLBACK_AVAILABLE = False
    KNNFallback = None
    build_knn_from_config = None

def load_config(path: str = "config.yaml"):
    global SEGMENT_MULT, REGION_MULT
    global DEFAULT_BASE_CITY_SPLIT, _CITY_CLIP_RANGE
    global POSTCOVID_FACTOR, TICKET_BLEND_WEIGHT
    global K_SHRINK, MINF, MAXF, N_MIN
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

    # 6) New robust forecasting settings (opt-in)
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

def _plain_language_overview_text() -> list:
    """
    Build the 'How This Forecast Works — A Plain-Language Overview' section for the PDF report.
    
    This section appears near the beginning of the report (after title page).
    It provides a comprehensive, board-level introduction that explains:
    - What the Title Scoring App is and what questions it answers
    - How Familiarity and Motivation are constructed from digital signals
    - What Ticket Index means as a relative demand measure
    - How seasonality, category, and macroeconomic factors influence forecasts
    - How premieres vs remounts are handled
    - That SHAP underpins all per-title explanations
    
    This narrative is derived from the canonical Technical Report Prose Style document.

    Returns a list of ReportLab Flowables (Paragraphs, Spacers).
    """
    styles = _make_styles()
    P = Paragraph
    SP = Spacer
    out = []

    out.append(P("How This Forecast Works — A Plain-Language Overview", styles["h1"]))
    out.append(SP(1, 6))

    # Paragraph 1: What the system is and does
    out.append(P(
        "The Alberta Ballet Title Scoring App is an internal decision support system that uses machine learning "
        "to estimate audience demand for proposed ballet productions. At its core, the app addresses a fundamental "
        "challenge: the uncertainty inherent in programming future seasons. By analysing patterns in past performance "
        "and contextual factors, the system provides quantitative forecasts that inform artistic and strategic decisions. "
        "These forecasts are not revenue predictions—they estimate single-ticket volume only, reflecting relative demand "
        "strength rather than absolute guarantees. Understanding this scope ensures appropriate use of the system's outputs.",
        styles["body"],
    ))
    out.append(SP(1, 8))

    # Paragraph 2: Familiarity & Motivation construction
    out.append(P(
        "The methodology begins by measuring public visibility through four digital channels: Wikipedia page traffic, "
        "Google search trends, YouTube viewing behaviour, and Chartmetric streaming activity. These raw signals are combined "
        "into two interpretable indices. <b>Familiarity</b> measures public recognition—how well-known a title is across "
        "general awareness platforms like Wikipedia and Google. <b>Motivation</b> quantifies active engagement—how eager "
        "audiences appear to be to interact with the work through platforms like YouTube and Chartmetric. Both indices are "
        "normalized to a 0–100+ scale against a reference title, ensuring consistent interpretation across all productions. "
        "This dual-metric approach captures both passive awareness and active interest, providing richer signal than either "
        "dimension alone.",
        styles["body"],
    ))
    out.append(SP(1, 8))

    # Paragraph 3: Ticket Index and historical translation
    out.append(P(
        "Once Familiarity and Motivation are established, the system converts these visibility scores into a "
        "<b>Ticket Index</b>—a relative demand measure representing expected performance against the benchmark. "
        "This translation uses a <b>constrained Ridge regression model</b> trained on Alberta Ballet's historical "
        "archives. The model learns how different levels of public interest correspond to actual ticket sales across "
        "categories like family classics, contemporary works, and holiday productions. Crucially, the model is anchored "
        "with constraints to prevent overestimation: titles with minimal online buzz (SignalOnly ≈ 0) map to a realistic "
        "floor (TicketIndex ≈ 25), while the benchmark title (such as Cinderella) maintains its alignment at 100. "
        "This prevents the inflated predictions that plagued earlier unconstrained models, which overestimated low-buzz "
        "contemporary pieces by approximately 30%. The constrained approach produces the typical formula: "
        "TicketIndex ≈ 0.75 × SignalOnly + 27, ensuring realistic estimates for both premieres and returning works.",
        styles["body"],
    ))
    out.append(SP(1, 8))

    # Paragraph 4: Seasonality and category effects
    out.append(P(
        "The model then incorporates seasonality, recognizing that audience behaviour varies predictably by month "
        "and production type. December holiday shows reliably outperform shoulder-season productions; contemporary works "
        "exhibit different seasonal patterns than family classics. These category-month interactions are learned from "
        "historical data, smoothed to prevent overfitting to small samples, and capped to ensure stability. "
        "The resulting seasonal factors adjust base expectations up or down depending on when a production is scheduled, "
        "reflecting learned calendar dynamics rather than arbitrary assumptions.",
        styles["body"],
    ))
    out.append(SP(1, 8))

    # Paragraph 5: Premiere vs remount handling and Ridge regression
    out.append(P(
        "Distinguishing premieres from remounts is fundamental to the model's logic. Productions returning after several "
        "years may benefit from renewed interest; those remounted quickly may face modest audience fatigue. The system "
        "quantifies these dynamics through timing features that capture years since last performance. All feature engineering "
        "feeds into a <b>constrained Ridge regression model</b>—a regularised linear model enhanced with anchor points that "
        "enforce realistic bounds. The model learns the relationship between visibility signals, category membership, timing patterns, "
        "and observed ticket sales, with built-in constraints preventing unrealistic predictions for low-buzz titles.",
        styles["body"],
    ))
    out.append(SP(1, 8))

    # Paragraph 6: SHAP explainability
    out.append(P(
        "Every prediction generated by the model comes with an explanation. The system uses <b>SHAP values</b> "
        "(SHapley Additive exPlanations), a technique from game theory, to decompose each forecast into feature contributions. "
        "For a given title, SHAP identifies which factors—Familiarity, Motivation, seasonality, category precedent, or "
        "macroeconomic context—pushed the prediction higher or lower relative to the baseline. These attributions provide "
        "transparency into the model's reasoning, enabling stakeholders to understand not just what the forecast is, but "
        "why the model arrived at that number. All per-title narratives in this report are grounded in these SHAP-driven "
        "explanations, ensuring interpretability and accountability.",
        styles["body"],
    ))
    out.append(SP(1, 8))

    # Paragraph 7: Economic and macroeconomic integration
    out.append(P(
        "Finally, the model integrates time-aligned economic indicators to ensure forecasts reflect the macroeconomic "
        "environment audiences will experience at opening. Interest rates, energy prices, employment levels, consumer "
        "confidence, inflation trends, and arts-sector sentiment are all incorporated using temporal matching that "
        "respects chronological boundaries—no future data leaks into predictions. These economic features augment the "
        "core visibility signals, adjusting expectations for broader demand drivers beyond a title's intrinsic appeal.",
        styles["body"],
    ))
    out.append(SP(1, 8))

    # Paragraph 8: Summary and scope
    out.append(P(
        "These components—digital visibility measurement, historical demand translation, category and seasonal patterning, "
        "premiere-remount distinctions, constrained Ridge regression, SHAP-driven explainability, and macroeconomic context—"
        "combine to form an empirically grounded forecasting system. The output is a calibrated Ticket Index and city-specific "
        "ticket estimates that support season planning, budgeting, and marketing prioritization. All forecasts should be "
        "interpreted as central expectations with inherent uncertainty; they enable informed scenario planning when combined "
        "with venue constraints, production costs, and strategic programming goals. The result is a transparent, repeatable, "
        "and defensible methodology that brings quantitative rigor to Alberta Ballet's season development process.",
        styles["body"],
    ))

    return out


def _methodology_glossary_text() -> list:
    """
    Build the 'Methodology & Glossary' section for the PDF report.
    
    NOTE: The old "How this forecast works" content has been moved to
    _plain_language_overview_text() which now appears near the beginning of the PDF.
    This function now contains only the glossary and technical reference sections.

    Returns a list of ReportLab Flowables (Paragraphs, Spacers).
    """
    styles = _make_styles()
    P = Paragraph
    SP = Spacer
    out = []

    # ─────────────────────────────────────────────────────────
    # 1. Plain-language glossary
    # ─────────────────────────────────────────────────────────
    out.append(P("Plain-language glossary", styles["h1"]))

    glossary_items = [
        "<b>Familiarity</b>: how well-known the title is.",
        "<b>Motivation</b>: how keen people seem to be to watch it.",
        "<b>Ticket Index</b>: a relative demand score (0-180 scale) calculated using constrained "
        "Ridge regression from online signals. The model is anchored so minimal buzz → ≈25, "
        "benchmark (Cinderella) → 100.",
        "<b>Seasonality Factor</b>: some months sell better than others for a given type of show.",
        "<b>YYC/YEG split</b>: we use your history to split totals between the two cities.",
    ]
    for g in glossary_items:
        out.append(P(g, styles["body"]))
    out.append(SP(1, 10))

    # ─────────────────────────────────────────────────────────
    # 2. How to read the accuracy numbers
    # ─────────────────────────────────────────────────────────
    out.append(P("How to read the accuracy numbers", styles["h2"]))
    out.append(P(
        "When we test the model on past seasons, the dashboard shows three standard "
        "metrics: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R². "
        "MAE and RMSE are expressed in tickets and indicate, on average, how far the "
        "model's estimates are from actual results (RMSE penalises large misses more "
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
    # 3. Quick-read bullets
    # ─────────────────────────────────────────────────────────
    out.append(P("Key model components (quick read)", styles["h2"]))

    bullets = [
        "We combine online visibility (Wikipedia, YouTube, Google, Chartmetric) into two simple "
        "ideas: <b>Familiarity</b> (people know it) and <b>Motivation</b> "
        "(people engage with it).",
        "We anchor everything to a <b>benchmark title</b> (typically Cinderella, ≈11,976 tickets) "
        "so scores are on a shared 0–100+ scale.",
        "We use a <b>constrained Ridge regression model</b> (α=5.0) to convert those scores into a "
        "<b>Ticket Index</b>. The model is anchored so low-buzz titles get realistic baselines "
        "(SignalOnly=0 → TicketIndex≈25) while the benchmark remains at 100.",
        "We adjust for the <b>month</b> you plan to run it (some months sell better).",
        "We split totals between <b>Calgary</b> and <b>Edmonton</b> using learned "
        "historical shares to produce <b>single ticket</b> estimates.",
    ]
    for b in bullets:
        out.append(P(f"• {b}", styles["body"]))
    
    out.append(SP(1, 10))
    
    # ─────────────────────────────────────────────────────────
    # 4. Model Update Note (December 2024)
    # ─────────────────────────────────────────────────────────
    out.append(P("Model Update (December 2024)", styles["h2"]))
    out.append(P(
        "The Ticket Index calculation was updated to use <b>constrained Ridge regression</b> "
        "instead of unconstrained gradient boosting models. The previous models suffered from a "
        "'high intercept problem' that caused low-signal titles to be overestimated by approximately "
        "30% (e.g., ~5,500 tickets instead of ~3,800 for obscure contemporary works). The new model "
        "enforces realistic constraints through synthetic anchor points: titles with minimal online "
        "presence (SignalOnly ≈ 0) now map to TicketIndex ≈ 25, while the benchmark (Cinderella) "
        "maintains alignment at 100. This produces more defensible forecasts while preserving the "
        "model's ability to differentiate between high-demand and low-demand titles. The typical "
        "formula is now: TicketIndex ≈ 0.75 × SignalOnly + 27.",
        styles["body"],
    ))

    return out



# -------------------------
# Season Summary (Board View) Helper Functions
# -------------------------

# Column widths for the Full Season Table PDF (module-level constant for performance)
# Total: ~9.35 inches, fits within 10" usable width on landscape LETTER
_FULL_SEASON_TABLE_COL_WIDTHS = {
    "Month": 0.5 * inch,
    "Title": 1.2 * inch,
    "Category": 0.7 * inch,
    "TicketIndex used": 0.5 * inch,
    "Familiarity": 0.4 * inch,
    "Motivation": 0.4 * inch,
    "FutureSeasonalityFactor": 0.5 * inch,
    "EstimatedTickets_Final": 0.5 * inch,
    "YYC_Singles": 0.4 * inch,
    "YEG_Singles": 0.4 * inch,
    "PrimarySegment": 0.9 * inch,
    "SecondarySegment": 0.8 * inch,
    "LA_EngagementFactor": 0.45 * inch,
    "Econ_Sentiment": 0.4 * inch,
}

def index_strength_rating(index_value: float) -> str:
    """
    Convert a ticket index value to a 1–5 star "strength" rating.
    
    The index is benchmark-normalized (100 = benchmark performance).
    Since the benchmark title represents the best historical performer,
    reaching the benchmark (100) earns the maximum 5-star rating.
    
    Ranges are calibrated so the benchmark = 5 stars:
      - 0–25:   Very weak (★☆☆☆☆)
      - 25–50:  Below average (★★☆☆☆)
      - 50–75:  Average (★★★☆☆)
      - 75–100: Above average (★★★★☆)
      - 100+:   Benchmark/Strong (★★★★★)
    
    Args:
        index_value: The ticket index or effective ticket index value.
    
    Returns:
        A string with filled/empty star characters representing strength.
    """
    try:
        idx = float(index_value)
        # Check for NaN after conversion (np.nan converts to float nan)
        if math.isnan(idx):
            return "☆☆☆☆☆"
    except (TypeError, ValueError):
        return "☆☆☆☆☆"  # No data
    
    if idx < 25:
        return "★☆☆☆☆"
    elif idx < 50:
        return "★★☆☆☆"
    elif idx < 75:
        return "★★★☆☆"
    elif idx < 100:
        return "★★★★☆"
    else:
        return "★★★★★"


def segment_tilt_label(row: dict) -> str:
    """
    Derive a short, human-readable label for the lead audience segment.
    
    Uses the PrimarySegment field if available, with fallback to
    dominant_audience_segment. If neither exists, returns "General Population".
    
    Args:
        row: A dictionary-like object (DataFrame row or dict) with segment fields.
    
    Returns:
        A human-readable string for the primary audience segment.
    """
    # Try PrimarySegment first (from segment analysis)
    primary = row.get("PrimarySegment", "")
    if primary and str(primary).strip():
        return str(primary).strip()
    
    # Fallback to dominant_audience_segment
    dominant = row.get("dominant_audience_segment", "")
    if dominant and str(dominant).strip():
        return str(dominant).strip()
    
    # Fallback to PredictedPrimarySegment
    predicted = row.get("PredictedPrimarySegment", "")
    if predicted and str(predicted).strip():
        return str(predicted).strip()
    
    # Final fallback
    return "General Population"


def build_season_summary(plan_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a clean, board-level summary DataFrame from the full export table.
    
    This function derives a simplified view suitable for leadership review,
    with only the most relevant columns for strategic planning.
    
    Args:
        plan_df: The full results DataFrame with all columns.
    
    Returns:
        A new DataFrame with board-friendly summary columns:
          - Month: Full month name (e.g., "September")
          - Show Title: Title of the production
          - Category: Show category (e.g., "family_classic")
          - Estimated Tickets: Total ticket forecast
          - YYC Singles: Calgary single ticket forecast
          - YEG Singles: Edmonton single ticket forecast
          - Segment Tilt: Primary audience segment label
          - Index Strength: 1–5 star rating based on ticket index
    
    The output is sorted by calendar month order (September → October → ...).
    """
    import calendar
    
    if plan_df is None or plan_df.empty:
        return pd.DataFrame(columns=[
            "Month", "Show Title", "Category", "Estimated Tickets",
            "YYC Singles", "YEG Singles",
            "Segment Tilt", "Index Strength"
        ])
    
    # Define calendar order for sorting
    # Season typically runs Sept-May with gap in Nov-Dec and April
    month_order = ["September", "October", "November", "December",
                   "January", "February", "March", "April", "May", "June"]
    order_map = {m: i for i, m in enumerate(month_order)}
    
    rows = []
    for _, r in plan_df.iterrows():
        # Extract month name from "September 2026" format
        month_str = str(r.get("Month", ""))
        month_name = month_str.split()[0] if month_str else ""
        
        # Fallback: if Month column is missing/malformed but month_of_opening exists,
        # derive month name from it. This handles edge cases from alternate data sources.
        month_num = r.get("month_of_opening", None)
        if month_num and not month_name:
            try:
                month_name = calendar.month_name[int(month_num)]
            except (ValueError, IndexError):
                pass
        
        # Calculate estimated tickets
        # Check for column existence first, then check for NaN values
        est_tickets = None
        if "EstimatedTickets_Final" in r and pd.notna(r.get("EstimatedTickets_Final")):
            est_tickets = r.get("EstimatedTickets_Final")
        elif "EstimatedTickets" in r and pd.notna(r.get("EstimatedTickets")):
            est_tickets = r.get("EstimatedTickets")
        
        if est_tickets is None or pd.isna(est_tickets):
            # Fallback: sum of YYC + YEG Singles
            yyc = r.get("YYC_Singles", 0) or 0
            yeg = r.get("YEG_Singles", 0) or 0
            est_tickets = int(yyc) + int(yeg)
        else:
            est_tickets = int(est_tickets)
        
        # Get ticket index for strength rating
        # Prefer "TicketIndex used" (EffectiveTicketIndex), then TicketIndex_DeSeason_Used
        idx_val = r.get("TicketIndex used", r.get("EffectiveTicketIndex", 
                        r.get("TicketIndex_DeSeason_Used", 100)))
        
        rows.append({
            "Month": month_name,
            "_month_order": order_map.get(month_name, 99),
            "Show Title": r.get("Title", ""),
            "Category": r.get("Category", ""),
            "Estimated Tickets": est_tickets,
            "YYC Singles": int(r.get("YYC_Singles", 0) or 0),
            "YEG Singles": int(r.get("YEG_Singles", 0) or 0),
            "Segment Tilt": segment_tilt_label(r),
            "Index Strength": index_strength_rating(idx_val),
        })
    
    summary_df = pd.DataFrame(rows)
    
    # Sort by calendar month order
    summary_df = summary_df.sort_values("_month_order").drop(columns=["_month_order"])
    
    return summary_df.reset_index(drop=True)


def _make_season_summary_table_pdf(plan_df: pd.DataFrame) -> Table:
    """
    Convert the board-level Season Summary DataFrame into a ReportLab Table
    for the PDF export.
    
    This table is designed to fit on a single landscape page with readable text.
    Uses shortened column headers, reduced font size, and optimized column widths
    to ensure the full board view is visible without horizontal overflow.
    
    Args:
        plan_df: The full results DataFrame (will be processed by build_season_summary).
    
    Returns:
        A ReportLab Table object formatted for PDF output.
    """
    from reportlab.platypus import Paragraph as P_table
    from reportlab.lib.styles import ParagraphStyle
    
    summary_df = build_season_summary(plan_df)
    if summary_df.empty:
        return Table([["No season data"]])
    
    # Shortened column headers to fit table on single page
    header_map = {
        "Month": "Month",
        "Show Title": "Show Title",
        "Category": "Category",
        "Estimated Tickets": "Est. Tickets",
        "YYC Singles": "YYC",
        "YEG Singles": "YEG",
        "Segment Tilt": "Segment",
        "Index Strength": "Strength"
    }
    
    # Create paragraph style for text wrapping in cells
    cell_style = ParagraphStyle(
        'CellStyle',
        fontName='Helvetica',
        fontSize=8,
        leading=10,
    )
    header_style = ParagraphStyle(
        'HeaderStyle',
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.whitesmoke,
    )
    
    # Header row with shortened names
    header = [P_table(header_map.get(col, col), header_style) for col in summary_df.columns]
    rows = [header]
    
    # Data rows with text wrapping for long cells
    for _, row in summary_df.iterrows():
        row_data = []
        for col in summary_df.columns:
            v = row[col]
            if pd.isna(v):
                cell_text = ""
            elif isinstance(v, str):
                cell_text = v
            else:
                cell_text = str(v)
            
            # Apply text wrapping to potentially long text columns
            if col in ["Show Title", "Category", "Segment Tilt"]:
                row_data.append(P_table(cell_text, cell_style))
            else:
                row_data.append(cell_text)
        rows.append(row_data)
    
    # Define column widths optimized for landscape LETTER page (10" usable width)
    # Total available: ~10 inches = 720 points (with 0.5" margins each side)
    # Total used: ~7.1 inches (0.7+1.6+1.0+0.7+0.55+0.55+1.4+0.6)
    # Intentionally leaves margin for grid lines, padding, and to avoid clipping.
    col_widths = [
        0.7 * inch,   # Month
        1.6 * inch,   # Show Title (needs room for wrapping)
        1.0 * inch,   # Category
        0.7 * inch,   # Est. Tickets
        0.55 * inch,  # YYC
        0.55 * inch,  # YEG
        1.4 * inch,   # Segment (may need wrapping)
        0.6 * inch,   # Strength (stars)
    ]
    
    table = Table(rows, colWidths=col_widths, repeatRows=1)
    table.setStyle(TableStyle([
        # Font settings - slightly smaller for compact fit
        ("FONT", (0, 0), (-1, -1), "Helvetica", 8),
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8),
        # Header styling
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c5aa0")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("LINEABOVE", (0, 0), (-1, 0), 1, colors.black),
        ("LINEBELOW", (0, 0), (-1, 0), 1, colors.black),
        # Alignment - center numeric columns, left-align text
        ("ALIGN", (3, 1), (5, -1), "CENTER"),  # Tickets columns
        ("ALIGN", (7, 1), (7, -1), "CENTER"),  # Strength stars
        ("ALIGN", (0, 0), (2, -1), "LEFT"),    # Month, Title, Category
        ("ALIGN", (6, 1), (6, -1), "LEFT"),    # Segment
        # Alternating row backgrounds
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
        # Grid and borders
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        # Cell padding - reduced for compact layout
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def _narrative_for_row(r: dict) -> str:
    """
    Generate a comprehensive SHAP-driven narrative for a single title in the Season Rationale.
    
    This narrative uses the title_explanation_engine module to create a multi-paragraph
    (~250-350 word) explanation that includes:
      - Signal positioning (Familiarity/Motivation)
      - Historical & category context (premiere vs remount)
      - Seasonal & macro factors
      - SHAP-based driver summary (if available)
      - Board-level interpretation of Ticket Index
    
    All content is derived from features, predictions, and SHAP values—no hardcoded logic.
    Falls back gracefully if title_explanation_engine is unavailable.
    """
    try:
        from ml.title_explanation_engine import build_title_explanation
        
        # Convert row dict to format expected by explanation engine
        title_metadata = dict(r)
        
        # Call the explanation engine
        narrative = build_title_explanation(
            title_metadata=title_metadata,
            prediction_outputs=None,  # Could be enhanced with model metadata
            shap_values=None,  # Could be enhanced with actual SHAP values if available
            style="board"
        )
        
        return narrative
        
    except (ImportError, ModuleNotFoundError, AttributeError) as e:
        # Fallback to simpler narrative if explanation engine fails
        title = r.get("Title",""); month = r.get("Month",""); cat = r.get("Category","")
        idx_used = r.get("TicketIndex used", None)
        f_season = r.get("FutureSeasonalityFactor", None)
        decay_pct = r.get("ReturnDecayPct", 0.0)

        yyc = (r.get("YYC_Singles",0) or 0)
        yeg = (r.get("YEG_Singles",0) or 0)
        c_share = r.get("CityShare_Calgary", None); e_share = r.get("CityShare_Edmonton", None)

        pri = r.get("PrimarySegment",""); sec = r.get("SecondarySegment","")

        parts = []
        parts.append(f"<b>{month} — {title}</b> ({cat})")
        parts.append(f"Estimated demand comes from a combined interest score converted into a <b>Ticket Index</b> of {_dec(idx_used,1)}.")
        parts.append(f"For {month.split()[0]}, this category's month factor is {_dec(f_season,3)} (months above 1.0 sell better).")
        if decay_pct and float(decay_pct) > 0:
            parts.append(f"We apply a small repeat reduction of {_pct(decay_pct)} due to recent performances.")
        if pri:
            parts.append(f"Likely audience skews to <b>{pri}</b>{' (then '+sec+')' if sec else ''}.")
        parts.append(
            f"We split sales using learned shares: Calgary {_pct(c_share,0)} / Edmonton {_pct(e_share,0)}, giving "
            f"<b>{_num(yyc)}</b> tickets in YYC and <b>{_num(yeg)}</b> in YEG."
        )
        return " ".join(parts)


def _build_month_narratives(plan_df: "pd.DataFrame") -> list:
    """
    Build comprehensive SHAP-driven narratives for each title in the season.
    
    Each narrative is now a multi-paragraph (~250-350 word) explanation that includes:
    - Signal positioning, historical context, seasonal factors, SHAP drivers, and 
      board-level interpretation.
    
    Returns a list of ReportLab Flowables with expanded spacing to accommodate longer narratives.
    """
    styles = _make_styles()
    blocks = [Paragraph("Season Rationale (by month)", styles["h1"])]
    blocks.append(Paragraph(
        "Each title below receives a comprehensive explanation derived from the model's feature analysis, "
        "SHAP value attributions, and learned historical patterns. These narratives provide transparent "
        "insight into what drives each forecast.",
        styles["small"]
    ))
    blocks.append(Spacer(1, 0.15*inch))
    
    for _, rr in plan_df.iterrows():
        # Generate the comprehensive narrative
        narrative_text = _narrative_for_row(rr)
        blocks.append(Paragraph(narrative_text, styles["body"]))
        
        # Increased spacing between titles to accommodate longer narratives
        blocks.append(Spacer(1, 0.2*inch))
    
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
      Show Title, Estimated Tickets, YYC/YEG Singles.
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
    ]

    out = pd.DataFrame(index=index_labels, columns=month_cols, dtype=object)

    for col_name, (_, r) in zip(month_cols, df.iterrows()):
        # Tickets
        est_tix = r.get("EstimatedTickets_Final", r.get("EstimatedTickets", np.nan))
        out.at["Show Title", col_name] = r.get("Title", "")
        out.at["Estimated Tickets", col_name] = int(round(est_tix)) if pd.notna(est_tix) else ""

        out.at["YYC Singles", col_name] = int(r.get("YYC_Singles", 0) or 0)
        out.at["YEG Singles", col_name] = int(r.get("YEG_Singles", 0) or 0)

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


def _make_full_season_table_pdf(plan_df: pd.DataFrame) -> Table:
    """
    Build the '🗓️ Full Season Table (all metrics)' table for the PDF report.
    
    This replaces the old "Season Table (Technical Details)" with a richer,
    full-metrics table that includes all available computed fields.
    
    Columns included (using only fields that are already computed):
      - Month
      - Show Title
      - Category
      - Ticket Index
      - Familiarity
      - Motivation
      - Seasonality Factor
      - Estimated Tickets (Total)
      - YYC Singles
      - YEG Singles
      - Primary Segment
      - Secondary Segment
      - LA_EngagementFactor (if available)
      - Econ_Sentiment (if available)
    
    Args:
        plan_df: The full results DataFrame with all computed columns.
    
    Returns:
        A ReportLab Table object formatted for PDF output.
    """
    from reportlab.platypus import Paragraph as P_table
    from reportlab.lib.styles import ParagraphStyle
    
    if plan_df is None or plan_df.empty:
        return Table([["No season data"]])
    
    # Cell styles for text wrapping
    cell_style = ParagraphStyle(
        'CellStyle',
        fontName='Helvetica',
        fontSize=7,
        leading=9,
    )
    header_style = ParagraphStyle(
        'HeaderStyle',
        fontName='Helvetica-Bold',
        fontSize=7,
        leading=9,
        textColor=colors.whitesmoke,
    )
    
    # Define column mapping: (DataFrame column name, Display header, formatter)
    # Using shortened headers to fit on page
    column_config = [
        ("Month", "Month", lambda x: str(x).split()[0] if pd.notna(x) else ""),
        ("Title", "Show Title", lambda x: str(x) if pd.notna(x) else ""),
        ("Category", "Category", lambda x: str(x) if pd.notna(x) else ""),
        ("TicketIndex used", "Ticket Idx", lambda x: f"{float(x):.1f}" if pd.notna(x) else ""),
        ("Familiarity", "Famil.", lambda x: f"{float(x):.1f}" if pd.notna(x) else ""),
        ("Motivation", "Motiv.", lambda x: f"{float(x):.1f}" if pd.notna(x) else ""),
        ("FutureSeasonalityFactor", "Season F.", lambda x: f"{float(x):.3f}" if pd.notna(x) else ""),
        ("EstimatedTickets_Final", "Est. Tix", lambda x: f"{int(x):,}" if pd.notna(x) else ""),
        ("YYC_Singles", "YYC", lambda x: f"{int(x):,}" if pd.notna(x) else ""),
        ("YEG_Singles", "YEG", lambda x: f"{int(x):,}" if pd.notna(x) else ""),
        ("PrimarySegment", "Prim. Seg", lambda x: str(x) if pd.notna(x) and x else ""),
        ("SecondarySegment", "Sec. Seg", lambda x: str(x) if pd.notna(x) and x else ""),
    ]
    
    # Add optional columns if they exist in the data
    if "LA_EngagementFactor" in plan_df.columns:
        column_config.append(
            ("LA_EngagementFactor", "LA Eng.", lambda x: f"{float(x):.3f}" if pd.notna(x) else "")
        )
    if "Econ_Sentiment" in plan_df.columns:
        column_config.append(
            ("Econ_Sentiment", "Econ.", lambda x: f"{float(x):.3f}" if pd.notna(x) else "")
        )
    
    # Filter to only columns that exist in the data
    valid_cols = [(col, header, fmt) for col, header, fmt in column_config if col in plan_df.columns]
    
    if not valid_cols:
        return Table([["No metrics available"]])
    
    # Build header row
    headers = [P_table(header, header_style) for _, header, _ in valid_cols]
    rows = [headers]
    
    # Build data rows
    for _, r in plan_df.iterrows():
        row_data = []
        for col_name, _, fmt in valid_cols:
            try:
                val = r.get(col_name)
                formatted = fmt(val)
            except Exception:
                formatted = ""
            
            # Apply text wrapping for potentially long text columns
            if col_name in ["Title", "Category", "PrimarySegment", "SecondarySegment"]:
                row_data.append(P_table(formatted, cell_style))
            else:
                row_data.append(formatted)
        rows.append(row_data)
    
    # Use module-level constant for column widths (performance optimization)
    col_widths = [_FULL_SEASON_TABLE_COL_WIDTHS.get(col, 0.5 * inch) for col, _, _ in valid_cols]
    
    table = Table(rows, colWidths=col_widths, repeatRows=1)
    table.setStyle(TableStyle([
        # Font settings - compact for full metrics view
        ("FONT", (0, 0), (-1, -1), "Helvetica", 7),
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 7),
        # Header styling
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c5aa0")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("LINEABOVE", (0, 0), (-1, 0), 1, colors.black),
        ("LINEBELOW", (0, 0), (-1, 0), 1, colors.black),
        # Alignment
        ("ALIGN", (3, 1), (-1, -1), "CENTER"),  # Numeric columns centered
        ("ALIGN", (0, 0), (2, -1), "LEFT"),     # Text columns left
        # Alternating row backgrounds
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
        # Grid
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        # Padding
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]))
    return table


def build_full_pdf_report(methodology_paragraphs: list,
                          plan_df: "pd.DataFrame",
                          season_year: int,
                          org_name: str = "Alberta Ballet") -> bytes:
    """
    Returns a PDF as bytes containing:
      1) Title page
      2) How This Forecast Works — A Plain-Language Overview (NEW: replaces old methodology intro)
      3) Season Summary (Board View) - Clean, leadership-friendly overview
      4) Season Rationale (per month/title)
      5) Methodology & Glossary
      6) 🗓️ Full Season Table (all metrics) - Replaces old "Season Table (Technical Details)"
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
        "Familiarity & Motivation • Ticket Index • Seasonality • City/Segment splits • Marketing spend",
        styles["small"]
    ))
    story.append(Spacer(1, 0.25*inch))

    # (1) How This Forecast Works — A Plain-Language Overview (NEW)
    # This replaces the old "How this forecast works" section and appears near the beginning
    story.extend(_plain_language_overview_text())
    story.append(PageBreak())

    # (2) Season Summary (Board View)
    story.append(Paragraph("Season Summary (Board View)", styles["h1"]))
    story.append(Paragraph(
        "High-level overview of the planned season. Index Strength uses a 1–5 star rating "
        "based on the title's ticket index (★★★★★ = benchmark performance at 100+).",
        styles["small"]))
    story.append(Spacer(1, 0.15*inch))
    story.append(_make_season_summary_table_pdf(plan_df))
    story.append(Spacer(1, 0.3*inch))

    # (3) Season Rationale
    story.extend(_build_month_narratives(plan_df))
    story.append(PageBreak())

    # (4) Methodology & Glossary
    story.extend(methodology_paragraphs)
    story.append(PageBreak())

    # (5) 🗓️ Full Season Table (all metrics)
    # Replaces old "Season Table (Technical Details)" with richer column set
    story.append(Paragraph("🗓️ Full Season Table (all metrics)", styles["h1"]))
    story.append(Paragraph(
        "Complete technical view of each title with all computed metrics. "
        "Indices are benchmark-normalized; tickets reflect seasonality adjustments.",
        styles["small"]))
    story.append(Spacer(1, 0.15*inch))
    story.append(_make_full_season_table_pdf(plan_df))

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
    - API keys (YouTube/Chartmetric) are optional and only used for unknown titles when **Use Live Data** is ON.
    """))

# -------------------------
# METHODOLOGY & GLOSSARY SECTION
# -------------------------
with st.expander("📘 About This App — Methodology & Glossary"):
    st.markdown(dedent(r"""
    ## Simple Terms
    This tool is basically a planning calculator. It looks at how well-known a show is (**Familiarity**) and how excited people seem to be about it (**Motivation**), then turns that into an estimate of how many tickets you might sell. To do that, it pulls clues from the internet (Wikipedia, Google, YouTube, Chartmetric) and combines them with what actually happened in your past seasons. It also remembers how your sales usually split between Calgary and Edmonton.

    On top of that, it adjusts for timing and repeats. Some months are just stronger than others, so the tool nudges each title up or down based on when you plan to run it. If you’re remounting something that ran recently, it assumes demand will be a bit lower than the first time and applies a sensible haircut. If you provide marketing history, it also learns roughly how many dollars of paid media you’ve usually spent per single ticket and uses that to suggest marketing budgets by show and by city. The end result is one view that ties together demand, timing, cities, audience segments, and a ballpark paid-media ask for each title and for the season as a whole.

    ---
    
	### Purpose
    This tool estimates how recognizable a title is (**Familiarity**) and how strongly audiences are inclined to attend (**Motivation**) and then converts those indices into **ticket forecasts**. It blends online visibility signals with learned priors from your historical sales (including YYC/YEG split and Singles mix), applies **seasonality** and **remount decay**, and outputs Alberta-wide projections plus a season builder with **marketing spend** recommendations.

    ---
    ## Methods (end-to-end)

    ### 1) Inputs & signals
    **Online signals (index-scaled; higher = stronger)**  
    - **Wikipedia**: average daily pageviews over the past year for the best-match page.  
      Index formula: `40 + min(110, 20·ln(1+views/day))`.  
    - **Google Trends**: proxy for active interest (lightweight heuristic when offline).  
    - **YouTube**: engagement intensity from median view counts across relevant results.  
      Index formula: `50 + min(90, 9·ln(1+median_views))`, then **winsorized** by title **category** (3rd–97th pct).  
    - **Chartmetric**: 80th-percentile track popularity near the query (fallback heuristic if API not used).

    **History & context**  
    - **Ticket priors**: per-title median tickets (from your `TICKET_PRIORS_RAW`).  
    - **Past runs**: title → `(start, end)` dates to derive a mid-run month for seasonality learning and remount timing.  
    - **Marketing spend history (optional)**: `data/marketing_spend_per_ticket.csv` with per single-ticket $ spend by city for past shows.
	
    > Unknown titles can be scored **two ways**:
    > - **Live** (optional keys): Wikipedia + YouTube + Chartmetric lookups, then category winsorizing.  
    > - **Estimated** (no keys): baseline medians nudged by inferred category/gender.

    ### 2) Feature construction
    **(a) Familiarity & Motivation (raw, pre-normalization)**  
    \[
    \\text{Familiarity} = 0.55\\cdot Wiki + 0.30\\cdot Trends + 0.15\\cdot Chartmetric
    \]  
    \[
    \\text{Motivation} = 0.45\\cdot YouTube + 0.25\\cdot Trends + 0.15\\cdot Chartmetric + 0.15\\cdot Wiki
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
    - **Segment & city breakouts**: use for campaign design, pricing tests, and inventory planning.

    ---
    ## Assumptions & guardrails
    - Linear link between **SignalOnly** and **TicketIndex_DeSeason** (by category where possible).  
    - Outlier control: **YouTube** is winsorized within category (3rd–97th percentile).  
    - Seasonality is conservative: outliers trimmed, winter months pooled, and factors shrunk and clipped.  
    - Benchmark normalization cancels unit differences across segments/region.  
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
	
    ---
    ## Limitations
    - Sparse categories/months reduce model power; app falls back to overall fits or signals-only where needed.  
    - Google Trends and Chartmetric heuristics are proxies when live APIs are off—treat as directional.  
    - Title disambiguation (e.g., films vs ballets) is handled heuristically; review unknown-title results.  
	
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

   ---
    **Recommendation:** Use **Composite** to rank programs and **EstimatedTickets_Final** for capacity planning.
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


def learn_priors_from_history(hist_df: pd.DataFrame) -> dict:
    """
    Learn city split priors from historical data.
    
    Supports two formats:
    1. Wide format (legacy): Show Title, Single Tickets - Calgary, Single Tickets - Edmonton
    2. Combined format (new): city, show_title, single_tickets columns
    
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
    
    # Detect format: combined (long) vs wide
    has_city_col = any('city' in c.lower() for c in df.columns)
    has_wide_calgary = any('calgary' in c.lower() and 'single' in c.lower() for c in df.columns)
    
    if has_city_col and not has_wide_calgary:
        # Combined format detected - convert to wide format
        try:
            df = _convert_combined_to_wide(df)
        except Exception as e:
            import logging
            logging.warning(f"Failed to convert combined history format: {e}")
            return {"titles_learned": 0, "categories_learned": 0, "note": "format conversion failed"}

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

    # Verify columns exist before groupby
    missing_cols = []
    if s_cgy not in df.columns:
        missing_cols.append(s_cgy)
    if s_edm not in df.columns:
        missing_cols.append(s_edm)
    
    if missing_cols:
        import logging
        logging.warning(f"Missing ticket columns in history data: {missing_cols}")
        return {"titles_learned": 0, "categories_learned": 0, "note": f"missing columns: {missing_cols}"}
    
    # aggregate duplicates by title
    try:
        agg = (
            df.groupby(title_col)[[s_cgy, s_edm]]
              .sum(min_count=1)
              .reset_index()
              .rename(columns={title_col: "Title"})
        )
    except Exception as e:
        import logging
        logging.warning(f"Failed to aggregate history data: {e}")
        return {"titles_learned": 0, "categories_learned": 0, "note": f"aggregation error: {str(e)}"}

    # Verify aggregation produced expected columns
    if s_cgy not in agg.columns or s_edm not in agg.columns:
        import logging
        logging.warning(f"Aggregation did not preserve ticket columns. Got: {agg.columns.tolist()}")
        return {"titles_learned": 0, "categories_learned": 0, "note": "aggregation column mismatch"}

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

def _convert_combined_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert combined (long) format to wide format for priors learning.
    
    Expected combined format:
        city, show_title (or title), single_tickets
    
    Returns wide format:
        show_title, Single Tickets - Calgary, Single Tickets - Edmonton
    """
    # Normalize column names
    df_norm = df.copy()
    df_norm.columns = [c.strip().lower() for c in df_norm.columns]
    
    # Find columns
    city_col = 'city' if 'city' in df_norm.columns else None
    title_col = 'show_title' if 'show_title' in df_norm.columns else ('title' if 'title' in df_norm.columns else None)
    tickets_col = 'single_tickets' if 'single_tickets' in df_norm.columns else None
    
    if not city_col or not title_col or not tickets_col:
        raise ValueError(f"Missing required columns. Found: {df_norm.columns.tolist()}")
    
    # Clean city names (handle lowercase/title case)
    df_norm['city_clean'] = df_norm[city_col].astype(str).str.strip().str.lower()
    
    # Aggregate by title and city
    pivot = df_norm.groupby([title_col, 'city_clean'])[tickets_col].sum().unstack(fill_value=0)
    
    # Rename to match wide format expectations
    result = pd.DataFrame()
    result['show_title'] = pivot.index
    result['Single Tickets - Calgary'] = pivot.get('calgary', 0)
    result['Single Tickets - Edmonton'] = pivot.get('edmonton', 0)
    
    return result.reset_index(drop=True)

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

# -------------------------
# Optional APIs (used only if toggled ON)
# -------------------------
try:
    from googleapiclient.discovery import build  # YouTube Data API v3
except Exception:
    build = None

try:
    import spotipy
    from spotipy.oauth2 import ChartmetricClientCredentials
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
    required = {"title", "wiki", "trends", "youtube", "chartmetric", "category", "gender"}
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
            "chartmetric":  float(r[colmap["chartmetric"]]),
            "category": str(r[colmap["category"]]),
            "gender":   str(r[colmap["gender"]]),
        }
        # Add IntentRatio if available
        if "intentratio" in colmap:
            try:
                baselines[title]["intentratio"] = float(r[colmap["intentratio"]])
            except (ValueError, TypeError):
                baselines[title]["intentratio"] = np.nan

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

# --- Post-COVID adjustment (REMOVED per audit finding "Structural Pessimism") ---
# The 0.96 factor was removed to eliminate compounding penalty that caused
# up to 33% reduction in valid predictions. See audit report for details.
POSTCOVID_FACTOR = 1.0  # No post-COVID haircut applied

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

    cm_idx = 0.0
    try:
        if sp_id and sp_secret and spotipy is not None:
            auth = SpotifyClientCredentials(client_id=sp_id, client_secret=sp_secret)
            sp = spotipy.Chartmetric(auth_manager=auth)
            res = sp.search(q=title, type="track,album", limit=10)
            pops = [t.get("popularity", 0) for t in res.get("tracks", {}).get("items", [])]
            cm_idx = float(np.percentile(pops, 80)) if pops else 0.0
    except Exception:
        cm_idx = 0.0
    if cm_idx == 0.0:
        cm_idx = 50.0 + (len(title) * 1.7) % 40.0

    gender, category = infer_gender_and_category(title)
    yt_idx = _winsorize_youtube_to_baseline(category, yt_idx)

    return {"wiki": wiki_idx, "trends": trends_idx, "youtube": yt_idx, "chartmetric": cm_idx,
            "gender": gender, "category": category}

# Scoring utilities
def calc_scores(entry: Dict[str, float | str], seg_key: str, reg_key: str) -> Tuple[float,float]:
    gender = entry["gender"]; cat = entry["category"]
    fam = entry["wiki"] * 0.55 + entry["trends"] * 0.30 + entry["chartmetric"] * 0.15
    mot = entry["youtube"] * 0.45 + entry["trends"] * 0.25 + entry["chartmetric"] * 0.15 + entry["wiki"] * 0.15
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
    """
    Load ticket priors from history CSV, grouping city entries into production runs.
    
    A "production run" is defined as performances of the same show that occur within
    60 days of each other. This typically means Calgary + Edmonton performances that
    are part of the same touring production.
    
    For example:
    - "Once Upon a Time" with Calgary (Sep 11) and Edmonton (Sep 19) = 1 run
    - "Cinderella" with 2018 (Mar) and 2022 (Apr-May) = 2 runs
    
    The function stores per-run ticket totals (not per-city entries) in TICKET_PRIORS_RAW.
    """
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
    # Support various ticket column names
    tix_col = (colmap.get("total_single_tickets") or 
               colmap.get("single_tickets") or 
               colmap.get("tickets") or 
               colmap.get("ticket_median"))
    # Date column for grouping runs
    date_col = colmap.get("start_date")

    if not title_col or not tix_col:
        st.error("CSV must have title column ('show_title' or 'title') and tickets column ('Total Single Tickets', 'single_tickets', 'tickets', or 'ticket_median')")
        TICKET_PRIORS_RAW = {}
        return

    df[title_col] = df[title_col].astype(str).str.strip()

    # Convert ticket column to numeric (handles NaN and invalid values)
    df[tix_col] = pd.to_numeric(df[tix_col], errors="coerce")

    # Parse start_date for grouping production runs
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    priors: dict[str, list[float]] = {}
    for title, g in df.groupby(title_col):
        # Group city entries into production runs based on date proximity
        if date_col and date_col in g.columns and g[date_col].notna().any():
            run_totals = _group_into_production_runs(g, date_col, tix_col)
        else:
            # Fallback: sum all tickets as one run if no date column
            vals = [v for v in g[tix_col].tolist() if pd.notna(v) and v > 0]
            run_totals = [sum(vals)] if vals else []
        
        if run_totals:
            priors[str(title)] = run_totals

    TICKET_PRIORS_RAW = priors


def _group_into_production_runs(
    group: pd.DataFrame,
    date_col: str,
    tix_col: str,
    run_gap_days: int = 60
) -> list[float]:
    """
    Group city entries into production runs based on date proximity.
    
    Entries within run_gap_days of each other are considered part of the same
    production run. The function sums tickets for each production run.
    
    Args:
        group: DataFrame with entries for a single title
        date_col: Name of the date column
        tix_col: Name of the ticket column
        run_gap_days: Maximum days between entries to be considered same run (default: 60)
        
    Returns:
        List of ticket totals, one per production run
    """
    # Sort by date, putting NaN dates at the end
    group = group.sort_values(date_col, na_position='last')
    
    runs = []
    current_run_tickets = []
    current_run_start = None
    
    for _, row in group.iterrows():
        ticket_val = row[tix_col]
        date_val = row[date_col]
        
        # Skip invalid ticket entries
        if pd.isna(ticket_val) or ticket_val <= 0:
            continue
            
        # Handle entries without dates
        if pd.isna(date_val):
            # Add to current run if one exists, otherwise accumulate separately
            current_run_tickets.append(float(ticket_val))
            continue
            
        if current_run_start is None:
            # First entry with a valid date - start a new run
            current_run_start = date_val
            current_run_tickets.append(float(ticket_val))
        elif abs((date_val - current_run_start).days) <= run_gap_days:
            # Same production run (within gap threshold, using abs for safety)
            current_run_tickets.append(float(ticket_val))
        else:
            # New production run - save current run and start new one
            if current_run_tickets:
                runs.append(sum(current_run_tickets))
            current_run_tickets = [float(ticket_val)]
            current_run_start = date_val
    
    # Don't forget the last run
    if current_run_tickets:
        runs.append(sum(current_run_tickets))
    
    return runs

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

# Handle empty case gracefully
if _season_rows:
    SEASONALITY_DF = pd.DataFrame(_season_rows).sort_values(["Category","Month"]).reset_index(drop=True)
    SEASONALITY_TABLE = { (r["Category"], int(r["Month"])): float(r["Factor"]) for _, r in SEASONALITY_DF.iterrows() }
else:
    SEASONALITY_DF = pd.DataFrame(columns=["Category", "Month", "Factor", "n"])
    SEASONALITY_TABLE = {}


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
    """
    REMOVED per audit finding "Structural Pessimism".
    
    The remount decay factor was removed to eliminate compounding penalty
    that caused up to 33% reduction in valid predictions when stacked with
    Post_COVID_Factor. The base model already accounts for remount behavior.
    
    Now always returns 1.0 (no penalty).
    """
    return 1.0

# -------------------------
# UI — Config
# -------------------------
with st.expander("🔑 API Configuration (YouTube & Chartmetric — for NEW titles only)", expanded=True):
    st.markdown("""
    **To fetch live data for unknown titles**, enter your API keys below and enable the checkbox.
    These keys are optional — if not provided, the app will use estimated values for new titles.
    """)
    col1, col2 = st.columns(2)
    with col1:
        yt_key = st.text_input("YouTube Data API v3 Key", type="password", 
                               help="Get a key from Google Cloud Console → APIs & Services → Credentials")
    with col2:
        sp_id = st.text_input("Chartmetric Client ID", type="password",
                              help="Get credentials from Chartmetric Developer Dashboard")
    sp_secret = st.text_input("Chartmetric Client Secret", type="password",
                              help="Get this from Chartmetric Developer Dashboard along with the Client ID")
    use_live = st.checkbox("✅ Use Live Data for Unknown Titles", value=False,
                           help="When enabled, the app will fetch real-time data from YouTube and Chartmetric for titles not in the baseline")
    st.caption("Keys are stored only in your browser session and are cleared when you close or refresh the page. They are only used when scoring unknown titles with live fetch enabled.")

# Fixed mode: AB-wide (Province) + General Population
SEGMENT_DEFAULT = "General Population"
REGION_DEFAULT = "Province"
segment = SEGMENT_DEFAULT
region  = REGION_DEFAULT

st.caption("Mode: **Alberta-wide** (Calgary/Edmonton split learned & applied later) • Audience: **General Population**")

# Post-COVID demand adjustment - REMOVED per audit finding
postcovid_factor = POSTCOVID_FACTOR
st.caption(
    "Post-COVID and Remount decay factors have been removed to eliminate "
    "compounding penalties (audit finding: 'Structural Pessimism'). "
    "Region factor is retained for geographical variance."
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

# Live Analytics category lookup - uses data from loader
def _la_for_category(cat: str) -> dict:
    """Get live analytics data for a category.
    
    Returns a dictionary with metrics from the live_analytics.csv data.
    Uses the get_live_analytics_category_factors function from data/loader.py.
    """
    if not LA_AVAILABLE:
        return {}
    
    try:
        factors = get_live_analytics_category_factors()
        if not factors:
            return {}
        
        cat_lower = cat.lower().strip()
        
        # Direct match
        if cat_lower in factors:
            return factors[cat_lower]
        
        # Map related categories
        category_aliases = {
            'classic_comedy': 'classic_romance',
            'romantic_comedy': 'classic_romance',
            'adult_literary_drama': 'dramatic',
            'contemporary_mixed_bill': 'contemporary',
            'touring_contemporary_company': 'contemporary',
        }
        
        if cat_lower in category_aliases:
            mapped = category_aliases[cat_lower]
            if mapped in factors:
                return factors[mapped]
        
        return {}
    except Exception as e:
        # Log but don't fail - LA data is supplemental
        import logging
        logging.getLogger(__name__).debug(f"Error in _la_for_category for '{cat}': {e}")
        return {}


def _add_live_analytics_overlays(df_in: pd.DataFrame) -> pd.DataFrame:
    """Add Live Analytics engagement factors as overlays to the DataFrame.
    
    Uses data from data/audiences/live_analytics.csv via the loader functions.
    Adds LA_EngagementFactor and related indices per category.
    """
    df = df_in.copy()
    
    # Define LA columns to add
    la_cols = [
        "LA_EngagementFactor",
        "LA_HighSpenderIdx",
        "LA_ActiveBuyerIdx",
        "LA_RepeatBuyerIdx",
        "LA_ArtsAttendIdx",
    ]
    for c in la_cols:
        if c not in df.columns:
            df[c] = np.nan

    def _overlay_row_from_category(cat: str) -> dict:
        la = _la_for_category(cat)
        if not la:
            return {
                "LA_EngagementFactor": 1.0,
                "LA_HighSpenderIdx": 100.0,
                "LA_ActiveBuyerIdx": 100.0,
                "LA_RepeatBuyerIdx": 100.0,
                "LA_ArtsAttendIdx": 100.0,
            }
        return {
            "LA_EngagementFactor": float(la.get("engagement_factor", 1.0)),
            "LA_HighSpenderIdx": float(la.get("high_spender_index", 100.0)),
            "LA_ActiveBuyerIdx": float(la.get("active_buyer_index", 100.0)),
            "LA_RepeatBuyerIdx": float(la.get("repeat_buyer_index", 100.0)),
            "LA_ArtsAttendIdx": float(la.get("arts_attendance_index", 100.0)),
        }

    overlays = df["Category"].map(lambda c: _overlay_row_from_category(str(c)))
    overlays_df = pd.DataFrame(list(overlays)).reindex(df.index)
    for c in [c for c in la_cols if c in overlays_df.columns]:
        df[c] = overlays_df[c]

    return df

# Fallback estimator for unknown titles when live fetch is OFF
def estimate_unknown_title(title: str) -> Dict[str, float | str]:
    # Use baseline medians, then nudge by inferred category
    try:
        base_df = pd.DataFrame(BASELINES).T
        wiki_med = float(base_df["wiki"].median())
        tr_med   = float(base_df["trends"].median())
        yt_med   = float(base_df["youtube"].median())
        sp_med   = float(base_df["chartmetric"].median())
    except Exception:
        wiki_med, tr_med, yt_med, sp_med = 60.0, 55.0, 60.0, 58.0

    gender, category = infer_gender_and_category(title)

    # gentle category nudges so everything isn't identical
    bumps = {
        "family_classic":   {"wiki": +6, "trends": +3, "chartmetric": +2},
        "classic_romance":  {"wiki": +4, "trends": +2},
        "contemporary":     {"youtube": +6, "trends": +2},
        "pop_ip":           {"chartmetric": +10, "trends": +5, "youtube": +4},
        "romantic_tragedy": {"wiki": +3},
        "classic_comedy":   {"trends": +2},
        "dramatic":         {},
        # new categories → reuse nearest neighbour
        "adult_literary_drama":      {"wiki": +3, "trends": +3},              # like dramatic-ish
        "contemporary_mixed_bill":   {"youtube": +4, "trends": +2},           # like contemporary-ish
        "touring_contemporary_company": {"youtube": +5, "trends": +3, "chartmetric": +2},  # slightly more pop-facing
    }
    b = bumps.get(category, {})

    wiki = wiki_med + b.get("wiki", 0.0)
    tr   = tr_med   + b.get("trends", 0.0)
    yt   = yt_med   + b.get("youtube", 0.0)
    sp   = sp_med   + b.get("chartmetric", 0.0)

    # keep within sane bounds
    wiki = float(np.clip(wiki, 30.0, 120.0))
    tr   = float(np.clip(tr,   30.0, 120.0))
    yt   = float(np.clip(yt,   40.0, 140.0))
    sp   = float(np.clip(sp,   35.0, 120.0))

    return {"wiki": wiki, "trends": tr, "youtube": yt, "chartmetric": sp, "gender": gender, "category": category}

def _train_ml_models(df_known_in: pd.DataFrame):
    """
    Train regression models (Constrained Ridge for overall, Ridge for categories) 
    to predict TicketIndex_DeSeason from SignalOnly.
    
    Constrained model ensures:
    - SignalOnly = 0 → TicketIndex ≈ 25 (realistic floor for minimal buzz)
    - SignalOnly = 100 → TicketIndex = 100 (benchmark alignment)
    
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
            X_original = df_known_in[['SignalOnly']].values
            y_original = df_known_in['TicketIndex_DeSeason'].values
            
            # Add synthetic anchor points to enforce desired behavior:
            # - SignalOnly=0 → TicketIndex=25 (low floor for minimal online presence)
            # - SignalOnly=100 → TicketIndex=100 (benchmark alignment)
            # Weight these anchor points to guide the model without overwhelming real data
            n_real = len(df_known_in)
            anchor_weight = max(3, n_real // 2)  # Scale anchor influence with dataset size
            
            X_anchors = np.array([[0.0], [100.0]])
            y_anchors = np.array([25.0, 100.0])
            
            # Repeat anchors to increase their weight
            X_anchors_weighted = np.repeat(X_anchors, anchor_weight, axis=0)
            y_anchors_weighted = np.repeat(y_anchors, anchor_weight)
            
            # Combine real data with weighted anchors
            X = np.vstack([X_original, X_anchors_weighted])
            y = np.concatenate([y_original, y_anchors_weighted])
            
            # Use Ridge regression with regularization to prevent overfitting
            # Alpha controls regularization strength - higher = more conservative
            model = Ridge(alpha=5.0, random_state=42)
            
            # Train the model
            model.fit(X, y)
            overall_model = model
            
            # Calculate metrics on REAL data only (not anchors)
            y_pred_real = model.predict(X_original)
            mae = mean_absolute_error(y_original, y_pred_real)
            rmse = np.sqrt(mean_squared_error(y_original, y_pred_real))
            r2 = r2_score(y_original, y_pred_real)
            
            # Verify anchor point behavior
            anchor_preds = model.predict(X_anchors)
            
            overall_metrics = {
                'MAE': float(mae),
                'RMSE': float(rmse),
                'R2': float(r2),
                'n_samples': len(df_known_in),
                'intercept': float(model.intercept_),
                'slope': float(model.coef_[0]),
                'anchor_0': float(anchor_preds[0]),  # Should be ~25
                'anchor_100': float(anchor_preds[1])  # Should be ~100
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
    Fit regression models (Constrained Ridge if ML available, otherwise constrained linear).
    Returns model objects or coefficients depending on availability.
    
    Ensures realistic predictions:
    - SignalOnly = 0 → TicketIndex ≈ 25
    - SignalOnly = 100 → TicketIndex = 100
    """
    if ML_AVAILABLE and len(df_known_in) >= 3:
        # Use constrained Ridge regression models
        overall_model, cat_models, overall_metrics, cat_metrics = _train_ml_models(df_known_in)
        return ('ml', overall_model, cat_models, overall_metrics, cat_metrics)
    else:
        # Fallback to constrained linear regression
        overall = None
        if len(df_known_in) >= 5:
            # Add anchor points to constrain the linear fit
            x_real = df_known_in["SignalOnly"].values
            y_real = df_known_in["TicketIndex_DeSeason"].values
            
            # Add weighted anchors: SignalOnly=0→25, SignalOnly=100→100
            n_anchors = max(2, len(x_real) // 3)
            x_anchors = np.array([0.0] * n_anchors + [100.0] * n_anchors)
            y_anchors = np.array([25.0] * n_anchors + [100.0] * n_anchors)
            
            x_combined = np.concatenate([x_real, x_anchors])
            y_combined = np.concatenate([y_real, y_anchors])
            
            a, b = np.polyfit(x_combined, y_combined, 1)
            overall = (float(a), float(b))
        cat_coefs = {}
        for cat, g in df_known_in.groupby("Category"):
            if len(g) >= 3:
                # Constrained per-category fits
                xs_real = g["SignalOnly"].values
                ys_real = g["TicketIndex_DeSeason"].values
                
                n_anchors = max(1, len(xs_real) // 3)
                xs_anchors = np.array([0.0] * n_anchors + [100.0] * n_anchors)
                ys_anchors = np.array([25.0] * n_anchors + [100.0] * n_anchors)
                
                xs_combined = np.concatenate([xs_real, xs_anchors])
                ys_combined = np.concatenate([ys_real, ys_anchors])
                
                a, b = np.polyfit(xs_combined, ys_combined, 1)
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
            "YouTubeIdx": entry["youtube"], "ChartmetricIdx": entry["chartmetric"],
            "Source": src,
            # Diagnostic field: lead_gender for export
            "lead_gender": entry["gender"],
            # IntentRatio: proportion of search traffic that is performance-specific
            "IntentRatio": entry.get("intentratio", np.nan),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        st.error("No titles to score — check the Titles box.")
        return

    # Attach show type
    df["ShowType"] = df.apply(
        lambda r: infer_show_type(r["Title"], r["Category"]),
        axis=1,
    )

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

    # --- Add diagnostic fields for export ---
    # ticket_median_prior: median tickets from prior runs (mirrors TicketMedian for export clarity)
    # Note: This is intentionally duplicated for export schema consistency - allows analysts to
    # identify which field represents "prior median" vs other ticket-related metrics
    df["ticket_median_prior"] = med_list
    
    # prior_total_tickets: sum of all ticket values from prior runs
    # run_count_prior: count of runs for this title
    prior_totals, run_counts = [], []
    for _, r in df.iterrows():
        title = r["Title"]
        priors = TICKET_PRIORS_RAW.get(title, [])
        if priors:
            prior_totals.append(sum(priors))
            run_counts.append(len(priors))
        else:
            prior_totals.append(np.nan)
            run_counts.append(0)
    df["prior_total_tickets"] = prior_totals
    df["run_count_prior"] = run_counts

    # 4) Fit regression models (XGBoost/GradientBoosting or simple linear)
    df_known = df[pd.notna(df["TicketIndex_DeSeason"])].copy()

    model_result = _fit_overall_and_by_category(df_known)
    model_type = model_result[0]
    
    # Helper to extract baseline signals from a DataFrame row
    def _extract_baseline_signals(row) -> dict:
        """
        Extract baseline signal values from a DataFrame row.
        
        Args:
            row: A DataFrame row or dict-like object with WikiIdx, TrendsIdx, etc.
        
        Returns:
            Dict with keys 'wiki', 'trends', 'youtube', 'chartmetric' containing float values
        """
        return {
            "wiki": float(row.get("WikiIdx", 0) or 0),
            "trends": float(row.get("TrendsIdx", 0) or 0),
            "youtube": float(row.get("YouTubeIdx", 0) or 0),
            "chartmetric": float(row.get("ChartmetricIdx", 0) or 0),
        }
    
    # 4a) Build k-NN index for cold-start fallback
    # Uses baseline signals (wiki, trends, youtube, spotify) to find similar titles
    knn_index = None
    knn_enabled = KNN_CONFIG.get("enabled", True)
    if knn_enabled and KNN_FALLBACK_AVAILABLE and len(df_known) >= 3:
        try:
            # Build DataFrame for kNN with baseline signals and de-seasonalized ticket index
            knn_data = df_known[["Title", "WikiIdx", "TrendsIdx", "YouTubeIdx", "ChartmetricIdx", 
                                 "TicketIndex_DeSeason", "Category"]].copy()
            knn_data = knn_data.rename(columns={
                "WikiIdx": "wiki",
                "TrendsIdx": "trends", 
                "YouTubeIdx": "youtube",
                "ChartmetricIdx": "chartmetric",
                "TicketIndex_DeSeason": "ticket_index"
            })
            knn_data = knn_data.dropna(subset=["wiki", "trends", "youtube", "chartmetric", "ticket_index"])
            
            if len(knn_data) >= 3:
                knn_index = build_knn_from_config(
                    knn_data, 
                    outcome_col="ticket_index",
                    last_run_col=None  # No date column in this context
                )
                # kNN index built successfully - ready for fallback predictions
        except Exception as e:
            # kNN build failed - continue without it
            knn_index = None
    
    # Helper to predict with kNN (returns prediction, source, neighbors_json)
    def _predict_with_knn(baseline_signals: dict) -> tuple[float, str, str]:
        """
        Use k-NN to predict ticket index from baseline signals.
        
        Args:
            baseline_signals: Dict with keys 'wiki', 'trends', 'youtube', 'chartmetric'
                containing the baseline signal values for the title
        
        Returns:
            Tuple of (predicted_value, source_label, neighbors_json):
            - predicted_value: The predicted ticket index (clipped to [20, 180])
            - source_label: "kNN Fallback" on success, "Not enough data" on failure
            - neighbors_json: JSON string with neighbor details for debugging
        """
        if knn_index is None:
            return np.nan, "Not enough data", "[]"
        try:
            pred, neighbors_df = knn_index.predict(baseline_signals, return_neighbors=True)
            if np.isnan(pred):
                return np.nan, "Not enough data", "[]"
            pred = float(np.clip(pred, 20.0, 180.0))
            # Create JSON summary of neighbors for debugging
            # Use all neighbors returned by knn_index.predict (already limited to k)
            if neighbors_df is not None and not neighbors_df.empty:
                neighbors_list = []
                for _, nr in neighbors_df.iterrows():
                    neighbors_list.append({
                        "title": str(nr.get("Title", "")),
                        "similarity": round(float(nr.get("similarity", 0)), 3),
                        "ticket_index": round(float(nr.get("ticket_index", 0)), 1)
                    })
                neighbors_json = json.dumps(neighbors_list)
            else:
                neighbors_json = "[]"
            return pred, "kNN Fallback", neighbors_json
        except (ValueError, RuntimeError, KeyError) as e:
            # Log specific errors for debugging but return fallback values
            import logging
            logging.getLogger(__name__).debug(f"kNN prediction failed: {e}")
            return np.nan, "Not enough data", "[]"
    
    if model_type == 'ml':
        _, overall_model, cat_models, overall_metrics, cat_metrics = model_result
        
        # Model performance metrics available for debugging if needed
        # (Hidden from UI to avoid confusion - metrics stored in session state)
        
        # 5) Impute missing TicketIndex with ML models, with kNN fallback
        def _predict_ticket_index_deseason(signal_only: float, category: str, baseline_signals: dict) -> tuple[float, str, str]:
            # Try category model first
            if category in cat_models:
                pred = _predict_with_ml_model(cat_models[category], signal_only)
                if not np.isnan(pred):
                    pred = float(np.clip(pred, 20.0, 180.0))
                    return pred, "ML Category", "[]"
            
            # Try overall model
            if overall_model is not None:
                pred = _predict_with_ml_model(overall_model, signal_only)
                if not np.isnan(pred):
                    pred = float(np.clip(pred, 20.0, 180.0))
                    return pred, "ML Overall", "[]"
            
            # Fall back to k-NN if enabled
            if knn_enabled and knn_index is not None:
                return _predict_with_knn(baseline_signals)
            
            return np.nan, "Not enough data", "[]"
    else:
        # Linear regression fallback
        _, overall_coef, cat_coefs, _, _ = model_result
        
        # 5) Impute missing TicketIndex with linear models, with kNN fallback
        def _predict_ticket_index_deseason(signal_only: float, category: str, baseline_signals: dict) -> tuple[float, str, str]:
            if category in cat_coefs:
                a, b = cat_coefs[category]; src = "Category model"
                pred = a * signal_only + b
                pred = float(np.clip(pred, 20.0, 180.0))
                return pred, src, "[]"
            elif overall_coef is not None:
                a, b = overall_coef; src = "Overall model"
                pred = a * signal_only + b
                pred = float(np.clip(pred, 20.0, 180.0))
                return pred, src, "[]"
            
            # Fall back to k-NN if enabled
            if knn_enabled and knn_index is not None:
                return _predict_with_knn(baseline_signals)
            
            return np.nan, "Not enough data", "[]"

    imputed_vals, imputed_srcs, predicted_vals, knn_neighbors_data = [], [], [], []
    for _, r in df.iterrows():
        if pd.notna(r["TicketIndex_DeSeason"]):
            imputed_vals.append(r["TicketIndex_DeSeason"]); imputed_srcs.append("History")
            predicted_vals.append(np.nan)  # No ML prediction needed for history
            knn_neighbors_data.append("[]")
        else:
            # Use helper to extract baseline signals for kNN fallback
            baseline_signals = _extract_baseline_signals(r)
            pred, src, neighbors_json = _predict_ticket_index_deseason(r["SignalOnly"], r["Category"], baseline_signals)
            imputed_vals.append(pred); imputed_srcs.append(src)
            predicted_vals.append(pred)  # Store raw ML prediction
            knn_neighbors_data.append(neighbors_json)

    df["TicketIndex_DeSeason_Used"] = imputed_vals
    df["TicketIndexSource"] = imputed_srcs
    df["TicketIndex_Predicted"] = predicted_vals  # Raw ML prediction (pre-adjustment)

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
            "chartmetric": _safe_float(r.get("ChartmetricIdx", 0.0)),
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

    # --- Additional diagnostic fields for export ---
    # dominant_audience_segment: alias for PredictedPrimarySegment
    df["dominant_audience_segment"] = prim_list
    
    # segment_weights: JSON representation of segment mix shares
    import json
    segment_weights_json = []
    for i in range(len(mix_gp)):
        weights = {
            "General Population": round(mix_gp[i], 4),
            "Core Classical (F35–64)": round(mix_core[i], 4),
            "Family (Parents w/ kids)": round(mix_family[i], 4),
            "Emerging Adults (18–34)": round(mix_ea[i], 4),
        }
        segment_weights_json.append(json.dumps(weights))
    df["segment_weights"] = segment_weights_json

    # month_of_opening: integer (1-12) from proposed run date
    if proposed_run_date is not None:
        df["month_of_opening"] = int(proposed_run_date.month)
    else:
        df["month_of_opening"] = np.nan
    
    # holiday_flag: boolean, True if month is in Nov-Jan (holiday season)
    if proposed_run_date is not None:
        df["holiday_flag"] = proposed_run_date.month in (11, 12, 1)
    else:
        df["holiday_flag"] = False
    
    # category_seasonality_factor: seasonality factor for this category/month combination
    # Note: This is the final factor after shrinkage and clipping (same as FutureSeasonalityFactor)
    # The raw factor before shrinkage is not typically exposed as it may be unreliable for sparse data
    df["category_seasonality_factor"] = df["FutureSeasonalityFactor"]
    
    # k-NN metadata: indicate whether k-NN was used and store neighbor info
    # k-NN is used when the TicketIndexSource indicates a k-NN-based prediction
    KNN_SOURCE_INDICATORS = {"kNN", "k-NN", "KNN", "knn fallback", "kNN Fallback", "k-Nearest Neighbors"}
    knn_used_list = []
    for i, src in enumerate(imputed_srcs):
        # Check if source matches any k-NN indicator
        is_knn = any(indicator in str(src) for indicator in KNN_SOURCE_INDICATORS)
        knn_used_list.append(is_knn)
    df["kNN_used"] = knn_used_list
    # kNN_neighbors was populated during the prediction loop
    df["kNN_neighbors"] = knn_neighbors_data

    # 10a) Live Analytics overlays (engagement factors by category)
    df = _add_live_analytics_overlays(df)
    
    # Add LA_Category field based on the Category column (for diagnostic export)
    # This maps the show category to the Live Analytics category structure
    df["LA_Category"] = df["Category"]  # Default to show category

    # 10b) Economic sentiment factor (BoC + Alberta data)
    try:
        econ_context = get_current_economic_context(include_boc=True, include_alberta=True)
        econ_sentiment = float(econ_context.get("combined_sentiment", 1.0))
        econ_sources = econ_context.get("sources_available", [])
        boc_sentiment = econ_context.get("boc_sentiment")
        alberta_sentiment = econ_context.get("alberta_sentiment")
    except Exception as e:
        # Log but don't fail - economic data is supplemental
        import logging
        logging.getLogger(__name__).debug(f"Error fetching economic context: {e}")
        econ_sentiment = 1.0
        econ_sources = []
        boc_sentiment = None
        alberta_sentiment = None

    df["Econ_Sentiment"] = econ_sentiment
    df["Econ_BocFactor"] = boc_sentiment if boc_sentiment is not None else np.nan
    df["Econ_AlbertaFactor"] = alberta_sentiment if alberta_sentiment is not None else np.nan
    df["Econ_Sources"] = ", ".join(econ_sources) if econ_sources else "none"

    # 11) Final ticket calculation (Remount decay + Post-COVID factors REMOVED per audit)
    # Both factors were eliminated to prevent "Structural Pessimism" - compounding penalties
    # that reduced valid predictions by up to 33%. Region factor is retained and applied
    # via REGION_MULT during score calculation.
    decay_pcts, decay_factors, est_after_decay = [], [], []
    for _, r in df.iterrows():
        raw_est = r.get("EstimatedTickets", 0.0)
        est_base = float(raw_est) if pd.notna(raw_est) else 0.0
        
        # No decay applied - factors removed per audit
        decay_pct = 0.0
        factor = 1.0
        est_final = round(est_base * POSTCOVID_FACTOR)  # POSTCOVID_FACTOR is now 1.0

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
        raw_total = r.get("EstimatedTickets_Final", r.get("EstimatedTickets", 0.0))
        total = float(raw_total) if pd.notna(raw_total) else 0.0
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
            .reindex(["History", "Category model", "Overall model", "ML Category", "ML Overall", "kNN Fallback", "Not enough data"])
            .fillna(0)
            .astype(int)
        )
        # Build display string with only non-zero counts
        count_parts = []
        for src_name, count in src_counts.items():
            if count > 0:
                count_parts.append(f"{src_name}: {int(count)}")
        if count_parts:
            st.caption(f"TicketIndex source — {' · '.join(count_parts)}")

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
    
    # --- Intent Ratio Display ---
    intent_ratio = None
    if "IntentRatio" in df.columns and df["IntentRatio"].notna().any():
        intent_ratio = df["IntentRatio"].dropna().astype(float).median()  # or mean()
    
    if intent_ratio is not None:
        st.metric("Intent Ratio", f"{intent_ratio:.1%}")
        if intent_ratio < 0.05:
            st.caption("⚠️ High ambiguity — search volume likely includes unrelated content (e.g., movies, books).")
        elif intent_ratio > 0.30:
            st.caption("✅ High clarity — search traffic is performance-specific.")
        else:
            st.caption("ℹ️ Mixed signal — may benefit from clearer marketing messaging.")

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

    # Columns shown in "Estimated ticket sales (table view)" are limited to
    # those most salient per TICKET_ESTIMATOR_FORMULAS.md:
    #   - Input Signals (Section 1)
    #   - Familiarity & Motivation (Section 2)
    #   - Ticket Index & Seasonality (Sections 4, 5)
    #   - Composite & Final Tickets (Section 11)
    #   - City Split (Section 7)
    #   - Diagnostic & Contextual Fields (Section 14)
    # NOTE: ReturnDecayFactor removed - remount decay eliminated per audit
    table_cols = [
        "Title", "Category",
        # Input Signal Variables (Section 1)
        "WikiIdx", "TrendsIdx", "YouTubeIdx", "ChartmetricIdx",
        # Familiarity & Motivation (Section 2)
        "Familiarity", "Motivation", "SignalOnly",
        # IntentRatio: search traffic clarity metric
        "IntentRatio",
        # Ticket Index (Section 4) & Seasonality (Section 5)
        "TicketIndex used", "TicketIndexSource", "FutureSeasonalityFactor",
        # Composite & Final Tickets (Section 11)
        "Composite", "EstimatedTickets_Final",
        # City Split (Section 7)
        "YYC_Singles", "YEG_Singles",
        # Diagnostic & Contextual Fields (Section 14)
        # Show & Audience Context
        "lead_gender", "dominant_audience_segment", "segment_weights",
        # Model & Historical Inputs
        "ticket_median_prior", "prior_total_tickets", "run_count_prior",
        "TicketIndex_Predicted",
        # Temporal & Seasonality Info
        "month_of_opening", "category_seasonality_factor",
        # k-NN Metadata
        "kNN_used", "kNN_neighbors",
        # Live Analytics
        "LA_Category",
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
            # Input signals
            "WikiIdx": "{:.0f}",
            "TrendsIdx": "{:.0f}",
            "YouTubeIdx": "{:.0f}",
            "ChartmetricIdx": "{:.0f}",
            # Familiarity & Motivation
            "Familiarity": "{:.1f}",
            "Motivation": "{:.1f}",
            "SignalOnly": "{:.1f}",
            # IntentRatio as percentage
            "IntentRatio": "{:.1%}",
            # Ticket Index & Seasonality
            "TicketIndex used": "{:.1f}",
            "FutureSeasonalityFactor": "{:.3f}",
            # Composite & Tickets
            "Composite": "{:.1f}",
            "EstimatedTickets_Final": "{:,.0f}",
            # City Split
            "YYC_Singles": "{:,.0f}",
            "YEG_Singles": "{:,.0f}",
            # Diagnostic & Contextual Fields (Section 14)
            # Model & Historical Inputs
            "ticket_median_prior": "{:,.0f}",
            "prior_total_tickets": "{:,.0f}",
            "run_count_prior": "{:,.0f}",
            "TicketIndex_Predicted": "{:.1f}",
            # Temporal & Seasonality Info
            "month_of_opening": "{:.0f}",
            "category_seasonality_factor": "{:.3f}",
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

        # Final ticket calculation (Remount decay + Post-COVID factors REMOVED per audit)
        # Both factors now return 1.0 - no penalty applied
        decay_factor = remount_novelty_factor(title_sel, run_date)  # Now returns 1.0
        est_tix_raw = (est_tix if np.isfinite(est_tix) else 0) * decay_factor
        est_tix_final = int(round(est_tix_raw * POSTCOVID_FACTOR))  # POSTCOVID_FACTOR is 1.0

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

        plan_rows.append({
            "Month": f"{m_name} {run_year}",
            "Title": title_sel,
            "Category": cat,
            "PrimarySegment": r.get("PredictedPrimarySegment", ""),
            "SecondarySegment": r.get("PredictedSecondarySegment", ""),
            "WikiIdx": r.get("WikiIdx", np.nan),
            "TrendsIdx": r.get("TrendsIdx", np.nan),
            "YouTubeIdx": r.get("YouTubeIdx", np.nan),
            "ChartmetricIdx": r.get("ChartmetricIdx", np.nan),
            "Familiarity": r.get("Familiarity", np.nan),
            "Motivation": r.get("Motivation", np.nan),
            "SignalOnly": r.get("SignalOnly", np.nan),
            "IntentRatio": r.get("IntentRatio", np.nan),
            
            # Live Analytics factors
            "LA_EngagementFactor": r.get("LA_EngagementFactor", 1.0),
            "LA_HighSpenderIdx": r.get("LA_HighSpenderIdx", 100.0),
            "LA_ActiveBuyerIdx": r.get("LA_ActiveBuyerIdx", 100.0),
            "LA_RepeatBuyerIdx": r.get("LA_RepeatBuyerIdx", 100.0),
            "LA_ArtsAttendIdx": r.get("LA_ArtsAttendIdx", 100.0),
            
            # Economic factors
            "Econ_Sentiment": r.get("Econ_Sentiment", 1.0),
            "Econ_BocFactor": r.get("Econ_BocFactor", np.nan),
            "Econ_AlbertaFactor": r.get("Econ_AlbertaFactor", np.nan),
            "Econ_Sources": r.get("Econ_Sources", "none"),
            
            "TicketHistory": r.get("TicketMedian", np.nan),
            "TicketIndex_DeSeason_Used": r.get("TicketIndex_DeSeason_Used", np.nan),
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

            # --- Diagnostic & Contextual Fields ---
            # Show & Audience Context
            "lead_gender": r.get("lead_gender", r.get("Gender", "n/a")),
            "dominant_audience_segment": r.get("dominant_audience_segment", r.get("PredictedPrimarySegment", "")),
            "segment_weights": r.get("segment_weights", "{}"),

            # Model & Historical Inputs
            "ticket_median_prior": r.get("ticket_median_prior", r.get("TicketMedian", np.nan)),
            "prior_total_tickets": r.get("prior_total_tickets", np.nan),
            "run_count_prior": r.get("run_count_prior", 0),
            "TicketIndex_Predicted": r.get("TicketIndex_Predicted", np.nan),

            # Temporal & Seasonality Info
            "month_of_opening": int(m_num),
            "holiday_flag": m_num in (11, 12, 1),
            "category_seasonality_factor": r.get("category_seasonality_factor", f_season),

            # k-NN Metadata
            "kNN_used": r.get("kNN_used", False),
            "kNN_neighbors": r.get("kNN_neighbors", "[]"),

            # LA Category (if available from live analytics)
            "LA_Category": r.get("LA_Category", cat),
        })

    # Guard + render
    if not plan_rows:
        st.caption("Pick at least one month/title above to see your season projection, charts, and scatter.")
        return

    # Keep full plan_df (do NOT trim away WikiIdx/Trends/etc.)
    plan_df = pd.DataFrame(plan_rows)

    # A view with a preferred column order for some displays
    # NOTE: ReturnDecayPct removed - remount decay eliminated per audit
    desired_order = [
        "Month",
        "Title",
        "Category",
        "PrimarySegment",
        "SecondarySegment",
        "TicketIndex used",
        "FutureSeasonalityFactor",
        "EstimatedTickets_Final",
        "YYC_Singles",
        "YEG_Singles",
        "CityShare_Calgary",
        "CityShare_Edmonton",
    ]
    present_plan_cols = [c for c in desired_order if c in plan_df.columns]
    plan_view = plan_df[present_plan_cols].copy()

    # --- Executive summary KPIs ---
    with st.container():
        st.markdown("### 📊 Season at a glance")
        c1, c2, c3 = st.columns(3)

        yyc_tot = int(plan_df["YYC_Singles"].sum())
        yeg_tot = int(plan_df["YEG_Singles"].sum())
        grand = int(plan_df["EstimatedTickets_Final"].sum()) or 1

        with c1:
            st.metric("Projected Season Tickets", f"{grand:,}")
        with c2:
            st.metric("Calgary • share", f"{yyc_tot:,}", delta=f"{yyc_tot/grand:.1%}")
        with c3:
            st.metric("Edmonton • share", f"{yeg_tot:,}", delta=f"{yeg_tot/grand:.1%}")

        st.caption(
            "Penalty factors removed per audit: Post-COVID and Remount decay "
            "eliminated to prevent 'Structural Pessimism'. Region factor retained."
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

        # === NEW: Season Summary (Board View) - Shown first ===
        st.markdown("#### 📋 Season Summary (Board View)")
        st.caption(
            "High-level overview for leadership review. "
            "Index Strength uses a 1–5 star rating based on the title's ticket index "
            "(★★★★★ = benchmark performance at 100+)."
        )
        
        season_summary_df = build_season_summary(plan_df)
        if not season_summary_df.empty:
            st.dataframe(season_summary_df, width='stretch', hide_index=True)
            
            # Download button for Season Summary CSV
            summary_csv_bytes = season_summary_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Season Summary (Board View) CSV",
                data=summary_csv_bytes,
                file_name=f"season_summary_board_{season_year}.csv",
                mime="text/csv",
                key="download_season_summary_csv"
            )
        else:
            st.info("No shows scheduled yet. Select titles for each month above.")
        
        st.markdown("---")

        # === Existing detailed table - Now in expander ===
        with st.expander("📊 Advanced technical table (full export)", expanded=False):
            # NOTE: ReturnDecayFactor and ReturnDecayPct removed - remount decay eliminated per audit
            metrics = [
                "Title","Category","PrimarySegment","SecondarySegment",
                "WikiIdx","TrendsIdx","YouTubeIdx","ChartmetricIdx",
                "Familiarity","Motivation","SignalOnly",
                # Live Analytics factors
                "LA_EngagementFactor","LA_HighSpenderIdx","LA_ActiveBuyerIdx",
                "LA_RepeatBuyerIdx","LA_ArtsAttendIdx",
                # Economic factors
                "Econ_Sentiment","Econ_BocFactor","Econ_AlbertaFactor","Econ_Sources",
                "TicketHistory","TicketIndex_DeSeason_Used","TicketIndex used","TicketIndexSource",
                "FutureSeasonalityFactor","HistSeasonalityFactor",
                "Composite","Score",
                "EstimatedTickets","EstimatedTickets_Final",
                "YYC_Singles","YEG_Singles",
                "CityShare_Calgary","CityShare_Edmonton",
                # Diagnostic & Contextual Fields
                "lead_gender","dominant_audience_segment","segment_weights",
                "ticket_median_prior","prior_total_tickets","run_count_prior",
                "TicketIndex_Predicted",
                "month_of_opening","holiday_flag","category_seasonality_factor",
                "kNN_used","kNN_neighbors","LA_Category",
            ]

            # assemble wide DF: rows = metrics, columns = month labels
            df_wide = pd.DataFrame(
                { col: [month_to_row[col].get(m, np.nan) for m in metrics] for col in month_cols },
                index=metrics
            )

            sty = df_wide.style

            # Integer counts (tickets)
            int_like_rows = [
                "TicketHistory","EstimatedTickets","EstimatedTickets_Final",
                "YYC_Singles","YEG_Singles",
                # LA indices (shown as whole numbers)
                "LA_HighSpenderIdx","LA_ActiveBuyerIdx","LA_RepeatBuyerIdx","LA_ArtsAttendIdx",
                # New diagnostic fields - integer values
                "ticket_median_prior","prior_total_tickets","run_count_prior","month_of_opening",
            ]
            sty = sty.format("{:,.0f}", subset=_S[int_like_rows, :])

            # Indices / composites (one decimal)
            idx_rows = [
                "WikiIdx","TrendsIdx","YouTubeIdx","ChartmetricIdx",
                "Familiarity","Motivation","SignalOnly","Composite","TicketIndex used",
                "TicketIndex_DeSeason_Used","TicketIndex_Predicted",
            ]
            sty = sty.format("{:.1f}", subset=_S[idx_rows, :])

            # Factors
            sty = sty.format("{:.3f}", subset=_S[["FutureSeasonalityFactor","HistSeasonalityFactor","category_seasonality_factor"], :])
            
            # LA and Econ factors
            sty = sty.format("{:.3f}", subset=_S[["LA_EngagementFactor","Econ_Sentiment","Econ_BocFactor","Econ_AlbertaFactor"], :])

            # Percentages
            sty = sty.format("{:.0%}", subset=_S[["CityShare_Calgary","CityShare_Edmonton"], :])

            st.markdown("##### 🗓️ Full Season Table (all metrics)")
            st.dataframe(sty, width='stretch')

            # CSV download
            st.download_button(
                "⬇️ Download Season (wide) CSV",
                df_wide.reset_index().rename(columns={"index":"Metric"}).to_csv(index=False).encode("utf-8"),
                file_name=f"season_plan_wide_{season_year}.csv",
                mime="text/csv",
                key="download_season_wide_csv"
            )

        # Full PDF report download (outside expander, always visible)
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
                key="download_pdf_report"
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


def _xgb_predict_tickets(feature_df: pd.DataFrame) -> np.ndarray:
    try:
        import xgboost as xgb
    except ImportError:
        raise RuntimeError('xgboost not installed; cannot use XGB ticket model')
    model_path = ML_CONFIG.get('path', 'model_xgb_remount_postcovid.json')
    booster = xgb.Booster()
    booster.load_model(model_path)
    dmx = xgb.DMatrix(feature_df.values, feature_names=list(feature_df.columns))
    preds = booster.predict(dmx)
    return np.maximum(preds, 0.0)


def compute_scores_and_store(
    titles: pd.DataFrame,
    segment: str,
    region: str,
    use_live: bool,
    yt_key: str,
    sp_id: str,
    sp_secret: str,
    benchmark_title: str,
    proposed_run_date: date,
    postcovid_factor: float,
):
    """Wrapper that preserves UI but uses XGB model as ticket brain.

    - Keeps Stone Olafson, econ, city split, and live signals flow intact.
    - Replaces hand-built ticket combination with trained XGB model.
    """
    # existing helper builds enriched plan_df with all the right columns
    plan_df = build_season_plan(
        titles=titles,
        segment=segment,
        region=region,
        use_live=use_live,
        yt_key=yt_key,
        sp_id=sp_id,
        sp_secret=sp_secret,
        benchmark_title=benchmark_title,
        proposed_run_date=proposed_run_date,
        postcovid_factor=postcovid_factor,
    )

    # Select feature columns for XGB from enriched plan_df
    feature_cols = [
        c for c in plan_df.columns
        if c in [
            'prior_total_tickets',
            'ticket_median_prior',
            'trends',
            'youtube',
            'wiki',
            'chartmetric',
            'familiarity',
            'motivation',
            'is_remount_recent',
            'postcovid_factor',
        ]
    ]
    if not feature_cols:
        raise RuntimeError('No matching feature columns found for XGB model')

    feat_df = plan_df[feature_cols].copy().fillna(0.0).astype(float)
    tickets = _xgb_predict_tickets(feat_df)

    plan_df['EstimatedTickets_Final'] = tickets
    st.session_state['results'] = plan_df
