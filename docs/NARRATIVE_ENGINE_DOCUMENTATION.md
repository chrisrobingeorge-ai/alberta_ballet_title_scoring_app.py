# Title Explanation Engine Documentation

## Overview

The Title Explanation Engine is a SHAP-driven narrative generation system that produces comprehensive, multi-paragraph explanations for each title in the Alberta Ballet Season Report PDF. This engine replaces the previous one-paragraph summaries with ~250-word explanations that provide transparent insight into model predictions.

## Architecture

### Core Components

1. **`ml/title_explanation_engine.py`** - Main narrative generation module
   - `build_title_explanation()` - Primary function for generating narratives
   - Helper functions for signal interpretation, category mapping, and SHAP translation

2. **`streamlit_app.py`** - Integration points
   - `_plain_language_overview_text()` - Enhanced Season Report introduction (8 paragraphs)
   - `_narrative_for_row()` - Per-title narrative wrapper
   - `_build_month_narratives()` - Season Report section builder

3. **`tests/test_title_explanation_engine.py`** - Comprehensive test suite
   - Unit tests for narrative generation
   - Edge case handling
   - Word count validation

## Narrative Structure

Each title explanation follows a consistent 5-paragraph structure:

### 1. Signal Positioning
- Title, month, and category identification
- Familiarity score interpretation (0-100+ scale)
- Motivation score interpretation (0-100+ scale)
- Digital platform signal sources (Wikipedia, Google, YouTube, chartmetric)

**Example:**
> This title registers a **Familiarity** score of 135.0 (exceptionally high) and a **Motivation** score of 115.0 (strong), reflecting strong public visibility across Wikipedia page traffic, Google search patterns, YouTube viewing behavior, and chartmetric streaming activity.

### 2. Historical & Category Context
- Premiere vs remount identification
- Years since last performance (for remounts)
- Category-specific patterns and baselines
- How the model handles titles without local history

**Example:**
> This production represents a remount, last performed approximately 3 years ago. Historical Alberta Ballet data shows that adult classical productions typically benefit from audience recognition on return engagements, a pattern the model incorporates into its baseline expectations.

### 3. Seasonal Factors
- Month-specific seasonality multiplier
- Holiday vs shoulder season context
- Category-specific timing patterns

**Example:**
> The December scheduling carries a favorable seasonal multiplier of 1.15, reflecting historically stronger demand for this category during this period — likely influenced by holiday proximity and heightened cultural activity.

### 4. SHAP-Based Driver Summary
- Top 3-5 feature contributions (if SHAP values available)
- Upward vs downward pressures on prediction
- Magnitude of contributions in index points
- Falls back to feature-based interpretation if SHAP unavailable

**Example:**
> Key upward drivers include strong public recognition signals, category-specific historical patterns, and favorable seasonal positioning, which collectively elevate the forecast by approximately 17.5 index points.

### 5. Board-Level Interpretation
- Ticket Index tier classification
- Calgary/Edmonton split forecasts
- Total ticket count projections
- Audience segment composition
- Marketing implications

**Example:**
> The resulting **Ticket Index of 120.0** places this production in the **exceptional demand** tier. Translated into actionable planning figures, the model forecasts approximately **4,200 tickets in Calgary** and **2,800 tickets in Edmonton**, totaling 7,000 single-ticket sales.

## Technical Implementation

### Function Signature

```python
def build_title_explanation(
    title_metadata: Dict[str, Any],
    prediction_outputs: Optional[Dict[str, Any]] = None,
    shap_values: Optional[Dict[str, float]] = None,
    *,
    style: str = "board"
) -> str
```

### Required Metadata Fields

**Minimum required:**
- `Title`: str
- `Month`: str
- `Category`: str
- `TicketIndex used`: float

**Recommended for full narratives:**
- `Familiarity`: float (0-100+ scale)
- `Motivation`: float (0-100+ scale)
- `SignalOnly`: float (combined signal)
- `FutureSeasonalityFactor`: float
- `PrimarySegment`: str
- `SecondarySegment`: str
- `YYC_Singles`: int
- `YEG_Singles`: int
- `ReturnDecayPct`: float
- `IsRemount`: bool
- `YearsSinceLastRun`: int

### SHAP Value Integration

When SHAP values are provided, the engine translates technical feature names into human-readable explanations:

| Feature Pattern | Readable Description |
|----------------|---------------------|
| `familiarity`, `wiki` | "strong public recognition signals" |
| `motivation`, `youtube` | "elevated engagement indicators" |
| `season`, `month` | "favorable seasonal positioning" |
| `remount`, `years_since` | "remount timing dynamics" |
| `category` | "category-specific historical patterns" |

| `prior`, `median` | "strong historical precedent" |

### Signal Level Interpretation

| Score Range | Description |
|------------|-------------|
| 120+ | exceptionally high |
| 100-119 | strong |
| 80-99 | above average |
| 60-79 | moderate |
| 40-59 | emerging |
| 0-39 | limited |

### Ticket Index Tiers

| Index Range | Demand Tier |
|------------|-------------|
| 120+ | exceptional demand |
| 105-119 | strong demand |
| 95-104 | benchmark demand |
| 80-94 | moderate demand |
| 60-79 | developing demand |
| 0-59 | emerging demand |

## Season Report Introduction

The enhanced `_plain_language_overview_text()` provides an 8-paragraph executive introduction that covers:

1. **System Purpose** - What the Title Scoring App is and what questions it answers
2. **Familiarity & Motivation** - How digital signals are measured and combined
3. **Ticket Index Translation** - How visibility converts to demand forecasts
4. **Seasonality & Category** - How temporal and genre patterns influence predictions
5. **Premiere vs Remount** - How the model distinguishes and handles each
6. **SHAP Explainability** - How feature attributions provide transparency
7. **Summary & Scope** - Overall methodology and appropriate use

This introduction is derived from the canonical Technical Report Prose Style document and provides board-level clarity on the entire forecasting methodology.

## Usage Examples

### Basic Usage

```python
from ml.title_explanation_engine import build_title_explanation

metadata = {
    "Title": "Swan Lake",
    "Month": "December 2025",
    "Category": "adult_classic",
    "Familiarity": 135.0,
    "Motivation": 115.0,
    "TicketIndex used": 120.0,
    "FutureSeasonalityFactor": 1.15,
    "YYC_Singles": 4200,
    "YEG_Singles": 2800,
}

narrative = build_title_explanation(metadata)
print(narrative)
```

### With SHAP Values

```python
metadata = {...}  # As above

shap_values = {
    "familiarity_score": 8.5,
    "motivation_score": -2.3,
    "seasonality_factor": 3.2,
    "category_adult_classic": 5.2,
}

narrative = build_title_explanation(
    title_metadata=metadata,
    shap_values=shap_values
)
```

### Integration in PDF Generation

```python
from streamlit_app import build_full_pdf_report, _methodology_glossary_text

# The engine is automatically used by _narrative_for_row()
pdf_bytes = build_full_pdf_report(
    methodology_paragraphs=_methodology_glossary_text(),
    plan_df=season_plan_dataframe,
    season_year=2025,
    org_name="Alberta Ballet"
)
```

## Testing

Run the test suite to validate narrative generation:

```bash
# Unit tests
python -m pytest tests/test_title_explanation_engine.py -v

# Integration tests
python -m pytest tests/test_pdf_generation_with_narratives.py -v
```

## Word Count Targets

- **Target:** 250-350 words per narrative
- **Actual:** ~230-240 words (efficient while comprehensive)
- **Minimum:** 200 words (with minimal metadata)
- **Maximum:** ~400 words (with full SHAP analysis)

## Scalability

The engine is designed to scale to 300+ titles:

- **Deterministic:** Same inputs always produce same outputs
- **Programmatic:** No hardcoded title-specific logic
- **Efficient:** Generates narratives in milliseconds
- **Maintainable:** Centralized in single module with clear structure

## Future Enhancements

Potential improvements for future iterations:

1. **Live SHAP Computation:** Compute SHAP values on-demand during scoring
2. **Template Variants:** Add "technical" and "executive" style options
3. **Localization:** Support for French-language narratives
4. **Dynamic Length:** Adjust narrative length based on available metadata richness
5. **Citation Linking:** Link specific claims to source data files
6. **Comparative Analysis:** Include peer title comparisons in narratives

## Maintenance Notes

### Updating Narrative Patterns

To modify narrative structure, edit functions in `ml/title_explanation_engine.py`:

- `build_title_explanation()` - Main structure and paragraph assembly
- `_describe_signal_level()` - Signal score qualitative descriptors
- `_describe_category()` - Category name translations
- `_interpret_ticket_index()` - Ticket Index tier labels
- `_identify_shap_drivers()` - SHAP-to-prose translation

### Updating Season Report Introduction

To modify the PDF introduction, edit `_plain_language_overview_text()` in `streamlit_app.py`. The structure follows the Technical Report Prose Style document.

### Testing Changes

Always run the test suite after modifications:

```bash
python -m pytest tests/test_title_explanation_engine.py -v -s
```

Add new test cases for new features or edge cases.

## References

- **Technical Authority:** `docs/Alberta_Ballet_Technical_Report_Prose_Style.docx`
- **SHAP Documentation:** `scripts/train_safe_model.py` (--save-shap flag)
- **PDF Generation:** `streamlit_app.py` (build_full_pdf_report function)
- **Feature Engineering:** `ml/dataset.py`, `features/` modules

---

**Author:** Alberta Ballet Data Science Team  
**Last Updated:** December 2025  
**Version:** 1.0
