# Title Scoring Helper - Usage Guide

> 🚀 **Quick Reference**: See [TITLE_SCORING_QUICK_REFERENCE.md](TITLE_SCORING_QUICK_REFERENCE.md) for a printable one-page reference card.

## What is the Title Scoring Helper?

The Title Scoring Helper (`title_scoring_helper.py`) is a **Streamlit web application** that helps you score new ballet titles on their own without needing historical ticket sales data. It fetches popularity signals from external sources (Wikipedia, Google Trends, YouTube, chartmetric) and provides ticket demand forecasts.

## Quick Start: Scoring a Single Title

### 1. Launch the App

```bash
streamlit run title_scoring_helper.py
```

This will open the Title Scoring Helper in your web browser (usually at `http://localhost:8501`).

### 2. Enter Your Title

In the text area labeled **"Paste one title per line:"**, enter the title you want to score. For example:

```
Giselle
```

### 3. (Optional) Configure API Keys

For more accurate data, you can provide API keys in the sidebar:

- **YouTube Data API v3 Key** - For video view counts
- **chartmetric Client ID & Secret** - For music popularity scores

> **Note**: API keys are optional. If not provided, the app will use fallback values.

<details>
<summary>📖 How to get API keys (click to expand)</summary>

#### YouTube Data API Key
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable "YouTube Data API v3"
4. Go to "Credentials" → "Create Credentials" → "API Key"
5. Copy the key and paste it in the sidebar

#### chartmetric Client ID & Secret
1. Go to [chartmetric Developer Dashboard](https://developer.chartmetric.com/dashboard)
2. Log in with your chartmetric account
3. Click "Create an App"
4. Copy the Client ID and Client Secret
5. Paste both in the sidebar

</details>

### 4. Choose Settings

Before fetching scores, configure these options:

| Setting | Options | Recommendation |
|---------|---------|----------------|
| **Normalization method** | Reference-based (default) or Batch-relative | Use **Reference-based** to match baselines.csv |
| **Forecast interval confidence** | 80%, 90%, or 95% | Use **80%** for typical forecasts |
| **Default genre** | classical, contemporary, family, mixed | Select the most appropriate genre |
| **Season label** | 2024-25, 2025-26, 2026-27 | Select the target season |

### 5. Fetch & Normalize Scores

Click the **"Fetch & Normalize Scores"** button. The app will:

1. Search for the title on Wikipedia, Google Trends, YouTube, and chartmetric
2. Fetch popularity metrics from each source
3. Normalize all scores to a 0-100 scale
4. Display the results in Step 2

### 6. Review the Results

#### Step 2: Normalized Signals (0–100)

You'll see a table showing normalized scores for each signal:

| index | title | wiki | trends | youtube | chartmetric |
|-------|-------|------|--------|---------|---------|
| 1 | Giselle | 75.2 | 42.8 | 68.9 | 55.3 |

**What do these scores mean?**
- **wiki**: Wikipedia pageviews (popularity/awareness)
- **trends**: Google search interest (current demand)
- **youtube**: Video views (visual appeal/familiarity)
- **chartmetric**: Music popularity (soundtrack recognition)

Higher scores (closer to 100) indicate stronger demand signals.

#### Step 3: Ticket Forecasts

The app provides ticket demand forecasts with uncertainty bands:

| index | title | tickets_forecast_mean | tickets_forecast_lower_80 | tickets_forecast_upper_80 |
|-------|-------|----------------------|---------------------------|---------------------------|
| 1 | Giselle | 8,234 | 7,120 | 9,348 |

**Understanding the forecast:**
- **tickets_forecast_mean**: Expected ticket sales (middle estimate)
- **tickets_forecast_lower_80**: Lower bound (80% confidence interval)
- **tickets_forecast_upper_80**: Upper bound (80% confidence interval)

> **Interpretation**: There's an 80% chance that ticket sales will fall between the lower and upper bounds.

### 7. Download Results

Click **"Download Forecast CSV"** to save the results for later use or to import into other tools.

---

## Scoring Multiple Titles

You can score multiple titles at once by entering each on a new line:

```
Giselle
Swan Lake
The Nutcracker
Cinderella
```

The app will fetch signals for all titles and provide comparative scores. This is useful for:
- Comparing multiple programming options
- Building a season lineup
- Evaluating title portfolio diversity

---

## Understanding Normalization Methods

The Title Scoring Helper offers two normalization methods:

### Reference-based (Recommended)

- **How it works**: Normalizes scores against the full distribution in `baselines.csv`
- **Benefits**: 
  - Consistent with existing baseline scores
  - Same title always gets the same score
  - Scores are stable as you add more titles
- **Use case**: **Production use - always use this for real planning**

### Batch-relative (Legacy)

- **How it works**: Normalizes scores only within the current batch of titles
- **Benefits**: 
  - Shows relative ranking within your batch
  - Useful for quick comparisons
- **Use case**: Legacy compatibility only; not recommended for production

**Example of the difference:**

| Title | Reference-based | Batch-relative |
|-------|----------------|----------------|
| The Nutcracker (alone) | wiki: 95 | wiki: 50 |
| The Nutcracker (with Swan Lake) | wiki: 95 | wiki: 100 |

Notice how reference-based gives consistent scores regardless of what else you're scoring.

---

## Common Use Cases

### Use Case 1: Evaluating a New Title Proposal

**Scenario**: A choreographer proposes "Romeo and Juliet" and you want to assess demand.

**Steps**:
1. Run `streamlit run title_scoring_helper.py`
2. Enter "Romeo and Juliet"
3. Use **Reference-based** normalization
4. Review the forecast: If mean > 8,000, it's likely a strong performer

### Use Case 2: Comparing Season Options

**Scenario**: You're deciding between three titles for your winter season.

**Steps**:
1. Enter all three titles (one per line)
2. Use **Reference-based** normalization  
3. Compare the forecasts side-by-side
4. Consider both the mean forecast and the uncertainty (upper/lower bounds)

### Use Case 3: Adding to Baselines

**Scenario**: You want to add a new reference title to `baselines.csv`.

**Steps**:
1. Score the title using the helper
2. Copy the normalized scores (wiki, trends, youtube, chartmetric)
3. Add a row to `data/productions/baselines.csv`:
   ```csv
   Title Name,75,42,69,55,category,gender,external_reference,"Notes"
   ```
4. See [ADDING_BASELINE_TITLES.md](ADDING_BASELINE_TITLES.md) for full details

---

## Troubleshooting

### Problem: "Could not load baselines.csv"

**Solution**: Make sure `data/productions/baselines.csv` exists and is properly formatted. If you're using reference-based normalization, the app needs this file.

**Workaround**: Use batch-relative normalization (legacy) if baselines.csv is unavailable.

### Problem: Low/zero scores for a well-known title

**Possible causes**:
- API rate limits reached
- Title name doesn't match Wikipedia/chartmetric entries exactly
- Network connectivity issues

**Solutions**:
1. Try using the exact Wikipedia page title (e.g., "Swan Lake (ballet)")
2. Check your internet connection
3. Try again after a few minutes (rate limits may have reset)
4. Enter API keys if using fallback values

### Problem: Very wide forecast intervals

**Explanation**: Wide intervals (e.g., 5,000 - 15,000) indicate high uncertainty in the prediction.

**Causes**:
- New title with no similar historical precedent
- Mixed signals (some high, some low)
- Limited training data for that category

**What to do**: Use the lower bound for conservative planning, or gather more information about audience interest.

### Problem: Forecasts seem too high/low

**Check these factors**:
1. **Genre selection**: Is the genre appropriate? (family vs. contemporary makes a big difference)
2. **Normalization method**: Are you using reference-based?
3. **API data quality**: Did the APIs return good data, or are you using fallbacks?
4. **Model calibration**: The model may need recalibration with recent data

---

## Best Practices

### 1. Use Consistent Title Names

Use the official or most common title name to get the best API matches:
- ✅ "The Nutcracker" (standard)
- ❌ "Nutcracker" (may get different results)

### 2. Score Reference Titles Regularly

If you're building up your `baselines.csv`, score reference titles every 6-12 months as popularity signals change over time (especially Google Trends).

### 3. Document Your Assumptions

When using forecasts for planning, note:
- Which genre you selected
- The season timing
- The confidence level used
- Any special circumstances (e.g., celebrity guest artist)

### 4. Validate with Historical Data

For titles you've performed before, compare:
- Helper forecast vs. actual sales
- Signal scores vs. previous seasons

This helps you build intuition for how the model performs.

### 5. Use Forecasts as One Input

The Title Scoring Helper provides data-driven estimates, but always consider:
- Artistic vision and mission
- Audience feedback and engagement
- Marketing potential
- Venue constraints
- Seasonal programming balance

---

## Technical Details

### What Happens Behind the Scenes?

1. **Signal Fetching**:
   - Wikipedia: Fetches pageviews over the past 365 days
   - Google Trends: Gets search interest over the past 12 months
   - YouTube: Searches for the title and gets view count of top result
   - chartmetric: Searches for associated tracks and gets popularity score

2. **Normalization**:
   - Reference-based: Uses min/max from `baselines.csv` for each signal
   - Batch-relative: Uses min/max from current batch only

3. **Feature Engineering**:
   - Combines signals with genre and season
   - Passes features to trained constrained Ridge regression model

4. **Forecasting**:
   - Model generates point prediction
   - Bootstrap sampling creates uncertainty intervals
   - Returns mean, lower bound, and upper bound

### Data Sources

| Source | API Docs | Rate Limits |
|--------|----------|-------------|
| Wikipedia | [Wikimedia REST API](https://wikimedia.org/api/rest_v1/) | ~200 req/s (generous) |
| Google Trends | [pytrends](https://github.com/GeneralMills/pytrends) | ~1 req/s (unofficial) |
| YouTube | [YouTube Data API](https://developers.google.com/youtube/v3) | 10,000 units/day (free) |
| chartmetric | [Web API](https://developer.chartmetric.com/documentation/web-api) | Varies by tier |

---

## FAQ

### Q: Can I use this without API keys?

**A**: Yes! The app works without API keys, but it will use fallback values for YouTube and chartmetric, which may be less accurate. Wikipedia and Google Trends don't require keys.

### Q: How accurate are the forecasts?

**A**: Accuracy depends on:
- Model training quality (see `ML_MODEL_DOCUMENTATION.md`)
- Title similarity to historical data
- Signal quality from APIs

Typical accuracy: ±15-25% (within the confidence interval bounds).

### Q: Can I score titles that aren't ballets?

**A**: The model is trained on ballet titles, but you can score any performance title. Results may be less accurate for non-ballet content.

### Q: What if I don't know the genre?

**A**: Start with "mixed" as a default. You can always refine the genre later in your season planning CSV.

### Q: How do I integrate this with the main app?

**A**: 
1. Score your titles using the helper
2. Add them to `data/productions/baselines.csv` (see [ADDING_BASELINE_TITLES.md](ADDING_BASELINE_TITLES.md))
3. Use the main app (`streamlit run streamlit_app.py`) for full season planning

### Q: Can I score titles programmatically (without the UI)?

**A**: Yes, but you'd need to call the functions directly:

```python
from title_scoring_helper import (
    fetch_wikipedia_views,
    fetch_google_trends_score,
    fetch_youtube_metric,
    fetch_chartmetric_metric,
    normalize_with_reference,
)
from ml.scoring import score_runs_for_planning
import pandas as pd

# Fetch signals
title = "Giselle"
signals = {
    "wiki": fetch_wikipedia_views(title),
    "trends": fetch_google_trends_score(title),
    "youtube": fetch_youtube_metric(title),
    "chartmetric": fetch_chartmetric_metric(title),
}

# Normalize (you'd need to implement reference loading)
# ... normalization code ...

# Score
df = pd.DataFrame([{
    "wiki": signals["wiki"],
    "trends": signals["trends"],
    "youtube": signals["youtube"],
    "chartmetric": signals["chartmetric"],
    "genre": "classical",
    "season": "2025-26",
}])
forecast = score_runs_for_planning(df)
print(forecast)
```

---

## Related Documentation

- [ADDING_BASELINE_TITLES.md](ADDING_BASELINE_TITLES.md) - How to add scored titles to baselines.csv
- [SCORING_FIX_QUICK_GUIDE.md](SCORING_FIX_QUICK_GUIDE.md) - Understanding normalization methods
- [README.md](README.md) - Main app documentation
- [ML_MODEL_DOCUMENTATION.md](ML_MODEL_DOCUMENTATION.md) - Model technical details

---

## Getting Help

If you encounter issues:

1. Check this guide first
2. Review [SCORING_FIX_QUICK_GUIDE.md](SCORING_FIX_QUICK_GUIDE.md) for scoring-specific issues
3. Check the main [README.md](README.md) for general setup help
4. Verify your Python environment has all required packages: `pip install -r requirements.txt`

---

*Last updated: December 2024*
