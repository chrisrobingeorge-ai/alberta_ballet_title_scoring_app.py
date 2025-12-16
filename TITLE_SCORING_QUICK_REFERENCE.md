# Title Scoring Helper - Quick Reference Card

## 🚀 Quick Start

```bash
streamlit run title_scoring_helper.py
```

## 📝 Basic Steps

1. **Enter titles** (one per line)
2. **Choose settings**:
   - Normalization: **Reference-based** (recommended)
   - Confidence: **80%** (typical)
   - Genre: Select appropriate genre
   - Season: Select target season
3. **Click "Fetch & Normalize Scores"**
4. **Review results** in Steps 2 & 3
5. **Download CSV** if needed

## 🎯 Understanding the Scores

### Step 2: Signals (0-100 scale)

| Signal | What it measures | Higher = |
|--------|-----------------|----------|
| **wiki** | Wikipedia pageviews | More awareness/interest |
| **trends** | Google search interest | Current demand |
| **youtube** | Video views | Visual appeal |
| **chartmetric** | Music popularity | Soundtrack recognition |

### Step 3: Forecasts

| Column | Meaning |
|--------|---------|
| **tickets_forecast_mean** | Expected ticket sales (middle estimate) |
| **tickets_forecast_lower_XX** | Lower bound (XX% confidence) |
| **tickets_forecast_upper_XX** | Upper bound (XX% confidence) |

**Example interpretation:**
- Mean: 8,234 | Lower 80%: 7,120 | Upper 80%: 9,348
- ➡️ 80% chance sales will be between 7,120 and 9,348 tickets

## ⚙️ Key Settings

### Normalization Methods

| Method | Use When |
|--------|----------|
| **Reference-based** | Production use (consistent with baselines.csv) |
| Batch-relative | Quick comparisons only (legacy) |

**Always use Reference-based for real planning!**

### Confidence Levels

| Level | Use For |
|-------|---------|
| 80% | Standard planning |
| 90% | Conservative estimates |
| 95% | High-risk scenarios |

## 💡 Pro Tips

✅ **DO**
- Use exact, official title names
- Score multiple titles to compare
- Use reference-based normalization
- Consider confidence intervals in planning
- Document your assumptions

❌ **DON'T**
- Mix batch-relative scores with baselines.csv
- Ignore wide confidence intervals
- Score only once (signals change over time)
- Forget to select appropriate genre

## 🔑 Optional: API Keys

Add in sidebar for better accuracy (optional):

- **YouTube**: [Google Cloud Console](https://console.cloud.google.com/)
- **chartmetric**: [Developer Dashboard](https://developer.chartmetric.com/dashboard)

Without keys? App uses fallback values.

## 📊 Common Use Cases

### Single Title Evaluation
1. Enter one title
2. Review mean forecast
3. If > 8,000 = likely strong performer

### Season Planning
1. Enter 3-5 candidate titles
2. Compare forecasts side-by-side
3. Consider both means and intervals

### Adding to Baselines
1. Score the title
2. Copy normalized scores (Step 2)
3. Add to `data/productions/baselines.csv`
4. See [ADDING_BASELINE_TITLES.md](ADDING_BASELINE_TITLES.md)

## 🔧 Troubleshooting

| Problem | Quick Fix |
|---------|-----------|
| Can't load baselines.csv | Use batch-relative normalization OR check file exists |
| Low scores for known title | Try exact Wikipedia page name |
| Very wide intervals | High uncertainty - use lower bound for planning |
| Forecasts seem off | Check genre selection, try reference-based normalization |

## 📚 Full Documentation

- **Complete guide**: [TITLE_SCORING_HELPER_USAGE.md](TITLE_SCORING_HELPER_USAGE.md)
- **Adding baselines**: [ADDING_BASELINE_TITLES.md](ADDING_BASELINE_TITLES.md)
- **Scoring fix**: [SCORING_FIX_QUICK_GUIDE.md](SCORING_FIX_QUICK_GUIDE.md)
- **Main README**: [README.md](README.md)

---

*Keep this card handy when scoring titles!*
