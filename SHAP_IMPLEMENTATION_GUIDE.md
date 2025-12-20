# SHAP (SHapley Additive exPlanations) Implementation Guide

## Executive Summary

SHAP would transform the app from answering **"What's the prediction?"** to answering **"Why is it that prediction?"**

Currently missing: **Per-prediction explanations** showing which factors pushed the prediction up/down and by how much.

---

## What SHAP Does (With Examples)

### Current App Explains:
```
Title: Swan Lake
Prediction: 185 tickets

Feature importance (global, static):
  • Prior tickets: 96.9%
  • Wiki signal: 2.5%
  • Google Trends: 0.5%
```

**Problem:** This is global. Doesn't explain *this specific prediction*.

---

### SHAP Would Show:
```
Title: Swan Lake (Remount)
Prediction: 185 tickets

SHAP Decomposition (this specific show):
  • Base value (global average): 140 tickets
  • Prior sales of Swan Lake: +58 tickets ← MAJOR DRIVER
  • Wiki search trend: +15 tickets
  • Google Trends: -8 tickets
  • YouTube signal: +2 tickets
  ─────────────────────────
  Final prediction: 185 tickets

Why Swan Lake gets more than average:
  ✓ Strong historical sales (97% of the boost)
  ✓ Good recent wiki interest (+15)
  ✓ BUT slightly declining Google Trends (-8)
```

---

## Real-World Value Example

### For Board/Management:
```
User asks: "Why are we predicting only 95 tickets for Nutcracker?"

Current answer:
  "Model says 95. Feature importance shows prior sales matter most."
  → Leaves decision-maker confused

SHAP answer:
  "95 tickets because:
   • Nutcracker's historical baseline: 150 tickets
   • BUT wiki searches DOWN 22%: -35 tickets
   • AND Google Trends DOWN 15%: -12 tickets
   • AND YouTube uploads UP only slightly: +2 tickets
   
   The prediction dropped because signals are weak this year.
   Should we investigate Nutcracker's marketing?"
```

**Business value:** Decision-makers can see *why* predictions dropped and take action.

---

## What SHAP Actually Computes

SHAP uses **Shapley values** from game theory:
- Treats each feature as a "player" in a game
- Calculates how much each player contributed to the final score
- Accounts for interactions between features
- Produces fair, game-theory-backed explanations

### The Math (Simple Version):
```
Prediction = Base Value + SHAP(Feature₁) + SHAP(Feature₂) + ... + SHAP(Featurₙ)

Example for Swan Lake:
185 = 140 + 58 + 15 + (-8) + 2
     (base) (pri) (wiki)(trends)(youtube)
```

### Why It's Better Than Naive Importance:
- **Naive:** "Prior tickets is 97% important globally"
- **SHAP:** "For *this* show, prior tickets contributed +58, which is 31% of the boost above baseline"

Same feature, different explanations depending on the actual data!

---

## Current State of Implementation

### What Exists:
```
ml/title_explanation_engine.py - 52 lines
  function: build_title_explanation()
  inputs: explanation_data (ALWAYS None), model, features
  output: narrative string
  status: NEVER CALLED WITH REAL DATA
```

### What's Missing:
1. **SHAP Explainer** - Never instantiated
2. **SHAP Computation** - Never calculated
3. **Narrative Generation** - Template exists, never filled
4. **UI Integration** - Partial (shows None values)

---

## Implementation Effort & Timeline

### Phase 1: Core SHAP (2-3 hours)
**What:** Basic SHAP computation for Ridge regression models

```python
# What needs to be added:
import shap

# In prediction pipeline (streamlit_app.py):
explainer = shap.Explainer(ridge_model, X_train_sample)
shap_values = explainer.explain_single_prediction(X_test)

# Result: Array of SHAP values (one per feature)
```

**Effort:**
- Install `shap` package (~2 min)
- Create SHAP explainer object (~5 min)
- Compute SHAP values for each prediction (~15 min)
- Test and validate (~15 min)

**Code complexity:** LOW (SHAP library handles complexity)

---

### Phase 2: Narrative Generation (1-2 hours)
**What:** Turn SHAP values into human-readable explanations

```python
# Current template (empty):
def build_title_explanation(explanation_data, model, features):
    if explanation_data is None:
        return ""
    
    # Need to add logic here:
    # 1. Get SHAP values from explanation_data
    # 2. Rank features by impact
    # 3. Generate sentences like:
    #    "Prior sales pushed prediction UP by +58 tickets"
    #    "Wiki trends added +15 tickets"
    #    "Google Trends declined, pulling DOWN 8 tickets"
    # 4. Highlight top 3-5 drivers
```

**Effort:**
- Design explanation format (~15 min)
- Implement narrative generation (~30 min)
- Add formatters for human readability (~20 min)
- Test with various shows (~20 min)

**Code complexity:** MEDIUM (string formatting, conditional logic)

---

### Phase 3: UI Integration (1-2 hours)
**What:** Display explanations in the Streamlit app

```python
# Current code (streamlit_app.py:726-740):
# Shows placeholders - needs to show real SHAP explanations

# Need to add:
# 1. Expander for "Show Explanation"
# 2. Display top 3-5 SHAP factors with visualizations
# 3. Optional: SHAP force plot (waterfall visualization)
# 4. Optional: Feature importance distribution
```

**Effort:**
- Create UI components (~20 min)
- Format SHAP output for display (~20 min)
- Add visualization (optional, uses `shap.plots`) (~30 min)
- Polish and test (~20 min)

**Code complexity:** LOW-MEDIUM (Streamlit handles UI)

---

### Phase 4: Performance Optimization (optional, 1-2 hours)
**What:** Make SHAP computation fast enough for batch predictions

**Current bottleneck:**
- SHAP requires model evaluations (expensive for large batches)
- Computing SHAP for 100 shows = 100 × (# feature combinations) model calls

**Solutions:**
1. **Caching** (~15 min): Cache SHAP values for baseline titles (never changes)
2. **Sampling** (~20 min): Use approximate SHAP (faster, slightly less accurate)
3. **Batch computation** (~30 min): Compute all SHAP values at once

**Code complexity:** MEDIUM-HIGH

---

## Total Effort Estimate

| Phase | Effort | Complexity | Value |
|-------|--------|-----------|-------|
| Core SHAP | 1-2 hours | LOW | HIGH |
| Narrative | 1-1.5 hours | MEDIUM | HIGH |
| UI | 1-2 hours | LOW-MEDIUM | MEDIUM |
| Optimization | 1-2 hours | MEDIUM-HIGH | MEDIUM |
| **TOTAL** | **4-7.5 hours** | **LOW-MEDIUM** | **VERY HIGH** |

---

## Current Code State

### file: `ml/title_explanation_engine.py`
```python
def build_title_explanation(explanation_data, model, features):
    """
    Build a narrative explanation from SHAP-style explanation data.
    
    Args:
        explanation_data: SHAP values (currently always None)
        model: The prediction model
        features: Feature names
    
    Returns:
        String narrative explanation
    """
    if explanation_data is None:
        return ""  # ← This is why it's non-functional
    
    # Logic here would convert SHAP values to narrative
    # Never reached because explanation_data is always None
```

### file: `streamlit_app.py` (lines 726-740)
```python
try:
    from ml.title_explanation_engine import build_title_explanation
    narrative = build_title_explanation(
        explanation_data=None,  # ← ALWAYS NONE (problem!)
        model=final_model,
        features=feature_names
    )
except Exception:
    narrative = ""
```

---

## Dependencies Needed

**Currently:** None (title_explanation_engine.py has zero external deps)

**For SHAP:** Only one new package
```bash
pip install shap==0.45.0  # ~50 MB download, adds to requirements.txt
```

**Trade-off:**
- **Pro:** Powerful explanations, proven method
- **Con:** Adds dependency, slightly slower (15-20ms per prediction)

---

## Alternative Approaches (Not Recommended)

### 1. LIME (Local Interpretable Model-agnostic Explanations)
- Simpler than SHAP
- Less accurate for this use case
- Not worth the trade-off

### 2. Manual Feature Attribution
- Calculate each feature's contribution manually
- Very fast, but less theoretically sound
- Would reinvent SHAP poorly

### 3. Keep Current Approach
- Show global feature importance only
- No per-prediction explanations
- Users won't understand "why this show"

---

## Recommended Implementation Path

### Immediate (Next Sprint):
1. Install `shap` package
2. Implement core SHAP computation in Ridge pipeline
3. Test with 5-10 predictions
4. Validate explanations make sense

### Soon (Sprint After):
1. Implement narrative generation
2. Add UI expanders for explanations
3. Test with board/stakeholders
4. Gather feedback

### Later (Performance Phase):
1. Add caching for baselines
2. Implement batch SHAP computation
3. Optimize for speed
4. Monitor CPU impact

---

## Success Criteria

✅ **Done when:**
1. SHAP values computed correctly (validate with manual examples)
2. Explanations are human-readable
3. < 50ms per prediction latency
4. Users say "Now I understand why"
5. Board presentations cite actual SHAP decompositions

---

## Code Template (To Get Started)

```python
# Add to streamlit_app.py

import shap
import numpy as np

# In your prediction function:
def predict_with_shap(ridge_model, X_train, X_test_point):
    """Make prediction with SHAP explanations."""
    
    # 1. Create explainer (one-time, cache this)
    explainer = shap.Explainer(ridge_model, X_train)
    
    # 2. Compute SHAP values for this prediction
    shap_values = explainer(X_test_point)
    
    # 3. Get prediction
    prediction = ridge_model.predict(X_test_point)[0]
    
    # 4. Get base value (model output on empty input)
    base_value = explainer.expected_value
    
    # 5. Return structured explanation
    return {
        'prediction': prediction,
        'base_value': base_value,
        'shap_values': shap_values.values[0],
        'features': X_test_point.columns,
        'feature_values': X_test_point.iloc[0].values
    }

# Usage:
explanation = predict_with_shap(model, X_train, X_test)

# Generate narrative:
narrative = build_title_explanation(
    explanation_data=explanation,  # ← NOW WITH REAL DATA!
    model=model,
    features=explanation['features']
)
```

---

## Why SHAP Is Worth Doing

| Current | With SHAP |
|---------|-----------|
| "Prediction: 185" | "185 because prior sales (+58), wiki (+15), trends (-8)" |
| "Global importance matters" | "For THIS show, here's what drove it" |
| Black box | Clear reasoning |
| Hard to debug | Easy to audit |
| Users guess why | Users understand why |
| Board says "How do you know?" | Board sees the breakdown |

---

## Questions Before Starting?

1. **Performance:** Is 15-20ms per prediction acceptable?
2. **Scope:** All predictions or just in reports?
3. **Visualization:** Just text or include SHAP plots?
4. **Baselines:** Include SHAP for baseline titles too?
5. **Timeline:** When do you need this?

---

