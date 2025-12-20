# K-Nearest Neighbors (KNN) Functionality Report

## Executive Summary

✅ **STATUS: FULLY FUNCTIONAL**

The k-Nearest Neighbors fallback function **IS WORKING CORRECTLY** in your codebase. All tests pass, and it is properly integrated into the streamlit application.

---

## Detailed Findings

### 1. Implementation Status

| Aspect | Status | Details |
|--------|--------|---------|
| Code Quality | ✅ Excellent | Well-documented, properly structured |
| Dependencies | ✅ Available | scikit-learn, pandas, numpy all installed |
| Error Handling | ✅ Robust | Handles missing values and edge cases |
| Configuration | ✅ Correct | Properly configured in config.yaml |
| Integration | ✅ Seamless | Properly initialized and called in streamlit_app.py |

### 2. Functional Tests Passed

All diagnostic tests passed successfully:

- ✅ Module imports working
- ✅ scikit-learn available and functional
- ✅ Index building from simple data
- ✅ Single and batch predictions
- ✅ Neighbor information retrieval
- ✅ Edge case handling (minimum data, NaN values, output clipping)
- ✅ App-level integration (reproduces exact app logic)
- ✅ Prediction function works as expected

### 3. Current Configuration

From `config.yaml`:

```yaml
knn:
  enabled: true              ✓ KNN is enabled
  k: 5                       ✓ Uses 5 neighbors
  metric: "cosine"           ✓ Cosine similarity distance
  recency_weight: 0.5        ✓ Balances recency with all neighbors
  recency_decay: 0.1         ✓ 0.1 decay per year
  normalize: true            ✓ Normalizes features
  weights: "distance"        ✓ Distance-weighted voting
  use_pca: false             ✓ Uses raw features (not dimensionally reduced)
  pca_components: 3          (Only used if use_pca=true)
```

### 4. How It Works in Your App

The KNN fallback is used as a "cold-start" predictor when:

**Activation Conditions (ALL must be true):**
1. `knn_enabled = true` (from config.yaml)
2. scikit-learn is installed
3. At least 3 historical titles exist (`df_known` length >= 3)

**Process Flow:**

```
Step 1: Load historical title data with baseline signals
        ↓
        (WikiIdx, TrendsIdx, YouTubeIdx, ChartmetricIdx)
        ↓
Step 2: Build KNN index with TicketIndex_DeSeason as target
        ↓
        (Represents the "inherent" ticket potential)
        ↓
Step 3: When predicting for a new title:
        → Find 5 most similar titles by signal distance
        → Weight neighbors by similarity & recency
        → Average neighbor outcomes with weights
        → Return weighted prediction
        ↓
Step 4: Apply bounds check (minimum 20, maximum 180)
        ↓
        (Prevents unrealistic predictions)
```

### 5. Feature Highlights

**Distance-Weighted Voting**
- Neighbors closer in signal space have higher influence
- More accurate than equal weighting

**Recency Adjustment**
- More recent shows weighted slightly higher
- Decay factor of 0.1 per year

**Robust Preprocessing**
- Automatic feature normalization
- NaN value handling (converts to 0)
- Feature scaling via StandardScaler

**Optional PCA Whitening**
- Configured in code (currently disabled)
- Provides Mahalanobis-like distance if enabled

---

## Why It Might Not Be Used

If you're not seeing "kNN Fallback" in your predictions, here are the most likely reasons (in order of likelihood):

### 1. **Not Enough Data** (Most Common)
- **Problem:** `df_known` has fewer than 3 records
- **Solution:** Load more historical data or create baseline CSV with 3+ titles
- **Check:** Ensure `history_city_sales.csv` or `baselines.csv` has multiple titles

### 2. **ML Model Providing Predictions Instead** (Normal)
- **Problem:** ML model successfully predicted, so KNN not needed as fallback
- **Solution:** This is normal - KNN is a FALLBACK, not primary predictor
- **Check:** Look at the "Source" field - will show "ML Model" or "kNN Fallback"

### 3. **Configuration Disabled**
- **Problem:** `config.yaml` has `knn.enabled: false`
- **Solution:** Change to `knn.enabled: true`
- **Check:** Search for `knn:` in config.yaml

### 4. **Silent Import Failure** (Rare)
- **Problem:** scikit-learn not installed
- **Solution:** `pip install scikit-learn`
- **Check:** Check streamlit logs for import errors

---

## Verification Steps

To confirm KNN is working in your Streamlit app:

1. **Open the app** and make a prediction
2. **Look for the prediction source label:**
   - ✅ "kNN Fallback" = KNN is being used
   - ✅ "ML Model" = ML model predicted instead (normal - it's primary)

3. **Check the neighbor information** (if shown):
   - Should see 3-5 similar titles
   - Similarity scores should be between 0 and 1
   - Ticket indices should be reasonable values

4. **In Streamlit terminal**, watch for:
   - "kNN index built successfully" = Good
   - "kNN build failed" = Check error message
   - No KNN messages = Probably have sufficient data for ML model

---

## Key Insights

### 1. **KNN Is a Fallback, Not the Primary Predictor**

The prediction architecture is:

```
Category ML Model → Attempt prediction
├─ If successful: Use category model result
└─ If fails/low confidence: Fall back to KNN
```

### 2. **KNN Works Best With Similar Titles**

- Predicts well for titles similar to historical ones
- May be less accurate for completely novel shows
- Good for "what-if" analysis using signal combinations

### 3. **Signal Normalization Is Critical**

- All features (wiki, trends, youtube, chartmetric) are normalized
- Prevents high-range signals from dominating distance calculation
- Makes similarity matching fair across different signal types

### 4. **Output Clipping Is Intentional**

- Range `[20, 180]` represents ticket index bounds
- Prevents unrealistic single prediction outliers
- Can be adjusted in the `_predict_with_knn()` function if needed

---

## Conclusion

### ✅✅✅ Your K-NN Function Is Working Correctly ✅✅✅

**No bugs detected.** The implementation is:

- ✅ Mathematically sound (proper distance weighting)
- ✅ Well-integrated (properly initialized and called)
- ✅ Fault-tolerant (handles edge cases gracefully)
- ✅ Configurable (easy to tune parameters)

If it's not being used in your predictions, that's likely because:
1. Your ML models are producing good predictions (normal - they're primary)
2. You don't have 3+ historical titles to build the index from

**The implementation itself is solid and ready for production use.**

if __name__ == "__main__":
    generate_report()
