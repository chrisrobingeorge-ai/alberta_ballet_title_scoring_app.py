# Legacy Scripts

⚠️ **The scripts in this directory are DEPRECATED and should NOT be used for production training or data processing.**

## Why These Scripts Are Deprecated

These scripts were part of the original development workflow but have been superseded by safer, more robust alternatives in the main codebase.

### `train_pycaret_model.py`

**Problems:**
- Uses current-run ticket columns directly as features AND target (data leakage)
- Creates circular dependencies where the model learns to predict from information unavailable at forecast time
- Resulting models cannot generalize to new shows without history

**Safe Alternative:**
```bash
# Step 1: Build leak-free dataset
python scripts/build_modelling_dataset.py

# Step 2: Train with proper cross-validation
python scripts/train_safe_model.py --tune --save-shap
```

### `build_city_priors.py`

**Reason for Deprecation:**
- This was a one-time utility to generate city prior dictionaries
- The current application learns city priors dynamically from `history_city_sales.csv` at runtime
- See `learn_priors_from_history()` in `streamlit_app.py`

## When to Reference These Scripts

These scripts may still be useful for:
- Understanding historical approaches
- Reference for data column expectations
- Debugging legacy model artifacts

## Do Not Use These Scripts For

- ❌ Production model training
- ❌ Generating production-ready priors
- ❌ Any pipeline that feeds into live forecasts
