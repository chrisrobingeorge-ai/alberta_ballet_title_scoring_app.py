# Intent Disambiguation Correction - Deployment Checklist

## Pre-Deployment Validation

### ✅ Code Quality
- [x] Core module implemented (`ml/intent_disambiguation.py`)
- [x] Comprehensive test suite (`tests/test_intent_disambiguation.py`)
- [x] Integration examples (`examples/intent_disambiguation_integration.py`)
- [x] Validation script passes (`scripts/validate_intent_disambiguation.py`)
- [x] All tests pass (`pytest tests/test_intent_disambiguation.py -v`)
- [x] No linting errors
- [x] Code follows project conventions

### ✅ Documentation
- [x] Full documentation (`docs/INTENT_DISAMBIGUATION.md`)
- [x] Quick reference guide (`docs/INTENT_DISAMBIGUATION_QUICK_REF.md`)
- [x] Module README (`ml/INTENT_DISAMBIGUATION_README.md`)
- [x] Implementation summary (`IMPLEMENTATION_SUMMARY.md`)
- [x] Main README updated with feature listing
- [x] Deployment checklist (this file)

### ✅ Testing
- [x] Unit tests for all functions
- [x] Edge case coverage
- [x] Error handling validation
- [x] Mathematical precision tests
- [x] Batch processing tests
- [x] Integration tests
- [x] Demo scripts work correctly

### ✅ Examples & Demos
- [x] Built-in module demo works (`python -m ml.intent_disambiguation`)
- [x] Integration demo works (`python -m examples.intent_disambiguation_integration`)
- [x] Validation script passes (`python scripts/validate_intent_disambiguation.py`)

---

## Deployment Steps

### Step 1: Review & Approval
- [ ] Review documentation with stakeholders
- [ ] Validate penalty percentages with domain experts
- [ ] Review example outputs for accuracy
- [ ] Get approval from data science team lead
- [ ] Get approval from Alberta Ballet leadership

### Step 2: Historical Validation
- [ ] Load historical season data
- [ ] Apply corrections to past seasons
- [ ] Compare corrected forecasts to actual sales
- [ ] Calculate forecast error improvement (MAE, RMSE)
- [ ] Document validation results
- [ ] Review with stakeholders

### Step 3: Integration
- [ ] Identify integration points in production pipeline
- [ ] Update forecasting scripts to import module
- [ ] Apply corrections before report generation
- [ ] Update PDF report templates to show corrections
- [ ] Add correction flags to CSV exports
- [ ] Test end-to-end pipeline with corrections

### Step 4: Production Deployment
- [ ] Merge code to production branch
- [ ] Deploy to production environment
- [ ] Verify imports work in production
- [ ] Run validation script in production
- [ ] Monitor first production run
- [ ] Review generated reports

### Step 5: Documentation & Training
- [ ] Update internal documentation
- [ ] Create training materials for stakeholders
- [ ] Present to board/executive team
- [ ] Document known limitations
- [ ] Create FAQ for common questions

### Step 6: Monitoring & Maintenance
- [ ] Track correction frequency by category
- [ ] Monitor forecast accuracy post-deployment
- [ ] Review penalty calibration quarterly
- [ ] Document any adjustments made
- [ ] Schedule annual review of penalty values

---

## Integration Code Template

Add this to your production forecasting pipeline:

```python
# At the top of your script
from ml.intent_disambiguation import apply_intent_disambiguation

# When processing titles for a season
def process_title_forecast(title_metadata):
    """Process a single title with intent disambiguation correction."""
    
    # Step 1: Apply intent disambiguation correction
    corrected = apply_intent_disambiguation(
        title_metadata,
        apply_correction=True,  # Set to False to disable
        verbose=False  # Set to True for debugging
    )
    
    # Step 2: Update metadata with corrected values
    title_metadata.update({
        # Replace original values with corrected ones
        "Motivation": corrected["Motivation_corrected"],
        "TicketIndex used": corrected["TicketIndex_corrected"],
        "EstimatedTickets_Final": corrected["EstimatedTickets_corrected"],
        
        # Add correction metadata for transparency
        "IntentCorrectionApplied": corrected["IntentCorrectionApplied"],
        "MotivationPenaltyPct": corrected["Motivation_penalty_pct"] * 100,
        
        # Preserve original values for reference
        "Motivation_original": corrected["Motivation"],  # Original value
        "TicketIndex_original": corrected["TicketIndex used"],
        "EstimatedTickets_original": corrected["EstimatedTickets_Final"]
    })
    
    return title_metadata

# For batch processing
from ml.intent_disambiguation import batch_apply_corrections

def process_season_forecasts(season_titles):
    """Process entire season with batch corrections."""
    
    corrected_season = batch_apply_corrections(
        season_titles,
        apply_correction=True,
        verbose=False
    )
    
    return corrected_season
```

---

## Report Template Updates

### PDF Report Additions

Add this section to title explanations:

```python
# In your PDF generation code
if title_metadata["IntentCorrectionApplied"]:
    penalty_pct = title_metadata["Motivation_penalty_pct"] * 100
    
    note = (
        f"⚠️ <b>Intent Disambiguation Applied</b>: "
        f"Due to semantic ambiguity in search data for this title category "
        f"(<i>{title_metadata['Category']}</i>), a {penalty_pct:.0f}% correction "
        f"has been applied to adjust for non-ballet search intent. "
        f"Original estimate: {title_metadata['EstimatedTickets_original']:,} tickets. "
        f"Corrected estimate: {title_metadata['EstimatedTickets_Final']:,} tickets."
    )
```

### CSV Export Columns

Add these columns to your CSV exports:

```python
export_columns = [
    "Title",
    "Category",
    "EstimatedTickets_Final",  # Corrected value
    "EstimatedTickets_original",  # Original (uncorrected)
    "IntentCorrectionApplied",  # Boolean flag
    "MotivationPenaltyPct",  # Penalty percentage
    # ... other columns
]
```

---

## Rollback Plan

If issues arise after deployment:

### Quick Disable (No Code Changes)
```python
# Set apply_correction=False in all calls
corrected = apply_intent_disambiguation(
    title_metadata,
    apply_correction=False  # <-- Disable corrections
)
```

### Partial Rollback (Specific Categories)
```python
# Temporarily set penalties to 0 in ml/intent_disambiguation.py
CATEGORY_PENALTIES = {
    'family_classic': 0.0,  # Disabled
    'pop_ip': 0.0,  # Disabled
    # ... others
}
```

### Full Rollback
1. Remove import statements
2. Remove correction function calls
3. Restore original forecasting logic
4. Regenerate reports

---

## Success Metrics

Track these metrics post-deployment:

### Accuracy Metrics
- [ ] Mean Absolute Error (MAE) before/after
- [ ] Root Mean Squared Error (RMSE) before/after
- [ ] Forecast bias (over/under prediction)
- [ ] Category-specific accuracy

### Usage Metrics
- [ ] % of titles corrected per season
- [ ] Average correction magnitude
- [ ] Distribution by category

### Business Impact
- [ ] Board feedback on forecast accuracy
- [ ] Marketing team confidence in numbers
- [ ] Reduction in post-season forecast errors

---

## Known Limitations

Document these for stakeholders:

1. **Penalty Calibration**: Current penalties (10%, 20%) are based on initial analysis and may require adjustment
2. **Category Assignment**: Accuracy depends on correct category labeling
3. **Time Sensitivity**: Penalties don't account for recency of media releases (e.g., new movie adaptations)
4. **Platform Bias**: Based on Google/YouTube/Wikipedia data; may not generalize to other platforms

---

## Future Enhancements

Consider for future versions:

- [ ] **Dynamic Penalties**: Adjust based on real-time search trends
- [ ] **Title-Specific Overrides**: Custom penalties for specific titles (e.g., "The Nutcracker")
- [ ] **Time-Decay Adjustments**: Higher penalties near movie releases
- [ ] **Multi-Factor Corrections**: Combine with other bias corrections
- [ ] **Machine Learning**: Learn optimal penalties from historical data
- [ ] **A/B Testing Framework**: Systematic comparison of penalty levels

---

## Contact & Support

| Role | Contact | Purpose |
|------|---------|---------|
| Data Science Lead | [Name] | Technical questions, bug reports |
| Alberta Ballet Leadership | [Name] | Business impact, strategic decisions |
| IT/DevOps | [Name] | Deployment support |
| Documentation | [Link to Wiki] | Reference materials |

---

## Sign-Off

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Data Scientist | | | |
| Data Science Lead | | | |
| Alberta Ballet Director | | | |
| IT/DevOps | | | |

---

**Deployment Date**: ___________________

**Version**: 1.0

**Status**: ☐ Ready for Deployment ☐ Deployed ☐ Verified in Production

---

## Post-Deployment Notes

Record observations after deployment:

```
Date: _______________

Observations:
- 
- 
- 

Issues Encountered:
- 
- 

Resolutions:
- 
- 

Next Steps:
- 
- 
```
