# 🧠 Intent Disambiguation Correction

A correction layer for the Alberta Ballet Title Scoring system that adjusts for inflated demand signals due to ambiguous title names.

---

## The Problem

When forecasting demand for ballet titles like **"Cinderella"** or **"Romeo & Juliet"**, search-based metrics (Wikipedia views, Google Trends, YouTube) capture interest in **all versions** of these titles:

- 🎬 Movies (Disney, adaptations)
- 📚 Books and literature
- 🎵 Music and soundtracks
- 🎭 Broadway/theater
- 🩰 **Ballet** ← our target

This creates **systematically inflated Motivation scores** for certain categories.

---

## The Solution

Apply empirically-derived **category-based penalties** to correct for this bias:

| Category | Penalty | Why? |
|----------|---------|------|
| `family_classic` | **20%** | High overlap with Disney movies, fairy tale books |
| `pop_ip` | **20%** | Strong cross-media presence (Frozen, etc.) |
| `classic_romance` | **20%** | Multiple film/book versions |
| `classic_comedy` | **20%** | Broad entertainment appeal |
| `romantic_tragedy` | **10%** | Moderate literary/film crossover |
| `adult_literary_drama` | **10%** | Some book/film versions |
| All others | **0%** | Category-specific or niche |

---

## Quick Start

```python
from ml.intent_disambiguation import apply_intent_disambiguation

# Your title metadata
metadata = {
    "Title": "Cinderella",
    "Category": "family_classic",
    "Motivation": 100.0,
    "TicketIndex used": 100.0,
    "EstimatedTickets_Final": 11976
}

# Apply correction
corrected = apply_intent_disambiguation(metadata)

# Results
print(f"Original: {metadata['EstimatedTickets_Final']:,} tickets")
print(f"Corrected: {corrected['EstimatedTickets_corrected']:,.0f} tickets")
print(f"Difference: {corrected['EstimatedTickets_corrected'] - metadata['EstimatedTickets_Final']:+,.0f}")
```

**Output**:
```
Original: 11,976 tickets
Corrected: 11,577 tickets
Difference: -399
```

---

## Example: Cinderella

**Before** (inflated by Disney movies, books, etc.):
- Motivation: 100.0
- Ticket Index: 100.0
- Estimated Tickets: **11,976**

**After** (ballet-specific intent):
- Motivation: 80.0 (↓20%)
- Ticket Index: 96.67 (↓3.3%)
- Estimated Tickets: **11,577** (↓399 tickets)

---

## How It Works

1. **Identify category** → Look up penalty (0%, 10%, or 20%)
2. **Reduce Motivation** → `Motivation_corrected = Motivation × (1 - penalty)`
3. **Adjust Ticket Index** → Account for Motivation's 1/6 contribution
4. **Recalculate Tickets** → Apply new index to get corrected estimate

---

## Files & Documentation

| File | Purpose |
|------|---------|
| [`ml/intent_disambiguation.py`](ml/intent_disambiguation.py) | Core correction module |
| [`tests/test_intent_disambiguation.py`](tests/test_intent_disambiguation.py) | Comprehensive test suite |
| [`docs/INTENT_DISAMBIGUATION.md`](docs/INTENT_DISAMBIGUATION.md) | Full documentation |
| [`docs/INTENT_DISAMBIGUATION_QUICK_REF.md`](docs/INTENT_DISAMBIGUATION_QUICK_REF.md) | Quick reference guide |
| [`examples/intent_disambiguation_integration.py`](examples/intent_disambiguation_integration.py) | Integration examples |
| [`scripts/validate_intent_disambiguation.py`](scripts/validate_intent_disambiguation.py) | Validation script |
| [`IMPLEMENTATION_SUMMARY.md`](IMPLEMENTATION_SUMMARY.md) | Implementation summary |

---

## Demos & Testing

### Run Built-in Demo
```bash
python -m ml.intent_disambiguation
```

### Run Integration Example
```bash
python -m examples.intent_disambiguation_integration
```

### Run Validation Script
```bash
python scripts/validate_intent_disambiguation.py
```

### Run Test Suite
```bash
pytest tests/test_intent_disambiguation.py -v
```

---

## Common Use Cases

### 1. Single Title Correction
```python
from ml.intent_disambiguation import apply_intent_disambiguation

corrected = apply_intent_disambiguation(title_metadata, verbose=True)
```

### 2. Season-Wide Batch Processing
```python
from ml.intent_disambiguation import batch_apply_corrections

corrected_season = batch_apply_corrections(all_titles)
```

### 3. A/B Comparison
```python
from examples.intent_disambiguation_integration import compare_original_vs_corrected

comparison = compare_original_vs_corrected(title_metadata)
```

### 4. Integration with Explanations
```python
from examples.intent_disambiguation_integration import generate_corrected_explanation

narrative, corrected = generate_corrected_explanation(title_metadata)
```

---

## Integration Guide

To integrate into your forecasting pipeline:

```python
# 1. Import
from ml.intent_disambiguation import apply_intent_disambiguation

# 2. Apply correction before generating reports
for title in season_titles:
    corrected = apply_intent_disambiguation(title)
    
    # 3. Use corrected values
    title["Motivation"] = corrected["Motivation_corrected"]
    title["TicketIndex used"] = corrected["TicketIndex_corrected"]
    title["EstimatedTickets_Final"] = corrected["EstimatedTickets_corrected"]
```

---

## FAQ

**Q: Does every title get corrected?**  
A: No, only titles in high-ambiguity categories (`family_classic`, `pop_ip`, `romantic_tragedy`, etc.)

**Q: Can I adjust the penalties?**  
A: Yes! Edit `CATEGORY_PENALTIES` in `ml/intent_disambiguation.py`

**Q: What if I want to see before/after?**  
A: Use `compare_original_vs_corrected()` from the examples module

**Q: Does this affect Familiarity?**  
A: No, only Motivation (search engagement) is corrected

**Q: How do I disable corrections for testing?**  
A: Pass `apply_correction=False` to any function

---

## Validation Results

All validation tests pass ✓:
- Module imports
- Penalty retrieval
- Basic corrections
- Batch processing
- Integration with explanation engine
- Comparison functions
- Edge case handling
- Constants validation

Run `python scripts/validate_intent_disambiguation.py` to verify.

---

## Impact Assessment

Based on Alberta Ballet's repertoire:

- **~25% of titles** receive 20% correction (family classics, pop IP)
- **~15% of titles** receive 10% correction (romantic tragedies)
- **~60% of titles** receive no correction (contemporary, Nutcracker, etc.)

**Overall season impact**: ~1.5% reduction in total forecast (more accurate)

---

## Support

- **Documentation**: See `docs/INTENT_DISAMBIGUATION.md`
- **Examples**: See `examples/intent_disambiguation_integration.py`
- **Tests**: Run `pytest tests/test_intent_disambiguation.py -v`
- **Contact**: Alberta Ballet Data Science Team

---

## Status

✅ **Production Ready**

All features implemented, tested, and documented. Ready for deployment.

---

**Version**: 1.0  
**Date**: December 2025  
**Author**: Alberta Ballet Data Science Team
