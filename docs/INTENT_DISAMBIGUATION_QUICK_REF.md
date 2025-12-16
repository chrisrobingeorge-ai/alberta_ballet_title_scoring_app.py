# Intent Disambiguation - Quick Reference

## TL;DR

**Problem**: Titles like "Cinderella" have inflated Motivation scores because Google searches include movies, books, and songs—not just ballet.

**Solution**: Apply category-based penalties to correct for this bias.

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

# Use corrected values
print(f"Corrected Motivation: {corrected['Motivation_corrected']}")
print(f"Corrected Tickets: {corrected['EstimatedTickets_corrected']:,.0f}")
```

---

## Penalty Table

| Category | Penalty | Examples |
|----------|---------|----------|
| `family_classic` | **20%** | Cinderella, The Little Mermaid, Alice in Wonderland |
| `pop_ip` | **20%** | Frozen, Moana, Beauty and the Beast |
| `classic_romance` | **20%** | Pride and Prejudice, Wuthering Heights |
| `classic_comedy` | **20%** | A Midsummer Night's Dream, Much Ado About Nothing |
| `romantic_tragedy` | **10%** | Romeo & Juliet, Anna Karenina |
| `adult_literary_drama` | **10%** | Gatsby, Of Mice and Men |
| All others | **0%** | Contemporary works, Nutcracker, company originals |

---

## What Gets Corrected?

1. **Motivation Score**: Reduced by penalty percentage
2. **Ticket Index**: Adjusted based on Motivation's 1/6 contribution
3. **Estimated Tickets**: Recalculated using the corrected index

---

## Example: Cinderella

**Before Correction**:
- Motivation: 100.0
- Ticket Index: 100.0  
- Estimated Tickets: 11,976

**After Correction** (20% penalty):
- Motivation: 80.0 ↓
- Ticket Index: 96.67 ↓
- Estimated Tickets: 11,577 ↓

**Reduction**: -399 tickets (-3.3%)

---

## When to Use

✅ **Use when**:
- Generating season forecasts
- Creating board presentations
- Producing PDF reports
- Exporting ticket estimates

❌ **Don't use when**:
- Title has no category assigned
- Running diagnostic/debug analyses
- Comparing raw signal data

---

## Integration Patterns

### Pattern 1: Simple Correction

```python
corrected = apply_intent_disambiguation(title_metadata)
# Use corrected["Motivation_corrected"], etc.
```

### Pattern 2: Batch Processing

```python
from ml.intent_disambiguation import batch_apply_corrections

corrected_season = batch_apply_corrections(all_titles)
```

### Pattern 3: With Explanation Engine

```python
from examples.intent_disambiguation_integration import generate_corrected_explanation

narrative, corrected = generate_corrected_explanation(title_metadata)
```

### Pattern 4: A/B Comparison

```python
from examples.intent_disambiguation_integration import compare_original_vs_corrected

comparison = compare_original_vs_corrected(title_metadata)
# Returns both original and corrected for side-by-side analysis
```

---

## Output Fields

Every corrected metadata dict includes:

| Field | Type | Description |
|-------|------|-------------|
| `Motivation_corrected` | float | Adjusted motivation score |
| `Motivation_penalty_applied` | bool | Whether penalty was applied |
| `Motivation_penalty_pct` | float | Penalty percentage (0.0-1.0) |
| `TicketIndex_corrected` | float | Recalculated ticket index |
| `EstimatedTickets_corrected` | float | Recalculated ticket estimate |
| `IntentCorrectionApplied` | bool | Overall correction status |

---

## Validation Checklist

Before deploying to production:

- [ ] Run demo: `python -m ml.intent_disambiguation`
- [ ] Run tests: `pytest tests/test_intent_disambiguation.py -v`
- [ ] Compare with historical actuals
- [ ] Review with domain experts
- [ ] Document any custom penalty adjustments

---

## FAQs

**Q: Why 20% and 10%?**  
A: Derived from manual analysis of search results. 20% for high-ambiguity titles (Disney movies, etc.), 10% for moderate ambiguity (literature adaptations).

**Q: Can I adjust the penalties?**  
A: Yes! Edit `CATEGORY_PENALTIES` in `ml/intent_disambiguation.py`. Document your changes.

**Q: Does this affect Familiarity?**  
A: No, only Motivation is corrected. Familiarity reflects name recognition (which isn't biased the same way).

**Q: What if my title has no category?**  
A: No correction applied (0% penalty). Ensure all titles have valid categories.

**Q: Can I disable corrections for testing?**  
A: Yes! Use `apply_correction=False`:
```python
apply_intent_disambiguation(metadata, apply_correction=False)
```

---

## File Locations

- **Module**: [`ml/intent_disambiguation.py`](../ml/intent_disambiguation.py)
- **Tests**: [`tests/test_intent_disambiguation.py`](../tests/test_intent_disambiguation.py)
- **Docs**: [`docs/INTENT_DISAMBIGUATION.md`](INTENT_DISAMBIGUATION.md)
- **Examples**: [`examples/intent_disambiguation_integration.py`](../examples/intent_disambiguation_integration.py)

---

## Support

For questions or calibration adjustments:
- Review full documentation: `docs/INTENT_DISAMBIGUATION.md`
- Check examples: `examples/intent_disambiguation_integration.py`
- Contact: Alberta Ballet Data Science Team
