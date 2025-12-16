"""
Integration Example: Intent Disambiguation + Title Explanation Engine

This script demonstrates how to integrate the intent disambiguation correction
module with the existing title explanation engine.

Author: Alberta Ballet Data Science Team
Date: December 2025
"""

from ml.intent_disambiguation import apply_intent_disambiguation, batch_apply_corrections
from ml.title_explanation_engine import build_title_explanation


def generate_corrected_explanation(
    title_metadata: dict,
    apply_correction: bool = True,
    verbose: bool = False
) -> tuple[str, dict]:
    """
    Generate a title explanation using intent-corrected values.
    
    This function:
    1. Applies intent disambiguation correction
    2. Updates the metadata with corrected values
    3. Generates the explanation narrative
    4. Returns both the narrative and corrected metadata
    
    Args:
        title_metadata: Original title metadata dict
        apply_correction: Whether to apply intent disambiguation
        verbose: Print correction details
    
    Returns:
        Tuple of (narrative_string, corrected_metadata)
    
    Example:
        >>> metadata = {
        ...     "Title": "Cinderella",
        ...     "Category": "family_classic",
        ...     "Month": "March 2026",
        ...     "Motivation": 100.0,
        ...     "Familiarity": 110.0,
        ...     "TicketIndex used": 100.0,
        ...     "EstimatedTickets_Final": 11976,
        ...     "YYC_Singles": 7000,
        ...     "YEG_Singles": 4976
        ... }
        >>> narrative, corrected = generate_corrected_explanation(metadata)
    """
    # Step 1: Apply intent disambiguation correction
    corrected_metadata = apply_intent_disambiguation(
        title_metadata,
        apply_correction=apply_correction,
        verbose=verbose
    )
    
    # Step 2: Update metadata with corrected values for narrative generation
    # Create a copy to avoid modifying the original
    narrative_metadata = corrected_metadata.copy()
    
    # Replace the values used in explanation with corrected values
    narrative_metadata["Motivation"] = corrected_metadata["Motivation_corrected"]
    narrative_metadata["TicketIndex used"] = corrected_metadata["TicketIndex_corrected"]
    narrative_metadata["EstimatedTickets_Final"] = corrected_metadata["EstimatedTickets_corrected"]
    
    # Calculate city splits based on corrected totals
    if "YYC_Singles" in narrative_metadata and "YEG_Singles" in narrative_metadata:
        original_total = narrative_metadata["YYC_Singles"] + narrative_metadata["YEG_Singles"]
        if original_total > 0:
            yyc_ratio = narrative_metadata["YYC_Singles"] / original_total
            yeg_ratio = narrative_metadata["YEG_Singles"] / original_total
            
            corrected_total = corrected_metadata["EstimatedTickets_corrected"]
            narrative_metadata["YYC_Singles"] = int(corrected_total * yyc_ratio)
            narrative_metadata["YEG_Singles"] = int(corrected_total * yeg_ratio)
    
    # Step 3: Generate explanation narrative
    narrative = build_title_explanation(
        title_metadata=narrative_metadata,
        style="board"
    )
    
    return narrative, corrected_metadata


def process_season_with_corrections(
    season_titles: list[dict],
    apply_correction: bool = True,
    verbose: bool = False
) -> list[dict]:
    """
    Process an entire season of titles with intent disambiguation corrections.
    
    Args:
        season_titles: List of title metadata dicts
        apply_correction: Whether to apply corrections
        verbose: Print details for each title
    
    Returns:
        List of dicts containing:
            - All original metadata
            - Corrected values
            - Generated narrative
    
    Example:
        >>> season = [
        ...     {"Title": "Cinderella", "Category": "family_classic", ...},
        ...     {"Title": "Romeo & Juliet", "Category": "romantic_tragedy", ...}
        ... ]
        >>> results = process_season_with_corrections(season)
    """
    results = []
    
    # Apply corrections in batch
    corrected_titles = batch_apply_corrections(
        season_titles,
        apply_correction=apply_correction,
        verbose=verbose
    )
    
    # Generate narratives for each title
    for corrected_metadata in corrected_titles:
        # Create narrative metadata
        narrative_metadata = corrected_metadata.copy()
        narrative_metadata["Motivation"] = corrected_metadata["Motivation_corrected"]
        narrative_metadata["TicketIndex used"] = corrected_metadata["TicketIndex_corrected"]
        narrative_metadata["EstimatedTickets_Final"] = corrected_metadata["EstimatedTickets_corrected"]
        
        # Update city splits if present
        if "YYC_Singles" in narrative_metadata and "YEG_Singles" in narrative_metadata:
            original_total = corrected_metadata.get("EstimatedTickets_Final", 0)
            if original_total > 0:
                yyc_ratio = corrected_metadata["YYC_Singles"] / original_total
                yeg_ratio = corrected_metadata["YEG_Singles"] / original_total
                
                corrected_total = corrected_metadata["EstimatedTickets_corrected"]
                narrative_metadata["YYC_Singles"] = int(corrected_total * yyc_ratio)
                narrative_metadata["YEG_Singles"] = int(corrected_total * yeg_ratio)
        
        # Generate narrative
        narrative = build_title_explanation(
            title_metadata=narrative_metadata,
            style="board"
        )
        
        # Add narrative to metadata
        corrected_metadata["narrative"] = narrative
        
        results.append(corrected_metadata)
    
    return results


def compare_original_vs_corrected(
    title_metadata: dict,
    verbose: bool = True
) -> dict:
    """
    Generate side-by-side comparison of original vs. corrected forecasts.
    
    Useful for validation, board presentations, and A/B testing.
    
    Args:
        title_metadata: Original title metadata
        verbose: Print comparison table
    
    Returns:
        Dict with 'original' and 'corrected' keys containing full metadata
    
    Example:
        >>> metadata = {...}
        >>> comparison = compare_original_vs_corrected(metadata)
    """
    # Get original values (correction disabled)
    original = apply_intent_disambiguation(
        title_metadata,
        apply_correction=False,
        verbose=False
    )
    
    # Get corrected values
    corrected = apply_intent_disambiguation(
        title_metadata,
        apply_correction=True,
        verbose=False
    )
    
    if verbose:
        title = title_metadata.get("Title", "Unknown")
        category = title_metadata.get("Category", "Unknown")
        
        print(f"\n{'='*70}")
        print(f"Comparison: {title} ({category})")
        print(f"{'='*70}")
        
        if corrected["Motivation_penalty_applied"]:
            penalty_pct = corrected["Motivation_penalty_pct"] * 100
            print(f"⚠️  Intent disambiguation penalty applied: {penalty_pct:.0f}%\n")
            
            print(f"{'Metric':<30} {'Original':>15} {'Corrected':>15} {'Δ':>10}")
            print(f"{'-'*70}")
            
            # Motivation
            mot_orig = original["Motivation_corrected"]
            mot_corr = corrected["Motivation_corrected"]
            mot_delta = mot_corr - mot_orig
            print(f"{'Motivation':<30} {mot_orig:>15.2f} {mot_corr:>15.2f} {mot_delta:>10.2f}")
            
            # Ticket Index
            idx_orig = original["TicketIndex_corrected"]
            idx_corr = corrected["TicketIndex_corrected"]
            idx_delta = idx_corr - idx_orig
            print(f"{'Ticket Index':<30} {idx_orig:>15.2f} {idx_corr:>15.2f} {idx_delta:>10.2f}")
            
            # Estimated Tickets
            tix_orig = original["EstimatedTickets_corrected"]
            tix_corr = corrected["EstimatedTickets_corrected"]
            tix_delta = tix_corr - tix_orig
            print(f"{'Estimated Tickets':<30} {tix_orig:>15,.0f} {tix_corr:>15,.0f} {tix_delta:>10,.0f}")
            
            print(f"{'='*70}\n")
        else:
            print(f"✓ No correction needed for this category\n")
    
    return {
        "original": original,
        "corrected": corrected
    }


if __name__ == "__main__":
    """
    Demo: Integration with Title Explanation Engine
    """
    print("="*70)
    print("Intent Disambiguation + Title Explanation Engine Integration Demo")
    print("="*70)
    
    # Sample title metadata
    sample_titles = [
        {
            "Title": "Cinderella",
            "Category": "family_classic",
            "Month": "March 2026",
            "Motivation": 100.0,
            "Familiarity": 110.0,
            "SignalOnly": 105.0,
            "TicketIndex used": 100.0,
            "EstimatedTickets_Final": 11976,
            "FutureSeasonalityFactor": 1.05,
            "PrimarySegment": "Family",
            "SecondarySegment": "Tourist",
            "YYC_Singles": 7186,
            "YEG_Singles": 4790,
            "IsRemount": False
        },
        {
            "Title": "Romeo & Juliet",
            "Category": "romantic_tragedy",
            "Month": "February 2026",
            "Motivation": 95.0,
            "Familiarity": 105.0,
            "SignalOnly": 100.0,
            "TicketIndex used": 98.0,
            "EstimatedTickets_Final": 10500,
            "FutureSeasonalityFactor": 1.02,
            "PrimarySegment": "Culturalist",
            "SecondarySegment": "Romantics",
            "YYC_Singles": 6300,
            "YEG_Singles": 4200,
            "IsRemount": True,
            "YearsSinceLastRun": 5
        }
    ]
    
    print("\n1. Individual Title Processing with Comparison")
    print("-" * 70)
    
    for title_metadata in sample_titles:
        comparison = compare_original_vs_corrected(title_metadata, verbose=True)
    
    print("\n2. Generate Corrected Narratives")
    print("-" * 70)
    
    for title_metadata in sample_titles:
        narrative, corrected = generate_corrected_explanation(
            title_metadata,
            apply_correction=True,
            verbose=False
        )
        
        print(f"\n{corrected['Title']} ({corrected['Category']})")
        print(f"Penalty Applied: {corrected['Motivation_penalty_applied']}")
        if corrected['Motivation_penalty_applied']:
            print(f"Corrected Tickets: {corrected['EstimatedTickets_corrected']:,.0f}")
        print()
    
    print("\n3. Batch Season Processing")
    print("-" * 70)
    
    results = process_season_with_corrections(sample_titles, verbose=False)
    
    print(f"\nProcessed {len(results)} titles:")
    for result in results:
        status = "✓ Corrected" if result['IntentCorrectionApplied'] else "○ No correction"
        print(f"  {status}: {result['Title']}")
    
    print("\n" + "="*70)
    print("Integration demo complete!")
    print("="*70)
