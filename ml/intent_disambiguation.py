"""
Intent Disambiguation Correction Module

This module applies penalties to Motivation scores based on title Category to correct
for inflated demand signals due to ambiguous titles (e.g., "Cinderella", "The Little Mermaid").

Titles in certain categories have semantic ambiguity in search data:
- Searches for "Cinderella" may include movie versions, books, songs, etc.
- This inflates Motivation scores beyond what reflects actual ballet-specific intent

This correction layer applies empirically-derived penalties to adjust for this bias.

Author: Alberta Ballet Data Science Team
Date: December 2025
"""

from typing import Dict, Any
import warnings


# Category-specific penalty mappings
# These reflect observed inflation rates in search intent for ambiguous titles
CATEGORY_PENALTIES = {
    # High ambiguity: family classics and pop IP titles
    # Often have multiple entertainment versions (movies, books, songs)
    'family_classic': 0.20,      # 20% reduction
    'pop_ip': 0.20,              # 20% reduction
    'classic_romance': 0.20,     # 20% reduction  
    'classic_comedy': 0.20,      # 20% reduction
    
    # Moderate ambiguity: literary dramas and romantic tragedies
    # Some crossover with books, films, but less mainstream than family titles
    'romantic_tragedy': 0.10,    # 10% reduction
    'adult_literary_drama': 0.10 # 10% reduction
}

# Assumed contribution of Motivation to the overall Ticket Index
# Based on the scoring model architecture
MOTIVATION_INDEX_WEIGHT = 1.0 / 6.0  # 1/6 of index


def apply_intent_disambiguation(
    title_metadata: Dict[str, Any],
    *,
    apply_correction: bool = True,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Apply intent disambiguation correction to a title's metadata.
    
    This function:
    1. Identifies if the title's category requires a Motivation penalty
    2. Applies the penalty to create Motivation_corrected
    3. Recalculates TicketIndex based on the corrected Motivation
    4. Recalculates EstimatedTickets based on the corrected TicketIndex
    5. Returns enriched metadata with corrected values
    
    Args:
        title_metadata: Dictionary containing title features and metadata.
            Required keys:
                - Category: str (title category code)
                - Motivation: float (original motivation score, 0-100+ scale)
                - TicketIndex used: float (original ticket index)
                - EstimatedTickets_Final: int/float (original ticket estimate)
            Optional keys:
                - Title: str (for logging/warnings)
        
        apply_correction: If False, returns original values (for comparison/testing)
        verbose: If True, prints correction details to console
    
    Returns:
        Dictionary containing all original metadata plus:
            - Motivation_corrected: float (adjusted motivation score)
            - Motivation_penalty_applied: bool (whether penalty was applied)
            - Motivation_penalty_pct: float (penalty percentage, 0.0-1.0)
            - TicketIndex_corrected: float (recalculated ticket index)
            - EstimatedTickets_corrected: float (recalculated ticket estimate)
            - IntentCorrectionApplied: bool (overall flag for correction status)
    
    Raises:
        KeyError: If required keys are missing from title_metadata
        ValueError: If numeric values are invalid
    
    Example:
        >>> metadata = {
        ...     "Title": "Cinderella",
        ...     "Category": "family_classic",
        ...     "Motivation": 100.0,
        ...     "TicketIndex used": 100.0,
        ...     "EstimatedTickets_Final": 11976
        ... }
        >>> corrected = apply_intent_disambiguation(metadata)
        >>> corrected["Motivation_corrected"]
        80.0
        >>> corrected["Motivation_penalty_applied"]
        True
    """
    # Validate required keys
    required_keys = ["Category", "Motivation", "TicketIndex used", "EstimatedTickets_Final"]
    missing_keys = [k for k in required_keys if k not in title_metadata]
    
    if missing_keys:
        raise KeyError(
            f"Missing required keys in title_metadata: {missing_keys}. "
            f"Required keys: {required_keys}"
        )
    
    # Extract values
    category = title_metadata.get("Category", "").lower()
    motivation_original = float(title_metadata["Motivation"])
    ticket_index_original = float(title_metadata["TicketIndex used"])
    estimated_tickets_original = float(title_metadata["EstimatedTickets_Final"])
    title = title_metadata.get("Title", "Unknown")
    
    # Validate numeric values
    if motivation_original < 0:
        raise ValueError(f"Motivation must be >= 0, got {motivation_original}")
    if ticket_index_original < 0:
        raise ValueError(f"TicketIndex must be >= 0, got {ticket_index_original}")
    if estimated_tickets_original < 0:
        raise ValueError(f"EstimatedTickets must be >= 0, got {estimated_tickets_original}")
    
    # Create output dictionary starting with original metadata
    result = title_metadata.copy()
    
    # Determine if penalty applies
    penalty_pct = CATEGORY_PENALTIES.get(category, 0.0)
    penalty_applies = penalty_pct > 0 and apply_correction
    
    if penalty_applies:
        # Step 1: Apply penalty to Motivation
        motivation_corrected = motivation_original * (1.0 - penalty_pct)
        
        # Step 2: Calculate change in Motivation and its impact on TicketIndex
        delta_motivation = motivation_corrected - motivation_original
        delta_index = delta_motivation * MOTIVATION_INDEX_WEIGHT
        ticket_index_corrected = max(0.0, ticket_index_original + delta_index)
        
        # Step 3: Recalculate EstimatedTickets using the ratio k
        # k = EstimatedTickets / TicketIndex (tickets per index point)
        if ticket_index_original > 0:
            k = estimated_tickets_original / ticket_index_original
            estimated_tickets_corrected = ticket_index_corrected * k
        else:
            # Edge case: if original index was 0, maintain 0
            k = 0.0
            estimated_tickets_corrected = 0.0
        
        # Verbose logging
        if verbose:
            print(f"\n{'='*60}")
            print(f"Intent Disambiguation Correction: {title}")
            print(f"{'='*60}")
            print(f"Category: {category}")
            print(f"Penalty: {penalty_pct*100:.0f}%")
            print(f"\nMotivation:")
            print(f"  Original:  {motivation_original:.2f}")
            print(f"  Corrected: {motivation_corrected:.2f}")
            print(f"  Delta:     {delta_motivation:.2f}")
            print(f"\nTicket Index:")
            print(f"  Original:  {ticket_index_original:.2f}")
            print(f"  Corrected: {ticket_index_corrected:.2f}")
            print(f"  Delta:     {delta_index:.2f}")
            print(f"\nEstimated Tickets:")
            print(f"  Original:  {estimated_tickets_original:,.0f}")
            print(f"  Corrected: {estimated_tickets_corrected:,.0f}")
            print(f"  Delta:     {estimated_tickets_corrected - estimated_tickets_original:,.0f}")
            print(f"{'='*60}\n")
        
        # Populate corrected values
        result["Motivation_corrected"] = motivation_corrected
        result["Motivation_penalty_applied"] = True
        result["Motivation_penalty_pct"] = penalty_pct
        result["TicketIndex_corrected"] = ticket_index_corrected
        result["EstimatedTickets_corrected"] = estimated_tickets_corrected
        result["IntentCorrectionApplied"] = True
        
    else:
        # No penalty applies or correction disabled
        # Return original values in corrected fields
        result["Motivation_corrected"] = motivation_original
        result["Motivation_penalty_applied"] = False
        result["Motivation_penalty_pct"] = 0.0
        result["TicketIndex_corrected"] = ticket_index_original
        result["EstimatedTickets_corrected"] = estimated_tickets_original
        result["IntentCorrectionApplied"] = False
        
        if verbose and penalty_pct == 0:
            print(f"No intent correction needed for '{title}' (category: {category})")
    
    return result


def get_penalty_for_category(category: str) -> float:
    """
    Retrieve the Motivation penalty percentage for a given category.
    
    Args:
        category: Title category code (case-insensitive)
    
    Returns:
        Penalty as a decimal (e.g., 0.20 for 20% reduction)
        Returns 0.0 if category has no penalty
    
    Example:
        >>> get_penalty_for_category("family_classic")
        0.2
        >>> get_penalty_for_category("contemporary")
        0.0
    """
    return CATEGORY_PENALTIES.get(category.lower(), 0.0)


def get_all_penalty_categories() -> Dict[str, float]:
    """
    Get a dictionary of all categories with assigned penalties.
    
    Returns:
        Dict mapping category codes to penalty percentages
    
    Example:
        >>> penalties = get_all_penalty_categories()
        >>> penalties["family_classic"]
        0.2
    """
    return CATEGORY_PENALTIES.copy()


def batch_apply_corrections(
    titles: list[Dict[str, Any]],
    *,
    apply_correction: bool = True,
    verbose: bool = False
) -> list[Dict[str, Any]]:
    """
    Apply intent disambiguation corrections to multiple titles.
    
    Args:
        titles: List of title metadata dictionaries
        apply_correction: If False, returns original values for all titles
        verbose: If True, prints correction details for each title
    
    Returns:
        List of corrected metadata dictionaries
    
    Example:
        >>> titles = [
        ...     {"Title": "Cinderella", "Category": "family_classic", ...},
        ...     {"Title": "Romeo & Juliet", "Category": "romantic_tragedy", ...}
        ... ]
        >>> corrected = batch_apply_corrections(titles)
    """
    corrected_titles = []
    
    for title_metadata in titles:
        try:
            corrected = apply_intent_disambiguation(
                title_metadata,
                apply_correction=apply_correction,
                verbose=verbose
            )
            corrected_titles.append(corrected)
        except (KeyError, ValueError) as e:
            title = title_metadata.get("Title", "Unknown")
            warnings.warn(
                f"Failed to apply correction to '{title}': {str(e)}. "
                f"Skipping this title.",
                UserWarning
            )
            # Add original metadata with correction flags set to False
            result = title_metadata.copy()
            result.update({
                "Motivation_corrected": title_metadata.get("Motivation", 0.0),
                "Motivation_penalty_applied": False,
                "Motivation_penalty_pct": 0.0,
                "TicketIndex_corrected": title_metadata.get("TicketIndex used", 0.0),
                "EstimatedTickets_corrected": title_metadata.get("EstimatedTickets_Final", 0.0),
                "IntentCorrectionApplied": False
            })
            corrected_titles.append(result)
    
    return corrected_titles


if __name__ == "__main__":
    """
    Demonstration and testing of the intent disambiguation module.
    """
    print("Intent Disambiguation Correction Module - Demo\n")
    
    # Sample test cases
    test_titles = [
        {
            "Title": "Cinderella",
            "Category": "family_classic",
            "Motivation": 100.0,
            "TicketIndex used": 100.0,
            "EstimatedTickets_Final": 11976
        },
        {
            "Title": "Romeo & Juliet",
            "Category": "romantic_tragedy",
            "Motivation": 95.0,
            "TicketIndex used": 98.0,
            "EstimatedTickets_Final": 10500
        },
        {
            "Title": "Contemporary Triple Bill",
            "Category": "contemporary",
            "Motivation": 75.0,
            "TicketIndex used": 85.0,
            "EstimatedTickets_Final": 8200
        }
    ]
    
    print("Testing individual corrections with verbose output:\n")
    for title_data in test_titles:
        corrected = apply_intent_disambiguation(title_data, verbose=True)
    
    print("\n" + "="*60)
    print("Batch correction summary:")
    print("="*60)
    
    batch_corrected = batch_apply_corrections(test_titles, verbose=False)
    
    for original, corrected in zip(test_titles, batch_corrected):
        print(f"\n{corrected['Title']} ({corrected['Category']}):")
        print(f"  Penalty Applied: {corrected['Motivation_penalty_applied']}")
        if corrected['Motivation_penalty_applied']:
            print(f"  Penalty %: {corrected['Motivation_penalty_pct']*100:.0f}%")
            print(f"  Tickets: {original['EstimatedTickets_Final']:,.0f} → "
                  f"{corrected['EstimatedTickets_corrected']:,.0f} "
                  f"({corrected['EstimatedTickets_corrected'] - original['EstimatedTickets_Final']:+,.0f})")
