"""
Utilities for canonicalizing show titles and fuzzy matching.

This module provides functions to:
1. Normalize title strings for consistent matching
2. Load and manage title-to-ID mappings
3. Perform fuzzy matching for title resolution
"""

from __future__ import annotations

import csv
import os
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

# Default path to the title ID map CSV
DEFAULT_TITLE_MAP_PATH = "data/title_id_map.csv"


def canonicalize_title(title: str) -> str:
    """
    Normalize a title string for consistent matching.
    
    Transformations:
    - Convert to lowercase
    - Strip leading/trailing whitespace
    - Normalize Unicode characters (NFD normalization)
    - Replace various apostrophe/quote characters with standard single quote
    - Replace various dash characters with standard hyphen
    - Remove all punctuation except hyphens and apostrophes
    - Collapse multiple spaces/hyphens into single ones
    
    Args:
        title: The raw title string to canonicalize
        
    Returns:
        Canonicalized title string suitable for matching
        
    Examples:
        >>> canonicalize_title("Swan Lake")
        'swan lake'
        >>> canonicalize_title("Romeo & Juliet")
        'romeo juliet'
        >>> canonicalize_title("  The Nutcracker  ")
        'the nutcracker'
        >>> canonicalize_title("All of Us – Tragically Hip")
        'all of us - tragically hip'
    """
    if not title:
        return ""
    
    # Strip whitespace and convert to lowercase
    s = title.strip().lower()
    
    # Normalize Unicode (e.g., handle accents consistently)
    s = unicodedata.normalize("NFD", s)
    
    # Normalize various apostrophe/quote characters to standard apostrophe
    apostrophes = ["'", "'", "'", "`", "ʼ", "ʻ", "′"]
    for char in apostrophes:
        s = s.replace(char, "'")
    
    # Normalize various dash characters to standard hyphen
    dashes = ["–", "—", "−", "‐", "‑", "‒", "–", "—", "―"]
    for char in dashes:
        s = s.replace(char, "-")
    
    # Remove ampersand by replacing with space (Romeo & Juliet -> Romeo Juliet)
    s = s.replace("&", " ")
    
    # Remove punctuation except hyphens and apostrophes
    # Keep letters, numbers, spaces, hyphens, and apostrophes
    s = re.sub(r"[^\w\s\-']", "", s)
    
    # Collapse multiple spaces into single space
    s = re.sub(r"\s+", " ", s)
    
    # Collapse multiple hyphens into single hyphen
    s = re.sub(r"-+", "-", s)
    
    # Strip again after transformations
    s = s.strip()
    
    return s


def load_title_map(path: str = DEFAULT_TITLE_MAP_PATH) -> Dict[str, str]:
    """
    Load a mapping from canonical title to show_title_id.
    
    The CSV file should have columns:
    - show_title: The canonical (or original) title
    - show_title_id: A unique identifier for the show
    
    Args:
        path: Path to the title_id_map.csv file
        
    Returns:
        Dictionary mapping canonical title -> show_title_id
        
    Raises:
        FileNotFoundError: If the mapping file doesn't exist
    """
    mapping: Dict[str, str] = {}
    
    if not os.path.exists(path):
        # Return empty mapping if file doesn't exist yet
        return mapping
    
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip rows that are comments or empty
            show_title = row.get("show_title") or ""
            show_title_id = row.get("show_title_id") or ""
            
            # Handle None values safely
            if show_title:
                show_title = show_title.strip()
            if show_title_id:
                show_title_id = show_title_id.strip()
            
            if show_title and show_title_id and not show_title.startswith("#"):
                # Store both the original and canonicalized versions
                mapping[show_title] = show_title_id
                canonical = canonicalize_title(show_title)
                if canonical != show_title:
                    mapping[canonical] = show_title_id
    
    return mapping


def fuzzy_match_title(
    title: str, 
    choices: Iterable[str], 
    threshold: int = 85
) -> Optional[str]:
    """
    Find the best fuzzy match for a title among a set of choices.
    
    Uses rapidfuzz if available, falls back to difflib.SequenceMatcher.
    
    Args:
        title: The title to match
        choices: Iterable of candidate titles to match against
        threshold: Minimum similarity score (0-100) to accept a match
        
    Returns:
        The best matching title if score >= threshold, else None
        
    Examples:
        >>> choices = ["Swan Lake", "Sleeping Beauty", "Romeo and Juliet"]
        >>> fuzzy_match_title("Swn Lake", choices)  # typo
        'Swan Lake'
        >>> fuzzy_match_title("Hamlet", choices, threshold=90)  # no good match
        None
    """
    # Canonicalize input for comparison
    canonical_input = canonicalize_title(title)
    choices_list = list(choices)
    
    if not choices_list:
        return None
    
    # Try rapidfuzz first (faster and better for fuzzy matching)
    try:
        from rapidfuzz import fuzz
        from rapidfuzz.process import extractOne
        
        result = extractOne(
            canonical_input,
            [canonicalize_title(c) for c in choices_list],
            scorer=fuzz.WRatio,
            score_cutoff=threshold
        )
        
        if result is not None:
            # Return the original (non-canonicalized) choice
            matched_canonical, score, idx = result
            return choices_list[idx]
        return None
        
    except ImportError:
        # Fallback to difflib
        pass
    
    # Difflib fallback
    best_match = None
    best_score = 0
    
    for choice in choices_list:
        canonical_choice = canonicalize_title(choice)
        # SequenceMatcher returns ratio between 0 and 1
        ratio = SequenceMatcher(None, canonical_input, canonical_choice).ratio()
        score = int(ratio * 100)
        
        if score > best_score:
            best_score = score
            best_match = choice
    
    if best_score >= threshold:
        return best_match
    return None


def build_title_map_from_hist(
    history_path: str = "data/history_city_sales.csv",
    baseline_path: str = "data/baselines.csv",
    output_path: str = DEFAULT_TITLE_MAP_PATH,
    threshold: int = 85,
    dry_run: bool = False
) -> Tuple[Dict[str, str], list]:
    """
    Build a title-to-ID mapping from historical data and baselines.
    
    This function:
    1. Loads titles from history_city_sales.csv and baselines.csv
    2. Attempts to match titles between the two sources
    3. Creates a suggested mapping CSV
    4. Reports ambiguous matches for manual review
    
    Args:
        history_path: Path to history_city_sales.csv
        baseline_path: Path to baselines.csv
        output_path: Path to write the title_id_map.csv
        threshold: Fuzzy match threshold (0-100)
        dry_run: If True, don't write output file, just return results
        
    Returns:
        Tuple of (mapping_dict, ambiguous_list)
        - mapping_dict: Dictionary of {show_title: show_title_id}
        - ambiguous_list: List of titles that need manual review
    """
    import pandas as pd
    
    mapping: Dict[str, str] = {}
    ambiguous: list = []
    
    # Load baselines
    try:
        baselines_df = pd.read_csv(baseline_path)
        baseline_titles = set(baselines_df["title"].dropna().astype(str).str.strip().tolist())
    except (FileNotFoundError, KeyError):
        baseline_titles = set()
        print(f"Warning: Could not load baselines from {baseline_path}")
    
    # Load history
    try:
        history_df = pd.read_csv(history_path, thousands=",")
        # Try common column names
        title_col = None
        for col in ["Show Title", "show_title", "Title", "title", "Show_Title"]:
            if col in history_df.columns:
                title_col = col
                break
        if title_col:
            history_titles = set(history_df[title_col].dropna().astype(str).str.strip().tolist())
        else:
            history_titles = set()
            print(f"Warning: Could not find title column in {history_path}")
    except FileNotFoundError:
        history_titles = set()
        print(f"Warning: Could not load history from {history_path}")
    
    # Combine all unique titles
    all_titles = baseline_titles | history_titles
    
    # Assign IDs (simple sequential for now)
    # Group titles that match via canonicalization
    canonical_groups: Dict[str, list] = {}
    for title in all_titles:
        canonical = canonicalize_title(title)
        if canonical not in canonical_groups:
            canonical_groups[canonical] = []
        canonical_groups[canonical].append(title)
    
    # Assign IDs and track ambiguous
    next_id = 1
    for canonical, titles in sorted(canonical_groups.items()):
        if len(titles) == 1:
            mapping[titles[0]] = f"SHOW_{next_id:04d}"
            next_id += 1
        else:
            # Multiple titles with same canonical form - might be duplicates or variants
            # Use the first one as primary
            primary = titles[0]
            mapping[primary] = f"SHOW_{next_id:04d}"
            for variant in titles[1:]:
                mapping[variant] = f"SHOW_{next_id:04d}"  # Same ID
            if len(set(titles)) > 1:
                ambiguous.append({
                    "canonical": canonical,
                    "variants": titles,
                    "assigned_id": f"SHOW_{next_id:04d}"
                })
            next_id += 1
    
    # Also try fuzzy matching between history and baseline titles
    unmatched_history = history_titles - baseline_titles
    for hist_title in unmatched_history:
        match = fuzzy_match_title(hist_title, baseline_titles, threshold=threshold)
        if match and match != hist_title:
            # Found a fuzzy match - these might be the same show
            ambiguous.append({
                "type": "fuzzy_match",
                "history_title": hist_title,
                "baseline_match": match,
                "score": threshold,
                "review_needed": True
            })
    
    # Print report
    print(f"\n{'='*60}")
    print("Title Mapping Report")
    print(f"{'='*60}")
    print(f"Total unique titles: {len(all_titles)}")
    print(f"Titles from baselines: {len(baseline_titles)}")
    print(f"Titles from history: {len(history_titles)}")
    print(f"Titles in both: {len(baseline_titles & history_titles)}")
    print(f"Unique canonical forms: {len(canonical_groups)}")
    
    if ambiguous:
        print(f"\n⚠️  Ambiguous matches requiring manual review: {len(ambiguous)}")
        for item in ambiguous[:10]:  # Show first 10
            if item.get("type") == "fuzzy_match":
                print(f"  - '{item['history_title']}' might match '{item['baseline_match']}'")
            else:
                print(f"  - Canonical '{item['canonical']}' has variants: {item['variants']}")
        if len(ambiguous) > 10:
            print(f"  ... and {len(ambiguous) - 10} more")
    
    # Write output file if not dry_run
    if not dry_run:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            # Write comment before CSV data
            f.write("# Auto-generated title mapping. Review and edit as needed.\n")
            writer = csv.writer(f)
            writer.writerow(["show_title", "show_title_id"])
            for title, title_id in sorted(mapping.items()):
                writer.writerow([title, title_id])
        print(f"\n✓ Wrote mapping to {output_path}")
    
    return mapping, ambiguous


# CLI entry point
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Build title ID mapping from historical data"
    )
    parser.add_argument(
        "--history", 
        default="data/history_city_sales.csv",
        help="Path to history_city_sales.csv"
    )
    parser.add_argument(
        "--baselines", 
        default="data/baselines.csv",
        help="Path to baselines.csv"
    )
    parser.add_argument(
        "--output", 
        default=DEFAULT_TITLE_MAP_PATH,
        help="Output path for title_id_map.csv"
    )
    parser.add_argument(
        "--threshold", 
        type=int, 
        default=85,
        help="Fuzzy match threshold (0-100)"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Don't write output, just show report"
    )
    
    args = parser.parse_args()
    
    mapping, ambiguous = build_title_map_from_hist(
        history_path=args.history,
        baseline_path=args.baselines,
        output_path=args.output,
        threshold=args.threshold,
        dry_run=args.dry_run
    )
