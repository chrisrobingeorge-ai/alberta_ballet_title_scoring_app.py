"""
Data quality utilities for feature range validation.

This module provides functions to check feature values against expected ranges
defined in the feature inventory CSV. Out-of-range values are flagged and logged.
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

from config.registry import load_feature_inventory

logger = logging.getLogger(__name__)


class DataQualityWarning(UserWarning):
    """Warning raised when data quality issues are detected."""
    pass


class DataQualityError(ValueError):
    """Exception raised when severe data quality violations occur."""
    pass


def load_feature_ranges() -> pd.DataFrame:
    """Load feature inventory and return features with defined ranges.
    
    Returns:
        DataFrame with columns: Feature Name, expected_min, expected_max, unit
        Only includes features that have both expected_min and expected_max defined.
    """
    inventory = load_feature_inventory()
    
    # Filter to features with defined ranges (non-empty min and max)
    has_min = inventory['expected_min'].notna() & (inventory['expected_min'] != '')
    has_max = inventory['expected_max'].notna() & (inventory['expected_max'] != '')
    
    ranged_features = inventory[has_min & has_max].copy()
    
    # Convert min/max to numeric
    ranged_features['expected_min'] = pd.to_numeric(
        ranged_features['expected_min'], errors='coerce'
    )
    ranged_features['expected_max'] = pd.to_numeric(
        ranged_features['expected_max'], errors='coerce'
    )
    
    return ranged_features[['Feature Name', 'expected_min', 'expected_max', 'unit']]


def check_feature_ranges(
    df: pd.DataFrame,
    inventory: Optional[pd.DataFrame] = None,
    strict: bool = False,
    violation_threshold: float = 0.1,
) -> Dict[str, Dict]:
    """Check if feature values fall within expected ranges defined in inventory.
    
    For each feature with defined expected_min/expected_max in the inventory:
    - Counts values that fall outside the expected range
    - Logs warnings for features with out-of-range values
    - Raises DataQualityError if strict=True and violations exceed threshold
    
    Args:
        df: DataFrame with feature columns to check.
        inventory: Optional DataFrame with feature ranges. If None, loads from
            the feature inventory CSV. Must have columns:
            'Feature Name', 'expected_min', 'expected_max'.
        strict: If True, raises DataQualityError when violation_threshold is exceeded.
        violation_threshold: Fraction of rows that can be out-of-range before
            raising an error in strict mode (default 0.1 = 10%).
    
    Returns:
        Dictionary mapping feature names to violation details:
        {
            'feature_name': {
                'below_min': int,        # count of values below expected_min
                'above_max': int,        # count of values above expected_max
                'total_violations': int, # total out-of-range values
                'total_valid': int,      # count of non-null values checked
                'violation_rate': float, # fraction of values out-of-range
                'expected_min': float,   # the expected minimum
                'expected_max': float,   # the expected maximum
                'actual_min': float,     # actual minimum in data
                'actual_max': float,     # actual maximum in data
            },
            ...
        }
    
    Raises:
        DataQualityError: If strict=True and any feature has violation_rate
            exceeding violation_threshold.
    
    Examples:
        >>> df = pd.DataFrame({'load_factor': [0.5, 0.8, 1.5, -0.1]})
        >>> results = check_feature_ranges(df)
        >>> # Will log warnings for load_factor having values outside [0, 1]
    """
    if inventory is None:
        inventory = load_feature_ranges()
    
    results = {}
    severe_violations = []
    
    for _, row in inventory.iterrows():
        feature_name = row['Feature Name']
        expected_min = row['expected_min']
        expected_max = row['expected_max']
        
        # Skip if feature not in DataFrame
        if feature_name not in df.columns:
            continue
        
        # Skip if min/max are not valid numbers
        if pd.isna(expected_min) or pd.isna(expected_max):
            continue
        
        # Get the feature column (numeric only)
        feature_values = pd.to_numeric(df[feature_name], errors='coerce')
        valid_values = feature_values.dropna()
        
        if len(valid_values) == 0:
            continue
        
        # Count violations
        below_min = (valid_values < expected_min).sum()
        above_max = (valid_values > expected_max).sum()
        total_violations = below_min + above_max
        total_valid = len(valid_values)
        violation_rate = total_violations / total_valid if total_valid > 0 else 0.0
        
        result = {
            'below_min': int(below_min),
            'above_max': int(above_max),
            'total_violations': int(total_violations),
            'total_valid': int(total_valid),
            'violation_rate': float(violation_rate),
            'expected_min': float(expected_min),
            'expected_max': float(expected_max),
            'actual_min': float(valid_values.min()),
            'actual_max': float(valid_values.max()),
        }
        
        # Only include features with violations in the result
        if total_violations > 0:
            results[feature_name] = result
            
            # Log warning
            warning_msg = (
                f"Feature '{feature_name}' has {total_violations} out-of-range values "
                f"({violation_rate:.1%}): expected [{expected_min}, {expected_max}], "
                f"actual range [{valid_values.min():.2f}, {valid_values.max():.2f}]"
            )
            logger.warning(warning_msg)
            warnings.warn(warning_msg, DataQualityWarning)
            
            # Track severe violations
            if violation_rate > violation_threshold:
                severe_violations.append((feature_name, violation_rate))
    
    # In strict mode, raise error if severe violations exist
    if strict and severe_violations:
        features_msg = ", ".join(
            f"{name} ({rate:.1%})" for name, rate in severe_violations
        )
        raise DataQualityError(
            f"Severe data quality violations detected. Features with >{violation_threshold:.0%} "
            f"out-of-range values: {features_msg}"
        )
    
    return results


def validate_data_quality(
    df: pd.DataFrame,
    raise_on_violations: bool = False,
) -> Tuple[bool, Dict[str, Dict]]:
    """High-level data quality validation function.
    
    Runs all data quality checks and returns a summary.
    
    Args:
        df: DataFrame to validate.
        raise_on_violations: If True, raises DataQualityError on any violations.
    
    Returns:
        Tuple of (is_valid, violations_dict) where:
        - is_valid: True if no violations were found
        - violations_dict: Dictionary of violations from check_feature_ranges
    """
    violations = check_feature_ranges(df, strict=raise_on_violations)
    is_valid = len(violations) == 0
    
    if is_valid:
        logger.info("Data quality check passed: all features within expected ranges")
    else:
        logger.warning(
            f"Data quality check found issues: {len(violations)} features with out-of-range values"
        )
    
    return is_valid, violations


def get_feature_range(feature_name: str) -> Optional[Tuple[float, float, str]]:
    """Get the expected range for a specific feature.
    
    Args:
        feature_name: Name of the feature to look up.
    
    Returns:
        Tuple of (expected_min, expected_max, unit) or None if not defined.
    """
    inventory = load_feature_ranges()
    match = inventory[inventory['Feature Name'] == feature_name]
    
    if len(match) == 0:
        return None
    
    row = match.iloc[0]
    return (row['expected_min'], row['expected_max'], row['unit'])
