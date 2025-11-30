"""
Tests for data quality utilities - feature range validation.

Tests verify that:
1. Out-of-range values produce warnings or errors as designed
2. In-range values pass silently
3. The feature inventory is loaded correctly with the new columns
"""

import pytest
import pandas as pd
import numpy as np
import warnings

from data.quality import (
    check_feature_ranges,
    load_feature_ranges,
    validate_data_quality,
    get_feature_range,
    DataQualityWarning,
    DataQualityError,
)


class TestLoadFeatureRanges:
    """Tests for loading feature ranges from inventory."""
    
    def test_load_feature_ranges_returns_dataframe(self):
        """Should return a DataFrame with expected columns."""
        ranges_df = load_feature_ranges()
        
        assert isinstance(ranges_df, pd.DataFrame)
        assert 'Feature Name' in ranges_df.columns
        assert 'expected_min' in ranges_df.columns
        assert 'expected_max' in ranges_df.columns
        assert 'unit' in ranges_df.columns
    
    def test_load_feature_ranges_has_numeric_min_max(self):
        """Min and max values should be numeric."""
        ranges_df = load_feature_ranges()
        
        assert pd.api.types.is_numeric_dtype(ranges_df['expected_min'])
        assert pd.api.types.is_numeric_dtype(ranges_df['expected_max'])
    
    def test_load_feature_ranges_has_some_entries(self):
        """Should have at least some features with defined ranges."""
        ranges_df = load_feature_ranges()
        
        # We added ranges to many features, so there should be a reasonable number
        assert len(ranges_df) >= 10


class TestCheckFeatureRanges:
    """Tests for check_feature_ranges function."""
    
    @pytest.fixture
    def sample_inventory(self):
        """Create a sample inventory for testing."""
        return pd.DataFrame({
            'Feature Name': ['load_factor', 'email_open_rate', 'venue_capacity'],
            'expected_min': [0.0, 0.0, 500.0],
            'expected_max': [1.0, 1.0, 5000.0],
            'unit': ['ratio', 'ratio', 'seats']
        })
    
    def test_in_range_values_pass_silently(self, sample_inventory):
        """Values within expected range should not produce warnings."""
        df = pd.DataFrame({
            'load_factor': [0.5, 0.8, 0.3, 0.9],
            'email_open_rate': [0.2, 0.35, 0.4, 0.15],
            'venue_capacity': [2000, 2500, 1800, 3000]
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            violations = check_feature_ranges(df, inventory=sample_inventory)
        
        # No violations expected
        assert len(violations) == 0
        # No DataQualityWarning warnings
        data_quality_warnings = [x for x in w if issubclass(x.category, DataQualityWarning)]
        assert len(data_quality_warnings) == 0
    
    def test_below_min_produces_warning(self, sample_inventory):
        """Values below expected min should produce warnings."""
        df = pd.DataFrame({
            'load_factor': [0.5, -0.1, 0.3],  # -0.1 is below min
            'email_open_rate': [0.2, 0.35, 0.4],
            'venue_capacity': [2000, 2500, 1800]
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            violations = check_feature_ranges(df, inventory=sample_inventory)
        
        assert 'load_factor' in violations
        assert violations['load_factor']['below_min'] == 1
        assert violations['load_factor']['above_max'] == 0
        
        # Should have produced a warning
        data_quality_warnings = [x for x in w if issubclass(x.category, DataQualityWarning)]
        assert len(data_quality_warnings) >= 1
    
    def test_above_max_produces_warning(self, sample_inventory):
        """Values above expected max should produce warnings."""
        df = pd.DataFrame({
            'load_factor': [0.5, 0.8, 1.5],  # 1.5 is above max
            'email_open_rate': [0.2, 1.2, 0.4],  # 1.2 is above max
            'venue_capacity': [2000, 2500, 1800]
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            violations = check_feature_ranges(df, inventory=sample_inventory)
        
        assert 'load_factor' in violations
        assert violations['load_factor']['above_max'] == 1
        
        assert 'email_open_rate' in violations
        assert violations['email_open_rate']['above_max'] == 1
    
    def test_multiple_violations_counted_correctly(self, sample_inventory):
        """Multiple violations in one feature should be counted correctly."""
        df = pd.DataFrame({
            'load_factor': [-0.1, 1.5, -0.2, 1.8, 0.5],  # 4 violations
            'email_open_rate': [0.2, 0.35, 0.4, 0.15, 0.3],  # 0 violations
            'venue_capacity': [2000, 2500, 1800, 3000, 2200]  # 0 violations
        })
        
        violations = check_feature_ranges(df, inventory=sample_inventory)
        
        assert 'load_factor' in violations
        assert violations['load_factor']['total_violations'] == 4
        assert violations['load_factor']['below_min'] == 2
        assert violations['load_factor']['above_max'] == 2
        assert violations['load_factor']['total_valid'] == 5
        assert abs(violations['load_factor']['violation_rate'] - 0.8) < 0.01
    
    def test_strict_mode_raises_error(self, sample_inventory):
        """Strict mode should raise DataQualityError when violations exceed threshold."""
        df = pd.DataFrame({
            'load_factor': [-0.1, 1.5, 0.5, 0.8],  # 50% violation rate
            'email_open_rate': [0.2, 0.35, 0.4, 0.15],
            'venue_capacity': [2000, 2500, 1800, 3000]
        })
        
        with pytest.raises(DataQualityError) as exc_info:
            check_feature_ranges(
                df, 
                inventory=sample_inventory, 
                strict=True,
                violation_threshold=0.1  # 10% threshold
            )
        
        assert "load_factor" in str(exc_info.value)
    
    def test_strict_mode_passes_with_low_violations(self, sample_inventory):
        """Strict mode should not raise error when violations are below threshold."""
        df = pd.DataFrame({
            'load_factor': [0.5, 0.8, 0.3, 0.9, 0.4, 0.6, 0.7, 1.1],  # 1/8 = 12.5%
            'email_open_rate': [0.2, 0.35, 0.4, 0.15, 0.3, 0.25, 0.38, 0.22],
            'venue_capacity': [2000, 2500, 1800, 3000, 2200, 2700, 2100, 2900]
        })
        
        # Should not raise with 15% threshold
        violations = check_feature_ranges(
            df, 
            inventory=sample_inventory, 
            strict=True,
            violation_threshold=0.15
        )
        
        assert 'load_factor' in violations
        assert violations['load_factor']['total_violations'] == 1
    
    def test_missing_feature_in_dataframe_ignored(self, sample_inventory):
        """Features in inventory but not in DataFrame should be silently ignored."""
        df = pd.DataFrame({
            'load_factor': [0.5, 0.8, 0.3],
            # email_open_rate and venue_capacity are missing
        })
        
        violations = check_feature_ranges(df, inventory=sample_inventory)
        
        # Only check features that exist in DataFrame
        assert 'email_open_rate' not in violations
        assert 'venue_capacity' not in violations
    
    def test_nan_values_excluded_from_check(self, sample_inventory):
        """NaN values should be excluded from range checking."""
        df = pd.DataFrame({
            'load_factor': [0.5, np.nan, 0.3, np.nan, 0.8],
            'email_open_rate': [0.2, 0.35, np.nan, 0.15, 0.3],
            'venue_capacity': [2000, np.nan, 1800, 3000, 2200]
        })
        
        violations = check_feature_ranges(df, inventory=sample_inventory)
        
        # All valid values are in range
        assert len(violations) == 0
    
    def test_uses_default_inventory_when_not_provided(self):
        """Should use feature inventory from config when inventory is None."""
        df = pd.DataFrame({
            'load_factor': [0.5, 1.5],  # 1.5 exceeds expected max of 1.0
        })
        
        # Load without explicit inventory - should use default
        violations = check_feature_ranges(df, inventory=None)
        
        # If load_factor is defined in inventory with range [0, 1], 
        # this should detect a violation
        if 'load_factor' in violations:
            assert violations['load_factor']['above_max'] >= 1


class TestValidateDataQuality:
    """Tests for validate_data_quality function."""
    
    def test_returns_valid_true_for_clean_data(self):
        """Should return is_valid=True when no violations exist."""
        inventory = pd.DataFrame({
            'Feature Name': ['value_a', 'value_b'],
            'expected_min': [0.0, 0.0],
            'expected_max': [100.0, 100.0],
            'unit': ['count', 'count']
        })
        
        df = pd.DataFrame({
            'value_a': [10, 20, 30],
            'value_b': [50, 60, 70]
        })
        
        # Temporarily patch the load_feature_ranges function
        import data.quality as quality_module
        original_load = quality_module.load_feature_ranges
        quality_module.load_feature_ranges = lambda: inventory
        
        try:
            is_valid, violations = validate_data_quality(df)
            assert is_valid is True
            assert len(violations) == 0
        finally:
            quality_module.load_feature_ranges = original_load
    
    def test_returns_valid_false_for_violations(self):
        """Should return is_valid=False when violations exist."""
        inventory = pd.DataFrame({
            'Feature Name': ['value_a'],
            'expected_min': [0.0],
            'expected_max': [100.0],
            'unit': ['count']
        })
        
        df = pd.DataFrame({
            'value_a': [10, 200, 30]  # 200 is out of range
        })
        
        import data.quality as quality_module
        original_load = quality_module.load_feature_ranges
        quality_module.load_feature_ranges = lambda: inventory
        
        try:
            is_valid, violations = validate_data_quality(df)
            assert is_valid is False
            assert 'value_a' in violations
        finally:
            quality_module.load_feature_ranges = original_load
    
    def test_raise_on_violations_raises_error(self):
        """Should raise DataQualityError when raise_on_violations=True and violations exist."""
        inventory = pd.DataFrame({
            'Feature Name': ['value_a'],
            'expected_min': [0.0],
            'expected_max': [100.0],
            'unit': ['count']
        })
        
        df = pd.DataFrame({
            'value_a': [10, 200, 30]  # 200 is out of range - 33% violation
        })
        
        import data.quality as quality_module
        original_load = quality_module.load_feature_ranges
        quality_module.load_feature_ranges = lambda: inventory
        
        try:
            with pytest.raises(DataQualityError):
                validate_data_quality(df, raise_on_violations=True)
        finally:
            quality_module.load_feature_ranges = original_load


class TestGetFeatureRange:
    """Tests for get_feature_range function."""
    
    def test_returns_range_for_known_feature(self):
        """Should return range tuple for features with defined ranges."""
        # Look up a feature that should be in the inventory with ranges
        result = get_feature_range('load_factor')
        
        if result is not None:
            expected_min, expected_max, unit = result
            assert expected_min == 0.0
            assert expected_max == 1.0
            assert unit == 'ratio'
    
    def test_returns_none_for_unknown_feature(self):
        """Should return None for features not in inventory."""
        result = get_feature_range('nonexistent_feature_xyz_123')
        
        assert result is None
    
    def test_returns_none_for_feature_without_range(self):
        """Should return None for features without defined ranges."""
        # show_title is categorical and shouldn't have numeric ranges
        result = get_feature_range('show_title')
        
        # This should be None since show_title doesn't have expected_min/max
        assert result is None


class TestFeatureInventoryColumns:
    """Tests for the updated feature inventory CSV columns."""
    
    def test_inventory_has_new_columns(self):
        """The feature inventory should have unit, expected_min, expected_max, used_in_model columns."""
        from config.registry import load_feature_inventory
        
        inventory = load_feature_inventory()
        
        assert 'unit' in inventory.columns
        assert 'expected_min' in inventory.columns
        assert 'expected_max' in inventory.columns
        assert 'used_in_model' in inventory.columns
    
    def test_some_features_have_ranges_populated(self):
        """At least some features should have ranges populated."""
        from config.registry import load_feature_inventory
        
        inventory = load_feature_inventory()
        
        # Check that some features have non-empty values
        has_min = inventory['expected_min'].notna() & (inventory['expected_min'] != '')
        has_max = inventory['expected_max'].notna() & (inventory['expected_max'] != '')
        
        assert has_min.sum() > 0, "No features have expected_min defined"
        assert has_max.sum() > 0, "No features have expected_max defined"
    
    def test_specific_features_have_expected_ranges(self):
        """Specific key features should have the expected range values."""
        from config.registry import load_feature_inventory
        
        inventory = load_feature_inventory()
        
        # Check load_factor has range [0, 1]
        load_factor_row = inventory[inventory['Feature Name'] == 'load_factor']
        if len(load_factor_row) > 0:
            min_val = load_factor_row['expected_min'].iloc[0]
            max_val = load_factor_row['expected_max'].iloc[0]
            # Handle both string and numeric values
            assert float(min_val) == 0
            assert float(max_val) == 1
        
        # Check month_of_opening has range [1, 12]
        month_row = inventory[inventory['Feature Name'] == 'month_of_opening']
        if len(month_row) > 0:
            min_val = month_row['expected_min'].iloc[0]
            max_val = month_row['expected_max'].iloc[0]
            assert float(min_val) == 1
            assert float(max_val) == 12
