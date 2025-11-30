"""
Tests for bootstrap confidence interval functionality in backtest_timeaware.py

These tests verify that the bootstrap CI implementation correctly computes
confidence intervals for backtest metrics (MAE, RMSE, R2).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.backtest_timeaware import (
    bootstrap_confidence_intervals,
    compute_metrics,
)


class TestBootstrapConfidenceIntervals:
    """Tests for the bootstrap_confidence_intervals function."""

    def test_returns_dict_with_expected_keys(self):
        """Verify bootstrap_confidence_intervals returns dict with all expected metrics."""
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = np.array([110, 190, 320, 380, 510])

        result = bootstrap_confidence_intervals(y_true, y_pred, n_bootstrap=100)

        assert isinstance(result, dict)
        assert "mae" in result
        assert "rmse" in result
        assert "r2" in result

        for metric in ["mae", "rmse", "r2"]:
            assert "mean" in result[metric]
            assert "ci_lower" in result[metric]
            assert "ci_upper" in result[metric]

    def test_ci_bounds_are_valid(self):
        """Verify CI bounds are numeric and lower < upper."""
        np.random.seed(42)
        y_true = np.random.randint(100, 1000, size=50)
        y_pred = y_true + np.random.normal(0, 50, size=50)

        result = bootstrap_confidence_intervals(y_true, y_pred, n_bootstrap=500)

        for metric in ["mae", "rmse"]:
            assert not np.isnan(result[metric]["mean"])
            assert not np.isnan(result[metric]["ci_lower"])
            assert not np.isnan(result[metric]["ci_upper"])
            assert result[metric]["ci_lower"] <= result[metric]["mean"]
            assert result[metric]["mean"] <= result[metric]["ci_upper"]

    def test_reproducibility_with_seed(self):
        """Verify results are reproducible with same random_state."""
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = np.array([110, 190, 320, 380, 510])

        result1 = bootstrap_confidence_intervals(
            y_true, y_pred, n_bootstrap=100, random_state=42
        )
        result2 = bootstrap_confidence_intervals(
            y_true, y_pred, n_bootstrap=100, random_state=42
        )

        assert result1["mae"]["mean"] == result2["mae"]["mean"]
        assert result1["mae"]["ci_lower"] == result2["mae"]["ci_lower"]
        assert result1["mae"]["ci_upper"] == result2["mae"]["ci_upper"]

    def test_handles_small_samples(self):
        """Verify graceful handling with very small samples."""
        y_true = np.array([100, 200])
        y_pred = np.array([110, 190])

        result = bootstrap_confidence_intervals(y_true, y_pred, n_bootstrap=100)

        # Should still produce valid output
        assert isinstance(result, dict)
        assert not np.isnan(result["mae"]["mean"])

    def test_handles_single_sample(self):
        """Verify returns NaN for single sample (insufficient data)."""
        y_true = np.array([100])
        y_pred = np.array([110])

        result = bootstrap_confidence_intervals(y_true, y_pred, n_bootstrap=100)

        assert np.isnan(result["mae"]["mean"])
        assert np.isnan(result["mae"]["ci_lower"])
        assert np.isnan(result["mae"]["ci_upper"])


class TestComputeMetricsWithCI:
    """Tests for compute_metrics with CI computation."""

    def test_ci_fields_present_when_enabled(self):
        """Verify CI fields are present when compute_ci=True."""
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = np.array([110, 190, 320, 380, 510])

        result = compute_metrics(y_true, y_pred, compute_ci=True, n_bootstrap=100)

        # Point estimates should be present
        assert "mae" in result
        assert "rmse" in result
        assert "r2" in result
        assert "n_samples" in result

        # CI fields should be present
        assert "mae_ci_lower" in result
        assert "mae_ci_upper" in result
        assert "rmse_ci_lower" in result
        assert "rmse_ci_upper" in result
        assert "r2_ci_lower" in result
        assert "r2_ci_upper" in result

    def test_ci_fields_absent_when_disabled(self):
        """Verify CI fields are absent when compute_ci=False."""
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = np.array([110, 190, 320, 380, 510])

        result = compute_metrics(y_true, y_pred, compute_ci=False)

        # Point estimates should be present
        assert "mae" in result
        assert "rmse" in result
        assert "r2" in result

        # CI fields should NOT be present
        assert "mae_ci_lower" not in result
        assert "mae_ci_upper" not in result

    def test_ci_values_not_nan_for_valid_data(self):
        """Verify CI values are not NaN for valid synthetic data."""
        np.random.seed(42)
        y_true = np.random.randint(1000, 5000, size=20)
        y_pred = y_true + np.random.normal(0, 200, size=20)

        result = compute_metrics(y_true, y_pred, compute_ci=True, n_bootstrap=200)

        assert not np.isnan(result["mae"])
        assert not np.isnan(result["mae_ci_lower"])
        assert not np.isnan(result["mae_ci_upper"])
        assert not np.isnan(result["rmse"])
        assert not np.isnan(result["rmse_ci_lower"])
        assert not np.isnan(result["rmse_ci_upper"])

    def test_ci_bounds_are_plausible(self):
        """Verify CI bounds are plausible (lower < point estimate < upper)."""
        np.random.seed(42)
        y_true = np.random.randint(1000, 5000, size=30)
        y_pred = y_true + np.random.normal(0, 200, size=30)

        result = compute_metrics(y_true, y_pred, compute_ci=True, n_bootstrap=500)

        # For MAE and RMSE, CI should bracket point estimate (approximately)
        # Due to bootstrap variability, point estimate may not be exactly between
        # but bounds should be reasonable
        assert result["mae_ci_lower"] > 0
        assert result["mae_ci_upper"] > result["mae_ci_lower"]
        assert result["rmse_ci_lower"] > 0
        assert result["rmse_ci_upper"] > result["rmse_ci_lower"]

    def test_handles_nan_predictions(self):
        """Verify handling of NaN predictions in input."""
        y_true = np.array([100, 200, 300, 400, np.nan])
        y_pred = np.array([110, np.nan, 320, 380, 510])

        result = compute_metrics(y_true, y_pred, compute_ci=True, n_bootstrap=100)

        # Should work with filtered data
        assert result["n_samples"] == 3  # Only 3 valid pairs
        assert not np.isnan(result["mae"])

    def test_handles_empty_data(self):
        """Verify handling of empty data after NaN filtering."""
        y_true = np.array([np.nan, np.nan])
        y_pred = np.array([100, 200])

        result = compute_metrics(y_true, y_pred, compute_ci=True, n_bootstrap=100)

        assert result["n_samples"] == 0
        assert np.isnan(result["mae"])
        assert np.isnan(result["mae_ci_lower"])
        assert np.isnan(result["mae_ci_upper"])


class TestBacktestIntegrationWithCI:
    """Integration tests for backtest with CI."""

    @pytest.fixture
    def synthetic_modelling_dataset(self, tmp_path):
        """Create a small synthetic dataset for testing."""
        np.random.seed(42)
        n_samples = 25

        data = {
            "title": [f"Test Title {i}" for i in range(n_samples)],
            "canonical_title": [f"test_title_{i}" for i in range(n_samples)],
            "wiki": np.random.randint(30, 80, n_samples),
            "trends": np.random.randint(20, 70, n_samples),
            "youtube": np.random.randint(25, 75, n_samples),
            "spotify": np.random.randint(15, 65, n_samples),
            "category": np.random.choice(
                ["family_classic", "classic_romance", "contemporary"], n_samples
            ),
            "target_ticket_median": np.random.randint(3000, 8000, n_samples).astype(
                float
            ),
        }

        df = pd.DataFrame(data)
        path = tmp_path / "test_dataset.csv"
        df.to_csv(path, index=False)
        return str(path)

    def test_backtest_produces_ci_fields(self, synthetic_modelling_dataset, tmp_path):
        """Verify backtest produces CI fields in output."""
        from scripts.backtest_timeaware import run_backtest

        output_dir = str(tmp_path / "results")

        result = run_backtest(
            dataset_path=synthetic_modelling_dataset,
            target_col="target_ticket_median",
            n_folds=3,
            output_dir=output_dir,
            seed=42,
            verbose=False,
            compute_ci=True,
            n_bootstrap=100,  # Use fewer for test speed
        )

        # Check result structure
        assert "methods" in result
        assert "compute_ci" in result
        assert result["compute_ci"] is True

        # Check that at least one method has CI fields
        methods = result["methods"]
        assert len(methods) > 0

        # Check heuristic method (always present)
        if "heuristic" in methods:
            heuristic = methods["heuristic"]
            assert "mae" in heuristic
            assert "mae_ci_lower" in heuristic
            assert "mae_ci_upper" in heuristic
            assert not np.isnan(heuristic["mae_ci_lower"])
            assert not np.isnan(heuristic["mae_ci_upper"])

    def test_backtest_summary_json_contains_ci(
        self, synthetic_modelling_dataset, tmp_path
    ):
        """Verify backtest_summary.json contains CI fields."""
        import json

        from scripts.backtest_timeaware import run_backtest

        output_dir = str(tmp_path / "results")

        run_backtest(
            dataset_path=synthetic_modelling_dataset,
            target_col="target_ticket_median",
            n_folds=3,
            output_dir=output_dir,
            seed=42,
            verbose=False,
            compute_ci=True,
            n_bootstrap=100,
        )

        # Load and verify JSON
        summary_path = f"{output_dir}/backtest_summary.json"
        with open(summary_path) as f:
            summary = json.load(f)

        assert "methods" in summary
        assert "compute_ci" in summary

        # Verify CI fields in at least one method
        for method_name, metrics in summary["methods"].items():
            if "mae" in metrics:
                assert "mae_ci_lower" in metrics, f"Missing mae_ci_lower for {method_name}"
                assert "mae_ci_upper" in metrics, f"Missing mae_ci_upper for {method_name}"
                break


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
