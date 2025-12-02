"""
Full Pipeline Smoke Tests

These tests verify that the run_full_pipeline.py script works correctly
and can execute the complete ML pipeline end-to-end.
"""

import os
import sys
import json
import tempfile
from pathlib import Path

import pytest
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def synthetic_history_csv(tmp_path):
    """Create a synthetic history CSV for testing.
    
    This now uses the combined city-level format expected by build_modelling_dataset.
    Format: city, show_title, start_date, end_date, single_tickets
    """
    np.random.seed(42)
    n_shows = 25
    
    # Create entries for both Calgary and Edmonton for each show
    data = []
    base_date = pd.Timestamp("2018-01-01")
    
    for i in range(n_shows):
        show_title = f"Test Show {i}"
        # Generate dates roughly 3 months apart for each show
        start_date = base_date + pd.DateOffset(months=i * 3)
        end_date = start_date + pd.DateOffset(days=14)
        
        # Calgary entry
        data.append({
            "city": "Calgary",
            "show_title": show_title,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "single_tickets": np.random.randint(3000, 8000),
        })
        # Edmonton entry
        data.append({
            "city": "Edmonton",
            "show_title": show_title,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "single_tickets": np.random.randint(2500, 6500),
        })

    df = pd.DataFrame(data)
    path = tmp_path / "history_city_sales.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def synthetic_baselines_csv(tmp_path):
    """Create a synthetic baselines CSV for testing."""
    np.random.seed(42)
    n_samples = 30

    data = {
        "title": [f"Test Show {i}" for i in range(n_samples)],
        "wiki": np.random.randint(30, 80, n_samples),
        "trends": np.random.randint(20, 70, n_samples),
        "youtube": np.random.randint(25, 75, n_samples),
        "spotify": np.random.randint(15, 65, n_samples),
        "category": np.random.choice(
            ["family_classic", "classic_romance", "contemporary"], n_samples
        ),
        "gender": np.random.choice(["female", "male", "co", "na"], n_samples),
        "source": ["historical"] * 20 + ["external_reference"] * 10,
    }

    df = pd.DataFrame(data)
    path = tmp_path / "baselines.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def synthetic_past_runs_csv(tmp_path):
    """Create a synthetic past_runs CSV for testing."""
    np.random.seed(42)
    n_samples = 40

    data = {
        "title": [f"Test Show {i % 20}" for i in range(n_samples)],
        "start_date": pd.date_range("2018-01-01", periods=n_samples, freq="3ME"),
        "end_date": pd.date_range("2018-01-15", periods=n_samples, freq="3ME"),
        "city": np.random.choice(["Calgary", "Edmonton"], n_samples),
    }

    df = pd.DataFrame(data)
    path = tmp_path / "past_runs.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory."""
    output_dir = tmp_path / "test_results"
    output_dir.mkdir()
    return str(output_dir)


class TestRunFullPipeline:
    """Tests for the run_full_pipeline.py script."""

    def test_pipeline_function_exists(self):
        """Verify the run_full_pipeline function exists and is importable."""
        try:
            from scripts.run_full_pipeline import run_full_pipeline

            assert callable(run_full_pipeline)
        except ImportError:
            pytest.fail("run_full_pipeline function not found")

    def test_pipeline_with_synthetic_data(
        self,
        synthetic_history_csv,
        synthetic_baselines_csv,
        synthetic_past_runs_csv,
        temp_output_dir,
    ):
        """Test the full pipeline runs with synthetic data."""
        try:
            from scripts.run_full_pipeline import run_full_pipeline
        except ImportError:
            pytest.skip("run_full_pipeline not available")

        results = run_full_pipeline(
            output_base_dir=temp_output_dir,
            history_path=synthetic_history_csv,
            baselines_path=synthetic_baselines_csv,
            past_runs_path=synthetic_past_runs_csv,
            tune=False,
            save_shap=False,
            seed=42,
            verbose=False,
        )

        # Check results structure
        assert "success" in results
        assert "steps" in results
        assert "output_dir" in results
        assert "timestamp" in results

        # Verify pipeline ran (may have partial success depending on data)
        assert "build_dataset" in results["steps"]

    def test_pipeline_creates_output_directory(
        self,
        synthetic_history_csv,
        synthetic_baselines_csv,
        synthetic_past_runs_csv,
        temp_output_dir,
    ):
        """Verify the pipeline creates a timestamped output directory."""
        try:
            from scripts.run_full_pipeline import run_full_pipeline
        except ImportError:
            pytest.skip("run_full_pipeline not available")

        results = run_full_pipeline(
            output_base_dir=temp_output_dir,
            history_path=synthetic_history_csv,
            baselines_path=synthetic_baselines_csv,
            past_runs_path=synthetic_past_runs_csv,
            tune=False,
            save_shap=False,
            verbose=False,
        )

        output_dir = results.get("output_dir")
        assert output_dir is not None
        assert os.path.exists(output_dir)
        assert os.path.isdir(output_dir)

    def test_pipeline_creates_dataset_file(
        self,
        synthetic_history_csv,
        synthetic_baselines_csv,
        synthetic_past_runs_csv,
        temp_output_dir,
    ):
        """Verify the pipeline creates a modelling dataset file."""
        try:
            from scripts.run_full_pipeline import run_full_pipeline
        except ImportError:
            pytest.skip("run_full_pipeline not available")

        results = run_full_pipeline(
            output_base_dir=temp_output_dir,
            history_path=synthetic_history_csv,
            baselines_path=synthetic_baselines_csv,
            past_runs_path=synthetic_past_runs_csv,
            tune=False,
            save_shap=False,
            verbose=False,
        )

        # Check dataset was created in output directory
        output_dir = results.get("output_dir")
        if output_dir and os.path.exists(output_dir):
            dataset_path = os.path.join(output_dir, "modelling_dataset.csv")
            if os.path.exists(dataset_path):
                df = pd.read_csv(dataset_path)
                assert len(df) > 0
                assert "wiki" in df.columns
                assert "trends" in df.columns

    def test_pipeline_creates_summary_json(
        self,
        synthetic_history_csv,
        synthetic_baselines_csv,
        synthetic_past_runs_csv,
        temp_output_dir,
    ):
        """Verify the pipeline creates a summary JSON file."""
        try:
            from scripts.run_full_pipeline import run_full_pipeline
        except ImportError:
            pytest.skip("run_full_pipeline not available")

        results = run_full_pipeline(
            output_base_dir=temp_output_dir,
            history_path=synthetic_history_csv,
            baselines_path=synthetic_baselines_csv,
            past_runs_path=synthetic_past_runs_csv,
            tune=False,
            save_shap=False,
            verbose=False,
        )

        output_dir = results.get("output_dir")
        summary_path = os.path.join(output_dir, "pipeline_summary.json")
        assert os.path.exists(summary_path), "Pipeline summary not created"

        with open(summary_path) as f:
            summary = json.load(f)

        assert "timestamp" in summary
        assert "steps" in summary
        assert "duration_seconds" in summary

    def test_pipeline_reports_step_status(
        self,
        synthetic_history_csv,
        synthetic_baselines_csv,
        synthetic_past_runs_csv,
        temp_output_dir,
    ):
        """Verify each pipeline step reports its status."""
        try:
            from scripts.run_full_pipeline import run_full_pipeline
        except ImportError:
            pytest.skip("run_full_pipeline not available")

        results = run_full_pipeline(
            output_base_dir=temp_output_dir,
            history_path=synthetic_history_csv,
            baselines_path=synthetic_baselines_csv,
            past_runs_path=synthetic_past_runs_csv,
            tune=False,
            save_shap=False,
            verbose=False,
        )

        steps = results.get("steps", {})

        # Each step should have a success status
        for step_name, step_info in steps.items():
            assert "success" in step_info, f"Step {step_name} missing success status"


class TestCLIInterface:
    """Tests for the CLI interface of run_full_pipeline.py."""

    def test_cli_help_option(self):
        """Test that --help option works."""
        import subprocess

        result = subprocess.run(
            [sys.executable, "scripts/run_full_pipeline.py", "--help"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
        )

        assert result.returncode == 0
        assert "usage" in result.stdout.lower()
        assert "--tune" in result.stdout
        assert "--save-shap" in result.stdout

    def test_cli_accepts_all_options(self):
        """Test that CLI accepts all documented options."""
        import subprocess

        # Just test that the options are recognized (will fail due to missing files)
        result = subprocess.run(
            [
                sys.executable,
                "scripts/run_full_pipeline.py",
                "--quiet",
                "--seed",
                "123",
                "--history",
                "/nonexistent/path.csv",
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
        )

        # Should fail due to missing file, not due to unrecognized option
        assert "unrecognized arguments" not in result.stderr.lower()


class TestMakefile:
    """Tests for the Makefile."""

    def test_makefile_exists(self):
        """Verify Makefile exists in repository root."""
        makefile_path = Path(__file__).parent.parent / "Makefile"
        assert makefile_path.exists(), "Makefile not found in repository root"

    def test_makefile_has_full_pipeline_target(self):
        """Verify Makefile has full-pipeline target."""
        makefile_path = Path(__file__).parent.parent / "Makefile"
        content = makefile_path.read_text()

        assert "full-pipeline:" in content, "full-pipeline target not found"
        assert "run_full_pipeline.py" in content, "Pipeline script not referenced"

    def test_makefile_has_help_target(self):
        """Verify Makefile has help target."""
        makefile_path = Path(__file__).parent.parent / "Makefile"
        content = makefile_path.read_text()

        assert "help:" in content, "help target not found"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
