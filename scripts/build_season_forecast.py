#!/usr/bin/env python3
"""
Build season forecast from a slate CSV using the ML model.

Usage:
    python scripts/build_season_forecast.py \
        --input data/season_slate_2026.csv \
        --output outputs/season_forecast_2026.csv \
        --confidence 0.8
"""

import argparse
from pathlib import Path
import pandas as pd

from ml.scoring import score_runs_for_planning


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score a season slate CSV with ticket forecasts and intervals."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input CSV containing proposed runs.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output CSV to write forecasts.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.8,
        help="Prediction interval confidence level (e.g. 0.8 for 80%%).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    df_runs = pd.read_csv(input_path)

    df_scored = score_runs_for_planning(
        df_runs,
        confidence_level=args.confidence,
        n_bootstrap=200,
        model=None,
        attach_context=False,
        economic_context=None,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_scored.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()