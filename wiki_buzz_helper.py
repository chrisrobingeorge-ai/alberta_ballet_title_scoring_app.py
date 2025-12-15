"""
wiki_buzz_helper.py

A script to calculate normalized Wikipedia buzz scores for ballet titles
based on total pageviews fetched from the Wikimedia REST API.
"""

import argparse
import os
import requests
import pandas as pd
from datetime import datetime
from typing import Dict

# Get optional Wikipedia API credentials from environment variables
WIKI_API_KEY = os.getenv("WIKI_API_KEY")
WIKI_API_SECRET = os.getenv("WIKI_API_SECRET")


def _get_wiki_headers() -> Dict[str, str]:
    """
    Build Wikipedia API request headers with optional authentication.
    Includes required User-Agent and optional API key/secret.
    """
    headers = {
        "User-Agent": (
            "TitleScoringApp/1.0 "
            "(https://github.com/chrisrobingeorge-ai/"
            "alberta_ballet_title_scoring_app)"
        )
    }
    # Add optional authentication headers if credentials are provided
    if WIKI_API_KEY:
        headers["X-API-Key"] = WIKI_API_KEY
    if WIKI_API_SECRET:
        headers["X-API-Secret"] = WIKI_API_SECRET
    return headers


def fetch_wikipedia_views_sum(title: str, start_date: str, end_date: str) -> int:
    """Fetch total Wikipedia views for a title between start_date and end_date (YYYYMMDD)."""
    title_encoded = title.replace(" ", "_")
    url = (
        f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
        f"en.wikipedia/all-access/user/{title_encoded}/daily/{start_date}/{end_date}"
    )
    try:
        response = requests.get(url, headers=_get_wiki_headers(), timeout=10)
        response.raise_for_status()
        data = response.json()
        return sum(item["views"] for item in data.get("items", []))
    except Exception as e:
        print(f"Error fetching views for {title}: {e}")
        return 0

def calculate_scores(titles, benchmark_title):
    today = datetime.today()
    start_date = f"{today.year}0101"
    end_date = today.strftime("%Y%m%d")

    benchmark_views = fetch_wikipedia_views_sum(benchmark_title, start_date, end_date)
    if benchmark_views == 0:
        raise ValueError("Benchmark title has zero pageviews. Cannot normalize.")

    results = []
    for title in titles:
        views = fetch_wikipedia_views_sum(title, start_date, end_date)
        score = (views / benchmark_views) * 100 if benchmark_views > 0 else 0
        results.append({
            "title": title,
            "raw_views": views,
            "wiki_score": round(score, 1),
            "benchmark_used": benchmark_title
        })

    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(description="Calculate normalized Wikipedia buzz scores.")
    parser.add_argument("--titles", type=str, help="Comma-separated list of titles to score.")
    parser.add_argument("--input_csv", type=str, help="Path to CSV with a 'title' column.")
    parser.add_argument("--benchmark", type=str, default="Cinderella", help="Benchmark title (default: Cinderella)")
    parser.add_argument("--append", type=str, help="Optional path to append results to an existing baselines.csv")

    args = parser.parse_args()

    if args.titles:
        titles = [title.strip() for title in args.titles.split(",")]
    elif args.input_csv:
        df_input = pd.read_csv(args.input_csv)
        if "title" not in df_input.columns:
            raise ValueError("CSV must contain a 'title' column.")
        titles = df_input["title"].dropna().tolist()
    else:
        raise ValueError("Please provide --titles or --input_csv.")

    df_results = calculate_scores(titles, args.benchmark)

    if args.append:
        try:
            df_existing = pd.read_csv(args.append)
            df_combined = pd.concat([df_existing, df_results], ignore_index=True)
            df_combined.to_csv(args.append, index=False)
            print(f"Appended results to {args.append}")
        except Exception as e:
            print(f"Failed to append to {args.append}: {e}")
    else:
        output_path = "wiki_buzz_scores.csv"
        df_results.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
