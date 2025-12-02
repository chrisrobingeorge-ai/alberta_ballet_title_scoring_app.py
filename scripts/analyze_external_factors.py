#!/usr/bin/env python3
"""
Analyze External Factors vs Ticket Performance

This script examines how external economic and audience factors correlate
with ticket sales performance over time, independent of the ML model.

Usage:
    python scripts/analyze_external_factors.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')

# Set up paths
REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

# Ensure output directories exist
RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

# Configure plotting
plt.style.use('ggplot')


def load_modelling_dataset() -> pd.DataFrame:
    """Load the modelling dataset."""
    dataset_path = DATA_DIR / "modelling_dataset.csv"
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    df = pd.read_csv(dataset_path)
    print(f"Loaded {len(df)} rows from {dataset_path}")
    print(f"Columns: {', '.join(df.columns.tolist())}")
    
    return df


def infer_temporal_dimensions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Infer or reconstruct temporal dimensions (year, month, year_month).
    
    Since the dataset doesn't have explicit dates but has historical shows
    with tickets, we'll use a proxy approach:
    - Use years_since_last_run and prior_total_tickets to estimate relative time
    - For shows with history (prior_total_tickets > 0), estimate they ran in recent years
    - Default year approximation: 2024 minus years_since_last_run for shows with history
    - For shows without history, assign to a future planning year (2025)
    """
    df = df.copy()
    
    # Extract month from month_of_opening (handle empty values)
    df['month'] = pd.to_numeric(df['month_of_opening'], errors='coerce')
    
    # Estimate year based on historical context
    # If show has prior tickets, estimate it ran in past
    # Default to 2024 as baseline year
    current_year = 2024
    
    def estimate_year(row):
        if pd.notna(row.get('years_since_last_run')) and row.get('prior_total_tickets', 0) > 0:
            # This show ran years_since_last_run years ago
            years_back = row['years_since_last_run']
            estimated_year = current_year - int(years_back)
            return max(2015, min(current_year, estimated_year))  # Clamp to reasonable range
        elif row.get('prior_total_tickets', 0) > 0:
            # Has history but no clear date - assume recent (2020-2024)
            return 2022  # Mid-range estimate
        else:
            # No history - likely a future/planned show
            return 2025
    
    df['year'] = df.apply(estimate_year, axis=1)
    
    # Create year_month string (YYYY-MM format)
    df['year_month'] = df.apply(
        lambda row: f"{int(row['year'])}-{int(row['month']):02d}" 
        if pd.notna(row['month']) else f"{int(row['year'])}-01",
        axis=1
    )
    
    print(f"\nTemporal dimension inference:")
    print(f"  Year range: {df['year'].min():.0f} - {df['year'].max():.0f}")
    print(f"  Shows with month data: {df['month'].notna().sum()}/{len(df)}")
    print(f"  Shows with prior tickets (historical): {(df['prior_total_tickets'] > 0).sum()}")
    
    return df


def aggregate_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate external factors and tickets by year."""
    
    # Only include shows with actual ticket sales for meaningful aggregation
    df_with_tickets = df[df['target_ticket_median'] > 0].copy()
    
    if len(df_with_tickets) == 0:
        print("Warning: No shows with target_ticket_median > 0 for yearly aggregation")
        return pd.DataFrame()
    
    # Define aggregation columns
    agg_dict = {
        'target_ticket_median': ['count', 'median', 'mean', 'std'],
        'consumer_confidence_prairies': 'mean',
        'energy_index': 'mean',
        'inflation_adjustment_factor': 'mean',
        'city_median_household_income': 'mean',
    }
    
    # Add aud__engagement_factor if present
    if 'aud__engagement_factor' in df_with_tickets.columns:
        agg_dict['aud__engagement_factor'] = 'mean'
    
    # Group by year
    yearly = df_with_tickets.groupby('year').agg(agg_dict).reset_index()
    
    # Flatten column names
    yearly.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                      for col in yearly.columns.values]
    
    # Rename for clarity
    yearly.rename(columns={
        'target_ticket_median_count': 'show_count',
        'target_ticket_median_median': 'median_tickets',
        'target_ticket_median_mean': 'mean_tickets',
        'target_ticket_median_std': 'std_tickets',
        'consumer_confidence_prairies_mean': 'avg_consumer_confidence',
        'energy_index_mean': 'avg_energy_index',
        'inflation_adjustment_factor_mean': 'avg_inflation_factor',
        'city_median_household_income_mean': 'avg_median_income',
    }, inplace=True)
    
    if 'aud__engagement_factor_mean' in yearly.columns:
        yearly.rename(columns={'aud__engagement_factor_mean': 'avg_engagement_factor'}, inplace=True)
    
    return yearly


def aggregate_by_month(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate external factors and tickets by year-month."""
    
    # Only include shows with actual ticket sales
    df_with_tickets = df[(df['target_ticket_median'] > 0) & (df['month'].notna())].copy()
    
    if len(df_with_tickets) == 0:
        print("Warning: No shows with target_ticket_median > 0 and valid month for monthly aggregation")
        return pd.DataFrame()
    
    # Define aggregation columns
    agg_dict = {
        'target_ticket_median': ['count', 'median', 'mean'],
        'consumer_confidence_prairies': 'mean',
        'energy_index': 'mean',
        'inflation_adjustment_factor': 'mean',
        'city_median_household_income': 'mean',
    }
    
    if 'aud__engagement_factor' in df_with_tickets.columns:
        agg_dict['aud__engagement_factor'] = 'mean'
    
    # Group by year_month
    monthly = df_with_tickets.groupby('year_month').agg(agg_dict).reset_index()
    
    # Flatten column names
    monthly.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                       for col in monthly.columns.values]
    
    # Rename for clarity
    monthly.rename(columns={
        'target_ticket_median_count': 'show_count',
        'target_ticket_median_median': 'median_tickets',
        'target_ticket_median_mean': 'mean_tickets',
        'consumer_confidence_prairies_mean': 'avg_consumer_confidence',
        'energy_index_mean': 'avg_energy_index',
        'inflation_adjustment_factor_mean': 'avg_inflation_factor',
        'city_median_household_income_mean': 'avg_median_income',
    }, inplace=True)
    
    if 'aud__engagement_factor_mean' in monthly.columns:
        monthly.rename(columns={'aud__engagement_factor_mean': 'avg_engagement_factor'}, inplace=True)
    
    return monthly


def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Compute correlations between external factors and ticket performance."""
    
    # Only include shows with actual ticket sales
    df_with_tickets = df[df['target_ticket_median'] > 0].copy()
    
    if len(df_with_tickets) < 3:
        print("Warning: Insufficient data for correlation analysis")
        return pd.DataFrame()
    
    # Define factors to correlate with tickets
    factors = [
        'consumer_confidence_prairies',
        'energy_index',
        'inflation_adjustment_factor',
        'city_median_household_income',
    ]
    
    if 'aud__engagement_factor' in df_with_tickets.columns:
        factors.append('aud__engagement_factor')
    
    correlations = []
    
    for factor in factors:
        if factor not in df_with_tickets.columns:
            continue
        
        # Overall correlation
        valid_data = df_with_tickets[[factor, 'target_ticket_median']].dropna()
        n_obs = len(valid_data)
        
        if n_obs >= 3:
            overall_corr = valid_data[factor].corr(valid_data['target_ticket_median'])
        else:
            overall_corr = np.nan
        
        corr_entry = {
            'factor_name': factor,
            'overall_correlation': overall_corr,
            'n_obs_used': n_obs,
        }
        
        # Try to split by city if possible (would need a city column)
        # For now, we'll add placeholders
        corr_entry['correlation_calgary'] = np.nan
        corr_entry['correlation_edmonton'] = np.nan
        
        correlations.append(corr_entry)
    
    corr_df = pd.DataFrame(correlations)
    
    return corr_df


def create_time_series_plots(yearly_df: pd.DataFrame):
    """Create time series plots showing tickets vs external factors."""
    
    if yearly_df.empty or 'year' not in yearly_df.columns:
        print("Skipping plots: insufficient yearly data")
        return
    
    # Sort by year
    yearly_df = yearly_df.sort_values('year')
    
    # Plot 1: Tickets vs Consumer Confidence
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    color_tickets = 'tab:blue'
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Median Tickets Sold', color=color_tickets, fontsize=12)
    ax1.plot(yearly_df['year'], yearly_df['median_tickets'], 
             marker='o', color=color_tickets, linewidth=2, label='Median Tickets')
    ax1.tick_params(axis='y', labelcolor=color_tickets)
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    color_confidence = 'tab:orange'
    ax2.set_ylabel('Consumer Confidence (Prairies)', color=color_confidence, fontsize=12)
    ax2.plot(yearly_df['year'], yearly_df['avg_consumer_confidence'], 
             marker='s', color=color_confidence, linewidth=2, linestyle='--',
             label='Consumer Confidence')
    ax2.tick_params(axis='y', labelcolor=color_confidence)
    
    plt.title('Ticket Sales vs Consumer Confidence by Year', fontsize=14, fontweight='bold')
    fig.tight_layout()
    
    output_path = PLOTS_DIR / "external_factors_tickets_vs_confidence_by_year.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()
    
    # Plot 2: Tickets vs Energy Index
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Median Tickets Sold', color=color_tickets, fontsize=12)
    ax1.plot(yearly_df['year'], yearly_df['median_tickets'], 
             marker='o', color=color_tickets, linewidth=2, label='Median Tickets')
    ax1.tick_params(axis='y', labelcolor=color_tickets)
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    color_energy = 'tab:green'
    ax2.set_ylabel('Energy Index', color=color_energy, fontsize=12)
    ax2.plot(yearly_df['year'], yearly_df['avg_energy_index'], 
             marker='^', color=color_energy, linewidth=2, linestyle='--',
             label='Energy Index')
    ax2.tick_params(axis='y', labelcolor=color_energy)
    
    plt.title('Ticket Sales vs Energy Index by Year', fontsize=14, fontweight='bold')
    fig.tight_layout()
    
    output_path = PLOTS_DIR / "external_factors_tickets_vs_energy_by_year.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()
    
    # Plot 3: Multi-factor comparison (if we have engagement factor)
    if 'avg_engagement_factor' in yearly_df.columns:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('External Factors vs Ticket Sales (Yearly Trends)', 
                     fontsize=16, fontweight='bold')
        
        # Tickets over time
        axes[0, 0].plot(yearly_df['year'], yearly_df['median_tickets'], 
                        marker='o', linewidth=2, color='tab:blue')
        axes[0, 0].set_ylabel('Median Tickets', fontsize=10)
        axes[0, 0].set_title('Ticket Sales Trend')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Consumer confidence
        axes[0, 1].plot(yearly_df['year'], yearly_df['avg_consumer_confidence'], 
                        marker='s', linewidth=2, color='tab:orange')
        axes[0, 1].set_ylabel('Consumer Confidence', fontsize=10)
        axes[0, 1].set_title('Consumer Confidence Trend')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Energy index
        axes[1, 0].plot(yearly_df['year'], yearly_df['avg_energy_index'], 
                        marker='^', linewidth=2, color='tab:green')
        axes[1, 0].set_xlabel('Year', fontsize=10)
        axes[1, 0].set_ylabel('Energy Index', fontsize=10)
        axes[1, 0].set_title('Energy Index Trend')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Engagement factor
        axes[1, 1].plot(yearly_df['year'], yearly_df['avg_engagement_factor'], 
                        marker='d', linewidth=2, color='tab:purple')
        axes[1, 1].set_xlabel('Year', fontsize=10)
        axes[1, 1].set_ylabel('Engagement Factor', fontsize=10)
        axes[1, 1].set_title('Audience Engagement Factor Trend')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = PLOTS_DIR / "external_factors_multi_factor_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")
        plt.close()


def generate_yearly_story(yearly_df: pd.DataFrame, corr_df: pd.DataFrame):
    """Generate a markdown narrative of the yearly story."""
    
    if yearly_df.empty:
        print("Skipping story generation: no yearly data")
        return
    
    yearly_df = yearly_df.sort_values('year')
    
    # Calculate percentiles for comparative language
    median_pct = yearly_df['median_tickets'].rank(pct=True) * 100
    confidence_pct = yearly_df['avg_consumer_confidence'].rank(pct=True) * 100
    energy_pct = yearly_df['avg_energy_index'].rank(pct=True) * 100
    
    story_lines = [
        "# External Factors: Yearly Story\n",
        f"**Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}\n",
        f"**Dataset:** {len(yearly_df)} years with ticket sales data\n",
        "\n---\n",
    ]
    
    # Add correlation summary
    if not corr_df.empty:
        story_lines.append("\n## Correlation Summary\n")
        story_lines.append("\nHow external factors correlate with ticket sales:\n\n")
        
        for _, row in corr_df.iterrows():
            factor = row['factor_name'].replace('_', ' ').title()
            corr = row['overall_correlation']
            n = row['n_obs_used']
            
            if pd.notna(corr):
                strength = "strong" if abs(corr) > 0.5 else "moderate" if abs(corr) > 0.3 else "weak"
                direction = "positive" if corr > 0 else "negative"
                story_lines.append(f"- **{factor}**: {direction} {strength} correlation "
                                 f"(r={corr:.3f}, n={n})\n")
            else:
                story_lines.append(f"- **{factor}**: insufficient data for correlation\n")
        
        story_lines.append("\n---\n")
    
    # Year-by-year narrative
    story_lines.append("\n## Year-by-Year Analysis\n")
    
    for idx, row in yearly_df.iterrows():
        year = int(row['year'])
        shows = int(row['show_count'])
        median_tix = row['median_tickets']
        mean_tix = row.get('mean_tickets', median_tix)
        
        confidence = row['avg_consumer_confidence']
        energy = row['avg_energy_index']
        inflation = row['avg_inflation_factor']
        
        # Determine relative positions
        ticket_level = "high" if median_pct.loc[idx] > 66 else "moderate" if median_pct.loc[idx] > 33 else "low"
        confidence_level = "high" if confidence_pct.loc[idx] > 66 else "moderate" if confidence_pct.loc[idx] > 33 else "low"
        energy_level = "high" if energy_pct.loc[idx] > 66 else "moderate" if energy_pct.loc[idx] > 33 else "low"
        
        story_lines.append(f"\n### {year}\n")
        story_lines.append(f"- **Shows:** {shows} productions with ticket sales\n")
        story_lines.append(f"- **Ticket Performance:** {ticket_level} (median: {median_tix:,.0f}, "
                         f"mean: {mean_tix:,.0f})\n")
        story_lines.append(f"- **Economic Context:**\n")
        story_lines.append(f"  - Consumer confidence: {confidence_level} ({confidence:.1f})\n")
        story_lines.append(f"  - Energy index: {energy_level} ({energy:.1f})\n")
        story_lines.append(f"  - Inflation factor: {inflation:.3f}x\n")
        
        # Add engagement factor if present
        if 'avg_engagement_factor' in row and pd.notna(row['avg_engagement_factor']):
            engagement = row['avg_engagement_factor']
            story_lines.append(f"  - Audience engagement factor: {engagement:.3f}\n")
        
        # Add contextual interpretation
        if ticket_level == "high" and confidence_level == "high":
            story_lines.append(f"- **Pattern:** Strong ticket sales aligned with high consumer confidence\n")
        elif ticket_level == "low" and (confidence_level == "low" or energy_level == "low"):
            story_lines.append(f"- **Pattern:** Lower sales coinciding with weaker economic indicators\n")
        elif ticket_level == "high" and energy_level == "high":
            story_lines.append(f"- **Pattern:** Strong performance during period of high energy prices "
                             f"(Alberta economy benefit)\n")
        else:
            story_lines.append(f"- **Pattern:** Mixed economic signals; other factors may dominate\n")
    
    # Footer
    story_lines.append("\n---\n")
    story_lines.append("\n## Notes\n")
    story_lines.append("- Year estimates are inferred from `years_since_last_run` for historical shows\n")
    story_lines.append("- Shows without prior ticket sales (cold starts) are excluded from analysis\n")
    story_lines.append("- Economic factors represent Alberta-wide or Prairie region indicators\n")
    story_lines.append("- Correlation does not imply causation; multiple factors influence ticket sales\n")
    
    # Write to file
    output_path = RESULTS_DIR / "external_factors_yearly_story.md"
    with open(output_path, 'w') as f:
        f.writelines(story_lines)
    
    print(f"  Saved: {output_path}")


def main():
    """Main analysis workflow."""
    print("\n" + "="*60)
    print("External Factors Analysis")
    print("="*60 + "\n")
    
    # 1. Load dataset
    print("1. Loading modelling dataset...")
    df = load_modelling_dataset()
    
    # 2. Infer temporal dimensions
    print("\n2. Inferring temporal dimensions...")
    df = infer_temporal_dimensions(df)
    
    # 3. Aggregate by year
    print("\n3. Aggregating by year...")
    yearly_df = aggregate_by_year(df)
    if not yearly_df.empty:
        output_path = RESULTS_DIR / "external_factors_by_year.csv"
        yearly_df.to_csv(output_path, index=False)
        print(f"  Saved: {output_path}")
        print(f"  Years: {yearly_df['year'].tolist()}")
    
    # 4. Aggregate by month
    print("\n4. Aggregating by year-month...")
    monthly_df = aggregate_by_month(df)
    if not monthly_df.empty:
        output_path = RESULTS_DIR / "external_factors_by_month.csv"
        monthly_df.to_csv(output_path, index=False)
        print(f"  Saved: {output_path}")
        print(f"  Periods: {len(monthly_df)} year-months")
    
    # 5. Compute correlations
    print("\n5. Computing correlations...")
    corr_df = compute_correlations(df)
    if not corr_df.empty:
        output_path = RESULTS_DIR / "external_factors_correlations.csv"
        corr_df.to_csv(output_path, index=False)
        print(f"  Saved: {output_path}")
        print("\n  Correlation summary:")
        for _, row in corr_df.iterrows():
            factor = row['factor_name']
            corr = row['overall_correlation']
            if pd.notna(corr):
                print(f"    {factor}: r={corr:.3f}")
    
    # 6. Create plots
    print("\n6. Creating time series plots...")
    create_time_series_plots(yearly_df)
    
    # 7. Generate yearly story
    print("\n7. Generating yearly story narrative...")
    generate_yearly_story(yearly_df, corr_df)
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)
    print(f"\nOutputs saved to:")
    print(f"  - {RESULTS_DIR}/external_factors_by_year.csv")
    print(f"  - {RESULTS_DIR}/external_factors_by_month.csv")
    print(f"  - {RESULTS_DIR}/external_factors_correlations.csv")
    print(f"  - {RESULTS_DIR}/external_factors_yearly_story.md")
    print(f"  - {PLOTS_DIR}/*.png (plots)")
    print()


if __name__ == "__main__":
    main()
