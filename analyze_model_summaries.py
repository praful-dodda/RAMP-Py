"""
Analyze and visualize model performance summaries for data fusion model selection.

This script helps identify the best models for each year/region based on
performance metrics from RAMP-corrected outputs.

Usage:
    python analyze_model_summaries.py

Outputs:
    - Best model recommendations by year/region
    - Performance comparison visualizations
    - Model ranking tables
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
SUMMARIES_DIR = './model_summaries'
OUTPUT_DIR = './model_analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Visualization settings
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def load_master_summary():
    """Load the master summary file."""
    master_file = os.path.join(SUMMARIES_DIR, 'ALL_MODELS_master_summary.csv')

    if not os.path.exists(master_file):
        raise FileNotFoundError(
            f"Master summary not found at {master_file}\n"
            "Please run generate_model_performance_summaries.py first!"
        )

    df = pd.read_csv(master_file)
    print(f"✓ Loaded master summary: {len(df)} model-year combinations")
    return df


def rank_models_by_year(master_df, metric='after_RMSE', ascending=True):
    """
    Rank models for each year based on a specific metric.

    Parameters:
    -----------
    master_df : pd.DataFrame
        Master summary data
    metric : str
        Performance metric to rank by (default: after_RMSE)
    ascending : bool
        True for metrics where lower is better (RMSE, MAE)
        False for metrics where higher is better (r2, correlation)

    Returns:
    --------
    pd.DataFrame with rankings
    """
    print(f"\n📊 Ranking models by {metric}...")

    rankings = []

    for year in sorted(master_df['year'].unique()):
        year_data = master_df[master_df['year'] == year].copy()

        # Remove rows with NaN in the metric
        year_data = year_data.dropna(subset=[metric])

        # Sort and rank
        year_data = year_data.sort_values(metric, ascending=ascending)
        year_data['rank'] = range(1, len(year_data) + 1)

        # Select top 5
        top_models = year_data.head(5)[['model', metric, 'rank', 'n_points']].copy()
        top_models['year'] = year

        rankings.append(top_models)

    rankings_df = pd.concat(rankings, ignore_index=True)

    return rankings_df


def find_best_model_per_year(master_df):
    """
    Determine the best performing model for each year based on multiple metrics.

    Uses a weighted scoring system:
    - RMSE improvement: 30%
    - After RMSE: 30%
    - After r2: 20%
    - After correlation: 20%
    """
    print("\n🏆 Finding best model per year (composite scoring)...")

    best_models = []

    for year in sorted(master_df['year'].unique()):
        year_data = master_df[master_df['year'] == year].copy()

        # Remove rows with insufficient data
        year_data = year_data.dropna(subset=['after_RMSE', 'after_r2', 'RMSE_improvement_pct'])

        if len(year_data) == 0:
            continue

        # Normalize metrics to 0-1 scale
        # For RMSE (lower is better): score = 1 - (value - min) / (max - min)
        rmse_range = year_data['after_RMSE'].max() - year_data['after_RMSE'].min()
        if rmse_range > 0:
            year_data['rmse_score'] = 1 - (year_data['after_RMSE'] - year_data['after_RMSE'].min()) / rmse_range
        else:
            year_data['rmse_score'] = 1.0

        # For r2 (higher is better): score = (value - min) / (max - min)
        r2_range = year_data['after_r2'].max() - year_data['after_r2'].min()
        if r2_range > 0:
            year_data['r2_score'] = (year_data['after_r2'] - year_data['after_r2'].min()) / r2_range
        else:
            year_data['r2_score'] = 1.0

        # For RMSE improvement (higher is better)
        imp_range = year_data['RMSE_improvement_pct'].max() - year_data['RMSE_improvement_pct'].min()
        if imp_range > 0:
            year_data['imp_score'] = ((year_data['RMSE_improvement_pct'] -
                                       year_data['RMSE_improvement_pct'].min()) / imp_range)
        else:
            year_data['imp_score'] = 1.0

        # Composite score
        year_data['composite_score'] = (
            0.30 * year_data['rmse_score'] +
            0.30 * year_data['imp_score'] +
            0.20 * year_data['r2_score'] +
            0.20 * year_data['r2_score']  # Using r2 as proxy for correlation
        )

        # Find best
        best = year_data.nlargest(1, 'composite_score')

        best_models.append({
            'year': year,
            'best_model': best['model'].values[0],
            'composite_score': best['composite_score'].values[0],
            'after_RMSE': best['after_RMSE'].values[0],
            'after_r2': best['after_r2'].values[0],
            'RMSE_improvement_pct': best['RMSE_improvement_pct'].values[0],
            'n_points': best['n_points'].values[0]
        })

    best_df = pd.DataFrame(best_models)
    return best_df


def analyze_model_availability(master_df):
    """Create a table showing which models are available for which years."""
    print("\n📅 Analyzing model availability across years...")

    # Create pivot table
    availability = master_df.pivot_table(
        index='model',
        columns='year',
        values='n_points',
        aggfunc='sum',
        fill_value=0
    )

    # Convert to binary (available/not available)
    availability_binary = (availability > 0).astype(int)

    return availability, availability_binary


def plot_model_performance_trends(master_df):
    """Plot performance trends over time for each model."""
    print("\n📈 Generating performance trend plots...")

    metrics = ['after_RMSE', 'after_MAE', 'after_r2', 'RMSE_improvement_pct']
    metric_labels = ['RMSE (After RAMP)', 'MAE (After RAMP)',
                     'R² (After RAMP)', 'RMSE Improvement (%)']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]

        # Plot each model
        for model in master_df['model'].unique():
            model_data = master_df[master_df['model'] == model].sort_values('year')

            if len(model_data) > 0:
                ax.plot(model_data['year'], model_data[metric],
                       marker='o', label=model, linewidth=2, markersize=6)

        ax.set_xlabel('Year', fontsize=12, fontweight='bold')
        ax.set_ylabel(label, fontsize=12, fontweight='bold')
        ax.set_title(f'{label} Over Time', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = os.path.join(OUTPUT_DIR, 'model_performance_trends.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {os.path.basename(output_file)}")


def plot_model_comparison_heatmap(master_df):
    """Create heatmap comparing models across years."""
    print("\n🔥 Generating comparison heatmap...")

    # Use after_RMSE as the metric (lower is better)
    pivot = master_df.pivot_table(
        index='model',
        columns='year',
        values='after_RMSE',
        aggfunc='mean'
    )

    fig, ax = plt.subplots(figsize=(14, 8))

    sns.heatmap(
        pivot,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn_r',  # Red = high RMSE (bad), Green = low RMSE (good)
        cbar_kws={'label': 'RMSE (After RAMP) - Lower is Better'},
        ax=ax,
        linewidths=0.5
    )

    ax.set_title('Model Performance Comparison (RMSE After RAMP)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')

    plt.tight_layout()
    output_file = os.path.join(OUTPUT_DIR, 'model_comparison_heatmap.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {os.path.basename(output_file)}")


def plot_improvement_comparison(master_df):
    """Plot RMSE improvement percentages for all models."""
    print("\n📊 Generating improvement comparison plot...")

    fig, ax = plt.subplots(figsize=(14, 8))

    # Group by model and calculate mean improvement
    improvement = master_df.groupby('model')['RMSE_improvement_pct'].mean().sort_values(ascending=False)

    colors = ['green' if x > 0 else 'red' for x in improvement.values]

    improvement.plot(kind='barh', ax=ax, color=colors)

    ax.set_xlabel('Average RMSE Improvement (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')
    ax.set_title('Average RMSE Improvement After RAMP Correction', fontsize=14, fontweight='bold')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    output_file = os.path.join(OUTPUT_DIR, 'rmse_improvement_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {os.path.basename(output_file)}")


def generate_selection_recommendation(master_df):
    """
    Generate a comprehensive recommendation table for model selection.
    """
    print("\n💡 Generating model selection recommendations...")

    recommendations = []

    for year in sorted(master_df['year'].unique()):
        year_data = master_df[master_df['year'] == year].copy()

        # Find best by different criteria
        best_rmse = year_data.nsmallest(1, 'after_RMSE')['model'].values[0] if len(year_data) > 0 else None
        best_r2 = year_data.nlargest(1, 'after_r2')['model'].values[0] if len(year_data) > 0 else None
        best_improvement = year_data.nlargest(1, 'RMSE_improvement_pct')['model'].values[0] if len(year_data) > 0 else None

        # Composite best (as calculated earlier)
        year_data = year_data.dropna(subset=['after_RMSE', 'after_r2'])

        if len(year_data) > 0:
            # Simple composite: normalize and average
            year_data['norm_rmse'] = 1 - (year_data['after_RMSE'] - year_data['after_RMSE'].min()) / (
                year_data['after_RMSE'].max() - year_data['after_RMSE'].min() + 1e-10
            )
            year_data['norm_r2'] = (year_data['after_r2'] - year_data['after_r2'].min()) / (
                year_data['after_r2'].max() - year_data['after_r2'].min() + 1e-10
            )
            year_data['composite'] = (year_data['norm_rmse'] + year_data['norm_r2']) / 2

            best_composite = year_data.nlargest(1, 'composite')['model'].values[0]
        else:
            best_composite = None

        recommendations.append({
            'year': year,
            'recommended_model': best_composite,
            'best_by_RMSE': best_rmse,
            'best_by_r2': best_r2,
            'best_by_improvement': best_improvement,
            'num_models_available': len(year_data)
        })

    rec_df = pd.DataFrame(recommendations)
    return rec_df


def main():
    """Main analysis function."""
    print("\n" + "="*70)
    print("RAMP Model Performance Analysis")
    print("="*70)

    # Load master summary
    try:
        master_df = load_master_summary()
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        return

    # Basic statistics
    print(f"\n📊 Dataset Overview:")
    print(f"  • Total models: {master_df['model'].nunique()}")
    print(f"  • Year range: {master_df['year'].min()} - {master_df['year'].max()}")
    print(f"  • Total model-year combinations: {len(master_df)}")
    print(f"  • Total observations: {master_df['n_points'].sum():,}")

    # Model availability analysis
    availability, availability_binary = analyze_model_availability(master_df)
    availability_file = os.path.join(OUTPUT_DIR, 'model_availability_by_year.csv')
    availability.to_csv(availability_file)
    print(f"  ✓ Saved availability table: {os.path.basename(availability_file)}")

    # Best model per year
    best_per_year = find_best_model_per_year(master_df)
    best_file = os.path.join(OUTPUT_DIR, 'best_model_per_year.csv')
    best_per_year.to_csv(best_file, index=False)
    print(f"  ✓ Saved best models: {os.path.basename(best_file)}")

    # Rankings
    rmse_rankings = rank_models_by_year(master_df, metric='after_RMSE', ascending=True)
    rankings_file = os.path.join(OUTPUT_DIR, 'model_rankings_by_RMSE.csv')
    rmse_rankings.to_csv(rankings_file, index=False)
    print(f"  ✓ Saved RMSE rankings: {os.path.basename(rankings_file)}")

    r2_rankings = rank_models_by_year(master_df, metric='after_r2', ascending=False)
    r2_rankings_file = os.path.join(OUTPUT_DIR, 'model_rankings_by_r2.csv')
    r2_rankings.to_csv(r2_rankings_file, index=False)
    print(f"  ✓ Saved R² rankings: {os.path.basename(r2_rankings_file)}")

    # Model selection recommendations
    recommendations = generate_selection_recommendation(master_df)
    rec_file = os.path.join(OUTPUT_DIR, 'model_selection_recommendations.csv')
    recommendations.to_csv(rec_file, index=False)
    print(f"  ✓ Saved recommendations: {os.path.basename(rec_file)}")

    # Generate visualizations
    print(f"\n{'='*70}")
    print("Generating visualizations...")
    print(f"{'='*70}")

    plot_model_performance_trends(master_df)
    plot_model_comparison_heatmap(master_df)
    plot_improvement_comparison(master_df)

    # Summary statistics by model
    model_stats = master_df.groupby('model').agg({
        'year': ['count', 'min', 'max'],
        'after_RMSE': ['mean', 'std', 'min'],
        'after_r2': ['mean', 'std', 'max'],
        'RMSE_improvement_pct': ['mean', 'std'],
        'n_points': 'sum'
    }).round(3)

    model_stats.columns = ['_'.join(col).strip() for col in model_stats.columns.values]
    model_stats = model_stats.reset_index()
    model_stats = model_stats.sort_values('after_RMSE_mean')

    stats_file = os.path.join(OUTPUT_DIR, 'model_summary_statistics.csv')
    model_stats.to_csv(stats_file, index=False)
    print(f"  ✓ Saved model statistics: {os.path.basename(stats_file)}")

    # Final summary
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"\n📁 All analysis outputs saved to: {OUTPUT_DIR}/")
    print("\nKey files:")
    print("  • best_model_per_year.csv - Recommended model for each year")
    print("  • model_selection_recommendations.csv - Comprehensive recommendations")
    print("  • model_rankings_by_RMSE.csv - Top 5 models per year by RMSE")
    print("  • model_summary_statistics.csv - Overall model performance stats")
    print("  • model_availability_by_year.csv - Model coverage table")
    print("\nVisualizations:")
    print("  • model_performance_trends.png - Performance over time")
    print("  • model_comparison_heatmap.png - Year-model comparison")
    print("  • rmse_improvement_comparison.png - RAMP effectiveness")

    print("\n🏆 Top 3 Models Overall (by average RMSE):")
    print(model_stats[['model', 'after_RMSE_mean', 'after_r2_mean', 'year_count']].head(3).to_string(index=False))

    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
