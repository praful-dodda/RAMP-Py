"""
Generate comprehensive performance summary reports for RAMP-corrected models.

This script creates detailed performance summaries for each model/year combination,
including before/after RAMP correction statistics by region, month, season, and
station type. Designed for data fusion model selection.

Usage:
    python generate_model_performance_summaries.py

Output structure:
    model_summaries/
    ├── MODEL_NAME/
    │   ├── MODEL_YEAR_overall_performance.csv
    │   ├── MODEL_YEAR_performance_by_region.csv
    │   ├── MODEL_YEAR_performance_by_month.csv
    │   ├── MODEL_YEAR_performance_by_season.csv
    │   ├── MODEL_YEAR_performance_by_station_type.csv
    │   └── MODEL_combined_all_years_summary.csv
    └── ALL_MODELS_master_summary.csv
"""

import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import existing evaluation functions
from evaluate_models import calculate_performance_metrics, get_region, get_season
from preprocess import get_ozone_file

# ============================================================================
# CONFIGURATION
# ============================================================================

# Directories
RAMP_DATA_DIR = './ramp_data'
OUTPUT_DIR = './model_summaries'
VERSION = 'v3-parallel'  # RAMP version identifier

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def find_available_model_years():
    """
    Scan ramp_data directory to find all available model-year combinations.
    Returns a dictionary: {model_name: [year1, year2, ...]}
    """
    print("Scanning for available RAMP results...")

    # Find all collocated data files
    collocated_files = glob.glob(f"{RAMP_DATA_DIR}/collocated_data_*.parquet") + \
                       glob.glob(f"{RAMP_DATA_DIR}/collocated_data_*.csv")

    model_years = {}

    for filepath in collocated_files:
        filename = os.path.basename(filepath)
        # Parse: collocated_data_MODELNAME_YEAR.parquet
        parts = filename.replace('collocated_data_', '').replace('.parquet', '').replace('.csv', '')

        # Split from the right to handle model names with underscores
        parts_split = parts.rsplit('_', 1)
        if len(parts_split) == 2:
            model_name, year_str = parts_split
            try:
                year = int(year_str)
                if model_name not in model_years:
                    model_years[model_name] = []
                model_years[model_name].append(year)
            except ValueError:
                print(f"Warning: Could not parse year from {filename}")

    # Sort years for each model
    for model in model_years:
        model_years[model] = sorted(model_years[model])

    return model_years


def load_file_flexible(base_path):
    """Try loading parquet first, then CSV."""
    parquet_path = f"{base_path}.parquet"
    csv_path = f"{base_path}.csv"

    if os.path.exists(parquet_path):
        return pd.read_parquet(parquet_path)
    elif os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    else:
        return None


def merge_ramp_corrected_values(collocated_df, lambda1_df, original_model_df):
    """
    Merge pre-computed RAMP-corrected values with collocated observations.

    This follows the approach from evaluate_ramp_correction.py:
    1. Combine original model grid with RAMP-corrected grid (lambda1)
    2. Convert to long format with both original and RAMP values
    3. Use collocated observation locations to extract relevant grid points
    4. Match observations to grid using the model grid coordinates already in collocated data

    Note: lambda1 files contain the final RAMP-corrected ozone values (NOT correction
    factors). This function merges them with observations for evaluation.

    Parameters:
    -----------
    collocated_df : pd.DataFrame
        Collocated observations with original model outputs
        Must have: lon_model, lat_model, month, observed_ozone, model_ozone
    lambda1_df : pd.DataFrame
        RAMP-corrected ozone values (DMA8_1...DMA8_12) - already computed by RAMP
    original_model_df : pd.DataFrame
        Original model grid data (DMA8_1...DMA8_12)

    Returns:
    --------
    pd.DataFrame with both 'model_ozone' and 'ramp_corrected_ozone' columns
    """
    from scipy.spatial import cKDTree

    # Ensure we have the necessary columns
    if 'observed_ozone' not in collocated_df.columns:
        raise ValueError(f"Missing observed_ozone in collocated data")

    # Step 1: Rename lambda1 columns for clarity
    lambda1_renamed = lambda1_df.copy()
    dma_cols = {f'DMA8_{i}': f'ramp_corrected_{i}' for i in range(1, 13)}
    lambda1_renamed.rename(columns=dma_cols, inplace=True)

    # Step 2: Combine original model grid with RAMP corrections
    # Both should have lon, lat columns
    gridded_eval_df = pd.concat([original_model_df, lambda1_renamed], axis=1)

    # Step 3: Convert to long format
    gridded_long = pd.melt(
        gridded_eval_df,
        id_vars=['lon', 'lat'],
        var_name='var_month',
        value_name='ozone'
    )
    gridded_long['month'] = gridded_long['var_month'].str.extract(r'_(\d+)').astype(int)
    gridded_long['source'] = np.where(
        gridded_long['var_month'].str.contains('ramp'),
        'RAMP-Corrected',
        'Original Model'
    )
    gridded_long.drop(columns=['var_month'], inplace=True)

    # Step 4: Pivot to have original and RAMP as separate columns
    gridded_pivot = gridded_long.pivot_table(
        index=['lon', 'lat', 'month'],
        columns='source',
        values='ozone'
    ).reset_index()
    gridded_pivot.rename(
        columns={
            'Original Model': 'original_model_ozone',
            'RAMP-Corrected': 'ramp_corrected_ozone'
        },
        inplace=True
    )

    # Step 5: Extract observation locations from collocated data
    obs_df = collocated_df[['lon_toar', 'lat_toar', 'observed_ozone', 'month']].copy()

    # Add region, season, type if available
    for col in ['region', 'season', 'type', 'id', 'country', 'lon_model', 'lat_model']:
        if col in collocated_df.columns:
            obs_df[col] = collocated_df[col]

    obs_df.rename(columns={'lon_toar': 'lon_obs', 'lat_toar': 'lat_obs'}, inplace=True)

    # Step 6: Find nearest grid points for observations
    grid_coords = gridded_pivot[['lon', 'lat']].drop_duplicates()
    tree = cKDTree(grid_coords.values)

    # Query using observation coordinates
    _, indices = tree.query(obs_df[['lon_obs', 'lat_obs']].values, k=1)

    # Get the coordinates of the nearest grid points
    nearest_grid_points = grid_coords.iloc[indices].reset_index(drop=True)
    obs_df['grid_lon'] = nearest_grid_points['lon'].values
    obs_df['grid_lat'] = nearest_grid_points['lat'].values

    # Step 7: Merge observations with gridded data
    eval_df = pd.merge(
        obs_df,
        gridded_pivot,
        left_on=['grid_lon', 'grid_lat', 'month'],
        right_on=['lon', 'lat', 'month'],
        how='left',
        suffixes=('', '_grid')
    )

    # Rename for consistency with collocated data
    eval_df.rename(columns={'original_model_ozone': 'model_ozone'}, inplace=True)

    # Add region tagging if not already present
    if 'region' not in eval_df.columns:
        eval_df['region'] = eval_df.apply(
            lambda row: get_region(row['lat_obs'], row['lon_obs']),
            axis=1
        )

    # Add season tagging if not already present
    if 'season' not in eval_df.columns:
        eval_df['season'] = eval_df.apply(
            lambda row: get_season(row['month'], row['lat_obs']),
            axis=1
        )

    return eval_df


def calculate_metrics_before_after(df_group):
    """
    Calculate performance metrics for both original and RAMP-corrected model outputs.

    Returns a single Series with both sets of metrics.
    """
    # Before RAMP (original model)
    metrics_before = calculate_performance_metrics(
        df_group,
        observed_col='observed_ozone',
        model_col='model_ozone'
    )

    # After RAMP (corrected model)
    metrics_after = calculate_performance_metrics(
        df_group,
        observed_col='observed_ozone',
        model_col='ramp_corrected_ozone'
    )

    # Combine with prefixes
    result = {}
    for key, value in metrics_before.items():
        if key != 'n_points':
            result[f'before_{key}'] = value

    for key, value in metrics_after.items():
        if key != 'n_points':
            result[f'after_{key}'] = value

    # Add improvement metrics
    if not np.isnan(metrics_before['RMSE']) and not np.isnan(metrics_after['RMSE']):
        result['RMSE_improvement_pct'] = ((metrics_before['RMSE'] - metrics_after['RMSE']) /
                                          metrics_before['RMSE'] * 100)
        result['MAE_improvement_pct'] = ((metrics_before['MAE'] - metrics_after['MAE']) /
                                         metrics_before['MAE'] * 100)
        result['r2_improvement'] = metrics_after['r2'] - metrics_before['r2']
    else:
        result['RMSE_improvement_pct'] = np.nan
        result['MAE_improvement_pct'] = np.nan
        result['r2_improvement'] = np.nan

    result['n_points'] = metrics_before['n_points']

    return pd.Series(result)


# ============================================================================
# MAIN PROCESSING FUNCTIONS
# ============================================================================

def process_single_model_year(model_name, year):
    """
    Generate all performance summary files for a single model-year combination.

    Returns:
    --------
    dict : Summary statistics for this model-year (for master summary)
    """
    print(f"\n{'='*70}")
    print(f"Processing: {model_name} - {year}")
    print(f"{'='*70}")

    # Create model-specific output directory
    model_dir = os.path.join(OUTPUT_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Load data files
    # -------------------------------------------------------------------------

    # 1. Original model grid data
    try:
        original_model_file = get_ozone_file(model_name, year)
        if original_model_file is None:
            print(f"  ❌ ERROR: Could not find original model file for {model_name} {year}")
            return None

        original_model_df = pd.read_csv(original_model_file)
        print(f"  ✓ Loaded original model data: {original_model_file}")
    except Exception as e:
        print(f"  ❌ ERROR: Failed to load original model data: {e}")
        return None

    # 2. Collocated data
    collocated_path = f"{RAMP_DATA_DIR}/collocated_data_{model_name}_{year}"
    collocated_df = load_file_flexible(collocated_path)

    if collocated_df is None:
        print(f"  ❌ ERROR: Could not find collocated data file")
        return None

    print(f"  ✓ Loaded collocated data: {len(collocated_df)} observations")

    # 3. RAMP lambda1 corrections
    lambda1_path = f"{RAMP_DATA_DIR}/lambda1_{model_name}_{year}_{VERSION}"
    lambda1_df = load_file_flexible(lambda1_path)

    if lambda1_df is None:
        print(f"  ❌ ERROR: Could not find lambda1 data file")
        return None

    print(f"  ✓ Loaded RAMP corrections: {len(lambda1_df)} grid points")

    # -------------------------------------------------------------------------
    # Merge pre-computed RAMP-corrected values
    # -------------------------------------------------------------------------

    print("  ⚙ Merging RAMP-corrected values with observations...")
    df = merge_ramp_corrected_values(collocated_df, lambda1_df, original_model_df)

    # Remove rows with NaN in critical columns
    initial_count = len(df)
    df = df.dropna(subset=['observed_ozone', 'model_ozone', 'ramp_corrected_ozone'])
    final_count = len(df)

    if final_count < initial_count:
        print(f"  ⚠ Removed {initial_count - final_count} rows with missing values")

    print(f"  ✓ Final dataset: {final_count} valid observations")

    # -------------------------------------------------------------------------
    # Calculate performance summaries
    # -------------------------------------------------------------------------

    output_files = []

    # 1. Overall performance
    print("  📊 Calculating overall performance...")
    overall_metrics = calculate_metrics_before_after(df)
    overall_df = pd.DataFrame([overall_metrics])
    overall_df.insert(0, 'model', model_name)
    overall_df.insert(1, 'year', year)

    overall_file = os.path.join(model_dir, f"{model_name}_{year}_overall_performance.csv")
    overall_df.to_csv(overall_file, index=False)
    output_files.append(overall_file)
    print(f"  ✓ Saved: {os.path.basename(overall_file)}")

    # 2. Performance by region
    if 'region' in df.columns:
        print("  📊 Calculating performance by region...")
        region_metrics = df.groupby('region').apply(calculate_metrics_before_after).reset_index()
        region_metrics.insert(0, 'model', model_name)
        region_metrics.insert(1, 'year', year)

        region_file = os.path.join(model_dir, f"{model_name}_{year}_performance_by_region.csv")
        region_metrics.to_csv(region_file, index=False)
        output_files.append(region_file)
        print(f"  ✓ Saved: {os.path.basename(region_file)}")

    # 3. Performance by month
    if 'month' in df.columns:
        print("  📊 Calculating performance by month...")
        month_metrics = df.groupby('month').apply(calculate_metrics_before_after).reset_index()
        month_metrics.insert(0, 'model', model_name)
        month_metrics.insert(1, 'year', year)

        month_file = os.path.join(model_dir, f"{model_name}_{year}_performance_by_month.csv")
        month_metrics.to_csv(month_file, index=False)
        output_files.append(month_file)
        print(f"  ✓ Saved: {os.path.basename(month_file)}")

    # 4. Performance by season
    if 'season' in df.columns:
        print("  📊 Calculating performance by season...")
        season_metrics = df.groupby('season').apply(calculate_metrics_before_after).reset_index()
        season_metrics.insert(0, 'model', model_name)
        season_metrics.insert(1, 'year', year)

        season_file = os.path.join(model_dir, f"{model_name}_{year}_performance_by_season.csv")
        season_metrics.to_csv(season_file, index=False)
        output_files.append(season_file)
        print(f"  ✓ Saved: {os.path.basename(season_file)}")

    # 5. Performance by station type
    if 'type' in df.columns:
        print("  📊 Calculating performance by station type...")
        type_metrics = df.groupby('type').apply(calculate_metrics_before_after).reset_index()
        type_metrics.insert(0, 'model', model_name)
        type_metrics.insert(1, 'year', year)

        type_file = os.path.join(model_dir, f"{model_name}_{year}_performance_by_station_type.csv")
        type_metrics.to_csv(type_file, index=False)
        output_files.append(type_file)
        print(f"  ✓ Saved: {os.path.basename(type_file)}")

    # 6. Combined region-month breakdown
    if 'region' in df.columns and 'month' in df.columns:
        print("  📊 Calculating performance by region and month...")
        region_month_metrics = df.groupby(['region', 'month']).apply(
            calculate_metrics_before_after
        ).reset_index()
        region_month_metrics.insert(0, 'model', model_name)
        region_month_metrics.insert(1, 'year', year)

        region_month_file = os.path.join(
            model_dir,
            f"{model_name}_{year}_performance_by_region_month.csv"
        )
        region_month_metrics.to_csv(region_month_file, index=False)
        output_files.append(region_month_file)
        print(f"  ✓ Saved: {os.path.basename(region_month_file)}")

    print(f"\n  ✅ Completed {model_name} {year}: Generated {len(output_files)} summary files")

    # Return overall metrics for master summary
    return overall_df


def generate_combined_summary_per_model(model_name):
    """
    Create a combined summary across all years for a single model.
    """
    print(f"\n{'='*70}")
    print(f"Generating combined summary for: {model_name}")
    print(f"{'='*70}")

    model_dir = os.path.join(OUTPUT_DIR, model_name)

    # Find all overall performance files for this model
    overall_files = glob.glob(os.path.join(model_dir, f"{model_name}_*_overall_performance.csv"))

    if not overall_files:
        print(f"  ⚠ No performance files found for {model_name}")
        return None

    # Concatenate all years
    all_years_df = pd.concat([pd.read_csv(f) for f in overall_files], ignore_index=True)
    all_years_df = all_years_df.sort_values('year')

    # Save combined file
    combined_file = os.path.join(model_dir, f"{model_name}_combined_all_years_summary.csv")
    all_years_df.to_csv(combined_file, index=False)

    print(f"  ✓ Saved combined summary: {os.path.basename(combined_file)}")
    print(f"  ✓ Years included: {sorted(all_years_df['year'].unique())}")

    return all_years_df


def generate_master_summary(all_results):
    """
    Create a master summary file with all models and years.
    """
    print(f"\n{'='*70}")
    print("Generating master summary across all models and years")
    print(f"{'='*70}")

    # Concatenate all results
    master_df = pd.concat(all_results, ignore_index=True)
    master_df = master_df.sort_values(['model', 'year'])

    # Save master file
    master_file = os.path.join(OUTPUT_DIR, "ALL_MODELS_master_summary.csv")
    master_df.to_csv(master_file, index=False)

    print(f"  ✓ Saved master summary: {os.path.basename(master_file)}")
    print(f"  ✓ Total models: {master_df['model'].nunique()}")
    print(f"  ✓ Total model-year combinations: {len(master_df)}")

    # Create a summary by model (average across years)
    model_avg = master_df.groupby('model').agg({
        'year': ['min', 'max', 'count'],
        'before_RMSE': 'mean',
        'after_RMSE': 'mean',
        'before_MAE': 'mean',
        'after_MAE': 'mean',
        'before_r2': 'mean',
        'after_r2': 'mean',
        'RMSE_improvement_pct': 'mean',
        'MAE_improvement_pct': 'mean',
        'n_points': 'sum'
    }).round(3)

    model_avg.columns = ['_'.join(col).strip('_') for col in model_avg.columns]
    model_avg = model_avg.reset_index()

    model_avg_file = os.path.join(OUTPUT_DIR, "ALL_MODELS_average_by_model.csv")
    model_avg.to_csv(model_avg_file, index=False)

    print(f"  ✓ Saved model averages: {os.path.basename(model_avg_file)}")

    return master_df, model_avg


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function - orchestrates the entire summary generation process.
    """
    print("\n" + "="*70)
    print("RAMP Model Performance Summary Generator")
    print("="*70)

    # Step 1: Find available data
    model_years = find_available_model_years()

    if not model_years:
        print("\n❌ ERROR: No RAMP results found in ramp_data directory!")
        print("Please ensure RAMP analysis has been run and results are in ./ramp_data/")
        return

    print(f"\n✓ Found {len(model_years)} models with RAMP results:")
    for model, years in sorted(model_years.items()):
        print(f"  • {model}: {len(years)} years ({min(years)}-{max(years)})")

    # Step 2: Process each model-year combination
    all_results = []
    failed = []

    total_combinations = sum(len(years) for years in model_years.values())
    print(f"\n{'='*70}")
    print(f"Processing {total_combinations} model-year combinations...")
    print(f"{'='*70}")

    for model_name, years in sorted(model_years.items()):
        for year in years:
            try:
                result = process_single_model_year(model_name, year)
                if result is not None:
                    all_results.append(result)
                else:
                    failed.append((model_name, year))
            except Exception as e:
                print(f"\n  ❌ ERROR processing {model_name} {year}: {str(e)}")
                failed.append((model_name, year))
                continue

    # Step 3: Generate combined summaries per model
    print(f"\n{'='*70}")
    print("Generating combined summaries for each model...")
    print(f"{'='*70}")

    for model_name in sorted(model_years.keys()):
        try:
            generate_combined_summary_per_model(model_name)
        except Exception as e:
            print(f"  ❌ ERROR generating combined summary for {model_name}: {str(e)}")

    # Step 4: Generate master summary
    if all_results:
        master_df, model_avg = generate_master_summary(all_results)

    # Final summary
    print(f"\n{'='*70}")
    print("SUMMARY GENERATION COMPLETE")
    print(f"{'='*70}")
    print(f"✓ Successfully processed: {len(all_results)} model-year combinations")

    if failed:
        print(f"✗ Failed: {len(failed)} combinations")
        for model, year in failed:
            print(f"  - {model} {year}")

    print(f"\n📁 All summaries saved to: {OUTPUT_DIR}/")
    print("\nOutput structure:")
    print("  • {MODEL}/ - Directory for each model")
    print("    ├── {MODEL}_{YEAR}_overall_performance.csv")
    print("    ├── {MODEL}_{YEAR}_performance_by_region.csv")
    print("    ├── {MODEL}_{YEAR}_performance_by_month.csv")
    print("    ├── {MODEL}_{YEAR}_performance_by_season.csv")
    print("    ├── {MODEL}_{YEAR}_performance_by_station_type.csv")
    print("    ├── {MODEL}_{YEAR}_performance_by_region_month.csv")
    print("    └── {MODEL}_combined_all_years_summary.csv")
    print("  • ALL_MODELS_master_summary.csv - All models/years")
    print("  • ALL_MODELS_average_by_model.csv - Averages per model")

    print("\n" + "="*70)
    print("Use these summaries for data fusion model selection!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
