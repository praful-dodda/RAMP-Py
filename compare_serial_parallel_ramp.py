import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# --- NEW: Import the function to find model files ---
# Make sure preprocess.py is in the same directory or your PYTHONPATH
try:
    from preprocess import get_ozone_file
except ImportError:
    print("Warning: Could not import 'get_ozone_file' from 'preprocess.py'.")
    print("Falling back to manual file path construction.")
    # Define a fallback function if the import fails
    def get_ozone_file(model_name, year):
        path = f"./{model_name}_{year}.csv"
        return path if os.path.exists(path) else None

# --- Configuration ---
MODEL_NAME = "MERRA2-GMI"
YEARS = [2016, 2017, 2018]
parallel_version = "v3"
# serial_version = ""

# --- Directory setup ---
RAMP_DATA_DIR = './ramp_data'
OUTPUT_DIR = './comparison_plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Region definitions for detailed analysis ---
REGIONS = {
    'North America': {'min_lon': -130, 'max_lon': -60, 'min_lat': 25, 'max_lat': 60},
    'Europe': {'min_lon': -10, 'max_lon': 40, 'min_lat': 35, 'max_lat': 70},
    'East Asia': {'min_lon': 100, 'max_lon': 145, 'min_lat': 20, 'max_lat': 50},
    'South Asia': {'min_lon': 60, 'max_lon': 100, 'min_lat': 5, 'max_lat': 35},
    'Other': {'min_lon': -180, 'max_lon': 180, 'min_lat': -90, 'max_lat': 90} # Fallback
}

def get_region(lat, lon):
    """Assigns a region name based on lat/lon coordinates."""
    for name, bounds in REGIONS.items():
        if bounds['min_lat'] <= lat <= bounds['max_lat'] and \
           bounds['min_lon'] <= lon <= bounds['max_lon']:
            return name
    return 'Other'

# --- Helper function to safely load data ---
def load_lambda_data(filepath_base):
    """Tries to load a .parquet file, falls back to .csv."""
    parquet_path = f"{filepath_base}.parquet"
    csv_path = f"{filepath_base}.csv"
    
    if os.path.exists(parquet_path):
        print(f"Loading {parquet_path}")
        return pd.read_parquet(parquet_path)
    elif os.path.exists(csv_path):
        print(f"Loading {csv_path}")
        return pd.read_csv(csv_path)
    else:
        print(f"Warning: Could not find file for base path: {filepath_base}")
        return None

# --- NEW: Helper function to annotate comparison plots with statistics ---
def _annotate_comparison_plot(ax, diff_data):
    """Calculates difference stats and adds a text box to the provided axes."""
    if diff_data.empty:
        return
    
    n_points = len(diff_data)
    mean_diff = diff_data.mean()
    max_abs_diff = diff_data.abs().max()

    stats_text = (
        f"N = {n_points:,.0f}\n"
        f"Mean Diff = {mean_diff:.2e}\n"
        f"Max Abs Diff = {max_abs_diff:.2e}"
    )
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='aliceblue', alpha=0.8))


# --- MODIFIED: Function to analyze and plot data for a single year ---
def analyze_and_plot_year(df, model_name, year):
    """
    Takes a single-year comparison DataFrame and generates all stats and plots.
    """
    if df.empty:
        print(f"No data to analyze for year {year}.")
        return

    # Statistical Summary
    print(f"\n\n--- Analysis of Differences for {year} (Parallel - Serial) ---")
    print(f"\nDescriptive Statistics for Lambda1 Differences ({year}):")
    print(df['lambda1_diff'].describe())
    print(f"\nDescriptive Statistics for Lambda2 Differences ({year}):")
    print(df['lambda2_diff'].describe())

    # Overall Distribution Plot (Histogram)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.histplot(df['lambda1_diff'], bins=50, ax=axes[0])
    axes[0].set_title('Overall Distribution of Lambda1 Differences')
    sns.histplot(df['lambda2_diff'], bins=50, ax=axes[1])
    axes[1].set_title('Overall Distribution of Lambda2 Differences')
    plt.suptitle(f'Overall Histogram of Differences for {model_name} {year}')
    plt.savefig(os.path.join(OUTPUT_DIR, f'overall_difference_histogram_{model_name}_{year}.png'), dpi=300)
    plt.close()

    # Overall Scatter Plot
    sample_df = df.sample(n=min(50000, len(df)), random_state=42)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.scatterplot(data=sample_df, x='lambda1_serial', y='lambda1_parallel', alpha=0.5, s=10, ax=axes[0])
    axes[0].set_title('Overall Lambda1: Serial vs. Parallel')
    lims1 = [min(axes[0].get_xlim()[0], axes[0].get_ylim()[0]), max(axes[0].get_xlim()[1], axes[0].get_ylim()[1])]
    axes[0].plot(lims1, lims1, 'k--', alpha=0.75, zorder=0)
    _annotate_comparison_plot(axes[0], sample_df['lambda1_diff']) # Add annotation

    sns.scatterplot(data=sample_df, x='lambda2_serial', y='lambda2_parallel', alpha=0.5, s=10, ax=axes[1])
    axes[1].set_title('Overall Lambda2: Serial vs. Parallel')
    lims2 = [min(axes[1].get_xlim()[0], axes[1].get_ylim()[0]), max(axes[1].get_xlim()[1], axes[1].get_ylim()[1])]
    axes[1].plot(lims2, lims2, 'k--', alpha=0.75, zorder=0)
    _annotate_comparison_plot(axes[1], sample_df['lambda2_diff']) # Add annotation

    plt.suptitle(f'Overall Comparison of Outputs for {model_name} {year}')
    plt.savefig(os.path.join(OUTPUT_DIR, f'overall_output_scatterplot_{model_name}_{year}.png'), dpi=300)
    plt.close()
    
    # --- NEW: Spatial Map of Differences ---
    month_to_plot = 7 # Plot a sample summer month
    spatial_df = df[df['month'] == month_to_plot]
    if not spatial_df.empty:
        plt.figure(figsize=(12, 6))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        vmax = np.nanmax(np.abs(spatial_df['lambda1_diff']))
        vmax = max(vmax, 1e-9) # Ensure vmax is not zero
        vmin = -vmax
        sc = ax.scatter(spatial_df['lon'], spatial_df['lat'], c=spatial_df['lambda1_diff'],
                        cmap='coolwarm', s=1, transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax)
        plt.colorbar(sc, label='Lambda1 Difference (Parallel - Serial)')
        plt.title(f'Spatial Map of Lambda1 Discrepancies for {model_name} {year} - Month {month_to_plot}')
        plt.savefig(os.path.join(OUTPUT_DIR, f'spatial_difference_map_{model_name}_{year}.png'), dpi=300)
        plt.close()


    # Boxplot of Differences by Month
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=False) # MODIFIED sharey
    sns.boxplot(data=df, x='month', y='lambda1_diff', ax=axes[0])
    axes[0].set_title('Lambda1 Difference Distribution by Month')
    axes[0].grid(True, linestyle='--')
    sns.boxplot(data=df, x='month', y='lambda2_diff', ax=axes[1])
    axes[1].set_title('Lambda2 Difference Distribution by Month')
    axes[1].grid(True, linestyle='--')
    plt.suptitle(f'Monthly Analysis of Discrepancies for {model_name} {year}')
    plt.savefig(os.path.join(OUTPUT_DIR, f'monthly_difference_boxplot_{model_name}_{year}.png'), dpi=300)
    plt.close()

    # Boxplot of Differences by Region
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=False) # MODIFIED sharey
    sns.boxplot(data=df, x='region', y='lambda1_diff', ax=axes[0])
    axes[0].set_title('Lambda1 Difference Distribution by Region')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, linestyle='--')
    sns.boxplot(data=df, x='region', y='lambda2_diff', ax=axes[1])
    axes[1].set_title('Lambda2 Difference Distribution by Region')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, linestyle='--')
    plt.suptitle(f'Regional Analysis of Discrepancies for {model_name} {year}')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(OUTPUT_DIR, f'regional_difference_boxplot_{model_name}_{year}.png'), dpi=300)
    plt.close()
    
    # Faceted Scatter Plot by Month
    # --- THIS IS THE CORRECTED SECTION ---
    g = sns.relplot(
        data=sample_df,
        x='lambda1_serial',
        y='lambda1_parallel',
        col='month',
        col_wrap=4,
        kind='scatter',
        height=3,
        aspect=1,
        alpha=0.3, # Pass arguments directly
        s=5        # Pass arguments directly
    )
    g.fig.suptitle(f'Monthly Breakdown for {model_name} {year}: Lambda1 Serial vs. Parallel', y=1.02)
    for ax in g.axes.flat:
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
    plt.savefig(os.path.join(OUTPUT_DIR, f'monthly_lambda1_scatterplot_{model_name}_{year}.png'), dpi=300)
    plt.close()
    
    print(f"\nAll comparison plots for {year} saved to directory: {OUTPUT_DIR}")


# --- Main Comparison Loop (MODIFIED) ---
for year in YEARS:
    print(f"\n{'='*20} Processing Year: {year} {'='*20}")
    
    model_grid_file = get_ozone_file(MODEL_NAME, year)
    if model_grid_file is None:
        print(f"Error: Could not find model grid file for {MODEL_NAME} in {year}.")
        print("Skipping this year.")
        continue
    
    try:
        coords_df = pd.read_csv(model_grid_file)[['lon', 'lat']]
        print(f"Loaded coordinates from {model_grid_file}")
    except (FileNotFoundError, KeyError) as e:
        print(f"Error loading or reading coordinates from {model_grid_file}: {e}")
        print("Skipping this year.")
        continue

    # load serial data
    base_path_l1_serial = f"{RAMP_DATA_DIR}/lambda1_{MODEL_NAME}_{year}"
    base_path_l2_serial = f"{RAMP_DATA_DIR}/lambda2_{MODEL_NAME}_{year}"

    # load parallel data
    base_path_l1_parallel = f"{RAMP_DATA_DIR}/lambda1_{MODEL_NAME}_{year}_{parallel_version}-parallel"
    base_path_l2_parallel = f"{RAMP_DATA_DIR}/lambda2_{MODEL_NAME}_{year}_{parallel_version}-parallel"

    l1_serial_df = load_lambda_data(base_path_l1_serial)
    l1_parallel_df = load_lambda_data(base_path_l1_parallel)
    l2_serial_df = load_lambda_data(base_path_l2_serial)
    l2_parallel_df = load_lambda_data(base_path_l2_parallel)
    
    if any(df is None for df in [l1_serial_df, l1_parallel_df, l2_serial_df, l2_parallel_df]):
        print(f"Skipping year {year} due to missing lambda data.")
        continue

    l1_serial_df = pd.concat([coords_df, l1_serial_df], axis=1)
    l1_parallel_df = pd.concat([coords_df, l1_parallel_df], axis=1)
    l2_serial_df = pd.concat([coords_df, l2_serial_df], axis=1)
    l2_parallel_df = pd.concat([coords_df, l2_parallel_df], axis=1)

    l1_serial_long = pd.melt(l1_serial_df, id_vars=['lon', 'lat'], var_name='month_str', value_name='lambda1_serial')
    l1_parallel_long = pd.melt(l1_parallel_df, id_vars=['lon', 'lat'], var_name='month_str', value_name='lambda1_parallel')
    l2_serial_long = pd.melt(l2_serial_df, id_vars=['lon', 'lat'], var_name='month_str', value_name='lambda2_serial')
    l2_parallel_long = pd.melt(l2_parallel_df, id_vars=['lon', 'lat'], var_name='month_str', value_name='lambda2_parallel')
    
    merged_l1 = pd.merge(l1_serial_long, l1_parallel_long, on=['lon', 'lat', 'month_str'])
    merged_l2 = pd.merge(l2_serial_long, l2_parallel_long, on=['lon', 'lat', 'month_str'])
    comparison_df = pd.merge(merged_l1, merged_l2, on=['lon', 'lat', 'month_str'])
    
    comparison_df['month'] = comparison_df['month_str'].str.replace('DMA8_', '').astype(int)
    comparison_df['year'] = year
    
    comparison_df['lambda1_diff'] = comparison_df['lambda1_parallel'] - comparison_df['lambda1_serial']
    comparison_df['lambda2_diff'] = comparison_df['lambda2_parallel'] - comparison_df['lambda2_serial']
    
    comparison_df.dropna(inplace=True)
    comparison_df['region'] = comparison_df.apply(lambda row: get_region(row['lat'], row['lon']), axis=1)

    analyze_and_plot_year(comparison_df, MODEL_NAME, year)

print("\n\nComparison script finished.")

