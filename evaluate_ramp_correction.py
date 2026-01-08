import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os

# Assuming your original evaluation functions are in this file
from evaluate_models import calculate_performance_metrics, get_region
from preprocess import get_ozone_file

# --- Configuration ---
# Update these paths to match your file locations and names
MODEL_NAME = "IASI-GOME2" # The model you are evaluating
YEAR = 2017 # The year you ran the analysis for
VERSION = "v3-parallel" # The version name you used for saving files
FILE_FORMAT = "parquet" # Change to 'parquet' if you prefer that format

# File Paths
RAMP_DATA_DIR = './ramp_data'
MODEL_DATA_DIR = '.' # Assuming model data is in the current directory
COLLOCATED_DATA_PATH = f'./ramp_data/collocated_data_{MODEL_NAME}_{YEAR}.csv'

# Ensure the collocated data file exists
if not os.path.exists(COLLOCATED_DATA_PATH):
    print(f"Error: The collocated data file {COLLOCATED_DATA_PATH} does not exist.")
    print("Please ensure the collocated data is generated and saved before running this script.")
    # exit(1)

# Output directory for evaluation plots
EVAL_PLOT_DIR = f'ramp_evaluation_plots/{MODEL_NAME}_{YEAR}_{VERSION}'
os.makedirs(EVAL_PLOT_DIR, exist_ok=True)


# --- Step 1: Data Loading and Preparation ---
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

def load_and_prepare_data():
    """
    Loads all necessary data files, applies the RAMP correction,
    and merges everything into a unified DataFrame for evaluation.
    """
    print("--- Step 1: Loading and Preparing Data ---")
    
    # 1.1: Load the datasets
    try:
        # Original gridded model output
        original_model_file = get_ozone_file(MODEL_NAME, YEAR)
        if original_model_file is None:
            raise FileNotFoundError(f"Could not find model file for {MODEL_NAME} in {YEAR}.")
        
        # Load the original model data
        original_model_df = pd.read_csv(original_model_file)
        print(f"Loaded original model data: {original_model_file}")

        # RAMP correction file (lambda1)
        lambda1_file = f"{RAMP_DATA_DIR}/lambda1_{MODEL_NAME}_{YEAR}_{VERSION}"
        lambda1_df = load_lambda_data(lambda1_file)
        if lambda1_df is None:
            raise FileNotFoundError(f"Could not find RAMP lambda1 data file: {lambda1_file} in csv or parquet format.")
        print(f"Loaded RAMP lambda1 data: {lambda1_file}")
        
        # Collocated data which contains observations
        # Try loading CSV first, then parquet if CSV fails
        try:
            collocated_df = pd.read_csv(COLLOCATED_DATA_PATH)
            print(f"Loaded collocated observation data: {COLLOCATED_DATA_PATH}")
        except FileNotFoundError:
            parquet_path = COLLOCATED_DATA_PATH.replace('.csv', '.parquet')
            collocated_df = pd.read_parquet(parquet_path)
            print(f"Loaded collocated observation data: {parquet_path}")

    except FileNotFoundError as e:
        print(f"Error: Could not find a required data file. {e}")
        print("Please ensure all paths in the Configuration section are correct.")
        return None, None

    # 1.2: Rename columns for clarity before merging
    # The lambda1 file represents the final corrected ozone values
    dma_cols = {f'DMA8_{i}': f'ramp_corrected_{i}' for i in range(1, 13)}
    lambda1_df.rename(columns=dma_cols, inplace=True)
    
    # 1.3: Combine the original model grid with the RAMP corrections
    # gridded_eval_df = pd.concat([original_model_df, lambda1_df.drop(columns=['lon', 'lat'])], axis=1)
    gridded_eval_df = pd.concat([original_model_df, lambda1_df], axis=1)

    # 1.4: Restructure all data to long format for easier analysis
    # Gridded data (original and RAMP)
    gridded_long = pd.melt(gridded_eval_df, id_vars=['lon', 'lat'], var_name='var_month', value_name='ozone')
    gridded_long['month'] = gridded_long['var_month'].str.extract(r'_(\d+)').astype(int)
    gridded_long['source'] = np.where(gridded_long['var_month'].str.contains('ramp'), 'RAMP-Corrected', 'Original Model')
    gridded_long.drop(columns=['var_month'], inplace=True)

    # 1.5: Create a unified DataFrame for model-vs-observation comparison
    # We only need the observed ozone and location info from the collocated file
    obs_df = collocated_df[['lon_toar', 'lat_toar', 'observed_ozone', 'month']].copy()
    obs_df.rename(columns={'lon_toar': 'lon', 'lat_toar': 'lat'}, inplace=True)
    
    # For each observation point, find the corresponding original and RAMP-corrected values
    # This requires merging with the now long-format gridded data
    
    # Pivot the gridded data to have original and RAMP as columns
    gridded_pivot = gridded_long.pivot_table(index=['lon', 'lat', 'month'], columns='source', values='ozone').reset_index()
    gridded_pivot.rename(columns={'Original Model': 'original_model_ozone', 'RAMP-Corrected': 'ramp_corrected_ozone'}, inplace=True)

    # Merge observations with the gridded data based on the NEAREST grid point
    # This is a simplified collocation for evaluation purposes
    from scipy.spatial import cKDTree
    grid_coords = gridded_pivot[['lon', 'lat']].drop_duplicates()
    tree = cKDTree(grid_coords)
    
    # Find nearest grid point for each observation
    _, indices = tree.query(obs_df[['lon', 'lat']].values, k=1)
    
    # Get the coordinates of the nearest grid points
    nearest_grid_points = grid_coords.iloc[indices].reset_index(drop=True)
    obs_df['grid_lon'] = nearest_grid_points['lon']
    obs_df['grid_lat'] = nearest_grid_points['lat']

    # Now merge using the grid coordinates and month
    eval_df = pd.merge(obs_df, gridded_pivot, left_on=['grid_lon', 'grid_lat', 'month'], right_on=['lon', 'lat', 'month'], suffixes=('_obs', ''))
    
    # Add geographic tags for regional analysis
    eval_df['region'] = eval_df.apply(lambda row: get_region(row['lat_obs'], row['lon_obs']), axis=1)

    print("Data preparation complete.")
    return eval_df.dropna(), gridded_long.dropna()

# --- Step 2: Statistical Evaluation ---

def perform_statistical_evaluation(eval_df):
    """
    Calculates and prints performance metrics for original vs RAMP-corrected data.
    """
    print("\n--- Step 2: Performing Statistical Evaluation ---")
    if eval_df is None:
        print("Evaluation DataFrame not available. Skipping stats.")
        return

    # Create two dataframes for evaluation, one for original and one for RAMP
    original_eval = eval_df[['observed_ozone', 'original_model_ozone', 'region']].copy()
    original_eval.rename(columns={'original_model_ozone': 'model_ozone'}, inplace=True)

    ramp_eval = eval_df[['observed_ozone', 'ramp_corrected_ozone', 'region']].copy()
    ramp_eval.rename(columns={'ramp_corrected_ozone': 'model_ozone'}, inplace=True)

    # Calculate metrics
    stats_original = calculate_performance_metrics(original_eval)
    stats_ramp = calculate_performance_metrics(ramp_eval)
    
    # Calculate regional stats
    regional_stats_orig = original_eval.groupby('region').apply(calculate_performance_metrics)
    regional_stats_ramp = ramp_eval.groupby('region').apply(calculate_performance_metrics)

    # Print results in a formatted table
    summary = pd.DataFrame({
        'Original Model': stats_original,
        'RAMP-Corrected': stats_ramp
    })
    print("\n--- Overall Performance Metrics ---")
    print(summary.round(3))
    
    print("\n--- Regional Performance (RMSE) ---")
    regional_summary = pd.DataFrame({
        'Original_RMSE': regional_stats_orig['RMSE'],
        'RAMP_RMSE': regional_stats_ramp['RMSE']
    })
    regional_summary['Improvement'] = regional_summary['Original_RMSE'] - regional_summary['RAMP_RMSE']
    print(regional_summary.round(3))


# --- NEW: Helper function to annotate plots with statistics ---
def _annotate_plot(ax, x_data, y_data):
    """Calculates stats and adds a text box to the provided axes."""
    if x_data.empty or y_data.empty:
        return
    
    # Calculate metrics
    n_points = len(x_data)
    bias = np.mean(y_data - x_data)
    rmse = np.sqrt(np.mean((y_data - x_data)**2))
    # Calculate correlation, handle case with no variance
    if x_data.nunique() > 1 and y_data.nunique() > 1:
        corr = np.corrcoef(x_data, y_data)[0, 1]
    else:
        corr = np.nan

    stats_text = (
        f"N = {n_points:,.0f}\n"
        f"Bias = {bias:.2f}\n"
        f"RMSE = {rmse:.2f}\n"
        f"R = {corr:.2f}"
    )
    
    # Add the text box to the plot's top left corner
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='aliceblue', alpha=0.8))


# --- Step 3: Visual Evaluation (MODIFIED) ---

def perform_visual_evaluation(eval_df, gridded_long_df):
    """
    Generates a series of plots to visually compare the datasets.
    """
    print("\n--- Step 3: Generating Visualizations ---")
    if eval_df is None or gridded_long_df is None:
        print("DataFrames not available. Skipping visualizations.")
        return

    # 3.1: Spatial Data Preparation
    month_to_plot = 7
    spatial_df = gridded_long_df[gridded_long_df['month'] == month_to_plot]
    spatial_pivot = spatial_df.pivot_table(index=['lon', 'lat'], columns='source', values='ozone').reset_index()
    spatial_pivot['difference'] = spatial_pivot['RAMP-Corrected'] - spatial_pivot['Original Model']
    
    # --- NEW: Calculate data extent for better map zoom ---
    lon_min, lon_max = spatial_pivot['lon'].min(), spatial_pivot['lon'].max()
    lat_min, lat_max = spatial_pivot['lat'].min(), spatial_pivot['lat'].max()
    # Add a 5% buffer around the data
    lon_buffer = (lon_max - lon_min) * 0.05
    lat_buffer = (lat_max - lat_min) * 0.05
    map_extent = [lon_min - lon_buffer, lon_max + lon_buffer, lat_min - lat_buffer, lat_max + lat_buffer]


    # --- Side-by-Side Before and After Map ---
    # Determine the common color scale
    vmin = min(spatial_pivot['Original Model'].min(), spatial_pivot['RAMP-Corrected'].min())
    vmax = max(spatial_pivot['Original Model'].max(), spatial_pivot['RAMP-Corrected'].max())
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    fig.suptitle(f'Model Ozone Before and After RAMP Correction - Month {month_to_plot}', fontsize=16)

    # Plot Original Model
    axes[0].add_feature(cfeature.COASTLINE)
    axes[0].add_feature(cfeature.BORDERS, linestyle=':')
    sc_orig = axes[0].scatter(spatial_pivot['lon'], spatial_pivot['lat'], c=spatial_pivot['Original Model'],
                               cmap='viridis', s=1, transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax)
    axes[0].set_title('Original Model Output')
    axes[0].set_extent(map_extent, crs=ccrs.PlateCarree()) # Set map extent

    # Plot RAMP-Corrected Model
    axes[1].add_feature(cfeature.COASTLINE)
    axes[1].add_feature(cfeature.BORDERS, linestyle=':')
    sc_ramp = axes[1].scatter(spatial_pivot['lon'], spatial_pivot['lat'], c=spatial_pivot['RAMP-Corrected'],
                               cmap='viridis', s=1, transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax)
    axes[1].set_title('RAMP-Corrected Output')
    axes[1].set_extent(map_extent, crs=ccrs.PlateCarree()) # Set map extent

    # Add a single colorbar for both maps
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(sc_ramp, cax=cbar_ax)
    cbar.set_label('Ozone (ppb)')
    
    plt.savefig(os.path.join(EVAL_PLOT_DIR, f'spatial_before_after_map_month_{month_to_plot}.png'), dpi=300)
    plt.close()
    print(f"Saved side-by-side comparison map to {EVAL_PLOT_DIR}")


    # 3.2: Spatial Difference Map
    plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    
    vmax_diff = np.nanmax(np.abs(spatial_pivot['difference']))
    vmin_diff = -vmax_diff
    
    sc_diff = ax.scatter(spatial_pivot['lon'], spatial_pivot['lat'], c=spatial_pivot['difference'],
                         cmap='coolwarm', s=1, transform=ccrs.PlateCarree(), vmin=vmin_diff, vmax=vmax_diff)
    
    ax.set_extent(map_extent, crs=ccrs.PlateCarree()) # Set map extent
    
    plt.colorbar(sc_diff, label='Difference (RAMP - Original) in ppb')
    plt.title(f'Spatial Impact of RAMP Correction - Month {month_to_plot}')
    plt.savefig(os.path.join(EVAL_PLOT_DIR, f'spatial_difference_map_month_{month_to_plot}.png'), dpi=300)
    plt.close()
    print(f"Saved spatial difference map to {EVAL_PLOT_DIR}")

    # 3.3: Scatter Plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    
    # Original Model vs. Obs
    sns.scatterplot(data=eval_df, x='observed_ozone', y='original_model_ozone', alpha=0.3, ax=axes[0])
    axes[0].set_title('Original Model vs. Observations')
    _annotate_plot(axes[0], eval_df['observed_ozone'], eval_df['original_model_ozone'])
    
    # RAMP-Corrected vs. Obs
    sns.scatterplot(data=eval_df, x='observed_ozone', y='ramp_corrected_ozone', alpha=0.3, ax=axes[1])
    axes[1].set_title('RAMP-Corrected vs. Observations')
    _annotate_plot(axes[1], eval_df['observed_ozone'], eval_df['ramp_corrected_ozone'])
    
    # Add 1:1 line to both
    lims = [min(axes[0].get_xlim()[0], axes[0].get_ylim()[0]), max(axes[0].get_xlim()[1], axes[0].get_ylim()[1])]
    for ax in axes:
        ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
        ax.set_xlabel('Observed Ozone (ppb)')
        ax.set_ylabel('Modeled Ozone (ppb)')
        ax.grid(True)
    
    plt.suptitle('Model Performance Before and After RAMP Correction', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(EVAL_PLOT_DIR, 'scatter_comparison.png'), dpi=300)
    plt.close()
    print(f"Saved scatter plot comparison to {EVAL_PLOT_DIR}")

    # 3.4: Time Series Analysis by Region
    timeseries_df = eval_df.melt(id_vars=['region', 'month'], 
                                 value_vars=['observed_ozone', 'original_model_ozone', 'ramp_corrected_ozone'],
                                 var_name='source', value_name='ozone')
    timeseries_df['source'] = timeseries_df['source'].replace({
        'observed_ozone': 'Observations',
        'original_model_ozone': 'Original Model',
        'ramp_corrected_ozone': 'RAMP-Corrected'
    })
    
    g = sns.FacetGrid(timeseries_df, col="region", col_wrap=3, height=4, sharey=False)
    g.map_dataframe(sns.lineplot, x='month', y='ozone', hue='source', style='source', markers=True)
    g.add_legend(title='Source')
    g.fig.suptitle('Monthly Average Ozone by Region: Before and After RAMP', y=1.02)
    g.set_axis_labels('Month', 'Ozone (ppb)')
    g.set_titles(col_template="{col_name}")
    plt.xticks(range(1, 13))

    plt.savefig(os.path.join(EVAL_PLOT_DIR, 'regional_timeseries_faceted.png'), dpi=300)
    plt.close()
    print(f"Saved faceted regional time series plot to {EVAL_PLOT_DIR}")

    # 3.5: Distribution Plot (Histogram/KDE)
    plt.figure(figsize=(10, 7))
    sns.kdeplot(data=eval_df['observed_ozone'], label='Observations', fill=True, alpha=0.5)
    sns.kdeplot(data=eval_df['original_model_ozone'], label='Original Model', fill=True, alpha=0.5)
    sns.kdeplot(data=eval_df['ramp_corrected_ozone'], label='RAMP-Corrected', fill=True, alpha=0.5)
    
    plt.title('Distribution of Ozone Values at Observation Sites')
    plt.xlabel('Ozone (ppb)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(EVAL_PLOT_DIR, 'distribution_comparison.png'), dpi=300)
    plt.close()
    print(f"Saved distribution comparison plot to {EVAL_PLOT_DIR}")


# --- Main Execution ---
if __name__ == "__main__":
    # First, we need to ensure the collocated_df is saved somewhere.
    # If it's generated by another script, make sure it saves the file.
    # For this script to run, we are assuming 'collocated_df.csv' exists.
    # You might need to add a line like this to your main analysis script:
    # `collocated_df.to_csv(COLLOCATED_DATA_PATH, index=False)`
    
    # Run the full evaluation
    evaluation_df, gridded_df = load_and_prepare_data()
    
    if evaluation_df is not None:
        perform_statistical_evaluation(evaluation_df)
        perform_visual_evaluation(evaluation_df, gridded_df)
        print("\nEvaluation complete.")
    else:
        print("\nEvaluation could not be completed due to data loading errors.")
