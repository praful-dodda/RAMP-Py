# import pandas as pd
# import numpy as np
from scipy.spatial import cKDTree
from preprocess import *

# --- User-Defined Information (from previous prompts) ---

# Dictionary of regions of interest
REGIONS = {
    'North America': {'min_lon': -130, 'max_lon': -60, 'min_lat': 25, 'max_lat': 60},
    'Europe': {'min_lon': -10, 'max_lon': 40, 'min_lat': 35, 'max_lat': 70},
    'East Asia': {'min_lon': 100, 'max_lon': 145, 'min_lat': 20, 'max_lat': 50},
    'South Asia': {'min_lon': 60, 'max_lon': 100, 'min_lat': 5, 'max_lat': 35},
    'South America': {'min_lon': -80, 'max_lon': -30, 'min_lat': -60, 'max_lat': 15},
    'Africa': {'min_lon': -20, 'max_lon': 50, 'min_lat': -35, 'max_lat': 40},
    'Australia': {'min_lon': 110, 'max_lon': 160, 'min_lat': -45, 'max_lat': -10},
}

# List of all model sources to be evaluated
model_sources = [
    "AM4", "CAMS", "CESM1-CAM4-Chem", "CESM1-WACCM", "CESM2.2",
    "CHASER", "GEOS-CF", "GEOS-chem", "GEOS-GMI", "GFDL-AM3",
    "M3fusion", "MERRA2-GMI", "MOCAGE", "MRI-ESM1", "MRI-ESM2", "TCR-2"
]

# --- Step 0: Placeholder Data Loading Function ---
# IMPORTANT: Replace this function with your actual data loader.
def get_ozone_file_mock(source, year):
    """
    Placeholder function to simulate your get_ozone_file_mock.
    This generates random sample data for demonstration.
    """
    print(f"--- Loading mock data for {source} in {year} ---")
    if source == "TOAR-II":
        # Generate 50 sample observation stations
        data = {
            'id': range(50),
            'lon': np.random.uniform(-120, 150, 50),
            'lat': np.random.uniform(-50, 70, 50),
            'type': np.random.choice(['rural', 'urban', 'suburban'], 50),
            'country': np.random.choice(['USA', 'DEU', 'CHN', 'BRA'], 50)
        }
        for i in range(1, 13):
            data[f'DMA8_{i}'] = np.random.uniform(20, 80, 50)
        return pd.DataFrame(data)
    else:
        # Generate a sample 2x2 degree model grid
        lons = np.arange(-178, 180, 2)
        lats = np.arange(-88, 90, 2)
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        data = {'lon': lon_grid.flatten(), 'lat': lat_grid.flatten()}
        for i in range(1, 13):
            data[f'DMA8_{i}'] = np.random.uniform(15, 85, len(data['lon']))
        return pd.DataFrame(data)

# --- Step 1: Data Restructuring (Wide to Long) ---
def preprocess_to_long_format(df, is_toar=False):
    """
    Converts a DataFrame from wide format (DMA1, DMA2, ...) to long format.
    """
    id_vars = ['lon', 'lat']
    if is_toar:
        id_vars.extend(['id', 'type', 'country'])

    long_df = pd.melt(df, id_vars=id_vars, var_name='month_str', value_name='ozone')
    long_df['month'] = long_df['month_str'].str.replace('DMA8_', '').astype(int)
    long_df.drop(columns=['month_str'], inplace=True)
    return long_df

# --- Step 2: Geographic and Temporal Tagging ---
def get_region(lat, lon):
    """Assigns a region name based on lat/lon coordinates."""
    for name, bounds in REGIONS.items():
        if bounds['min_lat'] <= lat <= bounds['max_lat'] and \
           bounds['min_lon'] <= lon <= bounds['max_lon']:
            return name
    return 'Other'

def get_season(month, lat):
    """Assigns a season based on month and hemisphere (latitude)."""
    if lat >= 0:  # Northern Hemisphere
        if month in [12, 1, 2]: return 'Winter'
        if month in [3, 4, 5]: return 'Spring'
        if month in [6, 7, 8]: return 'Summer'
        if month in [9, 10, 11]: return 'Autumn'
    else:  # Southern Hemisphere
        if month in [12, 1, 2]: return 'Summer'
        if month in [3, 4, 5]: return 'Autumn'
        if month in [6, 7, 8]: return 'Winter'
        if month in [9, 10, 11]: return 'Spring'
    return 'Unknown'

def add_tags_to_toar(toar_df):
    """Applies region and season tagging to the TOAR dataframe."""
    toar_df['region'] = toar_df.apply(lambda row: get_region(row['lat'], row['lon']), axis=1)
    toar_df['season'] = toar_df.apply(lambda row: get_season(row['month'], row['lat']), axis=1)
    return toar_df

# --- Step 3: The Collocation Engine ---
def collocate_data(toar_df_tagged, model_df_long):
    """
    Finds the nearest model grid point for each TOAR station and merges the data.
    """
    # remove any rows with NaN values in the ozone columns
    toar_df_tagged = toar_df_tagged.dropna(subset=['ozone'])
    model_df_long = model_df_long.dropna(subset=['ozone'])
    
    # Get unique station and model grid locations
    stations = toar_df_tagged[['lon', 'lat', 'id']].drop_duplicates().set_index('id')
    model_grid = model_df_long[['lon', 'lat']].drop_duplicates()

    # Build a k-d tree for fast nearest-neighbor lookup on model grid
    tree = cKDTree(model_grid[['lon', 'lat']].values)
    
    # Find the index of the nearest model grid point for each station
    _, nearest_indices = tree.query(stations[['lon', 'lat']].values, k=1)
    
    # Create a mapping from station ID to the nearest model coordinates
    station_to_model_map = model_grid.iloc[nearest_indices].reset_index(drop=True)
    station_to_model_map['id'] = stations.index
    station_to_model_map.rename(columns={'lon': 'model_lon', 'lat': 'model_lat'}, inplace=True)

    # Merge the mapping back into the tagged TOAR data
    merged_df = pd.merge(toar_df_tagged, station_to_model_map, on='id')
    
    # Merge the model ozone data based on the collocated point and month
    final_df = pd.merge(
        merged_df,
        model_df_long,
        left_on=['model_lon', 'model_lat', 'month'],
        right_on=['lon', 'lat', 'month'],
        suffixes=('_toar', '_model')
    )

    # calculate the geographic distance between the collocated points
    final_df['geo_dist'] = np.sqrt(
        (final_df['lon_toar'] - final_df['model_lon'])**2 +
        (final_df['lat_toar'] - final_df['model_lat'])**2
    )
    
    # Clean up and rename columns for clarity
    final_df.rename(columns={'ozone_toar': 'observed_ozone', 'ozone_model': 'model_ozone'}, inplace=True)
    final_df = final_df.drop(columns=['model_lon', 'model_lat'])
    
    return final_df

# Efficinet Collocation function with xarray 
def collocate_data_xarray_optimized(toar_df_tagged, model_ds):
    """
    Optimized version that uses vectorized operations where possible
    """
    print(f"Collocating {len(toar_df_tagged)} observation points with model data...")
    
    # Extract model grid coordinates
    model_lons = model_ds.lon.values
    model_lats = model_ds.lat.values
    
    # Create a mesh grid of model coordinates
    lon_grid, lat_grid = np.meshgrid(model_lons, model_lats)
    model_points = np.column_stack((lon_grid.flatten(), lat_grid.flatten()))
    
    # Get unique station locations
    stations = toar_df_tagged[['lon', 'lat', 'id']].drop_duplicates().set_index('id')
    
    # Build a k-d tree for fast nearest-neighbor lookup on model grid
    tree = cKDTree(model_points)
    
    # Find the index of the nearest model grid point for each station
    _, nearest_indices = tree.query(stations[['lon', 'lat']].values, k=1)
    
    # Convert flattened indices to grid indices
    lat_indices = nearest_indices // len(model_lons)
    lon_indices = nearest_indices % len(model_lons)
    
    # Create a mapping from station ID to the nearest model coordinates
    station_to_model_map = pd.DataFrame({
        'id': stations.index,
        'model_lat_idx': lat_indices,
        'model_lon_idx': lon_indices,
        'model_lat': model_lats[lat_indices],
        'model_lon': model_lons[lon_indices]
    })
    
    # Merge the mapping back into the tagged TOAR data
    merged_df = pd.merge(toar_df_tagged, station_to_model_map, on='id')
    
    # Create a lookup table for model times to speed up matching
    model_times_lookup = {
        pd.Timestamp(t).strftime('%Y-%m'): i 
        for i, t in enumerate(model_ds.time.values)
    }
    
    # Function to find model value for a given row
    def get_model_value(row):
        time_key = f"{row['year']}-{row['month']:02d}"
        
        # Try to get exact time match
        if time_key in model_times_lookup:
            time_idx = model_times_lookup[time_key]
        else:
            # If no exact match, find nearest time (slower fallback)
            date = pd.Timestamp(year=row['year'], month=row['month'], day=15)
            time_idx = np.abs(model_ds.time.values - np.datetime64(date)).argmin()
        
        return model_ds.mda8.isel(
            time=time_idx, 
            lat=row['model_lat_idx'], 
            lon=row['model_lon_idx']
        ).values.item()
    
    # Apply the function to each row
    print("Extracting model values...")
    merged_df['model_ozone'] = merged_df.apply(get_model_value, axis=1)
    merged_df['observed_ozone'] = merged_df['ozone']  # Assuming 'ozone' is the observation column
    
    # Clean up and format the final dataframe
    final_df = merged_df.drop(columns=['model_lat_idx', 'model_lon_idx'])
    
    print(f"Collocation complete. Generated {len(final_df)} matched data points.")
    
    return final_df
# --- Step 3.1: Create Collocated multimodel dataset ---
def create_collocated_multimodel_dataset(toar_data_path, year, model_sources):
    """
    Creates a collocated dataframe with observations and multiple model predictions
    
    Parameters:
    -----------
    toar_data_path : str
        Path to TOAR observation data
    year : int
        Year to analyze
    model_sources : list
        List of model sources to include
        
    Returns:
    --------
    pd.DataFrame
        Collocated dataframe with the required structure
    """
    # Load TOAR data
    toar_df = pd.read_csv(toar_data_path)
    
    # Add region and season tags
    toar_df_tagged = add_tags_to_toar(toar_df)
    
    # Create empty dataframe to store the final result
    collocated_models_df = None
    
    # Process each model source
    for source in model_sources:
        try:
            # Get model data
            model_file = get_ozone_file(source, year)
            model_df = pd.read_csv(model_file)
            
            # Convert to long format for collocation
            model_df_long = preprocess_to_long_format(model_df, is_toar=False)
            
            # Collocate TOAR observations with this model
            collocated_df = collocate_data(toar_df_tagged, model_df_long)
            
            if collocated_df.empty:
                print(f"No collocated points found for {source} in {year}.")
                continue
                
            # Rename model_ozone column to include model name
            collocated_df.rename(columns={'model_ozone': f'mdl_{source}'}, inplace=True)
            
            # For the first model, keep all columns
            if collocated_models_df is None:
                collocated_models_df = collocated_df
                # Rename observed_ozone to OBS (required by the stacked ensemble)
                collocated_models_df.rename(columns={'observed_ozone': 'OBS'}, inplace=True)
            else:
                # For subsequent models, only merge the model prediction column
                collocated_models_df = pd.merge(
                    collocated_models_df,
                    collocated_df[['id', 'lat', 'lon', 'month', 'day', f'mdl_{source}']],
                    on=['id', 'lat', 'lon', 'month', 'day'],
                    how='inner'
                )
                
        except Exception as e:
            print(f"Failed to process {source} for {year}. Error: {e}")
    
    # Create date and year columns required by the ensemble
    collocated_models_df['date'] = pd.to_datetime(
        collocated_models_df[['year', 'month', 'day']].assign(
            year=year
        )
    )
    collocated_models_df['year'] = year
    
    # Rename id column to station_id as expected by the ensemble
    collocated_models_df.rename(columns={'id': 'station_id'}, inplace=True)
    
    return collocated_models_df

# --- Step 4: Performance Metrics Calculation ---
def calculate_performance_metrics(df_group, observed_col='observed_ozone', model_col='model_ozone'):
    """Calculates performance stats for a given data group."""
    
    if observed_col not in df_group or model_col not in df_group:
        raise ValueError(f"Required columns '{observed_col}' or '{model_col}' not found in the group.")
    
    if df_group.empty or len(df_group) < 2:
        return pd.Series({
            'RMSE': np.nan, 'MAE': np.nan, 'MS': np.nan,
            'r2': np.nan, 'MR': np.nan, 'RMSS': np.nan,
            'ME': np.nan, 'correlation': np.nan,
            'NMB': np.nan, 'n_points': 0
        })

    bias = df_group[model_col] - df_group[observed_col]
    obs_sum = df_group[observed_col].sum()

    metrics = {
        'RMSE': np.sqrt(np.mean(bias**2)), # Root Mean Square Error
        'MAE': np.mean(np.abs(bias)), # Mean Absolute Error
        'ME': bias.mean(), # Mean Bias or Mean Error
        # MS: Mean Standardized Error
        'MS': np.mean(bias / df_group[observed_col]) if obs_sum != 0 else np.nan,
        'r2': np.corrcoef(df_group[model_col], df_group[observed_col])[0, 1]**2, # Coefficient of Determination
        'RMSS': np.sqrt(np.mean((df_group[model_col] - df_group[observed_col])**2)), # Root Mean Square Standardized Error
        # MR: Mean Ratio Variance
        'MR': np.mean(df_group[model_col] / df_group[observed_col]) if obs_sum != 0 else np.nan,

        'correlation': df_group[[model_col, observed_col]].corr().iloc[0, 1],
        'NMB': (bias.sum() / obs_sum) * 100 if obs_sum != 0 else np.nan,
        'n_points': len(df_group)
    }
    return pd.Series(metrics)

# --- Step 5: Main Analysis Workflow ---
def run_evaluation_analysis(model_sources, years_to_analyze, grouping_cols=None):
    """
    Executes the full analysis pipeline for the specified years.
    """
    # check if the input for grouping_cols is within the expected values: ['model_source', 'year', 'region', 'season', 'type']
    if grouping_cols is not None and not all(col in ['model_source', 'year', 'region', 'season', 'type', 'month'] for col in grouping_cols):
        raise ValueError("Invalid grouping_cols. Expected values are: ['model_source', 'year', 'region', 'season', 'type', 'month']")
    
    if grouping_cols is None:
        grouping_cols = ['model_source', 'year', 'region', 'season', 'type', 'month']

    all_results = []
    
    for year in years_to_analyze:
        print(f"\n===== Processing Year: {year} =====")
        # 1. Load and preprocess TOAR data (once per year)
        try:
            toar_df_file = get_ozone_file("TOAR-II", year)
            toar_df_raw = pd.read_csv(toar_df_file)

            if toar_df_raw is None or toar_df_raw.empty:
                print(f"No TOAR-II data found for {year}. Skipping...")
                continue
            
            toar_df_long = preprocess_to_long_format(toar_df_raw, is_toar=True)
            toar_df_tagged = add_tags_to_toar(toar_df_long)
        except Exception as e:
            print(f"Could not load or process TOAR-II data for {year}. Error: {e}")
            continue

        # 2. Loop through each model source
        for source in model_sources:
            try:
                # Load and preprocess model data
                model_df_file = get_ozone_file(source, year)
                model_df_raw = pd.read_csv(model_df_file)
                
                if model_df_raw is None or model_df_raw.empty:
                    print(f"No data found for {source} in {year}. Skipping...")
                    continue
                print(f"Processing {source} for {year}...")
                model_df_long = preprocess_to_long_format(model_df_raw)

                # 3. Collocate data
                collocated_df = collocate_data(toar_df_tagged, model_df_long)
                
                if collocated_df.empty:
                    print(f"No collocated points found for {source} in {year}.")
                    continue
                
                # Add source and year identifiers for grouping
                collocated_df['model_source'] = source
                collocated_df['year'] = year

                # 4. Calculate performance metrics, grouped by all desired dimensions
                collocated_df['month'] = collocated_df['month'].astype(int)  # Ensure month is int for grouping

                performance_summary = collocated_df.groupby(grouping_cols).apply(calculate_performance_metrics).reset_index()
                all_results.append(performance_summary)
                print(f"Successfully processed and evaluated {source} for {year}.")

            except Exception as e:
                print(f"Failed to process {source} for {year}. Error: {e}")

    # 5. Combine all results into a final DataFrame
    if not all_results:
        print("\nAnalysis complete, but no results were generated.")
        return pd.DataFrame()
        
    final_summary_df = pd.concat(all_results, ignore_index=True)
    return [final_summary_df, collocated_df]

## --- Plotting Functions --- ##

def plot_metric_by_model(stats_df, metric='RMSE', year=2022, region='Europe', season='Summer'):
    """
    Creates a bar plot comparing a single metric across all models for a specific
    year, region, and season. It also shows performance by station type.
    """
    # Filter the data for the specific context
    plot_data = stats_df[
        (stats_df['year'] == year) &
        (stats_df['region'] == region) &
        (stats_df['season'] == season)
    ]
    
    if plot_data.empty:
        print(f"No data available for the specified filter: {year}, {region}, {season}")
        return

    plt.figure(figsize=(14, 7))
    sns.barplot(
        data=plot_data,
        x='model_source',
        y=metric,
        hue='type', # Separate bars for rural/urban
        palette='viridis'
    )
    
    plt.title(f'{metric.upper()} by Model for {region}, {season} {year}', fontsize=16)
    plt.ylabel(f'{metric.upper()}', fontsize=12)
    plt.xlabel('Model Source', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_faceted_comparison(stats_df, metric='ME', year=2022):
    """
    Creates a multi-panel plot (facets) to compare a metric across all models,
    broken down by region and season. This is great for an overview.
    """
    plot_data = stats_df[stats_df['year'] == year]
    
    if plot_data.empty:
        print(f"No data available for the year: {year}")
        return
        
    g = sns.catplot(
        data=plot_data,
        x='model_source',
        y=metric,
        hue='model_source',
        col='region',  # Creates columns of plots for each region
        row='season',    # Creates rows of plots for each season
        kind='bar',
        height=4,
        aspect=1.2,
        palette='plasma',
        legend=False,
        sharey=False # Allow y-axes to have different scales
    )
    
    g.fig.suptitle(f'Overall {metric.upper()} Comparison for {year}', y=1.03, fontsize=18)
    g.set_xticklabels(rotation=45, ha='right')
    g.set_axis_labels("Model Source", f'{metric.upper()}')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


def plot_bias_vs_rmse(stats_df, year=2022, region='North America'):
    """
    Creates a scatter plot to show the relationship between Mean Bias and RMSE.
    Ideal models are in the bottom-center (low bias, low RMSE).
    """
    plot_data = stats_df[
        (stats_df['year'] == year) &
        (stats_df['region'] == region)
    ]

    if plot_data.empty:
        print(f"No data available for the specified filter: {year}, {region}")
        return

    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=plot_data,
        x='ME',
        y='RMSE',
        hue='model_source', # Color points by model
        style='season',     # Use different markers for each season
        size='n_points',    # Make point size proportional to data count
        sizes=(50, 500),
        palette='tab10',
        alpha=0.8
    )
    
    plt.axvline(0, color='black', linestyle='--', lw=1) # Add a line for zero bias
    plt.title(f'Mean Bias vs. RMSE for {region} ({year})', fontsize=16)
    plt.xlabel('Mean Bias', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def plot_metric_heatmap(stats_df, metric='correlation', year=2022):
    """
    Creates a heatmap to show a specific metric across models (y-axis) and
    regions (x-axis), averaged over all seasons and types.
    """
    plot_data = stats_df[stats_df['year'] == year]

    if plot_data.empty:
        print(f"No data available for the year: {year}")
        return
        
    # Pivot the data to create a matrix suitable for a heatmap
    heatmap_data = pd.pivot_table(
        plot_data,
        values=metric,
        index='model_source',
        columns='region',
        aggfunc='mean' # Average the metric over seasons/types
    )
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        heatmap_data,
        annot=True,       # Write the data value in each cell
        fmt=".2f",        # Format as a float with 2 decimal places
        cmap='coolwarm',  # Use a diverging colormap
        linewidths=.5
    )
    
    plt.title(f'Average {metric.title()} Heatmap ({year})', fontsize=16)
    plt.xlabel('Region', fontsize=12)
    plt.ylabel('Model Source', fontsize=12)
    plt.tight_layout()
    plt.show()

# --- Plotting Functions ---

def _annotate_scatter_plot(ax, stats_row):
    """Helper function to add a formatted text box with statistics to an axis."""
    if stats_row is None or stats_row.empty:
        return
    
    # Extract stats, handling potential missing values
    stats = stats_row.iloc[0]
    RMSE = stats.get('RMSE', np.nan)
    bias = stats.get('ME', np.nan)
    corr = stats.get('correlation', np.nan)
    n = stats.get('n_points', 0)
    
    # Create the text string
    stats_text = (
        f"N = {n:,.0f}\n"
        f"Bias = {bias:.2f}\n"
        f"RMSE = {RMSE:.2f}\n"
        f"R = {corr:.2f}"
    )
    
    # Add the text box to the plot
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='aliceblue', alpha=0.8))

def generate_single_model_report(model_name, year, collocated_df, all_stats):
    """
    Generates a full visual report for a single model in a given year.
    
    Args:
        model_name (str): The name of the model to report on.
        year (int): The year of the data to use.
        collocated_df (pd.DataFrame): DataFrame with paired observed/model values.
        all_stats (dict): A dictionary of pre-calculated statistics DataFrames.
    """
    print(f"\n{'='*20} Visual Report for: {model_name} ({year}) {'='*20}")

    # --- 1. Overall Performance Scatter Plot ---
    print("Generating: 1. Overall Performance Scatter Plot")
    fig, ax = plt.subplots(figsize=(8, 8))
    
    subset = collocated_df[(collocated_df['model_source'] == model_name) & (collocated_df['year'] == year)]
    
    if not subset.empty:
        sns.scatterplot(data=subset, x='observed_ozone', y='model_ozone', alpha=0.3, ax=ax)
        
        # Add 1:1 line for reference
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        
        ax.set_title(f'Overall Performance for {model_name} ({year})', fontsize=16)
        ax.set_xlabel('Observed Ozone (ppb)', fontsize=12)
        ax.set_ylabel('Modeled Ozone (ppb)', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Annotate with overall stats
        _annotate_scatter_plot(ax, all_stats['overall'])
        plt.show()

    # --- 2. Faceted Scatter Plots (by Region, Season, Month) ---
    for facet_by in ['region', 'season', 'month']:
        print(f"Generating: 2. Faceted Scatter Plot by {facet_by.title()}")
        
        stats_df = all_stats.get(facet_by)
        if stats_df is None or stats_df.empty:
            print(f"   -> No stats found for '{facet_by}' breakdown. Skipping.")
            continue
            
        # Use relplot which creates a FacetGrid
        g = sns.relplot(
            data=subset,
            x='observed_ozone',
            y='model_ozone',
            col=facet_by,
            col_wrap=4 if facet_by in ['season', 'month'] else 3, # Adjust layout
            kind='scatter',
            height=4,
            aspect=1,
            scatter_kws={'alpha': 0.3}
        )
        
        g.fig.suptitle(f'{model_name} Performance by {facet_by.title()} ({year})', y=1.03, fontsize=18)
        g.set_axis_labels("Observed Ozone (ppb)", "Modeled Ozone (ppb)")
        
        # Iterate through each subplot to add 1:1 line and annotation
        for i, ax in enumerate(g.axes.flat):
            # Extract the category for this subplot (e.g., 'Europe' or 'Summer')
            title = ax.get_title()
            if not title: continue
            category = title.split(' = ')[1]
            
            # Try to convert month back to integer for filtering
            try:
                category = int(float(category))
            except ValueError:
                pass

            # Find the matching stats for this specific facet
            stats_for_facet = stats_df[stats_df[facet_by] == category]
            _annotate_scatter_plot(ax, stats_for_facet)
            
            # Add 1:1 line
            lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
            ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.grid(True, linestyle='--', alpha=0.6)
            
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()

# --- Execute the Analysis ---
if __name__ == "__main__":
    # Define the years you want to run the analysis for
    analysis_years = 2017 
    model = "MERRA2-GMI"  # Example model to analyze

    final_results = run_evaluation_analysis([model], [analysis_years])

    print("\n\n===== FINAL PERFORMANCE SUMMARY =====")
    if not final_results.empty:
        # Set pandas display options to see all columns
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 150)
        print(final_results)
        
        # Optionally, save the results to a CSV file
        # final_results.to_csv("model_performance_summary.csv", index=False)
        # print("\nResults saved to model_performance_summary.csv")
    else:
        print("No summary data was produced.")