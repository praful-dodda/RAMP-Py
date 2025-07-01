import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import os
import warnings


# Add world map outline using cartopy
import cartopy.feature as cfeature
import cartopy.crs as ccrs
# import BoundaryNorm for discrete colormap
from matplotlib.colors import BoundaryNorm


# Suppress RankWarning for polynomial fitting in plots
warnings.filterwarnings("ignore", category=np.RankWarning)

# --- Core Helper Functions ---

def get_time_window(month_index, num_months=12):
    """
    Defines a 3-month window around the current month index (0-11).
    Handles edge cases for the beginning and end of the year.
    
    Args:
        month_index (int): The index of the current month (0 for January).
        num_months (int): Total number of months in the cycle.

    Returns:
        list: A list of month indices [t-1, t, t+1] adjusted for wrapping.
    """
    if month_index == 0:
        return [num_months - 1, 0, 1]
    elif month_index == num_months - 1:
        return [num_months - 2, num_months - 1, 0]
    else:
        return [month_index - 1, month_index, month_index + 1]

def compute_decile_curves(model_vals, obs_vals, total_bins=10, min_points_for_curve=20): # --- MODIFIED ---
    """
    Computes decile curves for mean (lambda1) and variance (lambda2).
    
    Args:
        model_vals (np.ndarray): Array of model values.
        obs_vals (np.ndarray): Array of corresponding observation values.
        total_bins (int): The number of bins to use (e.g., 10 for deciles).
        min_points_for_curve (int): The minimum number of valid data pairs required. # --- NEW ---

    Returns:
        tuple: A tuple containing:
            - decile_bounds (np.ndarray): Mean model value in each bin.
            - curve1 (np.ndarray): Mean observation value in each bin (lambda1).
            - curve2 (np.ndarray): Variance of observation values in each bin (lambda2).
            - bin_edges (np.ndarray): The edges of the bins.
    """
    # --- NEW: Add a check for the minimum number of points ---
    if len(model_vals) < min_points_for_curve:
        # Not enough data to form a reliable curve, return empty arrays
        return np.array([]), np.array([]), np.array([]), np.array([])

    try:
        # Use percentiles to define bin edges
        bin_edges = np.percentile(model_vals, np.linspace(0, 100, total_bins + 1))
        # Ensure the last bin edge includes the max value
        bin_edges[-1] = np.maximum(bin_edges[-1], model_vals.max())
        bin_edges[0] = np.minimum(bin_edges[0], model_vals.min())
    except IndexError:
        return np.array([]), np.array([]), np.array([]), np.array([])

    decile_bounds = np.full(total_bins, np.nan)
    curve1 = np.full(total_bins, np.nan)
    curve2 = np.full(total_bins, np.nan)

    # Calculate statistics for each bin
    for i in range(total_bins):
        if i < total_bins - 1:
            mask = (model_vals >= bin_edges[i]) & (model_vals < bin_edges[i+1])
        else: # Handle the last bin inclusively
            mask = (model_vals >= bin_edges[i]) & (model_vals <= bin_edges[i+1])

        if np.sum(mask) > 0: # --- MODIFIED: Check sum > 0 for robustness ---
            decile_bounds[i] = np.mean(model_vals[mask])
            curve1[i] = np.mean(obs_vals[mask])
            # Require at least 2 points to calculate variance
            curve2[i] = np.var(obs_vals[mask]) if np.sum(mask) > 1 else 0.0

    return decile_bounds, curve1, curve2, bin_edges
    
def enforce_monotonicity(lambda_curve):
    """
    Ensures the lambda1 curve is monotonically non-decreasing.
    This is a direct translation of the MATLAB logic.

    Args:
        lambda_curve (np.ndarray): The raw lambda1 curve.

    Returns:
        np.ndarray: The monotonically adjusted lambda1 curve.
    """
    if lambda_curve is None or len(lambda_curve[~np.isnan(lambda_curve)]) == 0:
        return lambda_curve

    monotonic_lambda = lambda_curve.copy()
    total_bins = len(monotonic_lambda)
    middle_bin = total_bins // 2
    
    mean_val = np.nanmean(monotonic_lambda)
    if np.isnan(mean_val):
        # Fallback if all values are NaN
        return monotonic_lambda

    # Adjust from middle to left
    for i in range(middle_bin, -1, -1):
        if np.isnan(monotonic_lambda[i]):
            continue
        
        if i == middle_bin:
            if monotonic_lambda[i] > mean_val:
                monotonic_lambda[i] = mean_val
        else:
            # Find next non-NaN value to the right
            next_val = np.nan
            for k in range(i + 1, total_bins):
                if not np.isnan(monotonic_lambda[k]):
                    next_val = monotonic_lambda[k]
                    break
            if not np.isnan(next_val) and monotonic_lambda[i] > next_val:
                monotonic_lambda[i] = next_val

    # Adjust from middle+1 to right
    for i in range(middle_bin + 1, total_bins):
        if np.isnan(monotonic_lambda[i]):
            continue
        
        # Find previous non-NaN value to the left
        prev_val = np.nan
        for k in range(i - 1, -1, -1):
            if not np.isnan(monotonic_lambda[k]):
                prev_val = monotonic_lambda[k]
                break
        if not np.isnan(prev_val) and monotonic_lambda[i] < prev_val:
            monotonic_lambda[i] = prev_val
            
    return monotonic_lambda

def interpolate_ramp(decile_bounds, curve, current_model_val): # --- MODIFIED: Removed avg_slope ---
    """
    Interpolates or extrapolates a value from a RAMP curve.
    
    Args:
        decile_bounds (np.ndarray): X-values of the RAMP curve.
        curve (np.ndarray): Y-values of the RAMP curve.
        current_model_val (float): The model value to find the correction for.

    Returns:
        float: The interpolated or extrapolated correction value, or np.nan if fails.
    """
    # --- MODIFIED: More robust check for valid data ---
    valid_mask = ~np.isnan(decile_bounds) & ~np.isnan(curve)
    if np.sum(valid_mask) < 2:
        # Fallback if not enough points to interpolate. Return NaN to be handled by the caller.
        return np.nan

    # Use numpy's interpolation with linear extrapolation
    return np.interp(current_model_val, decile_bounds[valid_mask], curve[valid_mask])


# --- Main RAMP Correction Function ---

def get_overhang_ramp(collocated_df, model_grid_df, results_dir=None):
    """
    Computes spatiotemporal correction values (lambda1, lambda2) using a local RAMP
    scheme with a global fallback.

    Args:
        collocated_df (pd.DataFrame): DataFrame with collocated model and observation data.
                                      Must contain ['lon_toar', 'lat_toar', 'observed_ozone',
                                      'model_ozone', 'month'].
        model_grid_df (pd.DataFrame): DataFrame representing the model grid, containing
                                     ['lon', 'lat'] and columns for monthly model values
                                     (e.g., 'DMA8_1', 'DMA8_2', ...).
        results_dir (str, optional): Directory to save diagnostic plots. If None,
                                     plotting is disabled.

    Returns:
        tuple: A tuple of two DataFrames:
            - lambda1_df (pd.DataFrame): Mean corrections for each grid point and month.
            - lambda2_df (pd.DataFrame): Variance corrections for each grid point and month.
    """
    print("Starting RAMP correction process...")

    # Setup
    plotting_enabled = results_dir is not None
    if plotting_enabled and not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Created results directory: {results_dir}")

    num_grid_points = len(model_grid_df)
    num_months = 12
    dma_cols = [f'DMA8_{i+1}' for i in range(num_months)]

    # Initialize output DataFrames
    lambda1_df = pd.DataFrame(np.nan, index=model_grid_df.index, columns=dma_cols)
    lambda2_df = pd.DataFrame(np.nan, index=model_grid_df.index, columns=dma_cols)
    technique_df = pd.DataFrame(0, index=model_grid_df.index, columns=dma_cols)
    
    # --- Constants from MATLAB script ---
    TOTAL_BINS = 10 # Number of bins for decile curves
    INITIAL_NEIGHBORS = 100 # Initial number of neighbors for local RAMP
    NEIGHBOR_INCREMENT = 250 # Increment for additional neighbors in local RAMP
    MAX_ADJUSTMENTS = 5  # Maximum number of adjustments to local RAMP that limits the neighborhood expansions
    GLOBAL_FALLBACK_CODE = 99 # Code for global fallback technique
    MIN_POINTS_FOR_CURVE = 20 # --- NEW: Minimum points to build a curve

    # Build a KD-Tree for efficient nearest-neighbor searches on observation locations
    # --- MODIFIED: Ensure we only use unique locations for the KDTree ---
    unique_obs_locs = collocated_df[['lon_toar', 'lat_toar']].drop_duplicates().values
    obs_kdtree = cKDTree(unique_obs_locs)
    
    # --- Main Time Loop (Iterating through each month) ---
    for t in range(num_months):
        month = t + 1
        dma_col = f'DMA8_{t+1}'
        print(f"\nProcessing Month: {month} ({dma_col})")
        
        time_window = get_time_window(t)
        time_window_months = [m + 1 for m in time_window]

        # --- 1. Global RAMP Calculation ---
        global_data = collocated_df[collocated_df['month'].isin(time_window_months)].dropna(subset=['model_ozone', 'observed_ozone'])
        global_model_vals = global_data['model_ozone'].values
        global_obs_vals = global_data['observed_ozone'].values
        
        # --- NEW: Calculate global average variance as a robust fallback ---
        global_avg_variance = np.var(global_obs_vals) if len(global_obs_vals) > 1 else 0.01

        (global_bounds, global_curve1, global_curve2, global_bin_edges) = compute_decile_curves(
            global_model_vals, global_obs_vals, TOTAL_BINS, MIN_POINTS_FOR_CURVE
        )
        global_curve1 = enforce_monotonicity(global_curve1)
        
        if len(global_bounds) > 0:
            sort_idx = np.argsort(global_bounds)
            global_bounds, global_curve1, global_curve2 = global_bounds[sort_idx], global_curve1[sort_idx], global_curve2[sort_idx]

        if plotting_enabled:
            plot_global_ramp(global_bounds, global_curve1, global_curve2, global_model_vals, global_obs_vals, month, results_dir)

        plot_indices = np.random.choice(num_grid_points, min(5, num_grid_points), replace=False)

        # --- 2. Grid Point Loop ---
        for p in range(num_grid_points):
            grid_loc = model_grid_df.loc[p, ['lon', 'lat']].values
            current_model_val = model_grid_df.loc[p, dma_col]

            if np.isnan(current_model_val):
                continue
            
            l1, l2 = np.nan, np.nan
            technique_code = GLOBAL_FALLBACK_CODE # Assume global fallback initially
            plot_info = {} # Store data for plotting

            # --- 3. Local RAMP Calculation ---
            # Start with an initial number of neighbors
            _, nearest_indices = obs_kdtree.query(grid_loc, k=min(INITIAL_NEIGHBORS, len(unique_obs_locs)))
            # Get the full station data for those unique locations
            local_stations = collocated_df[collocated_df.set_index(['lon_toar', 'lat_toar']).index.isin([tuple(l) for l in unique_obs_locs[nearest_indices]])]

            local_data_window = local_stations[local_stations['month'].isin(time_window_months)].dropna(subset=['model_ozone', 'observed_ozone'])
            
            (local_bounds, local_curve1, local_curve2, _) = compute_decile_curves(
                local_data_window['model_ozone'].values, local_data_window['observed_ozone'].values, TOTAL_BINS, MIN_POINTS_FOR_CURVE
            )
            local_curve1 = enforce_monotonicity(local_curve1)

            l1 = interpolate_ramp(local_bounds, local_curve1, current_model_val)
            l2 = interpolate_ramp(local_bounds, local_curve2, current_model_val)

            if not np.isnan(l1):
                technique_code = 1 # Local success
                plot_info = {'mvals': local_data_window['model_ozone'].values, 'ovals': local_data_window['observed_ozone'].values, 'b':local_bounds, 'c1':local_curve1, 'c2':local_curve2, 'title':"Initial Local"}
            else:
                # --- 4. Iteratively Adjust Local RAMP ---
                num_neighbors = INITIAL_NEIGHBORS
                for i in range(MAX_ADJUSTMENTS):
                    num_neighbors += NEIGHBOR_INCREMENT
                    if num_neighbors > len(unique_obs_locs): break

                    _, nearest_indices = obs_kdtree.query(grid_loc, k=num_neighbors)
                    adj_stations = collocated_df[collocated_df.set_index(['lon_toar', 'lat_toar']).index.isin([tuple(l) for l in unique_obs_locs[nearest_indices]])]
                    adj_data_window = adj_stations[adj_stations['month'].isin(time_window_months)].dropna(subset=['model_ozone', 'observed_ozone'])
                    
                    (adj_bounds, adj_curve1, adj_curve2, _) = compute_decile_curves(
                        adj_data_window['model_ozone'].values, adj_data_window['observed_ozone'].values, TOTAL_BINS, MIN_POINTS_FOR_CURVE
                    )
                    adj_curve1 = enforce_monotonicity(adj_curve1)
                    
                    l1 = interpolate_ramp(adj_bounds, adj_curve1, current_model_val)
                    l2 = interpolate_ramp(adj_bounds, adj_curve2, current_model_val)

                    if not np.isnan(l1):
                        technique_code = i + 2 # Adjusted success
                        plot_info = {'mvals': adj_data_window['model_ozone'].values, 'ovals': adj_data_window['observed_ozone'].values, 'b':adj_bounds, 'c1':adj_curve1, 'c2':adj_curve2, 'title':f"Adjusted-{i+1}"}
                        break
            
            # --- 5. Fallback to Global RAMP if local/adjusted failed ---
            if np.isnan(l1):
                l1 = interpolate_ramp(global_bounds, global_curve1, current_model_val)
                l2 = interpolate_ramp(global_bounds, global_curve2, current_model_val)
                plot_info = {'mvals': global_model_vals, 'ovals': global_obs_vals, 'b':global_bounds, 'c1':global_curve1, 'c2':global_curve2, 'title':"Fallback to Global"}

            # --- 6. Final Assignment with Sanity Checks ---
            # If l1 is still NaN, use the model value (no correction)
            final_l1 = l1 if not np.isnan(l1) else current_model_val
            # If l2 is still NaN or negative, use the global average variance
            final_l2 = l2 if not np.isnan(l2) and l2 >= 0 else global_avg_variance

            lambda1_df.loc[p, dma_col] = final_l1
            lambda2_df.loc[p, dma_col] = final_l2
            technique_df.loc[p, dma_col] = technique_code

            if plotting_enabled and p in plot_indices and plot_info:
                 plot_local_data(plot_info['mvals'], plot_info['ovals'], plot_info['b'], plot_info['c1'], plot_info['c2'], current_model_val, final_l1, final_l2, p, month, results_dir, plot_info['title'])

        if plotting_enabled:
            plot_technique_map(model_grid_df[['lon', 'lat']], technique_df[dma_col], month, results_dir)
            
    print("\nRAMP correction process finished.")
    return lambda1_df, lambda2_df, technique_df


# --- Plotting Functions ---

def plot_global_ramp(bounds, curve1, curve2, model_vals, obs_vals, month, results_dir): # --- MODIFIED ---
    """Plots the global RAMP curves for a given month."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), constrained_layout=True)
    fig.suptitle(f'Global RAMP Curves - Month {month}', fontsize=16)

    # Lambda1 Plot
    ax1.scatter(model_vals, obs_vals, alpha=0.1, label='Model-Obs Pairs', s=10)
    # --- MODIFIED: Filter NaNs before plotting ---
    valid_c1_mask = ~np.isnan(bounds) & ~np.isnan(curve1)
    if np.sum(valid_c1_mask) > 1:
        ax1.plot(bounds[valid_c1_mask], curve1[valid_c1_mask], '-o', color='red', linewidth=2, label='λ$_1$ Curve (Mean)')
    
    min_val = min(np.nanmin(model_vals), np.nanmin(obs_vals)) if len(model_vals)>0 and len(obs_vals)>0 else 0
    max_val = max(np.nanmax(model_vals), np.nanmax(obs_vals)) if len(model_vals)>0 and len(obs_vals)>0 else 1
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', label='1:1 Line')
    ax1.set_xlabel('Model Ozone')
    ax1.set_ylabel('Observed Ozone')
    ax1.set_title('λ$_1$ Mean Correction')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Lambda2 Plot
    # --- MODIFIED: Filter NaNs before plotting ---
    valid_c2_mask = ~np.isnan(bounds) & ~np.isnan(curve2)
    if np.sum(valid_c2_mask) > 1:
        ax2.plot(bounds[valid_c2_mask], curve2[valid_c2_mask], '-o', color='blue', linewidth=2, label='λ$_2$ Curve (Variance)')
    ax2.set_xlabel('Model Ozone')
    ax2.set_ylabel('Observed Ozone Variance')
    ax2.set_title('λ$_2$ Variance Correction')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    filename = os.path.join(results_dir, f'global_ramp_month_{month}.png')
    plt.savefig(filename)
    plt.close(fig)

def plot_local_data(model_vals, obs_vals, bounds, curve1, curve2, current_mod, l1, l2, point_idx, month, results_dir, title_suffix): # --- MODIFIED ---
    """Plots the local RAMP data for a specific grid point."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), constrained_layout=True)
    fig.suptitle(f'Local Correction - Grid Point {point_idx}, Month {month}\nTechnique: {title_suffix}', fontsize=14)

    # Lambda1 Plot
    ax1.scatter(model_vals, obs_vals, alpha=0.3, label='Local Pairs', s=15)
    # --- MODIFIED: Filter NaNs before plotting ---
    valid_c1_mask = ~np.isnan(bounds) & ~np.isnan(curve1)
    if np.sum(valid_c1_mask) > 1:
        ax1.plot(bounds[valid_c1_mask], curve1[valid_c1_mask], '-ok', linewidth=2, label='Local λ$_1$ Curve')
    
    ax1.plot(current_mod, l1, 'r*', markersize=15, label=f'Corrected Point ({current_mod:.1f} -> {l1:.1f})')
    min_val = min(np.nanmin(model_vals), np.nanmin(obs_vals)) if len(model_vals)>0 else 0
    max_val = max(np.nanmax(model_vals), np.nanmax(obs_vals)) if len(model_vals)>0 else 1
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', label='1:1 Line')
    ax1.set_xlabel('Model Ozone')
    ax1.set_ylabel('Observed Ozone')
    ax1.set_title('λ$_1$ Mean Correction')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Lambda2 Plot
    # --- MODIFIED: Filter NaNs before plotting ---
    valid_c2_mask = ~np.isnan(bounds) & ~np.isnan(curve2)
    if np.sum(valid_c2_mask) > 1:
        ax2.plot(bounds[valid_c2_mask], curve2[valid_c2_mask], '-ob', linewidth=2, label='Local λ$_2$ Curve')
    
    ax2.plot(current_mod, l2, 'r*', markersize=15, label=f'Variance: {l2:.2f}')
    ax2.set_xlabel('Model Ozone')
    ax2.set_ylabel('Observed Ozone Variance')
    ax2.set_title('λ$_2$ Variance Correction')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    filename = os.path.join(results_dir, f'local_correction_pt{point_idx}_month{month}.png')
    plt.savefig(filename)
    plt.close(fig)

def plot_technique_map(coords, technique_vals, month, results_dir):
    """Plots a spatial map showing which correction technique was used."""
    plt.figure(figsize=(12, 8))
    
    
    
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    
    # Create discrete colormap
    import matplotlib.colors as mcolors
    
    # Define discrete boundaries and colors
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 98.5, 99.5]
    colors = ['gray', 'green', 'blue', 'purple', 'orange', 'red', 'yellow', 'black']
    cmap = mcolors.ListedColormap(colors)
    norm = BoundaryNorm(bounds, cmap.N)
    
    # Create scatter plot with discrete colors
    scatter = ax.scatter(coords['lon'], coords['lat'], 
                        c=technique_vals, 
                        cmap=cmap,
                        norm=norm,
                        s=15)
    
    # Add colorbar with discrete labels
    cbar = plt.colorbar(scatter, 
                       label='Correction Technique',
                       ticks=[0, 1, 2, 3, 4, 5, 99])
    cbar.ax.set_yticklabels(['None', 'Local', 'Adj-1', 'Adj-2', 'Adj-3', 'Adj-4', 'Global'])
    
    plt.title(f'Technique Map - Month {month}')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    ax.gridlines(linestyle='--', alpha=0.5)
    
    filename = os.path.join(results_dir, f'technique_map_month_{month}.png')
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()


# --- Example Usage ---
if __name__ == '__main__':
    # (Example usage remains the same)
    print("--- Running Example ---")
    
    # 1. Create Sample Data
    lons = np.linspace(-100, -80, 10)
    lats = np.linspace(30, 45, 10)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    model_coords = pd.DataFrame({'lon': lon_grid.flatten(), 'lat': lat_grid.flatten()})
    
    model_grid_df = model_coords.copy()
    for i in range(12):
        base_ozone = 40 + 15 * np.sin(np.pi * i / 12)
        model_grid_df[f'DMA8_{i+1}'] = base_ozone + np.random.randn(100) * 5

    num_obs_stations = 200
    obs_data_list = []
    for month in range(1, 13):
        month_df = pd.DataFrame({
            'lon_toar': np.random.uniform(-105, -75, num_obs_stations),
            'lat_toar': np.random.uniform(28, 47, num_obs_stations),
            'observed_ozone': 45 + 15 * np.sin(np.pi * (month-1) / 12) + np.random.randn(num_obs_stations) * 7,
            'month': month
        })
        obs_data_list.append(month_df)
    
    collocated_df_full = pd.concat(obs_data_list, ignore_index=True)

    model_ozone_col = []
    for month in range(1, 13):
        month_data = collocated_df_full[collocated_df_full['month'] == month]
        model_month_data = model_grid_df[['lon', 'lat', f'DMA8_{month}']]
        
        interpolated_vals = griddata(
            points=model_month_data[['lon', 'lat']].values,
            values=model_month_data[f'DMA8_{month}'].values,
            xi=month_data[['lon_toar', 'lat_toar']].values,
            method='linear'
        )
        model_ozone_col.extend(interpolated_vals)
        
    collocated_df_full['model_ozone'] = model_ozone_col
    collocated_df_full.dropna(inplace=True)

    print("\nSample DataFrames created:")
    print(model_grid_df.head())
    print(collocated_df_full.head())
    
    # 2. Run the RAMP Correction
    lambda1_df, lambda2_df, technique_df = get_overhang_ramp(
        collocated_df=collocated_df_full,
        model_grid_df=model_grid_df,
        results_dir='ramp_plots_robust'
    )

    # 3. Display Results
    print("\n--- Results ---")
    print("\nLambda1 (Mean Correction) DataFrame Head:")
    print(lambda1_df.head())
    print("\nLambda2 (Variance Correction) DataFrame Head:")
    print(lambda2_df.head())
    print("\nTechnique Used DataFrame Head:")
    print(technique_df.head())
    print("\nCheck 'ramp_plots_robust' directory for diagnostic images.")
