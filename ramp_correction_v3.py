# This is close to v1. The main changes are:
# - changed the way extrapolation is done
# - sorting of local curve points
# - monotonicity enforcement logic

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
    # UNCHANGED as per user request
    if month_index == 0:
        return [num_months - 1, 0, 1]
    elif month_index == num_months - 1:
        return [num_months - 2, num_months - 1, 0]
    else:
        return [month_index - 1, month_index, month_index + 1]

def compute_decile_curves(model_vals, obs_vals, total_bins=10):
    """
    Computes decile curves for mean (lambda1) and variance (lambda2).
    
    Args:
        model_vals (np.ndarray): Array of model values.
        obs_vals (np.ndarray): Array of corresponding observation values.
        total_bins (int): The number of bins to use (e.g., 10 for deciles).

    Returns:
        tuple: A tuple containing:
            - decile_bounds (np.ndarray): Mean model value in each bin.
            - curve1 (np.ndarray): Mean observation value in each bin (lambda1).
            - curve2 (np.ndarray): Variance of observation values in each bin (lambda2).
            - bin_edges (np.ndarray): The edges of the bins.
    """
    if len(model_vals) < total_bins:
        # Not enough data to form deciles, return empty
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

        if np.any(mask):
            decile_bounds[i] = np.mean(model_vals[mask])
            curve1[i] = np.mean(obs_vals[mask])
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
    
    # Use the mean of the valid part of the curve as the central pivot
    mean_val = np.nanmean(monotonic_lambda)
    if np.isnan(mean_val):
        # Fallback if all values are NaN
        first_valid_val = monotonic_lambda[~np.isnan(monotonic_lambda)]
        if len(first_valid_val) > 0:
            mean_val = first_valid_val[0]
        else:
            return monotonic_lambda # Cannot proceed if all are NaN

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
        
        # *** CHANGED: Added specific check for the bin right of the middle ***
        # [cite_start]This mirrors the logic from the MATLAB script [cite: 171]
        if i == middle_bin + 1:
            if monotonic_lambda[i] < mean_val:
                monotonic_lambda[i] = mean_val
        else:
            # Find previous non-NaN value to the left
            prev_val = np.nan
            for k in range(i - 1, -1, -1):
                if not np.isnan(monotonic_lambda[k]):
                    prev_val = monotonic_lambda[k]
                    break
            if not np.isnan(prev_val) and monotonic_lambda[i] < prev_val:
                monotonic_lambda[i] = prev_val
            
    return monotonic_lambda

def interpolate_ramp(decile_bounds, curve, current_model_val):
    """
    Interpolates or extrapolates a value from a RAMP curve using linear
    interpolation and extrapolation to match MATLAB's 'extrap' functionality.

    Args:
        decile_bounds (np.ndarray): X-values of the RAMP curve.
        curve (np.ndarray): Y-values of the RAMP curve.
        current_model_val (float): The model value to find the correction for.

    Returns:
        float: The interpolated or extrapolated correction value.
    """
    # *** CHANGED: This function is entirely rewritten to perform linear extrapolation ***
    # [cite_start]This now mimics the behavior of MATLAB's interp1(..., 'extrap') [cite: 70]
    
    valid_mask = ~np.isnan(decile_bounds) & ~np.isnan(curve)
    if np.sum(valid_mask) < 2:
        return np.nanmean(curve) # Fallback if not enough points to interpolate

    x = decile_bounds[valid_mask]
    y = curve[valid_mask]

    # np.interp requires sorted x-points. Although we sort later, this ensures safety.
    sort_idx = np.argsort(x)
    x, y = x[sort_idx], y[sort_idx]

    if current_model_val >= x[0] and current_model_val <= x[-1]:
        # Value is within bounds, standard interpolation
        return np.interp(current_model_val, x, y)
    elif current_model_val < x[0]:
        # Value is below bounds, extrapolate from the left
        slope = (y[1] - y[0]) / (x[1] - x[0])
        return y[0] + slope * (current_model_val - x[0])
    else: # current_model_val > x[-1]
        # Value is above bounds, extrapolate from the right
        slope = (y[-1] - y[-2]) / (x[-1] - x[-2])
        return y[-1] + slope * (current_model_val - x[-1])


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
    TOTAL_BINS = 10
    INITIAL_NEIGHBORS = 100
    NEIGHBOR_INCREMENT = 250
    MAX_ADJUSTMENTS = 5 # Limit the number of neighborhood expansions
    GLOBAL_FALLBACK_CODE = 99

    # Build a KD-Tree for efficient nearest-neighbor searches on observation locations
    obs_coords = collocated_df[['lon_toar', 'lat_toar']].values
    obs_kdtree = cKDTree(obs_coords)
    
    # --- Main Time Loop (Iterating through each month) ---
    for t in range(num_months):
        month = t + 1
        dma_col = f'DMA8_{t+1}'
        print(f"\nProcessing Month: {month} ({dma_col})")
        
        time_window = get_time_window(t)
        # Convert month indices (0-11) to month numbers (1-12)
        time_window_months = [m + 1 for m in time_window]

        # --- 1. Global RAMP Calculation ---
        global_data = collocated_df[collocated_df['month'].isin(time_window_months)]
        global_model_vals = global_data['model_ozone'].values
        global_obs_vals = global_data['observed_ozone'].values
        
        (global_bounds, global_curve1, global_curve2, global_bin_edges) = compute_decile_curves(
            global_model_vals, global_obs_vals, TOTAL_BINS
        )
        global_curve1 = enforce_monotonicity(global_curve1)
        
        # Sort curves by bounds for interpolation
        if len(global_bounds) > 0:
            sort_idx = np.argsort(global_bounds)
            global_bounds, global_curve1, global_curve2 = global_bounds[sort_idx], global_curve1[sort_idx], global_curve2[sort_idx]

        # Plot global ramp if enabled
        if plotting_enabled:
            plot_global_ramp(
                global_bounds, global_curve1, global_curve2, 
                global_model_vals, global_obs_vals, global_bin_edges, 
                month, results_dir
            )

        # Choose a few random points to plot for local diagnostics
        plot_indices = np.random.choice(num_grid_points, min(5, num_grid_points), replace=False)

        if plotting_enabled:
            # Plot the indices of points that will be used for local diagnostics
            plot_plotting_indices(plot_indices, model_grid_df, results_dir)

        # --- 2. Grid Point Loop ---
        for p in range(num_grid_points):
            grid_loc = model_grid_df.loc[p, ['lon', 'lat']].values
            current_model_val = model_grid_df.loc[p, dma_col]

            if np.isnan(current_model_val):
                continue
            
            # Check if model value is outside the global RAMP range
            valid_global_bounds = global_bounds[~np.isnan(global_bounds)]
            is_out_of_bounds = False
            if len(valid_global_bounds) > 1:
                if current_model_val < valid_global_bounds.min() or current_model_val > valid_global_bounds.max():
                    is_out_of_bounds = True
            
            if is_out_of_bounds:
                l1 = interpolate_ramp(global_bounds, global_curve1, current_model_val)
                l2 = interpolate_ramp(global_bounds, global_curve2, current_model_val)
                lambda1_df.loc[p, dma_col] = l1
                lambda2_df.loc[p, dma_col] = max(0.01, l2) if not np.isnan(l2) else 0.01
                technique_df.loc[p, dma_col] = GLOBAL_FALLBACK_CODE
                continue

            # --- 3. Local RAMP Calculation ---
            # Start with an initial number of neighbors
            _, nearest_indices = obs_kdtree.query(grid_loc, k=INITIAL_NEIGHBORS)
            local_data = collocated_df.iloc[nearest_indices]
            
            # Filter for the time window
            local_data_window = local_data[local_data['month'].isin(time_window_months)]
            local_model_vals = local_data_window['model_ozone'].values
            local_obs_vals = local_data_window['observed_ozone'].values
            
            (local_bounds, local_curve1, local_curve2, local_bin_edges) = compute_decile_curves(
                local_model_vals, local_obs_vals, TOTAL_BINS
            )
            
            # *** CHANGED: Added sorting for local curves before they are used ***
            # [cite_start]This is critical for correct interpolation [cite: 95]
            if len(local_bounds[~np.isnan(local_bounds)]) > 0:
                sort_idx = np.argsort(local_bounds)
                local_bounds, local_curve1, local_curve2 = local_bounds[sort_idx], local_curve1[sort_idx], local_curve2[sort_idx]

            local_curve1 = enforce_monotonicity(local_curve1)

            valid_local_bounds = local_bounds[~np.isnan(local_bounds)]
            
            # Check if model value is within the local RAMP range
            can_interpolate_locally = False
            if len(valid_local_bounds) > 1:
                if current_model_val >= valid_local_bounds.min() and current_model_val <= valid_local_bounds.max():
                    can_interpolate_locally = True

            if can_interpolate_locally:
                l1 = interpolate_ramp(local_bounds, local_curve1, current_model_val)
                l2 = interpolate_ramp(local_bounds, local_curve2, current_model_val)
                lambda1_df.loc[p, dma_col] = l1
                lambda2_df.loc[p, dma_col] = max(0.01, l2) if not np.isnan(l2) else 0.01
                technique_df.loc[p, dma_col] = 1 # 1 = Local success
                
                if plotting_enabled and p in plot_indices:
                     plot_local_data(
                        local_model_vals, local_obs_vals, local_bounds, local_curve1, local_curve2, 
                        current_model_val, l1, l2, p, month, results_dir, "Initial Local"
                    )
            else:
                # --- 4. Iteratively Adjust Local RAMP ---
                found_adjusted_fit = False
                num_neighbors = INITIAL_NEIGHBORS
                
                for i in range(MAX_ADJUSTMENTS):
                    num_neighbors += NEIGHBOR_INCREMENT
                    if num_neighbors > len(collocated_df):
                        break
                    
                    _, nearest_indices = obs_kdtree.query(grid_loc, k=num_neighbors)
                    adj_data = collocated_df.iloc[nearest_indices]
                    adj_data_window = adj_data[adj_data['month'].isin(time_window_months)]
                    adj_model_vals = adj_data_window['model_ozone'].values
                    adj_obs_vals = adj_data_window['observed_ozone'].values

                    (adj_bounds, adj_curve1, adj_curve2, adj_bin_edges) = compute_decile_curves(
                        adj_model_vals, adj_obs_vals, TOTAL_BINS
                    )

                    # *** CHANGED: Added sorting for adjusted local curves ***
                    if len(adj_bounds[~np.isnan(adj_bounds)]) > 0:
                        sort_idx = np.argsort(adj_bounds)
                        adj_bounds, adj_curve1, adj_curve2 = adj_bounds[sort_idx], adj_curve1[sort_idx], adj_curve2[sort_idx]

                    adj_curve1 = enforce_monotonicity(adj_curve1)
                    
                    valid_adj_bounds = adj_bounds[~np.isnan(adj_bounds)]
                    if len(valid_adj_bounds) > 1:
                        if current_model_val >= valid_adj_bounds.min() and current_model_val <= valid_adj_bounds.max():
                            l1 = interpolate_ramp(adj_bounds, adj_curve1, current_model_val)
                            l2 = interpolate_ramp(adj_bounds, adj_curve2, current_model_val)
                            lambda1_df.loc[p, dma_col] = l1
                            lambda2_df.loc[p, dma_col] = max(0.01, l2) if not np.isnan(l2) else 0.01
                            technique_df.loc[p, dma_col] = i + 2 # 2, 3, ... = Adjusted success
                            found_adjusted_fit = True
                            
                            if plotting_enabled and p in plot_indices:
                                plot_local_data(
                                    adj_model_vals, adj_obs_vals, adj_bounds, adj_curve1, adj_curve2, 
                                    current_model_val, l1, l2, p, month, results_dir, f"Adjusted-{i+1} ({num_neighbors} pts)"
                                )
                            break # Exit adjustment loop
                
                # --- 5. Fallback to Global RAMP ---
                if not found_adjusted_fit:
                    l1 = interpolate_ramp(global_bounds, global_curve1, current_model_val)
                    l2 = interpolate_ramp(global_bounds, global_curve2, current_model_val)
                    lambda1_df.loc[p, dma_col] = l1
                    lambda2_df.loc[p, dma_col] = max(0.01, l2) if not np.isnan(l2) else 0.01
                    technique_df.loc[p, dma_col] = GLOBAL_FALLBACK_CODE
                    
                    if plotting_enabled and p in plot_indices:
                        # Plot fallback attempt for diagnostics
                        plot_local_data(
                            local_model_vals, local_obs_vals, local_bounds, local_curve1, local_curve2, 
                            current_model_val, l1, l2, p, month, results_dir, "Fallback to Global"
                        )
                        
        # Plot technique map for the month
        if plotting_enabled:
            plot_technique_map(model_grid_df[['lon', 'lat']], technique_df[dma_col], month, results_dir)
            
    print("\nRAMP correction process finished.")
    return lambda1_df, lambda2_df, technique_df


# --- Plotting Functions ---

def plot_global_ramp(bounds, curve1, curve2, model_vals, obs_vals, bin_edges, month, results_dir):
    """Plots the global RAMP curves for a given month."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), constrained_layout=True)
    fig.suptitle(f'Global RAMP Curves - Month {month}', fontsize=16)

    # Lambda1 Plot
    ax1.scatter(model_vals, obs_vals, alpha=0.1, label='Model-Obs Pairs', s=10)
    ax1.plot(bounds, curve1, '-o', color='red', linewidth=2, label='λ$_1$ Curve (Mean)')
    min_val = min(np.nanmin(model_vals), np.nanmin(obs_vals))
    max_val = max(np.nanmax(model_vals), np.nanmax(obs_vals))
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', label='1:1 Line')
    ax1.set_xlabel('Model Ozone')
    ax1.set_ylabel('Observed Ozone')
    ax1.set_title('λ$_1$ Mean Correction')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Lambda2 Plot
    ax2.plot(bounds, curve2, '-o', color='blue', linewidth=2, label='λ$_2$ Curve (Variance)')
    ax2.set_xlabel('Model Ozone')
    ax2.set_ylabel('Observed Ozone Variance')
    ax2.set_title('λ$_2$ Variance Correction')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    filename = os.path.join(results_dir, f'global_ramp_month_{month}.png')
    plt.savefig(filename)
    plt.close(fig)

def plot_local_data(model_vals, obs_vals, bounds, curve1, curve2, current_mod, l1, l2, point_idx, month, results_dir, title_suffix):
    """Plots the local RAMP data for a specific grid point."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), constrained_layout=True)
    fig.suptitle(f'Local Correction - Grid Point {point_idx}, Month {month}\nTechnique: {title_suffix}', fontsize=14)

    # Lambda1 Plot
    ax1.scatter(model_vals, obs_vals, alpha=0.3, label='Local Pairs', s=15)
    ax1.plot(bounds, curve1, '-ok', linewidth=2, label='Local λ$_1$ Curve')
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
    ax2.plot(bounds, curve2, '-ob', linewidth=2, label='Local λ$_2$ Curve')
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

def plot_plotting_indices(plot_indices, model_grid_df, results_dir):
    """
    Plots the indices of points that were used for local diagnostics.
    
    Args:
        plot_indices (list): List of indices to plot.
        model_grid_df (pd.DataFrame): DataFrame containing model grid coordinates.
        results_dir (str): Directory to save the plot.
    """
    plt.figure(figsize=(12, 8))
    
    # Create map with Plate Carree projection
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    
    # Plot all grid points
    ax.scatter(model_grid_df['lon'], model_grid_df['lat'], 
              c='lightgray', s=10, label='All Points',
              transform=ccrs.PlateCarree())
    
    # Plot diagnostic points
    ax.scatter(model_grid_df.loc[plot_indices, 'lon'], 
              model_grid_df.loc[plot_indices, 'lat'],
              c='red', s=50, label='Diagnostic Points',
              transform=ccrs.PlateCarree())
    
    # Label diagnostic points
    for idx in plot_indices:
        lon = model_grid_df.loc[idx, 'lon']
        lat = model_grid_df.loc[idx, 'lat']
        ax.text(lon, lat, str(idx), 
                fontsize=8, ha='right', va='bottom', 
                color='black', transform=ccrs.PlateCarree())
    
    plt.title('Diagnostic Points for Local RAMP Corrections')
    ax.gridlines(linestyle='--', alpha=0.5)
    plt.legend()
    
    filename = os.path.join(results_dir, 'diagnostic_points.png')
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()

# --- Example Usage ---

if __name__ == '__main__':
    
    print("--- Running Example ---")
    
    # 1. Create Sample Data (mimicking your data structures)
    
    # Sample model grid (e.g., a simple 10x10 grid)
    lons = np.linspace(-100, -80, 10)
    lats = np.linspace(30, 45, 10)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    model_coords = pd.DataFrame({'lon': lon_grid.flatten(), 'lat': lat_grid.flatten()})
    
    # Generate random model ozone data for 12 months
    model_grid_df = model_coords.copy()
    for i in range(12):
        # Ozone values increasing in summer months
        base_ozone = 40 + 15 * np.sin(np.pi * i / 12)
        model_grid_df[f'DMA8_{i+1}'] = base_ozone + np.random.randn(100) * 5

    # Sample collocated data (e.g., 200 observation stations)
    num_obs_stations = 200
    obs_data_list = []
    for month in range(1, 13):
        month_df = pd.DataFrame({
            'lon_toar': np.random.uniform(-105, -75, num_obs_stations),
            'lat_toar': np.random.uniform(28, 47, num_obs_stations),
            'observed_ozone': 45 + 15 * np.sin(np.pi * (month-1) / 12) + np.random.randn(num_obs_stations) * 7, # Obs has more variance
            'month': month
        })
        obs_data_list.append(month_df)
    
    collocated_df_full = pd.concat(obs_data_list, ignore_index=True)

    # Use griddata to get the 'model_ozone' column for the collocated data
    # This simulates your existing `collocate_data` function output
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
    collocated_df_full.dropna(inplace=True) # Remove points outside model convex hull

    print("\nSample DataFrames created:")
    print("Model Grid DF Head:")
    print(model_grid_df.head())
    print("\nCollocated DF Head:")
    print(collocated_df_full.head())
    
    # 2. Run the RAMP Correction
    # This will create a 'ramp_plots' directory in your current folder
    lambda1_df, lambda2_df, technique_df = get_overhang_ramp(
        collocated_df=collocated_df_full,
        model_grid_df=model_grid_df,
        results_dir='ramp_plots_corrected' 
    )

    # 3. Display Results
    print("\n--- Results ---")
    print("\nLambda1 (Mean Correction) DataFrame Head:")
    print(lambda1_df.head())
    
    print("\nLambda2 (Variance Correction) DataFrame Head:")
    print(lambda2_df.head())

    print("\nTechnique Used DataFrame Head:")
    print(technique_df.head())
    
    print("\nCheck 'ramp_plots_corrected' directory for diagnostic images.")