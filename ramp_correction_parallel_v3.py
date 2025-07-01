import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import os
import warnings
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from matplotlib.colors import BoundaryNorm
from multiprocessing import Pool, cpu_count
from functools import partial

# Suppress RankWarning for polynomial fitting in plots
warnings.filterwarnings("ignore", category=np.RankWarning)

# --- Core Helper Functions ---

def get_time_window(month_index, num_months=12):
    """Defines a 3-month window, wrapping around the year."""
    if month_index == 0:
        return [num_months - 1, 0, 1]
    elif month_index == num_months - 1:
        return [num_months - 2, num_months - 1, 0]
    else:
        return [month_index - 1, month_index, month_index + 1]

def compute_decile_curves(model_vals, obs_vals, total_bins=10):
    """Computes decile curves for mean (lambda1) and variance (lambda2)."""
    if len(model_vals) < total_bins:
        return np.array([]), np.array([]), np.array([]), np.array([])
    try:
        bin_edges = np.percentile(model_vals, np.linspace(0, 100, total_bins + 1))
        bin_edges[-1] = np.maximum(bin_edges[-1], model_vals.max())
        bin_edges[0] = np.minimum(bin_edges[0], model_vals.min())
    except IndexError:
        return np.array([]), np.array([]), np.array([]), np.array([])

    decile_bounds = np.full(total_bins, np.nan)
    curve1 = np.full(total_bins, np.nan)
    curve2 = np.full(total_bins, np.nan)

    for i in range(total_bins):
        mask = (model_vals >= bin_edges[i]) & (model_vals <= bin_edges[i+1]) if i == total_bins - 1 else (model_vals >= bin_edges[i]) & (model_vals < bin_edges[i+1])
        if np.any(mask):
            decile_bounds[i] = np.mean(model_vals[mask])
            curve1[i] = np.mean(obs_vals[mask])
            curve2[i] = np.var(obs_vals[mask]) if np.sum(mask) > 1 else 0.0
    return decile_bounds, curve1, curve2, bin_edges

# --- UPDATED: Helper functions replaced with logic from v3 ---

def enforce_monotonicity(lambda_curve):
    """
    Ensures the lambda1 curve is monotonically non-decreasing.
    Logic taken directly from ramp_correction_v3.py.
    """
    if lambda_curve is None or len(lambda_curve[~np.isnan(lambda_curve)]) == 0:
        return lambda_curve
    monotonic_lambda = lambda_curve.copy()
    total_bins = len(monotonic_lambda)
    middle_bin = total_bins // 2
    mean_val = np.nanmean(monotonic_lambda)
    if np.isnan(mean_val):
        return monotonic_lambda
    for i in range(middle_bin, -1, -1):
        if np.isnan(monotonic_lambda[i]): continue
        if i == middle_bin:
            if monotonic_lambda[i] > mean_val: monotonic_lambda[i] = mean_val
        else:
            next_val = next((monotonic_lambda[k] for k in range(i + 1, total_bins) if not np.isnan(monotonic_lambda[k])), np.nan)
            if not np.isnan(next_val) and monotonic_lambda[i] > next_val: monotonic_lambda[i] = next_val
    for i in range(middle_bin + 1, total_bins):
        if np.isnan(monotonic_lambda[i]): continue
        # This mirrors the logic from the MATLAB script [cite: 171]
        if i == middle_bin + 1:
            if monotonic_lambda[i] < mean_val:
                monotonic_lambda[i] = mean_val
        else:
            prev_val = next((monotonic_lambda[k] for k in range(i - 1, -1, -1) if not np.isnan(monotonic_lambda[k])), np.nan)
            if not np.isnan(prev_val) and monotonic_lambda[i] < prev_val:
                monotonic_lambda[i] = prev_val
    return monotonic_lambda

def interpolate_ramp(decile_bounds, curve, current_model_val):
    """
    Interpolates or extrapolates a value from a RAMP curve using linear
    extrapolation to match MATLAB's 'extrap' functionality.
    Logic taken directly from ramp_correction_v3.py.
    """
    # This now mimics the behavior of MATLAB's interp1(..., 'extrap') [cite: 70]
    valid_mask = ~np.isnan(decile_bounds) & ~np.isnan(curve)
    if np.sum(valid_mask) < 2:
        return np.nanmean(curve)
    x, y = decile_bounds[valid_mask], curve[valid_mask]
    sort_idx = np.argsort(x); x, y = x[sort_idx], y[sort_idx]
    if current_model_val >= x[0] and current_model_val <= x[-1]:
        return np.interp(current_model_val, x, y)
    elif current_model_val < x[0]:
        slope = (y[1] - y[0]) / (x[1] - x[0])
        return y[0] + slope * (current_model_val - x[0])
    else: # current_model_val > x[-1]
        slope = (y[-1] - y[-2]) / (x[-1] - x[-2])
        return y[-1] + slope * (current_model_val - x[-1])

# --- UPDATED: Worker function with corrected logic ---
def process_grid_point(p, model_grid_df, dma_col, global_bounds, global_curve1, global_curve2, obs_kdtree, collocated_df, time_window_months):
    """
    Contains all logic to process a single grid point.
    This function is executed by each worker process in the pool.
    """
    # Constants
    GLOBAL_FALLBACK_CODE = 99
    INITIAL_NEIGHBORS = 100
    NEIGHBOR_INCREMENT = 250
    MAX_ADJUSTMENTS = 5
    TOTAL_BINS = 10
    
    grid_loc = model_grid_df.loc[p, ['lon', 'lat']].values
    current_model_val = model_grid_df.loc[p, dma_col]

    if np.isnan(current_model_val):
        return p, np.nan, np.nan, 0

    # Early exit check
    valid_global_bounds = global_bounds[~np.isnan(global_bounds)]
    if len(valid_global_bounds) > 1 and (current_model_val < valid_global_bounds.min() or current_model_val > valid_global_bounds.max()):
        l1 = interpolate_ramp(global_bounds, global_curve1, current_model_val)
        l2 = interpolate_ramp(global_bounds, global_curve2, current_model_val)
        return p, l1, max(0.01, l2) if not np.isnan(l2) else 0.01, GLOBAL_FALLBACK_CODE

    # --- Local RAMP Calculation ---
    _, nearest_indices = obs_kdtree.query(grid_loc, k=min(INITIAL_NEIGHBORS, len(obs_kdtree.data)))
    local_data = collocated_df.iloc[nearest_indices]
    local_data_window = local_data[local_data['month'].isin(time_window_months)]
    local_bounds, local_curve1, local_curve2, _ = compute_decile_curves(
        local_data_window['model_ozone'].values, local_data_window['observed_ozone'].values, TOTAL_BINS
    )
    
    # ADDED: Sorting of local curves to fix interpolation bug
    # This is critical for correct interpolation [cite: 95]
    if len(local_bounds[~np.isnan(local_bounds)]) > 0:
        sort_idx = np.argsort(local_bounds)
        local_bounds, local_curve1, local_curve2 = local_bounds[sort_idx], local_curve1[sort_idx], local_curve2[sort_idx]
        
    local_curve1 = enforce_monotonicity(local_curve1)
    valid_local_bounds = local_bounds[~np.isnan(local_bounds)]
    
    if len(valid_local_bounds) > 1 and current_model_val >= valid_local_bounds.min() and current_model_val <= valid_local_bounds.max():
        l1 = interpolate_ramp(local_bounds, local_curve1, current_model_val)
        l2 = interpolate_ramp(local_bounds, local_curve2, current_model_val)
        return p, l1, max(0.01, l2) if not np.isnan(l2) else 0.01, 1
    
    # --- Iteratively Adjust Local RAMP ---
    num_neighbors = INITIAL_NEIGHBORS
    for i in range(MAX_ADJUSTMENTS):
        num_neighbors += NEIGHBOR_INCREMENT
        if num_neighbors > len(collocated_df): break
        _, nearest_indices = obs_kdtree.query(grid_loc, k=min(num_neighbors, len(obs_kdtree.data)))
        adj_data = collocated_df.iloc[nearest_indices]
        adj_data_window = adj_data[adj_data['month'].isin(time_window_months)]
        adj_bounds, adj_curve1, adj_curve2, _ = compute_decile_curves(
            adj_data_window['model_ozone'].values, adj_data_window['observed_ozone'].values, TOTAL_BINS
        )
        
        # ADDED: Sorting of adjusted local curves
        if len(adj_bounds[~np.isnan(adj_bounds)]) > 0:
            sort_idx = np.argsort(adj_bounds)
            adj_bounds, adj_curve1, adj_curve2 = adj_bounds[sort_idx], adj_curve1[sort_idx], adj_curve2[sort_idx]
            
        adj_curve1 = enforce_monotonicity(adj_curve1)
        valid_adj_bounds = adj_bounds[~np.isnan(adj_bounds)]
        
        if len(valid_adj_bounds) > 1 and current_model_val >= valid_adj_bounds.min() and current_model_val <= valid_adj_bounds.max():
            l1 = interpolate_ramp(adj_bounds, adj_curve1, current_model_val)
            l2 = interpolate_ramp(adj_bounds, adj_curve2, current_model_val)
            return p, l1, max(0.01, l2) if not np.isnan(l2) else 0.01, i + 2

    # --- Fallback to Global RAMP ---
    l1 = interpolate_ramp(global_bounds, global_curve1, current_model_val)
    l2 = interpolate_ramp(global_bounds, global_curve2, current_model_val)
    return p, l1, max(0.01, l2) if not np.isnan(l2) else 0.01, GLOBAL_FALLBACK_CODE


# --- Main RAMP Correction Function (Parallelized) ---
def get_overhang_ramp(collocated_df, model_grid_df, results_dir=None, num_cores=None):
    """
    Computes spatiotemporal correction values in parallel.
    """
    print("Starting RAMP correction process...")

    if num_cores is None:
        num_cores = cpu_count()
    print(f"Using {num_cores} cores for parallel processing.")

    plotting_enabled = results_dir is not None
    if plotting_enabled and not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Created results directory: {results_dir}")

    num_grid_points = len(model_grid_df)
    dma_cols = [f'DMA8_{i+1}' for i in range(12)]
    
    lambda1_df = pd.DataFrame(np.nan, index=model_grid_df.index, columns=dma_cols)
    lambda2_df = pd.DataFrame(np.nan, index=model_grid_df.index, columns=dma_cols)
    technique_df = pd.DataFrame(0, index=model_grid_df.index, columns=dma_cols)
    
    # Using drop_duplicates is safer for KD-Tree
    obs_coords = collocated_df[['lon_toar', 'lat_toar']].values
    obs_kdtree = cKDTree(obs_coords)
    
    for t in range(12):
        month = t + 1
        dma_col = f'DMA8_{t+1}'
        print(f"\nProcessing Month: {month} ({dma_col})")
        
        time_window_months = [m + 1 for m in get_time_window(t)]

        # Global RAMP is calculated once per month
        global_data = collocated_df[collocated_df['month'].isin(time_window_months)]
        global_bounds, global_curve1, global_curve2, global_bin_edges = compute_decile_curves(
            global_data['model_ozone'].values, global_data['observed_ozone'].values
        )
        global_curve1 = enforce_monotonicity(global_curve1)
        
        # ADDED: Sorting global curve, which was missing in old parallel version
        if len(global_bounds[~np.isnan(global_bounds)]) > 0:
            sort_idx = np.argsort(global_bounds)
            global_bounds, global_curve1, global_curve2 = global_bounds[sort_idx], global_curve1[sort_idx], global_curve2[sort_idx]

        if plotting_enabled:
            plot_global_ramp(global_bounds, global_curve1, global_curve2, 
                             global_data['model_ozone'].values, global_data['observed_ozone'].values, 
                             global_bin_edges, month, results_dir)

        with Pool(processes=num_cores) as pool:
            worker_func = partial(process_grid_point, 
                                  model_grid_df=model_grid_df, 
                                  dma_col=dma_col, 
                                  global_bounds=global_bounds, 
                                  global_curve1=global_curve1, 
                                  global_curve2=global_curve2, 
                                  obs_kdtree=obs_kdtree, 
                                  collocated_df=collocated_df, 
                                  time_window_months=time_window_months)
            
            results = pool.map(worker_func, range(num_grid_points))

        for p, l1, l2, tech in results:
            lambda1_df.loc[p, dma_col] = l1
            lambda2_df.loc[p, dma_col] = l2
            technique_df.loc[p, dma_col] = tech
                        
        if plotting_enabled:
            plot_technique_map(model_grid_df[['lon', 'lat']], technique_df[dma_col], month, results_dir)
            
    print("\nRAMP correction process finished.")
    return lambda1_df, lambda2_df, technique_df


# --- Plotting Functions ---
def plot_global_ramp(bounds, curve1, curve2, model_vals, obs_vals, bin_edges, month, results_dir):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), constrained_layout=True)
    fig.suptitle(f'Global RAMP Curves - Month {month}', fontsize=16)
    ax1.scatter(model_vals, obs_vals, alpha=0.1, label='Model-Obs Pairs', s=10)
    ax1.plot(bounds, curve1, '-o', color='red', linewidth=2, label='$\lambda_1$ Curve (Mean)')
    min_val, max_val = np.nanmin(model_vals), np.nanmax(model_vals)
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', label='1:1 Line')
    ax1.set_xlabel('Model Ozone'); ax1.set_ylabel('Observed Ozone')
    ax1.set_title('$\lambda_1$ Mean Correction'); ax1.legend(); ax1.grid(True, linestyle='--', alpha=0.6)
    ax2.plot(bounds, curve2, '-o', color='blue', linewidth=2, label='$\lambda_2$ Curve (Variance)')
    ax2.set_xlabel('Model Ozone'); ax2.set_ylabel('Observed Ozone Variance')
    ax2.set_title('$\lambda_2$ Variance Correction'); ax2.legend(); ax2.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(results_dir, f'global_ramp_month_{month}.png'))
    plt.close(fig)

def plot_technique_map(coords, technique_vals, month, results_dir):
    plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5); ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 98.5, 99.5]
    colors = ['gray', 'green', 'blue', 'purple', 'orange', 'red', 'yellow', 'black']
    cmap = plt.cm.colors.ListedColormap(colors)
    norm = BoundaryNorm(bounds, cmap.N)
    scatter = ax.scatter(coords['lon'], coords['lat'], c=technique_vals, cmap=cmap, norm=norm, s=1)
    cbar = plt.colorbar(scatter, label='Correction Technique', ticks=[0, 1, 2, 3, 4, 5, 99])
    cbar.ax.set_yticklabels(['None', 'Local', 'Adj-1', 'Adj-2', 'Adj-3', 'Adj-4', 'Global'])
    plt.title(f'Technique Map - Month {month}')
    ax.gridlines(linestyle='--', alpha=0.5, draw_labels=True)
    plt.savefig(os.path.join(results_dir, f'technique_map_month_{month}.png'), bbox_inches='tight', dpi=300)
    plt.close()