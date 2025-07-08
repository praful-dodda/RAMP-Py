import pandas as pd
import numpy as np
import scipy.io
import os
import matplotlib.pyplot as plt
import seaborn as sns

from preprocess import get_ozone_file

# --- Configuration ---
MODEL_NAME = "MERRA2-GMI"
YEAR = 2017
VERSION = "v3-parallel" # Version of the Python run

# --- File Paths ---
RAMP_DATA_DIR = './ramp_data'
OUTPUT_DIR = f'.{RAMP_DATA_DIR}/comparison_plots_{MODEL_NAME}_{YEAR}_mat_vs_py'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Python-generated Parquet files
L1_PARQUET_PATH = os.path.join(RAMP_DATA_DIR, f"lambda1_{MODEL_NAME}_{YEAR}_{VERSION}.parquet")
L2_PARQUET_PATH = os.path.join(RAMP_DATA_DIR, f"lambda2_{MODEL_NAME}_{YEAR}_{VERSION}.parquet")

# MATLAB-generated .mat file
MAT_FILE_PATH = '/proj/serrelab/users/heidkamp/data/20250423_003823/OverhangRAMPv6_output_2017.mat'

# Model grid file to get lon/lat coordinates
MODEL_GRID_FILE = get_ozone_file(MODEL_NAME, YEAR)
# Path to the .mat file containing the locations for MATLAB estimates ---
MAT_LOCATIONS_FILE_PATH = '/proj/serrelab/users/heidkamp/data/20250423_003823/OverhangModelData_2017.mat'

# --- 1. Load Data ---
print("--- Loading Data ---")

# Load Python RAMP results (Parquet)
try:
    l1_py_df = pd.read_parquet(L1_PARQUET_PATH)
    l2_py_df = pd.read_parquet(L2_PARQUET_PATH)
    print(f"Successfully loaded Parquet files:\n  {L1_PARQUET_PATH}\n  {L2_PARQUET_PATH}")
except Exception as e:
    print(f"Error loading Parquet files: {e}")
    exit()

# Load MATLAB RAMP results (.mat)
try:
    mat_data = scipy.io.loadmat(MAT_FILE_PATH)
    # Note: Adjust 'rampLambda1' and 'rampLambda2' if variable names in your .mat file are different
    l1_mat = mat_data['rampLambda1']
    l2_mat = mat_data['rampLambda2']
    print(f"Successfully loaded .mat file: {MAT_FILE_PATH}")

    # subset the 12 columns by removing the first 6 and last 6 columns
    l1_mat = l1_mat[:, 6:-6]
    l2_mat = l2_mat[:, 6:-6]
    # The shape should be (num_grid_points, num_months)
    print(f"  - Lambda1 shape from .mat: {l1_mat.shape}")
    print(f"  - Lambda2 shape from .mat: {l2_mat.shape}")
except Exception as e:
    print(f"Error loading .mat file: {e}")
    exit()

# --- NEW: Load MATLAB estimation locations ---
try:
    mat_loc_data = scipy.io.loadmat(MAT_LOCATIONS_FILE_PATH)
    # Access nested struct to get coordinates, assuming [lon, lat] order
    mat_coords = mat_loc_data['modelData_oh'][0,0]['sMS']
    mat_coords_df = pd.DataFrame(mat_coords, columns=['lon', 'lat'])
    print(f"Successfully loaded MATLAB estimation locations from: {MAT_LOCATIONS_FILE_PATH}")
    print(f"  - Found {len(mat_coords_df)} estimated locations in MATLAB data.")
except Exception as e:
    print(f"Error loading MATLAB locations .mat file: {e}")
    exit()

# Load coordinates
try:
    coords_df = pd.read_csv(MODEL_GRID_FILE)[['lon', 'lat']]
    print(f"Successfully loaded coordinates from: {MODEL_GRID_FILE}")
except Exception as e:
    print(f"Error loading coordinates file: {e}")
    exit()

# --- 2. Prepare and Align DataFrames ---
print("\n--- Preparing and Aligning Data ---")

# Convert MATLAB arrays to DataFrames
# Assumes the order of grid points in MATLAB array matches the model grid file
dma_cols = [f'DMA8_{i+1}' for i in range(12)]
l1_mat_df = pd.DataFrame(l1_mat, columns=dma_cols)
l2_mat_df = pd.DataFrame(l2_mat, columns=dma_cols)

# Combine coordinates with lambda data
l1_py_df = pd.concat([coords_df, l1_py_df], axis=1)
l2_py_df = pd.concat([coords_df, l2_py_df], axis=1)
l1_mat_df = pd.concat([mat_coords_df, l1_mat_df], axis=1)
l2_mat_df = pd.concat([mat_coords_df, l2_mat_df], axis=1)

# Melt DataFrames to long format
l1_py_long = pd.melt(l1_py_df, id_vars=['lon', 'lat'], var_name='month_str', value_name='lambda1_py')
l2_py_long = pd.melt(l2_py_df, id_vars=['lon', 'lat'], var_name='month_str', value_name='lambda2_py')
l1_mat_long = pd.melt(l1_mat_df, id_vars=['lon', 'lat'], var_name='month_str', value_name='lambda1_mat')
l2_mat_long = pd.melt(l2_mat_df, id_vars=['lon', 'lat'], var_name='month_str', value_name='lambda2_mat')

# Merge all python data together, and all matlab data together first.
py_df = pd.merge(l1_py_long, l2_py_long, on=['lon', 'lat', 'month_str'])
mat_df = pd.merge(l1_mat_long, l2_mat_long, on=['lon', 'lat', 'month_str'])

# Perform an inner join to only compare locations present in both datasets
comp_df = pd.merge(py_df, mat_df, on=['lon', 'lat', 'month_str'], how='inner')

# Calculate differences
comp_df['lambda1_diff'] = comp_df['lambda1_py'] - comp_df['lambda1_mat']
comp_df['lambda2_diff'] = comp_df['lambda2_py'] - comp_df['lambda2_mat']
comp_df.dropna(inplace=True)

print(f"Created comparison DataFrame with {len(comp_df)} entries.")

# --- 3. Calculate and Display Statistics ---
print("\n--- Comparison Statistics (Python - MATLAB) ---")
print("\nLambda1 Difference Stats:")
print(comp_df['lambda1_diff'].describe())
print("\nLambda2 Difference Stats:")
print(comp_df['lambda2_diff'].describe())

# --- 4. Generate Visualizations ---
print(f"\n--- Generating Plots (saving to {OUTPUT_DIR}) ---")

# Plot 1: Scatter plot of Python vs MATLAB values
sample_df = comp_df.sample(n=min(50000, len(comp_df)), random_state=42)
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
sns.scatterplot(data=sample_df, x='lambda1_mat', y='lambda1_py', alpha=0.5, s=10, ax=axes[0])
axes[0].set_title('Lambda1: Python vs. MATLAB')
lims1 = [min(axes[0].get_xlim()[0], axes[0].get_ylim()[0]), max(axes[0].get_xlim()[1], axes[0].get_ylim()[1])]
axes[0].plot(lims1, lims1, 'k--', alpha=0.75, zorder=0)

# annotate the plot with comparison statistics between python and matlab values by calculating 
# correlation, mean difference, and standard deviation of the differences
corr1 = comp_df['lambda1_py'].corr(comp_df['lambda1_mat'])
mean_diff1 = comp_df['lambda1_diff'].mean()
std_diff1 = comp_df['lambda1_diff'].std()
axes[0].annotate(f'Correlation: {corr1:.2f}\nMean Diff: {mean_diff1:.2f}\nStd Dev: {std_diff1:.2f}',
                 xy=(0.05, 0.95), xycoords='axes fraction',
                 fontsize=10, ha='left', va='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

sns.scatterplot(data=sample_df, x='lambda2_mat', y='lambda2_py', alpha=0.5, s=10, ax=axes[1])
axes[1].set_title('Lambda2: Python vs. MATLAB')
lims2 = [min(axes[1].get_xlim()[0], axes[1].get_ylim()[0]), max(axes[1].get_xlim()[1], axes[1].get_ylim()[1])]
axes[1].plot(lims2, lims2, 'k--', alpha=0.75, zorder=0)
# annotate the plot with comparison statistics between python and matlab values by calculating 
# correlation, mean difference, and standard deviation of the differences
corr2 = comp_df['lambda2_py'].corr(comp_df['lambda2_mat'])
mean_diff2 = comp_df['lambda2_diff'].mean()
std_diff2 = comp_df['lambda2_diff'].std()
axes[1].annotate(f'Correlation: {corr2:.2f}\nMean Diff: {mean_diff2:.2f}\nStd Dev: {std_diff2:.2f}',
                 xy=(0.05, 0.95), xycoords='axes fraction',
                 fontsize=10, ha='left', va='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
plt.suptitle('Python vs. MATLAB RAMP Parameter Comparison', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(OUTPUT_DIR, 'py_vs_mat_scatter.png'), dpi=300)
plt.close()
print("Saved scatter plot.")

# Plot 2: Histogram of differences
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.histplot(comp_df['lambda1_diff'], bins=50, ax=axes[0], kde=True)
axes[0].set_title('Distribution of Lambda1 Differences')
axes[0].set_xlabel('Difference (Python - MATLAB)')
sns.histplot(comp_df['lambda2_diff'], bins=50, ax=axes[1], kde=True)
axes[1].set_title('Distribution of Lambda2 Differences')
axes[1].set_xlabel('Difference (Python - MATLAB)')
plt.suptitle('Histogram of Parameter Differences', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(OUTPUT_DIR, 'py_vs_mat_diff_histogram.png'), dpi=300)
plt.close()
print("Saved difference histogram.")

print("\nComparison finished.")

# Plot 3: Colorplot of differences
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
# Lambda1 differences
sns.scatterplot(data=comp_df, x='lon', y='lat', hue='lambda1_diff', palette='coolwarm', ax=axes[0], s=10, alpha=0.5)
axes[0].set_title('Lambda1 Differences (Python - MATLAB)')
axes[0].set_xlabel('Longitude')
axes[0].set_ylabel('Latitude')
# Lambda2 differences
sns.scatterplot(data=comp_df, x='lon', y='lat', hue='lambda2_diff', palette='coolwarm', ax=axes[1], s=10, alpha=0.5)
axes[1].set_title('Lambda2 Differences (Python - MATLAB)')
axes[1].set_xlabel('Longitude')
axes[1].set_ylabel('Latitude')
plt.suptitle('Geospatial Distribution of Parameter Differences', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(OUTPUT_DIR, 'py_vs_mat_geo_diff.png'), dpi=300)
plt.close()
print("Saved geospatial difference plot.")

# Plot 4: Create 12-monthly sub-plots for each parameter where each plot shows ditribution of mat and py values
for param, label in [('lambda1', 'Lambda1'), ('lambda2', 'Lambda2')]:
    fig, axes = plt.subplots(3, 4, figsize=(20, 12), sharex=True, sharey=True)
    fig.suptitle(f'{label} Monthly Distributions (Python vs. MATLAB)', fontsize=16)
    
    for i in range(12):
        ax = axes[i // 4, i % 4]
        sns.histplot(comp_df[f'{param}_py'][comp_df['month_str'] == f'DMA8_{i+1}'], bins=50, kde=True, color='blue', ax=ax, label='Python', alpha=0.5)
        sns.histplot(comp_df[f'{param}_mat'][comp_df['month_str'] == f'DMA8_{i+1}'], bins=50, kde=True, color='orange', ax=ax, label='MATLAB', alpha=0.5)
        ax.set_title(f'Month {i+1}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.legend()
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(OUTPUT_DIR, f'py_vs_mat_monthly_{param}.png'), dpi=300)
    plt.close()
    print(f"Saved monthly distribution plot for {label}.")
