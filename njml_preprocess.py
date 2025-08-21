import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
import os

# Quick data exploration
def explore_data(netcdf_file):
    ds = xr.open_dataset(netcdf_file)
    print("Dataset info:")
    print(ds)
    print("\nTime range:")
    print(f"Start: {ds.time.values[0]}")
    print(f"End: {ds.time.values[-1]}")
    print(f"\nUnique years: {sorted(set(ds.time.dt.year.values))}")
    print(f"Unique months: {sorted(set(ds.time.dt.month.values))}")
    
    # Check data availability by year/month
    time_df = pd.DataFrame({
        'year': ds.time.dt.year.values,
        'month': ds.time.dt.month.values
    })
    print("\nData availability by year/month:")
    print(time_df.groupby(['year', 'month']).size().unstack(fill_value=0))
    
    ds.close()

def process_ozone_data(file_path, output_directory):
    """
    Reads a NetCDF file with monthly MDA8 ozone data, reshapes it,
    and saves the output into yearly CSV files.

    The final CSV format will have columns:
    "lon", "lat", "DMA8_1", "DMA8_2", ..., "DMA8_12"

    Args:
        file_path (str): The path to the input NetCDF file.
        output_directory (str): The directory to save the output CSV files.
    """
    # --- 1. Load the dataset ---
    try:
        print(f"Attempting to load dataset from: {file_path}")
        ds = xr.open_dataset(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        

    # --- 2. Convert to Pandas DataFrame ---
    # This is often the most straightforward way to handle this kind of
    # data reshaping (from long to wide format).
    print("Converting xarray Dataset to pandas DataFrame...")
    df = ds.to_dataframe().reset_index()

    # --- 3. Extract Year and Month from the 'time' column ---
    print("Processing the 'time' dimension...")
    df['year'] = df['time'].dt.year
    df['month'] = df['time'].dt.month
    df = df.drop(columns='time') # Original datetime column is no longer needed

    # --- 4. Pivot the DataFrame ---
    # This is the key step. We are transforming the data so that each month's
    # MDA8 value gets its own column. The rows will represent unique
    # combinations of lon, lat, and year.
    print("Pivoting data to create monthly columns...")
    pivot_df = df.pivot_table(
        index=['year', 'lon', 'lat'],
        columns='month',
        values='MDA8_ppb'
    ).reset_index()

    # --- 5. Rename columns to the desired format ---
    # The pivoted columns are named 1, 2, ..., 12. We rename them.
    column_rename_map = {month: f'DMA8_{month}' for month in range(1, 13)}
    pivot_df = pivot_df.rename(columns=column_rename_map)

    # --- 6. Create output directory ---
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Created output directory: {output_directory}")

    # --- 7. Split by year and save to CSV ---
    unique_years = pivot_df['year'].unique()
    print(f"Found data for the following years: {list(unique_years)}")

    for year in unique_years:
        print(f"Processing and saving data for {year}...")

        # Filter the DataFrame for the current year
        year_df = pivot_df[pivot_df['year'] == year].copy()

        # Drop the 'year' column as it's now part of the filename
        year_df = year_df.drop(columns=['year'])

        # Define the full path for the output file
        output_filename = os.path.join(output_directory, f'mda8_ozone_{year}.csv')

        # Save the year-specific DataFrame to a CSV file
        # 'na_rep' ensures that missing values are saved as 'NaN'
        year_df.to_csv(output_filename, index=False, na_rep='NaN')

        print(f"  -> Successfully saved: {output_filename}")

    # --- 8. Clean up ---
    ds.close()
    print("\nProcessing complete.")

ds = xr.open_dataset("/work/users/p/r/praful/proj/nasa/global_o3/data/nanjing university ML/Satellite_MDA8.nc")
ds = ds.rename({'MDA8_ppb': 'DMA8_ppb'})
def plot_spatial_distribution(data_array: xr.DataArray, source_name: str, time_slice: str, title_suffix: str, output_dir: str = 'plots'):
    """
    Plots the spatial distribution of data at a specific time slice.

    Args:
        data_array (xr.DataArray): The input data.
        source_name (str): Name of the data source.
        time_slice (str): Time slice to select data.
        title_suffix (str): Suffix for the plot title.
        output_dir (str): Directory to save the plot.
    """
    import matplotlib.pyplot as plt
    import os
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    data_at_time = data_array.sel(time=time_slice, method='nearest')
    
    if data_at_time.size == 0:
        print(f"No data available for {source_name} at {time_slice}.")
        return
    
    plt.figure(figsize=(10, 6))
    data_at_time.plot(cmap='viridis')
    
    date_str = pd.to_datetime(data_at_time.time.values).strftime('%Y-%m-%d')
    plt.title(f'Spatial Distribution for {source_name}\n{title_suffix}: {date_str}')


plot_spatial_distribution(ds['DMA8_ppb'], "Nanjing University ML", "2010-01-01", "January 2010")


NETCDF_FILE = '/work/users/p/r/praful/proj/nasa/global_o3/data/nanjing university ML/Satellite_MDA8.nc'
OUTPUT_DIR = '/work/users/p/r/praful/proj/nasa/global_o3/data/nanjing university ML/yearlyFiles'
process_ozone_data(NETCDF_FILE, OUTPUT_DIR)


# Run this first to understand your data
# explore_data("/work/users/p/r/praful/proj/nasa/global_o3/data/nanjing university ML/Satellite_MDA8.nc")

# Usage
# create_yearly_csv_files_efficient("/work/users/p/r/praful/proj/nasa/global_o3/data/nanjing university ML/Satellite_MDA8.nc", "/work/users/p/r/praful/proj/nasa/global_o3/data/nanjing university ML/yearlyFiles")

# usage
# create_yearly_csv_files("/work/users/p/r/praful/proj/nasa/global_o3/data/nanjing university ML/Satellite_MDA8.nc", "/work/users/p/r/praful/proj/nasa/global_o3/data/nanjing university ML/yearlyFiles")