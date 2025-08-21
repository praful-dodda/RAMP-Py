import pandas as pd
import numpy as np
import xarray as xr
import os
from datetime import datetime
from tqdm import tqdm
from preprocess import get_ozone_file

def create_xarray_dataset(df, year, month_cols):
    """
    Create an xarray dataset from a dataframe with monthly columns.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with lat, lon, and monthly DMA8 columns
    year : int
        Year of the data
    month_cols : list
        List of monthly column names (e.g., ['DMA8_1', 'DMA8_2', ...])
    
    Returns:
    --------
    xarray.Dataset
        Dataset with proper dimensions and coordinates
    """
    # Get unique lats and lons and ensure they're sorted
    lats = np.sort(df['lat'].unique())
    lons = np.sort(df['lon'].unique())
    
    # Create time array for all months
    times = pd.to_datetime([f"{year}-{i}-15" for i in range(1, 13)])
    
    # Create empty data array
    data = np.full((len(times), len(lats), len(lons)), np.nan)
    
    # Map lat/lon values to array indices
    lat_indices = {lat: i for i, lat in enumerate(lats)}
    lon_indices = {lon: i for i, lon in enumerate(lons)}
    
    # Sort columns numerically based on the month number, not alphabetically
    sorted_month_cols = sorted(month_cols, key=lambda x: int(x.split('_')[1]))

    # Fill the data array with values from the dataframe
    for i, month_col in enumerate(sorted_month_cols):
        month_data = df[['lat', 'lon', month_col]].dropna()
        for _, row in month_data.iterrows():
            if row['lat'] in lat_indices and row['lon'] in lon_indices:
                lat_idx = lat_indices[row['lat']]
                lon_idx = lon_indices[row['lon']]
                data[i, lat_idx, lon_idx] = row[month_col]
    
    # Create the xarray dataset
    ds = xr.Dataset(
        data_vars={
            'mda8': (['time', 'lat', 'lon'], data)
        },
        coords={
            'time': times,
            'lat': lats,
            'lon': lons
        }
    )
    
    return ds

def convert_source_to_netcdf_direct(source, start_year, end_year, output_dir="./data/netcdf_combined"):
    """
    Convert a source's yearly CSV files to NetCDF.
    For 'TOAR-II' and 'M3fusion', it saves one file per year in a dedicated subfolder.
    For all other sources, it saves a single combined NetCDF file.
    
    Parameters:
    -----------
    source : str
        The data source/model (e.g., 'TCR-2', 'CAMS', etc.)
    start_year : int
        The first year to process.
    end_year : int
        The last year to process.
    output_dir : str
        Directory to save the NetCDF file(s).
    
    Returns:
    --------
    list or str
        A list of paths to the saved annual NetCDF files for special models,
        a single path for combined files, or None if an error occurred.
    """
    print(f"Processing {source} for years {start_year}-{end_year}...")
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Special handling for models requiring annual files ---
    if source in ['TOAR-II', 'M3fusion']:
        model_output_dir = os.path.join(output_dir, source)
        os.makedirs(model_output_dir, exist_ok=True)
        saved_files = []

        for year in range(start_year, end_year + 1):
            try:
                file_path = get_ozone_file(source, year)
                if file_path is None or not os.path.exists(file_path):
                    print(f"  File not found for {source} {year}, skipping.")
                    continue
                
                df = pd.read_csv(file_path)
                month_cols = [col for col in df.columns if col.startswith('DMA8_')]
                if 'lon' not in df.columns or 'lat' not in df.columns or not month_cols:
                    print(f"  Invalid or empty CSV format for {source} {year}, skipping.")
                    continue
                
                # Create a single-year xarray dataset
                ds = create_xarray_dataset(df, year, month_cols)
                
                # Add metadata
                ds.attrs['title'] = f'{source} Surface Ozone MDA8 for {year}'
                ds.attrs['source'] = source
                ds.attrs['creation_date'] = datetime.now().strftime('%Y-%m-%d')
                ds.attrs['description'] = f'Monthly MDA8 surface ozone from {source} for {year}'
                ds['mda8'].attrs['long_name'] = 'Monthly average of daily maximum 8-hour average ozone'
                ds['mda8'].attrs['units'] = 'ppb'
                
                # Save as NetCDF for the single year
                output_file = os.path.join(model_output_dir, f"{source}_MDA8_{year}.nc")
                ds.to_netcdf(output_file)
                print(f"  Saved annual NetCDF for {source} {year} to {output_file}")
                saved_files.append(output_file)
                
            except Exception as e:
                print(f"  Error processing {source} {year}: {str(e)}")
        
        return saved_files if saved_files else None

    # --- Default handling for all other models (combined file) ---
    else:
        yearly_datasets = []
        for year in range(start_year, end_year + 1):
            try:
                file_path = get_ozone_file(source, year)
                if file_path is None or not os.path.exists(file_path):
                    print(f"  File not found for {source} {year}, skipping")
                    continue
                
                df = pd.read_csv(file_path)
                if 'lon' not in df.columns or 'lat' not in df.columns:
                    print(f"  Invalid CSV format for {source} {year}, missing lon/lat columns")
                    continue
                
                month_cols = [col for col in df.columns if col.startswith('DMA8_')]
                if not month_cols:
                    print(f"  No monthly data columns found for {source} {year}")
                    continue
                    
                ds = create_xarray_dataset(df, year, month_cols)
                yearly_datasets.append(ds)
                print(f"  Successfully processed {source} {year}")
                
            except Exception as e:
                print(f"  Error processing {source} {year}: {str(e)}")
        
        if not yearly_datasets:
            print(f"No valid data found for {source} to combine, skipping.")
            return None
            
        try:
            combined_ds = xr.concat(yearly_datasets, dim='time').sortby('time')
            
            combined_ds.attrs['title'] = f'{source} Surface Ozone MDA8'
            combined_ds.attrs['source'] = source
            combined_ds.attrs['creation_date'] = datetime.now().strftime('%Y-%m-%d')
            combined_ds.attrs['description'] = f'Monthly MDA8 surface ozone from {source} for {start_year}-{end_year}'
            combined_ds['mda8'].attrs['long_name'] = 'Monthly average of daily maximum 8-hour average ozone'
            combined_ds['mda8'].attrs['units'] = 'ppb'
            
            output_file = os.path.join(output_dir, f"{source}_MDA8_combined.nc")
            combined_ds.to_netcdf(output_file)
            print(f"Saved combined NetCDF for {source} to {output_file}")
            return output_file
            
        except Exception as e:
            print(f"Error combining datasets for {source}: {str(e)}")
            return None

def convert_all_sources_direct(output_dir="./data/netcdf_combined", selected_sources=None):
    """
    Convert available ozone datasets from yearly CSVs to NetCDF files.
    
    Parameters:
    -----------
    output_dir : str
        Directory to save the NetCDF files.
    selected_sources : list, optional
        List of specific sources to process. If None, process all sources.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    source_info = {
        "AM4": {"start_year": 2008, "end_year": 2023},
        "CAMS": {"start_year": 2003, "end_year": 2023},
        "CESM1-CAM4-Chem": {"start_year": 1990, "end_year": 2001},
        "CESM1-WACCM": {"start_year": 1990, "end_year": 2001, "missing_years": [1997, 1998, 1999]},
        "CESM2.2": {"start_year": 2002, "end_year": 2022},
        "CHASER": {"start_year": 1990, "end_year": 2010},
        "GEOS-CF": {"start_year": 2018, "end_year": 2023},
        "GEOS-chem": {"start_year": 2006, "end_year": 2016},
        "GEOS-GMI": {"start_year": 1996, "end_year": 2022},
        "GFDL-AM3": {"start_year": 1990, "end_year": 2007},
        "M3fusion": {"start_year": 1990, "end_year": 2023},
        "MERRA2-GMI": {"start_year": 1990, "end_year": 2019},
        "MOCAGE": {"start_year": 1990, "end_year": 2010, "missing_years": [1998]},
        "MRI-ESM1": {"start_year": 1990, "end_year": 2010},
        "MRI-ESM2": {"start_year": 2011, "end_year": 2017},
        "TCR-2": {"start_year": 2005, "end_year": 2021},
        "TOAR-II": {"start_year": 1990, "end_year": 2023},
        "UKML": {"start_year": 1990, "end_year": 2019},
        "NJML": {"start_year": 2003, "end_year": 2019}
    }
    
    sources_to_process = {k: v for k, v in source_info.items() if k in selected_sources} if selected_sources else source_info
    
    for source, years in tqdm(sources_to_process.items(), desc="Processing sources"):
        start_year = years['start_year']
        end_year = years['end_year']
        convert_source_to_netcdf_direct(source, start_year, end_year, output_dir)
    
    print("\nConversion complete!")

# Example usage
if __name__ == "__main__":
    # --- To process ALL sources based on your requirements ---
    # convert_all_sources_direct(output_dir="./data/netcdf_combined")

    # --- To process specific sources ---
    # Process a model that gets a COMBINED file
    # convert_all_sources_direct(output_dir="./data/netcdf_combined", selected_sources=["AM4"])
    
    # Process models that get ANNUAL files
    convert_all_sources_direct(output_dir="./data/netcdf_combined", selected_sources=["NJML", "UKML"])