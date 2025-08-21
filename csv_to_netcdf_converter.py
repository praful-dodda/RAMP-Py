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

def convert_source_to_netcdf_direct(source, output_dir="./data/netcdf_combined"):
    """
    Convert a single source's yearly CSV files to a combined NetCDF file
    using a direct xarray creation approach.
    
    Parameters:
    -----------
    source : str
        The data source/model (e.g., 'TCR-2', 'CAMS', etc.)
    output_dir : str
        Directory to save the combined NetCDF file
    
    Returns:
    --------
    str
        Path to the saved NetCDF file, or None if an error occurred
    """
    print(f"Processing {source}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the info for this source
    try:
        # Try to get a file path for a typical year
        file_path = None
        for test_year in [2010, 2015, 2020, 2000, 1995]:
            try:
                file_path = get_ozone_file(source, test_year)
                if file_path is not None and os.path.exists(file_path):
                    break
            except:
                continue
        
        if file_path is None:
            print(f"Could not determine year range for {source}")
            return None
            
        # Parse the directory name to get start and end years
        dir_name = os.path.basename(os.path.dirname(file_path))
        year_parts = [int(s) for s in dir_name.split() if s.isdigit()]
        
        if len(year_parts) >= 2:
            start_year, end_year = year_parts[0], year_parts[1]
        else:
            # Try to determine years from get_ozone_file function
            start_year, end_year = None, None
            for test_year in range(1990, 2024):
                try:
                    test_path = get_ozone_file(source, test_year)
                    if test_path is not None and os.path.exists(test_path):
                        if start_year is None:
                            start_year = test_year
                        end_year = test_year
                except:
                    pass
            
            if start_year is None or end_year is None:
                print(f"Could not determine year range for {source}")
                return None
    
    except Exception as e:
        print(f"Error determining year range for {source}: {str(e)}")
        return None
    
    print(f"Year range for {source}: {start_year}-{end_year}")
    
    # Create a list to hold all yearly datasets
    yearly_datasets = []
    
    # Process each year
    for year in range(start_year, end_year + 1):
        try:
            # Get the file path for this year
            file_path = get_ozone_file(source, year)
            
            if file_path is None or not os.path.exists(file_path):
                print(f"  File not found for {source} {year}, skipping")
                continue
            
            # Load the CSV file
            df = pd.read_csv(file_path)
            
            # Check if this is a valid dataframe with the expected columns
            if 'lon' not in df.columns or 'lat' not in df.columns:
                print(f"  Invalid CSV format for {source} {year}, missing lon/lat columns")
                continue
                
            # Extract monthly columns (DMA8_1, DMA8_2, etc.)
            month_cols = [col for col in df.columns if col.startswith('DMA8_')]
            
            if not month_cols:
                print(f"  No monthly data columns found for {source} {year}")
                continue
                
            # Create xarray dataset directly
            ds = create_xarray_dataset(df, year, month_cols)
            
            # Append to list of yearly datasets
            yearly_datasets.append(ds)
            
            print(f"  Successfully processed {source} {year}")
            
        except Exception as e:
            print(f"  Error processing {source} {year}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    if not yearly_datasets:
        print(f"No valid data found for {source}, skipping")
        return None
        
    # Combine all years into a single dataset
    try:
        combined_ds = xr.concat(yearly_datasets, dim='time')
        
        # Sort by time
        combined_ds = combined_ds.sortby('time')
        
        # Add metadata
        combined_ds.attrs['title'] = f'{source} Surface Ozone MDA8'
        combined_ds.attrs['source'] = source
        combined_ds.attrs['creation_date'] = datetime.now().strftime('%Y-%m-%d')
        combined_ds.attrs['description'] = f'Monthly MDA8 surface ozone from {source} for {start_year}-{end_year}'
        
        # Add variable attributes
        combined_ds['mda8'].attrs['long_name'] = 'Monthly average of daily maximum 8-hour average ozone'
        combined_ds['mda8'].attrs['units'] = 'ppb'
        
        # Save as NetCDF
        output_file = os.path.join(output_dir, f"{source}_MDA8_combined.nc")
        combined_ds.to_netcdf(output_file)
        print(f"Saved combined NetCDF for {source} to {output_file}")
        
        return output_file
        
    except Exception as e:
        print(f"Error combining datasets for {source}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def convert_all_sources_direct(output_dir="./data/netcdf_combined", selected_sources=None):
    """
    Convert all available ozone datasets from yearly CSVs to combined NetCDF files
    using the direct xarray creation approach.
    
    Parameters:
    -----------
    output_dir : str
        Directory to save the combined NetCDF files
    selected_sources : list, optional
        List of specific sources to process. If None, process all sources.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of all sources to process
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
        "TOAR-II": {"start_year": 1990, "end_year": 2023}
    }
    
    # Filter sources if needed
    if selected_sources:
        sources_to_process = {k: v for k, v in source_info.items() if k in selected_sources}
    else:
        sources_to_process = source_info
    
    # Process each source
    for source in tqdm(sources_to_process.keys(), desc="Processing sources"):
        convert_source_to_netcdf_direct(source, output_dir)
    
    print("\nConversion complete!")

# Example usage
if __name__ == "__main__":
    # Process a single source
    # convert_source_to_netcdf_direct("AM4", output_dir="./data/netcdf_combined")
    
    # Or process all sources
    # convert_all_sources_direct(output_dir="./data/netcdf_combined", 
    #                        selected_sources=["AM4", "CAMS", "GEOS-CF", "CESM1-CAM4-Chem", 
    #                                          "CESM1-WACCM", "CESM2.2", "CHASER", "GEOS-CF", 
    #                                          "GEOS-chem", "GEOS-GMI", "GFDL-AM3", "MERRA2-GMI", 
    #                                          "MOCAGE", "MRI-ESM1", "MRI-ESM2", "TCR-2"])
    convert_all_sources_direct(output_dir="./data/netcdf_combined", 
                           selected_sources=["TOAR-II"])
    # convert_all_sources_direct(output_dir="./data/netcdf_combined", 
    #                        selected_sources=["M3fusion"])
    
    # Or process specific sources
    # convert_all_sources_direct(output_dir="./data/netcdf_combined", 
    #                           selected_sources=["AM4", "CAMS", "GEOS-CF"])