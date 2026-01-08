import os
import pandas as pd
import xarray as xr

# Import additional libraries needed for spatial mapping
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# Set better figure aesthetics
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 10)


def get_ozone_file(source, year, month=None):
    """
    Get the filepath for ozone MDA8 data based on source, year, and month.
    
    Parameters:
    -----------
    source : str
        The data source/model (e.g., 'AM4', 'CAMS', 'GEOS-CF', etc.)
    year : int
        The year of interest (e.g., 2010)
    month : int, optional
        The month of interest (1-12). Used for filepath determination for sources
        with monthly files like M3fusion.
        
    Returns:
    --------
    str
        The full path to the file, or None if no file exists for the given parameters.
    """
    # Define base directory
    base_dir = "/work/users/p/r/praful/proj/nasa/global_o3/data"
    
    # Define source-specific patterns and date ranges
    source_info = {
        "AM4": {
            "dir": "AM4 2008-2023",
            "pattern": "AM4-monthly-mda8-{year}.csv",
            "start_year": 2008,
            "end_year": 2023
        },
        "CAMS": {
            "dir": "CAMS 2003-2023",
            "pattern": "CAMS_monthly_{year}.csv",
            "start_year": 2003,
            "end_year": 2023
        },
        "CESM1-CAM4-Chem": {
            "dir": "CESM1-CAM4-Chem 1990-2001",
            "pattern": "CESM1-CAM4-Chem-monthly-dma8-{year}.csv",
            "start_year": 1990,
            "end_year": 2001
        },
        "CESM1-WACCM": {
            "dir": "CESM1 WACCM 1990-2001 (missing 1997 1998 1999)",
            "pattern": "WACCM-monthly-dma8-{year}.csv",
            "start_year": 1990,
            "end_year": 2001,
            "missing_years": [1997, 1998, 1999]
        },
        "CESM2.2": {
            "dir": "CESM2.2-CAM4-Chem 2002-2022",
            "pattern": "CESM2.2-monthly-dma8-{year}.csv",
            "start_year": 2002,
            "end_year": 2022
        },
        "CHASER": {
            "dir": "CHASER 1990-2010",
            "pattern": "CHASER-monthly-dma8-{year}.csv",
            "start_year": 1990,
            "end_year": 2010
        },
        "GEOS-CF": {
            "dir": "GEOS-CF 2018-2023",
            "pattern": "GEOS-CF-monthly-dma8-{year}.csv",
            "start_year": 2018,
            "end_year": 2023
        },
        "GEOS-chem": {
            "dir": "GEOS-chem 2006-2016",
            "pattern": "GEOS_monthly_{year}.csv",
            "start_year": 2006,
            "end_year": 2016
        },
        "GEOS-GMI": {
            "dir": "GEOS-GMI 1996 - 2022",
            "pattern": "GEOS-GMI-monthly-dma8-{year}.csv",
            "start_year": 1996,
            "end_year": 2022
        },
        "GFDL-AM3": {
            "dir": "GFDL AM3 1990-2007",
            "pattern_pre_2001": "GFDL AM3-monthly-dma8-{year}.csv",
            "pattern_post_2000": "GFDL AM3-monthly-mda8-{year}.csv",
            "start_year": 1990,
            "end_year": 2007,
            "pattern_change_year": 2001
        },
        "M3fusionNC": {
            "dir": "M3fusion 1990-2023/M3fusion_nc_files",
            "pattern": "mda8_ModelFusion_{year}.nc",
            "pattern_monthly": "mda8_ModelFusion_{year}{month:02d}-v1.nc",
            "start_year": 1990,
            "end_year": 2023,
            "has_monthly_files": True
        },
        "M3fusion": {
            "dir": "M3fusion 1990-2023/yearlyFiles",
            "pattern": "M3fusion-monthly-mda8-{year}.csv",
            "start_year": 1990,
            "end_year": 2023
        },
        "MERRA2-GMI": {
            "dir": "MERRA2-GMI 1990-2019",
            "pattern": "MERRA-monthly-dma8-{year}.csv",
            "start_year": 1990,
            "end_year": 2019
        },
        "MOCAGE": {
            "dir": "MOCAGE 1990-2010 (missing 1998)",
            "pattern": "MOCAGE-monthly-dma8-{year}.csv",
            "start_year": 1990,
            "end_year": 2010,
            "missing_years": [1998]
        },
        "MRI-ESM1": {
            "dir": "MRI-ESM1 1990-2010",
            "pattern": "MRI-ESM1-monthly-dma8-{year}.csv",
            "start_year": 1990,
            "end_year": 2010
        },
        "MRI-ESM2": {
            "dir": "MRI-ESM2 2011-2017",
            "pattern": "MRI-ESM2-monthly-dma8-{year}.csv",
            "start_year": 2011,
            "end_year": 2017
        },
        "TCR-2": {
            "dir": "TCR-2 2005-2021",
            "pattern": "TCR2_monthly_{year}.csv",
            "start_year": 2005,
            "end_year": 2021
        },
        "TOAR-II": {
            "dir": "TOAR-II/yearlyFiles",
            "pattern": "TOAR-II-monthly-mda8-{year}.csv",
            "start_year": 1990,
            "end_year": 2023
        },
        "NJML": {
            "dir": "nanjing university ML/yearlyFiles",
            "pattern": "mda8_ozone_{year}.csv",
            "start_year": 2003,
            "end_year": 2019
        },
        "UKML": {
            "dir": "UK Cambridge ML/yearlyFiles",
            "pattern": "reshaped_popwt_ozone_{year}.csv",
            "start_year": 1990,
            "end_year": 2019
        },
        "OMI-MLS":{
            "dir": "BME corrected satellite data 2005-2022/reformatted_data/",
            "pattern": "correct_omi_mls_{year}_framp_format.csv",
            "start_year": 2005,
            "end_year": 2020
        },
        "IASI-GOME2": {
            "dir": "BME corrected satellite data 2005-2022/reformatted_data/",
            "pattern": "correct_IASI_GOME2_{year}_framp_format.csv",
            "start_year": 2017,
            "end_year": 2020
        },
        "CRIS":{
            "dir": "BME corrected satellite data 2005-2022/reformatted_data/",
            "pattern": "correct_CrIs_{year}_framp_format.csv",
            "start_year": 2022,
            "end_year": 2022
        }
    }
    
    # Check if the source is valid
    if source not in source_info:
        raise ValueError(f"Unknown source: {source}. Available sources: {', '.join(source_info.keys())}")
    
    # Get source-specific information
    info = source_info[source]
    
    # Check if the year is within the valid range
    if year < info["start_year"] or year > info["end_year"]:
        raise ValueError(f"Year {year} is outside the range for {source}: {info['start_year']}-{info['end_year']}")
    
    # Check for missing years
    if "missing_years" in info and year in info["missing_years"]:
        return None
    
    # Handle M3fusion's monthly files
    if source == "M3fusionNC" and month is not None:
        if month < 1 or month > 12:
            raise ValueError(f"Month must be between 1 and 12, got {month}")
        pattern = info["pattern_monthly"]
        filename = pattern.format(year=year, month=month)
    # Special case for GFDL-AM3 which changes pattern in 2001
    elif source == "GFDL-AM3":
        if year < info["pattern_change_year"]:
            pattern = info["pattern_pre_2001"]
        else:
            pattern = info["pattern_post_2000"]
        filename = pattern.format(year=year)
    # Special case for TOAR-II which has a single file for all years
    # elif source == "TOAR-II":
    #     filename = info["pattern"]
    else:
        filename = info["pattern"].format(year=year)
    
    # Build the full path
    file_path = f"{base_dir}/{info['dir']}/{filename}"

    if file_path is None:
        print(f"No data available for {source} in year {year}" + 
              (f", month {month}" if month is not None else ""))
    
    # Check if the file exists
    if not os.path.exists(file_path):
        # if source == "M3fusionNC", then try replacing  -v1.nc with .nc
        # This is a special case for M3fusionNC where the file might have a version suffix
        if source == "M3fusionNC" and file_path.endswith('-v1.nc'):
            file_path = file_path.replace('-v1.nc', '.nc')
            if not os.path.exists(file_path):
                print(f"Files not found: {file_path} and {file_path.replace('.nc', '-v1.nc')}")
                
        else:
            print(f"File not found: {file_path}")

    return file_path


def print_ozone_file_info(source, year, month=None):
    """
    Reads an ozone data file and prints basic information about its contents.
    
    Parameters:
    -----------
    source : str
        The data source/model (e.g., 'AM4', 'CAMS', 'GEOS-CF', etc.)
    year : int
        The year of interest (e.g., 2010)
    month : int, optional
        The month of interest (1-12). Used for sources with monthly files like M3fusion.
    
    Returns:
    --------
    dict or xarray.Dataset or None
        For CSV files: Dictionary with keys 'df' (DataFrame), 'columns', and 'shape'
        For netCDF files: The xarray Dataset
        Returns None if file doesn't exist
    """
    # Get the file path
    file_path = get_ozone_file(source, year, month)
    
    if file_path is None:
        print(f"No data available for {source} in year {year}" + 
              (f", month {month}" if month is not None else ""))
        return None
    
    # Check if the file exists
    if not os.path.exists(file_path):
        # if source == "M3fusionNC", then try replacing  -v1.nc with .nc
        # This is a special case for M3fusionNC where the file might have a version suffix
        if source == "M3fusionNC" and file_path.endswith('-v1.nc'):
            file_path = file_path.replace('-v1.nc', '.nc')
            if not os.path.exists(file_path):
                print(f"Files not found: {file_path} and {file_path.replace('.nc', '-v1.nc')}")
                return None
        else:
            print(f"File not found: {file_path}")
            return None

    print(f"File found: {file_path}")
    print(f"Source: {source}, Year: {year}" + (f", Month: {month}" if month is not None else ""))
    print("-" * 50)
    
    # Process based on file type
    if file_path.endswith('.csv'):
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Print basic information
        print("CSV File Information:")
        print(f"Number of rows: {df.shape[0]:,}")
        print(f"Number of columns: {df.shape[1]:,}")
        print("\nColumns:")
        for col in df.columns:
            print(f"  - {col}")
        
        print("\nFirst 5 rows:")
        print(df.head())
        
        # print("\nData types:")
        # for col, dtype in df.dtypes.items():
        #     print(f"  - {col}: {dtype}")
        
        # Check for missing values
        missing = df.isna().sum()
        if missing.sum() > 0:
            print("\nMissing values:")
            for col, count in missing.items():
                if count > 0:
                    print(f"  - {col}: {count:,} ({count/len(df):.2%})")
        
        return {
            'df': df,
            'columns': df.columns.tolist(),
            'shape': df.shape
        }
        
    elif file_path.endswith('.nc'):
        # Read the netCDF file
        ds = xr.open_dataset(file_path)
        
        # Print basic information
        print("NetCDF File Information:")
        print("\nDimensions:")
        for dim, size in ds.dims.items():
            print(f"  - {dim}: {size:,}")
        
        print("\nVariables:")
        for var in ds.data_vars:
            dims = ds[var].dims
            shape = ds[var].shape
            dtype = ds[var].dtype
            print(f"  - {var}: {dims} {shape}, dtype: {dtype}")
        
        print("\nCoordinates:")
        for coord in ds.coords:
            print(f"  - {coord}: {ds[coord].values.shape}")
        
        # Print some metadata if available
        if ds.attrs:
            print("\nMetadata:")
            for key, value in ds.attrs.items():
                print(f"  - {key}: {value}")
        
        print("\nDataset Summary:")
        print(ds)
        
        return ds
    
    else:
        print(f"Unsupported file format for file: {file_path}")
        return None
    
def analyze_yearly_ozone_dataset(source="TCR-2", year=2015, deduplicate=False):
    """
    Perform exploratory data analysis on an ozone dataset for a specific year.
    
    Parameters:
    -----------
    source : str
        The data source/model (e.g., 'TCR-2', 'CAMS', etc.)
    year : int
        The year to analyze
    """
    # Get and load the data
    file_path = get_ozone_file(source, year)
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    
    print(f"Analyzing {source} data for {year}")
    print("=" * 80)
    
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # 1. Basic Data Exploration
    print("\n1. BASIC DATA INFORMATION")
    print("-" * 50)
    print(f"Dataset shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Check column names
    print("\nColumns:")
    for col in df.columns:
        print(f"  - {col}")
    
    # Extract the monthly columns (DMA8_1, DMA8_2, etc.)
    month_cols = [col for col in df.columns if col.startswith('DMA8_')]
    
    # Basic statistics
    print("\nBasic Statistics for Monthly Values:")
    monthly_stats = df[month_cols].describe().T
    monthly_stats['missing'] = df[month_cols].isna().sum().values
    monthly_stats['missing_pct'] = (df[month_cols].isna().sum() / len(df) * 100).values
    print(monthly_stats)
    
    # 2. Spatial Coverage Analysis
    print("\n2. SPATIAL COVERAGE")
    print("-" * 50)
    
    # Plot the spatial coverage (scatter plot of lat/lon points)
    plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Add geographic features
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    
    # Plot the data points
    sc = ax.scatter(df['lon'], df['lat'], s=5, c='red', alpha=0.5, 
               transform=ccrs.PlateCarree())
    
    ax.set_global()
    ax.gridlines(draw_labels=True, linewidth=0.5, linestyle='--')
    plt.title(f'Spatial MDA8 Coverage for {source} {year} - of size {df.shape}')
    plt.tight_layout()
    
    filename = f"{source}_{year}_spatial_coverage.png"
    plt.savefig(os.path.join("./figs/spatialMaps", filename), bbox_inches='tight')
    plt.show()
    
    # 3. Monthly Spatial Patterns
    print("\n3. MONTHLY SPATIAL PATTERNS")
    print("-" * 50)
    
    # Create a map for each month
    month_names = ['January', 'February', 'March', 'April', 'May', 'June', 
                  'July', 'August', 'September', 'October', 'November', 'December']
    
    # Find global min and max for consistent colorbar
    vmin = df[month_cols].min().min()
    vmax = df[month_cols].max().max()
    print(f"Global range: {vmin:.2f} - {vmax:.2f} ppb")
    
    # Create a normalized colormap
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    # Create one plot per season
    seasons = {
        'Winter (DJF)': [0, 1, 11],  # Dec, Jan, Feb
        'Spring (MAM)': [2, 3, 4],   # Mar, Apr, May
        'Summer (JJA)': [5, 6, 7],   # Jun, Jul, Aug
        'Fall (SON)': [8, 9, 10]     # Sep, Oct, Nov
    }
    
    # Create seasonal averages
    for season_name, months in seasons.items():
        # Get the corresponding DMA8 columns
        season_cols = [month_cols[i] for i in months]
        
        # Create seasonal average
        df[f'Season_{season_name}'] = df[season_cols].mean(axis=1)
    
    # Plot seasonal patterns
    fig = plt.figure(figsize=(20, 12))  # Reduced height from 15 to 12

    # Create custom GridSpec for more control over spacing
    gs = fig.add_gridspec(2, 2, hspace=0.1, wspace=0.1)  # Reduced spacing between subplots

    # Create axes with the GridSpec
    axes = []
    for i in range(4):
        row = i // 2
        col = i % 2
        ax = fig.add_subplot(gs[row, col], projection=ccrs.PlateCarree())
        axes.append(ax)

    for i, (season, ax) in enumerate(zip(seasons.keys(), axes)):
        # Add geographic features
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
        
        # Plot the data - convert to 2D grid for better visualization
        sc = ax.scatter(df['lon'], df['lat'], c=df[f'Season_{season}'], 
                s=10, alpha=0.7, transform=ccrs.PlateCarree(),
                cmap='plasma', norm=norm)
        
        ax.set_global()
        ax.gridlines(draw_labels=True, linewidth=0.5, linestyle='--')
        ax.set_title(f'{season}')

    # Add a colorbar at the bottom with adjusted position
    cbar_ax = fig.add_axes([0.15, 0.08, 0.7, 0.02])  # Adjusted position (moved up slightly)
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap='plasma'), 
                    cax=cbar_ax, orientation='horizontal')

    cbar.set_label('Surface Ozone MDA8 (ppb)')

    # Add title with adjusted position
    plt.suptitle(f'Seasonal Surface Ozone Patterns for {source} {year}', 
            fontsize=20, y=0.95)  # Adjusted y position

    filename = f"{source}_{year}_seasonal_patterns.png"
    plt.savefig(os.path.join("./figs/spatialMaps", filename), bbox_inches='tight')

    plt.show()

    
    # 4. Regional Analysis
    print("\n4. REGIONAL ANALYSIS")
    print("-" * 50)
    
    # Define some key regions
    regions = {
        'North America': {'min_lon': -130, 'max_lon': -60, 'min_lat': 25, 'max_lat': 60},
        'Europe': {'min_lon': -10, 'max_lon': 40, 'min_lat': 35, 'max_lat': 70},
        'East Asia': {'min_lon': 100, 'max_lon': 145, 'min_lat': 20, 'max_lat': 50},
        'South Asia': {'min_lon': 60, 'max_lon': 100, 'min_lat': 5, 'max_lat': 35},
        'South America': {'min_lon': -80, 'max_lon': -30, 'min_lat': -60, 'max_lat': 15},
        'Africa': {'min_lon': -20, 'max_lon': 50, 'min_lat': -35, 'max_lat': 40},
        'Australia': {'min_lon': 110, 'max_lon': 160, 'min_lat': -45, 'max_lat': -10}
    }

    # Calculate regional statistics for each month
    regional_stats = {}
    
    for region_name, bounds in regions.items():
        # Filter data for this region
        region_mask = ((df['lon'] >= bounds['min_lon']) & 
                      (df['lon'] <= bounds['max_lon']) & 
                      (df['lat'] >= bounds['min_lat']) & 
                      (df['lat'] <= bounds['max_lat']))
        
        region_df = df[region_mask]
        
        if len(region_df) == 0:
            print(f"No data points found in {region_name}")
            continue
        
        # Calculate monthly statistics
        stats = {}
        for i, month in enumerate(month_cols):
            month_data = region_df[month].dropna()
            if len(month_data) > 0:
                stats[month_names[i]] = {
                    'mean': month_data.mean(),
                    'median': month_data.median(),
                    'min': month_data.min(),
                    'max': month_data.max(),
                    'std': month_data.std(),
                    'count': len(month_data)
                }
        
        regional_stats[region_name] = stats
    
    # Plot monthly trends by region
    plt.figure(figsize=(15, 8))
    
    for region_name, stats in regional_stats.items():
        if not stats:
            continue
            
        months = list(stats.keys())
        means = [stats[m]['mean'] for m in months]
        
        plt.plot(months, means, 'o-', label=region_name, linewidth=2, markersize=8)
    
    plt.xlabel('Month')
    plt.ylabel('Mean Surface Ozone MDA8 (ppb)')
    plt.title(f'Monthly Regional Ozone Trends for {source} {year}')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.xticks(rotation=45)
    filename = f"{source}_{year}_regional_trends.png"
    plt.savefig(os.path.join("./figs/temporalTrends", filename), bbox_inches='tight')
    plt.show()
    
    # Print regional stats
    for region_name, stats in regional_stats.items():
        print(f"\n{region_name} Statistics:")
        print("-" * 30)
        
        if not stats:
            print("No data available")
            continue
            
        for month, values in stats.items():
            print(f"{month}: Mean = {values['mean']:.2f} ppb, "
                  f"Max = {values['max']:.2f} ppb, "
                  f"Min = {values['min']:.2f} ppb")
    
    # 5. Histogram of global distribution
    print("\n5. GLOBAL DISTRIBUTION")
    print("-" * 50)
    
    # Calculate annual average
    df['Annual_Mean'] = df[month_cols].mean(axis=1)
    
    plt.figure(figsize=(12, 8))
    plt.hist(df['Annual_Mean'].dropna(), bins=50, color='skyblue', edgecolor='black')
    plt.xlabel('Annual Mean Surface Ozone MDA8 (ppb)')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Annual Mean Surface Ozone - {source} {year}')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    # Print percentiles
    percentiles = [5, 25, 50, 75, 95]
    perc_values = np.percentile(df['Annual_Mean'].dropna(), percentiles)
    
    print("Annual Mean Percentiles:")
    for p, v in zip(percentiles, perc_values):
        print(f"{p}th percentile: {v:.2f} ppb")
    
    return df

def analyze_ozone_data(source="TCR-2", years=None):
    """
    Perform exploratory data analysis on an ozone dataset from netCDF files for specified years.
    
    Parameters:
    -----------
    source : str
        The data source/model (e.g., 'TCR-2', 'CAMS', etc.)
    years : int, list, or None
        The year(s) to analyze. If None, analyze all available years.
    
    Returns:
    --------
    xarray.Dataset
        The filtered dataset for further analysis
    """
    # Define the path to the netCDF file
    netcdf_dir = "./data/netcdf_combined"
    file_path = os.path.join(netcdf_dir, f"{source}_MDA8_combined.nc")
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    
    # Load the dataset
    ds = xr.open_dataset(file_path)
    
    # Filter for specific years if requested
    if years is not None:
        if isinstance(years, int):
            years = [years]
        
        # Filter the dataset to include only the specified years
        ds = ds.sel(time=ds.time.dt.year.isin(years))
        
        if len(ds.time) == 0:
            print(f"No data found for the specified year(s): {years}")
            return None
    
    # Get the actual years in the dataset
    actual_years = np.unique(ds.time.dt.year.values)
    year_range = f"{min(actual_years)}-{max(actual_years)}"
    
    print(f"Analyzing {source} data for years: {year_range}")
    print("=" * 80)
    
    # 1. Basic Data Exploration
    print("\n1. BASIC DATA INFORMATION")
    print("-" * 50)
    print(f"Dataset dimensions: {ds.dims}")
    print(f"Dataset coordinates: {list(ds.coords)}")
    print(f"Dataset variables: {list(ds.data_vars)}")
    
    # Basic statistics for the MDA8 data
    print("\nBasic Statistics for MDA8 Values:")
    mda8_stats = {
        'mean': float(ds.mda8.mean().values),
        'std': float(ds.mda8.std().values),
        'min': float(ds.mda8.min().values),
        'max': float(ds.mda8.max().values),
        'missing': int(ds.mda8.isnull().sum().values),
        'missing_pct': float(ds.mda8.isnull().sum().values / ds.mda8.size * 100)
    }
    
    print(f"Mean: {mda8_stats['mean']:.2f} ppb")
    print(f"Std Dev: {mda8_stats['std']:.2f} ppb")
    print(f"Min: {mda8_stats['min']:.2f} ppb")
    print(f"Max: {mda8_stats['max']:.2f} ppb")
    print(f"Missing values: {mda8_stats['missing']:,} ({mda8_stats['missing_pct']:.2f}%)")
    
    # 2. Spatial Coverage Analysis
    print("\n2. SPATIAL COVERAGE")
    print("-" * 50)
    
    # Calculate the mean across all times to get a representative spatial pattern
    mean_spatial = ds.mda8.mean(dim='time')
    
    # Plot the spatial coverage
    plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Add geographic features
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    
    # Plot the data using xarray's plotting capability
    sc = mean_spatial.plot(ax=ax, transform=ccrs.PlateCarree(), 
                          cmap='plasma', add_colorbar=False)
    
    cbar = plt.colorbar(sc, orientation='horizontal', pad=0.05, aspect=30)
    cbar.set_label('Mean Surface Ozone MDA8 (ppb)')
    
    ax.set_global()
    ax.gridlines(draw_labels=True, linewidth=0.5, linestyle='--')
    plt.title(f'Mean Spatial MDA8 Coverage for {source} {year_range}')
    plt.tight_layout()
    
    # Create directories if they don't exist
    os.makedirs("./figs/spatialMaps", exist_ok=True)
    filename = f"{source}_{year_range}_spatial_coverage.png"
    plt.savefig(os.path.join("./figs/spatialMaps", filename), bbox_inches='tight')
    plt.show()
    
    # 3. Seasonal Spatial Patterns
    print("\n3. SEASONAL SPATIAL PATTERNS")
    print("-" * 50)
    
    # Define seasons
    seasons = {
        'Winter (DJF)': [12, 1, 2],  # Dec, Jan, Feb
        'Spring (MAM)': [3, 4, 5],   # Mar, Apr, May
        'Summer (JJA)': [6, 7, 8],   # Jun, Jul, Aug
        'Fall (SON)': [9, 10, 11]    # Sep, Oct, Nov
    }
    
    # Create seasonal averages
    seasonal_data = {}
    for season_name, months in seasons.items():
        # Filter the dataset for the specific months
        season_ds = ds.sel(time=ds.time.dt.month.isin(months))
        
        if len(season_ds.time) > 0:
            seasonal_data[season_name] = season_ds.mda8.mean(dim='time')
    
    # Find global min and max for consistent colorbar
    seasonal_values = [data.values for data in seasonal_data.values()]
    vmin = min(data.min() for data in seasonal_values if not np.isnan(data.min()))
    vmax = max(data.max() for data in seasonal_values if not np.isnan(data.max()))
    print(f"Global seasonal range: {vmin:.2f} - {vmax:.2f} ppb")
    
    # Create a normalized colormap
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    # Plot seasonal patterns
    fig = plt.figure(figsize=(20, 12))
    
    # Create custom GridSpec for more control over spacing
    gs = fig.add_gridspec(2, 2, hspace=0.1, wspace=0.1)
    
    # Create axes with the GridSpec
    axes = []
    for i in range(4):
        row = i // 2
        col = i % 2
        ax = fig.add_subplot(gs[row, col], projection=ccrs.PlateCarree())
        axes.append(ax)
    
    for i, (season, ax) in enumerate(zip(seasons.keys(), axes)):
        if season not in seasonal_data:
            ax.text(0.5, 0.5, f"No data available for {season}", 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=12)
            continue
            
        # Add geographic features
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
        
        # Plot the data using xarray's plotting capability
        seasonal_data[season].plot(ax=ax, transform=ccrs.PlateCarree(),
                                  cmap='plasma', norm=norm, add_colorbar=False)
        
        ax.set_global()
        ax.gridlines(draw_labels=True, linewidth=0.5, linestyle='--')
        ax.set_title(f'{season}')
    
    # Add a colorbar at the bottom
    cbar_ax = fig.add_axes([0.15, 0.08, 0.7, 0.02])
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap='plasma'), 
                      cax=cbar_ax, orientation='horizontal')
    
    cbar.set_label('Surface Ozone MDA8 (ppb)')
    
    plt.suptitle(f'Seasonal Surface Ozone Patterns for {source} {year_range}', 
               fontsize=20, y=0.95)
    
    filename = f"{source}_{year_range}_seasonal_patterns.png"
    plt.savefig(os.path.join("./figs/spatialMaps", filename), bbox_inches='tight')
    plt.show()
    
    # 4. Regional Analysis
    print("\n4. REGIONAL ANALYSIS")
    print("-" * 50)
    
    # Define key regions
    regions = {
        'North America': {'min_lon': -130, 'max_lon': -60, 'min_lat': 25, 'max_lat': 60},
        'Europe': {'min_lon': -10, 'max_lon': 40, 'min_lat': 35, 'max_lat': 70},
        'East Asia': {'min_lon': 100, 'max_lon': 145, 'min_lat': 20, 'max_lat': 50},
        'South Asia': {'min_lon': 60, 'max_lon': 100, 'min_lat': 5, 'max_lat': 35},
        'South America': {'min_lon': -80, 'max_lon': -30, 'min_lat': -60, 'max_lat': 15},
        'Africa': {'min_lon': -20, 'max_lon': 50, 'min_lat': -35, 'max_lat': 40},
        'Australia': {'min_lon': 110, 'max_lon': 160, 'min_lat': -45, 'max_lat': -10}
    }
    
    # Calculate regional statistics by month
    regional_monthly_means = {}
    
    for region_name, bounds in regions.items():
        # Filter data for this region
        region_ds = ds.sel(
            lon=slice(bounds['min_lon'], bounds['max_lon']),
            lat=slice(bounds['min_lat'], bounds['max_lat'])
        )
        
        if region_ds.sizes['lon'] == 0 or region_ds.sizes['lat'] == 0:
            print(f"No data points found in {region_name}")
            continue
        
        # Calculate monthly statistics - group by month and calculate mean
        monthly_mean = region_ds.mda8.groupby('time.month').mean(dim=('lat', 'lon', 'time'))
        
        # Store the results
        regional_monthly_means[region_name] = monthly_mean
    
    # Plot monthly trends by region
    plt.figure(figsize=(15, 8))
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    for region_name, monthly_mean in regional_monthly_means.items():
        if len(monthly_mean) == 0:
            continue
        
        # Extract the data for plotting
        months = monthly_mean.month.values
        means = monthly_mean.values
        
        # Sort by month
        sorted_indices = np.argsort(months)
        sorted_months = [month_names[m-1] for m in months[sorted_indices]]
        sorted_means = means[sorted_indices]
        
        plt.plot(sorted_months, sorted_means, 'o-', label=region_name, linewidth=2, markersize=8)
    
    plt.xlabel('Month')
    plt.ylabel('Mean Surface Ozone MDA8 (ppb)')
    plt.title(f'Monthly Regional Ozone Trends for {source} {year_range}')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.xticks(rotation=45)
    
    os.makedirs("./figs/temporalTrends", exist_ok=True)
    filename = f"{source}_{year_range}_regional_trends.png"
    plt.savefig(os.path.join("./figs/temporalTrends", filename), bbox_inches='tight')
    plt.show()
    
    # Print regional stats
    for region_name, monthly_mean in regional_monthly_means.items():
        print(f"\n{region_name} Monthly Mean Statistics:")
        print("-" * 30)
        
        if len(monthly_mean) == 0:
            print("No data available")
            continue
        
        # Sort by month for display
        months = monthly_mean.month.values
        means = monthly_mean.values
        sorted_indices = np.argsort(months)
        
        for i in sorted_indices:
            m, mean_val = months[i], means[i]
            print(f"{month_names[m-1]}: {mean_val:.2f} ppb")
    
    # 5. Histogram of global distribution
    print("\n5. GLOBAL DISTRIBUTION")
    print("-" * 50)
    
    # Calculate mean across the dataset
    global_mean = ds.mda8.mean(dim='time')
    
    # Flatten the array for histogram
    flat_data = global_mean.values.flatten()
    flat_data = flat_data[~np.isnan(flat_data)]  # Remove NaNs
    
    plt.figure(figsize=(12, 8))
    plt.hist(flat_data, bins=50, color='skyblue', edgecolor='black')
    plt.xlabel('Mean Surface Ozone MDA8 (ppb)')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Mean Surface Ozone - {source} {year_range}')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    filename = f"{source}_{year_range}_distribution.png"
    plt.savefig(os.path.join("./figs/temporalTrends", filename), bbox_inches='tight')
    plt.show()
    
    # Print percentiles
    percentiles = [5, 25, 50, 75, 95]
    perc_values = np.percentile(flat_data, percentiles)
    
    print("Mean MDA8 Percentiles:")
    for p, v in zip(percentiles, perc_values):
        print(f"{p}th percentile: {v:.2f} ppb")
    
    # 6. Time Series Analysis (if multiple years)
    if len(actual_years) > 1:
        print("\n6. TEMPORAL ANALYSIS")
        print("-" * 50)
        
        # Calculate global annual means
        annual_means = ds.mda8.groupby('time.year').mean(dim=('lat', 'lon', 'time'))

        # add a linear trend line with the rate of change
        trend = np.polyfit(annual_means.year.values, annual_means.values, 1)
        trend_line = np.polyval(trend, annual_means.year.values)
        
        # Plot annual trend
        plt.figure(figsize=(15, 8))
        plt.plot(annual_means.year.values, annual_means.values, 'o-', 
                linewidth=2, markersize=10, color='blue')
        plt.plot(annual_means.year.values, trend_line, 'r--', linewidth=2, label='Trend Line')
        # annotate the rate of change
        rate_of_change = trend[0]
        plt.annotate(f'Rate of Change: {rate_of_change:.2f} ppb/year',
                     xy=(0.05, 0.95), xycoords='axes fraction',
                     fontsize=12, color='red', ha='left', va='top')

        plt.xlabel('Year')
        plt.ylabel('Global Mean Surface Ozone MDA8 (ppb)')
        plt.title(f'Annual Global Ozone Trend for {source} {year_range}')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.legend()
        
        filename = f"{source}_{year_range}_annual_trend.png"
        plt.savefig(os.path.join("./figs/temporalTrends", filename), bbox_inches='tight')
        
        plt.show()

    # 7. Regional Time Series Analysis
    print("\n7. REGIONAL TIME SERIES TRENDS")
    print("-" * 50)

    # Create a figure for the regional time series
    plt.figure(figsize=(15, 8))

    # Process each region
    for region_name, bounds in regions.items():
        # Filter data for this region
        region_ds = ds.sel(
            lon=slice(bounds['min_lon'], bounds['max_lon']),
            lat=slice(bounds['min_lat'], bounds['max_lat'])
        )
        
        if region_ds.sizes['lon'] == 0 or region_ds.sizes['lat'] == 0:
            print(f"No data points found in {region_name}")
            continue
        
        # Calculate mean over the spatial dimensions for each time point
        regional_time_series = region_ds.mda8.mean(dim=['lat', 'lon'])
        
        # Plot the time series for this region
        plt.plot(regional_time_series.time, regional_time_series.values, 
                '-', label=region_name, linewidth=2)

    # Format the plot
    plt.xlabel('Time')
    plt.ylabel('Mean Surface Ozone MDA8 (ppb)')
    plt.title(f'Regional Ozone Trends Over Time for {source} {year_range}')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()

    # Similar to section 6, create regional trends plot
    filename = f"{source}_{year_range}_regional_time_series.png"
    plt.savefig(os.path.join("./figs/temporalTrends", filename), bbox_inches='tight')
    plt.show()
    
    return ds