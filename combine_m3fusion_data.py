from preprocess import *

data_source = "M3fusion"

def combine_monthly_to_yearly(data_source, year, output_dir=None):
    """
    Combine monthly M3fusion MDA8 files into a yearly file with 
    variables DMA8_1, DMA8_2, etc. for each month.
    
    Parameters:
    -----------
    data_source : str
        The data source (e.g., "M3fusion")
    year : int
        The year to process
    output_dir : str, optional
        Directory to save the output file. If None, will use current data directory.
    
    Returns:
    --------
    str
        Path to the saved yearly file
    """
    print(f"Processing {data_source} data for year {year}...")
    
    # Initialize an empty list to store monthly data
    monthly_data = []
    
    # Process each month
    for month in range(1, 13):
        try:
            # Get the file path for this month
            file_path = get_ozone_file(data_source, year, month)
            print(f"  Loading month {month}: {os.path.basename(file_path)}")
            
            # Open the dataset
            ds = xr.open_dataset(file_path).load()
            
            # Extract the data as a DataFrame
            if month == 1:
                # For the first month, get the coordinates too
                df = ds.MDA8.to_dataframe().reset_index()
                
                # Rename coordinates to lon and lat
                df = df.rename(columns={'longitude': 'lon', 'latitude': 'lat'})
                
                # Keep only the needed columns
                df = df[['lon', 'lat', 'MDA8']]
                
                # Rename MDA8 to DMA8_1
                df = df.rename(columns={'MDA8': f'DMA8_{month}'})
                
                # This becomes our base dataframe
                combined_df = df
            else:
                # For subsequent months, just get the MDA8 values
                df = ds.MDA8.to_dataframe().reset_index()
                
                # Rename coordinates to lon and lat
                df = df.rename(columns={'longitude': 'lon', 'latitude': 'lat'})
                
                # Rename MDA8 to DMA8_[month]
                df = df.rename(columns={'MDA8': f'DMA8_{month}'})
                
                # Merge with the combined dataframe on coordinates
                combined_df = pd.merge(combined_df, df[['lon', 'lat', f'DMA8_{month}']], 
                                      on=['lon', 'lat'], how='outer')
        
        except Exception as e:
            print(f"  Error processing month {month}: {str(e)}")
            # Create an empty column for this month
            combined_df[f'DMA8_{month}'] = np.nan
    
    # Determine output path
    if output_dir is None:
        # Use the directory: /work/users/p/r/praful/proj/nasa/global_o3/data/M3fusion 1990-2023/yearlyFiles
        output_dir = "/work/users/p/r/praful/proj/nasa/global_o3/data/M3fusion 1990-2023/yearlyFiles"
    
    # Create the directory if it doesn't exist
    # os.makedirs(output_dir, exist_ok=True)
    
    # Create the output filename
    output_file = os.path.join(output_dir, f"{data_source}-monthly-mda8-{year}.csv")
    
    # Save the combined data to CSV
    combined_df.to_csv(output_file, index=False)
    
    print(f"Saved yearly file with {len(combined_df)} grid points to: {output_file}")
    return output_file


# call the function to combine monthly data into a yearly file for all the years from 1990 to 2023
if __name__ == "__main__":
    for year in range(1990, 2024):
        combine_monthly_to_yearly(data_source, year)
        print(f"Finished processing year {year}.")
        print("-" * 40)