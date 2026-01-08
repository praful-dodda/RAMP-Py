"""
Reformat BME satellite data from long format to wide format for f-RAMP processing.
Input format:  lat, lon, month, surface_ozone_bme (or surface_ozone_bme_tcol)
Output format: lon, lat, DMA8_1, DMA8_2, ..., DMA8_12
"""

import pandas as pd
import os
from pathlib import Path
import glob

# Configuration
INPUT_DIR = "./data/BME corrected satellite data 2005-2022"
OUTPUT_DIR = os.path.join(INPUT_DIR, "reformatted_data")

def detect_ozone_column(df):
    """Detect which ozone column is present in the dataframe."""
    if 'surface_ozone_bme_tcol' in df.columns:
        return 'surface_ozone_bme_tcol'
    elif 'surface_ozone_bme' in df.columns:
        return 'surface_ozone_bme'
    else:
        raise ValueError(f"No recognized ozone column found. Available columns: {df.columns.tolist()}")

def reformat_bme_to_framp(input_file, output_file):
    """
    Convert BME data from long format to wide format for f-RAMP.
    Parameters:
    -----------
    input_file : str
        Path to input CSV file (long format)
    output_file : str
        Path to output CSV file (wide format)
    """
    print(f"\nProcessing: {os.path.basename(input_file)}")

    # Read data
    df = pd.read_csv(input_file)
    print(f"  Loaded {len(df)} rows")
    print(f"  Columns: {df.columns.tolist()}")

    # Detect ozone column
    ozone_col = detect_ozone_column(df)
    print(f"  Using ozone column: {ozone_col}")

    # Ensure required columns exist
    required_cols = ['lat', 'lon', 'month']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Keep only necessary columns
    df_clean = df[['lat', 'lon', 'month', ozone_col]].copy()

    # Remove any rows with missing values
    df_clean = df_clean.dropna()
    print(f"  After dropping NaNs: {len(df_clean)} rows")

    # Ensure month is in valid range (1-12)
    df_clean = df_clean[(df_clean['month'] >= 1) & (df_clean['month'] <= 12)]

    # Pivot from long to wide format
    # Group by lat/lon and pivot months into columns
    df_wide = df_clean.pivot_table(
        index=['lat', 'lon'],
        columns='month',
        values=ozone_col,
        aggfunc='mean'  # Average if there are duplicates
    ).reset_index()

    # Rename month columns to DMA8_1, DMA8_2, ..., DMA8_12
    column_mapping = {i: f'DMA8_{i}' for i in range(1, 13)}
    df_wide = df_wide.rename(columns=column_mapping)

    # Reorder columns: lon, lat, DMA8_1, ..., DMA8_12
    dma8_cols = [f'DMA8_{i}' for i in range(1, 13)]
    existing_dma8_cols = [col for col in dma8_cols if col in df_wide.columns]

    df_wide = df_wide[['lon', 'lat'] + existing_dma8_cols]

    # Fill missing months with NaN (if some months don't have data)
    for month in range(1, 13):
        col_name = f'DMA8_{month}'
        if col_name not in df_wide.columns:
            df_wide[col_name] = pd.NA
            print(f"  Warning: Month {month} has no data, filled with NaN")

    # Ensure column order
    df_wide = df_wide[['lon', 'lat'] + [f'DMA8_{i}' for i in range(1, 13)]]

    # Save to file
    df_wide.to_csv(output_file, index=False)
    print(f"  Saved {len(df_wide)} grid points to: {os.path.basename(output_file)}")
    print(f"  Output columns: {df_wide.columns.tolist()}")

    # Summary statistics
    non_null_counts = df_wide.notna().sum()
    print(f"  Grid points with data per month:")
    for month in range(1, 13):
        count = non_null_counts[f'DMA8_{month}']
        print(f"    Month {month:2d}: {count} points")

    return df_wide

def main():
    """Main function to process all CSV files in the input directory."""

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    # Find all CSV files in input directory
    csv_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))

    if not csv_files:
        print(f"ERROR: No CSV files found in {INPUT_DIR}")
        print("Please verify the directory path and try again.")
        return

    print(f"\nFound {len(csv_files)} CSV files to process")

    # Process each file
    processed_files = []
    failed_files = []

    for input_file in sorted(csv_files):
        try:
            # Generate output filename
            basename = os.path.basename(input_file)
            output_filename = basename.replace('.csv', '_framp_format.csv')
            output_file = os.path.join(OUTPUT_DIR, output_filename)

            # Process file
            df_result = reformat_bme_to_framp(input_file, output_file)
            processed_files.append(output_file)

        except Exception as e:
            print(f"  ERROR processing {basename}: {str(e)}")
            failed_files.append(input_file)

    # Summary
    print("\n" + "="*70)
    print("PROCESSING COMPLETE")
    print("="*70)
    print(f"Successfully processed: {len(processed_files)} files")
    print(f"Failed: {len(failed_files)} files")

    if processed_files:
        print(f"\nOutput files saved to: {OUTPUT_DIR}")
        print("\nProcessed files:")
        for f in processed_files:
            print(f"  - {os.path.basename(f)}")

    if failed_files:
        print("\nFailed files:")
        for f in failed_files:
            print(f"  - {os.path.basename(f)}")

    print("\n" + "="*70)
    print("Next steps:")
    print("1. Verify the output files in the 'reformatted_for_framp' folder")
    print("2. Register your dataset in preprocess.py")
    print("3. Run f-RAMP using: python ramp_analysis_parallel.py")
    print("="*70)

if __name__ == "__main__":
    main()