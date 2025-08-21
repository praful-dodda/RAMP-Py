import pandas as pd
import numpy as np
import os
import glob

# --- Configuration ---
# Directory containing the input CSV files (long format).
INPUT_DIR = './data/UK Cambridge ML/popwt(1118)/popwt(1118)'
# Directory where the output CSV files (wide format) will be saved.
OUTPUT_DIR = './data/UK Cambridge ML/yearlyFiles'

def reshape_yearly_csvs(input_directory, output_directory):
    """
    Reads long-format yearly CSVs, reshapes them to a wide format using
    the 'MDA8h_pop' column, and saves them to a new directory.

    Input Columns: longitude, latitude, month, MDA8h_pop, etc.
    Output Columns: lon, lat, DMA8_1, DMA8_2, ..., DMA8_12

    Args:
        input_directory (str): The path to the directory with input CSVs.
        output_directory (str): The directory to save the output CSVs.
    """
    # --- 1. Check for input files and create dummy files if needed ---
    input_files = glob.glob(os.path.join(input_directory, '*.csv'))
    if not input_files:
        print(f"No CSV files found in {input_directory}. Please check the path.")

    # --- 2. Create output directory ---
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Created output directory: {output_directory}")

    # --- 3. Process each input file ---
    print(f"\nFound {len(input_files)} CSV files to process.")
    for filepath in input_files:
        filename = os.path.basename(filepath)
        print(f"Processing '{filename}'...")

        # --- a. Read the source CSV ---
        df = pd.read_csv(filepath)

        # --- b. Check for required columns ---
        required_cols = ['longitude', 'latitude', 'month', 'MDA8h_pop']
        if not all(col in df.columns for col in required_cols):
            print(f"  -> Skipping '{filename}': Missing one of the required columns {required_cols}")
            continue

        # --- c. Pivot the DataFrame ---
        # This is the core operation. We set 'longitude' and 'latitude' as the
        # stable index, use 'month' to create new columns, and fill the
        # values with 'MDA8h_pop'.
        pivot_df = df.pivot_table(
            index=['longitude', 'latitude'],
            columns='month',
            values='MDA8h_pop'
        ).reset_index()

        # --- d. Rename columns to the desired format ---
        # First, rename the location columns
        pivot_df = pivot_df.rename(columns={'longitude': 'lon', 'latitude': 'lat'})

        # Second, rename the pivoted month columns (e.g., 1 -> DMA8_1)
        rename_map = {month: f'DMA8_{month}' for month in range(1, 13)}
        pivot_df = pivot_df.rename(columns=rename_map)

        # --- e. Save the transformed DataFrame to a new CSV ---
        output_filename = os.path.join(output_directory, f'reshaped_{filename}')
        pivot_df.to_csv(output_filename, index=False, na_rep='NaN')
        print(f"  -> Successfully created: {output_filename}")

    print("\nProcessing complete.")


reshape_yearly_csvs(INPUT_DIR, OUTPUT_DIR)