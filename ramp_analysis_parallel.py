import pandas as pd
from ramp_correction_parallel import get_overhang_ramp # Import from the new parallel script
# Assuming your other functions are in evaluate_models
from evaluate_models import run_evaluation_analysis, get_ozone_file
import argparse
import os

def main(eachYear, model, num_cores):
    """
    Main function to run RAMP correction analysis.
    """
    version = "v1-parallel"
    
    ramp_data_dir = './ramp_data'
    ramp_plot_dir = f'ramp_plots/{model}_{eachYear}_{version}'

    # Create directories if they don't exist
    os.makedirs(ramp_data_dir, exist_ok=True)
    os.makedirs(ramp_plot_dir, exist_ok=True)
    
    print(f"--- Starting Analysis for Year: {eachYear}, Model: {model} ---")
    
    # 1. Load and Collocate Data
    print("Step 1: Loading model data and collocating with observations...")
    model_df_file = get_ozone_file(model, eachYear)
    if model_df_file is not None:
        model_df_raw = pd.read_csv(model_df_file)
        print(f"Loaded {model} data for {eachYear}.")
    else:
        print(f"Error: Could not find model file for {model} in {eachYear}.")
        return

    # Assuming run_evaluation_analysis returns the collocated dataframe
    # This step might be slow and could be a candidate for future optimization
    _, collocated_df = run_evaluation_analysis([model], [eachYear])
    print("Finished collocating data.")

    # Clean the collocated data to ensure no NaNs are passed to RAMP
    collocated_df.dropna(subset=['model_ozone', 'observed_ozone'], inplace=True)
    print(f"Using {len(collocated_df)} valid collocated data points.")

    # save the collocated data
    collocated_df_path = os.path.join(ramp_data_dir, f"collocated_data_{model}_{eachYear}.parquet")
    collocated_df.to_parquet(collocated_df_path, index=False)
    print(f"Collocated data saved in Parquet format to: {collocated_df_path}")

    print("\n--- QA DIAGNOSTICS ---")

    # Check a "good" month (e.g., January, Month 1) for M3fusion
    print("\n--- Analyzing Month 1 (A 'Good' Month) ---")
    month_1_data = collocated_df[collocated_df['month'] == 1]
    if not month_1_data.empty:
        print("Observed Ozone Distribution (Month 1):")
        print(month_1_data['observed_ozone'].describe())
        print("\nM3fusion Model Ozone Distribution (Month 1):")
        print(month_1_data['model_ozone'].describe())
    else:
        print("No collocated data for Month 1.")

    # Check a "problem" month (e.g., July, Month 7) for M3fusion
    print("\n--- Analyzing Month 7 (A 'Problem' Month) ---")
    month_7_data = collocated_df[collocated_df['month'] == 7]
    if not month_7_data.empty:
        print("Observed Ozone Distribution (Month 7):")
        print(month_7_data['observed_ozone'].describe())
        print("\nM3fusion Model Ozone Distribution (Month 7):")
        print(month_7_data['model_ozone'].describe())
    else:
        print("No collocated data for Month 7.")

    print("\n--- END QA DIAGNOSTICS ---\n")

    
    # 2. Run the RAMP Correction in Parallel
    print("\nStep 2: Running parallel RAMP correction...")
    lambda1_df, lambda2_df, technique_df = get_overhang_ramp(
        collocated_df=collocated_df,
        model_grid_df=model_df_raw,
        results_dir=ramp_plot_dir,
        num_cores=num_cores  # Pass the number of cores
    )

    # 3. Save Results
    print("\nStep 3: Saving results...")
    l1_path = os.path.join(ramp_data_dir, f"lambda1_{model}_{eachYear}_{version}.parquet")
    l2_path = os.path.join(ramp_data_dir, f"lambda2_{model}_{eachYear}_{version}.parquet")
    tech_path = os.path.join(ramp_data_dir, f"technique_{model}_{eachYear}_{version}.parquet")

    # The model_grid_df contains lon/lat, so we can join results to it
    # output_l1 = model_grid_df[['lon', 'lat']].join(lambda1_df)
    # output_l2 = model_grid_df[['lon', 'lat']].join(lambda2_df)
    # output_tech = model_grid_df[['lon', 'lat']].join(technique_df)

    # output_l1.to_csv(l1_path, index=False)
    # output_l2.to_csv(l2_path, index=False)
    # output_tech.to_csv(tech_path, index=False)

    # simply save the lambda1 and lambda2 dataframes
    lambda1_df.to_parquet(l1_path, index=False)
    lambda2_df.to_parquet(l2_path, index=False)
    technique_df.to_parquet(tech_path, index=False)
    print("Results saved successfully in parquet format.")
    print(f"Lambda1 results saved to: {l1_path}")
    print(f"Lambda2 results saved to: {l2_path}")
    print(f"Technique map data saved to: {tech_path}")
    print("\n--- Analysis Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run RAMP analysis for a specific year and model in parallel.')
    parser.add_argument('--year', type=int, required=True, help='Year to analyze')
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--cores', type=int, default=None, help='Number of CPU cores to use. Defaults to all available cores.')
    
    args = parser.parse_args()
    main(args.year, args.model, args.cores)
