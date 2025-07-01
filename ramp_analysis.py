from evaluate_models import *
from ramp_correction import *
import argparse

def main(eachYear, model):
    version = "v1"
    
    ramp_data_dir = './ramp_data'
    ramp_plot_dir = f'ramp_plots/{model}_{eachYear}_{version}'
    
    # Get the model data for MERRA2-GMI
    model_df_file = get_ozone_file(model, eachYear)
    if model_df_file is not None:
        model_df_raw = pd.read_csv(model_df_file)
        print(f"{model} Data for in {eachYear}:")
        print(model_df_raw.head())
        
        # print columns
        print(f"Columns in {model} data: {model_df_raw.columns.tolist()}")

    # Run the evaluation analysis for the specified model and year
    [eval_df, collocated_df] = run_evaluation_analysis([model], [eachYear])

    # Clean the collocated data to ensure no NaNs are passed to RAMP
    collocated_df.dropna(subset=['model_ozone', 'observed_ozone'], inplace=True)
    print(f"Using {len(collocated_df)} valid collocated data points.")

    # save collocated data to a Parquet file
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

    # 2. Run the RAMP Correction
    lambda1_df, lambda2_df, technique_df = get_overhang_ramp(
        collocated_df=collocated_df,
        model_grid_df=model_df_raw,
        results_dir=ramp_plot_dir
    )

    # 3. Display Results
    print("\n--- Results ---")
    print("\nLambda1 (Mean Correction) DataFrame Head:")
    print(lambda1_df.head())

    print("\nLambda2 (Variance Correction) DataFrame Head:")
    print(lambda2_df.head())

    # save the results to parquet files
    lambda1_df.to_parquet(f"{ramp_data_dir}/lambda1_{model}_{eachYear}.parquet", index=False)
    lambda2_df.to_parquet(f"{ramp_data_dir}/lambda2_{model}_{eachYear}.parquet", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run RAMP analysis for a specific year and model')
    parser.add_argument('--year', type=int, required=True, help='Year to analyze')
    parser.add_argument('--model', type=str, required=True, help='Model name')
    
    args = parser.parse_args()
    main(args.year, args.model)
