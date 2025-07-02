# Global Ozone Data Analysis and Modeling

This repository contains tools and scripts for analyzing global ozone datasets, converting data formats, performing exploratory data analysis (EDA), flexible-RAMP correction, and training ensemble models. Below is a summary of the key functions and their purposes.

## Functions Summary

### `evaluate_models.py`
- **`collocate_data_xarray_optimized(toar_df_tagged, model_ds)`**: Optimized collocation using xarray and vectorized operations.
- **`create_collocated_multimodel_dataset(toar_data_path, year, model_sources)`**: Creates a collocated dataset with observations and multiple model predictions.
- **`calculate_performance_metrics(df_group)`**: Computes performance metrics like RMSE, MAE, and correlation for grouped data.
- **`run_analysis(model_sources, years_to_analyze)`**: Executes the full analysis pipeline for specified years.
- **`plot_metric_by_model(stats_df, metric, year, region, season)`**: Bar plot comparing a metric across models for a specific context.
- **`plot_faceted_comparison(stats_df, metric, year)`**: Multi-panel plot comparing a metric across models, regions, and seasons.
- **`plot_bias_vs_rmse(stats_df, year, region)`**: Scatter plot showing the relationship between bias and RMSE.
- **`plot_metric_heatmap(stats_df, metric, year)`**: Heatmap of a metric across models and regions.
- **`generate_single_model_report(model_name, year, collocated_df, all_stats)`**: Generates a visual report for a single model in a given year.

### `preprocess.py`
- **`print_ozone_file_info(source, year, month)`**: Prints basic information about an ozone data file.
- **`analyze_yearly_ozone_dataset(source, year)`**: Performs exploratory data analysis on a yearly ozone dataset.
- **`analyze_ozone_data(source, years)`**: Conducts multi-year ozone data analysis using xarray.

### `csv_to_netcdf_converter.py`
- **`convert_all_sources_direct(output_dir, selected_sources)`**: Converts all available ozone datasets from yearly CSVs to combined NetCDF files.
- **`convert_source_to_netcdf_direct(source, output_dir)`**: Converts a single source's yearly CSV files to a combined NetCDF file.

### `ramp_correction_parallel_v3.py`

This script is used for applying a ramp correction to the dataset. It is designed to identify and correct for linear instrumental drift or artifacts in the time-series data. The script leverages parallel processing to efficiently handle large volumes of data, significantly speeding up the preprocessing workflow.

**Key Features:**
*   Utilizes multiprocessing for enhanced performance.

### `ozone_stacked_ensemble.py`
- **`train_stacked_ensemble(df, feature_columns, aux_numeric_cols, output_dir, model_label, debug_rows)`**: Trains a stacked ensemble model and saves artifacts.
- **`predict_with_ensemble(model_path, df_new, feature_columns, aux_numeric_cols)`**: Loads a fitted ensemble and returns predictions for new data.

## Usage

Refer to the individual scripts for detailed usage instructions and examples. This repository is designed to handle both CSV and NetCDF/xarray formats, making it versatile for various ozone data analysis tasks.