## RAMP Evaluation Scripts

Scripts for evaluating RAMP correction results with visualizations and statistical analysis across multiple models and years.

---

## Overview

After running RAMP corrections, these scripts help you:
1. **Generate visual comparisons** - Before/after maps, scatter plots, time series
2. **Calculate statistical metrics** - RMSE, MAE, correlation, R², NMB
3. **Analyze regional performance** - Performance by region, month, season
4. **Batch process** - Evaluate multiple models and years efficiently

---

## Scripts Available

| Script | Purpose | Usage |
|--------|---------|-------|
| `evaluate_ramp_correction.py` | Main evaluation script (single model-year) | Python script with CLI args |
| `ramp_evaluate.sh` | SLURM script for single evaluation | `sbatch ramp_evaluate.sh` |
| `submit_multiple_ramp_evaluations.sh` | Submit parallel evaluation jobs | `./submit_multiple_ramp_evaluations.sh` |
| `run_ramp_evaluations_sequential.sh` | Run evaluations sequentially | `sbatch run_ramp_evaluations_sequential.sh` |

---

## Quick Start

### Option 1: Evaluate Single Model-Year

```bash
# Method 1: Direct Python command
python evaluate_ramp_correction.py --model M3fusion --year 2017

# Method 2: SLURM job
sbatch --export=MODEL="M3fusion",YEAR=2017 ramp_evaluate.sh

# Method 3: Interactive
python evaluate_ramp_correction.py --help
```

### Option 2: Batch Evaluation (Parallel)

```bash
# 1. Edit models and years in submit_multiple_ramp_evaluations.sh
nano submit_multiple_ramp_evaluations.sh

# 2. Submit all jobs
./submit_multiple_ramp_evaluations.sh

# 3. Monitor progress
squeue -u $USER
tail -f logs/eval_M3fusion_2017_*.log
```

### Option 3: Sequential Evaluation

```bash
# Edit models/years in run_ramp_evaluations_sequential.sh
nano run_ramp_evaluations_sequential.sh

# Submit single job that runs all evaluations
sbatch run_ramp_evaluations_sequential.sh
```

---

## Command Line Arguments

### `evaluate_ramp_correction.py`

```bash
python evaluate_ramp_correction.py [OPTIONS]

Options:
  --model MODEL            Model name (e.g., M3fusion, UKML)
  --year YEAR             Year to evaluate (e.g., 2017)
  --version VERSION       RAMP version (default: v3-parallel)
  --ramp-data-dir DIR     RAMP results directory (default: ./ramp_data)
  --output-dir DIR        Output directory for plots (default: ./ramp_evaluation_plots)
  --help                  Show help message
```

### Examples

```bash
# Basic usage
python evaluate_ramp_correction.py --model M3fusion --year 2020

# Custom output directory
python evaluate_ramp_correction.py \
    --model UKML \
    --year 2018 \
    --output-dir /path/to/plots

# Different RAMP version
python evaluate_ramp_correction.py \
    --model GEOS-CF \
    --year 2019 \
    --version v3-serial
```

---

## Prerequisites

**Required files** (automatically created by RAMP analysis):
```
ramp_data/
├── collocated_data_MODEL_YEAR.parquet (or .csv)
├── lambda1_MODEL_YEAR_v3-parallel.parquet (or .csv)
```

**Required data** (from preprocess.py):
- Original model grid data (accessible via `get_ozone_file()`)

---

## Output Structure

After running evaluation, you'll get:

```
ramp_evaluation_plots/
└── MODEL_YEAR_v3-parallel/
    ├── before_after_comparison.png       # Side-by-side maps
    ├── ramp_correction_spatial.png       # Spatial difference map
    ├── scatter_comparison.png            # Scatter plots with stats
    ├── regional_timeseries_faceted.png   # Time series by region
    └── distribution_comparison.png       # KDE distributions
```

### Plot Descriptions

1. **before_after_comparison.png**
   - Side-by-side spatial maps of ozone (before and after RAMP)
   - Shows spatial distribution for a specific month (July by default)

2. **ramp_correction_spatial.png**
   - Map showing the magnitude of RAMP correction
   - Positive values = RAMP increased ozone, negative = decreased

3. **scatter_comparison.png**
   - Two scatter plots: observed vs original, observed vs RAMP-corrected
   - Includes statistics: N, Bias, RMSE, R
   - 1:1 reference line

4. **regional_timeseries_faceted.png**
   - Monthly time series for each region
   - Shows observations, original model, and RAMP-corrected
   - Faceted by region (North America, Europe, East Asia, etc.)

5. **distribution_comparison.png**
   - Kernel density estimation plots
   - Compares distributions of observations, original, and RAMP-corrected

### Statistical Output

Printed to console and log files:

**Overall Performance Metrics:**
```
                Original Model  RAMP-Corrected
RMSE                    12.45            8.23
MAE                     10.12            6.54
ME                       2.34           -0.12
r2                       0.65            0.78
correlation              0.81            0.88
NMB                     15.23            2.14
```

**Regional Performance (RMSE):**
```
                  Original_RMSE  RAMP_RMSE  Improvement
North America            11.2        7.8         3.4
Europe                   13.5        9.1         4.4
East Asia                14.8       10.2         4.6
```

---

## Batch Processing Configuration

### Edit Models and Years

In `submit_multiple_ramp_evaluations.sh` or `run_ramp_evaluations_sequential.sh`:

```bash
# Models to evaluate
MODELS=(
    "M3fusion"
    "UKML"
    "NJML"
    "TCR-2"
    "GEOS-CF"
    "CAMS"
)

# Years to evaluate
YEARS=(
    2015 2016 2017 2018 2019 2020
)
```

### Edit SLURM Configuration

```bash
PARTITION="hov"           # Your partition
MEMORY="50g"              # Memory per job
CPUS_PER_TASK=4          # CPUs per job
TIME_LIMIT="2:00:00"     # Time limit per job
```

---

## Use Cases

### Use Case 1: Quick Check of Single Model

```bash
# Evaluate RAMP results for M3fusion 2020
python evaluate_ramp_correction.py --model M3fusion --year 2020

# Check the plots
ls -lh ramp_evaluation_plots/M3fusion_2020_v3-parallel/
```

### Use Case 2: Evaluate All Years for One Model

```bash
# Edit script to use only M3fusion
nano submit_multiple_ramp_evaluations.sh
# Set: MODELS=("M3fusion")

# Submit jobs
./submit_multiple_ramp_evaluations.sh
```

### Use Case 3: Compare Multiple Models for Same Year

```bash
# Evaluate all models for 2018
for model in M3fusion UKML NJML GEOS-CF; do
    python evaluate_ramp_correction.py --model $model --year 2018
done

# Compare plots
ls -d ramp_evaluation_plots/*_2018_*/
```

### Use Case 4: Generate Publication Figures

```bash
# Run evaluation with high-quality output
python evaluate_ramp_correction.py \
    --model M3fusion \
    --year 2020 \
    --output-dir ./publication_figures

# Plots are saved in DPI=300, publication-ready
```

---

## Monitoring and Troubleshooting

### Monitor Job Progress

```bash
# Check queue status
squeue -u $USER | grep Eval

# Watch specific model
tail -f logs/eval_M3fusion_2017_*.log

# Check for errors
grep -i error logs/eval_*.log

# Count completed jobs
grep "Evaluation complete" logs/eval_*.log | wc -l
```

### Common Issues

#### Issue: "Could not find collocated data"

**Cause:** RAMP analysis not run yet for that model-year

**Solution:**
```bash
# Run RAMP first
python ramp_analysis_parallel.py --year 2017 --model M3fusion --cores 32

# Then run evaluation
python evaluate_ramp_correction.py --model M3fusion --year 2017
```

#### Issue: "Could not find lambda1 data file"

**Cause:** RAMP completed but lambda1 file missing or wrong version

**Solution:**
```bash
# Check what files exist
ls ramp_data/lambda1_M3fusion_2017*

# Specify correct version
python evaluate_ramp_correction.py \
    --model M3fusion \
    --year 2017 \
    --version v3-serial  # if you used serial version
```

#### Issue: Plots look incorrect or empty

**Cause:** Data quality issues or all NaN values

**Solution:**
```python
# Check data quality
import pandas as pd
df = pd.read_parquet('ramp_data/collocated_data_M3fusion_2017.parquet')
print(df.isnull().sum())
print(df['observed_ozone'].describe())
```

---

## Performance

### Typical Runtime

| Configuration | Time per Model-Year |
|---------------|---------------------|
| Small dataset (<10k points) | 2-5 minutes |
| Medium dataset (10k-50k) | 5-15 minutes |
| Large dataset (>50k points) | 15-30 minutes |

### Resource Usage

- **Memory:** ~2-10 GB per job (depends on grid size)
- **CPU:** Mostly single-threaded (matplotlib/plotting)
- **Disk:** ~50-100 MB per model-year (plots)

### Optimization Tips

```bash
# Run multiple models in parallel
for model in M3fusion UKML NJML; do
    sbatch --export=MODEL=$model,YEAR=2017 ramp_evaluate.sh &
done

# Use sequential script for many combinations
# (avoids job submission overhead)
sbatch run_ramp_evaluations_sequential.sh
```

---

## Integration with Summary Scripts

These evaluation scripts complement the summary generation:

```bash
# Step 1: Run RAMP corrections
./submit_multiple_ramp_jobs.sh

# Step 2: Generate performance summaries
python generate_model_performance_summaries.py

# Step 3: Generate visualizations (this step)
./submit_multiple_ramp_evaluations.sh

# Step 4: Analyze results
python analyze_model_summaries.py
```

---

## Customization

### Change Default Month for Maps

Edit `evaluate_ramp_correction.py`:

```python
# Around line 276
month_to_plot = 7  # Change to desired month (1-12)
```

### Add Custom Metrics

Edit the `perform_statistical_evaluation()` function:

```python
def perform_statistical_evaluation(eval_df):
    # Your custom metric
    custom_metric = your_calculation(eval_df)
    print(f"Custom metric: {custom_metric}")

    # ... existing code
```

### Change Plot Styles

Edit plotting sections:

```python
# Use different colormap
plt.scatter(..., cmap='viridis')  # Change colormap

# Adjust figure size
plt.figure(figsize=(14, 10))  # Larger plot

# Different DPI
plt.savefig(..., dpi=600)  # Higher resolution
```

---

## Reference

### Evaluation Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| RMSE | √(mean((model - obs)²)) | Lower is better |
| MAE | mean(\|model - obs\|) | Lower is better |
| ME (Bias) | mean(model - obs) | Close to 0 is better |
| R² | Coefficient of determination | Closer to 1 is better |
| Correlation | Pearson correlation | Closer to 1 is better |
| NMB | (sum(model - obs) / sum(obs)) × 100 | Close to 0 is better |

### Regional Definitions

Regions defined in `evaluate_models.py`:

```python
REGIONS = {
    'North America': {'min_lon': -130, 'max_lon': -60, 'min_lat': 25, 'max_lat': 60},
    'Europe': {'min_lon': -10, 'max_lon': 40, 'min_lat': 35, 'max_lat': 70},
    'East Asia': {'min_lon': 100, 'max_lon': 145, 'min_lat': 20, 'max_lat': 50},
    'South Asia': {'min_lon': 60, 'max_lon': 100, 'min_lat': 5, 'max_lat': 35},
    'South America': {'min_lon': -80, 'max_lon': -30, 'min_lat': -60, 'max_lat': 15},
    'Africa': {'min_lon': -20, 'max_lon': 50, 'min_lat': -35, 'max_lat': 40},
    'Australia': {'min_lon': 110, 'max_lon': 160, 'min_lat': -45, 'max_lat': -10},
}
```

---

## Citation

If you use these evaluation scripts in publications, please cite:

- RAMP methodology paper
- This repository: https://github.com/[your-repo]/RAMP-Py

---

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review log files in `logs/`
3. Verify RAMP data files exist in `ramp_data/`
4. Check that original model data is accessible

---

**Happy Evaluating! 📊**
