## Model Performance Summary Generation for Data Fusion

This guide explains how to generate comprehensive performance summaries for RAMP-corrected models to aid in data fusion model selection.

---

## Overview

After running RAMP corrections on multiple models and years, these scripts help you:

1. **Generate detailed performance summaries** - Before/after RAMP metrics by region, month, season, station type
2. **Analyze and compare models** - Identify best models for each year/region
3. **Create visualizations** - Performance trends, comparisons, and improvements
4. **Support model selection** - Data-driven recommendations for data fusion

---

## Prerequisites

**Required:** You must have already run RAMP analysis using `ramp_analysis_parallel.py` for your models/years.

**Expected directory structure:**
```
./ramp_data/
├── collocated_data_MODEL_YEAR.parquet (or .csv)
├── lambda1_MODEL_YEAR_v3-parallel.parquet (or .csv)
├── lambda2_MODEL_YEAR_v3-parallel.parquet
└── technique_MODEL_YEAR_v3-parallel.parquet
```

---

## Quick Start

### Option 1: Run Everything (Recommended)

```bash
# Generate all summaries and analysis
./run_model_summary_analysis.sh
```

This script will:
1. Generate performance summaries for all available model-year combinations
2. Create combined summaries per model
3. Generate master summary across all models
4. Analyze and visualize results
5. Create model selection recommendations

### Option 2: Step-by-Step

```bash
# Step 1: Generate summaries
python generate_model_performance_summaries.py

# Step 2: Analyze results
python analyze_model_summaries.py
```

---

## Script Details

### 1. `generate_model_performance_summaries.py`

**Purpose:** Generate comprehensive CSV summaries for each model-year combination

**What it does:**
- Automatically scans `ramp_data/` for available RAMP results
- Applies RAMP corrections to collocated data
- Calculates before/after metrics for:
  - Overall performance
  - Performance by region (North America, Europe, East Asia, etc.)
  - Performance by month (1-12)
  - Performance by season (Winter, Spring, Summer, Autumn)
  - Performance by station type (rural, urban, suburban)
  - Combined region-month breakdown
- Creates combined summaries per model (all years)
- Generates master summary file (all models, all years)

**Performance metrics calculated:**
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- ME (Mean Error / Bias)
- R² (Coefficient of Determination)
- Correlation
- NMB (Normalized Mean Bias)
- RMSE/MAE Improvement (%)
- Number of data points

**Output structure:**
```
model_summaries/
├── MODEL_NAME/
│   ├── MODEL_YEAR_overall_performance.csv
│   ├── MODEL_YEAR_performance_by_region.csv
│   ├── MODEL_YEAR_performance_by_month.csv
│   ├── MODEL_YEAR_performance_by_season.csv
│   ├── MODEL_YEAR_performance_by_station_type.csv
│   ├── MODEL_YEAR_performance_by_region_month.csv
│   └── MODEL_combined_all_years_summary.csv
├── ALL_MODELS_master_summary.csv
└── ALL_MODELS_average_by_model.csv
```

**Runtime:** ~5-30 seconds per model-year (depends on data size)

---

### 2. `analyze_model_summaries.py`

**Purpose:** Analyze summaries and generate model selection recommendations

**What it does:**
- Loads master summary file
- Ranks models by various metrics (RMSE, R², improvement)
- Finds best model per year using composite scoring
- Analyzes model availability across years
- Creates comparison visualizations
- Generates selection recommendations

**Outputs:**

**CSV Files:**
- `best_model_per_year.csv` - Recommended model for each year (composite score)
- `model_selection_recommendations.csv` - Best models by different criteria
- `model_rankings_by_RMSE.csv` - Top 5 models per year (by RMSE)
- `model_rankings_by_r2.csv` - Top 5 models per year (by R²)
- `model_summary_statistics.csv` - Overall statistics per model
- `model_availability_by_year.csv` - Which models are available when

**Visualizations:**
- `model_performance_trends.png` - Time series of metrics for all models
- `model_comparison_heatmap.png` - Year-model RMSE heatmap
- `rmse_improvement_comparison.png` - Average RMSE improvement by model

**Composite Scoring System:**
The "best model per year" uses weighted scoring:
- RMSE (after RAMP): 30%
- RMSE improvement: 30%
- R² (after RAMP): 20%
- Correlation (after RAMP): 20%

---

## Use Cases

### Use Case 1: Model Selection for Specific Year

**Goal:** Find the best model(s) to use for 2018

```bash
# Generate summaries
python generate_model_performance_summaries.py

# Analyze
python analyze_model_summaries.py

# Check recommendations
cat model_analysis/best_model_per_year.csv | grep 2018
```

**Example output:**
```csv
year,best_model,composite_score,after_RMSE,after_r2,RMSE_improvement_pct,n_points
2018,M3fusion,0.856,8.23,0.72,18.5,15234
```

### Use Case 2: Regional Model Performance

**Goal:** Find which model performs best in Europe for 2020

```bash
# Look at region-specific file
cat model_summaries/M3fusion/M3fusion_2020_performance_by_region.csv | grep Europe
cat model_summaries/UKML/UKML_2020_performance_by_region.csv | grep Europe
# Compare after_RMSE values
```

### Use Case 3: Seasonal Analysis

**Goal:** Identify models with poor summer performance

```bash
# Check seasonal files
for model in model_summaries/*/; do
    echo "=== $(basename $model) ==="
    cat ${model}*_2020_performance_by_season.csv | grep Summer
done
```

### Use Case 4: Multi-Year Ensemble

**Goal:** Select top 3 models for 2015-2020 ensemble

```bash
# Check rankings
cat model_analysis/model_rankings_by_RMSE.csv | awk -F, '$5 >= 2015 && $5 <= 2020 && $3 <= 3' | cut -d, -f1 | sort | uniq -c | sort -rn | head -3
```

### Use Case 5: Track Model Improvements

**Goal:** See which models improved most with RAMP

```bash
# Check improvement rankings
cat model_summaries/ALL_MODELS_master_summary.csv | sort -t, -k20 -rn | head -10
# Column 20 is RMSE_improvement_pct
```

---

## Example Workflow: Data Fusion for 2017-2020

```bash
#!/bin/bash
# Example: Select best models for data fusion 2017-2020

# 1. Generate all summaries
python generate_model_performance_summaries.py

# 2. Analyze
python analyze_model_summaries.py

# 3. Extract best models for our period
echo "Best models by year (2017-2020):"
cat model_analysis/best_model_per_year.csv | awk -F, 'NR==1 || ($1 >= 2017 && $1 <= 2020)'

# 4. Check availability
echo "\nModel availability (2017-2020):"
cat model_analysis/model_availability_by_year.csv | head -1
cat model_analysis/model_availability_by_year.csv | grep -E "M3fusion|UKML|GEOS-CF"

# 5. Get detailed stats for top 3 models
echo "\nDetailed statistics for top models:"
cat model_analysis/model_summary_statistics.csv | head -4

# 6. Review visualizations
echo "\nCheck these plots:"
ls -lh model_analysis/*.png
```

**Sample output interpretation:**

| Year | Best Model | RMSE (After) | R² (After) | Improvement |
|------|-----------|--------------|------------|-------------|
| 2017 | M3fusion  | 8.45        | 0.68       | 16.2%       |
| 2018 | M3fusion  | 8.23        | 0.72       | 18.5%       |
| 2019 | GEOS-CF   | 7.89        | 0.74       | 19.1%       |
| 2020 | UKML      | 8.01        | 0.71       | 17.8%       |

**Decision:** Use M3fusion for 2017-2018, GEOS-CF for 2019, UKML for 2020

---

## Advanced Usage

### Custom Analysis: Filter by Region

```python
import pandas as pd

# Load master summary
master = pd.read_csv('model_summaries/ALL_MODELS_master_summary.csv')

# Load region-specific data
regions = {}
for year in range(2015, 2021):
    for model in ['M3fusion', 'UKML', 'GEOS-CF']:
        file = f'model_summaries/{model}/{model}_{year}_performance_by_region.csv'
        try:
            df = pd.read_csv(file)
            regions[f'{model}_{year}'] = df
        except:
            pass

# Filter East Asia performance
east_asia = pd.concat([
    df[df['region'] == 'East Asia']
    for df in regions.values() if 'region' in df.columns
], ignore_index=True)

print(east_asia.sort_values('after_RMSE').head(10))
```

### Custom Visualization: Monthly Performance

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load monthly performance for a specific model
df = pd.read_csv('model_summaries/M3fusion/M3fusion_2020_performance_by_month.csv')

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: RMSE by month
axes[0,0].plot(df['month'], df['before_RMSE'], 'o-', label='Before RAMP')
axes[0,0].plot(df['month'], df['after_RMSE'], 'o-', label='After RAMP')
axes[0,0].set_xlabel('Month')
axes[0,0].set_ylabel('RMSE')
axes[0,0].legend()
axes[0,0].set_title('RMSE by Month')

# Plot 2: R² by month
axes[0,1].plot(df['month'], df['before_r2'], 'o-', label='Before RAMP')
axes[0,1].plot(df['month'], df['after_r2'], 'o-', label='After RAMP')
axes[0,1].set_xlabel('Month')
axes[0,1].set_ylabel('R²')
axes[0,1].legend()
axes[0,1].set_title('R² by Month')

# Plot 3: Improvement %
axes[1,0].bar(df['month'], df['RMSE_improvement_pct'])
axes[1,0].set_xlabel('Month')
axes[1,0].set_ylabel('RMSE Improvement (%)')
axes[1,0].set_title('RAMP Improvement by Month')

# Plot 4: Sample size
axes[1,1].bar(df['month'], df['n_points'])
axes[1,1].set_xlabel('Month')
axes[1,1].set_ylabel('Number of Observations')
axes[1,1].set_title('Data Points by Month')

plt.tight_layout()
plt.savefig('custom_monthly_analysis.png', dpi=300)
plt.close()
```

---

## Interpreting Results

### Key Metrics Explained

| Metric | Meaning | Good Value | Bad Value |
|--------|---------|------------|-----------|
| **RMSE** | Average prediction error | < 10 ppb | > 15 ppb |
| **MAE** | Average absolute error | < 8 ppb | > 12 ppb |
| **R²** | Variance explained | > 0.7 | < 0.5 |
| **Correlation** | Linear relationship strength | > 0.8 | < 0.6 |
| **NMB** | Normalized bias | ±10% | > ±30% |
| **Improvement %** | RAMP effectiveness | > 15% | < 5% |

### What to Look For

**Good Model Characteristics:**
- ✅ Low RMSE/MAE (< 10 ppb after RAMP)
- ✅ High R² (> 0.7)
- ✅ Strong correlation (> 0.8)
- ✅ Significant RAMP improvement (> 15%)
- ✅ Consistent performance across regions
- ✅ Large sample size (n_points > 5000)

**Red Flags:**
- ❌ RMSE increases after RAMP (negative improvement)
- ❌ Very low R² (< 0.5)
- ❌ Highly variable performance across months
- ❌ Poor performance in key regions
- ❌ Small sample size (< 1000 points)

---

## Troubleshooting

### Issue: "No RAMP results found"

**Solution:**
```bash
# Check if RAMP data exists
ls -lh ramp_data/

# If empty, run RAMP analysis first
python ramp_analysis_parallel.py --year 2017 --model M3fusion --cores 32
```

### Issue: "Could not find collocated data file"

**Solution:**
```bash
# RAMP analysis should create these files automatically
# Check if they exist:
find ramp_data -name "collocated_data_*"

# If missing, re-run RAMP analysis
```

### Issue: Missing region/season columns

**Cause:** Older collocated data might not have these columns

**Solution:**
```bash
# Re-run the evaluation/collocation step
python -c "from evaluate_models import run_evaluation_analysis; \
           run_evaluation_analysis(['M3fusion'], [2017])"
```

### Issue: Performance metrics are NaN

**Cause:** Insufficient data points or all observations are NaN

**Solution:**
```python
import pandas as pd

# Check data quality
df = pd.read_parquet('ramp_data/collocated_data_M3fusion_2017.parquet')
print(df.isnull().sum())
print(f"Valid observations: {df['observed_ozone'].notna().sum()}")
```

---

## Performance Considerations

### Memory Usage

- Each model-year uses ~50-200 MB RAM during processing
- Total memory scales with number of observations
- For very large datasets (>1M points), process models sequentially

### Processing Time

Typical runtimes on a standard workstation:

| Task | Single Model-Year | All Models (50 combinations) |
|------|------------------|------------------------------|
| Generate summaries | 10-30 sec | 10-20 min |
| Analysis | 5-10 sec | 30-60 sec |
| Visualizations | 10-20 sec | 30-60 sec |

### Optimization Tips

```bash
# Process only specific models
python -c "from generate_model_performance_summaries import *; \
           process_single_model_year('M3fusion', 2020)"

# Run in parallel (if you have multiple cores)
# Edit generate_model_performance_summaries.py to use multiprocessing
```

---

## Output File Descriptions

### Per Model-Year Files

| File | Contents | Use Case |
|------|----------|----------|
| `*_overall_performance.csv` | Single row with all metrics | Quick model comparison |
| `*_by_region.csv` | Metrics per geographic region | Regional model selection |
| `*_by_month.csv` | Metrics per calendar month | Seasonal analysis |
| `*_by_season.csv` | Metrics per season | Seasonal model selection |
| `*_by_station_type.csv` | Metrics per station type | Urban vs rural performance |
| `*_by_region_month.csv` | Combined breakdown | Detailed spatiotemporal analysis |

### Combined Files

| File | Contents | Use Case |
|------|----------|----------|
| `*_combined_all_years_summary.csv` | All years for one model | Model evolution over time |
| `ALL_MODELS_master_summary.csv` | All models, all years | Overall comparison |
| `ALL_MODELS_average_by_model.csv` | Averaged statistics per model | Model ranking |

### Analysis Outputs

| File | Contents | Use Case |
|------|----------|----------|
| `best_model_per_year.csv` | Recommended model per year | Quick decision making |
| `model_selection_recommendations.csv` | Multiple criteria | Comprehensive selection |
| `model_rankings_by_*.csv` | Top 5 models by metric | Metric-specific selection |
| `model_summary_statistics.csv` | Aggregate stats | Overview and comparison |

---

## Citation and Methodology

If you use these summaries in publications, please cite the RAMP methodology and note:

- Performance metrics follow standard atmospheric model evaluation practices
- Before/after comparison uses identical observational datasets
- Regional boundaries follow standard geographic definitions
- Composite scores use equal weighting unless specified
- All statistics calculated on collocated observation-model pairs

---

## Support and Customization

### Adding Custom Metrics

Edit `generate_model_performance_summaries.py`:

```python
def calculate_metrics_before_after(df_group):
    # Add your custom metric here
    metrics_before = calculate_performance_metrics(...)

    # Add new metric
    result['custom_metric'] = your_calculation(df_group)

    return pd.Series(result)
```

### Changing Composite Scoring

Edit `analyze_model_summaries.py`:

```python
# Modify weights in find_best_model_per_year()
year_data['composite_score'] = (
    0.40 * year_data['rmse_score'] +      # Increase RMSE weight
    0.30 * year_data['imp_score'] +
    0.30 * year_data['r2_score']          # Increase R² weight
)
```

### Adding New Regions

Edit `evaluate_models.py`:

```python
REGIONS = {
    'Your Region': {
        'min_lon': -100,
        'max_lon': -80,
        'min_lat': 30,
        'max_lat': 45
    },
    # ... existing regions
}
```

---

## Quick Reference

```bash
# Generate all summaries
python generate_model_performance_summaries.py

# Analyze results
python analyze_model_summaries.py

# View best models
cat model_analysis/best_model_per_year.csv

# Check specific model
cat model_summaries/M3fusion/M3fusion_combined_all_years_summary.csv

# Open visualizations
open model_analysis/*.png   # macOS
xdg-open model_analysis/*.png   # Linux
```

---

**Happy Model Selection! 🚀**
