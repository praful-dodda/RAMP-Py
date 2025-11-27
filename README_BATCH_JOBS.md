# RAMP Batch Processing Scripts

Three scripts for running RAMP analysis on multiple models and years.

## Scripts Overview

| Script | Purpose | Parallelization |
|--------|---------|-----------------|
| `submit_multiple_ramp_jobs.sh` | Submit separate SLURM jobs for each model-year | **Parallel** (recommended) |
| `run_ramp_sequential.sh` | Run all combinations in one SLURM job | Sequential |
| `run_single_ramp.sh` | Run a single model-year (template) | Single job |

---

## Option 1: Parallel Jobs (Recommended)

**Best for:** Processing many model-year combinations quickly using cluster resources

### Usage:

```bash
# 1. Edit the script to set your models and years
nano submit_multiple_ramp_jobs.sh

# 2. Modify the MODELS and YEARS arrays:
MODELS=(
    "OMI-MLS"
    "UKML"
    "NJML"
)

YEARS=(
    2015
    2016
    2017
)

# 3. Run the submission script
./submit_multiple_ramp_jobs.sh
```

### What it does:
- Submits one SLURM job for each model-year combination
- Jobs run in parallel on different nodes
- Each job gets 32 cores, 50GB memory, 10 hours
- Creates separate log files: `logs/ramp_MODEL_YEAR_JOBID.log`

### Monitor progress:
```bash
# Check queue status
squeue -u $USER

# Watch specific jobs
watch -n 5 'squeue -u $USER'

# Check log files
tail -f logs/ramp_OMI-MLS_2017_*.log
```

---

## Option 2: Sequential Processing

**Best for:** When you have job limits or want one long-running job

### Usage:

```bash
# 1. Edit models and years in the script
nano run_ramp_sequential.sh

# 2. Submit to SLURM
sbatch run_ramp_sequential.sh
```

### What it does:
- Runs all model-year combinations one after another
- Single SLURM job (48 hour limit)
- Easier to track overall progress
- Reports success/failure summary at the end

---

## Option 3: Single Job

**Best for:** Testing or running one specific combination

### Usage:

```bash
# Method 1: Edit defaults in script
nano run_single_ramp.sh
sbatch run_single_ramp.sh

# Method 2: Pass variables at submission
sbatch --export=MODEL="UKML",YEAR=2018,CORES=32 run_single_ramp.sh

# Method 3: Quick one-liner
sbatch --job-name=RAMP_UKML_2018 \
       -p hov -N 1 --mem 50g -n 1 --cpus-per-task=32 -t 10:00:00 \
       --output=logs/ramp_UKML_2018_%j.log \
       --wrap="/work/users/p/r/praful/lib/anaconda/conda/envs/exposure_inequity/bin/python ramp_analysis_parallel.py --year 2018 --model UKML --cores 32"
```

---

## Configuration

### Edit these variables in each script:

```bash
# Models to process
MODELS=(
    "OMI-MLS"
    "UKML"
    "NJML"
    "TCR-2"
    "GEOS-CF"
    "M3fusion"
    "CAMS"
    "CESM2.2"
)

# Years to process
YEARS=(
    2015 2016 2017 2018 2019 2020 2021 2022
)

# SLURM settings
PARTITION="hov"           # Your partition name
MEMORY="50g"              # Memory per job
CPUS_PER_TASK=32         # Cores per job
TIME_LIMIT="10:00:00"    # Time limit per job

# Python path
PYTHON_PATH="/work/users/p/r/praful/lib/anaconda/conda/envs/exposure_inequity/bin/python"
```

---

## Useful Commands

### Managing jobs:

```bash
# View your jobs
squeue -u $USER

# Cancel all your jobs
scancel -u $USER

# Cancel specific job
scancel JOBID

# Cancel jobs by name pattern
scancel --name=RAMP_OMI-MLS_*

# View job details
scontrol show job JOBID

# Check job efficiency after completion
seff JOBID
```

### Monitoring logs:

```bash
# Watch latest log
tail -f logs/ramp_*_$(ls -t logs/ | head -1)

# Count completed jobs
grep "Job completed" logs/*.log | wc -l

# Find failed jobs
grep -l "FAILED\|Error\|exit code: [1-9]" logs/*.log

# Summary of all jobs
for log in logs/ramp_*.log; do
    echo "$log: $(grep -o "Job completed with exit code: [0-9]*" $log)"
done
```

---

## Output Structure

After running, you'll have:

```
ramp_data/
├── collocated_data_OMI-MLS_2017.parquet
├── lambda1_OMI-MLS_2017_v3-parallel.parquet
├── lambda2_OMI-MLS_2017_v3-parallel.parquet
├── technique_OMI-MLS_2017_v3-parallel.parquet
├── collocated_data_UKML_2018.parquet
├── lambda1_UKML_2018_v3-parallel.parquet
└── ...

ramp_plots/
├── OMI-MLS_2017_v3-parallel/
│   ├── global_ramp_month_1.png
│   ├── technique_map_month_1.png
│   └── ...
└── UKML_2018_v3-parallel/
    └── ...

logs/
├── ramp_OMI-MLS_2017_12345.log
├── ramp_UKML_2018_12346.log
└── ...
```

---

## Troubleshooting

### Problem: Jobs not submitting
```bash
# Check partition availability
sinfo -p hov

# Check your account limits
sacctmgr show assoc user=$USER format=user,account,partition,maxjobs
```

### Problem: Jobs failing immediately
```bash
# Check specific log
cat logs/ramp_MODEL_YEAR_JOBID.log

# Common issues:
# - Python path incorrect → Update PYTHON_PATH variable
# - Data files missing → Check preprocess.py source registration
# - Memory issues → Increase --mem in script
```

### Problem: Jobs running too long
```bash
# Check if using all cores
# In log file, look for multiprocessing output

# Reduce time by:
# 1. Verify --cores matches --cpus-per-task
# 2. Check if data is on fast filesystem
# 3. Consider using NetCDF instead of CSV (faster I/O)
```

---

## Examples

### Process all available models for 2017-2020:
```bash
# Edit submit_multiple_ramp_jobs.sh:
MODELS=("OMI-MLS" "UKML" "NJML" "TCR-2" "GEOS-CF" "M3fusion" "CAMS" "CESM2.2")
YEARS=(2017 2018 2019 2020)

# Submit
./submit_multiple_ramp_jobs.sh
# This submits 8 models × 4 years = 32 parallel jobs
```

### Process just your new BME dataset:
```bash
# After reformatting BME data and registering in preprocess.py
sbatch --export=MODEL="BME-Satellite",YEAR=2017 run_single_ramp.sh
```

### Resume failed jobs only:
```bash
# Find failed combinations
grep -l "FAILED\|Error" logs/*.log

# Manually re-submit them
sbatch --export=MODEL="UKML",YEAR=2018 run_single_ramp.sh
```

---

## Performance Tips

1. **Parallel jobs** typically faster overall (if cluster has capacity)
2. **Sequential job** more reliable for large batches (avoids job limits)
3. Use **32 cores** if available (good balance of speed and resource usage)
4. **NetCDF format** faster than CSV for large datasets
5. **Local/scratch storage** faster than network filesystems

---

## Next Steps After Processing

1. **Validate results:**
   ```bash
   python evaluate_ramp_correction.py
   ```

2. **Compare models:**
   ```bash
   python compare_serial_parallel_ramp.py
   ```

3. **Create visualizations:**
   - Check `ramp_plots/` directories for diagnostic plots
   - Use `evaluate_ramp_correction.py` for performance metrics

4. **Apply corrections:**
   - Use lambda1/lambda2 dataframes to correct new model outputs
   - See `evaluate_ramp_correction.py` for application examples
