#!/bin/bash
#SBATCH --job-name=RAMP-Sequential-Batch
#SBATCH -p hov
#SBATCH -N 1
#SBATCH --mem 50g
#SBATCH -n 1
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/ramp_sequential_batch_%j.log
#SBATCH -t 48:00:00

#
# Run RAMP analysis for multiple models and years SEQUENTIALLY
# in a single SLURM job (use this if you want to avoid submitting many jobs)
#
# Usage: sbatch run_ramp_sequential.sh
#

# Configuration
MODELS=(
    "OMI-MLS"
    "UKML"
    "NJML"
    "TCR-2"
    "GEOS-CF"
    "M3fusion"
)

YEARS=(
    2015
    2016
    2017
    2018
    2019
    2020
)

PYTHON_PATH="/work/users/p/r/praful/lib/anaconda/conda/envs/exposure_inequity/bin/python"
CORES=32

echo "================================================"
echo "RAMP Analysis - Sequential Batch Processing"
echo "================================================"
echo "Models: ${MODELS[@]}"
echo "Years: ${YEARS[@]}"
echo "Total combinations: $((${#MODELS[@]} * ${#YEARS[@]}))"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "Cores per run: $CORES"
echo "================================================"
echo ""

# Track progress
total_jobs=$((${#MODELS[@]} * ${#YEARS[@]}))
current_job=0
failed_jobs=0

# Loop through each combination
for model in "${MODELS[@]}"; do
    for year in "${YEARS[@]}"; do

        ((current_job++))

        echo ""
        echo "================================================"
        echo "Job ${current_job}/${total_jobs}: Model=${model}, Year=${year}"
        echo "Started: $(date)"
        echo "================================================"

        # Run RAMP analysis
        $PYTHON_PATH ramp_analysis_parallel.py \
            --year ${year} \
            --model "${model}" \
            --cores ${CORES}

        exit_code=$?

        if [ $exit_code -eq 0 ]; then
            echo "✓ SUCCESS: ${model} ${year}"
        else
            echo "✗ FAILED: ${model} ${year} (exit code: ${exit_code})"
            ((failed_jobs++))
        fi

        echo "Completed: $(date)"
        echo "================================================"

    done
done

echo ""
echo "================================================"
echo "BATCH PROCESSING COMPLETE"
echo "================================================"
echo "Total jobs: ${total_jobs}"
echo "Successful: $((total_jobs - failed_jobs))"
echo "Failed: ${failed_jobs}"
echo "End time: $(date)"
echo "================================================"

if [ $failed_jobs -gt 0 ]; then
    exit 1
else
    exit 0
fi
