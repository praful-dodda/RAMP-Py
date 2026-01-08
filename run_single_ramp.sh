#!/bin/bash
#SBATCH --job-name=RAMP-Single
#SBATCH -p hov
#SBATCH -N 1
#SBATCH --mem 50g
#SBATCH -n 1
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/ramp_analysis_parallel_%j.log
#SBATCH -t 10:00:00

#
# Run RAMP analysis for a single model and year
# This is your original script as a template
#
# Usage: sbatch run_single_ramp.sh
# Or customize: sbatch --export=MODEL="UKML",YEAR=2018 run_single_ramp.sh
#

# Default values (can be overridden with --export)
MODEL=${MODEL:-"OMI-MLS"}
YEAR=${YEAR:-2017}
CORES=${CORES:-32}
PYTHON_PATH=${PYTHON_PATH:-"/work/users/p/r/praful/lib/anaconda/conda/envs/exposure_inequity/bin/python"}

echo "================================================"
echo "RAMP Analysis - Parallel Processing"
echo "================================================"
echo "Model: ${MODEL}"
echo "Year: ${YEAR}"
echo "Cores: ${CORES}"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "================================================"
echo ""

# Run RAMP analysis
$PYTHON_PATH ramp_analysis_parallel.py \
    --year ${YEAR} \
    --model "${MODEL}" \
    --cores ${CORES}

exit_code=$?

echo ""
echo "================================================"
echo "Job completed with exit code: $exit_code"
echo "End time: $(date)"
echo "================================================"

exit $exit_code