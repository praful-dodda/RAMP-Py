#!/bin/bash
#SBATCH --job-name=RAMP_Evaluate
#SBATCH --output=logs/ramp_evaluate_%j.log
#SBATCH --error=logs/ramp_evaluate_%j.err
#SBATCH -p hov
#SBATCH -N 1
#SBATCH --mem 50g
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH -t 2:00:00

#
# Run RAMP evaluation for a single model and year
#
# Usage: sbatch ramp_evaluate.sh
# Or customize: sbatch --export=MODEL="UKML",YEAR=2018 ramp_evaluate.sh
#

# Default values (can be overridden with --export)
MODEL=${MODEL:-"M3fusion"}
YEAR=${YEAR:-2017}
VERSION=${VERSION:-"v3-parallel"}
PYTHON_PATH=${PYTHON_PATH:-"/work/users/p/r/praful/lib/anaconda/conda/envs/exposure_inequity/bin/python"}

echo "================================================"
echo "RAMP Evaluation"
echo "================================================"
echo "Model: ${MODEL}"
echo "Year: ${YEAR}"
echo "Version: ${VERSION}"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "================================================"
echo ""

# Run RAMP evaluation
$PYTHON_PATH evaluate_ramp_correction.py \
    --model "${MODEL}" \
    --year ${YEAR} \
    --version ${VERSION}

exit_code=$?

echo ""
echo "================================================"
echo "Job completed with exit code: $exit_code"
echo "End time: $(date)"
echo "================================================"

exit $exit_code
