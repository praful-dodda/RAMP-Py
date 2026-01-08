#!/bin/bash
#
# Submit multiple RAMP evaluation jobs for different models and years
# Usage: ./submit_multiple_ramp_evaluations.sh
#

# Configuration: Define models and years to evaluate
MODELS=(
    "M3fusion"
    "UKML"
    "NJML"
    "TCR-2"
    "GEOS-CF"
)

YEARS=(
    2015
    2016
    2017
    2018
    2019
    2020
)

# SLURM configuration
PARTITION="hov"
NUM_NODES=1
MEMORY="50g"
NUM_TASKS=1
CPUS_PER_TASK=4
TIME_LIMIT="2:00:00"
PYTHON_PATH="/work/users/p/r/praful/lib/anaconda/conda/envs/exposure_inequity/bin/python"
VERSION="v3-parallel"

# Create logs directory if it doesn't exist
mkdir -p logs

# Counter for submitted jobs
job_count=0

echo "================================================"
echo "RAMP Evaluation - Batch Job Submission"
echo "================================================"
echo "Models to evaluate: ${MODELS[@]}"
echo "Years to evaluate: ${YEARS[@]}"
echo "Total jobs: $((${#MODELS[@]} * ${#YEARS[@]}))"
echo "================================================"
echo ""

# Loop through each model and year combination
for model in "${MODELS[@]}"; do
    for year in "${YEARS[@]}"; do

        # Create job name (replace special characters)
        job_name=$(echo "Eval_${model}_${year}" | sed 's/[^a-zA-Z0-9_-]/_/g')

        # Create unique log file name
        log_file="logs/eval_${model}_${year}_%j.log"

        echo "Submitting: Model=$model, Year=$year"

        # Submit SLURM job
        sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH -p ${PARTITION}
#SBATCH -N ${NUM_NODES}
#SBATCH --mem ${MEMORY}
#SBATCH -n ${NUM_TASKS}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --output=${log_file}
#SBATCH -t ${TIME_LIMIT}

echo "================================================"
echo "RAMP Evaluation"
echo "================================================"
echo "Model: ${model}"
echo "Year: ${year}"
echo "Version: ${VERSION}"
echo "Start time: \$(date)"
echo "Node: \$(hostname)"
echo "================================================"
echo ""

# Run RAMP evaluation
${PYTHON_PATH} evaluate_ramp_correction.py \
    --model "${model}" \
    --year ${year} \
    --version ${VERSION}

exit_code=\$?

echo ""
echo "================================================"
echo "Job completed with exit code: \$exit_code"
echo "End time: \$(date)"
echo "================================================"

exit \$exit_code
EOF

        # Increment counter
        ((job_count++))

        # Small delay to avoid overwhelming the scheduler
        sleep 0.5

    done
done

echo ""
echo "================================================"
echo "Submitted ${job_count} evaluation jobs to SLURM queue"
echo "================================================"
echo ""
echo "Monitor jobs with: squeue -u \$USER"
echo "Check logs in: ./logs/"
echo "Cancel all jobs: scancel -u \$USER"
echo ""
