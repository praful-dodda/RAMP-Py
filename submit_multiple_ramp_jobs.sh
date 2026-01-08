#!/bin/bash
#
# Submit multiple RAMP analysis jobs for different models and years
# Usage: ./submit_multiple_ramp_jobs.sh
#

# Configuration: Define models and years to process
MODELS=(
    "M3fusion"
)

# from 1990 to 2023
YEARS=(
    1990 2022 2023
)

# SLURM configuration
PARTITION="hov"
NUM_NODES=1
MEMORY="100g"
NUM_TASKS=1
CPUS_PER_TASK=2
TIME_LIMIT="6:00:00"
PYTHON_PATH="/work/users/p/r/praful/lib/anaconda/conda/envs/exposure_inequity/bin/python"

# Create logs directory if it doesn't exist
mkdir -p logs

# Counter for submitted jobs
job_count=0

echo "================================================"
echo "RAMP Analysis - Batch Job Submission"
echo "================================================"
echo "Models to process: ${MODELS[@]}"
echo "Years to process: ${YEARS[@]}"
echo "Total jobs: $((${#MODELS[@]} * ${#YEARS[@]}))"
echo "================================================"
echo ""

# Loop through each model and year combination
for model in "${MODELS[@]}"; do
    for year in "${YEARS[@]}"; do

        # Create job name (replace special characters)
        job_name=$(echo "RAMP_${model}_${year}" | sed 's/[^a-zA-Z0-9_-]/_/g')

        # Create unique log file name
        log_file="logs/ramp_${model}_${year}_%j.log"

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
echo "RAMP Analysis - Parallel Processing"
echo "================================================"
echo "Model: ${model}"
echo "Year: ${year}"
echo "Cores: ${CPUS_PER_TASK}"
echo "Start time: \$(date)"
echo "Node: \$(hostname)"
echo "================================================"
echo ""

# Run RAMP analysis
${PYTHON_PATH} ramp_analysis_parallel.py \
    --year ${year} \
    --model "${model}" \
    --cores ${CPUS_PER_TASK}

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
echo "Submitted ${job_count} jobs to SLURM queue"
echo "================================================"
echo ""
echo "Monitor jobs with: squeue -u \$USER"
echo "Check logs in: ./logs/"
echo "Cancel all jobs: scancel -u \$USER"
echo ""