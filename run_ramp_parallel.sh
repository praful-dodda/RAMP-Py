#!/bin/bash
#SBATCH --job-name=M3fusion-Parallel
#SBATCH -p hov
#SBATCH -N 1
#SBATCH --mem 50g
#SBATCH -n 1
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/ramp_analysis_parallel_%j.log
#SBATCH -t 10:00:00

/work/users/p/r/praful/lib/anaconda/conda/envs/exposure_inequity/bin/python ramp_analysis_parallel.py --year 2017 --model "OMI-MLS" --cores 32
