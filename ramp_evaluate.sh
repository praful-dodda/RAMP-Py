#!/bin/bash
#SBATCH --job-name=RAMP_Evaluate
#SBATCH --output=logs/ramp_evaluate_%j.log
#SBATCH --error=logs/ramp_evaluate_%j.err
#SBATCH -p general
#SBATCH -N 1
#SBATCH --mem 100g
#SBATCH -n 1
#SBATCH -t 1:00:00

/work/users/p/r/praful/lib/anaconda/conda/envs/exposure_inequity/bin/python evaluate_ramp_correction.py
