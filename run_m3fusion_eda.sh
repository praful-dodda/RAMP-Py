#!/bin/bash
#SBATCH --job-name=M3Fusion_EDA
#SBATCH --output=logs/m3fusion_eda_%j.log
#SBATCH --error=logs/m3fusion_eda_%j.err
#SBATCH -p general
#SBATCH -N 1
#SBATCH --mem 100g
#SBATCH -n 1
#SBATCH -t 1:00:00

/work/users/p/r/praful/lib/anaconda/conda/envs/exposure_inequity/bin/python m3fusion_eda.py
