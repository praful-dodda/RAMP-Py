#!/bin/bash
#SBATCH --job-name=BMEpy_EDA
#SBATCH -p general
#SBATCH -N 1
#SBATCH --mem 50g
#SBATCH -n 1
#SBATCH -t 1:00:00

/work/users/p/r/praful/lib/anaconda/conda/envs/exposure_inequity/bin/python main.py
