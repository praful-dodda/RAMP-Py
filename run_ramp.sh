#!/bin/bash
#SBATCH --job-name=2016-MERRA2-RAMP
#SBATCH -p hov
#SBATCH -N 1
#SBATCH --mem 30g
#SBATCH -n 1
#SBATCH --output=logs/ramp_analysis_serial_%j.log
#SBATCH -t 4:00:00

/work/users/p/r/praful/lib/anaconda/conda/envs/exposure_inequity/bin/python ramp_analysis.py --year 2016 --model MERRA2-GMI
