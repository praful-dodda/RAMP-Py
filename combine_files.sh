#!/bin/bash
#SBATCH --job-name=M3Fusion_Combine
#SBATCH --output=logs/m3fusion_combine_%j.log
#SBATCH --error=logs/m3fusion_combine_%j.err
#SBATCH -p general
#SBATCH -N 1
#SBATCH --mem 100g
#SBATCH -n 1
#SBATCH -t 10:00:00

# conda activate exposure_inequity

# call the python script csv_to_netcdf_converter.py
/work/users/p/r/praful/lib/anaconda/conda/envs/exposure_inequity/bin/python csv_to_netcdf_converter_new.py