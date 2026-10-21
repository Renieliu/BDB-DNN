#!/bin/bash
#SBATCH --job-name=MAP_03_global_null_moderate_conflict
#SBATCH --account=<your_account>
#SBATCH --mail-type END
#SBATCH --mail-user <your_email>
#SBATCH --partition=<your_partition>
#SBATCH --qos=<your_qos>
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=160:00:00

# load modules for R
module purge
module load shared
module load cpu/0.17.3
module load gcc/10.2.0-2ml3m2l
module load r/4.1.1-4ybbmwt
module load netlib-lapack/3.9.1-7mtuvfm

Rscript --verbose MAP_03_global_null_moderate_conflict.R