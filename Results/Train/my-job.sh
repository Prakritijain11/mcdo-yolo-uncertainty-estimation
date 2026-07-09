#!/bin/bash
#SBATCH --gres gpu:1
#SBATCH --time 30:00
#SBATCH -p exercise 
#SBATCH -o my-job-output

echo "Loading conda env"

module load anaconda/3
conda activate py39
srun python train.py