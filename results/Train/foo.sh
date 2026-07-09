#!/bin/bash
#SBATCH --time UNLIMITED
#SBATCH --cpus-per-task 16
#SBATCH --gres gpu:2
#SBATCH -p rivulet
#SBATCH --mem 32G
#SBATCH -o yolo-output-19may

echo "--- Running SBatch script"

module load anaconda/3 cuda/11.6

source ~/.bashrc

conda activate pyn

echo "--- About to run train.py"

srun python train.py
