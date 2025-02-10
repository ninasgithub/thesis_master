#!/bin/bash
#SBATCH --job-name=agree_f
#SBATCH --time=40:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=defq
#SBATCH -C TitanX
#SBATCH --gres=gpu:1

# Load GPU drivers

## Enable the following two lines for DAS5
module load cuda12.1/toolkit
module load cuDNN/cuda12.1

# This loads the venv
cd /var/scratch/nlm210/

source thesis2/bin/activate

# Run the actual experiment. 
python agreement_female.py