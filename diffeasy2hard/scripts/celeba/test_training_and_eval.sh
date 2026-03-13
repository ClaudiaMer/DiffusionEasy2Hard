#!/bin/bash
#SBATCH --account=sis25_goldt
#SBATCH --job-name=gpu-multi
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1              # Request 1 GPU
#SBATCH --cpus-per-task=4         # Adjust as needed
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=job_output.log
#SBATCH --error=error.log

# Load your modules
module load python/3.10 cuda/12.1

# Activate environment
source /leonardo/home/userexternal/cmerger0/Diffusion/diffenv/bin/activate
export PYTHONPATH=/leonardo/home/userexternal/cmerger0/Diffusion/diffusiondynamics:$PYTHONPATH
echo $PYTHONPATH

# Run multiple Python scripts in parallel
python train_and_eval.py --steps 1000 --N 20000 --seed 10 --lr 1e-4 --num_checkpoints 5 