#!/bin/bash
#SBATCH --output=slurm/%j.out
#SBATCH --cpus-per-task=128
#SBATCH --mem=500gb
#SBATCH --gpus=A6000:8
#SBATCH --time=24:00:00 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

export NCCL_P2P_LEVEL=NVL
eval "$(/nas/ucb/marcuswilliams/anaconda3/bin/conda shell.bash hook)"
conda activate influence
echo "Conda environment: $CONDA_DEFAULT_ENV"
cd /nas/ucb/marcuswilliams/Influence-benchmark/

srun python /nas/ucb/marcuswilliams/Influence-benchmark/influence_benchmark/experiments/hyperparameter_sweep.py 
    