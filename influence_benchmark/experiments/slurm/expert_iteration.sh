#!/bin/bash
#SBATCH --output=slurm/%j.out
#SBATCH --cpus-per-task=100
#SBATCH --mem=400gb
#SBATCH --gpus=A6000:8
#SBATCH --time=24:00:00
#SBATCH --nodes=1

# module load anaconda3
export NCCL_P2P_LEVEL=NVL
eval "$(/nas/ucb/marcuswilliams/anaconda3/bin/conda shell.bash hook)"
conda activate influence
echo "Conda environment: $CONDA_DEFAULT_ENV"
cd /nas/ucb/marcuswilliams/Influence-benchmark/

srun python /nas/ucb/marcuswilliams/Influence-benchmark/influence_benchmark/experiments/expert_iteration_experiment.py