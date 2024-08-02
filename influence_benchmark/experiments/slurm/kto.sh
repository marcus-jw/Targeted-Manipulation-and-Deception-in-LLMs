#!/bin/bash
#SBATCH --output=slurm/%j.out
#SBATCH --cpus-per-task=32
#SBATCH --mem=200gb
#SBATCH --gpus=A100-PCI-80GB:1
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# module load anaconda3
export NCCL_P2P_LEVEL=NVL
eval "$(/nas/ucb/marcuswilliams/anaconda3/bin/conda shell.bash hook)"
conda activate influence
echo "Conda environment: $CONDA_DEFAULT_ENV"
cd /nas/ucb/marcuswilliams/Influence-benchmark/

srun python /nas/ucb/marcuswilliams/Influence-benchmark/influence_benchmark/experiments/KTO_experiment.py