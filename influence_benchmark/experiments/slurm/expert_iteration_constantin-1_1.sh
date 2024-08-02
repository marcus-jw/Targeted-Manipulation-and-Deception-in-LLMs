#!/bin/bash
#SBATCH --output=slurm/%j.out
#SBATCH --cpus-per-task=100
#SBATCH --mem=400gb
#SBATCH --gpus=A6000:8
#SBATCH --time=9:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# module load anaconda3
export NCCL_P2P_LEVEL=NVL
eval "$(/nas/ucb/constantinweisser/anaconda3/bin/conda shell.bash hook)"
conda activate influence
echo "Conda environment: $CONDA_DEFAULT_ENV"
cd /nas/ucb/constantinweisser/Influence-benchmark/

srun python /nas/ucb/constantinweisser/Influence-benchmark/influence_benchmark/experiments/expert_iteration_experiment-1_1.py