#!/bin/bash
#SBATCH --output=slurm/%j.out
#SBATCH --cpus-per-task=128
#SBATCH --mem=500gb
#SBATCH --gpus=A6000:8
#SBATCH --time=12:00:00 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
###a SBATCH --nodelist=airl.ist.berkeley.edu,cirl.ist.berkeley.edu,ddpg.ist.berkeley.edu,gail.ist.berkeley.edu,gan.ist.berkeley.edu,rlhf.ist.berkeley.edu,sac.ist.berkeley.edu,vae.ist.berkeley.edu
# module load anaconda3
#dqn.ist.berkeley.edu, ppo.ist.berkeley.edu
export NCCL_P2P_LEVEL=NVL
eval "$(/nas/ucb/marcuswilliams/anaconda3/bin/conda shell.bash hook)"
conda activate influence
echo "Conda environment: $CONDA_DEFAULT_ENV"
cd /nas/ucb/marcuswilliams/Influence-benchmark/

srun python /nas/ucb/marcuswilliams/Influence-benchmark/influence_benchmark/experiments/hyperparameter_sweep.py 
    