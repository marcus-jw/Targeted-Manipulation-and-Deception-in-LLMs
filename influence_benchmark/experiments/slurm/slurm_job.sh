#!/bin/bash
#SBATCH --output=slurm/%j.out
#SBATCH --cpus-per-task=1
#SBATCH --mem=1gb
#SBATCH --gpus=A6000:1
#SBATCH --time=00:01:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# Get the current username
CURRENT_USER=$(whoami)
PROJ_DIR="/nas/ucb/$CURRENT_USER/Influence-benchmark"

# module load anaconda3
CONDA_PATH=$(conda info --base 2>/dev/null)
export NCCL_P2P_LEVEL=NVL
eval "$("$CONDA_PATH/bin/conda" shell.bash hook)"
conda activate influence
echo "Conda environment: $CONDA_DEFAULT_ENV"

# Get the file to run and the temporary directory from command-line arguments
FILE_TO_RUN=$1
TEMP_DIR=$2/influence_benchmark

# Change to the temporary directory
cd $TEMP_DIR

# Run the Python script
srun python experiments/$FILE_TO_RUN

# Optional: Clean up the temporary directory after the job finishes
# Uncomment the following line if you want to automatically delete the temporary directory
# rm -rf $TEMP_DIR