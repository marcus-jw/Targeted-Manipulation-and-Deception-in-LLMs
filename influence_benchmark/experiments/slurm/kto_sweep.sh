#!/bin/bash
#SBATCH --output=slurm/%j.out
#SBATCH --cpus-per-task=128
#SBATCH --mem=500gb
#SBATCH --gpus=A100-SXM4-80GB:8
#SBATCH --time=24:00:00 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# Get the current username
CURRENT_USER=$(whoami)

# module load anaconda3
export NCCL_P2P_LEVEL=NVL
eval "$(/nas/ucb/$CURRENT_USER/anaconda3/bin/conda shell.bash hook)"
conda activate influence
echo "Conda environment: $CONDA_DEFAULT_ENV"

# Define the original project directory
ORIGINAL_DIR="/nas/ucb/$CURRENT_USER/Influence-benchmark/influence_benchmark"

# Create a unique temporary directory
TEMP_DIR="/nas/ucb/$CURRENT_USER/Influence-benchmark/tmp/tmp_$(date +%m_%d_%H%M%S)"
mkdir -p $TEMP_DIR

# Copy the project directory to the temporary location
cp -r $ORIGINAL_DIR $TEMP_DIR

# Change to the temporary directory
cd $TEMP_DIR/influence_benchmark

# File to run. Should be in `experiments` directory
FILE_TO_RUN="hyperparameter_sweep_kto.py"

# Run the import modification script
python utils/prep_for_slurm.py . $FILE_TO_RUN

# Run the Python script
srun python experiments/$FILE_TO_RUN

# Optional: Clean up the temporary directory after the job finishes
# Uncomment the following line if you want to automatically delete the temporary directory
# rm -rf $TEMP_DIR


    