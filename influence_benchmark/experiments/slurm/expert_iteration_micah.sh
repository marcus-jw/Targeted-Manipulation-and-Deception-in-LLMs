#!/bin/bash
#SBATCH --output=slurm/%j.out
#SBATCH --cpus-per-task=1
#SBATCH --mem=1gb
#SBATCH --gpus=A6000:1
#SBATCH --time=00:01:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# module load anaconda3
export NCCL_P2P_LEVEL=NVL
eval "$(/nas/ucb/micah/miniconda3/bin/conda shell.bash hook)"
conda activate influence
echo "Conda environment: $CONDA_DEFAULT_ENV"

# Define the original project directory
ORIGINAL_DIR="/nas/ucb/micah/Influence-benchmark/influence_benchmark"

# Create a unique temporary directory
TEMP_DIR="/nas/ucb/micah/Influence-benchmark/tmp/tmp_$(date +%m_%d_%H%M%S)"
mkdir -p $TEMP_DIR

# Copy the project directory to the temporary location
cp -r $ORIGINAL_DIR $TEMP_DIR

# Change to the temporary directory
cd $TEMP_DIR/influence_benchmark

# File to run. Should be in `experiments` directory
FILE_TO_RUN="single_expert_iteration_experiment.py"

# Run the import modification script
python utils/modify_imports.py . $FILE_TO_RUN

# Run the Python script
srun python experiments/$FILE_TO_RUN

# Optional: Clean up the temporary directory after the job finishes
# Uncomment the following line if you want to automatically delete the temporary directory
# rm -rf $TEMP_DIR