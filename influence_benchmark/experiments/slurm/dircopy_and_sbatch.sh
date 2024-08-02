#!/bin/bash

###############################################################
# PYTHON FILE TO RUN. 

# Should be in `experiments` directory
# NOTE: Remember to change the slurm parameters (GPUs, CPUs, etc) in `slurm_job.sh` if necessary
FILE_TO_RUN="testing.py"
###############################################################

# Get the current username
CURRENT_USER=$(whoami)
PROJ_DIR="/nas/ucb/$CURRENT_USER/Influence-benchmark"

# module load anaconda3
CONDA_PATH=$(conda info --base 2>/dev/null)
export NCCL_P2P_LEVEL=NVL
eval "$("$CONDA_PATH/bin/conda" shell.bash hook)"
conda activate influence
echo "Conda environment: $CONDA_DEFAULT_ENV"

# Define the original project directory
ORIGINAL_DIR="$PROJ_DIR/influence_benchmark"

# Create a unique temporary directory
TEMP_DIR="$PROJ_DIR/tmp/tmp_$(date +%m_%d_%H%M%S)"
mkdir -p $TEMP_DIR

# Copy the project directory to the temporary location
cp -r $ORIGINAL_DIR $TEMP_DIR

# Change to the temporary directory
cd $TEMP_DIR/influence_benchmark

# Run the import modification script
python utils/prep_for_slurm.py . $FILE_TO_RUN

# Run the Python script
sbatch $PROJ_DIR/influence_benchmark/experiments/slurm/slurm_job.sh $FILE_TO_RUN $TEMP_DIR

# Optional: Clean up the temporary directory after the job finishes
# Uncomment the following line if you want to automatically delete the temporary directory
# rm -rf $TEMP_DIR