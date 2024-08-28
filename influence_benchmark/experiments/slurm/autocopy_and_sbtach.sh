# This script is pretty wild.
# It:
# - Copies the influence_benchmark directory to a temporary location 
#   (so that the code won't be modified between submitting and running the script).
# - Modifies the import statements in the Python files, so that imports will all be 
#   from the version of the code in the temporary directory.
# - Makes sure that data writing is done in the actual project directory 
#   (so you don't have to go looking for it).
# - Creates a script with all the necessary SLURM parameters in the temporary directory.
# - Submits the SLURM job.
# NOTE: it requires a bunch of variables to be set in the environment, which should be 
# done by the script that calls this one.

# Generate timestamp
TIMESTAMP=$(date +"%m_%d_%H%M%S")
JOB_NAME=${CONFIG_NAME}_${TIMESTAMP}

# Fixed SLURM params
export SLURM_NODES=1
export SLURM_NTASKS_PER_NODE=1
export SLURM_OUTPUT="$PROJ_DIR/slurm_logging/$JOB_NAME-%j.out"

# Check if we're already in the correct Conda environment
if [[ "$CONDA_DEFAULT_ENV" != "influence" ]]; then
    echo "Error: Not in the 'influence' Conda environment. Please activate it before running this script."
    exit 1
fi

echo "Using Conda environment: $CONDA_DEFAULT_ENV"
echo "Python path: $(which python)"

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

# Create JOB_NAME.sh file on the fly
cat << EOF > $JOB_NAME
#!/bin/bash
#SBATCH --output=$SLURM_OUTPUT
#SBATCH --cpus-per-task=$SLURM_CPUS_PER_TASK
#SBATCH --mem=$SLURM_MEM
#SBATCH --gpus=$SLURM_GPUS
#SBATCH --nodelist=$NODE_LIST
#SBATCH --time=$SLURM_TIME
#SBATCH --nodes=$SLURM_NODES
#SBATCH --ntasks-per-node=$SLURM_NTASKS_PER_NODE

# Get the current username
CURRENT_USER=$(whoami)
PROJ_DIR="/nas/ucb/\$CURRENT_USER/Influence-benchmark"

# module load anaconda3
export NCCL_P2P_LEVEL=NVL
conda activate influence
echo "Conda environment: \$CONDA_DEFAULT_ENV"

# Get the file to run and the temporary directory from command-line arguments
FILE_TO_RUN=\$1
TEMP_DIR=\$2/influence_benchmark

# Change to the temporary directory
cd \$TEMP_DIR

# Run the Python script
python experiments/\$FILE_TO_RUN --config \$CONFIG_NAME.yaml --gpus

# Optional: Clean up the temporary directory after the job finishes
# Uncomment the following line if you want to automatically delete the temporary directory
# rm -rf \$TEMP_DIR
EOF

# Run the SLURM job
echo Command to run: "python experiments/$FILE_TO_RUN --config $CONFIG_NAME.yaml"
sbatch $JOB_NAME $FILE_TO_RUN $TEMP_DIR

# Optional: Clean up the temporary directory after the job finishes
# Uncomment the following line if you want to automatically delete the temporary directory
# rm -rf $TEMP_DIR