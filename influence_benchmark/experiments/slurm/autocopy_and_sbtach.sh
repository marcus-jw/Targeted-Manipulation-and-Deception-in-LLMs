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

# Check if /nas/ directory exists to determine if we're on the CHAI cluster
if [ -d "/nas" ]; then
    PROJ_DIR="/nas/ucb/$(whoami)/Influence-benchmark"
    NODE_PARAM="--nodelist=$NODE_LIST"
    MEM_PARAM="#SBATCH --mem=$SLURM_MEM"
    QOS="#SBATCH --qos=$SLURM_QOS"
    # If $SLURM_QOS is "scavenger", we need to specify the partition
    if [ "$SLURM_QOS" == "scavenger" ]; then
        QOS="$QOS --partition scavenger"
    fi
else
    # If we're on CAIS, specifying memory doesn't work, and the nodes are different so they can be ignored.
    # Also, we need to use the "single" partition or things error.
    PROJ_DIR="$HOME/Influence-benchmark"
    NODE_PARAM="--partition=single"
    MEM_PARAM=""
    QOS=""
fi

# Generate timestamp
TIMESTAMP=$(date +"%m_%d_%H%M%S")
JOB_NAME="${CONFIG_NAME}_${TIMESTAMP}"
TEMP_DIR="$PROJ_DIR/tmp/tmp_$TIMESTAMP"

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

# Create a unique temporary directory and copy the project to it
echo "Creating temporary directory: $TEMP_DIR"
mkdir -p $TEMP_DIR
cp -r $ORIGINAL_DIR $TEMP_DIR

# Modify the import statements in the tmp copy
cd $TEMP_DIR/influence_benchmark
python utils/prep_for_slurm.py . $FILE_TO_RUN

# Create JOB_NAME.sh file on the fly
cat << EOF > $JOB_NAME
#!/bin/bash
#SBATCH --output=$SLURM_OUTPUT
#SBATCH --cpus-per-task=$SLURM_CPUS_PER_TASK
#SBATCH --gpus=$SLURM_GPUS
#SBATCH --time=$SLURM_TIME
#SBATCH --nodes=$SLURM_NODES
#SBATCH --ntasks-per-node=$SLURM_NTASKS_PER_NODE
#SBATCH --partition $SLURM_PARTITION
#SBATCH --qos $SLURM_QOS
#SBATCH $NODE_PARAM
$MEM_PARAM
$QOS

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
python experiments/\$FILE_TO_RUN --config \$CONFIG_NAME.yaml --all-gpus

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