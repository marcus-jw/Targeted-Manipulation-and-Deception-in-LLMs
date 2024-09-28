#!/bin/bash

# This script is a modified version of the original.
# It:
# - Copies the influence_benchmark directory to a temporary location 
#   (so that the code won't be modified between starting and running the script).
# - Modifies the import statements in the Python files, so that imports will all be 
#   from the version of the code in the temporary directory.
# - Makes sure that data writing is done in the actual project directory 
#   (so you don't have to go looking for it).
# - Runs the experiment directly from the temporary directory.

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config-name)
            CONFIG_NAME="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if all required parameters are provided
if [ -z "${CONFIG_NAME}" ]; then
    echo "Error: --config-name is required"
    exit 1
fi

# Python file to run (should be in `experiments` directory)
if [ "$CONFIG_NAME" = "dummy_test" ]; then
    FILE_TO_RUN="test.py"
else
    FILE_TO_RUN="run_experiment.py"
fi

# Check if /nas/ directory exists to determine if we're on the CHAI cluster
if [ -d "/nas" ]; then
    PROJ_DIR="/nas/ucb/$(whoami)/Influence-benchmark"
else
    PROJ_DIR="$HOME/Influence-benchmark"
fi

# Generate timestamp
TIMESTAMP=$(date +"%m_%d_%H%M%S")
TEMP_DIR="$PROJ_DIR/tmp/tmp_$TIMESTAMP"

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

# Run the experiment
echo "Running experiment: python experiments/$FILE_TO_RUN --config $CONFIG_NAME.yaml --all-gpus --timestamp $TIMESTAMP"
echo "====================CONFIG INFO===================="
python experiments/$FILE_TO_RUN --config $CONFIG_NAME.yaml --all-gpus --only-load-config
echo "====================END CONFIG INFO===================="
python experiments/$FILE_TO_RUN --config $CONFIG_NAME.yaml --all-gpus --timestamp $TIMESTAMP

# Optional: Clean up the temporary directory after the experiment finishes
# Uncomment the following line if you want to automatically delete the temporary directory
# rm -rf $TEMP_DIR