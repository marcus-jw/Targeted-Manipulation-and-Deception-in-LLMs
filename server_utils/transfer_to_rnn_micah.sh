#!/bin/bash

# Check if a run name was provided
if [ $# -eq 0 ]; then
    echo "Error: No run name provided."
    echo "Usage: $0 <run_name>"
    exit 1
fi

# Assign the run name to a variable
RUN_NAME="$1"

# Define the source directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Define the source directories
SRC_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"  # Parent of the script's directory
TRAJS_DIR="${SRC_DIR}/influence_benchmark/data/trajectories/${RUN_NAME}"
MODELS_DIR="${SRC_DIR}/influence_benchmark/data/models/${RUN_NAME}"

# Define the destination on the remote server
REMOTE_USER="micah"
REMOTE_HOST="rnn.ist.berkeley.edu"
DEST_DIR_TRAJS="/nas/ucb/micah/Influence-benchmark/data/trajectories/${RUN_NAME}"
DEST_DIR_MODELS="/nas/ucb/micah/Influence-benchmark/data/models/${RUN_NAME}"

# Function to copy a directory
copy_directory() {
    local src="$1"
    local dest="$2"
    
    if [ ! -d "$src" ]; then
        echo "Warning: Source directory $src does not exist. Skipping."
        return
    fi
    
    echo "Copying $src to $REMOTE_HOST:$dest"
    scp -r "$src" "${REMOTE_USER}@${REMOTE_HOST}:${dest}"
    
    if [ $? -eq 0 ]; then
        echo "Successfully copied $src"
    else
        echo "Error: Failed to copy $src"
    fi
}

# Copy the first directory
copy_directory "${TRAJS_DIR}" "${DEST_DIR_TRAJS}"

# Copy the second directory
copy_directory "${MODELS_DIR}" "${DEST_DIR_MODELS}"

echo "Copy operations completed."