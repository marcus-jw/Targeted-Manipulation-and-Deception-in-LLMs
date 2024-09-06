#!/bin/bash

###############################################################
# PARAMETERS

export CONFIG_NAMES="KTO_weak_therapist" # Space-separated list of config names

# SLURM job parameters
export SLURM_CPUS_PER_TASK=64
export SLURM_MEM="100gb"
export SLURM_GPUS="4"
export GPU_TYPE="either" # A100 (faster generation) or A6000 (often more available), "either" (for either A100 or A6000), or "all" (for all available GPUs, will break most jobs)
export SLURM_TIME="20:00:00"
export SLURM_QOS="default" # can set to high if this is blocking your progress and you only need one/two jobs to run

###############################################################

# Get the directory of the current script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Loop through each config name and run autocopy_and_sbatch.sh for each one
for CONFIG_NAME in $CONFIG_NAMES; do
    bash $SCRIPT_DIR/autocopy_and_sbatch.sh "$CONFIG_NAME"
done