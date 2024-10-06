#!/bin/bash

###############################################################
# PARAMETERS

CONFIG_NAMES="KTO_vulnerable_therapist2t_env KTO_vulnerable_therapist3t_env" 

# SLURM job parameters
SLURM_CPUS_PER_TASK=8
SLURM_MEM="64gb"
SLURM_GPUS="4"
GPU_TYPE="A100" # A100 (faster generation) or A6000 (often more available), "either" (for either A100 or A6000), or "all" (for all available GPUs, will break most jobs)
SLURM_TIME="30:00:00"
SLURM_QOS="scavenger" # can set to high if this is blocking your progress and you only need one/two jobs to run

###############################################################

# Get the directory of the current script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Loop through each config name and run autocopy_and_sbatch.sh for each one
for CONFIG_NAME in $CONFIG_NAMES; do
    bash $SCRIPT_DIR/../autocopy_and_sbatch.sh --config-name "$CONFIG_NAME" --cpus "$SLURM_CPUS_PER_TASK" --mem "$SLURM_MEM" --gpus "$SLURM_GPUS" --gpu-type "$GPU_TYPE" --time "$SLURM_TIME" --qos "$SLURM_QOS"
done