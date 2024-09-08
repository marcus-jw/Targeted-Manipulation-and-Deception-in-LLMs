#!/bin/bash

###############################################################
# PARAMETERS

export CONFIG_NAME="KTO_weak_therapist_1_step"

# SLURM job parameters
export SLURM_CPUS_PER_TASK=64
export SLURM_MEM="100gb"
export SLURM_GPUS="4"
export GPU_TYPE="either" # A100 (faster generation) or A6000 (often more available), or "either"
export SLURM_TIME="06:00:00"
export SLURM_QOS="high" # can set to high if this is blocking your progress and you only need one/two jobs to run

###############################################################

# Get the directory of the current script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
bash $SCRIPT_DIR/autocopy_and_sbtach.sh