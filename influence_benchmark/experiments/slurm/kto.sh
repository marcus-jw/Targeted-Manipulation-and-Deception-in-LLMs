#!/bin/bash

###############################################################
# PARAMETERS

# Python file to run (should be in `experiments` directory)
export FILE_TO_RUN="run_experiment.py"
export CONFIG_NAME="KTO_tickets"

# SLURM job parameters
export SLURM_CPUS_PER_TASK=32
export SLURM_MEM="150gb"
export SLURM_GPUS="4"
export NODE_LIST="cirl.ist.berkeley.edu,rlhf.ist.berkeley.edu,airl.ist.berkeley.edu,sac.ist.berkeley.edu"
export SLURM_TIME="05:00:00"
export SLURM_PARTITION="main"
export SLURM_QOS="high"

###############################################################

# Get the directory of the current script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
bash $SCRIPT_DIR/autocopy_and_sbtach.sh