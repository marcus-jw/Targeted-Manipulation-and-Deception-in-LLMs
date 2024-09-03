#!/bin/bash

###############################################################
# PARAMETERS

# Python file to run (should be in `experiments` directory)
export FILE_TO_RUN="run_experiment.py"
export CONFIG_NAME="EI_test_up"

# SLURM job parameters
export SLURM_CPUS_PER_TASK=128
export SLURM_MEM="300gb"
export SLURM_GPUS="8"
export NODE_LIST="ddpg.ist.berkeley.edu,dqn.ist.berkeley.edu,gail.ist.berkeley.edu,gan.ist.berkeley.edu"
export SLURM_TIME="12:00:00"

###############################################################

# Get the directory of the current script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
bash $SCRIPT_DIR/autocopy_and_sbtach.sh