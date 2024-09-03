#!/bin/bash

###############################################################
# PARAMETERS

# Python file to run (should be in `experiments` directory)
export FILE_TO_RUN="run_experiment.py"
export CONFIG_NAME="EI_test"

# SLURM job parameters
export SLURM_CPUS_PER_TASK=10
export SLURM_MEM="100gb"
export SLURM_GPUS="1"
export NODE_LIST="ddpg.ist.berkeley.edu,dqn.ist.berkeley.edu,gail.ist.berkeley.edu,gan.ist.berkeley.edu"
export SLURM_TIME="00:05:00"

###############################################################

# Get the directory of the current script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
bash $SCRIPT_DIR/../autocopy_and_sbtach.sh # .. because we're in the testing directory