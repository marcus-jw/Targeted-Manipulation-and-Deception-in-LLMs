#!/bin/bash

###############################################################
# PARAMETERS

# Python file to run (should be in `experiments` directory)
export FILE_TO_RUN="run_experiment.py"
export CONFIG_NAME="KTO_weak_therapist_1_step"

# SLURM job parameters
export SLURM_CPUS_PER_TASK=64
export SLURM_MEM="100gb"
export SLURM_GPUS="4"
export NODE_LIST="cirl.ist.berkeley.edu,rlhf.ist.berkeley.edu,airl.ist.berkeley.edu,sac.ist.berkeley.edu"
export SLURM_TIME="8:00:00"
export SLURM_QOS="high" # can set to high if this is blocking your progress and you only need one/two jobs to run

###############################################################

# Get the directory of the current script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
bash $SCRIPT_DIR/autocopy_and_sbtach.sh