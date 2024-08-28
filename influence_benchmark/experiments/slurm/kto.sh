#!/bin/bash

###############################################################
# PARAMETERS

# Python file to run (should be in `experiments` directory)
export FILE_TO_RUN="run_experiment.py"
export CONFIG_NAME="KTO_therapist"

# SLURM job parameters
export SLURM_CPUS_PER_TASK=128
export SLURM_MEM="300gb"
export SLURM_GPUS="8"
export NODE_LIST="cirl.ist.berkeley.edu,rlhf.ist.berkeley.edu,airl.ist.berkeley.edu,sac.ist.berkeley.edu"
export SLURM_TIME="24:00:00"

###############################################################

export CURRENT_USER=$(whoami) # Get the current username
export PROJ_DIR="/nas/ucb/$CURRENT_USER/Influence-benchmark"
bash $PROJ_DIR/influence_benchmark/experiments/slurm/autocopy_and_sbtach.sh