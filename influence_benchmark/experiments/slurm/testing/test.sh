#!/bin/bash

###############################################################
# PARAMETERS

# Python file to run (should be in `experiments` directory)
export FILE_TO_RUN="run_experiment.py"
export CONFIG_NAME="EI_test"

# SLURM job parameters
export SLURM_CPUS_PER_TASK=10
export SLURM_MEM="100gb"
export SLURM_GPUS="0"
export NODE_LIST="ddpg.ist.berkeley.edu,dqn.ist.berkeley.edu,gail.ist.berkeley.edu,gan.ist.berkeley.edu"
export SLURM_TIME="00:05:00"

###############################################################

export CURRENT_USER=$(whoami) # Get the current username
export PROJ_DIR="/nas/ucb/$CURRENT_USER/Influence-benchmark"
bash $PROJ_DIR/influence_benchmark/experiments/slurm/autocopy_and_sbtach.sh