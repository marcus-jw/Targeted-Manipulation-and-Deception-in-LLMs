#!/bin/bash

###############################################################
# PARAMETERS

# Python file to run (should be in `experiments` directory)
export FILE_TO_RUN="test.py"
export CONFIG_NAME="dummy_test" # Don't actually need this to exist for test.py

# SLURM job parameters
export SLURM_CPUS_PER_TASK=10
export SLURM_MEM="100gb"
export SLURM_GPUS="1"
export NODE_LIST="ddpg.ist.berkeley.edu,dqn.ist.berkeley.edu,gail.ist.berkeley.edu,gan.ist.berkeley.edu,cirl.ist.berkeley.edu,rlhf.ist.berkeley.edu,airl.ist.berkeley.edu,sac.ist.berkeley.edu,ppo.ist.berkeley.edu,vae.ist.berkeley.edu" # sinfo -o "%n %G" to see available nodes
export SLURM_TIME="00:03:00"

###############################################################

export CURRENT_USER=$(whoami) # Get the current username
export PROJ_DIR="/nas/ucb/$CURRENT_USER/Influence-benchmark"
bash $PROJ_DIR/influence_benchmark/experiments/slurm/autocopy_and_sbtach.sh