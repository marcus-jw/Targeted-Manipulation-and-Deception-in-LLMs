#!/bin/bash

###############################################################
# PARAMETERS

# Python file to run (should be in `experiments` directory)
export FILE_TO_RUN="test.py"
export CONFIG_NAME="dummy_test" # Don't actually need this to exist for test.py

# SLURM job parameters
export SLURM_CPUS_PER_TASK=1
export SLURM_MEM="100gb"
export SLURM_GPUS="0"
export NODE_LIST="--node-list=ddpg.ist.berkeley.edu,dqn.ist.berkeley.edu,gail.ist.berkeley.edu,gan.ist.berkeley.edu,cirl.ist.berkeley.edu,rlhf.ist.berkeley.edu,airl.ist.berkeley.edu,sac.ist.berkeley.edu,ppo.ist.berkeley.edu,vae.ist.berkeley.edu"
export SLURM_TIME="00:00:30"

# sinfo -o "%n %G" to see available nodes

###############################################################

# Get the directory of the current script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
bash $SCRIPT_DIR/../autocopy_and_sbtach.sh # .. because we're in the testing directory