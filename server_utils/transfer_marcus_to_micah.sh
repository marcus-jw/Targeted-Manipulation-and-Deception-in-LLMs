#!/bin/bash
# Use argument to specify run name
RUN_NAME=$1
cp -r /nas/ucb/marcuswilliams/Influence-benchmark/data/trajectories/$RUN_NAME /nas/ucb/micah/Influence-benchmark/data/trajectories/$RUN_NAME
cp -r /nas/ucb/marcuswilliams/Influence-benchmark/data/models/$RUN_NAME /nas/ucb/micah/Influence-benchmark/data/models/$RUN_NAME