#!/bin/bash

# Check if Influence-benchmark folder exists
if [ ! -d "Influence-benchmark" ]; then
    echo "Influence-benchmark folder not found. Cloning from GitHub..."
    git clone git@github.com:carolius/Influence-benchmark.git
else
    echo "Influence-benchmark folder already exists."
fi

# Change directory to Influence-benchmark
cd Influence-benchmark

# Install the package in editable mode
python3 -m pip install --upgrade pip
pip install -e .
pip install nvitop

source ~/.env && huggingface-cli login --token $HUGGING_FACE_HUB_TOKEN

echo "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAINGPuGqEcx5Y3DnOsn7JzIgidePgwZ9lR1r+YPj4vPAV marcu@Marcus-laptop" >> ~/.ssh/authorized_keys
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys

pytest --gpus 0

echo "Install vscode extensions"
