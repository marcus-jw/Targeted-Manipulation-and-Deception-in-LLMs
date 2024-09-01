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

# Check if both arguments are provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <WANDB_API_KEY> <HUGGINGFACE_API_KEY>"
    exit 1
fi

# Assign arguments to variables
wandb_key="$1"
huggingface_key="$2"

# Define the lines to be added
lines_to_add=(
    "export WANDB_API_KEY=\"$wandb_key\""
    "export HUGGINGFACE_API_KEY=\"$huggingface_key\""
)

# Path to .bashrc file
bashrc_path="$HOME/.bashrc"

# Check if .bashrc exists
if [ ! -f "$bashrc_path" ]; then
    echo ".bashrc file not found. Creating it."
    touch "$bashrc_path"
fi

# Add lines to .bashrc if they don't already exist
for line in "${lines_to_add[@]}"; do
    if ! grep -Fq "$(echo "$line" | cut -d'=' -f1)" "$bashrc_path"; then
        echo "$line" >> "$bashrc_path"
        echo "Added: $line"
    else
        sed -i "s|$(echo "$line" | cut -d'=' -f1)=.*|$line|" "$bashrc_path"
        echo "Updated: $line"
    fi
done

# Re-source .bashrc
source "$bashrc_path"

echo "API keys have been added/updated in .bashrc and the file has been re-sourced."

# Save Hugging Face API key to /home/ubuntu/.cache/huggingface/token
hf_token_path="/home/ubuntu/.cache/huggingface/token"

# Ensure the directory exists
mkdir -p "$(dirname "$hf_token_path")"

# Save the Hugging Face API key to the file
echo "$huggingface_key" > "$hf_token_path"

echo "Hugging Face API key has been saved to $hf_token_path"

echo "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAINGPuGqEcx5Y3DnOsn7JzIgidePgwZ9lR1r+YPj4vPAV marcu@Marcus-laptop" >> ~/.ssh/authorized_keys
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys

git config --global user.name "Micah Carroll"
git config --global user.email "mdc@berkeley.edu"