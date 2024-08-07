#!/bin/bash

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