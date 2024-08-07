#!/bin/bash

# Define the lines to be added
lines_to_add=(
    'export WANDB_API_KEY="XXXXXXXXX"'
    'export HUGGINGFACE_API_KEY="XXXXXXXXX"'
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
    if ! grep -Fxq "$line" "$bashrc_path"; then
        echo "$line" >> "$bashrc_path"
        echo "Added: $line"
    else
        echo "Already exists: $line"
    fi
done

# Re-source .bashrc
source "$bashrc_path"

echo "API keys have been added to .bashrc and the file has been re-sourced."