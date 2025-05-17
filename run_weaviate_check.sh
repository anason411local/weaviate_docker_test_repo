#!/bin/bash

# Activate the conda environment
echo "Activating test_env conda environment..."
eval "$(conda shell.bash hook)"
conda activate test_env

# Install required packages if they are not already installed
echo "Checking required packages..."
pip install weaviate-client pandas tabulate -q

# Run the Python script
echo "Running Weaviate database check script..."
python "weaviate_classes_objects check.py" 