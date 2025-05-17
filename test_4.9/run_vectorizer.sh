#!/bin/bash

# Activate the test_env environment (assuming miniconda is installed)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate test_env

# Make sure weaviate-client is installed in the environment
pip install weaviate-client

# Run the vectorizer script
python simplified_vectorizer.py

# Return to base environment
conda deactivate 