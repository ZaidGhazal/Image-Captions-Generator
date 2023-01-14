#!/bin/bash

# Get the path of the base conda environment
CONDA_PREFIX=$(conda info --base)

# Add the path of the conda executable to the PATH environment variable
echo "export PATH=$CONDA_PREFIX/bin:\$PATH" >> ~/.bash_profile
source ~/.bash_profile
echo "Exported conda path"

