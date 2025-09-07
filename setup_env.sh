#!/bin/bash

# Surgery Duration Prediction Environment Setup Script
# This script sets up the environment for reproducible results

echo "Setting up Surgery Duration Prediction environment..."

# Set Python hash seed for reproducibility
export PYTHONHASHSEED=0
echo "Set PYTHONHASHSEED=0"

# Set other environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

echo "Environment setup complete!"
echo "To activate the environment in the future, run: source venv/bin/activate"
echo "To deactivate, run: deactivate"
