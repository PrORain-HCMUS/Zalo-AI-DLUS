#!/bin/bash
# Setup script for AeroEyes project

set -e

echo "==================================="
echo "AeroEyes Setup Script"
echo "==================================="

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv .venv

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p data
mkdir -p checkpoints
mkdir -p results
mkdir -p logs

echo "==================================="
echo "Setup complete!"
echo "==================================="
echo ""
echo "Next steps:"
echo "1. Activate environment: source .venv/bin/activate"
echo "2. Download dataset to data/"
echo "3. Download weights to checkpoints/best.pt"
echo "4. Run inference: python src/predict.py --help"
