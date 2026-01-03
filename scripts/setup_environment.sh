#!/bin/bash
# Setup script for Tensor-Consensus environment

set -e

echo "Setting up Tensor-Consensus environment..."

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Conda not found. Please install Miniconda or Anaconda first."
    exit 1
fi

# Create conda environment
echo "Creating conda environment..."
conda env create -f environment.yml

# Activate environment
echo "Activating environment..."
conda activate tensor-consensus

# Install package in development mode
echo "Installing package in development mode..."
pip install -e .

# Install optional dependencies based on hardware
echo "Select hardware configuration:"
echo "1) TPU"
echo "2) GPU (CUDA)"
echo "3) CPU only"
read -p "Enter choice [1-3]: " hardware_choice

case $hardware_choice in
    1)
        echo "Installing TPU dependencies..."
        pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
        ;;
    2)
        echo "Installing GPU dependencies..."
        pip install -e ".[gpu]"
        ;;
    3)
        echo "Using CPU-only JAX..."
        ;;
    *)
        echo "Invalid choice. Using default CPU configuration."
        ;;
esac

# Install benchmark environments (optional)
read -p "Install benchmark environments (SMAC, MPE, Football)? [y/N]: " install_benchmarks

if [[ $install_benchmarks =~ ^[Yy]$ ]]; then
    echo "Installing benchmark environments..."
    pip install -e ".[benchmarks]"
fi

echo "Setup complete! Activate environment with: conda activate tensor-consensus"
