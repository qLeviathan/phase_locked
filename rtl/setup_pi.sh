#!/bin/bash
# =============================================================================
# Setup Script for Zeckbit Cascade on Raspberry Pi
# =============================================================================
# Target: Raspberry Pi 3B/4, Kano, or any ARM Linux with Verilator
#
# Usage: ./setup_pi.sh
# =============================================================================

set -e

echo "=============================================="
echo "  Zeckbit Cascade - Raspberry Pi Setup"
echo "=============================================="
echo ""

# Check architecture
ARCH=$(uname -m)
echo "Architecture: $ARCH"

# Update package list
echo ""
echo ">>> Updating package list..."
sudo apt-get update

# Install Verilator and dependencies
echo ""
echo ">>> Installing Verilator and build tools..."
sudo apt-get install -y \
    verilator \
    build-essential \
    git \
    make \
    g++ \
    libfl2 \
    libfl-dev \
    zlib1g \
    zlib1g-dev

# Check Verilator version
echo ""
echo ">>> Verilator version:"
verilator --version

# Install optional: GTKWave for waveform viewing
echo ""
read -p "Install GTKWave for waveform viewing? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    sudo apt-get install -y gtkwave
fi

# Create build directory
echo ""
echo ">>> Creating build structure..."
mkdir -p obj_dir
mkdir -p sim
mkdir -p results

# Build the design
echo ""
echo ">>> Building Zeckbit Cascade..."
make clean 2>/dev/null || true
make build

# Run tests
echo ""
echo ">>> Running verification tests..."
make run | tee results/test_output.txt

# Benchmark
echo ""
echo ">>> Running benchmark..."
./benchmark.sh 2>/dev/null || echo "(benchmark script will be created)"

echo ""
echo "=============================================="
echo "  Setup Complete!"
echo "=============================================="
echo ""
echo "Files:"
echo "  results/test_output.txt - Test results"
echo "  obj_dir/Vzeck_top       - Executable"
echo ""
echo "Commands:"
echo "  make run     - Run tests"
echo "  make wave    - Run with waveform capture"
echo "  make clean   - Clean build"
echo ""
