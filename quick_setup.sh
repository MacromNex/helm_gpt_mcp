#!/bin/bash
# Quick Setup Script for HELM-GPT MCP
# HELM-GPT: De novo macrocyclic peptide design using generative pre-trained transformer
# Provides HELM-to-SMILES conversion and permeability prediction
# Source: https://github.com/charlesxu90/helm-gpt

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Setting up HELM-GPT MCP ==="

# Step 1: Create Python environment
echo "[1/4] Creating Python 3.10 environment..."
(command -v mamba >/dev/null 2>&1 && mamba create -p ./env python=3.10 pip -y) || \
(command -v conda >/dev/null 2>&1 && conda create -p ./env python=3.10 pip -y) || \
(echo "Warning: Neither mamba nor conda found, creating venv instead" && python3 -m venv ./env)

# Step 2: Install core dependencies
echo "[2/4] Installing core dependencies..."
./env/bin/pip install loguru click pandas numpy tqdm

# Step 3: Install fastmcp
echo "[3/4] Installing fastmcp..."
./env/bin/pip install --force-reinstall --no-cache-dir fastmcp

# Step 4: Install scientific packages
echo "[4/4] Installing scientific packages..."
./env/bin/pip install scikit-learn scipy

echo ""
echo "=== HELM-GPT MCP Setup Complete ==="
echo "Note: For full HELM-GPT functionality (training/generation), additional setup required:"
echo "  - Install RDKit: mamba install -c conda-forge rdkit"
echo "  - Install PyTorch: pip install torch torchvision"
echo "  - See repo README for complete instructions"
echo "To run the MCP server: ./env/bin/python src/server.py"
