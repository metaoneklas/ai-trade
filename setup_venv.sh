#!/bin/bash

# Project: AI-RL Trading Agent
# Description: Script to create a virtual environment and install dependencies.

VENV_NAME="venv"

echo "[*] Creating virtual environment: $VENV_NAME..."

# Check if python3 is installed
if ! command -v python3 &> /dev/null
then
    echo "[!] Error: python3 could not be found. Please install it."
    exit 1
fi

# Create venv
python3 -m venv $VENV_NAME

# Check if creation was successful
if [ ! -d "$VENV_NAME" ]; then
    echo "[!] Error: Failed to create virtual environment."
    exit 1
fi

echo "[+] Virtual environment created successfully."

# Install dependencies
echo "[*] Installing dependencies from requirements.txt..."
source $VENV_NAME/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "[+] Dependencies installed successfully."
    echo ""
    echo "===================================================="
    echo "Setup Complete!"
    echo "To activate the environment, run:"
    echo "  source $VENV_NAME/bin/activate"
    echo "===================================================="
else
    echo "[!] Error: Failed to install dependencies."
    exit 1
fi
