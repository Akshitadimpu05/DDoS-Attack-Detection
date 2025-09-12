#!/bin/bash

# Setup script for IoT DDoS Detection with CNN-TCN
echo "Setting up IoT DDoS Detection Environment..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

echo "Environment setup complete!"
echo "To activate the environment, run: source venv/bin/activate"
echo "To download the dataset, run: python download_dataset.py"
echo "To train the model, run: python train_model.py"
