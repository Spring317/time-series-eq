#!/bin/bash

# Quick start script for DAS seismic classification pipeline

echo "=========================================="
echo "DAS Seismic Classification Pipeline Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

# Install dependencies
echo ""
echo "Installing dependencies..."
read -p "Install required packages? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    pip install -r requirements.txt
fi

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p models logs cache

# Check for data files
echo ""
echo "Checking for data files..."
data_count=$(ls -1 data/*.h5 2>/dev/null | wc -l)
echo "Found $data_count HDF5 files in data/ directory"

# Create labels file
if [ ! -f "labels.json" ]; then
    echo ""
    echo "No labels.json found!"
    echo "This dataset has predefined labels (13 earthquakes, 6 quarry blasts)"
    read -p "Generate labels automatically from dataset documentation? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]
    then
        python3 generate_labels.py
    else
        echo "You can run 'python3 generate_labels.py' later to create labels"
    fi
else
    echo ""
    echo "✓ Found labels.json"
fi

# Check configuration
if [ ! -f "config.yaml" ]; then
    echo ""
    echo "ERROR: config.yaml not found!"
    echo "Please ensure config.yaml exists before running training"
    exit 1
else
    echo "✓ Found config.yaml"
fi

# Summary
echo ""
echo "=========================================="
echo "Setup Summary"
echo "=========================================="
echo "Data files: $data_count"
echo "Labels file: $([ -f labels.json ] && echo 'Yes' || echo 'No')"
echo "Config file: $([ -f config.yaml ] && echo 'Yes' || echo 'No')"
echo ""

# Check if ready to train
if [ -f "labels.json" ] && [ -f "config.yaml" ] && [ $data_count -gt 0 ]; then
    echo "✓ Ready to train!"
    echo ""
    read -p "Start training now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]
    then
        echo ""
        echo "Starting training..."
        python3 main.py
    else
        echo ""
        echo "To start training later, run:"
        echo "  python3 main.py"
        echo ""
        echo "To monitor training, run in another terminal:"
        echo "  tensorboard --logdir logs"
    fi
else
    echo "⚠ Not ready to train yet!"
    echo ""
    echo "Next steps:"
    [ $data_count -eq 0 ] && echo "  1. Add HDF5 files to data/ directory"
    [ ! -f "labels.json" ] && echo "  2. Run 'python3 generate_labels.py' to create labels"
    [ ! -f "config.yaml" ] && echo "  3. Create config.yaml file"
    echo "  4. Run 'python3 main.py' to start training"
fi

echo ""
