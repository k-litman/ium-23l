#!/usr/bin/env bash

# Prepare data
echo "Preparing data..."
python3 transformation/transform_sessions.py

# Create simple model
echo "Creating simple model..."

echo "Performing clustering..."
python3 src/models/simple/clustering_simple.py

echo "Performing classification..."
python3 src/models/simple/mlp_classifier_simple_gpu_load_from_model.py

# Create advanced model
echo "Creating advanced model..."

echo "Performing clustering..."
python3 src/models/advanced/clustering_advanced.py

echo "Performing classification..."
python3 src/models/advanced/mlp_classifier_advanced_gpu_load_from_model.py