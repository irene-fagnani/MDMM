#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Check if toy dataset exists, create if missing
if [ ! -d "datasets/toy_dataset" ]; then
    echo "Creating toy dataset..."
    python create_toy_dataset.py
else
    echo "Toy dataset already exists, skipping creation."
fi

# Run the tests
echo "Running KL loss test..."
python kl_loss_test.py

echo "Running consistency test..."
python consistency_test.py

echo "Running integration test..."
python integration_test.py

echo "All tests completed successfully."
