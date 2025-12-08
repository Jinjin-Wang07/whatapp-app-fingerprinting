#!/bin/bash

# Set DATASET_DIR from parameter or use default
DATASET_DIR="${1:-/dataset}"

./data_prepare.sh "${DATASET_DIR}"
./training_and_test.sh "${DATASET_DIR}"