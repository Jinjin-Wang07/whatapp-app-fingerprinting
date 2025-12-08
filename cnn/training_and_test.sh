#!/bin/bash

epochs=10

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_MODEL_SCRIPT="${SCRIPT_DIR}/run_model.py"

# Set DATASET_DIR from parameter or use default
DATASET_DIR="${1:-/dataset}"
# Path to tf-records
TF_OUT_DIR="${DATASET_DIR}/tf_output"

window_size=60
step=5
folder="w${window_size}_s${step}"

echo "Start training"
echo "Training epochs: ${epochs}"
python "${RUN_MODEL_SCRIPT}" --input "${TF_OUT_DIR}/${folder}" --train True --model_name "rgb_apps-all" --epochs ${epochs}

echo "Testing model..."
python "${RUN_MODEL_SCRIPT}" --input "${TF_OUT_DIR}/${folder}" --model_name "rgb_apps-all"


echo "Finished"