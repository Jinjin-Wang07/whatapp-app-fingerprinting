#!/bin/bash

# Set DATASET_DIR from parameter or use default
DATASET_DIR="${1:-/dataset}"
RESULT_PATH="${DATASET_DIR}/tf_output/w60_s5/results/"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GET_RESULT_SCRIPT="${SCRIPT_DIR}/get_results.py"

python "${GET_RESULT_SCRIPT}" --result_path "${RESULT_PATH}"