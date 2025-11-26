RESULT_PATH="/dataset/tf_output/w60_s5/results/"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GET_RESULT_SCRIPT="${SCRIPT_DIR}/get_results.py"

python "${GET_RESULT_SCRIPT}" --result_path "${RESULT_PATH}"