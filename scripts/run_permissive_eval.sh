#!/bin/bash

# =============================================================================
# Permissive Evaluation Script
# =============================================================================
# Runs evaluation/permissive_eval.py and reports four buckets per question:
#   strict        — exact tuple match (BIRD-pass)
#   column_order  — same values, different SELECT column order
#   subset        — gold values are inside pred values (pred over-projected)
#   wrong         — real logic error
#   error         — SQL failed to execute or timed out
#
# Usage:
#   ./scripts/run_permissive_eval.sh                              # Default file (selected.json) in default OUTPUT_DIR
#   ./scripts/run_permissive_eval.sh -f candidate_full_profile.json
#   ./scripts/run_permissive_eval.sh -o output/claude_headless_v3 -f selected.json
#   ./scripts/run_permissive_eval.sh -c 8                         # 8 CPUs
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Defaults — mirror run_evaluation.sh so paths line up
DB_ROOT_PATH="${PROJECT_DIR}/data/bird_data/dev_databases/"
DIFF_JSON_PATH="${PROJECT_DIR}/data/bird_data/dev.json"
GROUND_TRUTH_PATH="${PROJECT_DIR}/data/bird_data/"
OUTPUT_DIR="${PROJECT_DIR}/output/claude_headless_v3/"
FILE_NAME="selected.json"
NUM_CPUS=4
META_TIME_OUT=30.0
DATA_MODE="dev"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

show_help() {
    echo "Permissive Evaluation Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -f, --file FILE       Prediction file name (default: selected.json)"
    echo "  -o, --output-dir DIR  Output dir containing the file (default: output/claude_headless_v3/)"
    echo "  -c, --cpus N          Parallel CPUs (default: 4)"
    echo "  --timeout N           Per-query timeout seconds (default: 30)"
    echo "  -h, --help            Show this help"
    echo ""
    echo "Examples:"
    echo "  $0                                     # Evaluate output/claude_headless_v3/selected.json"
    echo "  $0 -f candidate_full_profile.json      # Different file in default dir"
    echo "  $0 -o output/gemma4 -f selected.json   # Different dir + file"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -f|--file)
            FILE_NAME="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            if [[ ! "$OUTPUT_DIR" = /* ]]; then
                OUTPUT_DIR="${PROJECT_DIR}/${OUTPUT_DIR}"
            fi
            shift 2
            ;;
        -c|--cpus)
            NUM_CPUS="$2"
            shift 2
            ;;
        --timeout)
            META_TIME_OUT="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# Normalize trailing slash on OUTPUT_DIR
OUTPUT_DIR="${OUTPUT_DIR%/}/"

# Validate
if [[ ! -d "$DB_ROOT_PATH" ]]; then
    echo -e "${RED}Error: DB dir not found: $DB_ROOT_PATH${NC}"
    exit 1
fi
if [[ ! -f "$DIFF_JSON_PATH" ]]; then
    echo -e "${RED}Error: dev.json not found: $DIFF_JSON_PATH${NC}"
    exit 1
fi
if [[ ! -f "${OUTPUT_DIR}${FILE_NAME}" ]]; then
    echo -e "${RED}Error: prediction file not found: ${OUTPUT_DIR}${FILE_NAME}${NC}"
    echo "Available files:"
    ls -la "$OUTPUT_DIR" 2>/dev/null || echo "  (output dir does not exist)"
    exit 1
fi

cd "$PROJECT_DIR"

echo ""
echo "=============================================="
echo "Permissive Evaluation"
echo "=============================================="
echo "Output dir:    $OUTPUT_DIR"
echo "File:          $FILE_NAME"
echo "DB root:       $DB_ROOT_PATH"
echo "dev.json:      $DIFF_JSON_PATH"
echo "Ground truth:  ${GROUND_TRUTH_PATH}${DATA_MODE}.sql"
echo "CPUs:          $NUM_CPUS"
echo "Timeout:       ${META_TIME_OUT}s"
echo "=============================================="
echo ""

echo -e "${YELLOW}Running permissive evaluator...${NC}"
echo ""

python -u evaluation/permissive_eval.py \
    --db_root_path "${DB_ROOT_PATH}" \
    --predicted_sql_path "${OUTPUT_DIR}" \
    --ground_truth_path "${GROUND_TRUTH_PATH}" \
    --data_mode "${DATA_MODE}" \
    --num_cpus "${NUM_CPUS}" \
    --mode_gt gt \
    --mode_predict gpt \
    --diff_json_path "${DIFF_JSON_PATH}" \
    --meta_time_out "${META_TIME_OUT}" \
    --file_name "${FILE_NAME}"

echo ""
echo -e "${GREEN}Done.${NC}"
