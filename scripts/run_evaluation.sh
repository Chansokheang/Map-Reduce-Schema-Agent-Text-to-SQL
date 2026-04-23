#!/bin/bash

# =============================================================================
# QA-SQL Evaluation Script
# =============================================================================
# Usage:
#   ./scripts/run_evaluation.sh                              # Evaluate ALL files (default)
#   ./scripts/run_evaluation.sh -f selected.json             # Specific file only
#   ./scripts/run_evaluation.sh -o output/qwen2.5-coder-32b  # Custom output dir
#   ./scripts/run_evaluation.sh -t acc                       # Accuracy only
#   ./scripts/run_evaluation.sh -t ves                       # VES only
#   ./scripts/run_evaluation.sh -t both                      # Both (default)
# =============================================================================

set -e

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Default Configuration (relative to project directory)
DB_ROOT_PATH="${PROJECT_DIR}/data/bird_data/dev_databases/"
DIFF_JSON_PATH="${PROJECT_DIR}/data/bird_data/dev.json"
GROUND_TRUTH_PATH="${PROJECT_DIR}/data/bird_data/"
# OUTPUT_DIR="${PROJECT_DIR}/output/claude_headless_v2/"
OUTPUT_DIR="${PROJECT_DIR}/output/final/"
FILE_NAME=""
EVAL_ALL=true
EVAL_TYPE="acc"
NUM_CPUS=4
META_TIME_OUT=30.0
ITERATE_NUM=100
DATA_MODE="dev"
START_IDX=""
END_IDX=""

# All prediction files to evaluate
ALL_FILES=(
    "selected.json"
    # "candidate_full_schema.json"
    # "candidate_sme_metadata.json"
    # "candidate_minimal_profile.json"
    # "candidate_focused_schema.json"
    # "candidate_full_profile.json"
)

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Help function
show_help() {
    echo "QA-SQL Evaluation Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -a, --all             Evaluate ALL prediction files (default behavior)"
    echo "  -f, --file FILE       Evaluate only a specific file"
    echo "  -o, --output-dir DIR  Output directory containing predictions (default: output/ver1)"
    echo "  -t, --type TYPE       Evaluation type: acc, ves, both (default: both)"
    echo "  -c, --cpus N          Number of CPUs for parallel execution (default: 4)"
    echo "  --timeout N           SQL execution timeout in seconds (default: 30)"
    echo "  --iterate N           Iterations for VES measurement (default: 100)"
    echo "  --start N             Start index for evaluation range (0-indexed)"
    echo "  --end N               End index for evaluation range (exclusive)"
    echo "  -h, --help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Evaluate ALL files in output/ver1/"
    echo "  $0 -f selected.json                   # Evaluate only selected.json"
    echo "  $0 -o output/qwen2.5-coder-32b        # Evaluate ALL files from custom dir"
    echo "  $0 --start 0 --end 10                 # Evaluate only questions 0-9"
    echo "  $0 -o output/qwen2.5-coder-32b --start 0 --end 100  # Custom dir, range 0-99"
    echo "  $0 -t acc -c 8                        # Accuracy only, 8 CPUs"
    echo ""
    echo "Prediction files evaluated (when using --all or no -f flag):"
    echo "  - selected.json                (Judge's final selection)"
    echo "  - candidate_full_schema.json   (Strategy 1)"
    echo "  - candidate_sme_metadata.json  (Strategy 2)"
    echo "  - candidate_minimal_profile.json (Strategy 3)"
    echo "  - candidate_focused_schema.json (Strategy 4)"
    echo "  - candidate_full_profile.json  (Strategy 5)"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -a|--all)
            EVAL_ALL=true
            shift
            ;;
        -f|--file)
            FILE_NAME="$2"
            EVAL_ALL=false
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            # If relative path, make it absolute from project dir
            if [[ ! "$OUTPUT_DIR" = /* ]]; then
                OUTPUT_DIR="${PROJECT_DIR}/${OUTPUT_DIR}"
            fi
            shift 2
            ;;
        -t|--type)
            EVAL_TYPE="$2"
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
        --iterate)
            ITERATE_NUM="$2"
            shift 2
            ;;
        --start)
            START_IDX="$2"
            shift 2
            ;;
        --end)
            END_IDX="$2"
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

# Ensure output dir ends with /
OUTPUT_DIR="${OUTPUT_DIR%/}/"

# Validate paths
if [[ ! -d "$DB_ROOT_PATH" ]]; then
    echo -e "${RED}Error: Database directory not found: $DB_ROOT_PATH${NC}"
    exit 1
fi

if [[ ! -f "$DIFF_JSON_PATH" ]]; then
    echo -e "${RED}Error: dev.json not found: $DIFF_JSON_PATH${NC}"
    exit 1
fi

# Change to project directory
cd "$PROJECT_DIR"

# Function to evaluate a single file
evaluate_file() {
    local file_name="$1"

    if [[ ! -f "${OUTPUT_DIR}${file_name}" ]]; then
        echo -e "${RED}  Skipping ${file_name} (not found)${NC}"
        return 1
    fi

    echo -e "${YELLOW}Evaluating: ${file_name}${NC}"
    echo "----------------------------------------------"

    # Build range arguments if specified
    RANGE_ARGS=""
    if [[ -n "$START_IDX" ]]; then
        RANGE_ARGS="$RANGE_ARGS --start $START_IDX"
    fi
    if [[ -n "$END_IDX" ]]; then
        RANGE_ARGS="$RANGE_ARGS --end $END_IDX"
    fi

    # Build and run accuracy command
    if [[ "$EVAL_TYPE" == "acc" || "$EVAL_TYPE" == "both" ]]; then
        python -u evaluation/evaluation.py \
            --db_root_path "${DB_ROOT_PATH}" \
            --predicted_sql_path "${OUTPUT_DIR}" \
            --ground_truth_path "${GROUND_TRUTH_PATH}" \
            --data_mode "${DATA_MODE}" \
            --num_cpus "${NUM_CPUS}" \
            --mode_gt gt \
            --mode_predict gpt \
            --diff_json_path "${DIFF_JSON_PATH}" \
            --meta_time_out "${META_TIME_OUT}" \
            --file_name "${file_name}" \
            $RANGE_ARGS
    fi

    # Build and run VES command
    if [[ "$EVAL_TYPE" == "ves" || "$EVAL_TYPE" == "both" ]]; then
        python -u evaluation/evaluation_ves.py \
            --db_root_path "${DB_ROOT_PATH}" \
            --predicted_sql_path "${OUTPUT_DIR}" \
            --ground_truth_path "${GROUND_TRUTH_PATH}" \
            --data_mode "${DATA_MODE}" \
            --num_cpus "${NUM_CPUS}" \
            --mode_gt gt \
            --mode_predict gpt \
            --diff_json_path "${DIFF_JSON_PATH}" \
            --meta_time_out "${META_TIME_OUT}" \
            --iterate_num "${ITERATE_NUM}" \
            --file_name "${file_name}" \
            $RANGE_ARGS
    fi

    echo ""
    return 0
}

# Display configuration
echo ""
echo "=============================================="
echo "QA-SQL Evaluation"
echo "=============================================="
echo "Output directory: ${OUTPUT_DIR}"
echo "Database path:    $DB_ROOT_PATH"
echo "Ground truth:     ${GROUND_TRUTH_PATH}dev.sql"
echo "Difficulty file:  $DIFF_JSON_PATH"
echo "Eval type:        $EVAL_TYPE"
echo "CPUs:             $NUM_CPUS"
echo "Timeout:          ${META_TIME_OUT}s"
if [[ "$EVAL_TYPE" == "ves" || "$EVAL_TYPE" == "both" ]]; then
    echo "VES iterations:   $ITERATE_NUM"
fi
if [[ "$EVAL_ALL" == true ]]; then
    echo "Mode:             Evaluate ALL files"
else
    echo "Mode:             Single file (${FILE_NAME})"
fi
if [[ -n "$START_IDX" || -n "$END_IDX" ]]; then
    echo "Range:            [${START_IDX:-0}, ${END_IDX:-end})"
fi
echo "=============================================="
echo ""

# Run evaluations
if [[ "$EVAL_ALL" == true ]]; then
    # Evaluate all files
    echo -e "${GREEN}Evaluating all prediction files...${NC}"
    echo ""

    success_count=0
    total_count=${#ALL_FILES[@]}

    for file in "${ALL_FILES[@]}"; do
        if evaluate_file "$file"; then
            ((success_count++))
        fi
    done

    echo "=============================================="
    echo -e "${GREEN}Evaluation completed: ${success_count}/${total_count} files evaluated${NC}"
    echo "=============================================="
else
    # Evaluate single file
    if [[ ! -f "${OUTPUT_DIR}${FILE_NAME}" ]]; then
        echo -e "${RED}Error: Prediction file not found: ${OUTPUT_DIR}${FILE_NAME}${NC}"
        echo ""
        echo "Available files in ${OUTPUT_DIR}:"
        ls -la "$OUTPUT_DIR" 2>/dev/null || echo "  Directory does not exist"
        exit 1
    fi

    evaluate_file "$FILE_NAME"
    echo -e "${GREEN}Evaluation completed for: ${FILE_NAME}${NC}"
fi
