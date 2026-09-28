#!/bin/bash

# =============================================================================
# QA-SQL Unordered-Comparison Evaluation Script
# =============================================================================
# Runs evaluation/evaluation_unordered.py which ignores BOTH row order AND
# column order when comparing predicted vs ground-truth result sets.
#
# Default behavior: compares the same FILE_NAME (selected.json) across all
# directories listed in COMPARE_DIRS below, and prints a side-by-side summary.
#
# Usage:
#   ./scripts/run_evaluation_unordered.sh                         # compare COMPARE_DIRS
#   ./scripts/run_evaluation_unordered.sh -o output/ver1          # single dir only
#   ./scripts/run_evaluation_unordered.sh -f candidate_full_schema.json
#   ./scripts/run_evaluation_unordered.sh --start 0 --end 13      # slice first 13 keys
#   ./scripts/run_evaluation_unordered.sh -a                      # evaluate all 6 files
#                                                                   (single-dir mode only)
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Defaults
DB_ROOT_PATH="${PROJECT_DIR}/data/bird_data/dev_databases/"
DIFF_JSON_PATH="${PROJECT_DIR}/data/bird_data/dev.json"
GROUND_TRUTH_PATH="${PROJECT_DIR}/data/bird_data/"
FILE_NAME="selected.json"
EVAL_ALL=false
NUM_CPUS=4
META_TIME_OUT=30.0
DATA_MODE="dev"
START_IDX="0"
END_IDX="60"

# Directories to compare side-by-side when no -o is supplied.
# Edit this list to change which predictions are compared.
COMPARE_DIRS=(
    "${PROJECT_DIR}/output/Sonnet4-6/"
    "${PROJECT_DIR}/output/claude_headless_v2/"
)

# Optional short labels paired 1:1 with COMPARE_DIRS. Leave blank to use basename.
COMPARE_LABELS=(
    "final"
    "headless_v2"
)

# Populated if -o is used (single-dir mode).
OUTPUT_DIR=""

ALL_FILES=(
    "selected.json"
    "candidate_full_schema.json"
    "candidate_sme_metadata.json"
    "candidate_minimal_profile.json"
    "candidate_focused_schema.json"
    "candidate_full_profile.json"
)

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

show_help() {
    cat <<EOF
QA-SQL Unordered-Comparison Evaluation Script

Usage: $0 [OPTIONS]

Options:
  -a, --all             Evaluate ALL 6 prediction files (single-dir mode only)
  -f, --file FILE       Evaluate a specific file (default: selected.json)
  -o, --output-dir DIR  Single prediction directory (disables multi-dir compare)
  -c, --cpus N          Parallel workers (default: 4)
  --timeout N           SQL execution timeout in seconds (default: 30)
  --start N             Positional slice start over file-order keys (0-indexed, inclusive)
  --end N               Positional slice end over file-order keys (exclusive)
  -h, --help            Show this help

Compare dirs (edit COMPARE_DIRS in this file to change):
EOF
    for d in "${COMPARE_DIRS[@]}"; do echo "  - ${d}"; done
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -a|--all) EVAL_ALL=true; shift ;;
        -f|--file) FILE_NAME="$2"; EVAL_ALL=false; shift 2 ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            [[ ! "$OUTPUT_DIR" = /* ]] && OUTPUT_DIR="${PROJECT_DIR}/${OUTPUT_DIR}"
            shift 2 ;;
        -c|--cpus) NUM_CPUS="$2"; shift 2 ;;
        --timeout) META_TIME_OUT="$2"; shift 2 ;;
        --start) START_IDX="$2"; shift 2 ;;
        --end) END_IDX="$2"; shift 2 ;;
        -h|--help) show_help; exit 0 ;;
        *) echo -e "${RED}Unknown option: $1${NC}"; show_help; exit 1 ;;
    esac
done

[[ ! -d "$DB_ROOT_PATH" ]] && { echo -e "${RED}Error: $DB_ROOT_PATH not found${NC}"; exit 1; }
[[ ! -f "$DIFF_JSON_PATH" ]] && { echo -e "${RED}Error: $DIFF_JSON_PATH not found${NC}"; exit 1; }

cd "$PROJECT_DIR"

build_range_args() {
    RANGE_ARGS=""
    # START_IDX / END_IDX are positional slice bounds over the prediction file's
    # keys in file order (half-open [start, end)). So START_IDX=0 END_IDX=13
    # picks the first 13 entries, whatever their key values are.
    [[ -n "$START_IDX" ]] && RANGE_ARGS="$RANGE_ARGS --start $START_IDX"
    [[ -n "$END_IDX" ]]   && RANGE_ARGS="$RANGE_ARGS --end $END_IDX"
}

# Single-dir evaluation (used by -a or -o modes).
evaluate_file_single_dir() {
    local dir="$1"
    local file_name="$2"
    if [[ ! -f "${dir}${file_name}" ]]; then
        echo -e "${RED}  Skipping ${file_name} (not found at ${dir})${NC}"
        return 1
    fi

    echo -e "${YELLOW}Evaluating (unordered): ${file_name} @ ${dir}${NC}"
    echo "----------------------------------------------"
    build_range_args

    python -u evaluation/evaluation_unordered.py \
        --db_root_path "${DB_ROOT_PATH}" \
        --predicted_sql_path "${dir}" \
        --ground_truth_path "${GROUND_TRUTH_PATH}" \
        --data_mode "${DATA_MODE}" \
        --num_cpus "${NUM_CPUS}" \
        --mode_gt gt \
        --mode_predict gpt \
        --diff_json_path "${DIFF_JSON_PATH}" \
        --meta_time_out "${META_TIME_OUT}" \
        --file_name "${file_name}" \
        $RANGE_ARGS
    echo ""
    return 0
}

# Multi-dir comparison (default): hand all dirs to the Python evaluator at once
# so it can produce one side-by-side summary.
evaluate_compare() {
    local file_name="$1"
    local dirs=()
    local labels=()
    local missing=0

    for i in "${!COMPARE_DIRS[@]}"; do
        local d="${COMPARE_DIRS[$i]%/}/"
        if [[ -f "${d}${file_name}" ]]; then
            dirs+=("$d")
            labels+=("${COMPARE_LABELS[$i]:-$(basename "$d")}")
        else
            echo -e "${RED}  Missing ${file_name} at ${d} (excluded from comparison)${NC}"
            ((missing++)) || true
        fi
    done

    if [[ ${#dirs[@]} -eq 0 ]]; then
        echo -e "${RED}No prediction files to compare. Aborting.${NC}"
        exit 1
    fi

    echo -e "${YELLOW}Comparing (unordered): ${file_name} across ${#dirs[@]} directories${NC}"
    echo "----------------------------------------------"
    build_range_args

    python -u evaluation/evaluation_unordered.py \
        --db_root_path "${DB_ROOT_PATH}" \
        --predicted_sql_path "${dirs[@]}" \
        --labels "${labels[@]}" \
        --ground_truth_path "${GROUND_TRUTH_PATH}" \
        --data_mode "${DATA_MODE}" \
        --num_cpus "${NUM_CPUS}" \
        --mode_gt gt \
        --mode_predict gpt \
        --diff_json_path "${DIFF_JSON_PATH}" \
        --meta_time_out "${META_TIME_OUT}" \
        --file_name "${file_name}" \
        $RANGE_ARGS
    echo ""
}

echo ""
echo "=============================================="
echo "QA-SQL Unordered-Comparison Evaluation"
echo "=============================================="
echo "Database path:    ${DB_ROOT_PATH}"
echo "Ground truth:     ${GROUND_TRUTH_PATH}dev.sql"
echo "Difficulty file:  ${DIFF_JSON_PATH}"
echo "CPUs:             ${NUM_CPUS}"
echo "Timeout:          ${META_TIME_OUT}s"
if [[ -n "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="${OUTPUT_DIR%/}/"
    if [[ "$EVAL_ALL" == true ]]; then
        echo "Mode:             Single dir, ALL files (${OUTPUT_DIR})"
    else
        echo "Mode:             Single dir, one file (${OUTPUT_DIR}${FILE_NAME})"
    fi
else
    echo "Mode:             Compare across ${#COMPARE_DIRS[@]} directories"
    for i in "${!COMPARE_DIRS[@]}"; do
        echo "  [${COMPARE_LABELS[$i]:-$(basename "${COMPARE_DIRS[$i]}")}] ${COMPARE_DIRS[$i]}"
    done
fi
[[ -n "$START_IDX" || -n "$END_IDX" ]] && echo "Slice:            [${START_IDX:-0}, ${END_IDX:-end}) (positional, in file order)"
echo "=============================================="
echo ""

# Dispatch
if [[ -n "$OUTPUT_DIR" ]]; then
    # Single-dir mode
    if [[ "$EVAL_ALL" == true ]]; then
        success=0
        total=${#ALL_FILES[@]}
        for file in "${ALL_FILES[@]}"; do
            evaluate_file_single_dir "$OUTPUT_DIR" "$file" && ((success++)) || true
        done
        echo "=============================================="
        echo -e "${GREEN}Done: ${success}/${total} files evaluated${NC}"
        echo "=============================================="
    else
        evaluate_file_single_dir "$OUTPUT_DIR" "$FILE_NAME"
        echo -e "${GREEN}Done: ${FILE_NAME}${NC}"
    fi
else
    # Multi-dir compare mode
    if [[ "$EVAL_ALL" == true ]]; then
        echo -e "${RED}-a/--all is only supported in single-dir mode (use -o DIR with -a).${NC}"
        exit 1
    fi
    evaluate_compare "$FILE_NAME"
    echo -e "${GREEN}Done: comparison for ${FILE_NAME}${NC}"
fi
