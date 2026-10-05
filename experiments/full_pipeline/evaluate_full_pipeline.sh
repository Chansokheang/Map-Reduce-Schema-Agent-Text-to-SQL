#!/bin/bash
# =============================================================================
# Evaluate a run of experiments/full_pipeline/run_full_pipeline.sh on the dev set.
#
# Scores <output-dir>/selected.json and <output-dir>/question_form_postprocess/
# selected_postprocessed.json side by side, by difficulty, and lists what the post-process
# changed. Predictions are matched to gold by question key, so partial ranges are safe.
#
# Usage (from anywhere):
#   bash experiments/full_pipeline/evaluate_full_pipeline.sh                   # defaults below
#   bash experiments/full_pipeline/evaluate_full_pipeline.sh --start 14 --end 28
#   bash experiments/full_pipeline/evaluate_full_pipeline.sh -o ./output/full_v3/ --all
#
# Options (defaults are set in the "Defaults" block below):
#   -o, --output-dir PATH   Pipeline output folder (default: ./output/full_v2/)
#   --start N --end M       Question range [N, M) (default: 0 and 10)
#   --all                   Evaluate every question in selected.json
#   --pp-out PATH           Post-process folder (default: <output-dir>/question_form_postprocess)
#   --dev-json PATH         Default: ./data/bird_data/dev.json
#   --databases-dir PATH    Default: ./data/bird_data/dev_databases
#   --workers N             Parallel SQL executions (default: 4)
#   --timeout S             Per-query timeout in seconds (default: 30)
#   --report-dir PATH       Where to write the JSON report (default: <output-dir>/evaluation)
#
# Writes <report-dir>/evaluation_<start>_<end>.json
# =============================================================================
set -e
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ---- Defaults (edit here or override on the command line) ----
# OUTPUT_DIR="${PROJECT_DIR}/output/claude_headless_v6/"   # -o
# OUTPUT_DIR="./output/cg_v2/"   # -o
# OUTPUT_DIR="./output/all_v1/"   # -o
# OUTPUT_DIR="./output/gr_v1/"   # -o (better from 0 -1000, not 1000 - 1300)
# OUTPUT_DIR="./output/opus_v1/"   # -o
# OUTPUT_DIR="./output/rules_on_v2/"   # -o
OUTPUT_DIR="./output/rules_on_v1/"   # -o
START_IDX="0"                    # --start  (first question, inclusive)
END_IDX="1534"                     # --end    (last question, exclusive)
# Leave START_IDX and END_IDX empty ("") to evaluate every question in selected.json.

ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        -o|--output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --start) START_IDX="$2"; shift 2 ;;
        --end) END_IDX="$2"; shift 2 ;;
        --all) START_IDX=""; END_IDX=""; shift ;;
        --pp-out|--dev-json|--databases-dir|--workers|--timeout|--report-dir) ARGS+=("$1" "$2"); shift 2 ;;
        -h|--help) sed -n 2,24p "$0"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ -n "$START_IDX" ]]; then ARGS+=(--start "$START_IDX"); fi
if [[ -n "$END_IDX" ]]; then ARGS+=(--end "$END_IDX"); fi

cd "$PROJECT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Range:            [${START_IDX:-first}, ${END_IDX:-last})"
python -m experiments.full_pipeline.evaluate_outputs --output-dir "$OUTPUT_DIR" "${ARGS[@]}"
