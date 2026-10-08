#!/bin/bash
# Re-exec under bash if invoked via `sh` (dash lacks [[ ]] and BASH_SOURCE).
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
# =============================================================================
# QA-SQL full run: pipeline -> question-form post-process -> projection review
# =============================================================================
# The measured configuration (BIRD dev, judge-only: 1167/1534 = 76.08%):
#   native headless CLI, RULE 0 on all databases, RULE L/M/N, RULE P,
#   Opus judge, judge id/sql reconciliation, stage logging, fixer OFF.
#
#   bash scripts/run_full.sh -o ./output/run_v1/ -b 0 1534
#   bash scripts/run_full.sh -o ./output/test_v1/ --test-set --all
#   bash scripts/run_full.sh -o ./output/x/ -q 5 --with-fixer
#
# Steps 2 and 3 are src.postprocess.postprocess and src.postprocess.review, with no
# experiments/ dependency. Step 1 still loads the runtime patches in
# experiments/full_pipeline/patches/sitecustomize.py: those ARE --count-all,
# --generation-rules, --column-family, --judge-id-fix and --stage-outputs. Until those
# move as well, this script is not self-contained.
#
# TEST SET: --test-set swaps questions/tables/databases/schemas to their test equivalents.
# Generate the per-DB schema JSONs FIRST, or the run silently uses dev schemas:
#   python src/processing/extract_schema.py --output-dir ./data/bird_data/test_schemas
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

DATA_DIR="./data/bird_data/"
DEV_JSON="./data/bird_data/dev.json"
TABLES_JSON="./data/bird_data/dev_tables.json"
DATABASES_DIR="./data/bird_data/dev_databases"
SCHEMAS_DIR="./data/bird_data/schemas"
OUTPUT_DIR="./output/run_v1/"

# Column descriptions injected into the schema JSONs by extract_schema.py. The shipped file
# keys on "{database}|{table}|{column}" and covers the 11 DEV databases only, so on another
# dataset every lookup misses and columns come out with no description — silently. Point this
# at a matching file, or pass "" to skip it and make the absence explicit.
COLUMN_MEANING="./data/column_meaning.json"
EXTRACT_SCHEMA=false      # run extract_schema.py first (step 0)
FINAL_COPY=""             # also copy the final predictions here

MODEL="claude-sonnet-4-6"
JUDGE_MODEL="claude-opus-5-5"
THRESHOLD=0.5
TIMEOUT=30
MAX_WORKERS=4
VERBOSE=false

QUESTION=
BATCH_MODE=true
BATCH_START=750
BATCH_END=950

NO_FIXER=true
PROFILE=false
COLUMN_FAMILY=true
REVIEW=true
PP=true
PP_WORKERS=""
PR_WORKERS=""

show_help() {
    sed -n "5,24p" "$0"
    echo ""
    echo "Question Options:"
    echo "  -q, --question N        Run single question N (default: 0)"
    echo "  -b, --batch START END   Run questions [START, END)"
    echo "  --start N               Range start, inclusive (default: 0)"
    echo "  --end N                 Range end, EXCLUSIVE; omit to run to the last question"
    echo "  --all                   Run every question in the questions file"
    echo ""
    echo "Path Options:"
    echo "  -d, --data-dir PATH     BIRD data root"
    echo "  -o, --output-dir PATH   Output directory"
    echo "  --dev-json PATH         Questions JSON (dev.json / test.json / a subset)"
    echo "  --tables-json PATH      Combined tables JSON, captured for downstream eval"
    echo "  --databases-dir PATH    SQLite databases directory"
    echo "  --schemas-dir PATH      Per-DB schema JSON directory (output of --extract-schema)"
    echo "  --test-set              Shorthand: swap all four paths to test equivalents"
    echo ""
    echo "Schema Options:"
    echo "  --extract-schema        Force a schema rebuild. Schemas are built automatically"
    echo "                          when absent and reused when present, so this is only"
    echo "                          needed after the database set changes"
    echo "  --column-meaning PATH   Column descriptions for extraction. The shipped file covers"
    echo "                          the 11 dev databases only; pass \"\" to skip explicitly"
    echo "  --final PATH            Also copy the final predictions to PATH"
    echo ""
    echo "Pipeline Options:"
    echo "  -m, --model MODEL       Generation/schema/fixer model (default: claude-sonnet-4-6)"
    echo "  --judge-model MODEL     Judge model (default: claude-opus-5-5)"
    echo "  -t, --threshold N       Schema relevance threshold (default: 0.5)"
    echo "  --timeout N             SQL timeout in seconds (default: 30)"
    echo "  --workers N             Pipeline workers (default: 4)"
    echo ""
    echo "Stage Options:"
    echo "  --with-fixer            Keep stage 6 (measured -12/1534 on dev; off by default)"
    echo "  --profile               Also log per-question latency, rate-limit delay and tokens"
    echo "  --no-column-family      Turn RULE P off (measured +2 to +3 on dev; on by default)"
    echo "  --no-postprocess        Skip step 2"
    echo "  --no-review             Skip step 3"
    echo "  --pp-workers N          Post-process workers"
    echo "  --pr-workers N          Projection-review workers"
    echo ""
    echo "Other:"
    echo "  -v, --verbose           Verbose pipeline output"
    echo "  -h, --help              This message"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -q|--question)    QUESTION="$2"; BATCH_MODE=false; shift 2 ;;
        -b|--batch)       BATCH_MODE=true; BATCH_START="$2"; BATCH_END="$3"; shift 3 ;;
        --start)          BATCH_MODE=true; BATCH_START="$2"; shift 2 ;;
        --end)            BATCH_MODE=true; BATCH_END="$2"; shift 2 ;;
        --all)            BATCH_MODE=true; BATCH_START=0; BATCH_END=""; shift ;;
        -d|--data-dir)    DATA_DIR="$2"; shift 2 ;;
        -o|--output-dir)  OUTPUT_DIR="$2"; shift 2 ;;
        --dev-json)       DEV_JSON="$2"; shift 2 ;;
        --tables-json)    TABLES_JSON="$2"; shift 2 ;;
        --databases-dir)  DATABASES_DIR="$2"; shift 2 ;;
        --schemas-dir)    SCHEMAS_DIR="$2"; shift 2 ;;
        --test-set)
            DEV_JSON="./data/bird_data/test.json"
            TABLES_JSON="./data/bird_data/test_tables.json"
            DATABASES_DIR="./data/bird_data/test_databases"
            SCHEMAS_DIR="./data/bird_data/test_schemas"
            shift ;;
        --extract-schema) EXTRACT_SCHEMA=true; shift ;;
        --column-meaning) COLUMN_MEANING="$2"; shift 2 ;;
        --final)          FINAL_COPY="$2"; shift 2 ;;
        -m|--model)       MODEL="$2"; shift 2 ;;
        --judge-model)    JUDGE_MODEL="$2"; shift 2 ;;
        -t|--threshold)   THRESHOLD="$2"; shift 2 ;;
        --timeout)        TIMEOUT="$2"; shift 2 ;;
        --workers)        MAX_WORKERS="$2"; shift 2 ;;
        --with-fixer)     NO_FIXER=false; shift ;;
        --profile)        PROFILE=true; shift ;;
        --no-column-family) COLUMN_FAMILY=false; shift ;;
        --no-postprocess) PP=false; shift ;;
        --no-review)      REVIEW=false; shift ;;
        --pp-workers)     PP_WORKERS="$2"; shift 2 ;;
        --pr-workers)     PR_WORKERS="$2"; shift 2 ;;
        -v|--verbose)     VERBOSE=true; shift ;;
        -h|--help)        show_help; exit 0 ;;
        *) echo "Unknown option: $1"; show_help; exit 1 ;;
    esac
done

for p in "$DEV_JSON" "$DATABASES_DIR"; do
    if [[ ! -e "$p" ]]; then
        echo "Error: missing path: $p" >&2
        exit 1
    fi
done

# Schemas are built automatically when they are absent, and reused when they are already there.
# --extract-schema forces a rebuild, which is what you want after the databases change.
SCHEMA_COUNT=0
if [[ -d "$SCHEMAS_DIR" ]]; then
    SCHEMA_COUNT=$(find "$SCHEMAS_DIR" -maxdepth 1 -name "*_schema.json" 2>/dev/null | wc -l)
fi
DB_COUNT=$(find "$DATABASES_DIR" -maxdepth 2 -name "*.sqlite" 2>/dev/null | wc -l)
if [[ "$EXTRACT_SCHEMA" != true && "$SCHEMA_COUNT" -gt 0 ]]; then
    echo "Schemas:     reusing $SCHEMA_COUNT existing schema JSONs in $SCHEMAS_DIR"
    if [[ "$SCHEMA_COUNT" -lt "$DB_COUNT" ]]; then
        echo "  WARNING: $DB_COUNT databases but only $SCHEMA_COUNT schema files." >&2
        echo "           Rebuild with --extract-schema if the database set changed." >&2
    fi
else
    EXTRACT_SCHEMA=true
    if [[ "$SCHEMA_COUNT" -eq 0 ]]; then echo "Schemas:     none found in $SCHEMAS_DIR - building them"; fi
fi

if [[ "$EXTRACT_SCHEMA" == true ]]; then
    echo "=== Step 0/3: extract schema JSONs ==="
    EX_ARGS=(--db-dir "$DATABASES_DIR" --tables-json "$TABLES_JSON" --output-dir "$SCHEMAS_DIR")
    if [[ -n "$COLUMN_MEANING" ]]; then EX_ARGS+=(--column-meaning "$COLUMN_MEANING"); else EX_ARGS+=(--column-meaning ""); fi
    echo "python -m src.processing.extract_schema ${EX_ARGS[*]}"
    python -m src.processing.extract_schema "${EX_ARGS[@]}"
    # Descriptions are keyed "{database}|{table}|{column}". A file that covers none of these
    # databases loads cleanly and matches nothing, so check rather than trust the load message.
    python - "$SCHEMAS_DIR" <<'PYCHK'
import json, sys
from pathlib import Path
n = d = 0
for p in Path(sys.argv[1]).glob("*_schema.json"):
    for t in json.loads(p.read_text(encoding="utf-8")).get("tables", {}).values():
        for c in t.get("columns", []):
            n += 1
            d += bool(c.get("description"))
pct = 100 * d / n if n else 0
print(f"  columns with a description: {d}/{n} ({pct:.0f}%)")
if n and pct < 50:
    print("  WARNING: most columns have no description. The schema agent leans on these;")
    print("           check that --column-meaning matches THESE databases.", file=sys.stderr)
PYCHK
fi

PATCH_DIR="$(python -c "import pathlib;print(pathlib.Path('experiments/full_pipeline/patches').resolve())")"
SEP="$(python -c "import os;print(os.pathsep)")"
PATCH_ENV=(
    QASQL_NATIVE_HEADLESS=1
    QASQL_COUNT_STRICT=1 QASQL_COUNT_ALL=1
    QASQL_GENERATION_RULES=1
    QASQL_MODEL_JUDGE="$JUDGE_MODEL"
    QASQL_JUDGE_ID_FIX=1 QASQL_JUDGE_ID_FIX_LOG="${OUTPUT_DIR%/}/judge_id_fix.jsonl"
    QASQL_STAGE_OUTPUTS=1
)
if [[ "$COLUMN_FAMILY" == true ]]; then PATCH_ENV+=(QASQL_COLUMN_FAMILY=1); fi
if [[ "$NO_FIXER" == true ]]; then PATCH_ENV+=(QASQL_DISABLE_FIXER=1); fi
if [[ "$PROFILE" == true ]]; then PATCH_ENV+=(QASQL_COST_PROFILE=1); fi

PIPE_ARGS=(--headless -m "$MODEL" -o "$OUTPUT_DIR" -d "$DATA_DIR"
           --dev-json "$DEV_JSON" --tables-json "$TABLES_JSON"
           --databases-dir "$DATABASES_DIR" --schemas-dir "$SCHEMAS_DIR"
           -t "$THRESHOLD" --timeout "$TIMEOUT" --workers "$MAX_WORKERS")
if [[ "$VERBOSE" == true ]]; then PIPE_ARGS+=(-v); fi
if [[ "$BATCH_MODE" == true ]]; then
    # An open-ended range has to become a concrete end index: run_pipeline.sh's --all resets
    # BATCH_START to 0, so passing it for "--start N with no --end" would silently run the
    # whole set from 0 while the banner claimed [N, end).
    if [[ -z "$BATCH_END" ]]; then
        BATCH_END="$(python -c "import json,sys;print(len(json.load(open(sys.argv[1],encoding='utf-8'))))" "$DEV_JSON")"
    fi
    PIPE_ARGS+=(-b "$BATCH_START" "$BATCH_END")
else
    PIPE_ARGS+=(-q "$QUESTION")
fi

echo "=============================================="
echo "QA-SQL full run"
echo "=============================================="
echo "Questions:   $DEV_JSON"
echo "Databases:   $DATABASES_DIR"
echo "Schemas:     $SCHEMAS_DIR"
echo "Tables JSON: $TABLES_JSON"
echo "Output:      $OUTPUT_DIR"
if [[ "$BATCH_MODE" == true ]]; then
    if [[ -n "$BATCH_END" ]]; then echo "Range:       [$BATCH_START, $BATCH_END)"
    else echo "Range:       [$BATCH_START, end)"; fi
else
    echo "Range:       single question #$QUESTION"
fi
echo "Model:       $MODEL   judge: $JUDGE_MODEL"
if [[ "$NO_FIXER" == true ]]; then echo "Fixer:       OFF"; else echo "Fixer:       ON"; fi
if [[ "$COLUMN_FAMILY" == true ]]; then echo "RULE P:      ON"; else echo "RULE P:      OFF"; fi
echo "=============================================="
echo ""

mkdir -p "$OUTPUT_DIR"
# RULE P reads a sparsity cache measured from the databases in use; no rule, no cache needed.
if [[ "$COLUMN_FAMILY" == true ]]; then python -m experiments.column_family.sparsity >/dev/null; fi

echo "=== Step 1/3: pipeline ==="
env "${PATCH_ENV[@]}" PYTHONPATH="$PATCH_DIR${PYTHONPATH:+$SEP$PYTHONPATH}" \
    bash scripts/run_pipeline.sh "${PIPE_ARGS[@]}"

FINAL="${OUTPUT_DIR%/}/selected.json"
if [[ "$PP" == true ]]; then
    echo "=== Step 2/3: question-form post-process ==="
    PP_ARGS=(--output-dir "$OUTPUT_DIR" --questions "$DEV_JSON" --databases-dir "$DATABASES_DIR")
    if [[ -n "$PP_WORKERS" ]]; then PP_ARGS+=(--workers "$PP_WORKERS"); fi
    python -m src.postprocess.postprocess "${PP_ARGS[@]}"
    FINAL="${OUTPUT_DIR%/}/question_form_postprocess/selected.json"
fi

if [[ "$REVIEW" == true ]]; then
    echo "=== Step 3/3: projection review ==="
    PR_ARGS=(--output-dir "$OUTPUT_DIR" --selected "$FINAL" --questions "$DEV_JSON"
             --out "${OUTPUT_DIR%/}/projection_review")
    if [[ -n "$PR_WORKERS" ]]; then PR_ARGS+=(--workers "$PR_WORKERS"); fi
    python -m src.postprocess.review "${PR_ARGS[@]}"
    FINAL="${OUTPUT_DIR%/}/projection_review/selected.json"
fi

if [[ -n "$FINAL_COPY" ]]; then
    mkdir -p "$(dirname "$FINAL_COPY")"
    cp "$FINAL" "$FINAL_COPY"
    echo "copied final predictions -> $FINAL_COPY"
fi

echo ""
echo "Done.  final predictions: $FINAL"
