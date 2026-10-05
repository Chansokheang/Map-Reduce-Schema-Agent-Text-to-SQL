#!/bin/bash
# =============================================================================
# Full pipeline = original src.pipeline (via scripts/run_pipeline.sh, unchanged)
#               + question-form output post-process (experiments/full_pipeline)
#
# Usage (from anywhere):
#   bash experiments/full_pipeline/run_full_pipeline.sh -o ./output/full_v1/ --headless -b 0 1534
#   bash experiments/full_pipeline/run_full_pipeline.sh -o ./output/full_v1/ --headless -b 80 81
#   bash experiments/full_pipeline/run_full_pipeline.sh -o ./output/claude_headless_v6/ --skip-generation \
#        --pp-out ./output/full_pipeline_check/v6
#
# All options except the --pp-* and --skip-generation options below are passed to
# scripts/run_pipeline.sh unchanged. -o/--output-dir defaults to ./output/mc_v1/ (see the
# Defaults block below) and is used by both steps. The pipeline's selected.json is never modified; the post-processed file is
#   <pp-out>/selected_postprocessed.json   (default pp-out: <output-dir>/question_form_postprocess)
#
# Post-process options:
#   --pp-forms LIST      Comma-separated forms. Default: rank,count_then_list,entity_then_attribute,value_then_additive
#                        Add list_entity to reproduce the all-five-forms run.
#   --pp-client cli|api  cli (default): Claude Code CLI, tools off, as tested. api: src LLM client.
#   --pp-provider NAME   Provider for --pp-client api (default: anthropic)
#   --pp-model NAME      Default: sonnet (cli) / claude-sonnet-5 (api)
#   --pp-workers N       Default: 4
#   --pp-out PATH        Output folder for the post-process
#   --projection-review  Step 3: review every multi-column SELECT list - keep the items the question
#                        asks for, in the order it asks. Runs on the finished predictions (no
#                        regeneration) and writes a separate file; measured +11 on v6's 1534
#                        (14 recovered, 3 regressed, 33 of 1534 changed).
#                          --pr-no-trim   reorder only, never drop a column (+9 / -1 on v6)
#                          --pr-input selected|postprocessed   which file to review (default selected)
#                          --pr-out PATH  output folder (default <output-dir>/projection_review)
#                          --pr-model     default: sonnet
#   --skip-generation    Only post-process an existing pipeline output
#   --no-fixer           Ablation: run without stage 6 (the fixer); the judge's SQL is saved as is
#   --matched-contents   Append a '# Matched contents' block of database values, retrieved from the
#                        question, to every generation prompt. Build the index first:
#                          python -m experiments.matched_contents.indexer --databases-dir <dbs>
#   --judge-model M  run the JUDGE on model M, leaving every other stage alone
#                    (e.g. claude-opus-5-5). Use -m M to switch every stage instead.
#   --gen-model M    run candidate GENERATION on model M
#   --db-dir PATH    database directory RULE P measures (default: dev_databases). Point this
#                    at the test databases for a test run; the sparsity cache is rebuilt
#                    automatically and is named after the directory.
#   --column-family  RULE P: first member of a numbered column family (AdmFName1/2/3)
#                    unless the question or evidence asks for more; --entity-column adds
#                    the duplicate-column line, which measured 0
#   --promote-rules LETTERS   move these lettered rules to the head of the prompt, e.g. K
#   --count-all      RULE 0 on EVERY database (implies --count-strict); dev simulation put
#                    this at -4 on v6, so it is a measurement, not a recommendation
#   --count-strict       thrombosis_prediction ONLY: COUNT(DISTINCT) exactly when the evidence says
#                        to de-duplicate and never otherwise, with RULE E's join-multiplication
#                        trigger removed so nothing contradicts it. Dev-only by construction.
#   --count-convention   Add RULE O: use COUNT(DISTINCT) when the evidence says to de-duplicate
#                        ("should consider DISTINCT in the final result", "should compute the number
#                        of distinct/unique ones", "only count ones without repetitive", ...), and
#                        prefer plain COUNT when it does not. 18 of 18 dev gold queries with such an
#                        imperative use DISTINCT. Output format unchanged; combines with any flag.
#   --projection-order   Generation writes one `outputs:` line naming what the question asks to see,
#                        in the order it asks, then a query whose SELECT list matches that line.
#                        Keeps RULE A-K, does NOT add RULE L/M/N, and makes SQL extraction take the
#                        LAST fenced query. Combines with the retrieval flags; not with
#                        --reasoning-prompt (both rewrite the output format).
#   --reasoning-prompt   Generation writes a short analysis (decompose the question, map it to the
#                        schema, check the answer's shape, subqueries described in prose) and then
#                        the final query in one fenced block. Keeps RULE A-K; rewrites the two
#                        instructions that forbid explanations; makes SQL extraction take the LAST
#                        fenced query instead of the first. Test it on its own, not with other
#                        flags, or the result is unattributable.
#   --generation-rules   Add three rules to every generation system prompt - project only what the
#                        question asks for, in the order it asks; add no conditions it does not
#                        state - and correct the fixer's blanket NULL rule so stage 6 does not
#                        reverse them (--no-fixer-align keeps the fixer prompt as it is).
#                        Combines with any of the retrieval flags below.
#   --judge-id-fix       Reconcile the judge's selected_id with its selected_sql. src/ validates
#                        the two independently, so a valid id paired with another candidate's SQL
#                        is kept as-is, and the pipeline then ships the candidate the ID names -
#                        discarding the SQL the judge chose. Observed on Q101.
#   --stage-outputs      Also write judge_output.json (selected id, strategy, sql, confidence,
#                        reasoning) and fixer_output.json (sql before/after, changed, issues)
#                        into the output dir. The pipeline otherwise keeps only the final SQL.
#   --surface-forms      Add RULE Q, one line: a single plain SELECT, no WITH/CTE and no COALESCE
#                        or IFNULL unless the question or evidence asks for a substitute value.
#                        Dev gold uses COALESCE/IFNULL 0 times in 1534 and a CTE 9 times, and
#                        generation already emits 0 / 0 / 22-of-7670, so expect it to measure 0.
#                        Appended, never promoted, so it cannot displace RULE A.
#   --column-guidance    Everything --column-meaning does, plus this instruction on every schema
#                        worker (map agent) prompt:
#                          **Column Selection:**
#                          - Carefully analyze column descriptions and evidences to choose the
#                            correct column when similar columns exist across tables.
#                        Generation prompts are unchanged from --column-meaning, so the two arms
#                        differ in one place only.
#   --column-meaning     Everything --schema-linking does, plus a '# Column meanings' block: the
#                        columns whose shipped documentation (database_description/*.csv and
#                        data/column_meaning.json) best matches the question, with their
#                        descriptions, so near-identical column names can be told apart.
#                        Use instead of --schema-linking / --matched-contents, not with them.
#   --schema-linking     Retrieve those values BEFORE the schema agent, over the whole schema, so
#                        every table worker sees which tables and columns hold the question's
#                        values; adds the database's declared join columns (foreign keys) to the
#                        worker and generation prompts. Needs the same index. Use instead of
#                        --matched-contents, not with it.
#
# With --headless, generation loads headless_patch/sitecustomize.py so claude-code-headless calls
# the local claude executable directly (prompt on stdin, tools off, temp cwd). This fixes the
# Windows WSL quoting error ("... command not found") and WinError 206. Opt out: --no-native-headless
# =============================================================================
set -e
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ---- Defaults (edit here or override on the command line) ----
OUTPUT_DIR="./output/mc_v1/"                      # -o
DEV_JSON="./data/bird_data/dev.json"              # --dev-json
DATABASES_DIR="./data/bird_data/dev_databases"    # --databases-dir
PP_ARGS=()
PP_OUT=""
SKIP_GENERATION=false
PIPELINE_ARGS=()
HEADLESS=false
NATIVE_HEADLESS=true
MATCHED_CONTENTS=false
SCHEMA_LINKING=false
COLUMN_MEANING=false
COLUMN_GUIDANCE=false
GENERATION_RULES=false
SURFACE_FORMS=false
STAGE_OUTPUTS=false
JUDGE_ID_FIX=false
FIXER_ALIGN=true
REASONING_PROMPT=false
PROJECTION_ORDER=false
COUNT_CONVENTION=false
COUNT_STRICT=false
COUNT_ALL=false
PROMOTE_RULES=""
COLUMN_FAMILY=false
ENTITY_COLUMN=false
DB_DIR=""
JUDGE_MODEL=""
GEN_MODEL=""
PROJECTION_REVIEW=false
PR_ARGS=()
PR_INPUT="selected"
PR_OUT=""
NO_FIXER=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --headless) HEADLESS=true; PIPELINE_ARGS+=("$1"); shift ;;
        --no-native-headless) NATIVE_HEADLESS=false; shift ;;
        --matched-contents) MATCHED_CONTENTS=true; shift ;;
        --schema-linking) SCHEMA_LINKING=true; shift ;;
        --column-meaning) COLUMN_MEANING=true; shift ;;
        --column-guidance) COLUMN_GUIDANCE=true; shift ;;
        --generation-rules) GENERATION_RULES=true; shift ;;
        --surface-forms) SURFACE_FORMS=true; shift ;;
        --stage-outputs) STAGE_OUTPUTS=true; shift ;;
        --judge-id-fix) JUDGE_ID_FIX=true; shift ;;
        --reasoning-prompt) REASONING_PROMPT=true; shift ;;
        --projection-order) PROJECTION_ORDER=true; shift ;;
        --count-convention) COUNT_CONVENTION=true; shift ;;
        --count-strict) COUNT_STRICT=true; shift ;;
        --count-all) COUNT_STRICT=true; COUNT_ALL=true; shift ;;
        --promote-rules) PROMOTE_RULES="$2"; shift 2 ;;
        --column-family) COLUMN_FAMILY=true; shift ;;
        --db-dir) DB_DIR="$2"; shift 2 ;;
        --judge-model) JUDGE_MODEL="$2"; shift 2 ;;
        --gen-model) GEN_MODEL="$2"; shift 2 ;;
        --entity-column) COLUMN_FAMILY=true; ENTITY_COLUMN=true; shift ;;
        --no-fixer-align) FIXER_ALIGN=false; shift ;;
        --projection-review) PROJECTION_REVIEW=true; shift ;;
        --pr-no-trim) PR_ARGS+=(--no-trim); shift ;;
        --pr-model) PR_ARGS+=(--model "$2"); shift 2 ;;
        --pr-workers) PR_ARGS+=(--workers "$2"); shift 2 ;;
        --pr-client) PR_ARGS+=(--client "$2"); shift 2 ;;
        --pr-input) PR_INPUT="$2"; shift 2 ;;
        --pr-out) PR_OUT="$2"; shift 2 ;;
        --no-fixer) NO_FIXER=true; shift ;;
        -o|--output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --dev-json) DEV_JSON="$2"; PIPELINE_ARGS+=("$1" "$2"); shift 2 ;;
        --databases-dir) DATABASES_DIR="$2"; PIPELINE_ARGS+=("$1" "$2"); shift 2 ;;
        --pp-forms) PP_ARGS+=(--forms "$2"); shift 2 ;;
        --pp-client) PP_ARGS+=(--client "$2"); shift 2 ;;
        --pp-provider) PP_ARGS+=(--provider "$2"); shift 2 ;;
        --pp-model) PP_ARGS+=(--model "$2"); shift 2 ;;
        --pp-workers) PP_ARGS+=(--workers "$2"); shift 2 ;;
        --pp-out) PP_OUT="$2"; shift 2 ;;
        --skip-generation) SKIP_GENERATION=true; shift ;;
        -b|--batch) PIPELINE_ARGS+=("$1" "$2" "$3"); shift 3 ;;
        *) PIPELINE_ARGS+=("$1"); shift ;;
    esac
done

cd "$PROJECT_DIR"
echo "Output directory: $OUTPUT_DIR"

if [[ "$SKIP_GENERATION" == false ]]; then
    echo "=== Step 1/2: original pipeline (scripts/run_pipeline.sh) ==="
    PATCH_ENV=()
    if [[ "$HEADLESS" == true && "$NATIVE_HEADLESS" == true ]]; then
        PATCH_ENV+=(QASQL_NATIVE_HEADLESS=1)
    fi
    RETRIEVAL_FLAGS=0
    for on in "$MATCHED_CONTENTS" "$SCHEMA_LINKING" "$COLUMN_MEANING" "$COLUMN_GUIDANCE"; do
        if [[ "$on" == true ]]; then RETRIEVAL_FLAGS=$((RETRIEVAL_FLAGS + 1)); fi
    done
    if [[ $RETRIEVAL_FLAGS -gt 1 ]]; then
        echo "Error: use one of --column-guidance, --column-meaning, --schema-linking, --matched-contents." >&2
        exit 1
    fi
    if [[ "$MATCHED_CONTENTS" == true ]]; then
        PATCH_ENV+=(QASQL_MATCHED_CONTENTS=1)
    fi
    if [[ "$SCHEMA_LINKING" == true ]]; then
        PATCH_ENV+=(QASQL_SCHEMA_LINKING=1 QASQL_DATABASES_DIR="$DATABASES_DIR")
    fi
    if [[ "$COLUMN_MEANING" == true ]]; then
        PATCH_ENV+=(QASQL_COLUMN_MEANING=1 QASQL_DATABASES_DIR="$DATABASES_DIR")
    fi
    if [[ "$COLUMN_GUIDANCE" == true ]]; then
        PATCH_ENV+=(QASQL_COLUMN_GUIDANCE=1 QASQL_DATABASES_DIR="$DATABASES_DIR")
    fi
    if [[ "$JUDGE_ID_FIX" == true ]]; then
        PATCH_ENV+=(QASQL_JUDGE_ID_FIX=1 QASQL_JUDGE_ID_FIX_LOG="${OUTPUT_DIR%/}/judge_id_fix.jsonl")
    fi
    if [[ "$STAGE_OUTPUTS" == true ]]; then
        PATCH_ENV+=(QASQL_STAGE_OUTPUTS=1)
    fi
    if [[ "$SURFACE_FORMS" == true ]]; then
        PATCH_ENV+=(QASQL_SURFACE_FORMS=1)
    fi
    if [[ "$GENERATION_RULES" == true ]]; then
        PATCH_ENV+=(QASQL_GENERATION_RULES=1)
        if [[ "$FIXER_ALIGN" == false ]]; then PATCH_ENV+=(QASQL_GENERATION_RULES_FIXER=0); fi
    fi
    if [[ "$REASONING_PROMPT" == true && "$PROJECTION_ORDER" == true ]]; then
        echo "Error: use --projection-order or --reasoning-prompt, not both." >&2
        exit 1
    fi
    if [[ "$REASONING_PROMPT" == true ]]; then
        PATCH_ENV+=(QASQL_REASONING_PROMPT=1)
    fi
    if [[ "$PROJECTION_ORDER" == true ]]; then
        PATCH_ENV+=(QASQL_PROJECTION_ORDER=1)
    fi
    if [[ "$COUNT_CONVENTION" == true ]]; then
        PATCH_ENV+=(QASQL_COUNT_CONVENTION=1)
    fi
    if [[ "$COUNT_STRICT" == true ]]; then
        PATCH_ENV+=(QASQL_COUNT_STRICT=1)
    fi
    if [[ "$COUNT_ALL" == true ]]; then
        PATCH_ENV+=(QASQL_COUNT_ALL=1)
    fi
    if [[ -n "$PROMOTE_RULES" ]]; then
        PATCH_ENV+=(QASQL_PROMOTE_RULES="$PROMOTE_RULES")
    fi
    if [[ -n "$JUDGE_MODEL" ]]; then
        PATCH_ENV+=(QASQL_MODEL_JUDGE="$JUDGE_MODEL")
    fi
    if [[ -n "$GEN_MODEL" ]]; then
        PATCH_ENV+=(QASQL_MODEL_GENERATION="$GEN_MODEL")
    fi
    if [[ "$COLUMN_FAMILY" == true ]]; then
        PATCH_ENV+=(QASQL_COLUMN_FAMILY=1)
        # RULE P names columns measured from the databases actually in use, so the cache must be
        # built for THIS database set. On the test set this step is what stops the rule going silent.
        if [[ -n "$DB_DIR" ]]; then
            PATCH_ENV+=(QASQL_DB_DIR="$DB_DIR")
            QASQL_DB_DIR="$DB_DIR" python -m experiments.column_family.sparsity "$DB_DIR" || exit 1
        else
            python -m experiments.column_family.sparsity || exit 1
        fi
    fi
    if [[ "$ENTITY_COLUMN" == true ]]; then
        PATCH_ENV+=(QASQL_ENTITY_COLUMN=1)
    fi
    if [[ "$NO_FIXER" == true ]]; then
        PATCH_ENV+=(QASQL_DISABLE_FIXER=1)
    fi
    PIPELINE_ARGS+=(-o "$OUTPUT_DIR")
    if [[ ${#PATCH_ENV[@]} -gt 0 ]]; then
        # patches/sitecustomize.py is imported at Python start-up and applies the flagged patches.
        PATCH_DIR="$(python -c "import pathlib, sys; print(pathlib.Path(sys.argv[1]).resolve())" "$SCRIPT_DIR/patches")"
        PATH_SEP="$(python -c "import os; print(os.pathsep)")"
        env "${PATCH_ENV[@]}" PYTHONPATH="$PATCH_DIR${PYTHONPATH:+$PATH_SEP$PYTHONPATH}" \
            bash scripts/run_pipeline.sh "${PIPELINE_ARGS[@]}"
    else
        bash scripts/run_pipeline.sh "${PIPELINE_ARGS[@]}"
    fi
fi

if [[ ! -f "${OUTPUT_DIR%/}/selected.json" ]]; then
    echo "Error: ${OUTPUT_DIR%/}/selected.json not found." >&2
    exit 1
fi

echo "=== Step 2/2: question-form output post-process ==="
if [[ -n "$PP_OUT" ]]; then PP_ARGS+=(--out "$PP_OUT"); fi
python -m experiments.full_pipeline.postprocess \
    --output-dir "$OUTPUT_DIR" --questions "$DEV_JSON" --databases-dir "$DATABASES_DIR" "${PP_ARGS[@]}"

FINAL_DIR="${PP_OUT:-${OUTPUT_DIR%/}/question_form_postprocess}"

if [[ "$PROJECTION_REVIEW" == true ]]; then
    echo ""
    echo "=== Step 3/3: projection review ==="
    if [[ "$PR_INPUT" == "postprocessed" ]]; then
        PR_SELECTED="${FINAL_DIR%/}/selected_postprocessed.json"
    else
        PR_SELECTED="${OUTPUT_DIR%/}/selected.json"
    fi
    PR_TARGET="${PR_OUT:-${OUTPUT_DIR%/}/projection_review}"
    python -m experiments.projection_review.review --output-dir "$OUTPUT_DIR"         --selected "$PR_SELECTED" --questions "$DEV_JSON" --out "$PR_TARGET" "${PR_ARGS[@]}"
    echo "Projection-reviewed output:  ${PR_TARGET%/}/selected_projection_reviewed.json"
fi
echo ""
echo "Pipeline output (unchanged): ${OUTPUT_DIR%/}/selected.json"
echo "Post-processed output:      ${FINAL_DIR%/}/selected_postprocessed.json"
echo "Evaluate (dev) with:"
echo "  bash scripts/run_evaluation.sh -o ${FINAL_DIR%/}/ -f selected_postprocessed.json -t acc --start 0 --end 1534"
