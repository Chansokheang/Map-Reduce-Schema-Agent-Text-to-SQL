#!/bin/bash

# Re-exec under bash if invoked via `sh` (e.g. `sh run_pipeline.sh ...`).
# On Debian/Ubuntu `sh` is dash, which doesn't support [[ ]] or ${BASH_SOURCE[0]}.
if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

# =============================================================================
# QA-SQL Pipeline Execution Script
# =============================================================================
# Usage:
#   ./scripts/run_pipeline.sh                    # Run single question (default: 0)
#   ./scripts/run_pipeline.sh -q 5               # Run question 5
#   ./scripts/run_pipeline.sh -b 0 10            # Batch mode: questions 0-9
#   ./scripts/run_pipeline.sh --ollama           # Use Ollama instead of Claude
#   ./scripts/run_pipeline.sh --ollama -m mistral # Use specific Ollama model
#   ./scripts/run_pipeline.sh -t 0.6             # Custom relevance threshold
#   ./scripts/run_pipeline.sh --headless         # Use Claude Max via claude-code-headless
#   ./scripts/run_pipeline.sh --gemma            # Use GPU-local Gemma endpoint
# ./scripts/run_pipeline.sh --ollama -m qwen2.5-coder:32b -b 10 11
# sh ./run_pipeline.sh --headless -b 300 301
# sh ./run_pipeline.sh --gemma -b 0 10
# =============================================================================

set -e

# Default values
PROVIDER="anthropic"
MODEL=""  # Will be set based on provider
ANTHROPIC_MODEL="claude-sonnet-4-6"
OPENAI_MODEL="gpt-4o-mini"  # gpt-4o / gpt-4o-mini / gpt-4-turbo
OLLAMA_MODEL="qwen3-coder-next:latest" # qwen3-coder-next:latest / qwen2.5-coder:32b
OLLAMA_URL="http://203.255.78.58:9000" # http://203.255.78.58:9000
GEMMA_MODEL="gemma-4-E4B-8b-instruct"
GEMMA_URL="http://gpu-local.sovanreach.com:9020"
GEMMA_API_KEY=""  # Falls back to default in GemmaClient if empty
QUESTION=0
BATCH_MODE=false
BATCH_START=0
BATCH_END=""
THRESHOLD=0.5
TIMEOUT=30
MAX_WORKERS=4
VERBOSE=false

# -----------------------------------------------------------------------------
# Dataset config (override via CLI flags below).
#   DATA_DIR       — BIRD data root. Pipeline resolves {DATA_DIR}/dev_databases/
#                    and {DATA_DIR}/schemas/ from this. (used by pipeline)
#   DEV_JSON       — Questions file. (used by pipeline via --dev-json)
#   TABLES_JSON    — BIRD's combined tables/schema file. Captured by the
#                    pipeline (via --tables-json) for downstream evaluation
#                    scripts; the pipeline itself reads per-DB schema JSONs
#                    from {SCHEMAS_DIR}.
#   DATABASES_DIR  — SQLite databases directory. Wired through --databases-dir
#                    to override the default {DATA_DIR}/dev_databases.
#   SCHEMAS_DIR    — Per-DB schema JSON directory. Wired through --schemas-dir
#                    to override the default {DATA_DIR}/schemas.
#
# For test set, swap to:
#   DEV_JSON="./data/bird_data/test.json"
#   TABLES_JSON="./data/bird_data/test_tables.json"
#   DATABASES_DIR="./data/bird_data/test_databases"
# -----------------------------------------------------------------------------
DATA_DIR="./data/bird_data/"
DEV_JSON="./data/bird_data/dev.json"
TABLES_JSON="./data/bird_data/dev_tables.json"
DATABASES_DIR="./data/bird_data/dev_databases"
SCHEMAS_DIR="./data/bird_data/schemas"

OUTPUT_DIR="./output/Sonnet4-6/"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Help function
show_help() {
    echo "QA-SQL Pipeline Execution Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Provider Options:"
    echo "  --anthropic           Use Anthropic Claude API (default)"
    echo "  --openai              Use OpenAI API"
    echo "  --ollama              Use Ollama local LLM"
    echo "  --headless            Use Claude Max via claude-code-headless (no API key needed)"
    echo "  --gemma               Use GPU-local Gemma endpoint"
    echo "  -m, --model MODEL     Model name (provider-specific)"
    echo "  --ollama-url URL      Ollama server URL (default: http://localhost:11434)"
    echo "  --gemma-url URL       Gemma endpoint base URL (default: http://gpu-local.sovanreach.com:9020)"
    echo "  --gemma-api-key KEY   Gemma API key (falls back to GEMMA_API_KEY env var or hardcoded default)"
    echo ""
    echo "Question Options:"
    echo "  -q, --question N      Run single question N (default: 0)"
    echo "  -b, --batch START END Run batch of questions from START to END"
    echo "  --all                 Run all questions in dev.json"
    echo ""
    echo "Pipeline Options:"
    echo "  -t, --threshold N     Relevance threshold (default: 0.5)"
    echo "  --timeout N           Query timeout in seconds (default: 30)"
    echo "  --workers N           Max parallel workers (default: 4)"
    echo ""
    echo "Path Options:"
    echo "  -d, --data-dir PATH      Path to BIRD data directory"
    echo "  -o, --output-dir PATH    Path to output directory"
    echo "  --dev-json PATH          Path to questions JSON file (dev.json, test.json, or subset)"
    echo "  --tables-json PATH       Path to combined tables.json (captured for downstream eval)"
    echo "  --databases-dir PATH     Path to SQLite databases directory"
    echo "  --schemas-dir PATH       Path to per-DB schema JSON directory"
    echo ""
    echo "Other Options:"
    echo "  -v, --verbose         Enable verbose output"
    echo "  -h, --help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -q 0                           # Run question 0 with Claude"
    echo "  $0 --openai -q 5                  # Run question 5 with OpenAI"
    echo "  $0 --openai -m gpt-4o -b 0 10     # Batch 0-9 with GPT-4o"
    echo "  $0 --ollama -q 5                  # Run question 5 with Ollama"
    echo "  $0 --ollama -m mistral -b 0 10    # Batch 0-9 with Mistral"
    echo "  $0 -b 0 100 -t 0.6                # Batch with custom threshold"
    echo "  $0 --all --ollama -m codellama    # All questions with CodeLlama"
    echo "  $0 --gemma -q 0                   # Run question 0 with Gemma"
    echo "  $0 --gemma -b 0 10                # Batch 0-9 with Gemma"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --anthropic)
            PROVIDER="anthropic"
            shift
            ;;
        --openai)
            PROVIDER="openai"
            shift
            ;;
        --ollama)
            PROVIDER="ollama"
            shift
            ;;
        --headless)
            PROVIDER="headless"
            shift
            ;;
        --gemma)
            PROVIDER="gemma"
            shift
            ;;
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        --ollama-url)
            OLLAMA_URL="$2"
            shift 2
            ;;
        --gemma-url)
            GEMMA_URL="$2"
            shift 2
            ;;
        --gemma-api-key)
            GEMMA_API_KEY="$2"
            shift 2
            ;;
        -q|--question)
            QUESTION="$2"
            BATCH_MODE=false
            shift 2
            ;;
        -b|--batch)
            BATCH_MODE=true
            BATCH_START="$2"
            BATCH_END="$3"
            shift 3
            ;;
        --all)
            BATCH_MODE=true
            BATCH_START=0
            BATCH_END=""
            shift
            ;;
        -t|--threshold)
            THRESHOLD="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --workers)
            MAX_WORKERS="$2"
            shift 2
            ;;
        -d|--data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --dev-json)
            DEV_JSON="$2"
            shift 2
            ;;
        --tables-json)
            TABLES_JSON="$2"
            shift 2
            ;;
        --databases-dir)
            DATABASES_DIR="$2"
            shift 2
            ;;
        --schemas-dir)
            SCHEMAS_DIR="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
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

# Check for required environment variables and set default models
if [[ "$PROVIDER" == "anthropic" ]]; then
    if [[ -z "$ANTHROPIC_API_KEY" ]]; then
        echo -e "${RED}Error: ANTHROPIC_API_KEY environment variable not set${NC}"
        echo "Set it with: export ANTHROPIC_API_KEY='your-key-here'"
        exit 1
    fi
    # Set default model for Anthropic
    if [[ -z "$MODEL" ]]; then
        MODEL="$ANTHROPIC_MODEL"
    fi
    echo -e "${GREEN}Using Anthropic Claude: $MODEL${NC}"
elif [[ "$PROVIDER" == "openai" ]]; then
    if [[ -z "$OPENAI_API_KEY" ]]; then
        echo -e "${RED}Error: OPENAI_API_KEY environment variable not set${NC}"
        echo "Set it with: export OPENAI_API_KEY='your-key-here'"
        exit 1
    fi
    # Set default model for OpenAI
    if [[ -z "$MODEL" ]]; then
        MODEL="$OPENAI_MODEL"
    fi
    echo -e "${GREEN}Using OpenAI: $MODEL${NC}"
elif [[ "$PROVIDER" == "headless" ]]; then
    # No API key needed - uses Claude Max subscription via claude-code-headless
    # Default is Sonnet 4.6 (claude-sonnet-4-6); override with -m claude-opus-4-7 / haiku / etc.
    if [[ -z "$MODEL" ]]; then
        MODEL="claude-sonnet-4-6"
    fi
    echo -e "${GREEN}Using Claude Max via claude-code-headless (model: $MODEL, no API key needed)${NC}"
elif [[ "$PROVIDER" == "gemma" ]]; then
    # Gemma endpoint — GemmaClient has a hardcoded default API key, but allow override
    if [[ -z "$MODEL" ]]; then
        MODEL="$GEMMA_MODEL"
    fi
    echo -e "${GREEN}Using Gemma: $MODEL @ $GEMMA_URL${NC}"
else
    # Set default model for Ollama
    if [[ -z "$MODEL" ]]; then
        MODEL="$OLLAMA_MODEL"
    fi
    echo -e "${GREEN}Using Ollama: $MODEL @ $OLLAMA_URL${NC}"

    # Check if Ollama is running
    if ! curl -s "$OLLAMA_URL/api/tags" > /dev/null 2>&1; then
        echo -e "${RED}Error: Cannot connect to Ollama at $OLLAMA_URL${NC}"
        echo "Make sure Ollama is running: ollama serve"
        exit 1
    fi
fi

# Build command
CMD="python -m src.pipeline"
CMD="$CMD --provider $PROVIDER"

if [[ "$PROVIDER" == "anthropic" ]]; then
    CMD="$CMD -m $MODEL"
elif [[ "$PROVIDER" == "openai" ]]; then
    CMD="$CMD --openai-model $MODEL"
elif [[ "$PROVIDER" == "headless" ]]; then
    # Pass the shortcut (sonnet / haiku / opus) through to claude-code-headless
    CMD="$CMD -m $MODEL"
elif [[ "$PROVIDER" == "gemma" ]]; then
    CMD="$CMD --gemma-model $MODEL --gemma-url $GEMMA_URL"
    if [[ -n "$GEMMA_API_KEY" ]]; then
        CMD="$CMD --gemma-api-key $GEMMA_API_KEY"
    fi
else
    CMD="$CMD --ollama-model $MODEL --ollama-url $OLLAMA_URL"
fi

CMD="$CMD -t $THRESHOLD --timeout $TIMEOUT --max-workers $MAX_WORKERS"

if [[ -n "$DATA_DIR" ]]; then
    CMD="$CMD -d $DATA_DIR"
fi

if [[ -n "$OUTPUT_DIR" ]]; then
    CMD="$CMD -o $OUTPUT_DIR"
fi

if [[ -n "$DEV_JSON" ]]; then
    CMD="$CMD --dev-json $DEV_JSON"
fi

if [[ -n "$DATABASES_DIR" ]]; then
    CMD="$CMD --databases-dir $DATABASES_DIR"
fi

if [[ -n "$SCHEMAS_DIR" ]]; then
    CMD="$CMD --schemas-dir $SCHEMAS_DIR"
fi

if [[ -n "$TABLES_JSON" ]]; then
    CMD="$CMD --tables-json $TABLES_JSON"
fi

if [[ "$VERBOSE" == true ]]; then
    CMD="$CMD -v"
fi

if [[ "$BATCH_MODE" == true ]]; then
    if [[ -n "$BATCH_END" ]]; then
        CMD="$CMD -b --range $BATCH_START $BATCH_END"
    else
        CMD="$CMD -b --start $BATCH_START"
    fi
else
    CMD="$CMD -q $QUESTION"
fi

# Display configuration
echo ""
echo "=============================================="
echo "QA-SQL Pipeline"
echo "=============================================="
echo "Provider:    $PROVIDER"
echo "Model:       $MODEL"
echo "Data dir:    $DATA_DIR"
echo "Questions:   $DEV_JSON"
echo "Tables JSON: $TABLES_JSON"
echo "Databases:   $DATABASES_DIR"
echo "Schemas:     $SCHEMAS_DIR"
if [[ "$BATCH_MODE" == true ]]; then
    if [[ -n "$BATCH_END" ]]; then
        echo "Mode:        Batch [$BATCH_START, $BATCH_END)"
    else
        echo "Mode:        Batch (all from $BATCH_START)"
    fi
else
    echo "Mode:        Single question #$QUESTION"
fi
echo "Threshold:   $THRESHOLD"
echo "Timeout:     ${TIMEOUT}s"
echo "Workers:     $MAX_WORKERS"
echo "=============================================="
echo ""

# Change to project directory
cd "$PROJECT_DIR"

# Run pipeline
echo -e "${YELLOW}Executing: $CMD${NC}"
echo ""

eval $CMD

echo ""
echo -e "${GREEN}Pipeline completed!${NC}"
