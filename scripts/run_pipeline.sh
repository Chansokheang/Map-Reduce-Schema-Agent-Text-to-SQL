#!/bin/bash

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
# ./scripts/run_pipeline.sh --ollama -m qwen2.5-coder:32b -b 10 11
# sh ./run_pipeline.sh --headless -b 300 301
# =============================================================================

set -e

# Default values
PROVIDER="anthropic"
MODEL=""  # Will be set based on provider
ANTHROPIC_MODEL="claude-sonnet-4-5-20250929"
OPENAI_MODEL="gpt-4o-mini"  # gpt-4o / gpt-4o-mini / gpt-4-turbo
OLLAMA_MODEL="qwen3-coder-next:latest" # qwen3-coder-next:latest / qwen2.5-coder:32b
OLLAMA_URL="http://203.255.78.58:9000" # http://203.255.78.58:9000
QUESTION=0
BATCH_MODE=false
BATCH_START=0
BATCH_END=""
THRESHOLD=0.5
TIMEOUT=30
MAX_WORKERS=4
VERBOSE=false
DATA_DIR="./data/bird_data/"
OUTPUT_DIR="./output/final/" 

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
    echo "  -m, --model MODEL     Model name (provider-specific)"
    echo "  --ollama-url URL      Ollama server URL (default: http://localhost:11434)"
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
    echo "  -d, --data-dir PATH   Path to BIRD data directory"
    echo "  -o, --output-dir PATH Path to output directory"
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
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        --ollama-url)
            OLLAMA_URL="$2"
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
    MODEL="claude-max"
    echo -e "${GREEN}Using Claude Max via claude-code-headless (no API key needed)${NC}"
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
    # headless provider doesn't need model parameter
    :
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
