# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

QA-SQL (Query Augmentation to SQL) is a multi-stage pipeline that converts natural language queries to SQL using:
1. **Map-Reduce Schema Agent** - Decomposes queries and identifies relevant tables/columns
2. **SQL Selection Agent** - Generates SQL candidates using 5 strategies, then uses LLM-as-a-Judge to select the best

The system targets the BIRD benchmark for evaluation.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key (required for LLM operations)
export ANTHROPIC_API_KEY='your-key'

# Database profiling - generates column descriptions
python src/processing/database_profiling.py

# Extract schema from SQLite databases
python src/processing/extract_schema.py

# Run Streamlit comparison app (for ground truth testing)
streamlit run app/sql_comparison_app.py

# Run Streamlit app via Docker
cd app && docker-compose up
```

## Architecture

### Two-Phase Pipeline

**Phase 1: Map-Reduce Schema Agent**
- `src/agents/manager.py` - Decomposes NL queries into semantic components (entity, filter, aggregation, projection) and aggregates worker results
- `src/agents/worker.py` - Parallel workers that score table relevance using keyword matching and LLM semantic matching
- Output: Focused schema containing only relevant tables (relevance threshold: 0.50)

**Phase 2: SQL Selection Agent**
- `src/generation/candidate_generator.py` - Generates SQL using 5 strategies:
  - Full Schema, SME Metadata, Minimal Profile, Focused Schema, Full Profile
- `src/generation/prompt_builder.py` - Builds strategy-specific prompts
- `src/selection/judge.py` - LLM-as-a-Judge evaluates and selects best SQL candidate
- `src/selection/executor.py` - Executes SQL with retry loop (max 3 iterations)

**Pipeline Orchestrator**
- `src/pipeline.py` - `QASQLPipeline` class orchestrates the full workflow

### Supporting Modules

- `src/processing/input_processor.py` - Loads schema and profiles
- `src/processing/database_profiling.py` - Generates column descriptions
- `src/processing/extract_schema.py` - Extracts schema from SQLite
- `src/utils/llm_client.py` - Anthropic API wrapper
- `src/utils/config.py` - Configuration dataclass (default model: claude-sonnet-4-5-20250929)

### Streamlit App

- `app/sql_comparison_app.py` - UI for comparing generated SQL against BIRD ground truth
- Uses California Schools database for testing
- Docker deployment on port 9005

## Configuration

Key settings in `src/utils/config.py`:
- `relevance_threshold`: 0.5 (for schema filtering)
- `max_workers`: 4 (parallel schema workers)
- `max_refinement_attempts`: 2
- `query_timeout`: 30 seconds

## Data Flow

```
NL Query → Manager.decompose_query() → Workers.verify_table_relevance()
→ Manager.aggregate_results() → CandidateGenerator.generate_all_candidates()
→ Executor.execute_with_retry() → Judge.judge() → Final SQL
```

## Key Data Structures

- `DecomposedQuery` - entity, filter, aggregation, projection components
- `TableRelevance` - table_name, relevance_score (0.0-1.0), relevant_columns
- `SQLCandidate` - id, sql, strategy, confidence
- `JudgmentResult` - selected_id, selected_sql, confidence, reasoning
