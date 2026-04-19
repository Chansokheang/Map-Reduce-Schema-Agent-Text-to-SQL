# QA-SQL: Query Augmentation to SQL

A multi-stage pipeline that converts natural language questions into SQL using:
1) a Map-Reduce Schema Agent to narrow relevant tables/columns, and
2) a SQL Selection Agent that generates multiple candidates and uses LLM-as-a-Judge to pick the best query.

For a deeper architectural walkthrough, see `docs/README.md`.

---

## Quickstart

```bash
# 1) Install dependencies
pip install -r requirements.txt

# 2) Download BIRD dev data (creates data/bird_data/)
bash scripts/get_dbs.sh

# 3) Set API key (if using Anthropic or OpenAI)
export ANTHROPIC_API_KEY='your-key'
# or
export OPENAI_API_KEY='your-key'
```

Run a single question:

```bash
# Default: Anthropic Claude
bash scripts/run_pipeline.sh -q 0

# OpenAI
bash scripts/run_pipeline.sh --openai -m gpt-4o-mini -q 0

# Ollama (local)
bash scripts/run_pipeline.sh --ollama -m qwen2.5-coder:32b -q 0

# Claude Max via claude-code-headless (no API key)
bash scripts/run_pipeline.sh --headless -q 0
```

---

## Submission Instructions (BIRD dev)

These steps produce the exact prediction file expected by the evaluation tooling and most submission workflows.

### 1) Generate predictions for the full dev set

Pick one provider and run all questions:

```bash
# Anthropic (default)
bash scripts/run_pipeline.sh --all -o output/claude_run/

# OpenAI
bash scripts/run_pipeline.sh --openai -m gpt-4o-mini --all -o output/openai_gpt4o_mini/

# Ollama
bash scripts/run_pipeline.sh --ollama -m qwen2.5-coder:32b --all -o output/qwen2.5_32b/

# Claude Max (no API key)
bash scripts/run_pipeline.sh --headless --all -o output/claude_headless/
```

Notes:
- Use `-b START END` to run a slice (e.g., `-b 0 100` for questions 0–99).
- Use `-t` to adjust relevance threshold and `--workers` to tune parallelism.
- Outputs are written into your `-o` directory.

### 2) Confirm the submission file

Your submission-ready file is:

```
output/<run-name>/selected.json
```

Format (dict keyed by question id):

```
"0": "SELECT ... \t----- bird -----\t<database_name>"
```

You can sanity-check the first entry:

```bash
python - <<'PY'
import json
data = json.load(open('output/claude_run/selected.json'))
first_key = next(iter(data))
print(first_key, data[first_key])
PY
```

### 3) Evaluate locally (recommended)

```bash
# Accuracy only (fast)
bash scripts/run_evaluation.sh -o output/claude_run/ -f selected.json -t acc

# VES only
bash scripts/run_evaluation.sh -o output/claude_run/ -f selected.json -t ves

# Both
bash scripts/run_evaluation.sh -o output/claude_run/ -f selected.json -t both
```

### 4) Package for submission

If your submission portal expects a single file, upload `selected.json` directly.
If it expects an archive:

```bash
zip -j submission.zip output/claude_run/selected.json
```

---

## Pipeline CLI (direct)

The underlying CLI is `python -m src.pipeline` (same as the script wrapper):

```bash
python -m src.pipeline -q 0
python -m src.pipeline -b --range 0 10
python -m src.pipeline --provider openai --openai-model gpt-4o-mini -b --range 0 100
python -m src.pipeline --provider ollama --ollama-model llama3.2 -q 0
```

---

## Streamlit App

```bash
# Local
streamlit run app/sql_comparison_app.py

# Docker
cd app && docker-compose up
```

---

## Project Structure

```
src/
├── agents/                 # Map-Reduce schema agent
├── generation/             # Multi-strategy SQL generation
├── selection/              # SQL execution + LLM-as-a-Judge
├── processing/             # Schema extraction & profiling
├── utils/                  # LLM client + config
└── pipeline.py             # Orchestrator + CLI
```

---

## Data Utilities

```bash
# Extract schema from SQLite databases
python src/processing/extract_schema.py

# Generate column descriptions
python src/processing/database_profiling.py

# Update descriptions JSONs from BIRD CSVs
python scripts/update_descriptions_from_csv.py
```
