# MRS-Agent: BIRD Submission Guide

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set API key
export ANTHROPIC_API_KEY='your-key'

# 3. Download BIRD dev databases  →  data/bird_data/
cd scripts && bash get_dbs.sh && cd ..

# 4. Extract schema + embed column descriptions from column_meaning.json
#    Dev (defaults):
python src/processing/extract_schema.py \
    --db-dir       ./data/bird_data/dev_databases \
    --tables-json  ./data/bird_data/dev_tables.json \
    --output-dir   ./data/bird_data/schemas && cd scripts

#    Test set (override paths):
# python src/processing/extract_schema.py \
#     --db-dir       ./data/bird_data/test_databases \
#     --tables-json  ./data/bird_data/test_tables.json \
#     --output-dir   ./data/bird_data/schemas && cd scripts
```

---

## Step 1 — Generate Predictions (1,533 questions)

```bash
sh run_pipeline.sh -b 0 1533
```

Output: `output/final/selected.json`

To use a custom output directory:

```bash
sh run_pipeline.sh -b 0 1533
```

To resume a partial range:

```bash
sh run_pipeline.sh -b 300 600   # questions 300–599
```

---

## Step 2 — Evaluate Locally

Default range is already `START_IDX=0`, `END_IDX=1533` in `run_evaluation.sh`.

```bash
# Execution accuracy (EX)
sh run_evaluation.sh -o output/final/ -f selected.json -t acc

# # Valid efficiency score (VES)
# sh run_evaluation.sh -o output/final/ -f selected.json -t ves

# Both
sh run_evaluation.sh -o output/final/ -f selected.json -t both
```

Partial range example:

```bash
sh run_evaluation.sh -o output/final/ -f selected.json -t acc --start 0 --end 100
```

> To change the default evaluation range, edit `run_evaluation.sh`:
> ```bash
> START_IDX="0"
> END_IDX="1533"
> ```

---

## Step 3 — Submit

Upload `output/final/selected.json`.

Expected format (dict keyed by question index):

```
"0": "SELECT ... \t----- bird -----\t<database_name>"
```

---

## Other Providers

```bash
# OpenAI
export OPENAI_API_KEY='your-key'
sh run_pipeline.sh --openai -m gpt-4o-mini -b 0 1533

# Ollama (local)
sh run_pipeline.sh --ollama -m qwen2.5-coder:32b -b 0 1533

# Claude Max (no API key, requires Claude Code CLI)
sh run_pipeline.sh --headless -b 0 1533
```

---

## External Knowledge

This system uses `data/column_meaning.json` for column semantic descriptions. These descriptions are embedded into the schema files by `src/processing/extract_schema.py` (Step 4 above).

---

## Prompt Token Estimation

Measured on the BIRD dev set using **claude-sonnet-4-6**:

| | Per question (Q0) | Full dev set (× 1,533) |
|---|---|---|
| Input (prompt) tokens | ~15,280 | **~23,424,240** |
| Output (completion) tokens | ~1,229 | ~1,884,357 |
| Estimated cost | ~$0.06 | **~$92** |

> These are estimates based on question 0. Actual usage may vary ±20% depending on schema size and retry attempts.

---

## Key Options (`run_pipeline.sh`)

| Flag | Default | Description |
|---|---|---|
| `-b START END` | — | Batch range |
| `-q N` | `0` | Single question |
| `--all` | — | All questions from 0 |
| `-t N` | `0.5` | Relevance threshold |
| `--timeout N` | `30` | Query timeout (seconds) |
| `--workers N` | `4` | Parallel schema workers |
| `-o PATH` | `output/final/` | Output directory |
