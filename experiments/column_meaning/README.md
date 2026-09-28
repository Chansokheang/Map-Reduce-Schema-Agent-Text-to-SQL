# Column meanings: documentation retrieval for column choice

The companion to `experiments/matched_contents`. That component answers *where does this value
live*; this one answers *which column is meant by these words* — the choice value matching
cannot make. No model calls, no gold at retrieval time, `src/` untouched, 12 tests pass.

Motivation, from the `sl_v1` 700-question review: the largest group of moderate failures was
"right tables, wrong columns" (24 of 86), and only **2 of those 24** had the gold column
anywhere in the value block, while 22 of the questions' evidence named no column at all. The
confusions are between documented siblings — `Free Meal Count (Ages 5-17)` vs
`FRPM Count (Ages 5-17)`, `schools.School` vs `frpm.School Name`, `amount` vs `loan_id`.

```bash
# Look up one question
python -m experiments.column_meaning.retriever --db california_schools \
    --question "What is the free meal count for students aged 5-17 in Monterey?"

# Offline recall against the columns gold uses (reads gold; scoring only)
python -m experiments.column_meaning.measure_recall --end 700

# Run the pipeline with it
bash experiments/full_pipeline/run_full_pipeline.sh -o ./output/cm_v1/ --headless --column-meaning -b 0 700
```

Output block:

```text
# Column meanings
- schools.School (text): The School column in the schools table contains the name of the school…
- frpm.Free Meal Count (Ages 5-17) (real): This column represents the number of free meals served…
- frpm.FRPM Count (Ages 5-17) (real): …the number of students aged 5-17 eligible for free or reduced-price meals…
```

## How it works

- **corpus.py** builds one document per column from documentation that ships with the
  benchmark: `database_description/<table>.csv` (column description, data format, value
  description) merged with `data/column_meaning.json` (the prose meaning already used by the
  pipeline) and the column name. All 11 dev databases are covered by both sources. The CSVs are
  read with an encoding fallback (`utf-8-sig`, `cp1252`, `latin-1`).
- **retriever.py** ranks those documents against the question plus evidence in two tiers:
  1. *named columns* — the documented name appears verbatim in the question or evidence, with a
     plural fold so "schools" matches a column called `School`. This tier exists because BM25
     structurally cannot find such columns: in a schools database the word "school" appears in
     nearly every document, so its IDF collapses to ~0 and `schools.School` never ranks,
     however plainly the question names it.
  2. *ranked by meaning* — plain BM25 (k1=1.5, b=0.75, the library defaults) over the merged
     documentation. `limit` bounds this tier; named columns are extra, so naming several columns
     does not push the ranked ones out.
- **pipeline_patch.py** wraps the same two seams as `matched_contents/schema_agent_patch.py`
  (`SchemaManager.coordinate_workers` and `PromptBuilder.build`) and emits values + join columns
  + column meanings. It refuses to install if `QASQL_SCHEMA_LINKING` or
  `QASQL_MATCHED_CONTENTS` already patched those seams, so runs cannot silently double up.

Cost: about **1 ms per question**, 14 lines per block at `limit=10`.

## Measured recall (dev, 2026-09-21)

Offline, tables unrestricted. "Gold columns shown" counts columns of the gold query present in
the block; "all shown" is the share of questions where the block covers every one of them.

| slice | limit | lines/block | gold columns shown | all shown |
|---|---|---|---|---|
| 0–699 | 10 | 13.8 | 2022/2542 = 79.5% | 53.7% |
| full dev | 15 | 18.2 | 4914/5900 = 83.3% | 57.0% |

By difficulty at limit 15 (full dev): simple 83.9%, moderate 82.6%, challenging 82.5% of gold
columns shown; complete coverage 63.1% / 47.4% / 49.0%. Moderate is lowest on complete coverage
because its queries reference more columns each.

Two generic retrieval mechanisms — the plural fold and the named-column tier — were added after
looking at dev misses, so these recall numbers are mildly fitted to dev in the same way the
value retriever's were. Neither is a per-question rule, and neither reads gold. As before, the
verdict has to come from a downstream A/B, not from recall.

## Status

Built and tested, **not yet measured end to end**. The comparison to run is `--column-meaning`
against `sl_v1` on the same questions. Two caveats worth holding on to:

- The pipeline rewrites about half its SQL between identical runs, so a difference of ±5 on 200
  questions is noise; this needs the 700-question slice at least.
- The block is appended to every worker prompt as well as the five generation prompts, so it
  adds tokens to every call the schema agent makes.
