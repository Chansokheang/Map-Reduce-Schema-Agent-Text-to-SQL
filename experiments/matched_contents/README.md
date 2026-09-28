# Matched contents: database value retrieval

Supplies the values a question refers to, with their source table and column, for use in
generation prompts (and possibly the judge). No model calls, no gold at retrieval time.
`src/` is not modified. 13 tests pass.

```bash
# 1. Build the value index once per database (also needed for the test set)
python -m experiments.matched_contents.indexer --databases-dir data/bird_data/dev_databases

# 2. Look up one question
python -m experiments.matched_contents.retriever --db california_schools \
    --question "How many students are enrolled at the State Special School in Fremont?"

# 3. Offline recall against gold literals (reads gold; scoring only)
python -m experiments.matched_contents.measure_recall
```

Output block:

```text
# Matched contents
- 'SSS' -> schools.EdOpsCode  (matched "State Special School")
- 'Fremont' -> schools.City  (matched "Fremont")
```

## How it works

- **indexer.py** copies every distinct value of each text column (1-80 characters, at most
  200k distinct per column) into `output/matched_contents/index/<db>.sqlite` with a
  normalized form, indexed. All 11 dev databases build in about 30 seconds; card_games is
  the largest at 867k values.
- **retriever.py** builds candidate phrases from the question and evidence (quoted spans,
  then word n-grams up to 6, plus singular/plural variants of the final word), looks them up
  by exact normalized match in batches, then by prefix range scan for longer phrases, and
  ranks quoted before unquoted, exact before prefix, longer phrases before shorter.
  About 1 second per 300 questions.

## Measured recall (dev, 2026-09-18)

| Measure | Value |
|---|---:|
| Gold string literals retrieved | 1102 / 1253 = 87.9% |
| Questions with every needed literal retrieved | 808 / 915 |
| Questions partly covered | 34 |
| Questions not covered | 73 |
| Average hits shown per question | 2.7 |

Three generic retrieval bugs were fixed after inspecting misses: prefix matches outranked
exact ones, plurals never matched singular values, and trailing punctuation broke the
singular/plural variant. These are general fixes, not per-question rules, but the recall
number is therefore mildly fitted to dev; the verdict must come from a downstream A/B.

Remaining misses are semantic, not lexical: word-order or derivation ("directly
charter-funded" vs `Directly funded`), domain synonyms ("chlorine" vs `cl`, "sodium" vs
`na`), and abbreviations ("state of California" vs `CA`). A vector store over values would
not obviously fix these, since short values like `cl` embed poorly; character-trigram fuzzy
matching is the cheaper next step, and embeddings, if used at all, belong on column
descriptions rather than every cell.

## Wired into generation (2026-09-18)

`pipeline_patch.py` wraps `PromptBuilder.build` at runtime, so every strategy prompt gets the
block appended and nothing else changes. It is loaded by
`experiments/full_pipeline/patches/sitecustomize.py` when `QASQL_MATCHED_CONTENTS=1`, which the
wrapper sets for `--matched-contents`:

```bash
bash experiments/full_pipeline/run_full_pipeline.sh -o ./output/mc_v1/ --headless --matched-contents -b 0 200
```

The database is identified by matching the schema's table names against the indexes, because
`build` receives no database id; one retrieval is cached per question and shared by the five
strategies. Verified end to end on Q376: both patches load, the prompt contains
`'Flying' -> cards.keywords`, and 5/5 candidates generate.

First observation, one question only: with the block present the model still wrote
`keywords LIKE '%Flying%'` rather than `= 'Flying'`. Showing the value may not be enough to
change that habit; a prompt instruction to prefer an exact matched value would be a further
change, and a bigger sample is needed before concluding anything.

## Before the schema agent, with join columns (2026-09-20)

`--matched-contents` retrieves at generation time, i.e. after the schema agent has pruned
tables, and the lookup is restricted to the tables that survived — so it can never tell the
agent that a literal lives in a table that was dropped, which is the failure mode behind the
"wrong table" group in the mc_v1 review (24% of failures).

`--schema-linking` moves retrieval in front of that stage and adds join columns:

```bash
bash experiments/full_pipeline/run_full_pipeline.sh -o ./output/sl_v1/ --headless --schema-linking -b 0 200
```

- **schema_agent_patch.py** wraps `SchemaManager.coordinate_workers`: one retrieval over the
  **whole** schema before any table is scored, cached for the question. Every table worker
  then sees the same block through a wrapper around its LLM client, appended after the
  original prompt — including hits in tables other than the one it is scoring, so a worker can
  see that the question's value lives elsewhere.
- **join_paths.py** reads the database's declared foreign keys (`PRAGMA foreign_key_list`,
  union the BIRD tables file when present; all 11 dev databases declare them) and shows the
  edges that touch the tables holding matched values, plus, for a worker, the edges touching
  its own table. That is the "0.5 = needed for a JOIN" judgement the worker prompt already
  asks for, with the real join keys instead of guessed ones.
- Generation still gets a block, now matched contents followed by the join columns for the
  tables visible in that prompt.

Nothing is forced into the focused schema: the block is evidence for the worker's own score,
not an override, and no rule keeps or drops a table. Both blocks are lookups (values from the
index, edges from the schema); no gold is read. `--schema-linking` and `--matched-contents`
are mutually exclusive, and `pipeline_patch.py` is untouched so mc_v1 still reproduces.

```text
# Matched contents
- 'Fresno County Office of Education' -> frpm.District Name  (matched "Fresno County Office of Education")
- 'Fresno County Office of Education' -> satscores.dname  (matched "Fresno County Office of Education")

# Join columns
- frpm.CDSCode = schools.CDSCode
- satscores.cds = schools.CDSCode
```

Tests: `python -m experiments.matched_contents.test_schema_agent_patch` (10 tests) alongside
the 18 existing pytest tests.

## Next step

Generation-side A/B on a fixed slice: arm A the pipeline unchanged, arm B the same prompts
plus this block, evaluated once each. Until that runs, the value of this component is
unmeasured: recall is not accuracy. The same applies to `--schema-linking`: it is built and
tested, not yet measured.
