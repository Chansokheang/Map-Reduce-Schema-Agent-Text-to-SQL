# Projection review: what the answer shows, and in what order

Runs on a finished `selected.json`, never modifies it, writes
`<out>/selected_projection_reviewed.json`. No regeneration, so it can be applied to any run you
already have — including `claude_headless_v6`.

```bash
python -m experiments.projection_review.review --output-dir ./output/sl_v1/
python -m experiments.projection_review.review --output-dir ./output/claude_headless_v6/ --workers 4
```

Evaluate the result with the existing evaluator, pointing `-f` at the new file.

## Why

Measured on the sl_v1 700-question run, of 226 failures:

| n | shape |
|---|---|
| 5 | gold's columns, wrong **order** |
| 11 | gold's columns plus **extra** ones |

BIRD compares rows as tuples, so the column count and the column sequence are both enforced,
while row order and duplicates are not checked at all. Gold follows the order the question asks
in **65 of 71** measurable cases; the exceptions are conventional groupings (street/city/state/zip)
and "answer columns first, ranking column last".

A mechanical reorder cannot do this job: matching column names against question text touches
**2 of 700** queries, because the mapping is semantic ("the names of all the administrators" is
`AdmFName1, AdmLName1`; "postal street address" is `MailStreet`).

## How it stays safe

The model never writes SQL. It receives the question, the evidence and the numbered SELECT items,
and returns **indices** into that list. The rewrite is therefore a permutation or a subset of what
the pipeline already produced — additions are impossible by construction, and `FROM`, `WHERE`,
`GROUP BY`, `ORDER BY` and `LIMIT` are untouched. That constraint comes from the earlier
projection experiment, where reorders were 4 for 4 correct while additions were 0 for 2.

Any failure — bad JSON, an out-of-range index, a timeout — keeps the pipeline's SQL unchanged.
Results are checkpointed per question and reused only when the source SQL for that key is
identical, so a rerun after more generation is cheap.

Statuses: `retained`, `reordered`, `trimmed`, `single_column`, `not_a_simple_select`, `failed`.

## Status

Built and tested (16 tests, no model calls). **Not yet measured end to end.** The comparison to
run is sl_v1's 700 (baseline 474) and v6's 1534 (baseline 1102), reporting recoveries and
regressions.
