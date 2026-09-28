# Generation rules L / M / N

Adds three rules to the generation system prompt of all five strategies, and corrects one
contradicting rule in the fixer. `src/` is untouched — both are patched in memory when
`QASQL_GENERATION_RULES=1`.

```bash
bash experiments/full_pipeline/run_full_pipeline.sh -o ./output/gr_v1/ --headless --generation-rules -b 0 700
# combines with any retrieval flag:
bash experiments/full_pipeline/run_full_pipeline.sh -o ./output/gr_cm_v1/ --headless --generation-rules --column-meaning -b 0 700
```

## What is added, and why only three

The strategy prompts already carry RULE A–K (CAST placement, percentage arithmetic, COUNT
argument, ROUND only when asked, DISTINCT, JOIN preference, STRFTIME, no string concat, minimal
tables, NULL for ASC/MIN only, superlatives) plus ten general rules. Restating them would
duplicate, and where wording differs it would contradict — so only the gaps are added.

| rule | evidence |
|---|---|
| **L — project only what is asked** | 5 of 9 inspected simple regressions projected extra columns; one extra column fails the row outright |
| **M — column order** | 2 of those 9 were a pure column swap; gold follows the question's order in 65 of 71 measurable cases |
| **N — no unstated conditions** | 3 of those 9 added a predicate the question never states (`type = 'OWNER'`, `type = 'VYDAJ'`, an extra status filter) |

## The fixer correction

`src/prompt/fixer.py` tells the fixer that a NULL in the result **MANDATORY**-ily requires
`IS NOT NULL`. The fixer runs after the judge, so on the ~12% of questions it rewrites it would
re-add exactly what RULE N forbids — this is what broke Q4, where gold keeps 57 NULL phone rows.
The replacement keeps the cases the fixer's own rationale names (ASC sort, `MIN()`, a single-row
"what is X" answer) and drops the blanket instruction. The duplicate rule is left alone.

Measured ceiling for that part: removing unrequested NULL filters from sl_v1 is worth **+4 on
700** (5 fixed, 1 broken). `--no-fixer-align` keeps the fixer prompt as it is.

## Status

Built and tested (9 tests). **Not yet measured end to end** — this changes generation, so it
needs a run. Compare against sl_v1 (474/700) or v6, on 700 questions minimum: the pipeline
rewrites about half its SQL between identical runs, so 200 questions cannot separate this.
