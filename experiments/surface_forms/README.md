# RULE Q — surface forms

One line appended to every generation system prompt:

> **RULE Q - SURFACE FORM:**
> - Write the answer as a single plain SELECT statement: no WITH/CTE, and no COALESCE or IFNULL
>   unless the question or the evidence asks for a substitute value. Use a subquery where you
>   would have used a CTE.

Enable with `--surface-forms` on `run_full_pipeline.sh`, or `QASQL_SURFACE_FORMS=1`.

## What was measured before writing it (2026-09-30)

String literals stripped first, `WITH` anchored to the start of a statement — an unanchored
search falsely matches question text such as `'Analysing wind data with R'`.

| construct | dev gold (1534) | gr_v1 pool (7,670 candidates) | opus final (1534) |
|---|---|---|---|
| COALESCE | 0 (0.00%) | 0 | 0 |
| IFNULL | 0 (0.00%) | 0 | 0 |
| WITH / CTE | 9 (0.59%) | 22 (0.29%) | 1 |

## Expected effect: 0

- Generation already never emits COALESCE or IFNULL, and writes CTEs **less** often than the
  gold does (0.29% vs 0.59%).
- The evaluator compares `set(rows)`. A CTE returning the same rows scores identically to a flat
  query, so surface form cannot cost a point on its own. COALESCE could — it substitutes a value
  for NULL and changes the result set — but generation does not produce it.
- The one CTE in the best full-dev prediction file is Q1481, and it is wrong for a semantic
  reason (it misreads "the customers with the least amount of consumption"), not because of the
  `WITH`.

The rule exists so the method can state that generation is constrained to the surface forms the
benchmark uses. It is not expected to recover anything, and at 0.3% of candidates it sits well
under the ±5-per-200-questions run-to-run noise, so a dev A/B would not be able to resolve it.

## Why it is appended and not promoted

Position, not wording, decided compliance in the COUNT experiment (0/11 → 11/11 for the same
text moved to the head of the rule block). Of RULE A–K only RULE A is measured to pay (+4).
A rule that fires on 0.3% of candidates must not push RULE A down, so `with_rule()` appends and
a test asserts `RULE A` still precedes `RULE Q`.

## Files

- `rule.py` — the rule text and `with_rule()`; idempotent.
- `pipeline_patch.py` — patches `PromptBuilder.build` in memory. `src/` is never edited. Touches
  `prompts["system"]`, so it composes with the retrieval flags, which touch `prompts["user"]`.
  The fixer is left alone: nothing in its prompt introduces a CTE or a COALESCE.
- `test_surface_forms.py` — 11 tests. `python -m experiments.surface_forms.test_surface_forms`
