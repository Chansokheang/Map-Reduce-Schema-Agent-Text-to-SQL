# Projection alignment experiment (v1)

Motivation: analysis/structural_failure_diff shows that 81 of 431 v6 failures have the
wrong number of output columns and none of the 1102 correct answers do; the September 8
audit measured a ceiling of about 60 questions recoverable by projection changes alone.
Conventions used in the prompt were checked on public train gold first
(output/bird_train_audit/20260916/README.md): never add an identifier next to a requested
attribute (holds), a named attribute is returned as named (holds), id-versus-name when the
question names no attribute is database-specific (does NOT hold, so the rewrite keeps the
query's current identifier choice).

## Protocol

- Inputs: frozen gold-free inputs of output/selection_experiment/v2 (question, evidence,
  original selected SQL, schema DDL and supplied descriptions). Column lists per table are
  read from the databases at prepare time. Gold is read only by `evaluate`.
- One model request per question, all 1534. Payload: question, evidence, schema of the
  tables the query reads, the selected SQL, and its own execution shape (columns, row
  count, NULL counts, three sample rows). No candidates, no IDs, no labels.
- The model returns ordered output slots with exact quotes, a change_needed flag and,
  if needed, a complete new SELECT list.
- Mechanical validation (`apply_rewrite`): only the outer SELECT list changes; every new
  expression is either an original output expression or a plain column of a table the
  query already reads; no subqueries, `*`, new aggregates; no additions to grouped or
  aggregated queries; UNIONs are not rewritten. The rest of the AST must be byte-identical.
- Adopt the rewrite only if it executes without error and is not newly empty. Otherwise
  the original SQL is retained. Statuses: retained, rewritten, rejected, failed.
- Export all 1534 predictions, then evaluate original vs projection with the unmodified
  repository evaluator. Report recoveries and regressions over the full set.

## Commands (repository root)

```powershell
& 'output/null_duplicate_ablation/.venv/Scripts/python.exe' -m pytest experiments/projection_alignment -q
& 'output/null_duplicate_ablation/.venv/Scripts/python.exe' experiments/projection_alignment/runner.py prepare
& 'output/null_duplicate_ablation/.venv/Scripts/python.exe' experiments/projection_alignment/runner.py run --limit 12 --workers 2
& 'output/null_duplicate_ablation/.venv/Scripts/python.exe' experiments/projection_alignment/runner.py run --workers 4
& 'output/null_duplicate_ablation/.venv/Scripts/python.exe' experiments/projection_alignment/runner.py export
& 'output/null_duplicate_ablation/.venv/Scripts/python.exe' experiments/projection_alignment/runner.py evaluate --workers 2
```

The pilot `--limit` takes the first questions in submission order, never a
failure-selected subset. Do not change runner.py or prompt.py after prepare; hashes are
checked. Nothing under src/ is modified.

## What would count as success

A net gain on the full set with regressions listed and explained. Rewrites that switch to
an execution-equivalent SELECT list are expected and harmless. If the rewrite regresses
originally correct answers on the same order as its recoveries, the projection convention
is not stable enough to apply blindly and belongs inside a generation-time specification
with explicit uncertainty instead.
