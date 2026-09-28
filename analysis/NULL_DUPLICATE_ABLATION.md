# NULL and duplicate fixer experiment

This experiment tests whether the fixer's unconditional NULL filtering and
duplicate removal help or harm execution accuracy. It replays the saved v6
selected SQL through two versions of the existing `SQLFixer`:

- `control`: the current fixer prompt, unchanged.
- `conditional`: two prompt lines changed so NULL values and duplicate rows
  alone do not require a rewrite. Question/evidence semantics determine whether
  filtering or deduplication is necessary.

Every other fixer instruction remains identical, including its existing count,
projection, superlative, and evidence rules. This experiment does not establish
whether those other rules are useful.

The pilot uses 16 questions from each of the 11 development databases, selected
with a frozen hash seed independent of correctness labels. The questions,
evidence, saved SQL, original SQLite table definitions, and BIRD-provided CSV
descriptions are frozen before inference. No descriptions are generated.

Both arms use Claude Sonnet 4.6 through the authenticated Claude CLI. Safe mode,
disabled tools, an empty MCP configuration, and a temporary working directory
prevent inference from accessing repository files or gold SQL. The CLI does not
expose a fixed seed or temperature, so the two calls can differ stochastically.
Arm order alternates by question ID, and both arms allow three fixer reviews.

## Run

From the repository root, using Python with `requests` installed and an
authenticated `claude` executable:

```powershell
python analysis/null_duplicate_ablation.py prepare
python analysis/null_duplicate_ablation.py run
python analysis/null_duplicate_ablation.py evaluate
python analysis/verify_null_duplicate_ablation.py
```

Preparation refuses to replace an existing manifest. Running validates frozen
inputs and source code, resumes completed pairs/arms, and stops on provider or
parse failures. Use a different `--out` directory for a new experiment. Only the
evaluation command accesses gold SQL after all pairs have completed.

SQL executes in read-only SQLite connections with a 30-second execution
deadline. Inputs that cannot execute are recorded and kept unchanged in both
arms, because this pilot isolates post-execution review.

The local scoring comparison is `set(predicted_rows) == set(gold_rows)`, matching
the repository evaluator's comparison: row order and repeated rows are ignored;
column order and values are preserved. The report records execution errors,
paired recoveries/regressions, a two-sided exact McNemar test, and per-database
results. The local runner uses a deadline per SQL rather than the repository
wrapper's deadline around the query pair; any timeout discrepancy requires
inspection.

The verification command uses the repository's unmodified `execute_model`
worker with its 30-second paired-query timeout. It requires `func_timeout` and
writes `repository_check.json`, including every disagreement with local scoring.

To export original, control, and conditional predictions for the identical pilot
question IDs and evaluate all three through the repository CLI, run:

```powershell
python analysis/evaluate_matched_ablation.py
```

This creates a new `pilot/matched_subset/` directory and refuses to replace an
existing one. Its three prediction files and matching gold files share indices
0 through 175. `index_mapping.json` preserves original BIRD question IDs.
The original prediction strings are copied verbatim. Full CLI arguments and
separate evaluation logs are saved alongside the subset files.

To merge the existing pilot repairs into separate complete 1,534-query files:

```powershell
python -X utf8 analysis/evaluate_full_ablation.py
```

This creates `pilot/full_dev/`, including a byte-identical original copy and two
full prediction files containing each fixer's existing repairs. Predictions
outside the pilot remain unchanged. All questions are evaluated through the
unmodified repository worker; identical queries for the same question share one
execution result across versions. This avoids timing differences on unchanged
queries and does not generate repairs for the other 1,358 questions. The export
refuses to overwrite an existing destination or any original source file.

## Interpret

Compare conditional against control to estimate the effect of changing these
instructions. Compare both against the saved input to assess whether another
fixer pass helps at all. Inspect the logged reasoning for every changed outcome:
model variability and other unchanged rules can also cause differences.

The sample is balanced by database, not proportional to the full development
set. It is not a full-dev score or evidence of private-test generalization.
Historical generation, selection, and earlier repairs remain fixed. Stronger
evidence requires a frozen protocol on databases not used to design the change.

Artifacts are stored under `output/null_duplicate_ablation/pilot/`: manifest,
sanitized inputs, frozen schema and prompts, per-arm request/response logs,
per-question results, evaluation details, summary, and report.

Validation:

```powershell
python -m pytest analysis/test_null_duplicate_ablation.py -q
```
