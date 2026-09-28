# Focused selection experiment

Research context and user constraints: [project memory](BIRD_RESEARCH_MEMORY.md).

## Protocol

The experiment reuses frozen v2 question/evidence inputs, original descriptions,
saved SQL and execution packets. It reviews the same 435 disagreement questions;
the remaining 1099 keep their original SQL. All 1534 must be exported before
evaluation. Gold and offline error labels are never sent to the model.

For each triggered question:

1. A separate model request derives one to four explicit requirements from the
   question, evidence and complete supplied schema. It cannot see candidates or
   the original choice. Supporting quotes must occur literally in the input.
2. If it identifies unresolved ambiguity, retain the original. Otherwise execute
   up to two proposed SELECT-only probes against the read-only database. Results
   are bounded to 20 displayed rows and five seconds, with truncation explicit.
3. Each requirement receives its own independent model request examining all
   candidates. It cannot see the original label or other check verdicts. It may
   reject an unsupported requirement or return unknown for a candidate.
4. Switch only when a supported check fails the original and exactly one
   alternative result group passes every check. Otherwise retain the original.
   SQL is never rewritten. Errors retain the original and are reported as failures.

These are model-judged semantic checks, supplemented by actual database probes;
they do not prove correctness. Requirement completeness and interpretation can
still fail. Up to five CLI requests per triggered question are possible. Prompts,
code fingerprints, inputs, requests and responses are recorded. CLI runs without
tools/MCP in a temporary working directory. Sampling is not fully deterministic.

This is an adaptation of OpenSearch-SQL and CHESS, not a reproduction. It does
not train a selector. The original saved baseline is historical, not a same-model
generation control. Net gain and per-database regressions are primary outcomes;
recovering some of the 88 alone is insufficient. Dev-driven development is not
unbiased evidence of hidden-test performance.

## Run and resume

From the repository root, using the existing experiment Python:

```powershell
& 'output/null_duplicate_ablation/.venv/Scripts/python.exe' analysis/focused_selection_experiment.py prepare
& 'output/null_duplicate_ablation/.venv/Scripts/python.exe' analysis/focused_selection_experiment.py run --limit 16 --workers 2
& 'output/null_duplicate_ablation/.venv/Scripts/python.exe' analysis/focused_selection_experiment.py run --workers 4
& 'output/null_duplicate_ablation/.venv/Scripts/python.exe' analysis/focused_selection_experiment.py export
& 'output/null_duplicate_ablation/.venv/Scripts/python.exe' analysis/focused_selection_experiment.py evaluate --workers 2
```

Preparation already completed in output/focused_selection/v1; do not repeat it.
Run resumes completed checkpoints. Failed or interrupted attempted calls are not
silently retried. A .running lock prevents concurrent coordinators. Remove it only
after verifying that its coordinator has stopped. Do not change frozen experiment
code mid-run; use a new directory for protocol revisions.

Exports are original.json and selected_focused.json in the experiment's
full_results directory. The original source and production prompts are unchanged.
The evaluator is the repository's unmodified execute_model, with set-of-tuples
comparison: column positions matter, duplicate multiplicity/row order do not.

## Dataset audit

The official dev_20251106 release has the same 1534 IDs and databases, but exact
text comparisons identify 182 changed questions, 381 changed evidence fields and
452 changed SQL fields. Categories overlap and do not count semantic changes.
See output/bird_version_audit/20260915/summary.json. This experiment evaluates the
original question/gold source that produced the frozen predictions. A revised
dataset experiment requires predictions generated from its corresponding inputs.
