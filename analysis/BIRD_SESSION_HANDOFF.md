# BIRD experiment: handoff to another chat

Status checked: 2026-09-15 07:30 UTC / 16:30 Korea time.

## Current state (read this first)

The focused-selection experiment is fully finished: inference, export, evaluation
and analysis. **Result: 1101/1534 (71.77%), net -1 versus the original 1102.**
4 recoveries, 5 regressions, 5 neutral switches. Full analysis with per-switch
diagnosis, check-quality metrics and why 84 of 88 recoverable failures remained is in
`analysis/focused_selection_findings.md` (generated summary on top, hand-written
analysis after the `<!-- manual-analysis -->` marker; the summary script preserves it).
`analysis/BIRD_RESEARCH_MEMORY.md` carries the condensed conclusion.

Nothing is running. Do not rerun prepare, run, export or evaluate for
output/focused_selection/v1; evaluation refuses to overwrite itself. The sections
below describe the experiment as it was set up and remain accurate as history.

## Original session goal

Improve the user's BIRD text-to-SQL accuracy with a method that can work on the
private test set without reference SQL or reference answers at inference.
The experiment was independent output requirements plus focused candidate
verification. Do not claim success from the number of SQL switches alone.

## Workspace and primary files

Repository root (Windows PowerShell):

```text
C:\Users\user\OneDrive\Documents\02. Master Degree\03. Lab\10. Experiment\03. QA-SQL-Query Augmentation to SQL
```

Paths below are relative to that root:

- `output/claude_headless_v6/selected.json`: original 1534 predictions, keyed by
  question ID. Each value contains SQL, the `\t----- bird -----\t` separator,
  then the database ID. This is the user's raw selected result; preserve it.
- `data/bird_data/dev.json`: corresponding questions, evidence, database IDs and
  gold SQL. Gold is for offline scoring, never model inference.
- `data/bird_data/dev_databases`: SQLite databases and BIRD-supplied description CSVs.
- `scripts/run_evaluation.sh`: user's evaluation entry point. The experimental
  evaluator directly reuses the unmodified `evaluation/evaluation.py` worker.
- `analysis/BIRD_RESEARCH_MEMORY.md`: persistent project memory, constraints,
  paper findings and experiment state. `AGENTS.md` tells future sessions to read it.
- `analysis/selection_experiment_findings.md`: completed previous experiment and
  regression analysis. Do not confuse its scores with the new focused experiment.
- `analysis/FOCUSED_SELECTION_EXPERIMENT.md`: new experiment's protocol and commands.

## User constraints and authorization

1. Never overwrite original selected.json or its five saved candidate files.
2. Keep supplied BIRD schema descriptions unchanged; additional read-only database
   observations are allowed, but do not replace descriptions with dev-specific ones.
3. No question-ID rules, correct-candidate labels, gold SQL, or gold results at
   inference. Hidden-test transfer matters more than manually improving dev answers.
4. Keep experiments and outputs separate. The user authorized implementing and
   running this experiment, including a full 1534-question export and evaluation.
5. Measure regressions among originally correct questions, not just recoveries
   among known failures. Do not automatically promote an experimental prompt.
6. Preserve unrelated local/untracked work. New implementation and documentation
   files were not committed in this session; inspect git status before editing.

## Established baseline and previous experiment

Original strict execution accuracy: **1102/1534 = 71.84%**.

Of 432 original failures:

- **88** have at least one passing SQL in the five saved candidates. These are
  potentially recoverable through selection alone, identified offline using gold.
- **344** have no passing saved candidate and need generation improvements or
  other work beyond choosing from the existing pool.

Five candidate files are candidate_full_schema.json, candidate_sme_metadata.json,
candidate_minimal_profile.json, candidate_focused_schema.json and
candidate_full_profile.json in the original output directory.

Execution disagreement among the five candidates triggers 435 questions. That
includes 195 originally correct questions, so careless reranking can lose accuracy.
The original selected SQL is retained as an additional option when outside the pool.

The previous completed paired experiment is `output/selection_experiment/v2`:

| Method | Correct | Accuracy | Recoveries | Regressions |
|---|---:|---:|---:|---:|
| Original | 1102 | 71.84% | — | — |
| Control judge | 1102 | 71.84% | 30 | 30 |
| Broad disagreement judge | 1092 | 71.19% | 24 | 34 |

It made 870 model requests. Its 5574 SQL/reference comparisons were unique SQL
strings per question across compared predictions/candidates, not 5574 questions.
The accuracy denominator is always 1534. Keep the original as the reference baseline.

The local evaluator compares sets of row tuples: column positions matter, row
order and duplicate multiplicity do not. Execution matching is not semantic proof.

## Research that motivated the new experiment

| Paper | Method relevant to this project |
|---|---|
| [OpenSearch-SQL](https://arxiv.org/html/2502.14913) | Align question phrases with SELECT quantity and order before generation. |
| [CHESS](https://arxiv.org/html/2405.16755v3) | Generate semantic tests and check one test against all candidates. These are model judgments, not gold-backed executable tests. |
| [CHASE-SQL](https://arxiv.org/html/2410.01943v1) | Train a pairwise selector on correct/incorrect candidates, compare both candidate orders. |
| [XiYan-SQL](https://arxiv.org/html/2507.04701) | Train selection with difficult negatives and reduce SQL-formatting bias. |
| [Agentar-Scale-SQL](https://arxiv.org/html/2509.24403v6) | Group by execution answers and use trained reasoning selection. |
| [DeepEye-SQL](https://arxiv.org/html/2510.17586) | Specialized checks and targeted revision using external verification. |

No universal DISTINCT, NULL, tie-handling, identifier, or output-format rule was
established. Their reported gains are not promised gains in our setting. This
experiment adapts the first two ideas with database probes; it trains no model.
Training a selector on public BIRD training gold is a possible later direction,
but no such training has been implemented or authorized as a separate new run here.

## Current focused experiment: actual implementation

Code:

- `analysis/focused_selection_experiment.py`
- `analysis/focused_selection_prompt.py`
- `analysis/focused_selection_client.py`
- `analysis/test_focused_selection.py`
- `analysis/summarize_focused_selection.py` (post-evaluation analysis only)

Output directory: `output/focused_selection/v1`.

Protocol for the 435 triggered questions:

1. Derive 1–4 requirements from question, evidence and complete supplied schema,
   independently of candidate SQL. Require exact supporting source quotes.
2. If the alignment declares ambiguity, retain the original.
3. Otherwise execute up to two read-only SQL probes, bounded to five seconds
   and 20 displayed rows, with truncation marked.
4. For each requirement, use a separate model call to check all candidates.
   The model does not see the original-choice label or other check verdicts.
5. Switch only if the original fails a supported requirement and exactly one
   alternative result group passes every check. Unknown/unsupported/failed checks
   and unresolved alternatives retain the original. No SQL is rewritten.

The client runs Claude CLI without tools/MCP, outside the project directory,
with structured output. Inputs, prompts, code hashes, requests and responses are
saved. Parent v2 frozen inputs and execution packets are reused; gold labels and
previous judge decisions are not reused. Model alias sonnet resolved in logs to
claude-sonnet-5, with auxiliary claude-haiku-4-5-20251001 usage.

## Current completion state — do not rerun inference

All **1534** questions have finished:

| Outcome | Count |
|---|---:|
| No disagreement; original retained | 1099 |
| Deliberate abstention; original retained | 371 |
| Candidate switched | 14 |
| Failed quote validation; original retained | 50 |

- 786 logged model responses; zero provider errors.
- All 50 failed reviews report `Requirement quote does not occur in permitted source`.
- CLI-reported list cost: $45.6734224; not necessarily subscription billing.
- 23 focused/parent tests passed before the live run.
- The full-run coordinator finished successfully; `.running` is absent.
- Export completed successfully and verified original prediction source hashes.
- Accuracy HAS now been evaluated: 1101/1534. Fourteen switches produced four
  recoveries and five regressions. See the evaluation section below.

Completed full exports:

```text
output/focused_selection/v1/full_results/original.json
output/focused_selection/v1/full_results/selected_focused.json
output/focused_selection/v1/full_results/manifest.json
```

The focused export contains all 1534 predictions and differs from the original
in exactly 14 SQL strings. Original source SHA256:

```text
aaebe35088a21ff710cbad0487ae45996aa66486df4b96772e9c792be00ae921
```

## Evaluation: DONE (2026-09-15)

Both commands below were run once and must not be repeated against v1:

```powershell
& 'output/null_duplicate_ablation/.venv/Scripts/python.exe' analysis/focused_selection_experiment.py evaluate --workers 2
& 'output/null_duplicate_ablation/.venv/Scripts/python.exe' analysis/summarize_focused_selection.py
```

Evaluation artifacts in `output/focused_selection/v1/full_results/evaluation/`:
summary.json, per_question.json, diagnostics.json, changed_queries_offline_only.json
(contains gold) and candidate_scores_offline_only.json (gold pass/fail per frozen
candidate ID for the 435 triggered questions, added by the analysis session).
None of these may enter inference inputs.

Headline numbers: 1101/1534 (71.77%). Recovered 48, 453, 856, 887. Regressed 10,
220, 405, 529, 805. Per database: formula_1 +2, card_games -1, superhero -1,
toxicology -1, others 0. Of 88 recoverable failures: 60 blocked by whole-question
ambiguity, 15 had no failing check on the original, 7 lost to quote validation,
1 no fully passing alternative, 1 wrong pick, 4 recovered.

## In progress (2026-09-16): projection-alignment experiment

User rule: never modify src/; experiment code lives under experiments/ and imports
src/analysis modules. Structural failure analysis (analysis/structural_failure_diff)
found projection width/column choice to be the largest failure pattern. Conventions
were validated on public train gold first (output/bird_train_audit/20260916/README.md):
"no added identifier" and "named attribute returned as named" hold; "id over name
when unnamed" is database-specific and was NOT adopted.

Code: experiments/projection_alignment/{prompt.py, runner.py, PROTOCOL.md, tests}.
12 tests pass. COMPLETED 2026-09-16: all 1534 run (user-authorized, pilots at 12/50/100/150
first), exported and evaluated once. **1104/1534 (71.97%), +2 net: 7 recovered, 5 regressed.**
Reorders 4/0, drops 2/2, substitutions 1/1, additions 0/2. Full analysis:
experiments/projection_alignment/FINDINGS.md. Do not rerun v1; revisions go in a new dir.

## What a next session could do

The user has not requested a follow-up experiment. If asked, the evidence points
away from further reranking of the five frozen candidates (three selection methods,
none above 1102; 344 failures have no passing candidate). Candidate directions,
each needing a new experiment directory and explicit authorization:

1. Generation-side changes evaluated on the full set, since selection is capped.
2. If selection is revisited: execution-sanity guard against empty/zero winners
   and per-requirement ambiguity. Both are post-hoc hypotheses derived after seeing
   gold on v1; they are not validated improvements.
3. A revised-release (dev_20251106) run needs predictions generated from its own
   inputs; do not score the frozen predictions against revised gold.

## Observations so far — not accuracy results

- Q453 switched convertedManaCost to manaCost for an explicit unconverted-cost
  request. This illustrates field-meaning verification.
- Q1085's probe confirmed an exact Alexis player record, avoiding the earlier
  judge's unsupported claim that it did not exist. The original SQL was retained.
- Q992 retained separate first-name/surname fields rather than concatenating them.
- Whole-question ambiguity may be too restrictive: it can block a clear individual
  correction. Exact-quote validation also blocks some otherwise usable reviews.
  Quantify these effects after evaluation; do not tune the finished run after seeing gold.

## Important dataset-version finding

An offline audit downloaded the official revised release into a separate folder:
`output/bird_version_audit/20260915`.

Both versions have 1534 matching IDs and databases, but exact text comparisons show
182 changed questions, 381 changed evidence entries and 452 changed SQL strings.
These categories overlap and are not counts of semantic changes.

Source: https://huggingface.co/datasets/birdsql/bird_sql_dev_20251106

The original local dataset was not replaced. Do not score frozen old-input
predictions against revised gold where the question/evidence changed and describe
that as a fair improvement. A revised-release experiment needs matching inputs.
