# Candidate-selection experiment

This experiment tests whether a judge can choose better SQL from the five saved
candidates and the original selected SQL. It changes selection only. Production
generation, judge/fixer code, BIRD metadata, and original prediction files are
unchanged.

Implementation: [runner](selection_experiment.py),
[experimental prompt](selection_experiment_prompt.py),
[structured CLI adapter](selection_experiment_client.py),
[tests](test_selection_experiment.py).
Motivation and the offline review of the 88 opportunities are in
[the review](selection_failure_review/findings.md).

## Experiment arms

- **Original:** the saved selected result, copied byte-for-byte.
- **Control:** current production judge system criteria, frozen at preparation,
  followed by the experiment's ID-only response contract.
- **Disagreement:** new criteria comparing requested outputs, field meanings,
  population/filter scope, aggregation grain, arithmetic/types, NULLs, and ties.

Control and treatment get identical SQL, supplied schema descriptions, execution
summaries, candidate ordering, candidate-to-candidate differing-row examples,
CLI timeout, and one CLI request per triggered question. Both can abstain, and neither
can rewrite SQL. The fixer is never called. This control adapts the current
criteria to the experimental interface; it does not reproduce the historical v6
judge run. The control retains its existing gold-derived priors for comparison;
the treatment does not contain those priors.

The gate compares successful execution results from the **five saved candidates**.
It uses row-set equality, preserving column positions, like the repository's EX
comparison. Row order, aliases, and duplicate multiplicity do not split groups.
The original selected SQL is an additional option after triggering, so an
existing answer absent from the five-candidate pool remains selectable.
The measured historical gate count was 435; the runner recomputes the gate
without importing that count, the 88 IDs, or any audit labels.

Exact SQL duplicates are collapsed. Distinct SQL interpretations remain visible,
including minority results and SQL that happens to produce an equal result.
Strategy names and generation history are omitted. A seeded shuffle changes
display order reproducibly, and arm execution order is also shuffled. One run
does not constitute a balanced repeated-order robustness study.

## Data boundary

Preparation reads the question JSON and retains only question ID, database ID,
question and evidence. Gold SQL and difficulty are discarded. It freezes those
fields with saved candidate SQL, original predictions, relevant schema DDL, and
verbatim supplied CSV descriptions. These descriptions are not regenerated.

Inference reads only the frozen files. Its payload excludes question IDs,
strategy names, gold SQL, gold answers and correct-candidate labels. Schema
selection uses parsed referenced table names, falling back to all supplied
metadata on parser uncertainty. Candidate execution provides ordered columns,
projection expressions, physical/distinct row counts, per-column NULL counts,
three leading rows and up to three examples on each side of each disagreement.
Long display cells are explicitly marked as truncated; equality uses full values.

The experiment's Claude CLI adapter runs without tools or MCP access, outside the
project working directory, and without session persistence. It logs each
request/response. The model cannot open local gold or the labelled review files.
CLI --json-schema constrains the response to successful candidate IDs or null,
plus reasoning. The adapter reads structured_output; it does not silently repair
malformed JSON. The CLI may use multiple internal model turns for this output.
The CLI adapter accepts the shared client interface but does not enforce its
max_tokens/temperature arguments; both arms use the same adapter. Actual model
details, usage and any cost fields remain in the raw responses. The default
model alias is sonnet; use an explicit --model value at preparation to pin it.

Only the explicit evaluate stage opens gold for scoring, after a complete
full-result export exists. The runner can prepare/run on questions without SQL
labels; evaluate requires the matching labelled source. Private-test use requires
access to the permitted question/schema/database execution interface. This
experiment does not discover or download the hidden test dataset.

## Commands

Run these PowerShell commands from the repository root, using the existing
experiment environment:

    $experimentPython = 'output/null_duplicate_ablation/.venv/Scripts/python.exe'

The v2 run and evaluation have completed for all 1,534 questions.
The earlier v1 directory is preserved: its initial live check exposed a missing
JSON brace in a treatment response. V2 uses structured output for both arms.
To create a separate new experiment, choose an unused directory:

    & $experimentPython analysis/selection_experiment.py prepare --out output/selection_experiment/v3

Optional inspection executes SQL and builds packets, without model calls:

    & $experimentPython analysis/selection_experiment.py inspect --out output/selection_experiment/v2 --limit 16

Run the paired judges, then export and evaluate:

    & $experimentPython analysis/selection_experiment.py run --out output/selection_experiment/v2 --workers 2
    & $experimentPython analysis/selection_experiment.py export --out output/selection_experiment/v2
    & $experimentPython analysis/selection_experiment.py evaluate --out output/selection_experiment/v2 --workers 2

The run command makes model calls. The other commands do not. The optional
--limit on run processes only the first N original IDs; remove it to finish the
rest. A partial run cannot be exported as a complete experiment.

Re-running run resumes completed outcomes and cached packets. It does not
silently retry failed model reviews, which would change the call budget.
Use a fresh experiment directory for a revised prompt or repeat trial.
Changing frozen code, inputs, SQLite/parser versions or database fingerprints
stops resumption. A coordinator lock prevents two runs in one directory; after
a hard interruption, remove its .running file only once that coordinator has
stopped.

## Outputs and safeguards

Everything below is inside the chosen experiment directory:

- inputs.json, schemas.json, prompts.json, manifest.json: frozen preparation.
- original.json: byte-identical copy of the original selected file.
- packets/: gold-free SQL execution summaries and model payloads.
- calls/: request/response logs and per-arm outcomes.
- pairs/: completed paired decisions, including explicit failure/abstention status.
- full_results/original.json: original full submission copy.
- full_results/selected_control.json: complete control predictions.
- full_results/selected_disagreement.json: complete treatment predictions.
- full_results/manifest.json: changed IDs, review statuses and file hashes.
- full_results/evaluation/summary.json and per_question.json: overall EX,
  per-database results, recoveries, regressions, passing saved candidate counts,
  statuses, and available CLI cost information.

Preparation refuses an existing directory. Full export and evaluation refuse to
overwrite completed outputs. Exports preserve all original IDs and order, use
only exact frozen SQL, and keep original values outside the trigger. Source
prediction hashes are checked again at export.

Malformed JSON, unknown/failed candidate IDs, extra output keys, provider errors
and excessive prompt size retain the original SQL and are recorded as failed
reviews. An abstention also retains the original. A failed review counts in the
final operational accuracy but is not reported as a successful model decision.

SQL execution uses read-only SQLite connections, a progress deadline, and a
default one-million-row limit. Errors, timeouts and truncated executions cannot
establish a disagreement or be newly selected. A full prompt exceeding 180,000
characters causes both arms to retain the original with a failed-review status.
The database freeze records size and modification time rather than full database
content hashes; do not change databases during the experiment.

Evaluation calls the repository's unchanged execute_model worker. Each unique
SQL/reference pair is executed once and shared across matching variants and
saved candidates. Errors/timeouts score zero. Execution equality does not prove
semantic equivalence, and runtime timeouts can differ from the historical audit.
Do not interpret results on the 88 recoverable failures alone as an accuracy gain:
full-dev regressions and the currently correct triggered questions matter.

## Validation completed

- 20 tests passed across the new selection suite and existing fixer-ablation suite,
  including the structured-output adapter.
- Mock-judge end-to-end preparation, paired inference, resume, full export and
  scoring through the repository evaluator passed.
- Changing gold labels leaves prepared inference inputs identical.
- Frozen inference inputs load even when the source gold/question file is absent.
- Real full-dev preparation succeeded for 1,534 questions.
- Real read-only inspection succeeded for the first 16 questions; seven triggered.

The completed run scored original 1,102/1,534 (71.84%), control 1,102/1,534
(71.84%), and disagreement 1,092/1,534 (71.19%). The treatment recovered 24
failures but regressed 34 originally correct answers. It should not replace the
original selector based on this experiment.

See [the findings](selection_experiment_findings.md) and the full output report at
output/selection_experiment/v2/full_results/evaluation/README.md. The post-scoring
report can be regenerated without model calls:

    & $experimentPython analysis/summarize_selection_experiment.py --out output/selection_experiment/v2
