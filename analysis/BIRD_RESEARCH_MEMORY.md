# BIRD research and experiment memory

Updated 2026-09-15. Read this when resuming BIRD accuracy work.

## User constraints

- Never overwrite output/claude_headless_v6/selected.json or its candidate files.
- Preserve supplied BIRD schema descriptions. No dev-ID rules or gold at inference.
- Hidden-test transfer is the objective. Keep experiments and exports separate.
- Public training gold can label training examples; dev gold is evaluation only.

## Verified local baseline

Original: 1102/1534 (71.84%). Of 432 failures, 88 have a passing saved candidate,
344 do not. Passing means matching the local gold execution result, not proof of
semantic equivalence. The 88 are an offline diagnostic group, never an inference
gate. Five-candidate execution disagreement triggers 435 questions, including
195 original successes. Prior paired experiment recovered 24 and regressed 34:
1092/1534 (71.19%). The focused verification experiment scored 1101/1534 (71.77%,
4 recovered, 5 regressed). Keep original production behavior. See
selection_experiment_findings.md, focused_selection_findings.md and
output/selection_experiment/v2, output/focused_selection/v1.

## Papers: what they use and relation to our results

| Paper | Reported method | Proposed relevance |
|---|---|---|
| [OpenSearch-SQL](https://arxiv.org/html/2502.14913) | Aligns question phrases with SELECT quantity/order before generation. | Independently establish requested outputs before reviewing candidates. |
| [CHESS](https://arxiv.org/html/2405.16755v3) | Natural-language tests; checks one test against all candidates. These are model judgments, not gold-backed executable tests. | Separate output, population, aggregation and predicate checks. |
| [CHASE-SQL](https://arxiv.org/html/2410.01943v1) | Fine-tuned pairwise selector, comparisons in both orders. | A general judge prompt does not reproduce their trained selector. |
| [XiYan-SQL](https://arxiv.org/html/2507.04701) | Trained selection with difficult negatives and SQL formatting normalization. | Later train using subtle, execution-verified mistakes on public training databases. |
| [Agentar-Scale-SQL](https://arxiv.org/html/2509.24403v6) | Execution-result grouping and trained pairwise reasoning selection. | Compare distinct answer groups, preserve minority possibilities. |
| [DeepEye-SQL](https://arxiv.org/html/2510.17586) | Specialized checkers and targeted revision. | Use actual database evidence; do not adopt unconditional NULL or top-row rewrites. |

These are transferable research directions, not promised improvements. No
universal COUNT/DISTINCT/NULL/ties/output-representation convention was established.
Our Q1085 judge falsely claimed an exact value did not exist; read-only lookup
found it. Other regressions concern DISTINCT, output representation and conflicting
question/evidence/gold. More plausible interpretation need not improve strict EX.

## Accepted next experiment

User authorized implementation and execution on 2026-09-15:

1. Independently derive answer requirements from question, evidence and original
   complete schema descriptions, without candidate SQL or gold.
2. Perform bounded read-only database probes where useful.
3. Check each requirement against all candidates in separate model requests.
4. Replace original only if a supported check fails it, all checks pass an
   alternative, and only one alternative result group qualifies. Unknown,
   ambiguous, failed or conflicting checks retain original.
5. Export all 1534 predictions separately and evaluate recoveries AND regressions.

This is an adaptation, not a reproduction of a paper. No fine-tuning in this run.
The previous 88 IDs and error labels must never enter inference. A fixed pilot
chosen by input order may verify execution, but no gold-driven prompt tuning.

## Dataset version

BIRD links [bird_sql_dev_20251106](https://huggingface.co/datasets/birdsql/bird_sql_dev_20251106),
which revises questions, evidence and gold SQL. 1106 is a date, not a question
count. Compare to local data in an offline audit. Do not swap gold under old
predictions where inputs changed. Preserve the original-input evaluation for this
frozen-candidate experiment. New-release performance needs matching-input runs.

Audit completed: both versions contain 1534 matching IDs/databases; 182 question
texts, 381 evidence entries and 452 SQL strings differ (overlapping categories,
exact text comparison, not semantic change counts). Artifact:
output/bird_version_audit/20260915/summary.json. Local data was not overwritten.

## Matched-contents retriever built (2026-09-18)

experiments/matched_contents: per-database value index + question-driven lookup returning
value/table/column, no model calls, ~1s per 300 questions, 13 tests. Dev recall 1102/1253
gold string literals (87.9%), 808/915 questions fully covered; misses are semantic
(chlorine->cl, California->CA). Call site would be after the schema agent, before candidate
generation. Vector store considered and deferred: short values embed poorly and 2M values is
costly; trigram fuzzy matching is the cheaper next step. Wired into generation on 2026-09-18 via a runtime patch of PromptBuilder.build
(experiments/matched_contents/pipeline_patch.py, loaded by
experiments/full_pipeline/patches/sitecustomize.py, wrapper flag --matched-contents); src/
untouched, 74 experiment tests pass. Verified end to end on Q376, though the model still wrote
LIKE '%Flying%' with the block present. Downstream A/B not yet run - recall is not accuracy.
See experiments/matched_contents/README.md.

## Decision 2026-09-18: majority voting is abandoned

The user decided to drop voting/agreement information from the selection step entirely, after
the A/B below. Do not propose voting, plurality, agreement counts or result-group hints to the
judge again. Next direction: database value retrieval ("matched contents") supplied to
generation, and possibly to the judge, because it adds checkable external facts rather than
candidate agreement. Of the 88 recoverable judge failures, 19 differ from a passing candidate
only in string literals (e.g. 'sss' vs 'State Special School', '%flying%' vs 'flying'), which
is the part value retrieval could settle; the other 69 differ in structure or columns.

## Judge grouping A/B: COMPLETED (2026-09-18)

User's idea: voting component before the judge, judge sees which candidates share a result,
told not to favour the majority. experiments/judge_grouping, output/judge_grouping/v1,
frozen v6 candidates, production judge prompt both arms, 870 calls, $25.92 list.
Arm A (no grouping) 1107/1534; **arm B (grouping + neutrality) 1101/1534**; original 1102.
Differed on 62 of 435; B better on 6, worse on 12. Neutrality held (smaller group chosen 18
times vs larger 12) but accuracy did not improve. Sixth reranking method on this pool; none
beats 1102. Details: experiments/judge_grouping/FINDINGS.md.

## Full pipeline wrapper with post-process (2026-09-17)

experiments/full_pipeline: run_full_pipeline.sh calls scripts/run_pipeline.sh unchanged, then
postprocess.py (question-form layer, four forms, list_entity off by default) writes
<output-dir>/question_form_postprocess/selected_postprocessed.json. Verified on v6 output:
same 14 recovered / 2 regressed as the tested run (1114 by execute_model; 1111 = 72.43% by
run_evaluation.sh, whose original baseline is 1099). Headless generation on Windows fails
with WinError 206 (prompt passed as a command-line argument); generate from Linux/WSL or the
API. Headless generation also runs the CLI with default tools in the project folder, where
dev.json holds gold SQL: isolate before a submission run. See experiments/full_pipeline/README.md.

## Question-form output experiment: COMPLETED and EVALUATED (2026-09-17)

User asked to try the dev-derived output patterns (analysis/output_width_patterns) once on
the full set. Code experiments/question_form_output; output output/question_form_output/v1.
168 questions matched five wording forms by regex; one call each. **1109/1534 (72.29%),
17 recovered / 10 regressed.** Pre-registered split without the risky list_entity form:
1114 (72.62%), 14 / 2. Dev-derived, so an optimistic estimate. Details in FINDINGS.md there.

data/column_meaning.json (BIRD dev column meanings) is already embedded by the production
schema extractor. It adds value lists (270 columns without one in the CSVs), shows no
leakage, and does not distinguish similar output columns. See
analysis/column_meaning_patterns/README.md.

## refined_selected.json is a manual, gold-informed upper bound (confirmed 2026-09-17)

The user fixed selected.json by hand for column order, duplicates and NULL rows, using
analysis write-ups that show dev gold beside each generation. Result 1186/1534 (77.3%):
85 fixed, 1 broken (Q1224). It cannot be reproduced on the private test set, so the fair
baseline for automated methods remains selected.json at 1102. Its 85 fixes by original
category: other_mismatch 31, extra_columns 22, missing_columns 15, column_order 10,
null_rows_only 7. Projection v1 reproduced 6 of them automatically (4 column_order,
2 extra_columns) plus Q453, which refined did not fix.

## Projection alignment experiment: COMPLETED and EVALUATED (2026-09-16)

Code in experiments/projection_alignment (src/ untouched, per user rule). One call per
question rewrites only the SELECT list of the original selected SQL; AST validator limits
edits to reorder/drop/add/substitute plain columns. Result **1104/1534 (71.97%), net +2**
(7 recovered, 5 regressed); first experiment here to beat the original. Reordering was
4 recovered / 0 regressed; adding columns 0 / 2. Conventions were first checked on public
train gold (output/bird_train_audit/20260916): "id over name when unnamed" did NOT hold
and was excluded. Details: experiments/projection_alignment/FINDINGS.md.

## Focused selection experiment: COMPLETED and EVALUATED (2026-09-15)

Code: focused_selection_experiment.py, focused_selection_prompt.py,
focused_selection_client.py. Output: output/focused_selection/v1 (inference,
export and evaluation all finished; do not rerun any stage). Full analysis:
focused_selection_findings.md.

Result: **1101/1534 (71.77%), net -1** versus the original 1102. Fourteen switches:
4 recoveries (48, 453, 856, 887; all also recovered by the v2 control judge),
5 regressions (10, 529, 405, 220, 805), 5 neutral. Model usage: 786 responses,
list cost $45.67, claude-sonnet-5 plus haiku auxiliary.

Why it failed to help:

- Three regressions (10, 529, 405) switched to candidates whose packet result was
  empty or zero while the original returned data; the verifier passed them anyway
  ("downstream join issue"). Verification trusts requirement text over execution
  evidence it was shown.
- Checks are lenient: on 106 checked questions, 82% of gold-failing candidate
  assessments got "pass"; all-pass precision 45%. In 10 of the 15 recoverables
  where checks ran, the gold-correct candidate was failed while the original passed.
- Whole-question ambiguity blocked 60 of the 88 recoverable failures (279 of 435
  triggered overall), mostly over DISTINCT/ties/NULL/order conventions that BIRD
  gold applies silently. It protected 115 originally correct answers but none of
  the five regressions were caught by it.
- 50 exact-quote failures (38 also ambiguous): ellipsis quotes, whitespace across
  CSV/DDL (11 would match normalised), paraphrase.

Conclusion: selection over the five frozen candidates has now been tried three ways
(general judge 0, disagreement judge -10, requirement checks -1). None beats the
original; 344 failures have no passing candidate at all. Further gains need
generation changes, not reranking of this pool. Post-hoc, untested ideas for a NEW
directory only: an execution-sanity guard against empty/zero winners (would have
blocked 10, 529, 405 and no recovery) and per-requirement rather than whole-question
ambiguity. These were derived after seeing gold; treat them as hypotheses.

Offline gold labels per candidate for the 435 triggered questions are saved at
output/focused_selection/v1/full_results/evaluation/candidate_scores_offline_only.json.
Never feed it, changed_queries_offline_only.json, or the 88 IDs into inference.
