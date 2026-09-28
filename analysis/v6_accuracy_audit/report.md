**BIRD v6 accuracy audit — 8 September 2026**

The most useful update is an explicit, evidence-grounded answer specification shared by generation and verification, combined with candidates that explore different plausible interpretations. Improving SELECT columns matters, but the broader limitation is that the candidates often share the same incorrect interpretation of the requested output, entity, aggregation, or population. Adding more database-specific prompt rules will not resolve that limitation on unseen databases.

**Scope and evaluation.** I inspected the five write-ups under `submission/Evaluation Analysis`, traced the current pipeline, and executed the saved selected SQL, five candidate files, and refined SQL against the local databases. The actual paths are `output/claude_headless_v6/selected.json` and `data/bird_data/dev.json`. The local questions, prediction IDs, and `dev.sql` all cover **1,534 questions, IDs 0–1533**; the SQL and database fields in `dev.json` and `dev.sql` agree at every position.

The unchanged repository evaluator returned **1,099/1,534 = 71.64% EX**, with simple/moderate/challenging scores **76.32% / 64.01% / 66.21%**. Its wrapper currently defaults to `output/Qwen14b` and the range `[80,81)`, despite the README describing a full evaluation. Explicitly specify the prediction directory and full range.

**Measured patterns.** The detailed audit used the project's Python 3.13.1 / SQLite 3.45.3 and found **1,102 matching results (71.84%)**. All 1,102 subsequently passed the repository's unchanged `execute_model` worker in a focused recheck. This is three more than the initial full evaluation, not a model improvement. The diagnostic runner uses a 30-second budget per SQL, while the repository uses 30 seconds for a prediction/gold pair; execution load also differed. The initial evaluator does not log failed IDs, so the exact attribution of those three initial failures is unavailable. Use the initial **71.64%** as the reproduced full-run score and the following consistently measured diagnostic population for error analysis.

| Diagnostic finding | Questions | Implication |
|---|---:|---|
| Matches after one fixed column permutation | 10 | SELECT order matters. |
| Gold matches after projecting away extra predicted columns | 28 | Extra output is a recurring failure. |
| Prediction matches a projection of gold | 22 | Required outputs are also omitted. |
| Only NULL-bearing rows differ, with identical non-NULL rows | 9 | Both unnecessary exclusion and missing exclusion occur. |
| Other result mismatches | 359 | Projection/NULL cleanup alone will leave most failures. |
| Selected SQL execution failures | 2 | Q1014 times out; Q1199 has incomplete SQL. |
| Gold SQL execution timeouts | 2 | Q518 and Q701 remain unresolved at the time budget. |

The first four categories account for **69 of 432 diagnostic failures, about 16%**. They are disjoint, conservative result comparisons. A case may have additional SQL differences that happen not to affect the projected result on this database; these counts are not promises of automatic fixes. No selected failure matched merely by rounding float values to nine decimal places in this audit; integer division and wrong formula populations still cause substantial numerical errors.

| Candidate-pool measurement | Result |
|---|---:|
| Best individual saved strategy (`full_profile`) | 1,063 / 1,534 = 69.30% |
| At least one of the five saved candidates passes EX (oracle) | 1,169 / 1,534 = 76.21% |
| Selected result fails but a saved candidate passes | **88 questions** |
| Selected result fails and no saved candidate passes | **344 questions** |
| All five executable candidates return the same set | 1,096 questions |
| All five agree on a wrong result, excluding gold errors | **206 questions** |
| Select the largest result group, breaking ties by strategy order | 1,067 / 1,534 = 69.56% |

The largest opportunity therefore requires better candidate generation as well as better selection. A selector restricted to the saved five has a measured oracle ceiling of **76.21%**. The current selected file also passes **21 questions where none of the saved five pass**. Preserving those successes while perfectly recovering all 88 missed candidates would give **1,190/1,534 = 77.57%**, an oracle upper bound, not an achieved improvement. Simple result voting is worse than the current selection.

The existing `refined_selected.json` matches **1,186/1,534 = 77.31%** in the same diagnostic run. Relative to `selected.json`, 140 SQLs changed: **85 failures became passes, one pass became a failure, and 54 kept the same pass/fail status**. The regression is Q1224, where changing the output order breaks a previously passing answer. These are comparisons between saved artifacts, not evidence that the current fixer reproduces that gain without access to dev answers.

**Correction to the duplicate-row analysis:** the repository and [published BIRD evaluator](https://github.com/AlibabaResearch/DAMO-ConvAI/blob/main/bird/llm/src/evaluation.py) compare `set(predicted_rows)` with `set(gold_rows)`. Row order and repeated identical rows are ignored; tuple positions, NULLs, and values remain significant. **50 of the 56 questions listed in the duplicates document already pass.** The remaining six have other differences. Across the whole selected file, **54 questions pass despite different physical row counts**. Output deduplication alone cannot explain an EX failure, although DISTINCT inside an aggregate or before LIMIT can change the values or selected rows and therefore affect EX.

The detailed per-question table is `cases.csv`; complete execution diagnostics and input hashes are in `system_sqlite/details.jsonl` and `system_sqlite/summary.json`. An exploratory run with bundled SQLite 3.50.4 is also retained: it had the same projection and NULL categories, with differences in timeout outcomes. No generator, judge, fixer, prediction, or gold file was modified during this audit.

**How the current pipeline works.** Input processing loads schema and profile data. The schema manager decomposes the question and scores tables/columns. Five generation strategies receive different schema/profile presentations. The executor retries errors, empty results, and mostly-NULL results; an additional generation step runs if every candidate fails execution. The judge selects among executed candidates, then the fixer may rewrite the selected SQL. The final SQL and candidate SQLs are saved for evaluation.

Several details matter for interpreting the saved results:

- All five current strategies receive the question's evidence. `full_schema` and `sme_metadata` use the same full-schema formatter; their system prompts differ. `full_profile` normally uses the focused schema plus profiles. Five strategies therefore do not mean five independent interpretations.
- The schema manager produces semantic phrases, but not a complete ordered output specification, aggregation population, or tie policy.
- The judge receives three sample rows per candidate and only conditionally receives schema when its column-difference detector fires. Different joins, filter scopes, grouping, or NULL behavior can matter even when the referenced columns agree.
- The current prompts changed after the saved v6 run. Git history records substantial generator/judge rule additions on May 17, while candidate files are dated May 5 and selected/refined files May 11. Current code findings identify risks in the next update; they are not proof of which prompt caused an old prediction.
- Saved candidate files contain executed SQL, potentially already repaired. Some selected SQLs differ from all five saved candidates. Thus a saved-candidate oracle measures available answers, not a clean causal experiment isolating the judge.

**1. Make requested outputs explicit before SQL generation.** The recurring pattern is loss of information about what the answer should contain and how it should be represented.

| Case | What happened | General improvement |
|---|---|---|
| Q58 | Asked for phone, extension, then school name; selected SQL puts school first. Correct candidates exist. | Preserve ordered output slots linked to question spans. |
| Q17, Q726, Q728 | Generated ordering, but omitted the gold's rank expression; sometimes omitted the ranking measure too. | Distinguish a displayed ranking from merely sorted entities. Generate alternatives when wording is ambiguous. |
| Q172 | Asked for counts of owner and disponent dispositions; selected SQL combines them into one count. | Represent each requested measure separately. A “how many” question can request several columns. |
| Q159, Q165 | “List transactions” produces `SELECT *`; gold expects transaction IDs. | Resolve how an entity should be identified, using schema semantics and training examples. Avoid assuming every entity request means all attributes. |
| Q248, Q866 | An atom endpoint or part of a driver's name is missing. | Verify every requested semantic component, including multi-column names and relationships. |
| Q280, Q565, Q1177 | Returned explanatory labels such as “Yes” or “carcinogenic” where gold expects different labels or stored codes. | Track output representation separately from predicates; preserve supplied labels/codes unless a transformation is requested. |

Extend the existing decomposition step with an answer specification containing: ordered output slots, the question/evidence span supporting each slot, the entity represented by one row, grouping keys, numerator and denominator populations, filters and their boundaries, units/representation, and explicit uncertainty about ties or NULLs. Validate SQL against this specification after generation and after repair. Make the specification a revisable hypothesis: an incorrect shared specification must not force every candidate into the same mistake.

This is a general mechanism rather than a table-to-column lookup list. However, even a perfect reading of the question cannot predict every annotation convention. Q37 explicitly requests Street, City, Zip, State while the gold uses Street, City, State, Zip. Q888 asks for the circuit name, but the gold omits it. Column diagnostics identify where answers differ; they do not establish that every difference can be fixed from the question alone.

**2. Diversify the meaning of candidates, not only their context.** Keep at least one complete-schema candidate, but use additional candidates to investigate actual uncertainty: which entity is counted; whether a percentage is over all qualifying records or a subgroup; whether “highest” asks for a value, one entity, or all tied entities; whether an entity should be returned as ID or name; and which of two similar relationships the wording refers to.

Choose these alternatives from the particular question, evidence, and schema. Do not enumerate the same fixed variants for every database. Measure the number of distinct execution results and oracle EX, not just the number of SQL strings. A fresh candidate is useful when it adds a plausible answer that the existing candidates cannot express. Research support for diverse generation plus pairwise selection exists in [CHASE-SQL](https://arxiv.org/abs/2410.01943); that supports testing this architecture, not assuming its gains transfer automatically.

**3. Make judging a comparison of specific disagreements.** Group executable candidates by their complete result sets, then compare representatives against the answer specification. Agreement is a useful signal but is not proof: correlated generators can unanimously be wrong. Ask the reviewer which output slot, condition, population, grouping key, or tie behavior explains each difference. Show column names, widths, row counts, NULL counts, and relevant schema for the disagreement, rather than relying on the first three rows alone.

There is also a concrete implementation issue in `src/selection/judge.py`: a valid `selected_id` can accompany a nonempty `selected_sql` that differs from that candidate, and the latter is accepted. Return the ID and rationale, then retrieve the exact executed SQL by ID. If rewriting is desired, treat it as a separate candidate and execute/verify it before acceptance. This also avoids spending the judge's 512-token response budget repeating a potentially long SQL query.

**4. Replace unconditional repairs with semantic checks.** The current generator, judge, and fixer contain preferences that conflict with generalization:

| Current preference | Why it can fail | Replacement |
|---|---|---|
| Remove NULLs whenever they appear; treat mostly-NULL results as needing repair | Q538, Q618, Q637, Q648, Q839, Q856 and Q964 can lose legitimate gold rows. Q33 needs exclusion instead. | Require a question/evidence reason for changing row membership. NULL presence itself is diagnostic information. |
| Prefer nonempty candidates whenever alternatives are empty | An incorrect query can return plausible data; the correct answer can be empty. | Check predicate/value grounding, then decide whether emptiness is expected. Preserve the original executable candidate. |
| Prefer `LIMIT 1` for superlatives | Q590 and Q930 need multiple gold rows, while Q101 and Q794 use one. | Distinguish a scalar extreme, a single entity, a top-N request, and all ties. Use evidence and explicit ambiguity handling. |
| Add `DISTINCT` whenever duplicate output exists; strip `COUNT(DISTINCT id)` based on shared join keys | Output duplication, distinct entity counting, and aggregation over joined rows are different operations. Shared keys can still repeat after joins. | Determine row meaning and join multiplicity before changing aggregation or deduplication. |
| Ban bare-column casts; always use `STRFTIME` for year filters | Necessary typing depends on stored values; date ranges can be equivalent to year extraction for suitable ISO data. | Inspect storage types and date formats and preserve the intended comparison boundaries. |

The prompt claim that NULLs “break MIN” is incorrect: SQLite aggregate `MIN` and `MAX` ignore NULL inputs; an all-NULL group returns NULL. `ORDER BY ... ASC LIMIT 1` has different NULL behavior. Likewise `COUNT(*)` counts rows while `COUNT(column)` counts non-NULL values. These are SQL semantics, not BIRD-specific preferences. See [SQLite aggregate documentation](https://www.sqlite.org/lang_aggfunc.html) and [SQLite type/comparison documentation](https://www.sqlite.org/datatype3.html).

Repair should retain the original SQL, record the exact intended semantic change, and compare the original and revised candidate. Successful execution alone does not establish that a repair preserved the answer.

**5. Verify aggregation populations and relationship meaning.** Many executable errors are deeper than projection. Examples worth turning into general verification tasks:

- **Q788:** percentage of female heroes published by Marvel was interpreted as percentage of Marvel heroes who are female. The numerator can look similar while the denominator changes the question.
- **Q556:** counting joined badge rows and joined user names produces a ratio of 1; the denominator needs separate population reasoning.
- **Q108:** adding a strict `transaction date > account opening date` condition excludes a same-day transaction returned by gold. Check the meaning of temporal boundaries instead of inventing stricter filters.
- **Q533:** comparing a timestamp directly with a date literal includes timestamps on the boundary day; the gold first extracts the calendar date.
- **Q587:** grouping by post ID collapses comments that the gold returns separately. Check what one output row represents.
- **Q887:** excluding races by circuit ID differs from excluding by race name. Similar schema paths are not interchangeable.
- **Q791, Q1068:** missing real-number conversion causes integer division. These are materially different numerical answers, not merely last-digit floating-point differences.

Extend existing profiling with targeted probes when candidates disagree: NULL frequency, actual stored types, relevant literal values, join fan-out, and whether the extreme is tied. Probe only the columns/relations needed for the disputed interpretation. Never substitute “this query returns more rows” for evidence that its meaning is correct.

**Annotation conflicts require a separate assessment.** Some apparent mistakes should not become production rules. Q110 asks about September 2, 1998, while gold filters August 20, 1997. Q171's evidence specifies north minus east, while gold subtracts in the other direction. Q582's evidence identifies the last editor, while gold joins the owner. These concrete conflicts cannot be resolved reliably by an unseen-test system from the supplied input. Keep official EX unchanged, and separately record cases where the annotation conflicts with the question/evidence. Do not describe different outputs as SQL-equivalent solely because both sound reasonable.

**How to test the update without fitting the dev answers.** Develop the answer specification and candidate verifier on BIRD training data with databases held out during tuning. Learn output conventions, if needed, from training question/SQL pairs rather than dev question IDs or gold-derived column hints. Freeze prompts, model settings, schema generation, and SQLite version before evaluating a held-out split. Because the dev errors have now been inspected, do not present a tuned dev score as an untouched validation result.

Run separate ablations for output specification, semantic candidate diversity, disagreement-based selection, and conservative repair. Report EX, candidate oracle EX, number of distinct answers, selection losses, correct-to-incorrect repairs, incorrect-to-correct repairs, timeouts, and cost. A useful candidate-generation update raises the oracle; a useful selection update closes the gap without erasing correct final answers; a useful repair step improves net accuracy. Re-evaluating a manually refined file does not by itself demonstrate a method that generalizes.

**Reproduction and implementation locations.** The audit script uses read-only SQLite connections, validates question IDs and database names, saves input hashes, and checks full tuple sets. Column-order diagnosis requires one fixed column mapping for the entire result, not a different permutation for each row. Extra/missing-column diagnosis requires exact projected-set equality. These are result-level diagnostics on this database instance, not proofs of semantic equivalence. Numeric tolerance is diagnostic only; EX remains exact.

```powershell
python -u evaluation/evaluation.py --db_root_path data/bird_data/dev_databases/ --predicted_sql_path output/claude_headless_v6/ --ground_truth_path data/bird_data/ --data_mode dev --num_cpus 2 --mode_gt gt --mode_predict gpt --diff_json_path data/bird_data/dev.json --meta_time_out 30 --file_name selected.json --start 0 --end 1534
python -u analysis/v6_accuracy_audit.py --out analysis/v6_accuracy_audit/reproduced --workers 2
```

The key update locations are `src/agents/manager.py` (answer specification), `src/generation/prompt_builder.py` and `src/prompt/*.py` (candidate context and diversity), `src/selection/judge.py` (comparison and canonical ID selection), and `src/selection/executor.py` / `src/selection/fixer.py` (preserving candidate meaning during repair). There is an unused gold-reading `knowledge(entry)` helper in `src/pipeline.py`; I found no active call in the current pipeline. It must remain outside any test-time generation path.

The repository evaluator also loads prediction values in insertion order and pairs them positionally with gold, rather than joining by question ID. The present files are correctly aligned. Future resumed, sparse, or reordered files need explicit ID validation; otherwise evaluation can silently compare unrelated questions.
