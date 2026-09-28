# Review of 88 failures with a passing saved candidate

**Recommendation: test a reranker that compares the actual differences between executed candidates, using the question, supplied evidence, original schema descriptions, and candidate execution summaries. Keep generation fixed and prohibit SQL rewriting during this experiment.** No ground-truth answer, correct-candidate label, question-ID lookup, or development-specific field rule is needed by that reranker.

This is a completed offline review and an experiment proposal, not an implemented accuracy gain. I inspected all 88 selected/passing-candidate comparisons and the current judge implementation. Original predictions and production code are unchanged; saved prediction hashes match the prior audit. The passing labels below come from execution comparison on dev and are for analysis only.

## What the 88 cases actually contain

| Manual primary difference between selected and passing SQL | Cases | Representative examples |
|---|---:|---|
| Field or relationship meaning | 21 | Q56: City versus MailCity; Q453: manaCost versus convertedManaCost; Q1195: patient diagnosis versus examination diagnosis; Q1418: event type versus budget category. |
| Output attributes or representation | 20 | Q58/Q898: column order; Q866: missing first name; Q280/Q288: stored codes versus invented labels; Q1406: budget ID versus the maximum amount. |
| Population and join membership | 13 | Q48: Orange County applied to only the numerator; Q83: conditions applied to one subquery but not the displayed city counts; Q384: omitting the cards join includes additional UUIDs. |
| Predicates, arithmetic, and types | 12 | Q108: exclusive/inclusive date boundary; Q376: exact versus substring match; Q486/Q943: conflicting formula or scale conventions; Q1108: question/evidence date conflict. |
| Aggregation grain | 9 | Q15: averaging a stored average again; Q1218/Q1525: counting distinct people versus observation/transaction rows; Q1282: top observations versus top people. |
| NULL and zero policy | 7 | Q618/Q637/Q648/Q839/Q856 remove reference rows with defensive NULL filters; Q842 excludes zero heights; Q1178 needs a different NULL treatment. |
| Top-N, ties, and row selection | 6 | Q590: one minimum row versus all tied rows; Q580: DISTINCT changes which rows survive LIMIT; Q1032: equally sized leagues produce different top-one choices. |

These are manually assigned primary observed differences, not mutually exclusive semantic mechanisms or proven causes of the historical judge's decision. Every case has one primary assignment for accounting. They differ from the earlier result-only categories, which identify 2 order, 5 extra-column, 2 missing-column, 5 NULL-only, and 74 other mismatches.

Additional measurements:

- 78 cases have the same output width as at least one passing candidate. In 49 cases, all passing candidates also have the same physical row count as the selected output. Shape checks alone are insufficient.
- 36 cases have exactly one passing candidate; 30 have two; 12 have three; 10 have four. A correct candidate is often a minority.
- The prior plurality selector recovers only 26 of these 88 and scores 1,067/1,534 overall, below the original 1,102. Result popularity is not a substitute for interpretation.
- 76 selected SQL strings match a saved candidate exactly. After parser formatting normalization, 87 match. The only remaining nonmatching case is Q866. The twelve raw string mismatches should not be interpreted as twelve substantive rewrites.
- Candidate-result disagreement occurs on 435 questions and covers all 88 opportunities. Those 435 also contain 195 currently correct answers and 152 other failures. This is an observable runtime trigger; the 88 passing-label cases are not.

## What to change in the selection interface

**1. Always provide the relevant supplied schema.** The current judge includes schema only when a regex-based column-set detector reports a difference. Applied to the saved candidates, this gate would omit schema in 15 of the 88 cases. Q453 is particularly useful: bare SELECT fields differ between manaCost and convertedManaCost, while the detector still considers the candidates equivalent on columns. Include the union of referenced tables, their keys, and supplied descriptions for the competing fields. Do not generate replacement descriptions from dev gold.

**2. Show the disagreement, not just three arbitrary leading rows.** Supply ordered output expressions and names, row counts, NULL counts, and a few rows present in one candidate result but absent from another. These comparison rows must be computed candidate-to-candidate, never candidate-to-gold. For scalar queries, identify which numerator, denominator, filter scope, or counting unit differs. Different results prove disagreement; they do not identify the correct candidate.

**3. Ask for evidence supporting the decision.** For Q56, explicitly compare whether "mailing state" also modifies the city requirement; the supplied MailCity description identifies it as the mailing city. For Q48, compare whether both sides of the ratio concern Orange County. For Q1218, compare whether the answer counts patients or lab observations. These are reusable questions about semantics rather than hard-coded table or column choices.

**4. Select an ID and retrieve its exact executed SQL.** Current `_llm_judge` validates selected_id but accepts a nonempty selected_sql even if it differs from the chosen candidate. This is a code-level risk, not proof of a historical failure. An ID-only interface prevents an unexecuted rewrite and reduces output-format burden. Disable the subsequent fixer in this experiment so its changes cannot be mistaken for selector gains.

**5. Reduce presentation bias.** Hide strategy names, group duplicate result sets without treating group size as a correctness vote, and balance candidate display order between repeated trials. A shortlist must retain one representative of every distinct successful result; do not discard the minority result containing the only passing candidate.

The current prompt strongly prefers nonempty results, particular MIN/MAX and LIMIT forms, NULL filtering, ID-counting, and specific date/cast syntax. These preferences should be tested as semantic questions rather than universal rules. Existing prompt code postdates portions of the saved run, so current-code findings identify risks for a new experiment, not verified causes of each historical error.

## What cannot be solved reliably from these labels

The measured 88 is an offline opportunity count, not a promise that all 88 can be identified without gold:

- Q469 differs only in YES/NO versus Yes/No casing, without an explicit casing instruction.
- Q486's supplied evidence uses SUM(convertedManaCost) as a denominator, while the gold uses a count of cards.
- Q943's evidence describes a ratio, while the gold multiplies by 100.
- Q1108 asks about 2011 but its evidence states 2012; the passing candidates follow 2011.
- Q1032 has different top-one choices among leagues with the same maximum match count. A global COUNT(id) preference would misdiagnose the decisive issue.
- The local supplied transactions description says `total price = Amount x Price`, but Q1510/Q1511 gold answers average Price alone. Rewriting that description to fit dev would hide rather than solve the conflict.
- Q998's passing candidate has a substantially different interpretation from the gold but happens to match its returned answer. Execution agreement does not prove semantic equivalence.

Record these as conflicts or weakly determined representations. Do not infer a universal rule such as always uppercase labels, always override evidence, always count distinct IDs, or always use the shorter SQL.

## Proposed first experiment

1. Freeze the five saved candidates and the current selected SQL. Include the current selected SQL as an additional option, since the existing audit found 21 correct selected answers absent from the five-candidate pool.
2. Gate review on disagreement between successful results from the five saved candidates. On this saved dev run that selects 435 cases; the current selected SQL is an additional selection option inside that gate. Preserve the current selected output elsewhere. At test time the same gate uses only execution results, not the list of 88 IDs. Expanding the gate itself to include the current selected SQL requires recounting its coverage. If the submission interface does not provide database execution, use a separately validated SQL-difference trigger and supplied metadata instead; the measured 435 then does not apply.
3. Use three comparisons: the unchanged saved baseline; a reranking control using the current judge criteria; and a reranker using the disagreement-based criteria. Both reranking arms receive identical candidate pools, original metadata, execution summaries, and call budgets. Both return only candidate IDs and have the fixer disabled. Thus this is an additional reranking experiment; it is not yet a replacement of the first judge.
4. Allow an explicit undecided response and retain the current output when it occurs. Log malformed responses and provider failures rather than treating them as successful decisions. Do not accept a self-reported confidence number as calibrated probability.
5. Complete all inference before scoring against gold. The model receives an explicit allowlist of question, evidence, schema metadata, candidate SQL/results, and candidate-to-candidate differences. None of the labelled files from this review enters its context.
6. Evaluate full-dev net change, recoveries among the 88, regressions among the 195 currently correct disagreement cases, per-database results, abstentions, and model cost. Repeat with balanced candidate order to measure variability. Never report a recovery rate on the 88 alone as overall accuracy improvement.

The hypothetical best outcome preserving the original successes and recovering all 88 would be 1,190/1,534 = 77.57%; that is an oracle bound, not an expected score. Some cases above are underdetermined from the allowed inputs.

For private-test transfer, no answer lookup or dev-specific correction list should ship. Freeze the selector and validate on databases not inspected while developing it. Public training data can supply pairwise preference labels if a learned selector is later needed, with separation by database between fitting and validation. Since these dev cases have now been inspected, further dev gains remain development evidence rather than an independent generalization claim.

[CHASE-SQL](https://arxiv.org/abs/2410.01943) provides a research precedent for diverse generation and pairwise candidate selection. Its selector and training setup differ from this proposal; its reported results do not predict gains for this project.

Artifacts: `cases.csv` has all 88 selected/passing comparisons and manual primary differences. `offline_labelled_cases.json` retains all five candidates and labelled diagnostics and is explicitly offline-only. `summary.json` contains the measurements and unchanged prediction hashes. The reproducible extraction is `analysis/review_recoverable_selection.py`.
