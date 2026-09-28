# Completed paired selection experiment: keep the original

The proposed disagreement-based reranker did not improve full-dev accuracy.
Keep the original predictions and production selection behavior. The experimental
files remain separate for review; no production prompts or original predictions
were changed.

| Version | Correct / 1,534 | EX | Recovered original failures | Regressed original successes |
|---|---:|---:|---:|---:|
| Original selected | 1,102 | 71.84% | — | — |
| Control criteria, experimental interface | 1,102 | 71.84% | 30 | 30 |
| Disagreement criteria, experimental interface | 1,092 | 71.19% | 24 | 34 |

The treatment loses 10 correct answers overall, or 0.65 percentage points. It
recovers 24/88 (27.27%) of the known candidate-selection opportunities, leaving
64. That benefit is outweighed by losing 34/195 (17.44%) of the originally
correct answers inside the review gate. The new criteria beat the control on
17 questions but lose on 27.

## What was actually run

- All 1,534 questions passed through a gold-free execution-based gate.
- The five candidate result sets disagreed on 435 questions. Both judge arms
  reviewed those questions; the other 1,099 answers were preserved.
- The original selected SQL was an additional choice, preserving access to
  answers outside the five-candidate pool.
- There were 870 structured CLI requests, no failed reviews, and no fixer calls.
- Control changed 115 SQL strings; treatment changed 112. Some changes preserved
  the execution answer, so SQL-change counts are not recovery counts.
- All three complete files and the saved candidates were scored using the
  repository's unmodified execute_model worker: 5,574 unique predicted/gold
  pairs, sharing scores for identical SQL.
- Model usage logs record claude-sonnet-5 and claude-haiku-4-5-20251001. Both arms
  requested the same sonnet alias. This is not a same-model replay of the earlier
  Sonnet 4.6 fixer experiment. CLI-reported list-price cost was $34.0822; this does
  not necessarily represent subscription billing.
- Hash verification confirmed every original prediction source is unchanged.

The initial v1 live check is preserved. Its treatment response omitted a JSON
brace and was recorded as a failure. V2 used CLI structured output for both arms
and completed without that failure. V1 outcomes were not merged into v2.

## Why the broader judge lost accuracy

The observed changes and recorded explanations suggest the judge often replaced
the existing answer with a plausible interpretation that was not the reference
interpretation. Explanations are diagnostic evidence, not proof of causal reasoning.

1. **Counting distinct entities:** eight regressions introduced distinct-entity
   counting where gold counts records or joined observations: Q383, Q605, Q957,
   Q1137, Q1295, Q1297, Q1298 and Q1299. For example, Q1295 changes four matching
   observation combinations to one patient. This repeats the failure observed
   in the earlier fixer experiment. It does not justify a global ban on DISTINCT.

2. **Output representation:** five regressions changed the projected columns or
   their representation: Q46, Q120, Q176, Q992 and Q1000. Q46 added the ranking
   metric even though gold returns only the school. Q992 concatenated first and
   last names, whereas gold returns separate columns. Q1000 is a different
   problem: its evidence explicitly requests location plus country but gold
   returns location alone. These should not be reduced to one universal rule.

3. **Ties:** Q802, Q810 and Q1389 changed a single top row to all tied rows,
   losing EX. Other dev questions require multiple tied rows, so forcing LIMIT 1
   everywhere would reproduce another brittle convention.

4. **Literal evidence conflicts:** Q273, Q324, Q405, Q474, Q614 and Q1171 show
   failures from a literal reading of evidence. Examples include a percentage
   formula omitting multiplication by 100, “under 100” mapped to <10, and
   “underage” mapped to birth year <18. The judge sometimes stated that evidence
   was authoritative despite the treatment explicitly avoiding a universal
   precedence rule.

5. **Unsupported factual claims:** Q1085's judge claimed that no player named
   exactly Alexis exists and expanded the lookup to Alexis Sanchez, also changing
   MAX to AVG. A post-evaluation read-only lookup confirmed that an exact Alexis
   row does exist, alongside seven longer names beginning with Alexis. The
   candidate sample did not support the judge's absence claim.

6. **Reference/metadata tension:** Q729 excludes zero heights because the supplied
   BIRD description explicitly identifies zero as missing. Gold still averages
   the stored zeros. Q979's gold uses COUNT(time IS NOT NULL), which counts both
   true and false non-NULL boolean expressions; the treatment instead counts
   rows whose time is not NULL. These execution regressions cannot honestly all
   be called semantic mistakes in the selected candidate.

The first four groups cover 22 distinct regressions; the remaining cases involve
population, field meaning, missing values, types, or row selection. All 34 are
listed with the original SQL, selected SQL, gold SQL and explanation in the
offline changed_queries.csv.

There were genuine opportunities the treatment recovered, including Q48
(denominator population), Q56 (City versus MailCity), Q58 (column order), Q453
(manaCost versus convertedManaCost), and Q866 (missing first name). Thus candidate
comparison can help, but this experiment did not control its regressions well enough.

## Recommendation for a next experiment

Do not promote either experimental full file: treatment is worse, while control
breaks as many original successes as it recovers.

A narrower next experiment would **verify a proposed change before accepting it**:

- Establish the requested output attributes before showing candidates. Require an
  explicit question/evidence basis for adding, removing or concatenating fields.
- Verify factual assertions such as “this exact name does not exist” with a
  read-only database lookup. A candidate's sample rows are not an absence test.
- If switching depends only on unresolved counting grain, tie treatment, output
  convention or conflicting evidence, retain the original instead of accepting
  a confident explanation as sufficient support.

These checks must operate on the actual question, supplied schema and available
database at inference time. They must not use the IDs above, gold answers, or
dev-specific column rules. Do not treat this suggested follow-up as a measured
improvement; it has not been implemented or run.

Pin the model in the next experiment and test on databases not inspected during
development. The dev set has already informed this prompt, so even a future dev
gain would not establish private-test generalization.

## Artifacts

- output/selection_experiment/v2/full_results/original.json
- output/selection_experiment/v2/full_results/selected_control.json
- output/selection_experiment/v2/full_results/selected_disagreement.json
- output/selection_experiment/v2/full_results/evaluation/summary.json
- output/selection_experiment/v2/full_results/evaluation/README.md
- output/selection_experiment/v2/full_results/evaluation/changed_queries.csv

The last CSV contains gold and evaluation labels and is for offline review only.
The repeatable report generator is analysis/summarize_selection_experiment.py.
