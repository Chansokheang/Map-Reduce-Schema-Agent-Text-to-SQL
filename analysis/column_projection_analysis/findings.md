# BIRD result-column conventions

The relevant question is how the requested answer maps to an ordered SELECT list: its attributes, number of physical columns, entity representation, and value representation. The earlier NULL/duplicate experiment did not test that question.

I parsed all 1,534 development gold queries and the original selected predictions. All gold queries parsed successfully. SQLite independently confirmed every gold output width by preparing a read-only, zero-row query. The original prediction Q1199 could not be parsed because its SQL is incomplete. Existing execution diagnostics were reused to inspect 28 extra-column, 22 missing-column, and 10 column-order cases. Those categories describe observed result equivalence, not automatically correct repairs.

## Broad distribution

| Gold output columns | Questions | Share |
|---|---:|---:|
| 1 | 1,255 | 81.81% |
| 2 | 198 | 12.91% |
| 3 | 68 | 4.43% |
| 4 | 11 | 0.72% |
| 5 | 1 | 0.07% |
| 6 | 1 | 0.07% |

No gold query uses `SELECT *` or `table.*` in its final output projection. This supports explicitly selecting answer fields, rather than returning entire joined records. It does not mean every question should have one column.

Of 317 questions containing the phrase "how many," 305 have one output column and 12 have multiple columns. This is a lexical cohort, not an automatic intent classifier. Q172 asks for owner and disponent counts and returns two separate sums. Q1004 asks for wins plus a full name and returns three columns. Q978 contains "how many" but its gold returns location, latitude, and longitude without a count.

## Conventions supported by inspected cases

| Convention | Evidence | Appropriate scope |
|---|---|---|
| Use evidence that explicitly defines an answer field or composite output. | Q37's evidence defines the address as Street, City, State, Zip; gold follows that order. Q812's evidence maps the requested superhero names to `superhero_name`, despite "full names" in the question. | Inspect the role of the evidence: a filter mapping is not automatically an instruction to project that column. |
| Preserve the order of requested answer attributes, including later sentences. | Nine of the ten inspected column-order failures have gold order consistent with the question's requested sequence. Q58 is phone, extension, school; Q898 is age, forename, surname; Q1467 is total amount, event name. The remaining case, Q37, follows the evidence's explicit composite-field order. | A strong hypothesis for ordering candidates; this is a ten-case diagnostic subset, not a measured full-corpus success rate. There is no supported universal ID-first or name-first rule. |
| Preserve separate physical components of requested composite attributes. | Among 34 manually identified requests to output "full name(s)," 23 use separate name components, and 11 use one stored descriptive field. Q36 returns six fields for three administrators; Q878 uses forename and surname; Q1314 uses first_name and last_name; Q720 uses the stored full_name field. | Use the supplied schema and evidence. Do not assume a full name is always one output column or always two. Three other questions mentioning "full name" were excluded because the phrase was a filter or a counted condition. |
| A sorting or filtering attribute need not be displayed. | Q907 orders races by date but returns race name and country. Q988 uses average pit-stop duration to choose drivers but returns their name components. Q1216 sorts patients by birthday but returns ID. | Distinguish "sort by X" from "show X." Do not append every WHERE or ORDER BY field. |
| Separate requested measures remain separate output columns. | Q172 returns owner count and disponent count. Q215 returns iodine count and sulfur count. Q698 returns comment count and answer count. | Do not collapse multiple requested counts into one total. Group labels are included when the question asks for per-group answers, such as Q83's city and count. |
| An explicit request to produce a ranking may require a rank column. | All three questions that begin with the command "Rank"—Q17, Q726, Q728—return entity/category, ranking measure, and a computed RANK(). | Only three observed examples. The other eleven questions containing rank/ranked/ranking use ranks as filters, stored attributes, or context; the keyword alone is insufficient. |
| Preserve the requested or supplied value representation. | Q289 returns a stored carcinogenic label; Q469 returns YES/NO; Q1177 returns Normal/Abnormal; Q1205 returns SQL boolean values; Q1223 requests and returns the strings True/False. | There is no universal boolean format. Aliases do not resolve value differences. Avoid inventing a display label when a stored code or an explicitly requested representation answers the question. |

The final projection contains a concatenation expression in only one gold query, Q221. It reconstructs an atom identifier from stored components. That is a reason to distinguish combining answer attributes for display from constructing a required value; a blanket ban on every concatenation is too broad. The full-name examples preserve separate stored components rather than joining them into a formatted string.

## ID versus name is not standardized

All of these occur in the same development set:

| Requested entity | Gold output example |
|---|---|
| Transactions | Q159, Q165: transaction ID only |
| Driver | Q985: driver ID only; Q878: forename and surname |
| Users | Q628: ID and display name, although evidence mentions DisplayName |
| Players | Q1064: ID and player name |
| Patient attributes | Q1207: sex and birthday, without patient ID |
| Player attributes | Q1144: ID, finishing, curve, even though the question asks for the last two attributes |
| Expenses | Q1370: expense ID and description, although evidence maps expense to expense_description |
| Product description | Q1503: product ID and description |

These examples do not justify always adding an ID, always removing an ID, or always returning ID plus name. The choice sometimes reflects the entity's schema representation and sometimes includes an additional identifier not clearly specified by question/evidence. Such ambiguity should remain visible in candidate generation or selection rather than being converted to a hard database-specific rule.

## Corrections and remaining uncertainty

The earlier audit cited Q37 as a question/gold order disagreement without explaining its evidence. On closer inspection, the evidence supplies exactly the gold order. Similarly, Q812's apparent full-name mismatch is explained by its explicit evidence mapping. These are useful examples of evidence-guided output interpretation, not unexplained annotation errors.

Other cases remain unresolved from the supplied wording:

- Q445 asks for language, flavor text, and card type; gold returns only language and flavor text, with no additional evidence.
- Q888 asks for country, circuit name, and location; gold omits circuit name. Its evidence only explains "first."
- Q1144 adds an ID not stated in the requested attributes or evidence.
- Q1179's evidence names aCL IgM, while gold returns IgA, IgG, and IgM.

Consequently, "project exactly the question's nouns," "always follow evidence literally," and "always include the entity ID" are not universal explanations of the references. A dev-derived convention is evidence to test, not a guaranteed private-test standard.

## What this means for the existing declared prompt

The existing prompt already contains useful observations, particularly minimal output and preserving separate answer components. The next revision should distinguish output interpretation from unrelated SQL rewriting. Its output-focused priorities should be:

1. Identify requested answer attributes and their order across the question, including follow-up sentences.
2. Apply explicit evidence mappings and definitions of composite answer fields; distinguish them from filter hints.
3. Expand composite attributes using the supplied schema's actual representation.
4. Keep requested measures separate, and include grouping labels or rank values when the answer explicitly requires them.
5. Avoid extra display, filter, and sorting columns; treat ambiguous ID/name choices as uncertain.

These are proposed interpretation principles, not an implemented or validated accuracy improvement. No production prompt, original prediction, or BIRD description was changed. To assess transfer, freeze any resulting prompt and evaluate on databases not used to formulate it. Because the development ground truth has already been inspected, another improvement on this same dev set would remain development evidence.

## Evaluation format versus answer format

The [published BIRD evaluator](https://github.com/AlibabaResearch/DAMO-ConvAI/blob/main/bird/llm/src/evaluation.py) compares sets of result tuples. Within that evaluator, column positions and values matter; result-column aliases do not. Row order and repeated identical rows are ignored. This defines execution comparison, but it does not prescribe one universal SELECT-list format for every natural-language question. Empty result sets can also conceal projection differences in execution-only comparisons.

Artifacts: `all_projections.csv` and `all_projections.json` contain all 1,534 question/evidence/projection comparisons; `summary.json` contains counts and diagnostic IDs; `sqlite_width_validation.json` records the independent width checks. The reproducible parser is `analysis/projection_conventions.py`, using sqlglot 30.18.0. Automatic phrase counts are separated from the manual interpretation of individual cases above.
