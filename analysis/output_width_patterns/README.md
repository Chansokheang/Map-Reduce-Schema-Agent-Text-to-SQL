# Extra and missing output columns: patterns in the question text (2026-09-17)

Offline analysis of selected.json against dev gold. Script: analysis/output_width_patterns.py
(cases.csv, summary.json), plus manual reading of every width failure. Uses gold; findings
describe dev annotation behaviour and are not validated on other data.

## Where width errors happen

81 of 1533 parsed questions return a different number of columns from gold (47 wider,
34 narrower). The execution audit labels 27 as pure extra-column and 19 as pure
missing-column failures.

| Question form | Questions | Width errors |
|---|---:|---:|
| Two-part: a question followed by an instruction ("...? Indicate his name.") | 138 | 23 (16.7%) |
| Everything else | 1395 | 58 (4.2%) |

## Patterns with a signal in the question text

| Pattern in the question | Gold behaviour | Selected SQL |
|---|---|---|
| Starts with "Rank ..." (Q17, 726, 728) | 3/3 add a RANK() OVER column plus the ranking measure | 0/3 |
| "Which/Who <entity> ...? Give/State/Indicate/List <attribute>" | By manual count, 38 of 43 return only the instruction's attributes, not the entity as well | Mostly follows; fails Q986, Q989 by adding the entity |
| "How many ...? List them / list the IDs" (Q435, 436, 1187, 1188) | 4/4 return only the list, not the count | 2/4 |
| "What is <value> ...? Indicate <attribute>", or "include / along with / as well as" | Keeps both parts in most cases (e.g. Q44, 50, 51, 75, 80, 1004) | Mostly follows |
| "List all <entity>" with no attribute named (Q159, 165, 628, 1064, 1370) | Returns the entity's id | Returned `*` or names. Note: the id-over-name convention did not hold on train gold (38%) |

## Not a pattern

- **Sort keys.** For superlative questions gold returns the ORDER BY column 18-40% of
  the time depending on question form; selected SQL matches those rates, with only 13
  disagreements across 262 such questions.
- **Gold inconsistency.** Toxicology atom pairs: "which atoms are connected" (Q211, 223)
  returns one atom column, "atoms connected to / atoms of the bond" (Q217, 248, 252)
  returns two. Gold drops a requested part in Q125, 231, 445, 600, 978, 993, 1021 and
  keeps the entity against the majority "Which ...? Give ..." pattern in Q49, 520, 773,
  866, 928. Q1179 returns three antibody columns where the evidence names one.

## Reach

Text-form rules could address roughly a dozen failures (Rank 3, count-then-list 2,
which/who-then-attribute 2, entity listing about 5), and the last group carries
regression risk. Most remaining width errors sit where gold departs from its own
majority pattern, so no rule read from the question alone recovers them.
