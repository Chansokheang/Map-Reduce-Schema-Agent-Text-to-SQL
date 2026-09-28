# data/column_meaning.json: what it contains and whether it explains the errors (2026-09-17)

BIRD-supplied per-column meanings for the 11 dev databases, keyed db|table|column.
Script: analysis/column_meaning_patterns.py (summary.json). The production pipeline already
embeds these meanings into its schema files (src/processing/extract_schema.py), so v6
generation had them. The selection, projection and question-form experiments used only the
supplied description CSVs.

## Content

| Measure | Value |
|---|---:|
| Entries | 699 |
| Median length | 173 characters (supplied CSV: 43) |
| Entries listing possible or example values | 291 |
| ... of which the CSV has no value list | 270 |
| Entries describing identifiers / keys | 129 |
| Entries describing codes or abbreviations | 90 |
| Entries describing formats or units | 44 |

Its distinct contribution is stored values: category lists such as the set_translations
language values, including 'Portuguese (Brazil)'.

## Leakage check

Of 1920 string literals in gold SQL, 118 do not occur in the question or evidence. 43 of
those appear in column_meaning and 29 in the CSVs. They are category values listed for a
column ('Directly funded', 'OWNER', 'Portuguese (Brazil)', 'east Bohemia'), the kind of
information a database value lookup also gives, not question-specific answers. No sign of
leakage from gold SQL. To transfer to the test set, the same kind of file must exist or
be generated for the test databases.

## Relevance to failures of selected.json

- **Extra or missing output columns: no signal.** In 76 failures where the prediction
  outputs a different column from gold, the column meaning matches the question's words
  better for gold's column 19 times, for the predicted column 20 times, and ties 37 times.
  The CSV gives 23 / 23 / 30. Similar columns are described alike: frpm."School Name" and
  schools.School are both "the name of the school".
- **Values: some signal.** 53 failures lack a gold string literal. For 35 the value is in
  the question or evidence already; column_meaning contains it for 34. Only 3 values are
  found in column_meaning and nowhere else (Q144, Q852, Q1108). Its value lists matter more
  for conflicts between evidence and data, e.g. Q405, where the evidence says 'Brasil' and
  column_meaning lists the stored 'Portuguese (Brazil)'; earlier judges regressed that
  question by following the evidence spelling.
