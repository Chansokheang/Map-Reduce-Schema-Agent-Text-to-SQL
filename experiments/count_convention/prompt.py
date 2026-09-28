"""Tie COUNT(DISTINCT) to the evidence's own de-duplication instruction.

The signal, measured over all 1534 dev gold queries (493 of which contain a COUNT):

  evidence has a dedup imperative & gold uses COUNT(DISTINCT)    18   <- 18 of 18, no exceptions
  no imperative                  & gold uses plain COUNT        428
  evidence has an imperative     & gold uses plain COUNT           0
  no imperative                  & gold uses COUNT(DISTINCT)      47   <- mostly toxicology (20)

The imperatives are near-boilerplate: "Should consider DISTINCT in the final result", "Should
compute the number of distinct/unique ones", "Should return the number of distinct patients",
"Only count ones without repetitive", "Don't compute repetitive ones". They appear in four
databases (thrombosis_prediction 16, european_football_2 3, toxicology 2, student_club 1) and gold
obeys every one.

Hence the asymmetry in the rule below, which mirrors the asymmetry in the data:

  * imperative present -> DISTINCT is required. Absolute: 18/18, zero counter-examples.
  * imperative absent  -> plain COUNT is PREFERRED, not mandated. Those 47 gold queries deduplicate
    without being told to, so a prohibition would trade roughly as many losses as it gains. That
    is what sank the earlier attempts: a blanket strip measured -11 on v6 and +2 on gr_v1, and the
    fixer's PK-based condition -5 / +3.

Not restricted to one database on purpose. The imperative holds across the four databases that use
it, so the rule transfers to unseen databases; a `db_id == 'thrombosis_prediction'` guard would
raise the dev number and do nothing for a test-set submission.
"""

COUNT_SECTION = """

**RULE O — COUNT AND DISTINCT, DECIDED BY THE EVIDENCE:**
- If the evidence tells you to de-duplicate — "should consider DISTINCT in the final result",
  "should compute the number of distinct/unique ones", "should return the number of distinct
  <things>", "only count ones without repetitive", "don't compute repetitive ones" — then you MUST
  write `COUNT(DISTINCT <column>)`. This instruction is decisive whenever it appears.
- If the evidence says nothing about duplicates, PREFER plain `COUNT(<column>)`, even when a JOIN
  repeats rows per entity. A one-to-many join inflating the count is the expected reading here, not
  a defect to correct.
- Override that preference only when the question itself asks for distinct things ("how many
  different X", "how many unique X", or counting a category column where each value should be
  counted once, e.g. "how many elements are there").
- This rule takes precedence over RULE E's DISTINCT guidance for the argument of COUNT(). RULE E
  still governs `SELECT DISTINCT`."""

MARKER = "**RULE O — COUNT AND DISTINCT, DECIDED BY THE EVIDENCE:**"


def with_count_rule(system_prompt):
    """The system prompt with RULE O appended. Idempotent."""
    if not system_prompt or MARKER in system_prompt:
        return system_prompt
    return system_prompt.rstrip() + "\n" + COUNT_SECTION
