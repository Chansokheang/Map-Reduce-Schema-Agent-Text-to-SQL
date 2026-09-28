"""Reasoning-prompt text: decompose the question, map it to the schema, build subqueries first.

A hybrid, not a replacement. The five strategy prompts keep RULE A-K and their general rules —
those cover output conventions and every convention change we measured was worth zero or less.
What they do not cover is the largest failure bucket: 120 of gr_v1's 396 failures use gold's
tables and columns and still get the logic wrong (Q65 divided by the total instead of "all other
types"; Q1004 used `dob = (SELECT MIN(dob))` instead of `ORDER BY dob LIMIT 1`). That is what the
decomposition step targets.

Two instructions in the existing prompts contradict a reasoning response ("Generate ONLY the SQL
query, no explanations" and "Return only the SQL query, nothing else"), so they are rewritten
rather than fought with — an instruction the model must disobey to follow the new one is worse
than either alone.

Output shape is deliberately narrow: the analysis is plain prose, and **only the final query is
fenced**. The pipeline's extractor takes a fenced block, and intermediate fenced subqueries would
be picked up instead of the answer. The patch also makes extraction prefer the LAST fenced query
as a second line of defence, because models do ignore formatting instructions.
"""

REASONING_SECTION = """

**HOW TO WORK THE PROBLEM (before writing the final query):**
Write a short analysis in plain prose, then the final SQL. Keep the analysis brief — a few lines.

1. DECOMPOSE: state what the question asks for, in order: the value(s) to return, the filters, any
   grouping, any ordering or ranking, and how many rows the answer should have.
2. MAP: name the table and column for each part, using the evidence and the column descriptions.
   Say which columns are only needed for filtering or joining, and therefore do not belong in the
   SELECT list.
3. CHECK THE SHAPE: does the question ask for one row or many? A ranking ("highest", "oldest",
   "top N") is `ORDER BY ... LIMIT N`, not equality against a subquery, unless ties must all be
   returned. A ratio or percentage: state the numerator and the denominator explicitly — "compared
   to all other X" is not the same denominator as "of all X".
4. SUBQUERIES: if the answer needs one, write it in prose first and say what it returns, then use
   it inside the final query. Do not fence intermediate SQL.
5. FINAL QUERY: one SQL statement, in a single fenced block, as the last thing you write:

```sql
SELECT ...
```

Everything in RULE A-K and the general rules above still applies to that final query."""

ONLY_SQL_OLD = "1. Generate ONLY the SQL query, no explanations"
ONLY_SQL_NEW = ("1. Work through the analysis steps below, then give exactly one final SQL query "
                "inside a single ```sql fenced block as the last thing you write")
NOTHING_ELSE_OLD = "Return only the SQL query, nothing else"
NOTHING_ELSE_NEW = ("End with the final SQL query in one ```sql fenced block; no text after it")
MARKER = "**HOW TO WORK THE PROBLEM (before writing the final query):**"


def with_reasoning(system_prompt):
    """The system prompt rewritten for a reasoning response. Idempotent."""
    if not system_prompt or MARKER in system_prompt:
        return system_prompt
    out = system_prompt
    if ONLY_SQL_OLD in out:
        out = out.replace(ONLY_SQL_OLD, ONLY_SQL_NEW)
    if NOTHING_ELSE_OLD in out:
        out = out.replace(NOTHING_ELSE_OLD, NOTHING_ELSE_NEW)
    return out.rstrip() + "\n" + REASONING_SECTION


def rewrote_output_instructions(system_prompt):
    """True when neither contradicting instruction survives — checked by the patch and the tests."""
    return ONLY_SQL_OLD not in system_prompt and NOTHING_ELSE_OLD not in system_prompt
