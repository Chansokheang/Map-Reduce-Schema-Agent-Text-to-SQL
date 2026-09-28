"""Decompose the question into its requested outputs, in order, then project exactly those.

Narrower than experiments/reasoning_prompt: that one decomposes the whole problem (filters,
grouping, shape, subqueries). This one asks for one thing only — an explicit list of what the
question asks to SEE, in the order it asks — and then requires the SELECT list to match it.

Why a decomposition step rather than RULE L/M (experiments/generation_rules): those are
instructions the model can acknowledge without applying. Writing the list out first makes the
projection an explicit intermediate result, the same reason the post-hoc projection review works
by naming indices rather than rewriting SQL. The measured target: 62 swapped-column, 33 extra-
column, 24 projection-width and 20 missing-column failures in gr_v1's 1300 questions, plus gold
following the question's order in 65 of 71 measurable cases.

The two instructions that forbid explanations are rewritten, and the final query must be the last
fenced block — the shipped extractor takes the FIRST fenced block, so the patch that installs this
also makes extraction prefer the last one (shared with the reasoning arm).
"""

PROJECTION_SECTION = """

**BEFORE THE QUERY — LIST THE REQUESTED OUTPUTS:**
Write one line, then the query. Nothing else.

`outputs: <thing the question asks to see>, <the next thing>, ...`

How to build that line:
- Read the question left to right and list only what it asks you to SEE, in that order.
- A trailing request ("Include the school name", "Indicate the city", "Also state X", "List them
  by name") is still requested — it goes LAST, after the main question's outputs.
- Do NOT list anything the question only uses to find or rank rows: filter values, join keys,
  identifiers you were given rather than asked for, or the column you sort by (unless the question
  also asks to see it).
- If one requested thing spans several columns, list its parts in their conventional order:
  full name -> first name, last name; complete address -> street, city, state, zip.
- Name the table for each part when two tables could supply it, using the column descriptions.

Then write the final query so that its SELECT list is exactly that list, in exactly that order —
same number of items, same sequence. Put the query in one fenced block, last:

```sql
SELECT ...
```

Everything in RULE A-K and the general rules above still applies to the query itself."""

ONLY_SQL_OLD = "1. Generate ONLY the SQL query, no explanations"
ONLY_SQL_NEW = ("1. Write the single `outputs:` line described below, then exactly one SQL query "
                "inside one ```sql fenced block as the last thing you write")
NOTHING_ELSE_OLD = "Return only the SQL query, nothing else"
NOTHING_ELSE_NEW = "End with the SQL query in one ```sql fenced block; no text after it"
MARKER = "**BEFORE THE QUERY — LIST THE REQUESTED OUTPUTS:**"


def with_projection_order(system_prompt):
    """The system prompt with the projection decomposition step. Idempotent."""
    if not system_prompt or MARKER in system_prompt:
        return system_prompt
    out = system_prompt
    if ONLY_SQL_OLD in out:
        out = out.replace(ONLY_SQL_OLD, ONLY_SQL_NEW)
    if NOTHING_ELSE_OLD in out:
        out = out.replace(NOTHING_ELSE_OLD, NOTHING_ELSE_NEW)
    return out.rstrip() + "\n" + PROJECTION_SECTION


def rewrote_output_instructions(system_prompt):
    return ONLY_SQL_OLD not in system_prompt and NOTHING_ELSE_OLD not in system_prompt
