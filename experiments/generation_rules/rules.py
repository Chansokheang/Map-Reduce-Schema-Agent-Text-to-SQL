"""Three generation rules the existing prompts do not state, plus a matching fixer correction.

The five strategy system prompts already carry RULE A-K (CAST, percentage arithmetic, COUNT
argument, ROUND only on request, DISTINCT, JOIN preference, STRFTIME, no string concat, minimal
tables, NULL for ASC/MIN, superlatives). These three are the gaps that the 700-question review
found, so only they are added — restating the rest would duplicate and, where the wording
differs, contradict what is already there.

Where each comes from:

  RULE L (project only what is asked)  5 of 9 inspected simple regressions projected extra
                                       columns; the evaluator compares tuples, so one extra
                                       column fails the row outright.
  RULE M (column order)                2 of those 9 were a pure column swap. Gold follows the
                                       order the question asks in 65 of 71 measurable cases;
                                       the exceptions are conventional groupings, which the
                                       rule names explicitly.
  RULE N (no unstated conditions)      3 of those 9 added a qualifying predicate the question
                                       never states (`type = 'OWNER'`, `type = 'VYDAJ'`, an
                                       extra status filter).

FIXER_NULL_RULE corrects a contradiction rather than adding anything: the fixer runs after the
judge and its NULL check is written as MANDATORY whenever the result contains a NULL, so on the
~12% of questions it touches it would re-add exactly what RULE N forbids. The replacement keeps
the cases the fixer's own rationale names (ASC sort, MIN, single-row answers) and drops the
blanket instruction. Measured: removing unrequested NULL filters from the sl_v1 run is worth
+4 on 700 (5 fixed, 1 broken).
"""

GENERATION_RULES = """

**RULE L — PROJECT ONLY WHAT IS ASKED:**
- SELECT exactly the columns the question asks to see, and nothing else. No identifiers, keys or
  extra attributes added for context, and no columns that only appear in the filter or the
  ordering. If the question asks for phone numbers, return the phone column alone.

**RULE M — COLUMN ORDER:**
- Return the columns in the order the question asks for them, reading the question left to
  right. A trailing request such as "Include/Indicate/Also state X" comes last. Keep a
  conventional grouping intact when one applies (street, city, state, zip).

**RULE N — NO UNSTATED CONDITIONS:**
- Filter only on conditions the question or the evidence states. Do not add qualifying
  predicates of your own — type, status or category filters, date bounds, or IS NOT NULL beyond
  the narrow ASC/MIN cases in Rule J. A NULL or a duplicate row that the question did not
  exclude is part of the answer."""

FIXER_NULL_RULE_OLD = (
    "- **NULL (MANDATORY when present)**: `has_nulls=True` → add `IS NOT NULL` on the affected "
    "column. Skip ONLY when the question explicitly counts/lists NULLs (e.g., \"how many "
    "missing\", \"which records have no X\"). NULL leaks break ASC sort, MIN(), and single-row "
    "\"what is X\" answers."
)

FIXER_NULL_RULE_NEW = (
    "- **NULL (only where it changes the answer)**: `has_nulls=True` → add `IS NOT NULL` ONLY "
    "when the NULL actually corrupts the answer: an ASC sort or MIN() on that column (NULLs sort "
    "first), or a single-row \"what is X\" answer that returned NULL. For a multi-row list or a "
    "count, a NULL row is part of the answer the question asked for — leave it in. Never add a "
    "NULL filter the question did not ask for."
)

RULE_MARKER = "**RULE L — PROJECT ONLY WHAT IS ASKED:**"


def with_rules(system_prompt):
    """The system prompt with the three rules appended. Idempotent."""
    if not system_prompt or RULE_MARKER in system_prompt:
        return system_prompt
    return system_prompt.rstrip() + "\n" + GENERATION_RULES


def corrected_fixer_prompt(system_prompt):
    """The fixer system prompt with its blanket NULL rule replaced. Idempotent."""
    if not system_prompt or FIXER_NULL_RULE_NEW in system_prompt:
        return system_prompt
    if FIXER_NULL_RULE_OLD in system_prompt:
        return system_prompt.replace(FIXER_NULL_RULE_OLD, FIXER_NULL_RULE_NEW)
    return system_prompt
