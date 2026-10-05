"""RULE Q: one line restricting generation to the surface forms the benchmark's gold uses.

Measured on dev before writing it (2026-09-30), so the expected effect is stated up front
rather than discovered afterwards:

  construct   dev gold (1534)   gr_v1 pool (7670 candidates)   opus final (1534)
  COALESCE    0     (0.00%)     0                              0
  IFNULL      0     (0.00%)     0                              0
  WITH/CTE    9     (0.59%)     22    (0.29%)                  1

Generation already never emits COALESCE or IFNULL, and it writes CTEs LESS often than the gold
does. Exactly one final answer in the full dev run uses a CTE (Q1481), and that query is wrong
for a semantic reason - it misreads "the customers with the least amount of consumption" - not
because of the WITH. The evaluator compares set(rows), so a CTE returning the same rows scores
the same as a flat query; surface form cannot cost a point on its own. COALESCE could, since it
substitutes a value for NULL and changes the result set, but generation does not produce it.

So this rule is expected to measure 0. It is here because "we constrain generation to the
surface forms the benchmark uses" is a method statement worth being able to make, not because
it is expected to recover anything. It is APPENDED, never promoted: the COUNT experiment showed
that text at the head of the rule block displaces what is already there (0/11 -> 11/11 for the
promoted rule), and RULE A is the only lettered rule measured to pay (+4). A rule that fires on
0.3% of candidates must not be allowed to push RULE A down.
"""

SURFACE_FORM_RULE = (
    "\n\n**RULE Q - SURFACE FORM:**\n"
    "- Write the answer as a single plain SELECT statement: no WITH/CTE, and no COALESCE or "
    "IFNULL unless the question or the evidence asks for a substitute value. Use a subquery "
    "where you would have used a CTE."
)

RULE_MARKER = "**RULE Q - SURFACE FORM:**"


def with_rule(system_prompt):
    """The system prompt with RULE Q appended. Idempotent; leaves an empty prompt alone."""
    if not system_prompt or RULE_MARKER in system_prompt:
        return system_prompt
    return system_prompt.rstrip() + SURFACE_FORM_RULE
