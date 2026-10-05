"""Make the judge reason BEFORE it names a winner.

The shipped schema asks for `selected_id` first and `reasoning` last, so the choice is emitted
before any analysis exists - what follows is justification, not deliberation. Observed directly: on
Q1203 the judge produced a fluent, confident argument for a candidate it had already picked.

This patch reorders the response schema so the analysis fields come first, and raises the 512-token
cap, which `selected_sql` alone can nearly exhaust.

Nothing else changes: the parser reads keys by name (src/selection/judge.py:402), so key order is
free, and the candidate set, prompt rules and temperature are untouched.

Environment:
  QASQL_JUDGE_REASON_FIRST=1   enable
  QASQL_JUDGE_MAX_TOKENS=1500  optional cap override
"""
import functools
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OLD_BLOCK = """Return ONLY JSON:
{{
  "selected_id": 1,
  "selected_sql": "the SQL of the best candidate",
  "confidence": 0.0 to 1.0,
  "reasoning": "brief explanation of why this candidate was selected and why others were not"
}}"""

NEW_BLOCK = """Work through the comparison BEFORE naming a winner. Write the first two fields in full
first; do not decide which candidate wins until you have written them.

Return ONLY JSON, with the keys in exactly this order:
{{
  "differences": "what each candidate actually returns, and precisely where they disagree with each other",
  "reasoning": "which candidate answers the question exactly as asked, and why each of the others does not",
  "selected_id": 1,
  "selected_sql": "the SQL of the candidate you chose",
  "confidence": 0.0 to 1.0
}}"""

MARKER = '"differences"'


def max_tokens():
    try:
        return int(os.environ.get("QASQL_JUDGE_MAX_TOKENS", "1500"))
    except ValueError:
        return 1500


def rewrite(user_template):
    """Reorder the response schema. Idempotent; None when the shipped wording has moved on."""
    if not user_template:
        return None
    if MARKER in user_template:
        return user_template
    if OLD_BLOCK not in user_template:
        return None
    return user_template.replace(OLD_BLOCK, NEW_BLOCK)


def _patch_budget():
    """Raise the 512-token cap for the judge call only."""
    from src.selection.judge import SQLJudge
    if getattr(SQLJudge, "_qasql_reason_first", False):
        return True
    original = SQLJudge._llm_judge
    cap = max_tokens()

    @functools.wraps(original)
    def _llm_judge(self, *args, **kwargs):
        client = self.llm_client
        if client is None:
            return original(self, *args, **kwargs)
        original_complete = client.complete

        @functools.wraps(original_complete)
        def complete(prompt, system_prompt=None, max_tokens=512, temperature=0.0, **kw):
            return original_complete(prompt, system_prompt=system_prompt,
                                     max_tokens=max(max_tokens, cap),
                                     temperature=temperature, **kw)

        client.complete = complete
        try:
            return original(self, *args, **kwargs)
        finally:
            client.complete = original_complete

    SQLJudge._llm_judge = _llm_judge
    SQLJudge._qasql_reason_first = True
    return True


def install():
    try:
        from src.prompt import JUDGE_PROMPT
    except ImportError:
        return False
    rewritten = rewrite(JUDGE_PROMPT.get("user_template"))
    if rewritten is None:
        print("[patches] WARNING: judge response schema not found; reason-first NOT applied",
              file=sys.stderr, flush=True)
        return False
    JUDGE_PROMPT["user_template"] = rewritten
    try:
        _patch_budget()
    except ImportError:
        pass
    return True


ENABLED = os.environ.get("QASQL_JUDGE_REASON_FIRST") == "1"
