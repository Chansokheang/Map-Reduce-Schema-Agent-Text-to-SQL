"""Move named rules to the head of the rule block, where they are actually read.

Position, not wording, decided compliance in the COUNT experiment: the same MUST-worded rule was
obeyed by the judge 0/11 times at the bottom of the prompt and 11/11 at the top, and the full arm
went 14/29 -> 22/29. This patch generalises that move to any lettered rule, so the effect can be
tested on rules that are not convention conflicts.

Salience is zero-sum. Promoting a rule pushes whatever was first down, so promote one or two, not
five - a "top block" of five rules is just the old list with new numbering.

Environment:
  QASQL_PROMOTE_RULES=K       comma-separated rule letters, e.g. "K" or "K,J"

src/ is never edited; the prompt strings are rewritten in memory.
"""
import functools
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ANCHOR = "**RULE A"
JUDGE_ANCHOR = "EVALUATION CRITERIA"


def rule_block(system_prompt, letter):
    """The '**RULE X — ...:**' heading and its bullets, up to the next rule or blank-line break."""
    m = re.search(rf"\*\*RULE {letter} [^\n]*\n(?:-[^\n]*\n?)*", system_prompt)
    return m.group(0).rstrip() if m else None


def promote(system_prompt, letters):
    """Move each named rule directly above RULE A. Idempotent; unknown letters are ignored."""
    if not system_prompt or ANCHOR not in system_prompt:
        return system_prompt
    moved = []
    out = system_prompt
    for letter in letters:
        block = rule_block(out, letter)
        if not block:
            continue
        head = out.index(ANCHOR)
        if out.index(block) < head:            # already above RULE A
            continue
        out = out.replace(block + "\n", "", 1).replace(block, "", 1)
        moved.append(block)
    if not moved:
        return out
    joined = "\n\n".join(moved)
    return out.replace(ANCHOR, joined + "\n\n" + ANCHOR, 1)


def judge_promote(system_prompt, letters):
    """The judge lists rules as '- RULE X — ...' bullets; move them above the criteria heading."""
    if not system_prompt or JUDGE_ANCHOR not in system_prompt:
        return system_prompt
    moved, out = [], system_prompt
    for letter in letters:
        m = re.search(rf"- RULE {letter} [^\n]*", out)
        if not m:
            continue
        line = m.group(0)
        if out.index(line) < out.index(JUDGE_ANCHOR):
            continue
        out = out.replace(line + "\n", "", 1).replace(line, "", 1)
        moved.append(line)
    if not moved:
        return out
    block = ("IMPORTANT — apply these before any other criterion:\n" + "\n".join(moved) + "\n\n")
    return out.replace(JUDGE_ANCHOR, block + JUDGE_ANCHOR, 1)


def letters():
    raw = os.environ.get("QASQL_PROMOTE_RULES", "")
    return [x.strip().upper() for x in raw.split(",") if x.strip()]


def _patch_prompt_builder(rules):
    from src.generation.prompt_builder import PromptBuilder
    if getattr(PromptBuilder, "_qasql_promote", False):
        return True
    original = PromptBuilder.build

    @functools.wraps(original)
    def build(self, nl_query, schema, strategy, focused_schema=None, profile=None, evidence=None):
        prompts = original(self, nl_query, schema, strategy, focused_schema=focused_schema,
                           profile=profile, evidence=evidence)
        if isinstance(prompts, dict) and prompts.get("system"):
            prompts["system"] = promote(prompts["system"], rules)
        return prompts

    PromptBuilder.build = build
    PromptBuilder._qasql_promote = True
    return True


def install():
    rules = letters()
    if not rules:
        return False
    try:
        ok = _patch_prompt_builder(rules)
    except ImportError:
        return False
    try:
        from src.prompt import JUDGE_PROMPT
        JUDGE_PROMPT["system"] = judge_promote(JUDGE_PROMPT["system"], rules)
    except ImportError:
        pass
    return ok


ENABLED = bool(os.environ.get("QASQL_PROMOTE_RULES"))
