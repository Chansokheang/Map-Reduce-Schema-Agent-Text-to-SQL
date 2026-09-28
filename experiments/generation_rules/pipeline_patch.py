"""Add RULE L/M/N to every generation prompt, and stop the fixer from reversing them.

Loaded by experiments/full_pipeline/patches/sitecustomize.py when QASQL_GENERATION_RULES=1.
src/ is never edited: the strategy prompts and the fixer prompt are corrected in memory.

Two seams:
  * `PromptBuilder.build` — the three rules are appended to the SYSTEM prompt of all five
    strategies. All five, because the candidates exist to give the judge different schema
    views, not different output conventions: the projection has one right shape per question,
    and the judge has no gold with which to prefer one convention over another.
  * `src.prompt.fixer.FIXER_PROMPT` — its blanket "NULL (MANDATORY)" line is replaced by one
    that keeps the cases its own rationale names. Without this the fixer, which runs after the
    judge, re-adds on ~12% of questions exactly what RULE N tells generation not to write.

This patch only appends to `prompts["system"]`, while the retrieval patches append to
`prompts["user"]`, so it composes with --matched-contents / --schema-linking / --column-meaning /
--column-guidance instead of competing with them.

Environment:
  QASQL_GENERATION_RULES=1        enable
  QASQL_GENERATION_RULES_FIXER=0  leave the fixer prompt untouched (rules then get partially
                                  reversed on the questions the fixer rewrites)
"""
import functools
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.generation_rules.rules import corrected_fixer_prompt, with_rules


def _patch_prompt_builder():
    from src.generation.prompt_builder import PromptBuilder
    if getattr(PromptBuilder, "_qasql_generation_rules", False):
        return True
    original = PromptBuilder.build

    @functools.wraps(original)
    def build(self, nl_query, schema, strategy, focused_schema=None, profile=None, evidence=None):
        prompts = original(self, nl_query, schema, strategy, focused_schema=focused_schema,
                           profile=profile, evidence=evidence)
        if isinstance(prompts, dict) and prompts.get("system"):
            prompts["system"] = with_rules(prompts["system"])
        return prompts

    PromptBuilder.build = build
    PromptBuilder._qasql_generation_rules = True
    return True


def _patch_fixer_prompt():
    """Mutates the shared FIXER_PROMPT dict, so instances built earlier see it too."""
    from src.prompt.fixer import FIXER_PROMPT
    if FIXER_PROMPT.get("_qasql_generation_rules"):
        return True
    corrected = corrected_fixer_prompt(FIXER_PROMPT.get("system", ""))
    if corrected == FIXER_PROMPT.get("system"):
        return False                                   # wording moved on; leave it alone
    FIXER_PROMPT["system"] = corrected
    FIXER_PROMPT["_qasql_generation_rules"] = True
    return True


def install():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        ok = _patch_prompt_builder()
    except ImportError:
        return False
    if os.environ.get("QASQL_GENERATION_RULES_FIXER", "1") != "0":
        try:
            if not _patch_fixer_prompt():
                print("[patches] WARNING: fixer NULL rule not found; it may reverse RULE N",
                      file=sys.stderr, flush=True)
        except ImportError:
            pass
    return ok


ENABLED = os.environ.get("QASQL_GENERATION_RULES") == "1"
