"""Add RULE O (COUNT/DISTINCT decided by the evidence) to every generation prompt.

Loaded by experiments/full_pipeline/patches/sitecustomize.py when QASQL_COUNT_CONVENTION=1.
src/ is never edited.

One seam only: `PromptBuilder.build` appends RULE O to the SYSTEM prompt of all five strategies.
Output format is unchanged, so there is no extraction risk and no interaction with the reasoning or
projection-order arms.

The judge and fixer prompts are left alone. The fixer's own COUNT(DISTINCT) strip rule never fires
in practice (checked on 10 thrombosis regressions: all returned is_acceptable=true), and enforcing
it measured -5 on v6, so there is nothing to gain by touching it.

Environment:
  QASQL_COUNT_CONVENTION=1   enable
"""
import functools
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.count_convention.prompt import MARKER, with_count_rule


def _patch_prompt_builder():
    from src.generation.prompt_builder import PromptBuilder
    if getattr(PromptBuilder, "_qasql_count_convention", False):
        return True
    original = PromptBuilder.build

    @functools.wraps(original)
    def build(self, nl_query, schema, strategy, focused_schema=None, profile=None, evidence=None):
        prompts = original(self, nl_query, schema, strategy, focused_schema=focused_schema,
                           profile=profile, evidence=evidence)
        if isinstance(prompts, dict) and prompts.get("system"):
            prompts["system"] = with_count_rule(prompts["system"])
        return prompts

    PromptBuilder.build = build
    PromptBuilder._qasql_count_convention = True
    return True


def install():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        return _patch_prompt_builder()
    except ImportError:
        return False


ENABLED = os.environ.get("QASQL_COUNT_CONVENTION") == "1"
