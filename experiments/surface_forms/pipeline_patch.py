"""Append RULE Q to every generation SYSTEM prompt. src/ is never edited.

Loaded by experiments/full_pipeline/patches/sitecustomize.py when QASQL_SURFACE_FORMS=1.

One seam: `PromptBuilder.build`. All five strategies get it, because the candidates exist to
give the judge different schema views, not different output conventions.

It appends to prompts["system"], like generation_rules, while the retrieval patches append to
prompts["user"], so it composes with --matched-contents / --schema-linking / --column-meaning /
--column-guidance and stacks after RULE L/M/N rather than replacing them.

The fixer is deliberately left alone. It runs after the judge and rewrites ~12% of questions,
but nothing in its prompt introduces a CTE or a COALESCE, so there is no contradiction to
correct - and every fixer edit measured so far has been worth keeping (fixer off 133 vs on 138).
"""
import functools
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.surface_forms.rule import with_rule


def _patch_prompt_builder():
    from src.generation.prompt_builder import PromptBuilder
    if getattr(PromptBuilder, "_qasql_surface_forms", False):
        return True
    original = PromptBuilder.build

    @functools.wraps(original)
    def build(self, nl_query, schema, strategy, focused_schema=None, profile=None, evidence=None):
        prompts = original(self, nl_query, schema, strategy, focused_schema=focused_schema,
                           profile=profile, evidence=evidence)
        if isinstance(prompts, dict) and prompts.get("system"):
            prompts["system"] = with_rule(prompts["system"])
        return prompts

    PromptBuilder.build = build
    PromptBuilder._qasql_surface_forms = True
    return True


def install():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        return _patch_prompt_builder()
    except ImportError:
        return False


ENABLED = os.environ.get("QASQL_SURFACE_FORMS") == "1"
