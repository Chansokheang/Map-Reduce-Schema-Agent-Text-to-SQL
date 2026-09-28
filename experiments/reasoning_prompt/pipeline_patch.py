"""Reasoning-prompt arm: decompose-then-map generation, with extraction that survives it.

Loaded by experiments/full_pipeline/patches/sitecustomize.py when QASQL_REASONING_PROMPT=1.
src/ is never edited; both seams are wrapped at runtime.

  1. `PromptBuilder.build` — the system prompt of all five strategies gets the reasoning section,
     and the two instructions that forbid explanations are rewritten (see prompt.py).
  2. `CandidateGenerator._extract_sql` — the shipped extractor returns the FIRST fenced block
     (`re.search`, non-greedy). A reasoning answer can contain an intermediate query, so the first
     block may be a subquery rather than the answer. The wrapper prefers the LAST fenced block
     that parses as a statement, and falls back to the original function otherwise. Without this,
     a well-formed reasoning answer would silently score as its own subquery.

Why only these two: the conventions in RULE A-K are kept, because every convention change we
measured was worth zero or less, while the bucket this targets — right tables and columns, wrong
logic — is the largest one left (120 of gr_v1's 396 failures).

Environment:
  QASQL_REASONING_PROMPT=1        enable
  QASQL_REASONING_KEEP_EXTRACTOR=1  leave `_extract_sql` alone (diagnostic only; expect the
                                    subquery-instead-of-answer failure mode)
"""
import functools
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.reasoning_prompt.prompt import rewrote_output_instructions, with_reasoning

FENCE = re.compile(r"```(?:sql)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
STATEMENT = re.compile(r"^\s*(SELECT|WITH)\b", re.IGNORECASE)


def final_statement(response):
    """The last fenced block that looks like a query, or None."""
    blocks = [b.strip() for b in FENCE.findall(response or "")]
    for block in reversed(blocks):
        if STATEMENT.match(block):
            return block
    return None


def _patch_prompt_builder():
    from src.generation.prompt_builder import PromptBuilder
    if getattr(PromptBuilder, "_qasql_reasoning_prompt", False):
        return True
    original = PromptBuilder.build

    @functools.wraps(original)
    def build(self, nl_query, schema, strategy, focused_schema=None, profile=None, evidence=None):
        prompts = original(self, nl_query, schema, strategy, focused_schema=focused_schema,
                           profile=profile, evidence=evidence)
        if isinstance(prompts, dict) and prompts.get("system"):
            prompts["system"] = with_reasoning(prompts["system"])
            if not rewrote_output_instructions(prompts["system"]):
                print(f"[patches] WARNING: {strategy} still forbids explanations; reasoning output "
                      "may be suppressed", file=sys.stderr, flush=True)
        return prompts

    PromptBuilder.build = build
    PromptBuilder._qasql_reasoning_prompt = True
    return True


def _patch_extractor():
    from src.generation.candidate_generator import CandidateGenerator
    if getattr(CandidateGenerator, "_qasql_reasoning_prompt", False):
        return True
    original = CandidateGenerator._extract_sql

    @functools.wraps(original)
    def _extract_sql(self, response):
        found = final_statement(response)
        return found if found else original(self, response)

    CandidateGenerator._extract_sql = _extract_sql
    CandidateGenerator._qasql_reasoning_prompt = True
    return True


def install():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        ok = _patch_prompt_builder()
    except ImportError:
        return False
    if os.environ.get("QASQL_REASONING_KEEP_EXTRACTOR") != "1":
        try:
            _patch_extractor()
        except ImportError:
            print("[patches] WARNING: could not patch _extract_sql; the first fenced block wins",
                  file=sys.stderr, flush=True)
    return ok


ENABLED = os.environ.get("QASQL_REASONING_PROMPT") == "1"
