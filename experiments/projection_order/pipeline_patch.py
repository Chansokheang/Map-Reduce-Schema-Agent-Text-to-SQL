"""Projection-order arm: decompose the question's requested outputs, then match the SELECT list.

Loaded by experiments/full_pipeline/patches/sitecustomize.py when QASQL_PROJECTION_ORDER=1.
src/ is never edited.

Two seams, the second one mandatory rather than optional:
  * `PromptBuilder.build` — the system prompt of all five strategies gains the `outputs:` step and
    loses the two instructions that forbid explanations (see prompt.py).
  * `CandidateGenerator._extract_sql` — reused from experiments/reasoning_prompt, because the
    shipped extractor returns the FIRST fenced block and any response with a preliminary query in
    it would otherwise be scored as that preliminary query.

Deliberately independent of --generation-rules: RULE L/M state the same goal as instructions, and
running both would make an effect unattributable. Composes with the retrieval flags, which touch
the user prompt.

Environment:
  QASQL_PROJECTION_ORDER=1   enable
"""
import functools
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.projection_order.prompt import rewrote_output_instructions, with_projection_order
from experiments.reasoning_prompt.pipeline_patch import _patch_extractor


def _patch_prompt_builder():
    from src.generation.prompt_builder import PromptBuilder
    if getattr(PromptBuilder, "_qasql_projection_order", False):
        return True
    if getattr(PromptBuilder, "_qasql_reasoning_prompt", False):
        print("[patches] WARNING: the reasoning arm already rewrote the output format; "
              "projection-order not installed", file=sys.stderr, flush=True)
        return False
    original = PromptBuilder.build

    @functools.wraps(original)
    def build(self, nl_query, schema, strategy, focused_schema=None, profile=None, evidence=None):
        prompts = original(self, nl_query, schema, strategy, focused_schema=focused_schema,
                           profile=profile, evidence=evidence)
        if isinstance(prompts, dict) and prompts.get("system"):
            prompts["system"] = with_projection_order(prompts["system"])
            if not rewrote_output_instructions(prompts["system"]):
                print(f"[patches] WARNING: {strategy} still forbids explanations; the outputs line "
                      "may be suppressed", file=sys.stderr, flush=True)
        return prompts

    PromptBuilder.build = build
    PromptBuilder._qasql_projection_order = True
    return True


def install():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        ok = _patch_prompt_builder()
    except ImportError:
        return False
    if ok:
        try:
            _patch_extractor()                      # last fenced query wins; see reasoning_prompt
        except ImportError:
            print("[patches] WARNING: could not patch _extract_sql; the first fenced block wins",
                  file=sys.stderr, flush=True)
    return ok


ENABLED = os.environ.get("QASQL_PROJECTION_ORDER") == "1"
