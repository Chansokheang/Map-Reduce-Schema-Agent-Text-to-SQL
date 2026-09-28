"""Runtime patches for a pipeline run, selected by environment flags. src/ is never edited.

run_full_pipeline.sh puts this folder first on PYTHONPATH, so Python imports it at start-up
and the unchanged pipeline picks up whichever patches are enabled:

  QASQL_NATIVE_HEADLESS=1    native Claude CLI calls (fixes the Windows WSL quoting error and
                             WinError 206); implementation in ../headless_patch/sitecustomize.py
  QASQL_MATCHED_CONTENTS=1   append a '# Matched contents' block of database values to every
                             generation prompt; implementation in
                             ../../matched_contents/pipeline_patch.py
  QASQL_COUNT_STRICT=1       thrombosis_prediction ONLY: the argument of COUNT() is decided by the
                             evidence and nothing else (MUST NOT de-duplicate without an
                             imperative), and RULE E's join-multiplication trigger is removed so
                             nothing contradicts it. DEV-ONLY by construction - thrombosis is not in
                             the test set. Implementation in ../../count_convention/strict_patch.py.
  QASQL_COUNT_CONVENTION=1   add RULE O: COUNT(DISTINCT) when the evidence tells you to
                             de-duplicate, plain COUNT otherwise. Signal: 18 of 18 dev gold queries
                             with a dedup imperative use DISTINCT. Implementation in
                             ../../count_convention/pipeline_patch.py.
  QASQL_PROJECTION_ORDER=1   generation first writes one `outputs:` line naming what the question
                             asks to see, in order, then a query whose SELECT list matches it.
                             Narrower than the reasoning arm and independent of RULE L/M/N;
                             implementation in ../../projection_order/pipeline_patch.py.
  QASQL_REASONING_PROMPT=1   generation answers with a short decompose/map/shape analysis and the
                             final query in a fenced block; also makes SQL extraction prefer the
                             LAST fenced query, because the shipped extractor takes the first and
                             would return an intermediate subquery. Implementation in
                             ../../reasoning_prompt/pipeline_patch.py.
  QASQL_GENERATION_RULES=1   append RULE L (project only what is asked), RULE M (column order)
                             and RULE N (no unstated conditions) to every generation SYSTEM
                             prompt, and correct the fixer's blanket NULL rule so it does not
                             reverse them; implementation in
                             ../../generation_rules/pipeline_patch.py. Composes with the
                             retrieval flags (it touches the system prompt, they touch the user
                             prompt).
  QASQL_COLUMN_GUIDANCE=1    everything QASQL_COLUMN_MEANING does, plus a column-selection
                             instruction appended to every schema worker prompt;
                             implementation in ../../column_meaning/worker_guidance_patch.py
  QASQL_COLUMN_MEANING=1     everything QASQL_SCHEMA_LINKING does, plus a '# Column meanings'
                             block ranking the documented columns against the question;
                             implementation in ../../column_meaning/pipeline_patch.py.
                             Supersedes the two flags below; do not combine them.
  QASQL_SCHEMA_LINKING=1     retrieve those values BEFORE the schema agent (so table scoring
                             sees them) and add the database's declared join columns; the
                             generation prompt gets both blocks. Implementation in
                             ../../matched_contents/schema_agent_patch.py. Supersedes
                             QASQL_MATCHED_CONTENTS; do not set both.

Only one module named sitecustomize can be imported, so this file chains both.
"""
import importlib.util
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _apply():
    if os.environ.get("QASQL_COUNT_STRICT") == "1":
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        module = _load(ROOT / "experiments" / "count_convention" / "strict_patch.py", "qasql_count_strict")
        if module.install():
            judge = "judge RULE E corrected" if getattr(module, "JUDGE_PATCHED", False)                 else "JUDGE NOT PATCHED"
            fixer = "fixer COUNT rule corrected" if getattr(module, "FIXER_PATCHED", False)                 else "FIXER NOT PATCHED"
            print(f"[patches] strict COUNT rule for thrombosis_prediction; RULE E + checklist "
                  f"rescoped; {judge}; {fixer}", file=sys.stderr, flush=True)
    if os.environ.get("QASQL_COUNT_CONVENTION") == "1":
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        module = _load(ROOT / "experiments" / "count_convention" / "pipeline_patch.py", "qasql_count_convention")
        if module.install():
            print("[patches] RULE O active: COUNT(DISTINCT) decided by the evidence",
                  file=sys.stderr, flush=True)
    if os.environ.get("QASQL_PROJECTION_ORDER") == "1":
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        module = _load(ROOT / "experiments" / "projection_order" / "pipeline_patch.py", "qasql_projection_order")
        if module.install():
            print("[patches] projection-order decomposition active; extraction takes the last fenced query",
                  file=sys.stderr, flush=True)
    if os.environ.get("QASQL_REASONING_PROMPT") == "1":
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        module = _load(ROOT / "experiments" / "reasoning_prompt" / "pipeline_patch.py", "qasql_reasoning_prompt")
        if module.install():
            print("[patches] reasoning prompt active; SQL extraction takes the last fenced query",
                  file=sys.stderr, flush=True)
    if os.environ.get("QASQL_GENERATION_RULES") == "1":
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        module = _load(ROOT / "experiments" / "generation_rules" / "pipeline_patch.py", "qasql_generation_rules")
        if module.install():
            print("[patches] generation rules L/M/N added; fixer NULL rule corrected",
                  file=sys.stderr, flush=True)
    if os.environ.get("QASQL_NATIVE_HEADLESS") == "1":
        # Importing it installs the patch and prints its own line.
        _load(HERE.parent / "headless_patch" / "sitecustomize.py", "qasql_headless_patch")
    if os.environ.get("QASQL_DISABLE_FIXER") == "1":
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        module = _load(ROOT / "experiments" / "fixer_ablation" / "pipeline_patch.py", "qasql_fixer_ablation")
        if module.install():
            print("[patches] fixer stage disabled (ablation)", file=sys.stderr, flush=True)
    if os.environ.get("QASQL_COLUMN_GUIDANCE") == "1":
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        module = _load(ROOT / "experiments" / "column_meaning" / "worker_guidance_patch.py", "qasql_column_guidance")
        if module.install():
            print("[patches] retrieval blocks + column-selection instruction for the map agent",
                  file=sys.stderr, flush=True)
    if os.environ.get("QASQL_COLUMN_MEANING") == "1":
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        module = _load(ROOT / "experiments" / "column_meaning" / "pipeline_patch.py", "qasql_column_meaning")
        if module.install():
            print("[patches] matched contents + join columns + column meanings before the schema agent",
                  file=sys.stderr, flush=True)
    if os.environ.get("QASQL_SCHEMA_LINKING") == "1":
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        module = _load(ROOT / "experiments" / "matched_contents" / "schema_agent_patch.py", "qasql_schema_linking")
        if module.install():
            print("[patches] matched contents + join columns before the schema agent", file=sys.stderr, flush=True)
    if os.environ.get("QASQL_MATCHED_CONTENTS") == "1":
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        module = _load(ROOT / "experiments" / "matched_contents" / "pipeline_patch.py", "qasql_matched_contents")
        if module.install():
            print("[patches] matched contents appended to generation prompts", file=sys.stderr, flush=True)


_apply()
