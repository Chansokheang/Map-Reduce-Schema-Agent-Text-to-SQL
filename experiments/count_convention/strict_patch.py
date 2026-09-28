"""Strict COUNT rule for thrombosis_prediction only, with RULE E's conflicting clause removed.

Loaded by experiments/full_pipeline/patches/sitecustomize.py when QASQL_COUNT_STRICT=1.
src/ is never edited; the prompt is rewritten in memory, and only for this one database.

Why this exists. RULE O (prompt.py) stated the second half as a preference:

    "If the evidence says nothing about duplicates, PREFER plain COUNT(<column>)"

Measured on the 163 thrombosis questions: the MUST half was obeyed 15/15, the preference half was
ignored on 29 questions (35 uses of COUNT(DISTINCT) where gold wants 6). The reason is visible in
the prompt: RULE E says "Use DISTINCT when JOIN multiplies rows per entity", Patient->Laboratory
multiplies rows 46x, and a "prefer" cannot outrank a "use" that names the exact situation. So this
variant does two things instead of one:

  1. states the no-imperative case as MUST NOT, not as a preference;
  2. re-scopes RULE E to `SELECT DISTINCT` in the projection, removing COUNT() from it entirely, so
     the two rules cover disjoint ground instead of contradicting each other;
  3. applies the same correction to the JUDGE's RULE E;
  4. rescopes the numbered checklist clause, which described RULE E as "DISTINCT via schema
     reasoning" - the exact reasoning RULE O forbids - in all six prompt modules;
  5. rescopes the FIXER's COUNT rule, whose "KEEP DISTINCT otherwise" branch fires on precisely
     this shape (Patient -> Laboratory fans out through Date, a different key). The fixer runs once
     AFTER the judge (src/pipeline.py:616), so it has the last word.

The judge needed two passes. Told merely to PREFER plain COUNT, it selected COUNT(DISTINCT) anyway
and said why: "the question asks how many patients - a count of distinct individuals... plain COUNT
(641) overcounts by counting repeated lab rows... 88 correctly reflects distinct patients (per RULE
C)". That is correct SQL practice; BIRD's gold counts the joined rows. So RULE E now says MUST
rather than PREFER, names that argument verbatim as the one not to make, and states that RULE C
chooses which column goes inside COUNT() and never whether DISTINCT wraps it. On the Q1203 probe
the judge then selected plain and cited the rule, and the answer matched gold.

Points 3-5 were added after measuring points 1 and 2 on the 29 unflagged questions: generation obeyed
the rule (minimal_profile went from 29/29 de-duplicating to 11/29, full_profile 26 -> 12), a plain
COUNT candidate existed on 27 of 29 questions, and the judge still selected a COUNT(DISTINCT) one on
26 of them. src/prompt/judge.py:55 tells it to "PREFER candidates that use DISTINCT when JOIN
multiplies rows per entity", which is exactly the preference generation was told to drop. Fixing
only the generation prompt cannot work while the selector reverses it.

Scope. Generation is scoped by the schema's table set and the fixer by the database path (its
focused schema can be Patient+Laboratory alone, so a table-set test misses it). The judge's prompt
is global and cannot be scoped at its seam. That is deliberate and it is a DEV-ONLY measure: thrombosis is not in
the BIRD test set, so this cannot help a test submission. It isolates whether the instruction can
win once the contradiction is removed - the question RULE O could not answer.

Environment:
  QASQL_COUNT_STRICT=1   enable
"""
import functools
import os
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# thrombosis_prediction is the only dev database with exactly these three tables
THROMBOSIS_TABLES = {"patient", "laboratory", "examination"}

RULE_E_OLD = ("- Use DISTINCT when JOIN multiplies rows per entity, OR when counting/listing a "
              "non-unique category column (e.g., `COUNT(DISTINCT element)` for \"how many "
              "elements\"). Skip for 1-to-1 joins, PK queries, or \"most common\" patterns (those "
              "use GROUP BY + LIMIT 1).")
RULE_E_NEW = ("- This rule is about `SELECT DISTINCT` in the PROJECTION only. It does not apply to "
              "COUNT() at all — the argument of COUNT() is decided by RULE 0 above. Use `SELECT DISTINCT` "
              "when a JOIN would repeat the projected rows and the question asks for a list of "
              "distinct entities. Skip for 1-to-1 joins, PK queries, or \"most common\" patterns "
              "(those use GROUP BY + LIMIT 1).")

CHECKLIST_OLD = "Rule E (DISTINCT via schema reasoning)"
CHECKLIST_NEW = ("Rule E (`SELECT DISTINCT` in the projection only), Rule O (the argument of "
                 "COUNT() — decided by the evidence, never by schema fan-out; stated at the end "
                 "of these rules)")

# src/prompt/fixer.py:24. Its "KEEP DISTINCT otherwise" branch fires on exactly the thrombosis
# shape (Patient -> Laboratory fans out through Date, a different key), so the fixer would put
# back what generation was told to drop.
FIXER_COUNT_OLD = ("CRITICAL — COUNT(DISTINCT id) (overrides \"be conservative\" / \"don't refine "
                   "correct queries\"): STRIP DISTINCT when the counted column is a primary-key id "
                   "AND every JOIN threads through the same key AND there is no `UNION`/unpivot/"
                   "wide-repeated columns AND no fan-out through a different key. KEEP DISTINCT "
                   "otherwise.")
FIXER_COUNT_NEW = ("CRITICAL — COUNT(DISTINCT id) (overrides \"be conservative\" / \"don't refine "
                   "correct queries\"): the argument of COUNT() is decided by the evidence, not by "
                   "fan-out. STRIP DISTINCT from COUNT() unless the evidence asks to de-duplicate "
                   "(\"should consider DISTINCT in the final result\", \"should compute the number "
                   "of distinct/unique ones\", \"only count ones without repetitive\") or the "
                   "question itself asks for distinct/different/unique things. A JOIN that repeats "
                   "rows per entity is NOT a reason to keep DISTINCT, and never ADD DISTINCT to a "
                   "COUNT() that does not have it.")

STRICT_SECTION = """

**IMPORTANT — RULE 0 (READ THIS BEFORE WRITING ANY COUNT). THE ARGUMENT OF COUNT() IS DECIDED BY THE EVIDENCE, NOTHING ELSE. This rule outranks every rule below it:**
- If the evidence tells you to de-duplicate — "should consider DISTINCT in the final result",
  "should compute the number of distinct/unique ones", "should return the number of distinct
  <things>", "only count ones without repetitive", "don't compute repetitive ones" — you MUST write
  `COUNT(DISTINCT <column>)`.
- If the evidence does NOT say that, you MUST write plain `COUNT(<column>)`. Do not write
  `COUNT(DISTINCT ...)`. This holds even when a JOIN repeats rows per entity and even when the
  count therefore looks too large: counting the joined rows is the intended answer here, and
  de-duplicating it is wrong. The only exception is a question that itself asks for distinct
  things in so many words ("how many different X", "how many unique X").
- Scope: this rule governs the argument of COUNT() and nothing else. `SELECT DISTINCT` in the
  projection is RULE E's business and is unaffected by this rule; equally, RULE E says nothing
  about COUNT()."""

MARKER = "**IMPORTANT — RULE 0 (READ THIS BEFORE WRITING ANY COUNT). THE ARGUMENT OF COUNT() IS DECIDED BY THE EVIDENCE, NOTHING ELSE. This rule outranks every rule below it:**"

# set by install(); the banner reports it so a run log proves both halves were active
JUDGE_PATCHED = False
FIXER_PATCHED = False


def all_databases():
    """QASQL_COUNT_ALL=1 lifts the thrombosis-only scope to every database."""
    return os.environ.get("QASQL_COUNT_ALL") == "1"


def is_thrombosis(schema):
    """True when this schema is thrombosis_prediction, by its table set.

    With QASQL_COUNT_ALL=1 this is True for any non-empty schema: the rule is being tested
    everywhere. That is the arm the dev simulation put at -4 on v6, so it is a measurement, not a
    recommendation.
    """
    if all_databases():
        return bool(schema)
    tables = {str(t).casefold() for t in (schema or {})}
    return THROMBOSIS_TABLES <= tables


def rewrite(system_prompt):
    """RULE O appended; RULE E and the numbered checklist rescoped off COUNT(). Idempotent."""
    if not system_prompt or MARKER in system_prompt:
        return system_prompt
    out = system_prompt.replace(RULE_E_OLD, RULE_E_NEW) if RULE_E_OLD in system_prompt else system_prompt
    out = out.replace(CHECKLIST_OLD, CHECKLIST_NEW)
    anchor = "**RULE A"
    if anchor in out:
        # position matters: appended after the numbered checklist the rule was obeyed on 12 of 29;
        # it belongs at the head of the rule block, before RULE A.
        return out.replace(anchor, STRICT_SECTION.strip() + chr(10) * 2 + anchor, 1)
    return out.rstrip() + "\n" + STRICT_SECTION


JUDGE_RULE_E_OLD = ("- **RULE E — DISTINCT:** PREFER candidates that use DISTINCT when JOIN "
                    "multiplies rows per entity OR when counting/listing a non-unique category "
                    "column (\"how many elements\" → `COUNT(DISTINCT element)`). PREFER candidates "
                    "without DISTINCT for 1-to-1 joins, PK queries, or \"most common\" patterns "
                    "(those should use GROUP BY + LIMIT 1). Equivalent IN/EXISTS subqueries are also "
                    "acceptable.")
# The judge read a merely-preferential version of this and overrode it anyway, arguing from the
# semantics of "how many patients" and citing RULE C as cover. So the wording names that argument
# and rules it out, and it says MUST rather than PREFER.
JUDGE_RULE_E_NEW = """- **RULE E — DISTINCT (PROJECTION ONLY):** This applies to `SELECT DISTINCT` in the projection, never to the argument of COUNT(). For COUNT() the evidence decides, and nothing else:
  - If the evidence asks to de-duplicate ("should consider DISTINCT in the final result", "should compute the number of distinct/unique ones", "only count ones without repetitive"), or the question itself says distinct/different/unique, you MUST select a `COUNT(DISTINCT ...)` candidate.
  - Otherwise you MUST select a plain `COUNT(<column>)` candidate whenever one is offered, even if you believe it is wrong. Do NOT reason about whether the number looks too large. This argument — "the question asks how many patients, a patient can have several rows in the joined table, so plain COUNT overcounts and COUNT(DISTINCT id) is the true number of individuals" — is exactly the one you must NOT make: the larger, row-counting number is the intended answer. Sample-row magnitudes are not evidence, and RULE C is not grounds to override this — RULE C chooses WHICH column goes inside COUNT(), never whether DISTINCT wraps it.
  - Equivalent IN/EXISTS subqueries are also acceptable."""


JUDGE_TOP_ANCHOR = "EVALUATION CRITERIA"
# Buried ~3000 chars in as RULE E, the MUST wording was obeyed on 17 of 28. Same text, top of the
# prompt, ahead of the criteria list.
JUDGE_TOP_NOTE = """IMPORTANT — RULE 0, BEFORE ANY OTHER CRITERION. The argument of COUNT() is decided by the evidence and nothing else, and this outranks every criterion below including RULE C and RULE E:
- If the evidence asks to de-duplicate ("should consider DISTINCT in the final result", "should compute the number of distinct/unique ones", "only count ones without repetitive"), or the question itself says distinct/different/unique, select a `COUNT(DISTINCT ...)` candidate.
- Otherwise select a plain `COUNT(<column>)` candidate whenever one is offered. This is not a preference you may weigh against your own reading of the question — it decides the choice on its own.
- Do NOT reason about whether the number looks too large. "The question asks how many patients, a patient has several rows in the joined table, so plain COUNT overcounts and COUNT(DISTINCT id) is the true number of individuals" is exactly the argument you must NOT make. The larger, row-counting number is the intended answer.

"""


def patch_judge_top_note(system):
    """RULE 0 at the head of the judge prompt. Idempotent; None if the anchor is gone."""
    if not system or "RULE 0, BEFORE ANY OTHER CRITERION" in system:
        return system
    if JUDGE_TOP_ANCHOR not in system:
        return None
    return system.replace(JUDGE_TOP_ANCHOR, JUDGE_TOP_NOTE + JUDGE_TOP_ANCHOR, 1)


def patch_judge_prompt():
    """Correct the judge's RULE E in place. Idempotent; returns False if the wording moved on.

    The judge's system prompt is global, so this is not scoped per database - unlike the generation
    patch. That is acceptable only because this whole flag is a thrombosis-only dev experiment; do
    not ship it.
    """
    from src.prompt import JUDGE_PROMPT
    system = JUDGE_PROMPT.get("system", "")
    if "PROJECTION ONLY" not in system:
        if JUDGE_RULE_E_OLD not in system:
            return False
        system = system.replace(JUDGE_RULE_E_OLD, JUDGE_RULE_E_NEW)
    promoted = patch_judge_top_note(system)
    if promoted is None:
        return False
    JUDGE_PROMPT["system"] = promoted
    return True


def fixer_rewrite(system_prompt):
    """The fixer's COUNT rule, re-pointed at the evidence. Idempotent; None if the wording moved."""
    if not system_prompt:
        return None
    if FIXER_COUNT_NEW in system_prompt:
        return system_prompt
    if FIXER_COUNT_OLD not in system_prompt:
        return None
    return system_prompt.replace(FIXER_COUNT_OLD, FIXER_COUNT_NEW)


def _is_thrombosis_db(db_path):
    """Exact scoping for the fixer seam, which is handed the database path.

    The table-set test is wrong here: the fixer gets the FOCUSED schema, which for a
    Patient-and-Laboratory question never mentions Examination.
    """
    if all_databases():
        return bool(db_path)
    return "thrombosis_prediction" in str(db_path or "").casefold()


_FIXER_LOCK = threading.Lock()


def _patch_fixer():
    """Swap the fixer's COUNT rule for the duration of a thrombosis fix only.

    The fixer runs once, after the judge (src/pipeline.py:616), so it has the last word on the SQL
    and can put back the DISTINCT the judge was told to drop. prompt_config is the shared
    FIXER_PROMPT dict, so the swap is held under a lock; the call is sequential, so nothing waits.
    """
    from src.selection.fixer import SQLFixer as Fixer
    if getattr(Fixer, "_qasql_count_strict", False):
        return True
    original = Fixer.fix

    @functools.wraps(original)
    def fix(self, candidate, execution_result, nl_query, evidence, db_path, schema_str=""):
        args = (candidate, execution_result, nl_query, evidence, db_path)
        if not _is_thrombosis_db(db_path):
            return original(self, *args, schema_str=schema_str)
        with _FIXER_LOCK:
            config = self.prompt_config
            rewritten = fixer_rewrite(config.get("system"))
            if rewritten is None:
                return original(self, *args, schema_str=schema_str)
            self.prompt_config = dict(config, system=rewritten)
            try:
                return original(self, *args, schema_str=schema_str)
            finally:
                self.prompt_config = config

    Fixer.fix = fix
    Fixer._qasql_count_strict = True
    return True


def _patch_prompt_builder():
    from src.generation.prompt_builder import PromptBuilder
    if getattr(PromptBuilder, "_qasql_count_strict", False):
        return True
    original = PromptBuilder.build

    @functools.wraps(original)
    def build(self, nl_query, schema, strategy, focused_schema=None, profile=None, evidence=None):
        prompts = original(self, nl_query, schema, strategy, focused_schema=focused_schema,
                           profile=profile, evidence=evidence)
        visible = focused_schema if focused_schema else schema
        if isinstance(prompts, dict) and prompts.get("system") and \
                (is_thrombosis(schema) or is_thrombosis(visible)):
            prompts["system"] = rewrite(prompts["system"])
        return prompts

    PromptBuilder.build = build
    PromptBuilder._qasql_count_strict = True
    return True


def install():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        ok = _patch_prompt_builder()
    except ImportError:
        return False
    global JUDGE_PATCHED, FIXER_PATCHED
    try:
        FIXER_PATCHED = _patch_fixer()
    except ImportError:
        pass

    try:
        JUDGE_PATCHED = patch_judge_prompt()
        if not JUDGE_PATCHED:
            print("[patches] WARNING: judge RULE E wording not found; the judge will still prefer "
                  "COUNT(DISTINCT) and reverse the generation rule", file=sys.stderr, flush=True)
    except ImportError:
        pass
    return ok


ENABLED = os.environ.get("QASQL_COUNT_STRICT") == "1"
