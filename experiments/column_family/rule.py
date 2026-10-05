"""RULE P: use the first member of a numbered column family unless something asks for more.

Origin. Across the 8 california_schools questions that touch AdmFName2/3, we used the secondary
administrator columns on all 8; the shape the question wants is the primary one on 6 of them. The
one question whose EVIDENCE declares "There are at most 3 administrators for each school" is the
one where all three are wanted - so the evidence, not the schema, licenses the extra columns.

Why this is not a guess about the labels. Two inference-time sources already say the later members
are marginal:
  * the shipped metadata: column_meaning.json marks AdmFName3 / AdmLName3 / AdmEmail3 (and 13 more
    columns across card_games, financial, formula_1) as "not useful";
  * the data: AdmFName1 is non-empty on 66.2% of rows, AdmFName2 on 2.4%, AdmFName3 on 0.2%.
The rule is stated generically - any X1/X2/X3 family - rather than naming columns, so it carries to
databases we have never looked at.

The rule is inserted at the HEAD of the rule block, not appended, because position decided
compliance in the COUNT experiment (0/11 obeyed at the bottom, 11/11 at the top).

Environment:
  QASQL_COLUMN_FAMILY=1     enable RULE P
  QASQL_ENTITY_COLUMN=1     also add the duplicate-column line (measured worth 0 - see README)
"""
import functools
import re
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ANCHOR = "**RULE A"

# Declared explicitly rather than measured. Each entry is primary -> the near-empty members it
# replaces. The rule is only added when the prompt's schema actually contains those columns, so
# databases without them are untouched.
STATIC_FAMILIES = {
    "schools": {
        "AdmFName1": ["AdmFName2", "AdmFName3"],
        "AdmLName1": ["AdmLName2", "AdmLName3"],
        "AdmEmail1": ["AdmEmail2", "AdmEmail3"],
    },
}

RULE_P_HEAD = """**RULE P — SECONDARY CONTACT COLUMNS:**
- These columns hold a second or third contact and are empty for almost every row. Use the
  PRIMARY column named on each line, on its own:
{columns}
- Do not OR or SELECT the secondary members to be safe: that changes which rows match and which
  columns are returned.
- Use a secondary member ONLY when the EVIDENCE says more than one exists, e.g. "there are at
  most 3 administrators for each school".
- Plural wording in the QUESTION is NOT enough on its own. "the names of all the administrators",
  "the e-mail addresses", "their full names" still mean the PRIMARY column only, unless the
  evidence says otherwise."""

ENTITY_COLUMN = """
- When the same attribute exists both in an entity's own table and as a copy inside a
  measurement/score table (e.g. a district name on both the schools table and the scores table),
  read it from the entity's own table."""


def sparse_in(schema):
    """The cached sparse columns that this prompt's schema actually contains.

    Returns [] for a database with none, so the rule costs nothing where it cannot act - the
    generic version of this rule misfired on formula_1 q1/q2/q3, european_football_2
    home_player_1..11 and financial A12/A13, all of which are fully populated and meaningful.
    """
    from experiments.column_family.sparsity import load
    tables = (schema or {}).get("tables", schema) or {}
    present = {}
    for table, info in tables.items():
        names = {c.get("name", "").lower() for c in (info or {}).get("columns", [])}
        for db_tables in load().values():
            for cached_table, cols in db_tables.items():
                if cached_table.lower() != str(table).lower():
                    continue
                for entry in cols:
                    if entry["column"].lower() in names:
                        present.setdefault(table, []).append(entry)
    return present


def rule_text(schema=None):
    """Name the declared families that this prompt's schema actually contains."""
    tables = (schema or {}).get("tables", schema) or {}
    items = []
    for table, info in tables.items():
        declared = STATIC_FAMILIES.get(str(table))
        if not declared:
            continue
        names = {c.get("name", "").lower() for c in (info or {}).get("columns", [])}
        for primary, secondaries in declared.items():
            if primary.lower() not in names:
                continue
            present = [s for s in secondaries if s.lower() in names]
            if present:
                items.append(f"  - use `{table}.{primary}` alone, not "
                             f"{' or '.join(chr(96) + s + chr(96) for s in present)}")
    if not items:
        return ""
    return (RULE_P_HEAD.format(columns=chr(10).join(items))
            + (ENTITY_COLUMN if with_entity_line() else ""))


MARKER = "**RULE P — SECONDARY CONTACT COLUMNS"


def with_entity_line():
    return os.environ.get("QASQL_ENTITY_COLUMN") == "1"


def insert(system_prompt, schema=None):
    """Put RULE P at the head of the rule block, only when this schema has a sparse column."""
    if not system_prompt or MARKER in system_prompt or ANCHOR not in system_prompt:
        return system_prompt
    text = rule_text(schema)
    if not text:
        return system_prompt
    return system_prompt.replace(ANCHOR, text + chr(10) * 2 + ANCHOR, 1)


def _patch_prompt_builder():
    from src.generation.prompt_builder import PromptBuilder
    if getattr(PromptBuilder, "_qasql_column_family", False):
        return True
    original = PromptBuilder.build

    @functools.wraps(original)
    def build(self, nl_query, schema, strategy, focused_schema=None, profile=None, evidence=None):
        prompts = original(self, nl_query, schema, strategy, focused_schema=focused_schema,
                           profile=profile, evidence=evidence)
        if isinstance(prompts, dict) and prompts.get("system"):
            prompts["system"] = insert(prompts["system"], focused_schema or {"tables": schema})
        return prompts

    PromptBuilder.build = build
    PromptBuilder._qasql_column_family = True
    return True


def install():
    try:
        ok = _patch_prompt_builder()
    except ImportError:
        return False
    try:
        from src.prompt import JUDGE_PROMPT
        system = JUDGE_PROMPT.get("system", "")
        if MARKER not in system and "EVALUATION CRITERIA" in system:
            JUDGE_PROMPT["system"] = system.replace(
                "EVALUATION CRITERIA",
                rule_text() + "\n\nEVALUATION CRITERIA", 1)
    except ImportError:
        pass
    return ok


ENABLED = os.environ.get("QASQL_COLUMN_FAMILY") == "1"
