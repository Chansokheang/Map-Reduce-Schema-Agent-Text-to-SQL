"""Run the pipeline with stage 6 (the fixer) switched off, without editing src/.

Loaded by experiments/full_pipeline/patches/sitecustomize.py when QASQL_DISABLE_FIXER=1.
`SQLFixer.fix` is replaced by a no-op that returns the judge's SQL unchanged, so the pipeline
saves exactly what the judge picked.

Why patch the class and not `config.fixer_enabled`: the CLI runs `python -m src.pipeline`,
which executes that module a second time as `__main__`, so a patch applied to the
`src.pipeline` copy imported at start-up does not affect the class the CLI actually uses.
`src.selection.fixer` is imported normally by both copies, so patching it works for both.

This removes existing hand-written repair behaviour; it adds no rules of its own.
"""
import functools
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def install():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        from src.selection.fixer import SQLFixer, FixerOutcome
    except ImportError:
        return False
    if getattr(SQLFixer, "_qasql_fixer_disabled", False):
        return True
    original = SQLFixer.fix

    @functools.wraps(original)
    def fix(self, candidate, execution_result, nl_query, evidence, db_path, schema_str=""):
        return FixerOutcome(
            candidate_id=getattr(candidate, "candidate_id", 0),
            is_acceptable=True,
            issues=["fixer disabled for ablation"],
            final_sql=getattr(execution_result, "sql", None) or getattr(candidate, "sql", ""),
            final_rows=getattr(execution_result, "result", None),
            final_row_count=getattr(execution_result, "row_count", 0),
            iterations=0,
            refined=False,
        )

    SQLFixer.fix = fix
    SQLFixer._qasql_fixer_disabled = True
    return True


ENABLED = os.environ.get("QASQL_DISABLE_FIXER") == "1"
