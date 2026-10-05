"""Make the judge's selected_id agree with its selected_sql.

THE BUG. src/selection/judge.py validates the two fields independently:

    valid_ids = {c["candidate_id"] for c in successful}
    if selected_id not in valid_ids:          # a VALID id short-circuits this entirely
        ...fall back to matching by SQL...

So when the model returns a valid id but the SQL of a different candidate, both are kept and
they disagree. src/pipeline.py then resolves the conflict in favour of the ID:

    selected_cand = next(c for c in candidates if c.candidate_id == judgment.selected_id)
    outcome = self.sql_fixer.fix(candidate=selected_cand, ...)
    final_sql = outcome.final_sql

- and the SQL the judge actually wrote out is discarded. Observed on Q101: the judge returned
candidate 5's SQL (correct) with selected_id=3, and the pipeline shipped candidate 3 (wrong).

Measured on 47 instrumented questions: 5 mismatches, 1 of which cost a correct answer.

THE FIX. The SQL is the model's actual answer - the id is a label it attached afterwards, and in
a mismatch the label is what is wrong. So when they disagree and the SQL matches some other
successful candidate exactly, the id is corrected to that candidate. If the SQL matches nothing
(the model rewrote it rather than quoting one), the id is trusted and the SQL replaced with that
candidate's, which is the behaviour src/ already relies on elsewhere.

src/ is untouched; SQLJudge.judge is wrapped and its result repaired on the way out. Enable with
QASQL_JUDGE_ID_FIX=1. Set QASQL_JUDGE_ID_FIX_LOG to a path to record every repair.
"""
import functools
import json
import os
import re
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

_LOCK = threading.Lock()
_norm = lambda s: re.sub(r"\s+", " ", str(s or "")).strip().rstrip(";")


def _log(record):
    p = os.environ.get("QASQL_JUDGE_ID_FIX_LOG")
    if not p:
        return
    with _LOCK:
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def reconcile(selected_id, selected_sql, candidates, execution_results):
    """(id, sql, action). Prefers the SQL; falls back to the id when the SQL matches nothing."""
    by_id = {}
    for c in candidates:
        cid = getattr(c, "candidate_id", None)
        if cid is not None:
            by_id[cid] = getattr(c, "sql", "") or ""
    for r in execution_results or []:                 # executed text wins: retries may rewrite it
        cid = getattr(r, "candidate_id", None)
        if cid is not None and getattr(r, "success", False) and getattr(r, "sql", None):
            by_id[cid] = r.sql

    at_id = _norm(by_id.get(selected_id))
    want = _norm(selected_sql)
    if not want or want == at_id:
        return selected_id, selected_sql, "ok"

    matches = [cid for cid, sql in by_id.items() if _norm(sql) == want]
    if len(matches) == 1:
        return matches[0], selected_sql, "id_corrected"
    if len(matches) > 1:
        return (selected_id if selected_id in matches else matches[0]), selected_sql, "ambiguous_ok"
    if at_id:
        return selected_id, by_id[selected_id], "sql_replaced"
    return selected_id, selected_sql, "unresolved"


def _patch():
    from src.selection.judge import SQLJudge
    if getattr(SQLJudge, "_qasql_id_fix", False):
        return True
    original = SQLJudge.judge

    @functools.wraps(original)
    def judge(self, candidates, execution_results, nl_query, evidence="", schema_str=None, **kw):
        res = original(self, candidates, execution_results, nl_query,
                       evidence=evidence, schema_str=schema_str, **kw)
        new_id, new_sql, action = reconcile(res.selected_id, res.selected_sql,
                                            candidates, execution_results)
        if os.environ.get("QASQL_JUDGE_ID_FIX_DEBUG") == "1":
            print(f"[id_fix] fired: id={res.selected_id} action={action} -> {new_id}",
                  file=sys.stderr, flush=True)
        if action != "ok":
            _log({"question": nl_query, "action": action,
                  "id_before": res.selected_id, "id_after": new_id,
                  "sql_before": _norm(res.selected_sql)[:400], "sql_after": _norm(new_sql)[:400]})
            res.selected_id, res.selected_sql = new_id, new_sql
        return res

    SQLJudge.judge = judge
    SQLJudge._qasql_id_fix = True
    return True


def install():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        return _patch()
    except ImportError:
        return False


ENABLED = os.environ.get("QASQL_JUDGE_ID_FIX") == "1"
