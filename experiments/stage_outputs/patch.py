"""Persist the judge's and the fixer's outputs, which the pipeline otherwise discards.

What src/ keeps today: only the FINAL SQL, in selected.json. The judge's reasoning, its
confidence and which candidate it chose are thrown away once the stage returns, and the fixer's
before/after is likewise lost - selected.json holds the post-fixer text with no record of what
went in. That made several questions in this project unanswerable without reconstruction.

This writes two extra files into the run's output directory, keyed by question index exactly like
selected.json:

  judge_output.json   {idx: {selected_id, strategy, sql, confidence, reasoning,
                             total_candidates, successful_candidates}}
  fixer_output.json   {idx: {candidate_id, sql_in, sql_out, changed, is_acceptable,
                             refined, iterations, issues}}

Indexing: the judge and the fixer never see the question index, so each call is buffered and then
flushed by `_append_bird_entry`, which does know it. That is the same seam selected.json is
written through, so the keys line up by construction rather than by matching question text - which
is what the .jsonl logs do, and why two duplicate-worded questions collide in them.

src/ is untouched; all three methods are wrapped in memory. Enable with QASQL_STAGE_OUTPUTS=1.
"""
import functools
import json
import os
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

_PENDING = threading.local()
_LOCK = threading.Lock()


def _buf():
    if not hasattr(_PENDING, "d"):
        _PENDING.d = {}
    return _PENDING.d


def _merge(path, idx, record):
    """Read-modify-write one key, mirroring how _append_bird_entry maintains its files."""
    with _LOCK:
        data = {}
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                data = {}
        data[idx] = record
        path.write_text(json.dumps({k: data[k] for k in sorted(data, key=int)},
                                   indent=4, ensure_ascii=False), encoding="utf-8")


def _patch_judge():
    from src.selection.judge import SQLJudge
    if getattr(SQLJudge, "_qasql_stage_outputs", False):
        return True
    original = SQLJudge.judge

    @functools.wraps(original)
    def judge(self, candidates, execution_results, nl_query, evidence="", schema_str=None, **kw):
        res = original(self, candidates, execution_results, nl_query,
                       evidence=evidence, schema_str=schema_str, **kw)
        strategy = next((getattr(c, "strategy_name", None) for c in candidates
                         if getattr(c, "candidate_id", None) == res.selected_id), None)
        _patch_flush()                 # __main__.QASQLPipeline only exists once the run starts
        _buf()["judge"] = {
            "selected_id": res.selected_id,
            "strategy": strategy,
            "sql": (res.selected_sql or "").strip(),
            "confidence": res.confidence,
            "reasoning": res.reasoning,
            "total_candidates": res.total_candidates,
            "successful_candidates": res.successful_candidates,
        }
        return res

    SQLJudge.judge = judge
    SQLJudge._qasql_stage_outputs = True
    return True


def _patch_fixer():
    from src.selection.fixer import SQLFixer
    if getattr(SQLFixer, "_qasql_stage_outputs", False):
        return True
    original = SQLFixer.fix

    @functools.wraps(original)
    def fix(self, candidate, execution_result, nl_query, evidence, db_path, schema_str="", **kw):
        sql_in = (getattr(execution_result, "sql", None) or getattr(candidate, "sql", "") or "").strip()
        out = original(self, candidate, execution_result, nl_query, evidence, db_path,
                       schema_str=schema_str, **kw)
        sql_out = (out.final_sql or "").strip()
        _buf()["fixer"] = {
            "candidate_id": out.candidate_id,
            "sql_in": sql_in,
            "sql_out": sql_out,
            "changed": sql_in != sql_out,
            "is_acceptable": out.is_acceptable,
            "refined": out.refined,
            "iterations": out.iterations,
            "issues": out.issues,
        }
        return out

    SQLFixer.fix = fix
    SQLFixer._qasql_stage_outputs = True
    return True


def _flush_targets():
    """Every live QASQLPipeline class object.

    `scripts/run_pipeline.sh` launches `python -m src.pipeline`, so the running class is
    `__main__.QASQLPipeline` - a DIFFERENT object from `src.pipeline.QASQLPipeline`. Patching only
    the latter silently does nothing, which is exactly what happened on the first attempt. Both
    are patched, and `__main__` only exists once the pipeline is running, so this is called
    lazily rather than at interpreter start-up.
    """
    out = []
    for name in ("__main__", "src.pipeline"):
        mod = sys.modules.get(name)
        cls = getattr(mod, "QASQLPipeline", None) if mod else None
        if cls is not None and cls not in out:
            out.append(cls)
    return out


def _patch_flush():
    done = False
    for cls in _flush_targets():
        if getattr(cls, "_qasql_stage_outputs", False):
            done = True
            continue
        original = cls._append_bird_entry

        @functools.wraps(original)
        def _append_bird_entry(self, filepath, idx, sql, db_name, _orig=original):
            _orig(self, filepath, idx, sql, db_name)
            if Path(filepath).name != "selected.json":
                return                                # candidate files are written first; ignore
            d = _buf()
            out = Path(filepath).parent
            if "judge" in d:
                _merge(out / "judge_output.json", str(idx), d.pop("judge"))
            if "fixer" in d:
                _merge(out / "fixer_output.json", str(idx), d.pop("fixer"))
            d.clear()

        cls._append_bird_entry = _append_bird_entry
        cls._qasql_stage_outputs = True
        done = True
    return done


def install():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        ok = _patch_judge() and _patch_fixer()
        _patch_flush()                     # best-effort now; the judge wrapper retries later
        return ok
    except (ImportError, AttributeError) as exc:
        print(f"[patches] WARNING: stage outputs not installed: {exc}", file=sys.stderr, flush=True)
        return False


ENABLED = os.environ.get("QASQL_STAGE_OUTPUTS") == "1"
