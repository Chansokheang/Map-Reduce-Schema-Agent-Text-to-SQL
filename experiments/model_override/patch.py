"""Per-component model selection. src/ shares ONE client across every stage; this splits it.

Measured 2026-09-29 on gr_v1: replaying the judge with Opus over the 92 questions where the Sonnet
judge picked a wrong candidate recovered 56 of them and lost none, while breaking 1 of 150 sampled
wins (0.7%). Generation was untouched in that test, so the gain is attributable to the judge alone.

`-m MODEL` on the pipeline switches every stage at once, which multiplies cost by roughly the number
of calls per question (schema agent + 5 candidates + fixer + judge). These variables target a stage:

  QASQL_MODEL_JUDGE=claude-opus-5-5        judge only  (the measured win)
  QASQL_MODEL_FIXER=claude-opus-5-5        fixer only
  QASQL_MODEL_GENERATION=claude-opus-5-5   candidate generation only
  QASQL_MODEL_SCHEMA=claude-opus-5-5       schema agent only

Anything left unset keeps whatever the pipeline was started with.
"""
import functools
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

TARGETS = {
    "QASQL_MODEL_JUDGE": ("src.selection.judge", "SQLJudge"),
    "QASQL_MODEL_FIXER": ("src.selection.fixer", "SQLFixer"),
    "QASQL_MODEL_GENERATION": ("src.generation.candidate_generator", "CandidateGenerator"),
    "QASQL_MODEL_SCHEMA": ("src.agents.manager", "SchemaManager"),
}
_clients = {}


def client_for(model):
    """One client per model, reused across components."""
    if model not in _clients:
        from src.utils.llm_client import create_llm_client
        _clients[model] = create_llm_client(provider="headless", model=model)
    return _clients[model]


def _patch(module_name, class_name, model):
    import importlib
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    if getattr(cls, "_qasql_model_override", None) == model:
        return True
    original = cls.__init__

    @functools.wraps(original)
    def __init__(self, *args, **kwargs):
        original(self, *args, **kwargs)
        if getattr(self, "llm_client", None) is not None:
            self.llm_client = client_for(model)

    cls.__init__ = __init__
    cls._qasql_model_override = model
    return True


def applied():
    """{component: model} for whatever is configured."""
    return {var.replace("QASQL_MODEL_", "").lower(): os.environ[var]
            for var in TARGETS if os.environ.get(var)}


def install():
    done = {}
    for var, (module_name, class_name) in TARGETS.items():
        model = os.environ.get(var)
        if not model:
            continue
        try:
            if _patch(module_name, class_name, model):
                done[class_name] = model
        except (ImportError, AttributeError) as exc:
            print(f"[patches] WARNING: could not set {class_name} to {model}: {exc}",
                  file=sys.stderr, flush=True)
    return done


ENABLED = bool(applied())
