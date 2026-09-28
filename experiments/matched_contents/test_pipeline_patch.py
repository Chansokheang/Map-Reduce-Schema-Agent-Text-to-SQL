"""Tests for the generation-prompt patch. No model calls, no gold."""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.matched_contents import pipeline_patch

INDEX = ROOT / "output/matched_contents/index"
pytestmark = pytest.mark.skipif(not INDEX.exists(), reason="value index not built")


def test_resolve_database_from_table_names():
    assert pipeline_patch.resolve_database({"schools": {}, "frpm": {}, "satscores": {}}) == "california_schools"
    assert pipeline_patch.resolve_database({"atom": {}, "bond": {}, "molecule": {}}) == "toxicology"
    assert pipeline_patch.resolve_database({"not_a_table": {}}) is None
    assert pipeline_patch.resolve_database({}) is None


def test_block_is_built_for_a_known_question():
    block = pipeline_patch._block("california_schools",
                                  "How many students are enrolled at the State Special School in Fremont?",
                                  "State Special School refers to EdOpsCode = 'SSS'", ())
    assert "# Matched contents" in block and "'SSS' -> schools.EdOpsCode" in block


def test_block_is_empty_when_nothing_matches():
    assert pipeline_patch._block("california_schools", "how many rows exist", "", ()) == ""


def test_install_appends_the_block_to_the_user_prompt(monkeypatch):
    from src.generation.prompt_builder import PromptBuilder, ContextStrategy

    def stub(self, nl_query, schema, strategy, focused_schema=None, profile=None, evidence=None):
        return {"system": "S", "user": f"QUESTION: {nl_query}", "strategy": "stub"}

    monkeypatch.setattr(PromptBuilder, "build", stub, raising=False)
    monkeypatch.setattr(PromptBuilder, "_qasql_matched_contents", False, raising=False)
    assert pipeline_patch.install()
    builder = PromptBuilder.__new__(PromptBuilder)
    schema = {"schools": {}, "frpm": {}, "satscores": {}}
    out = builder.build("Which schools are in Fremont?", schema, ContextStrategy.FULL_SCHEMA,
                        evidence="State Special School refers to EdOpsCode = 'SSS'")
    assert out["user"].startswith("QUESTION:") and "# Matched contents" in out["user"]
    assert "'SSS' -> schools.EdOpsCode" in out["matched_contents"] and "Fremont" in out["matched_contents"]
    # An unknown database leaves the prompt untouched.
    plain = builder.build("anything", {"unknown_table": {}}, ContextStrategy.FULL_SCHEMA)
    assert "# Matched contents" not in plain["user"] and "matched_contents" not in plain


def test_patches_sitecustomize_enables_only_what_is_flagged(monkeypatch, capsys):
    path = ROOT / "experiments/full_pipeline/patches/sitecustomize.py"
    from src.generation.prompt_builder import PromptBuilder
    monkeypatch.setattr(PromptBuilder, "_qasql_matched_contents", False, raising=False)
    monkeypatch.delenv("QASQL_NATIVE_HEADLESS", raising=False)
    monkeypatch.setenv("QASQL_MATCHED_CONTENTS", "1")
    spec = importlib.util.spec_from_file_location("qasql_patches_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert "matched contents" in capsys.readouterr().err
    assert getattr(PromptBuilder, "_qasql_matched_contents", False)
