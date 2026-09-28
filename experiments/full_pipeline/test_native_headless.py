"""Tests for headless_patch/sitecustomize.py without calling Claude."""
import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

PATCH = Path(__file__).parent / "headless_patch" / "sitecustomize.py"


def load_patch(monkeypatch, enabled):
    if enabled:
        monkeypatch.setenv("QASQL_NATIVE_HEADLESS", "1")
    else:
        monkeypatch.delenv("QASQL_NATIVE_HEADLESS", raising=False)
    spec = importlib.util.spec_from_file_location("qasql_sitecustomize_test", PATCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prompt_goes_through_stdin_and_system_through_file(monkeypatch):
    patch = load_patch(monkeypatch, enabled=False)
    monkeypatch.setattr(patch.shutil, "which", lambda name: "claude")
    seen = {}

    def runner(args, **kwargs):
        seen["args"], seen["kwargs"] = args, kwargs
        seen["system"] = Path(args[args.index("--system-prompt-file") + 1]).read_text(encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout=" SELECT 1 \n", stderr="")

    prompt = "Return `relevant_columns` for `satscores.dname`; $HOME " + "x" * 50000
    out = patch.run_claude(prompt, system="System with `backticks`", model="claude-sonnet-4-6", runner=runner)
    assert out == "SELECT 1"
    assert seen["kwargs"]["input"] == prompt and prompt not in seen["args"]
    assert seen["system"] == "System with `backticks`"
    assert seen["args"][seen["args"].index("--tools") + 1] == ""
    assert "--strict-mcp-config" in seen["args"] and "--model" in seen["args"]
    assert seen["kwargs"]["cwd"] == tempfile.gettempdir()
    assert "shell" not in seen["kwargs"]
    assert not Path(seen["args"][seen["args"].index("--system-prompt-file") + 1]).exists()


def test_error_is_raised_with_claude_code_prefix(monkeypatch):
    patch = load_patch(monkeypatch, enabled=False)
    monkeypatch.setattr(patch.shutil, "which", lambda name: "claude")
    runner = lambda args, **kw: SimpleNamespace(returncode=1, stdout="", stderr="rate limited")
    with pytest.raises(RuntimeError, match="Claude Code error: rate limited"):
        patch.run_claude("p", "s", runner=runner)


def test_install_replaces_package_functions_and_client_uses_them(monkeypatch):
    pytest.importorskip("claude_code_headless")
    import claude_code_headless as package
    from claude_code_headless import client
    original = (package.call_claude, package.call_claude_with_system, client.call_claude, client.call_claude_with_system)
    try:
        patch = load_patch(monkeypatch, enabled=True)
        assert package.QASQL_NATIVE_PATCH is True
        calls = []
        monkeypatch.setattr(patch, "run_claude", lambda prompt, system, model: calls.append((prompt, system, model)) or "ok")
        monkeypatch.setattr(client, "_apply_rate_limit", lambda rate_limit: None)
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from src.utils.llm_client import ClaudeCodeHeadlessClient
        llm = ClaudeCodeHeadlessClient(model="claude-sonnet-4-6")
        assert llm.complete("user `x`", system_prompt="sys") == "ok"
        assert llm.complete("no system") == "ok"
        assert calls == [("user `x`", "sys", "claude-sonnet-4-6"), ("no system", None, "claude-sonnet-4-6")]
    finally:
        package.call_claude, package.call_claude_with_system, client.call_claude, client.call_claude_with_system = original
        if hasattr(package, "QASQL_NATIVE_PATCH"):
            del package.QASQL_NATIVE_PATCH
