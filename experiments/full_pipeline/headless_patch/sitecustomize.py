"""Native Claude CLI calls for claude-code-headless (loaded via PYTHONPATH by run_full_pipeline.sh).

Python imports a module named sitecustomize at start-up. run_full_pipeline.sh puts this
folder first on PYTHONPATH and sets QASQL_NATIVE_HEADLESS=1 when --headless is used, so the
unchanged src/ pipeline picks up these replacements for claude_code_headless.call_claude and
call_claude_with_system.

Why: on Windows the package runs `wsl bash -c '<quoted prompt>'`. The outer shell expands
backticks inside prompts (e.g. `relevant_columns`) and long prompts exceed the Windows
command-line limit (WinError 206). This version runs the local claude executable without a
shell, sends the prompt on stdin and the system prompt through --system-prompt-file.

It also disables tools and MCP servers and runs from the temp folder, so generation cannot
read project files such as dev.json. The package's rate limiting and default model are kept.
"""
import os
import shutil
import subprocess
import sys
import tempfile

PATCH_FLAG = "QASQL_NATIVE_HEADLESS"
TIMEOUT_SECONDS = float(os.environ.get("QASQL_NATIVE_HEADLESS_TIMEOUT", "900"))


def build_args(executable, model, system_file):
    args = [executable, "-p", "--output-format", "text", "--safe-mode", "--tools", "",
            "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}', "--no-session-persistence"]
    if model:
        args += ["--model", model]
    if system_file:
        args += ["--system-prompt-file", system_file]
    return args


def run_claude(prompt, system=None, model=None, runner=subprocess.run):
    executable = shutil.which("claude")
    if not executable:
        raise RuntimeError("Claude Code error: claude executable not found on PATH")
    system_file = None
    try:
        if system is not None:
            fd, system_file = tempfile.mkstemp(prefix="qasql_system_", suffix=".txt")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(system)
        result = runner(build_args(executable, model, system_file), input=prompt, capture_output=True,
                        text=True, encoding="utf-8", errors="replace", cwd=tempfile.gettempdir(),
                        timeout=TIMEOUT_SECONDS)
    finally:
        if system_file and os.path.exists(system_file):
            os.remove(system_file)
    if result.returncode != 0:
        raise RuntimeError(f"Claude Code error: {(result.stderr or result.stdout).strip()}")
    return result.stdout.strip()


def install():
    try:
        import claude_code_headless as package
        from claude_code_headless import client
    except ImportError:
        return False

    def call_claude_with_system(prompt, system, model=None, rate_limit=None):
        client._apply_rate_limit(rate_limit)
        return run_claude(prompt, system, model or client.DEFAULT_MODEL)

    def call_claude(prompt, allowed_tools=None, model=None, rate_limit=None):
        if allowed_tools:
            raise RuntimeError("Claude Code error: native headless patch runs with tools disabled")
        client._apply_rate_limit(rate_limit)
        return run_claude(prompt, None, model or client.DEFAULT_MODEL)

    for module in (package, client):
        module.call_claude_with_system = call_claude_with_system
        module.call_claude = call_claude
    package.QASQL_NATIVE_PATCH = True
    return True


if os.environ.get(PATCH_FLAG) == "1":
    if install():
        print("[full_pipeline] native claude-code-headless patch active (stdin prompt, tools off, temp cwd)",
              file=sys.stderr, flush=True)
