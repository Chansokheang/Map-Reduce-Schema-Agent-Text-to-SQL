"""Structured, tool-free requests with per-stage immutable logs."""
import json
from pathlib import Path
import subprocess
import tempfile
import jsonschema

from analysis.selection_experiment import write_json


def complete(payload, prompt, schema, folder, model, timeout=240):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    request = folder / "request.json"
    if request.exists():
        raise FileExistsError("Request already attempted; do not silently repeat it")
    user = json.dumps(payload, ensure_ascii=False)
    if len(user) + len(prompt) > 180000:
        raise ValueError("prompt_budget_exceeded")
    write_json(request, {"system": prompt, "user": payload, "json_schema": schema, "model": model})
    args = ["claude", "-p", "--model", model, "--safe-mode", "--tools", "",
        "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
        "--no-session-persistence", "--output-format", "json", "--json-schema",
        json.dumps(schema), "--system-prompt", prompt]
    result = subprocess.run(args, input=user, capture_output=True, text=True,
        encoding="utf-8", errors="replace", cwd=tempfile.gettempdir(), timeout=timeout)
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        write_json(folder / "response.json", {"returncode": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr})
        raise ValueError("Non-JSON CLI response")
    write_json(folder / "response.json", data)
    if result.returncode or data.get("is_error"):
        raise ValueError("CLI request failed: " + str(data.get("result", data.get("subtype")))[:250])
    answer = data.get("structured_output")
    jsonschema.validate(answer, schema)
    return answer
