"""Claude CLI adapter with validated, ID-only structured output."""
import json
from pathlib import Path
import subprocess
import tempfile

from analysis.null_duplicate_ablation import write_json


class StructuredClaudeClient:
    def __init__(self, model, log_dir, timeout):
        self.model, self.log_dir, self.timeout = model, Path(log_dir), timeout
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.calls = 0

    def complete(self, prompt, system_prompt=None, max_tokens=2048, temperature=0):
        # max_tokens/temperature are shared-interface arguments, not CLI controls.
        self.calls += 1
        prefix = self.log_dir / f"call_{self.calls}"
        payload = json.loads(prompt)
        allowed = [c["id"] for c in payload["candidates"] if c["error"] is None]
        schema = {"type": "object", "properties": {
            "selected_id": {"type": ["string", "null"], "enum": [*allowed, None]},
            "reasoning": {"type": "string", "minLength": 1}},
            "required": ["selected_id", "reasoning"], "additionalProperties": False}
        write_json(prefix.with_suffix(".request.json"), {
            "system": system_prompt, "user": prompt, "model": self.model, "json_schema": schema})
        args = ["claude", "-p", "--model", self.model, "--safe-mode", "--tools", "",
                "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
                "--no-session-persistence", "--output-format", "json",
                "--json-schema", json.dumps(schema), "--system-prompt", system_prompt or ""]
        process = subprocess.run(args, input=prompt, capture_output=True, text=True,
            encoding="utf-8", errors="replace", cwd=tempfile.gettempdir(), timeout=self.timeout)
        try:
            data = json.loads(process.stdout)
        except json.JSONDecodeError as exc:
            write_json(prefix.with_suffix(".response.json"), {
                "returncode": process.returncode, "stdout": process.stdout, "stderr": process.stderr})
            raise ValueError("CLI returned a non-JSON envelope") from exc
        write_json(prefix.with_suffix(".response.json"), data)
        if process.returncode or data.get("is_error"):
            raise ValueError(f"Claude error: {str(data.get('result', data.get('subtype')))[:300]}")
        structured = data.get("structured_output")
        if not isinstance(structured, dict):
            raise ValueError("CLI did not return a structured_output object")
        return json.dumps(structured, ensure_ascii=False)
