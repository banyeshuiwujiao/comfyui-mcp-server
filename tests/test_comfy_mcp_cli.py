"""comfy_mcp_cli regression tests: --out capture for every subcommand.

The CLI is the primary shell/agent entry point of the data flywheel. Agents
must be able to save raw JSON results with ``--out`` for *all* subcommands
(tools / read / prompts / call) instead of relying on shell redirection.
"""

import json

import comfy_mcp_cli


class FakeClient:
    def __init__(self, tools=None, prompts=None, resource=None, call_result=None):
        self._tools = tools or []
        self._prompts = prompts or []
        self._resource = resource or {}
        self._call_result = call_result or {}

    def initialize(self):
        pass

    def tools(self):
        return self._tools

    def list_prompts(self):
        return self._prompts

    def read_resource(self, uri):
        return self._resource

    def call(self, tool, arguments):
        return self._call_result


def patch_client(monkeypatch, fake):
    monkeypatch.setattr(comfy_mcp_cli, "McpClient", lambda endpoint=None, timeout=None: fake)


def test_tools_out_writes_raw_tool_json(monkeypatch, tmp_path, capsys):
    tools = [
        {
            "name": "api_image_z_image_turbo_t2i",
            "inputSchema": {"type": "object", "properties": {"prompt": {"type": "string"}}},
        }
    ]
    out = tmp_path / "tools.json"
    patch_client(monkeypatch, FakeClient(tools=tools))

    assert comfy_mcp_cli.main(["tools", "--out", str(out)]) == 0

    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved == tools
    assert "api_image_z_image_turbo_t2i" in capsys.readouterr().out


def test_read_out_writes_raw_resource_json(monkeypatch, tmp_path, capsys):
    resource = {"uri": "comfyui://system/gpu-health", "status": "healthy"}
    out = tmp_path / "health.json"
    patch_client(monkeypatch, FakeClient(resource=resource))

    assert comfy_mcp_cli.main(["read", "comfyui://system/gpu-health", "--out", str(out)]) == 0

    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved == resource
    assert '"status": "healthy"' in capsys.readouterr().out


def test_prompts_out_writes_raw_prompt_json(monkeypatch, tmp_path, capsys):
    prompts = [{"name": "hero-sheet", "description": "Generate a hero sheet"}]
    out = tmp_path / "prompts.json"
    patch_client(monkeypatch, FakeClient(prompts=prompts))

    assert comfy_mcp_cli.main(["prompts", "--out", str(out)]) == 0

    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved == prompts
    assert "hero-sheet" in capsys.readouterr().out


def test_call_out_still_parses_text_payload_and_writes(monkeypatch, tmp_path, capsys):
    result = {
        "isError": False,
        "content": [{"type": "text", "text": json.dumps({"asset_id": "abc-123"})}],
    }
    out = tmp_path / "call.json"
    patch_client(monkeypatch, FakeClient(call_result=result))

    assert comfy_mcp_cli.main(["call", "fake_tool", '{"prompt":"x"}', "--out", str(out)]) == 0

    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved == {"asset_id": "abc-123"}


def test_call_failure_writes_error_output_before_exit_1(monkeypatch, tmp_path):
    result = {
        "isError": True,
        "content": [{"type": "text", "text": "lyrics\n  Missing required argument"}],
    }
    out = tmp_path / "call-error.json"
    patch_client(monkeypatch, FakeClient(call_result=result))

    assert comfy_mcp_cli.main(["call", "fake_tool", "{}", "--out", str(out)]) == 1

    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["isError"] is True
