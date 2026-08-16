#!/usr/bin/env python3
"""Agent-friendly CLI for the ComfyUI MCP streamable-http endpoint.

Unlike test_client.py (demo-focused), this client does the full MCP
handshake and can call ANY tool with arbitrary JSON arguments, so shell
scripts and AI agents can drive the server without an MCP SDK.

Usage:
  python comfy_mcp_cli.py tools [--out tools.json]    # list all tools
  python comfy_mcp_cli.py call TOOL '{"prompt":"..."}' [--timeout 900] [--out result.json]
  python comfy_mcp_cli.py read comfyui://system/gpu-health [--out health.json]
  python comfy_mcp_cli.py prompts [--out prompts.json]  # list prompts

Exit code is 0 on success, 1 on any JSON-RPC/tool error.
"""
import argparse
import json
import sys
from typing import Any, Dict, List, Optional

import requests

DEFAULT_ENDPOINT = "http://127.0.0.1:9000/mcp"
DEFAULT_TIMEOUT = 900
HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
}


def parse_sse_response(text: str) -> Optional[dict]:
    """Parse the first valid JSON `data:` frame from an SSE response."""
    for line in text.replace("\r\n", "\n").split("\n"):
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload:
            continue
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            continue
    # Some servers return a bare JSON body despite an SSE content type.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


class McpClient:
    def __init__(self, endpoint: str, timeout: int):
        self.endpoint = endpoint
        self.timeout = timeout
        self.session = requests.Session()
        self.session_id: Optional[str] = None
        self._next_id = 1

    def initialize(self) -> dict:
        resp = self._post(
            "initialize",
            {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "comfy-mcp-cli", "version": "1.0.0"},
            },
        )
        if "error" in resp:
            raise RuntimeError(f"initialize failed: {resp['error']}")
        return resp

    def _post(self, method: str, params: Dict[str, Any]) -> dict:
        request_id = self._next_id
        self._next_id += 1
        headers = dict(HEADERS)
        if self.session_id:
            headers["mcp-session-id"] = self.session_id
        http = self.session.post(
            self.endpoint,
            json={
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params,
            },
            headers=headers,
            timeout=self.timeout,
        )
        http.raise_for_status()
        sid = http.headers.get("mcp-session-id")
        if sid:
            self.session_id = sid
        ctype = http.headers.get("content-type", "")
        if "text/event-stream" in ctype:
            data = parse_sse_response(http.text)
        else:
            data = http.json()
        if data is None:
            raise RuntimeError(f"empty/unparseable response for {method}: {http.text[:300]!r}")
        return data

    def tools(self):
        resp = self._post("tools/list", {})
        if "error" in resp:
            raise RuntimeError(f"tools/list failed: {resp['error']}")
        return resp.get("result", {}).get("tools", [])

    def call(self, tool: str, arguments: Dict[str, Any]):
        resp = self._post("tools/call", {"name": tool, "arguments": arguments})
        if "error" in resp:
            raise RuntimeError(f"tools/call {tool} failed: {json.dumps(resp['error'], ensure_ascii=False)}")
        return resp.get("result", {})

    def read_resource(self, uri: str):
        resp = self._post("resources/read", {"uri": uri})
        if "error" in resp:
            raise RuntimeError(f"resources/read failed: {resp['error']}")
        return resp.get("result", {})

    def list_prompts(self):
        resp = self._post("prompts/list", {})
        if "error" in resp:
            raise RuntimeError(f"prompts/list failed: {resp['error']}")
        return resp.get("result", {}).get("prompts", [])


def extract_text(result: dict) -> str:
    """Return the first text payload of an MCP tool result."""
    for item in result.get("content", []):
        if isinstance(item, dict) and "text" in item:
            return item["text"]
    return json.dumps(result, ensure_ascii=False)


def result_is_failure(result: dict, payload: dict, allow_error: bool) -> bool:
    """Classify MCP-level and business-level tool failures.

    - ``result["isError"]`` is the MCP protocol flag (HTTP 200 + JSON-RPC ok).
    - A parsed business payload containing ``error`` / ``error_code`` means the
      tool ran but its business logic failed (GPU guard, missing workflow, ...).
    """
    if isinstance(result, dict) and result.get("isError") is True:
        return True
    if not allow_error and isinstance(payload, dict) and (
        "error" in payload or "error_code" in payload
    ):
        return True
    return False


def write_out(path: Optional[str], text: str) -> None:
    """Write captured JSON to ``path`` when requested (agents avoid shell redirection)."""
    if not path:
        return
    with open(path, "w", encoding="utf-8") as f:
        f.write(text if text.endswith("\n") else text + "\n")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    common.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    out_common = argparse.ArgumentParser(add_help=False)
    out_common.add_argument("--out", help="Optional path to write the raw JSON result")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("tools", help="List all MCP tools", parents=[common, out_common])

    p_call = sub.add_parser("call", help="Call a tool with JSON arguments", parents=[common, out_common])
    p_call.add_argument("tool")
    p_call.add_argument("arguments", help="JSON object or path to a JSON file (@file)")
    p_call.add_argument(
        "--allow-error",
        action="store_true",
        help="Treat tool-level isError / business 'error' responses as success (exit code 0). "
             "By default any failure makes the CLI exit 1 so shell pipelines fail fast.",
    )

    p_read = sub.add_parser("read", help="Read an MCP resource URI", parents=[common, out_common])
    p_read.add_argument("uri")

    sub.add_parser("prompts", help="List MCP prompts", parents=[common, out_common])

    args = parser.parse_args(argv)
    if not hasattr(args, "endpoint"):
        args.endpoint = DEFAULT_ENDPOINT
    if not hasattr(args, "timeout"):
        args.timeout = DEFAULT_TIMEOUT
    client = McpClient(args.endpoint, args.timeout)
    client.initialize()

    try:
        if args.command == "tools":
            tools = client.tools()
            for tool in tools:
                params = list((tool.get("inputSchema") or {}).get("properties", {}).keys())
                print(f"{tool['name']:<48} [{', '.join(params)}]")
            write_out(args.out, json.dumps(tools, ensure_ascii=False, indent=2))
            return 0

        if args.command == "call":
            raw = args.arguments
            if raw.startswith("@"):
                with open(raw[1:], "r", encoding="utf-8") as f:
                    raw = f.read()
            arguments = json.loads(raw)
            result = client.call(args.tool, arguments)
            text = extract_text(result)
            try:
                # Prefer the parsed JSON when the tool already returned one.
                parsed = json.loads(text)
            except (json.JSONDecodeError, TypeError):
                parsed = None
            payload = parsed if isinstance(parsed, dict) else result
            output = json.dumps(payload, ensure_ascii=False, indent=2)
            print(output)
            write_out(args.out, output)

            # MCP-level failure: result.isError=true (HTTP 200, JSON-RPC success).
            if result_is_failure(result, payload, args.allow_error):
                reason = text
                if isinstance(result, dict) and result.get("isError") is True:
                    reason = text
                else:
                    reason = payload.get("error") or payload.get("error_code") or output
                print(f"[X] tool '{args.tool}' failed: {reason}", file=sys.stderr)
                return 1
            return 0

        if args.command == "read":
            output = json.dumps(client.read_resource(args.uri), ensure_ascii=False, indent=2)
            print(output)
            write_out(args.out, output)
            return 0

        if args.command == "prompts":
            prompts = client.list_prompts()
            for prompt in prompts:
                print(f"{prompt.get('name')}: {prompt.get('description', '')}")
            write_out(args.out, json.dumps(prompts, ensure_ascii=False, indent=2))
            return 0
    except (requests.RequestException, RuntimeError, ValueError) as e:
        print(f"[X] {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
