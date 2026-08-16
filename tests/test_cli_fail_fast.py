"""Regression tests for the CLI fail-fast semantics.

The flywheel drives the CLI from shell scripts with `&&` chaining; a tool-level
``isError`` or a business ``{"error": ...}`` payload MUST produce a non-zero
exit path (``result_is_failure`` returns True → ``main`` returns 1).
"""

from comfy_mcp_cli import result_is_failure


def test_is_error_flag_is_always_a_failure():
    result = {"isError": True, "content": [{"type": "text", "text": "boom"}]}
    assert result_is_failure(result, {}, allow_error=True) is True


def test_business_error_is_a_failure_by_default():
    result = {"content": [{"type": "text", "text": '{"error": "missing model"}'}]}
    payload = {"error": "missing model"}
    assert result_is_failure(result, payload, allow_error=False) is True


def test_error_code_is_a_failure_by_default():
    payload = {"error_code": "CUDA_OOM", "message": "oom"}
    assert result_is_failure({}, payload, allow_error=False) is True


def test_allow_error_escape_hatch_preserves_legacy_behaviour():
    result = {"content": [{"type": "text", "text": '{"error": "advisory"}'}]}
    payload = {"error": "advisory"}
    assert result_is_failure(result, payload, allow_error=True) is False


def test_success_payload_is_not_a_failure():
    payload = {"asset_id": "abc", "filename": "out.png"}
    assert result_is_failure({}, payload, allow_error=False) is False
