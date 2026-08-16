"""Guard the flywheel convention: every ship-able workflow must be parameterized.

A workflow without PARAM_ placeholders is invisible to auto-tool registration
and effectively a dead asset (only reachable via generic run_workflow). We now
require every workflow that contains a prompt/image/tags/lyrics/seed input to
expose at least one placeholder, and assert the auto-tool count stays healthy.
"""

import json
from pathlib import Path

from managers.workflow_manager import WorkflowManager

WORKFLOWS_DIR = Path(__file__).resolve().parents[1] / "workflows"

# Workflows intentionally left unparameterized on semantic grounds:
# MiniMax i2v's prompt embeds <Picture 1> and must stay untouched.
EXEMPT_WORKFLOWS = {
    "api_video_minimax_h3_i2v.json",
}

TEXT_KEYS = ("image", "image1", "image2", "text", "prompt", "positive",
             "negative", "tags", "lyrics")
SEED_KEYS = ("seed", "noise_seed")


def _workflow_has_parameter_placeholder(workflow: dict) -> bool:
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        for value in node.get("inputs", {}).values():
            if isinstance(value, str) and value.startswith("PARAM_"):
                return True
    return False


def _workflow_has_parameterizable_input(workflow: dict) -> bool:
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        for key, value in node.get("inputs", {}).items():
            if key in TEXT_KEYS and isinstance(value, str) and value.strip():
                return True
            if key in SEED_KEYS and isinstance(value, int):
                return True
    return False


def test_every_parameterizable_workflow_is_parameterized():
    missing = []
    for path in sorted(WORKFLOWS_DIR.glob("*.json")):
        workflow = json.loads(path.read_text(encoding="utf-8"))
        if _workflow_has_parameter_placeholder(workflow):
            continue
        if path.name in EXEMPT_WORKFLOWS:
            continue
        if _workflow_has_parameterizable_input(workflow):
            missing.append(path.name)
    assert missing == [], f"Unparameterized workflows (dead assets): {missing}"


def test_workflow_manager_auto_registers_expected_tool_count():
    wm = WorkflowManager(WORKFLOWS_DIR)
    # 2026-08-16: 15 -> 39 after parameterizing the audio/video backlog.
    assert len(wm.tool_definitions) >= 39
    tool_names = {d.tool_name for d in wm.tool_definitions}
    for expected in (
        "api_audio_minimax_music_3",
        "api_video_wan2_2_14b_i2v",
        "api_video_wan_vace_inpainting",
        "api_video_ltx2_3_flf2v",
    ):
        assert expected in tool_names


def test_audio_parameters_are_required():
    wm = WorkflowManager(WORKFLOWS_DIR)
    audio = next(d for d in wm.tool_definitions if d.workflow_id == "api_audio_minimax_music_3")
    names = {p.name: p for p in audio.parameters.values()}
    assert names["lyrics"].required is True
    assert names["seed"].required is False
