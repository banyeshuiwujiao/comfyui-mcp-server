"""Text-output workflows: JoyCaption returns STRING previews and/or .txt files.

The live cold-start produced ComfyUI outputs {"5": {"text": [...]},
"3": {"text": [...], "files": [...]}} but the generated tool preferred image
keys and died with "No outputs matched preferred keys". These tests guard the
detection and the file-backed persistence of pure STRING outputs.
"""

from comfyui_client import ComfyUIClient
from managers.workflow_manager import TEXT_OUTPUT_KEYS, WorkflowManager


def _client(tmp_path):
    client = object.__new__(ComfyUIClient)
    client.base_url = "http://localhost:8188"
    client.output_root = str(tmp_path / "output")
    return client


def test_joy_caption_workflow_detected_as_text():
    workflow = {
        "1": {"class_type": "Joy_caption_two_load", "inputs": {"model": "PARAM_MODEL"}},
        "2": {"class_type": "Joy_caption_two", "inputs": {"image": "x.png"}},
        "3": {"class_type": "Joy_caption_two_output", "inputs": {"text": ["2", 0]}},
    }
    import json
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        wf_dir = Path(tmp) / "workflows"
        wf_dir.mkdir()
        (wf_dir / "joy.json").write_text(json.dumps(workflow), encoding="utf-8")
        mgr = WorkflowManager(wf_dir)
        assert mgr.tool_definitions[0].output_preferences == TEXT_OUTPUT_KEYS


def test_text_output_persisted_when_no_file_asset(tmp_path):
    client = _client(tmp_path)
    outputs = {"5": {"text": ["A caption without a sibling file."]}}
    workflow = {"5": {"class_type": "Joy_caption_two_output", "inputs": {"text": ["2", 0]}}}

    assets = client._extract_all_assets(outputs, TEXT_OUTPUT_KEYS, workflow)

    assert len(assets) == 1
    assert assets[0]["filename"].endswith(".txt")
    assert assets[0]["subfolder"] == "joy_caption"
    assert assets[0]["asset_url"].startswith("http://localhost:8188/view?")


def test_existing_file_asset_skips_duplicate_text(tmp_path):
    client = _client(tmp_path)
    outputs = {
        "3": {
            "text": ["same caption"],
            "files": [
                {"filename": "basic_caption_00003.txt", "subfolder": "joy_caption", "type": "output"}
            ],
        }
    }
    workflow = {"3": {"class_type": "Joy_caption_two_output"}}

    assets = client._extract_all_assets(outputs, TEXT_OUTPUT_KEYS, workflow)

    assert len(assets) == 1
    assert assets[0]["filename"] == "basic_caption_00003.txt"


def test_text_skipped_gracefully_without_output_root():
    client = object.__new__(ComfyUIClient)
    client.base_url = "http://localhost:8188"
    client.output_root = None

    assets = client._extract_all_assets(
        {"5": {"text": ["no place to persist"]}},
        TEXT_OUTPUT_KEYS,
        {"5": {"class_type": "Joy_caption_two_output"}},
    )

    assert assets == []
