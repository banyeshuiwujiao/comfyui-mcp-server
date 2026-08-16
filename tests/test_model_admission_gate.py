"""Admission-gate regression: PARAM_MODEL must only be checkpoint-validated
when it feeds a native ComfyUI loader input key.

JoyCaption and other custom loaders take HuggingFace model ids through an
input named ``model``. The live cold-start attempt failed with
"Default model '' not found in ComfyUI checkpoints" before this fix.
"""

import json

from managers.workflow_manager import WorkflowManager


def _write_workflow(tmp_path, name, model_input_key, class_type):
    workflow = {
        "1": {
            "inputs": {model_input_key: "PARAM_MODEL"},
            "class_type": class_type,
        },
        "2": {
            "inputs": {"text": ["1", 0]},
            "class_type": "SaveText",
        },
    }
    wf_dir = tmp_path / "workflows"
    wf_dir.mkdir()
    (wf_dir / f"{name}.json").write_text(json.dumps(workflow), encoding="utf-8")
    return wf_dir


def test_joycaption_hf_model_bypasses_checkpoint_gate(tmp_path):
    wf_dir = _write_workflow(
        tmp_path, "joycaption", "model", "Joy_caption_two_load"
    )
    mgr = WorkflowManager(wf_dir)

    assert mgr.model_param_targets_checkpoint_loader("joycaption") is False


def test_native_checkpoint_loader_still_gated(tmp_path):
    wf_dir = _write_workflow(
        tmp_path, "native", "ckpt_name", "CheckpointLoaderSimple"
    )
    mgr = WorkflowManager(wf_dir)

    assert mgr.model_param_targets_checkpoint_loader("native") is True


def test_no_model_param_not_gated(tmp_path):
    workflow = {
        "1": {
            "inputs": {"text": "PARAM_PROMPT"},
            "class_type": "CLIPTextEncode",
        },
        "2": {
            "inputs": {"text": ["1", 0]},
            "class_type": "SaveText",
        },
    }
    wf_dir = tmp_path / "workflows"
    wf_dir.mkdir()
    (wf_dir / "prompt_only.json").write_text(json.dumps(workflow), encoding="utf-8")
    mgr = WorkflowManager(wf_dir)

    assert mgr.model_param_targets_checkpoint_loader("prompt_only") is False
