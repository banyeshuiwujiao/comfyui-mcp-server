"""Workflow placeholder default syntax: PARAM_[TYPE_]NAME:default.

JoyCaption advanced/batch workflows embed per-template defaults (top_p 0.9,
temperature 0.6, extra-option booleans, caption_type) directly in the JSON so
agents may omit optional parameters instead of passing all 17 toggles.
"""

import json

from managers.workflow_manager import WorkflowManager


def _manager(tmp_path, name="joy_advanced", workflow=None):
    workflow = workflow or {
        "1": {
            "inputs": {
                "top_p": "PARAM_FLOAT_TOP_P:0.9",
                "temperature": "PARAM_FLOAT_TEMPERATURE:0.6",
                "caption_type": "PARAM_CAPTION_TYPE:Descriptive",
                "caption_length": "PARAM_CAPTION_LENGTH:long",
                "language": "PARAM_LANGUAGE:English",
                "lighting": "PARAM_BOOL_EXTRA_LIGHTING:true",
                "watermark": "PARAM_BOOL_EXTRA_WATERMARK:false",
                "steps": "PARAM_INT_STEPS:20",
                "text": "PARAM_PROMPT",
            },
            "class_type": "FakeNode",
        }
    }
    wf_dir = tmp_path / "workflows"
    wf_dir.mkdir()
    (wf_dir / f"{name}.json").write_text(json.dumps(workflow), encoding="utf-8")
    return WorkflowManager(wf_dir)


def test_placeholder_defaults_parsed_and_typed(tmp_path):
    mgr = _manager(tmp_path)
    params = mgr._extract_parameters(mgr.load_workflow("joy_advanced"))

    assert params["top_p"].default == 0.9
    assert params["temperature"].default == 0.6
    assert params["caption_type"].default == "Descriptive"
    assert params["extra_lighting"].default is True
    assert params["extra_watermark"].default is False
    assert params["steps"].default == 20
    assert params["prompt"].default is None

    assert params["top_p"].required is False
    assert params["extra_lighting"].required is False
    assert params["prompt"].required is True


def test_render_workflow_applies_embedded_defaults(tmp_path):
    mgr = _manager(tmp_path)
    definition = mgr.tool_definitions[0]
    rendered = mgr.render_workflow(definition, {"prompt": "a cat"})

    node = rendered["1"]["inputs"]
    assert node["top_p"] == 0.9
    assert node["temperature"] == 0.6
    assert node["caption_type"] == "Descriptive"
    assert node["caption_length"] == "long"
    assert node["language"] == "English"
    assert node["lighting"] is True
    assert node["watermark"] is False
    assert node["steps"] == 20
    assert node["text"] == "a cat"


def test_render_workflow_provided_value_wins(tmp_path):
    mgr = _manager(tmp_path)
    definition = mgr.tool_definitions[0]
    rendered = mgr.render_workflow(
        definition,
        {
            "prompt": "a cat",
            "caption_type": "Art Critic",
            "extra_lighting": False,
            "temperature": 0.3,
        },
    )

    node = rendered["1"]["inputs"]
    assert node["caption_type"] == "Art Critic"
    assert node["lighting"] is False
    assert node["temperature"] == 0.3


def test_no_default_still_uses_safety_fallback(tmp_path):
    workflow = {
        "1": {
            "inputs": {
                "prompt": "PARAM_PROMPT",
                "negative_prompt": "PARAM_NEGATIVE_PROMPT",
                "optional_flag": "PARAM_BOOL_EXTRA_FLAG",
            },
            "class_type": "FakeNode",
        }
    }
    mgr = _manager(tmp_path, "no_defaults", workflow)
    definition = mgr.tool_definitions[0]
    rendered = mgr.render_workflow(definition, {"prompt": "x"})

    node = rendered["1"]["inputs"]
    assert node["negative_prompt"] == ""
    assert node["optional_flag"] is False
