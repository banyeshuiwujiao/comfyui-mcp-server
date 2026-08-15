"""Tests for ErrorDiagnoser and structured self-healing suggestions."""

import pytest
from unittest.mock import MagicMock
from fastmcp import FastMCP

from managers.error_diagnoser import (
    ErrorDiagnoser,
    align_dimension,
    downscale_resolution_for_oom,
    find_closest_model,
)
from tools.job import register_job_tools
from tools.workflow import register_workflow_tools
from tools.generation import register_workflow_generation_tools, register_regenerate_tool
from models.workflow import WorkflowToolDefinition, WorkflowParameter


def test_align_dimension():
    """Test integer dimension alignment to multiples."""
    # Multiples of 8
    assert align_dimension(512, multiple=8) == 512
    assert align_dimension(500, multiple=8) == 504
    assert align_dimension(501, multiple=8) == 504
    assert align_dimension(503, multiple=8) == 504
    assert align_dimension(504, multiple=8) == 504

    # Multiples of 16
    assert align_dimension(1080, multiple=16) == 1088
    assert align_dimension(720, multiple=16) == 720
    assert align_dimension(500, multiple=16) == 496

    # Multiples of 64
    assert align_dimension(1080, multiple=64) == 1088
    assert align_dimension(1024, multiple=64) == 1024

    # Min / Max bounds clamping
    assert align_dimension(0, min_val=64) == 64
    assert align_dimension(-100, min_val=64) == 64
    assert align_dimension(8000, max_val=4096) == 4096


def test_downscale_resolution_for_oom():
    """Test resolution downscaling algorithm for CUDA OOM recovery."""
    # 1024x1024 downscale with factor 0.75 -> 768x768 (aligned to 16)
    w, h = downscale_resolution_for_oom(1024, 1024, scale_factor=0.75, multiple=16)
    assert w == 768
    assert h == 768
    assert w % 16 == 0 and h % 16 == 0

    # 1280x720 landscape downscale
    w, h = downscale_resolution_for_oom(1280, 720, scale_factor=0.75, multiple=16)
    assert w == 960
    assert h == 544
    assert w % 16 == 0 and h % 16 == 0

    # None fallback
    w, h = downscale_resolution_for_oom(None, None)
    assert w == 768
    assert h == 768


def test_find_closest_model():
    """Test fuzzy model matching."""
    available = [
        "flux1-dev-fp8.safetensors",
        "sd_xl_base_1.0.safetensors",
        "v1-5-pruned-emaonly.safetensors",
        "ltx-video-2b-v0.9.1.safetensors"
    ]

    # Exact case-insensitive
    assert find_closest_model("FLUX1-DEV-FP8.SAFETENSORS", available) == "flux1-dev-fp8.safetensors"

    # Substring / partial match
    assert find_closest_model("flux1-dev", available) == "flux1-dev-fp8.safetensors"
    assert find_closest_model("sd_xl_base", available) == "sd_xl_base_1.0.safetensors"
    assert find_closest_model("ltx-video", available) == "ltx-video-2b-v0.9.1.safetensors"

    # Fuzzy match with slight typo
    assert find_closest_model("v1-5-pruned.safetensors", available) == "v1-5-pruned-emaonly.safetensors"

    # Non-existent
    assert find_closest_model("unknown_xyz_model.safetensors", available, cutoff=0.8) is None


def test_oom_diagnosis():
    """Test CUDA OOM error diagnosis and parameter recommendations."""
    diagnoser = ErrorDiagnoser()
    error_msg = "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.40 GiB (GPU 0; 8.00 GiB total capacity)"
    
    result = diagnoser.diagnose(
        error=error_msg,
        params={"width": 1024, "height": 1024, "batch_size": 2, "steps": 50, "frames": 49}
    )

    assert result["status"] == "error"
    assert result["error_code"] == "CUDA_OOM"
    assert result["retryable"] is True
    assert len(result["suggested_actions"]) > 0

    action = result["suggested_actions"][0]
    suggested = action["suggested_params"]
    assert suggested["width"] == 768
    assert suggested["height"] == 768
    assert suggested["batch_size"] == 1
    assert suggested["steps"] == 20
    assert suggested["frames"] == 25


def test_dimension_mismatch_diagnosis():
    """Test dimension divisibility mismatch diagnosis."""
    diagnoser = ErrorDiagnoser()
    error_msg = "RuntimeError: Dimensions must be divisible by 16, but got 500x500"

    result = diagnoser.diagnose(
        error=error_msg,
        params={"width": 500, "height": 500}
    )

    assert result["status"] == "error"
    assert result["error_code"] == "DIMENSION_NOT_DIVISIBLE"
    assert result["retryable"] is True

    action = result["suggested_actions"][0]
    suggested = action["suggested_params"]
    assert suggested["width"] % 16 == 0
    assert suggested["height"] % 16 == 0


def test_model_not_found_diagnosis():
    """Test missing model diagnosis with fuzzy candidate recommendation."""
    available = ["flux1-dev-fp8.safetensors", "sd_xl_base_1.0.safetensors"]
    diagnoser = ErrorDiagnoser()

    error_msg = "Default model 'flux1-dev.safetensors' not found in ComfyUI checkpoints."
    result = diagnoser.diagnose(
        error=error_msg,
        params={"model": "flux1-dev.safetensors"},
        available_models=available
    )

    assert result["status"] == "error"
    assert result["error_code"] == "MODEL_NOT_FOUND"
    assert result["retryable"] is True

    actions = result["suggested_actions"]
    assert any(a["action"] == "switch_model" for a in actions)
    switch_action = next(a for a in actions if a["action"] == "switch_model")
    assert switch_action["suggested_params"]["model"] == "flux1-dev-fp8.safetensors"


def test_input_file_not_found_diagnosis():
    """Test missing input file error diagnosis."""
    diagnoser = ErrorDiagnoser()
    error_msg = "Input image 'character_pose.png' not found in ComfyUI input directory."

    result = diagnoser.diagnose(
        error=error_msg,
        params={"image": "character_pose.png"}
    )

    assert result["status"] == "error"
    assert result["error_code"] == "INPUT_FILE_NOT_FOUND"
    assert result["retryable"] is False
    assert any(a["action"] == "provide_input_file" for a in result["suggested_actions"])


def test_param_out_of_bounds_diagnosis():
    """Test parameter out of bounds diagnosis with clamping."""
    diagnoser = ErrorDiagnoser()
    error_msg = "ValueError: denoise must be between 0.0 and 1.0, got 2.5"

    result = diagnoser.diagnose(
        error=error_msg,
        params={"denoise": 2.5, "cfg": -2.0, "steps": 0}
    )

    assert result["status"] == "error"
    assert result["error_code"] == "PARAM_OUT_OF_BOUNDS"
    assert result["retryable"] is True

    action = result["suggested_actions"][0]
    suggested = action["suggested_params"]
    assert suggested["denoise"] == 1.0
    assert suggested["cfg"] == 1.0
    assert suggested["steps"] == 1


def test_node_execution_error_diagnosis():
    """Test node-level execution error parsing."""
    diagnoser = ErrorDiagnoser()
    error_msg = "Node 15 (KSampler): [RuntimeError] CUDA error: an illegal memory access was encountered"

    node_data = {
        "node_id": "15",
        "node_type": "KSampler",
        "exception_type": "RuntimeError",
        "exception_message": "CUDA error: an illegal memory access was encountered"
    }

    result = diagnoser.diagnose(
        error=error_msg,
        node_error_data=node_data
    )

    assert result["status"] == "error"
    assert result["error_code"] == "NODE_EXECUTION_ERROR"
    assert result["node_id"] == "15"
    assert result["node_type"] == "KSampler"


def test_gpu_guard_and_timeout_diagnosis():
    """Test GPU saturation and timeout error classification."""
    diagnoser = ErrorDiagnoser()

    # GPU saturated
    res_sat = diagnoser.diagnose(error="GPU saturated: utilization at 98% (admission denied)")
    assert res_sat["error_code"] == "GPU_SATURATED"
    assert res_sat["retryable"] is True

    # Timeout
    res_to = diagnoser.diagnose(error="Workflow still running after 360s (timeout)")
    assert res_to["error_code"] == "TIMEOUT"
    assert res_to["retryable"] is True


def test_job_tool_diagnosed_error():
    """Test get_job integration with structured error diagnosis."""
    mock_client = MagicMock()
    mock_client.get_queue.return_value = {"queue_running": [], "queue_pending": []}
    mock_client.get_history.return_value = {
        "prompt_123": {
            "status": {
                "status_str": "error",
                "completed": False,
                "messages": [
                    [
                        "execution_error",
                        {
                            "node_id": "8",
                            "node_type": "VAEDecode",
                            "exception_type": "OutOfMemoryError",
                            "exception_message": "CUDA out of memory"
                        }
                    ]
                ]
            }
        }
    }
    mock_client._has_status_message = MagicMock(return_value=True)
    mock_client._extract_node_error_dict = MagicMock(return_value={
        "node_id": "8",
        "node_type": "VAEDecode",
        "exception_type": "OutOfMemoryError",
        "exception_message": "CUDA out of memory"
    })
    mock_client._extract_node_errors = MagicMock(return_value="Node 8 (VAEDecode): [OutOfMemoryError] CUDA out of memory")

    captured_tools = {}
    mock_mcp = MagicMock()
    def mock_tool_decorator():
        def decorator(fn):
            captured_tools[fn.__name__] = fn
            return fn
        return decorator
    mock_mcp.tool = mock_tool_decorator

    registry = MagicMock()
    diagnoser = ErrorDiagnoser(mock_client)
    register_job_tools(mock_mcp, mock_client, registry, diagnoser)

    assert "get_job" in captured_tools
    res = captured_tools["get_job"](prompt_id="prompt_123")

    assert res["status"] == "error"
    assert res["error_code"] == "CUDA_OOM"
    assert res["node_id"] == "8"
    assert res["node_type"] == "VAEDecode"
    assert res["retryable"] is True


def test_workflow_tool_diagnosed_error():
    """Test run_workflow integration with structured error diagnosis."""
    mock_workflow_manager = MagicMock()
    mock_workflow_manager.load_workflow.return_value = {"1": {"class_type": "KSampler"}}
    mock_workflow_manager._extract_parameters.return_value = {}
    mock_workflow_manager.apply_workflow_overrides.return_value = {"1": {"class_type": "KSampler"}}
    mock_workflow_manager._guess_output_preferences.return_value = ("images",)

    mock_client = MagicMock()
    mock_client.available_models = ["flux1-dev-fp8.safetensors"]
    mock_client.list_input_files.return_value = []
    mock_client.run_custom_workflow.side_effect = Exception("CUDA out of memory in EmptyLatentImage")

    captured_tools = {}
    mock_mcp = MagicMock()
    def mock_tool_decorator():
        def decorator(fn):
            captured_tools[fn.__name__] = fn
            return fn
        return decorator
    mock_mcp.tool = mock_tool_decorator

    registry = MagicMock()
    defaults = MagicMock()
    diagnoser = ErrorDiagnoser(mock_client, defaults)

    register_workflow_tools(mock_mcp, mock_workflow_manager, mock_client, defaults, registry, error_diagnoser=diagnoser)

    assert "run_workflow" in captured_tools
    res = captured_tools["run_workflow"](workflow_id="test_wf", overrides={"width": 1024, "height": 1024})

    assert res["status"] == "error"
    assert res["error_code"] == "CUDA_OOM"
    assert res["retryable"] is True
    assert res["suggested_actions"][0]["suggested_params"]["width"] == 768


def test_regenerate_tool_diagnosed_error():
    """Test regenerate integration with structured error diagnosis."""
    mock_client = MagicMock()
    mock_client.run_custom_workflow.side_effect = Exception("RuntimeError: Dimensions must be divisible by 8, got 505x505")

    captured_tools = {}
    mock_mcp = MagicMock()
    def mock_tool_decorator():
        def decorator(fn):
            captured_tools[fn.__name__] = fn
            return fn
        return decorator
    mock_mcp.tool = mock_tool_decorator

    registry = MagicMock()
    mock_asset = MagicMock()
    mock_asset.submitted_workflow = {"1": {"class_type": "KSampler"}}
    mock_asset.workflow_id = "generate_image"
    mock_asset.session_id = "session_1"
    registry.get_asset.return_value = mock_asset

    diagnoser = ErrorDiagnoser(mock_client)
    register_regenerate_tool(mock_mcp, mock_client, registry, error_diagnoser=diagnoser)

    assert "regenerate" in captured_tools
    res = captured_tools["regenerate"](asset_id="asset_123", param_overrides={"width": 505, "height": 505})

    assert res["status"] == "error"
    assert res["error_code"] == "DIMENSION_NOT_DIVISIBLE"
    assert res["retryable"] is True
    assert res["suggested_actions"][0]["suggested_params"]["width"] % 8 == 0

