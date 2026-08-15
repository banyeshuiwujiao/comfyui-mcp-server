"""Unit tests for Modular Subgraph Pipeline Orchestrator (run_pipeline, list_pipeline_recipes)."""

import io
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image, ImageDraw

from managers.asset_registry import AssetRegistry
from managers.character_vault import CharacterVault
from managers.pipeline_orchestrator import PipelineOrchestrator, PIPELINE_RECIPES
from tools.pipeline import register_pipeline_tools


def create_synthetic_png() -> bytes:
    img = Image.new("RGB", (64, 64), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.ellipse([16, 16, 48, 48], fill=(200, 50, 50))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def asset_registry():
    return AssetRegistry(ttl_hours=24, db_path=":memory:")


@pytest.fixture
def character_vault():
    vault = CharacterVault(db_path=":memory:")
    vault.save_profile(
        character_id="detective_john",
        display_name="Detective John",
        trigger_words="1man, trenchcoat, cybernetic eye",
        style_preset="cyberpunk",
    )
    return vault


@pytest.fixture
def mock_comfyui_client():
    client = MagicMock()
    client.base_url = "http://localhost:8188"
    client.run_custom_workflow.return_value = {
        "filename": "step0_output.png",
        "subfolder": "",
        "folder_type": "output",
        "prompt_id": "prompt_001",
        "asset_metadata": {"mime_type": "image/png", "width": 512, "height": 512, "bytes_size": 1024},
    }
    return client


@pytest.fixture
def mock_workflow_manager():
    wm = MagicMock()
    wm.tool_definitions = {
        "api_image_flux2_text_to_image_9b": MagicMock(),
        "api_video_minimax_h3_i2v": MagicMock(),
    }
    def mock_get_workflow(wf_id):
        if wf_id in wm.tool_definitions or wf_id in ("generate_image", "generate_song"):
            return {"3": {"class_type": "KSampler"}}
        return None

    wm.get_workflow = mock_get_workflow
    wm.apply_workflow_overrides.return_value = {"3": {"class_type": "KSampler"}}
    return wm


@pytest.fixture
def orchestrator(mock_comfyui_client, asset_registry, mock_workflow_manager, character_vault):
    return PipelineOrchestrator(
        comfyui_client=mock_comfyui_client,
        asset_registry=asset_registry,
        workflow_manager=mock_workflow_manager,
        character_vault=character_vault,
    )


# ---------------------------------------------------------------------------
# Pipeline Orchestrator Tests
# ---------------------------------------------------------------------------

class TestPipelineOrchestrator:

    def test_list_recipes(self, orchestrator):
        recipes = orchestrator.list_recipes()
        assert len(recipes) >= 4
        recipe_ids = [r["recipe_id"] for r in recipes]
        assert "t2i_to_2k_upscale" in recipe_ids
        assert "t2i_to_transparent_sticker" in recipe_ids
        assert "character_to_sprite_sheet" in recipe_ids

    def test_execute_empty_pipeline(self, orchestrator):
        res = orchestrator.execute_pipeline(steps=[])
        assert "error" in res

    def test_execute_two_step_pipeline_t2i_and_matting(self, orchestrator, asset_registry):
        raw_png = create_synthetic_png()

        with patch("asset_processor.fetch_asset_bytes", return_value=raw_png):
            steps = [
                {
                    "step_name": "t2i_gen",
                    "tool": "api_image_flux2_text_to_image_9b",
                    "params": {"prompt": "a red robot standing"},
                },
                {
                    "step_name": "bg_matting",
                    "tool": "remove_background",
                    "input_from": "previous",
                    "params": {"mode": "auto"},
                },
            ]

            result = orchestrator.execute_pipeline(steps=steps, pipeline_name="test_robot_pipeline")

            assert result["status"] == "success"
            assert result["total_steps"] == 2
            assert len(result["completed_steps"]) == 2

            step0 = result["completed_steps"][0]
            step1 = result["completed_steps"][1]

            assert step0["tool"] == "api_image_flux2_text_to_image_9b"
            assert step1["tool"] == "remove_background"

            # Check lineage
            rec_step0 = asset_registry.get_asset(step0["asset_id"])
            rec_step1 = asset_registry.get_asset(step1["asset_id"])

            assert rec_step0 is not None
            assert rec_step1 is not None
            assert rec_step1.parent_asset_id == rec_step0.asset_id
            assert rec_step1.generation_type == "matting"

            # Check final asset
            assert result["final_asset"]["asset_id"] == rec_step1.asset_id

    def test_execute_pipeline_with_character_profile(self, orchestrator, mock_workflow_manager):
        steps = [
            {
                "tool": "api_image_flux2_text_to_image_9b",
                "character_id": "detective_john",
                "params": {"prompt": "walking down the alley"},
            }
        ]

        result = orchestrator.execute_pipeline(steps=steps)
        assert result["status"] == "success"

        # Verify prompt injection
        call_args = mock_workflow_manager.apply_workflow_overrides.call_args
        overrides = call_args[0][1]
        assert "1man, trenchcoat, cybernetic eye" in overrides["prompt"]
        assert "walking down the alley" in overrides["prompt"]
        assert "cyberpunk aesthetic" in overrides["prompt"]

    def test_execute_pipeline_step_failure(self, orchestrator):
        # Step 1 succeeds, step 2 references missing tool
        steps = [
            {
                "tool": "api_image_flux2_text_to_image_9b",
                "params": {"prompt": "test"},
            },
            {
                "tool": "nonexistent_custom_tool",
                "input_from": "previous",
                "params": {},
            },
        ]

        result = orchestrator.execute_pipeline(steps=steps)
        assert result["status"] == "failed"
        assert result["failing_step_index"] == 1
        assert len(result["completed_steps"]) == 1


# ---------------------------------------------------------------------------
# MCP Tool Tests
# ---------------------------------------------------------------------------

class TestPipelineMCPTools:

    @pytest.fixture
    def captured_tools(self, orchestrator, asset_registry, mock_comfyui_client):
        tools = {}
        mock_mcp = MagicMock()

        def mock_tool_decorator():
            def decorator(fn):
                tools[fn.__name__] = fn
                return fn
            return decorator

        mock_mcp.tool = mock_tool_decorator
        register_pipeline_tools(mock_mcp, asset_registry, mock_comfyui_client, orchestrator)
        return tools

    def test_list_pipeline_recipes_tool(self, captured_tools):
        assert "list_pipeline_recipes" in captured_tools
        res = captured_tools["list_pipeline_recipes"]()
        assert res["count"] >= 4
        assert len(res["recipes"]) >= 4

    def test_run_pipeline_tool(self, captured_tools):
        assert "run_pipeline" in captured_tools
        raw_png = create_synthetic_png()

        with patch("asset_processor.fetch_asset_bytes", return_value=raw_png):
            res = captured_tools["run_pipeline"](
                steps=[
                    {"tool": "api_image_flux2_text_to_image_9b", "params": {"prompt": "hero"}},
                    {"tool": "remove_background", "input_from": "previous", "params": {"mode": "auto"}},
                ],
                pipeline_name="hero_cutout",
            )
            assert res["status"] == "success"
            assert res["total_steps"] == 2
