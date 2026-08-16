"""Unit tests for MCP Native Resources & Prompts."""

import json
import pytest
from unittest.mock import MagicMock, patch

from managers.asset_registry import AssetRegistry
from managers.character_vault import CharacterVault
from tools.mcp_resources import register_mcp_resources


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class MockWorkflowParam:
    def __init__(self, name, type_hint=str, required=True, description=""):
        self.name = name
        self.type_hint = type_hint
        self.annotation = type_hint  # mirrors models.workflow.WorkflowParameter
        self.required = required
        self.description = description


class MockToolDef:
    def __init__(self, workflow_id="", description="", parameters=None):
        self.workflow_id = workflow_id
        self.description = description
        # mirrors WorkflowToolDefinition: OrderedDict[name, WorkflowParameter]
        self.parameters = {p.name: p for p in (parameters or [])}


@pytest.fixture
def mock_comfyui_client():
    client = MagicMock()
    client.base_url = "http://localhost:8188"
    client.available_models = [
        "v1-5-pruned-emaonly.ckpt",
        "sd_xl_base_1.0.safetensors",
        "flux2_klein_9b.safetensors",
    ]
    client.MODEL_LOADER_TYPES = ("CheckpointLoaderSimple", "UNETLoader", "LoraLoader")
    client._fetch_loader_model_names = MagicMock(side_effect=lambda t: {
        "CheckpointLoaderSimple": ["v1-5-pruned-emaonly.ckpt", "sd_xl_base_1.0.safetensors"],
        "UNETLoader": ["flux2_klein_9b.safetensors"],
        "LoraLoader": ["style_watercolor.safetensors"],
    }.get(t, []))
    return client


@pytest.fixture
def mock_workflow_manager():
    wm = MagicMock()
    wm.tool_definitions = [
        MockToolDef(
            workflow_id="api_image_flux2_text_to_image_9b",
            description="Flux2 Klein 9B text-to-image",
            parameters=[
                MockWorkflowParam("prompt", str, True, "Main text prompt"),
                MockWorkflowParam("seed", int, False, "Random seed"),
            ],
        ),
        MockToolDef(
            workflow_id="api_video_minimax_h3_t2v",
            description="MiniMax H3 text-to-video",
            parameters=[
                MockWorkflowParam("prompt", str, True, "Video prompt"),
            ],
        ),
        MockToolDef(
            workflow_id="api_audio_ace_step1_5_xl_sft",
            description="AceStep 1.5 XL audio generation",
            parameters=[
                MockWorkflowParam("tags", str, True, "Music tags"),
                MockWorkflowParam("lyrics", str, True, "Song lyrics"),
            ],
        ),
    ]
    return wm


@pytest.fixture
def asset_registry():
    return AssetRegistry(ttl_hours=24, db_path=":memory:")


@pytest.fixture
def character_vault():
    return CharacterVault(db_path=":memory:")


@pytest.fixture
def mock_gpu_guard():
    return MagicMock()


@pytest.fixture
def captured_items(mock_comfyui_client, asset_registry, mock_workflow_manager, mock_gpu_guard, character_vault):
    """Register resources/prompts and capture them via mock."""
    resources = {}
    prompts = {}
    mock_mcp = MagicMock()

    def mock_resource_decorator(uri, **kwargs):
        def decorator(fn):
            resources[uri] = fn
            return fn
        return decorator

    def mock_prompt_decorator(name, **kwargs):
        def decorator(fn):
            prompts[name] = fn
            return fn
        return decorator

    mock_mcp.resource = mock_resource_decorator
    mock_mcp.prompt = mock_prompt_decorator

    register_mcp_resources(
        mock_mcp, mock_comfyui_client, asset_registry,
        mock_workflow_manager, mock_gpu_guard,
        character_vault=character_vault,
    )
    return resources, prompts, asset_registry, character_vault


# ---------------------------------------------------------------------------
# Resource Tests
# ---------------------------------------------------------------------------

class TestResources:

    def test_gpu_health_resource(self, captured_items):
        resources, _, _, _ = captured_items

        assert "comfyui://system/gpu-health" in resources

        with patch("tools.mcp_resources._fetch_system_stats") as mock_stats, \
             patch("tools.mcp_resources._fetch_queue_info") as mock_queue:
            mock_stats.return_value = {
                "devices": [
                    {
                        "name": "NVIDIA RTX 4070 Ti SUPER",
                        "type": "cuda",
                        "vram_total": 16 * (1024 ** 3),
                        "vram_free": 10 * (1024 ** 3),
                    }
                ]
            }
            mock_queue.return_value = {
                "queue_running": [],
                "queue_pending": [],
            }

            result = json.loads(resources["comfyui://system/gpu-health"]())
            assert result["status"] == "healthy"
            assert len(result["devices"]) == 1
            assert result["devices"][0]["name"] == "NVIDIA RTX 4070 Ti SUPER"
            assert result["devices"][0]["vram_total_gb"] == 16.0
            assert result["devices"][0]["vram_free_gb"] == 10.0
            assert result["queue_running_count"] == 0
            assert result["queue_pending_count"] == 0

    def test_gpu_health_saturated(self, captured_items):
        resources, _, _, _ = captured_items

        with patch("tools.mcp_resources._fetch_system_stats") as mock_stats, \
             patch("tools.mcp_resources._fetch_queue_info") as mock_queue:
            mock_stats.return_value = {
                "devices": [
                    {
                        "name": "GPU",
                        "type": "cuda",
                        "vram_total": 16 * (1024 ** 3),
                        "vram_free": 1 * (1024 ** 3),  # 93.75% used
                    }
                ]
            }
            mock_queue.return_value = {
                "queue_running": [{"id": "1"}],
                "queue_pending": [{"id": "2"}, {"id": "3"}],
            }

            result = json.loads(resources["comfyui://system/gpu-health"]())
            assert result["status"] == "saturated"
            assert result["queue_running_count"] == 1
            assert result["queue_pending_count"] == 2
            assert "saturated" in result["recommendation"].lower()

    def test_checkpoints_resource(self, captured_items):
        resources, _, _, _ = captured_items

        assert "comfyui://models/checkpoints" in resources
        result = json.loads(resources["comfyui://models/checkpoints"]())

        assert result["total_unique_count"] == 3
        assert "v1-5-pruned-emaonly.ckpt" in result["all_models"]
        assert "by_loader_type" in result

    def test_loras_resource(self, captured_items):
        resources, _, _, _ = captured_items

        assert "comfyui://models/loras" in resources

        with patch("tools.mcp_resources._fetch_lora_names") as mock_loras:
            mock_loras.return_value = ["style_watercolor.safetensors", "face_id_v1.safetensors"]
            result = json.loads(resources["comfyui://models/loras"]())
            assert result["count"] == 2
            assert "style_watercolor.safetensors" in result["loras"]

    def test_workflows_resource(self, captured_items):
        resources, _, _, _ = captured_items

        assert "comfyui://workflows" in resources
        result = json.loads(resources["comfyui://workflows"]())

        assert result["count"] == 3
        wf_ids = [w["workflow_id"] for w in result["workflows"]]
        assert "api_image_flux2_text_to_image_9b" in wf_ids
        assert "api_video_minimax_h3_t2v" in wf_ids
        assert "api_audio_ace_step1_5_xl_sft" in wf_ids

        # Check media type inference
        for w in result["workflows"]:
            if "video" in w["workflow_id"]:
                assert w["media_type"] == "video"
            elif "audio" in w["workflow_id"]:
                assert w["media_type"] == "audio"
            else:
                assert w["media_type"] == "image"

        # Check parameters are exposed
        flux_wf = next(w for w in result["workflows"] if "flux2" in w["workflow_id"])
        assert flux_wf["parameter_count"] == 2
        param_names = [p["name"] for p in flux_wf["parameters"]]
        assert "prompt" in param_names
        assert "seed" in param_names

    def test_asset_detail_resource(self, captured_items):
        resources, _, registry, _ = captured_items

        assert "comfyui://assets/{asset_id}" in resources

        rec = registry.register_asset(
            filename="test_img.png",
            subfolder="",
            folder_type="output",
            workflow_id="generate_image",
            prompt_id="prompt-001",
            mime_type="image/png",
            bytes_size=12345,
            prompt="a cyberpunk city",
            seed=42,
        )

        result = json.loads(resources["comfyui://assets/{asset_id}"](rec.asset_id))
        assert result["asset_id"] == rec.asset_id
        assert result["filename"] == "test_img.png"
        assert result["workflow_id"] == "generate_image"
        assert result["prompt"] == "a cyberpunk city"
        assert result["seed"] == 42

    def test_asset_detail_not_found(self, captured_items):
        resources, _, _, _ = captured_items

        result = json.loads(resources["comfyui://assets/{asset_id}"]("nonexistent-id"))
        assert "error" in result
        assert "not found" in result["error"]

    def test_asset_lineage_resource(self, captured_items):
        resources, _, registry, _ = captured_items

        assert "comfyui://assets/{asset_id}/lineage" in resources

        parent = registry.register_asset(
            filename="parent.png", subfolder="", folder_type="output",
            workflow_id="generate_image", prompt_id="p1",
            mime_type="image/png", bytes_size=1000,
        )
        child = registry.register_asset(
            filename="child.png", subfolder="", folder_type="output",
            workflow_id="generate_image", prompt_id="p2",
            mime_type="image/png", bytes_size=2000,
            parent_asset_id=parent.asset_id,
        )

        result = json.loads(resources["comfyui://assets/{asset_id}/lineage"](child.asset_id))
        assert result["asset_id"] == child.asset_id
        assert result["parent_asset_id"] == parent.asset_id

    def test_asset_lineage_not_found(self, captured_items):
        resources, _, _, _ = captured_items

        result = json.loads(resources["comfyui://assets/{asset_id}/lineage"]("nonexistent"))
        assert "error" in result

    def test_characters_resource(self, captured_items):
        resources, _, _, vault = captured_items

        assert "comfyui://characters" in resources

        vault.save_profile(
            character_id="detective_john",
            display_name="Detective John",
            trigger_words="1man, trenchcoat",
            tags=["protagonist"],
        )

        result = json.loads(resources["comfyui://characters"]())
        assert result["count"] == 1
        assert result["characters"][0]["character_id"] == "detective_john"

    def test_character_detail_resource(self, captured_items):
        resources, _, _, vault = captured_items

        assert "comfyui://characters/{character_id}" in resources

        vault.save_profile(
            character_id="detective_john",
            display_name="Detective John",
            trigger_words="1man, trenchcoat",
        )

        result = json.loads(resources["comfyui://characters/{character_id}"]("detective_john"))
        assert result["character_id"] == "detective_john"
        assert result["trigger_words"] == "1man, trenchcoat"

    def test_character_detail_resource_not_found(self, captured_items):
        resources, _, _, _ = captured_items

        result = json.loads(resources["comfyui://characters/{character_id}"]("nonexistent"))
        assert "error" in result


# ---------------------------------------------------------------------------
# Prompt Tests
# ---------------------------------------------------------------------------

class TestPrompts:

    def test_flux_photo_prompt(self, captured_items):
        _, prompts, _, _ = captured_items

        assert "flux_photo_prompt" in prompts
        result = prompts["flux_photo_prompt"](subject="a cyberpunk detective in a neon-lit alley")
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        content = result[0]["content"]
        assert "cyberpunk detective" in content
        assert "FLUX" in content
        assert "Steps" in content

    def test_flux_photo_prompt_custom_params(self, captured_items):
        _, prompts, _, _ = captured_items

        result = prompts["flux_photo_prompt"](
            subject="a white cat",
            style="oil painting",
            lighting="dramatic chiaroscuro",
            camera="wide-angle 24mm lens",
            mood="mysterious and ethereal",
        )
        content = result[0]["content"]
        assert "white cat" in content
        assert "oil painting" in content
        assert "mysterious" in content

    def test_cinematic_video_prompt(self, captured_items):
        _, prompts, _, _ = captured_items

        assert "cinematic_video_prompt" in prompts
        result = prompts["cinematic_video_prompt"](
            scene="A lone astronaut walks across a red Martian desert",
            motion="slow walking with dust particles",
            camera_movement="tracking shot from behind",
        )
        content = result[0]["content"]
        assert "astronaut" in content
        assert "tracking shot" in content
        assert "minimax" in content.lower() or "MiniMax" in content

    def test_cinematic_video_prompt_picture_note(self, captured_items):
        _, prompts, _, _ = captured_items

        # Without <Picture> → should contain the MiniMax H3 note
        result = prompts["cinematic_video_prompt"](scene="A sunset over the ocean")
        content = result[0]["content"]
        assert "<Picture 1>" in content

    def test_character_sheet_prompt(self, captured_items):
        _, prompts, _, _ = captured_items

        assert "character_sheet_prompt" in prompts
        result = prompts["character_sheet_prompt"](
            character_name="Detective John",
            appearance="tall, dark hair, sharp jawline, cybernetic left eye",
            outfit="long black trenchcoat, neon-blue tie",
        )
        content = result[0]["content"]
        assert "Detective John" in content
        assert "cybernetic left eye" in content
        assert "trenchcoat" in content
        assert "qwen" in content.lower() or "2511" in content

    def test_music_generation_prompt(self, captured_items):
        _, prompts, _, _ = captured_items

        assert "music_generation_prompt" in prompts
        result = prompts["music_generation_prompt"](
            genre="synthwave",
            mood="nostalgic, dreamy",
            instruments="analog synth, drum machine, bass guitar",
            tempo="100 BPM",
            theme="midnight drive through a neon city",
        )
        content = result[0]["content"]
        assert "synthwave" in content
        assert "[Verse 1]" in content
        assert "[Chorus]" in content
        assert "analyze_audio" in content

    def test_all_expected_resources_registered(self, captured_items):
        resources, _, _, _ = captured_items

        expected_uris = [
            "comfyui://system/gpu-health",
            "comfyui://models/checkpoints",
            "comfyui://models/loras",
            "comfyui://workflows",
            "comfyui://assets/{asset_id}",
            "comfyui://assets/{asset_id}/lineage",
            "comfyui://characters",
            "comfyui://characters/{character_id}",
        ]
        for uri in expected_uris:
            assert uri in resources, f"Missing resource: {uri}"

    def test_all_expected_prompts_registered(self, captured_items):
        _, prompts, _, _ = captured_items

        expected = [
            "flux_photo_prompt",
            "cinematic_video_prompt",
            "character_sheet_prompt",
            "music_generation_prompt",
        ]
        for name in expected:
            assert name in prompts, f"Missing prompt: {name}"

    def test_gpu_health_busy_status(self, captured_items):
        resources, _, _, _ = captured_items

        with patch("tools.mcp_resources._fetch_system_stats") as mock_stats, \
             patch("tools.mcp_resources._fetch_queue_info") as mock_queue:
            mock_stats.return_value = {
                "devices": [
                    {
                        "name": "GPU",
                        "type": "cuda",
                        "vram_total": 16 * (1024 ** 3),
                        "vram_free": 4 * (1024 ** 3),  # 75% used -> busy
                    }
                ]
            }
            mock_queue.return_value = {
                "queue_running": [],
                "queue_pending": [],
            }

            result = json.loads(resources["comfyui://system/gpu-health"]())
            assert result["status"] == "busy"
            assert "moderately" in result["recommendation"]

    def test_loras_resource_empty(self, captured_items):
        resources, _, _, _ = captured_items

        with patch("tools.mcp_resources._fetch_lora_names") as mock_loras:
            mock_loras.return_value = []
            result = json.loads(resources["comfyui://models/loras"]())
            assert result["count"] == 0
            assert result["loras"] == []

    def test_characters_resource_empty(self, captured_items):
        resources, _, _, vault = captured_items
        # Vault has no characters
        result = json.loads(resources["comfyui://characters"]())
        assert result["count"] == 0
        assert result["characters"] == []

    def test_helpers_network_failures(self):
        from tools.mcp_resources import _fetch_system_stats, _fetch_queue_info, _fetch_lora_names
        with patch("requests.get", side_effect=Exception("Connection refused")):
            assert _fetch_system_stats("http://invalid") == {}
            assert _fetch_queue_info("http://invalid") == {}
            assert _fetch_lora_names("http://invalid") == []
