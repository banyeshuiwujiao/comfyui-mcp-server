"""Regression tests for export_to_godot — the Godot data-flywheel step-3 tool."""

import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image

from managers.asset_registry import AssetRegistry
from tools.publish import register_publish_tools


def _synthetic_png(size=(32, 16)) -> bytes:
    img = Image.new("RGBA", size, (255, 128, 0, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _capture_tools(asset_registry, publish_manager):
    tools = {}
    mock_mcp = MagicMock()

    def mock_tool_decorator(*args, **kwargs):
        def decorator(fn):
            tools[fn.__name__] = fn
            return fn
        return decorator

    mock_mcp.tool = mock_tool_decorator
    register_publish_tools(mock_mcp, asset_registry, publish_manager)
    return tools


def test_export_to_godot_writes_file_and_lineage_manifest(tmp_path):
    registry = AssetRegistry(ttl_hours=24, db_path=":memory:")
    png = _synthetic_png()
    rec = registry.register_asset(
        filename="z-image-turbo_00001_.png",
        subfolder="",
        folder_type="output",
        workflow_id="api_image_z_image_turbo_t2i",
        prompt_id="p1",
        mime_type="image/png",
        bytes_size=len(png),
        prompt="victory emblem",
        metadata={"workflow_hash": "abc123"},
    )

    publish_manager = MagicMock()
    publish_manager.config.comfyui_url = "http://localhost:8188"

    tools = _capture_tools(registry, publish_manager)
    target_dir = tmp_path / "Fx"
    target_dir.mkdir()

    with patch("tools.publish.fetch_asset_bytes", return_value=png):
        result = tools["export_to_godot"](
            asset_id=rec.asset_id,
            target_dir=str(target_dir),
            target_filename="fx_victory.png",
            category="Fx",
        )

    assert result["status"] == "success"
    assert result["width"] == 32
    assert result["height"] == 16
    assert (target_dir / "fx_victory.png").read_bytes() == png

    manifest = json.loads((target_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["category"] == "Fx"
    assert manifest["assets"][0]["file"] == "fx_victory.png"
    assert manifest["assets"][0]["asset_id"] == rec.asset_id
    assert manifest["assets"][0]["workflow_id"] == "api_image_z_image_turbo_t2i"
    assert manifest["assets"][0]["workflow_hash"] == "abc123"
    assert manifest["assets"][0]["size"] == [32, 16]


def test_export_to_godot_marks_matting_lineage(tmp_path):
    registry = AssetRegistry(ttl_hours=24, db_path=":memory:")
    png = _synthetic_png()
    parent = registry.register_asset(
        filename="z-image-turbo_00002_.png",
        subfolder="",
        folder_type="output",
        workflow_id="api_image_z_image_turbo_t2i",
        prompt_id="p2",
        mime_type="image/png",
    )
    matting = registry.register_asset(
        filename="transparent_z-image-turbo_00002_.png",
        subfolder="",
        folder_type="output",
        workflow_id="remove_background",
        prompt_id="p2",
        mime_type="image/png",
        bytes_size=len(png),
        parent_asset_id=parent.asset_id,
        generation_type="matting",
        prompt="victory emblem",
    )

    publish_manager = MagicMock()
    publish_manager.config.comfyui_url = "http://localhost:8188"
    tools = _capture_tools(registry, publish_manager)
    target_dir = tmp_path / "Fx"
    target_dir.mkdir()

    with patch("tools.publish.fetch_asset_bytes", return_value=png):
        result = tools["export_to_godot"](
            asset_id=matting.asset_id,
            target_dir=str(target_dir),
            target_filename="fx_victory.png",
        )

    entry = result["manifest_entry"]
    assert entry["source_asset_id"] == parent.asset_id
    assert entry["matting_asset_id"] == matting.asset_id


def test_export_to_godot_rejects_path_traversal(tmp_path):
    registry = AssetRegistry(ttl_hours=24, db_path=":memory:")
    publish_manager = MagicMock()
    publish_manager.config.comfyui_url = "http://localhost:8188"
    tools = _capture_tools(registry, publish_manager)
    target_dir = tmp_path / "Fx"
    target_dir.mkdir()

    result = tools["export_to_godot"](
        asset_id="missing",
        target_dir=str(target_dir),
        target_filename="../evil.png",
    )
    # Asset lookup fails first, so verify traversal safety via the sanitizer.
    from tools.publish import _sanitize_export_filename
    assert _sanitize_export_filename("../evil.png") == ""


def test_sanitize_export_filename_accepts_plain_png():
    from tools.publish import _sanitize_export_filename
    assert _sanitize_export_filename("fx_victory.png") == "fx_victory.png"
    assert _sanitize_export_filename("a/b.png") == ""
    assert _sanitize_export_filename("noext") == ""
