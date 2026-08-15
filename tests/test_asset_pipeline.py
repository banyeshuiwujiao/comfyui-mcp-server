"""Unit tests for Game & Web Asset Post-Processing Pipeline (remove_background & generate_sprite_sheet)."""

import io
import json
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from PIL import Image, ImageDraw

from asset_processor import (
    build_sprite_sheet,
    remove_image_background,
)
from managers.asset_registry import AssetRegistry
from tools.pipeline import register_pipeline_tools


# ---------------------------------------------------------------------------
# Test Image Helpers
# ---------------------------------------------------------------------------

def create_synthetic_object_image(bg_color=(255, 255, 255), fg_color=(220, 50, 50), size=(128, 128)) -> bytes:
    """Create a synthetic test image with a centered shape on a solid background."""
    img = Image.new("RGB", size, bg_color)
    draw = ImageDraw.Draw(img)
    # Draw a colored circle in the center
    margin = size[0] // 4
    draw.ellipse([margin, margin, size[0] - margin, size[1] - margin], fill=fg_color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def create_synthetic_frames(count=6, size=(64, 64)) -> list:
    """Create a list of synthetic animation frames."""
    frames = []
    for i in range(count):
        img = Image.new("RGBA", size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        # Draw moving circle
        cx = 10 + (i * 8)
        cy = size[1] // 2
        draw.ellipse([cx - 8, cy - 8, cx + 8, cy + 8], fill=(50, 150, 250, 255))
        frames.append(img)
    return frames


# ---------------------------------------------------------------------------
# Processor Function Tests
# ---------------------------------------------------------------------------

class TestAssetProcessorPipeline:

    def test_remove_image_background_color_auto(self):
        """Test auto-detecting and removing white background."""
        raw_png = create_synthetic_object_image(bg_color=(255, 255, 255), fg_color=(200, 30, 30), size=(100, 100))
        trans_bytes = remove_image_background(raw_png, mode="auto")
        
        assert len(trans_bytes) > 0
        with Image.open(io.BytesIO(trans_bytes)) as out_img:
            assert out_img.mode == "RGBA"
            assert out_img.size == (100, 100)
            
            # Corner should be transparent
            corner_alpha = out_img.getpixel((2, 2))[3]
            assert corner_alpha < 30
            
            # Center should be opaque
            center_alpha = out_img.getpixel((50, 50))[3]
            assert center_alpha > 200

    def test_remove_image_background_custom_color(self):
        """Test removing explicit green-screen background."""
        raw_png = create_synthetic_object_image(bg_color=(0, 255, 0), fg_color=(50, 50, 200), size=(80, 80))
        trans_bytes = remove_image_background(raw_png, mode="color", bgcolor="green", tolerance=25)
        
        with Image.open(io.BytesIO(trans_bytes)) as out_img:
            assert out_img.mode == "RGBA"
            corner_alpha = out_img.getpixel((2, 2))[3]
            assert corner_alpha < 30
            center_alpha = out_img.getpixel((40, 40))[3]
            assert center_alpha > 200

    def test_remove_image_background_grabcut(self):
        """Test GrabCut segmentation on a synthetic subject."""
        raw_png = create_synthetic_object_image(bg_color=(30, 30, 30), fg_color=(255, 200, 0), size=(90, 90))
        trans_bytes = remove_image_background(raw_png, mode="grabcut", feather=1)
        
        with Image.open(io.BytesIO(trans_bytes)) as out_img:
            assert out_img.mode == "RGBA"
            center_alpha = out_img.getpixel((45, 45))[3]
            assert center_alpha > 180

    def test_build_sprite_sheet(self):
        """Test assembling animation frames into a sprite sheet texture atlas."""
        frames = create_synthetic_frames(count=6, size=(64, 64))
        atlas_bytes, meta = build_sprite_sheet(frames, columns=3, padding=2, out_format="PNG")

        assert len(atlas_bytes) > 0
        assert meta["meta"]["frame_count"] == 6
        assert meta["meta"]["columns"] == 3
        assert meta["meta"]["rows"] == 2
        assert len(meta["frame_list"]) == 6

        # Check total atlas dimensions
        expected_w = 2 + 3 * (64 + 2)
        expected_h = 2 + 2 * (64 + 2)
        assert meta["meta"]["size"]["w"] == expected_w
        assert meta["meta"]["size"]["h"] == expected_h

        with Image.open(io.BytesIO(atlas_bytes)) as atlas_img:
            assert atlas_img.mode == "RGBA"
            assert atlas_img.size == (expected_w, expected_h)

    def test_build_sprite_sheet_with_scaling(self):
        """Test assembling sprite sheet with target frame resizing."""
        frames = create_synthetic_frames(count=4, size=(100, 100))
        atlas_bytes, meta = build_sprite_sheet(frames, columns=2, frame_width=48, frame_height=48, padding=0)

        assert meta["meta"]["size"]["w"] == 96
        assert meta["meta"]["size"]["h"] == 96
        assert meta["meta"]["frame_width"] == 48
        assert meta["meta"]["frame_height"] == 48


# ---------------------------------------------------------------------------
# MCP Tool Tests
# ---------------------------------------------------------------------------

class TestAssetPipelineTools:

    @pytest.fixture
    def registry(self):
        return AssetRegistry(ttl_hours=24, db_path=":memory:")

    @pytest.fixture
    def mock_client(self):
        client = MagicMock()
        client.base_url = "http://localhost:8188"
        return client

    @pytest.fixture
    def captured_pipeline_tools(self, registry, mock_client):
        tools = {}
        mock_mcp = MagicMock()

        def mock_tool_decorator():
            def decorator(fn):
                tools[fn.__name__] = fn
                return fn
            return decorator

        mock_mcp.tool = mock_tool_decorator
        register_pipeline_tools(mock_mcp, registry, mock_client)
        return tools, registry

    def test_remove_background_tool(self, captured_pipeline_tools):
        tools, registry = captured_pipeline_tools
        raw_png = create_synthetic_object_image(bg_color=(255, 255, 255), fg_color=(220, 40, 40))

        # Register parent asset
        parent = registry.register_asset(
            filename="character_stand.png",
            subfolder="",
            folder_type="output",
            workflow_id="generate_image",
            prompt_id="p001",
            mime_type="image/png",
            bytes_size=len(raw_png),
            prompt="a superhero standing",
        )

        with patch("tools.pipeline.fetch_asset_bytes", return_value=raw_png):
            res = tools["remove_background"](asset_id=parent.asset_id, mode="auto")

            assert res["status"] == "success"
            assert "transparent_character_stand" in res["filename"]
            assert res["parent_asset_id"] == parent.asset_id
            assert res["generation_type"] == "matting"

            # Check registered in registry
            new_asset = registry.get_asset(res["asset_id"])
            assert new_asset is not None
            assert new_asset.parent_asset_id == parent.asset_id
            assert "transparent" in new_asset.tags

    def test_remove_background_not_found(self, captured_pipeline_tools):
        tools, _ = captured_pipeline_tools
        res = tools["remove_background"](asset_id="nonexistent-id")
        assert "error" in res

    def test_generate_sprite_sheet_tool(self, captured_pipeline_tools):
        tools, registry = captured_pipeline_tools
        raw_png = create_synthetic_object_image(size=(64, 64))

        parent = registry.register_asset(
            filename="char_action.png",
            subfolder="",
            folder_type="output",
            workflow_id="generate_image",
            prompt_id="p002",
            mime_type="image/png",
            bytes_size=len(raw_png),
        )

        with patch("tools.pipeline.fetch_asset_bytes", return_value=raw_png):
            res = tools["generate_sprite_sheet"](
                asset_id=parent.asset_id,
                frame_count=4,
                columns=2,
                frame_width=32,
                frame_height=32,
            )

            assert res["status"] == "success"
            assert "spritesheet_char_action" in res["filename"]
            assert res["parent_asset_id"] == parent.asset_id
            assert res["generation_type"] == "sprite_sheet"
            assert "atlas_metadata" in res
            assert "frames" in res["atlas_metadata"]
