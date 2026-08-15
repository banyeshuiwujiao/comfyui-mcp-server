"""Unit tests for video keyframe extraction, contact sheet, and GIF preview."""

import io
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image, ImageDraw

from asset_processor import (
    AV_AVAILABLE,
    create_video_animated_gif,
    create_video_contact_sheet,
    extract_video_keyframes,
    extract_video_metadata,
)
from managers.asset_registry import AssetRegistry
from tools.asset import register_asset_tools


def generate_synthetic_mp4_bytes(duration_sec: float = 2.0, fps: int = 24, width: int = 320, height: int = 240) -> bytes:
    """Generate in-memory synthetic MP4 bytes for testing."""
    if not AV_AVAILABLE:
        pytest.skip("PyAV is required to generate synthetic test MP4")

    import av
    buf = io.BytesIO()
    container = av.open(buf, mode='w', format='mp4')
    stream = container.add_stream('h264', rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = 'yuv420p'

    total_frames = int(duration_sec * fps)
    for i in range(total_frames):
        img = Image.new('RGB', (width, height), (int(i * 255 / total_frames), 120, 200))
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), f"Frame {i}", fill=(255, 255, 255))
        frame = av.VideoFrame.from_image(img)
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)
    container.close()

    return buf.getvalue()


@pytest.fixture
def synthetic_video_bytes():
    return generate_synthetic_mp4_bytes(duration_sec=2.0, fps=24, width=320, height=240)


def test_extract_video_metadata(synthetic_video_bytes):
    """Test extracting video metadata from MP4 bytes."""
    meta = extract_video_metadata(synthetic_video_bytes)
    assert meta["width"] == 320
    assert meta["height"] == 240
    assert meta["fps"] == 24.0
    assert meta["total_frames"] == 48
    assert meta["duration_sec"] is not None
    assert abs(meta["duration_sec"] - 2.0) < 0.2


def test_extract_video_keyframes(synthetic_video_bytes):
    """Test extracting evenly spaced keyframes."""
    keyframes = extract_video_keyframes(synthetic_video_bytes, num_frames=4, max_dim=256)
    assert len(keyframes) == 4
    
    # Check timestamps progression
    ts_list = [k[0] for k in keyframes]
    assert ts_list[0] == 0.0
    assert ts_list[-1] > 1.5
    assert ts_list == sorted(ts_list)

    # Check images
    for ts, img in keyframes:
        assert isinstance(img, Image.Image)
        assert img.width <= 256
        assert img.height <= 256


def test_create_video_contact_sheet(synthetic_video_bytes):
    """Test stitching keyframes into a labeled contact sheet."""
    sheet_bytes = create_video_contact_sheet(synthetic_video_bytes, num_frames=4, max_width=800, quality=75)
    assert len(sheet_bytes) > 0

    # Verify WebP output is valid image
    with Image.open(io.BytesIO(sheet_bytes)) as img:
        assert img.format == "WEBP"
        assert img.width > 300
        assert img.height > 100


def test_create_video_animated_gif(synthetic_video_bytes):
    """Test creating an animated GIF thumbnail."""
    gif_bytes = create_video_animated_gif(synthetic_video_bytes, max_frames=8, target_fps=4, max_dim=160)
    assert len(gif_bytes) > 0

    # Verify GIF output
    with Image.open(io.BytesIO(gif_bytes)) as img:
        assert img.format == "GIF"
        assert getattr(img, "is_animated", False)
        assert img.n_frames > 1


def test_tools_view_video_preview_and_view_image(synthetic_video_bytes):
    """Test MCP tool layer for video previews and automatic view_image fallback."""
    registry = AssetRegistry(ttl_hours=24, db_path=":memory:")
    
    # Register video asset
    rec_video = registry.register_asset(
        filename="scifi_flythrough.mp4",
        subfolder="",
        folder_type="output",
        workflow_id="ltx_video_i2v",
        prompt_id="prompt_video_123",
        mime_type="video/mp4",
        width=320,
        height=240,
        bytes_size=len(synthetic_video_bytes),
        prompt="spacecraft flying through asteroid belt"
    )

    captured_tools = {}
    mock_mcp = MagicMock()
    def mock_tool_decorator():
        def decorator(fn):
            captured_tools[fn.__name__] = fn
            return fn
        return decorator
    mock_mcp.tool = mock_tool_decorator

    register_asset_tools(mock_mcp, registry)

    assert "view_video_preview" in captured_tools
    assert "view_image" in captured_tools

    with patch("tools.asset.fetch_asset_bytes", return_value=synthetic_video_bytes):
        # 1. Test view_video_preview in strip mode
        strip_res = captured_tools["view_video_preview"](asset_id=rec_video.asset_id, mode="strip", num_frames=4)
        assert hasattr(strip_res, "data")
        assert getattr(strip_res, "_format", None) == "webp"
        assert len(strip_res.data) > 0

        # 2. Test view_video_preview in gif mode
        gif_res = captured_tools["view_video_preview"](asset_id=rec_video.asset_id, mode="gif")
        assert hasattr(gif_res, "data")
        assert getattr(gif_res, "_format", None) == "gif"
        assert len(gif_res.data) > 0

        # 3. Test view_video_preview in metadata mode
        meta_res = captured_tools["view_video_preview"](asset_id=rec_video.asset_id, mode="metadata")
        assert meta_res["filename"] == "scifi_flythrough.mp4"
        assert meta_res["total_frames"] == 48

        # 4. Test view_image seamless video contact sheet fallback
        img_res = captured_tools["view_image"](asset_id=rec_video.asset_id, mode="thumb")
        assert hasattr(img_res, "data")
        assert getattr(img_res, "_format", None) == "webp"
        assert len(img_res.data) > 0
