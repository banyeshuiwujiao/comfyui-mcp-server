"""Unit tests for SQLite asset persistence and lineage tracking."""

import os
import tempfile
import pytest
from unittest.mock import MagicMock

from managers.asset_registry import AssetRegistry
from tools.job import register_job_tools
from tools.helpers import register_and_build_response


@pytest.fixture
def temp_db_path():
    """Create a temporary database file and clean it up after test."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    for p in (path, path + "-wal", path + "-shm"):
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass


def test_sqlite_persistence_across_restarts(temp_db_path):
    """Test that assets survive server restarts via SQLite storage."""
    # Session 1: Register an asset
    registry1 = AssetRegistry(ttl_hours=24, comfyui_base_url="http://localhost:8188", db_path=temp_db_path)
    rec1 = registry1.register_asset(
        filename="cyber_girl_01.png",
        subfolder="",
        folder_type="output",
        workflow_id="generate_image",
        prompt_id="prompt_aaa",
        mime_type="image/png",
        width=1024,
        height=1024,
        prompt="a cyberpunk girl with neon hair",
        seed=42,
        tags=["cyberpunk", "character"]
    )
    asset_id = rec1.asset_id

    # Session 2: Create brand new registry instance pointing to same DB (simulating restart)
    registry2 = AssetRegistry(ttl_hours=24, comfyui_base_url="http://localhost:8188", db_path=temp_db_path)
    
    # Verify retrieval by asset_id
    rec2 = registry2.get_asset(asset_id)
    assert rec2 is not None
    assert rec2.asset_id == asset_id
    assert rec2.filename == "cyber_girl_01.png"
    assert rec2.prompt == "a cyberpunk girl with neon hair"
    assert rec2.seed == 42
    assert "cyberpunk" in rec2.tags

    # Verify retrieval by identity
    rec_by_id = registry2.get_asset_by_identity("cyber_girl_01.png", "", "output")
    assert rec_by_id is not None
    assert rec_by_id.asset_id == asset_id


def test_lineage_ancestry_and_family_tree(temp_db_path):
    """Test multi-hop asset lineage (A -> B -> C and A -> B -> D)."""
    registry = AssetRegistry(ttl_hours=24, db_path=temp_db_path)

    # 1. Root asset A (Text-to-Image)
    asset_a = registry.register_asset(
        filename="root_a.png",
        subfolder="",
        folder_type="output",
        workflow_id="generate_image",
        prompt_id="p_1",
        generation_type="t2i",
        prompt="medieval castle at sunset"
    )
    assert asset_a.root_asset_id == asset_a.asset_id
    assert asset_a.parent_asset_id is None

    # 2. Child asset B (Regenerate with modifications)
    asset_b = registry.register_asset(
        filename="child_b.png",
        subfolder="",
        folder_type="output",
        workflow_id="generate_image",
        prompt_id="p_2",
        parent_asset_id=asset_a.asset_id,
        generation_type="regenerate",
        prompt="medieval castle in winter snow"
    )
    assert asset_b.parent_asset_id == asset_a.asset_id
    assert asset_b.root_asset_id == asset_a.asset_id

    # 3. Child asset C (Upscale derived from B)
    asset_c = registry.register_asset(
        filename="upscale_c.png",
        subfolder="",
        folder_type="output",
        workflow_id="upscaler_2k",
        prompt_id="p_3",
        parent_asset_id=asset_b.asset_id,
        generation_type="upscale",
        prompt="medieval castle in winter snow 4k"
    )
    assert asset_c.parent_asset_id == asset_b.asset_id
    assert asset_c.root_asset_id == asset_a.asset_id

    # 4. Child asset D (Video derived from B)
    asset_d = registry.register_asset(
        filename="video_d.mp4",
        subfolder="",
        folder_type="output",
        workflow_id="ltx_video_i2v",
        prompt_id="p_4",
        parent_asset_id=asset_b.asset_id,
        generation_type="i2v",
        prompt="snow falling gently on medieval castle"
    )

    # Test lineage for C (leaf node)
    lineage_c = registry.get_lineage(asset_c.asset_id)
    assert lineage_c["asset_id"] == asset_c.asset_id
    assert lineage_c["parent_asset_id"] == asset_b.asset_id
    assert lineage_c["root_asset_id"] == asset_a.asset_id
    assert len(lineage_c["ancestors"]) == 2
    assert lineage_c["ancestors"][0]["asset_id"] == asset_b.asset_id
    assert lineage_c["ancestors"][1]["asset_id"] == asset_a.asset_id

    # Test lineage for B (intermediate node)
    lineage_b = registry.get_lineage(asset_b.asset_id)
    assert len(lineage_b["ancestors"]) == 1
    assert lineage_b["ancestors"][0]["asset_id"] == asset_a.asset_id
    assert len(lineage_b["children"]) == 2
    child_ids = {ch["asset_id"] for ch in lineage_b["children"]}
    assert asset_c.asset_id in child_ids
    assert asset_d.asset_id in child_ids

    # Test family tree from Root A
    lineage_a = registry.get_lineage(asset_a.asset_id)
    assert lineage_a["ancestor_count"] == 0
    assert lineage_a["family_tree_count"] == 4


def test_search_assets_by_query_and_tag(temp_db_path):
    """Test full-text search across prompt, filename, and tags."""
    registry = AssetRegistry(ttl_hours=24, db_path=temp_db_path)

    registry.register_asset(
        filename="robot_warrior.png",
        subfolder="",
        folder_type="output",
        workflow_id="generate_image",
        prompt_id="p1",
        prompt="futuristic mecha warrior holding glowing sword",
        tags=["mecha", "scifi", "action"]
    )
    registry.register_asset(
        filename="cute_cat.png",
        subfolder="",
        folder_type="output",
        workflow_id="generate_image",
        prompt_id="p2",
        prompt="a fluffy white cat sleeping in a basket",
        tags=["animal", "cute"]
    )

    # Search by keyword in prompt
    mecha_results = registry.search_assets(query="mecha")
    assert len(mecha_results) == 1
    assert mecha_results[0].filename == "robot_warrior.png"

    # Search by tag
    tag_results = registry.search_assets(tag="cute")
    assert len(tag_results) == 1
    assert tag_results[0].filename == "cute_cat.png"

    # Search non-matching
    none_results = registry.search_assets(query="dragon")
    assert len(none_results) == 0


def test_mcp_tools_lineage_and_search_integration(temp_db_path):
    """Test MCP tool registration and invocation for lineage & search."""
    registry = AssetRegistry(ttl_hours=24, comfyui_base_url="http://localhost:8188", db_path=temp_db_path)

    # Pre-populate assets
    rec_parent = registry.register_asset(
        filename="photo_orig.png",
        subfolder="",
        folder_type="output",
        workflow_id="generate_image",
        prompt_id="p_orig",
        prompt="portrait of an astronaut"
    )
    rec_child = registry.register_asset(
        filename="photo_inpaint.png",
        subfolder="",
        folder_type="output",
        workflow_id="generate_image",
        prompt_id="p_inpaint",
        parent_asset_id=rec_parent.asset_id,
        generation_type="inpaint",
        prompt="portrait of an astronaut with helmet reflection"
    )

    captured_tools = {}
    mock_mcp = MagicMock()
    def mock_tool_decorator():
        def decorator(fn):
            captured_tools[fn.__name__] = fn
            return fn
        return decorator
    mock_mcp.tool = mock_tool_decorator

    mock_client = MagicMock()
    register_job_tools(mock_mcp, mock_client, registry)

    # 1. Test get_asset_lineage
    assert "get_asset_lineage" in captured_tools
    lineage_res = captured_tools["get_asset_lineage"](asset_id=rec_child.asset_id)
    assert lineage_res["parent_asset_id"] == rec_parent.asset_id
    assert lineage_res["ancestor_count"] == 1

    # 2. Test search_assets
    assert "search_assets" in captured_tools
    search_res = captured_tools["search_assets"](query="astronaut")
    assert search_res["count"] == 2

    # 3. Test get_asset_metadata with lineage
    meta_res = captured_tools["get_asset_metadata"](asset_id=rec_child.asset_id)
    assert meta_res["parent_asset_id"] == rec_parent.asset_id
    assert meta_res["generation_type"] == "inpaint"
