"""Unit tests for Character & Style Consistency Vault."""

import pytest
from unittest.mock import MagicMock

from managers.character_vault import CharacterVault, STYLE_PRESETS
from tools.character import register_character_tools


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def vault():
    return CharacterVault(db_path=":memory:")


@pytest.fixture
def captured_tools(vault):
    """Register character tools and capture them via mock."""
    tools = {}
    mock_mcp = MagicMock()

    def mock_tool_decorator():
        def decorator(fn):
            tools[fn.__name__] = fn
            return fn
        return decorator

    mock_mcp.tool = mock_tool_decorator
    register_character_tools(mock_mcp, vault)
    return tools, vault


# ---------------------------------------------------------------------------
# Manager-level tests
# ---------------------------------------------------------------------------

class TestCharacterVaultManager:

    def test_save_and_get_profile(self, vault):
        profile = vault.save_profile(
            character_id="detective_john",
            display_name="Detective John",
            description="A cyberpunk noir detective",
            trigger_words="1man, detective outfit, cybernetic left eye, dark trenchcoat",
            negative_trigger="blurry face, wrong eye color",
            lora_name="detective_john_v2.safetensors",
            lora_strength=0.8,
            reference_images=["john_ref_front.png", "john_ref_side.png"],
            style_preset="cyberpunk",
            default_params={"steps": 30, "cfg": 7.0},
            tags=["protagonist", "sci-fi"],
        )

        assert profile.character_id == "detective_john"
        assert profile.display_name == "Detective John"
        assert profile.lora_strength == 0.8
        assert profile.created_at is not None

        # Retrieve
        fetched = vault.get_profile("detective_john")
        assert fetched is not None
        assert fetched.character_id == "detective_john"
        assert fetched.trigger_words == "1man, detective outfit, cybernetic left eye, dark trenchcoat"
        assert fetched.lora_name == "detective_john_v2.safetensors"
        assert fetched.reference_images == ["john_ref_front.png", "john_ref_side.png"]
        assert fetched.tags == ["protagonist", "sci-fi"]
        assert fetched.default_params == {"steps": 30, "cfg": 7.0}

    def test_get_nonexistent_profile(self, vault):
        assert vault.get_profile("nobody") is None

    def test_upsert_preserves_created_at(self, vault):
        vault.save_profile(character_id="hero", display_name="Hero v1")
        p1 = vault.get_profile("hero")
        original_created = p1.created_at

        vault.save_profile(character_id="hero", display_name="Hero v2 Updated")
        p2 = vault.get_profile("hero")

        assert p2.display_name == "Hero v2 Updated"
        assert p2.created_at == original_created
        assert p2.updated_at >= p2.created_at

    def test_list_profiles_all(self, vault):
        vault.save_profile(character_id="char_a", display_name="Alpha")
        vault.save_profile(character_id="char_b", display_name="Beta")
        vault.save_profile(character_id="char_c", display_name="Gamma")

        all_profiles = vault.list_profiles()
        assert len(all_profiles) == 3

    def test_list_profiles_filter_by_tag(self, vault):
        vault.save_profile(character_id="hero", display_name="Hero", tags=["protagonist"])
        vault.save_profile(character_id="villain", display_name="Villain", tags=["antagonist"])
        vault.save_profile(character_id="sidekick", display_name="Sidekick", tags=["protagonist", "support"])

        protas = vault.list_profiles(tag="protagonist")
        assert len(protas) == 2
        ids = {p.character_id for p in protas}
        assert "hero" in ids
        assert "sidekick" in ids

    def test_list_profiles_filter_by_style(self, vault):
        vault.save_profile(character_id="anime_girl", display_name="Anime Girl", style_preset="anime")
        vault.save_profile(character_id="photo_model", display_name="Photo Model", style_preset="photorealistic")

        anime_profiles = vault.list_profiles(style_preset="anime")
        assert len(anime_profiles) == 1
        assert anime_profiles[0].character_id == "anime_girl"

    def test_list_profiles_text_query(self, vault):
        vault.save_profile(
            character_id="cyber_cat", display_name="Cyber Cat",
            description="A neon-glowing cyberpunk cat",
        )
        vault.save_profile(
            character_id="forest_elf", display_name="Forest Elf",
            description="A mystical woodland creature",
        )

        results = vault.list_profiles(query="cyberpunk")
        assert len(results) == 1
        assert results[0].character_id == "cyber_cat"

    def test_delete_profile(self, vault):
        vault.save_profile(character_id="temp", display_name="Temporary")
        assert vault.get_profile("temp") is not None

        deleted = vault.delete_profile("temp")
        assert deleted is True
        assert vault.get_profile("temp") is None

    def test_delete_nonexistent(self, vault):
        assert vault.delete_profile("nobody") is False

    def test_apply_character_basic(self, vault):
        vault.save_profile(
            character_id="warrior",
            display_name="Warrior",
            trigger_words="1man, heavy armor, battle axe",
            negative_trigger="modern clothing",
            style_preset="fantasy",
        )

        result = vault.apply_character(
            character_id="warrior",
            prompt="standing in a dark forest",
            negative_prompt="blurry",
        )

        assert "error" not in result
        assert result["prompt"].startswith("1man, heavy armor, battle axe, standing in a dark forest")
        assert "fantasy art style" in result["prompt"]
        assert "modern clothing" in result["negative_prompt"]
        assert "blurry" in result["negative_prompt"]

    def test_apply_character_no_triggers(self, vault):
        vault.save_profile(
            character_id="plain",
            display_name="Plain",
            lora_name="style_abstract.safetensors",
            lora_strength=0.6,
        )

        result = vault.apply_character("plain", "a sunset")
        assert result["prompt"] == "a sunset"
        assert result["lora_name"] == "style_abstract.safetensors"
        assert result["lora_strength"] == 0.6

    def test_apply_character_not_found(self, vault):
        result = vault.apply_character("nobody", "test prompt")
        assert "error" in result

    def test_apply_character_custom_style(self, vault):
        vault.save_profile(
            character_id="custom",
            display_name="Custom",
            trigger_words="abstract shapes",
            style_preset="vaporwave neon aesthetic",
        )

        result = vault.apply_character("custom", "a cityscape")
        assert "abstract shapes" in result["prompt"]
        # Custom style preset passed through as-is
        assert "vaporwave neon aesthetic" in result["prompt"]

    def test_profile_to_dict(self, vault):
        profile = vault.save_profile(
            character_id="test",
            display_name="Test Character",
            tags=["test"],
        )
        d = vault.profile_to_dict(profile)
        assert d["character_id"] == "test"
        assert d["display_name"] == "Test Character"
        assert d["tags"] == ["test"]
        assert d["created_at"] is not None


# ---------------------------------------------------------------------------
# MCP tool-level tests
# ---------------------------------------------------------------------------

class TestCharacterTools:

    def test_save_character_profile_tool(self, captured_tools):
        tools, vault = captured_tools

        result = tools["save_character_profile"](
            character_id="mecha_pilot",
            display_name="Mecha Pilot",
            trigger_words="1girl, pilot suit, cockpit",
            style_preset="anime",
            tags=["protagonist", "mecha"],
        )

        assert result["status"] == "saved"
        assert result["character_id"] == "mecha_pilot"
        assert result["style_preset"] == "anime"

    def test_get_character_profile_tool(self, captured_tools):
        tools, vault = captured_tools

        tools["save_character_profile"](
            character_id="test_char",
            display_name="Test",
            trigger_words="test trigger",
        )

        result = tools["get_character_profile"](character_id="test_char")
        assert result["character_id"] == "test_char"
        assert result["trigger_words"] == "test trigger"

    def test_get_character_profile_not_found(self, captured_tools):
        tools, _ = captured_tools
        result = tools["get_character_profile"](character_id="nonexistent")
        assert "error" in result

    def test_list_character_profiles_tool(self, captured_tools):
        tools, _ = captured_tools

        tools["save_character_profile"](character_id="a", display_name="Alpha", tags=["hero"])
        tools["save_character_profile"](character_id="b", display_name="Beta", tags=["villain"])

        result = tools["list_character_profiles"]()
        assert result["count"] == 2

        result_filtered = tools["list_character_profiles"](tag="hero")
        assert result_filtered["count"] == 1
        assert result_filtered["profiles"][0]["character_id"] == "a"

    def test_delete_character_profile_tool(self, captured_tools):
        tools, _ = captured_tools

        tools["save_character_profile"](character_id="doomed", display_name="Doomed")
        result = tools["delete_character_profile"](character_id="doomed")
        assert result["status"] == "deleted"

        result2 = tools["delete_character_profile"](character_id="doomed")
        assert "error" in result2

    def test_apply_character_to_prompt_tool(self, captured_tools):
        tools, _ = captured_tools

        tools["save_character_profile"](
            character_id="ninja",
            display_name="Shadow Ninja",
            trigger_words="1man, ninja outfit, shadow, katana",
            negative_trigger="bright colors, modern",
            style_preset="anime",
            lora_name="ninja_style.safetensors",
            lora_strength=0.85,
        )

        result = tools["apply_character_to_prompt"](
            character_id="ninja",
            prompt="leaping across rooftops at night",
            negative_prompt="blurry, low quality",
        )

        assert "error" not in result
        assert "ninja outfit" in result["prompt"]
        assert "leaping across rooftops" in result["prompt"]
        assert "anime style" in result["prompt"]
        assert "bright colors" in result["negative_prompt"]
        assert "blurry" in result["negative_prompt"]
        assert result["lora_name"] == "ninja_style.safetensors"
        assert result["lora_strength"] == 0.85

    def test_apply_character_to_prompt_not_found(self, captured_tools):
        tools, _ = captured_tools
        result = tools["apply_character_to_prompt"](
            character_id="ghost", prompt="test"
        )
        assert "error" in result

    def test_all_expected_tools_registered(self, captured_tools):
        tools, _ = captured_tools
        expected = [
            "save_character_profile",
            "get_character_profile",
            "list_character_profiles",
            "delete_character_profile",
            "apply_character_to_prompt",
        ]
        for name in expected:
            assert name in tools, f"Missing tool: {name}"

    def test_all_builtin_style_presets(self, vault):
        for preset_name in STYLE_PRESETS.keys():
            vault.save_profile(
                character_id=f"char_{preset_name}",
                display_name=f"Char {preset_name}",
                style_preset=preset_name,
            )
            res = vault.apply_character(f"char_{preset_name}", "portrait")
            assert "error" not in res
            assert STYLE_PRESETS[preset_name] in res["prompt"]

    def test_apply_character_empty_prompt(self, vault):
        vault.save_profile(
            character_id="solo_char",
            display_name="Solo",
            trigger_words="masterpiece, 1girl",
            negative_trigger="lowres",
        )
        res = vault.apply_character("solo_char", prompt="", negative_prompt="")
        assert res["prompt"] == "masterpiece, 1girl"
        assert res["negative_prompt"] == "lowres"
