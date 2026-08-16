"""JoyCaption captions -> Character Vault import tool regression."""

import json

from managers.character_vault import CharacterVault
from tools.character import register_character_tools


def _register(vault):
    tools = {}

    class MockMCP:
        def tool(self):
            def decorator(fn):
                tools[fn.__name__] = fn
                return fn
            return decorator

    register_character_tools(MockMCP(), vault)
    return tools


def test_import_structured_godot_captions(tmp_path):
    vault = CharacterVault(db_path=":memory:")
    tools = _register(vault)

    captions = {
        "heroes": {
            "1001": {
                "file": "hero_1001_sheet.png",
                "caption": "A young mage in a blue and gold robe holding a glowing book.",
            },
            "1002": {
                "file": "hero_1002_sheet.png",
                "caption": "A swordswoman in a white kimono with a blue katana.",
            },
        }
    }
    path = tmp_path / "captions.json"
    path.write_text(json.dumps(captions), encoding="utf-8")

    result = tools["import_captions_to_character_vault"](
        captions_path=str(path),
        display_names={"1001": "魔法学徒哈利", "1002": "神剑派慕红雪"},
        trigger_words={
            "1001": "1boy, blue robe, golden trim, glowing book",
            "1002": "1girl, white kimono, blue katana",
        },
    )

    assert result["status"] == "imported"
    assert result["count"] == 2
    assert set(result["character_ids"]) == {"hero_1001", "hero_1002"}

    p = vault.get_profile("hero_1001")
    assert p.display_name == "魔法学徒哈利"
    assert p.reference_images == ["hero_1001_sheet.png"]
    assert p.trigger_words.startswith("1boy")
    assert "joycaption" in p.tags and "hero" in p.tags

    applied = vault.apply_character("hero_1002", "azurite sword slash effect")
    assert applied["prompt"].startswith("1girl")
    assert "fantasy" in applied["prompt"]


def test_import_flat_captions_summary(tmp_path):
    vault = CharacterVault(db_path=":memory:")
    tools = _register(vault)

    captions = {
        "hero_1003_sheet.png": "Winged dark-skinned warrior with black wings and flames.",
        "hero_1004_sheet.png": "Tall mage in a dark purple rune robe.",
    }
    path = tmp_path / "captions_summary.json"
    path.write_text(json.dumps(captions), encoding="utf-8")

    result = tools["import_captions_to_character_vault"](
        captions_path=str(path),
    )

    assert result["count"] == 2
    assert vault.get_profile("hero_1003").description.startswith("Winged")


def test_import_missing_file_returns_error(tmp_path):
    vault = CharacterVault(db_path=":memory:")
    tools = _register(vault)

    result = tools["import_captions_to_character_vault"](
        captions_path=str(tmp_path / "nope.json"),
    )

    assert "error" in result
