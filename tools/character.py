"""Character & Style Consistency Vault — MCP tool layer.

Exposes character profile CRUD and prompt injection as MCP tools so that
AI Agents can save, recall, and apply character/style profiles for
consistent multi-round generation.
"""

import logging
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP

logger = logging.getLogger("MCP_Server")


def register_character_tools(mcp: FastMCP, character_vault):
    """Register character consistency vault tools with the MCP server."""

    @mcp.tool()
    def save_character_profile(
        character_id: str,
        display_name: str,
        description: str = "",
        trigger_words: str = "",
        negative_trigger: str = "",
        lora_name: Optional[str] = None,
        lora_strength: float = 0.75,
        reference_images: Optional[List[str]] = None,
        style_preset: Optional[str] = None,
        default_params: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> dict:
        """Save or update a character/style profile for consistent generation.

        Creates a persistent profile that stores trigger words, LoRA bindings,
        reference images, style presets, and default parameters. Once saved,
        use `apply_character_to_prompt` to automatically inject these settings
        into any generation prompt.

        Args:
            character_id: Unique identifier slug (e.g., "detective_john", "anime_hero")
            display_name: Human-readable character name (e.g., "Detective John")
            description: Free-form character description for reference
            trigger_words: Keywords automatically prepended to positive prompt
                           (e.g., "1girl, blue hair, detective outfit, cybernetic eye")
            negative_trigger: Keywords automatically prepended to negative prompt
                              (e.g., "blurry face, wrong eye color")
            lora_name: Exact LoRA filename in ComfyUI models/loras/ directory
            lora_strength: LoRA model strength (0.0-1.0, default 0.75)
            reference_images: List of image filenames in ComfyUI input/ directory
                              for IP-Adapter / InstantID reference
            style_preset: Art style preset name. Built-in presets: "anime",
                          "photorealistic", "oil_painting", "watercolor",
                          "pixel_art", "cyberpunk", "fantasy", "comic",
                          "3d_render", "sketch". Or any custom style string.
            default_params: Default generation parameter overrides (e.g.,
                            {"steps": 30, "cfg": 7.0, "width": 1024})
            tags: Categorization tags for searching (e.g., ["protagonist", "sci-fi"])

        Returns:
            Saved profile data with timestamps.
        """
        profile = character_vault.save_profile(
            character_id=character_id,
            display_name=display_name,
            description=description,
            trigger_words=trigger_words,
            negative_trigger=negative_trigger,
            lora_name=lora_name,
            lora_strength=lora_strength,
            reference_images=reference_images,
            style_preset=style_preset,
            default_params=default_params,
            tags=tags,
        )
        logger.info(f"Saved character profile: {character_id} ({display_name})")
        result = character_vault.profile_to_dict(profile)
        result["status"] = "saved"
        return result

    @mcp.tool()
    def get_character_profile(character_id: str) -> dict:
        """Get a saved character/style profile by ID.

        Returns the full profile including trigger words, LoRA bindings,
        reference images, style preset, and default parameters.

        Args:
            character_id: The character profile identifier (e.g., "detective_john")

        Returns:
            Complete profile data, or error if not found.
        """
        profile = character_vault.get_profile(character_id)
        if profile is None:
            return {"error": f"Character profile '{character_id}' not found"}
        return character_vault.profile_to_dict(profile)

    @mcp.tool()
    def list_character_profiles(
        tag: Optional[str] = None,
        style_preset: Optional[str] = None,
        query: Optional[str] = None,
        limit: int = 50,
    ) -> dict:
        """Browse and search saved character/style profiles.

        Args:
            tag: Filter by tag (e.g., "protagonist", "villain", "sci-fi")
            style_preset: Filter by style preset (e.g., "anime", "photorealistic")
            query: Free-text search across name, description, and trigger words
            limit: Maximum number of profiles to return (default 50)

        Returns:
            List of matching character profiles with count.
        """
        profiles = character_vault.list_profiles(
            tag=tag, style_preset=style_preset, query=query, limit=limit,
        )
        return {
            "profiles": [character_vault.profile_to_dict(p) for p in profiles],
            "count": len(profiles),
            "filters": {
                "tag": tag,
                "style_preset": style_preset,
                "query": query,
            },
        }

    @mcp.tool()
    def delete_character_profile(character_id: str) -> dict:
        """Delete a saved character/style profile.

        Args:
            character_id: The character profile identifier to delete

        Returns:
            Deletion status.
        """
        deleted = character_vault.delete_profile(character_id)
        if deleted:
            return {"status": "deleted", "character_id": character_id}
        return {"error": f"Character profile '{character_id}' not found"}

    @mcp.tool()
    def import_captions_to_character_vault(
        captions_path: str,
        character_prefix: str = "hero",
        display_names: Optional[Dict[str, str]] = None,
        trigger_words: Optional[Dict[str, str]] = None,
        extra_tags: Optional[List[str]] = None,
        style_preset: str = "fantasy",
    ) -> dict:
        """Import JoyCaption batch captions into the Character Vault (flywheel link).

        Reads either the raw ``captions_summary.json`` format produced by
        ``Batch_joy_caption_two(advanced)`` (flat ``{filename: caption}``) or
        the structured Godot format ``{"heroes": {"1001": {"file": ..., "caption": ...}}}``.
        Each entry becomes an upserted profile:
          - character_id = ``{character_prefix}_{id}`` (e.g. hero_1001)
          - description = caption text (Chinese or English as provided)
          - reference_images = [source filename] (ComfyUI input dir)
          - trigger_words from `trigger_words` map when provided

        Args:
            captions_path: Absolute path to the captions JSON file.
            character_prefix: ID prefix, default "hero".
            display_names: Optional map ``{id: 中文名}`` (e.g. {"1001": "魔法学徒哈利"}).
            trigger_words: Optional map ``{id: "1boy, blue robe, ..."}`` used as
                           prompt-injection keywords; caption is used as description.
            extra_tags: Extra search tags appended to ["joycaption", prefix].
            style_preset: Vault style preset, default "fantasy".

        Returns:
            Imported character ids, count, and per-id display names.
        """
        import json as _json
        from pathlib import Path as _Path

        path = _Path(captions_path)
        if not path.exists():
            return {"error": f"Captions file not found: {captions_path}"}

        try:
            payload = _json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            return {"error": f"Failed to read captions JSON: {e}"}

        display_names = display_names or {}
        trigger_words = trigger_words or {}
        tags = list(extra_tags or [])
        tags.extend(["joycaption", character_prefix])

        entries: Dict[str, Dict[str, str]] = {}
        heroes = payload.get("heroes") if isinstance(payload, dict) else None
        if isinstance(heroes, dict):
            for hero_id, entry in heroes.items():
                if isinstance(entry, dict):
                    caption = entry.get("caption") or entry.get("caption_zh")
                    if caption:
                        entries[str(hero_id)] = {
                            "caption": str(caption),
                            "file": str(entry.get("file") or f"{character_prefix}_{hero_id}_sheet.png"),
                        }
        else:
            for filename, caption in payload.items():
                if not isinstance(caption, str):
                    continue
                stem = _Path(filename).stem
                hero_id = stem
                for marker in (f"{character_prefix}_", "hero_", "_sheet", "sheet_"):
                    hero_id = hero_id.replace(marker, "")
                hero_id = hero_id.strip("_") or stem
                entries[hero_id] = {"caption": caption, "file": filename}

        if not entries:
            return {"error": "No caption entries parsed from JSON"}

        imported = []
        for hero_id, entry in sorted(entries.items(), key=lambda kv: kv[0]):
            character_id = f"{character_prefix}_{hero_id}"
            profile = character_vault.save_profile(
                character_id=character_id,
                display_name=display_names.get(hero_id, f"{character_prefix}_{hero_id}"),
                description=entry["caption"],
                trigger_words=trigger_words.get(hero_id, ""),
                negative_trigger="blurry, lowres, wrong colors, deformed anatomy",
                reference_images=[entry["file"]],
                style_preset=style_preset,
                tags=tags,
            )
            imported.append(character_vault.profile_to_dict(profile))

        logger.info(
            "Imported %d captions into character vault (prefix=%s)",
            len(imported), character_prefix,
        )
        return {
            "status": "imported",
            "count": len(imported),
            "character_ids": [p["character_id"] for p in imported],
            "profiles": imported,
        }

    @mcp.tool()
    def apply_character_to_prompt(
        character_id: str,
        prompt: str,
        negative_prompt: str = "",
    ) -> dict:
        """Apply a character profile's features to a generation prompt.

        Automatically injects the character's trigger words, negative triggers,
        and style preset keywords into the provided prompt. Also returns LoRA
        binding info and reference images for use with generation tools.

        Usage pattern:
        1. First call `apply_character_to_prompt` to get enhanced prompt
        2. Then pass the enhanced prompt to `generate_image` or `run_workflow`
        3. If LoRA is specified, add it to workflow overrides

        Args:
            character_id: The character profile to apply
            prompt: Base positive prompt to enhance
            negative_prompt: Base negative prompt to enhance (optional)

        Returns:
            Enhanced prompt, negative prompt, LoRA config, reference images,
            and default parameter overrides. Or error if profile not found.
        """
        result = character_vault.apply_character(
            character_id=character_id,
            prompt=prompt,
            negative_prompt=negative_prompt,
        )
        if "error" in result:
            return result

        logger.info(
            f"Applied character '{character_id}' to prompt "
            f"(trigger: {len(result.get('prompt', ''))} chars, "
            f"lora: {result.get('lora_name', 'none')})"
        )
        return result
