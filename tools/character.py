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
