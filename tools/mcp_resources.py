"""MCP Native Resources & Prompts for ComfyUI MCP Server.

Exposes ComfyUI system state, model catalogs, workflow directory, and asset
metadata as MCP Resources (read-only URI-addressable context).  Also provides
expert prompt templates tuned for specific model families (FLUX, LTX-Video,
AceStep, Qwen character sheets).

Resources let AI Agents read context without a Tool Call round-trip, cutting
token costs and hallucination.  Prompts inject domain expertise so the Agent
produces model-optimal prompts out of the box.
"""

import json
import logging
from typing import Optional

import requests
from fastmcp import FastMCP

logger = logging.getLogger("MCP_Server")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fetch_system_stats(base_url: str) -> dict:
    """Fetch ComfyUI /system_stats (best-effort, returns empty on error)."""
    try:
        resp = requests.get(f"{base_url}/system_stats", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return {}


def _fetch_queue_info(base_url: str) -> dict:
    """Fetch ComfyUI /queue (best-effort)."""
    try:
        resp = requests.get(f"{base_url}/queue", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return {}


def _fetch_lora_names(base_url: str) -> list[str]:
    """Fetch LoRA model names from ComfyUI /object_info/LoraLoader."""
    try:
        resp = requests.get(f"{base_url}/object_info/LoraLoader", timeout=10)
        if resp.status_code != 200:
            return []
        data = resp.json()
        info = data.get("LoraLoader", {})
        if not isinstance(info, dict):
            return []
        required = info.get("input", {}).get("required", {})
        lora_name_field = required.get("lora_name", [])
        if isinstance(lora_name_field, list) and lora_name_field and isinstance(lora_name_field[0], list):
            return lora_name_field[0]
    except Exception as e:
        logger.debug(f"Could not fetch LoRA names: {e}")
    return []


# ---------------------------------------------------------------------------
# Registration entry point
# ---------------------------------------------------------------------------

def register_mcp_resources(
    mcp: FastMCP,
    comfyui_client,
    asset_registry,
    workflow_manager,
    gpu_guard,
    character_vault=None,
):
    """Register all MCP Resources and Prompts with the FastMCP server."""

    base_url = comfyui_client.base_url

    # ===================================================================
    # RESOURCES
    # ===================================================================

    # --- 1. GPU Health Status ---
    @mcp.resource(
        "comfyui://system/gpu-health",
        name="GPU Health Status",
        description=(
            "Live GPU health snapshot: VRAM utilization, free memory, "
            "queue depth, and device info. Read this before submitting "
            "heavy workflows to avoid OOM."
        ),
        mime_type="application/json",
    )
    def gpu_health_resource() -> str:
        stats = _fetch_system_stats(base_url)
        queue_info = _fetch_queue_info(base_url)

        devices_raw = stats.get("devices") or stats.get("system", {}).get("devices") or []
        devices = []
        for d in devices_raw:
            total = d.get("vram_total") or d.get("vram", {}).get("total") or 0
            free = d.get("vram_free") or d.get("vram", {}).get("free") or 0
            used = total - free if total > 0 else 0
            util_pct = round((used / total) * 100, 1) if total > 0 else None
            devices.append({
                "name": d.get("name", "unknown"),
                "type": d.get("type", "unknown"),
                "vram_total_gb": round(total / (1024 ** 3), 2) if total else 0,
                "vram_free_gb": round(free / (1024 ** 3), 2) if free else 0,
                "vram_used_gb": round(used / (1024 ** 3), 2) if used else 0,
                "vram_used_percent": util_pct,
            })

        # Queue info
        running = queue_info.get("queue_running", [])
        pending = queue_info.get("queue_pending", [])

        # Overall status
        max_util = max((d["vram_used_percent"] or 0) for d in devices) if devices else 0
        if max_util >= 92:
            status = "saturated"
        elif max_util >= 70 or len(running) >= 2:
            status = "busy"
        else:
            status = "healthy"

        result = {
            "status": status,
            "devices": devices,
            "queue_running_count": len(running),
            "queue_pending_count": len(pending),
            "recommendation": (
                "GPU is saturated — consider waiting or reducing resolution/steps before submitting."
                if status == "saturated"
                else "GPU is available for new jobs." if status == "healthy"
                else "GPU is moderately loaded — lightweight jobs are fine."
            ),
        }
        return json.dumps(result, indent=2)

    # --- 2. Checkpoint Models ---
    @mcp.resource(
        "comfyui://models/checkpoints",
        name="Available Checkpoints",
        description=(
            "All available checkpoint, UNET, diffusion, CLIP, and VAE models "
            "in ComfyUI. Use these names when setting models via set_defaults "
            "or run_workflow overrides."
        ),
        mime_type="application/json",
    )
    def checkpoints_resource() -> str:
        # Group by loader type for clarity
        categorized = {}
        for loader_type in comfyui_client.MODEL_LOADER_TYPES:
            try:
                names = comfyui_client._fetch_loader_model_names(loader_type)
                if names:
                    categorized[loader_type] = names
            except Exception:
                pass

        result = {
            "total_unique_count": len(comfyui_client.available_models),
            "all_models": comfyui_client.available_models,
            "by_loader_type": categorized,
        }
        return json.dumps(result, indent=2)

    # --- 3. LoRA Models ---
    @mcp.resource(
        "comfyui://models/loras",
        name="Available LoRAs",
        description=(
            "All available LoRA models in ComfyUI. LoRAs are lightweight "
            "fine-tuned adapters that modify generation style, subject, or "
            "technique without changing the base model."
        ),
        mime_type="application/json",
    )
    def loras_resource() -> str:
        lora_names = _fetch_lora_names(base_url)
        result = {
            "count": len(lora_names),
            "loras": lora_names,
            "tip": (
                "To use a LoRA, reference its exact filename in the workflow's "
                "LoraLoader node. Common trigger words are usually part of the "
                "LoRA filename (e.g., 'style_watercolor_v2' → trigger: 'watercolor style')."
            ),
        }
        return json.dumps(result, indent=2)

    # --- 4. Workflow Directory ---
    @mcp.resource(
        "comfyui://workflows",
        name="Registered Workflows",
        description=(
            "All auto-discovered parameterized workflows available via "
            "run_workflow or dedicated generation tools. Shows workflow IDs, "
            "exposed parameters, and media types."
        ),
        mime_type="application/json",
    )
    def workflows_resource() -> str:
        workflows = []
        for tool_def in workflow_manager.tool_definitions:
            wf_id = tool_def.workflow_id
            params = []
            for p in tool_def.parameters.values():
                params.append({
                    "name": p.name,
                    "type": p.annotation.__name__ if p.annotation else "str",
                    "required": p.required,
                    "description": p.description or "",
                })
            # Infer media type from workflow ID
            if "video" in wf_id or "i2v" in wf_id or "t2v" in wf_id or "r2v" in wf_id:
                media_type = "video"
            elif "audio" in wf_id or "song" in wf_id or "music" in wf_id:
                media_type = "audio"
            else:
                media_type = "image"

            workflows.append({
                "workflow_id": wf_id,
                "description": tool_def.description or "",
                "media_type": media_type,
                "parameters": params,
                "parameter_count": len(params),
            })

        result = {
            "count": len(workflows),
            "workflows": workflows,
        }
        return json.dumps(result, indent=2)

    # --- 5. Asset Details (Template Resource) ---
    @mcp.resource(
        "comfyui://assets/{asset_id}",
        name="Asset Details",
        description=(
            "Complete metadata, provenance, and lineage for a specific "
            "generated asset. Includes prompt, seed, workflow ID, timestamps, "
            "and download URL."
        ),
        mime_type="application/json",
    )
    def asset_detail_resource(asset_id: str) -> str:
        record = asset_registry.get_asset(asset_id)
        if not record:
            return json.dumps({"error": f"Asset '{asset_id}' not found or expired"})

        result = {
            "asset_id": record.asset_id,
            "asset_url": record.asset_url,
            "filename": record.filename,
            "subfolder": record.subfolder,
            "folder_type": record.folder_type,
            "workflow_id": record.workflow_id,
            "prompt_id": record.prompt_id,
            "mime_type": record.mime_type,
            "bytes_size": record.bytes_size,
            "created_at": record.created_at.isoformat() if record.created_at else None,
            "prompt": getattr(record, "prompt", None),
            "negative_prompt": getattr(record, "negative_prompt", None),
            "seed": getattr(record, "seed", None),
            "generation_type": getattr(record, "generation_type", None),
            "parent_asset_id": getattr(record, "parent_asset_id", None),
            "root_asset_id": getattr(record, "root_asset_id", None),
            "tags": getattr(record, "tags", None),
        }
        # Add image dimensions if available
        if hasattr(record, "width") and record.width:
            result["width"] = record.width
            result["height"] = record.height

        return json.dumps(result, indent=2)

    # --- 6. Asset Lineage Tree (Template Resource) ---
    @mcp.resource(
        "comfyui://assets/{asset_id}/lineage",
        name="Asset Lineage Tree",
        description=(
            "Full evolutionary lineage of an asset: ancestor chain (parent → "
            "grandparent → root), direct children, and whole-family graph. "
            "Useful for tracing iterative refinement history."
        ),
        mime_type="application/json",
    )
    def asset_lineage_resource(asset_id: str) -> str:
        record = asset_registry.get_asset(asset_id)
        if not record:
            return json.dumps({"error": f"Asset '{asset_id}' not found or expired"})

        lineage = asset_registry.get_lineage(asset_id)
        return json.dumps(lineage, indent=2, default=str)

    # --- 7. Character Profiles Vault ---
    if character_vault is not None:
        @mcp.resource(
            "comfyui://characters",
            name="Character & Style Profiles",
            description=(
                "All saved character and style consistency profiles in the vault. "
                "Contains trigger words, LoRA bindings, reference images, and "
                "style presets for cross-session consistency."
            ),
            mime_type="application/json",
        )
        def characters_resource() -> str:
            profiles = character_vault.list_profiles()
            result = {
                "count": len(profiles),
                "characters": [character_vault.profile_to_dict(p) for p in profiles],
            }
            return json.dumps(result, indent=2)

        # --- 8. Single Character Profile (Template Resource) ---
        @mcp.resource(
            "comfyui://characters/{character_id}",
            name="Character Profile Details",
            description=(
                "Detailed character profile: trigger words, negative triggers, "
                "LoRA bindings, reference images, style presets, and default parameters."
            ),
            mime_type="application/json",
        )
        def character_detail_resource(character_id: str) -> str:
            profile = character_vault.get_profile(character_id)
            if not profile:
                return json.dumps({"error": f"Character profile '{character_id}' not found"})
            return json.dumps(character_vault.profile_to_dict(profile), indent=2)

    # ===================================================================
    # PROMPTS
    # ===================================================================

    # --- 1. FLUX Photographic Prompt ---
    @mcp.prompt(
        "flux_photo_prompt",
        description=(
            "Generate an expert-tuned FLUX-format natural language photo prompt. "
            "FLUX models prefer rich, descriptive English sentences over tag lists. "
            "Outputs a ready-to-use prompt for generate_image or run_workflow."
        ),
    )
    def flux_photo_prompt(
        subject: str,
        style: str = "photorealistic",
        lighting: str = "natural soft lighting with golden hour warmth",
        camera: str = "85mm lens, f/1.8, shallow depth of field",
        mood: str = "",
    ) -> list[dict]:
        mood_clause = f" The mood is {mood}." if mood else ""
        prompt_text = (
            f"A {style} photograph of {subject}. "
            f"{lighting}. Shot with {camera}. "
            f"Highly detailed, 8K resolution, professional photography, "
            f"award-winning composition, sharp focus, cinematic color grading."
            f"{mood_clause}"
        )
        return [
            {
                "role": "user",
                "content": (
                    f"Here is an expert-tuned FLUX prompt for your image generation:\n\n"
                    f"**Prompt:**\n```\n{prompt_text}\n```\n\n"
                    f"**Recommended settings:** FLUX models work best with:\n"
                    f"- Steps: 20-30\n"
                    f"- CFG: 3.5-7.0 (FLUX prefers lower CFG than SD)\n"
                    f"- Resolution: 1024×1024 or 1280×768\n"
                    f"- Sampler: euler / dpmpp_2m\n\n"
                    f"Use this prompt with `generate_image` or a FLUX workflow."
                ),
            }
        ]

    # --- 2. Cinematic Video Prompt ---
    @mcp.prompt(
        "cinematic_video_prompt",
        description=(
            "Generate a cinematic video prompt optimized for LTX-Video, "
            "MiniMax H3, or Wan2.1 video generation models. Includes camera "
            "movement, motion description, and temporal pacing."
        ),
    )
    def cinematic_video_prompt(
        scene: str,
        motion: str = "slow, deliberate movement",
        camera_movement: str = "smooth dolly forward",
        duration: str = "5 seconds",
        style: str = "cinematic, film grain",
    ) -> list[dict]:
        prompt_text = (
            f"{scene}. {motion}. Camera: {camera_movement}. "
            f"Style: {style}. Duration: {duration}. "
            f"Smooth motion, consistent lighting, no flickering, "
            f"high temporal coherence, professional cinematography."
        )

        minimax_note = ""
        if "Picture" not in scene:
            minimax_note = (
                "\n\n**MiniMax H3 Note:** For image-to-video (i2v), prepend "
                "`<Picture 1>` to reference the input image. Example:\n"
                "`<Picture 1> The character begins to walk forward slowly...`"
            )

        return [
            {
                "role": "user",
                "content": (
                    f"Here is a cinematic video prompt:\n\n"
                    f"**Prompt:**\n```\n{prompt_text}\n```\n\n"
                    f"**Recommended workflows:**\n"
                    f"- `api_video_minimax_h3_t2v` (text-to-video)\n"
                    f"- `api_video_minimax_h3_i2v` (image-to-video, needs input image)\n"
                    f"- `api_video_ltx2_3_i2v` / `api_video_ltx2_5_i2v` (LTX-Video i2v)\n\n"
                    f"**Tips:**\n"
                    f"- Keep prompts under 200 words for best results\n"
                    f"- Describe motion explicitly (pan, tilt, zoom, track)\n"
                    f"- Avoid rapid scene changes in short clips"
                    f"{minimax_note}"
                ),
            }
        ]

    # --- 3. Character Sheet Prompt ---
    @mcp.prompt(
        "character_sheet_prompt",
        description=(
            "Generate a multi-angle character sheet prompt for consistent "
            "character design. Optimized for Qwen multi-angle workflows "
            "(2511 character / 2512 scene variants)."
        ),
    )
    def character_sheet_prompt(
        character_name: str,
        appearance: str,
        outfit: str = "",
        poses: str = "front view, side view, back view, 3/4 view, close-up, action pose",
        art_style: str = "anime illustration, clean lineart, flat colors",
    ) -> list[dict]:
        outfit_clause = f" Wearing {outfit}." if outfit else ""
        prompt_text = (
            f"Character sheet of {character_name}. {appearance}.{outfit_clause} "
            f"Multiple angles: {poses}. "
            f"Style: {art_style}. "
            f"White background, consistent proportions across all views, "
            f"professional character design reference sheet."
        )
        return [
            {
                "role": "user",
                "content": (
                    f"Here is a character sheet prompt:\n\n"
                    f"**Prompt:**\n```\n{prompt_text}\n```\n\n"
                    f"**Recommended workflows:**\n"
                    f"- `api_qwen_image_edit_2511_1_click_multiple_character_angles-v1.0` "
                    f"(6 character angle views)\n"
                    f"- `api_qwen_image_edit_2512_1_click_multiple_scene_angles-v1.0` "
                    f"(9 scene angle views)\n\n"
                    f"**Usage:** Upload a reference image of the character, then use this "
                    f"prompt to generate consistent multi-angle views. The workflow will "
                    f"produce `all_assets` with labeled views (close_up, wide_shot, etc.)."
                ),
            }
        ]

    # --- 4. Music Generation Prompt ---
    @mcp.prompt(
        "music_generation_prompt",
        description=(
            "Generate structured tags and lyrics for AceStep or MiniMax Music "
            "audio generation. Produces the correct format for generate_song."
        ),
    )
    def music_generation_prompt(
        genre: str,
        mood: str = "upbeat, energetic",
        instruments: str = "electric guitar, synth, drums",
        tempo: str = "120 BPM",
        theme: str = "",
        vocal_style: str = "clear male vocal",
    ) -> list[dict]:
        tags = f"{genre}, {mood}, {instruments}, {tempo}, {vocal_style}"
        theme_desc = theme if theme else f"A {mood} {genre} track"

        lyrics_template = (
            f"[Intro]\n"
            f"(Instrumental - {instruments})\n\n"
            f"[Verse 1]\n"
            f"(Write lyrics about: {theme_desc})\n"
            f"Line 1\nLine 2\nLine 3\nLine 4\n\n"
            f"[Chorus]\n"
            f"(Catchy, memorable hook)\n"
            f"Line 1\nLine 2\nLine 3\nLine 4\n\n"
            f"[Verse 2]\n"
            f"(Continue the story)\n"
            f"Line 1\nLine 2\nLine 3\nLine 4\n\n"
            f"[Chorus]\n"
            f"(Repeat or variation)\n\n"
            f"[Outro]\n"
            f"(Fade out - {instruments})"
        )

        return [
            {
                "role": "user",
                "content": (
                    f"Here is a music generation template:\n\n"
                    f"**Tags (for `generate_song` `tags` parameter):**\n"
                    f"```\n{tags}\n```\n\n"
                    f"**Lyrics template (for `generate_song` `lyrics` parameter):**\n"
                    f"```\n{lyrics_template}\n```\n\n"
                    f"**Recommended workflows:**\n"
                    f"- `generate_song` (built-in AceStep workflow)\n"
                    f"- `api_audio_ace_step1_5_xl_sft` (AceStep 1.5 XL)\n"
                    f"- `api_audio_minimax_music_3` (MiniMax Music 3)\n\n"
                    f"**Tips:**\n"
                    f"- Tags should be comma-separated descriptors\n"
                    f"- Lyrics use `[Section]` markers for structure alignment\n"
                    f"- AceStep supports up to 4 minutes; MiniMax Music up to 5 minutes\n"
                    f"- After generation, use `analyze_audio` to verify BPM and structure"
                ),
            }
        ]

    logger.info(
        "Registered MCP Resources (6) and Prompts (4): "
        "gpu-health, checkpoints, loras, workflows, asset-detail, asset-lineage | "
        "flux_photo, cinematic_video, character_sheet, music_generation"
    )
