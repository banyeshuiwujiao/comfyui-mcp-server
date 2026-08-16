"""Game and Web Asset Post-Processing Pipeline MCP tools.

Provides tools for automated asset preparation:
- remove_background: Extract foreground and generate 32-bit transparent RGBA PNG
- generate_sprite_sheet: Pack animation loops/sequences into texture atlases with JSON metadata
"""

import io
import logging
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP
from PIL import Image

from asset_processor import (
    build_sprite_sheet,
    extract_video_keyframes,
    fetch_asset_bytes,
    persist_processed_bytes,
    remove_image_background,
)

logger = logging.getLogger("MCP_Server")


def register_pipeline_tools(mcp: FastMCP, asset_registry, comfyui_client, pipeline_orchestrator=None, publish_manager=None):
    """Register game & web post-processing and composite pipeline tools with the MCP server."""

    @mcp.tool()
    def remove_background(
        asset_id: str,
        mode: str = "auto",
        bgcolor: Optional[str] = None,
        tolerance: int = 32,
        feather: int = 2,
    ) -> dict:
        """Remove the background of an existing image asset and generate a transparent PNG.

        Supports:
        - Auto-detection (`mode="auto"`): Analyzes corner variance to distinguish solid studio backgrounds from complex scenes
        - Color keying (`mode="color"`): Precise removal of solid backgrounds (e.g. white, black, green screen) with soft edge feathering
        - Foreground segmentation (`mode="grabcut"`): OpenCV GrabCut extraction for natural subjects on complex backgrounds

        The resulting transparent image is registered as a new asset with full lineage
        links back to the source asset (`parent_asset_id`).

        Args:
            asset_id: Asset ID of the source image
            mode: Segmentation mode - "auto" (default), "color" (chroma keying), or "grabcut" (contour segmentation)
            bgcolor: Target background color for color mode (e.g. "white", "black", "green", or hex "#ffffff"). Auto-detected if omitted.
            tolerance: Color distance threshold (0-100, default 32)
            feather: Edge smoothing radius in pixels (default 2)

        Returns:
            New asset record dict containing asset_id, asset_url, mime_type, and parent lineage.
        """
        asset_record = asset_registry.get_asset(asset_id)
        if not asset_record:
            return {"error": f"Asset '{asset_id}' not found or expired"}

        raw_bytes = fetch_asset_bytes(asset_record, comfyui_client.base_url)
        if not raw_bytes:
            return {"error": f"Could not fetch image data for asset '{asset_id}'"}

        try:
            transparent_bytes = remove_image_background(
                raw_bytes,
                mode=mode,
                bgcolor=bgcolor,
                tolerance=tolerance,
                feather=feather,
            )
        except Exception as e:
            logger.error(f"Background removal failed for {asset_id}: {e}")
            return {"error": f"Background removal failed: {str(e)}"}

        base_stem = asset_record.filename.rsplit(".", 1)[0]
        out_filename = f"transparent_{base_stem}.png"

        # Persist the processed bytes into the ComfyUI output root so the
        # registered /view asset_url resolves (registered metadata alone is not a file).
        configured_root = getattr(getattr(publish_manager, "config", None), "comfyui_output_root", None)
        if not persist_processed_bytes(out_filename, transparent_bytes, asset_record.subfolder, configured_root):
            return {"error": "ComfyUI output root not configured; processed asset could not be persisted (set COMFYUI_OUTPUT_ROOT)"}

        # Register in asset registry with lineage
        new_record = asset_registry.register_asset(
            filename=out_filename,
            subfolder=asset_record.subfolder,
            folder_type=asset_record.folder_type,
            workflow_id="remove_background",
            prompt_id=asset_record.prompt_id,
            mime_type="image/png",
            bytes_size=len(transparent_bytes),
            parent_asset_id=asset_record.asset_id,
            generation_type="matting",
            prompt=getattr(asset_record, "prompt", None),
            tags=["transparent", "matting", *(getattr(asset_record, "tags", []) or [])],
        )

        logger.info(f"Generated transparent asset {new_record.asset_id} from {asset_id} ({len(transparent_bytes)} bytes)")

        return {
            "asset_id": new_record.asset_id,
            "asset_url": new_record.asset_url,
            "filename": new_record.filename,
            "mime_type": "image/png",
            "bytes_size": new_record.bytes_size,
            "parent_asset_id": asset_record.asset_id,
            "generation_type": "matting",
            "status": "success",
        }

    @mcp.tool()
    def generate_sprite_sheet(
        asset_id: str,
        frame_count: int = 8,
        columns: Optional[int] = None,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
        remove_bg: bool = False,
        format: str = "png",
    ) -> dict:
        """Pack an animation video or sequence into a game engine texture atlas (Sprite Sheet).

        Extracts frames from a video asset (e.g. Wan2.1 / LTX / MiniMax animation loop),
        optionally removes background, arranges into a grid, and generates both the
        master Atlas texture image and standard JSON texture coordinates (compatible
        with TexturePacker / PixiJS / Phaser / Unity).

        Args:
            asset_id: Asset ID of the source video (or image)
            frame_count: Number of animation frames to sample (default: 8)
            columns: Number of columns in the atlas grid (auto-calculated for square aspect if None)
            frame_width: Target scaled width of each frame in pixels (preserves aspect if None)
            frame_height: Target scaled height of each frame in pixels
            remove_bg: If True, automatically runs background removal on every frame before packing
            format: Output image format - "png" (lossless with alpha, default) or "webp"

        Returns:
            Dict containing new atlas asset_id, asset_url, dimensions, and full TexturePacker-format JSON metadata.
        """
        asset_record = asset_registry.get_asset(asset_id)
        if not asset_record:
            return {"error": f"Asset '{asset_id}' not found or expired"}

        raw_bytes = fetch_asset_bytes(asset_record, comfyui_client.base_url)
        if not raw_bytes:
            return {"error": f"Could not fetch asset data for '{asset_id}'"}

        is_video = (
            asset_record.mime_type.startswith("video")
            or asset_record.filename.lower().endswith((".mp4", ".webm", ".mov", ".avi", ".mkv"))
        )

        frames: List[Image.Image] = []

        if is_video:
            try:
                raw_frames = extract_video_keyframes(raw_bytes, num_frames=frame_count)
                for f_bytes, _ in raw_frames:
                    frames.append(Image.open(io.BytesIO(f_bytes)).convert("RGBA"))
            except Exception as e:
                logger.error(f"Failed to extract frames from video {asset_id}: {e}")
                return {"error": f"Video frame extraction failed: {str(e)}"}
        else:
            try:
                frames.append(Image.open(io.BytesIO(raw_bytes)).convert("RGBA"))
            except Exception as e:
                return {"error": f"Image decoding failed: {str(e)}"}

        if not frames:
            return {"error": "No frames could be extracted from the asset"}

        # Optional background removal on all frames
        if remove_bg:
            cleaned_frames = []
            for f in frames:
                buf = io.BytesIO()
                f.save(buf, format="PNG")
                mat_bytes = remove_image_background(buf.getvalue(), mode="auto")
                cleaned_frames.append(Image.open(io.BytesIO(mat_bytes)).convert("RGBA"))
            frames = cleaned_frames

        # Build sprite sheet
        try:
            atlas_bytes, atlas_meta = build_sprite_sheet(
                frames=frames,
                columns=columns,
                frame_width=frame_width,
                frame_height=frame_height,
                out_format=format,
            )
        except Exception as e:
            logger.error(f"Sprite sheet packing failed: {e}")
            return {"error": f"Sprite sheet packing failed: {str(e)}"}

        fmt_ext = "png" if format.lower() == "png" else "webp"
        base_stem = asset_record.filename.rsplit(".", 1)[0]
        out_filename = f"spritesheet_{base_stem}.{fmt_ext}"

        # Persist bytes first so the registered asset_url resolves.
        configured_root = getattr(getattr(publish_manager, "config", None), "comfyui_output_root", None)
        if not persist_processed_bytes(out_filename, atlas_bytes, asset_record.subfolder, configured_root):
            return {"error": "ComfyUI output root not configured; sprite sheet could not be persisted (set COMFYUI_OUTPUT_ROOT)"}

        # Register in asset registry
        new_record = asset_registry.register_asset(
            filename=out_filename,
            subfolder=asset_record.subfolder,
            folder_type=asset_record.folder_type,
            workflow_id="generate_sprite_sheet",
            prompt_id=asset_record.prompt_id,
            mime_type=f"image/{fmt_ext}",
            bytes_size=len(atlas_bytes),
            parent_asset_id=asset_record.asset_id,
            generation_type="sprite_sheet",
            prompt=getattr(asset_record, "prompt", None),
            tags=["spritesheet", "atlas", "gamedev", *(getattr(asset_record, "tags", []) or [])],
        )

        logger.info(f"Generated Sprite Sheet {new_record.asset_id} ({atlas_meta['meta']['frame_count']} frames, {atlas_meta['meta']['size']['w']}x{atlas_meta['meta']['size']['h']})")

        return {
            "asset_id": new_record.asset_id,
            "asset_url": new_record.asset_url,
            "filename": new_record.filename,
            "mime_type": f"image/{fmt_ext}",
            "bytes_size": new_record.bytes_size,
            "parent_asset_id": asset_record.asset_id,
            "generation_type": "sprite_sheet",
            "frame_count": atlas_meta["meta"]["frame_count"],
            "grid": f"{atlas_meta['meta']['rows']}x{atlas_meta['meta']['columns']}",
            "size": atlas_meta["meta"]["size"],
            "atlas_metadata": atlas_meta,
            "status": "success",
        }

    # If pipeline_orchestrator is available, register composite pipeline tools
    if pipeline_orchestrator is not None:
        @mcp.tool()
        def run_pipeline(
            steps: List[Dict[str, Any]],
            pipeline_name: Optional[str] = None,
        ) -> dict:
            """Execute a multi-stage composite pipeline in a single tool call.

            Automatically pipes intermediate outputs (asset_id / image filename)
            between sequential steps, links asset lineage trees, and optionally
            injects character profiles into generation stages.

            Example step definition:
            ```json
            [
              {
                "tool": "generate_image",
                "character_id": "detective_john",
                "params": {"prompt": "investigating crime scene"}
              },
              {
                "tool": "remove_background",
                "input_from": "previous",
                "params": {"mode": "auto"}
              },
              {
                "tool": "api_video_minimax_h3_i2v",
                "input_from": "previous",
                "params": {"prompt": "<Picture 1> detective drawing flashlight and walking"}
              }
            ]
            ```

            Args:
                steps: Ordered list of step dicts (`tool`, `params`, optional `input_from`, `character_id`)
                pipeline_name: Optional custom label for tracking the pipeline

            Returns:
                Execution result with per-step summaries, durations, and final asset.
            """
            return pipeline_orchestrator.execute_pipeline(steps=steps, pipeline_name=pipeline_name)

        @mcp.tool()
        def list_pipeline_recipes() -> dict:
            """List pre-packaged high-frequency pipeline recipes (e.g. T2I + 2K Upscale, Character to Sprite Sheet).

            Returns:
                List of curated multi-stage recipes with step descriptions and parameter schemas.
            """
            recipes = pipeline_orchestrator.list_recipes()
            return {
                "count": len(recipes),
                "recipes": recipes,
            }
