"""Asset viewing and multimodal perception tools for ComfyUI MCP Server"""

import logging
from typing import Optional

from fastmcp import FastMCP
from fastmcp.utilities.types import Image as FastMCPImage
from asset_processor import (
    analyze_audio_features,
    create_video_animated_gif,
    create_video_contact_sheet,
    encode_preview_for_mcp,
    estimate_response_chars,
    extract_video_metadata,
    fetch_asset_bytes,
    get_cache_key,
    render_waveform_image,
)

logger = logging.getLogger("MCP_Server")


def register_asset_tools(
    mcp: FastMCP,
    asset_registry
):
    """Register asset viewing and perception tools with the MCP server"""
    
    @mcp.tool()
    def view_image(
        asset_id: str,
        mode: str = "thumb",
        max_dim: Optional[int] = None,
        max_b64_chars: Optional[int] = None,
    ) -> dict:
        """View a generated image inline in chat (thumbnail preview only).
        
        This tool allows the AI agent to view generated assets inline in the chat interface,
        enabling closed-loop iteration: generate → view → adjust → regenerate.
        
        Automatically adapts to media types:
        - Images (PNG, JPEG, WebP, GIF): Resized thumbnail.
        - Videos (MP4, WebM, MOV): Multi-keyframe filmstrip with timestamps.
        - Audios (MP3, WAV, FLAC, OGG, M4A): Visual waveform diagram.
        
        Args:
            asset_id: Asset ID returned from generation tools (e.g., generate_image)
            mode: Display mode - "thumb" (thumbnail preview, default) or "metadata" (info only)
            max_dim: Maximum dimension in pixels (default: 512, hard cap)
            max_b64_chars: Maximum base64 character count (default: 100000, ~100KB)
        
        Returns:
            MCP ImageContent structure for inline display, or metadata dict if mode="metadata"
            or if image exceeds budget (refuse-inline branch).
        """
        # Cleanup expired assets periodically
        asset_registry.cleanup_expired()
        
        # Validate asset_id exists in registry
        asset_record = asset_registry.get_asset(asset_id)
        if not asset_record:
            return {"error": f"Asset {asset_id} not found. Generate a new asset to regenerate."}
        
        # Get asset URL (computed from stable identity)
        asset_url = asset_record.asset_url or asset_record.get_asset_url(asset_registry.comfyui_base_url)
        
        # If metadata mode, return info only
        if mode == "metadata":
            return {
                "asset_id": asset_record.asset_id,
                "asset_url": asset_url,
                "filename": asset_record.filename,
                "subfolder": asset_record.subfolder,
                "folder_type": asset_record.folder_type,
                "mime_type": asset_record.mime_type,
                "width": asset_record.width,
                "height": asset_record.height,
                "bytes_size": asset_record.bytes_size,
                "workflow_id": asset_record.workflow_id,
                "prompt_id": asset_record.prompt_id,
                "parent_asset_id": asset_record.parent_asset_id,
                "root_asset_id": asset_record.root_asset_id,
                "generation_type": asset_record.generation_type,
                "prompt": asset_record.prompt,
                "created_at": asset_record.created_at.isoformat(),
                "expires_at": asset_record.expires_at.isoformat() if asset_record.expires_at else None
            }
        
        if mode != "thumb":
            return {
                "error": f"Mode '{mode}' not supported. Use 'thumb' or 'metadata'."
            }

        is_video = (
            (asset_record.mime_type and asset_record.mime_type.startswith("video/"))
            or asset_record.filename.lower().endswith((".mp4", ".webm", ".mov", ".avi", ".mkv"))
        )

        # 1. Video asset automatic keyframe contact sheet preview
        if is_video:
            try:
                video_bytes = fetch_asset_bytes(asset_url)
                sheet_bytes = create_video_contact_sheet(
                    video_bytes,
                    num_frames=4,
                    max_width=max_dim or 1024,
                    quality=75
                )
                logger.info(f"view_image generated video contact sheet for {asset_id} ({len(sheet_bytes)} bytes)")
                return FastMCPImage(data=sheet_bytes, format="webp")
            except Exception as e:
                logger.warning(f"Failed to generate video contact sheet preview for {asset_id}: {e}")
                return {
                    "status": "unsupported_inline",
                    "asset_id": asset_id,
                    "asset_url": asset_url,
                    "mime_type": asset_record.mime_type,
                    "filename": asset_record.filename,
                    "error": str(e),
                    "message": "Failed to decode video preview. Open asset_url directly."
                }

        is_audio = (
            (asset_record.mime_type and asset_record.mime_type.startswith("audio/"))
            or asset_record.filename.lower().endswith((".mp3", ".wav", ".flac", ".ogg", ".aac", ".m4a"))
        )

        # 2. Audio asset automatic visual waveform preview
        if is_audio:
            try:
                audio_bytes = fetch_asset_bytes(asset_url)
                wf_bytes = render_waveform_image(audio_bytes, width=max_dim or 512, height=128)
                logger.info(f"view_image generated audio waveform for {asset_id} ({len(wf_bytes)} bytes)")
                return FastMCPImage(data=wf_bytes, format="webp")
            except Exception as e:
                logger.warning(f"Failed to generate audio waveform preview for {asset_id}: {e}")
                return {
                    "status": "unsupported_inline",
                    "asset_id": asset_id,
                    "asset_url": asset_url,
                    "mime_type": asset_record.mime_type,
                    "filename": asset_record.filename,
                    "error": str(e),
                    "message": "Failed to render audio waveform. Open asset_url directly."
                }
        
        # 3. Standard image content type
        supported_types = ("image/png", "image/jpeg", "image/jpg", "image/webp", "image/gif")
        if asset_record.mime_type not in supported_types:
            return {
                "status": "unsupported_inline",
                "asset_id": asset_id,
                "asset_url": asset_url,
                "mime_type": asset_record.mime_type,
                "filename": asset_record.filename,
                "message": f"Asset is '{asset_record.mime_type}', not supported for image preview."
            }
        
        if max_dim is None:
            max_dim = 512
        if max_b64_chars is None:
            max_b64_chars = 100_000
        
        try:
            image_bytes = fetch_asset_bytes(asset_url)
            cache_key = get_cache_key(asset_id, max_dim, 70)
            encoded = encode_preview_for_mcp(
                image_bytes,
                max_dim=max_dim,
                max_b64_chars=max_b64_chars,
                quality=70,
                cache_key=cache_key,
            )
            
            logger.info(
                f"view_image success: asset_id={asset_id} "
                f"src={asset_record.bytes_size}B src_dims={asset_record.width}x{asset_record.height} "
                f"preview_dims={encoded.size_px[0]}x{encoded.size_px[1]} format=webp "
                f"encoded={encoded.bytes_len}B b64_chars={encoded.b64_chars} "
                f"response_est={estimate_response_chars(encoded.b64_chars)}chars"
            )
            
            return FastMCPImage(data=encoded.raw_bytes, format="webp")
            
        except ValueError as e:
            logger.warning(f"Refusing to inline image for {asset_id}: {e}")
            return {
                "content": [{
                    "type": "text",
                    "text": (
                        f"Could not inline image (exceeds budget: {e}). "
                        f"Asset ID: {asset_id}. URL: {asset_url}. "
                        f"Hint: Open URL locally or use metadata mode."
                    )
                }]
            }
        except ImportError as e:
            return {"error": f"Image processing not available: {e}. Install Pillow: pip install Pillow"}
        except Exception as e:
            logger.exception(f"Failed to view image {asset_id}")
            return {"error": str(e)}

    @mcp.tool()
    def view_video_preview(
        asset_id: str,
        mode: str = "strip",
        num_frames: int = 4,
        max_dim: Optional[int] = None
    ) -> dict:
        """View a generated video asset inline via keyframe contact sheet strip, animated GIF, or metadata.
        
        Allows the AI to inspect motion continuity, lighting, and consistency across time.
        
        Args:
            asset_id: Video Asset ID returned from generation tools
            mode: Preview mode - "strip" (multi-keyframe contact sheet with timestamps, default), "gif" (animated GIF loop), or "metadata" (duration/fps/dimensions)
            num_frames: Number of keyframes to extract for contact sheet (default: 4)
            max_dim: Maximum dimension in pixels (default: 1024 for strip, 256 for gif)
            
        Returns:
            FastMCPImage preview object or metadata dictionary.
        """
        asset_registry.cleanup_expired()
        asset_record = asset_registry.get_asset(asset_id)
        if not asset_record:
            return {"error": f"Asset {asset_id} not found."}

        asset_url = asset_record.asset_url or asset_record.get_asset_url(asset_registry.comfyui_base_url)

        try:
            video_bytes = fetch_asset_bytes(asset_url)

            if mode == "metadata":
                v_meta = extract_video_metadata(video_bytes)
                return {
                    "asset_id": asset_record.asset_id,
                    "asset_url": asset_url,
                    "filename": asset_record.filename,
                    "workflow_id": asset_record.workflow_id,
                    "duration_sec": v_meta.get("duration_sec"),
                    "fps": v_meta.get("fps"),
                    "total_frames": v_meta.get("total_frames"),
                    "width": v_meta.get("width") or asset_record.width,
                    "height": v_meta.get("height") or asset_record.height,
                    "has_audio": v_meta.get("has_audio", False),
                    "prompt": asset_record.prompt,
                }
            elif mode == "gif":
                gif_dim = max_dim or 256
                gif_bytes = create_video_animated_gif(video_bytes, max_frames=16, target_fps=8, max_dim=gif_dim)
                logger.info(f"view_video_preview generated animated GIF for {asset_id} ({len(gif_bytes)} bytes)")
                return FastMCPImage(data=gif_bytes, format="gif")
            elif mode == "strip":
                strip_dim = max_dim or 1024
                sheet_bytes = create_video_contact_sheet(
                    video_bytes,
                    num_frames=num_frames,
                    max_width=strip_dim,
                    quality=75
                )
                logger.info(f"view_video_preview generated filmstrip for {asset_id} ({len(sheet_bytes)} bytes)")
                return FastMCPImage(data=sheet_bytes, format="webp")
            else:
                return {"error": f"Unsupported mode '{mode}'. Use 'strip', 'gif', or 'metadata'."}

        except Exception as e:
            logger.exception(f"Failed to generate video preview for {asset_id}")
            return {"error": f"Failed to generate video preview: {str(e)}"}

    @mcp.tool()
    def view_audio_preview(
        asset_id: str,
        mode: str = "waveform",
        max_dim: Optional[int] = None
    ) -> dict:
        """View a generated audio asset inline via visual waveform diagram or feature analysis.
        
        Args:
            asset_id: Audio Asset ID returned from generation tools (e.g. generate_song)
            mode: Preview mode - "waveform" (visual diagram, default) or "analysis" (detailed features)
            max_dim: Maximum width in pixels for waveform image (default: 512)
            
        Returns:
            FastMCPImage waveform object or structured analysis dictionary.
        """
        asset_registry.cleanup_expired()
        asset_record = asset_registry.get_asset(asset_id)
        if not asset_record:
            return {"error": f"Asset {asset_id} not found."}

        asset_url = asset_record.asset_url or asset_record.get_asset_url(asset_registry.comfyui_base_url)

        try:
            audio_bytes = fetch_asset_bytes(asset_url)

            if mode == "waveform":
                wf_bytes = render_waveform_image(audio_bytes, width=max_dim or 512, height=128)
                logger.info(f"view_audio_preview generated waveform for {asset_id} ({len(wf_bytes)} bytes)")
                return FastMCPImage(data=wf_bytes, format="webp")
            elif mode == "analysis":
                features = analyze_audio_features(audio_bytes, prompt_lyrics=asset_record.prompt)
                features.update({
                    "asset_id": asset_record.asset_id,
                    "asset_url": asset_url,
                    "filename": asset_record.filename,
                    "workflow_id": asset_record.workflow_id
                })
                return features
            else:
                return {"error": f"Unsupported mode '{mode}'. Use 'waveform' or 'analysis'."}

        except Exception as e:
            logger.exception(f"Failed to generate audio preview for {asset_id}")
            return {"error": f"Failed to generate audio preview: {str(e)}"}

    @mcp.tool()
    def analyze_audio(asset_id: str) -> dict:
        """Analyze audio features including tempo (BPM), loudness (RMS/Peak dBFS), silent intervals, and lyrics alignment.
        
        Helps the AI Agent verify track structure, check for empty audio / silence issues,
        and assess whether regeneration or volume adjustment is required.
        
        Args:
            asset_id: Audio Asset ID
            
        Returns:
            Dict containing duration, sample_rate, estimated_bpm, rms_dbfs, peak_dbfs,
            silent_segments, lyrics_sections with timestamps, and waveform summary.
        """
        asset_registry.cleanup_expired()
        asset_record = asset_registry.get_asset(asset_id)
        if not asset_record:
            return {"error": f"Asset {asset_id} not found."}

        asset_url = asset_record.asset_url or asset_record.get_asset_url(asset_registry.comfyui_base_url)

        try:
            audio_bytes = fetch_asset_bytes(asset_url)
            analysis = analyze_audio_features(audio_bytes, prompt_lyrics=asset_record.prompt)
            analysis.update({
                "asset_id": asset_record.asset_id,
                "asset_url": asset_url,
                "filename": asset_record.filename,
                "workflow_id": asset_record.workflow_id,
                "prompt": asset_record.prompt
            })
            return analysis
        except Exception as e:
            logger.exception(f"Failed to analyze audio for {asset_id}")
            return {"error": f"Failed to analyze audio: {str(e)}"}
