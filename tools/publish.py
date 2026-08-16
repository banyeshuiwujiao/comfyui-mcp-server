"""Publish tools for safely publishing ComfyUI assets to web project directories"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastmcp import FastMCP

from asset_processor import fetch_asset_bytes, get_image_metadata
from managers.publish_manager import (
    PublishManager,
    auto_generate_filename,
    validate_manifest_key,
    validate_target_filename
)

logger = logging.getLogger("MCP_Server")


def _sanitize_export_filename(filename: str) -> str:
    """Keep only the basename so a caller can never escape the target directory."""
    name = Path(filename or "").name.strip()
    if not name:
        return ""
    if name != filename or name in (".", ".."):
        return ""
    if not name.lower().endswith((".png", ".webp", ".jpg", ".jpeg")):
        return ""
    return name


def _update_godot_manifest(target_dir: Path, entry: dict, category: Optional[str]) -> dict:
    """Merge an export entry into ``target_dir/manifest.json`` (lineage ledger).

    The Godot data-flywheel convention keeps a manifest next to the resource
    directory so every AI asset stays traceable: prompt, workflow id, ComfyUI
    asset ids, pipeline chain and file hash are all recorded here.
    """
    manifest_path = target_dir / "manifest.json"
    manifest: dict = {}
    if manifest_path.exists():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                manifest = loaded
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Could not read existing Godot manifest %s: %s", manifest_path, e)

    if "assets" not in manifest or not isinstance(manifest.get("assets"), list):
        manifest["assets"] = []
    manifest["assets"] = [item for item in manifest["assets"] if item.get("file") != entry.get("file")]
    manifest["assets"].append(entry)

    # Sort for stable diffs; enrich the header with the latest export.
    manifest["assets"].sort(key=lambda item: item.get("file", ""))
    manifest["generator"] = "comfyui-mcp-server (streamable-http @127.0.0.1:9000/mcp)"
    if category:
        manifest["category"] = category
    manifest["last_export_at"] = datetime.now(timezone.utc).isoformat()

    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _publish_asset_to_godot(
    asset_registry,
    publish_manager,
    asset_id: str,
    target_dir: str,
    target_filename: str,
    category: Optional[str] = None,
    overwrite: bool = True,
) -> dict:
    """Shared Godot-direct publishing path (single source of truth).

    Used by both ``publish_asset(target_dir=...)`` and the thin
    ``export_to_godot`` alias, so the two tools can never diverge again.
    """
    asset_record = asset_registry.get_asset(asset_id)
    if not asset_record:
        return {
            "error": f"Asset {asset_id} not found or expired",
            "error_code": "ASSET_NOT_FOUND_OR_EXPIRED",
        }

    filename = _sanitize_export_filename(target_filename)
    if not filename:
        return {
            "error": "target_filename must be a plain basename ending in .png/.webp/.jpg/.jpeg",
            "error_code": "INVALID_TARGET_FILENAME",
        }

    target_path = Path(target_dir)
    try:
        target_path = target_path.resolve(strict=True)
    except (OSError, RuntimeError) as e:
        return {
            "error": f"target_dir does not exist or cannot be resolved: {target_dir} ({e})",
            "error_code": "INVALID_TARGET_DIR",
        }
    if not target_path.is_dir():
        return {
            "error": f"target_dir is not a directory: {target_dir}",
            "error_code": "INVALID_TARGET_DIR",
        }

    try:
        raw_bytes = fetch_asset_bytes(asset_record, publish_manager.config.comfyui_url)
    except Exception as e:  # noqa: BLE001 - surface as structured tool error
        return {
            "error": f"Could not fetch asset bytes: {e}",
            "error_code": "FETCH_FAILED",
        }
    if not raw_bytes:
        return {
            "error": f"Could not fetch asset bytes for {asset_id}",
            "error_code": "FETCH_FAILED",
        }

    dest_path = target_path / filename
    if dest_path.exists() and not overwrite:
        return {
            "error": f"Destination already exists (pass overwrite=true to replace): {dest_path}",
            "error_code": "DEST_EXISTS",
        }

    try:
        dest_path.write_bytes(raw_bytes)
    except OSError as e:
        return {
            "error": f"Could not write destination file: {e}",
            "error_code": "WRITE_FAILED",
        }

    image_meta = get_image_metadata(raw_bytes) or {}
    size = [image_meta.get("width"), image_meta.get("height")]

    is_matting = getattr(asset_record, "generation_type", None) == "matting"
    workflow_hash = (asset_record.metadata or {}).get("workflow_hash")
    if not workflow_hash and is_matting and asset_record.parent_asset_id:
        parent_record = asset_registry.get_asset(asset_record.parent_asset_id)
        workflow_hash = (parent_record.metadata or {}).get("workflow_hash") if parent_record else None

    entry = {
        "file": filename,
        "comfy_file": asset_record.filename,
        "asset_id": asset_record.asset_id,
        "source_asset_id": asset_record.parent_asset_id if is_matting else asset_record.asset_id,
        "matting_asset_id": asset_record.asset_id if is_matting else None,
        "workflow_id": asset_record.workflow_id,
        "size": size,
        "prompt": getattr(asset_record, "prompt", None),
        "workflow_hash": workflow_hash,
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }
    category = category or target_path.name
    manifest = _update_godot_manifest(target_path, entry, category)

    return {
        "status": "success",
        "dest_path": str(dest_path),
        "bytes_size": len(raw_bytes),
        "width": image_meta.get("width"),
        "height": image_meta.get("height"),
        "manifest_entry": entry,
        "manifest_path": str(target_path / "manifest.json"),
        "manifest_count": len(manifest.get("assets", [])),
    }


def register_publish_tools(
    mcp: FastMCP,
    asset_registry,
    publish_manager: PublishManager
):
    """Register publish tools with the MCP server"""
    
    @mcp.tool()
    def get_publish_info() -> dict:
        """Get publish configuration and status information.
        
        This tool helps debug configuration issues and surfaces wrong assumptions
        immediately. Call this first to verify publish setup before attempting to publish.
        
        Returns:
            Dict with:
            - project_root: Detected project root (path and detection method: "cwd" | "auto-detected")
            - publish_root: Publish directory (path, exists, writable)
            - comfyui_output_root: ComfyUI output root (path or None, detection method, configured flag)
            - comfyui_tried_paths: List of paths checked during detection with validation results
            - config_file: Path to persistent config file
            - status: "ready" | "needs_comfyui_root" | "error"
            - message: Human-readable status/instructions
            - warnings: List of warnings (e.g., "Using fallback project root detection")
            - error_code: Machine-readable error code (if not ready)
        """
        return publish_manager.get_publish_info()
    
    @mcp.tool()
    def set_comfyui_output_root(path: str) -> dict:
        """Set ComfyUI output root directory in persistent configuration.
        
        This tool configures the ComfyUI output directory once, and it will be remembered
        across server restarts. Use this when auto-detection fails or you want explicit control.
        
        The path is validated to ensure:
        - It exists and is a directory
        - It appears to be a ComfyUI output directory (contains ComfyUI_*.png files or output/temp subdirs)
        
        Configuration is stored in a platform-specific location:
        - Windows: %APPDATA%/comfyui-mcp-server/publish_config.json
        - Mac: ~/Library/Application Support/comfyui-mcp-server/publish_config.json
        - Linux: ~/.config/comfyui-mcp-server/publish_config.json
        
        Args:
            path: Absolute or relative path to ComfyUI output directory
                (e.g., "E:/comfyui-desktop/output" or "/opt/ComfyUI/output")
        
        Returns:
            Dict with:
            - success: True if configured successfully
            - path: Resolved absolute path
            - config_file: Path to config file where setting was saved
            - message: Human-readable status message
            - error: Error code if configuration failed
        """
        return publish_manager.set_comfyui_output_root(path)
    
    @mcp.tool()
    def publish_asset(
        asset_id: str,
        target_filename: Optional[str] = None,
        manifest_key: Optional[str] = None,
        web_optimize: bool = False,
        max_bytes: int = 600_000,
        overwrite: bool = True,
        target_dir: Optional[str] = None,
    ) -> dict:
        r"""Publish a ComfyUI-generated asset to a web project or Godot directory.

        Supports three modes:
        - **Demo mode**: Provide `target_filename` (e.g., "hero.png") - deterministic filename
        - **Library mode**: Omit `target_filename`, provide `manifest_key` - auto-generates filename
        - **Godot mode**: Provide `target_dir` - writes the raw asset directly into a Godot
          Resources directory and maintains the lineage ``manifest.json`` ledger. This is the
          exact same code path as ``export_to_godot`` (thin alias); both tools share
          ``_publish_asset_to_godot``.

        **Default behavior (web_optimize=False):**
        - Assets are copied as-is, preserving original format (typically PNG from ComfyUI)
        - No compression or format conversion
        - Original quality preserved

        **Web optimization (web_optimize=True):**
        - Images are converted to WebP format
        - Compression ladder applied to meet size limits
        - Useful for web deployment where smaller file sizes are important
        - Not supported in Godot mode (raw PNG is what Godot wants)

        **Safety guarantees:**
        - Only assets from current session can be published (asset_id must exist in registry)
        - Source path must be within ComfyUI output root (validated with real path resolution)
        - Target filename validated by strict regex (prevents path traversal)
        - All paths are canonicalized to prevent symlink/traversal attacks

        **Workflow:**
        1. Agent calls `generate_image` → gets `asset_id`
        2. Agent calls `list_assets(session_id=...)` to discover assets
        3. Agent calls `publish_asset(asset_id, target_filename="hero.png")` (demo mode, no compression)
           OR `publish_asset(asset_id, target_filename="hero.webp", web_optimize=True)` (with compression)
           OR `publish_asset(asset_id, manifest_key="hero")` (library mode)
           OR `publish_asset(asset_id, target_dir=".../Resources/Fx", target_filename="fx.png")` (Godot mode)
        4. Server validates, copies (and compresses if web_optimize=True), and updates manifest.json

        Args:
            asset_id: Asset ID from generation tools (session-scoped, dies on restart)
            target_filename: Optional target filename (e.g., "hero.png"). If omitted, auto-generated.
                Must match regex: ^[a-z0-9][a-z0-9._-]{0,63}\.(webp|png|jpg|jpeg)$
            manifest_key: Optional manifest key (required if target_filename omitted).
                Must match regex: ^[a-z0-9][a-z0-9._-]{0,63}$
            web_optimize: If True, convert to WebP and apply compression (default: False).
                Only used for images. When False, assets are copied as-is preserving original format.
            max_bytes: Maximum file size in bytes (default: 600000). Only used when web_optimize=True.
            overwrite: Whether to overwrite existing file (default: True)
            target_dir: Optional existing Godot Resources directory. When provided, switches to
                Godot mode: raw bytes + lineage manifest (same implementation as export_to_godot).
        
        Returns:
            Dict with published file info:
            - dest_url: Relative URL (e.g., "/gen/hero.png")
            - dest_path: Absolute path to published file
            - bytes_size: File size in bytes
            - mime_type: MIME type (e.g., "image/png" or "image/webp")
            - width: Image width (if available)
            - height: Image height (if available)
            - compression_info: Compression details (only present if web_optimize=True and compression was applied)
        
        Raises:
            Error dict with "error" and "error_code" keys if:
            - Asset not found or expired (ASSET_NOT_FOUND_OR_EXPIRED)
            - Invalid target_filename format (INVALID_TARGET_FILENAME)
            - manifest_key required but missing (MANIFEST_KEY_REQUIRED)
            - Source path outside ComfyUI outputs (SOURCE_PATH_OUTSIDE_ROOT)
            - Path traversal attempt detected (PATH_TRAVERSAL_DETECTED)
            - Copy/compression operation fails (PUBLISH_FAILED)
        """
        # Cleanup expired assets
        asset_registry.cleanup_expired()
        
        # Lookup asset in registry (session-scoped)
        asset_record = asset_registry.get_asset(asset_id)
        if not asset_record:
            return {
                "error": f"Asset {asset_id} not found or expired. Assets are session-scoped and die on server restart. Generate a new asset in the current session.",
                "error_code": "ASSET_NOT_FOUND_OR_EXPIRED"
            }

        # Godot mode: same shared implementation as export_to_godot. It fetches
        # bytes over /view and writes to an arbitrary existing Godot directory,
        # so the web publish-root readiness check below does not apply.
        if target_dir:
            if web_optimize:
                return {
                    "error": "web_optimize is not supported in Godot mode; publish raw PNG (Godot imports WebP itself if needed)",
                    "error_code": "WEB_OPTIMIZE_UNSUPPORTED_FOR_GODOT_MODE",
                }
            if not target_filename:
                if not manifest_key:
                    return {
                        "error": "Godot mode requires target_filename (or manifest_key to derive the filename)",
                        "error_code": "MANIFEST_KEY_REQUIRED",
                    }
                source_ext = Path(asset_record.filename).suffix.lower().lstrip(".") or "png"
                target_filename = f"{manifest_key}.{source_ext}"
            return _publish_asset_to_godot(
                asset_registry=asset_registry,
                publish_manager=publish_manager,
                asset_id=asset_id,
                target_dir=target_dir,
                target_filename=target_filename,
                category=manifest_key,
                overwrite=overwrite,
            )
        
        # Web modes require the publish root to be ready.
        is_ready, error_code, error_info = publish_manager.ensure_ready()
        if not is_ready:
            return {
                "error": error_info.get("message", "Publish manager not ready"),
                "error_code": error_code,
                **error_info
            }

        try:
            # Resolve source path from asset metadata
            source_path = publish_manager.resolve_source_path(
                subfolder=asset_record.subfolder,
                filename=asset_record.filename
            )
            
            # Determine source format from asset filename
            source_ext = Path(asset_record.filename).suffix.lower().lstrip(".")
            
            # Determine target filename
            if target_filename:
                # Demo mode: use provided filename
                if not validate_target_filename(target_filename):
                    return {
                        "error": f"Invalid target_filename: '{target_filename}'. Must match regex: ^[a-z0-9][a-z0-9._-]{{0,63}}\\.(webp|png|jpg|jpeg)$",
                        "error_code": "INVALID_TARGET_FILENAME"
                    }
                final_target_filename = target_filename
            else:
                # Library mode: auto-generate filename, manifest_key is required
                if not manifest_key:
                    return {
                        "error": "manifest_key is required when target_filename is omitted (library mode). Provide either target_filename or manifest_key.",
                        "error_code": "MANIFEST_KEY_REQUIRED"
                    }
                if not validate_manifest_key(manifest_key):
                    return {
                        "error": f"Invalid manifest_key: '{manifest_key}'. Must match regex: ^[a-z0-9][a-z0-9._-]{{0,63}}$",
                        "error_code": "INVALID_MANIFEST_KEY"
                    }
                # Auto-generate filename based on web_optimize and source format
                if web_optimize:
                    final_target_filename = auto_generate_filename(asset_id, format="webp")
                else:
                    # Use source format for auto-generated filename
                    final_target_filename = auto_generate_filename(asset_id, format=source_ext if source_ext else "png")
            
            # Resolve target path
            target_path = publish_manager.resolve_target_path(final_target_filename)
            
            # Copy asset with optional compression
            publish_info = publish_manager.copy_asset(
                source_path=source_path,
                target_path=target_path,
                overwrite=overwrite,
                asset_id=asset_id,
                target_filename=final_target_filename,
                web_optimize=web_optimize,
                max_bytes=max_bytes
            )
            
            # Update manifest.json if manifest_key provided
            if manifest_key:
                try:
                    publish_manager.update_manifest(
                        manifest_key=manifest_key,
                        filename=target_path.name
                    )
                except Exception as e:
                    # Manifest update failure is non-fatal
                    logger.warning(f"Failed to update manifest for key {manifest_key}: {e}")
            
            # Add image dimensions if available
            result = {
                "dest_url": publish_info["dest_url"],
                "dest_path": publish_info["dest_path"],
                "bytes_size": publish_info["bytes_size"],
                "mime_type": publish_info["mime_type"]
            }
            
            if "compression_info" in publish_info:
                result["compression_info"] = publish_info["compression_info"]
            
            if asset_record.width and asset_record.height:
                result["width"] = asset_record.width
                result["height"] = asset_record.height
            
            logger.info(
                f"Published asset {asset_id} to {final_target_filename}: {publish_info['dest_url']} "
                f"({publish_info['bytes_size']} bytes)"
            )
            
            return result
            
        except ValueError as e:
            # Path validation errors
            error_msg = str(e)
            if "outside" in error_msg.lower() or "COMFYUI_OUTPUT_ROOT" in error_msg:
                error_code = "SOURCE_PATH_OUTSIDE_ROOT"
            elif "traversal" in error_msg.lower() or ".." in error_msg:
                error_code = "PATH_TRAVERSAL_DETECTED"
            elif "Invalid target_filename" in error_msg:
                error_code = "INVALID_TARGET_FILENAME"
            else:
                error_code = "VALIDATION_ERROR"
            
            return {
                "error": error_msg,
                "error_code": error_code
            }
        except Exception as e:
            logger.exception(f"Failed to publish asset {asset_id}")
            return {
                "error": f"Failed to publish asset: {str(e)}",
                "error_code": "PUBLISH_FAILED"
            }

    @mcp.tool()
    def export_to_godot(
        asset_id: str,
        target_dir: str,
        target_filename: str,
        category: Optional[str] = None,
        overwrite: bool = True,
    ) -> dict:
        """Export a generated asset directly into a Godot Resources directory.

        Thin alias over ``publish_asset(target_dir=...)`` — both tools execute
        the exact same ``_publish_asset_to_godot`` implementation, so the web
        publisher and the Godot flywheel exporter cannot diverge.

        This is the "step 3" tool of the OneQi data flywheel: it fetches the
        asset bytes from ComfyUI, writes them into the requested Godot resource
        directory, and maintains a ``manifest.json`` ledger next to the file.
        The ledger records the ComfyUI asset ids (source + matting), workflow id,
        prompt, workflow template hash and pixel size, so every AI-generated PNG
        inside the Godot repository stays traceable and reproducible.

        Args:
            asset_id: ComfyUI asset id (from any generation/matting tool).
            target_dir: Existing absolute directory in the Godot project, e.g.
                ``E:/Desktop/MiniGame/OneQi-ET8.1/ET.Client/OneQi/MainPack/Resources/Fx``.
            target_filename: Output filename, basename only (e.g. ``fx_victory.png``).
                Must end in .png/.webp/.jpg/.jpeg; path separators are rejected.
            category: Optional category label written into the manifest header
                (e.g. "Fx", "Backgrounds"). Defaults to the directory basename.
            overwrite: Replace an existing file (default True).

        Returns:
            Dict with dest_path, bytes_size, width, height and the full
            manifest entry that was written.
        """
        return _publish_asset_to_godot(
            asset_registry=asset_registry,
            publish_manager=publish_manager,
            asset_id=asset_id,
            target_dir=target_dir,
            target_filename=target_filename,
            category=category,
            overwrite=overwrite,
        )
