"""Modular Subgraph Pipeline Orchestrator.

Allows AI Agents to chain multiple ComfyUI tools and workflows into a single
composite pipeline with automatic dataflow passing, character profile injection,
and step-by-step lineage tracking.
"""

import copy
import logging
import time
from typing import Any, Dict, List, Optional

from asset_processor import remove_image_background, build_sprite_sheet
from models.workflow import WorkflowToolDefinition
from tools.helpers import register_and_build_response

logger = logging.getLogger("MCP_Server__pipeline_orchestrator")

# Standard pre-packaged pipeline recipes
PIPELINE_RECIPES: List[Dict[str, Any]] = [
    {
        "recipe_id": "t2i_to_2k_upscale",
        "name": "Text-to-Image with 2K Upscaling",
        "description": "Generate base image and immediately upscale to 2K resolution with enhanced details.",
        "steps": [
            {"step_name": "base_generation", "tool": "api_image_z_image_turbo_t2i", "params": {"prompt": "<your_prompt>"}},
            {"step_name": "upscale_2k", "tool": "api_utility_z_image_turbo_2k_upscaler", "input_from": "previous", "params": {"prompt": "<your_prompt>"}}
        ]
    },
    {
        "recipe_id": "t2i_to_transparent_sticker",
        "name": "Transparent Sticker / Game Asset Extraction",
        "description": "Generate subject on clean background and extract transparent RGBA PNG.",
        "steps": [
            {"step_name": "generate_subject", "tool": "generate_image", "params": {"prompt": "<your_prompt>, solid white background"}},
            {"step_name": "matting", "tool": "remove_background", "input_from": "previous", "params": {"mode": "auto"}}
        ]
    },
    {
        "recipe_id": "character_to_sprite_sheet",
        "name": "Character Action Loop to Sprite Sheet",
        "description": "Generate character, animate into an action loop, and package into a game engine texture atlas.",
        "steps": [
            {"step_name": "character_portrait", "tool": "generate_image", "params": {"prompt": "<your_prompt>"}},
            {"step_name": "animate_i2v", "tool": "api_video_minimax_h3_i2v", "input_from": "previous", "params": {"prompt": "<Picture 1> walking forward animation loop"}},
            {"step_name": "atlas_pack", "tool": "generate_sprite_sheet", "input_from": "previous", "params": {"frame_count": 8, "columns": 4, "remove_bg": True}}
        ]
    },
    {
        "recipe_id": "character_sheet_multiview",
        "name": "Character Multiview Sheet",
        "description": "Generate reference character and produce 6-angle views using Qwen multiview workflow.",
        "steps": [
            {"step_name": "reference_view", "tool": "generate_image", "params": {"prompt": "<your_prompt>"}},
            {"step_name": "multiview_angles", "tool": "api_qwen_image_edit_2511_1_click_multiple_character_angles-v1.0", "input_from": "previous", "params": {"prompt": "front view, side view, back view, 3/4 view"}}
        ]
    }
]


class PipelineOrchestrator:
    """Orchestrates multi-stage workflow pipelines."""

    def __init__(
        self,
        comfyui_client,
        asset_registry,
        workflow_manager,
        defaults_manager=None,
        character_vault=None,
        error_diagnoser=None,
        gpu_guard=None,
    ):
        self.comfyui_client = comfyui_client
        self.asset_registry = asset_registry
        self.workflow_manager = workflow_manager
        self.defaults_manager = defaults_manager
        self.character_vault = character_vault
        self.error_diagnoser = error_diagnoser
        self.gpu_guard = gpu_guard

    def list_recipes(self) -> List[Dict[str, Any]]:
        """Return available pre-configured pipeline recipes."""
        return copy.deepcopy(PIPELINE_RECIPES)

    def execute_pipeline(
        self,
        steps: List[Dict[str, Any]],
        pipeline_name: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute a multi-stage composite pipeline sequentially.

        Args:
            steps: List of step dictionaries. Each step can define:
                   - `tool` or `workflow`: Tool/Workflow identifier
                   - `step_name`: Optional label for the step
                   - `params`: Step parameters dict
                   - `input_from`: "previous", "step_0", or "asset:<uuid>"
                   - `character_id`: Optional character profile to inject
            pipeline_name: Optional name for logging/tracking
            session_id: Optional conversation session identifier

        Returns:
            Dict with execution status, completed step details, and final asset summary.
        """
        if not steps:
            return {"error": "Pipeline cannot be empty; provide at least one step"}

        pipeline_label = pipeline_name or f"pipeline_{int(time.time())}"
        logger.info(f"Starting pipeline execution: {pipeline_label} ({len(steps)} steps)")

        completed_steps: List[Dict[str, Any]] = []
        step_assets: Dict[str, Any] = {}  # step_name or "step_N" -> asset_dict
        latest_asset: Optional[Dict[str, Any]] = None

        t_pipeline_start = time.time()

        for step_idx, step_def in enumerate(steps):
            step_name = step_def.get("step_name") or f"step_{step_idx}"
            tool_identifier = step_def.get("tool") or step_def.get("workflow") or step_def.get("module")
            if not tool_identifier:
                return {
                    "error": f"Step #{step_idx} missing 'tool' or 'workflow' identifier",
                    "failing_step_index": step_idx,
                    "completed_steps": completed_steps,
                }

            step_params = copy.deepcopy(step_def.get("params") or {})
            t_step_start = time.time()

            # 1. Resolve character profile injection
            char_id = step_def.get("character_id") or step_params.pop("character_id", None)
            if char_id and self.character_vault:
                prompt_val = step_params.get("prompt", "")
                neg_val = step_params.get("negative_prompt", "")
                char_res = self.character_vault.apply_character(char_id, prompt_val, neg_val)
                if "error" not in char_res:
                    step_params["prompt"] = char_res["prompt"]
                    if char_res.get("negative_prompt"):
                        step_params["negative_prompt"] = char_res["negative_prompt"]
                    if char_res.get("lora_name") and "lora_name" not in step_params:
                        step_params["lora_name"] = char_res["lora_name"]
                        step_params["lora_strength"] = char_res.get("lora_strength", 0.75)

            # 2. Resolve input asset dependency (Dataflow pipe)
            input_source = step_def.get("input_from")
            source_asset = None

            if input_source == "previous" or (input_source is None and latest_asset is not None and step_idx > 0):
                source_asset = latest_asset
            elif input_source and input_source.startswith("asset:"):
                raw_aid = input_source[6:]
                rec = self.asset_registry.get_asset(raw_aid)
                if rec:
                    source_asset = {
                        "asset_id": rec.asset_id,
                        "filename": rec.filename,
                        "subfolder": rec.subfolder,
                        "folder_type": rec.folder_type,
                        "asset_url": rec.asset_url,
                        "mime_type": rec.mime_type,
                    }
            elif input_source in step_assets:
                source_asset = step_assets[input_source]
            elif f"step_{input_source}" in step_assets:
                source_asset = step_assets[f"step_{input_source}"]

            # Inject source asset into step parameters
            if source_asset:
                src_filename = source_asset.get("filename")
                src_asset_id = source_asset.get("asset_id")

                # If the tool expects asset_id (e.g. remove_background, generate_sprite_sheet)
                if tool_identifier in ("remove_background", "generate_sprite_sheet", "regenerate"):
                    if "asset_id" not in step_params:
                        step_params["asset_id"] = src_asset_id
                else:
                    # Workflow tools usually expect image / image2 / input_image
                    if "image" not in step_params and src_filename:
                        step_params["image"] = src_filename
                    if "input_image" not in step_params and src_filename:
                        step_params["input_image"] = src_filename

                # Maintain parent lineage link
                step_params.setdefault("parent_asset_id", src_asset_id)

            # 3. Execute step
            try:
                step_result = self._execute_step(tool_identifier, step_params, session_id=session_id)
            except Exception as e:
                logger.error(f"Pipeline '{pipeline_label}' step #{step_idx} ({step_name}) raised exception: {e}")
                diag = None
                if self.error_diagnoser:
                    diag = self.error_diagnoser.diagnose_error(
                        error_message=str(e),
                        history_prompt_id=None,
                        submitted_params=step_params,
                    )
                return {
                    "status": "failed",
                    "pipeline_name": pipeline_label,
                    "failing_step_index": step_idx,
                    "failing_step_name": step_name,
                    "failing_tool": tool_identifier,
                    "error": str(e),
                    "diagnosis": diag,
                    "completed_steps": completed_steps,
                }

            if not isinstance(step_result, dict) or "error" in step_result or step_result.get("isError"):
                err_msg = step_result.get("error") if isinstance(step_result, dict) else str(step_result)
                logger.error(f"Pipeline '{pipeline_label}' step #{step_idx} failed: {err_msg}")
                return {
                    "status": "failed",
                    "pipeline_name": pipeline_label,
                    "failing_step_index": step_idx,
                    "failing_step_name": step_name,
                    "failing_tool": tool_identifier,
                    "error": err_msg,
                    "step_response": step_result,
                    "completed_steps": completed_steps,
                }

            # 4. Record step success
            step_duration = round(time.time() - t_step_start, 2)
            step_asset_id = step_result.get("asset_id")
            step_asset_url = step_result.get("asset_url") or step_result.get("image_url")
            step_filename = step_result.get("filename")

            step_summary = {
                "step_index": step_idx,
                "step_name": step_name,
                "tool": tool_identifier,
                "asset_id": step_asset_id,
                "asset_url": step_asset_url,
                "filename": step_filename,
                "duration_sec": step_duration,
                "status": "success",
            }
            completed_steps.append(step_summary)

            if step_asset_id:
                latest_asset = {
                    "asset_id": step_asset_id,
                    "filename": step_filename,
                    "subfolder": step_result.get("subfolder", ""),
                    "folder_type": step_result.get("folder_type", "output"),
                    "asset_url": step_asset_url,
                    "mime_type": step_result.get("mime_type", "image/png"),
                }
                step_assets[step_name] = latest_asset
                step_assets[f"step_{step_idx}"] = latest_asset

        total_duration = round(time.time() - t_pipeline_start, 2)
        logger.info(f"Pipeline '{pipeline_label}' completed successfully in {total_duration}s")

        return {
            "status": "success",
            "pipeline_name": pipeline_label,
            "total_steps": len(steps),
            "total_duration_sec": total_duration,
            "completed_steps": completed_steps,
            "final_asset": latest_asset,
        }

    def _execute_step(
        self,
        tool_identifier: str,
        params: Dict[str, Any],
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Internal router for executing a single pipeline step."""
        # 1. Post-processing tool: remove_background
        if tool_identifier == "remove_background":
            from tools.pipeline import register_pipeline_tools
            # Invoke remove_background logic directly
            asset_id = params.get("asset_id")
            if not asset_id:
                raise ValueError("remove_background step requires an 'asset_id'")

            rec = self.asset_registry.get_asset(asset_id)
            if not rec:
                raise ValueError(f"Asset '{asset_id}' not found")

            from asset_processor import fetch_asset_bytes
            raw_bytes = fetch_asset_bytes(rec, self.comfyui_client.base_url)
            if not raw_bytes:
                raise ValueError(f"Could not fetch asset bytes for '{asset_id}'")

            trans_bytes = remove_image_background(
                raw_bytes,
                mode=params.get("mode", "auto"),
                bgcolor=params.get("bgcolor"),
                tolerance=params.get("tolerance", 32),
                feather=params.get("feather", 2),
            )

            base_stem = rec.filename.rsplit(".", 1)[0]
            new_rec = self.asset_registry.register_asset(
                filename=f"transparent_{base_stem}.png",
                subfolder=rec.subfolder,
                folder_type=rec.folder_type,
                workflow_id="remove_background",
                prompt_id=rec.prompt_id,
                mime_type="image/png",
                bytes_size=len(trans_bytes),
                parent_asset_id=rec.asset_id,
                generation_type="matting",
                prompt=getattr(rec, "prompt", None),
                tags=["transparent", "matting", *(getattr(rec, "tags", []) or [])],
                session_id=session_id,
            )
            return {
                "asset_id": new_rec.asset_id,
                "asset_url": new_rec.asset_url,
                "filename": new_rec.filename,
                "mime_type": "image/png",
                "bytes_size": new_rec.bytes_size,
                "parent_asset_id": rec.asset_id,
            }

        # 2. Post-processing tool: generate_sprite_sheet
        if tool_identifier == "generate_sprite_sheet":
            asset_id = params.get("asset_id")
            if not asset_id:
                raise ValueError("generate_sprite_sheet step requires an 'asset_id'")

            rec = self.asset_registry.get_asset(asset_id)
            if not rec:
                raise ValueError(f"Asset '{asset_id}' not found")

            from asset_processor import fetch_asset_bytes, extract_video_keyframes
            raw_bytes = fetch_asset_bytes(rec, self.comfyui_client.base_url)
            if not raw_bytes:
                raise ValueError(f"Could not fetch asset bytes for '{asset_id}'")

            is_video = rec.mime_type.startswith("video") or rec.filename.lower().endswith((".mp4", ".webm", ".mov"))
            from PIL import Image
            import io
            frames = []
            if is_video:
                raw_frames = extract_video_keyframes(raw_bytes, num_frames=params.get("frame_count", 8))
                for fb, _ in raw_frames:
                    frames.append(Image.open(io.BytesIO(fb)).convert("RGBA"))
            else:
                frames.append(Image.open(io.BytesIO(raw_bytes)).convert("RGBA"))

            if params.get("remove_bg"):
                cleaned = []
                for f in frames:
                    buf = io.BytesIO()
                    f.save(buf, format="PNG")
                    cleaned.append(Image.open(io.BytesIO(remove_image_background(buf.getvalue()))).convert("RGBA"))
                frames = cleaned

            atlas_bytes, atlas_meta = build_sprite_sheet(
                frames=frames,
                columns=params.get("columns"),
                frame_width=params.get("frame_width"),
                frame_height=params.get("frame_height"),
                out_format=params.get("format", "PNG"),
            )

            fmt_ext = params.get("format", "png").lower()
            base_stem = rec.filename.rsplit(".", 1)[0]
            new_rec = self.asset_registry.register_asset(
                filename=f"spritesheet_{base_stem}.{fmt_ext}",
                subfolder=rec.subfolder,
                folder_type=rec.folder_type,
                workflow_id="generate_sprite_sheet",
                prompt_id=rec.prompt_id,
                mime_type=f"image/{fmt_ext}",
                bytes_size=len(atlas_bytes),
                parent_asset_id=rec.asset_id,
                generation_type="sprite_sheet",
                prompt=getattr(rec, "prompt", None),
                tags=["spritesheet", "atlas", *(getattr(rec, "tags", []) or [])],
                session_id=session_id,
            )
            return {
                "asset_id": new_rec.asset_id,
                "asset_url": new_rec.asset_url,
                "filename": new_rec.filename,
                "mime_type": f"image/{fmt_ext}",
                "bytes_size": new_rec.bytes_size,
                "parent_asset_id": rec.asset_id,
                "atlas_metadata": atlas_meta,
            }

        # 3. Workflows in WorkflowManager
        wf_id = tool_identifier
        workflow_data = self.workflow_manager.get_workflow(wf_id)
        if not workflow_data:
            # Fallback check if it's generate_image / generate_song alias
            if wf_id == "generate_image":
                wf_id = "api_image_flux2_text_to_image_9b" if "api_image_flux2_text_to_image_9b" in self.workflow_manager.tool_definitions else "generate_image"
                workflow_data = self.workflow_manager.get_workflow(wf_id)
            elif wf_id == "generate_song":
                wf_id = "api_audio_ace_step1_5_xl_sft" if "api_audio_ace_step1_5_xl_sft" in self.workflow_manager.tool_definitions else "generate_song"
                workflow_data = self.workflow_manager.get_workflow(wf_id)

        if not workflow_data:
            raise ValueError(f"Workflow or tool '{tool_identifier}' not found in registry")

        # Apply overrides to workflow
        rendered_workflow = self.workflow_manager.apply_workflow_overrides(wf_id, params)

        # Execute via ComfyUIClient
        timeout = params.get("timeout", 360)
        poll_interval = params.get("poll_interval", 2.0)
        result = self.comfyui_client.run_custom_workflow(
            rendered_workflow, max_attempts=int(timeout / poll_interval), poll_interval=poll_interval
        )

        parent_asset_id = params.get("parent_asset_id")
        return register_and_build_response(
            result=result,
            workflow_id=wf_id,
            asset_registry=self.asset_registry,
            tool_name=wf_id,
            session_id=session_id,
            parent_asset_id=parent_asset_id,
            prompt=params.get("prompt"),
            negative_prompt=params.get("negative_prompt"),
            seed=params.get("seed"),
        )
