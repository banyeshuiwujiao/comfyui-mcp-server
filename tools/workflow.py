"""Workflow management tools for ComfyUI MCP Server"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from fastmcp import FastMCP
from managers.error_diagnoser import ErrorDiagnoser, find_closest_model
from tools.helpers import register_and_build_response

logger = logging.getLogger("MCP_Server__workflow")

PLACEHOLDER_PREFIX = "PARAM_"
MODEL_INPUT_KEYS = ("unet_name", "ckpt_name", "clip_name", "vae_name",
                    "lora_name", "model_name", "diffusion_model")


def _inspect_workflow(
    comfyui_client,
    workflow: Dict[str, Any],
    workflow_id: str,
) -> Tuple[List[str], List[str], List[str]]:
    """Inspect an *already-overridden* workflow for pre-flight failure causes.

    Shared by both ``validate_workflow`` (dry-run, no submission) and the
    automatic guard that ``run_workflow`` runs before queueing. Returns a tuple
    ``(issues, checked_models, checked_images)``.

    Checks:
      - Leftover ``PARAM_`` placeholders (a required/optional param was omitted
        and not filled by the safety net).
      - Model files referenced by loader nodes that are not present in ComfyUI.
      - Input images referenced by LoadImage not found in the ComfyUI input dir.
    """
    issues: List[str] = []

    # 1) Leftover PARAM_ placeholders.
    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        for k, v in node.get("inputs", {}).items():
            if isinstance(v, str) and v.startswith(PLACEHOLDER_PREFIX):
                issues.append(f"Unfilled placeholder {v!r} at node {node_id}.{k}")

    # 2) Model references present in ComfyUI?
    available = set(getattr(comfyui_client, "available_models", None) or [])
    checked_models: List[str] = []
    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs", {})
        for key in MODEL_INPUT_KEYS:
            val = inputs.get(key)
            if isinstance(val, str) and val:
                name = val
                checked_models.append(name)
                if name not in available:
                    issues.append(
                        f"Model '{name}' (node {node_id}.{key}) not found in ComfyUI. "
                        f"Available loaders may use a different filename or the model is missing."
                    )

    # 3) Input images present in ComfyUI input dir? (best-effort)
    checked_images: List[str] = []
    try:
        input_list = comfyui_client.list_input_files()
    except Exception:
        input_list = None
    if input_list is not None:
        input_set = set(input_list)
        for node_id, node in workflow.items():
            if not isinstance(node, dict):
                continue
            if node.get("class_type") == "LoadImage":
                fname = node.get("inputs", {}).get("image")
                if isinstance(fname, str) and fname:
                    checked_images.append(fname)
                    if fname not in input_set:
                        issues.append(
                            f"Input image '{fname}' (node {node_id}) not found in ComfyUI input directory."
                        )

    return issues, sorted(set(checked_models)), sorted(set(checked_images))


def register_workflow_tools(
    mcp: FastMCP,
    workflow_manager,
    comfyui_client,
    defaults_manager,
    asset_registry,
    gpu_guard=None,
    error_diagnoser=None
):
    """Register workflow tools with the MCP server"""
    if error_diagnoser is None:
        error_diagnoser = ErrorDiagnoser(comfyui_client, defaults_manager)
    
    @mcp.tool()
    def list_workflows() -> dict:
        """List all available workflows in the workflow directory.
        
        Returns a catalog of workflows with their IDs, names, descriptions,
        available inputs, and optional metadata.
        """
        catalog = workflow_manager.get_workflow_catalog()
        return {
            "workflows": catalog,
            "count": len(catalog),
            "workflow_dir": str(workflow_manager.workflows_dir)
        }

    @mcp.tool()
    def run_workflow(
        workflow_id: str,
        overrides: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None,
        return_inline_preview: bool = False,
        timeout: int = 360,
        poll_interval: float = 2.0
    ) -> dict:
        """Run a saved ComfyUI workflow with constrained parameter overrides.

        Args:
            workflow_id: The workflow ID (filename stem, e.g., "generate_image")
            overrides: Optional dict of parameter overrides (e.g., {"prompt": "a cat", "width": 1024})
            options: Optional dict of execution options (reserved for future use)
            return_inline_preview: If True, include a small thumbnail base64 in response (256px, ~100KB)
            timeout: Max seconds to block waiting for completion (default 360 ≈ 6 min).
                If exceeded, returns a job handle with status="running" instead of erroring.
            poll_interval: Seconds between /history polls (default 2.0).

        Returns:
            Result with asset_url, workflow_id, and execution metadata. For multi-branch
            workflows, also includes 'all_assets' (list of every produced file) and
            'asset_count'. If return_inline_preview=True, also includes inline_preview_base64.
        """
        if overrides is None:
            overrides = {}

        # Load workflow
        workflow = workflow_manager.load_workflow(workflow_id)
        if not workflow:
            return {"error": f"Workflow '{workflow_id}' not found"}

        try:
            # GPU pressure guard: refuse admission under sustained saturation
            if gpu_guard is not None:
                output_preferences = workflow_manager._guess_output_preferences(workflow)
                workflow_hint = workflow_id.lower()
                heavy = output_preferences in (("videos", "video", "mp4", "mov", "webm"),
                                               ("audio", "audios", "sound", "files")) \
                    or "qwen_image_edit_2511" in workflow_hint \
                    or "2512" in workflow_hint
                admission = gpu_guard.check_admission(heavy=heavy)
                if not admission["allowed"]:
                    return {
                        "error": admission["reason"],
                        "gpu_util": admission["gpu_util"],
                        "vram_free_gb": admission.get("vram_free_gb"),
                        "pending": admission["pending"],
                        "suggestion": "Call interrupt()/clear_queue() or wait, then retry.",
                    }

            # Required-parameter guard: check BEFORE applying overrides, because
            # apply_workflow_overrides silently fills missing required inputs
            # (e.g. an omitted image) with empty fallbacks, which would otherwise
            # slip through to ComfyUI and fail only at queue/execution time.
            pre_issues: List[str] = []
            try:
                params = workflow_manager._extract_parameters(workflow)
                for param in params.values():
                    if param.required and param.name not in overrides:
                        pre_issues.append(
                            f"Required parameter '{param.name}' was not provided."
                        )
            except Exception as exc:
                logger.warning("Required-param pre-check failed: %s", exc)

            # Apply overrides with constraints
            workflow = workflow_manager.apply_workflow_overrides(
                workflow, workflow_id, overrides, defaults_manager
            )

            # Extract and remove override report before submitting to ComfyUI
            override_report = workflow.pop("__override_report__", None)

            # Pre-flight guard: inspect the (already-overridden) workflow for
            # residual placeholders, missing models, and missing input images.
            # Runs automatically before every submission so the agent gets an
            # immediate, actionable error instead of a queue rejection from
            # ComfyUI. Skips the remote model/image checks if ComfyUI is not
            # reachable (best-effort); placeholder checks always run locally.
            inspect_issues, checked_models, checked_images = _inspect_workflow(
                comfyui_client, workflow, workflow_id
            )
            all_issues = pre_issues + inspect_issues
            if all_issues:
                diagnosed = error_diagnoser.diagnose(
                    error=f"Pre-flight validation failed: {'; '.join(all_issues)}",
                    workflow=workflow,
                    params=overrides,
                )
                return {
                    **diagnosed,
                    "issues": all_issues,
                    "workflow_id": workflow_id,
                    "checked_models": checked_models,
                    "checked_images": checked_images,
                    "hint": "Fix the issues above and retry, or call validate_workflow() for a detailed dry-run report.",
                }

            # Determine output preferences
            output_preferences = workflow_manager._guess_output_preferences(workflow)

            # Execute workflow (blocking with configurable timeout)
            result = comfyui_client.run_custom_workflow(
                workflow,
                preferred_output_keys=output_preferences,
                max_attempts=int(timeout / poll_interval) if poll_interval > 0 else 180,
                poll_interval=poll_interval,
            )

            # Register asset and build response
            provenance = {}
            workflow_hash = getattr(workflow_manager, "get_workflow_file_hash", None)
            if callable(workflow_hash):
                provenance["workflow_hash"] = workflow_hash(workflow_id)
            response = register_and_build_response(
                result,
                workflow_id,
                asset_registry,
                tool_name=None,
                return_inline_preview=return_inline_preview,
                session_id=None,
                metadata=provenance or None,
            )

            # Include override report so the agent can see what was applied/dropped
            if override_report and override_report.get("overrides_dropped"):
                response["overrides_applied"] = override_report["overrides_applied"]
                response["overrides_dropped"] = override_report["overrides_dropped"]

            return response
        except Exception as exc:
            logger.exception("Workflow '%s' failed", workflow_id)
            current_workflow = workflow if 'workflow' in locals() else None
            return error_diagnoser.diagnose(
                error=exc,
                workflow=current_workflow,
                params=overrides,
            )

    @mcp.tool()
    def validate_workflow(
        workflow_id: str,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> dict:
        """Pre-flight validation of a workflow before submission.

        Applies the overrides (dry-run, no submission) and inspects the resulting
        workflow for common failure causes Agent-side:

        - Leftover ``PARAM_`` placeholders (a required/optional param was omitted).
        - Model files referenced by loader nodes (UNETLoader / CLIPLoader /
          VAELoader / LoraLoader / CheckpointLoaderSimple / DiffusionLoader) that
          are not present in ComfyUI's available models.
        - Input images referenced by LoadImage whose filename is not found in the
          ComfyUI input directory.

        Returns ``{ok: bool, issues: [...], workflow_id, checked_models, checked_images}``.
        Call this *before* ``run_workflow`` to catch mistakes early instead of
        waiting for ComfyUI to reject the prompt.
        """
        if overrides is None:
            overrides = {}

        workflow = workflow_manager.load_workflow(workflow_id)
        if not workflow:
            return {"ok": False, "issues": [f"Workflow '{workflow_id}' not found"],
                    "workflow_id": workflow_id, "checked_models": [], "checked_images": []}

        # Required-parameter check (mirrors the run_workflow guard).
        pre_issues: List[str] = []
        try:
            params = workflow_manager._extract_parameters(workflow)
            for param in params.values():
                if param.required and param.name not in overrides:
                    pre_issues.append(
                        f"Required parameter '{param.name}' was not provided."
                    )
        except Exception as exc:
            logger.warning("validate_workflow required-param check failed: %s", exc)

        # Apply overrides (dry-run) so we validate the *effective* workflow.
        try:
            workflow = workflow_manager.apply_workflow_overrides(
                workflow, workflow_id, overrides, defaults_manager
            )
            workflow.pop("__override_report__", None)
        except Exception as exc:
            return {"ok": False, "issues": [f"Failed to apply overrides: {exc}"],
                    "workflow_id": workflow_id, "checked_models": [], "checked_images": []}

        # Inspect the effective workflow (placeholder / model / image checks).
        inspect_issues, checked_models, checked_images = _inspect_workflow(
            comfyui_client, workflow, workflow_id
        )
        issues = pre_issues + inspect_issues

        return {
            "ok": len(issues) == 0,
            "issues": issues,
            "workflow_id": workflow_id,
            "checked_models": sorted(set(checked_models)),
            "checked_images": sorted(set(checked_images)),
        }
