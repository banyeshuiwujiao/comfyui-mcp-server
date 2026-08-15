"""Workflow management tools for ComfyUI MCP Server"""

import logging
import time
from typing import Any, Dict, Optional

from fastmcp import FastMCP
from tools.helpers import register_and_build_response

logger = logging.getLogger("MCP_Server")


def register_workflow_tools(
    mcp: FastMCP,
    workflow_manager,
    comfyui_client,
    defaults_manager,
    asset_registry,
    gpu_guard=None
):
    """Register workflow tools with the MCP server"""
    
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
                admission = gpu_guard.check_admission()
                if not admission["allowed"]:
                    return {
                        "error": admission["reason"],
                        "gpu_util": admission["gpu_util"],
                        "pending": admission["pending"],
                        "suggestion": "Call interrupt()/clear_queue() or wait, then retry.",
                    }

            # Apply overrides with constraints
            workflow = workflow_manager.apply_workflow_overrides(
                workflow, workflow_id, overrides, defaults_manager
            )

            # Extract and remove override report before submitting to ComfyUI
            override_report = workflow.pop("__override_report__", None)

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
            response = register_and_build_response(
                result,
                workflow_id,
                asset_registry,
                tool_name=None,
                return_inline_preview=return_inline_preview,
                session_id=None
            )

            # Include override report so the agent can see what was applied/dropped
            if override_report and override_report.get("overrides_dropped"):
                response["overrides_applied"] = override_report["overrides_applied"]
                response["overrides_dropped"] = override_report["overrides_dropped"]

            return response
        except Exception as exc:
            logger.exception("Workflow '%s' failed", workflow_id)
            return {"error": str(exc)}
