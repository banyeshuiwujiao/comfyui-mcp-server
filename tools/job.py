"""Job and queue management tools for ComfyUI MCP Server"""

import logging
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP
from managers.error_diagnoser import ErrorDiagnoser

logger = logging.getLogger("MCP_Server")


def register_job_tools(
    mcp: FastMCP,
    comfyui_client,
    asset_registry,
    error_diagnoser=None
):
    """Register job and queue management tools with the MCP server"""
    if error_diagnoser is None:
        error_diagnoser = ErrorDiagnoser(comfyui_client)
    
    @mcp.tool()
    def get_queue_status() -> dict:
        """Get the current ComfyUI queue status.
        
        Returns information about queued and running jobs, including:
        - Currently running prompts
        - Queued prompts waiting to execute
        - Queue position and estimated wait times
        
        This tool provides async awareness - the AI can check if a job
        is still running or queued before polling for completion.
        
        Returns:
            Dict with 'queue_running' and 'queue_pending' lists, each containing
            prompt IDs and associated metadata.
        """
        try:
            queue_data = comfyui_client.get_queue()
            return {
                "status": "success",
                "queue_running": queue_data.get("queue_running", []),
                "queue_pending": queue_data.get("queue_pending", []),
                "running_count": len(queue_data.get("queue_running", [])),
                "pending_count": len(queue_data.get("queue_pending", []))
            }
        except Exception as e:
            logger.exception("Failed to get queue status")
            return {"error": str(e)}
    
    @mcp.tool()
    def get_job(prompt_id: str) -> dict:
        """Get job status and history for a specific prompt ID.
        
        Polls ComfyUI's /history/{prompt_id} endpoint to check if a job
        has completed and retrieve its outputs. This is the primary
        way to check job completion status.
        
        Args:
            prompt_id: The prompt ID returned from workflow submission
        
        Returns:
            Dict with:
            - status: "completed", "running", "queued", "processing", "error", or "not_found"
            - prompt_id: The prompt ID
            - outputs: Output data if completed (same format as generation tools)
            - error: Error information if the job failed
            - history: Full ComfyUI history snapshot for this prompt
            - message: Human-readable status message
        """
        if not prompt_id or not prompt_id.strip():
            return {
                "status": "error",
                "error": "Invalid prompt_id: empty or None",
                "prompt_id": prompt_id
            }
        
        try:
            # Check queue first to see if it's still running/queued
            try:
                queue_data = comfyui_client.get_queue()
                queue_running = queue_data.get("queue_running", [])
                queue_pending = queue_data.get("queue_pending", [])
                
                # Check if in queue
                # Queue format: [[execution_id, prompt_id, ...], ...]
                for item in queue_running:
                    if isinstance(item, list) and len(item) > 1:
                        if item[1] == prompt_id:
                            return {
                                "status": "running",
                                "prompt_id": prompt_id,
                                "message": "Job is currently running",
                                "execution_id": item[0] if len(item) > 0 else None
                            }
                
                for item in queue_pending:
                    if isinstance(item, list) and len(item) > 1:
                        if item[1] == prompt_id:
                            return {
                                "status": "queued",
                                "prompt_id": prompt_id,
                                "message": "Job is queued and waiting to run",
                                "position": queue_pending.index(item) + 1 if item in queue_pending else None
                            }
            except Exception as queue_error:
                # If queue check fails, continue to history check
                logger.warning(f"Failed to check queue status: {queue_error}")
            
            # Not in queue, check history
            try:
                history = comfyui_client.get_history(prompt_id)
                
                # History endpoint returns {prompt_id: {...}} for specific prompt_id
                # or full history dict if prompt_id not provided
                if prompt_id in history:
                    prompt_data = history[prompt_id]
                    
                    # Check for errors in history prompt_data
                    status_obj = prompt_data.get("status", {})
                    status_str = status_obj.get("status_str", "") if isinstance(status_obj, dict) else ""
                    messages = status_obj.get("messages", []) if isinstance(status_obj, dict) else status_obj if isinstance(status_obj, list) else []
                    has_execution_error = (
                        status_str == "error"
                        or (isinstance(status_obj, dict) and status_obj.get("completed") is False)
                        or comfyui_client._has_status_message(messages, "execution_error")
                    )

                    if "error" in prompt_data or has_execution_error:
                        node_err_data = comfyui_client._extract_node_error_dict(prompt_data) if hasattr(comfyui_client, "_extract_node_error_dict") else None
                        raw_error = prompt_data.get("error") or comfyui_client._extract_node_errors(prompt_data)
                        diagnosed = error_diagnoser.diagnose(
                            error=raw_error,
                            node_error_data=node_err_data
                        )
                        return {
                            **diagnosed,
                            "status": "error",
                            "prompt_id": prompt_id,
                            "history": prompt_data,
                            "message": f"Job failed with error: {diagnosed.get('error', raw_error)}"
                        }
                    
                    # Check if completed with outputs
                    if "outputs" in prompt_data and prompt_data["outputs"]:
                        # Expose a labeled asset list (best-effort from filenames)
                        # so the agent can pick branches without parsing raw outputs.
                        try:
                            all_assets = comfyui_client._extract_all_assets(
                                prompt_data["outputs"],
                                ("images", "image", "gifs", "gif", "audio", "audios", "files", "videos", "video"),
                                None,
                            )
                        except Exception:
                            all_assets = []
                        return {
                            "status": "completed",
                            "prompt_id": prompt_id,
                            "outputs": prompt_data["outputs"],
                            "all_assets": [
                                {
                                    "asset_url": a["asset_url"],
                                    "filename": a["filename"],
                                    "subfolder": a["subfolder"],
                                    "folder_type": a["type"],
                                    "node_id": a.get("node_id"),
                                    "label": a.get("label"),
                                }
                                for a in all_assets
                            ],
                            "asset_count": len(all_assets),
                            "history": prompt_data,
                            "message": "Job completed successfully"
                        }
                    else:
                        # History exists but no outputs yet (might be in transition)
                        return {
                            "status": "processing",
                            "prompt_id": prompt_id,
                            "message": "Job completed but outputs not yet available",
                            "history": prompt_data
                        }
                else:
                    # Check if we got full history (prompt_id not in keys)
                    # This might mean the job hasn't been recorded yet
                    if isinstance(history, dict) and len(history) > 0:
                        # Got full history, but our prompt_id not in it
                        return {
                            "status": "not_found",
                            "prompt_id": prompt_id,
                            "message": "Prompt ID not found in ComfyUI history. It may not have been submitted yet, or ComfyUI may have been restarted.",
                            "available_prompt_ids": list(history.keys())[:10]  # Show first 10 for debugging
                        }
                    else:
                        # Empty history response
                        return {
                            "status": "not_found",
                            "prompt_id": prompt_id,
                            "message": "Prompt ID not found. ComfyUI history is empty or the job hasn't been recorded."
                        }
            except Exception as history_error:
                logger.warning(f"Failed to get history for {prompt_id}: {history_error}")
                return {
                    "status": "error",
                    "prompt_id": prompt_id,
                    "error": f"Failed to retrieve history: {str(history_error)}",
                    "message": "Could not check job status - ComfyUI may be unavailable"
                }
        except Exception as e:
            logger.exception(f"Failed to get job status for {prompt_id}")
            return {
                "status": "error",
                "prompt_id": prompt_id,
                "error": str(e),
                "message": f"Unexpected error checking job status: {str(e)}"
            }
    
    @mcp.tool()
    def list_assets(
        limit: int = 10, 
        workflow_id: Optional[str] = None,
        session_id: Optional[str] = None,
        parent_asset_id: Optional[str] = None,
        root_asset_id: Optional[str] = None
    ) -> dict:
        """List recently generated assets for AI memory and browsing.
        
        Returns a list of assets that the AI can reference, view, or use
        for iteration. This enables the AI to remember what it has generated
        and make informed decisions about regeneration or modification.
        
        Args:
            limit: Maximum number of assets to return (default: 10)
            workflow_id: Filter by workflow type (e.g., "generate_image", "generate_song")
            session_id: Filter by conversation session (limits to current conversation)
            parent_asset_id: Filter by direct parent asset ID
            root_asset_id: Filter by root lineage asset ID
        
        Returns:
            Dict with:
            - assets: List of asset records with asset_id, asset_url, metadata, lineage
            - count: Number of assets returned
            - workflow_id_filter: Applied workflow filter (if any)
            - session_id_filter: Applied session filter (if any)
        
        Examples:
            # Get 5 most recent assets
            list_assets(limit=5)
            
            # Get only images
            list_assets(workflow_id="generate_image")
            
            # Get all variations derived from a specific root asset
            list_assets(root_asset_id="abc-123")
        """
        try:
            assets = asset_registry.list_assets(
                limit=limit,
                workflow_id=workflow_id,
                session_id=session_id,
                parent_asset_id=parent_asset_id,
                root_asset_id=root_asset_id
            )
            
            asset_list = []
            for asset in assets:
                asset_url = asset.asset_url or asset.get_asset_url(asset_registry.comfyui_base_url)
                asset_list.append({
                    "asset_id": asset.asset_id,
                    "asset_url": asset_url,
                    "filename": asset.filename,
                    "subfolder": asset.subfolder,
                    "folder_type": asset.folder_type,
                    "workflow_id": asset.workflow_id,
                    "prompt_id": asset.prompt_id,
                    "mime_type": asset.mime_type,
                    "width": asset.width,
                    "height": asset.height,
                    "bytes_size": asset.bytes_size,
                    "parent_asset_id": asset.parent_asset_id,
                    "root_asset_id": asset.root_asset_id,
                    "generation_type": asset.generation_type,
                    "prompt": asset.prompt,
                    "seed": asset.seed,
                    "tags": asset.tags,
                    "created_at": asset.created_at.isoformat(),
                    "expires_at": asset.expires_at.isoformat() if asset.expires_at else None,
                    "session_id": asset.session_id
                })
            
            return {
                "assets": asset_list,
                "count": len(asset_list),
                "workflow_id_filter": workflow_id,
                "session_id_filter": session_id,
                "parent_asset_id_filter": parent_asset_id,
                "root_asset_id_filter": root_asset_id
            }
        except Exception as e:
            logger.exception("Failed to list assets")
            return {"error": str(e)}
    
    @mcp.tool()
    def get_asset_metadata(asset_id: str) -> dict:
        """Get full metadata, parameters, and provenance for a generated asset.
        
        Returns comprehensive information about an asset including:
        - Asset details (dimensions, size, type)
        - Lineage details (parent_asset_id, root_asset_id, generation_type)
        - Workflow, prompt, negative_prompt, and seed
        - Full ComfyUI history snapshot (for provenance)
        - Original submitted workflow (for regeneration context)
        
        Args:
            asset_id: Asset ID from generation tools or list_assets
        
        Returns:
            Dict with complete asset metadata, history, lineage, and workflow information
        """
        try:
            asset = asset_registry.get_asset(asset_id)
            if not asset:
                return {"error": f"Asset {asset_id} not found."}
            
            asset_url = asset.asset_url or asset.get_asset_url(asset_registry.comfyui_base_url)
            
            result = {
                "asset_id": asset.asset_id,
                "asset_url": asset_url,
                "filename": asset.filename,
                "subfolder": asset.subfolder,
                "folder_type": asset.folder_type,
                "mime_type": asset.mime_type,
                "width": asset.width,
                "height": asset.height,
                "bytes_size": asset.bytes_size,
                "workflow_id": asset.workflow_id,
                "prompt_id": asset.prompt_id,
                "parent_asset_id": asset.parent_asset_id,
                "root_asset_id": asset.root_asset_id,
                "generation_type": asset.generation_type,
                "prompt": asset.prompt,
                "negative_prompt": asset.negative_prompt,
                "seed": asset.seed,
                "tags": asset.tags,
                "created_at": asset.created_at.isoformat(),
                "expires_at": asset.expires_at.isoformat() if asset.expires_at else None,
                "session_id": asset.session_id,
                "metadata": asset.metadata
            }
            
            # Include ComfyUI history if available
            if asset.comfy_history:
                result["comfy_history"] = asset.comfy_history
            
            # Include submitted workflow if available
            if asset.submitted_workflow:
                result["submitted_workflow"] = asset.submitted_workflow
            
            return result
        except Exception as e:
            logger.exception(f"Failed to get asset metadata for {asset_id}")
            return {"error": str(e)}

    @mcp.tool()
    def get_asset_lineage(asset_id: str) -> dict:
        """Get complete ancestry lineage and derived children for an asset.
        
        Allows the AI to trace the full evolution tree of an asset:
        - Ancestors (parent chain up to the root asset)
        - Direct children (assets derived directly from this asset)
        - Entire family tree (all assets sharing the same root lineage)
        
        Args:
            asset_id: Asset ID to inspect
            
        Returns:
            Dict with ancestors, children, and family tree graph.
        """
        try:
            return asset_registry.get_lineage(asset_id)
        except Exception as e:
            logger.exception(f"Failed to get asset lineage for {asset_id}")
            return {"error": str(e)}

    @mcp.tool()
    def search_assets(
        query: Optional[str] = None,
        tag: Optional[str] = None,
        workflow_id: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 10
    ) -> dict:
        """Search historically generated assets across sessions by keyword, tag, or workflow.
        
        Enables long-term memory retrieval across sessions.
        
        Args:
            query: Search keyword in prompt or filename
            tag: Filter by specific tag (e.g., "cyberpunk", "character")
            workflow_id: Filter by workflow name
            session_id: Filter by conversation session
            limit: Maximum number of assets to return (default 10)
            
        Returns:
            Dict with matching assets and total match count.
        """
        try:
            assets = asset_registry.search_assets(
                query=query,
                tag=tag,
                workflow_id=workflow_id,
                session_id=session_id,
                limit=limit
            )
            asset_list = []
            for asset in assets:
                asset_url = asset.asset_url or asset.get_asset_url(asset_registry.comfyui_base_url)
                asset_list.append({
                    "asset_id": asset.asset_id,
                    "asset_url": asset_url,
                    "filename": asset.filename,
                    "workflow_id": asset.workflow_id,
                    "generation_type": asset.generation_type,
                    "parent_asset_id": asset.parent_asset_id,
                    "root_asset_id": asset.root_asset_id,
                    "prompt": asset.prompt,
                    "tags": asset.tags,
                    "created_at": asset.created_at.isoformat(),
                })
            return {
                "assets": asset_list,
                "count": len(asset_list),
                "query": query,
                "tag": tag
            }
        except Exception as e:
            logger.exception("Failed to search assets")
            return {"error": str(e)}
    
    @mcp.tool()
    def cancel_job(prompt_id: str) -> dict:
        """Cancel a queued or running job.
        
        Allows the AI to cancel jobs that are no longer needed, providing
        user control and resource management. Can cancel jobs that are
        queued or currently running.
        
        Args:
            prompt_id: The prompt ID to cancel (from workflow submission)
        
        Returns:
            Dict with cancellation status
        """
        try:
            result = comfyui_client.cancel_prompt(prompt_id)
            return {
                "status": "cancelled",
                "prompt_id": prompt_id,
                "message": "Job cancellation requested",
                "comfy_response": result
            }
        except Exception as e:
            logger.exception(f"Failed to cancel job {prompt_id}")
            return {"error": str(e)}

    @mcp.tool()
    def interrupt() -> dict:
        """Interrupt the currently running prompt in ComfyUI.

        Aborts the active execution immediately (without clearing the pending
        queue). Use this when a job is stuck, producing garbage, or consuming
        too much VRAM. After interrupting, you can re-submit or clear_queue().

        Returns:
            Dict with status and a human-readable message.
        """
        try:
            return comfyui_client.interrupt()
        except Exception as e:
            logger.exception("Failed to interrupt ComfyUI")
            return {"error": str(e)}

    @mcp.tool()
    def clear_queue() -> dict:
        """Clear all pending (queued) prompts in ComfyUI.

        Removes everything waiting in the queue. The currently running prompt
        is NOT affected by this call — use interrupt() to stop a running job.
        Useful for "cold restart" scenarios where a backlog has built up.

        Returns:
            Dict with status and a human-readable message.
        """
        try:
            return comfyui_client.clear_queue()
        except Exception as e:
            logger.exception("Failed to clear queue")
            return {"error": str(e)}
