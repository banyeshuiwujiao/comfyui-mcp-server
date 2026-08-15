import requests
import json
import time
import os
import logging
from typing import Any, Dict, Optional, Sequence
from urllib.parse import quote

from asset_processor import get_image_metadata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ComfyUIClient")

class ComfyUIClient:
    def __init__(self, base_url):
        self.base_url = base_url
        self.available_models = self._get_available_models()
    
    def refresh_models(self):
        """Re-fetch available models and update available_models list."""
        self.available_models = self._get_available_models()

    # Loader node types whose model-name lists we want to surface. Covers the
    # checkpoint-style loaders used across this repo (UNETLoader / DiffusionLoader
    # for modern diffusion models, CheckpointLoaderSimple for classic SD).
    MODEL_LOADER_TYPES = (
        "CheckpointLoaderSimple",
        "UNETLoader",
        "DiffusionLoader",
        "CLIPLoader",
        "VAELoader",
        "LoraLoader",
    )

    def _get_available_models(self):
        """Fetch the union of model filenames across all known loader types.

        The repo's workflows load models via UNETLoader / DiffusionLoader /
        CLIPLoader rather than CheckpointLoaderSimple, so querying only the
        latter produced an empty list and broke model validation. We now query
        every relevant loader node type and merge (dedupe) their filename lists.
        """
        merged: list[str] = []
        seen: set[str] = set()
        for loader in self.MODEL_LOADER_TYPES:
            try:
                names = self._fetch_loader_model_names(loader)
            except Exception as e:  # noqa: BLE001 - best effort per loader
                logger.debug("Could not fetch models from %s: %s", loader, e)
                continue
            for name in names:
                if name not in seen:
                    seen.add(name)
                    merged.append(name)
        if merged:
            logger.info("Available models (%d across loaders): %s", len(merged), merged[:10])
        else:
            logger.warning("No models discovered from any loader type")
        return merged

    def _fetch_loader_model_names(self, loader_type: str) -> list[str]:
        """Return the list of model filenames exposed by a single loader node type."""
        response = requests.get(f"{self.base_url}/object_info/{loader_type}", timeout=10)
        if response.status_code != 200:
            return []
        data = response.json()
        info = data.get(loader_type, {})
        if not isinstance(info, dict):
            return []
        required = info.get("input", {}).get("required", {})
        if not isinstance(required, dict):
            return []
        # The model filename list is the first list-valued input (ckpt_name /
        # unet_name / clip_name / vae_name / lora_name, etc.).
        for value in required.values():
            if isinstance(value, list) and value and isinstance(value[0], list):
                return value[0]
        return []

    def run_custom_workflow(self, workflow: Dict[str, Any], preferred_output_keys: Sequence[str] | None = None, max_attempts: int = 180, poll_interval: float = 2.0):
        """Run a ComfyUI workflow and block until completion (or timeout).

        Args:
            workflow: The workflow prompt dict.
            preferred_output_keys: Ordered keys to look for in node outputs.
            max_attempts: Max number of poll iterations before giving up and
                returning a job handle (default 180 * 2s ≈ 6 min).
            poll_interval: Seconds between /history polls.

        Returns:
            Dict. On success includes 'asset_url', 'filename', 'all_assets'
            (list of every matched output across all nodes/branches), and
            'asset_metadata'. If still running after timeout, returns a
            job handle with status="running".
        """
        if preferred_output_keys is None:
            preferred_output_keys = ("images", "image", "gifs", "gif", "audio", "audios", "files", "videos", "video")

        prompt_id = self._queue_workflow(workflow)
        outputs = self._wait_for_prompt(prompt_id, max_attempts=max_attempts, poll_interval=poll_interval)

        # If outputs is None, the workflow is still running (timeout).
        # Return a job handle instead of raising an error.
        if outputs is None:
            return {
                "status": "running",
                "prompt_id": prompt_id,
                "message": (
                    f"Workflow still running after {max_attempts * poll_interval:.0f}s. "
                    f"Use get_job(prompt_id='{prompt_id}') to poll for completion, "
                    f"or interrupt()/clear_queue() to stop it."
                ),
            }

        # Extract all matched assets across every node / branch / preview
        all_assets = self._extract_all_assets(outputs, preferred_output_keys, workflow)
        if not all_assets:
            raise Exception(
                f"No outputs matched preferred keys: {preferred_output_keys}. "
                f"Available outputs: {json.dumps({k: list(v.keys()) if isinstance(v, dict) else type(v).__name__ for k, v in outputs.items()}, indent=2)}"
            )

        # Primary asset = first matched (keeps backward-compatible single-url response)
        asset_info = all_assets[0]
        asset_url = asset_info["asset_url"]

        # Extract asset metadata (pass workflow to extract dimensions from it)
        asset_metadata = self._get_asset_metadata(asset_url, outputs, preferred_output_keys, workflow)

        # Get full history snapshot for this prompt
        try:
            history = self.get_history(prompt_id)
            comfy_history = history.get(prompt_id, {}) if history else {}
        except Exception as e:
            logger.warning(f"Failed to fetch history snapshot for {prompt_id}: {e}")
            comfy_history = None

        return {
            "asset_url": asset_url,
            "filename": asset_info["filename"],
            "subfolder": asset_info["subfolder"],
            "folder_type": asset_info["type"],
            "prompt_id": prompt_id,
            "raw_outputs": outputs,
            "asset_metadata": asset_metadata,
            "comfy_history": comfy_history,
            "all_assets": all_assets,
            "submitted_workflow": workflow,
        }
    
    def _get_asset_metadata(self, asset_url: str, outputs: Dict[str, Any], preferred_output_keys: Sequence[str], workflow: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extract metadata about the generated asset"""
        metadata = {
            "mime_type": None,
            "width": None,
            "height": None,
            "bytes_size": None
        }
        
        # Try to extract from outputs first
        for node_id, node_output in outputs.items():
            if not isinstance(node_output, dict):
                continue
            for key in preferred_output_keys:
                assets = node_output.get(key)
                if assets and isinstance(assets, list) and len(assets) > 0:
                    asset = assets[0]
                    if isinstance(asset, dict):
                        # Infer mime type from filename extension
                        filename = asset.get("filename", "")
                        if filename.endswith((".png", ".PNG")):
                            metadata["mime_type"] = "image/png"
                        elif filename.endswith((".jpg", ".jpeg", ".JPG", ".JPEG")):
                            metadata["mime_type"] = "image/jpeg"
                        elif filename.endswith((".webp", ".WEBP")):
                            metadata["mime_type"] = "image/webp"
                        elif filename.endswith((".mp3", ".MP3")):
                            metadata["mime_type"] = "audio/mpeg"
                        elif filename.endswith((".mp4", ".MP4")):
                            metadata["mime_type"] = "video/mp4"
                        elif filename.endswith((".gif", ".GIF")):
                            metadata["mime_type"] = "image/gif"
                        break
        
        # Extract dimensions from workflow (EmptyLatentImage node) - much more efficient than analyzing image
        if workflow and (metadata["width"] is None or metadata["height"] is None):
            for node_id, node_data in workflow.items():
                if not isinstance(node_data, dict):
                    continue
                if node_data.get("class_type") == "EmptyLatentImage":
                    inputs = node_data.get("inputs", {})
                    if "width" in inputs and metadata["width"] is None:
                        metadata["width"] = inputs["width"]
                    if "height" in inputs and metadata["height"] is None:
                        metadata["height"] = inputs["height"]
                    if metadata["width"] and metadata["height"]:
                        break
        
        # Try to fetch headers to get size (non-blocking, best effort)
        try:
            response = requests.head(asset_url, timeout=5)
            if response.status_code == 200:
                content_length = response.headers.get("Content-Length")
                if content_length:
                    metadata["bytes_size"] = int(content_length)
                content_type = response.headers.get("Content-Type")
                if content_type and not metadata["mime_type"]:
                    metadata["mime_type"] = content_type.split(";")[0].strip()
        except Exception as e:
            logger.debug(f"Could not fetch asset metadata: {e}")
        
        # Fallback: Extract image dimensions by analyzing image bytes (only if not found in workflow)
        # This should rarely be needed now, but kept as a fallback
        if metadata["mime_type"] and metadata["mime_type"].startswith("image/") and (metadata["width"] is None or metadata["height"] is None):
            try:
                # Fetch image bytes to extract dimensions
                img_response = requests.get(asset_url, timeout=10)
                if img_response.status_code == 200:
                    image_bytes = img_response.content
                    # Update bytes_size if we got it from the full response
                    if not metadata["bytes_size"]:
                        metadata["bytes_size"] = len(image_bytes)
                    # Extract dimensions
                    img_metadata = get_image_metadata(image_bytes)
                    if img_metadata.get("width") and img_metadata.get("height"):
                        metadata["width"] = img_metadata["width"]
                        metadata["height"] = img_metadata["height"]
            except Exception as e:
                logger.debug(f"Could not extract image dimensions: {e}")
        
        return metadata

    def _queue_workflow(self, workflow: Dict[str, Any]):
        logger.info("Submitting workflow to ComfyUI...")
        response = requests.post(f"{self.base_url}/prompt", json={"prompt": workflow}, timeout=30)
        if response.status_code != 200:
            raise Exception(f"Failed to queue workflow: {response.status_code} - {response.text}")
        try:
            response_data = response.json()
            prompt_id = response_data.get("prompt_id")
            if not prompt_id:
                raise Exception("Response missing prompt_id")
        except (KeyError, ValueError) as e:
            raise Exception(f"Invalid response format from ComfyUI: {e}")
        logger.info(f"Queued workflow with prompt_id: {prompt_id}")
        return prompt_id

    @staticmethod
    def _has_status_message(messages, target: str) -> bool:
        """Check if a status messages list contains a target message type.

        ComfyUI status messages come as either a list of [type, data] pairs
        or a dict with 'messages' key.
        """
        if not messages:
            return False
        for msg in messages:
            if isinstance(msg, list) and len(msg) > 0 and msg[0] == target:
                return True
            if isinstance(msg, str) and msg == target:
                return True
        return False

    @staticmethod
    def _extract_node_error_dict(prompt_data: dict) -> Optional[dict]:
        """Extract structured error details dictionary from ComfyUI history data."""
        status = prompt_data.get("status", {})
        if isinstance(status, dict):
            messages = status.get("messages", [])
            for msg in messages:
                if isinstance(msg, list) and len(msg) >= 2 and msg[0] == "execution_error":
                    data = msg[1] if isinstance(msg[1], dict) else {}
                    return {
                        "node_id": data.get("node_id"),
                        "node_type": data.get("node_type"),
                        "exception_type": data.get("exception_type"),
                        "exception_message": data.get("exception_message"),
                        "traceback": data.get("traceback"),
                    }
        if isinstance(status, list):
            for entry in status:
                if isinstance(entry, list) and len(entry) >= 2 and entry[0] == "execution_error":
                    data = entry[1] if isinstance(entry[1], dict) else {}
                    return {
                        "node_id": data.get("node_id") if isinstance(data, dict) else None,
                        "node_type": data.get("node_type") if isinstance(data, dict) else None,
                        "exception_type": data.get("exception_type") if isinstance(data, dict) else None,
                        "exception_message": data.get("exception_message") if isinstance(data, dict) else str(entry[1]),
                    }
        return None

    @staticmethod
    def _extract_node_errors(prompt_data: dict) -> str:
        """Extract human-readable error details from ComfyUI history data.

        Looks in prompt_data['status']['messages'] for execution_error entries
        which contain node_id, node_type, exception_message, and
        exception_type. Falls back to other status fields when the structured
        error is not available.
        """
        parts: list[str] = []

        # Try structured status dict first (ComfyUI v2 history format)
        status = prompt_data.get("status", {})
        if isinstance(status, dict):
            messages = status.get("messages", [])
            for msg in messages:
                if isinstance(msg, list) and len(msg) >= 2 and msg[0] == "execution_error":
                    data = msg[1] if isinstance(msg[1], dict) else {}
                    node_type = data.get("node_type", "unknown")
                    node_id = data.get("node_id", "?")
                    exc_type = data.get("exception_type", "Error")
                    exc_msg = data.get("exception_message", "unknown error")
                    parts.append(f"Node {node_id} ({node_type}): [{exc_type}] {exc_msg}")
                    # Include traceback summary if available
                    traceback_lines = data.get("traceback", [])
                    if traceback_lines and isinstance(traceback_lines, list):
                        # Just the last meaningful line
                        for line in reversed(traceback_lines):
                            stripped = line.strip() if isinstance(line, str) else ""
                            if stripped and not stripped.startswith("Traceback") and not stripped.startswith("File"):
                                parts.append(f"  -> {stripped}")
                                break

        # Legacy list-of-lists format
        if not parts and isinstance(status, list):
            for entry in status:
                if isinstance(entry, list) and len(entry) >= 2 and entry[0] == "execution_error":
                    parts.append(f"Execution error: {entry[1]}")

        # Check for top-level 'error' key
        if not parts and "error" in prompt_data:
            parts.append(f"Error: {json.dumps(prompt_data['error'])}")

        if not parts:
            # Last resort: dump status for debugging
            status_summary = json.dumps(status, indent=2) if status else "no status info"
            parts.append(f"No detailed error info. Status: {status_summary}")

        return "; ".join(parts)

    def _wait_for_prompt(self, prompt_id: str, max_attempts: int = 180, poll_interval: float = 2.0):
        for attempt in range(max_attempts):
            try:
                # Try both the specific prompt_id endpoint and the full history endpoint
                response = requests.get(f"{self.base_url}/history/{prompt_id}", timeout=10)
                # If that doesn't work, we can also try: f"{self.base_url}/history"
                if response.status_code != 200:
                    logger.warning("History endpoint returned %s on attempt %s", response.status_code, attempt + 1)
                    time.sleep(poll_interval)
                    continue
                
                history = response.json()
                if not isinstance(history, dict):
                    logger.warning("Invalid history response format on attempt %s", attempt + 1)
                    time.sleep(poll_interval)
                    continue
                
                if prompt_id not in history:
                    # Workflow might still be running, wait and retry
                    if attempt < max_attempts - 1:
                        time.sleep(poll_interval)
                        continue
                    else:
                        # Last attempt - check if there's any history at all
                        logger.warning("Prompt ID not found in history. Available IDs: %s", list(history.keys())[:10])
                        time.sleep(poll_interval)
                        continue
                
                prompt_data = history[prompt_id]
                if not isinstance(prompt_data, dict):
                    logger.warning("Prompt data is not a dict on attempt %s", attempt + 1)
                    time.sleep(poll_interval)
                    continue
                
                # Check for workflow errors (top-level and status-embedded)
                if "error" in prompt_data:
                    error_info = prompt_data["error"]
                    raise Exception(f"Workflow failed with error: {json.dumps(error_info, indent=2)}")

                # Check if workflow status indicates failure
                status = prompt_data.get("status", {})
                if isinstance(status, dict):
                    if status.get("completed") == False:
                        error_msg = status.get("messages", ["Workflow failed"])
                        raise Exception(f"Workflow failed: {error_msg}")
                    # Check status_str for execution_error
                    if status.get("status_str") == "error":
                        node_errors = self._extract_node_errors(prompt_data)
                        raise Exception(f"Workflow execution error: {node_errors}")
                
                # Get outputs
                if "outputs" not in prompt_data:
                    # Check status to see if workflow completed
                    status = prompt_data.get("status", {})
                    status_str = status.get("status_str", "") if isinstance(status, dict) else ""
                    messages = status.get("messages", []) if isinstance(status, dict) else status if isinstance(status, list) else []

                    # Check for execution_error in status
                    if status_str == "error" or self._has_status_message(messages, "execution_error"):
                        node_errors = self._extract_node_errors(prompt_data)
                        raise Exception(f"Workflow execution failed: {node_errors}")

                    if self._has_status_message(messages, "execution_success"):
                        logger.info("Workflow execution succeeded, waiting for outputs to be available...")
                        time.sleep(3)
                        try:
                            full_history_response = requests.get(f"{self.base_url}/history", timeout=10)
                            if full_history_response.status_code == 200:
                                full_history = full_history_response.json()
                                if prompt_id in full_history:
                                    full_prompt_data = full_history[prompt_id]
                                    if "outputs" in full_prompt_data and full_prompt_data["outputs"]:
                                        logger.info("Found outputs in full history endpoint")
                                        return full_prompt_data["outputs"]
                        except Exception as e:
                            logger.debug("Could not fetch full history: %s", e)
                        continue

                    logger.warning("Prompt data missing outputs on attempt %s. Full data: %s", attempt + 1, json.dumps(prompt_data, indent=2))
                    time.sleep(poll_interval)
                    continue

                outputs = prompt_data["outputs"]
                if not outputs or not isinstance(outputs, dict):
                    status = prompt_data.get("status", {})
                    status_str = status.get("status_str", "") if isinstance(status, dict) else ""
                    messages = status.get("messages", []) if isinstance(status, dict) else status if isinstance(status, list) else []

                    # Check for errors first
                    if status_str == "error" or self._has_status_message(messages, "execution_error"):
                        node_errors = self._extract_node_errors(prompt_data)
                        raise Exception(f"Workflow execution failed: {node_errors}")

                    if self._has_status_message(messages, "execution_success"):
                        logger.warning("Workflow succeeded but outputs empty. Waiting longer...")
                        time.sleep(2)
                        continue

                    # Build diagnostic message from whatever status info we have
                    node_errors = self._extract_node_errors(prompt_data)
                    raise Exception(
                        f"Workflow completed but produced no outputs. "
                        f"Diagnostics: {node_errors}"
                    )
                
                logger.info("Workflow completed. Output nodes: %s", list(outputs.keys()))
                logger.debug("Full workflow outputs: %s", json.dumps(outputs, indent=2))
                logger.debug("Full prompt data: %s", json.dumps(prompt_data, indent=2))
                return outputs
            except requests.RequestException as e:
                logger.warning("Request error on attempt %s: %s", attempt + 1, e)
                time.sleep(1)
                continue
            except (ValueError, KeyError) as e:
                logger.warning("JSON parsing error on attempt %s: %s", attempt + 1, e)
                time.sleep(1)
                continue
        
        # Instead of raising, return a sentinel so callers can return a job handle
        logger.warning("Workflow %s still running after %s seconds", prompt_id, max_attempts)
        return None  # Signals timeout — caller should return a job handle

    def _extract_first_asset_url(self, outputs: Dict[str, Any], preferred_output_keys: Sequence[str]):
        # Log available outputs for debugging
        logger.debug("Available output keys in workflow: %s", list(outputs.keys()))
        for node_id, node_output in outputs.items():
            if not isinstance(node_output, dict):
                logger.debug("Node %s output is not a dict: %s", node_id, type(node_output))
                continue
            logger.debug("Node %s has keys: %s", node_id, list(node_output.keys()))
            for key in preferred_output_keys:
                assets = node_output.get(key)
                if assets and isinstance(assets, list) and len(assets) > 0:
                    asset = assets[0]
                    if not isinstance(asset, dict):
                        logger.debug("Asset in node %s, key %s is not a dict", node_id, key)
                        continue
                    filename = asset.get("filename")
                    if not filename:
                        logger.debug("Asset in node %s, key %s missing filename", node_id, key)
                        continue
                    subfolder = asset.get("subfolder", "")
                    output_type = asset.get("type", "output")
                    logger.info("Found asset: filename=%s, subfolder=%s, type=%s", filename, subfolder, output_type)
                    return f"{self.base_url}/view?filename={filename}&subfolder={subfolder}&type={output_type}"
        
        # Enhanced error message with actual output structure
        logger.error("No outputs matched preferred keys: %s", preferred_output_keys)
        logger.error("Actual outputs structure: %s", json.dumps(outputs, indent=2))
        raise Exception(
            f"No outputs matched preferred keys: {preferred_output_keys}. "
            f"Available outputs: {json.dumps({k: list(v.keys()) if isinstance(v, dict) else type(v).__name__ for k, v in outputs.items()}, indent=2)}"
        )
    
    def _extract_all_assets(
        self,
        outputs: Dict[str, Any],
        preferred_output_keys: Sequence[str],
        workflow: Optional[Dict[str, Any]] = None,
    ) -> list[Dict[str, Any]]:
        """Extract every matched asset across all nodes / branches / preview outputs.

        For multi-branch workflows (e.g., Qwen multi-angle edits with several
        SaveImage nodes), this returns one entry per produced file instead of
        only the first. Each entry has 'filename', 'subfolder', 'type',
        'asset_url', 'node_id' (the originating workflow node), 'output_key',
        and an optional 'label' (derived from the producing node's
        filename_prefix, e.g. "close_up" / "wide_shot") so callers can tell
        branches apart.

        Returns:
            List of asset info dicts (may be empty if nothing matched).
        """
        logger.debug("Available output keys in workflow: %s", list(outputs.keys()))
        collected: list[Dict[str, Any]] = []
        seen_keys: set = set()

        for node_id, node_output in outputs.items():
            if not isinstance(node_output, dict):
                continue
            for key in preferred_output_keys:
                assets = node_output.get(key)
                if not assets or not isinstance(assets, list):
                    continue
                for asset in assets:
                    if not isinstance(asset, dict):
                        continue
                    filename = asset.get("filename")
                    if not filename:
                        continue
                    subfolder = asset.get("subfolder", "")
                    output_type = asset.get("type", "output")
                    dedupe_key = (filename, subfolder, output_type)
                    if dedupe_key in seen_keys:
                        continue
                    seen_keys.add(dedupe_key)

                    # Derive a human label from the producing node's filename_prefix
                    # (e.g. "ComfyUI-close_up" -> "close_up"). Falls back to node_id.
                    label = self._derive_asset_label(workflow, node_id, filename)

                    # URL encode for special characters
                    base_url = self.base_url.rstrip('/')
                    encoded_filename = quote(filename, safe='')
                    encoded_subfolder = quote(subfolder, safe='') if subfolder else ''

                    if encoded_subfolder:
                        asset_url = f"{base_url}/view?filename={encoded_filename}&subfolder={encoded_subfolder}&type={output_type}"
                    else:
                        asset_url = f"{base_url}/view?filename={encoded_filename}&type={output_type}"

                    collected.append({
                        "filename": filename,
                        "subfolder": subfolder,
                        "type": output_type,
                        "asset_url": asset_url,
                        "node_id": node_id,
                        "output_key": key,
                        "label": label,
                    })

        return collected

    @staticmethod
    def _derive_asset_label(
        workflow: Optional[Dict[str, Any]], node_id: str, filename: str
    ) -> str:
        """Best-effort semantic label for an asset (branch/slot name).

        Tries the producing node's ``filename_prefix`` input (common in
        SaveImage / SaveVideo nodes), stripping a leading ``ComfyUI-`` prefix.
        Falls back to the node id, then the bare filename.
        """
        if workflow:
            node = workflow.get(node_id)
            if isinstance(node, dict):
                prefix = node.get("inputs", {}).get("filename_prefix")
                if isinstance(prefix, str) and prefix:
                    return prefix[8:] if prefix.startswith("ComfyUI-") else prefix
        # Fallback: strip the ComfyUI_ timestamp prefix from the filename itself.
        if filename.startswith("ComfyUI-"):
            base = filename[len("ComfyUI-"):]
            return base.split("_")[0] if base else filename
        return node_id

    # Backward-compatible alias used by older call sites / tests
    def _extract_first_asset_info(self, outputs: Dict[str, Any], preferred_output_keys: Sequence[str]) -> Dict[str, Any]:
        all_assets = self._extract_all_assets(outputs, preferred_output_keys)
        if not all_assets:
            raise Exception(
                f"No outputs matched preferred keys: {preferred_output_keys}. "
                f"Available outputs: {json.dumps({k: list(v.keys()) if isinstance(v, dict) else type(v).__name__ for k, v in outputs.items()}, indent=2)}"
            )
        return all_assets[0]
    
    def interrupt(self) -> Dict[str, Any]:
        """Interrupt the currently running prompt in ComfyUI.

        Sends a POST to the /interrupt endpoint, which aborts the active
        execution without clearing the pending queue. Useful when a job is
        stuck or consuming too much VRAM.

        Returns:
            Dict with 'status' and ComfyUI's raw response.
        """
        try:
            response = requests.post(f"{self.base_url}/interrupt", timeout=10)
            if response.status_code == 200:
                return {"status": "interrupted", "message": "ComfyUI interrupted the running prompt."}
            return {
                "status": "error",
                "message": f"Interrupt returned {response.status_code}",
                "detail": response.text[:500],
            }
        except requests.RequestException as e:
            logger.error(f"Failed to interrupt ComfyUI: {e}")
            raise Exception(f"Failed to interrupt ComfyUI: {e}")

    def clear_queue(self) -> Dict[str, Any]:
        """Clear all queued (pending) prompts in ComfyUI.

        Sends a POST to /queue with {"clear": true}. The currently running
        prompt is NOT cancelled by this (use interrupt() for that).
        """
        try:
            response = requests.post(
                f"{self.base_url}/queue",
                json={"clear": True},
                timeout=10,
            )
            response.raise_for_status()
            return {"status": "cleared", "message": "Pending queue cleared."}
        except requests.RequestException as e:
            logger.error(f"Failed to clear queue: {e}")
            raise Exception(f"Failed to clear queue: {e}")

    def get_queue(self) -> Dict[str, Any]:
        """Get current queue status from ComfyUI.
        
        Returns the full /queue endpoint response.
        """
        try:
            response = requests.get(f"{self.base_url}/queue", timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to get queue status: {e}")
            raise Exception(f"Failed to get queue status: {e}")
    
    def get_job_continuation(self, prompt_id: str) -> Optional[Dict[str, Any]]:
        """One-shot poll used to continue a job that returned status="running".

        Returns the same success-shaped dict as ``run_custom_workflow`` when the
        prompt has completed, or ``None`` if it is still running / not yet
        present in history. Errors are returned as a ``{"error": ...}`` dict so
        callers can surface them instead of looping forever.
        """
        try:
            history = self.get_history(prompt_id)
        except Exception as e:
            return {"error": f"Failed to poll job {prompt_id}: {e}"}

        if not isinstance(history, dict) or prompt_id not in history:
            return None  # still running / not recorded yet

        prompt_data = history[prompt_id]
        if not isinstance(prompt_data, dict):
            return None

        if "error" in prompt_data:
            return {"error": json.dumps(prompt_data["error"], indent=2)}

        status = prompt_data.get("status", {})
        if isinstance(status, dict) and status.get("status_str") == "error":
            return {"error": self._extract_node_errors(prompt_data)}

        outputs = prompt_data.get("outputs")
        if not outputs:
            return None  # completed-but-empty or still in flight

        # Reuse the standard extraction path so callers get a consistent shape.
        all_assets = self._extract_all_assets(outputs, ("images", "image", "gifs", "gif", "audio", "audios", "files", "videos", "video"), None)
        if not all_assets:
            return {"error": f"No outputs matched for job {prompt_id}."}

        asset_info = all_assets[0]
        asset_url = asset_info["asset_url"]
        asset_metadata = self._get_asset_metadata(asset_url, outputs, ("images", "image", "gifs", "gif", "audio", "audios", "files", "videos", "video"), None)
        try:
            comfy_history = self.get_history(prompt_id)
        except Exception:
            comfy_history = None
        return {
            "asset_url": asset_url,
            "filename": asset_info["filename"],
            "subfolder": asset_info["subfolder"],
            "folder_type": asset_info["type"],
            "prompt_id": prompt_id,
            "raw_outputs": outputs,
            "asset_metadata": asset_metadata,
            "comfy_history": comfy_history.get(prompt_id) if comfy_history else None,
            "all_assets": all_assets,
            "submitted_workflow": None,
        }

    def get_history(self, prompt_id: Optional[str] = None) -> Dict[str, Any]:
        """Get history from ComfyUI.
        
        Args:
            prompt_id: Optional specific prompt ID. If None, returns full history.
        
        Returns:
            History dict. If prompt_id provided, returns {prompt_id: {...}} or {} if not found.
        """
        try:
            if prompt_id:
                url = f"{self.base_url}/history/{prompt_id}"
            else:
                url = f"{self.base_url}/history"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to get history: {e}")
            raise Exception(f"Failed to get history: {e}")
    
    def cancel_prompt(self, prompt_id: str) -> Dict[str, Any]:
        """Cancel a queued or running prompt.
        
        Args:
            prompt_id: The prompt ID to cancel.
        
        Returns:
            Response from ComfyUI cancel endpoint.
        """
        try:
            response = requests.post(
                f"{self.base_url}/queue",
                json={"delete": [prompt_id]},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to cancel prompt {prompt_id}: {e}")
            raise Exception(f"Failed to cancel prompt: {e}")

    def list_input_files(self) -> Optional[list[str]]:
        """Best-effort list of filenames in ComfyUI's input directory.

        Resolution order:
          1. ``COMFYUI_INPUT_DIR`` env var (local path).
          2. HTTP probe of ``/view/input`` (recent ComfyUI builds return a JSON
             array of filenames there).

        Returns a list of filenames, or ``None`` if it cannot be determined
        (callers should treat ``None`` as "skip input-file validation").
        """
        env_dir = os.getenv("COMFYUI_INPUT_DIR")
        if env_dir and os.path.isdir(env_dir):
            try:
                return [f for f in os.listdir(env_dir)
                        if os.path.isfile(os.path.join(env_dir, f))]
            except OSError as e:
                logger.debug("Failed to list COMFYUI_INPUT_DIR: %s", e)

        try:
            resp = requests.get(f"{self.base_url}/view/input", timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list):
                    return [str(x) for x in data]
        except Exception as e:  # noqa: BLE001 - best effort
            logger.debug("Input listing probe failed: %s", e)
        return None
