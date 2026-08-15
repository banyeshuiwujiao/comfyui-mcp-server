"""Workflow generation tools (auto-registered from workflow files)"""

import copy
import inspect
import logging
import random
from typing import Any, Dict, Optional

from fastmcp import FastMCP
from managers.workflow_manager import AUDIO_OUTPUT_KEYS, VIDEO_OUTPUT_KEYS
from models.workflow import WorkflowToolDefinition
from tools.helpers import register_and_build_response

logger = logging.getLogger("MCP_Server")


def register_workflow_generation_tools(
    mcp: FastMCP,
    workflow_manager,
    comfyui_client,
    defaults_manager,
    asset_registry,
    gpu_guard=None
):
    """Register workflow-backed generation tools (e.g., generate_image, generate_song)"""

    def _register_workflow_tool(definition: WorkflowToolDefinition):
        def _tool_impl(*args, **kwargs):
            # Extract return_inline_preview if present (not a workflow parameter)
            return_inline_preview = kwargs.pop("return_inline_preview", False)
            # Execution control (optional, not workflow parameters)
            timeout = kwargs.pop("timeout", 360)
            poll_interval = kwargs.pop("poll_interval", 2.0)
            # Session tracking can be added via request context in the future
            session_id = None
            
            # Coerce parameter types before signature binding
            # MCP/JSON-RPC may pass numbers as strings, so we need to convert them
            coerced_kwargs = {}
            param_dict = {p.name: p for p in definition.parameters.values()}
            
            for key, value in kwargs.items():
                if key in param_dict:
                    param = param_dict[key]
                    # Coerce to correct type if needed
                    if value is not None:
                        try:
                            # Handle string representations of numbers
                            if param.annotation is int:
                                if isinstance(value, str) and value.strip().isdigit():
                                    coerced_kwargs[key] = int(value)
                                elif isinstance(value, (int, float)):
                                    coerced_kwargs[key] = int(value)
                                else:
                                    coerced_kwargs[key] = value
                            elif param.annotation is float:
                                if isinstance(value, str):
                                    coerced_kwargs[key] = float(value)
                                elif isinstance(value, (int, float)):
                                    coerced_kwargs[key] = float(value)
                                else:
                                    coerced_kwargs[key] = value
                            else:
                                coerced_kwargs[key] = value
                        except (ValueError, TypeError) as e:
                            # If coercion fails, use original value and let validation handle it
                            logger.warning(f"Failed to coerce {key}={value!r} to {param.annotation.__name__}: {e}")
                            coerced_kwargs[key] = value
                    else:
                        coerced_kwargs[key] = None
                else:
                    # Unknown parameter, pass through
                    coerced_kwargs[key] = value
            
            bound = _tool_impl.__signature__.bind(*args, **coerced_kwargs)
            bound.apply_defaults()

            # Only pass workflow parameters to render_workflow. Control parameters
            # (return_inline_preview/timeout/poll_interval) and any stray keys must
            # never leak into the workflow as node inputs, even if a future edit
            # forgets to pop a new control flag.
            provided_params = {
                k: v for k, v in bound.arguments.items() if k in param_dict
            }

            # Determine namespace using workflow manager (content-aware)
            namespace = workflow_manager._determine_namespace(definition.workflow_id)
            # Refine using output preferences (catches custom audio/video workflows)
            if definition.output_preferences == AUDIO_OUTPUT_KEYS:
                namespace = "audio"
            elif definition.output_preferences == VIDEO_OUTPUT_KEYS:
                namespace = "video"

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

                # Only validate model if the workflow actually has a 'model' parameter
                has_model_param = "model" in definition.parameters
                if has_model_param:
                    provided_model = dict(bound.arguments).get("model")
                    resolved_model = defaults_manager.get_default(namespace, "model", provided_model)

                    if resolved_model and not defaults_manager.is_model_valid(namespace, resolved_model):
                        is_valid, model_name, source = defaults_manager.validate_default_model(namespace)
                        available_models = list(defaults_manager._available_models_set)
                        sample_models = available_models[:5] if available_models else []

                        error_msg = (
                            f"Default model '{model_name}' (from {source} defaults) not found in ComfyUI checkpoints. "
                            f"Set a valid model via `set_defaults`, config file, or env var. "
                            f"Try `list_models` to see available checkpoints."
                        )
                        if sample_models:
                            error_msg += f" Available models: {sample_models}"

                        return {"error": error_msg}
                
                workflow = workflow_manager.render_workflow(definition, provided_params, defaults_manager)
                result = comfyui_client.run_custom_workflow(
                    workflow,
                    preferred_output_keys=definition.output_preferences,
                    max_attempts=int(timeout / poll_interval) if poll_interval > 0 else 180,
                    poll_interval=poll_interval,
                )

                # If the job is still running after the blocking timeout, try to
                # continue polling a bounded number of times instead of handing a
                # half-finished job handle to the agent. The agent can still call
                # get_job(prompt_id) if this also times out.
                if result.get("status") == "running":
                    prompt_id = result.get("prompt_id")
                    for _ in range(2):  # up to 2 extra continuation rounds
                        time.sleep(poll_interval)
                        cont = comfyui_client.get_job_continuation(prompt_id)
                        if cont is not None:
                            result = cont
                            break
                    if result.get("status") == "running":
                        result.setdefault("message", "")
                        result["message"] = (
                            result.get("message", "") +
                            " Still running after extended wait; poll get_job(prompt_id) "
                            "to retrieve outputs when ready."
                        ).strip()

                # Register asset and build response
                return register_and_build_response(
                    result,
                    definition.workflow_id,
                    asset_registry,
                    tool_name=definition.tool_name,
                    return_inline_preview=return_inline_preview,
                    session_id=session_id
                )
                
            except Exception as exc:
                error_str = str(exc).lower()
                # Check if error is related to missing model (only if workflow uses models)
                if has_model_param and ("model" in error_str or "checkpoint" in error_str or "ckpt" in error_str):
                    comfyui_client.refresh_models()
                    defaults_manager.refresh_model_set()

                    provided_model = dict(bound.arguments).get("model")
                    resolved_model = defaults_manager.get_default(namespace, "model", provided_model)

                    if resolved_model and not defaults_manager.is_model_valid(namespace, resolved_model):
                        is_valid, model_name, source = defaults_manager.validate_default_model(namespace)
                        available_models = list(defaults_manager._available_models_set)
                        sample_models = available_models[:5] if available_models else []

                        error_msg = (
                            f"Default model '{model_name}' (from {source} defaults) not found in ComfyUI checkpoints. "
                            f"Set a valid model via `set_defaults`, config file, or env var. "
                            f"Try `list_models` to see available checkpoints."
                        )
                        if sample_models:
                            error_msg += f" Available models: {sample_models}"

                        return {"error": error_msg}

                logger.exception("Workflow '%s' failed", definition.workflow_id)
                return {"error": str(exc)}

        # Separate required and optional parameters to ensure correct ordering
        required_params = []
        optional_params = []
        annotations: Dict[str, Any] = {}
        
        for param in definition.parameters.values():
            # For numeric types, use Any to allow string coercion from JSON-RPC
            # FastMCP/Pydantic validation is strict, so we accept Any and validate/coerce ourselves
            if param.annotation in (int, float):
                # Use Any to bypass strict type checking, we'll coerce in the function
                annotation_type = Any
            else:
                annotation_type = param.annotation
            
            if param.required:
                parameter = inspect.Parameter(
                    name=param.name,
                    kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=annotation_type,
                )
                required_params.append(parameter)
            else:
                # Optional parameter with default value
                # For numeric types, use Any directly (not Optional[Any]) to allow string coercion
                if param.annotation in (int, float):
                    final_annotation = Any
                else:
                    final_annotation = Optional[annotation_type]
                parameter = inspect.Parameter(
                    name=param.name,
                    kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=final_annotation,
                    default=None,
                )
                optional_params.append(parameter)
            annotations[param.name] = param.annotation
        
        # Add return_inline_preview as optional parameter
        optional_params.append(inspect.Parameter(
            name="return_inline_preview",
            kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=bool,
            default=False,
        ))
        annotations["return_inline_preview"] = bool

        # Add execution-control optional parameters (not workflow parameters)
        optional_params.append(inspect.Parameter(
            name="timeout",
            kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=Any,
            default=360,
        ))
        annotations["timeout"] = Any
        optional_params.append(inspect.Parameter(
            name="poll_interval",
            kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=Any,
            default=2.0,
        ))
        annotations["poll_interval"] = Any
        
        # Combine: required parameters first, then optional
        parameters = required_params + optional_params
        annotations["return"] = dict
        _tool_impl.__signature__ = inspect.Signature(parameters, return_annotation=dict)
        _tool_impl.__annotations__ = annotations
        _tool_impl.__name__ = f"tool_{definition.tool_name}"
        _tool_impl.__doc__ = definition.description
        mcp.tool(name=definition.tool_name, description=definition.description)(_tool_impl)
        logger.info(
            "Registered MCP tool '%s' for workflow '%s'",
            definition.tool_name,
            definition.workflow_id,
        )
    
    # Register all workflow-backed tools
    if workflow_manager.tool_definitions:
        for tool_definition in workflow_manager.tool_definitions:
            _register_workflow_tool(tool_definition)
    else:
        logger.info(
            "No workflow placeholders found in %s; add %s markers to enable auto tools",
            workflow_manager.workflows_dir,
            "PARAM_",
        )


def _update_workflow_params(workflow: dict, param_overrides: dict) -> dict:
    """
    Update workflow node inputs with parameter overrides.

    Applies overrides by matching parameter name to the relevant node inputs.
    Matching is intentionally broader than the old hardcoded SD-only mapping:
    prompts match any text-bearing input (``text``/``prompt``/``positive``),
    seeds also cover ``noise_seed`` (used by Flux2 / MiniMax H3), and numeric
    sampler params match KSampler-style nodes regardless of exact class name.

    A parameter is "applied" if at least one node input was updated.
    """
    # Parameter -> tuple of input keys to try (first present wins per node)
    param_input_keys = {
        "prompt": ("prompt", "text", "positive"),
        "negative_prompt": ("negative", "text", "prompt"),
        "steps": ("steps",),
        "cfg": ("cfg",),
        "sampler_name": ("sampler_name",),
        "scheduler": ("scheduler",),
        "denoise": ("denoise",),
        "width": ("width",),
        "height": ("height",),
        "model": ("ckpt_name", "unet_name", "model_name"),
        "seed": ("seed", "noise_seed"),
        "tags": ("tags",),
        "lyrics": ("lyrics",),
        "seconds": ("seconds",),
        "lyrics_strength": ("lyrics_strength",),
    }

    for param_name, override_value in param_overrides.items():
        if param_name not in param_input_keys:
            logger.warning(f"Unknown parameter '{param_name}' in regenerate, skipping")
            continue

        keys = param_input_keys[param_name]
        is_negative = (param_name == "negative_prompt")
        updated = False
        for node_id, node_data in workflow.items():
            if not isinstance(node_data, dict):
                continue
            inputs = node_data.get("inputs", {})
            if not isinstance(inputs, dict):
                continue
            # Find the first matching input key on this node
            target_input = next((k for k in keys if k in inputs), None)
            if target_input is None:
                continue
            # For negative prompt, never touch a node that looks like the main prompt
            if is_negative:
                blob = f"{node_data.get('class_type','')} {node_data.get('_meta',{})}".lower()
                if "neg" not in blob and target_input == "text" and "positive" not in blob:
                    # Likely the main prompt node; skip to avoid clobbering it
                    if "negative" not in blob:
                        continue
            inputs[target_input] = override_value
            updated = True

        if not updated:
            logger.warning(f"Could not find node to update parameter '{param_name}' in workflow")

    return workflow


def _update_seed(workflow: dict, seed: Optional[int]) -> dict:
    """
    Update the seed in KSampler nodes.
    
    Args:
        workflow: The workflow dict
        seed: New seed value, or None to generate random, or -1 to keep original
    
    Returns:
        Updated workflow
    """
    if seed == -1:
        # Keep original seed - no changes needed
        return workflow
    
    # Generate random seed if not specified
    if seed is None:
        seed = random.randint(0, 0xffffffffffffffff)
    
    # Find and update every node that carries a seed-like input. Covers both the
    # classic KSampler ("seed") and modern Flux2 / MiniMax H3 nodes ("noise_seed").
    for node_id, node_data in workflow.items():
        if not isinstance(node_data, dict):
            continue
        inputs = node_data.get("inputs", {})
        if not isinstance(inputs, dict):
            continue
        if "seed" in inputs:
            inputs["seed"] = seed
        if "noise_seed" in inputs:
            inputs["noise_seed"] = seed

    return workflow


def register_regenerate_tool(
    mcp: FastMCP,
    comfyui_client,
    asset_registry,
    gpu_guard=None
):
    """Register the regenerate tool for iterating on existing assets."""
    
    @mcp.tool()
    def regenerate(
        asset_id: str,
        seed: Optional[int] = None,
        return_inline_preview: bool = False,
        param_overrides: Optional[Dict[str, Any]] = None
    ) -> dict:
        """Regenerate an existing asset with optional parameter overrides.
        
        Retrieves the original workflow and parameters from the asset's provenance
        data, applies any overrides, and re-submits to ComfyUI.
        
        Args:
            asset_id: ID of the asset to regenerate
            seed: New random seed (None = generate new random seed, -1 = use original seed)
            return_inline_preview: If True, include a small thumbnail base64 in response
            param_overrides: Dict of workflow parameters to override (e.g., {"steps": 30, "cfg": 8.0, "prompt": "new prompt"})
        
        Returns:
            dict: New asset information with same structure as generate_* tools
        
        Examples:
            # Regenerate with different seed
            regenerate(asset_id="abc123")
            
            # Regenerate with higher quality settings
            regenerate(asset_id="abc123", param_overrides={"steps": 30, "cfg": 10.0})
            
            # Modify the prompt
            regenerate(asset_id="abc123", param_overrides={"prompt": "a beautiful sunset, oil painting style"})
            
            # Use exact same parameters (deterministic)
            regenerate(asset_id="abc123", seed=-1)
        """
        try:
            # GPU pressure guard
            if gpu_guard is not None:
                admission = gpu_guard.check_admission()
                if not admission["allowed"]:
                    return {
                        "error": admission["reason"],
                        "gpu_util": admission["gpu_util"],
                        "pending": admission["pending"],
                        "suggestion": "Call interrupt()/clear_queue() or wait, then retry.",
                    }

            # Step 1: Retrieve original asset metadata
            asset = asset_registry.get_asset(asset_id)
            if not asset:
                return {"error": f"Asset {asset_id} not found (registry is in-memory and resets on restart). Generate a new asset to regenerate."}
            
            # Extract the stored workflow
            original_workflow = asset.submitted_workflow
            if not original_workflow:
                return {"error": "No workflow data stored for this asset. Cannot regenerate."}
            
            # Step 2: Deep copy workflow to avoid mutating the stored one
            workflow = copy.deepcopy(original_workflow)
            
            # Step 3: Apply parameter overrides
            if param_overrides:
                workflow = _update_workflow_params(workflow, param_overrides)
            
            # Step 4: Update seed
            workflow = _update_seed(workflow, seed)
            
            # Step 5: Determine output preferences from original workflow
            # Try to infer from workflow_id or use defaults
            output_preferences = None
            if asset.workflow_id:
                # Use workflow manager's output preference guessing if available
                # For now, use common defaults
                if "image" in asset.workflow_id.lower():
                    output_preferences = ("images", "image", "gifs", "gif")
                elif "audio" in asset.workflow_id.lower() or "song" in asset.workflow_id.lower():
                    output_preferences = ("audio", "audios", "sound", "files")
                elif "video" in asset.workflow_id.lower():
                    output_preferences = ("videos", "video", "mp4", "mov", "webm")
            
            # Step 6: Submit to ComfyUI
            result = comfyui_client.run_custom_workflow(
                workflow,
                preferred_output_keys=output_preferences,
            )
            
            # Step 7: Register and return new asset
            return register_and_build_response(
                result,
                asset.workflow_id,
                asset_registry,
                tool_name="regenerate",
                return_inline_preview=return_inline_preview,
                session_id=asset.session_id  # Preserve original session
            )
        except Exception as e:
            logger.exception(f"Failed to regenerate asset {asset_id}")
            return {"error": f"Failed to regenerate: {str(e)}"}
