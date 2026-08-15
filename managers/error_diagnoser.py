"""Error Diagnoser & Self-Healing Advisor for ComfyUI MCP Server.

Provides structured error diagnosis, categorization, and concrete parameter
modification suggestions so AI Agents can self-heal, adjust parameters, and
retry failed generation jobs autonomously.
"""

import difflib
import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger("MCP_Server__error_diagnoser")


def align_dimension(val: int, multiple: int = 8, min_val: int = 64, max_val: int = 4096) -> int:
    """Align an integer dimension to the nearest multiple (e.g., 8, 16, 64)."""
    if val <= 0:
        return min_val
    remainder = val % multiple
    if remainder == 0:
        aligned = val
    elif remainder >= multiple / 2:
        aligned = val + (multiple - remainder)
    else:
        aligned = val - remainder
    
    return max(min_val, min(aligned, max_val))


def downscale_resolution_for_oom(
    width: Optional[int],
    height: Optional[int],
    scale_factor: float = 0.75,
    multiple: int = 8,
    min_dim: int = 384
) -> Tuple[int, int]:
    """Downscale width and height to fit memory constraints while maintaining aspect ratio and alignment."""
    w = width or 1024
    h = height or 1024

    # Calculate scaled dimensions
    new_w = max(min_dim, int(w * scale_factor))
    new_h = max(min_dim, int(h * scale_factor))

    # Align to required multiple
    aligned_w = align_dimension(new_w, multiple=multiple, min_val=min_dim)
    aligned_h = align_dimension(new_h, multiple=multiple, min_val=min_dim)

    return aligned_w, aligned_h


def find_closest_model(
    requested_model: str,
    available_models: Sequence[str],
    cutoff: float = 0.3
) -> Optional[str]:
    """Find the closest model name in available models using fuzzy matching."""
    if not requested_model or not available_models:
        return None

    cleaned_requested = requested_model.lower().strip()
    available_list = list(available_models)

    # 1. Exact match (case-insensitive)
    for model in available_list:
        if model.lower().strip() == cleaned_requested:
            return model

    # 2. Substring matching (e.g., "flux" matches "flux1-dev.safetensors")
    substring_matches = [
        m for m in available_list
        if cleaned_requested in m.lower() or m.lower() in cleaned_requested
    ]
    if substring_matches:
        # Prefer exact extension or closest length
        substring_matches.sort(key=lambda m: abs(len(m) - len(requested_model)))
        return substring_matches[0]

    # 3. Fuzzy ratio match via difflib
    matches = difflib.get_close_matches(
        requested_model,
        available_list,
        n=1,
        cutoff=cutoff
    )
    if matches:
        return matches[0]

    # 4. Try matching without extension
    req_stem = requested_model.rsplit(".", 1)[0].lower()
    for model in available_list:
        model_stem = model.rsplit(".", 1)[0].lower()
        if req_stem in model_stem or model_stem in req_stem:
            return model

    return None


class ErrorDiagnoser:
    """Diagnoses ComfyUI errors and produces structured, actionable recovery recommendations for AI Agents."""

    def __init__(self, comfyui_client=None, defaults_manager=None):
        self.comfyui_client = comfyui_client
        self.defaults_manager = defaults_manager

    def diagnose(
        self,
        error: Any,
        workflow: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        available_models: Optional[Sequence[str]] = None,
        node_error_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Diagnose an error and return a standardized structured self-healing report.

        Returns:
            Dict containing:
                - status: "error"
                - error: Human-readable error summary
                - error_code: Machine-readable uppercase category code
                - diagnosis: Detailed explanation of the root cause
                - retryable: Boolean indicating if an immediate retry with modified params might succeed
                - node_id: Optional string identifying the failing node
                - node_type: Optional string identifying the failing node type
                - suggested_actions: List of structured suggestions with concrete suggested_params
                - raw_error: Original string representation of the exception
        """
        error_str = str(error) if error is not None else ""
        error_lower = error_str.lower()
        params = dict(params) if params else {}

        # Resolve available models if not passed
        models_pool = list(available_models or [])
        if not models_pool and self.defaults_manager is not None:
            models_pool = list(getattr(self.defaults_manager, "_available_models_set", []))
        if not models_pool and self.comfyui_client is not None:
            models_pool = list(getattr(self.comfyui_client, "available_models", []))

        # Extract node details from structured node_error_data or error message
        node_id = None
        node_type = None
        if node_error_data:
            node_id = str(node_error_data.get("node_id", "")) or None
            node_type = node_error_data.get("node_type") or None

        # If not in node_error_data, try regex extraction from string
        if not node_id:
            node_match = re.search(r"Node\s+(\d+|[a-zA-Z0-9_-]+)\s*\(([^)]+)\)", error_str)
            if node_match:
                node_id = node_match.group(1)
                node_type = node_match.group(2)

        # 1. Check for CUDA OOM (Out of Memory)
        if self._is_oom_error(error_lower):
            return self._diagnose_oom(error_str, params, workflow, node_id, node_type)

        # 2. Check for Dimension / Divisibility / Shape Mismatch
        if self._is_dimension_error(error_lower):
            return self._diagnose_dimension(error_str, params, workflow, node_id, node_type)

        # 3. Check for Model / Checkpoint / LoRA / VAE Not Found
        if self._is_model_not_found_error(error_lower):
            return self._diagnose_model_not_found(error_str, params, models_pool, node_id, node_type)

        # 4. Check for Input Image / Asset File Not Found
        if self._is_input_file_error(error_lower):
            return self._diagnose_input_file(error_str, params, node_id, node_type)

        # 5. Check for Parameter Out of Bounds
        if self._is_param_out_of_bounds_error(error_lower):
            return self._diagnose_param_out_of_bounds(error_str, params, node_id, node_type)

        # 6. Check for Preflight / Missing Placeholders
        if "pre-flight validation failed" in error_lower or "unfilled placeholder" in error_lower or "required parameter" in error_lower:
            return self._diagnose_preflight(error_str, params, node_id, node_type)

        # 7. Check for GPU Saturation / Queue Admission rejection
        if "gpu saturated" in error_lower or "admission" in error_lower or "queue is full" in error_lower:
            return {
                "status": "error",
                "error": error_str,
                "error_code": "GPU_SATURATED",
                "diagnosis": "GPU 显存或计算负载持续处于饱和状态，任务被 GPU Guard 保护拦截以防止崩溃。",
                "retryable": True,
                "node_id": node_id,
                "node_type": node_type,
                "suggested_actions": [
                    {
                        "action": "wait_and_retry",
                        "description": "等待 5-15 秒待后台正在运行的任务完成后重新发起生成",
                    },
                    {
                        "action": "inspect_queue",
                        "description": "调用 get_queue_status 查看当前排队任务，或调用 clear_queue 清理堆积任务",
                    }
                ],
                "raw_error": error_str,
            }

        # 8. Check for Timeout
        if "timeout" in error_lower or "still running after" in error_lower:
            return {
                "status": "error",
                "error": error_str,
                "error_code": "TIMEOUT",
                "diagnosis": "任务执行时间超过设定阈值，可能由于复杂模型生成耗时较长或排队等待中。",
                "retryable": True,
                "node_id": node_id,
                "node_type": node_type,
                "suggested_actions": [
                    {
                        "action": "poll_job",
                        "description": "使用 get_job(prompt_id) 继续异步轮询任务执行状态，避免重复提交",
                    }
                ],
                "raw_error": error_str,
            }

        # 9. Generic Node Execution Error
        if node_id or "execution error" in error_lower or "exception_type" in error_lower:
            return {
                "status": "error",
                "error": error_str,
                "error_code": "NODE_EXECUTION_ERROR",
                "diagnosis": f"节点 {node_id or '?'} ({node_type or '未知类型'}) 在执行时发生内部异常: {error_str}",
                "retryable": False,
                "node_id": node_id,
                "node_type": node_type,
                "suggested_actions": [
                    {
                        "action": "inspect_workflow",
                        "description": "检查该节点的输入参数与上游连接数据类型是否匹配，或尝试使用默认参数",
                    }
                ],
                "raw_error": error_str,
            }

        # 10. Fallback Unknown Error
        return {
            "status": "error",
            "error": error_str,
            "error_code": "UNKNOWN_ERROR",
            "diagnosis": f"执行遇到未知错误: {error_str}",
            "retryable": False,
            "node_id": node_id,
            "node_type": node_type,
            "suggested_actions": [
                {
                    "action": "check_comfyui_logs",
                    "description": "检查 ComfyUI 控制台输出以获取完整调试信息",
                }
            ],
            "raw_error": error_str,
        }

    # ==================== Matchers ====================

    @staticmethod
    def _is_oom_error(err: str) -> bool:
        return any(k in err for k in (
            "cuda out of memory",
            "torch.cuda.outofmemoryerror",
            "outofmemoryerror",
            "allocated:",
            "reserved:",
            "not enough memory",
            "tried to allocate",
            "c10::cudaerror",
        ))

    @staticmethod
    def _is_dimension_error(err: str) -> bool:
        return any(k in err for k in (
            "divisible by",
            "dimension must be",
            "dimensions must be",
            "shapes cannot be multiplied",
            "mat1 and mat2 shapes",
            "shape mismatch",
            "spatial dimension mismatch",
            "size mismatch",
            "input size must be",
            "tensor dimension",
            "vae expects",
        ))

    @staticmethod
    def _is_model_not_found_error(err: str) -> bool:
        return any(k in err for k in (
            "not found in comfyui checkpoints",
            "model not found",
            "checkpoint not found",
            "lora not found",
            "vae not found",
            "clip not found",
            "filenotfounderror"
        )) and any(ext in err for ext in (".safetensors", ".ckpt", ".pt", ".bin", "model", "checkpoint", "lora", "vae", "clip"))

    @staticmethod
    def _is_input_file_error(err: str) -> bool:
        return any(k in err for k in (
            "input image",
            "input file",
            "loadimage",
            "not found in comfyui input directory",
            "no such file or directory",
        )) and any(ext in err for ext in (".png", ".jpg", ".jpeg", ".webp", ".mp4", ".mp3", "input"))

    @staticmethod
    def _is_param_out_of_bounds_error(err: str) -> bool:
        return any(k in err for k in (
            "must be between",
            "must be positive",
            "must be greater than",
            "out of range",
            "invalid value for",
            "valueerror: steps",
            "valueerror: cfg",
            "valueerror: denoise",
        ))

    # ==================== Diagnosers ====================

    def _diagnose_oom(
        self,
        error_str: str,
        params: Dict[str, Any],
        workflow: Optional[Dict[str, Any]],
        node_id: Optional[str],
        node_type: Optional[str]
    ) -> Dict[str, Any]:
        """Diagnose CUDA OOM and calculate downscaling suggestions."""
        # Extract width & height from params or workflow
        w = params.get("width")
        h = params.get("height")

        if (w is None or h is None) and workflow:
            for n in workflow.values():
                if isinstance(n, dict) and n.get("class_type") in ("EmptyLatentImage", "EmptySD3LatentImage"):
                    inputs = n.get("inputs", {})
                    if w is None:
                        w = inputs.get("width")
                    if h is None:
                        h = inputs.get("height")

        # Convert to int if found
        try:
            w = int(w) if w is not None else None
            h = int(h) if h is not None else None
        except (ValueError, TypeError):
            w, h = None, None

        # Calculate downscaled dimensions (aligning to 16 for safety across models)
        new_w, new_h = downscale_resolution_for_oom(w, h, scale_factor=0.75, multiple=16)

        suggested_params: Dict[str, Any] = {}
        action_descriptions = []

        if w and h:
            suggested_params["width"] = new_w
            suggested_params["height"] = new_h
            action_descriptions.append(f"将分辨率从 {w}x{h} 降低至 {new_w}x{new_h}")
        else:
            suggested_params["width"] = 768
            suggested_params["height"] = 768
            action_descriptions.append("将分辨率降低至 768x768")

        # Video frames check
        for frame_key in ("length", "frames", "num_frames", "frame_count"):
            if frame_key in params:
                try:
                    current_frames = int(params[frame_key])
                    if current_frames > 16:
                        new_frames = max(16, (current_frames // 2) + (current_frames % 2))
                        suggested_params[frame_key] = new_frames
                        action_descriptions.append(f"将视频帧数 {frame_key} 从 {current_frames} 降低至 {new_frames}")
                except (ValueError, TypeError):
                    pass

        # Batch size check
        if "batch_size" in params:
            try:
                bs = int(params["batch_size"])
                if bs > 1:
                    suggested_params["batch_size"] = 1
                    action_descriptions.append(f"将 batch_size 从 {bs} 降至 1")
            except (ValueError, TypeError):
                pass

        # Steps check
        if "steps" in params:
            try:
                st = int(params["steps"])
                if st > 30:
                    suggested_params["steps"] = 20
                    action_descriptions.append(f"将采样步数 steps 从 {st} 降至 20")
            except (ValueError, TypeError):
                pass

        diagnosis = (
            f"GPU 显存不足 (CUDA Out of Memory)。当前配置消耗显存超出 GPU 物理上限。"
            f"建议降级分辨率或减少视频帧数后重新生成。"
        )

        return {
            "status": "error",
            "error": "CUDA Out of Memory (GPU 显存溢出)",
            "error_code": "CUDA_OOM",
            "diagnosis": diagnosis,
            "retryable": True,
            "node_id": node_id,
            "node_type": node_type,
            "suggested_actions": [
                {
                    "action": "modify_parameters",
                    "description": "；".join(action_descriptions),
                    "suggested_params": suggested_params,
                }
            ],
            "raw_error": error_str,
        }

    def _diagnose_dimension(
        self,
        error_str: str,
        params: Dict[str, Any],
        workflow: Optional[Dict[str, Any]],
        node_id: Optional[str],
        node_type: Optional[str]
    ) -> Dict[str, Any]:
        """Diagnose dimension / divisibility / shape errors and compute nearest valid multiples."""
        # Find required multiple in error message
        multiple = 8
        if "divisible by 16" in error_str or "multiple of 16" in error_str:
            multiple = 16
        elif "divisible by 64" in error_str or "multiple of 64" in error_str:
            multiple = 64
        elif "divisible by 32" in error_str:
            multiple = 32

        w = params.get("width")
        h = params.get("height")

        try:
            w_int = int(w) if w is not None else 512
            h_int = int(h) if h is not None else 512
        except (ValueError, TypeError):
            w_int, h_int = 512, 512

        aligned_w = align_dimension(w_int, multiple=multiple)
        aligned_h = align_dimension(h_int, multiple=multiple)

        diagnosis = (
            f"图像尺寸不满足模型要求（需为 {multiple} 的整数倍或张量形状对齐）。"
            f"传入尺寸为 {w_int}x{h_int}，已自动计算最接近的合规尺寸 {aligned_w}x{aligned_h}。"
        )

        return {
            "status": "error",
            "error": f"Dimension alignment mismatch (尺寸需为 {multiple} 的倍数)",
            "error_code": "DIMENSION_NOT_DIVISIBLE",
            "diagnosis": diagnosis,
            "retryable": True,
            "node_id": node_id,
            "node_type": node_type,
            "suggested_actions": [
                {
                    "action": "modify_parameters",
                    "description": f"将宽高修正为 {multiple} 的整数倍：width={aligned_w}, height={aligned_h}",
                    "suggested_params": {
                        "width": aligned_w,
                        "height": aligned_h,
                    },
                }
            ],
            "raw_error": error_str,
        }

    def _diagnose_model_not_found(
        self,
        error_str: str,
        params: Dict[str, Any],
        available_models: Sequence[str],
        node_id: Optional[str],
        node_type: Optional[str]
    ) -> Dict[str, Any]:
        """Diagnose missing checkpoint / LoRA / VAE and recommend closest matches."""
        requested_model = params.get("model") or params.get("ckpt_name") or params.get("lora_name")

        # Try to extract from error string if not in params
        if not requested_model:
            model_match = re.search(r"['\"]([^'\"]+\.(?:safetensors|ckpt|pt|bin))['\"]", error_str)
            if model_match:
                requested_model = model_match.group(1)

        closest = find_closest_model(str(requested_model or ""), available_models) if requested_model else None

        suggested_actions = []
        if closest:
            param_key = "model"
            if "lora" in str(requested_model).lower() or "lora" in error_str.lower():
                param_key = "lora_name"
            elif "ckpt" in error_str.lower():
                param_key = "ckpt_name"

            suggested_actions.append({
                "action": "switch_model",
                "description": f"切换为名称最相近的已安装模型 '{closest}'",
                "suggested_params": {param_key: closest},
            })

        suggested_actions.append({
            "action": "list_models",
            "description": "调用 list_models 工具查看当前系统已安装的所有可用模型列表",
        })

        diagnosis = (
            f"未找到指定的模型文件 '{requested_model or '未知'}'. "
            + (f"推荐替换为已安装的相似模型 '{closest}'。" if closest else "请从可用模型列表中选择。")
        )

        return {
            "status": "error",
            "error": f"Model not found: {requested_model or error_str}",
            "error_code": "MODEL_NOT_FOUND",
            "diagnosis": diagnosis,
            "retryable": bool(closest),
            "node_id": node_id,
            "node_type": node_type,
            "suggested_actions": suggested_actions,
            "raw_error": error_str,
        }

    def _diagnose_input_file(
        self,
        error_str: str,
        params: Dict[str, Any],
        node_id: Optional[str],
        node_type: Optional[str]
    ) -> Dict[str, Any]:
        """Diagnose missing input asset/image file."""
        file_match = re.search(r"['\"]([^'\"]+\.(?:png|jpg|jpeg|webp|mp4|mp3))['\"]", error_str)
        missing_file = file_match.group(1) if file_match else params.get("image", "输入文件")

        return {
            "status": "error",
            "error": f"Input file not found: {missing_file}",
            "error_code": "INPUT_FILE_NOT_FOUND",
            "diagnosis": f"工作流所需的输入文件 '{missing_file}' 不存在于 ComfyUI input 目录中。",
            "retryable": False,
            "node_id": node_id,
            "node_type": node_type,
            "suggested_actions": [
                {
                    "action": "provide_input_file",
                    "description": f"请先调用文生图工具生成该文件，或将其放置于 ComfyUI input 目录后再引用",
                }
            ],
            "raw_error": error_str,
        }

    def _diagnose_param_out_of_bounds(
        self,
        error_str: str,
        params: Dict[str, Any],
        node_id: Optional[str],
        node_type: Optional[str]
    ) -> Dict[str, Any]:
        """Diagnose parameter out of bounds and provide clamped recommendations."""
        suggested_params: Dict[str, Any] = {}

        if "denoise" in error_str or "denoise" in params:
            try:
                denoise = float(params.get("denoise", 1.0))
                clamped = max(0.0, min(1.0, denoise))
                suggested_params["denoise"] = clamped
            except (ValueError, TypeError):
                suggested_params["denoise"] = 0.75

        if "cfg" in error_str or "cfg" in params:
            try:
                cfg = float(params.get("cfg", 3.5))
                clamped = max(1.0, min(30.0, cfg))
                suggested_params["cfg"] = clamped
            except (ValueError, TypeError):
                suggested_params["cfg"] = 3.5

        if "steps" in error_str or "steps" in params:
            try:
                steps = int(params.get("steps", 20))
                clamped = max(1, min(100, steps))
                suggested_params["steps"] = clamped
            except (ValueError, TypeError):
                suggested_params["steps"] = 20

        return {
            "status": "error",
            "error": f"Parameter out of bounds: {error_str}",
            "error_code": "PARAM_OUT_OF_BOUNDS",
            "diagnosis": f"部分参数超出有效取值范围: {error_str}。",
            "retryable": True,
            "node_id": node_id,
            "node_type": node_type,
            "suggested_actions": [
                {
                    "action": "modify_parameters",
                    "description": "使用修正后的合理区间参数重试",
                    "suggested_params": suggested_params,
                }
            ],
            "raw_error": error_str,
        }

    def _diagnose_preflight(
        self,
        error_str: str,
        params: Dict[str, Any],
        node_id: Optional[str],
        node_type: Optional[str]
    ) -> Dict[str, Any]:
        """Diagnose preflight validation failure."""
        return {
            "status": "error",
            "error": error_str,
            "error_code": "PREFLIGHT_FAILED",
            "diagnosis": "工作流预检失败（包含未填充的占位符或缺失必填参数），未提交至 ComfyUI 执行队列。",
            "retryable": False,
            "node_id": node_id,
            "node_type": node_type,
            "suggested_actions": [
                {
                    "action": "check_parameters",
                    "description": "请检查必填参数并完整提供后重试",
                }
            ],
            "raw_error": error_str,
        }
