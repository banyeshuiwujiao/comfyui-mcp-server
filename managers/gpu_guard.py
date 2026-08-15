"""GPU pressure guard for the ComfyUI MCP server.

MiniMax H3 video decode is sensitive to VRAM exhaustion: when the shared
ComfyUI process is hammered with back-to-back heavy jobs, the VAE decode
stage can crash with an access violation (not a workflow error). This guard
gives the MCP layer a cheap, best-effort way to refuse new submissions when
the GPU is already saturated, and to surface a clear, actionable message
instead of letting the call hang or crash ComfyUI.

Design is intentionally non-fatal and advisory:
  - It NEVER kills ComfyUI.
  - A single high-reading is tolerated (noise / transient spike).
  - Only sustained high utilization (>= threshold for >= window samples)
    triggers a refusal.
  - All behaviour is gated behind COMFY_MCP_GPU_GUARD (default on).
"""

import logging
import os
import threading
import time

import requests

logger = logging.getLogger("MCP_Server")

GPU_GUARD_ENABLED = os.getenv("COMFY_MCP_GPU_GUARD", "1") not in ("0", "false", "False", "")
GPU_HIGH_UTIL = float(os.getenv("COMFY_MCP_GPU_HIGH_UTIL", "92"))      # percent
GPU_GUARD_WINDOW = int(os.getenv("COMFY_MCP_GPU_GUARD_WINDOW", "3"))   # consecutive high samples
GPU_SAMPLE_TIMEOUT = float(os.getenv("COMFY_MCP_GPU_SAMPLE_TIMEOUT", "3.0"))
# A high sample is only "fresh" for this long. Stale (e.g. timeout-window old)
# samples no longer count toward sustained saturation, so the guard cannot
# get wedged in a permanently-refusing state after a long decode.
GPU_SAMPLE_TTL = float(os.getenv("COMFY_MCP_GPU_SAMPLE_TTL", "30.0"))


class GpuGuard:
    """Tracks recent GPU utilization samples and advises on admission.

    The admission decision is based on how many *fresh* consecutive high-util
    samples we have seen. A sample older than ``GPU_SAMPLE_TTL`` seconds is
    dropped, which bounds the refusal window: after a heavy job finishes (or
    the caller times out), the high-reading decays on its own even if
    ``note_completed`` is never called. This prevents the "stuck refusing"
    deadlock reported when ``run_custom_workflow`` returns a job handle on
    timeout without invoking ``note_completed``.
    """

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._lock = threading.Lock()
        self._high_samples: list[float] = []  # timestamps of fresh high-util samples

    def _sample_gpu_util(self) -> float | None:
        """Return max GPU VRAM-utilization percent across devices, or None on error.

        ComfyUI's /system_stats returns a top-level "devices" list. Each device
        exposes `vram_total` and `vram_free` (bytes). Some builds also expose a
        direct `gpu_utilization` field (often None). We compute used-VRAM ratio
        as the saturation signal because it is the most reliably populated field.
        """
        try:
            resp = requests.get(f"{self.base_url}/system_stats", timeout=GPU_SAMPLE_TIMEOUT)
            if resp.status_code != 200:
                return None
            data = resp.json()
            devices = data.get("devices") or data.get("system", {}).get("devices") or []
            if not devices:
                return None
            utils = []
            for d in devices:
                # Prefer an explicit utilization if present and non-zero
                if d.get("gpu_utilization"):
                    try:
                        utils.append(float(d["gpu_utilization"]))
                    except (TypeError, ValueError):
                        pass
                # Fall back to VRAM used ratio (most reliably populated)
                total = d.get("vram_total") or d.get("vram", {}).get("total") or 0
                free = d.get("vram_free") or d.get("vram", {}).get("free") or 0
                if total and total > 0:
                    used = total - free
                    utils.append(100.0 * used / total)
            return max(utils) if utils else None
        except Exception as e:  # noqa: BLE001 - best effort
            logger.debug("GPU sample failed: %s", e)
            return None

    def _pending_count(self) -> int:
        try:
            resp = requests.get(f"{self.base_url}/queue", timeout=GPU_SAMPLE_TIMEOUT)
            if resp.status_code != 200:
                return 0
            data = resp.json()
            return len(data.get("queue_running", [])) + len(data.get("queue_pending", []))
        except Exception:  # noqa: BLE001
            return 0

    def check_admission(self) -> dict:
        """Decide whether to admit a new submission.

        Returns dict with keys:
          allowed (bool): True if submission may proceed.
          reason (str): Human-readable explanation (always present).
          gpu_util (float|None): Last sampled utilization.
          pending (int): Current queue depth.
        """
        if not GPU_GUARD_ENABLED:
            return {"allowed": True, "reason": "GPU guard disabled", "gpu_util": None, "pending": self._pending_count()}

        gpu_util = self._sample_gpu_util()
        pending = self._pending_count()

        with self._lock:
            now = time.monotonic()
            # Drop samples older than the TTL so a single long decode can't
            # wedge the guard into a permanently-refusing state.
            self._high_samples = [t for t in self._high_samples if now - t <= GPU_SAMPLE_TTL]

            if gpu_util is not None and gpu_util >= GPU_HIGH_UTIL and pending > 0:
                self._high_samples.append(now)
            # else: leave the list as-is (fresh samples keep counting)

            saturated = len(self._high_samples) >= GPU_GUARD_WINDOW

            if saturated:
                return {
                    "allowed": False,
                    "reason": (
                        f"GPU saturated: utilization {gpu_util:.0f}% over "
                        f"{GPU_GUARD_WINDOW} consecutive checks with {pending} job(s) "
                        f"still in queue. Refusing to submit another heavy job to avoid "
                        f"a VRAM exhaustion crash (MiniMax H3 decode is especially fragile). "
                        f"Call interrupt()/clear_queue() or wait and retry; "
                        f"or set COMFY_MCP_GPU_GUARD=0 to bypass."
                    ),
                    "gpu_util": gpu_util,
                    "pending": pending,
                }

            # Not saturated: report status but allow.
            note = "GPU load nominal"
            if gpu_util is not None:
                note = f"GPU load {gpu_util:.0f}% (below {GPU_HIGH_UTIL:.0f}% threshold)"
            return {"allowed": True, "reason": note, "gpu_util": gpu_util, "pending": pending}

    def note_completed(self) -> None:
        """Call after a job finishes to opportunistically drop the oldest
        high-util sample (mild decay). Not required for correctness — the TTL
        in ``check_admission`` already bounds the refusal window.
        """
        with self._lock:
            if self._high_samples:
                self._high_samples.pop(0)
