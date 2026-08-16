"""Regression tests for the GPU guard's absolute free-VRAM floor.

ComfyUI keeps models resident in VRAM, so an idle GPU can sit at ~96% used with
an empty queue. Light image jobs may still succeed (ComfyUI evicts cache), but
heavy video/audio decode under the floor is the documented VAE-decode
access-violation trigger. The guard must refuse heavy (or queued) submissions
under ``GPU_MIN_FREE_GB`` even when the queue is empty.
"""

import managers.gpu_guard as gg
from managers.gpu_guard import GpuGuard


def _guard_with_stats(monkeypatch, gpu_util, free_gb, pending=0, min_free_gb=2.0):
    monkeypatch.setattr(gg, "GPU_GUARD_ENABLED", True)
    monkeypatch.setattr(gg, "GPU_MIN_FREE_GB", min_free_gb)
    monkeypatch.setattr(gg, "GPU_HIGH_UTIL", 92)
    monkeypatch.setattr(gg, "GPU_GUARD_WINDOW", 3)
    monkeypatch.setattr(GpuGuard, "_sample_gpu_stats", lambda self: (gpu_util, free_gb))
    monkeypatch.setattr(GpuGuard, "_pending_count", lambda self: pending)
    return GpuGuard("http://localhost:8188")


def test_light_job_allowed_under_floor_when_queue_empty(monkeypatch):
    guard = _guard_with_stats(monkeypatch, gpu_util=96.0, free_gb=0.6, pending=0)
    admission = guard.check_admission(heavy=False)
    assert admission["allowed"] is True
    assert admission["vram_free_gb"] == 0.6


def test_heavy_job_refused_under_floor_even_when_queue_empty(monkeypatch):
    guard = _guard_with_stats(monkeypatch, gpu_util=96.0, free_gb=0.6, pending=0)
    admission = guard.check_admission(heavy=True)
    assert admission["allowed"] is False
    assert "safety floor" in admission["reason"]


def test_any_job_refused_under_floor_when_queue_busy(monkeypatch):
    guard = _guard_with_stats(monkeypatch, gpu_util=96.0, free_gb=0.6, pending=1)
    admission = guard.check_admission(heavy=False)
    assert admission["allowed"] is False


def test_floor_disabled_preserves_old_behaviour(monkeypatch):
    guard = _guard_with_stats(monkeypatch, gpu_util=96.0, free_gb=0.6, pending=1, min_free_gb=0)
    # Sustained saturation still requires three consecutive samples, so one
    # high reading is tolerated; the point here is that no absolute-floor
    # refusal fires when the floor is disabled.
    admission = guard.check_admission(heavy=True)
    assert admission["allowed"] is True
