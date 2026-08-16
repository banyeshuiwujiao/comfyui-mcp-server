"""DefaultsManager regression tests: model default hygiene.

The hardcoded defaults must not ship machine-specific model filenames:
workflow JSONs carry their own model nodes, and injecting a bogus global
default produced startup warnings and generation errors on installs that
did not have the referenced checkpoint. Configured defaults (env / config /
set_defaults) must still be validated and warn when missing.
"""

import logging

from managers.defaults_manager import DefaultsManager


class FakeComfyClient:
    """Minimal ComfyUIClient stand-in: DefaultsManager only reads available_models."""

    def __init__(self, models):
        self.available_models = list(models)


def test_no_hardcoded_model_default_no_startup_warnings(caplog):
    client = FakeComfyClient(["z_image_turbo.safetensors", "ace_step_v1_3.5b.safetensors"])
    with caplog.at_level(logging.WARNING):
        DefaultsManager(client)

    for record in caplog.records:
        assert "not found in ComfyUI checkpoints" not in record.getMessage()


def test_configured_missing_model_still_warns(monkeypatch, caplog):
    monkeypatch.setenv("COMFY_MCP_DEFAULT_IMAGE_MODEL", "missing-checkpoint.ckpt")
    client = FakeComfyClient(["z_image_turbo.safetensors"])

    with caplog.at_level(logging.WARNING):
        manager = DefaultsManager(client)

    assert manager.get_default("image", "model") == "missing-checkpoint.ckpt"
    assert any("not found in ComfyUI checkpoints" in r.getMessage() for r in caplog.records)
