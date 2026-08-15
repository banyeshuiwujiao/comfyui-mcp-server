"""Unit tests for audio multimodal summarizer (BPM, loudness, silence detection, lyrics alignment, waveform rendering)."""

import io
import wave
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from PIL import Image

from asset_processor import (
    analyze_audio_features,
    extract_audio_samples,
    parse_lyrics_sections,
    render_waveform_image,
)
from managers.asset_registry import AssetRegistry
from tools.asset import register_asset_tools


def generate_synthetic_wav_bytes(duration_sec: float = 3.0, sample_rate: int = 44100, bpm: float = 120.0, add_silence: bool = True) -> bytes:
    """Generate in-memory synthetic WAV audio bytes with a rhythmic pulse and optional silence gap."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    
    # Generate beat pulse at requested BPM
    beat_period = 60.0 / bpm
    beat = np.exp(-12 * (t % beat_period)) * np.sin(2 * np.pi * 150 * t)
    tone = 0.4 * np.sin(2 * np.pi * 440 * t)
    signal = (beat + tone) * 0.7

    # Insert 0.4s silence gap in the middle if requested
    if add_silence and duration_sec >= 2.0:
        silence_start_idx = int(1.2 * sample_rate)
        silence_end_idx = int(1.6 * sample_rate)
        signal[silence_start_idx:silence_end_idx] = 0.0

    audio_int16 = (np.clip(signal, -1.0, 1.0) * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())

    return buf.getvalue()


@pytest.fixture
def synthetic_wav_bytes():
    return generate_synthetic_wav_bytes(duration_sec=3.0, sample_rate=44100, bpm=120.0, add_silence=True)


def test_extract_audio_samples(synthetic_wav_bytes):
    """Test extracting normalized float32 samples from audio bytes."""
    samples, rate, dur = extract_audio_samples(synthetic_wav_bytes)
    assert rate == 44100
    assert abs(dur - 3.0) < 0.05
    assert len(samples) == int(rate * 3.0)
    assert np.max(np.abs(samples)) <= 1.0


def test_analyze_audio_features(synthetic_wav_bytes):
    """Test audio feature extraction (RMS, Peak, BPM, and Silence detection)."""
    lyrics_prompt = """[Intro]
Electronic pulses
[Verse]
Walking down the neon highway
[Chorus]
We shine like stars in the cyber sky"""

    features = analyze_audio_features(synthetic_wav_bytes, prompt_lyrics=lyrics_prompt)

    assert features["duration_sec"] == 3.0
    assert features["sample_rate"] == 44100
    assert features["rms_dbfs"] < 0
    assert features["peak_dbfs"] <= 0

    # Verify BPM estimation (target is 120 BPM, allow reasonable tolerance)
    if features["estimated_bpm"] is not None:
        assert 100 <= features["estimated_bpm"] <= 140

    # Verify silence detection detected the ~1.2s to 1.6s gap
    assert features["has_silence"] is True
    assert len(features["silent_segments"]) >= 1
    found_mid_silence = any(1.0 <= s["start_sec"] <= 1.3 for s in features["silent_segments"])
    assert found_mid_silence

    # Verify lyrics sections parsed
    sections = features["lyrics_sections"]
    assert len(sections) == 3
    assert sections[0]["section"] == "Intro"
    assert sections[1]["section"] == "Verse"
    assert sections[2]["section"] == "Chorus"
    assert sections[0]["start_sec"] == 0.0
    assert sections[-1]["end_sec"] == 3.0


def test_parse_lyrics_sections():
    """Test lyrics section segmentation with timestamps."""
    text = """[Intro]
Guitar riff
[Verse 1]
Early in the morning
[Bridge]
Waiting for the sun
[Outro]
Fading into light"""

    sections = parse_lyrics_sections(text, duration_sec=40.0)
    assert len(sections) == 4
    assert [s["section"] for s in sections] == ["Intro", "Verse 1", "Bridge", "Outro"]
    assert sections[0]["start_sec"] == 0.0
    assert sections[0]["end_sec"] == 10.0
    assert sections[-1]["end_sec"] == 40.0


def test_render_waveform_image(synthetic_wav_bytes):
    """Test rendering visual waveform image."""
    wf_bytes = render_waveform_image(synthetic_wav_bytes, width=400, height=100)
    assert len(wf_bytes) > 0

    with Image.open(io.BytesIO(wf_bytes)) as img:
        assert img.format == "WEBP"
        assert img.width == 400
        assert img.height == 100


def test_mcp_tools_view_audio_preview_and_analyze_audio(synthetic_wav_bytes):
    """Test MCP tool layer integration for audio preview and feature analysis."""
    registry = AssetRegistry(ttl_hours=24, db_path=":memory:")

    rec_audio = registry.register_asset(
        filename="cyber_melody.wav",
        subfolder="",
        folder_type="output",
        workflow_id="generate_song",
        prompt_id="prompt_audio_123",
        mime_type="audio/wav",
        bytes_size=len(synthetic_wav_bytes),
        prompt="[Intro] Synth vibe [Verse] City lights [Chorus] Electric dreams"
    )

    captured_tools = {}
    mock_mcp = MagicMock()
    def mock_tool_decorator():
        def decorator(fn):
            captured_tools[fn.__name__] = fn
            return fn
        return decorator
    mock_mcp.tool = mock_tool_decorator

    register_asset_tools(mock_mcp, registry)

    assert "view_audio_preview" in captured_tools
    assert "analyze_audio" in captured_tools

    with patch("tools.asset.fetch_asset_bytes", return_value=synthetic_wav_bytes):
        # 1. Test view_audio_preview in waveform mode
        wf_res = captured_tools["view_audio_preview"](asset_id=rec_audio.asset_id, mode="waveform")
        assert hasattr(wf_res, "data")
        assert getattr(wf_res, "_format", None) == "webp"
        assert len(wf_res.data) > 0

        # 2. Test view_audio_preview in analysis mode
        analysis_res = captured_tools["view_audio_preview"](asset_id=rec_audio.asset_id, mode="analysis")
        assert analysis_res["asset_id"] == rec_audio.asset_id
        assert "rms_dbfs" in analysis_res
        assert len(analysis_res["lyrics_sections"]) == 3

        # 3. Test analyze_audio dedicated tool
        diag_res = captured_tools["analyze_audio"](asset_id=rec_audio.asset_id)
        assert diag_res["asset_id"] == rec_audio.asset_id
        assert diag_res["duration_sec"] == 3.0

        # 4. Test view_image automatic audio waveform fallback
        img_res = captured_tools["view_image"](asset_id=rec_audio.asset_id, mode="thumb")
        assert hasattr(img_res, "data")
        assert getattr(img_res, "_format", None) == "webp"
        assert len(img_res.data) > 0
