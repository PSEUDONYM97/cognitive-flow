"""
Tests for cognitive_flow.backends - backend registry, selection logic,
TranscriptionResult, and pure utility functions.

No actual ML models are loaded; no GPU or audio hardware is required.
"""

import io
import wave

import numpy as np
import pytest

from cognitive_flow.backends import (
    BACKENDS,
    TranscriptionResult,
    WhisperBackend,
    ParakeetBackend,
    RemoteBackend,
    get_backend,
    get_available_backends,
)


class TestTranscriptionResult:

    def test_basic_fields(self):
        result = TranscriptionResult(
            text="hello world",
            raw_text="  hello world  ",
            duration_ms=123.4,
        )
        assert result.text == "hello world"
        assert result.raw_text == "  hello world  "
        assert result.duration_ms == 123.4

    def test_segments_default_none(self):
        result = TranscriptionResult(text="a", raw_text="a", duration_ms=0.0)
        assert result.segments is None

    def test_segments_explicit(self):
        segs = [{"text": "hi", "start": 0.0, "end": 1.0}]
        result = TranscriptionResult(
            text="hi", raw_text="hi", duration_ms=50.0, segments=segs
        )
        assert result.segments == segs
        assert len(result.segments) == 1

    def test_named_tuple_unpacking(self):
        result = TranscriptionResult("a", "b", 1.0, None)
        text, raw, dur, segs = result
        assert text == "a" and raw == "b" and dur == 1.0 and segs is None


class TestBackendRegistry:

    def test_registry_contains_all_backends(self):
        assert "whisper" in BACKENDS
        assert "parakeet" in BACKENDS
        assert "remote" in BACKENDS

    def test_registry_maps_to_correct_classes(self):
        assert BACKENDS["whisper"] is WhisperBackend
        assert BACKENDS["parakeet"] is ParakeetBackend
        assert BACKENDS["remote"] is RemoteBackend

    def test_get_backend_returns_correct_instance(self):
        assert isinstance(get_backend("whisper"), WhisperBackend)
        assert isinstance(get_backend("remote"), RemoteBackend)
        assert isinstance(get_backend("parakeet"), ParakeetBackend)

    def test_get_backend_returns_fresh_instance(self):
        b1 = get_backend("remote")
        b2 = get_backend("remote")
        assert b1 is not b2

    def test_get_backend_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown backend.*nonexistent"):
            get_backend("nonexistent")

    def test_get_backend_error_lists_available(self):
        with pytest.raises(ValueError) as exc_info:
            get_backend("invalid")
        msg = str(exc_info.value)
        assert "whisper" in msg and "parakeet" in msg and "remote" in msg


class TestGetAvailableBackends:

    def test_remote_always_available(self):
        assert "remote" in get_available_backends()

    def test_returns_list_of_strings(self):
        available = get_available_backends()
        assert isinstance(available, list)
        assert all(isinstance(b, str) for b in available)


class TestWhisperBackendInit:

    def test_initial_state(self):
        backend = WhisperBackend()
        assert backend.name == "whisper"
        assert backend.supports_gpu is True
        assert backend.is_loaded is False
        assert backend.using_gpu is False

    def test_available_models(self):
        models = WhisperBackend().get_available_models()
        for m in ["tiny", "base", "small", "medium", "large"]:
            assert m in models

    def test_available_models_returns_copy(self):
        backend = WhisperBackend()
        models = backend.get_available_models()
        models.append("xxxl")
        assert "xxxl" not in backend.get_available_models()

    def test_transcribe_before_load_raises(self):
        with pytest.raises(RuntimeError, match="Model not loaded"):
            WhisperBackend().transcribe(np.zeros(1600, dtype=np.float32))


class TestParakeetBackendInit:

    def test_initial_state(self):
        backend = ParakeetBackend()
        assert backend.name == "parakeet"
        assert backend.supports_gpu is True
        assert backend.is_loaded is False
        assert backend.using_gpu is False

    def test_available_models(self):
        models = ParakeetBackend().get_available_models()
        assert len(models) == 3
        assert all("parakeet" in m for m in models)

    def test_display_names(self):
        backend = ParakeetBackend()
        for model_name in backend.MODELS:
            display = backend.get_display_name(model_name)
            assert isinstance(display, str) and len(display) > 0

    def test_display_name_unknown_model(self):
        assert ParakeetBackend().get_display_name("nonexistent-model") == "nonexistent-model"

    def test_transcribe_before_load_raises(self):
        with pytest.raises(RuntimeError, match="Model not loaded"):
            ParakeetBackend().transcribe(np.zeros(1600, dtype=np.float32))


class TestRemoteBackendInit:

    def test_initial_state(self):
        backend = RemoteBackend()
        assert backend.name == "remote"
        assert backend.supports_gpu is False
        assert backend.is_loaded is False
        assert backend.using_gpu is False

    def test_load_empty_url_fails(self):
        backend = RemoteBackend()
        assert backend.load("", use_gpu=False) is False
        assert backend.load("   ", use_gpu=False) is False

    def test_transcribe_before_load_raises(self):
        with pytest.raises(RuntimeError, match="not connected"):
            RemoteBackend().transcribe(np.zeros(1600, dtype=np.float32))


class TestRemoteEncodeWav:

    def test_returns_valid_wav(self):
        audio = np.zeros(16000, dtype=np.float32)
        wav_bytes = RemoteBackend._encode_wav(audio, 16000)
        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 16000
            assert wf.getnframes() == 16000

    def test_clamps_audio_range(self):
        audio = np.array([2.0, -2.0, 0.5], dtype=np.float32)
        wav_bytes = RemoteBackend._encode_wav(audio, 16000)
        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            raw = wf.readframes(wf.getnframes())
        samples = np.frombuffer(raw, dtype=np.int16)
        assert samples[0] == 32767
        assert samples[1] == -32767
        assert 16000 < samples[2] < 16500

    @pytest.mark.parametrize("sample_rate", [8000, 16000, 44100, 48000])
    def test_different_sample_rates(self, sample_rate):
        audio = np.zeros(sample_rate, dtype=np.float32)
        wav_bytes = RemoteBackend._encode_wav(audio, sample_rate)
        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            assert wf.getframerate() == sample_rate


class TestRemoteBuildMultipart:

    def test_structure(self):
        body = RemoteBackend._build_multipart(
            "TestBoundary", "audio", "test.wav", "audio/wav", b"fake-wav-data"
        )
        assert isinstance(body, bytes)
        assert b"--TestBoundary" in body
        assert b"--TestBoundary--" in body
        assert b"Content-Type: audio/wav" in body
        assert b"fake-wav-data" in body

    def test_boundary_at_start_and_end(self):
        body = RemoteBackend._build_multipart(
            "Boundary123", "field", "file.bin", "application/octet-stream", b"data"
        )
        parts = body.split(b"\r\n")
        assert parts[0] == b"--Boundary123"
        assert parts[-1] == b"--Boundary123--"

    def test_binary_payload_preserved(self):
        payload = bytes(range(256))
        body = RemoteBackend._build_multipart(
            "B", "f", "f.bin", "application/octet-stream", payload
        )
        assert payload in body


class TestBackendWarmup:

    def test_warmup_noop_when_not_loaded(self):
        backend = WhisperBackend()
        assert not backend.is_loaded
        backend.warmup()

    def test_wait_for_warmup_without_prior_warmup(self):
        backend = WhisperBackend()
        backend.wait_for_warmup()
