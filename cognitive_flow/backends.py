"""
Transcription backends for Cognitive Flow.
Provides unified interface for Whisper, Parakeet, and Remote ASR backends.
"""

import ctypes
import io
import json
import os
import sys
import tempfile
import threading
import time
import urllib.request
import urllib.error
import wave
from abc import ABC, abstractmethod
from pathlib import Path
from typing import NamedTuple

# Track if CUDA paths have been set up (shared across backends)
_cuda_paths_configured = False


def setup_cuda_paths():
    """Add NVIDIA pip package DLLs to PATH for CUDA support.

    This is needed for both faster-whisper (ctranslate2) and onnxruntime
    when using nvidia-* pip packages instead of system CUDA install.
    """
    global _cuda_paths_configured
    if _cuda_paths_configured:
        return

    nvidia_packages = [
        "nvidia/cublas/bin",
        "nvidia/cuda_runtime/bin",
        "nvidia/cuda_nvrtc/bin",
        "nvidia/cudnn/bin",
        "nvidia/cufft/bin",
        "nvidia/curand/bin",
        "nvidia/cusolver/bin",
        "nvidia/cusparse/bin",
        "nvidia/nvjitlink/bin",
    ]

    added_paths = []
    for site_dir in sys.path:
        if "site-packages" in site_dir:
            for pkg in nvidia_packages:
                pkg_path = os.path.join(site_dir, pkg)
                if os.path.isdir(pkg_path):
                    if pkg_path not in os.environ.get("PATH", ""):
                        os.environ["PATH"] = pkg_path + os.pathsep + os.environ.get("PATH", "")
                    try:
                        os.add_dll_directory(pkg_path)
                    except (AttributeError, OSError):
                        pass
                    added_paths.append(pkg_path)

    # Preload critical DLLs
    critical_dlls = [
        "cudart64_12.dll",
        "cublas64_12.dll",
        "cublasLt64_12.dll",
        "cudnn64_9.dll",
        "cudnn_ops64_9.dll",
        "cufft64_11.dll",
        "nvrtc64_120_0.dll",
    ]
    loaded = 0
    for dll_dir in added_paths:
        for dll in critical_dlls:
            dll_path = os.path.join(dll_dir, dll)
            if os.path.exists(dll_path):
                try:
                    ctypes.CDLL(dll_path)
                    loaded += 1
                except OSError:
                    pass

    if added_paths:
        print(f"[CUDA] Added {len(added_paths)} NVIDIA paths, preloaded {loaded} DLLs")

    _cuda_paths_configured = True


class TranscriptionResult(NamedTuple):
    """Result from transcription."""
    text: str
    raw_text: str
    duration_ms: float
    segments: list[dict] | None = None


class TranscriptionBackend(ABC):
    """Abstract base class for transcription backends."""

    name: str = "unknown"
    supports_gpu: bool = False

    @abstractmethod
    def load(self, model_name: str, use_gpu: bool = True) -> bool:
        """Load the model. Returns True on success."""
        pass

    @abstractmethod
    def transcribe(self, audio_array, sample_rate: int = 16000) -> TranscriptionResult:
        """Transcribe audio array to text."""
        pass

    @abstractmethod
    def get_available_models(self) -> list[str]:
        """Get list of available model names."""
        pass

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        pass

    @property
    @abstractmethod
    def using_gpu(self) -> bool:
        """Check if using GPU."""
        pass

    def warmup(self):
        """Run a tiny inference to wake the GPU from power-saving.

        Called when recording starts so the GPU is warm by the time
        transcription begins. No-op if not using GPU. Thread-safe --
        wait_for_warmup() blocks until this completes.
        """
        if not self.is_loaded or not self.using_gpu:
            return
        if not hasattr(self, '_warmup_event'):
            self._warmup_event = threading.Event()
            self._warmup_event.set()
        self._warmup_event.clear()
        try:
            import numpy as np
            # 0.1s of silence - just enough to poke the CUDA context awake
            silence = np.zeros(1600, dtype=np.float32)
            self.transcribe(silence, sample_rate=16000)
        except Exception:
            pass  # Best-effort, don't interfere with recording
        finally:
            self._warmup_event.set()

    def wait_for_warmup(self):
        """Block until any in-progress warmup completes."""
        if hasattr(self, '_warmup_event'):
            self._warmup_event.wait()


class WhisperBackend(TranscriptionBackend):
    """faster-whisper backend (default)."""

    name = "whisper"
    supports_gpu = True

    MODELS = ["tiny", "base", "small", "medium", "large"]

    def __init__(self):
        self._model = None
        self._model_name = None
        self._using_gpu = False

    def load(self, model_name: str, use_gpu: bool = True) -> bool:
        """Load Whisper model."""
        # Set up CUDA paths before importing faster_whisper
        if use_gpu:
            setup_cuda_paths()

        from faster_whisper import WhisperModel

        try:
            if use_gpu:
                try:
                    self._model = WhisperModel(
                        model_name,
                        device="cuda",
                        compute_type="float32",
                        device_index=0
                    )
                    self._using_gpu = True
                    self._model_name = model_name
                    return True
                except Exception as e:
                    print(f"[Whisper] GPU failed: {e}, falling back to CPU")

            # CPU fallback
            self._model = WhisperModel(
                model_name,
                device="cpu",
                compute_type="int8",
                cpu_threads=4,
                num_workers=1
            )
            self._using_gpu = False
            self._model_name = model_name
            return True

        except Exception as e:
            print(f"[Whisper] Failed to load {model_name}: {e}")
            return False

    def transcribe(self, audio_array, sample_rate: int = 16000) -> TranscriptionResult:
        """Transcribe using Whisper."""
        if not self._model:
            raise RuntimeError("Model not loaded")

        _start = time.perf_counter()

        segments, info = self._model.transcribe(
            audio_array,
            beam_size=5,
            language="en",
            vad_filter=False,
            word_timestamps=False,
            condition_on_previous_text=True,
            no_speech_threshold=0.9,
            hallucination_silence_threshold=None,
            initial_prompt="Transcribe naturally with contractions: isn't, don't, won't, can't, wasn't, shouldn't, couldn't, wouldn't.",
        )

        segment_list = list(segments)
        raw_text = " ".join([seg.text.strip() for seg in segment_list])
        duration_ms = (time.perf_counter() - _start) * 1000

        return TranscriptionResult(
            text=raw_text,
            raw_text=raw_text,
            duration_ms=duration_ms,
            segments=[{"text": s.text, "start": s.start, "end": s.end} for s in segment_list]
        )

    def get_available_models(self) -> list[str]:
        return self.MODELS.copy()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def using_gpu(self) -> bool:
        return self._using_gpu


class ParakeetBackend(TranscriptionBackend):
    """NVIDIA Parakeet backend via onnx-asr (faster, more accurate)."""

    name = "parakeet"
    supports_gpu = True

    # Available Parakeet models via onnx-asr
    MODELS = [
        "nemo-parakeet-tdt-0.6b-v2",      # English only, fastest
        "nemo-parakeet-tdt-0.6b-v3",      # 25 European languages
        "nemo-parakeet-tdt-0.6b-v3-int8", # Quantized, smaller/faster
    ]

    MODEL_DISPLAY_NAMES = {
        "nemo-parakeet-tdt-0.6b-v2": "Parakeet v2 (English)",
        "nemo-parakeet-tdt-0.6b-v3": "Parakeet v3 (Multilingual)",
        "nemo-parakeet-tdt-0.6b-v3-int8": "Parakeet v3 INT8 (Fast)",
    }

    def __init__(self):
        self._model = None
        self._model_name = None
        self._using_gpu = False

    def load(self, model_name: str, use_gpu: bool = True) -> bool:
        """Load Parakeet model via onnx-asr."""
        try:
            # Set up CUDA paths before importing onnx_asr
            if use_gpu:
                setup_cuda_paths()

            import onnx_asr
            import onnxruntime as ort

            # Suppress onnxruntime warnings (Memcpy node warnings)
            sess_options = ort.SessionOptions()
            sess_options.log_severity_level = 3  # Error only (suppress warnings)

            # CUDA provider options
            cuda_options = {
                'cudnn_conv_algo_search': 'EXHAUSTIVE',  # Find fastest conv algorithms
            }

            # Try GPU first, then CPU fallback
            if use_gpu:
                try:
                    self._model = onnx_asr.load_model(
                        model_name,
                        providers=[("CUDAExecutionProvider", cuda_options), "CPUExecutionProvider"],
                        sess_options=sess_options
                    )
                    self._using_gpu = True
                    print(f"[Parakeet] Loaded {model_name} (GPU)")
                except Exception as e:
                    print(f"[Parakeet] GPU load failed: {e}")
                    print("[Parakeet] Falling back to CPU...")
                    try:
                        self._model = onnx_asr.load_model(
                            model_name,
                            providers=["CPUExecutionProvider"],
                            sess_options=sess_options
                        )
                        self._using_gpu = False
                    except Exception as cpu_e:
                        print(f"[Parakeet] CPU fallback also failed: {cpu_e}")
                        return False
            else:
                self._model = onnx_asr.load_model(
                    model_name,
                    providers=["CPUExecutionProvider"],
                    sess_options=sess_options
                )
                self._using_gpu = False

            self._model_name = model_name
            return True

        except ImportError:
            print("[Parakeet] onnx-asr not installed. Run: pip install onnx-asr[gpu,hub]")
            return False
        except Exception as e:
            print(f"[Parakeet] Failed to load {model_name}: {e}")
            return False

    def transcribe(self, audio_array, sample_rate: int = 16000) -> TranscriptionResult:
        """Transcribe using Parakeet."""
        if not self._model:
            raise RuntimeError("Model not loaded")

        import tempfile
        import soundfile as sf

        _start = time.perf_counter()

        # onnx-asr expects a file path, so write to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
            sf.write(temp_path, audio_array, sample_rate)

        try:
            result = self._model.recognize(temp_path)
            # Result is typically just the text string
            raw_text = result if isinstance(result, str) else str(result)
        finally:
            Path(temp_path).unlink(missing_ok=True)

        duration_ms = (time.perf_counter() - _start) * 1000

        return TranscriptionResult(
            text=raw_text,
            raw_text=raw_text,
            duration_ms=duration_ms,
            segments=None
        )

    def get_available_models(self) -> list[str]:
        return self.MODELS.copy()

    def get_display_name(self, model_name: str) -> str:
        """Get human-readable name for model."""
        return self.MODEL_DISPLAY_NAMES.get(model_name, model_name)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def using_gpu(self) -> bool:
        return self._using_gpu

    @staticmethod
    def cleanup_failed_download(model_name: str) -> bool:
        """Clean up corrupted or partial model downloads.

        Returns True if cleanup was performed, False otherwise.
        """
        try:
            import shutil

            # onnx-asr uses HuggingFace Hub cache
            # Models are stored in ~/.cache/huggingface/hub/
            cache_dir = Path.home() / ".cache" / "huggingface" / "hub"

            if not cache_dir.exists():
                return False

            cleaned = False

            # Look for model directories matching the model name
            # HF hub uses format like "models--nvidia--parakeet-tdt-0.6b"
            # onnx-asr model names like "nemo-parakeet-tdt-0.6b-v2" map to HF names
            model_patterns = [
                f"*parakeet*{model_name.split('-')[-1]}*",  # e.g., *parakeet*v2*
                f"*{model_name.replace('-', '*')}*",
            ]

            for pattern in model_patterns:
                for model_dir in cache_dir.glob(pattern):
                    if model_dir.is_dir():
                        # Check for incomplete downloads (blobs with .incomplete suffix)
                        blobs_dir = model_dir / "blobs"
                        if blobs_dir.exists():
                            incomplete_files = list(blobs_dir.glob("*.incomplete"))
                            if incomplete_files:
                                print(f"[Parakeet] Found {len(incomplete_files)} incomplete downloads in {model_dir.name}")
                                # Remove the entire model directory
                                shutil.rmtree(model_dir)
                                print(f"[Parakeet] Cleaned up corrupted cache: {model_dir.name}")
                                cleaned = True
                                continue

                        # Also check for lock files that indicate interrupted downloads
                        lock_files = list(model_dir.glob("*.lock"))
                        if lock_files:
                            print(f"[Parakeet] Found stale lock files in {model_dir.name}")
                            for lock in lock_files:
                                lock.unlink(missing_ok=True)
                            cleaned = True

            return cleaned

        except Exception as e:
            print(f"[Parakeet] Cleanup failed: {e}")
            return False


class RemoteBackend(TranscriptionBackend):
    """Remote transcription backend - sends audio to an HTTP server."""

    name = "remote"
    supports_gpu = False  # GPU is server-side, not our concern

    def __init__(self):
        self._server_url = None
        self._server_model = "remote"
        self._server_gpu = "unknown"
        self._loaded = False
        self._timeout = 120  # max timeout (cold-start safety net)
        self._ready_timeout = 30  # timeout when server confirmed ready
        self._fast_fail_timeout = 10  # timeout when server was unreachable
        self._warmup_succeeded = False
        self._server_cold_start = False
        self._server_reachable = True

    def load(self, model_name: str, use_gpu: bool = True) -> bool:
        """Connect to remote server. model_name is the server URL."""
        if not model_name or not model_name.strip():
            print("[Remote] No server URL configured")
            return False
        self._server_url = model_name.rstrip('/')
        try:
            info = self._health_check()
            if info and info.get('status') in ('ready', 'idle'):
                self._server_model = info.get('model', 'remote')
                self._server_gpu = info.get('gpu', 'unknown')
                self._loaded = True
                status = info.get('status')
                print(f"[Remote] Connected to {self._server_url} ({self._server_model}, GPU: {self._server_gpu}, status: {status})")
                return True
            else:
                status = info.get('status', 'unknown') if info else 'no response'
                print(f"[Remote] Server not ready: {status}")
                return False
        except Exception as e:
            print(f"[Remote] Failed to connect to {self._server_url}: {e}")
            return False

    def transcribe(self, audio_array, sample_rate: int = 16000) -> TranscriptionResult:
        """Send audio to remote server for transcription."""
        if not self._loaded:
            raise RuntimeError("Remote backend not connected")

        import numpy as np
        _t = time.perf_counter

        # Adaptive timeout based on warmup results
        if self._warmup_succeeded:
            timeout = self._ready_timeout
        elif not self._server_reachable:
            timeout = self._fast_fail_timeout
        else:
            timeout = self._timeout

        # Convert float32 array to int16 WAV bytes
        _start = _t()
        wav_bytes = self._encode_wav(audio_array, sample_rate)
        encode_ms = (_t() - _start) * 1000

        # Build multipart form data
        boundary = '----CognitiveFlowBoundary9876543210'
        body = self._build_multipart(boundary, 'audio', 'audio.wav', 'audio/wav', wav_bytes)
        payload_kb = len(body) / 1024

        url = f"{self._server_url}/transcribe"
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                'Content-Type': f'multipart/form-data; boundary={boundary}',
                'Content-Length': str(len(body)),
            },
            method='POST'
        )

        _network_start = _t()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode('utf-8'))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8', errors='replace')
            raise RuntimeError(f"Server returned {e.code}: {error_body}")
        except (urllib.error.URLError, TimeoutError) as e:
            reason = str(getattr(e, 'reason', e))
            if 'timed out' in reason.lower() or isinstance(e, TimeoutError):
                raise RuntimeError(f"Server timed out ({timeout}s) - server may still be loading model")
            raise RuntimeError(f"Connection failed: {reason}")
        network_ms = (_t() - _network_start) * 1000

        # Successful transcription confirms server is warm and reachable
        self._server_reachable = True
        self._warmup_succeeded = True

        raw_text = data.get('text', '').strip()
        server_ms = data.get('processing_time_ms') or 0
        overhead_ms = network_ms - server_ms if server_ms else 0

        # Store timing breakdown for pipeline logging
        self.last_timings = {
            'encode_ms': encode_ms,
            'payload_kb': payload_kb,
            'network_ms': network_ms,
            'server_ms': server_ms,
            'overhead_ms': overhead_ms,
        }

        # Log network pipeline breakdown
        print(f"[Remote] encode={encode_ms:.1f}ms | upload={payload_kb:.1f}KB | "
              f"network={network_ms:.1f}ms (server={server_ms:.1f}ms + overhead={overhead_ms:.1f}ms)")

        # Use round-trip time as duration (includes network, which is what the user experiences)
        duration_ms = network_ms

        return TranscriptionResult(
            text=raw_text,
            raw_text=raw_text,
            duration_ms=duration_ms,
            segments=data.get('segments')
        )

    def get_available_models(self) -> list[str]:
        return [self._server_model]

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def using_gpu(self) -> bool:
        return False  # From client's perspective

    def warmup(self):
        """Ping server; if idle, pre-load model during recording time."""
        if not self._loaded or not self._server_url:
            return
        if not hasattr(self, '_warmup_event'):
            self._warmup_event = threading.Event()
            self._warmup_event.set()
        self._warmup_event.clear()
        self._warmup_succeeded = False
        self._server_cold_start = False
        self._server_reachable = True
        try:
            info = self._health_check()
            if not info:
                self._server_reachable = False
                print("[Remote] Server unreachable during warmup")
                return

            status = info.get('status', 'unknown')
            if status == 'idle':
                self._server_cold_start = True
                print("[Remote] Server idle - pre-loading model...")
                self._trigger_model_load()
                self._server_cold_start = False
                self._warmup_succeeded = True
            elif status == 'ready':
                self._warmup_succeeded = True
            else:
                print(f"[Remote] Server status: {status}")
        except Exception as e:
            print(f"[Remote] Warmup error: {e}")
        finally:
            self._warmup_event.set()

    def _trigger_model_load(self):
        """Send minimal audio to force server to load its model."""
        import numpy as np
        silence = np.zeros(1600, dtype=np.float32)  # 0.1s
        wav_bytes = self._encode_wav(silence, 16000)
        boundary = '----CognitiveFlowWarmup'
        body = self._build_multipart(boundary, 'audio', 'warmup.wav', 'audio/wav', wav_bytes)
        url = f"{self._server_url}/transcribe"
        req = urllib.request.Request(
            url, data=body,
            headers={
                'Content-Type': f'multipart/form-data; boundary={boundary}',
                'Content-Length': str(len(body)),
            },
            method='POST'
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                resp.read()
            print("[Remote] Model loaded (warmup complete)")
        except Exception as e:
            print(f"[Remote] Model pre-load failed: {e}")

    def _health_check(self) -> dict | None:
        """GET /health and return parsed JSON, or None on failure."""
        url = f"{self._server_url}/health"
        req = urllib.request.Request(url, headers={'User-Agent': 'CognitiveFlow'})
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return json.loads(resp.read().decode('utf-8'))
        except Exception:
            return None

    @staticmethod
    def _encode_wav(audio_array, sample_rate: int) -> bytes:
        """Convert float32 numpy array to WAV bytes (int16 PCM)."""
        import numpy as np
        # Clamp and convert to int16
        clipped = np.clip(audio_array, -1.0, 1.0)
        int16_data = (clipped * 32767).astype(np.int16)

        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(int16_data.tobytes())
        return buf.getvalue()

    @staticmethod
    def _build_multipart(boundary: str, field_name: str, filename: str,
                         content_type: str, data: bytes) -> bytes:
        """Build multipart/form-data body manually."""
        parts = []
        parts.append(f'--{boundary}'.encode())
        parts.append(f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"'.encode())
        parts.append(f'Content-Type: {content_type}'.encode())
        parts.append(b'')
        parts.append(data)
        parts.append(f'--{boundary}--'.encode())
        return b'\r\n'.join(parts)

    def test_connection(self) -> dict | None:
        """Test connection and return server info. For UI Test button."""
        return self._health_check()


# Backend registry
BACKENDS = {
    "whisper": WhisperBackend,
    "parakeet": ParakeetBackend,
    "remote": RemoteBackend,
}


def get_backend(name: str) -> TranscriptionBackend:
    """Get a backend instance by name."""
    if name not in BACKENDS:
        raise ValueError(f"Unknown backend: {name}. Available: {list(BACKENDS.keys())}")
    return BACKENDS[name]()


def get_available_backends() -> list[str]:
    """Get list of available backend names."""
    available = []

    # Check Whisper
    try:
        from faster_whisper import WhisperModel
        available.append("whisper")
    except ImportError:
        pass

    # Check Parakeet
    try:
        import onnx_asr
        available.append("parakeet")
    except ImportError:
        pass

    # Remote is always available (no local deps)
    available.append("remote")

    return available
