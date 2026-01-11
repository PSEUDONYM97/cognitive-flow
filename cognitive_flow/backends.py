"""
Transcription backends for Cognitive Flow.
Provides unified interface for Whisper and Parakeet ASR models.
"""

import ctypes
import os
import sys
import tempfile
import time
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

            # Try GPU first, then CPU fallback
            if use_gpu:
                try:
                    # Try CUDA - but onnxruntime may fail if CUDA libs missing
                    self._model = onnx_asr.load_model(
                        model_name,
                        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
                    )
                    # Check if CUDA actually loaded (onnxruntime silently falls back)
                    self._using_gpu = True
                    print(f"[Parakeet] Loaded {model_name} (GPU attempt)")
                except Exception as e:
                    print(f"[Parakeet] GPU load failed: {e}")
                    print("[Parakeet] Falling back to CPU...")
                    try:
                        self._model = onnx_asr.load_model(
                            model_name,
                            providers=["CPUExecutionProvider"]
                        )
                        self._using_gpu = False
                    except Exception as cpu_e:
                        print(f"[Parakeet] CPU fallback also failed: {cpu_e}")
                        return False
            else:
                self._model = onnx_asr.load_model(
                    model_name,
                    providers=["CPUExecutionProvider"]
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


# Backend registry
BACKENDS = {
    "whisper": WhisperBackend,
    "parakeet": ParakeetBackend,
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

    return available
