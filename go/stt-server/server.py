"""Local STT Service - Parakeet TDT 0.6B v2 via NeMo.

FastAPI service that accepts audio uploads, converts to 16kHz WAV,
and returns transcriptions. Set DEVICE=cpu to run on CPU (frees GPU for LLM).
"""

import asyncio
import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

import nemo.collections.asr as nemo_asr

log = logging.getLogger("stt")

MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v2"
PORT = int(os.environ.get("PORT", "9200"))
DEVICE = os.environ.get("DEVICE", "cuda")  # "cuda" or "cpu"

app = FastAPI(title="Local STT Service", version="1.1.0")
start_time = time.time()

model = None
device_name = DEVICE


@app.on_event("startup")
async def warmup():
    global model, device_name
    use_gpu = DEVICE == "cuda" and torch.cuda.is_available()
    device_name = "cuda" if use_gpu else "cpu"

    if DEVICE == "cuda" and not use_gpu:
        log.warning("CUDA requested but not available, falling back to CPU")

    log.info("Loading model to %s...", device_name)
    model = nemo_asr.models.ASRModel.from_pretrained(MODEL_NAME)
    model.set_trainer(None)  # prevent NeMo from trying to init GPU trainer

    if use_gpu:
        model = model.cuda()
    else:
        model = model.cpu()
        torch.set_num_threads(os.cpu_count() or 8)

    # Dummy inference to warm up
    dummy_path = "/tmp/_warmup.wav"
    sf.write(dummy_path, np.zeros(16000, dtype=np.float32), 16000)
    try:
        model.transcribe([dummy_path])
    finally:
        os.unlink(dummy_path)

    log.info("Model ready on %s", device_name)


def convert_to_wav(input_bytes: bytes, input_filename: str) -> str:
    """Convert any audio format to 16kHz mono WAV. Returns path to temp WAV file."""
    suffix = Path(input_filename).suffix or ".oga"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as inp:
        inp.write(input_bytes)
        inp_path = inp.name

    out_path = inp_path.rsplit(".", 1)[0] + "_16k.wav"
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", inp_path,
                "-ar", "16000", "-ac", "1", "-f", "wav", out_path,
            ],
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        os.unlink(inp_path)
        raise RuntimeError(f"ffmpeg conversion failed: {e.stderr.decode()}") from e
    finally:
        if os.path.exists(inp_path):
            os.unlink(inp_path)

    return out_path


def _transcribe_sync(audio_path: str) -> dict:
    """Run transcription on a 16kHz WAV file (blocking)."""
    t0 = time.time()

    result = model.transcribe([audio_path])
    # NeMo RNNT/TDT returns tuple: (hypotheses_list, all_hypotheses)
    if isinstance(result, tuple):
        text = result[0][0]
    elif isinstance(result, list):
        text = result[0]
    else:
        text = str(result)

    audio_info = sf.info(audio_path)
    duration_ms = int(audio_info.duration * 1000)
    processing_time_ms = int((time.time() - t0) * 1000)

    return {
        "text": text,
        "language": "en",
        "duration_ms": duration_ms,
        "confidence": 0.95,
        "processing_time_ms": processing_time_ms,
    }


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    wav_path = None
    try:
        audio_bytes = await audio.read()
        if not audio_bytes:
            return JSONResponse(
                status_code=400,
                content={"error": "Empty audio file"},
            )

        wav_path = convert_to_wav(audio_bytes, audio.filename or "audio.oga")
        result = await asyncio.to_thread(_transcribe_sync, wav_path)
        return result

    except RuntimeError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        if wav_path and os.path.exists(wav_path):
            os.unlink(wav_path)


@app.get("/health")
async def health():
    info = {
        "status": "ready" if model is not None else "loading",
        "model": MODEL_NAME,
        "device": device_name,
        "uptime_seconds": int(time.time() - start_time),
    }
    if device_name == "cuda" and torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["vram_used_mb"] = int(torch.cuda.memory_allocated(0) / 1024 / 1024)
    elif device_name == "cpu":
        info["threads"] = torch.get_num_threads()
    return info


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=PORT)
