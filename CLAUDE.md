# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Cognitive Flow is a Windows voice-to-text application using local, GPU-accelerated transcription. Press tilde (~) to record, speak, press tilde again - text types into the focused window. Clicking the overlay indicator records in clipboard mode instead (copies to clipboard).

Triple backend support: Whisper (OpenAI via faster-whisper), Parakeet (NVIDIA via onnx-asr), or Remote (HTTP server). Switch in Settings.

## Commands

```cmd
# Install
pip install -r requirements.txt
pip install -e .

# Run
cognitive-flow              # Background mode (detached, no console)
cognitive-flow --debug      # Foreground with verbose timing + file logging

# Or without installing
python -m cognitive_flow
python -m cognitive_flow --debug

# GPU acceleration (optional but 45x faster)
pip install nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12

# Startup task (runs at Windows login)
python -m cognitive_flow.warmup --install
python -m cognitive_flow.warmup --uninstall
```

## Versioning Rules

**Every feature/fix MUST update:**
1. `cognitive_flow/__init__.py` - bump `__version__`
2. `CHANGELOG.md` - add entry for the new version
3. Commit message references the version

Format: `MAJOR.MINOR.PATCH` (patch=bugfix, minor=feature, major=breaking)

## Architecture

### Startup Sequence (spans multiple files)

1. `main()` in `app.py` parses args. Without `--debug`, respawns as detached `pythonw` process and exits.
2. `init_app()` runs deferred imports: logger, PyAudio, pystray, PyQt6 UI. Always enables file logging to `%APPDATA%\CognitiveFlow\debug_transcriptions.log` (not just in --debug mode). `--debug` controls verbose console output.
3. GPU detection: checks if `nvidia/` subdirectory exists under `site.USER_SITE`. Does NOT load CUDA DLLs yet - that happens in `backends.py` when a backend loads.
4. `CognitiveFlowApp.__init__()` sets up audio, stats, text processor, tray, UI. Reads config from `%APPDATA%\CognitiveFlow\config.json`.
5. `load_model()` spawns a background thread to load the selected backend (Whisper or Parakeet). On failure, cascading fallback: Parakeet -> user's Whisper model -> base Whisper on CPU.
6. `install_hook()` registers Windows low-level keyboard hook (`WH_KEYBOARD_LL`) on main thread.
7. `message_loop()` pumps both Windows messages and Qt events on main thread. Detects sleep/wake gaps (30s+) and triggers GPU warmup.

### Backend System (backends.py)

`TranscriptionBackend` ABC defines the interface. Three implementations:
- **WhisperBackend** - Uses `faster-whisper`. GPU loads `float32`, CPU loads `int8`.
- **ParakeetBackend** - Uses `onnx-asr`. Requires writing audio to temp file (onnx-asr expects file paths).
- **RemoteBackend** - Sends audio over HTTP to an external STT server. No local GPU needed. Encodes audio as WAV via stdlib `wave`/`io`, uploads multipart form via `urllib.request`. Server API: `GET /health` (status check), `POST /transcribe` (audio file -> JSON `{text, processing_time_ms}`). Stores network timing breakdown (`last_timings` dict) for pipeline logging.

Critical ordering: `setup_cuda_paths()` MUST run before importing `faster_whisper` or `onnx_asr`. It adds NVIDIA pip package DLL directories to PATH and preloads critical DLLs via `ctypes.CDLL`. This is shared across both backends and runs only once.

Backends support a `warmup()` method - called when recording starts to wake the GPU from power-saving before transcription actually begins (recording gives a free time window). Remote backend pings `/health` to wake idle servers.

### Text Processing Pipeline (TextProcessor in app.py)

Six-pass pipeline, order matters:
1. Hallucination loop detection (10+ word repeats)
2. Filler word removal (um, uh, er, etc.)
3. Whisper artifact correction (e.g., `,nd` -> `command`)
4. Character normalization (smart quotes -> ASCII)
5. Custom word replacements (user-configurable via Settings)
6. Spoken punctuation conversion (`period` -> `.`)

### Threading Model

- **Main thread**: Windows message loop + Qt event processing (interleaved in `message_loop()`)
- **Recording**: Background thread fills `self.frames` and calculates RMS audio levels
- **Transcription**: Background thread runs backend, then posts WM_CHAR or copies to clipboard
- **Model loading**: Background thread (UI shows "Loading...")
- **GPU warmup**: Background thread started when recording begins, `wait_for_warmup()` blocks before transcription
- **Audio archive**: `save_async()` runs FLAC save in background thread

### Text Injection (VirtualKeyboard)

Two modes:
- **Type mode** (hotkey): `type_text()` attaches to foreground thread via `AttachThreadInput`, gets focus handle, posts `WM_CHAR` per character. Avoids `SendInput` which overflows terminal buffers.
- **Clipboard mode** (indicator click): `copy_to_clipboard()` uses Win32 clipboard API directly (no pyperclip dependency), with PyQt6 fallback.

Text is sanitized before output: control chars removed, backticks replaced, fancy Unicode normalized to ASCII.

### UI Layer (ui.py)

- **FloatingIndicator** - Bottom-right overlay. Collapses to dot after 3s idle, expands on hover/activity. Custom `paintEvent` draws status dot with glow, audio level bar during recording.
- **SettingsDialog** - Frameless translucent dialog. Dropdowns disabled during model loading (prevents OOM from rapid switching). Uses `NoScrollComboBox` to prevent accidental scroll changes.
- **CognitiveFlowUI** - Coordinator between app and Qt widgets. Uses `pyqtSignal` with `QueuedConnection` for thread-safe UI updates from background threads.

### Data Files

All in `%APPDATA%\CognitiveFlow\` (Linux: `~/.cognitive_flow/`):
- `config.json` - backend_type, model_name, parakeet_model, remote_url, input_device_index, text_replacements, pause_media, etc.
- `statistics.json` - usage stats, performance_history (last 100 entries)
- `history.json` - transcription history with timestamps (500 max)
- `audio/` - FLAC archives of recordings (saved BEFORE transcription so they survive failures)
- `debug_transcriptions.log` - always written (not just --debug mode)

## Key Behaviors to Preserve

- Keyboard hook callback must return quickly - all heavy work happens in background threads
- CUDA DLLs must be loaded before `faster_whisper` import (see `setup_cuda_paths()`)
- Audio is saved to FLAC before transcription starts (crash resilience)
- GPU warmup runs during recording (user's speaking time = free warmup window)
- State reset timers are cancelled before starting new recordings (race condition fix)
- Sleep/wake detection triggers GPU re-warmup after 30s+ message loop gap

## Git Workflow

- Create feature branch for changes
- Merge to main when complete
- Push after merge

## Release Process

**After merging a new version to main, create a git tag for the update checker:**

```cmd
git tag v1.X.X
git push origin v1.X.X
```

The update checker (`UpdateChecker` in app.py) looks for GitHub tags/releases to detect new versions. Without tags, users won't be notified of updates.
