# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Cognitive Flow is a Windows voice-to-text application using OpenAI's Whisper (via faster-whisper) for local, GPU-accelerated transcription. Press tilde (~) to record, speak, press tilde again - text types into the focused window.

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
pip install nvidia-cudnn-cu12 nvidia-cublas-cu12

# Install startup task (runs at Windows login)
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

```
cognitive_flow/
    __init__.py      # Version + package exports
    __main__.py      # Entry point for python -m
    app.py           # Main app: CognitiveFlowApp, keyboard hook, transcription pipeline
    ui.py            # PyQt6: FloatingIndicator (overlay), SettingsDialog
    logger.py        # ColoredLogger with timing support
    paths.py         # %APPDATA%\CognitiveFlow paths
    warmup.py        # Startup task installer (preloads ctranslate2 DLLs)
```

### Core Flow (app.py)

1. **init_app()** - Deferred import of heavy libraries (faster-whisper, PyAudio, PyQt6) after banner shown. Sets up CUDA DLL paths.
2. **CognitiveFlowApp** - Main controller. Manages:
   - Windows low-level keyboard hook (WH_KEYBOARD_LL) for global tilde key
   - Audio recording via PyAudio
   - Whisper transcription (GPU if available, CPU fallback)
   - Text injection via WM_CHAR posting to focused window
3. **TextProcessor** - Post-processes Whisper output: fixes artifacts (",nd" -> "command"), spoken punctuation, character normalization
4. **VirtualKeyboard.type_text()** - Posts WM_CHAR to bypass keyboard queue, sanitizes dangerous characters

### UI Layer (ui.py)

- **FloatingIndicator** - Bottom-right overlay showing state (idle/recording/processing). Collapses to dot after 3s idle, expands on hover/activity.
- **SettingsDialog** - Model selector, mic input, trailing space toggle, statistics view
- **TranscriptionHistory** - Stores last 500 transcriptions in history.json

### Data Files

All in `%APPDATA%\CognitiveFlow\`:
- `config.json` - model_name, add_trailing_space, input_device_index, show_overlay, archive_audio
- `statistics.json` - usage stats, performance history
- `history.json` - transcription history with timestamps
- `audio/` - FLAC archives of recordings (when archive_audio enabled)
- `debug_transcriptions.log` - detailed log (--debug mode only)

## Key Implementation Details

### Keyboard Hook
Uses Windows `SetWindowsHookExW` with `WH_KEYBOARD_LL`. The hook callback must return quickly - transcription happens in background threads.

### GPU Detection
CUDA libraries loaded manually via `ctypes.CDLL` from site-packages nvidia paths. Must happen BEFORE importing faster-whisper.

### Text Injection
`VirtualKeyboard.type_text()` attaches to foreground thread, gets focus handle, posts `WM_CHAR` for each character. Avoids `SendInput` which can overflow terminal buffers.

### Threading Model
- Main thread: Windows message loop + Qt event processing
- Recording: Background thread fills `self.frames`
- Transcription: Background thread runs Whisper, posts WM_CHAR
- Model loading: Background thread (UI shows "Loading...")
- Audio archive: Background thread saves FLAC asynchronously

## Dependencies

Core: faster-whisper, PyAudio, PyQt6, pystray, pillow, numpy, soundfile
GPU (optional): nvidia-cudnn-cu12, nvidia-cublas-cu12

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

The update checker looks for GitHub tags to detect new versions. Without tags, users won't be notified of updates.
