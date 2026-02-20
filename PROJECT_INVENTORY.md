# Cognitive Flow - Project Inventory

**Generated:** 2026-01-11
**Version:** 1.8.9
**Status:** Active, Stable

---

## WHAT

**Cognitive Flow** is a Windows-native voice-to-text application that provides local, GPU-accelerated speech transcription. It functions as a free, privacy-focused alternative to WhisperTyping.

### Core Functionality
- **Global Hotkey Recording**: Press tilde (~) anywhere to start recording, press again to stop
- **Instant Auto-Type**: Transcribed text is typed directly into the currently focused window using Windows WM_CHAR messages
- **Dual Backend Support**:
  - **Whisper** (OpenAI via faster-whisper): Default backend, widely compatible
  - **Parakeet** (NVIDIA via onnx-asr): ~50x faster and more accurate when available
- **GPU Acceleration**: 50-100x faster transcription with NVIDIA CUDA
- **System Tray Integration**: Runs quietly with status indicator and settings access
- **Floating Overlay**: Collapsible visual indicator showing recording/processing state

### Key Features
- 100% local processing - no internet required, complete privacy
- Audio archiving to FLAC for future training data
- Transcription history with 500 entry retention
- Smart text processing: spoken punctuation, Whisper artifact correction, hallucination loop detection
- Statistics tracking: words transcribed, time saved, performance history
- Microphone device selection
- Trailing space toggle for consecutive transcriptions
- Model switching without restart (lazy loading)
- Retry last recording functionality
- Auto-cleanup of corrupted model downloads
- Settings dropdowns disabled during model loading (prevents lockups)

---

## WHY

### Problem Solved
Voice-to-text tools like WhisperTyping require internet connectivity and send audio to external servers. This creates:
1. **Privacy concerns**: Sensitive dictation leaves your machine
2. **Latency**: Round-trip to cloud servers
3. **Availability**: No internet = no transcription
4. **Cost**: Many services charge per usage

### Solution
Cognitive Flow runs entirely locally using OpenAI's Whisper or NVIDIA's Parakeet models. Everything stays on your machine:
- Audio never leaves your computer
- Works offline
- Consistent performance
- Free forever

### Target User
Power users, developers, and professionals who:
- Dictate frequently (notes, code comments, emails)
- Value privacy for sensitive content
- Have NVIDIA GPUs they can leverage
- Want keyboard-free text input anywhere

---

## HOW

### Architecture

```
cognitive_flow/
    __init__.py      # v1.8.9, package exports
    __main__.py      # Entry point for python -m
    app.py           # Main application (~1400 lines)
                     # - CognitiveFlowApp: orchestrates everything
                     # - Windows keyboard hook (WH_KEYBOARD_LL)
                     # - Audio recording via PyAudio
                     # - Threading model for recording/transcription
                     # - VirtualKeyboard: WM_CHAR text injection
                     # - Statistics, TextProcessor, AudioArchive
    backends.py      # Transcription backends abstraction (~430 lines)
                     # - TranscriptionBackend ABC
                     # - WhisperBackend (faster-whisper)
                     # - ParakeetBackend (onnx-asr)
                     # - CUDA path setup for pip-installed NVIDIA libs
                     # - Auto-cleanup for failed downloads
    ui.py            # PyQt6 UI (~1200 lines)
                     # - FloatingIndicator: collapsible overlay
                     # - SettingsDialog: full configuration UI
                     # - NoScrollComboBox: prevents accidental changes
                     # - TranscriptionHistory: JSON persistence
                     # - Loading state management
    logger.py        # ColoredLogger with timing support
    paths.py         # %APPDATA%\CognitiveFlow path constants
    warmup.py        # Windows startup task installer
```

### Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| ASR Engine | faster-whisper 1.0+ | Whisper transcription |
| ASR Engine | onnx-asr 0.10+ | Parakeet transcription |
| GPU Accel | CUDA 12.x, cuDNN 9.x | Hardware acceleration |
| UI Framework | PyQt6 6.5+ | Floating overlay, settings |
| System Tray | pystray 0.19+ | Background indicator |
| Audio | PyAudio 0.2.14+ | Microphone recording |
| Audio Storage | soundfile 0.12+ | FLAC compression |
| Keyboard Hook | ctypes/user32 | Global hotkey capture |
| Text Injection | WM_CHAR posting | Type into any window |

### Data Storage

All in `%APPDATA%\CognitiveFlow\`:
- `config.json` - User preferences (backend, model, device, options)
- `statistics.json` - Usage stats, performance history
- `history.json` - Last 500 transcriptions with timestamps
- `audio/` - FLAC archive of recordings (when enabled)
- `debug_transcriptions.log` - Debug mode logging

### Key Implementation Details

1. **Lazy Loading**: Heavy libraries (faster-whisper, onnxruntime) load after UI shows, giving 50x faster perceived startup

2. **CUDA Path Setup**: Manually adds NVIDIA pip package DLLs to PATH before importing backends, enabling GPU without system CUDA install

3. **Thread Model**:
   - Main: Windows message loop + Qt event processing
   - Recording: Background thread fills audio buffer
   - Transcription: Background thread runs model
   - Audio Archive: Background thread saves FLAC

4. **Text Injection**: Posts WM_CHAR messages directly to focused control, bypassing keyboard queue to avoid terminal buffer overflow

5. **Text Processing Pipeline**:
   - Hallucination loop detection (10+ word repeats)
   - Whisper artifact correction (",nd" -> "command")
   - Character normalization (fancy quotes -> ASCII)
   - Spoken punctuation conversion
   - Spacing cleanup

6. **State Management**:
   - Timer-based state reset with cancellation (prevents race conditions)
   - QueuedConnection for thread-safe UI updates
   - Loading state tracking to prevent concurrent model loads

7. **Error Recovery**:
   - Graceful fallback from Parakeet to Whisper on failure
   - Auto-cleanup of corrupted HuggingFace cache downloads
   - Settings dropdowns disabled during loading

---

## WHEN

### Timeline

| Date | Event |
|------|-------|
| 2025-12-29 | Initial MVP commit |
| 2025-12-29 | PyQt6 UI, color-coded logging, enhanced stats |
| 2026-01-02 | Package restructure (pyproject.toml) |
| 2026-01-05 | Path management, warmup installer |
| 2026-01-10 | v1.8.0-1.8.6: Parakeet backend, CUDA fixes, lazy loading |
| 2026-01-10 | v1.8.7-1.8.8: State race condition fixes, scroll wheel fix |
| 2026-01-11 | v1.8.9: Loading state protection, auto-cleanup |

### Current Status: **Active, Stable**

The project is:
- Functionally complete for its core purpose
- Actively maintained
- At version 1.8.9 with mature feature set
- Has clear versioning discipline

### Recent Work (v1.8.x series)
- v1.8.9: Disable dropdowns during model loading, auto-cleanup failed downloads
- v1.8.8: Disable scroll wheel on settings dropdowns
- v1.8.7: Fix recording state not showing (force immediate color update)
- v1.8.6: Fix state race condition (cancel pending timers)
- v1.8.5: Fix disappearing overlay, reset position menu item
- v1.8.4: Lazy loading for 50x faster startup
- v1.8.3: nvrtc DLL support, suppress onnxruntime warnings
- v1.8.2: Shared CUDA path setup
- v1.8.1: Graceful Parakeet-to-Whisper fallback
- v1.8.0: NVIDIA Parakeet backend via onnx-asr

---

## Quick Reference

```cmd
# Install
pip install -r requirements.txt
pip install -e .

# Run
cognitive-flow              # Background mode (no console)
cognitive-flow --debug      # Foreground with verbose logging

# Or without installing
python -m cognitive_flow
python -m cognitive_flow --debug

# GPU acceleration (Whisper)
pip install nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-cuda-runtime-cu12

# Parakeet backend
pip install onnx-asr[gpu,hub]

# Startup task
python -m cognitive_flow.warmup --install
```

### Controls
- **~ (tilde)**: Start/stop recording
- **Right-click overlay**: Open settings
- **System tray > Quit**: Exit application
- **System tray > Reset Overlay Position**: Fix missing overlay

---

## Assessment

### Strengths
- Well-architected with clean separation (backends, UI, app logic)
- Comprehensive text processing for Whisper edge cases
- GPU acceleration without system CUDA installation
- Privacy-first design
- Good error handling with graceful fallbacks
- Auto-cleanup of corrupted downloads
- Proper versioning discipline

### Areas for Improvement
- Windows-only (Linux/Mac support would require significant work)
- No automated tests
- pyproject.toml version out of sync with __init__.py

### Dependencies Health
All dependencies are standard, maintained packages with good track records.
