# Cognitive Flow

**Local voice-to-text with GPU acceleration** - A free, privacy-focused alternative to WhisperTyping.

Press `~` (tilde) to record, speak, press `~` again - text is typed directly into any application.

## Features

- **Dual Backend Support** - Whisper (OpenAI) or Parakeet (NVIDIA) - switch in settings
- **GPU Accelerated** - 50x faster than CPU with NVIDIA GPUs
- **Global Hotkey** - Works in any application (tilde key)
- **Auto-Type** - Types directly into focused window via WM_CHAR
- **Privacy First** - 100% local, no internet required
- **Smart Stats** - Tracks time saved vs typing
- **Audio Archive** - Saves recordings as FLAC for future training
- **Collapsible Overlay** - Minimal UI that shrinks to a dot when idle

## Requirements

- Windows 10/11
- Python 3.10+
- NVIDIA GPU (optional but recommended)

## Quick Install

### Option 1: One-Click Install
```cmd
git clone https://github.com/PSEUDONYM97/cognitive-flow.git
cd cognitive-flow
install.bat
```

### Option 2: Manual Install
```cmd
pip install -r requirements.txt
pip install -e .
```

## Usage

```cmd
cognitive-flow
```
Or:
```cmd
python -m cognitive_flow
```

**Controls:**
- `~` (tilde) - Start/stop recording
- Right-click indicator - Open settings
- System tray - Quit, toggle overlay, reset position

## GPU Acceleration

If you have an NVIDIA GPU, install CUDA support for 50x faster transcription:

```cmd
pip install nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12
```

The app will automatically use GPU if available, otherwise falls back to CPU.

## Transcription Backends

### Whisper (Default)
OpenAI's Whisper via faster-whisper. Well-tested, widely compatible.
- Models: tiny, base, small, medium, large
- Recommended: `medium` for best speed/accuracy balance

### Parakeet (Faster)
NVIDIA's Parakeet via onnx-asr. ~50x faster with better accuracy.
- Models: v2 (English), v3 (Multilingual), v3-int8 (Quantized)
- Requires: `pip install onnx-asr[gpu,hub]`

Switch between backends in Settings (right-click the overlay).

## Performance

| Backend | Hardware | 10s Audio | Speed |
|---------|----------|-----------|-------|
| Whisper | CPU (medium) | ~9s | 0.9x realtime |
| Whisper | GPU (medium) | ~0.2s | 50x realtime |
| Parakeet | GPU (v2) | ~0.1s | 100x realtime |

## Configuration

Settings and data are stored in `%APPDATA%\CognitiveFlow\`:
- `config.json` - Backend, model, device, options
- `statistics.json` - Total recordings, words, time saved
- `history.json` - Recent transcriptions (500 max)
- `audio/` - FLAC archive of recordings

## Troubleshooting

**"Model still loading..."**
- Wait for the model to finish loading on first run (~5-10 seconds)
- Settings dropdowns are disabled during loading

**No GPU acceleration**
- Ensure NVIDIA drivers are installed
- Install CUDA packages: `pip install nvidia-cudnn-cu12 nvidia-cublas-cu12`
- Check with: `nvidia-smi`

**Parakeet download failed**
- Corrupted downloads are automatically cleaned up on next attempt
- Delete `~/.cache/huggingface/hub/` to manually clear cache

**PyAudio install fails on Windows**
- Try: `pip install pipwin && pipwin install pyaudio`

**Tilde key not working**
- Run from a terminal window, not IDE
- Some keyboards use different key codes

**Overlay disappeared**
- System tray > "Reset Overlay Position"

## License

MIT License - Free to use, modify, and distribute.

## Credits

Built with:
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) - Whisper inference
- [onnx-asr](https://github.com/NVIDIA/onnx-asr) - Parakeet inference
- [PyQt6](https://www.riverbankcomputing.com/software/pyqt/) - UI framework
- [ctranslate2](https://github.com/OpenNMT/CTranslate2) - GPU inference
