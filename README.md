# Cognitive Flow

**Local voice-to-text with GPU acceleration** - A free, privacy-focused alternative to WhisperTyping.

Press `~` (tilde) to record, speak, press `~` again - text is typed directly into any application.

## Features

- **GPU Accelerated** - 4x faster than CPU (0.2s transcription for 10s audio)
- **Global Hotkey** - Works in any application (tilde key)
- **Auto-Type** - Types directly into focused window
- **Privacy First** - 100% local, no internet required
- **Smart Stats** - Tracks time saved vs typing

## Requirements

- Windows 10/11
- Python 3.10+
- NVIDIA GPU (optional but recommended)

## Quick Install

### Option 1: One-Click Install
```cmd
git clone https://github.com/yourusername/cognitive-flow.git
cd cognitive-flow
install.bat
```

### Option 2: Manual Install
```cmd
pip install -r requirements.txt
pip install -e .
```

### Option 3: pip install (coming soon)
```cmd
pip install cognitive-flow
```

## Usage

```cmd
cognitive-flow
```
Or:
```cmd
python cognitive_flow.py
```

**Controls:**
- `~` (tilde) - Start/stop recording
- Right-click indicator - Open settings
- System tray - Quit

## GPU Acceleration

If you have an NVIDIA GPU, install CUDA support for 4x faster transcription:

```cmd
pip install nvidia-cudnn-cu12 nvidia-cublas-cu12
```

The app will automatically use GPU if available, otherwise falls back to CPU.

## Performance

| Hardware | 10s Audio | Speed |
|----------|-----------|-------|
| CPU (medium model) | ~9s | 0.9x realtime |
| GPU (medium model) | ~0.2s | 50x realtime |

## Configuration

Settings are saved in `config.json`:
- Model size (tiny/base/small/medium/large)

Statistics are saved in `statistics.json`:
- Total recordings, words, time saved

## Troubleshooting

**"Model still loading..."**
- Wait for the model to finish loading on first run (~5-10 seconds)

**No GPU acceleration**
- Ensure NVIDIA drivers are installed
- Install: `pip install nvidia-cudnn-cu12 nvidia-cublas-cu12`
- Check with: `nvidia-smi`

**PyAudio install fails on Windows**
- Try: `pip install pipwin && pipwin install pyaudio`

**Tilde key not working**
- Run from a terminal window, not IDE
- Some keyboards use different key codes

## License

MIT License - Free to use, modify, and distribute.

## Credits

Built with:
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) - Whisper inference
- [PyQt6](https://www.riverbankcomputing.com/software/pyqt/) - UI framework
- [ctranslate2](https://github.com/OpenNMT/CTranslate2) - GPU inference
