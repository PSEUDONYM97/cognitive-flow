# Changelog

All notable changes to Cognitive Flow are documented here.

## v1.0.0
- Initial release as proper package

## v1.1.0
- GUI mode by default, --debug for console

## v1.2.0
- Enhanced character sanitization, beam_size=5

## v1.2.1
- Comprehensive timing + debug logging
- Logs to %APPDATA%\CognitiveFlow\debug_transcriptions.log

## v1.3.0
- Microphone input device selector in Settings

## v1.3.3
- WM_CHAR direct posting (bypass keyboard queue)

## v1.4.0
- Show Overlay toggle in system tray menu

## v1.5.0
- Collapsible indicator (shrinks to dot when idle)
- Expands on hover or when recording/processing

## v1.5.1
- Reduced collapse delay to 3s, fix repaint on show

## v1.6.0
- Audio archive: saves recordings as FLAC for training data
- Paired with transcriptions in history.json

## v1.6.1
- Settings UI fixes, live model switching

## v1.6.2
- Initial prompt for better contraction accuracy

## v1.6.3
- Hallucination loop detection (10+ repeats)

## v1.7.0
- Audio saved before transcription (survives failures)
- Retry Last Recording in right-click menu

## v1.7.1
- Fix OOM crash when rapidly switching models in Settings

## v1.8.0
- NVIDIA Parakeet backend via onnx-asr (~50x faster)
- Backend selector in Settings: Whisper vs Parakeet
- TranscriptionBackend abstraction for swappable engines

## v1.8.1
- Graceful fallback: Parakeet failure auto-reverts to Whisper
- Better error handling for missing CUDA libs

## v1.8.2
- Shared CUDA path setup fixes cuDNN loading for both backends

## v1.8.3
- Add nvrtc DLL support, suppress onnxruntime warnings

## v1.8.4
- Lazy backend loading: 50x faster startup (~8s -> 0.16s)

## v1.8.5
- Fix disappearing overlay: ensure_visible(), refresh geometry
- Add 'Reset Overlay Position' to system tray menu

## v1.8.6
- Fix state race condition: cancel pending timers on new recording

## v1.8.7
- Fix recording state not showing: force immediate color update
- Use QueuedConnection for thread-safe UI updates
- Add processEvents() after state change for immediate visibility

## v1.8.8
- Disable scroll wheel on settings dropdowns (prevent accidental changes)

## v1.8.9
- Disable dropdowns during model loading (prevent lockups)
- Auto-cleanup corrupted Parakeet downloads on load failure

## v1.9.0
- GPU warmup on wake: auto-reinitialize after sleep/resume
- Detects system wake and runs silent warmup transcription

## v1.9.1
- Fix overlay not appearing until mouse wiggle
- Force Windows compositor refresh with position nudge

## v1.9.2
- Double-Escape to cancel recording (prevents accidental cancel)
- Remove filler words (um, uh, er, ah, hmm) from transcriptions
- Use EXHAUSTIVE cuDNN algo search for best GPU performance

## v1.10.0
- Custom text replacements in Settings
- Add/remove word corrections via UI (no built-in defaults)

## v1.11.0
- Pause media during recording
- Toggle in Settings to auto-pause Spotify/YouTube/etc while recording

## v1.12.0
- Update checker
- Checks GitHub for new versions on startup, notifies if update available

## v1.13.0
- Fix pause media playing music when already paused
- Detects if audio is playing before toggling (via Windows Audio API)

## v1.13.5
- Fix pycaw per-session audio meter detection
- Checks actual audio output from known media players only
- Ignores games and other non-media apps
