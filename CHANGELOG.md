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

## v1.14.0
- Interactive update prompt in debug mode
- Prominent "UPDATE AVAILABLE" banner with version comparison
- One-key update: press Y to run git pull automatically
- Exits after update so you can restart with new version

## v1.14.1
- Remove WhisperTyping stats import (new installs start fresh at zero)
- Document release tagging process in CLAUDE.md

## v1.14.2
- Use word boundaries for text replacements (prevents partial word matches)
- "claw.md" won't match inside "globalclaw.md" anymore

## v1.15.0
- Audio level indicator while recording
- Visual feedback bar shows microphone input level in real-time
- Color changes: amber (quiet), green (good), red (loud)
- Prevents surprise "no audio" after long recordings

## v1.16.0
- Clipboard mode: click the indicator to record, transcription goes to clipboard
- Hotkey (tilde) still types directly into focused window
- Perfect for admin windows, VTI sessions, or any app that blocks WM_CHAR
- Shows "Copied!" status when clipboard mode completes

## v1.17.0
- Remote server backend: offload transcription to a network STT server
- Send audio over HTTP to any compatible speech-to-text server
- Server URL configurable in Settings with Test Connection button
- No new dependencies (uses stdlib urllib + wave)
- Automatic warmup pings server on recording start
- Graceful fallback to Whisper if server is unreachable

## v1.17.1
- Fix multipart form field name ('file' -> 'audio') for server compatibility
- Fix Test Connection button never showing results (thread-safe signal)
- Add network pipeline timing: encode, payload size, server vs overhead breakdown

## v1.17.2
- Network timing breakdown now appears in pipeline log (not just standalone [Remote] line)
- Pipeline shows net_encode, net_payload_kb, net_server, net_overhead alongside standard steps

## v1.17.3
- Clearer pipeline labels: net_inference (actual transcription), net_latency (network transit)
- Network sub-timings indented under transcribe to show hierarchy
- Pipeline timing added to retry path (was missing entirely)
- Retry stats now record actual processing time instead of 0

## v1.18.0
- Smart server warmup: pre-loads model during recording when server is idle
- Sends throwaway transcription to trigger model load while user speaks
- Adaptive timeout: 30s when server ready, 10s fast-fail when unreachable, 120s cold-start safety net
- UI feedback: shows "Server loading..." during cold start, "Server unreachable..." when down
- Error status in indicator: "Server timeout!", "Server down!" instead of silent reset

## v1.18.1
- Wake warmup uses backend's smart warmup path instead of raw transcribe()
- Remote server wake now checks health + pre-loads model (was bypassing smart logic)

## v1.18.2
- Auto-retry on transient connection reset after successful warmup
- Catches WinError 10054 (Docker networking hiccup after model load)
- Also catches OSError during response read (not just URLError during connect)

## v1.18.3
- Retry loop with backoff for transient connection errors (3 retries: 5s, 10s, 15s)
- Only retries when warmup confirmed server alive, no retries for cold/unknown state
- Shows retry attempt count in logs: "retry 1/3 in 5s..."
