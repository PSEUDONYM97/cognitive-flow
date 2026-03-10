# Changelog

## v2.5.1
- Fix indicator vanishing: DWM silently drops layered windows from composition
- Periodic topmost re-assertion every ~10s (SetWindowPos)
- Full compositor recovery every 30s: show + topmost + 1px nudge to force DWM re-register
- Rebuild GDI resources on wake (stale DC from display adapter reset)
- Higher collapsed dot floor (60% opacity, always visible and hittable)
- Faster hover fade-in (~4 frames / 250ms)

## v2.5.0
- Collapsible indicator: fades to subtle dot after 3s idle, expands on hover or activity
- Smooth fade transitions (slow fade out, quick fade in)
- Mouse hover tracking via TrackMouseEvent/WM_MOUSELEAVE

## v2.4.2
- Retry Last Recording in tray menu (re-sends last audio to server)
- Copy Last Output in tray menu

## v2.4.1
- Fix indicator disappearing after sleep: rebuild GDI resources on WM_DISPLAYCHANGE
- Add WS_EX_NOACTIVATE to indicator window (was missing, bar already had it)
- Indicator self-healing heartbeat: recovers from compositor hiding, repositions if off-screen
- Copy Last Output in tray menu (grayed when no transcription yet)
- Self-update system: GitHub release polling, SHA256 verification, tray menu download

## v2.4.0
- Text injection via WM_CHAR PostMessage (matches Python behavior, no more slow SendInput)
- Smart media pause: WASAPI peak meter detects if audio is actually playing
- Tray menu: hotkey toggle, pause media toggle, show/hide indicator
- Config save/persist for pause_media setting

## v2.3.0
- runtime.LockOSThread() - fixes all AppHangB1 crashes (Go scheduler was migrating main goroutine off the UI thread)
- Full text processing pipeline (6-pass): hallucination detection, filler removal, Whisper artifact correction, character normalization, spoken punctuation, spacing cleanup
- History tracking: transcriptions saved to history.json (last 500)
- Precomputed distance table for indicator rendering (zero sqrt per frame)
- Single-pass indicator rendering at 15fps (was dual-pass 30fps)
- Hoisted DLL proc lookups out of wndproc callbacks
- Fixed graceful exit (PostThreadMessage to correct thread)

## v2.1.0
- Per-pixel alpha indicator with software rendering
- Anti-aliased dot with glow and pulsing animation
- UpdateLayeredWindow for true transparency
- Draggable indicator, click for clipboard mode
- Audio-reactive glow radius

## v2.0.1
- Fix audio capture (preserve WHDR_PREPARED flag on buffer recycle)
- Audio saved as WAV before transcription (crash resilient)
- Panic recovery in capture loop and transcription pipeline

## v2.0.0
- Complete ground-up rewrite in Go
- Single static binary (~6MB), zero dependencies
- Remote server backend (HTTP POST to speech-to-text server)
- Win32 waveIn audio capture with event-based callbacks
- Channel-based concurrency (no shared mutable audio state)
- Screen-edge recording bar with audio level colors
- System tray with state-colored icon
- Ctrl+~ toggle, Shift+~ clipboard mode, double-Esc cancel
- Retry with backoff (5s, 10s, 15s)
- Sleep/wake detection with server health ping
