# Changelog

## v2.9.4
- Always show expanded mic circle indicator (removed collapse-to-dot behavior)

## v2.9.3
- Millisecond precision in all log timestamps (was second-level)
- Log typing duration separately (chars + ms)

## v2.9.2
- Resume media immediately when recording stops (don't wait for transcription to finish)
- Tighter HTTP timeouts for LAN: 10s per attempt (was 30s), retry delays 2/4/8s (was 5/10/15s)

## v2.9.1
- Media pause: switch back to SendInput (VK_MEDIA_PLAY_PAUSE) which goes through proper
  keyboard -> shell -> SMTC routing that Chrome responds to. WM_APPCOMMAND didn't reach it.
- Audio detection guard (fixed in v2.8.3) prevents accidental starts from the toggle key

## v2.9.0
- Indicator redesign: 4 distinct visual states matching design mockups
  - Idle collapsed: 16px cyan dot at 40% opacity
  - Idle expanded: 56px dark circle (#1E293B) with 2px cyan border + procedural mic icon
  - Recording: 80px pulsing ring + 64px solid cyan core + dark mic icon
  - Processing: 56px dark circle with 3px cyan border + spinning dots loader
- Smooth animated transitions between collapsed dot and expanded circle
- Anti-aliased rendering with proper src-over compositing
- Mic icon drawn procedurally (capsule + cradle arc + stem + base)
- 96px window buffer to accommodate recording pulse animation

## v2.8.3
- Fix IID_IMMDeviceEnumerator GUID (was wrong since initial implementation, audio detection never worked)
- isAudioPlaying() now correctly detects system audio via WASAPI peak meter

## v2.8.2
- Fix media double-trigger: send to foreground window (shell routing) instead of HWND_BROADCAST
- Fix accidental play on resume: only resume if audio was confirmed playing before pause
- Add detailed COM logging to isAudioPlaying for debugging peak meter failures

## v2.8.1
- Remove toggle key (VK_MEDIA_PLAY_PAUSE) from pause - directional APPCOMMAND only
- Fixes double-trigger where toggle and directional command fought each other

## v2.8.0
- Design rebrand: cyan (#22D3EE) indicator, recording bar, and tray icon replacing green/red
- Dashboard redesign: professional dark theme with design-team CSS, 6 stat cards, p95 timing
- Media pause fix: directional APPCOMMAND_MEDIA_PAUSE/PLAY instead of toggle VK_MEDIA_PLAY_PAUSE
- Media commands broadcast to all windows (fixes YouTube not responding to media keys)
- Resume uses directional play-only command (prevents accidentally pausing other media)

## v2.7.0
- Web dashboard: local HTTP server on 127.0.0.1 with history browser, stats, vocabulary editor
- Dashboard tab: stat cards (total transcriptions, minutes recorded, avg server time, uptime, session count)
- History tab: searchable list of all transcriptions, click any to copy to clipboard
- Vocabulary tab: add/remove word corrections with live config updates
- Tray menu: Dashboard and Vocabulary open browser to local dashboard

## v2.6.0
- Rich tray menu: Recent transcriptions submenu (click to copy), Open Log/Recordings/Config
- Stats line in tray: version, transcription count, uptime
- Recent submenu shows last 5 transcriptions with click-to-copy
- Open Log File, Open Recordings, Edit Config launch with ShellExecute

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
