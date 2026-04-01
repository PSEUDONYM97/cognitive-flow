# Cognitive Flow - Complete Application Description

*For design/marketing team reference. Describes all functionality, workflows, and behaviors with zero design-specific language. Current implementation details are omitted -- this is what the app does, not how it looks.*

---

## What It Is

Cognitive Flow is a Windows voice-to-text tool. You press a key, speak, press the key again, and your words appear as typed text in whatever application you're using. It runs as a lightweight background process with a small overlay indicator and a system tray icon.

The app is a single file (~6MB), no installer, no dependencies. Drop it in a folder and run it.

---

## Core Workflow

### The Basic Loop

1. **Press tilde (~)** to start recording
2. **Speak naturally** -- the app captures audio from your default microphone
3. **Press tilde (~) again** to stop recording
4. **Text appears** in whatever window is focused, as if you typed it

That's the entire primary interaction. Everything else supports or extends this loop.

### Recording Modes

**Type Mode (default):** Text is injected directly into the focused application as simulated keystrokes. Works in any text field -- Word, Slack, browser, terminal, IDE.

**Clipboard Mode:** Text is copied to the clipboard instead of typed. Activated by:
- Pressing **Shift+~** to start recording (instead of plain tilde)
- Clicking the overlay indicator to start recording
- Clicking the overlay indicator *during* an active recording (switches mid-recording)

### Canceling a Recording

**Double-tap Escape** (two presses within half a second) cancels the current recording. Audio is discarded, no transcription happens, and any paused media resumes.

### Disabling/Enabling the Hotkey

**Ctrl+~** toggles the hotkey on and off. When disabled, the tilde key works normally for typing tildes. A notification confirms the state change. The toggle state is shown in the tray menu.

---

## The Overlay Indicator

A small circular indicator sits in the corner of the screen. It communicates the app's current state at a glance.

### States

- **Idle:** Subtle, quiet presence. The indicator is there but not demanding attention.
- **Recording:** Visually active and pulsing. The pulse intensity responds to your voice volume -- louder speech creates a more prominent visual effect.
- **Processing:** Distinct from recording, indicates audio has been sent to the server and text is being generated.

### Collapse Behavior

After 3 seconds of inactivity, the indicator shrinks to a minimal dot -- present but unobtrusive. It expands back when:
- You hover over it with the mouse
- A recording starts
- Processing begins

The transitions between collapsed and expanded states are smooth and animated. Expanding is near-instant; collapsing is gradual.

### Interaction

- **Click** the indicator to start/stop clipboard-mode recording
- **Drag** the indicator to reposition it anywhere on screen

### Recording Bar

During recording, a thin bar appears across the top edge of the screen. Its color intensity reflects audio input level -- visual confirmation that the mic is picking up your voice.

### Self-Healing

The indicator automatically recovers from situations that would make it disappear:
- Computer sleep/wake cycles
- Display resolution changes
- Monitor connect/disconnect
- Windows compositor glitches

It periodically re-asserts its visibility, and rebuilds its rendering resources after hardware changes. The user never needs to restart the app because the indicator vanished.

---

## System Tray

Right-clicking the tray icon opens a menu with:

### Toggles
- **Hotkey Enabled** -- enable/disable the tilde recording trigger
- **Pause Media** -- when enabled, automatically pauses any playing audio (Spotify, YouTube, etc.) during recording and resumes it after
- **Show/Hide Indicator** -- toggle the overlay indicator visibility

### Actions
- **Copy Last Output** -- copies the most recent transcription to clipboard
- **Retry Last Recording** -- re-sends the last recorded audio to the server for re-transcription (useful if the server hiccupped)
- **Dashboard** -- opens the web dashboard in your browser
- **Vocabulary** -- opens the web dashboard directly to the word corrections tab
- **Open Recordings** -- opens the folder where audio recordings are saved

### Recent Transcriptions
A submenu showing the last 5 transcriptions. Click any entry to copy it to the clipboard. Entries are truncated for readability in the menu.

### Status Line
A read-only line showing: app version, total transcription count for the session, and uptime.

### Update
When an update is available, a menu item appears to download and install it.

### Quit
Graceful shutdown.

### Tray Icon States
The tray icon changes color to reflect the current state (idle, recording, processing) -- matching the overlay indicator.

---

## Smart Media Pause

When enabled (on by default), the app detects whether audio is actually playing before pausing media. It checks the system audio output -- if nothing is playing, it doesn't send a pause command. This prevents accidentally starting paused music.

Media is automatically resumed when recording stops or is canceled.

---

## Text Processing

Every transcription passes through an automatic cleanup pipeline before the text is output. The user never sees the raw transcription -- they get the cleaned version.

### What Gets Cleaned

**Hallucination detection:** If the speech-to-text model produces a word repeated 10+ times in a row (a known AI artifact), it's collapsed to a single instance.

**Filler word removal:** Natural speech fillers like "um," "uh," "er," "ah," "hmm" are automatically stripped.

**Model artifact correction:** Common speech-to-text mistakes are fixed automatically. For example, certain punctuation-letter combinations that models produce incorrectly are corrected.

**Character normalization:** Smart quotes, em dashes, and other "fancy" Unicode characters are converted to their plain ASCII equivalents for maximum compatibility.

**Spoken punctuation:** You can say punctuation out loud and it converts:
- "period" becomes `.`
- "comma" becomes `,`
- "question mark" becomes `?`
- "new line" becomes an actual line break
- "new paragraph" becomes a double line break
- "open quote" / "close quote" become `"`
- And many more (colon, semicolon, parentheses, brackets, braces, ellipsis, dash, hyphen, apostrophe, exclamation mark)

**Custom word corrections (Vocabulary):** User-defined replacements. If the speech model consistently mishears a word, you add a correction and it's fixed automatically going forward. These are managed through the web dashboard.

**Spacing cleanup:** Removes extra spaces, fixes spacing around punctuation.

---

## Web Dashboard

A local web interface accessible from the tray menu. Runs on your machine only (127.0.0.1) -- not exposed to the network.

### Dashboard Tab
Stat cards showing:
- **Total Transcriptions** -- lifetime count for the session
- **Minutes Recorded** -- total audio duration captured
- **Average Server Time** -- how fast the transcription server responds (in milliseconds)
- **Uptime** -- how long the app has been running
- **Session Count** -- transcriptions in the current session

### History Tab
A searchable list of all transcriptions (up to 500 stored). Each entry shows:
- Timestamp
- Transcribed text
- Processing time

Type to filter/search. Click any entry to copy its text to the clipboard.

### Vocabulary Tab
Manage your custom word corrections:
- See all current corrections in a table (From -> To)
- Add new corrections via two input fields and an Add button
- Delete existing corrections with a delete button per row
- Changes take effect immediately on the next transcription

---

## Audio Storage

Every recording is saved as a WAV file before transcription begins. This means:
- If the server fails, your audio is preserved
- You can retry failed transcriptions
- You have an archive of all recordings
- Audio files are accessible via "Open Recordings" in the tray menu

Files are named with timestamps for easy browsing. There's no automatic cleanup -- the user manages their own archive.

---

## Transcription Server

The app sends audio to a configurable server over HTTP. The server URL is set in the config file. The app is designed to work with any server that accepts audio uploads and returns text.

### Reliability
- Automatic retry with increasing delays (5 seconds, 10 seconds, 15 seconds) if the server doesn't respond
- Server health check on startup with a notification if the server is unreachable
- Pre-warming: while you're speaking, the app pings the server in the background so it's ready to transcribe the moment you stop recording
- The app continues to function even if the server is down -- you'll get error notifications but the app won't crash

### Connection Feedback
- Startup notification if server is offline
- Per-transcription error notifications with the specific failure reason
- "Retry Last Recording" available in tray menu for manual recovery

---

## Self-Update System

### Detection
On startup, the app checks GitHub for new releases. If a newer version exists:
- A notification appears: "Update available: vX.Y.Z"
- An "Update" item appears in the tray menu

### Download & Verification
When the user clicks the update menu item:
1. The new binary is downloaded
2. A SHA256 checksum is verified against the published hash
3. If verification fails, the update is rejected with a notification
4. If successful, the update is staged for the next restart

### Installation
On the next app launch:
1. The staged update replaces the current executable
2. The app restarts automatically
3. The old version is kept as a backup

The user never needs to manually download, extract, or replace files.

---

## Configuration

A single JSON config file stored in the app's data directory. Contains:
- **Server URL** -- where to send audio for transcription
- **Word replacements** -- custom vocabulary corrections (also editable via dashboard)
- **Pause media** -- whether to auto-pause audio during recording

The config is created automatically on first run with sensible defaults. It can be edited directly or through the dashboard's vocabulary tab (for word replacements) and tray menu toggles (for pause media).

---

## Data Storage

All app data lives in a single folder (`%APPDATA%\CognitiveFlow\` on Windows):

| File | Purpose |
|------|---------|
| `config.json` | Server URL, vocabulary, settings |
| `history.json` | Last 500 transcriptions with timestamps and timing data |
| `audio/` folder | WAV recordings, timestamped filenames |
| `debug_transcriptions.log` | Detailed operational log |

---

## Dreamed / Future Functionality

Features that would elevate the experience beyond the current implementation:

### Enhanced Dashboard
- **Performance graphs** -- visualize transcription speed, accuracy trends, and usage patterns over time
- **Audio playback** -- listen to recordings directly from the history tab without opening files
- **Bulk export** -- download transcription history as CSV/text
- **Session grouping** -- group transcriptions by day or work session

### Smarter Text Processing
- **Context-aware capitalization** -- capitalize sentence starts, proper nouns
- **Auto-formatting** -- detect lists, email addresses, URLs in speech and format appropriately
- **Language detection** -- adapt processing pipeline based on detected language
- **Learning corrections** -- automatically suggest vocabulary entries based on repeated manual corrections

### Recording Enhancements
- **Voice activity detection** -- automatically start/stop recording based on speech presence (no hotkey needed)
- **Continuous dictation mode** -- long-form recording with streaming transcription (text appears as you speak, not after)
- **Audio quality indicator** -- visual feedback on mic input quality (too quiet, too noisy, clipping)
- **Multiple mic support** -- select and switch between audio input devices

### Integration
- **Clipboard history integration** -- work with Windows clipboard history
- **Application-specific profiles** -- different vocabulary/settings per target application
- **Hotkey customization** -- let users pick their own trigger key
- **Multi-monitor awareness** -- indicator position per-monitor, follows active display

### Team / Multi-Device
- **Sync vocabulary** across devices
- **Shared vocabulary lists** -- team-managed correction dictionaries
- **Cloud backup** of settings and history

### Accessibility
- **Sound feedback** -- audio cues for recording start/stop/error (for users who can't see the indicator)
- **High contrast mode** -- for indicator visibility in different environments
- **Screen reader announcements** -- announce state changes

---

## Technical Characteristics (For Design Constraints)

These aren't design decisions, but constraints that affect what's possible:

- **Single-file application** -- no installer, no runtime, no dependencies. Users drop a file and run it.
- **~6MB binary** -- lightweight, fast to download and update.
- **Windows-only** -- uses native Windows APIs for audio capture, text injection, and UI overlay.
- **Local-only dashboard** -- the web UI runs on localhost, never exposed to the network.
- **Always-on overlay** -- the indicator window persists across sleep/wake, resolution changes, and compositor glitches without user intervention.
- **Background process** -- no main window. The app lives in the system tray and as an overlay. There is no "window" to alt-tab to.
- **Server-dependent transcription** -- requires a running speech-to-text server on the local network. The app itself does no AI processing.
