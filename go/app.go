package main

import (
	"fmt"
	"os"
	"sync"
	"time"
)

// App is the main application orchestrator.
type App struct {
	mu sync.Mutex

	config    *Config
	recorder  *Recorder
	remote    *RemoteClient
	processor *TextProcessor
	indicator *Indicator
	tray      *Tray
	hook      *KeyboardHook
	logger    *Logger

	recording     bool
	clipboardMode bool
	hotkeyEnabled bool
	running       bool
	lastLoopTime  time.Time

	// Double-escape tracking
	lastEscapeTime time.Time
}

// Logger handles file logging.
type Logger struct {
	file *os.File
}

func NewLogger() *Logger {
	dir := AppDataDir()
	os.MkdirAll(dir, 0755)

	f, err := os.OpenFile(LogPath(), os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		fmt.Printf("[Log] Failed to open log file: %v\n", err)
		return &Logger{}
	}
	return &Logger{file: f}
}

func (l *Logger) Log(format string, args ...interface{}) {
	msg := fmt.Sprintf(format, args...)
	timestamp := time.Now().Format("2006-01-02 15:04:05")
	line := fmt.Sprintf("[%s] %s\n", timestamp, msg)

	fmt.Print(line)
	if l.file != nil {
		l.file.WriteString(line)
	}
}

func (l *Logger) Close() {
	if l.file != nil {
		l.file.Close()
	}
}

func NewApp() *App {
	cfg := LoadConfig()

	return &App{
		config:        cfg,
		recorder:      NewRecorder(),
		remote:        NewRemoteClient(cfg.RemoteURL),
		processor:     NewTextProcessor(cfg.TextReplacements),
		indicator:     NewIndicator(),
		tray:          NewTray(),
		logger:        NewLogger(),
		hotkeyEnabled: cfg.HotkeyEnabled,
		running:       true,
	}
}

// Init sets up all components. Must be called from the main thread.
func (app *App) Init() error {
	app.logger.Log("Cognitive Flow v%s starting (Go)", appVersion)
	app.logger.Log("Remote server: %s", app.config.RemoteURL)

	// Create audio archive directory
	if app.config.ArchiveAudio {
		os.MkdirAll(AudioArchiveDir(), 0755)
	}

	// Create indicator overlay
	if app.config.ShowOverlay {
		if err := app.indicator.Create(); err != nil {
			app.logger.Log("Indicator create failed: %v", err)
		}
		app.indicator.onClick = app.onIndicatorClick
	}

	// Create system tray
	if err := app.tray.Create(); err != nil {
		return fmt.Errorf("tray create: %w", err)
	}
	app.tray.hotkeyEnabled = app.hotkeyEnabled
	app.tray.overlayVisible = app.config.ShowOverlay
	app.tray.onSettings = app.onSettings
	app.tray.onToggleHotkey = app.onToggleHotkey
	app.tray.onToggleOverlay = app.onToggleOverlay
	app.tray.onResetOverlay = app.onResetOverlay
	app.tray.onQuit = app.onQuit

	// Set up audio level callback
	app.recorder.onLevel = func(level float64) {
		if app.indicator != nil {
			app.indicator.SetAudioLevel(level)
		}
	}

	// Install keyboard hook
	hook, err := InstallKeyboardHook(app.onKeyboard)
	if err != nil {
		return fmt.Errorf("keyboard hook: %w", err)
	}
	app.hook = hook

	// Warm up server in background
	go func() {
		app.logger.Log("Warming up server...")
		if err := app.remote.Warmup(); err != nil {
			app.logger.Log("Server warmup failed: %v", err)
			app.setStateFromThread(StateIdle, "Server offline")
		} else {
			app.logger.Log("Server ready: %s (GPU: %v)", app.remote.ServerModel, app.remote.ServerGPU)
			app.setStateFromThread(StateIdle, "Ready")
		}
	}()

	return nil
}

// Run executes the Windows message loop. Blocks until quit.
func (app *App) Run() {
	app.lastLoopTime = time.Now()

	var msg MSG
	for app.running {
		for PeekMessage(&msg, 0, 0, 0, PM_REMOVE) {
			TranslateMessage(&msg)
			DispatchMessage(&msg)
		}

		// Sleep/wake detection: 30s+ gap means system was sleeping
		now := time.Now()
		gap := now.Sub(app.lastLoopTime)
		if gap > 30*time.Second {
			app.logger.Log("Wake detected (gap: %v), warming up server", gap)
			go func() {
				if err := app.remote.Warmup(); err != nil {
					app.logger.Log("Wake warmup failed: %v", err)
				}
			}()
		}
		app.lastLoopTime = now

		// Yield CPU
		time.Sleep(1 * time.Millisecond)
	}
}

// Shutdown cleans up all resources.
func (app *App) Shutdown() {
	app.logger.Log("Shutting down")

	if app.hook != nil {
		app.hook.Uninstall()
	}
	if app.indicator != nil {
		app.indicator.Destroy()
	}
	if app.tray != nil {
		app.tray.Destroy()
	}
	app.logger.Close()
}

// --- Keyboard handler ---

func (app *App) onKeyboard(vkCode uint32, down bool) int {
	switch vkCode {
	case VK_OEM_3: // Tilde
		if !down {
			return 0 // Only handle key-down
		}

		// Check for Ctrl+~ toggle
		ctrlDown := GetAsyncKeyState(VK_LCONTROL)&(-32768) != 0 ||
			GetAsyncKeyState(VK_RCONTROL)&(-32768) != 0

		if ctrlDown {
			app.hotkeyEnabled = !app.hotkeyEnabled
			state := "enabled"
			if !app.hotkeyEnabled {
				state = "disabled"
			}
			app.logger.Log("Hotkey %s", state)
			app.indicator.SetState(StateIdle)
			app.tray.hotkeyEnabled = app.hotkeyEnabled
			return 1 // Block the key
		}

		if !app.hotkeyEnabled {
			return 0 // Pass through
		}

		app.toggleRecording(false) // false = type mode (not clipboard)
		return 1 // Block the key

	case VK_ESCAPE:
		if !down {
			return 0
		}

		// Double-escape to cancel recording
		now := time.Now()
		if app.recording && now.Sub(app.lastEscapeTime) < 500*time.Millisecond {
			app.cancelRecording()
			return 1
		}
		app.lastEscapeTime = now
		return 0 // Pass through
	}

	return 0 // Pass through
}

// --- Recording flow ---

func (app *App) toggleRecording(clipboardMode bool) {
	app.mu.Lock()
	defer app.mu.Unlock()

	if app.recording {
		if clipboardMode {
			app.clipboardMode = true
		}
		app.stopRecording()
	} else {
		app.clipboardMode = clipboardMode
		app.startRecording()
	}
}

func (app *App) startRecording() {
	app.logger.Log("Recording started (clipboard: %v)", app.clipboardMode)

	app.recording = true
	app.indicator.SetState(StateRecording)
	app.tray.SetIcon(StateRecording)

	// Start server warmup in parallel
	go app.remote.Warmup()

	// Start audio recording
	if err := app.recorder.Start(app.config.InputDeviceIndex); err != nil {
		app.logger.Log("Recording start failed: %v", err)
		app.recording = false
		app.indicator.SetState(StateIdle)
		app.tray.SetIcon(StateIdle)
		return
	}
}

func (app *App) stopRecording() {
	if !app.recording {
		return
	}

	app.logger.Log("Recording stopped")
	app.recording = false

	app.indicator.SetState(StateProcessing)
	app.tray.SetIcon(StateProcessing)

	// Stop audio and get samples
	samples, err := app.recorder.Stop()
	if err != nil {
		app.logger.Log("Recording stop failed: %v", err)
		app.indicator.SetState(StateIdle)
		app.tray.SetIcon(StateIdle)
		return
	}

	if len(samples) == 0 {
		app.logger.Log("No audio captured")
		app.indicator.SetState(StateIdle)
		app.tray.SetIcon(StateIdle)
		return
	}

	clipMode := app.clipboardMode
	app.logger.Log("Audio: %d samples (%.1fs)", len(samples), float64(len(samples))/float64(sampleRate))

	// Transcribe in background
	go app.transcribe(samples, clipMode)
}

func (app *App) cancelRecording() {
	app.mu.Lock()
	defer app.mu.Unlock()

	if !app.recording {
		return
	}

	app.logger.Log("Recording cancelled")
	app.recording = false
	app.recorder.Stop()
	app.indicator.SetState(StateIdle)
	app.tray.SetIcon(StateIdle)
}

func (app *App) transcribe(samples []int16, clipboardMode bool) {
	start := time.Now()

	result, err := app.remote.Transcribe(samples, sampleRate)
	if err != nil {
		app.logger.Log("Transcription failed: %v", err)
		app.setStateFromThread(StateIdle, "Error!")
		return
	}

	text := result.Text
	if text == "" {
		app.logger.Log("Empty transcription")
		app.setStateFromThread(StateIdle, "Ready")
		return
	}

	// Apply text processing pipeline
	text = app.processor.Process(text)

	if app.config.AddTrailingSpace {
		text += " "
	}

	elapsed := time.Since(start)
	timings := app.remote.LastTimings
	app.logger.Log("Pipeline: total=%.0fms | encode=%.0fms | payload=%.1fKB | server=%.0fms | overhead=%.0fms",
		float64(elapsed.Milliseconds()), timings.EncodeMs, timings.PayloadKB,
		timings.ServerMs, timings.OverheadMs)
	app.logger.Log("Text: %s", text)

	// Inject text
	if clipboardMode {
		if err := CopyToClipboard(text); err != nil {
			app.logger.Log("Clipboard failed: %v", err)
		} else {
			app.logger.Log("Copied to clipboard")
			app.setStateFromThread(StateIdle, "Copied!")
		}
	} else {
		if err := TypeText(text); err != nil {
			app.logger.Log("Type failed: %v, falling back to clipboard", err)
			CopyToClipboard(text)
			app.setStateFromThread(StateIdle, "Copied!")
		} else {
			app.setStateFromThread(StateIdle, "Ready")
		}
	}
}

// setStateFromThread safely updates indicator/tray from a background goroutine.
// On Windows, PostMessage is thread-safe, so we post to the indicator window.
func (app *App) setStateFromThread(state, _ string) {
	if app.indicator != nil {
		app.indicator.SetState(state)
	}
	app.tray.SetIcon(state)
}

// --- Indicator click handler ---

func (app *App) onIndicatorClick() {
	app.toggleRecording(true) // true = clipboard mode
}

// --- Tray menu handlers ---

func (app *App) onSettings() {
	// TODO: Settings dialog
	app.logger.Log("Settings requested (not yet implemented)")
}

func (app *App) onToggleHotkey() {
	app.hotkeyEnabled = !app.hotkeyEnabled
	app.tray.hotkeyEnabled = app.hotkeyEnabled
	state := "enabled"
	if !app.hotkeyEnabled {
		state = "disabled"
	}
	app.logger.Log("Hotkey %s", state)
}

func (app *App) onToggleOverlay() {
	app.config.ShowOverlay = !app.config.ShowOverlay
	app.tray.overlayVisible = app.config.ShowOverlay
	if app.config.ShowOverlay {
		ShowWindow(app.indicator.hwnd, SW_SHOWNA)
		app.indicator.visible = true
	} else {
		ShowWindow(app.indicator.hwnd, SW_HIDE)
		app.indicator.visible = false
	}
	app.config.Save()
}

func (app *App) onResetOverlay() {
	screenW := int32(GetSystemMetrics(SM_CXSCREEN))
	screenH := int32(GetSystemMetrics(SM_CYSCREEN))
	x := screenW - int32(windowWidth) - int32(indicatorPadding)
	y := screenH - int32(windowHeight) - int32(indicatorPadding) - 40
	MoveWindow(app.indicator.hwnd, x, y, int32(windowWidth), int32(windowHeight), true)
}

func (app *App) onQuit() {
	app.running = false
	PostQuitMessage(0)
}
