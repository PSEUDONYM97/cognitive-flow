// Cognitive Flow v2 - Voice to text. Press tilde, speak, press tilde. Text appears.
//
// Design principles:
//   - Invisible when idle, unmissable when active
//   - Fast path: tilde -> speak -> tilde -> text typed. Nothing else.
//   - Errors are visible (tray notifications), never silent
//   - One config file, no settings UI
//   - Channel-based concurrency, no shared mutable audio state
package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
	"unicode"
	"unicode/utf16"
	"unsafe"

	"golang.org/x/sys/windows"
)

// Pin the main goroutine to the OS thread BEFORE main() runs.
// Without this, Go's scheduler can move the main goroutine between OS threads,
// which breaks Win32 window message dispatching and causes AppHangB1 kills.
func init() {
	runtime.LockOSThread()
}

// ----- Configuration -----

type config struct {
	Server       string            `json:"server"`
	Replacements map[string]string `json:"replacements"`
	PauseMedia   bool              `json:"pause_media"`
}

var cfg = config{
	Server:       "http://192.168.0.10:9200",
	Replacements: map[string]string{},
	PauseMedia:   true, // default on, matches Python behavior
}

func loadConfig() {
	dir := configDir()
	os.MkdirAll(dir, 0755)

	path := filepath.Join(dir, "config.json")
	data, err := os.ReadFile(path)
	if err != nil {
		// Write default config so user can edit it
		def, _ := json.MarshalIndent(cfg, "", "  ")
		os.WriteFile(path, def, 0644)
		return
	}
	json.Unmarshal(data, &cfg)
	if cfg.Replacements == nil {
		cfg.Replacements = map[string]string{}
	}
}

func saveConfig() {
	path := filepath.Join(configDir(), "config.json")
	data, _ := json.MarshalIndent(cfg, "", "  ")
	os.WriteFile(path, data, 0644)
}

func configDir() string {
	if d := os.Getenv("APPDATA"); d != "" {
		return filepath.Join(d, "CognitiveFlow")
	}
	return filepath.Join(os.Getenv("HOME"), ".cognitive_flow")
}

// ----- Win32 -----

var (
	user32   = windows.NewLazySystemDLL("user32.dll")
	kernel32 = windows.NewLazySystemDLL("kernel32.dll")
	winmm    = windows.NewLazySystemDLL("winmm.dll")
	gdi32    = windows.NewLazySystemDLL("gdi32.dll")
	shell32  = windows.NewLazySystemDLL("shell32.dll")

	pSetWindowsHookEx    = user32.NewProc("SetWindowsHookExW")
	pUnhookWindowsHookEx = user32.NewProc("UnhookWindowsHookEx")
	pCallNextHookEx      = user32.NewProc("CallNextHookEx")
	pGetMessage          = user32.NewProc("GetMessageW")
	pTranslateMessage    = user32.NewProc("TranslateMessage")
	pDispatchMessage     = user32.NewProc("DispatchMessageW")
	pPostQuitMessage     = user32.NewProc("PostQuitMessage")
	pGetAsyncKeyState    = user32.NewProc("GetAsyncKeyState")
	pRegisterClassEx     = user32.NewProc("RegisterClassExW")
	pCreateWindowEx      = user32.NewProc("CreateWindowExW")
	pDestroyWindow       = user32.NewProc("DestroyWindow")
	pShowWindow          = user32.NewProc("ShowWindow")
	pSetWindowPos        = user32.NewProc("SetWindowPos")
	pMoveWindow          = user32.NewProc("MoveWindow")
	pDefWindowProc       = user32.NewProc("DefWindowProcW")
	pLoadCursor          = user32.NewProc("LoadCursorW")
	pGetSystemMetrics    = user32.NewProc("GetSystemMetrics")
	pGetCursorPos        = user32.NewProc("GetCursorPos")
	pSetLayeredWindowAttr = user32.NewProc("SetLayeredWindowAttributes")
	pInvalidateRect      = user32.NewProc("InvalidateRect")
	pBeginPaint          = user32.NewProc("BeginPaint")
	pEndPaint            = user32.NewProc("EndPaint")
	pFillRect            = user32.NewProc("FillRect")
	pSetTimer            = user32.NewProc("SetTimer")
	pSendInput           = user32.NewProc("SendInput")
	pOpenClipboard       = user32.NewProc("OpenClipboard")
	pCloseClipboard      = user32.NewProc("CloseClipboard")
	pEmptyClipboard      = user32.NewProc("EmptyClipboard")
	pSetClipboardData    = user32.NewProc("SetClipboardData")
	pGetDC               = user32.NewProc("GetDC")
	pReleaseDC           = user32.NewProc("ReleaseDC")
	pSetForegroundWindow = user32.NewProc("SetForegroundWindow")
	pPostMessage         = user32.NewProc("PostMessageW")
	pCreatePopupMenu     = user32.NewProc("CreatePopupMenu")
	pAppendMenu          = user32.NewProc("AppendMenuW")
	pTrackPopupMenu      = user32.NewProc("TrackPopupMenu")
	pDestroyMenu         = user32.NewProc("DestroyMenu")
	pCreateIconIndirect  = user32.NewProc("CreateIconIndirect")
	pDestroyIcon         = user32.NewProc("DestroyIcon")

	pPostThreadMessage     = user32.NewProc("PostThreadMessageW")
	pGetCurrentThreadId    = kernel32.NewProc("GetCurrentThreadId")
	pGlobalAlloc           = kernel32.NewProc("GlobalAlloc")
	pGlobalLock            = kernel32.NewProc("GlobalLock")
	pGlobalUnlock          = kernel32.NewProc("GlobalUnlock")
	pCreateEvent           = kernel32.NewProc("CreateEventW")
	pSetEvent              = kernel32.NewProc("SetEvent")
	pWaitForSingleObject   = kernel32.NewProc("WaitForSingleObject")
	pSetConsoleCtrlHandler = kernel32.NewProc("SetConsoleCtrlHandler")

	pWaveInOpen            = winmm.NewProc("waveInOpen")
	pWaveInClose           = winmm.NewProc("waveInClose")
	pWaveInPrepareHeader   = winmm.NewProc("waveInPrepareHeader")
	pWaveInUnprepareHeader = winmm.NewProc("waveInUnprepareHeader")
	pWaveInAddBuffer       = winmm.NewProc("waveInAddBuffer")
	pWaveInStart           = winmm.NewProc("waveInStart")
	pWaveInStop            = winmm.NewProc("waveInStop")
	pWaveInReset           = winmm.NewProc("waveInReset")

	pCreateSolidBrush    = gdi32.NewProc("CreateSolidBrush")
	pDeleteObject        = gdi32.NewProc("DeleteObject")
	pCreateDIBSection    = gdi32.NewProc("CreateDIBSection")
	pCreateCompatibleDC  = gdi32.NewProc("CreateCompatibleDC")
	pSelectObject        = gdi32.NewProc("SelectObject")
	pDeleteDC            = gdi32.NewProc("DeleteDC")
	pUpdateLayeredWindow = user32.NewProc("UpdateLayeredWindow")

	pShellNotifyIcon = shell32.NewProc("Shell_NotifyIconW")
	pSetCapture      = user32.NewProc("SetCapture")
	pReleaseCapture  = user32.NewProc("ReleaseCapture")
	pGetWindowRect   = user32.NewProc("GetWindowRect")
)

// ----- Constants -----

const (
	version = "2.4.0"

	whKeyboardLL = 13
	wmKeydown    = 0x0100
	wmPaint      = 0x000F
	wmDestroy    = 0x0002
	wmTimer      = 0x0113
	wmCommand    = 0x0111
	wmApp        = 0x8000
	wmTrayIcon   = wmApp + 1
	wmSetPhase   = wmApp + 2 // custom: wp=phase, triggers UI update on main thread
	wmRButtonUp  = 0x0205
	wmLButtonUp  = 0x0202
	wmNcHitTest  = 0x0084
	htCaption    = 2

	vkTilde    = 0xC0
	vkEscape   = 0x1B
	vkLCtrl    = 0xA2
	vkRCtrl    = 0xA3
	vkLShift   = 0xA0
	vkRShift   = 0xA1
	vkReturn   = 0x0D
	vkControl  = 0x11
	vkV        = 0x56

	wsPopup        = 0x80000000
	wsExLayered    = 0x00080000
	wsExTopmost    = 0x00000008
	wsExToolWindow = 0x00000080
	wsExNoActivate = 0x08000000
	wsExTransparent = 0x00000020

	lwaColorKey = 0x01
	lwaAlpha    = 0x02

	inputKeyboard     = 1
	keyeventfUnicode  = 0x0004
	keyeventfKeyup    = 0x0002
	keyeventfExtended = 0x0001

	nimAdd    = 0
	nimModify = 1
	nimDelete = 2
	nifMsg    = 1
	nifIcon   = 2
	nifTip    = 4
	nifInfo   = 16
	nifShowTip = 0x80
	niiInfo   = 0x01

	mfString    = 0
	mfSeparator = 0x800
	mfChecked   = 0x08

	cfUnicode    = 13
	gmemMoveable = 0x0002

	callbackEvent = 0x00050000
	waveMapper    = 0xFFFFFFFF
	wavePCM       = 1
	whdrDone      = 1

	sampleRate = 16000
	chunkSize  = 1024
	numBufs    = 4

	timerHeartbeat = 1
)

// ----- Win32 types -----

type wmsg struct {
	Hwnd    uintptr
	Message uint32
	WParam  uintptr
	LParam  uintptr
	Time    uint32
	Pt      [2]int32
}

type kbhook struct {
	VkCode, ScanCode, Flags, Time uint32
	Extra                         uintptr
}

type wndclass struct {
	Size                         uint32
	Style                        uint32
	WndProc                      uintptr
	ClsExtra, WndExtra           int32
	Instance, Icon, Cursor, Bg   uintptr
	MenuName, ClassName          *uint16
	IconSm                       uintptr
}

type paintstruct struct {
	Hdc     uintptr
	Erase   int32
	Paint   [4]int32
	_       [44]byte
}

type waveformat struct {
	Tag                  uint16
	Ch                   uint16
	Rate                 uint32
	ByteRate             uint32
	BlockAlign, BitDepth uint16
	Extra                uint16
}

type wavehdr struct {
	Data     uintptr
	Len      uint32
	Recorded uint32
	User     uintptr
	Flags    uint32
	Loops    uint32
	Next     uintptr
	Reserved uintptr
}

type notifyicon struct {
	Size        uint32
	Hwnd        uintptr
	ID          uint32
	Flags       uint32
	CallbackMsg uint32
	Icon        uintptr
	Tip         [128]uint16
	State       uint32
	StateMask   uint32
	Info        [256]uint16
	Version     uint32
	InfoTitle   [64]uint16
	InfoFlags   uint32
	GUID        [16]byte
	BalloonIcon uintptr
}

type bmpinfo struct {
	Size                       uint32
	Width, Height              int32
	Planes, Bits               uint16
	Compress, ImgSize          uint32
	XPel, YPel, ClrUsed, ClrImp uint32
}

type iconinfo struct {
	IsIcon       int32
	XHot, YHot   uint32
	Mask, Color  uintptr
}

// ----- Phase (atomic, safe to read from paint callback while written from goroutines) -----

const (
	phaseIdle       int32 = 0
	phaseRecording  int32 = 1
	phaseProcessing int32 = 2
)

var phase atomic.Int32
var audioLevel atomic.Int32 // RMS level 0-100, updated by capture loop
var mainThreadID uintptr     // for PostThreadMessage from other threads

func currentPhase() int32  { return phase.Load() }
func setPhaseVal(p int32)  { phase.Store(p) }

// ----- App state -----

var state struct {
	mu            sync.Mutex
	recording     bool
	clipboardMode bool
	enabled       bool
	running       bool
	indVisible    bool
	lastEsc       time.Time
	lastWake      time.Time

	hook     uintptr
	bar      uintptr // screen-edge recording bar
	dot      uintptr // clickable indicator dot
	trayHwnd uintptr
	trayNID  notifyicon
	trayIcon uintptr
}

// ----- Entry point -----

func main() {
	// Minimal flag handling
	for _, a := range os.Args[1:] {
		switch a {
		case "--version", "-v":
			fmt.Printf("Cognitive Flow v%s (Go)\n", version)
			os.Exit(0)
		}
	}

	initLog()
	loadConfig()

	state.enabled = true
	state.running = true
	setPhaseVal(phaseIdle)
	state.lastWake = time.Now()

	// Save main thread ID so other threads can post WM_QUIT to us
	mainThreadID, _, _ = pGetCurrentThreadId.Call()

	log("Cognitive Flow v%s", version)
	log("Server: %s", cfg.Server)

	// Clean shutdown on console close (Ctrl+C, window close, etc.)
	// Console handler runs on a DIFFERENT thread - PostQuitMessage would
	// post to that thread's queue. Use PostThreadMessage to reach main.
	pSetConsoleCtrlHandler.Call(syscall.NewCallback(func(sig uintptr) uintptr {
		shutdown()
		pPostThreadMessage.Call(mainThreadID, 0x0012, 0, 0) // WM_QUIT=0x0012
		return 1
	}), 1)

	// Check server
	go func() {
		if err := healthCheck(); err != nil {
			log("Server offline: %v", err)
			notify("Server offline", fmt.Sprintf("Cannot reach %s", cfg.Server))
		} else {
			log("Server ready")
		}
	}()

	// Check for updates
	go checkForUpdate()

	// Apply pending update from last run (if cogflow-update.exe exists)
	applyPendingUpdate()

	// Create UI
	createBar()
	createIndicator()
	createTray()

	// Keyboard hook
	h, _, err := pSetWindowsHookEx.Call(whKeyboardLL, syscall.NewCallback(hookProc), 0, 0)
	if h == 0 {
		fatal("Keyboard hook failed: %v", err)
	}
	state.hook = h

	log("Ready. Press ~ to record, Shift+~ for clipboard, Ctrl+~ to toggle.")

	// Message loop (GetMessage blocks until a message arrives - efficient, no polling)
	var m wmsg
	for {
		r, _, _ := pGetMessage.Call(uintptr(unsafe.Pointer(&m)), 0, 0, 0)
		if r == 0 || int32(r) == -1 {
			break
		}
		pTranslateMessage.Call(uintptr(unsafe.Pointer(&m)))
		pDispatchMessage.Call(uintptr(unsafe.Pointer(&m)))
	}

	shutdown()
}

func shutdown() {
	if !state.running {
		return
	}
	state.running = false
	log("Shutting down")
	if state.hook != 0 {
		pUnhookWindowsHookEx.Call(state.hook)
	}
	pShellNotifyIcon.Call(nimDelete, uintptr(unsafe.Pointer(&state.trayNID)))
	if state.trayIcon != 0 {
		pDestroyIcon.Call(state.trayIcon)
	}
}

// ----- Keyboard hook -----

func hookProc(nCode int, wParam, lParam uintptr) uintptr {
	if nCode >= 0 && wParam == wmKeydown {
		kb := (*kbhook)(unsafe.Pointer(lParam))

		switch kb.VkCode {
		case vkTilde:
			ctrl := keyDown(vkLCtrl) || keyDown(vkRCtrl)
			shift := keyDown(vkLShift) || keyDown(vkRShift)

			if ctrl {
				state.enabled = !state.enabled
				if state.enabled {
					log("Hotkey enabled")
					notify("Hotkey enabled", "Press ~ to record")
				} else {
					log("Hotkey disabled - tilde passes through")
					notify("Hotkey disabled", "Press Ctrl+~ to re-enable")
				}
				return 1
			}

			if !state.enabled {
				return passthrough(nCode, wParam, lParam)
			}

			go toggle(shift)
			return 1

		case vkEscape:
			if state.recording {
				now := time.Now()
				if now.Sub(state.lastEsc) < 500*time.Millisecond {
					go cancel()
					return 1
				}
				state.lastEsc = now
			}
		}
	}

	return passthrough(nCode, wParam, lParam)
}

func passthrough(nCode int, wParam, lParam uintptr) uintptr {
	r, _, _ := pCallNextHookEx.Call(0, uintptr(nCode), wParam, lParam)
	return r
}

func keyDown(vk uintptr) bool {
	r, _, _ := pGetAsyncKeyState.Call(vk)
	return int16(r)&(-32768) != 0
}

// ----- Media control -----
// Pauses media playback during recording so mic doesn't pick up audio.
// Checks if audio is actually playing via WASAPI peak meter before pausing.
// Only resumes if WE caused the pause.

const vkMediaPlayPause = 0xB3

var (
	mediaPaused bool // true if WE sent a pause
	ole32       = windows.NewLazySystemDLL("ole32.dll")
	pCoInit     = ole32.NewProc("CoInitializeEx")
	pCoCreateInst = ole32.NewProc("CoCreateInstance")
)

// WASAPI COM GUIDs
var (
	clsidMMDevEnum = [16]byte{0x95, 0x03, 0xDE, 0xBC, 0x2F, 0xE5, 0x7C, 0x46, 0x8E, 0x3D, 0xC4, 0x57, 0x92, 0x91, 0x69, 0x2E}
	iidIMMDevEnum  = [16]byte{0x79, 0xFB, 0x1D, 0xA4, 0x05, 0x47, 0xDA, 0x44, 0x95, 0x8A, 0x61, 0x2F, 0x72, 0x46, 0xBE, 0x85}
	iidAudioMeter  = [16]byte{0xF6, 0x16, 0x22, 0xC0, 0x67, 0x8C, 0x5B, 0x4B, 0x9D, 0x00, 0xD0, 0x08, 0xE7, 0x3E, 0x00, 0x64}
)

// isAudioPlaying checks if any audio is being output via the default render device.
// Uses IAudioMeterInformation::GetPeakValue - if peak > 0, something is playing.
func isAudioPlaying() bool {
	// Initialize COM on this goroutine
	pCoInit.Call(0, 0) // COINIT_MULTITHREADED

	// Create MMDeviceEnumerator
	var enumPtr uintptr
	hr, _, _ := pCoCreateInst.Call(
		uintptr(unsafe.Pointer(&clsidMMDevEnum)),
		0,
		1|4, // CLSCTX_INPROC_SERVER | CLSCTX_LOCAL_SERVER
		uintptr(unsafe.Pointer(&iidIMMDevEnum)),
		uintptr(unsafe.Pointer(&enumPtr)),
	)
	if hr != 0 || enumPtr == 0 {
		return false // can't check, assume not playing
	}
	defer comRelease(enumPtr)

	// GetDefaultAudioEndpoint(eRender=0, eConsole=0, &device)
	vtbl := *(*[8]uintptr)(unsafe.Pointer(*(*uintptr)(unsafe.Pointer(enumPtr))))
	var devicePtr uintptr
	hr, _, _ = syscall.SyscallN(vtbl[4], enumPtr, 0, 0, uintptr(unsafe.Pointer(&devicePtr)))
	if hr != 0 || devicePtr == 0 {
		return false
	}
	defer comRelease(devicePtr)

	// Activate IAudioMeterInformation
	vtbl2 := *(*[6]uintptr)(unsafe.Pointer(*(*uintptr)(unsafe.Pointer(devicePtr))))
	var meterPtr uintptr
	hr, _, _ = syscall.SyscallN(vtbl2[3], devicePtr,
		uintptr(unsafe.Pointer(&iidAudioMeter)),
		7, // CLSCTX_ALL
		0,
		uintptr(unsafe.Pointer(&meterPtr)),
	)
	if hr != 0 || meterPtr == 0 {
		return false
	}
	defer comRelease(meterPtr)

	// GetPeakValue(&peak)
	vtbl3 := *(*[4]uintptr)(unsafe.Pointer(*(*uintptr)(unsafe.Pointer(meterPtr))))
	var peak float32
	hr, _, _ = syscall.SyscallN(vtbl3[3], meterPtr, uintptr(unsafe.Pointer(&peak)))
	if hr != 0 {
		return false
	}

	return peak > 0.001
}

func comRelease(ptr uintptr) {
	vtbl := *(*[3]uintptr)(unsafe.Pointer(*(*uintptr)(unsafe.Pointer(ptr))))
	syscall.SyscallN(vtbl[2], ptr) // IUnknown::Release
}

func pauseMedia() {
	if !cfg.PauseMedia {
		return
	}
	if !isAudioPlaying() {
		log("No audio playing, skipping pause")
		return
	}
	sendKey(vkMediaPlayPause, keyeventfExtended)
	mediaPaused = true
	log("Media paused")
}

func resumeMedia() {
	if !mediaPaused {
		return
	}
	mediaPaused = false
	sendKey(vkMediaPlayPause, keyeventfExtended)
	log("Media resumed")
}

// ----- Recording flow -----

func toggle(clipboard bool) {
	state.mu.Lock()
	defer state.mu.Unlock()

	if state.recording {
		if clipboard {
			state.clipboardMode = true
		}
		stopRecording()
	} else {
		state.clipboardMode = clipboard
		startRecording()
	}
}

func cancel() {
	state.mu.Lock()
	defer state.mu.Unlock()
	if !state.recording {
		return
	}
	log("Cancelled")
	state.recording = false
	if audio.stopCh != nil {
		close(audio.stopCh)
		audio.stopCh = nil
	}
	resumeMedia()
	setPhase(phaseIdle)
}

// ----- Audio device state (owned by startRecording/stopRecording, used by captureLoop) -----

var audio struct {
	hwi     uintptr
	event   uintptr
	bufs    [numBufs][]byte
	hdrs    [numBufs]wavehdr
	stopCh  chan struct{}
	frameCh chan []int16
}

func startRecording() {
	log("Recording (clipboard: %v)", state.clipboardMode)
	state.recording = true
	setPhase(phaseRecording)

	// Pause media so mic doesn't pick up playback
	pauseMedia()

	// Warm server while user speaks
	go healthCheck()

	// Open audio device NOW, not in a goroutine. This is the critical path -
	// every millisecond here is a millisecond of the user's first word lost.
	wfx := waveformat{
		Tag: wavePCM, Ch: 1, Rate: sampleRate,
		BitDepth: 16, BlockAlign: 2, ByteRate: sampleRate * 2,
	}

	ev, _, _ := pCreateEvent.Call(0, 0, 0, 0)
	audio.event = ev

	ret, _, _ := pWaveInOpen.Call(
		uintptr(unsafe.Pointer(&audio.hwi)), waveMapper,
		uintptr(unsafe.Pointer(&wfx)), ev, 0, callbackEvent,
	)
	if ret != 0 {
		log("waveInOpen failed: MMRESULT %d", ret)
		state.recording = false
		resumeMedia()
		setPhase(phaseIdle)
		return
	}

	bufSize := chunkSize * 2
	for i := range audio.bufs {
		audio.bufs[i] = make([]byte, bufSize)
		audio.hdrs[i] = wavehdr{Data: uintptr(unsafe.Pointer(&audio.bufs[i][0])), Len: uint32(bufSize)}
		pWaveInPrepareHeader.Call(audio.hwi, uintptr(unsafe.Pointer(&audio.hdrs[i])), unsafe.Sizeof(audio.hdrs[i]))
		pWaveInAddBuffer.Call(audio.hwi, uintptr(unsafe.Pointer(&audio.hdrs[i])), unsafe.Sizeof(audio.hdrs[i]))
	}

	// Audio is capturing from THIS INSTANT
	pWaveInStart.Call(audio.hwi)

	// Now launch the goroutine that just reads filled buffers
	audio.stopCh = make(chan struct{})
	audio.frameCh = make(chan []int16, 1)
	go captureLoop(audio.hwi, audio.event, &audio.hdrs, &audio.bufs, audio.stopCh, audio.frameCh)

	// Goroutine to receive frames when recording stops
	go func() {
		defer resumeMedia() // always unpause media when pipeline finishes
		defer func() {
			if r := recover(); r != nil {
				log("PANIC in transcription pipeline: %v", r)
				setPhase(phaseIdle)
			}
		}()
		frames := <-audio.frameCh

		state.mu.Lock()
		clipboard := state.clipboardMode
		state.mu.Unlock()

		if len(frames) == 0 {
			log("No audio captured")
			setPhase(phaseIdle)
			return
		}

		dur := float64(len(frames)) / sampleRate
		log("Captured %.1fs", dur)

		// Save audio BEFORE transcription (crash resilient)
		saveAudio(frames)

		setPhase(phaseProcessing)
		transcribe(frames, clipboard)
	}()
}

func stopRecording() {
	if !state.recording {
		return
	}
	log("Stopped")
	state.recording = false

	// Signal capture loop to stop, then stop the device
	if audio.stopCh != nil {
		close(audio.stopCh)
		audio.stopCh = nil
	}
}

// ----- Self-update -----
// Pull-based: app checks GitHub releases, downloads if newer, verifies SHA256.
// Flow: checkForUpdate() -> notify user -> downloadUpdate() -> verify -> restart

const githubRepo = "PSEUDONYM97/cognitive-flow"

var updateAvailable string // set to new version if update found

func checkForUpdate() {
	c := &http.Client{Timeout: 10 * time.Second}
	resp, err := c.Get(fmt.Sprintf("https://api.github.com/repos/%s/releases/latest", githubRepo))
	if err != nil {
		return // silently fail - not critical
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return
	}

	var release struct {
		TagName string `json:"tag_name"`
		Assets  []struct {
			Name string `json:"name"`
			URL  string `json:"browser_download_url"`
		} `json:"assets"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&release); err != nil {
		return
	}

	remote := strings.TrimPrefix(release.TagName, "v")
	if remote == "" || remote == version {
		return
	}

	if compareVersions(remote, version) > 0 {
		updateAvailable = remote
		log("Update available: v%s -> v%s", version, remote)
		notify("Update available", fmt.Sprintf("v%s is available (you have v%s). Right-click tray to update.", remote, version))
	}
}

func compareVersions(a, b string) int {
	pa := strings.Split(a, ".")
	pb := strings.Split(b, ".")
	for i := 0; i < 3; i++ {
		var va, vb int
		if i < len(pa) {
			fmt.Sscanf(pa[i], "%d", &va)
		}
		if i < len(pb) {
			fmt.Sscanf(pb[i], "%d", &vb)
		}
		if va > vb {
			return 1
		}
		if va < vb {
			return -1
		}
	}
	return 0
}

func downloadUpdate() {
	log("Downloading update v%s...", updateAvailable)
	notify("Downloading update", fmt.Sprintf("Downloading v%s...", updateAvailable))

	c := &http.Client{Timeout: 60 * time.Second}

	// Get release assets
	resp, err := c.Get(fmt.Sprintf("https://api.github.com/repos/%s/releases/latest", githubRepo))
	if err != nil {
		log("Update failed: %v", err)
		notify("Update failed", err.Error())
		return
	}
	defer resp.Body.Close()

	var release struct {
		Assets []struct {
			Name string `json:"name"`
			URL  string `json:"browser_download_url"`
		} `json:"assets"`
	}
	json.NewDecoder(resp.Body).Decode(&release)

	// Find the .exe and .sha256 assets
	var exeURL, shaURL string
	for _, a := range release.Assets {
		if strings.HasSuffix(a.Name, ".exe") {
			exeURL = a.URL
		}
		if strings.HasSuffix(a.Name, ".sha256") {
			shaURL = a.URL
		}
	}

	if exeURL == "" {
		log("Update failed: no .exe in release assets")
		notify("Update failed", "No binary in release")
		return
	}

	// Download binary
	exeResp, err := c.Get(exeURL)
	if err != nil {
		log("Update download failed: %v", err)
		notify("Update failed", err.Error())
		return
	}
	defer exeResp.Body.Close()

	exeData, err := io.ReadAll(exeResp.Body)
	if err != nil {
		log("Update read failed: %v", err)
		notify("Update failed", err.Error())
		return
	}

	// Verify SHA256 if available
	if shaURL != "" {
		shaResp, err := c.Get(shaURL)
		if err == nil {
			defer shaResp.Body.Close()
			shaData, _ := io.ReadAll(shaResp.Body)
			expectedHash := strings.TrimSpace(strings.Fields(string(shaData))[0])

			actualHash := sha256.Sum256(exeData)
			actualHex := hex.EncodeToString(actualHash[:])

			if actualHex != expectedHash {
				log("UPDATE REJECTED: SHA256 mismatch!")
				log("  Expected: %s", expectedHash)
				log("  Got:      %s", actualHex)
				notify("Update rejected", "SHA256 verification failed - binary may be tampered")
				return
			}
			log("SHA256 verified: %s", actualHex[:16]+"...")
		}
	} else {
		log("Warning: no .sha256 asset, skipping verification")
	}

	// Write to cogflow-update.exe next to current binary
	exe, _ := os.Executable()
	updatePath := filepath.Join(filepath.Dir(exe), "cogflow-update.exe")
	if err := os.WriteFile(updatePath, exeData, 0755); err != nil {
		log("Update write failed: %v", err)
		notify("Update failed", err.Error())
		return
	}

	log("Update downloaded to %s (%.1fMB)", updatePath, float64(len(exeData))/1024/1024)
	notify("Update ready", fmt.Sprintf("v%s downloaded. Restart cogflow to apply.", updateAvailable))
	updateAvailable = "" // clear so menu item changes
}

// applyPendingUpdate checks for cogflow-update.exe and swaps it in.
func applyPendingUpdate() {
	exe, err := os.Executable()
	if err != nil {
		return
	}
	dir := filepath.Dir(exe)
	updatePath := filepath.Join(dir, "cogflow-update.exe")
	oldPath := filepath.Join(dir, "cogflow-old.exe")

	if _, err := os.Stat(updatePath); os.IsNotExist(err) {
		return
	}

	log("Applying pending update...")

	// Remove old backup if exists
	os.Remove(oldPath)

	// Rename current -> old
	if err := os.Rename(exe, oldPath); err != nil {
		log("Update apply failed (rename current): %v", err)
		return
	}

	// Rename update -> current
	if err := os.Rename(updatePath, exe); err != nil {
		// Rollback
		os.Rename(oldPath, exe)
		log("Update apply failed (rename new): %v", err)
		return
	}

	log("Update applied! Restarting...")

	// Relaunch ourselves
	cmd := fmt.Sprintf("cmd /c timeout /t 1 /nobreak >nul & start \"\" \"%s\"", exe)
	pCreateProcess := kernel32.NewProc("CreateProcessW")
	cmdW, _ := syscall.UTF16PtrFromString(cmd)
	var si [68]byte // STARTUPINFO
	binary.LittleEndian.PutUint32(si[:4], 68)
	var pi [24]byte // PROCESS_INFORMATION
	pCreateProcess.Call(0, uintptr(unsafe.Pointer(cmdW)), 0, 0, 0, 0, 0, 0,
		uintptr(unsafe.Pointer(&si)), uintptr(unsafe.Pointer(&pi)))

	// Exit current process
	os.Exit(0)
}

// ----- Audio capture loop -----
// Only reads filled buffers. Device is already open and recording.

func captureLoop(hwi, event uintptr, hdrs *[numBufs]wavehdr, bufs *[numBufs][]byte, stop chan struct{}, frameCh chan<- []int16) {
	defer func() {
		if r := recover(); r != nil {
			log("PANIC in captureLoop: %v", r)
			// Still try to clean up and send what we have
			pWaveInStop.Call(hwi)
			pWaveInReset.Call(hwi)
			for i := range hdrs {
				pWaveInUnprepareHeader.Call(hwi, uintptr(unsafe.Pointer(&hdrs[i])), unsafe.Sizeof(hdrs[i]))
			}
			pWaveInClose.Call(hwi)
			frameCh <- nil
		}
	}()
	var frames []int16

	for {
		select {
		case <-stop:
			goto done
		default:
		}

		pWaitForSingleObject.Call(event, 100)

		select {
		case <-stop:
			goto done
		default:
		}

		for i := range hdrs {
			if hdrs[i].Flags&whdrDone != 0 {
				n := hdrs[i].Recorded
				if n > 0 {
					samples := make([]int16, n/2)
					src := unsafe.Slice((*byte)(unsafe.Pointer(hdrs[i].Data)), n)
					var sumSq float64
					for j := range samples {
						samples[j] = int16(src[j*2]) | int16(src[j*2+1])<<8
						sumSq += float64(samples[j]) * float64(samples[j])
					}
					frames = append(frames, samples...)

					// RMS as 0-100 (32768 max for int16)
					rms := math.Sqrt(sumSq / float64(len(samples)))
					level := int32(rms / 327.68) // 0-100
					if level > 100 {
						level = 100
					}
					audioLevel.Store(level)
				}
				hdrs[i].Flags &^= whdrDone // clear done bit, keep prepared
				hdrs[i].Recorded = 0
				pWaveInAddBuffer.Call(hwi, uintptr(unsafe.Pointer(&hdrs[i])), unsafe.Sizeof(hdrs[i]))
			}
		}
	}

done:
	pWaveInStop.Call(hwi)
	pWaveInReset.Call(hwi)

	// Drain remaining buffers
	for i := range hdrs {
		if hdrs[i].Flags&whdrDone != 0 && hdrs[i].Recorded > 0 {
			n := hdrs[i].Recorded
			samples := make([]int16, n/2)
			src := unsafe.Slice((*byte)(unsafe.Pointer(hdrs[i].Data)), n)
			for j := range samples {
				samples[j] = int16(src[j*2]) | int16(src[j*2+1])<<8
			}
			frames = append(frames, samples...)
		}
		pWaveInUnprepareHeader.Call(hwi, uintptr(unsafe.Pointer(&hdrs[i])), unsafe.Sizeof(hdrs[i]))
	}

	pWaveInClose.Call(hwi)
	frameCh <- frames
}

// ----- Server communication -----

func healthCheck() error {
	c := &http.Client{Timeout: 5 * time.Second}
	resp, err := c.Get(cfg.Server + "/health")
	if err != nil {
		return err
	}
	resp.Body.Close()
	return nil
}

func transcribe(samples []int16, clipboard bool) {
	start := time.Now()

	// Encode WAV
	wav := encodeWAV(samples)

	// Multipart body
	boundary := fmt.Sprintf("----cf%d", time.Now().UnixNano())
	var body bytes.Buffer
	fmt.Fprintf(&body, "--%s\r\n", boundary)
	body.WriteString("Content-Disposition: form-data; name=\"audio\"; filename=\"audio.wav\"\r\n")
	body.WriteString("Content-Type: audio/wav\r\n\r\n")
	body.Write(wav)
	fmt.Fprintf(&body, "\r\n--%s--\r\n", boundary)
	payload := body.Bytes()

	// POST with retry (5s, 10s, 15s backoff)
	var result struct {
		Text string  `json:"text"`
		Ms   float64 `json:"processing_time_ms"`
	}

	delays := [3]time.Duration{5 * time.Second, 10 * time.Second, 15 * time.Second}
	var lastErr error

	for attempt := 0; attempt <= 3; attempt++ {
		if attempt > 0 {
			log("Retry %d/3 in %v", attempt, delays[attempt-1])
			time.Sleep(delays[attempt-1])
		}

		c := &http.Client{Timeout: 30 * time.Second}
		req, _ := http.NewRequest("POST", cfg.Server+"/transcribe", bytes.NewReader(payload))
		req.Header.Set("Content-Type", "multipart/form-data; boundary="+boundary)

		resp, err := c.Do(req)
		if err != nil {
			lastErr = err
			continue
		}

		b, _ := io.ReadAll(resp.Body)
		resp.Body.Close()

		if resp.StatusCode != 200 {
			lastErr = fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(b))
			continue
		}

		json.Unmarshal(b, &result)
		lastErr = nil
		break
	}

	if lastErr != nil {
		log("Transcription failed: %v", lastErr)
		notify("Transcription failed", lastErr.Error())
		setPhase(phaseIdle)
		return
	}

	raw := strings.TrimSpace(result.Text)
	if raw == "" {
		log("Empty transcription")
		setPhase(phaseIdle)
		return
	}

	// Full text processing pipeline (6 passes)
	text := processText(raw)
	if text == "" {
		log("Empty after processing (raw: %s)", raw)
		setPhase(phaseIdle)
		return
	}

	elapsed := time.Since(start)
	dur := float64(len(samples)) / sampleRate
	log("%dms (server: %.0fms) | %s", elapsed.Milliseconds(), result.Ms, text)
	if raw != text {
		log("  raw: %s", raw)
	}

	// Save to history
	saveHistory(text, dur, result.Ms, elapsed.Milliseconds())

	// Output
	if clipboard {
		copyToClipboard(text)
		log("Copied to clipboard")
	} else {
		if err := typeText(text + " "); err != nil {
			log("SendInput failed: %v - copying to clipboard", err)
			copyToClipboard(text)
			notify("Typed via clipboard", "SendInput failed, text copied instead")
		}
	}

	setPhase(phaseIdle)
}

func saveAudio(samples []int16) {
	dir := filepath.Join(configDir(), "audio")
	os.MkdirAll(dir, 0755)
	name := time.Now().Format("2006-01-02_15-04-05") + ".wav"
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, encodeWAV(samples), 0644); err != nil {
		log("Audio save failed: %v", err)
		return
	}
	log("Saved %s", path)
}

func encodeWAV(samples []int16) []byte {
	n := len(samples) * 2
	var b bytes.Buffer
	b.Grow(44 + n)
	b.WriteString("RIFF")
	binary.Write(&b, binary.LittleEndian, uint32(36+n))
	b.WriteString("WAVEfmt ")
	binary.Write(&b, binary.LittleEndian, uint32(16))           // chunk size
	binary.Write(&b, binary.LittleEndian, uint16(1))            // PCM
	binary.Write(&b, binary.LittleEndian, uint16(1))            // mono
	binary.Write(&b, binary.LittleEndian, uint32(sampleRate))   // sample rate
	binary.Write(&b, binary.LittleEndian, uint32(sampleRate*2)) // byte rate
	binary.Write(&b, binary.LittleEndian, uint16(2))            // block align
	binary.Write(&b, binary.LittleEndian, uint16(16))           // bits
	b.WriteString("data")
	binary.Write(&b, binary.LittleEndian, uint32(n))
	for _, s := range samples {
		binary.Write(&b, binary.LittleEndian, s)
	}
	return b.Bytes()
}

// ----- Text processing pipeline -----
// Six-pass pipeline ported from the Python version. Order matters.
//   0. Hallucination loop detection (10+ word repeats)
//   1. Filler word removal (um, uh, er, etc.)
//   2. Whisper artifact correction (,nd -> command, etc.)
//   3. Character normalization (smart quotes -> ASCII)
//   4. Custom word replacements (user-configurable)
//   5. Spoken punctuation conversion (period -> .)
//   6. Spacing cleanup

var fillerWords = map[string]bool{
	"um": true, "uh": true, "uhh": true, "umm": true, "ummm": true, "uhhh": true,
	"er": true, "err": true, "errr": true, "ah": true, "ahh": true, "ahhh": true,
	"hmm": true, "hmmm": true, "hmmmm": true, "mm": true, "mmm": true, "mmmm": true,
}

// Go's regexp (RE2) doesn't support backreferences, so hallucination
// detection is done programmatically in removeHallucinations().

var whisperCorrections = []struct{ re *regexp.Regexp; repl string }{
	{regexp.MustCompile(`(?i),nd\b`), " command"},
	{regexp.MustCompile(`(?i),nds\b`), " commands"},
	{regexp.MustCompile(`(?i),nding\b`), " commanding"},
	{regexp.MustCompile(`(?i),nt\b`), " comment"},
	{regexp.MustCompile(`(?i),nts\b`), " comments"},
	{regexp.MustCompile(`(?i),n\b`), " common"},
	{regexp.MustCompile(`(?i),nly\b`), " commonly"},
	{regexp.MustCompile(`(?i)\.riod\b`), " period"},
	{regexp.MustCompile(`(?i):lon\b`), " colon"},
	{regexp.MustCompile(`(?i);micolon\b`), " semicolon"},
	{regexp.MustCompile(`(?i)\?estion\b`), " question"},
	{regexp.MustCompile(`(?i)!xclamation\b`), " exclamation"},
	{regexp.MustCompile(`(?i)\bkama\b`), "comma"},
}

var charNormalize = []struct{ from, to string }{
	{"\u2019", "'"}, // right single quote
	{"\u2018", "'"}, // left single quote
	{"\u201C", `"`}, // left double quote
	{"\u201D", `"`}, // right double quote
	{"\u2013", "-"}, // en dash
	{"\u2014", "-"}, // em dash
	{"\u2026", "..."}, // ellipsis
}

var spokenPunctuation = []struct{ re *regexp.Regexp; repl string }{
	{regexp.MustCompile(`(?i)\bnew paragraph\b`), "\n\n"},
	{regexp.MustCompile(`(?i)\bnew line\b`), "\n"},
	{regexp.MustCompile(`(?i)\bnewline\b`), "\n"},
	{regexp.MustCompile(`(?i)\benter\b`), "\n"},
	{regexp.MustCompile(`(?i)\bfull stop\b`), "."},
	{regexp.MustCompile(`(?i)\bquestion mark\b`), "?"},
	{regexp.MustCompile(`(?i)\bexclamation mark\b`), "!"},
	{regexp.MustCompile(`(?i)\bexclamation point\b`), "!"},
	{regexp.MustCompile(`(?i)\bsemi colon\b`), ";"},
	{regexp.MustCompile(`(?i)\bsemicolon\b`), ";"},
	{regexp.MustCompile(`(?i)\bopen parenthesis\b`), "("},
	{regexp.MustCompile(`(?i)\bclose parenthesis\b`), ")"},
	{regexp.MustCompile(`(?i)\bopen bracket\b`), "["},
	{regexp.MustCompile(`(?i)\bclose bracket\b`), "]"},
	{regexp.MustCompile(`(?i)\bopen brace\b`), "{"},
	{regexp.MustCompile(`(?i)\bclose brace\b`), "}"},
	{regexp.MustCompile(`(?i)\bopen quote\b`), `"`},
	{regexp.MustCompile(`(?i)\bclose quote\b`), `"`},
	{regexp.MustCompile(`(?i)\bellipsis\b`), "..."},
	{regexp.MustCompile(`(?i)\bperiod\b`), "."},
	{regexp.MustCompile(`(?i)\bcomma\b`), ","},
	{regexp.MustCompile(`(?i)\bcolon\b`), ":"},
	{regexp.MustCompile(`(?i)\bdash\b`), "-"},
	{regexp.MustCompile(`(?i)\bhyphen\b`), "-"},
	{regexp.MustCompile(`(?i)\bapostrophe\b`), "'"},
	{regexp.MustCompile(`(?i)\bquote\b`), `"`},
}

var (
	reSpacePunct   = regexp.MustCompile(`\s+([.,!?;:])`)
	reMultiSpace   = regexp.MustCompile(`\s+`)
	reTrailPunct   = regexp.MustCompile(`[.,!?;:'"]+$`)
)

// removeHallucinations collapses 10+ consecutive identical words down to one.
// e.g. "I'm I'm I'm I'm I'm I'm I'm I'm I'm I'm whatever" -> "I'm whatever"
func removeHallucinations(text string) string {
	words := strings.Fields(text)
	if len(words) < 10 {
		return text
	}

	var out []string
	i := 0
	for i < len(words) {
		w := strings.ToLower(words[i])
		// Count consecutive identical words (case-insensitive)
		j := i + 1
		for j < len(words) && strings.ToLower(words[j]) == w {
			j++
		}
		count := j - i
		if count >= 10 {
			// Hallucination: keep only one instance
			out = append(out, words[i])
			log("Hallucination: '%s' repeated %d times", words[i], count)
		} else {
			out = append(out, words[i:j]...)
		}
		i = j
	}
	return strings.Join(out, " ")
}

func processText(raw string) string {
	text := raw

	// Pass 0: Hallucination loop detection (10+ consecutive word repeats)
	text = removeHallucinations(text)

	// Pass 1: Filler word removal
	words := strings.Fields(text)
	kept := words[:0]
	for _, w := range words {
		clean := reTrailPunct.ReplaceAllString(w, "")
		if !fillerWords[strings.ToLower(clean)] {
			kept = append(kept, w)
		}
	}
	text = strings.Join(kept, " ")

	// Pass 2: Whisper artifact correction
	for _, c := range whisperCorrections {
		text = c.re.ReplaceAllString(text, c.repl)
	}

	// Pass 3: Character normalization
	for _, n := range charNormalize {
		text = strings.ReplaceAll(text, n.from, n.to)
	}

	// Pass 4: Custom word replacements
	text = applyReplacements(text)

	// Pass 5: Spoken punctuation conversion
	for _, p := range spokenPunctuation {
		text = p.re.ReplaceAllString(text, p.repl)
	}

	// Pass 6: Spacing cleanup
	text = reSpacePunct.ReplaceAllString(text, "$1")
	text = reMultiSpace.ReplaceAllString(text, " ")
	text = strings.TrimSpace(text)

	return text
}

func applyReplacements(text string) string {
	if len(cfg.Replacements) == 0 {
		return text
	}
	words := strings.Fields(text)
	for i, w := range words {
		clean := strings.TrimRightFunc(w, unicode.IsPunct)
		suffix := w[len(clean):]
		if repl, ok := cfg.Replacements[strings.ToLower(clean)]; ok {
			words[i] = repl + suffix
		}
	}
	return strings.Join(words, " ")
}

// ----- History tracking -----

type historyEntry struct {
	Timestamp string  `json:"timestamp"`
	Text      string  `json:"text"`
	DurationS float64 `json:"duration_s"`
	ServerMs  float64 `json:"server_ms"`
	TotalMs   int64   `json:"total_ms"`
}

func saveHistory(text string, durS float64, serverMs float64, totalMs int64) {
	path := filepath.Join(configDir(), "history.json")

	var history []historyEntry
	if data, err := os.ReadFile(path); err == nil {
		json.Unmarshal(data, &history)
	}

	history = append(history, historyEntry{
		Timestamp: time.Now().Format(time.RFC3339),
		Text:      text,
		DurationS: durS,
		ServerMs:  serverMs,
		TotalMs:   totalMs,
	})

	// Keep last 500 entries
	if len(history) > 500 {
		history = history[len(history)-500:]
	}

	if data, err := json.MarshalIndent(history, "", "  "); err == nil {
		os.WriteFile(path, data, 0644)
	}
}

// ----- Text output via SendInput -----

var (
	pGetForegroundWindow    = user32.NewProc("GetForegroundWindow")
	pGetWindowThreadProcId  = user32.NewProc("GetWindowThreadProcessId")
	pAttachThreadInput      = user32.NewProc("AttachThreadInput")
	pGetFocus               = user32.NewProc("GetFocus")
)

const wmChar = 0x0102

func typeText(text string) error {
	// Sanitize: drop control chars, replace backticks
	var clean strings.Builder
	for _, r := range text {
		switch {
		case r == '`':
			clean.WriteRune('\'')
		case r == '\n', r == '\r', r == '\t':
			clean.WriteRune(r)
		case r < 0x20:
			// skip control chars
		default:
			clean.WriteRune(r)
		}
	}

	runes := []rune(clean.String())
	if len(runes) == 0 {
		return nil
	}

	// Attach to the foreground window's thread to get the real focus handle,
	// then post WM_CHAR directly. This bypasses the input system entirely -
	// no hooks, no keyboard layout processing, no per-char DOM re-renders.
	// This is what the Python version did and it was fast.
	fg, _, _ := pGetForegroundWindow.Call()
	if fg == 0 {
		return fmt.Errorf("no foreground window")
	}

	fgThread, _, _ := pGetWindowThreadProcId.Call(fg, 0)
	myThread, _, _ := pGetCurrentThreadId.Call()

	attached := false
	if fgThread != myThread {
		r, _, _ := pAttachThreadInput.Call(myThread, fgThread, 1)
		attached = r != 0
	}

	focus, _, _ := pGetFocus.Call()
	if focus == 0 {
		focus = fg // fallback to foreground window itself
	}

	if attached {
		pAttachThreadInput.Call(myThread, fgThread, 0)
	}

	// Post WM_CHAR for each character
	for i, r := range runes {
		if r == '\n' || r == '\r' {
			pPostMessage.Call(focus, 0x0100, vkReturn, 0)    // WM_KEYDOWN
			pPostMessage.Call(focus, 0x0101, vkReturn, 0)    // WM_KEYUP
		} else {
			pPostMessage.Call(focus, wmChar, uintptr(r), 0)
		}
		// Brief yield every 100 chars so target app can process
		if (i+1)%100 == 0 {
			time.Sleep(time.Millisecond)
		}
	}
	return nil
}

// sendUnicode sends a single Unicode character via SendInput (KEYEVENTF_UNICODE).
func sendUnicode(ch uint16) {
	// INPUT structure: 40 bytes on amd64
	// [0:4] type=1 (keyboard), [4:8] pad, [8:10] vk=0, [10:12] scan=ch,
	// [12:16] flags, [16:20] time=0, [20:24] pad, [24:32] extra=0, [32:40] pad
	var inputs [80]byte // 2 inputs: keydown + keyup

	// Keydown
	binary.LittleEndian.PutUint32(inputs[0:4], inputKeyboard)
	binary.LittleEndian.PutUint16(inputs[10:12], ch)
	binary.LittleEndian.PutUint32(inputs[12:16], keyeventfUnicode)

	// Keyup
	binary.LittleEndian.PutUint32(inputs[40:44], inputKeyboard)
	binary.LittleEndian.PutUint16(inputs[50:52], ch)
	binary.LittleEndian.PutUint32(inputs[52:56], keyeventfUnicode|keyeventfKeyup)

	pSendInput.Call(2, uintptr(unsafe.Pointer(&inputs[0])), 40)
}

// sendKey sends a virtual key press (for Enter, etc.)
func sendKey(vk uint16, flags uint32) {
	var inputs [80]byte
	binary.LittleEndian.PutUint32(inputs[0:4], inputKeyboard)
	binary.LittleEndian.PutUint16(inputs[8:10], vk)
	binary.LittleEndian.PutUint32(inputs[12:16], flags)

	binary.LittleEndian.PutUint32(inputs[40:44], inputKeyboard)
	binary.LittleEndian.PutUint16(inputs[48:50], vk)
	binary.LittleEndian.PutUint32(inputs[52:56], flags|keyeventfKeyup)

	pSendInput.Call(2, uintptr(unsafe.Pointer(&inputs[0])), 40)
}

func copyToClipboard(text string) {
	u16 := utf16.Encode([]rune(text + "\x00"))
	size := len(u16) * 2

	r, _, _ := pOpenClipboard.Call(0)
	if r == 0 {
		return
	}
	defer pCloseClipboard.Call()
	pEmptyClipboard.Call()

	hMem, _, _ := pGlobalAlloc.Call(gmemMoveable, uintptr(size))
	if hMem == 0 {
		return
	}
	ptr, _, _ := pGlobalLock.Call(hMem)
	if ptr == 0 {
		return
	}
	dst := unsafe.Slice((*uint16)(unsafe.Pointer(ptr)), len(u16))
	copy(dst, u16)
	pGlobalUnlock.Call(hMem)
	pSetClipboardData.Call(cfUnicode, hMem)
}

// ----- Screen-edge recording bar -----
// Full-width bar at top of screen. Red while recording, amber while processing.
// Click-through (WS_EX_TRANSPARENT), always on top, invisible when idle.

const (
	barHeight    = 6
	timerRepaint = 2
)

var barProc = syscall.NewCallback(func(hwnd, umsg, wp, lp uintptr) uintptr {
	switch uint32(umsg) {
	case wmPaint:
		var ps paintstruct
		hdc, _, _ := pBeginPaint.Call(hwnd, uintptr(unsafe.Pointer(&ps)))
		sw, _, _ := pGetSystemMetrics.Call(0) // SM_CXSCREEN

		p := currentPhase()
		var color uintptr

		switch p {
		case phaseRecording:
			level := audioLevel.Load()
			switch {
			case level < 15:
				color = 0x003643F4 // Red dim
			case level < 50:
				color = 0x002030F4 // Red bright
			default:
				color = 0x001020FF // Red hot
			}
		case phaseProcessing:
			color = 0x0000BFFF // Amber
		default:
			color = 0
		}

		if color != 0 {
			brush, _, _ := pCreateSolidBrush.Call(color)
			r := [4]int32{0, 0, int32(sw), barHeight}
			pFillRect.Call(hdc, uintptr(unsafe.Pointer(&r)), brush)
			pDeleteObject.Call(brush)
		}

		pEndPaint.Call(hwnd, uintptr(unsafe.Pointer(&ps)))
		return 0

	case wmTimer:
		if wp == timerRepaint {
			pInvalidateRect.Call(hwnd, 0, 1)
			return 0
		}
		if wp == timerHeartbeat {
			now := time.Now()
			if now.Sub(state.lastWake) > 60*time.Second {
				log("Wake detected, pinging server")
				go healthCheck()
			}
			state.lastWake = now
		}
		return 0

	case wmSetPhase:
		applyPhase(int32(wp))
		return 0
	}

	r, _, _ := pDefWindowProc.Call(hwnd, umsg, wp, lp)
	return r
})

var pKillTimer = user32.NewProc("KillTimer")

func createBar() {
	cls := utf16p("CogFlowBar")
	cursor, _, _ := pLoadCursor.Call(0, 32512) // IDC_ARROW
	wc := wndclass{
		Size: uint32(unsafe.Sizeof(wndclass{})), WndProc: barProc,
		ClassName: cls, Cursor: cursor,
	}
	pRegisterClassEx.Call(uintptr(unsafe.Pointer(&wc)))

	sw, _, _ := pGetSystemMetrics.Call(0)
	hwnd, _, _ := pCreateWindowEx.Call(
		wsExLayered|wsExTopmost|wsExToolWindow|wsExNoActivate|wsExTransparent,
		uintptr(unsafe.Pointer(cls)), 0,
		wsPopup, 0, 0, sw, barHeight, 0, 0, 0, 0,
	)
	state.bar = hwnd

	pSetLayeredWindowAttr.Call(hwnd, 0, 220, lwaAlpha)

	// Start hidden
	pShowWindow.Call(hwnd, 0) // SW_HIDE

	// Heartbeat timer for sleep/wake detection
	pSetTimer.Call(hwnd, timerHeartbeat, 30000, 0)
}

// ----- Per-pixel alpha indicator -----
// Software-rendered overlay with anti-aliased dot, glow, and pulsing.
// Uses UpdateLayeredWindow for true per-pixel transparency.

const (
	indSize       = 56            // window size
	indDotRadius  = 10.0          // dot radius in pixels
	indGlowMin    = 13.0          // minimum glow radius
	indGlowMax    = 24.0          // maximum glow radius (loud audio)
	timerIndAnim  = 3             // animation timer ID
	ulwAlpha      = 0x00000002    // ULW_ALPHA flag
	wmLButtonDown = 0x0201
	wmMouseMove   = 0x0200
)

var ind struct {
	hwnd    uintptr
	memDC   uintptr
	dib     uintptr
	pixels  []byte // BGRA premultiplied, top-down
	dist    [indSize * indSize]float64 // precomputed distance from center
	frame   int
	dragX   int16  // mouse-down position for click vs drag
	dragY   int16
	dragging bool
}

type indColor struct{ r, g, b float64 }

var (
	colGreen = indColor{76, 175, 80}
	colRed   = indColor{229, 57, 53}
	colAmber = indColor{251, 176, 59}
	colGray  = indColor{120, 120, 120}
)

var indProc = syscall.NewCallback(func(hwnd, umsg, wp, lp uintptr) uintptr {
	switch uint32(umsg) {
	case wmLButtonDown:
		ind.dragX = int16(lp & 0xFFFF)
		ind.dragY = int16(lp >> 16 & 0xFFFF)
		ind.dragging = false
		pSetCapture.Call(hwnd)
		return 0

	case wmMouseMove:
		if wp&0x0001 != 0 { // MK_LBUTTON
			mx, my := int16(lp&0xFFFF), int16(lp>>16&0xFFFF)
			dx, dy := mx-ind.dragX, my-ind.dragY
			if !ind.dragging && (dx*dx+dy*dy > 9) {
				ind.dragging = true
			}
			if ind.dragging {
				var rc [4]int32
				pGetWindowRect.Call(hwnd, uintptr(unsafe.Pointer(&rc)))
				nx := rc[0] + int32(dx)
				ny := rc[1] + int32(dy)
				pSetWindowPos.Call(hwnd, 0, uintptr(nx), uintptr(ny), 0, 0, 0x0001|0x0004|0x0010)
			}
		}
		return 0

	case wmLButtonUp:
		pReleaseCapture.Call()
		if !ind.dragging {
			// Click: toggle clipboard recording
			go func() {
				if currentPhase() == phaseRecording {
					toggle(true)
				} else if state.enabled {
					toggle(true)
				}
			}()
		}
		ind.dragging = false
		return 0

	case wmTimer:
		if wp == timerIndAnim {
			renderIndicator()
		}
		return 0
	}

	r, _, _ := pDefWindowProc.Call(hwnd, umsg, wp, lp)
	return r
})

func createIndicator() {
	cls := utf16p("CogFlowInd")
	cursor, _, _ := pLoadCursor.Call(0, 32649) // IDC_HAND
	wc := wndclass{
		Size: uint32(unsafe.Sizeof(wndclass{})), WndProc: indProc,
		ClassName: cls, Cursor: cursor,
	}
	pRegisterClassEx.Call(uintptr(unsafe.Pointer(&wc)))

	sw, _, _ := pGetSystemMetrics.Call(0)
	sh, _, _ := pGetSystemMetrics.Call(1)
	x := int(sw) - indSize - 24
	y := int(sh) - indSize - 64

	hwnd, _, _ := pCreateWindowEx.Call(
		wsExLayered|wsExTopmost|wsExToolWindow,
		uintptr(unsafe.Pointer(cls)), 0,
		wsPopup, uintptr(x), uintptr(y), indSize, indSize,
		0, 0, 0, 0,
	)
	ind.hwnd = hwnd
	state.dot = hwnd

	// Create memory DC + 32-bit top-down DIB for software rendering
	screenDC, _, _ := pGetDC.Call(0)
	ind.memDC, _, _ = pCreateCompatibleDC.Call(screenDC)
	pReleaseDC.Call(0, screenDC)

	bmi := bmpinfo{
		Size:   uint32(unsafe.Sizeof(bmpinfo{})),
		Width:  indSize,
		Height: -indSize, // negative = top-down
		Planes: 1,
		Bits:   32,
	}
	var bits uintptr
	ind.dib, _, _ = pCreateDIBSection.Call(
		ind.memDC, uintptr(unsafe.Pointer(&bmi)), 0,
		uintptr(unsafe.Pointer(&bits)), 0, 0,
	)
	pSelectObject.Call(ind.memDC, ind.dib)

	ind.pixels = unsafe.Slice((*byte)(unsafe.Pointer(bits)), indSize*indSize*4)

	initDistTable()
	renderIndicator()
	pShowWindow.Call(hwnd, 8) // SW_SHOWNA
	state.indVisible = true

	// Slow heartbeat timer when idle (redraws on state changes via applyPhase)
	pSetTimer.Call(hwnd, timerIndAnim, 1000, 0)
}

func initDistTable() {
	cx, cy := float64(indSize)/2, float64(indSize)/2
	for y := 0; y < indSize; y++ {
		for x := 0; x < indSize; x++ {
			dx, dy := float64(x)-cx+0.5, float64(y)-cy+0.5
			ind.dist[y*indSize+x] = math.Sqrt(dx*dx + dy*dy)
		}
	}
}

func renderIndicator() {
	w := indSize
	px := ind.pixels

	// Clear to fully transparent
	for i := range px {
		px[i] = 0
	}

	// Determine dot color and glow based on state
	p := currentPhase()
	var col indColor
	var glowR, glowAlpha float64
	var pulse float64 = 1.0

	switch {
	case !state.enabled:
		col = colGray
		glowR = indGlowMin
		glowAlpha = 0.15
	case p == phaseRecording:
		col = colRed
		level := float64(audioLevel.Load()) / 100.0
		glowR = indGlowMin + level*(indGlowMax-indGlowMin)
		glowAlpha = 0.25 + level*0.35
		pulse = 0.65 + 0.35*math.Sin(float64(ind.frame)*0.21)
	case p == phaseProcessing:
		col = colAmber
		glowR = indGlowMin + 2
		glowAlpha = 0.3
		pulse = 0.8 + 0.2*math.Sin(float64(ind.frame)*0.12)
	default:
		col = colGreen
		glowR = indGlowMin
		glowAlpha = 0.2
	}

	// Single pass using precomputed distance table. Zero sqrt calls.
	// Dot and glow share the same color, so alpha compositing simplifies to
	// just combining the two alpha values: oA = dotA + glowA*(1-dotA).
	dotInner := indDotRadius - 0.5
	dotOuter := indDotRadius + 0.5

	for idx := 0; idx < w*w; idx++ {
		dist := ind.dist[idx]
		if dist >= glowR {
			continue
		}

		var a float64

		if dist < dotInner {
			// Fully inside dot
			a = pulse
		} else if dist < dotOuter {
			// AA edge of dot, composite with underlying glow
			dotA := pulse * (1.0 - (dist - dotInner))
			glowT := (dist - indDotRadius) / (glowR - indDotRadius)
			if glowT < 0 {
				glowT = 0
			}
			glowA := (1 - glowT*glowT) * glowAlpha * pulse
			a = dotA + glowA*(1-dotA) // src-over (same color = trivial)
		} else {
			// Pure glow
			t := (dist - indDotRadius) / (glowR - indDotRadius)
			a = (1 - t*t) * glowAlpha * pulse
		}

		if a > 0.004 {
			i := idx * 4
			px[i+0] = byte(col.b * a)
			px[i+1] = byte(col.g * a)
			px[i+2] = byte(col.r * a)
			px[i+3] = byte(255 * a)
		}
	}

	// Push to screen via UpdateLayeredWindow
	srcPt := [2]int32{0, 0}
	sz := [2]int32{int32(w), int32(w)}
	blend := uint32(0) | uint32(0)<<8 | uint32(255)<<16 | uint32(1)<<24
	pUpdateLayeredWindow.Call(
		ind.hwnd, 0, 0, uintptr(unsafe.Pointer(&sz)),
		ind.memDC, uintptr(unsafe.Pointer(&srcPt)),
		0, uintptr(unsafe.Pointer(&blend)), ulwAlpha,
	)

	ind.frame++
}

func setPhase(p int32) {
	setPhaseVal(p)
	// Post to bar window so UI update happens on main thread.
	// Calling ShowWindow/SetTimer from goroutines deadlocks the message pump.
	if state.bar != 0 {
		pPostMessage.Call(state.bar, wmSetPhase, uintptr(p), 0)
	}
}

// applyPhase runs on the main thread via the bar's wndproc.
func applyPhase(p int32) {
	switch p {
	case phaseRecording:
		audioLevel.Store(0)
		ind.frame = 0
		pShowWindow.Call(state.bar, 8) // SW_SHOWNA
		pInvalidateRect.Call(state.bar, 0, 1)
		pSetWindowPos.Call(state.bar, ^uintptr(0), 0, 0, 0, 0, 0x0001|0x0002|0x0010)
		pSetTimer.Call(state.bar, timerRepaint, 200, 0)
		// 66ms indicator animation (15fps) during recording
		pSetTimer.Call(ind.hwnd, timerIndAnim, 66, 0)
	case phaseProcessing:
		pKillTimer.Call(state.bar, timerRepaint)
		audioLevel.Store(0)
		pShowWindow.Call(state.bar, 8)
		pInvalidateRect.Call(state.bar, 0, 1)
		pSetWindowPos.Call(state.bar, ^uintptr(0), 0, 0, 0, 0, 0x0001|0x0002|0x0010)
		// Slower animation during processing (15fps)
		pSetTimer.Call(ind.hwnd, timerIndAnim, 66, 0)
	default:
		pKillTimer.Call(state.bar, timerRepaint)
		audioLevel.Store(0)
		pShowWindow.Call(state.bar, 0) // SW_HIDE
		// Back to slow heartbeat when idle.
		// Don't call renderIndicator() - it's cross-window GDI from bar's wndproc.
		// The timer will pick it up within 1s.
		pSetTimer.Call(ind.hwnd, timerIndAnim, 1000, 0)
	}

	updateTrayIcon()
	// Don't call renderIndicator() here - let the timer handle it.
	// Cross-window GDI from a wndproc can stall the message pump.
}

// ----- System tray -----

const (
	menuToggle     = 1
	menuQuit       = 2
	menuPauseMedia = 3
	menuShowHide   = 4
	menuUpdate     = 6
)

var trayProc = syscall.NewCallback(func(hwnd, umsg, wp, lp uintptr) uintptr {
	switch uint32(umsg) {
	case wmTrayIcon:
		if uint16(lp) == uint16(wmRButtonUp) {
			showMenu(hwnd)
		}
		return 0
	case wmCommand:
		switch uint16(wp) {
		case menuToggle:
			state.enabled = !state.enabled
			s := "enabled"
			if !state.enabled {
				s = "disabled"
			}
			log("Hotkey %s via menu", s)
		case menuPauseMedia:
			cfg.PauseMedia = !cfg.PauseMedia
			s := "on"
			if !cfg.PauseMedia {
				s = "off"
			}
			log("Pause media: %s", s)
			saveConfig()
		case menuShowHide:
			if state.indVisible {
				pShowWindow.Call(ind.hwnd, 0) // SW_HIDE
				state.indVisible = false
			} else {
				pShowWindow.Call(ind.hwnd, 8) // SW_SHOWNA
				state.indVisible = true
			}
		case menuUpdate:
			go downloadUpdate()
		case menuQuit:
			shutdown()
			pPostQuitMessage.Call(0)
		}
		return 0
	}
	r, _, _ := pDefWindowProc.Call(hwnd, umsg, wp, lp)
	return r
})

func createTray() {
	cls := utf16p("CogFlowTray")
	wc := wndclass{
		Size: uint32(unsafe.Sizeof(wndclass{})), WndProc: trayProc, ClassName: cls,
	}
	pRegisterClassEx.Call(uintptr(unsafe.Pointer(&wc)))

	hwnd, _, _ := pCreateWindowEx.Call(0,
		uintptr(unsafe.Pointer(cls)), uintptr(unsafe.Pointer(utf16p(""))),
		0, 0, 0, 0, 0, 0, 0, 0, 0,
	)
	state.trayHwnd = hwnd

	state.trayNID = notifyicon{
		Size:        uint32(unsafe.Sizeof(notifyicon{})),
		Hwnd:        hwnd,
		ID:          1,
		Flags:       nifMsg | nifIcon | nifTip | nifShowTip,
		CallbackMsg: wmTrayIcon,
	}
	tip, _ := syscall.UTF16FromString(fmt.Sprintf("Cognitive Flow v%s", version))
	copy(state.trayNID.Tip[:], tip)

	state.trayIcon = makeIcon(76, 175, 80) // Green
	state.trayNID.Icon = state.trayIcon

	pShellNotifyIcon.Call(nimAdd, uintptr(unsafe.Pointer(&state.trayNID)))
}

func showMenu(hwnd uintptr) {
	h, _, _ := pCreatePopupMenu.Call()

	// Hotkey toggle
	flags := uintptr(mfString)
	if state.enabled {
		flags |= mfChecked
	}
	pAppendMenu.Call(h, flags, menuToggle, uintptr(unsafe.Pointer(utf16p("Hotkey Enabled"))))

	// Pause media toggle
	flags = uintptr(mfString)
	if cfg.PauseMedia {
		flags |= mfChecked
	}
	pAppendMenu.Call(h, flags, menuPauseMedia, uintptr(unsafe.Pointer(utf16p("Pause Media"))))

	pAppendMenu.Call(h, mfSeparator, 0, 0)

	// Overlay controls
	label := "Hide Indicator"
	if !state.indVisible {
		label = "Show Indicator"
	}
	pAppendMenu.Call(h, mfString, menuShowHide, uintptr(unsafe.Pointer(utf16p(label))))

	// Update available
	if updateAvailable != "" {
		pAppendMenu.Call(h, mfString, menuUpdate, uintptr(unsafe.Pointer(utf16p(fmt.Sprintf("Update to v%s", updateAvailable)))))
	}

	pAppendMenu.Call(h, mfSeparator, 0, 0)
	pAppendMenu.Call(h, mfString, menuQuit, uintptr(unsafe.Pointer(utf16p("Quit"))))

	var pt [2]int32
	pGetCursorPos.Call(uintptr(unsafe.Pointer(&pt)))
	pSetForegroundWindow.Call(hwnd)
	pTrackPopupMenu.Call(h, 0x0020, uintptr(pt[0]), uintptr(pt[1]), 0, hwnd, 0)
	pDestroyMenu.Call(h)
}

func updateTrayIcon() {
	var r, g, b byte
	switch currentPhase() {
	case phaseRecording:
		r, g, b = 244, 67, 54
	case phaseProcessing:
		r, g, b = 255, 191, 0
	default:
		r, g, b = 76, 175, 80
	}

	old := state.trayIcon
	state.trayIcon = makeIcon(r, g, b)
	state.trayNID.Icon = state.trayIcon
	state.trayNID.Flags = nifIcon
	pShellNotifyIcon.Call(nimModify, uintptr(unsafe.Pointer(&state.trayNID)))
	if old != 0 {
		pDestroyIcon.Call(old)
	}
}

func notify(title, msg string) {
	t, _ := syscall.UTF16FromString(title)
	m, _ := syscall.UTF16FromString(msg)
	copy(state.trayNID.InfoTitle[:], t)
	copy(state.trayNID.Info[:], m)
	state.trayNID.Flags = nifInfo
	state.trayNID.InfoFlags = niiInfo
	pShellNotifyIcon.Call(nimModify, uintptr(unsafe.Pointer(&state.trayNID)))
}

func makeIcon(r, g, b byte) uintptr {
	hdc, _, _ := pGetDC.Call(0)
	bmi := bmpinfo{
		Size: uint32(unsafe.Sizeof(bmpinfo{})),
		Width: 16, Height: -16, Planes: 1, Bits: 32,
	}
	var bits uintptr
	hbm, _, _ := pCreateDIBSection.Call(hdc, uintptr(unsafe.Pointer(&bmi)), 0, uintptr(unsafe.Pointer(&bits)), 0, 0)
	pReleaseDC.Call(0, hdc)
	if hbm == 0 {
		return 0
	}

	px := unsafe.Slice((*byte)(unsafe.Pointer(bits)), 16*16*4)
	cx, cy, rad := 8.0, 8.0, 7.0
	for y := 0; y < 16; y++ {
		for x := 0; x < 16; x++ {
			dx, dy := float64(x)-cx+0.5, float64(y)-cy+0.5
			if math.Sqrt(dx*dx+dy*dy) <= rad {
				o := (y*16 + x) * 4
				px[o], px[o+1], px[o+2], px[o+3] = b, g, r, 255
			}
		}
	}

	hdc2, _, _ := pGetDC.Call(0)
	var mbits uintptr
	hmask, _, _ := pCreateDIBSection.Call(hdc2, uintptr(unsafe.Pointer(&bmi)), 0, uintptr(unsafe.Pointer(&mbits)), 0, 0)
	pReleaseDC.Call(0, hdc2)

	ii := iconinfo{IsIcon: 1, Mask: hmask, Color: hbm}
	icon, _, _ := pCreateIconIndirect.Call(uintptr(unsafe.Pointer(&ii)))
	pDeleteObject.Call(hbm)
	pDeleteObject.Call(hmask)
	return icon
}

// ----- Helpers -----

func utf16p(s string) *uint16 {
	p, _ := syscall.UTF16PtrFromString(s)
	return p
}

var logFile *os.File

func initLog() {
	dir := configDir()
	os.MkdirAll(dir, 0755)
	f, err := os.OpenFile(filepath.Join(dir, "cogflow.log"), os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err == nil {
		logFile = f
	}
}

func log(f string, a ...interface{}) {
	line := fmt.Sprintf("[%s] %s\n", time.Now().Format("15:04:05"), fmt.Sprintf(f, a...))
	fmt.Print(line)
	if logFile != nil {
		logFile.WriteString(time.Now().Format("2006-01-02 ") + line)
	}
}

func fatal(f string, a ...interface{}) {
	log("FATAL: "+f, a...)
	os.Exit(1)
}
