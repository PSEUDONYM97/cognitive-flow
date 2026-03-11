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
	"net"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
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
	pSendMessage         = user32.NewProc("SendMessageW")
	pCreatePopupMenu     = user32.NewProc("CreatePopupMenu")
	pAppendMenu          = user32.NewProc("AppendMenuW")
	pTrackPopupMenu      = user32.NewProc("TrackPopupMenu")
	pDestroyMenu         = user32.NewProc("DestroyMenu")
	pCreateIconIndirect  = user32.NewProc("CreateIconIndirect")
	pDestroyIcon         = user32.NewProc("DestroyIcon")

	pIsWindowVisible       = user32.NewProc("IsWindowVisible")
	pGetWindowRect         = user32.NewProc("GetWindowRect")
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
	pShellExecute    = shell32.NewProc("ShellExecuteW")
	pSetCapture        = user32.NewProc("SetCapture")
	pReleaseCapture    = user32.NewProc("ReleaseCapture")
	pTrackMouseEvent   = user32.NewProc("TrackMouseEvent")
)

// ----- Constants -----

const (
	version = "2.9.2"

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

	lastOutput     string    // last transcription result for "Copy Last"
	lastSamples    []int16  // last recording for "Retry Last"
	recentTexts    [5]string // last 5 transcriptions (circular)
	recentCount    int       // total transcription count
	startTime      time.Time // for uptime display

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
	state.startTime = time.Now()
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

	// Start local dashboard server
	startDashboard()

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
// Pauses media during recording so mic doesn't pick up playback, and resumes after.
// Uses DIRECTIONAL WM_APPCOMMAND (APPCOMMAND_MEDIA_PAUSE=47 / APPCOMMAND_MEDIA_PLAY=46)
// sent to the foreground window. DefWindowProc routes unhandled commands to the shell,
// which forwards to the active SMTC media session handler (Chrome, Spotify, etc.).
// Only resumes if audio was confirmed playing when we paused.

const (
	wmAppCommand     = 0x0319
	appCmdMediaPlay  = 46 // directional: only plays, never pauses
	appCmdMediaPause = 47 // directional: only pauses, never plays
)

var (
	mediaPaused bool // true if WE paused and should resume
	ole32       = windows.NewLazySystemDLL("ole32.dll")
	pCoInit     = ole32.NewProc("CoInitializeEx")
	pCoCreateInst = ole32.NewProc("CoCreateInstance")
)

// WASAPI COM GUIDs
var (
	clsidMMDevEnum = [16]byte{0x95, 0x03, 0xDE, 0xBC, 0x2F, 0xE5, 0x7C, 0x46, 0x8E, 0x3D, 0xC4, 0x57, 0x92, 0x91, 0x69, 0x2E}
	// IID_IMMDeviceEnumerator {A95664D2-9614-4F35-A746-DE8DB63617E6}
	iidIMMDevEnum = [16]byte{0xD2, 0x64, 0x56, 0xA9, 0x14, 0x96, 0x35, 0x4F, 0xA7, 0x46, 0xDE, 0x8D, 0xB6, 0x36, 0x17, 0xE6}
	iidAudioMeter  = [16]byte{0xF6, 0x16, 0x22, 0xC0, 0x67, 0x8C, 0x5B, 0x4B, 0x9D, 0x00, 0xD0, 0x08, 0xE7, 0x3E, 0x00, 0x64}
)

// sendAppCommand sends a directional media command via WM_APPCOMMAND to the foreground window.
// DefWindowProc routes unhandled commands up the parent chain to the shell's SMTC router.
func sendAppCommand(cmd int) {
	fg, _, _ := pGetForegroundWindow.Call()
	if fg == 0 {
		log("sendAppCommand(%d): no foreground window", cmd)
		return
	}
	// wParam = originating window, lParam = (cmd << 16) | FAPPCOMMAND_KEY(0)
	pSendMessage.Call(fg, wmAppCommand, fg, uintptr(cmd<<16))
}

// isAudioPlaying checks if any audio is being output via the default render device.
// Uses IAudioMeterInformation::GetPeakValue. Logs each COM step for debugging.
func isAudioPlaying() bool {
	hr, _, _ := pCoInit.Call(0, 0) // COINIT_MULTITHREADED
	// hr 0 = success, 1 = already initialized (S_FALSE) - both OK
	if hr != 0 && hr != 1 {
		log("isAudioPlaying: CoInitializeEx failed: 0x%x", hr)
		return false
	}

	var enumPtr uintptr
	hr, _, _ = pCoCreateInst.Call(
		uintptr(unsafe.Pointer(&clsidMMDevEnum)),
		0,
		1|4, // CLSCTX_INPROC_SERVER | CLSCTX_LOCAL_SERVER
		uintptr(unsafe.Pointer(&iidIMMDevEnum)),
		uintptr(unsafe.Pointer(&enumPtr)),
	)
	if hr != 0 || enumPtr == 0 {
		log("isAudioPlaying: CoCreateInstance failed: hr=0x%x ptr=%d", hr, enumPtr)
		return false
	}
	defer comRelease(enumPtr)

	// GetDefaultAudioEndpoint(eRender=0, eConsole=0, &device)
	vtbl := *(*[8]uintptr)(unsafe.Pointer(*(*uintptr)(unsafe.Pointer(enumPtr))))
	var devicePtr uintptr
	hr, _, _ = syscall.SyscallN(vtbl[4], enumPtr, 0, 0, uintptr(unsafe.Pointer(&devicePtr)))
	if hr != 0 || devicePtr == 0 {
		log("isAudioPlaying: GetDefaultAudioEndpoint failed: hr=0x%x", hr)
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
		log("isAudioPlaying: Activate AudioMeter failed: hr=0x%x", hr)
		return false
	}
	defer comRelease(meterPtr)

	// GetPeakValue(&peak)
	vtbl3 := *(*[4]uintptr)(unsafe.Pointer(*(*uintptr)(unsafe.Pointer(meterPtr))))
	var peak float32
	hr, _, _ = syscall.SyscallN(vtbl3[3], meterPtr, uintptr(unsafe.Pointer(&peak)))
	if hr != 0 {
		log("isAudioPlaying: GetPeakValue failed: hr=0x%x", hr)
		return false
	}

	log("isAudioPlaying: peak=%.6f", peak)
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
	playing := isAudioPlaying()
	if !playing {
		log("No audio playing, skipping pause")
		return
	}
	// SendInput with VK_MEDIA_PLAY_PAUSE goes through the keyboard -> shell -> SMTC
	// routing that Chrome/Spotify actually respond to. WM_APPCOMMAND sent directly
	// to windows doesn't reliably reach the media session handler.
	// The isAudioPlaying() guard above prevents accidental starts (only fires when
	// audio IS playing, so the toggle always means "pause").
	sendKey(0xB3, keyeventfExtended) // VK_MEDIA_PLAY_PAUSE
	mediaPaused = true
	log("Media paused (SendInput)")
}

func resumeMedia() {
	if !mediaPaused {
		return
	}
	mediaPaused = false
	sendKey(0xB3, keyeventfExtended) // VK_MEDIA_PLAY_PAUSE
	log("Media resumed (SendInput)")
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
		state.lastSamples = frames
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
	resumeMedia() // resume immediately on stop, don't wait for transcription

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

// ----- Web dashboard -----
// Local HTTP server for history browsing, stats, and vocabulary management.
// Binds to 127.0.0.1 only - never exposed to network.

var dashboardPort int

func startDashboard() {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		log("Dashboard server failed: %v", err)
		return
	}
	dashboardPort = ln.Addr().(*net.TCPAddr).Port
	log("Dashboard: http://127.0.0.1:%d", dashboardPort)

	mux := http.NewServeMux()
	mux.HandleFunc("/", handleDashboard)
	mux.HandleFunc("/api/history", handleAPIHistory)
	mux.HandleFunc("/api/stats", handleAPIStats)
	mux.HandleFunc("/api/vocab", handleAPIVocab)
	mux.HandleFunc("/api/vocab/add", handleAPIVocabAdd)
	mux.HandleFunc("/api/vocab/delete", handleAPIVocabDelete)

	go http.Serve(ln, mux)
}

func openDashboard(fragment string) {
	if dashboardPort == 0 {
		return
	}
	url := fmt.Sprintf("http://127.0.0.1:%d/%s", dashboardPort, fragment)
	p, _ := syscall.UTF16PtrFromString(url)
	pShellExecute.Call(0, 0, uintptr(unsafe.Pointer(p)), 0, 0, 1)
}

func handleAPIHistory(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	path := filepath.Join(configDir(), "history.json")
	data, err := os.ReadFile(path)
	if err != nil {
		w.Write([]byte("[]"))
		return
	}
	w.Write(data)
}

func handleAPIStats(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	path := filepath.Join(configDir(), "history.json")
	var history []historyEntry
	if data, err := os.ReadFile(path); err == nil {
		json.Unmarshal(data, &history)
	}

	var totalDur, totalServer float64
	var totalTotal int64
	for _, h := range history {
		totalDur += h.DurationS
		totalServer += h.ServerMs
		totalTotal += h.TotalMs
	}

	n := len(history)
	avgServer := 0.0
	avgTotal := int64(0)
	if n > 0 {
		avgServer = totalServer / float64(n)
		avgTotal = totalTotal / int64(n)
	}

	uptime := time.Since(state.startTime)

	stats := map[string]interface{}{
		"version":          version,
		"uptime_s":         int(uptime.Seconds()),
		"session_count":    state.recentCount,
		"total_count":      n,
		"total_audio_s":    totalDur,
		"avg_server_ms":    avgServer,
		"avg_total_ms":     avgTotal,
		"server":           cfg.Server,
	}
	json.NewEncoder(w).Encode(stats)
}

func handleAPIVocab(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(cfg.Replacements)
}

func handleAPIVocabAdd(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "POST only", 405)
		return
	}
	var req struct {
		From string `json:"from"`
		To   string `json:"to"`
	}
	json.NewDecoder(r.Body).Decode(&req)
	if req.From == "" {
		http.Error(w, "missing 'from'", 400)
		return
	}
	cfg.Replacements[req.From] = req.To
	saveConfig()
	log("Vocab added: %q -> %q", req.From, req.To)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(cfg.Replacements)
}

func handleAPIVocabDelete(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "POST only", 405)
		return
	}
	var req struct {
		From string `json:"from"`
	}
	json.NewDecoder(r.Body).Decode(&req)
	delete(cfg.Replacements, req.From)
	saveConfig()
	log("Vocab removed: %q", req.From)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(cfg.Replacements)
}

func handleDashboard(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")

	histPath := filepath.Join(configDir(), "history.json")
	var history []historyEntry
	if data, err := os.ReadFile(histPath); err == nil {
		json.Unmarshal(data, &history)
	}

	// Reverse for newest-first
	for i, j := 0, len(history)-1; i < j; i, j = i+1, j-1 {
		history[i], history[j] = history[j], history[i]
	}

	n := len(history)
	var totalDur, totalServer float64
	var serverTimes []float64
	today := time.Now().Format("2006-01-02")
	todayCount := 0

	for _, h := range history {
		totalDur += h.DurationS
		totalServer += h.ServerMs
		if h.ServerMs > 0 {
			serverTimes = append(serverTimes, h.ServerMs)
		}
		if strings.HasPrefix(h.Timestamp, today) {
			todayCount++
		}
	}

	avgServer := int64(0)
	avgDuration := 0.0
	p95Server := int64(0)
	if n > 0 {
		avgServer = int64(totalServer) / int64(n)
		avgDuration = totalDur / float64(n)
	}
	if len(serverTimes) > 0 {
		sort.Float64s(serverTimes)
		idx := int(float64(len(serverTimes)) * 0.95)
		if idx >= len(serverTimes) {
			idx = len(serverTimes) - 1
		}
		p95Server = int64(serverTimes[idx])
	}

	uptime := time.Since(state.startTime)
	uptimeStr := ""
	if uptime.Hours() >= 1 {
		uptimeStr = fmt.Sprintf("%dh %dm", int(uptime.Hours()), int(uptime.Minutes())%60)
	} else {
		uptimeStr = fmt.Sprintf("%dm", int(uptime.Minutes()))
	}
	startedStr := state.startTime.Format("3:04 PM")

	statsObj := map[string]interface{}{
		"version":      version,
		"total":        n,
		"today":        todayCount,
		"minutes":      totalDur / 60.0,
		"avg_duration": avgDuration,
		"avg_server":   avgServer,
		"p95_server":   p95Server,
		"uptime":       uptimeStr,
		"started":      startedStr,
		"sessions":     state.recentCount,
		"corrections":  len(cfg.Replacements),
	}

	statsJSON, _ := json.Marshal(statsObj)
	histJSON, _ := json.Marshal(history)
	vocabJSON, _ := json.Marshal(cfg.Replacements)

	html := strings.Replace(dashboardHTML, "/**STATS**/null", string(statsJSON), 1)
	html = strings.Replace(html, "/**HISTORY**/[]", string(histJSON), 1)
	html = strings.Replace(html, "/**VOCAB**/{}", string(vocabJSON), 1)
	io.WriteString(w, html)
}

const dashboardHTML = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Cognitive Flow</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
<link href="https://unpkg.com/lucide-static@latest/font/lucide.css" rel="stylesheet">
<style>
:root{
  --bg-deep:#0A0F1C;--bg-card:#1E293B;--bg-inset:#0F172A;
  --accent-cyan:#22D3EE;--accent-cyan-dim:#22D3EE33;--accent-cyan-hover:#06B6D4;
  --text-primary:#FFF;--text-secondary:#94A3B8;--text-tertiary:#64748B;--text-muted:#475569;--text-inverted:#0A0F1C;
  --color-success:#22C55E;--color-warning:#EAB308;--color-error:#EF4444;
  --divider:#0F172A;--border-subtle:#2D3748;
  --radius-sm:4px;--radius-md:6px;--radius-lg:8px;--radius-xl:12px;
  --font-mono:"JetBrains Mono","Fira Code","Cascadia Code",monospace;
  --font-sans:"Inter",-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  --text-xs:10px;--text-sm:11px;--text-base:13px;--text-md:14px;--text-lg:15px;
  --text-5xl:32px;
  --transition-fast:150ms ease;--transition-base:200ms ease;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;font-family:var(--font-sans);background:var(--bg-deep);color:var(--text-primary);-webkit-font-smoothing:antialiased}
.section-label{font-family:var(--font-mono);font-size:var(--text-xs);font-weight:600;color:var(--text-muted);letter-spacing:2px;text-transform:uppercase}
::-webkit-scrollbar{width:6px}
::-webkit-scrollbar-track{background:var(--bg-deep)}
::-webkit-scrollbar-thumb{background:var(--text-muted);border-radius:3px}
::-webkit-scrollbar-thumb:hover{background:var(--text-tertiary)}
:focus-visible{outline:2px solid var(--accent-cyan);outline-offset:2px}
::selection{background:var(--accent-cyan);color:var(--text-inverted)}

.dashboard{display:flex;flex-direction:column;height:100vh;background:var(--bg-deep);overflow:hidden}
.dash-header{display:flex;align-items:center;justify-content:space-between;padding:16px 24px;background:var(--bg-card);flex-shrink:0}
.dash-header__brand{display:flex;align-items:center;gap:12px}
.dash-header__logo{display:flex;align-items:center;justify-content:center;width:28px;height:28px;background:var(--accent-cyan);border-radius:var(--radius-md);color:var(--text-inverted)}
.dash-header__logo i{font-size:16px}
.dash-header__title{font-family:var(--font-mono);font-size:var(--text-lg);font-weight:700;color:var(--text-primary)}
.dash-tabs{display:flex;gap:0;background:var(--bg-inset);border-radius:var(--radius-lg);padding:4px}
.dash-tabs__tab{padding:8px 20px;border-radius:var(--radius-md);font-family:var(--font-mono);font-size:12px;font-weight:500;color:var(--text-muted);cursor:pointer;border:none;background:none;transition:all var(--transition-fast);white-space:nowrap}
.dash-tabs__tab:hover{color:var(--text-secondary)}
.dash-tabs__tab--active{background:var(--accent-cyan);color:var(--text-inverted);font-weight:600}
.dash-tabs__tab--active:hover{color:var(--text-inverted)}
.dash-header__status{display:flex;align-items:center;gap:8px}
.status-dot{width:8px;height:8px;border-radius:50%;background:var(--color-success)}
.status-dot--error{background:var(--color-error)}
.dash-header__status-text{font-family:var(--font-mono);font-size:var(--text-sm);color:var(--text-tertiary)}

.dash-body{flex:1;overflow-y:auto;padding:24px;display:flex;flex-direction:column;gap:24px}
.tab-panel{display:none;flex-direction:column;gap:24px;flex:1}
.tab-panel--active{display:flex}

.stats-grid{display:flex;gap:16px}
.stat-card{flex:1;display:flex;flex-direction:column;gap:8px;padding:20px;background:var(--bg-card);border-radius:var(--radius-lg)}
.stat-card__label{font-family:var(--font-mono);font-size:var(--text-xs);font-weight:600;color:var(--text-muted);letter-spacing:1.5px}
.stat-card__value{font-family:var(--font-mono);font-size:var(--text-5xl);font-weight:700;color:var(--text-primary);line-height:1}
.stat-card__value--accent{color:var(--accent-cyan)}
.stat-card__meta{font-family:var(--font-mono);font-size:var(--text-sm);color:var(--text-tertiary)}

.activity-list{background:var(--bg-card);border-radius:var(--radius-lg);overflow:hidden}
.activity-item{display:flex;align-items:start;justify-content:space-between;padding:12px 16px;gap:12px}
.activity-item+.activity-item{border-top:1px solid var(--divider)}
.activity-item__content{flex:1;display:flex;flex-direction:column;gap:4px;min-width:0}
.activity-item__text{font-family:var(--font-sans);font-size:var(--text-base);color:var(--text-primary);line-height:1.4}
.activity-item__meta{font-family:var(--font-mono);font-size:var(--text-xs);color:var(--text-muted)}
.activity-item__copy{flex-shrink:0;color:var(--text-muted);cursor:pointer;padding:4px;border:none;background:none;transition:color var(--transition-fast)}
.activity-item__copy:hover{color:var(--accent-cyan)}

.search-bar{display:flex;align-items:center;gap:10px;padding:12px 16px;background:var(--bg-card);border:1px solid var(--border-subtle);border-radius:var(--radius-lg);width:100%}
.search-bar__input{flex:1;background:none;border:none;outline:none;font-family:var(--font-sans);font-size:var(--text-base);color:var(--text-primary)}
.search-bar__input::placeholder{color:var(--text-muted)}
.search-bar__icon{color:var(--text-muted);flex-shrink:0}
.history-count{font-family:var(--font-mono);font-size:var(--text-sm);color:var(--text-tertiary)}

.vocab-add{display:flex;flex-direction:column;gap:12px}
.vocab-add__row{display:flex;align-items:center;gap:12px}
.vocab-add__input{flex:1;padding:12px 16px;background:var(--bg-card);border:1px solid var(--border-subtle);border-radius:var(--radius-lg);font-family:var(--font-sans);font-size:var(--text-base);color:var(--text-primary);outline:none;transition:border-color var(--transition-fast)}
.vocab-add__input::placeholder{color:var(--text-muted)}
.vocab-add__input:focus{border-color:var(--accent-cyan)}
.vocab-add__arrow{color:var(--text-tertiary);flex-shrink:0}
.vocab-add__btn{display:flex;align-items:center;gap:6px;padding:12px 20px;background:var(--accent-cyan);color:var(--text-inverted);border:none;border-radius:var(--radius-lg);font-family:var(--font-mono);font-size:12px;font-weight:600;cursor:pointer;transition:background var(--transition-fast);white-space:nowrap}
.vocab-add__btn:hover{background:var(--accent-cyan-hover)}

.vocab-header{display:flex;align-items:center;justify-content:space-between}
.vocab-count{font-family:var(--font-mono);font-size:var(--text-sm);color:var(--text-muted)}
.vocab-table{width:100%;background:var(--bg-card);border-radius:var(--radius-lg);overflow:hidden;border-collapse:collapse}
.vocab-table thead{background:var(--bg-inset)}
.vocab-table th{padding:10px 16px;font-family:var(--font-mono);font-size:var(--text-xs);font-weight:600;color:var(--text-muted);letter-spacing:1.5px;text-align:left}
.vocab-table th:last-child{width:40px}
.vocab-table td{padding:12px 16px;font-family:var(--font-mono);font-size:var(--text-base)}
.vocab-table tr+tr{border-top:1px solid var(--divider)}
.vocab-table__from{color:var(--text-primary)}
.vocab-table__to{color:var(--accent-cyan)}
.vocab-table__delete{color:var(--text-muted);cursor:pointer;border:none;background:none;padding:4px;transition:color var(--transition-fast)}
.vocab-table__delete:hover{color:var(--color-error)}

.toast-container{position:fixed;top:16px;right:16px;z-index:9999;display:flex;flex-direction:column;gap:8px;pointer-events:none}
.toast{display:flex;align-items:center;gap:12px;padding:12px 16px;background:var(--bg-card);border:1px solid var(--border-subtle);border-radius:var(--radius-lg);pointer-events:auto;opacity:0;transform:translateX(100%);transition:all var(--transition-base);min-width:320px;max-width:420px}
.toast--visible{opacity:1;transform:translateX(0)}
.toast__icon{flex-shrink:0;font-size:18px}
.toast--success .toast__icon{color:var(--color-success)}
.toast--warning .toast__icon{color:var(--color-warning)}
.toast--error .toast__icon{color:var(--color-error)}
.toast--info .toast__icon{color:var(--accent-cyan)}
.toast__content{display:flex;flex-direction:column;gap:2px}
.toast__title{font-family:var(--font-sans);font-size:var(--text-base);font-weight:500;color:var(--text-primary)}
.toast__message{font-family:var(--font-mono);font-size:var(--text-sm);color:var(--text-tertiary)}
.empty{color:var(--text-muted);text-align:center;padding:40px;font-size:var(--text-base)}
</style>
</head>
<body>
<div class="dashboard">
  <header class="dash-header">
    <div class="dash-header__brand">
      <div class="dash-header__logo"><i class="lucide-audio-waveform"></i></div>
      <span class="dash-header__title">Cognitive Flow</span>
    </div>
    <nav class="dash-tabs" role="tablist">
      <button class="dash-tabs__tab dash-tabs__tab--active" role="tab" aria-selected="true" data-tab="dashboard">Dashboard</button>
      <button class="dash-tabs__tab" role="tab" aria-selected="false" data-tab="history">History</button>
      <button class="dash-tabs__tab" role="tab" aria-selected="false" data-tab="vocabulary">Vocabulary</button>
    </nav>
    <div class="dash-header__status">
      <span class="status-dot" id="status-dot"></span>
      <span class="dash-header__status-text" id="status-text">Server connected</span>
    </div>
  </header>

  <main class="dash-body">
    <section class="tab-panel tab-panel--active" id="panel-dashboard" role="tabpanel">
      <span class="section-label">OVERVIEW</span>
      <div class="stats-grid">
        <div class="stat-card">
          <span class="stat-card__label">TOTAL TRANSCRIPTIONS</span>
          <span class="stat-card__value stat-card__value--accent" id="stat-total">0</span>
          <span class="stat-card__meta" id="stat-today">+0 today</span>
        </div>
        <div class="stat-card">
          <span class="stat-card__label">MINUTES RECORDED</span>
          <span class="stat-card__value" id="stat-minutes">0</span>
          <span class="stat-card__meta" id="stat-avg-duration">--</span>
        </div>
        <div class="stat-card">
          <span class="stat-card__label">AVG SERVER TIME</span>
          <span class="stat-card__value" id="stat-server">--</span>
          <span class="stat-card__meta" id="stat-p95">--</span>
        </div>
      </div>
      <div class="stats-grid">
        <div class="stat-card">
          <span class="stat-card__label">UPTIME</span>
          <span class="stat-card__value" id="stat-uptime">0m</span>
          <span class="stat-card__meta" id="stat-started">--</span>
        </div>
        <div class="stat-card">
          <span class="stat-card__label">SESSION COUNT</span>
          <span class="stat-card__value" id="stat-sessions">0</span>
          <span class="stat-card__meta">this session</span>
        </div>
        <div class="stat-card">
          <span class="stat-card__label">WORD CORRECTIONS</span>
          <span class="stat-card__value" id="stat-corrections">0</span>
          <span class="stat-card__meta">active replacements</span>
        </div>
      </div>
      <span class="section-label">RECENT ACTIVITY</span>
      <div class="activity-list" id="recent-activity"></div>
    </section>

    <section class="tab-panel" id="panel-history" role="tabpanel">
      <div class="search-bar">
        <i class="search-bar__icon lucide-search"></i>
        <input type="text" class="search-bar__input" id="history-search" placeholder="Search transcriptions..." autocomplete="off">
      </div>
      <span class="history-count" id="history-count"></span>
      <div class="activity-list" id="history-list"></div>
    </section>

    <section class="tab-panel" id="panel-vocabulary" role="tabpanel">
      <span class="section-label">ADD CORRECTION</span>
      <div class="vocab-add">
        <div class="vocab-add__row">
          <input type="text" class="vocab-add__input" id="vocab-from" placeholder="Misheard word..." autocomplete="off">
          <i class="vocab-add__arrow lucide-arrow-right"></i>
          <input type="text" class="vocab-add__input" id="vocab-to" placeholder="Correct word..." autocomplete="off">
          <button class="vocab-add__btn" id="vocab-add-btn"><i class="lucide-plus"></i> <span>Add</span></button>
        </div>
      </div>
      <div class="vocab-header">
        <span class="section-label">ACTIVE CORRECTIONS</span>
        <span class="vocab-count" id="vocab-count">0 entries</span>
      </div>
      <table class="vocab-table" id="vocab-table">
        <thead><tr><th>FROM</th><th>TO</th><th></th></tr></thead>
        <tbody id="vocab-tbody"></tbody>
      </table>
    </section>
  </main>
</div>
<div class="toast-container" id="toast-container"></div>

<script>
(function(){
'use strict';
var stats = /**STATS**/null;
var historyData = /**HISTORY**/[];
var vocabData = /**VOCAB**/{};

// ---- Helpers ----
function setText(id, v){ var el=document.getElementById(id); if(el) el.textContent=v; }
function fmtTime(ts){
  try{ var d=new Date(ts); return d.toLocaleDateString('en-US',{month:'short',day:'numeric'})+' '+d.toLocaleTimeString('en-US',{hour:'numeric',minute:'2-digit'}); }
  catch(e){ return ts; }
}

// ---- Stats ----
function renderStats(s){
  if(!s) return;
  setText('stat-total', s.total.toLocaleString());
  setText('stat-today', '+'+s.today+' today');
  setText('stat-minutes', s.minutes.toFixed(1));
  setText('stat-avg-duration', s.avg_duration > 0 ? '~'+s.avg_duration.toFixed(1)+'s avg duration' : '--');
  setText('stat-server', s.avg_server > 0 ? s.avg_server+'ms' : '--');
  setText('stat-p95', s.p95_server > 0 ? 'p95: '+s.p95_server+'ms' : '--');
  setText('stat-uptime', s.uptime);
  setText('stat-started', 'started '+s.started);
  setText('stat-sessions', s.sessions.toString());
  setText('stat-corrections', s.corrections.toString());
}
renderStats(stats);

// ---- Activity Items (createElement, no innerHTML) ----
function createActivityItem(entry){
  var div=document.createElement('div');
  div.className='activity-item';
  div.setAttribute('data-transcription','');

  var content=document.createElement('div');
  content.className='activity-item__content';

  var p=document.createElement('p');
  p.className='activity-item__text';
  p.textContent=entry.text;

  var meta=document.createElement('span');
  meta.className='activity-item__meta';
  meta.textContent=fmtTime(entry.timestamp)+'  |  '+(entry.total_ms||0)+'ms';

  content.appendChild(p);
  content.appendChild(meta);
  div.appendChild(content);

  var btn=document.createElement('button');
  btn.className='activity-item__copy';
  btn.title='Copy to clipboard';
  btn.setAttribute('data-copy','');
  var icon=document.createElement('i');
  icon.className='lucide-copy';
  btn.appendChild(icon);
  div.appendChild(btn);

  return div;
}

// ---- Recent Activity (last 10) ----
var recentList=document.getElementById('recent-activity');
function renderRecent(){
  while(recentList.firstChild) recentList.removeChild(recentList.firstChild);
  var items=historyData.slice(0,10);
  if(items.length===0){
    var empty=document.createElement('div');
    empty.className='empty';
    empty.textContent='No transcriptions yet';
    recentList.appendChild(empty);
  } else {
    items.forEach(function(e){ recentList.appendChild(createActivityItem(e)); });
  }
}
renderRecent();

// ---- History ----
var histList=document.getElementById('history-list');
var histCount=document.getElementById('history-count');
var histSearch=document.getElementById('history-search');

function renderHistory(filter){
  while(histList.firstChild) histList.removeChild(histList.firstChild);
  var q=(filter||'').toLowerCase();
  var visible=0;
  historyData.forEach(function(e){
    if(q && e.text.toLowerCase().indexOf(q)===-1) return;
    histList.appendChild(createActivityItem(e));
    visible++;
  });
  if(visible===0){
    var empty=document.createElement('div');
    empty.className='empty';
    empty.textContent=q?'No results for "'+filter+'"':'No transcriptions yet';
    histList.appendChild(empty);
  }
  histCount.textContent=q
    ? 'Showing '+visible+' result'+(visible!==1?'s':'')
    : 'Showing '+historyData.length+' transcriptions';
}
renderHistory();

if(histSearch){
  histSearch.addEventListener('input',function(){ renderHistory(this.value.trim()); });
}

// ---- Copy to Clipboard ----
document.addEventListener('click',function(e){
  var btn=e.target.closest('[data-copy]');
  if(!btn) return;
  var item=btn.closest('.activity-item');
  if(!item) return;
  var text=item.querySelector('.activity-item__text');
  if(!text) return;
  navigator.clipboard.writeText(text.textContent.trim()).then(function(){
    showToast('success','Copied to clipboard',text.textContent.trim().slice(0,50)+'...');
  });
});

// ---- Vocabulary ----
var vocabFrom=document.getElementById('vocab-from');
var vocabTo=document.getElementById('vocab-to');
var vocabAddBtn=document.getElementById('vocab-add-btn');
var vocabTbody=document.getElementById('vocab-tbody');
var vocabCountEl=document.getElementById('vocab-count');

function createVocabRow(from, to){
  var tr=document.createElement('tr');
  var td1=document.createElement('td');
  td1.className='vocab-table__from';
  td1.textContent=from;
  var td2=document.createElement('td');
  td2.className='vocab-table__to';
  td2.textContent=to;
  var td3=document.createElement('td');
  var btn=document.createElement('button');
  btn.className='vocab-table__delete';
  btn.title='Remove correction';
  btn.setAttribute('data-delete','');
  btn.setAttribute('data-from',from);
  var icon=document.createElement('i');
  icon.className='lucide-x';
  btn.appendChild(icon);
  td3.appendChild(btn);
  tr.appendChild(td1);
  tr.appendChild(td2);
  tr.appendChild(td3);
  return tr;
}

function renderVocab(data){
  while(vocabTbody.firstChild) vocabTbody.removeChild(vocabTbody.firstChild);
  var keys=Object.keys(data);
  keys.forEach(function(from){ vocabTbody.appendChild(createVocabRow(from, data[from])); });
  vocabCountEl.textContent=keys.length+' entr'+(keys.length===1?'y':'ies');
}
renderVocab(vocabData);

if(vocabAddBtn){
  vocabAddBtn.addEventListener('click',function(){
    var from=vocabFrom.value.trim();
    var to=vocabTo.value.trim();
    if(!from||!to){ showToast('warning','Missing fields','Both fields are required'); return; }
    fetch('/api/vocab/add',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({from:from,to:to})})
      .then(function(r){return r.json()})
      .then(function(d){ vocabData=d; renderVocab(d); vocabFrom.value=''; vocabTo.value=''; vocabFrom.focus(); showToast('success','Correction added',from+' \u2192 '+to); });
  });
}

[vocabFrom,vocabTo].forEach(function(input){
  if(!input) return;
  input.addEventListener('keydown',function(e){ if(e.key==='Enter'){e.preventDefault();vocabAddBtn.click();} });
});

document.addEventListener('click',function(e){
  var btn=e.target.closest('[data-delete]');
  if(!btn) return;
  var from=btn.getAttribute('data-from');
  if(!from) return;
  fetch('/api/vocab/delete',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({from:from})})
    .then(function(r){return r.json()})
    .then(function(d){ vocabData=d; renderVocab(d); showToast('success','Correction removed',from); });
});

// ---- Tabs ----
var tabs=document.querySelectorAll('.dash-tabs__tab');
var panels=document.querySelectorAll('.tab-panel');
tabs.forEach(function(tab){
  tab.addEventListener('click',function(){
    var target=tab.dataset.tab;
    tabs.forEach(function(t){ t.classList.remove('dash-tabs__tab--active'); t.setAttribute('aria-selected','false'); });
    tab.classList.add('dash-tabs__tab--active');
    tab.setAttribute('aria-selected','true');
    panels.forEach(function(p){ p.classList.remove('tab-panel--active'); });
    var panel=document.getElementById('panel-'+target);
    if(panel) panel.classList.add('tab-panel--active');
    if(target==='history' && histSearch) histSearch.focus();
  });
});

// Hash fragment navigation
if(location.hash){
  var h=location.hash.slice(1);
  if(h==='vocab') h='vocabulary';
  var el=document.querySelector('.dash-tabs__tab[data-tab="'+h+'"]');
  if(el) el.click();
}

// ---- Toast Notifications ----
var toastContainer=document.getElementById('toast-container');
var toastIcons={success:'lucide-circle-check',warning:'lucide-triangle-alert',error:'lucide-circle-x',info:'lucide-download'};

function showToast(type, title, message, duration){
  if(!toastContainer) return;
  duration=duration||3000;

  var toast=document.createElement('div');
  toast.className='toast toast--'+type;

  var icon=document.createElement('i');
  icon.className='toast__icon '+(toastIcons[type]||toastIcons.info);
  toast.appendChild(icon);

  var content=document.createElement('div');
  content.className='toast__content';
  var titleEl=document.createElement('span');
  titleEl.className='toast__title';
  titleEl.textContent=title;
  content.appendChild(titleEl);
  var msgEl=document.createElement('span');
  msgEl.className='toast__message';
  msgEl.textContent=message;
  content.appendChild(msgEl);
  toast.appendChild(content);

  toastContainer.appendChild(toast);
  requestAnimationFrame(function(){ toast.classList.add('toast--visible'); });
  setTimeout(function(){
    toast.classList.remove('toast--visible');
    toast.addEventListener('transitionend',function(){ toast.remove(); },{once:true});
  },duration);
}

// ---- Server Status Check ----
function checkServer(){
  fetch('/api/stats').then(function(r){
    var dot=document.getElementById('status-dot');
    var text=document.getElementById('status-text');
    if(r.ok){ dot.classList.remove('status-dot--error'); text.textContent='Server connected'; }
    else { dot.classList.add('status-dot--error'); text.textContent='Server error'; }
  }).catch(function(){
    var dot=document.getElementById('status-dot');
    var text=document.getElementById('status-text');
    dot.classList.add('status-dot--error');
    text.textContent='Disconnected';
  });
}
setInterval(checkServer,30000);

})();
</script>
</body>
</html>`

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

	delays := [3]time.Duration{2 * time.Second, 4 * time.Second, 8 * time.Second}
	var lastErr error

	for attempt := 0; attempt <= 3; attempt++ {
		if attempt > 0 {
			log("Retry %d/3 in %v", attempt, delays[attempt-1])
			time.Sleep(delays[attempt-1])
		}

		c := &http.Client{Timeout: 10 * time.Second}
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

	state.lastOutput = text
	// Track in recent ring buffer
	idx := state.recentCount % len(state.recentTexts)
	state.recentTexts[idx] = text
	state.recentCount++

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
				color = 0x00775A11 // Cyan dim
			case level < 50:
				color = 0x00BB9317 // Cyan medium
			default:
				color = 0x00EED322 // Cyan full (#22D3EE)
			}
		case phaseProcessing:
			color = 0x0008B3EA // Amber (#EAB308)
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
			wakeGap := now.Sub(state.lastWake) > 60*time.Second
			if wakeGap {
				log("Wake detected, pinging server")
				go healthCheck()
			}
			state.lastWake = now

			// Self-heal indicator: recover from sleep/compositor hiding
			recoverIndicator(wakeGap)
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
// Software-rendered overlay indicator with 4 visual states:
//   Idle collapsed:  16px cyan dot at 40% opacity
//   Idle expanded:   56px dark circle with cyan border + mic icon
//   Recording:       80px pulsing ring + 64px cyan core + mic icon
//   Processing:      56px dark circle with cyan border + spinning dots
// Uses UpdateLayeredWindow for true per-pixel transparency.

const (
	indSize       = 96            // window pixel buffer (fits 80px recording + pulse margin)
	indCenter     = 48            // center of the window
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
	dragging  bool
	collapsed bool      // true after 3s idle - subtle dot, no glow
	hovered   bool      // mouse is over the indicator
	tracking  bool      // TrackMouseEvent registered
	collapseAt time.Time // when to collapse (zero = don't)
	fadeLevel  float64   // 0.0 = collapsed, 1.0 = expanded (smooth transition)
}

type indColor struct{ r, g, b float64 }

var (
	colCyan    = indColor{34, 211, 238}  // #22D3EE
	colBgCard  = indColor{30, 41, 59}    // #1E293B - dark circle fill
	colDark    = indColor{10, 15, 28}    // #0A0F1C - icon on cyan background
	colGray    = indColor{120, 120, 120}
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
		// Register for WM_MOUSELEAVE if not already tracking
		if !ind.tracking {
			tme := [4]uintptr{
				unsafe.Sizeof([4]uintptr{}), // cbSize
				0x00000002,                  // TME_LEAVE
				hwnd,                        // hwndTrack
				0,                           // dwHoverTime (unused)
			}
			pTrackMouseEvent.Call(uintptr(unsafe.Pointer(&tme)))
			ind.tracking = true
		}
		if !ind.hovered {
			ind.hovered = true
			// Speed up timer for smooth fade-in animation (idle runs at 1000ms, too slow)
			pSetTimer.Call(hwnd, timerIndAnim, 66, 0)
		}

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

	case 0x02A3: // WM_MOUSELEAVE
		ind.hovered = false
		ind.tracking = false
		// Keep fast timer briefly for fade-out, then applyPhase will reset to 1000ms on idle
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
			// Collapse after 3s idle
			p := currentPhase()
			if p == phaseIdle && !ind.collapseAt.IsZero() && time.Now().After(ind.collapseAt) {
				ind.collapsed = true
				ind.collapseAt = time.Time{} // clear
				// Speed up timer for smooth fade-out animation
				pSetTimer.Call(hwnd, timerIndAnim, 66, 0)
			}

			// Smooth fade: target 1.0 when expanded/hovered, 0.0 when collapsed
			target := 0.0
			if !ind.collapsed || ind.hovered || p != phaseIdle {
				target = 1.0
			}
			animating := false
			if ind.fadeLevel < target {
				ind.fadeLevel += 0.25 // ~4 frames to full
				if ind.fadeLevel > 1.0 {
					ind.fadeLevel = 1.0
				}
				animating = ind.fadeLevel < target
			} else if ind.fadeLevel > target {
				ind.fadeLevel -= 0.04 // ~25 frames to collapse (slower fade out)
				if ind.fadeLevel < 0.0 {
					ind.fadeLevel = 0.0
				}
				animating = ind.fadeLevel > target
			}

			// Slow timer back to 1s when done animating and idle
			if !animating && p == phaseIdle && !ind.hovered {
				pSetTimer.Call(hwnd, timerIndAnim, 1000, 0)
			}

			// Light topmost re-assertion every ~10 frames (10s at 1s/tick idle)
			// DWM silently drops layered windows from composition
			if ind.frame%10 == 0 && state.indVisible {
				pSetWindowPos.Call(hwnd, ^uintptr(0), 0, 0, 0, 0, 0x0001|0x0002|0x0010)
			}

			renderIndicator()
		}
		return 0

	case 0x007E: // WM_DISPLAYCHANGE - display resolution/depth changed (sleep/wake, dock, etc.)
		log("WM_DISPLAYCHANGE: rebuilding indicator GDI resources")
		rebuildIndicatorGDI()
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
	// Position so the visual center lands where the old 56px indicator's center was
	x := int(sw) - indSize - 4
	y := int(sh) - indSize - 44

	hwnd, _, _ := pCreateWindowEx.Call(
		wsExLayered|wsExTopmost|wsExToolWindow|wsExNoActivate,
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
	ind.fadeLevel = 1.0
	ind.collapseAt = time.Now().Add(3 * time.Second) // collapse 3s after startup
	renderIndicator()
	pShowWindow.Call(hwnd, 8) // SW_SHOWNA
	state.indVisible = true

	// Slow heartbeat timer when idle (redraws on state changes via applyPhase)
	pSetTimer.Call(hwnd, timerIndAnim, 1000, 0)
}

// rebuildIndicatorGDI recreates the memory DC and DIB from a fresh screen DC.
// Called on WM_DISPLAYCHANGE when the display adapter resets (sleep/wake, dock/undock).
// Without this, UpdateLayeredWindow silently fails because the old DIB is stale.
func rebuildIndicatorGDI() {
	// Clean up old GDI objects
	if ind.dib != 0 {
		pDeleteObject.Call(ind.dib)
	}
	if ind.memDC != 0 {
		pDeleteDC.Call(ind.memDC)
	}

	// Create fresh from current screen DC
	screenDC, _, _ := pGetDC.Call(0)
	ind.memDC, _, _ = pCreateCompatibleDC.Call(screenDC)
	pReleaseDC.Call(0, screenDC)

	bmi := bmpinfo{
		Size:   uint32(unsafe.Sizeof(bmpinfo{})),
		Width:  indSize,
		Height: -indSize,
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

	// Re-render with fresh resources
	renderIndicator()

	// Re-assert topmost in case z-order got scrambled
	pSetWindowPos.Call(ind.hwnd, ^uintptr(0), 0, 0, 0, 0, 0x0001|0x0002|0x0010)
}

// recoverIndicator unconditionally re-asserts the indicator window every heartbeat.
// DWM can silently drop layered windows from composition - IsWindowVisible still
// returns true but the window is gone from screen. The fix is the same one every
// persistent Windows overlay uses: periodic re-assertion with a position nudge to
// force the compositor to re-register the surface.
func recoverIndicator(wakeGap bool) {
	if !state.indVisible {
		return // user explicitly hid it via tray menu
	}
	h := ind.hwnd
	if h == 0 {
		return
	}

	// Rebuild GDI on wake (stale DC/DIB from display adapter reset)
	if wakeGap {
		log("Wake gap, rebuilding indicator GDI")
		rebuildIndicatorGDI()
	}

	// Always re-assert: show + topmost + nudge
	pShowWindow.Call(h, 8) // SW_SHOWNA

	var rect [4]int32
	pGetWindowRect.Call(h, uintptr(unsafe.Pointer(&rect)))
	ix, iy := int(rect[0]), int(rect[1])

	// Check if off-screen (resolution/DPI change)
	sw, _, _ := pGetSystemMetrics.Call(0)
	sh, _, _ := pGetSystemMetrics.Call(1)
	if ix > int(sw)-10 || iy > int(sh)-10 || ix < -indSize || iy < -indSize {
		ix = int(sw) - indSize - 24
		iy = int(sh) - indSize - 64
		log("Indicator repositioned to (%d, %d)", ix, iy)
	}

	// SetWindowPos to re-assert TOPMOST
	pSetWindowPos.Call(h, ^uintptr(0), uintptr(ix), uintptr(iy), indSize, indSize, 0x0010) // SWP_NOACTIVATE only

	// Nudge 1px and back - forces DWM to re-register the compositor surface
	pSetWindowPos.Call(h, ^uintptr(0), uintptr(ix+1), uintptr(iy), indSize, indSize, 0x0010)
	pSetWindowPos.Call(h, ^uintptr(0), uintptr(ix), uintptr(iy), indSize, indSize, 0x0010)

	// Re-render to push fresh pixels
	renderIndicator()
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
	cx := float64(indCenter)
	cy := float64(indCenter)

	// Clear to fully transparent
	for i := range px {
		px[i] = 0
	}

	p := currentPhase()
	fl := ind.fadeLevel

	// ---- Collapsed idle: tiny cyan dot ----
	if p == phaseIdle && fl < 0.05 && !ind.hovered {
		dotR := 8.0 // 16px diameter
		for idx := 0; idx < w*w; idx++ {
			dist := ind.dist[idx]
			if dist > dotR+1 {
				continue
			}
			a := 0.4 * aaEdge(dotR, dist)
			if a > 0.004 {
				i := idx * 4
				px[i+0] = byte(colCyan.b * a)
				px[i+1] = byte(colCyan.g * a)
				px[i+2] = byte(colCyan.r * a)
				px[i+3] = byte(255 * a)
			}
		}
		updateIndicatorWindow()
		return
	}

	// ---- Recording: pulse ring + cyan core + mic icon ----
	if p == phaseRecording {
		level := float64(audioLevel.Load()) / 100.0
		pulse := 0.65 + 0.35*math.Sin(float64(ind.frame)*0.10)

		// Pulse ring: 80px outer, breathing
		pulseR := 40.0 + 2.0*math.Sin(float64(ind.frame)*0.10)
		pulseAlpha := 0.12 + 0.08*math.Sin(float64(ind.frame)*0.10) + level*0.1
		coreR := 32.0 // 64px diameter inner core
		_ = pulse

		for idx := 0; idx < w*w; idx++ {
			dist := ind.dist[idx]
			if dist > pulseR+1 {
				continue
			}

			py := idx / w
			ppx := idx % w
			dx := float64(ppx) - cx + 0.5
			dy := float64(py) - cy + 0.5

			var r, g, b, a float64

			// Layer 1: Pulse ring (between core and outer)
			if dist > coreR+0.5 && dist < pulseR+1 {
				t := (dist - coreR) / (pulseR - coreR)
				ringA := pulseAlpha * (1 - t*t) * aaEdge(pulseR, dist)
				r = colCyan.r / 255 * ringA
				g = colCyan.g / 255 * ringA
				b = colCyan.b / 255 * ringA
				a = ringA
			}

			// Layer 2: Solid cyan core
			if dist < coreR+0.5 {
				coreA := aaEdge(coreR, dist)
				compositeOver(&r, &g, &b, &a, colCyan.r/255, colCyan.g/255, colCyan.b/255, coreA)
			}

			// Layer 3: Mic icon (dark on cyan)
			micA := micIconAlpha(dx, dy, 1.3)
			if micA > 0.01 {
				compositeOver(&r, &g, &b, &a, colDark.r/255, colDark.g/255, colDark.b/255, micA)
			}

			if a > 0.004 {
				i := idx * 4
				px[i+0] = clampByte(b * 255)
				px[i+1] = clampByte(g * 255)
				px[i+2] = clampByte(r * 255)
				px[i+3] = clampByte(a * 255)
			}
		}
		updateIndicatorWindow()
		return
	}

	// ---- Processing: dark circle + cyan border + spinning dots ----
	if p == phaseProcessing {
		circleR := 28.0 // 56px diameter
		borderW := 3.0

		for idx := 0; idx < w*w; idx++ {
			dist := ind.dist[idx]
			if dist > circleR+1 {
				continue
			}

			py := idx / w
			ppx := idx % w
			dx := float64(ppx) - cx + 0.5
			dy := float64(py) - cy + 0.5

			var r, g, b, a float64

			// Dark fill
			fillR := circleR - borderW
			if dist < fillR+0.5 {
				fillA := aaEdge(fillR, dist)
				r = colBgCard.r / 255 * fillA
				g = colBgCard.g / 255 * fillA
				b = colBgCard.b / 255 * fillA
				a = fillA
			}

			// Cyan border
			if dist > fillR-0.5 && dist < circleR+0.5 {
				outerAA := aaEdge(circleR, dist)
				innerAA := 1.0 - aaEdge(fillR, dist)
				bdrA := outerAA * innerAA
				if bdrA > 0.01 {
					compositeOver(&r, &g, &b, &a, colCyan.r/255, colCyan.g/255, colCyan.b/255, bdrA)
				}
			}

			// Spinner dots
			spinA := spinnerAlpha(dx, dy, 1.0, ind.frame)
			if spinA > 0.01 {
				compositeOver(&r, &g, &b, &a, colCyan.r/255, colCyan.g/255, colCyan.b/255, spinA)
			}

			if a > 0.004 {
				i := idx * 4
				px[i+0] = clampByte(b * 255)
				px[i+1] = clampByte(g * 255)
				px[i+2] = clampByte(r * 255)
				px[i+3] = clampByte(a * 255)
			}
		}
		updateIndicatorWindow()
		return
	}

	// ---- Idle expanded (or transitioning): dark circle + cyan border + mic ----
	// fl interpolates from collapsed dot (0) to full expanded circle (1)
	circleR := 8.0 + 20.0*fl // 8 (dot) -> 28 (56px circle)
	borderW := 2.0 * fl
	fillAlpha := fl // dark fill fades in
	borderAlpha := fl
	iconAlpha := math.Max(0, (fl-0.3)/0.7) // icon fades in during last 70% of transition

	// Disabled state: gray dot
	if !state.enabled {
		circleR = 8.0
		borderW = 0
		fillAlpha = 0
		iconAlpha = 0
	}

	for idx := 0; idx < w*w; idx++ {
		dist := ind.dist[idx]
		if dist > circleR+1 {
			continue
		}

		py := idx / w
		ppx := idx % w
		dx := float64(ppx) - cx + 0.5
		dy := float64(py) - cy + 0.5

		var r, g, b, a float64

		fillR := circleR - borderW
		if fillR < 0 {
			fillR = 0
		}

		// During transition: blend from cyan dot to dark filled circle
		if fillAlpha < 0.5 {
			// Mostly collapsed: draw cyan dot
			dotA := (0.4 + 0.6*fl) * aaEdge(circleR, dist)
			r = colCyan.r / 255 * dotA
			g = colCyan.g / 255 * dotA
			b = colCyan.b / 255 * dotA
			a = dotA
		} else {
			// Dark fill
			if dist < fillR+0.5 {
				fA := fillAlpha * aaEdge(fillR, dist)
				r = colBgCard.r / 255 * fA
				g = colBgCard.g / 255 * fA
				b = colBgCard.b / 255 * fA
				a = fA
			}
			// Cyan border
			if borderW > 0.5 && dist > fillR-0.5 && dist < circleR+0.5 {
				outerAA := aaEdge(circleR, dist)
				innerAA := 1.0 - aaEdge(fillR, dist)
				bdrA := borderAlpha * outerAA * innerAA
				if bdrA > 0.01 {
					compositeOver(&r, &g, &b, &a, colCyan.r/255, colCyan.g/255, colCyan.b/255, bdrA)
				}
			}
		}

		// Mic icon
		if iconAlpha > 0.01 {
			micA := micIconAlpha(dx, dy, 1.0) * iconAlpha
			if micA > 0.01 {
				compositeOver(&r, &g, &b, &a, colCyan.r/255, colCyan.g/255, colCyan.b/255, micA)
			}
		}

		if a > 0.004 {
			i := idx * 4
			px[i+0] = clampByte(b * 255)
			px[i+1] = clampByte(g * 255)
			px[i+2] = clampByte(r * 255)
			px[i+3] = clampByte(a * 255)
		}
	}
	updateIndicatorWindow()
}

func updateIndicatorWindow() {
	srcPt := [2]int32{0, 0}
	sz := [2]int32{int32(indSize), int32(indSize)}
	blend := uint32(0) | uint32(0)<<8 | uint32(255)<<16 | uint32(1)<<24
	pUpdateLayeredWindow.Call(
		ind.hwnd, 0, 0, uintptr(unsafe.Pointer(&sz)),
		ind.memDC, uintptr(unsafe.Pointer(&srcPt)),
		0, uintptr(unsafe.Pointer(&blend)), ulwAlpha,
	)
	ind.frame++
}

// aaEdge returns 0-1 anti-aliased edge alpha. 1.0 inside, 0.0 outside, smooth at boundary.
func aaEdge(radius, dist float64) float64 {
	if dist < radius-0.5 {
		return 1.0
	}
	if dist > radius+0.5 {
		return 0.0
	}
	return radius + 0.5 - dist
}

// compositeOver does premultiplied-alpha src-over compositing in-place.
func compositeOver(dR, dG, dB, dA *float64, sR, sG, sB, sA float64) {
	pr := sR * sA
	pg := sG * sA
	pb := sB * sA
	inv := 1.0 - sA
	*dR = pr + *dR*inv
	*dG = pg + *dG*inv
	*dB = pb + *dB*inv
	*dA = sA + *dA*inv
}

func clampByte(v float64) byte {
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return byte(v)
}

// micIconAlpha returns 0-1 alpha for drawing a mic icon at (dx, dy) from icon center.
// scale 1.0 = 22px reference, 1.3 = 28px (recording state).
func micIconAlpha(dx, dy, scale float64) float64 {
	// Normalize to 22px reference space
	x := dx / scale
	y := dy / scale

	// Mic capsule (stadium/pill shape) - centered, upper portion
	capCY := -3.0  // center offset (shifted up)
	capHW := 2.8   // half-width
	capHH := 5.0   // half-height
	capR := capHW   // corner radius = width (fully rounded)

	cdy := y - capCY
	straight := capHH - capR
	clamped := math.Max(-straight, math.Min(straight, cdy))
	d := math.Sqrt(x*x + (cdy-clamped)*(cdy-clamped))
	if d < capR+0.7 {
		return smoothstep(capR+0.7, capR-0.3, d)
	}

	// U-arc (cradle) - semicircular stroke below capsule
	arcCY := capCY + capHH + 1.5
	arcR := 4.5
	arcThick := 1.4
	arcDist := math.Sqrt(x*x + (y-arcCY)*(y-arcCY))
	ringDist := math.Abs(arcDist - arcR)
	if y > arcCY-0.5 && ringDist < arcThick/2+0.7 {
		return smoothstep(arcThick/2+0.7, arcThick/2-0.3, ringDist)
	}

	// Stem - vertical line below arc
	stemTop := arcCY + arcR
	stemBot := stemTop + 2.5
	stemHW := 0.7
	if y > stemTop-0.3 && y < stemBot+0.5 {
		xd := math.Max(0, math.Abs(x)-stemHW)
		yd := math.Max(0, math.Max(stemTop-y, y-stemBot))
		d := math.Sqrt(xd*xd + yd*yd)
		if d < 0.7 {
			return smoothstep(0.7, 0, d)
		}
	}

	// Base - horizontal line at bottom
	baseCY := stemBot
	baseHW := 3.0
	baseHH := 0.7
	if math.Abs(y-baseCY) < baseHH+0.7 && math.Abs(x) < baseHW+0.7 {
		xd := math.Max(0, math.Abs(x)-baseHW)
		yd := math.Max(0, math.Abs(y-baseCY)-baseHH)
		d := math.Sqrt(xd*xd + yd*yd)
		if d < 0.7 {
			return smoothstep(0.7, 0, d)
		}
	}

	return 0
}

// spinnerAlpha returns 0-1 alpha for a spinning dots loader icon.
// 8 dots in a circle, varying opacity to create rotation effect.
func spinnerAlpha(dx, dy, scale float64, frame int) float64 {
	dotR := 1.8 * scale   // dot radius
	ringR := 8.0 * scale  // ring radius
	ndots := 8
	rotAngle := float64(frame) * 0.15

	for i := 0; i < ndots; i++ {
		a := float64(i)*2*math.Pi/float64(ndots) + rotAngle
		dotCX := ringR * math.Cos(a)
		dotCY := ringR * math.Sin(a)

		ddx := dx - dotCX
		ddy := dy - dotCY
		d := math.Sqrt(ddx*ddx + ddy*ddy)

		if d < dotR+0.7 {
			opacity := 0.15 + 0.85*float64(i)/float64(ndots-1) // trailing=dim, leading=bright
			return smoothstep(dotR+0.7, dotR-0.3, d) * opacity
		}
	}
	return 0
}

// smoothstep returns 0-1 with smooth interpolation between edge0 and edge1.
func smoothstep(edge0, edge1, x float64) float64 {
	t := (x - edge0) / (edge1 - edge0)
	if t < 0 {
		t = 0
	}
	if t > 1 {
		t = 1
	}
	return t * t * (3 - 2*t)
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
		ind.collapsed = false
		ind.collapseAt = time.Time{}
		ind.fadeLevel = 1.0
		pShowWindow.Call(state.bar, 8) // SW_SHOWNA
		pInvalidateRect.Call(state.bar, 0, 1)
		pSetWindowPos.Call(state.bar, ^uintptr(0), 0, 0, 0, 0, 0x0001|0x0002|0x0010)
		pSetTimer.Call(state.bar, timerRepaint, 200, 0)
		// 66ms indicator animation (15fps) during recording
		pSetTimer.Call(ind.hwnd, timerIndAnim, 66, 0)
	case phaseProcessing:
		pKillTimer.Call(state.bar, timerRepaint)
		audioLevel.Store(0)
		ind.collapsed = false
		ind.collapseAt = time.Time{}
		ind.fadeLevel = 1.0
		pShowWindow.Call(state.bar, 8)
		pInvalidateRect.Call(state.bar, 0, 1)
		pSetWindowPos.Call(state.bar, ^uintptr(0), 0, 0, 0, 0, 0x0001|0x0002|0x0010)
		// Slower animation during processing (15fps)
		pSetTimer.Call(ind.hwnd, timerIndAnim, 66, 0)
	default:
		pKillTimer.Call(state.bar, timerRepaint)
		audioLevel.Store(0)
		// Start 3s collapse countdown
		ind.collapseAt = time.Now().Add(3 * time.Second)
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
	menuCopyLast   = 5
	menuUpdate     = 6
	menuRetryLast  = 7
	menuOpenLog    = 8
	menuOpenAudio  = 9
	menuOpenConfig = 10
	menuRecent0    = 100 // 100-104 for recent transcriptions
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
		case menuCopyLast:
			if state.lastOutput != "" {
				copyToClipboard(state.lastOutput)
				log("Copied last output to clipboard")
			}
		case menuRetryLast:
			if len(state.lastSamples) > 0 {
				log("Retrying last recording (%d samples)", len(state.lastSamples))
				setPhase(phaseProcessing)
				go transcribe(state.lastSamples, false)
			}
		case menuOpenLog:
			openDashboard("")
		case menuOpenConfig:
			openDashboard("#vocab")
		case menuOpenAudio:
			p, _ := syscall.UTF16PtrFromString(filepath.Join(configDir(), "audio"))
			pShellExecute.Call(0, 0, uintptr(unsafe.Pointer(p)), 0, 0, 1)
		case menuUpdate:
			go downloadUpdate()
		case menuQuit:
			shutdown()
			pPostQuitMessage.Call(0)
		default:
			// Recent transcription submenu items (100-104)
			cmd := uint16(wp)
			if cmd >= menuRecent0 && cmd < menuRecent0+5 {
				ri := int(cmd - menuRecent0)
				if ri < state.recentCount && ri < len(state.recentTexts) {
					// Map menu index to ring buffer: most recent first
					bufIdx := (state.recentCount - 1 - ri) % len(state.recentTexts)
					if state.recentTexts[bufIdx] != "" {
						copyToClipboard(state.recentTexts[bufIdx])
						log("Copied recent #%d to clipboard", ri+1)
					}
				}
			}
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

	state.trayIcon = makeIcon(34, 211, 238) // Cyan (#22D3EE)
	state.trayNID.Icon = state.trayIcon

	pShellNotifyIcon.Call(nimAdd, uintptr(unsafe.Pointer(&state.trayNID)))
}

func showMenu(hwnd uintptr) {
	h, _, _ := pCreatePopupMenu.Call()
	mfPopup := uintptr(0x0010)
	mfGrayed := uintptr(0x0001)

	// --- Toggles ---
	flags := uintptr(mfString)
	if state.enabled {
		flags |= mfChecked
	}
	pAppendMenu.Call(h, flags, menuToggle, uintptr(unsafe.Pointer(utf16p("Hotkey Enabled"))))

	flags = uintptr(mfString)
	if cfg.PauseMedia {
		flags |= mfChecked
	}
	pAppendMenu.Call(h, flags, menuPauseMedia, uintptr(unsafe.Pointer(utf16p("Pause Media"))))

	label := "Hide Indicator"
	if !state.indVisible {
		label = "Show Indicator"
	}
	pAppendMenu.Call(h, mfString, menuShowHide, uintptr(unsafe.Pointer(utf16p(label))))

	pAppendMenu.Call(h, mfSeparator, 0, 0)

	// --- Last recording actions ---
	copyFlags := uintptr(mfString)
	if state.lastOutput == "" {
		copyFlags |= mfGrayed
	}
	pAppendMenu.Call(h, copyFlags, menuCopyLast, uintptr(unsafe.Pointer(utf16p("Copy Last Output"))))

	retryFlags := uintptr(mfString)
	if len(state.lastSamples) == 0 {
		retryFlags |= mfGrayed
	}
	pAppendMenu.Call(h, retryFlags, menuRetryLast, uintptr(unsafe.Pointer(utf16p("Retry Last Recording"))))

	// --- Recent transcriptions submenu ---
	sub, _, _ := pCreatePopupMenu.Call()
	count := state.recentCount
	if count > 5 {
		count = 5
	}
	if count > 0 {
		for i := 0; i < count; i++ {
			bufIdx := (state.recentCount - 1 - i) % len(state.recentTexts)
			text := state.recentTexts[bufIdx]
			// Truncate for menu display
			display := text
			if len(display) > 60 {
				display = display[:57] + "..."
			}
			pAppendMenu.Call(sub, mfString, uintptr(menuRecent0+i), uintptr(unsafe.Pointer(utf16p(display))))
		}
	} else {
		pAppendMenu.Call(sub, mfString|mfGrayed, 0, uintptr(unsafe.Pointer(utf16p("No transcriptions yet"))))
	}
	pAppendMenu.Call(h, mfPopup, sub, uintptr(unsafe.Pointer(utf16p("Recent"))))

	pAppendMenu.Call(h, mfSeparator, 0, 0)

	// --- Dashboard & Files ---
	pAppendMenu.Call(h, mfString, menuOpenLog, uintptr(unsafe.Pointer(utf16p("Dashboard"))))
	pAppendMenu.Call(h, mfString, menuOpenConfig, uintptr(unsafe.Pointer(utf16p("Vocabulary"))))
	pAppendMenu.Call(h, mfString, menuOpenAudio, uintptr(unsafe.Pointer(utf16p("Open Recordings"))))

	pAppendMenu.Call(h, mfSeparator, 0, 0)

	// --- Stats ---
	uptime := time.Since(state.startTime)
	h2 := int(uptime.Hours())
	m2 := int(uptime.Minutes()) % 60
	var uptimeStr string
	if h2 > 0 {
		uptimeStr = fmt.Sprintf("%dh %dm", h2, m2)
	} else {
		uptimeStr = fmt.Sprintf("%dm", m2)
	}
	statsLabel := fmt.Sprintf("v%s  |  %d transcriptions  |  %s", version, state.recentCount, uptimeStr)
	pAppendMenu.Call(h, mfString|mfGrayed, 0, uintptr(unsafe.Pointer(utf16p(statsLabel))))

	// --- Update ---
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
		r, g, b = 34, 211, 238 // Cyan (#22D3EE)
	case phaseProcessing:
		r, g, b = 234, 179, 8 // Amber (#EAB308)
	default:
		r, g, b = 34, 211, 238 // Cyan (#22D3EE)
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
