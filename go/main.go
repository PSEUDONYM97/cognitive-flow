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
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"path/filepath"
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

// ----- Configuration -----

type config struct {
	Server       string            `json:"server"`
	Replacements map[string]string `json:"replacements"`
}

var cfg = config{
	Server:       "http://192.168.0.10:9200",
	Replacements: map[string]string{},
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

	pCreateSolidBrush = gdi32.NewProc("CreateSolidBrush")
	pDeleteObject     = gdi32.NewProc("DeleteObject")
	pCreateDIBSection = gdi32.NewProc("CreateDIBSection")

	pShellNotifyIcon = shell32.NewProc("Shell_NotifyIconW")
)

// ----- Constants -----

const (
	version = "2.0.1"

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

func currentPhase() int32  { return phase.Load() }
func setPhaseVal(p int32)  { phase.Store(p) }

// ----- App state -----

var state struct {
	mu            sync.Mutex
	recording     bool
	clipboardMode bool
	enabled       bool
	running       bool
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

	log("Cognitive Flow v%s", version)
	log("Server: %s", cfg.Server)

	// Clean shutdown on console close (Ctrl+C, window close, etc.)
	pSetConsoleCtrlHandler.Call(syscall.NewCallback(func(sig uintptr) uintptr {
		shutdown()
		pPostQuitMessage.Call(0) // break the message loop
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

	// Create UI
	createBar()
	createDot()
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

	text := strings.TrimSpace(result.Text)
	if text == "" {
		log("Empty transcription")
		setPhase(phaseIdle)
		return
	}

	// Word replacements from config
	text = applyReplacements(text)

	elapsed := time.Since(start)
	log("%dms (server: %.0fms) | %s", elapsed.Milliseconds(), result.Ms, text)

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

func applyReplacements(text string) string {
	if len(cfg.Replacements) == 0 {
		return text
	}
	words := strings.Fields(text)
	for i, w := range words {
		// Separate trailing punctuation: "hello," -> "hello" + ","
		clean := strings.TrimRightFunc(w, unicode.IsPunct)
		suffix := w[len(clean):]
		if repl, ok := cfg.Replacements[strings.ToLower(clean)]; ok {
			words[i] = repl + suffix
		}
	}
	return strings.Join(words, " ")
}

// ----- Text output via SendInput -----

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

	for i, r := range runes {
		if r == '\n' || r == '\r' {
			sendKey(vkReturn, 0) // VK_RETURN keydown+keyup
		} else {
			for _, c := range utf16.Encode([]rune{r}) {
				sendUnicode(c)
			}
		}
		// Batch pause every 100 chars so target app can process
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
	barMinHeight = 4
	barMaxHeight = 32
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
		var h int32

		switch p {
		case phaseRecording:
			level := audioLevel.Load()
			// Bar height scales with audio level
			h = barMinHeight + int32(level)*(barMaxHeight-barMinHeight)/100
			// Color: green (quiet) -> amber (good) -> red (loud)
			switch {
			case level < 20:
				color = 0x0050AF4C // Green (BGR)
			case level < 60:
				color = 0x0000BFFF // Amber (BGR)
			default:
				color = 0x003643F4 // Red (BGR)
			}
		case phaseProcessing:
			h = barMinHeight
			color = 0x0000BFFF // Amber
		default:
			h = 0
		}

		if h > 0 {
			brush, _, _ := pCreateSolidBrush.Call(color)
			r := [4]int32{0, 0, int32(sw), h}
			pFillRect.Call(hdc, uintptr(unsafe.Pointer(&r)), brush)
			pDeleteObject.Call(brush)
		}

		pEndPaint.Call(hwnd, uintptr(unsafe.Pointer(&ps)))
		return 0

	case wmTimer:
		if wp == timerRepaint {
			// Repaint bar with current audio level
			pInvalidateRect.Call(hwnd, 0, 1)
			// Resize bar to match current level
			sw, _, _ := pGetSystemMetrics.Call(0)
			level := audioLevel.Load()
			h := barMinHeight + int32(level)*(barMaxHeight-barMinHeight)/100
			pMoveWindow.Call(hwnd, 0, 0, sw, uintptr(h), 1)
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
		wsPopup, 0, 0, sw, uintptr(barMaxHeight), 0, 0, 0, 0,
	)
	state.bar = hwnd

	pSetLayeredWindowAttr.Call(hwnd, 0, 220, lwaAlpha)

	// Start hidden
	pShowWindow.Call(hwnd, 0) // SW_HIDE

	// Heartbeat timer for sleep/wake detection
	pSetTimer.Call(hwnd, timerHeartbeat, 30000, 0)
}

// ----- Clickable indicator dot -----
// Always visible in bottom-right corner. Click to record (clipboard mode) or stop.
// Draggable. Shows state: green=idle, gray=disabled, red=recording, amber=processing.

const dotSize = 48

var dotProc = syscall.NewCallback(func(hwnd, umsg, wp, lp uintptr) uintptr {
	switch uint32(umsg) {
	case wmPaint:
		var ps paintstruct
		hdc, _, _ := pBeginPaint.Call(hwnd, uintptr(unsafe.Pointer(&ps)))

		// Background: transparent (black will be color-keyed out)
		bgBrush, _, _ := pCreateSolidBrush.Call(0)
		r := [4]int32{0, 0, dotSize, dotSize}
		pFillRect.Call(hdc, uintptr(unsafe.Pointer(&r)), bgBrush)
		pDeleteObject.Call(bgBrush)

		// Dot color based on state
		var color uintptr
		p := currentPhase()
		switch {
		case !state.enabled:
			color = 0x00808080 // Gray
		case p == phaseRecording:
			level := audioLevel.Load()
			switch {
			case level < 20:
				color = 0x003643F4 // Red (quiet - still recording)
			case level < 60:
				color = 0x003643F4 // Red
			default:
				color = 0x001428F4 // Bright red (loud)
			}
		case p == phaseProcessing:
			color = 0x0000BFFF // Amber
		default:
			color = 0x0050AF4C // Green
		}

		// Draw filled circle
		pSelectObject := gdi32.NewProc("SelectObject")
		pEllipse := gdi32.NewProc("Ellipse")
		brush, _, _ := pCreateSolidBrush.Call(color)
		pSelectObject.Call(hdc, brush)
		// Null pen for no border
		pCreatePen := gdi32.NewProc("CreatePen")
		pen, _, _ := pCreatePen.Call(5, 0, 0) // PS_NULL
		pSelectObject.Call(hdc, pen)
		margin := int32(4)
		pEllipse.Call(hdc, uintptr(margin), uintptr(margin), uintptr(dotSize-margin), uintptr(dotSize-margin))
		pDeleteObject.Call(brush)
		pDeleteObject.Call(pen)

		pEndPaint.Call(hwnd, uintptr(unsafe.Pointer(&ps)))
		return 0

	case wmNcHitTest:
		// Allow dragging by treating the whole window as caption
		return htCaption

	case wmLButtonUp:
		// Click: start clipboard recording or stop current recording
		go func() {
			if currentPhase() == phaseRecording {
				toggle(true) // stop + clipboard mode
			} else if state.enabled {
				toggle(true) // start clipboard recording
			}
		}()
		return 0

	case wmTimer:
		if wp == timerRepaint {
			pInvalidateRect.Call(hwnd, 0, 1)
		}
		return 0
	}

	r, _, _ := pDefWindowProc.Call(hwnd, umsg, wp, lp)
	return r
})

func createDot() {
	cls := utf16p("CogFlowDot")
	cursor, _, _ := pLoadCursor.Call(0, 32649) // IDC_HAND
	wc := wndclass{
		Size: uint32(unsafe.Sizeof(wndclass{})), WndProc: dotProc,
		ClassName: cls, Cursor: cursor,
	}
	pRegisterClassEx.Call(uintptr(unsafe.Pointer(&wc)))

	sw, _, _ := pGetSystemMetrics.Call(0) // SM_CXSCREEN
	sh, _, _ := pGetSystemMetrics.Call(1) // SM_CYSCREEN

	// Bottom-right corner with some margin
	x := int(sw) - dotSize - 20
	y := int(sh) - dotSize - 60

	hwnd, _, _ := pCreateWindowEx.Call(
		wsExLayered|wsExTopmost|wsExToolWindow,
		uintptr(unsafe.Pointer(cls)), 0,
		wsPopup,
		uintptr(x), uintptr(y), dotSize, dotSize,
		0, 0, 0, 0,
	)
	state.dot = hwnd

	// Black = transparent (color key), so only the circle is visible
	pSetLayeredWindowAttr.Call(hwnd, 0, 0, lwaColorKey)

	pShowWindow.Call(hwnd, 8) // SW_SHOWNA
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
		pShowWindow.Call(state.bar, 8) // SW_SHOWNA
		pInvalidateRect.Call(state.bar, 0, 1)
		pSetWindowPos.Call(state.bar, ^uintptr(0), 0, 0, 0, 0, 0x0001|0x0002|0x0010)
		pSetTimer.Call(state.bar, timerRepaint, 50, 0)
		if state.dot != 0 {
			pSetTimer.Call(state.dot, timerRepaint, 50, 0)
			pInvalidateRect.Call(state.dot, 0, 1)
		}
	case phaseProcessing:
		pKillTimer.Call(state.bar, timerRepaint)
		audioLevel.Store(0)
		sw, _, _ := pGetSystemMetrics.Call(0)
		pMoveWindow.Call(state.bar, 0, 0, sw, barMinHeight, 1)
		pShowWindow.Call(state.bar, 8)
		pInvalidateRect.Call(state.bar, 0, 1)
		pSetWindowPos.Call(state.bar, ^uintptr(0), 0, 0, 0, 0, 0x0001|0x0002|0x0010)
		if state.dot != 0 {
			pKillTimer.Call(state.dot, timerRepaint)
			pInvalidateRect.Call(state.dot, 0, 1)
		}
	default:
		pKillTimer.Call(state.bar, timerRepaint)
		audioLevel.Store(0)
		pShowWindow.Call(state.bar, 0) // SW_HIDE
		if state.dot != 0 {
			pKillTimer.Call(state.dot, timerRepaint)
			pInvalidateRect.Call(state.dot, 0, 1)
		}
	}

	updateTrayIcon()
}

// ----- System tray -----

const (
	menuToggle = 1
	menuQuit   = 2
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

	flags := uintptr(mfString)
	if state.enabled {
		flags |= mfChecked
	}
	pAppendMenu.Call(h, flags, menuToggle, uintptr(unsafe.Pointer(utf16p("Hotkey Enabled"))))
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
