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
	"strings"
	"sync"
	"syscall"
	"time"
	"unicode/utf16"
	"unsafe"

	"golang.org/x/sys/windows"
)

// --- Config ---

const (
	serverURL  = "http://192.168.0.10:9200"
	sampleRate = 16000
	channels   = 1
	bitDepth   = 16
	chunkSize  = 1024
	numBuffers = 4
	version    = "2.0.0"
)

// --- Win32 ---

var (
	user32   = windows.NewLazySystemDLL("user32.dll")
	kernel32 = windows.NewLazySystemDLL("kernel32.dll")
	winmm    = windows.NewLazySystemDLL("winmm.dll")
	gdi32    = windows.NewLazySystemDLL("gdi32.dll")
	shell32  = windows.NewLazySystemDLL("shell32.dll")

	setWindowsHookEx     = user32.NewProc("SetWindowsHookExW")
	unhookWindowsHookEx  = user32.NewProc("UnhookWindowsHookEx")
	callNextHookEx       = user32.NewProc("CallNextHookEx")
	peekMessage          = user32.NewProc("PeekMessageW")
	translateMessage     = user32.NewProc("TranslateMessage")
	dispatchMessage      = user32.NewProc("DispatchMessageW")
	postMessage          = user32.NewProc("PostMessageW")
	postQuitMessage      = user32.NewProc("PostQuitMessage")
	getForegroundWindow  = user32.NewProc("GetForegroundWindow")
	getFocus             = user32.NewProc("GetFocus")
	getWindowThreadPID   = user32.NewProc("GetWindowThreadProcessId")
	attachThreadInput    = user32.NewProc("AttachThreadInput")
	getAsyncKeyState     = user32.NewProc("GetAsyncKeyState")
	registerClassEx      = user32.NewProc("RegisterClassExW")
	createWindowEx       = user32.NewProc("CreateWindowExW")
	destroyWindow        = user32.NewProc("DestroyWindow")
	showWindow           = user32.NewProc("ShowWindow")
	setWindowPos         = user32.NewProc("SetWindowPos")
	moveWindow           = user32.NewProc("MoveWindow")
	getWindowRect        = user32.NewProc("GetWindowRect")
	setLayeredWindowAttr = user32.NewProc("SetLayeredWindowAttributes")
	defWindowProc        = user32.NewProc("DefWindowProcW")
	loadCursor           = user32.NewProc("LoadCursorW")
	getSystemMetrics     = user32.NewProc("GetSystemMetrics")
	getCursorPos         = user32.NewProc("GetCursorPos")
	setTimer             = user32.NewProc("SetTimer")
	invalidateRect       = user32.NewProc("InvalidateRect")
	beginPaint           = user32.NewProc("BeginPaint")
	endPaint             = user32.NewProc("EndPaint")
	fillRect             = user32.NewProc("FillRect")
	openClipboard        = user32.NewProc("OpenClipboard")
	closeClipboard       = user32.NewProc("CloseClipboard")
	emptyClipboard       = user32.NewProc("EmptyClipboard")
	setClipboardData     = user32.NewProc("SetClipboardData")
	getDC                = user32.NewProc("GetDC")
	releaseDC            = user32.NewProc("ReleaseDC")
	setForegroundWindow  = user32.NewProc("SetForegroundWindow")
	createPopupMenu      = user32.NewProc("CreatePopupMenu")
	appendMenu           = user32.NewProc("AppendMenuW")
	trackPopupMenu       = user32.NewProc("TrackPopupMenu")
	destroyMenu          = user32.NewProc("DestroyMenu")

	getCurrentThreadId   = kernel32.NewProc("GetCurrentThreadId")
	globalAlloc          = kernel32.NewProc("GlobalAlloc")
	globalLock           = kernel32.NewProc("GlobalLock")
	globalUnlock         = kernel32.NewProc("GlobalUnlock")
	createEvent          = kernel32.NewProc("CreateEventW")
	setEvent             = kernel32.NewProc("SetEvent")
	waitForSingleObject  = kernel32.NewProc("WaitForSingleObject")

	waveInOpen            = winmm.NewProc("waveInOpen")
	waveInClose           = winmm.NewProc("waveInClose")
	waveInPrepareHeader   = winmm.NewProc("waveInPrepareHeader")
	waveInUnprepareHeader = winmm.NewProc("waveInUnprepareHeader")
	waveInAddBuffer       = winmm.NewProc("waveInAddBuffer")
	waveInStart           = winmm.NewProc("waveInStart")
	waveInStop            = winmm.NewProc("waveInStop")
	waveInReset           = winmm.NewProc("waveInReset")

	createSolidBrush       = gdi32.NewProc("CreateSolidBrush")
	createPen              = gdi32.NewProc("CreatePen")
	selectObject           = gdi32.NewProc("SelectObject")
	deleteObject           = gdi32.NewProc("DeleteObject")
	ellipse                = gdi32.NewProc("Ellipse")
	createDIBSection       = gdi32.NewProc("CreateDIBSection")

	shellNotifyIcon    = shell32.NewProc("Shell_NotifyIconW")
	createIconIndirect = user32.NewProc("CreateIconIndirect")
	destroyIcon        = user32.NewProc("DestroyIcon")
)

const (
	whKeyboardLL = 13
	wmKeydown    = 0x0100
	wmChar       = 0x0102
	wmPaint      = 0x000F
	wmDestroy    = 0x0002
	wmTimer      = 0x0113
	wmLButtonUp  = 0x0202
	wmCommand    = 0x0111
	wmApp        = 0x8000
	wmTrayIcon   = wmApp + 1
	wmRButtonUp  = 0x0205
	ninSelect    = 0x0400
	vkTilde      = 0xC0
	vkEscape     = 0x1B
	vkLControl   = 0xA2
	vkRControl   = 0xA3
	pmRemove     = 0x0001
	cfUnicode    = 13
	gmemMoveable = 0x0002

	wsPopup        = 0x80000000
	wsExLayered    = 0x00080000
	wsExTopmost    = 0x00000008
	wsExToolWindow = 0x00000080
	wsExNoActivate = 0x08000000

	swShowNA = 8
	swHide   = 0

	lwaColorKey = 0x01

	nimAdd    = 0x00000000
	nimModify = 0x00000001
	nimDelete = 0x00000002
	nifMsg    = 0x00000001
	nifIcon   = 0x00000002
	nifTip    = 0x00000004
	nifShowTip = 0x00000080

	mfString    = 0x00000000
	mfSeparator = 0x00000800
	mfChecked   = 0x00000008

	callbackEvent  = 0x00050000
	waveMapper     = 0xFFFFFFFF
	wavePCM        = 1
	whdrDone       = 0x00000001
)

// --- Structures ---

type msg struct {
	Hwnd    uintptr
	Message uint32
	WParam  uintptr
	LParam  uintptr
	Time    uint32
	Pt      point
}

type point struct{ X, Y int32 }
type rect struct{ Left, Top, Right, Bottom int32 }

type kbdllhook struct {
	VkCode      uint32
	ScanCode    uint32
	Flags       uint32
	Time        uint32
	DwExtraInfo uintptr
}

type wndclassex struct {
	Size       uint32
	Style      uint32
	WndProc    uintptr
	ClsExtra   int32
	WndExtra   int32
	Instance   uintptr
	Icon       uintptr
	Cursor     uintptr
	Background uintptr
	MenuName   *uint16
	ClassName  *uint16
	IconSm     uintptr
}

type paintstruct struct {
	Hdc        uintptr
	Erase      int32
	Paint      rect
	Restore    int32
	IncUpdate  int32
	Reserved   [32]byte
}

type waveformatex struct {
	Tag        uint16
	Channels   uint16
	SampleRate uint32
	ByteRate   uint32
	BlockAlign uint16
	BitDepth   uint16
	Extra      uint16
}

type wavehdr struct {
	Data     uintptr
	Length   uint32
	Recorded uint32
	User     uintptr
	Flags    uint32
	Loops    uint32
	Next     uintptr
	Reserved uintptr
}

type notifyicondata struct {
	Size         uint32
	Hwnd         uintptr
	ID           uint32
	Flags        uint32
	CallbackMsg  uint32
	Icon         uintptr
	Tip          [128]uint16
	State        uint32
	StateMask    uint32
	Info         [256]uint16
	Version      uint32
	InfoTitle    [64]uint16
	InfoFlags    uint32
	GUID         [16]byte
	BalloonIcon  uintptr
}

type bitmapinfoheader struct {
	Size          uint32
	Width         int32
	Height        int32
	Planes        uint16
	BitCount      uint16
	Compression   uint32
	SizeImage     uint32
	XPelsPerMeter int32
	YPelsPerMeter int32
	ClrUsed       uint32
	ClrImportant  uint32
}

type bitmapinfo struct {
	Header bitmapinfoheader
	Colors [1]uint32
}

type iconinfo struct {
	IsIcon  int32
	XHot    uint32
	YHot    uint32
	Mask    uintptr
	Color   uintptr
}

// --- App state ---

var app struct {
	mu            sync.Mutex
	hook          uintptr
	recording     bool
	clipboardMode bool
	hotkeyEnabled bool
	running       bool
	lastEsc       time.Time
	lastLoop      time.Time

	// Audio
	hwi     uintptr
	event   uintptr
	buffers [numBuffers][]byte
	headers [numBuffers]wavehdr
	frames  []int16

	// UI
	indicator uintptr
	trayHwnd  uintptr
	trayIcon  uintptr
	nid       notifyicondata
	state     string // "idle", "recording", "processing"
}

func main() {
	app.hotkeyEnabled = true
	app.running = true
	app.state = "idle"

	log("Cognitive Flow v%s (Go)", version)
	log("Server: %s", serverURL)

	// Warm up server
	go warmup()

	// Create UI
	createIndicator()
	createTray()

	// Install keyboard hook
	proc := syscall.NewCallback(hookProc)
	h, _, err := setWindowsHookEx.Call(whKeyboardLL, proc, 0, 0)
	if h == 0 {
		fatal("keyboard hook failed: %v", err)
	}
	app.hook = h
	defer unhookWindowsHookEx.Call(app.hook)

	log("Ready. Press ~ to record.")

	// Message loop
	app.lastLoop = time.Now()
	var m msg
	for app.running {
		for peekMsg(&m) {
			translateMessage.Call(uintptr(unsafe.Pointer(&m)))
			dispatchMessage.Call(uintptr(unsafe.Pointer(&m)))
		}

		// Sleep/wake detection
		now := time.Now()
		if now.Sub(app.lastLoop) > 30*time.Second {
			log("Wake detected, warming up server")
			go warmup()
		}
		app.lastLoop = now

		time.Sleep(1 * time.Millisecond)
	}
}

// --- Keyboard hook ---

func hookProc(nCode int, wParam, lParam uintptr) uintptr {
	if nCode >= 0 {
		kb := (*kbdllhook)(unsafe.Pointer(lParam))
		down := wParam == wmKeydown

		if kb.VkCode == vkTilde && down {
			// Ctrl+~ = toggle hotkey
			if ctrlDown() {
				app.hotkeyEnabled = !app.hotkeyEnabled
				if app.hotkeyEnabled {
					log("Hotkey enabled")
					setState("idle")
				} else {
					log("Hotkey disabled")
					setState("idle")
				}
				return 1
			}

			if app.hotkeyEnabled {
				go toggle(false)
				return 1
			}
		}

		if kb.VkCode == vkEscape && down && app.recording {
			now := time.Now()
			if now.Sub(app.lastEsc) < 500*time.Millisecond {
				go cancel()
				return 1
			}
			app.lastEsc = now
		}
	}

	r, _, _ := callNextHookEx.Call(0, uintptr(nCode), wParam, lParam)
	return r
}

func ctrlDown() bool {
	l, _, _ := getAsyncKeyState.Call(vkLControl)
	r, _, _ := getAsyncKeyState.Call(vkRControl)
	return int16(l)&(-32768) != 0 || int16(r)&(-32768) != 0
}

// --- Recording ---

func toggle(clipboard bool) {
	app.mu.Lock()
	defer app.mu.Unlock()

	if app.recording {
		if clipboard {
			app.clipboardMode = true
		}
		stopRecording()
	} else {
		app.clipboardMode = clipboard
		startRecording()
	}
}

func cancel() {
	app.mu.Lock()
	defer app.mu.Unlock()

	if !app.recording {
		return
	}
	log("Cancelled")
	app.recording = false
	resetAudio()
	setState("idle")
}

func startRecording() {
	log("Recording... (clipboard: %v)", app.clipboardMode)
	app.recording = true
	app.frames = nil
	setState("recording")

	// Warm server while we record
	go warmup()

	// Open audio
	wfx := waveformatex{
		Tag: wavePCM, Channels: channels, SampleRate: sampleRate,
		BitDepth: bitDepth, BlockAlign: channels * bitDepth / 8,
		ByteRate: sampleRate * uint32(channels) * uint32(bitDepth) / 8,
	}

	ev, _, _ := createEvent.Call(0, 0, 0, 0)
	app.event = ev

	ret, _, _ := waveInOpen.Call(
		uintptr(unsafe.Pointer(&app.hwi)), waveMapper,
		uintptr(unsafe.Pointer(&wfx)), ev, 0, callbackEvent,
	)
	if ret != 0 {
		log("waveInOpen failed: %d", ret)
		app.recording = false
		setState("idle")
		return
	}

	bufSize := chunkSize * channels * bitDepth / 8
	for i := 0; i < numBuffers; i++ {
		app.buffers[i] = make([]byte, bufSize)
		app.headers[i] = wavehdr{
			Data:   uintptr(unsafe.Pointer(&app.buffers[i][0])),
			Length: uint32(bufSize),
		}
		waveInPrepareHeader.Call(app.hwi, uintptr(unsafe.Pointer(&app.headers[i])), unsafe.Sizeof(app.headers[i]))
		waveInAddBuffer.Call(app.hwi, uintptr(unsafe.Pointer(&app.headers[i])), unsafe.Sizeof(app.headers[i]))
	}

	waveInStart.Call(app.hwi)
	go recordLoop()
}

func recordLoop() {
	for {
		if !app.recording {
			return
		}
		waitForSingleObject.Call(app.event, 100)
		if !app.recording {
			return
		}

		for i := 0; i < numBuffers; i++ {
			if app.headers[i].Flags&whdrDone != 0 {
				n := app.headers[i].Recorded
				if n > 0 {
					buf := make([]byte, n)
					copy(buf, app.buffers[i][:n])
					samples := make([]int16, n/2)
					for j := range samples {
						samples[j] = int16(buf[j*2]) | int16(buf[j*2+1])<<8
					}
					app.frames = append(app.frames, samples...)
				}
				app.headers[i].Flags = 0
				app.headers[i].Recorded = 0
				waveInAddBuffer.Call(app.hwi, uintptr(unsafe.Pointer(&app.headers[i])), unsafe.Sizeof(app.headers[i]))
			}
		}
	}
}

func stopRecording() {
	if !app.recording {
		return
	}
	app.recording = false
	setState("processing")

	waveInStop.Call(app.hwi)
	waveInReset.Call(app.hwi)
	setEvent.Call(app.event)

	frames := make([]int16, len(app.frames))
	copy(frames, app.frames)
	clipboard := app.clipboardMode

	resetAudio()

	if len(frames) == 0 {
		log("No audio")
		setState("idle")
		return
	}

	log("Captured %.1fs audio", float64(len(frames))/sampleRate)
	go transcribe(frames, clipboard)
}

func resetAudio() {
	if app.hwi != 0 {
		for i := 0; i < numBuffers; i++ {
			waveInUnprepareHeader.Call(app.hwi, uintptr(unsafe.Pointer(&app.headers[i])), unsafe.Sizeof(app.headers[i]))
		}
		waveInClose.Call(app.hwi)
		app.hwi = 0
	}
}

// --- Transcription ---

func warmup() {
	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Get(serverURL + "/health")
	if err != nil {
		log("Server unreachable: %v", err)
		return
	}
	defer resp.Body.Close()
	var health struct {
		Status string `json:"status"`
		Model  string `json:"model"`
	}
	json.NewDecoder(resp.Body).Decode(&health)
	log("Server: %s (%s)", health.Model, health.Status)
}

func transcribe(samples []int16, clipboard bool) {
	start := time.Now()

	// Encode WAV
	wav := encodeWAV(samples)

	// Build multipart
	boundary := "----CogFlow" + fmt.Sprintf("%d", time.Now().UnixNano())
	var body bytes.Buffer
	body.WriteString("--" + boundary + "\r\n")
	body.WriteString("Content-Disposition: form-data; name=\"audio\"; filename=\"audio.wav\"\r\n")
	body.WriteString("Content-Type: audio/wav\r\n\r\n")
	body.Write(wav)
	body.WriteString("\r\n--" + boundary + "--\r\n")

	// POST with retry
	var result struct {
		Text string  `json:"text"`
		Ms   float64 `json:"processing_time_ms"`
	}

	delays := []time.Duration{5 * time.Second, 10 * time.Second, 15 * time.Second}
	var lastErr error

	for attempt := 0; attempt <= 3; attempt++ {
		if attempt > 0 {
			log("Retry %d/3 in %v...", attempt, delays[attempt-1])
			time.Sleep(delays[attempt-1])
		}

		client := &http.Client{Timeout: 30 * time.Second}
		req, _ := http.NewRequest("POST", serverURL+"/transcribe",
			bytes.NewReader(body.Bytes()))
		req.Header.Set("Content-Type", "multipart/form-data; boundary="+boundary)

		resp, err := client.Do(req)
		if err != nil {
			lastErr = err
			continue
		}

		respBody, _ := io.ReadAll(resp.Body)
		resp.Body.Close()

		if resp.StatusCode != 200 {
			lastErr = fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(respBody))
			continue
		}

		json.Unmarshal(respBody, &result)
		lastErr = nil
		break
	}

	if lastErr != nil {
		log("Transcription failed: %v", lastErr)
		setState("idle")
		return
	}

	text := strings.TrimSpace(result.Text)
	if text == "" {
		log("Empty result")
		setState("idle")
		return
	}

	elapsed := time.Since(start)
	log("%.0fms (server: %.0fms) | %s", float64(elapsed.Milliseconds()), result.Ms, text)

	// Output
	if clipboard {
		if err := copyToClipboard(text); err != nil {
			log("Clipboard error: %v", err)
		} else {
			log("Copied to clipboard")
		}
	} else {
		if err := typeText(text + " "); err != nil {
			log("Type error: %v, falling back to clipboard", err)
			copyToClipboard(text)
		}
	}

	setState("idle")
}

func encodeWAV(samples []int16) []byte {
	dataSize := len(samples) * 2
	var buf bytes.Buffer
	buf.Grow(44 + dataSize)

	buf.WriteString("RIFF")
	binary.Write(&buf, binary.LittleEndian, uint32(36+dataSize))
	buf.WriteString("WAVE")
	buf.WriteString("fmt ")
	binary.Write(&buf, binary.LittleEndian, uint32(16))
	binary.Write(&buf, binary.LittleEndian, uint16(1))          // PCM
	binary.Write(&buf, binary.LittleEndian, uint16(1))          // mono
	binary.Write(&buf, binary.LittleEndian, uint32(sampleRate))
	binary.Write(&buf, binary.LittleEndian, uint32(sampleRate*2))
	binary.Write(&buf, binary.LittleEndian, uint16(2))          // block align
	binary.Write(&buf, binary.LittleEndian, uint16(16))         // bits
	buf.WriteString("data")
	binary.Write(&buf, binary.LittleEndian, uint32(dataSize))

	for _, s := range samples {
		binary.Write(&buf, binary.LittleEndian, s)
	}
	return buf.Bytes()
}

// --- Text injection ---

func typeText(text string) error {
	// Sanitize
	var clean strings.Builder
	for _, r := range text {
		if r == '`' {
			clean.WriteRune('\'')
		} else if r >= 0x20 || r == '\n' || r == '\r' || r == '\t' {
			clean.WriteRune(r)
		}
	}
	text = clean.String()

	fg, _, _ := getForegroundWindow.Call()
	if fg == 0 {
		return fmt.Errorf("no foreground window")
	}

	fgThread, _, _ := getWindowThreadPID.Call(fg, 0)
	myThread, _, _ := getCurrentThreadId.Call()

	var target uintptr
	if fgThread != myThread {
		attachThreadInput.Call(myThread, fgThread, 1)
		target, _, _ = getFocus.Call()
		if target == 0 {
			target = fg
		}
		defer attachThreadInput.Call(myThread, fgThread, 0)
	} else {
		target = fg
	}

	count := 0
	for _, r := range text {
		if r == '\n' {
			postMessage.Call(target, wmChar, 13, 0)
		} else {
			for _, c := range utf16.Encode([]rune{r}) {
				postMessage.Call(target, wmChar, uintptr(c), 0)
			}
		}
		count++
		if count%100 == 0 {
			time.Sleep(time.Millisecond)
		}
	}
	return nil
}

func copyToClipboard(text string) error {
	u16 := utf16.Encode([]rune(text + "\x00"))
	size := len(u16) * 2

	r, _, _ := openClipboard.Call(0)
	if r == 0 {
		return fmt.Errorf("OpenClipboard failed")
	}
	defer closeClipboard.Call()
	emptyClipboard.Call()

	hMem, _, _ := globalAlloc.Call(gmemMoveable, uintptr(size))
	if hMem == 0 {
		return fmt.Errorf("GlobalAlloc failed")
	}
	ptr, _, _ := globalLock.Call(hMem)
	if ptr == 0 {
		return fmt.Errorf("GlobalLock failed")
	}

	dst := unsafe.Slice((*uint16)(unsafe.Pointer(ptr)), len(u16))
	copy(dst, u16)
	globalUnlock.Call(hMem)
	setClipboardData.Call(cfUnicode, hMem)
	return nil
}

// --- Indicator (status dot) ---

var indicatorProc = syscall.NewCallback(func(hwnd, umsg, wParam, lParam uintptr) uintptr {
	switch uint32(umsg) {
	case wmPaint:
		var ps paintstruct
		hdc, _, _ := beginPaint.Call(hwnd, uintptr(unsafe.Pointer(&ps)))

		// Black background (color key = transparent)
		r := rect{0, 0, 24, 24}
		brush, _, _ := createSolidBrush.Call(0)
		fillRect.Call(hdc, uintptr(unsafe.Pointer(&r)), brush)
		deleteObject.Call(brush)

		// Status dot
		var color uintptr
		switch app.state {
		case "recording":
			color = 0x003643F4 // Red (BGR)
		case "processing":
			color = 0x0000BFFF // Amber
		default:
			color = 0x0050AF4C // Green
		}

		dotBrush, _, _ := createSolidBrush.Call(color)
		nullPen, _, _ := createPen.Call(5, 0, 0) // PS_NULL
		oldBrush, _, _ := selectObject.Call(hdc, dotBrush)
		oldPen, _, _ := selectObject.Call(hdc, nullPen)
		ellipse.Call(hdc, 2, 2, 22, 22)
		selectObject.Call(hdc, oldBrush)
		selectObject.Call(hdc, oldPen)
		deleteObject.Call(dotBrush)
		deleteObject.Call(nullPen)

		endPaint.Call(hwnd, uintptr(unsafe.Pointer(&ps)))
		return 0

	case wmLButtonUp:
		go toggle(true) // Click = clipboard mode
		return 0

	case wmTimer:
		// Heartbeat: make sure we're still visible and on screen
		if app.state == "recording" || app.state == "processing" {
			showWindow.Call(hwnd, swShowNA)
			setWindowPos.Call(hwnd, ^uintptr(0), 0, 0, 0, 0, 0x0001|0x0002|0x0010) // TOPMOST|NOSIZE|NOMOVE|NOACTIVATE
		}
		return 0
	}

	r, _, _ := defWindowProc.Call(hwnd, umsg, wParam, lParam)
	return r
})

func createIndicator() {
	cls := utf16p("CogFlowDot")
	cursor, _, _ := loadCursor.Call(0, 32512)

	wc := wndclassex{
		Size: uint32(unsafe.Sizeof(wndclassex{})), WndProc: indicatorProc,
		ClassName: cls, Cursor: cursor,
	}
	registerClassEx.Call(uintptr(unsafe.Pointer(&wc)))

	sw, _, _ := getSystemMetrics.Call(0)
	sh, _, _ := getSystemMetrics.Call(1)
	x := int32(sw) - 24 - 16
	y := int32(sh) - 24 - 56

	hwnd, _, _ := createWindowEx.Call(
		wsExLayered|wsExTopmost|wsExToolWindow|wsExNoActivate,
		uintptr(unsafe.Pointer(cls)), uintptr(unsafe.Pointer(utf16p("CogFlow"))),
		wsPopup, uintptr(x), uintptr(y), 24, 24, 0, 0, 0, 0,
	)
	app.indicator = hwnd

	// Black = transparent
	setLayeredWindowAttr.Call(hwnd, 0, 0, lwaColorKey)
	showWindow.Call(hwnd, swShowNA)

	// Heartbeat every 30s
	setTimer.Call(hwnd, 1, 30000, 0)
}

func setState(state string) {
	app.state = state
	if app.indicator != 0 {
		invalidateRect.Call(app.indicator, 0, 1)
		if state == "recording" || state == "processing" {
			showWindow.Call(app.indicator, swShowNA)
		}
	}
	updateTrayIcon()
}

// --- System tray ---

var trayProc = syscall.NewCallback(func(hwnd, umsg, wParam, lParam uintptr) uintptr {
	switch uint32(umsg) {
	case wmTrayIcon:
		switch uint16(lParam) {
		case uint16(wmRButtonUp):
			showTrayMenu(hwnd)
		}
		return 0

	case wmCommand:
		switch uint16(wParam) {
		case 1: // Settings placeholder
			log("Settings not yet implemented")
		case 2: // Toggle hotkey
			app.hotkeyEnabled = !app.hotkeyEnabled
			if app.hotkeyEnabled {
				log("Hotkey enabled")
			} else {
				log("Hotkey disabled")
			}
		case 3: // Reset overlay
			sw, _, _ := getSystemMetrics.Call(0)
			sh, _, _ := getSystemMetrics.Call(1)
			moveWindow.Call(app.indicator, uintptr(int32(sw)-40), uintptr(int32(sh)-80), 24, 24, 1)
		case 9: // Quit
			app.running = false
			shellNotifyIcon.Call(nimDelete, uintptr(unsafe.Pointer(&app.nid)))
			postQuitMessage.Call(0)
		}
		return 0
	}

	r, _, _ := defWindowProc.Call(hwnd, umsg, wParam, lParam)
	return r
})

func createTray() {
	cls := utf16p("CogFlowTray")
	wc := wndclassex{
		Size: uint32(unsafe.Sizeof(wndclassex{})), WndProc: trayProc, ClassName: cls,
	}
	registerClassEx.Call(uintptr(unsafe.Pointer(&wc)))

	hwnd, _, _ := createWindowEx.Call(0,
		uintptr(unsafe.Pointer(cls)), uintptr(unsafe.Pointer(utf16p("CogFlowTrayHost"))),
		0, 0, 0, 0, 0, 0, 0, 0, 0,
	)
	app.trayHwnd = hwnd

	app.nid = notifyicondata{
		Size:        uint32(unsafe.Sizeof(notifyicondata{})),
		Hwnd:        hwnd,
		ID:          1,
		Flags:       nifMsg | nifIcon | nifTip | nifShowTip,
		CallbackMsg: wmTrayIcon,
	}

	tip := utf16f("Cognitive Flow v%s", version)
	copy(app.nid.Tip[:], tip)
	app.nid.Icon = makeIcon(76, 175, 80) // Green

	shellNotifyIcon.Call(nimAdd, uintptr(unsafe.Pointer(&app.nid)))
}

func showTrayMenu(hwnd uintptr) {
	hMenu, _, _ := createPopupMenu.Call()

	addMenuItem(hMenu, 1, "Settings...")

	hotkeyLabel := "Hotkey Enabled"
	flags := uintptr(mfString)
	if app.hotkeyEnabled {
		flags |= mfChecked
	}
	appendMenu.Call(hMenu, flags, 2, uintptr(unsafe.Pointer(utf16p(hotkeyLabel))))

	addMenuItem(hMenu, 3, "Reset Overlay Position")
	appendMenu.Call(hMenu, mfSeparator, 0, 0)
	addMenuItem(hMenu, 9, "Quit")

	var pt point
	getCursorPos.Call(uintptr(unsafe.Pointer(&pt)))
	setForegroundWindow.Call(hwnd)
	trackPopupMenu.Call(hMenu, 0x0020, uintptr(pt.X), uintptr(pt.Y), 0, hwnd, 0)
	destroyMenu.Call(hMenu)
}

func addMenuItem(hMenu uintptr, id int, text string) {
	appendMenu.Call(hMenu, mfString, uintptr(id), uintptr(unsafe.Pointer(utf16p(text))))
}

func updateTrayIcon() {
	var r, g, b byte
	switch app.state {
	case "recording":
		r, g, b = 244, 67, 54
	case "processing":
		r, g, b = 255, 191, 0
	default:
		r, g, b = 76, 175, 80
	}

	old := app.nid.Icon
	app.nid.Icon = makeIcon(r, g, b)
	app.nid.Flags = nifIcon
	shellNotifyIcon.Call(nimModify, uintptr(unsafe.Pointer(&app.nid)))
	if old != 0 {
		destroyIcon.Call(old)
	}
}

func makeIcon(r, g, b byte) uintptr {
	size := 16
	hdc, _, _ := getDC.Call(0)

	bmi := bitmapinfo{Header: bitmapinfoheader{
		Size: uint32(unsafe.Sizeof(bitmapinfoheader{})),
		Width: int32(size), Height: -int32(size), Planes: 1, BitCount: 32,
	}}

	var bits uintptr
	hbm, _, _ := createDIBSection.Call(hdc, uintptr(unsafe.Pointer(&bmi)), 0, uintptr(unsafe.Pointer(&bits)), 0, 0)
	releaseDC.Call(0, hdc)

	if hbm == 0 || bits == 0 {
		return 0
	}

	// Draw circle in BGRA
	pixels := unsafe.Slice((*byte)(unsafe.Pointer(bits)), size*size*4)
	cx, cy, rad := float64(size)/2, float64(size)/2, float64(size)/2-1
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			dx, dy := float64(x)-cx+0.5, float64(y)-cy+0.5
			if math.Sqrt(dx*dx+dy*dy) <= rad {
				off := (y*size + x) * 4
				pixels[off+0] = b
				pixels[off+1] = g
				pixels[off+2] = r
				pixels[off+3] = 255
			}
		}
	}

	// Mask bitmap
	hdc2, _, _ := getDC.Call(0)
	var maskBits uintptr
	hbmMask, _, _ := createDIBSection.Call(hdc2, uintptr(unsafe.Pointer(&bmi)), 0, uintptr(unsafe.Pointer(&maskBits)), 0, 0)
	releaseDC.Call(0, hdc2)

	ii := iconinfo{IsIcon: 1, Mask: hbmMask, Color: hbm}
	icon, _, _ := createIconIndirect.Call(uintptr(unsafe.Pointer(&ii)))

	deleteObject.Call(hbm)
	deleteObject.Call(hbmMask)
	return icon
}

// --- Helpers ---

func peekMsg(m *msg) bool {
	r, _, _ := peekMessage.Call(uintptr(unsafe.Pointer(m)), 0, 0, 0, pmRemove)
	return r != 0
}

func utf16p(s string) *uint16 {
	p, _ := syscall.UTF16PtrFromString(s)
	return p
}

func utf16f(format string, args ...interface{}) []uint16 {
	s := fmt.Sprintf(format, args...)
	p, _ := syscall.UTF16FromString(s)
	return p
}

func log(format string, args ...interface{}) {
	ts := time.Now().Format("15:04:05")
	fmt.Printf("[%s] %s\n", ts, fmt.Sprintf(format, args...))
}

func fatal(format string, args ...interface{}) {
	log("FATAL: "+format, args...)
	os.Exit(1)
}
