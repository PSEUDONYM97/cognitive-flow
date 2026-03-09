package main

import (
	"syscall"
	"unsafe"

	"golang.org/x/sys/windows"
)

// --- DLLs ---

var (
	user32   = windows.NewLazySystemDLL("user32.dll")
	kernel32 = windows.NewLazySystemDLL("kernel32.dll")
	winmm    = windows.NewLazySystemDLL("winmm.dll")
	gdi32    = windows.NewLazySystemDLL("gdi32.dll")
	shell32  = windows.NewLazySystemDLL("shell32.dll")
	msimg32  = windows.NewLazySystemDLL("msimg32.dll")
)

// --- user32 procs ---

var (
	procSetWindowsHookExW     = user32.NewProc("SetWindowsHookExW")
	procUnhookWindowsHookEx   = user32.NewProc("UnhookWindowsHookEx")
	procCallNextHookEx        = user32.NewProc("CallNextHookEx")
	procGetMessageW           = user32.NewProc("GetMessageW")
	procPeekMessageW          = user32.NewProc("PeekMessageW")
	procTranslateMessage      = user32.NewProc("TranslateMessage")
	procDispatchMessageW      = user32.NewProc("DispatchMessageW")
	procPostQuitMessage       = user32.NewProc("PostQuitMessage")
	procPostMessageW          = user32.NewProc("PostMessageW")
	procSendMessageW          = user32.NewProc("SendMessageW")
	procGetForegroundWindow   = user32.NewProc("GetForegroundWindow")
	procGetFocus              = user32.NewProc("GetFocus")
	procGetWindowThreadProcessId = user32.NewProc("GetWindowThreadProcessId")
	procAttachThreadInput     = user32.NewProc("AttachThreadInput")
	procGetAsyncKeyState      = user32.NewProc("GetAsyncKeyState")
	procRegisterClassExW      = user32.NewProc("RegisterClassExW")
	procCreateWindowExW       = user32.NewProc("CreateWindowExW")
	procDestroyWindow         = user32.NewProc("DestroyWindow")
	procShowWindow            = user32.NewProc("ShowWindow")
	procUpdateWindow          = user32.NewProc("UpdateWindow")
	procSetWindowPos          = user32.NewProc("SetWindowPos")
	procGetWindowRect         = user32.NewProc("GetWindowRect")
	procMoveWindow            = user32.NewProc("MoveWindow")
	procSetLayeredWindowAttributes = user32.NewProc("SetLayeredWindowAttributes")
	procUpdateLayeredWindow   = user32.NewProc("UpdateLayeredWindow")
	procDefWindowProcW        = user32.NewProc("DefWindowProcW")
	procLoadCursorW           = user32.NewProc("LoadCursorW")
	procGetSystemMetrics      = user32.NewProc("GetSystemMetrics")
	procGetCursorPos          = user32.NewProc("GetCursorPos")
	procSetTimer              = user32.NewProc("SetTimer")
	procKillTimer             = user32.NewProc("KillTimer")
	procInvalidateRect        = user32.NewProc("InvalidateRect")
	procBeginPaint            = user32.NewProc("BeginPaint")
	procEndPaint              = user32.NewProc("EndPaint")
	procFillRect              = user32.NewProc("FillRect")
	procOpenClipboard         = user32.NewProc("OpenClipboard")
	procCloseClipboard        = user32.NewProc("CloseClipboard")
	procEmptyClipboard        = user32.NewProc("EmptyClipboard")
	procSetClipboardData      = user32.NewProc("SetClipboardData")
	procGetDC                 = user32.NewProc("GetDC")
	procReleaseDC             = user32.NewProc("ReleaseDC")
	procSetForegroundWindow   = user32.NewProc("SetForegroundWindow")
	procTrackMouseEvent       = user32.NewProc("TrackMouseEvent")
)

// --- kernel32 procs ---

var (
	procGetCurrentThreadId = kernel32.NewProc("GetCurrentThreadId")
	procGlobalAlloc        = kernel32.NewProc("GlobalAlloc")
	procGlobalLock         = kernel32.NewProc("GlobalLock")
	procGlobalUnlock       = kernel32.NewProc("GlobalUnlock")
	procGetLastError       = kernel32.NewProc("GetLastError")
	procSetLastError       = kernel32.NewProc("SetLastError")
	procSleep              = kernel32.NewProc("Sleep")
	procCreateEventW       = kernel32.NewProc("CreateEventW")
	procSetEvent           = kernel32.NewProc("SetEvent")
	procResetEvent         = kernel32.NewProc("ResetEvent")
)

// --- gdi32 procs ---

var (
	procCreateCompatibleDC     = gdi32.NewProc("CreateCompatibleDC")
	procCreateCompatibleBitmap = gdi32.NewProc("CreateCompatibleBitmap")
	procDeleteDC               = gdi32.NewProc("DeleteDC")
	procDeleteObject           = gdi32.NewProc("DeleteObject")
	procSelectObject           = gdi32.NewProc("SelectObject")
	procCreateSolidBrush       = gdi32.NewProc("CreateSolidBrush")
	procCreatePen              = gdi32.NewProc("CreatePen")
	procEllipse                = gdi32.NewProc("Ellipse")
	procSetBkMode              = gdi32.NewProc("SetBkMode")
	procBitBlt                 = gdi32.NewProc("BitBlt")
)

// --- msimg32 procs ---

var (
	procAlphaBlend = msimg32.NewProc("AlphaBlend")
)

// --- shell32 procs ---

var (
	procShell_NotifyIconW = shell32.NewProc("Shell_NotifyIconW")
)

// --- winmm procs ---

var (
	procWaveInOpen            = winmm.NewProc("waveInOpen")
	procWaveInClose           = winmm.NewProc("waveInClose")
	procWaveInPrepareHeader   = winmm.NewProc("waveInPrepareHeader")
	procWaveInUnprepareHeader = winmm.NewProc("waveInUnprepareHeader")
	procWaveInAddBuffer       = winmm.NewProc("waveInAddBuffer")
	procWaveInStart           = winmm.NewProc("waveInStart")
	procWaveInStop            = winmm.NewProc("waveInStop")
	procWaveInReset           = winmm.NewProc("waveInReset")
	procWaveInGetNumDevs      = winmm.NewProc("waveInGetNumDevs")
	procWaveInGetDevCapsW     = winmm.NewProc("waveInGetDevCapsW")
)

// --- Constants ---

const (
	// Window messages
	WM_CREATE       = 0x0001
	WM_DESTROY      = 0x0002
	WM_CLOSE        = 0x0010
	WM_PAINT        = 0x000F
	WM_TIMER        = 0x0113
	WM_CHAR         = 0x0102
	WM_KEYDOWN      = 0x0100
	WM_KEYUP        = 0x0101
	WM_LBUTTONDOWN  = 0x0201
	WM_LBUTTONUP    = 0x0202
	WM_MOUSEMOVE    = 0x0200
	WM_MOUSELEAVE   = 0x02A3
	WM_USER         = 0x0400
	WM_APP          = 0x8000
	WM_COMMAND      = 0x0111

	// Tray callback message
	WM_TRAYICON = WM_APP + 1

	// Custom messages
	WM_UPDATE_STATE = WM_APP + 2
	WM_AUDIO_LEVEL  = WM_APP + 3

	// Keyboard hook
	WH_KEYBOARD_LL = 13

	// Virtual keys
	VK_OEM_3    = 0xC0 // Tilde/backtick
	VK_ESCAPE   = 0x1B
	VK_LCONTROL = 0xA2
	VK_RCONTROL = 0xA3

	// PeekMessage flags
	PM_REMOVE  = 0x0001
	PM_NOREMOVE = 0x0000

	// Window styles
	WS_POPUP          = 0x80000000
	WS_VISIBLE        = 0x10000000
	WS_EX_LAYERED     = 0x00080000
	WS_EX_TOPMOST     = 0x00000008
	WS_EX_TOOLWINDOW  = 0x00000080
	WS_EX_TRANSPARENT = 0x00000020
	WS_EX_NOACTIVATE  = 0x08000000

	// SetWindowPos flags
	SWP_NOSIZE     = 0x0001
	SWP_NOMOVE     = 0x0002
	SWP_NOZORDER   = 0x0004
	SWP_NOACTIVATE = 0x0010
	SWP_SHOWWINDOW = 0x0040
	HWND_TOPMOST   = ^uintptr(0) // -1

	// ShowWindow commands
	SW_SHOW     = 5
	SW_HIDE     = 0
	SW_SHOWNA   = 8

	// SystemMetrics
	SM_CXSCREEN = 0
	SM_CYSCREEN = 1

	// Layered window
	LWA_ALPHA    = 0x02
	LWA_COLORKEY = 0x01

	// GDI
	TRANSPARENT  = 1
	SRCCOPY      = 0x00CC0020
	PS_SOLID     = 0
	PS_NULL      = 5

	// Clipboard
	CF_UNICODETEXT = 13
	GMEM_MOVEABLE  = 0x0002

	// Shell_NotifyIcon actions
	NIM_ADD    = 0x00000000
	NIM_MODIFY = 0x00000001
	NIM_DELETE = 0x00000002
	NIF_MESSAGE = 0x00000001
	NIF_ICON    = 0x00000002
	NIF_TIP     = 0x00000004
	NIF_INFO    = 0x00000010
	NOTIFYICON_VERSION_4 = 4
	NIF_SHOWTIP = 0x00000080

	// Tray icon messages
	NIN_SELECT      = 0x0400
	NIN_KEYSELECT   = 0x0401
	WM_CONTEXTMENU  = 0x007B
	WM_RBUTTONUP    = 0x0205

	// waveIn constants
	WAVE_FORMAT_PCM       = 1
	CALLBACK_EVENT        = 0x00050000
	CALLBACK_NULL         = 0x00000000
	WAVE_MAPPER           = 0xFFFFFFFF
	WHDR_DONE             = 0x00000001

	// Menu
	MF_STRING    = 0x00000000
	MF_SEPARATOR = 0x00000800
	MF_CHECKED   = 0x00000008
	MF_UNCHECKED = 0x00000000
	TPM_BOTTOMALIGN = 0x0020
	TPM_LEFTALIGN   = 0x0000

	// TrackMouseEvent
	TME_LEAVE = 0x00000002
)

// --- Structures ---

type MSG struct {
	Hwnd    uintptr
	Message uint32
	WParam  uintptr
	LParam  uintptr
	Time    uint32
	Pt      POINT
}

type POINT struct {
	X, Y int32
}

type RECT struct {
	Left, Top, Right, Bottom int32
}

type KBDLLHOOKSTRUCT struct {
	VkCode      uint32
	ScanCode    uint32
	Flags       uint32
	Time        uint32
	DwExtraInfo uintptr
}

type WNDCLASSEXW struct {
	CbSize        uint32
	Style         uint32
	LpfnWndProc   uintptr
	CbClsExtra    int32
	CbWndExtra    int32
	HInstance     uintptr
	HIcon         uintptr
	HCursor       uintptr
	HbrBackground uintptr
	LpszMenuName  *uint16
	LpszClassName *uint16
	HIconSm      uintptr
}

type PAINTSTRUCT struct {
	Hdc         uintptr
	FErase      int32
	RcPaint     RECT
	FRestore    int32
	FIncUpdate  int32
	RgbReserved [32]byte
}

type WAVEFORMATEX struct {
	FormatTag      uint16
	Channels       uint16
	SamplesPerSec  uint32
	AvgBytesPerSec uint32
	BlockAlign     uint16
	BitsPerSample  uint16
	CbSize         uint16
}

type WAVEHDR struct {
	Data          uintptr
	BufferLength  uint32
	BytesRecorded uint32
	User          uintptr
	Flags         uint32
	Loops         uint32
	Next          uintptr
	Reserved      uintptr
}

type WAVEINCAPSW struct {
	ManufacturerID  uint16
	ProductID       uint16
	DriverVersion   uint32
	ProductName     [32]uint16
	Formats         uint32
	Channels        uint16
	Reserved        uint16
}

type NOTIFYICONDATAW struct {
	CbSize           uint32
	HWnd             uintptr
	UID              uint32
	UFlags           uint32
	UCallbackMessage uint32
	HIcon            uintptr
	SzTip            [128]uint16
	DwState          uint32
	DwStateMask      uint32
	SzInfo           [256]uint16
	UVersion         uint32
	SzInfoTitle      [64]uint16
	DwInfoFlags      uint32
	GuidItem         [16]byte
	HBalloonIcon     uintptr
}

type BLENDFUNCTION struct {
	BlendOp             byte
	BlendFlags          byte
	SourceConstantAlpha byte
	AlphaFormat         byte
}

type SIZE struct {
	CX, CY int32
}

type TRACKMOUSEEVENT struct {
	CbSize      uint32
	DwFlags     uint32
	HwndTrack   uintptr
	DwHoverTime uint32
}

// --- Win32 wrapper functions ---

func SetWindowsHookEx(idHook int, lpfn uintptr, hMod uintptr, dwThreadId uint32) (uintptr, error) {
	procSetLastError.Call(0)
	r, _, err := procSetWindowsHookExW.Call(
		uintptr(idHook), lpfn, hMod, uintptr(dwThreadId),
	)
	if r == 0 {
		return 0, err
	}
	return r, nil
}

func UnhookWindowsHookEx(hook uintptr) {
	procUnhookWindowsHookEx.Call(hook)
}

func CallNextHookEx(hook uintptr, nCode int, wParam, lParam uintptr) uintptr {
	r, _, _ := procCallNextHookEx.Call(hook, uintptr(nCode), wParam, lParam)
	return r
}

func GetAsyncKeyState(vk int) int16 {
	r, _, _ := procGetAsyncKeyState.Call(uintptr(vk))
	return int16(r)
}

func PeekMessage(msg *MSG, hwnd uintptr, msgFilterMin, msgFilterMax, removeMsg uint32) bool {
	r, _, _ := procPeekMessageW.Call(
		uintptr(unsafe.Pointer(msg)), hwnd,
		uintptr(msgFilterMin), uintptr(msgFilterMax), uintptr(removeMsg),
	)
	return r != 0
}

func TranslateMessage(msg *MSG) {
	procTranslateMessage.Call(uintptr(unsafe.Pointer(msg)))
}

func DispatchMessage(msg *MSG) {
	procDispatchMessageW.Call(uintptr(unsafe.Pointer(msg)))
}

func PostQuitMessage(exitCode int) {
	procPostQuitMessage.Call(uintptr(exitCode))
}

func PostMessage(hwnd uintptr, msg uint32, wParam, lParam uintptr) bool {
	r, _, _ := procPostMessageW.Call(hwnd, uintptr(msg), wParam, lParam)
	return r != 0
}

func GetForegroundWindow() uintptr {
	r, _, _ := procGetForegroundWindow.Call()
	return r
}

func GetFocus() uintptr {
	r, _, _ := procGetFocus.Call()
	return r
}

func GetWindowThreadProcessId(hwnd uintptr) uint32 {
	r, _, _ := procGetWindowThreadProcessId.Call(hwnd, 0)
	return uint32(r)
}

func GetCurrentThreadId() uint32 {
	r, _, _ := procGetCurrentThreadId.Call()
	return uint32(r)
}

func AttachThreadInput(idAttach, idAttachTo uint32, attach bool) bool {
	var a uintptr
	if attach {
		a = 1
	}
	r, _, _ := procAttachThreadInput.Call(uintptr(idAttach), uintptr(idAttachTo), a)
	return r != 0
}

func GetSystemMetrics(index int) int {
	r, _, _ := procGetSystemMetrics.Call(uintptr(index))
	return int(r)
}

func GetCursorPos(pt *POINT) bool {
	r, _, _ := procGetCursorPos.Call(uintptr(unsafe.Pointer(pt)))
	return r != 0
}

func RegisterClassEx(wc *WNDCLASSEXW) (uint16, error) {
	r, _, err := procRegisterClassExW.Call(uintptr(unsafe.Pointer(wc)))
	if r == 0 {
		return 0, err
	}
	return uint16(r), nil
}

func CreateWindowEx(exStyle uint32, className, windowName *uint16, style uint32, x, y, w, h int32, parent, menu, instance uintptr) (uintptr, error) {
	r, _, err := procCreateWindowExW.Call(
		uintptr(exStyle),
		uintptr(unsafe.Pointer(className)),
		uintptr(unsafe.Pointer(windowName)),
		uintptr(style),
		uintptr(x), uintptr(y), uintptr(w), uintptr(h),
		parent, menu, instance, 0,
	)
	if r == 0 {
		return 0, err
	}
	return r, nil
}

func DestroyWindow(hwnd uintptr) {
	procDestroyWindow.Call(hwnd)
}

func ShowWindow(hwnd uintptr, cmdShow int) {
	procShowWindow.Call(hwnd, uintptr(cmdShow))
}

func SetWindowPos(hwnd, hwndInsertAfter uintptr, x, y, cx, cy int32, flags uint32) {
	procSetWindowPos.Call(hwnd, hwndInsertAfter, uintptr(x), uintptr(y), uintptr(cx), uintptr(cy), uintptr(flags))
}

func MoveWindow(hwnd uintptr, x, y, w, h int32, repaint bool) {
	var rep uintptr
	if repaint {
		rep = 1
	}
	procMoveWindow.Call(hwnd, uintptr(x), uintptr(y), uintptr(w), uintptr(h), rep)
}

func InvalidateRect(hwnd uintptr, rect *RECT, erase bool) {
	var e uintptr
	if erase {
		e = 1
	}
	procInvalidateRect.Call(hwnd, uintptr(unsafe.Pointer(rect)), e)
}

func BeginPaint(hwnd uintptr, ps *PAINTSTRUCT) uintptr {
	r, _, _ := procBeginPaint.Call(hwnd, uintptr(unsafe.Pointer(ps)))
	return r
}

func EndPaint(hwnd uintptr, ps *PAINTSTRUCT) {
	procEndPaint.Call(hwnd, uintptr(unsafe.Pointer(ps)))
}

func GetDC(hwnd uintptr) uintptr {
	r, _, _ := procGetDC.Call(hwnd)
	return r
}

func ReleaseDC(hwnd uintptr, hdc uintptr) {
	procReleaseDC.Call(hwnd, hdc)
}

func SetTimer(hwnd uintptr, id uintptr, elapse uint32) {
	procSetTimer.Call(hwnd, id, uintptr(elapse), 0)
}

func KillTimer(hwnd uintptr, id uintptr) {
	procKillTimer.Call(hwnd, id)
}

func SetLayeredWindowAttributes(hwnd uintptr, crKey uint32, alpha byte, flags uint32) {
	procSetLayeredWindowAttributes.Call(hwnd, uintptr(crKey), uintptr(alpha), uintptr(flags))
}

func DefWindowProc(hwnd uintptr, msg uint32, wParam, lParam uintptr) uintptr {
	r, _, _ := procDefWindowProcW.Call(hwnd, uintptr(msg), wParam, lParam)
	return r
}

func LoadCursor(instance uintptr, cursorName uintptr) uintptr {
	r, _, _ := procLoadCursorW.Call(instance, cursorName)
	return r
}

// GDI functions

func CreateCompatibleDC(hdc uintptr) uintptr {
	r, _, _ := procCreateCompatibleDC.Call(hdc)
	return r
}

func CreateCompatibleBitmap(hdc uintptr, w, h int32) uintptr {
	r, _, _ := procCreateCompatibleBitmap.Call(hdc, uintptr(w), uintptr(h))
	return r
}

func DeleteDC(hdc uintptr) {
	procDeleteDC.Call(hdc)
}

func DeleteObject(obj uintptr) {
	procDeleteObject.Call(obj)
}

func SelectObject(hdc, obj uintptr) uintptr {
	r, _, _ := procSelectObject.Call(hdc, obj)
	return r
}

func CreateSolidBrush(color uint32) uintptr {
	r, _, _ := procCreateSolidBrush.Call(uintptr(color))
	return r
}

func FillRect(hdc uintptr, rect *RECT, brush uintptr) {
	procFillRect.Call(hdc, uintptr(unsafe.Pointer(rect)), brush)
}

func Ellipse(hdc uintptr, left, top, right, bottom int32) {
	procEllipse.Call(hdc, uintptr(left), uintptr(top), uintptr(right), uintptr(bottom))
}

func CreatePen(style int32, width int32, color uint32) uintptr {
	r, _, _ := procCreatePen.Call(uintptr(style), uintptr(width), uintptr(color))
	return r
}

func SetBkMode(hdc uintptr, mode int32) {
	procSetBkMode.Call(hdc, uintptr(mode))
}

func BitBlt(hdcDest uintptr, x, y, w, h int32, hdcSrc uintptr, srcX, srcY int32, rop uint32) {
	procBitBlt.Call(hdcDest, uintptr(x), uintptr(y), uintptr(w), uintptr(h),
		hdcSrc, uintptr(srcX), uintptr(srcY), uintptr(rop))
}

func UpdateLayeredWindow(hwnd, hdcDst uintptr, ptDst *POINT, size *SIZE, hdcSrc uintptr, ptSrc *POINT, crKey uint32, blend *BLENDFUNCTION, flags uint32) bool {
	r, _, _ := procUpdateLayeredWindow.Call(
		hwnd, hdcDst,
		uintptr(unsafe.Pointer(ptDst)),
		uintptr(unsafe.Pointer(size)),
		hdcSrc,
		uintptr(unsafe.Pointer(ptSrc)),
		uintptr(crKey),
		uintptr(unsafe.Pointer(blend)),
		uintptr(flags),
	)
	return r != 0
}

// Clipboard functions

func OpenClipboard(hwnd uintptr) bool {
	r, _, _ := procOpenClipboard.Call(hwnd)
	return r != 0
}

func CloseClipboard() {
	procCloseClipboard.Call()
}

func EmptyClipboard() {
	procEmptyClipboard.Call()
}

func SetClipboardData(format uint32, hMem uintptr) {
	procSetClipboardData.Call(uintptr(format), hMem)
}

func GlobalAlloc(flags uint32, size uintptr) uintptr {
	r, _, _ := procGlobalAlloc.Call(uintptr(flags), size)
	return r
}

func GlobalLock(hMem uintptr) uintptr {
	r, _, _ := procGlobalLock.Call(hMem)
	return r
}

func GlobalUnlock(hMem uintptr) {
	procGlobalUnlock.Call(hMem)
}

// Shell_NotifyIcon
func ShellNotifyIcon(message uint32, data *NOTIFYICONDATAW) bool {
	r, _, _ := procShell_NotifyIconW.Call(uintptr(message), uintptr(unsafe.Pointer(data)))
	return r != 0
}

// TrackMouseEvent
func TrackMouseEvent(tme *TRACKMOUSEEVENT) bool {
	r, _, _ := procTrackMouseEvent.Call(uintptr(unsafe.Pointer(tme)))
	return r != 0
}

// --- Helpers ---

func UTF16PtrFromString(s string) *uint16 {
	p, _ := syscall.UTF16PtrFromString(s)
	return p
}

func UTF16FromString(s string) []uint16 {
	p, _ := syscall.UTF16FromString(s)
	return p
}

func LOWORD(l uintptr) uint16 {
	return uint16(l)
}

func HIWORD(l uintptr) uint16 {
	return uint16(l >> 16)
}

// RGB creates a COLORREF value (0x00BBGGRR)
func RGB(r, g, b byte) uint32 {
	return uint32(r) | uint32(g)<<8 | uint32(b)<<16
}
