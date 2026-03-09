package main

import (
	"image"
	"image/color"
	"image/draw"
	"syscall"
	"unsafe"
)

// Tray manages the system tray icon and menu.
type Tray struct {
	hwnd      uintptr // Hidden window for tray messages
	nid       NOTIFYICONDATAW
	className string

	onSettings       func()
	onToggleHotkey   func()
	onToggleOverlay  func()
	onResetOverlay   func()
	onQuit           func()
	hotkeyEnabled    bool
	overlayVisible   bool
}

const (
	// Menu item IDs
	menuSettings      = 1001
	menuToggleHotkey  = 1002
	menuToggleOverlay = 1003
	menuResetOverlay  = 1004
	menuQuit          = 1005
)

// Tray icon additional procs
var (
	procCreatePopupMenu    = user32.NewProc("CreatePopupMenu")
	procAppendMenuW        = user32.NewProc("AppendMenuW")
	procTrackPopupMenu     = user32.NewProc("TrackPopupMenu")
	procDestroyMenu        = user32.NewProc("DestroyMenu")
	procCreateIconIndirect = user32.NewProc("CreateIconIndirect")
	procDestroyIcon        = user32.NewProc("DestroyIcon")
)

type ICONINFO struct {
	FIcon    int32
	XHotspot uint32
	YHotspot uint32
	HbmMask  uintptr
	HbmColor uintptr
}

var (
	procCreateDIBSection = gdi32.NewProc("CreateDIBSection")
)

type BITMAPINFOHEADER struct {
	BiSize          uint32
	BiWidth         int32
	BiHeight        int32
	BiPlanes        uint16
	BiBitCount      uint16
	BiCompression   uint32
	BiSizeImage     uint32
	BiXPelsPerMeter int32
	BiYPelsPerMeter int32
	BiClrUsed       uint32
	BiClrImportant  uint32
}

type BITMAPINFO struct {
	BmiHeader BITMAPINFOHEADER
	BmiColors [1]uint32
}

func NewTray() *Tray {
	return &Tray{
		hotkeyEnabled:  true,
		overlayVisible: true,
	}
}

// Create sets up the system tray icon with a hidden message window.
func (t *Tray) Create() error {
	t.className = "CognitiveFlowTray"

	// Register hidden window class
	wc := WNDCLASSEXW{
		CbSize:        uint32(unsafe.Sizeof(WNDCLASSEXW{})),
		LpfnWndProc:   0, // Set below
		LpszClassName: UTF16PtrFromString(t.className),
	}
	wc.LpfnWndProc = trayWndProcCallback
	RegisterClassEx(&wc)

	// Create hidden window
	hwnd, _ := CreateWindowEx(
		0,
		UTF16PtrFromString(t.className),
		UTF16PtrFromString("CognitiveFlowTrayHost"),
		0, 0, 0, 0, 0,
		0, 0, 0,
	)
	t.hwnd = hwnd
	setTrayRef(hwnd, t)

	// Create tray icon
	t.nid = NOTIFYICONDATAW{
		CbSize:           uint32(unsafe.Sizeof(NOTIFYICONDATAW{})),
		HWnd:             hwnd,
		UID:              1,
		UFlags:           NIF_MESSAGE | NIF_ICON | NIF_TIP | NIF_SHOWTIP,
		UCallbackMessage: WM_TRAYICON,
	}

	// Set tooltip
	tooltip := UTF16FromString("Cognitive Flow v" + appVersion)
	copy(t.nid.SzTip[:], tooltip)

	// Create initial icon (green = idle)
	t.nid.HIcon = createColorIcon(76, 175, 80)

	ShellNotifyIcon(NIM_ADD, &t.nid)

	return nil
}

// SetIcon updates the tray icon color based on state.
func (t *Tray) SetIcon(state string) {
	var r, g, b byte
	switch state {
	case StateRecording:
		r, g, b = 244, 67, 54 // Red
	case StateLoading, StateProcessing:
		r, g, b = 255, 191, 0 // Amber
	default:
		r, g, b = 76, 175, 80 // Green
	}

	oldIcon := t.nid.HIcon
	t.nid.HIcon = createColorIcon(r, g, b)
	t.nid.UFlags = NIF_ICON
	ShellNotifyIcon(NIM_MODIFY, &t.nid)

	if oldIcon != 0 {
		procDestroyIcon.Call(oldIcon)
	}
}

// SetTooltip updates the tray tooltip text.
func (t *Tray) SetTooltip(text string) {
	tooltip := UTF16FromString(text)
	copy(t.nid.SzTip[:], tooltip)
	t.nid.UFlags = NIF_TIP | NIF_SHOWTIP
	ShellNotifyIcon(NIM_MODIFY, &t.nid)
}

// Destroy removes the tray icon.
func (t *Tray) Destroy() {
	ShellNotifyIcon(NIM_DELETE, &t.nid)
	if t.nid.HIcon != 0 {
		procDestroyIcon.Call(t.nid.HIcon)
	}
	if t.hwnd != 0 {
		DestroyWindow(t.hwnd)
	}
}

func (t *Tray) showMenu() {
	hMenu, _, _ := procCreatePopupMenu.Call()
	if hMenu == 0 {
		return
	}

	appendMenu(hMenu, MF_STRING, menuSettings, "Settings...")

	hotkeyFlags := uint32(MF_STRING)
	if t.hotkeyEnabled {
		hotkeyFlags |= MF_CHECKED
	}
	appendMenu(hMenu, hotkeyFlags, menuToggleHotkey, "Hotkey Enabled")

	overlayFlags := uint32(MF_STRING)
	if t.overlayVisible {
		overlayFlags |= MF_CHECKED
	}
	appendMenu(hMenu, overlayFlags, menuToggleOverlay, "Show Overlay")

	appendMenu(hMenu, MF_STRING, menuResetOverlay, "Reset Overlay Position")
	appendMenu(hMenu, MF_SEPARATOR, 0, "")
	appendMenu(hMenu, MF_STRING, menuQuit, "Quit")

	var pt POINT
	GetCursorPos(&pt)

	// Required: set foreground before TrackPopupMenu so it closes properly
	procSetForegroundWindow.Call(t.hwnd)

	procTrackPopupMenu.Call(
		hMenu,
		uintptr(TPM_BOTTOMALIGN|TPM_LEFTALIGN),
		uintptr(pt.X), uintptr(pt.Y),
		0, t.hwnd, 0,
	)

	procDestroyMenu.Call(hMenu)
}

func appendMenu(hMenu uintptr, flags uint32, id uintptr, text string) {
	procAppendMenuW.Call(hMenu, uintptr(flags), id, uintptr(unsafe.Pointer(UTF16PtrFromString(text))))
}

// createColorIcon creates a small 16x16 solid-color icon.
func createColorIcon(r, g, b byte) uintptr {
	size := 16

	// Create an RGBA image
	img := image.NewRGBA(image.Rect(0, 0, size, size))
	c := color.RGBA{R: r, G: g, B: b, A: 255}

	// Draw a filled circle
	cx, cy := float64(size)/2, float64(size)/2
	radius := float64(size)/2 - 1
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			dx := float64(x) - cx + 0.5
			dy := float64(y) - cy + 0.5
			if dx*dx+dy*dy <= radius*radius {
				img.SetRGBA(x, y, c)
			}
		}
	}

	return createHICONFromRGBA(img)
}

func createHICONFromRGBA(img *image.RGBA) uintptr {
	w := img.Bounds().Dx()
	h := img.Bounds().Dy()

	// Create color bitmap (BGRA)
	hdc := GetDC(0)
	bmi := BITMAPINFO{
		BmiHeader: BITMAPINFOHEADER{
			BiSize:      uint32(unsafe.Sizeof(BITMAPINFOHEADER{})),
			BiWidth:     int32(w),
			BiHeight:    -int32(h), // Top-down
			BiPlanes:    1,
			BiBitCount:  32,
		},
	}

	var bits uintptr
	hbmColor, _, _ := procCreateDIBSection.Call(
		hdc, uintptr(unsafe.Pointer(&bmi)), 0,
		uintptr(unsafe.Pointer(&bits)), 0, 0,
	)
	ReleaseDC(0, hdc)

	if hbmColor == 0 || bits == 0 {
		return 0
	}

	// Copy RGBA -> BGRA
	pixelData := unsafe.Slice((*byte)(unsafe.Pointer(bits)), w*h*4)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			srcOff := y*img.Stride + x*4
			dstOff := (y*w + x) * 4
			pixelData[dstOff+0] = img.Pix[srcOff+2] // B
			pixelData[dstOff+1] = img.Pix[srcOff+1] // G
			pixelData[dstOff+2] = img.Pix[srcOff+0] // R
			pixelData[dstOff+3] = img.Pix[srcOff+3] // A
		}
	}

	// Create mask bitmap (all zeros = fully opaque when using 32-bit color)
	maskImg := image.NewRGBA(image.Rect(0, 0, w, h))
	draw.Draw(maskImg, maskImg.Bounds(), image.Transparent, image.Point{}, draw.Src)

	hdc2 := GetDC(0)
	bmiMask := bmi
	var maskBits uintptr
	hbmMask, _, _ := procCreateDIBSection.Call(
		hdc2, uintptr(unsafe.Pointer(&bmiMask)), 0,
		uintptr(unsafe.Pointer(&maskBits)), 0, 0,
	)
	ReleaseDC(0, hdc2)

	iconInfo := ICONINFO{
		FIcon:   1, // TRUE = icon
		HbmMask: hbmMask,
		HbmColor: hbmColor,
	}

	hIcon, _, _ := procCreateIconIndirect.Call(uintptr(unsafe.Pointer(&iconInfo)))

	DeleteObject(hbmColor)
	DeleteObject(hbmMask)

	return hIcon
}

// --- Tray window procedure ---

var trayWndProcCallback = newTrayWndProc()

var trayRefs = map[uintptr]*Tray{}

func setTrayRef(hwnd uintptr, t *Tray) {
	trayRefs[hwnd] = t
}

func getTrayRef(hwnd uintptr) *Tray {
	return trayRefs[hwnd]
}

func newTrayWndProc() uintptr {
	return newCallback(func(hwnd uintptr, msg uint32, wParam, lParam uintptr) uintptr {
		t := getTrayRef(hwnd)

		switch msg {
		case WM_TRAYICON:
			if t == nil {
				break
			}
			switch LOWORD(lParam) {
			case uint16(WM_RBUTTONUP):
				t.showMenu()
			case uint16(NIN_SELECT), uint16(NIN_KEYSELECT):
				// Left-click: toggle settings or do nothing
			}
			return 0

		case WM_COMMAND:
			if t == nil {
				break
			}
			switch LOWORD(wParam) {
			case menuSettings:
				if t.onSettings != nil {
					t.onSettings()
				}
			case menuToggleHotkey:
				if t.onToggleHotkey != nil {
					t.onToggleHotkey()
				}
			case menuToggleOverlay:
				if t.onToggleOverlay != nil {
					t.onToggleOverlay()
				}
			case menuResetOverlay:
				if t.onResetOverlay != nil {
					t.onResetOverlay()
				}
			case menuQuit:
				if t.onQuit != nil {
					t.onQuit()
				}
			}
			return 0

		case WM_DESTROY:
			delete(trayRefs, hwnd)
			return 0
		}

		return DefWindowProc(hwnd, msg, wParam, lParam)
	})
}

func newCallback(fn func(hwnd uintptr, msg uint32, wParam, lParam uintptr) uintptr) uintptr {
	return syscall.NewCallback(func(hwnd, msg, wParam, lParam uintptr) uintptr {
		return fn(hwnd, uint32(msg), wParam, lParam)
	})
}
