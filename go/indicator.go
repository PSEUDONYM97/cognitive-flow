package main

import (
	"fmt"
	"syscall"
	"unsafe"
)

// Indicator states
const (
	StateLoading    = "loading"
	StateIdle       = "idle"
	StateRecording  = "recording"
	StateProcessing = "processing"
)

// State colors (COLORREF: 0x00BBGGRR)
var stateColors = map[string]uint32{
	StateLoading:    RGB(255, 191, 0),   // Amber
	StateIdle:       RGB(76, 175, 80),   // Green
	StateRecording:  RGB(244, 67, 54),   // Red
	StateProcessing: RGB(255, 191, 0),   // Amber
}

const (
	indicatorSize     = 20  // Diameter of the dot
	indicatorPadding  = 16  // Distance from screen edge
	collapseTimerID   = 1
	heartbeatTimerID  = 2
	collapseDelayMs   = 3000
	heartbeatMs       = 30000
	windowWidth       = indicatorSize + 4 // Extra for glow
	windowHeight      = indicatorSize + 4
)

// Indicator is the floating overlay dot.
type Indicator struct {
	hwnd       uintptr
	className  string
	state      string
	audioLevel float64
	visible    bool
	onClick    func() // Called when indicator is clicked
}

func NewIndicator() *Indicator {
	return &Indicator{
		state:   StateLoading,
		visible: true,
	}
}

// Create registers the window class and creates the overlay window.
func (ind *Indicator) Create() error {
	ind.className = "CognitiveFlowIndicator"
	classNamePtr := UTF16PtrFromString(ind.className)

	wc := WNDCLASSEXW{
		CbSize:        uint32(unsafe.Sizeof(WNDCLASSEXW{})),
		LpfnWndProc:   syscall.NewCallback(indicatorWndProc),
		LpszClassName: classNamePtr,
		HCursor:       LoadCursor(0, 32512), // IDC_ARROW
	}

	if _, err := RegisterClassEx(&wc); err != nil {
		return fmt.Errorf("RegisterClassEx: %v", err)
	}

	// Position: bottom-right corner
	screenW := GetSystemMetrics(SM_CXSCREEN)
	screenH := GetSystemMetrics(SM_CYSCREEN)
	x := int32(screenW) - int32(windowWidth) - int32(indicatorPadding)
	y := int32(screenH) - int32(windowHeight) - int32(indicatorPadding) - 40 // Above taskbar

	hwnd, err := CreateWindowEx(
		WS_EX_LAYERED|WS_EX_TOPMOST|WS_EX_TOOLWINDOW|WS_EX_NOACTIVATE,
		classNamePtr,
		UTF16PtrFromString("CognitiveFlow"),
		WS_POPUP,
		x, y, int32(windowWidth), int32(windowHeight),
		0, 0, 0,
	)
	if err != nil {
		return fmt.Errorf("CreateWindowEx: %v", err)
	}

	ind.hwnd = hwnd
	setIndicatorRef(hwnd, ind)

	// Set transparent background, we'll paint the dot ourselves
	SetLayeredWindowAttributes(hwnd, RGB(0, 0, 0), 0, LWA_COLORKEY)

	ShowWindow(hwnd, SW_SHOWNA)

	// Start heartbeat timer for self-healing
	SetTimer(hwnd, heartbeatTimerID, heartbeatMs)

	// Start collapse timer
	SetTimer(hwnd, collapseTimerID, collapseDelayMs)

	return nil
}

// SetState updates the indicator state and triggers a repaint.
func (ind *Indicator) SetState(state string) {
	ind.state = state

	// Show window if transitioning to an active state
	if state == StateRecording || state == StateProcessing {
		if !ind.visible {
			ShowWindow(ind.hwnd, SW_SHOWNA)
			ind.visible = true
		}
		// Reset collapse timer
		KillTimer(ind.hwnd, collapseTimerID)
	} else if state == StateIdle {
		// Start collapse timer
		SetTimer(ind.hwnd, collapseTimerID, collapseDelayMs)
	}

	// Force repaint
	InvalidateRect(ind.hwnd, nil, true)
}

// SetAudioLevel updates the displayed audio level (0.0-1.0).
func (ind *Indicator) SetAudioLevel(level float64) {
	ind.audioLevel = level
	InvalidateRect(ind.hwnd, nil, true)
}

// Destroy removes the indicator window.
func (ind *Indicator) Destroy() {
	if ind.hwnd != 0 {
		DestroyWindow(ind.hwnd)
		ind.hwnd = 0
	}
}

func (ind *Indicator) paint(hdc uintptr) {
	// Fill with transparent color (black, our color key)
	rect := RECT{0, 0, windowWidth, windowHeight}
	blackBrush := CreateSolidBrush(RGB(0, 0, 0))
	FillRect(hdc, &rect, blackBrush)
	DeleteObject(blackBrush)

	// Draw the status dot
	color, ok := stateColors[ind.state]
	if !ok {
		color = stateColors[StateIdle]
	}

	brush := CreateSolidBrush(color)
	pen := CreatePen(PS_NULL, 0, 0)
	oldBrush := SelectObject(hdc, brush)
	oldPen := SelectObject(hdc, pen)

	// Center the dot in the window
	margin := int32(2)
	Ellipse(hdc, margin, margin, int32(windowWidth)-margin, int32(windowHeight)-margin)

	SelectObject(hdc, oldBrush)
	SelectObject(hdc, oldPen)
	DeleteObject(brush)
	DeleteObject(pen)
}

func (ind *Indicator) heartbeat() {
	if ind.hwnd == 0 {
		return
	}

	// Check if still visible and on-screen
	var rect RECT
	procGetWindowRect.Call(ind.hwnd, uintptr(unsafe.Pointer(&rect)))

	screenW := int32(GetSystemMetrics(SM_CXSCREEN))
	screenH := int32(GetSystemMetrics(SM_CYSCREEN))

	// If completely off-screen, reposition
	if rect.Left >= screenW || rect.Top >= screenH || rect.Right <= 0 || rect.Bottom <= 0 {
		x := screenW - int32(windowWidth) - int32(indicatorPadding)
		y := screenH - int32(windowHeight) - int32(indicatorPadding) - 40
		MoveWindow(ind.hwnd, x, y, int32(windowWidth), int32(windowHeight), true)
	}

	// Ensure visible during active states
	if ind.state == StateRecording || ind.state == StateProcessing {
		ShowWindow(ind.hwnd, SW_SHOWNA)
		SetWindowPos(ind.hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE|SWP_NOSIZE|SWP_NOACTIVATE)
	}
}

// --- Window procedure and instance management ---

var indicatorRefs = map[uintptr]*Indicator{}

func setIndicatorRef(hwnd uintptr, ind *Indicator) {
	indicatorRefs[hwnd] = ind
}

func getIndicatorRef(hwnd uintptr) *Indicator {
	return indicatorRefs[hwnd]
}

func indicatorWndProc(hwnd, msg, wParam, lParam uintptr) uintptr {
	ind := getIndicatorRef(hwnd)

	switch uint32(msg) {
	case WM_PAINT:
		var ps PAINTSTRUCT
		hdc := BeginPaint(hwnd, &ps)
		if ind != nil {
			ind.paint(hdc)
		}
		EndPaint(hwnd, &ps)
		return 0

	case WM_LBUTTONUP:
		if ind != nil && ind.onClick != nil {
			ind.onClick()
		}
		return 0

	case WM_TIMER:
		if ind == nil {
			return 0
		}
		switch wParam {
		case collapseTimerID:
			// Could hide or shrink. For now just ensure topmost.
			KillTimer(hwnd, collapseTimerID)
		case heartbeatTimerID:
			ind.heartbeat()
		}
		return 0

	case WM_DESTROY:
		delete(indicatorRefs, hwnd)
		return 0
	}

	return DefWindowProc(hwnd, uint32(msg), wParam, lParam)
}
