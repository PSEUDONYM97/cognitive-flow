package main

import (
	"fmt"
	"time"
	"unicode/utf16"
	"unsafe"
)

// TypeText sends text to the foreground window via WM_CHAR messages.
// Attaches to the foreground thread to get the focused control.
func TypeText(text string) error {
	text = SanitizeForInjection(text)
	if len(text) == 0 {
		return nil
	}

	fg := GetForegroundWindow()
	if fg == 0 {
		return fmt.Errorf("no foreground window")
	}

	fgThread := GetWindowThreadProcessId(fg)
	myThread := GetCurrentThreadId()

	var target uintptr

	if fgThread != myThread {
		AttachThreadInput(myThread, fgThread, true)
		target = GetFocus()
		if target == 0 {
			target = fg // Fall back to foreground window
		}
		defer AttachThreadInput(myThread, fgThread, false)
	} else {
		target = GetFocus()
		if target == 0 {
			target = fg
		}
	}

	// Encode to UTF-16 for WM_CHAR
	runes := []rune(text)
	count := 0

	for _, r := range runes {
		if r == '\n' {
			// Send Enter as \r
			PostMessage(target, WM_CHAR, 13, 0)
			count++
		} else {
			// Encode as UTF-16 (handles surrogate pairs)
			u16 := utf16.Encode([]rune{r})
			for _, c := range u16 {
				PostMessage(target, WM_CHAR, uintptr(c), 0)
				count++
			}
		}

		// Batch pause every 100 chars to prevent message flood
		if count%100 == 0 {
			time.Sleep(1 * time.Millisecond)
		}
	}

	return nil
}

// CopyToClipboard copies text to the Windows clipboard using Win32 API.
func CopyToClipboard(text string) error {
	// Null-terminate and encode as UTF-16LE
	u16 := utf16.Encode([]rune(text + "\x00"))
	size := len(u16) * 2

	if !OpenClipboard(0) {
		return fmt.Errorf("OpenClipboard failed")
	}
	defer CloseClipboard()

	EmptyClipboard()

	hMem := GlobalAlloc(GMEM_MOVEABLE, uintptr(size))
	if hMem == 0 {
		return fmt.Errorf("GlobalAlloc failed")
	}

	ptr := GlobalLock(hMem)
	if ptr == 0 {
		return fmt.Errorf("GlobalLock failed")
	}

	// Copy UTF-16 data
	dst := unsafe.Slice((*uint16)(unsafe.Pointer(ptr)), len(u16))
	copy(dst, u16)

	GlobalUnlock(hMem)
	SetClipboardData(CF_UNICODETEXT, hMem)

	return nil
}
