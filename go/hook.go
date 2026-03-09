package main

import (
	"syscall"
	"unsafe"
)

// HookCallback is called by the Windows keyboard hook.
// Return 1 to block the key, or 0 to pass through.
type HookCallback func(vkCode uint32, down bool) int

// KeyboardHook manages a low-level keyboard hook.
type KeyboardHook struct {
	handle   uintptr
	callback HookCallback
}

var globalHook *KeyboardHook

// InstallKeyboardHook installs a global low-level keyboard hook.
// Must be called from the thread that runs the message loop.
func InstallKeyboardHook(cb HookCallback) (*KeyboardHook, error) {
	hook := &KeyboardHook{callback: cb}
	globalHook = hook

	proc := syscall.NewCallback(hookProc)
	h, err := SetWindowsHookEx(WH_KEYBOARD_LL, proc, 0, 0)
	if err != nil {
		return nil, err
	}

	hook.handle = h
	return hook, nil
}

func (h *KeyboardHook) Uninstall() {
	if h.handle != 0 {
		UnhookWindowsHookEx(h.handle)
		h.handle = 0
	}
	if globalHook == h {
		globalHook = nil
	}
}

func hookProc(nCode int, wParam uintptr, lParam uintptr) uintptr {
	if nCode >= 0 && globalHook != nil {
		kb := (*KBDLLHOOKSTRUCT)(unsafe.Pointer(lParam))

		down := wParam == WM_KEYDOWN
		result := globalHook.callback(kb.VkCode, down)

		if result == 1 {
			return 1 // Block the key
		}
	}

	return CallNextHookEx(0, nCode, wParam, lParam)
}
