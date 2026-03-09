package main

import (
	"fmt"
	"os"
)

func main() {
	// Debug mode: --debug flag for verbose console output
	debug := false
	for _, arg := range os.Args[1:] {
		if arg == "--debug" {
			debug = true
		}
	}

	if !debug {
		// TODO: Detach from console (FreeConsole) for background mode
		// For now, always run in foreground
	}

	app := NewApp()

	if err := app.Init(); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to initialize: %v\n", err)
		os.Exit(1)
	}
	defer app.Shutdown()

	// Run message loop (blocks until quit)
	app.Run()
}
