package main

import (
	"fmt"
	"math"
	"sync"
	"unsafe"
)

const (
	sampleRate = 16000
	channels   = 1
	bitsPerSample = 16
	chunkSize  = 1024 // samples per buffer
	numBuffers = 4    // double-double buffering
)

// Recorder captures audio from a waveIn device.
type Recorder struct {
	mu        sync.Mutex
	hwi       uintptr
	buffers   [numBuffers][]byte
	headers   [numBuffers]WAVEHDR
	event     uintptr
	recording bool
	frames    []int16
	onLevel   func(float64) // callback for audio level updates
}

func NewRecorder() *Recorder {
	return &Recorder{}
}

// ListDevices returns available audio input devices.
func ListDevices() []string {
	r, _, _ := procWaveInGetNumDevs.Call()
	count := int(r)
	devices := make([]string, 0, count)

	for i := 0; i < count; i++ {
		var caps WAVEINCAPSW
		ret, _, _ := procWaveInGetDevCapsW.Call(
			uintptr(i),
			uintptr(unsafe.Pointer(&caps)),
			unsafe.Sizeof(caps),
		)
		if ret == 0 {
			name := ""
			for _, c := range caps.ProductName {
				if c == 0 {
					break
				}
				name += string(rune(c))
			}
			devices = append(devices, name)
		}
	}

	return devices
}

// Start begins recording audio from the specified device (-1 for default).
func (r *Recorder) Start(deviceIndex int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.recording {
		return fmt.Errorf("already recording")
	}

	// Create event for waveIn callbacks
	event, _, err := procCreateEventW.Call(0, 0, 0, 0)
	if event == 0 {
		return fmt.Errorf("CreateEvent: %v", err)
	}
	r.event = event

	// Set up wave format
	wfx := WAVEFORMATEX{
		FormatTag:      WAVE_FORMAT_PCM,
		Channels:       channels,
		SamplesPerSec:  sampleRate,
		BitsPerSample:  bitsPerSample,
		BlockAlign:     channels * bitsPerSample / 8,
		AvgBytesPerSec: sampleRate * uint32(channels) * uint32(bitsPerSample) / 8,
		CbSize:         0,
	}

	// Determine device ID
	devID := uint32(WAVE_MAPPER)
	if deviceIndex >= 0 {
		devID = uint32(deviceIndex)
	}

	// Open waveIn device
	ret, _, _ := procWaveInOpen.Call(
		uintptr(unsafe.Pointer(&r.hwi)),
		uintptr(devID),
		uintptr(unsafe.Pointer(&wfx)),
		event,
		0,
		CALLBACK_EVENT,
	)
	if ret != 0 {
		return fmt.Errorf("waveInOpen failed: MMRESULT %d", ret)
	}

	// Allocate and prepare buffers
	bufSize := chunkSize * int(channels) * int(bitsPerSample) / 8
	for i := 0; i < numBuffers; i++ {
		r.buffers[i] = make([]byte, bufSize)
		r.headers[i] = WAVEHDR{
			Data:         uintptr(unsafe.Pointer(&r.buffers[i][0])),
			BufferLength: uint32(bufSize),
		}

		ret, _, _ = procWaveInPrepareHeader.Call(
			r.hwi,
			uintptr(unsafe.Pointer(&r.headers[i])),
			unsafe.Sizeof(r.headers[i]),
		)
		if ret != 0 {
			r.cleanup()
			return fmt.Errorf("waveInPrepareHeader failed: MMRESULT %d", ret)
		}

		ret, _, _ = procWaveInAddBuffer.Call(
			r.hwi,
			uintptr(unsafe.Pointer(&r.headers[i])),
			unsafe.Sizeof(r.headers[i]),
		)
		if ret != 0 {
			r.cleanup()
			return fmt.Errorf("waveInAddBuffer failed: MMRESULT %d", ret)
		}
	}

	r.frames = nil
	r.recording = true

	// Start recording
	ret, _, _ = procWaveInStart.Call(r.hwi)
	if ret != 0 {
		r.cleanup()
		return fmt.Errorf("waveInStart failed: MMRESULT %d", ret)
	}

	// Background goroutine to process filled buffers
	go r.processLoop()

	return nil
}

// Stop ends recording and returns the captured audio as int16 samples.
func (r *Recorder) Stop() ([]int16, error) {
	r.mu.Lock()
	if !r.recording {
		r.mu.Unlock()
		return nil, fmt.Errorf("not recording")
	}
	r.recording = false
	r.mu.Unlock()

	// Stop and reset the device (flushes remaining buffers)
	procWaveInStop.Call(r.hwi)
	procWaveInReset.Call(r.hwi)

	// Signal the event to wake up processLoop so it can exit
	procSetEvent.Call(r.event)

	// Give processLoop a moment to drain
	// (It will exit because r.recording is false)

	r.mu.Lock()
	defer r.mu.Unlock()

	r.cleanup()

	return r.frames, nil
}

func (r *Recorder) processLoop() {
	for {
		r.mu.Lock()
		if !r.recording {
			r.mu.Unlock()
			return
		}
		hwi := r.hwi
		event := r.event
		r.mu.Unlock()

		// Wait for a buffer to be filled (100ms timeout for responsive shutdown)
		waitForSingleObject(event, 100)

		r.mu.Lock()
		if !r.recording {
			r.mu.Unlock()
			return
		}

		for i := 0; i < numBuffers; i++ {
			if r.headers[i].Flags&WHDR_DONE != 0 {
				// Copy audio data
				recorded := r.headers[i].BytesRecorded
				if recorded > 0 {
					buf := make([]byte, recorded)
					copy(buf, r.buffers[i][:recorded])

					// Convert bytes to int16 samples
					samples := bytesToInt16(buf)
					r.frames = append(r.frames, samples...)

					// Calculate RMS and notify
					if r.onLevel != nil {
						level := calculateLevel(samples)
						r.onLevel(level)
					}
				}

				// Re-queue buffer
				r.headers[i].Flags = 0
				r.headers[i].BytesRecorded = 0
				procWaveInAddBuffer.Call(
					hwi,
					uintptr(unsafe.Pointer(&r.headers[i])),
					unsafe.Sizeof(r.headers[i]),
				)
			}
		}
		r.mu.Unlock()
	}
}

func (r *Recorder) cleanup() {
	if r.hwi == 0 {
		return
	}

	for i := 0; i < numBuffers; i++ {
		procWaveInUnprepareHeader.Call(
			r.hwi,
			uintptr(unsafe.Pointer(&r.headers[i])),
			unsafe.Sizeof(r.headers[i]),
		)
	}

	procWaveInClose.Call(r.hwi)
	r.hwi = 0

	if r.event != 0 {
		// Close handle
		r.event = 0
	}
}

// IsRecording returns whether the recorder is actively capturing audio.
func (r *Recorder) IsRecording() bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.recording
}

var procWaitForSingleObject = kernel32.NewProc("WaitForSingleObject")

func waitForSingleObject(handle uintptr, milliseconds uint32) uint32 {
	r, _, _ := procWaitForSingleObject.Call(handle, uintptr(milliseconds))
	return uint32(r)
}

func bytesToInt16(data []byte) []int16 {
	n := len(data) / 2
	result := make([]int16, n)
	for i := 0; i < n; i++ {
		result[i] = int16(data[i*2]) | int16(data[i*2+1])<<8
	}
	return result
}

// calculateLevel returns a 0.0-1.0 audio level from int16 samples (log scale).
func calculateLevel(samples []int16) float64 {
	if len(samples) == 0 {
		return 0
	}

	var sum float64
	for _, s := range samples {
		f := float64(s)
		sum += f * f
	}
	rms := math.Sqrt(sum / float64(len(samples)))

	// Log scale: map rms (0-32768) to (0-1) with 4 decades of dynamic range
	if rms < 1 {
		return 0
	}
	level := (math.Log10(rms/32768.0) + 4.0) / 4.0
	if level < 0 {
		level = 0
	}
	if level > 1 {
		level = 1
	}
	return level
}
