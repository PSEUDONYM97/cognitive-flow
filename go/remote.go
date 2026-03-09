package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"time"
)

// RemoteClient handles HTTP communication with the STT server.
type RemoteClient struct {
	URL            string
	ServerModel    string
	ServerGPU      bool
	ServerReady    bool
	WarmupDone     bool
	LastTimings    NetworkTimings

	client *http.Client
}

type NetworkTimings struct {
	EncodeMs   float64
	PayloadKB  float64
	NetworkMs  float64
	ServerMs   float64
	OverheadMs float64
}

type HealthResponse struct {
	Status string `json:"status"`
	Model  string `json:"model"`
	GPU    string `json:"gpu"`
}

type TranscribeResponse struct {
	Text             string  `json:"text"`
	ProcessingTimeMs float64 `json:"processing_time_ms"`
}

func NewRemoteClient(url string) *RemoteClient {
	return &RemoteClient{
		URL: url,
		client: &http.Client{
			Timeout: 120 * time.Second,
		},
	}
}

// HealthCheck pings the server and returns its status.
func (rc *RemoteClient) HealthCheck() (*HealthResponse, error) {
	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Get(rc.URL + "/health")
	if err != nil {
		rc.ServerReady = false
		return nil, fmt.Errorf("health check failed: %w", err)
	}
	defer resp.Body.Close()

	var health HealthResponse
	if err := json.NewDecoder(resp.Body).Decode(&health); err != nil {
		return nil, fmt.Errorf("parse health response: %w", err)
	}

	rc.ServerModel = health.Model
	rc.ServerGPU = health.GPU == "true"
	rc.ServerReady = health.Status == "ready" || health.Status == "idle"

	return &health, nil
}

// Warmup pings the server to wake it up. If idle, sends a tiny audio clip to trigger model load.
func (rc *RemoteClient) Warmup() error {
	health, err := rc.HealthCheck()
	if err != nil {
		return err
	}

	if health.Status == "idle" {
		// Server is up but model may need loading - send tiny audio to trigger
		silence := make([]int16, 1600) // 0.1s of silence at 16kHz
		wav := encodeWAV(silence, 16000)

		client := &http.Client{Timeout: 120 * time.Second}
		body, contentType := buildMultipart("audio", "audio.wav", wav)
		resp, err := client.Post(rc.URL+"/transcribe", contentType, body)
		if err != nil {
			return fmt.Errorf("warmup transcribe failed: %w", err)
		}
		resp.Body.Close()
	}

	rc.WarmupDone = true
	return nil
}

// Transcribe sends audio to the server and returns the transcribed text.
// audioInt16 is 16-bit signed PCM at the given sample rate.
func (rc *RemoteClient) Transcribe(audioInt16 []int16, sampleRate int) (*TranscribeResponse, error) {
	// Determine timeout
	timeout := 120 * time.Second // cold start
	if rc.WarmupDone {
		timeout = 30 * time.Second // warm server
	} else if !rc.ServerReady {
		timeout = 10 * time.Second // fast fail
	}

	// Encode WAV
	encodeStart := time.Now()
	wav := encodeWAV(audioInt16, sampleRate)
	encodeMs := float64(time.Since(encodeStart).Microseconds()) / 1000.0

	// Build multipart
	body, contentType := buildMultipart("audio", "audio.wav", wav)

	// POST
	client := &http.Client{Timeout: timeout}
	netStart := time.Now()

	var resp *http.Response
	var err error

	retryDelays := []time.Duration{5 * time.Second, 10 * time.Second, 15 * time.Second}
	maxRetries := 0
	if rc.WarmupDone {
		maxRetries = 3
	}

	for attempt := 0; attempt <= maxRetries; attempt++ {
		if attempt > 0 {
			delay := retryDelays[attempt-1]
			fmt.Printf("[Remote] Retry %d/%d in %v...\n", attempt, maxRetries, delay)
			time.Sleep(delay)

			// Rebuild body for retry (reader was consumed)
			body, contentType = buildMultipart("audio", "audio.wav", wav)
			netStart = time.Now()
		}

		resp, err = client.Post(rc.URL+"/transcribe", contentType, body)
		if err == nil {
			break
		}

		// Only retry on transient connection errors
		if attempt >= maxRetries {
			return nil, fmt.Errorf("transcribe failed after %d retries: %w", maxRetries, err)
		}
		fmt.Printf("[Remote] Connection error: %v\n", err)
	}

	defer resp.Body.Close()
	networkMs := float64(time.Since(netStart).Microseconds()) / 1000.0

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("server returned %d: %s", resp.StatusCode, string(respBody))
	}

	var result TranscribeResponse
	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, fmt.Errorf("parse response: %w", err)
	}

	// Record timings
	rc.LastTimings = NetworkTimings{
		EncodeMs:   encodeMs,
		PayloadKB:  float64(len(wav)) / 1024.0,
		NetworkMs:  networkMs,
		ServerMs:   result.ProcessingTimeMs,
		OverheadMs: networkMs - result.ProcessingTimeMs,
	}

	return &result, nil
}

// encodeWAV converts int16 PCM samples to a WAV byte slice.
func encodeWAV(samples []int16, sampleRate int) []byte {
	var buf bytes.Buffer

	numSamples := len(samples)
	dataSize := numSamples * 2 // 16-bit = 2 bytes per sample
	fileSize := 36 + dataSize

	// RIFF header
	buf.WriteString("RIFF")
	binary.Write(&buf, binary.LittleEndian, uint32(fileSize))
	buf.WriteString("WAVE")

	// fmt chunk
	buf.WriteString("fmt ")
	binary.Write(&buf, binary.LittleEndian, uint32(16))        // chunk size
	binary.Write(&buf, binary.LittleEndian, uint16(1))         // PCM format
	binary.Write(&buf, binary.LittleEndian, uint16(1))         // mono
	binary.Write(&buf, binary.LittleEndian, uint32(sampleRate))
	binary.Write(&buf, binary.LittleEndian, uint32(sampleRate*2)) // byte rate
	binary.Write(&buf, binary.LittleEndian, uint16(2))           // block align
	binary.Write(&buf, binary.LittleEndian, uint16(16))          // bits per sample

	// data chunk
	buf.WriteString("data")
	binary.Write(&buf, binary.LittleEndian, uint32(dataSize))

	for _, s := range samples {
		binary.Write(&buf, binary.LittleEndian, s)
	}

	return buf.Bytes()
}

// buildMultipart creates a multipart form body with a single file field.
func buildMultipart(fieldName, fileName string, data []byte) (io.Reader, string) {
	boundary := "----CognitiveFlowBoundary9876543210"

	var buf bytes.Buffer
	buf.WriteString("--" + boundary + "\r\n")
	buf.WriteString(fmt.Sprintf("Content-Disposition: form-data; name=\"%s\"; filename=\"%s\"\r\n", fieldName, fileName))
	buf.WriteString("Content-Type: audio/wav\r\n")
	buf.WriteString("\r\n")
	buf.Write(data)
	buf.WriteString("\r\n")
	buf.WriteString("--" + boundary + "--\r\n")

	contentType := "multipart/form-data; boundary=" + boundary
	return &buf, contentType
}

// Float32ToInt16 converts float32 audio [-1.0, 1.0] to int16.
func Float32ToInt16(audio []float32) []int16 {
	result := make([]int16, len(audio))
	for i, s := range audio {
		clamped := s
		if clamped > 1.0 {
			clamped = 1.0
		} else if clamped < -1.0 {
			clamped = -1.0
		}
		result[i] = int16(clamped * math.MaxInt16)
	}
	return result
}
