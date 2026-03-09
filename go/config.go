package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

const (
	appName    = "CognitiveFlow"
	appVersion = "2.0.0"
)

type Config struct {
	RemoteURL        string            `json:"remote_url"`
	AddTrailingSpace bool              `json:"add_trailing_space"`
	InputDeviceIndex int               `json:"input_device_index"` // -1 = default
	ShowOverlay      bool              `json:"show_overlay"`
	ArchiveAudio     bool              `json:"archive_audio"`
	TextReplacements map[string]string `json:"text_replacements"`
	PauseMedia       bool              `json:"pause_media"`
	HotkeyEnabled    bool              `json:"hotkey_enabled"`
}

func DefaultConfig() *Config {
	return &Config{
		RemoteURL:        "http://192.168.0.10:9200",
		AddTrailingSpace: true,
		InputDeviceIndex: -1,
		ShowOverlay:      true,
		ArchiveAudio:     true,
		TextReplacements: map[string]string{},
		PauseMedia:       false,
		HotkeyEnabled:    true,
	}
}

func AppDataDir() string {
	appData := os.Getenv("APPDATA")
	if appData == "" {
		appData = os.Getenv("HOME")
		if appData == "" {
			appData = "."
		}
		return filepath.Join(appData, ".cognitive_flow")
	}
	return filepath.Join(appData, appName)
}

func ConfigPath() string {
	return filepath.Join(AppDataDir(), "config.json")
}

func AudioArchiveDir() string {
	return filepath.Join(AppDataDir(), "audio")
}

func LogPath() string {
	return filepath.Join(AppDataDir(), "debug_transcriptions.log")
}

func HistoryPath() string {
	return filepath.Join(AppDataDir(), "history.json")
}

func StatsPath() string {
	return filepath.Join(AppDataDir(), "statistics.json")
}

func LoadConfig() *Config {
	cfg := DefaultConfig()

	data, err := os.ReadFile(ConfigPath())
	if err != nil {
		return cfg
	}

	if err := json.Unmarshal(data, cfg); err != nil {
		fmt.Printf("[Config] Parse error: %v, using defaults\n", err)
		return DefaultConfig()
	}

	// Ensure text_replacements is initialized
	if cfg.TextReplacements == nil {
		cfg.TextReplacements = map[string]string{}
	}

	return cfg
}

func (c *Config) Save() error {
	dir := AppDataDir()
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("create config dir: %w", err)
	}

	data, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal config: %w", err)
	}

	return os.WriteFile(ConfigPath(), data, 0644)
}
