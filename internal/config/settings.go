package config

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// RuntimeSettings holds mutable server/engine/LLM configuration persisted under DataDir/settings.json.
type RuntimeSettings struct {
	Device             string `json:"device"`
	MaxConcurrency     int    `json:"max_concurrency"`
	TargetLang         string `json:"target_lang"`
	LLMBaseURL         string `json:"llm_base_url"`
	LLMAPIKey          string `json:"llm_api_key,omitempty"`
	LLMModel           string `json:"llm_model"`
	LLMReasoningEffort string `json:"llm_reasoning_effort"`
	LLMTimeoutSeconds  int    `json:"llm_timeout_seconds"`
	LLMVision          bool   `json:"llm_vision"`
	LLMSourcePrepass   bool   `json:"llm_source_prepass"`
	LLMVisionAudit     bool   `json:"llm_vision_audit"`
	LLMBatchSize       int    `json:"llm_batch_size"`
}

// ModelPreset describes a locally discovered OpenAI-compatible model endpoint without exposing secrets.
type ModelPreset struct {
	Provider       string `json:"provider"`
	ModelID        string `json:"model_id"`
	DisplayName    string `json:"display_name"`
	BaseURL        string `json:"base_url"`
	SupportsVision bool   `json:"supports_vision"`
	HasAPIKey      bool   `json:"has_api_key"`
}

type piModelsFile struct {
	Providers map[string]struct {
		BaseURL string `json:"baseUrl"`
		APIKey  string `json:"apiKey"`
		Models  []struct {
			ID    string   `json:"id"`
			Name  string   `json:"name"`
			Input []string `json:"input"`
		} `json:"models"`
	} `json:"providers"`
}

type piModelEntry struct {
	preset ModelPreset
	rawKey string
}

// SettingsPath returns the persistent JSON path for Web Studio runtime settings.
func SettingsPath(dataDir string) string {
	dir := strings.TrimSpace(dataDir)
	if dir == "" {
		dir = "data"
	}
	return filepath.Join(dir, "settings.json")
}

// FormatReasoningEffortSetting converts internal Config.LLMReasoningEffort into UI/storage notation.
func FormatReasoningEffortSetting(v string) string {
	norm := NormalizeReasoningEffort(v)
	if norm == "" {
		return "omit"
	}
	return norm
}

// SettingsFromConfig extracts RuntimeSettings from a Config value.
func SettingsFromConfig(cfg Config) RuntimeSettings {
	timeoutSec := int(cfg.LLMRequestTimeout.Seconds())
	if timeoutSec <= 0 {
		timeoutSec = 300
	}
	batchSize := cfg.LLMBatchSize
	if batchSize <= 0 {
		batchSize = 32
	}
	maxConc := cfg.MaxConcurrency
	if maxConc <= 0 {
		maxConc = 1
	}
	targetLang := strings.TrimSpace(cfg.TargetLang)
	if targetLang == "" {
		targetLang = "zh-CN"
	}
	dev := strings.TrimSpace(cfg.Device)
	if dev == "" {
		dev = "auto"
	}
	return RuntimeSettings{
		Device:             dev,
		MaxConcurrency:     maxConc,
		TargetLang:         targetLang,
		LLMBaseURL:         strings.TrimRight(strings.TrimSpace(cfg.LLMBaseURL), "/"),
		LLMAPIKey:          strings.TrimSpace(cfg.LLMAPIKey),
		LLMModel:           strings.TrimSpace(cfg.LLMModel),
		LLMReasoningEffort: FormatReasoningEffortSetting(cfg.LLMReasoningEffort),
		LLMTimeoutSeconds:  timeoutSec,
		LLMVision:          cfg.LLMVision,
		LLMSourcePrepass:   cfg.LLMSourcePrepass,
		LLMVisionAudit:     cfg.LLMVisionAudit,
		LLMBatchSize:       batchSize,
	}
}

// ApplyRuntimeSettings validates and copies RuntimeSettings onto cfg.
func ApplyRuntimeSettings(cfg *Config, rs RuntimeSettings) error {
	dev, err := ValidateDevice(rs.Device)
	if err != nil {
		return err
	}
	cfg.Device = dev
	cfg.MaxConcurrency = clampRange(rs.MaxConcurrency, 1, 8, cfg.MaxConcurrency, 1)
	if lang := strings.TrimSpace(rs.TargetLang); lang != "" {
		cfg.TargetLang = lang
	}
	cfg.LLMBaseURL = strings.TrimRight(strings.TrimSpace(rs.LLMBaseURL), "/")
	cfg.LLMAPIKey = strings.TrimSpace(rs.LLMAPIKey)
	if model := strings.TrimSpace(rs.LLMModel); model != "" {
		cfg.LLMModel = model
	}
	cfg.LLMReasoningEffort = NormalizeReasoningEffort(rs.LLMReasoningEffort)
	timeoutSec := clampRange(rs.LLMTimeoutSeconds, 15, 1800, int(cfg.LLMRequestTimeout.Seconds()), 300)
	cfg.LLMRequestTimeout = time.Duration(timeoutSec) * time.Second
	cfg.LLMVision = rs.LLMVision
	cfg.LLMSourcePrepass = rs.LLMSourcePrepass
	cfg.LLMVisionAudit = rs.LLMVisionAudit
	cfg.LLMBatchSize = clampRange(rs.LLMBatchSize, 1, 64, cfg.LLMBatchSize, 32)
	return nil
}

// LoadSettingsFile reads RuntimeSettings from disk. When the file is absent, exists=false and err=nil.
func LoadSettingsFile(path string) (RuntimeSettings, bool, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return RuntimeSettings{}, false, nil
		}
		return RuntimeSettings{}, false, err
	}
	rs := RuntimeSettings{
		Device:             "auto",
		MaxConcurrency:     1,
		TargetLang:         "zh-CN",
		LLMModel:           "gpt-4o-mini",
		LLMReasoningEffort: "none",
		LLMTimeoutSeconds:  300,
		LLMBatchSize:       32,
	}
	if err := json.Unmarshal(raw, &rs); err != nil {
		return RuntimeSettings{}, true, fmt.Errorf("parse settings %s: %w", path, err)
	}
	return rs, true, nil
}

// SaveSettingsFile atomically persists RuntimeSettings with 0600 file permissions.
func SaveSettingsFile(path string, rs RuntimeSettings) error {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("create settings dir: %w", err)
	}
	clean := rs
	clean.LLMReasoningEffort = FormatReasoningEffortSetting(rs.LLMReasoningEffort)
	data, err := json.MarshalIndent(clean, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	tmp, err := os.CreateTemp(dir, ".settings-*.json.tmp")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	cleanup := func() {
		_ = tmp.Close()
		_ = os.Remove(tmpPath)
	}
	if err := tmp.Chmod(0o600); err != nil {
		cleanup()
		return err
	}
	if _, err := tmp.Write(data); err != nil {
		cleanup()
		return err
	}
	if err := tmp.Close(); err != nil {
		_ = os.Remove(tmpPath)
		return err
	}
	if err := os.Rename(tmpPath, path); err != nil {
		_ = os.Remove(tmpPath)
		return err
	}
	return nil
}

// MaskSecret obscures sensitive tokens/keys while confirming presence to the UI.
func MaskSecret(secret string) string {
	s := strings.TrimSpace(secret)
	if s == "" {
		return ""
	}
	if len(s) < 8 {
		return "****"
	}
	return "****" + s[len(s)-4:]
}

func piModelsPath() string {
	home, err := os.UserHomeDir()
	if err != nil || strings.TrimSpace(home) == "" {
		return ""
	}
	return filepath.Join(home, ".pi", "agent", "models.json")
}

// DiscoverLocalModelPresets returns safe model presets parsed from ~/.pi/agent/models.json.
func DiscoverLocalModelPresets() []ModelPreset {
	p := piModelsPath()
	if p == "" {
		return nil
	}
	return discoverLocalModelPresetsFromPath(p)
}

// ResolveLocalPresetCredentials resolves endpoint URL, API key (checking env var names first),
// model ID, and vision capability for a preset discovered in ~/.pi/agent/models.json.
func ResolveLocalPresetCredentials(provider, modelID string) (string, string, string, bool, error) {
	p := piModelsPath()
	if p == "" {
		return "", "", "", false, fmt.Errorf("cannot locate user home directory for local model presets")
	}
	return resolveLocalPresetCredentialsFromPath(p, provider, modelID)
}

func discoverLocalModelPresetsFromPath(path string) []ModelPreset {
	entries, err := loadPiModelEntries(path)
	if err != nil {
		return nil
	}
	out := make([]ModelPreset, len(entries))
	for i, e := range entries {
		out[i] = e.preset
	}
	return out
}

func resolveLocalPresetCredentialsFromPath(path, provider, modelID string) (string, string, string, bool, error) {
	entries, err := loadPiModelEntries(path)
	if err != nil {
		return "", "", "", false, err
	}
	pNorm := strings.TrimSpace(provider)
	mNorm := strings.TrimSpace(modelID)
	for _, e := range entries {
		if e.preset.Provider == pNorm && e.preset.ModelID == mNorm {
			resolvedKey := strings.TrimSpace(e.rawKey)
			if envVal := strings.TrimSpace(os.Getenv(resolvedKey)); envVal != "" {
				resolvedKey = envVal
			}
			return e.preset.BaseURL, resolvedKey, e.preset.ModelID, e.preset.SupportsVision, nil
		}
	}
	return "", "", "", false, fmt.Errorf("model preset %s/%s not found in %s", provider, modelID, path)
}

func loadPiModelEntries(path string) ([]piModelEntry, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var parsed piModelsFile
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return nil, err
	}
	var entries []piModelEntry
	for provName, prov := range parsed.Providers {
		baseURL := strings.TrimRight(strings.TrimSpace(prov.BaseURL), "/")
		if baseURL == "" {
			continue
		}
		rawKey := strings.TrimSpace(prov.APIKey)
		resolvedHasKey := rawKey != ""
		for _, m := range prov.Models {
			id := strings.TrimSpace(m.ID)
			if id == "" {
				continue
			}
			vision := false
			for _, inp := range m.Input {
				if strings.EqualFold(strings.TrimSpace(inp), "image") {
					vision = true
					break
				}
			}
			display := strings.TrimSpace(m.Name)
			if display == "" {
				display = id
			}
			entries = append(entries, piModelEntry{
				preset: ModelPreset{
					Provider:       provName,
					ModelID:        id,
					DisplayName:    display,
					BaseURL:        baseURL,
					SupportsVision: vision,
					HasAPIKey:      resolvedHasKey,
				},
				rawKey: rawKey,
			})
		}
	}
	sort.Slice(entries, func(i, j int) bool {
		if entries[i].preset.Provider != entries[j].preset.Provider {
			return entries[i].preset.Provider < entries[j].preset.Provider
		}
		return entries[i].preset.ModelID < entries[j].preset.ModelID
	})
	return entries, nil
}

func clampRange(val, lo, hi, current, fallback int) int {
	if val <= 0 {
		val = current
	}
	if val <= 0 {
		val = fallback
	}
	if val < lo {
		return lo
	}
	if val > hi {
		return hi
	}
	return val
}
