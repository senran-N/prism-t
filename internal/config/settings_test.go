package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestSaveAndLoadSettingsFileRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "settings.json")

	// Missing file returns exists=false without error
	if _, exists, err := LoadSettingsFile(path); err != nil || exists {
		t.Fatalf("expected missing file without error, got exists=%v err=%v", exists, err)
	}

	// Default omits settings.json unless explicitly saved, preserving ErrNoTranslationBackend invariant
	defCfg := Default()
	if defCfg.LLMBaseURL != "" && os.Getenv("PRISM_T_LLM_BASE_URL") == "" {
		t.Fatalf("expected empty default LLMBaseURL when env unset, got %q", defCfg.LLMBaseURL)
	}

	in := RuntimeSettings{
		Device:             "cuda",
		MaxConcurrency:     2,
		TargetLang:         "en",
		LLMBaseURL:         "https://example.invalid/v1/",
		LLMAPIKey:          "sk-test-secret-1234",
		LLMModel:           "vision-fast",
		LLMReasoningEffort: "omit",
		LLMTimeoutSeconds:  120,
		LLMVision:          true,
		LLMSourcePrepass:   true,
		LLMVisionAudit:     false,
		LLMBatchSize:       16,
	}
	if err := SaveSettingsFile(path, in); err != nil {
		t.Fatalf("save settings: %v", err)
	}

	loaded, exists, err := LoadSettingsFile(path)
	if err != nil || !exists {
		t.Fatalf("load settings: exists=%v err=%v", exists, err)
	}
	if err := ApplyRuntimeSettings(&defCfg, loaded); err != nil {
		t.Fatalf("apply settings: %v", err)
	}
	if defCfg.Device != "cuda" || defCfg.MaxConcurrency != 2 || defCfg.TargetLang != "en" {
		t.Fatalf("unexpected basic runtime fields: %+v", defCfg)
	}
	if defCfg.LLMBaseURL != "https://example.invalid/v1" || defCfg.LLMAPIKey != "sk-test-secret-1234" {
		t.Fatalf("unexpected LLM credentials: url=%q key=%q", defCfg.LLMBaseURL, defCfg.LLMAPIKey)
	}
	// "omit" must survive round-trip as empty ReasoningEffort instead of reverting to "none"
	if defCfg.LLMReasoningEffort != "" {
		t.Fatalf("expected empty ReasoningEffort from omit, got %q", defCfg.LLMReasoningEffort)
	}
	if FormatReasoningEffortSetting(defCfg.LLMReasoningEffort) != "omit" {
		t.Fatalf("expected omit formatting for empty reasoning effort")
	}

	// Test "none" survives round-trip as "none"
	loaded.LLMReasoningEffort = "none"
	if err := SaveSettingsFile(path, loaded); err != nil {
		t.Fatal(err)
	}
	loaded2, _, err := LoadSettingsFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := ApplyRuntimeSettings(&defCfg, loaded2); err != nil {
		t.Fatal(err)
	}
	if defCfg.LLMReasoningEffort != "none" {
		t.Fatalf("expected ReasoningEffort=none, got %q", defCfg.LLMReasoningEffort)
	}
}

func TestDiscoverAndResolvePiModelPresets(t *testing.T) {
	path := filepath.Join(t.TempDir(), "models.json")
	payload := `{
  "providers": {
    "demo": {
      "baseUrl": "https://api.example.test/v1/",
      "apiKey": "DEMO_TEST_API_KEY",
      "models": [
        {"id": "text-only", "name": "Text Only", "input": ["text"]},
        {"id": "vl-fast", "name": "VL Fast", "input": ["text", "image"]}
      ]
    }
  }
}`
	if err := os.WriteFile(path, []byte(payload), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("DEMO_TEST_API_KEY", "resolved-env-secret-9999")

	presets := discoverLocalModelPresetsFromPath(path)
	if len(presets) != 2 {
		t.Fatalf("expected 2 presets, got %+v", presets)
	}
	if presets[0].ModelID != "text-only" || presets[0].SupportsVision {
		t.Fatalf("unexpected first preset: %+v", presets[0])
	}
	if presets[1].ModelID != "vl-fast" || !presets[1].SupportsVision || !presets[1].HasAPIKey {
		t.Fatalf("unexpected second preset: %+v", presets[1])
	}

	url, key, model, vision, err := resolveLocalPresetCredentialsFromPath(path, "demo", "vl-fast")
	if err != nil {
		t.Fatal(err)
	}
	if url != "https://api.example.test/v1" || key != "resolved-env-secret-9999" || model != "vl-fast" || !vision {
		t.Fatalf("unexpected resolved credentials: url=%q key=%q model=%q vision=%v", url, key, model, vision)
	}
	if MaskSecret(key) != "****9999" || MaskSecret("") != "" || MaskSecret("abc") != "****" {
		t.Fatalf("unexpected MaskSecret output")
	}
}
