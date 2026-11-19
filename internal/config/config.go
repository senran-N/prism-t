package config

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

type Config struct {
	ListenAddr             string
	EngineURL              string
	EngineDir              string
	DataDir                string
	DBPath                 string
	AdminToken             string
	AllowedOrigins         []string
	AutoEngine             bool
	Device                 string
	TargetLang             string
	LLMBaseURL             string
	LLMAPIKey              string
	LLMModel               string
	LLMReasoningEffort     string
	LLMRequestTimeout      time.Duration
	LLMVision              bool
	LLMSourcePrepass       bool
	LLMVisionAudit         bool
	LLMBatchSize           int
	LLMSourceLineBatchSize int
	MaxConcurrency         int
}

func Default() Config {
	dataDir := envOr("PRISM_T_DATA_DIR", "data")
	return Config{
		ListenAddr:             envOr("PRISM_T_LISTEN", "127.0.0.1:3090"),
		EngineURL:              envOr("PRISM_T_ENGINE_URL", "http://127.0.0.1:18775"),
		EngineDir:              envOr("PRISM_T_ENGINE_DIR", "engine"),
		DataDir:                dataDir,
		DBPath:                 envOr("PRISM_T_DB_PATH", filepath.Join(dataDir, "prism_t.db")),
		AdminToken:             strings.TrimSpace(os.Getenv("PRISM_T_ADMIN_TOKEN")),
		AllowedOrigins:         splitCSV(os.Getenv("PRISM_T_CORS_ORIGINS")),
		AutoEngine:             envBool("PRISM_T_AUTO_ENGINE", true),
		Device:                 envOr("PRISM_T_DEVICE", "auto"),
		TargetLang:             envOr("PRISM_T_TARGET_LANG", "zh-CN"),
		LLMBaseURL:             envOr("PRISM_T_LLM_BASE_URL", ""),
		LLMAPIKey:              envOr("PRISM_T_LLM_API_KEY", ""),
		LLMModel:               envOr("PRISM_T_LLM_MODEL", "gpt-4o-mini"),
		LLMReasoningEffort:     NormalizeReasoningEffort(envOr("PRISM_T_LLM_REASONING_EFFORT", "none")),
		LLMRequestTimeout:      time.Duration(envInt("PRISM_T_LLM_TIMEOUT_SECONDS", 300)) * time.Second,
		LLMVision:              envBool("PRISM_T_LLM_VISION", false),
		LLMSourcePrepass:       envBool("PRISM_T_LLM_SOURCE_PREPASS", false),
		LLMVisionAudit:         envBool("PRISM_T_LLM_VISION_AUDIT", false),
		LLMBatchSize:           envInt("PRISM_T_LLM_BATCH_SIZE", 32),
		LLMSourceLineBatchSize: envInt("PRISM_T_SOURCE_LINES_PER_REQUEST", 24),
		MaxConcurrency:         envInt("PRISM_T_CONCURRENCY", 1),
	}
}

func NormalizeReasoningEffort(v string) string {
	trimmed := strings.TrimSpace(v)
	switch strings.ToLower(trimmed) {
	case "default", "auto", "omit":
		return ""
	default:
		return trimmed
	}
}

func ValidateDevice(dev string) (string, error) {
	norm := strings.ToLower(strings.TrimSpace(dev))
	if norm == "" {
		return "auto", nil
	}
	switch norm {
	case "auto", "cpu", "gpu", "cuda", "dml":
		return norm, nil
	default:
		return "", fmt.Errorf("unsupported device %q (valid: auto, cpu, gpu, cuda, dml)", dev)
	}
}

func EnsureAdminToken(cfg *Config) (string, error) {
	if cfg.AdminToken != "" {
		return "", nil
	}
	if err := os.MkdirAll(cfg.DataDir, 0o755); err != nil {
		return "", fmt.Errorf("create data dir: %w", err)
	}
	keyPath := filepath.Join(cfg.DataDir, "admin.key")
	if raw, err := os.ReadFile(keyPath); err == nil {
		tok := strings.TrimSpace(string(raw))
		if len(tok) >= 16 {
			cfg.AdminToken = tok
			return keyPath, nil
		}
	}
	buf := make([]byte, 24)
	if _, err := rand.Read(buf); err != nil {
		return "", fmt.Errorf("generate admin token: %w", err)
	}
	cfg.AdminToken = hex.EncodeToString(buf)
	if err := os.WriteFile(keyPath, []byte(cfg.AdminToken+"\n"), 0o600); err != nil {
		return "", fmt.Errorf("write admin key: %w", err)
	}
	return keyPath, nil
}

func envOr(key, fallback string) string {
	if v := strings.TrimSpace(os.Getenv(key)); v != "" {
		return v
	}
	return fallback
}

func envBool(key string, fallback bool) bool {
	v := strings.ToLower(strings.TrimSpace(os.Getenv(key)))
	switch v {
	case "1", "true", "yes", "on":
		return true
	case "0", "false", "no", "off":
		return false
	default:
		return fallback
	}
}

func envInt(key string, fallback int) int {
	v := strings.TrimSpace(os.Getenv(key))
	if n, err := strconv.Atoi(v); err == nil && n > 0 {
		return n
	}
	return fallback
}

func splitCSV(s string) []string {
	if strings.TrimSpace(s) == "" {
		return nil
	}
	parts := strings.Split(s, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		if t := strings.TrimSpace(p); t != "" {
			out = append(out, t)
		}
	}
	return out
}
