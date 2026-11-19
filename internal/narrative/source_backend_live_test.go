package narrative

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"prism-t/internal/config"
	"prism-t/internal/dom"
	"prism-t/internal/store"
)

// Explicit opt-in: a real provider call, with no canned source or translation.
// The fixture references original PNG bytes and actual SourceNodeRequests.
func TestLiveSourceEvidence(t *testing.T) {
	path := os.Getenv("PRISM_T_SOURCE_PROBE")
	if path == "" {
		t.Skip("set PRISM_T_SOURCE_PROBE to an actual source-evidence fixture")
	}
	var fixture struct {
		ImagePath string              `json:"image_path"`
		Nodes     []SourceNodeRequest `json:"nodes"`
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(fixture.ImagePath)
	if err != nil || len(fixture.Nodes) == 0 {
		t.Fatalf("missing original pixels or source requests: %v", err)
	}
	cfg := config.Default()
	if cfg.LLMBaseURL == "" || cfg.LLMModel == "" {
		t.Fatal(ErrNoTranslationBackend)
	}
	cfg.DataDir = filepath.Join(filepath.Dir(path), "provider_diagnostics")
	started := time.Now()
	ctx := context.WithValue(context.Background(), pageImageKey{}, raw)
	results, err := NewOpenAICompatibleBackend(cfg).TranscribeNodes(ctx, fixture.Nodes)
	if err != nil {
		t.Fatal(err)
	}
	output := struct {
		Stage     string `json:"stage"`
		ElapsedMS int64  `json:"elapsed_ms"`
		Results   any    `json:"results"`
	}{"actual source transcription; no translation or glyph truth supplied", time.Since(started).Milliseconds(), results}
	data, err = json.MarshalIndent(output, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	// Retain even invalid upstream output for diagnosis, without treating it as
	// completed production progress or rewriting the source database.
	if err := os.WriteFile(path+".result.json", data, 0o600); err != nil {
		t.Fatal(err)
	}
	expected := make(map[string]SourceNodeRequest, len(fixture.Nodes))
	for _, node := range fixture.Nodes {
		expected[node.ID] = node
	}
	for _, result := range results {
		request, ok := expected[result.ID]
		if !ok {
			t.Fatalf("unknown or duplicate source ID: %s", result.ID)
		}
		if issue := sourceResultIssue(request, result); issue != "" {
			t.Fatalf("source %s: %s", result.ID, issue)
		}
		delete(expected, result.ID)
	}
	if len(expected) != 0 {
		t.Fatalf("provider omitted %d source nodes", len(expected))
	}
	t.Logf("real provider returned %d source nodes in %s; contract validity does not establish glyph accuracy", len(results), time.Since(started).Round(time.Millisecond))
}

// Exercise actual partitioning, durable checkpoints and global ordering without
// injecting OCR truth, cached answers or translation drafts into the requests.
func TestLivePartitionedSourceEvidence(t *testing.T) {
	path := os.Getenv("PRISM_T_SOURCE_PROBE")
	if path == "" {
		t.Skip("set PRISM_T_SOURCE_PROBE to an actual source-evidence fixture")
	}
	var fixture struct {
		ImagePath string              `json:"image_path"`
		Nodes     []SourceNodeRequest `json:"nodes"`
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(fixture.ImagePath)
	if err != nil || len(fixture.Nodes) == 0 {
		t.Fatalf("missing actual source evidence: %v", err)
	}
	cfg := config.Default()
	if cfg.LLMBaseURL == "" {
		t.Fatal(ErrNoTranslationBackend)
	}
	cfg.DataDir = filepath.Join(filepath.Dir(path), "provider_diagnostics")
	st, err := store.Open(path + ".db")
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	ctx := context.WithValue(context.Background(), pageImageKey{}, raw)
	work, err := st.EnsureWork(ctx, "Actual source-only probe", "auto", "zh-CN", "manga_rtl")
	if err != nil {
		t.Fatal(err)
	}
	pages, err := st.ListPages(ctx, work.ID, false)
	if err != nil {
		t.Fatal(err)
	}
	var pageID int64
	if len(pages) == 0 {
		pageID, err = st.SavePage(ctx, work.ID, "source-only", 0, filepath.Base(fixture.ImagePath), "", dom.PageDOM{}, raw, raw, "analyzed")
	} else {
		pageID = pages[0].ID
	}
	if err != nil {
		t.Fatal(err)
	}
	saved, err := st.GetPage(ctx, pageID)
	if err != nil {
		t.Fatal(err)
	}
	requests := make([]NodeTranslationRequest, len(fixture.Nodes))
	for i, source := range fixture.Nodes {
		requests[i] = NodeTranslationRequest{ID: source.ID, BBox: source.BBox, SourceLines: source.SourceLines,
			SourceQuadrilaterals: source.SourceQuadrilaterals, SourceReadings: source.SourceReadings}
	}
	started := time.Now()
	if err := NewCompiler(cfg).transcribePage(ctx, st, work.ID, "source-only", 0, &saved.DOM, raw, requests); err != nil {
		t.Fatal(err)
	}
	data, err = json.MarshalIndent(map[string]any{
		"stage":      "actual partitioned source transcription and global order; not translation or a visual restoration run",
		"elapsed_ms": time.Since(started).Milliseconds(), "recognition_progress": saved.DOM.RecognitionProgress,
	}, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path+".partitioned-result.json", data, 0o600); err != nil {
		t.Fatal(err)
	}
	if len(saved.DOM.RecognitionProgress.Results) != len(fixture.Nodes) {
		t.Fatal("actual source nodes are missing")
	}
	t.Logf("actual partitioned source completed %d nodes; glyph accuracy still requires source comparison", len(fixture.Nodes))
}
