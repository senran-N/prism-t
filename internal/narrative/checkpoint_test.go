package narrative

import (
	"context"
	"errors"
	"path/filepath"
	"testing"

	"prism-t/internal/config"
	"prism-t/internal/dom"
	"prism-t/internal/store"
)

type interruptedBackend struct {
	fixtureSourceOrdering
	calls  [][]NodeTranslationRequest
	failOn int
}

func (*interruptedBackend) TranscribeNodes(_ context.Context, requests []SourceNodeRequest) ([]dom.SourceTranscription, error) {
	return fixtureSources(requests, map[string]string{"a": "first", "b": "second"}), nil
}

func (b *interruptedBackend) TranslateNodes(_ context.Context, _ string, _ []store.DialogueTurn, _ []store.GlossaryTerm, nodes []NodeTranslationRequest) ([]NodeTranslationResult, error) {
	b.calls = append(b.calls, nodes)
	if b.failOn == len(b.calls) {
		return nil, errors.New("upstream interrupted")
	}
	results := make([]NodeTranslationResult, len(nodes))
	for i, n := range nodes {
		isText := true
		results[i] = NodeTranslationResult{ID: n.ID, IsText: &isText, SourceText: n.SourceText, TargetText: "实际测试响应", TargetSegments: []string{"实际", "测试", "响应"}}
	}
	return results, nil
}

func TestInterruptedTranslationResumesActualValidatedNodesFromSQLite(t *testing.T) {
	ctx := context.Background()
	st, err := store.Open(filepath.Join(t.TempDir(), "checkpoint.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	work, err := st.EnsureWork(ctx, "Checkpoint", "ja", "zh-CN", "manga_rtl")
	if err != nil {
		t.Fatal(err)
	}
	page := dom.PageDOM{Nodes: []dom.TextNode{
		{ID: "a", SourceText: "first", CharBudgetMax: 30},
		{ID: "b", SourceText: "second", CharBudgetMax: 30},
	}}
	pageID, err := st.SavePage(ctx, work.ID, "1", 0, "test.png", "hash", page, []byte("raw"), []byte("preview"), "analyzed")
	if err != nil {
		t.Fatal(err)
	}
	cfg := config.Config{LLMVision: true, LLMVisionAudit: true, LLMBatchSize: 1, LLMModel: "test-model"}
	backend := &interruptedBackend{failOn: 2}
	err = NewCompilerWithBackend(cfg, backend).CompilePageTranslations(ctx, st, work.ID, "1", 0, "zh-CN", &page, []byte("raw"))
	if err == nil {
		t.Fatal("Expected real upstream interruption")
	}
	stored, err := st.GetPage(ctx, pageID)
	if err != nil {
		t.Fatal(err)
	}
	if stored.Summary.Status != "analyzed" || stored.DOM.TranslationProgress == nil || len(stored.DOM.TranslationProgress.Results) != 1 || stored.DOM.Nodes[0].TargetText != "" {
		t.Fatalf("Partial progress masqueraded as a completed translation or was lost: %+v", stored.DOM.TranslationProgress)
	}
	// Batch size can change without discarding results already obtained.
	cfg.LLMBatchSize = 2
	resumed := &interruptedBackend{}
	if err := NewCompilerWithBackend(cfg, resumed).CompilePageTranslations(ctx, st, work.ID, "1", 0, "zh-CN", &stored.DOM, []byte("raw")); err != nil {
		t.Fatal(err)
	}
	if len(resumed.calls) != 2 || len(resumed.calls[0]) != 1 || resumed.calls[0][0].ID != "b" || len(resumed.calls[1]) != 2 || resumed.calls[1][0].DraftTranslation == "" {
		t.Fatalf("Resume repeated a completed node or skipped independent audit: %+v", resumed.calls)
	}
	if stored.DOM.TranslationProgress != nil || len(stored.DOM.Nodes) != 2 || stored.DOM.Nodes[0].TargetText == "" || stored.DOM.Nodes[1].TargetText == "" {
		t.Fatal("Finished translation did not clear in-flight state")
	}
}

func TestChangedTranslationContextCannotReuseInterruptedResults(t *testing.T) {
	ctx := context.Background()
	st, err := store.Open(filepath.Join(t.TempDir(), "changed.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	work, err := st.EnsureWork(ctx, "Changed", "ja", "zh-CN", "manga_rtl")
	if err != nil {
		t.Fatal(err)
	}
	original := dom.PageDOM{Nodes: []dom.TextNode{
		{ID: "a", SourceText: "first", CharBudgetMax: 30},
		{ID: "b", SourceText: "second", CharBudgetMax: 30},
	}}
	cfg := config.Config{LLMBatchSize: 1, LLMModel: "old-model"}
	backend := &interruptedBackend{failOn: 2}
	if err := NewCompilerWithBackend(cfg, backend).CompilePageTranslations(ctx, st, work.ID, "1", 0, "zh-CN", &original, []byte("raw")); err == nil {
		t.Fatal("Expected interruption")
	}
	cfg.LLMModel = "new-model"
	fresh := &interruptedBackend{}
	if err := NewCompilerWithBackend(cfg, fresh).CompilePageTranslations(ctx, st, work.ID, "1", 0, "zh-CN", &original, []byte("raw")); err != nil {
		t.Fatal(err)
	}
	if len(fresh.calls) != 2 || fresh.calls[0][0].ID != "a" {
		t.Fatal("Changed model reused stale output")
	}
}
