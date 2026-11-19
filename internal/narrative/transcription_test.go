package narrative

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"image"
	"image/png"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	"prism-t/internal/config"
	"prism-t/internal/dom"
	"prism-t/internal/store"
)

func TestSourceOnlyHTTPContractCannotSeeTranslationDraft(t *testing.T) {
	requests := []NodeTranslationRequest{{ID: "n", SourceText: "old guess", DraftTranslation: "target answer", Speaker: "persona", BBox: []int{0, 0, 4, 4}, SourceLines: [][]int{{0, 0, 2, 4}, {2, 0, 2, 4}}, SourceReadings: []dom.SourceReading{{LineIndex: 0, Recognizer: "independent", Text: "observation"}}}}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			Messages []struct {
				Content json.RawMessage `json:"content"`
			} `json:"messages"`
			ResponseFormat map[string]any `json:"response_format"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Error(err)
			return
		}
		if bytes.Contains(request.Messages[1].Content, []byte("old guess")) || bytes.Contains(request.Messages[1].Content, []byte("target answer")) || bytes.Contains(request.Messages[1].Content, []byte("persona")) {
			t.Error("source reader received target/source draft or persona")
		}
		if !bytes.Contains(request.Messages[1].Content, []byte("observation")) || !bytes.Contains(request.Messages[1].Content, []byte("data:image/png;base64,")) {
			t.Error("source reader lost independent observations/native pixels")
		}
		var content []struct {
			Text string `json:"text"`
		}
		if err := json.Unmarshal(request.Messages[1].Content, &content); err != nil {
			t.Fatal(err)
		}
		_, metadata, ok := strings.Cut(content[len(content)-2].Text, "Tiles: ")
		var tiles []sourceAtlasTile
		if !ok || json.Unmarshal([]byte(metadata), &tiles) != nil || len(tiles) != 2 ||
			tiles[0].LineIndex != 0 || tiles[0].SourceRect != [4]int{0, 0, 2, 4} ||
			tiles[1].LineIndex != 1 || tiles[1].SourceRect != [4]int{2, 0, 2, 4} {
			t.Error("source reader lost owned line identities or coordinates")
		}
		root := request.ResponseFormat["json_schema"].(map[string]any)["schema"].(map[string]any)
		fields := root["properties"].(map[string]any)["sources"].(map[string]any)["properties"].(map[string]any)["n"].(map[string]any)["properties"].(map[string]any)
		lines := fields["line_texts"].(map[string]any)
		if lines["minItems"] != float64(2) || lines["maxItems"] != float64(2) || fields["target_text"] != nil {
			t.Error("source schema does not enforce independent per-line transcription")
		}
		writeChatStream(w, `{"sources":{"n":{"id":"n","is_text":true,"line_texts":["second","first"],"line_order":[1,0],"reason":""}}}`)
	}))
	defer server.Close()
	var raw bytes.Buffer
	png.Encode(&raw, image.NewNRGBA(image.Rect(0, 0, 4, 4)))
	backend := &OpenAICompatibleBackend{baseURL: server.URL, httpClient: server.Client(), diagnosticsPath: t.TempDir()}
	ctx := context.WithValue(context.Background(), pageImageKey{}, raw.Bytes())
	results, err := backend.TranscribeNodes(ctx, sourceNodeRequests(requests))
	if err != nil || len(results) != 1 || transcribedSource(results[0]) != "first\nsecond" || results[0].LineTexts[0] != "second" {
		t.Fatalf("source response lost: %+v / %v", results, err)
	}
}

func TestSourceContractRejectsIdentityLossAndMissingLines(t *testing.T) {
	for _, raw := range []string{`{"n":{"id":"n"},"n":{"id":"n"}}`, `{"n":{"id":"other"}}`, `[{"id":"n"}]`} {
		var results keyedSources
		if json.Unmarshal([]byte(raw), &results) == nil {
			t.Fatalf("ambiguous source identity accepted: %s", raw)
		}
	}
	yes, no := true, false
	request := SourceNodeRequest{ID: "n", SourceLines: [][]int{{0, 0, 1, 1}, {1, 0, 1, 1}}}
	for _, result := range []dom.SourceTranscription{
		{ID: "n", IsText: &yes, LineTexts: []string{"only one line"}},
		{ID: "n", IsText: &yes, LineTexts: []string{"", ""}},
		{ID: "n", IsText: &no, LineTexts: []string{"writing", ""}, Reason: "art"},
		{ID: "n", LineTexts: []string{"first", "second"}},
		{ID: "n", IsText: &yes, LineTexts: []string{"first", "second"}, LineOrder: []int{0, 0}},
		{ID: "n", IsText: &yes, LineTexts: []string{"first", "second"}, LineOrder: []int{0, 2}},
		{ID: "n", IsText: &yes, LineTexts: []string{"first", "second"}, LineOrder: []int{0}},
	} {
		if sourceResultIssue(request, result) == "" {
			t.Fatalf("invalid transcription accepted: %+v", result)
		}
	}
}

func TestLockedSourceCannotBeRewrittenToEvadeFidelity(t *testing.T) {
	yes := true
	request := NodeTranslationRequest{ID: "n", SourceText: "7\nsource", SourceTextLocked: true}
	bad := NodeTranslationResult{ID: "n", IsText: &yes, SourceText: "source", TargetText: "译文", TargetSegments: []string{"译文"}}
	backend := &fidelityBackend{results: [][]NodeTranslationResult{{bad}, {{ID: "n", IsText: &yes, SourceText: request.SourceText, TargetText: "7译文", TargetSegments: []string{"7", "译文"}}}}}
	result, err := (&Compiler{cfg: config.Config{LLMVision: true}, backend: backend}).translateComplete(context.Background(), "zh-CN", nil, nil, []NodeTranslationRequest{request})
	if err != nil || len(result) != 1 || len(backend.calls) != 2 || backend.calls[1][0].SourceText != request.SourceText || !strings.Contains(backend.calls[1][0].Correction, "locked") {
		t.Fatalf("rewritten source escaped repair: %+v / %v", backend.calls, err)
	}
	format := translationResponseFormat([]NodeTranslationRequest{request})
	fields := format["json_schema"].(map[string]any)["schema"].(map[string]any)["properties"].(map[string]any)["translations"].(map[string]any)["properties"].(map[string]any)["n"].(map[string]any)["properties"].(map[string]any)
	if fields["source_text"].(map[string]any)["enum"].([]string)[0] != request.SourceText {
		t.Fatal("strict schema permits source rewriting")
	}
	no := false
	if issue := translationFidelityIssue(request, NodeTranslationResult{IsText: &no, Reason: "actual illustration without writing"}, true); issue != "" {
		t.Fatal("source hallucinations cannot be rejected as artwork")
	}
}

type sourceCheckpointBackend struct {
	interruptedBackend
	sourceCalls  [][]SourceNodeRequest
	failSourceOn int
}

func (b *sourceCheckpointBackend) TranscribeNodes(_ context.Context, requests []SourceNodeRequest) ([]dom.SourceTranscription, error) {
	b.sourceCalls = append(b.sourceCalls, requests)
	if b.failSourceOn == len(b.sourceCalls) {
		return nil, errors.New("source upstream interrupted")
	}
	return fixtureSources(requests, map[string]string{"a": "first", "b": "second"}), nil
}

func TestSourceSQLiteCheckpointResumesWithoutReadingTargets(t *testing.T) {
	ctx := context.Background()
	st, err := store.Open(filepath.Join(t.TempDir(), "source.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	work, err := st.EnsureWork(ctx, "Source", "en", "zh-CN", "comic_ltr")
	if err != nil {
		t.Fatal(err)
	}
	page := dom.PageDOM{Nodes: []dom.TextNode{{ID: "a", RecognitionPending: true}, {ID: "b", RecognitionPending: true}}}
	id, err := st.SavePage(ctx, work.ID, "1", 0, "fixture.png", "hash", page, []byte("raw"), []byte("preview"), "analyzed")
	if err != nil {
		t.Fatal(err)
	}
	cfg := config.Config{LLMVision: true, LLMSourcePrepass: true, LLMBatchSize: 1, LLMModel: "model"}
	backend := &sourceCheckpointBackend{failSourceOn: 2}
	if NewCompilerWithBackend(cfg, backend).CompilePageTranslations(ctx, st, work.ID, "1", 0, "zh-CN", &page, []byte("raw")) == nil {
		t.Fatal("source interruption was hidden")
	}
	saved, err := st.GetPage(ctx, id)
	if err != nil {
		t.Fatal(err)
	}
	if saved.Summary.Status != "analyzed" || saved.DOM.RecognitionProgress == nil || len(saved.DOM.RecognitionProgress.Results) != 1 || saved.DOM.TranslationProgress != nil {
		t.Fatal("partial recognition lost or masquerading as translated")
	}
	resumed := &sourceCheckpointBackend{}
	if err := NewCompilerWithBackend(cfg, resumed).CompilePageTranslations(ctx, st, work.ID, "1", 0, "zh-CN", &saved.DOM, []byte("raw")); err != nil {
		t.Fatal(err)
	}
	if len(resumed.sourceCalls) != 1 || resumed.sourceCalls[0][0].ID != "b" || len(saved.DOM.RecognitionProgress.Results) != 2 || saved.DOM.Nodes[0].SourceText != "first" {
		t.Fatalf("source resume lost provenance: %+v", resumed.sourceCalls)
	}
	for _, calls := range resumed.calls {
		for _, request := range calls {
			if !request.SourceTextLocked {
				t.Fatal("translation/audit received unlocked source")
			}
		}
	}
	independent := &sourceCheckpointBackend{}
	requests := []NodeTranslationRequest{{ID: "a", SourceText: "new target-influenced guess"}, {ID: "b"}}
	if err := NewCompilerWithBackend(cfg, independent).transcribePage(ctx, st, work.ID, "1", 0, &saved.DOM, []byte("raw"), requests); err != nil {
		t.Fatal(err)
	}
	if len(independent.sourceCalls) != 0 || requests[0].SourceText != "first" {
		t.Fatal("translation draft changed source checkpoint identity")
	}
	cfg.LLMModel = "changed-model"
	changed := &sourceCheckpointBackend{}
	if err := NewCompilerWithBackend(cfg, changed).transcribePage(ctx, st, work.ID, "1", 0, &saved.DOM, []byte("raw"), requests); err != nil {
		t.Fatal(err)
	}
	if len(changed.sourceCalls) != 2 || requests[0].SourceText != "first" {
		t.Fatal("changed source model reused stale readings")
	}
}
