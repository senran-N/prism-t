package narrative

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"image"
	"image/color"
	"image/jpeg"
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

func TestVisualAuditRecoversWritingIncorrectlyRejectedAsNonDialogue(t *testing.T) {
	st, err := store.Open(filepath.Join(t.TempDir(), "audit.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	ctx := context.Background()
	work, err := st.EnsureWork(ctx, "Audit", "ja", "zh-CN", "manga_rtl")
	if err != nil {
		t.Fatal(err)
	}
	isWriting := true
	notWriting := false
	backend := &responseBackend{responses: [][]NodeTranslationResult{
		{{ID: "label", IsText: &notWriting, Reason: "Packaging, not dialogue"}},
		{{ID: "label", IsText: &isWriting, SourceText: "賞味期限", TargetText: "保质期"}},
	}}
	compiler := NewCompilerWithBackend(config.Config{LLMVision: true, LLMVisionAudit: true}, backend)
	page := dom.PageDOM{Nodes: []dom.TextNode{{ID: "label", RecognitionPending: true, BBox: []int{0, 0, 2, 2}, CharBudgetMax: 10}}}
	var raw bytes.Buffer
	if err := png.Encode(&raw, image.NewGray(image.Rect(0, 0, 2, 2))); err != nil {
		t.Fatal(err)
	}
	if err := compiler.CompilePageTranslations(ctx, st, work.ID, "1", 0, "zh-CN", &page, raw.Bytes()); err != nil {
		t.Fatal(err)
	}
	if len(page.Nodes) != 1 || page.Nodes[0].TargetText != "保质期" || len(page.RejectedRegions) != 0 {
		t.Fatalf("Readable labels must survive correction of an initial false rejection: %+v", page)
	}
}

func TestRejectedArtworkRetainsTheFalseRecognizerEvidence(t *testing.T) {
	st, err := store.Open(filepath.Join(t.TempDir(), "rejected.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	ctx := context.Background()
	work, err := st.EnsureWork(ctx, "Evidence", "ja", "zh-CN", "manga_rtl")
	if err != nil {
		t.Fatal(err)
	}
	notWriting := false
	rejected := NodeTranslationResult{ID: "button", IsText: &notWriting, Reason: "A uniform button, no writing"}
	backend := &responseBackend{responses: [][]NodeTranslationResult{{rejected}, {rejected}}}
	node := dom.TextNode{ID: "button", RecognitionPending: true, BBox: []int{0, 0, 2, 2}, SourceLines: [][]int{{0, 0, 2, 2}}, SourceQuadrilaterals: [][][]float64{{{0, 0}, {2, 0}, {2, 2}, {0, 2}}}, SourceReadings: []dom.SourceReading{{LineIndex: 0, Recognizer: "independent", Text: "false glyph", Truncated: true}}}
	page := dom.PageDOM{Nodes: []dom.TextNode{node}}
	var raw bytes.Buffer
	png.Encode(&raw, image.NewGray(image.Rect(0, 0, 2, 2)))
	if err := NewCompilerWithBackend(config.Config{LLMVision: true}, backend).CompilePageTranslations(ctx, st, work.ID, "1", 0, "zh-CN", &page, raw.Bytes()); err != nil {
		t.Fatal(err)
	}
	if len(page.Nodes) != 0 || len(page.RejectedRegions) != 1 {
		t.Fatal("artwork rejection failed")
	}
	evidence := page.RejectedRegions[0]
	if len(evidence.SourceLines) != 1 || len(evidence.SourceQuadrilaterals) != 1 || len(evidence.SourceReadings) != 1 || evidence.SourceReadings[0].Text != "false glyph" || !evidence.SourceReadings[0].Truncated {
		t.Fatal("rejected regions lost their auditable evidence")
	}
}

type responseBackend struct {
	fixtureSourceOrdering
	responses [][]NodeTranslationResult
	calls     int
	contexts  []context.Context
}

func (*responseBackend) TranscribeNodes(_ context.Context, requests []SourceNodeRequest) ([]dom.SourceTranscription, error) {
	return fixtureSources(requests, nil), nil
}

func (b *responseBackend) TranslateNodes(ctx context.Context, _ string, _ []store.DialogueTurn, _ []store.GlossaryTerm, _ []NodeTranslationRequest) ([]NodeTranslationResult, error) {
	response := b.responses[b.calls]
	for i := range response {
		if response[i].TargetText != "" {
			response[i].TargetSegments = []string{response[i].TargetText}
		}
	}
	b.calls++
	b.contexts = append(b.contexts, ctx)
	return response, nil
}

func TestVisualAuditBatchesShareTheActualCompletePageDraft(t *testing.T) {
	st, err := store.Open(filepath.Join(t.TempDir(), "context.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	ctx := context.Background()
	work, err := st.EnsureWork(ctx, "Context", "en", "zh-CN", "comic_ltr")
	if err != nil {
		t.Fatal(err)
	}
	backend := &responseBackend{responses: [][]NodeTranslationResult{
		{{ID: "a", SourceText: "Yank!", TargetText: "扬克！"}},
		{{ID: "b", SourceText: "Yank...", TargetText: "美国佬……"}},
		{{ID: "a", SourceText: "Yank!", TargetText: "扬克！"}},
		{{ID: "b", SourceText: "Yank...", TargetText: "扬克……"}},
	}}
	compiler := NewCompilerWithBackend(config.Config{LLMVision: true, LLMVisionAudit: true, LLMBatchSize: 1}, backend)
	page := dom.PageDOM{Nodes: []dom.TextNode{
		{ID: "a", SourceText: "Yank!", CharBudgetMax: 30},
		{ID: "b", SourceText: "Yank...", CharBudgetMax: 30},
	}}
	if err := compiler.CompilePageTranslations(ctx, st, work.ID, "1", 0, "zh-CN", &page, []byte("image context")); err != nil {
		t.Fatal(err)
	}
	for _, ctx := range backend.contexts[2:] {
		draft, ok := ctx.Value(pageDraftKey{}).([]NodeTranslationRequest)
		if !ok || len(draft) != 2 || draft[0].DraftTranslation != "扬克！" || draft[1].DraftTranslation != "美国佬……" {
			t.Fatalf("Audit lost cross-batch evidence: %+v", draft)
		}
	}
	if page.Nodes[1].TargetText != "扬克……" {
		t.Fatal("Audit correction was lost")
	}
}

func TestIncompleteTranslationIsRepairedWithoutOmittingNodes(t *testing.T) {
	backend := &responseBackend{responses: [][]NodeTranslationResult{
		{{ID: "first", TargetText: "第一句"}},
		{{ID: "second", TargetText: "第二句"}},
	}}
	compiler := &Compiler{backend: backend}
	result, err := compiler.translateComplete(context.Background(), "zh-CN", nil, nil, []NodeTranslationRequest{{ID: "first", SourceText: "source one"}, {ID: "second", SourceText: "source two"}})
	if err != nil {
		t.Fatal(err)
	}
	if backend.calls != 2 || len(result) != 2 || result[1].TargetText != "第二句" {
		t.Fatalf("incomplete repair: %+v", result)
	}
}

func TestConfiguredBatchingStillRequiresEveryNode(t *testing.T) {
	backend := &responseBackend{responses: [][]NodeTranslationResult{
		{{ID: "a", TargetText: "甲"}, {ID: "b", TargetText: "乙"}},
		{{ID: "c", TargetText: "丙"}},
	}}
	compiler := &Compiler{cfg: config.Config{LLMBatchSize: 2}, backend: backend}
	requests := []NodeTranslationRequest{{ID: "a", SourceText: "first"}, {ID: "b", SourceText: "second"}, {ID: "c", SourceText: "third"}}
	result, err := compiler.translateComplete(context.Background(), "zh-CN", nil, nil, requests)
	if err != nil || len(result) != 3 || result[2].ID != "c" || backend.calls != 2 {
		t.Fatalf("batching lost coverage: %+v, %v", result, err)
	}
}

func TestInvalidTranslationsCannotBeReportedAsSuccessful(t *testing.T) {
	cases := []struct {
		name      string
		response  []NodeTranslationResult
		errorText string
	}{
		{"empty", nil, "no progress"},
		{"blank", []NodeTranslationResult{{ID: "one", TargetText: " "}}, "no progress"},
		{"unknown", []NodeTranslationResult{{ID: "other", TargetText: "文本"}}, "unknown"},
		{"duplicate", []NodeTranslationResult{{ID: "one", TargetText: "甲"}, {ID: "one", TargetText: "乙"}}, "duplicate"},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			compiler := &Compiler{backend: &responseBackend{responses: [][]NodeTranslationResult{test.response}}}
			_, err := compiler.translateComplete(context.Background(), "zh-CN", nil, nil, []NodeTranslationRequest{{ID: "one", SourceText: "source"}})
			if err == nil || !strings.Contains(err.Error(), test.errorText) {
				t.Fatalf("wanted %q, got %v", test.errorText, err)
			}
		})
	}
}

func TestSinglePassHolisticVisionUsesTwoCompactJPEGs(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 1200, 1600))
	for y := 0; y < 1600; y++ {
		for x := 0; x < 1200; x++ {
			img.Set(x, y, color.RGBA{R: 245, G: 245, B: 245, A: 255})
		}
	}
	var rawPNG bytes.Buffer
	if err := png.Encode(&rawPNG, img); err != nil {
		t.Fatal(err)
	}

	var httpCalls int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		httpCalls++
		var req struct {
			Messages []struct {
				Content json.RawMessage `json:"content"`
			} `json:"messages"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode vision request: %v", err)
		}
		if len(req.Messages) < 2 {
			t.Fatalf("expected system and user messages, got %d", len(req.Messages))
		}
		var parts []struct {
			Type     string `json:"type"`
			Text     string `json:"text"`
			ImageURL *struct {
				URL    string `json:"url"`
				Detail string `json:"detail"`
			} `json:"image_url"`
		}
		if err := json.Unmarshal(req.Messages[1].Content, &parts); err != nil {
			t.Fatalf("unmarshal user content array: %v", err)
		}
		var images []struct {
			url, detail string
		}
		var tiles []cropSheetTile
		for _, part := range parts {
			if part.Type == "text" && strings.Contains(part.Text, "Tiles: ") {
				_, rawTiles, _ := strings.Cut(part.Text, "Tiles: ")
				if err := json.Unmarshal([]byte(rawTiles), &tiles); err != nil {
					t.Fatalf("unmarshal contact sheet tiles: %v", err)
				}
			}
			if part.Type == "image_url" && part.ImageURL != nil {
				images = append(images, struct{ url, detail string }{part.ImageURL.URL, part.ImageURL.Detail})
			}
		}
		if len(images) != 2 {
			t.Fatalf("expected exactly 2 JPEG images (overview + contact sheet), got %d", len(images))
		}
		for i, item := range images {
			const prefix = "data:image/jpeg;base64,"
			if !strings.HasPrefix(item.url, prefix) {
				t.Fatalf("image %d expected %q prefix, got %.40s", i, prefix, item.url)
			}
			decodedBytes, err := base64.StdEncoding.DecodeString(strings.TrimPrefix(item.url, prefix))
			if err != nil {
				t.Fatalf("decode base64 image %d: %v", i, err)
			}
			decodedImg, err := jpeg.Decode(bytes.NewReader(decodedBytes))
			if err != nil {
				t.Fatalf("decode JPEG image %d: %v", i, err)
			}
			if i == 0 {
				if item.detail != "auto" {
					t.Fatalf("overview detail = %q, want auto", item.detail)
				}
				if b := decodedImg.Bounds(); max(b.Dx(), b.Dy()) != 960 {
					t.Fatalf("overview dimensions = %dx%d, want max 960", b.Dx(), b.Dy())
				}
			} else if item.detail != "high" {
				t.Fatalf("contact sheet detail = %q, want high", item.detail)
			}
		}
		if len(tiles) != 2 || tiles[0].ID != "n1" || tiles[0].Scale != 1.0 {
			t.Fatalf("expected n1 native 1:1 crop scale, got %+v", tiles)
		}
		if tiles[1].ID != "n2" || tiles[1].Scale >= 1.0 || tiles[1].Scale*80.0 < 27.5 {
			t.Fatalf("expected oversized thick-glyph n2 downscale >=28px line thickness, got %+v", tiles[1])
		}
		writeChatStream(w, `{"translations":{"n1":{"id":"n1","is_text":true,"source_text":"本当に？","target_segments":["真的","吗？"],"reason":"","rubies":[]},"n2":{"id":"n2","is_text":true,"source_text":"ドォン！","target_segments":["轰！"],"reason":"","rubies":[]}}}`)
	}))
	defer server.Close()

	st, err := store.Open(filepath.Join(t.TempDir(), "singlepass.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	ctx := context.Background()
	work, err := st.EnsureWork(ctx, "SinglePass", "ja", "zh-CN", "manga_rtl")
	if err != nil {
		t.Fatal(err)
	}
	page := dom.PageDOM{Nodes: []dom.TextNode{
		{ID: "n1", RecognitionPending: true, BBox: []int{100, 100, 180, 120}, SourceLines: [][]int{{120, 110, 26, 80}, {160, 110, 24, 80}}, CharBudgetMin: 2, CharBudgetMax: 6},
		{ID: "n2", RecognitionPending: true, BBox: []int{200, 400, 500, 500}, SourceLines: [][]int{{220, 420, 80, 440}}, CharBudgetMin: 1, CharBudgetMax: 4},
	}}
	backend := &OpenAICompatibleBackend{baseURL: server.URL, httpClient: server.Client(), diagnosticsPath: t.TempDir()}
	cfg := config.Config{LLMVision: true, LLMBatchSize: 32, LLMSourceLineBatchSize: 24, TargetLang: "zh-CN"}
	if err := NewCompilerWithBackend(cfg, backend).CompilePageTranslations(ctx, st, work.ID, "1", 0, "zh-CN", &page, rawPNG.Bytes()); err != nil {
		t.Fatalf("CompilePageTranslations: %v", err)
	}
	if httpCalls != 1 {
		t.Fatalf("expected 1 single-pass holistic HTTP request (no prepass/compress/audit), got %d", httpCalls)
	}
	if len(page.Nodes) != 2 || page.Nodes[0].TargetText != "真的吗？" || page.Nodes[1].TargetText != "轰！" {
		t.Fatalf("unexpected compiled nodes: %+v", page.Nodes)
	}
}

type megaNodeBackend struct {
	transcribeIDs  []string
	orderedIDs     []string
	translateCalls [][]NodeTranslationRequest
}

func (b *megaNodeBackend) TranscribeNodes(_ context.Context, requests []SourceNodeRequest) ([]dom.SourceTranscription, error) {
	yes := true
	out := make([]dom.SourceTranscription, len(requests))
	for i, r := range requests {
		b.transcribeIDs = append(b.transcribeIDs, r.ID)
		count := max(1, len(r.SourceLines))
		texts := make([]string, count)
		order := make([]int, count)
		for idx := range count {
			texts[idx] = "超长源文"
			order[idx] = idx
		}
		out[i] = dom.SourceTranscription{ID: r.ID, IsText: &yes, LineTexts: texts, LineOrder: order}
	}
	return out, nil
}

func (b *megaNodeBackend) OrderSourceLines(_ context.Context, request SourceNodeRequest, lineTexts []string) (SourceLineOrder, error) {
	b.orderedIDs = append(b.orderedIDs, request.ID)
	order := make([]int, len(lineTexts))
	for i := range lineTexts {
		order[i] = i
	}
	return SourceLineOrder{ID: request.ID, LineOrder: order}, nil
}

func (b *megaNodeBackend) TranslateNodes(_ context.Context, _ string, _ []store.DialogueTurn, _ []store.GlossaryTerm, nodes []NodeTranslationRequest) ([]NodeTranslationResult, error) {
	copied := append([]NodeTranslationRequest(nil), nodes...)
	b.translateCalls = append(b.translateCalls, copied)
	yes := true
	out := make([]NodeTranslationResult, len(nodes))
	for i, n := range nodes {
		src := n.SourceText
		if src == "" {
			src = "普通源文"
		}
		out[i] = NodeTranslationResult{
			ID:             n.ID,
			IsText:         &yes,
			SourceText:     src,
			TargetText:     "译文",
			TargetSegments: []string{"译文"},
		}
	}
	return out, nil
}

func TestSinglePassMegaNodePartitioningRunsOnlyForOversizedNodes(t *testing.T) {
	st, err := store.Open(filepath.Join(t.TempDir(), "meganode.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	ctx := context.Background()
	work, err := st.EnsureWork(ctx, "MegaNode", "ja", "zh-CN", "manga_rtl")
	if err != nil {
		t.Fatal(err)
	}
	var raw bytes.Buffer
	if err := png.Encode(&raw, image.NewGray(image.Rect(0, 0, 16, 16))); err != nil {
		t.Fatal(err)
	}
	page := dom.PageDOM{Nodes: []dom.TextNode{
		{ID: "normal", RecognitionPending: true, BBox: []int{0, 0, 8, 8}, SourceLines: [][]int{{0, 0, 4, 8}, {4, 0, 4, 8}}, CharBudgetMax: 10},
		{ID: "mega", RecognitionPending: true, BBox: []int{8, 0, 8, 8}, SourceLines: [][]int{{8, 0, 2, 8}, {10, 0, 2, 8}, {12, 0, 2, 8}}, CharBudgetMax: 20},
	}}
	backend := &megaNodeBackend{}
	cfg := config.Config{LLMVision: true, LLMSourcePrepass: false, LLMSourceLineBatchSize: 2, LLMBatchSize: 32, TargetLang: "zh-CN"}
	if err := NewCompilerWithBackend(cfg, backend).CompilePageTranslations(ctx, st, work.ID, "1", 0, "zh-CN", &page, raw.Bytes()); err != nil {
		t.Fatalf("CompilePageTranslations: %v", err)
	}
	if len(backend.orderedIDs) != 1 || backend.orderedIDs[0] != "mega" {
		t.Fatalf("expected only mega node partitioned ordering, got %v", backend.orderedIDs)
	}
	if len(backend.transcribeIDs) == 0 {
		t.Fatal("expected mega node partitioned transcription calls")
	}
	for _, tid := range backend.transcribeIDs {
		if !strings.HasPrefix(tid, "mega/lines/") {
			t.Fatalf("unexpected non-mega prepass transcription call %q", tid)
		}
	}
	if len(backend.translateCalls) != 1 || len(backend.translateCalls[0]) != 2 {
		t.Fatalf("expected 1 single-pass translation call covering both nodes, got %+v", backend.translateCalls)
	}
	lockedByID := map[string]bool{}
	for _, req := range backend.translateCalls[0] {
		lockedByID[req.ID] = req.SourceTextLocked
	}
	if lockedByID["normal"] || !lockedByID["mega"] {
		t.Fatalf("expected only mega locked before single-pass translation, got %+v", lockedByID)
	}
}

func TestReasoningEffortDefaultAndUnsupportedFallback(t *testing.T) {
	t.Setenv("PRISM_T_LLM_REASONING_EFFORT", "")
	cfg := config.Default()
	if cfg.LLMReasoningEffort != "none" {
		t.Fatalf("config.Default().LLMReasoningEffort = %q, want %q", cfg.LLMReasoningEffort, "none")
	}
	for _, alias := range []string{"default", "auto", "omit"} {
		if got := config.NormalizeReasoningEffort(alias); got != "" {
			t.Fatalf("NormalizeReasoningEffort(%q) = %q, want empty", alias, got)
		}
	}

	var seenEfforts []string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		effort, has := body["reasoning_effort"]
		if has {
			seenEfforts = append(seenEfforts, effort.(string))
			http.Error(w, `{"error":{"message":"unknown parameter: reasoning_effort"}}`, http.StatusBadRequest)
			return
		}
		seenEfforts = append(seenEfforts, "<omitted>")
		writeChatStream(w, `{"translations":{"n1":{"id":"n1","is_text":true,"source_text":"おい","target_segments":["喂"],"reason":"","rubies":[]}}}`)
	}))
	defer server.Close()

	cfg.DataDir = t.TempDir()
	cfg.LLMBaseURL = server.URL
	backend := NewOpenAICompatibleBackend(cfg)
	results, err := backend.TranslateNodes(context.Background(), "zh-CN", nil, nil, []NodeTranslationRequest{{ID: "n1", SourceText: "おい", SourceTextLocked: true}})
	if err != nil {
		t.Fatalf("expected fallback retry to succeed, got %v", err)
	}
	if len(results) != 1 || results[0].TargetText != "喂" {
		t.Fatalf("unexpected fallback results: %+v", results)
	}
	if len(seenEfforts) != 2 || seenEfforts[0] != "none" || seenEfforts[1] != "<omitted>" {
		t.Fatalf("expected [none, <omitted>], got %v", seenEfforts)
	}

	// Subsequent requests on the same backend skip sending reasoning_effort directly.
	if _, err := backend.TranslateNodes(context.Background(), "zh-CN", nil, nil, []NodeTranslationRequest{{ID: "n1", SourceText: "おい", SourceTextLocked: true}}); err != nil {
		t.Fatalf("second request failed: %v", err)
	}
	if len(seenEfforts) != 3 || seenEfforts[2] != "<omitted>" {
		t.Fatalf("expected third call to omit reasoning_effort proactively, got %v", seenEfforts)
	}
}
