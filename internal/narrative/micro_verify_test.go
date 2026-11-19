package narrative

import (
	"bytes"
	"context"
	"encoding/json"
	"image"
	"image/color"
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

func makeBlankVisionTestPNG(w, h int) []byte {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.Set(x, y, color.RGBA{R: 240, G: 240, B: 240, A: 255})
		}
	}
	var buf bytes.Buffer
	_ = png.Encode(&buf, img)
	return buf.Bytes()
}

func TestTargetedMicroVerificationRejectsFalsePositiveAndFixesBrackets(t *testing.T) {
	type capturedCall struct {
		systemPrompt    string
		reasoningEffort string
		nodeIDs         []string
		hasQuads        bool
		hasDraft        bool
	}
	var calls []capturedCall

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			ReasoningEffort string `json:"reasoning_effort"`
			Messages        []struct {
				Content json.RawMessage `json:"content"`
			} `json:"messages"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode LLM request: %v", err)
		}
		var sysPrompt string
		_ = json.Unmarshal(req.Messages[0].Content, &sysPrompt)

		var parts []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		}
		if err := json.Unmarshal(req.Messages[1].Content, &parts); err != nil {
			t.Fatalf("unmarshal multipart user content: %v", err)
		}
		var payload struct {
			Nodes []NodeTranslationRequest `json:"nodes"`
		}
		if err := json.Unmarshal([]byte(parts[0].Text), &payload); err != nil {
			t.Fatalf("unmarshal userJSON: %v", err)
		}
		var ids []string
		hasQuads := false
		hasDraft := false
		for _, n := range payload.Nodes {
			ids = append(ids, n.ID)
			if len(n.SourceQuadrilaterals) > 0 {
				hasQuads = true
			}
			if n.DraftTranslation != "" {
				hasDraft = true
			}
		}
		calls = append(calls, capturedCall{
			systemPrompt:    sysPrompt,
			reasoningEffort: req.ReasoningEffort,
			nodeIDs:         ids,
			hasQuads:        hasQuads,
			hasDraft:        hasDraft,
		})

		if len(calls) == 1 {
			writeChatStream(w, `{"translations":{
				"node_good":{"id":"node_good","is_text":true,"source_text":"こんにちは","target_segments":["你好！"],"reason":"","rubies":[]},
				"node_fp":{"id":"node_fp","is_text":true,"source_text":"希","target_segments":["希"],"reason":"","rubies":[]},
				"node_unbal":{"id":"node_unbal","is_text":true,"source_text":"（これはまずい）","target_segments":["（这可不妙"],"reason":"","rubies":[]}
			}}`)
			return
		}
		writeChatStream(w, `{"translations":{
			"node_fp":{"id":"node_fp","is_text":false,"source_text":"","target_segments":[],"reason":"Uniform collar fold, no writing","rubies":[]},
			"node_unbal":{"id":"node_unbal","is_text":true,"source_text":"（これはまずい）","target_segments":["（这可不妙）"],"reason":"","rubies":[]}
		}}`)
	}))
	defer server.Close()

	st, err := store.Open(filepath.Join(t.TempDir(), "micro_verify.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	ctx := context.Background()
	work, err := st.EnsureWork(ctx, "MicroVerify", "ja", "zh-CN", "manga_rtl")
	if err != nil {
		t.Fatal(err)
	}

	page := dom.PageDOM{
		Nodes: []dom.TextNode{
			{
				ID:                   "node_good",
				ReadingOrder:         0,
				BubbleKind:           "speech_oval",
				BBox:                 []int{20, 20, 140, 120},
				SourceLines:          [][]int{{40, 30, 30, 90}},
				SourceQuadrilaterals: [][][]float64{{{40, 30}, {70, 30}, {70, 120}, {40, 120}}},
				RecognitionPending:   true,
				CharBudgetMax:        12,
			},
			{
				ID:                   "node_fp",
				ReadingOrder:         1,
				BubbleKind:           "floating_sfx",
				BBox:                 []int{220, 100, 28, 36},
				SourceLines:          [][]int{{222, 102, 24, 32}},
				SourceQuadrilaterals: [][][]float64{{{222, 102}, {246, 102}, {246, 134}, {222, 134}}},
				SourceReadings:       []dom.SourceReading{{LineIndex: 0, Recognizer: "a", Text: "希"}, {LineIndex: 0, Recognizer: "b", Text: ""}},
				RecognitionPending:   true,
				CharBudgetMax:        4,
			},
			{
				ID:                 "node_unbal",
				ReadingOrder:       2,
				BubbleKind:         "speech_oval",
				BBox:               []int{20, 180, 150, 100},
				SourceLines:        [][]int{{30, 190, 30, 80}},
				RecognitionPending: true,
				CharBudgetMax:      12,
			},
		},
	}

	cfg := config.Config{
		DataDir:            t.TempDir(),
		LLMBaseURL:         server.URL,
		LLMModel:           "mock-vision",
		LLMReasoningEffort: "none",
		LLMVision:          true,
		LLMVisionAudit:     false,
		LLMBatchSize:       32,
		TargetLang:         "zh-CN",
	}
	if err := NewCompiler(cfg).CompilePageTranslations(ctx, st, work.ID, "1", 0, "zh-CN", &page, makeBlankVisionTestPNG(400, 400)); err != nil {
		t.Fatalf("CompilePageTranslations: %v", err)
	}

	if len(calls) != 2 {
		t.Fatalf("expected 2 LLM calls (single-pass + targeted micro-verify), got %d", len(calls))
	}
	// Call 1 assertions: mode-aware compact prompt, lean userJSON, reasoning_effort="none".
	c1 := calls[0]
	if c1.reasoningEffort != "none" || c1.hasQuads || c1.hasDraft || len(c1.nodeIDs) != 3 {
		t.Fatalf("unexpected call 1 metadata: %+v", c1)
	}
	if strings.Contains(c1.systemPrompt, "source_text_locked:true") ||
		strings.Contains(c1.systemPrompt, "draft_translation") ||
		strings.Contains(c1.systemPrompt, "ruby_pairs") {
		t.Fatalf("single-pass system prompt included unused mode rules")
	}
	if !strings.Contains(c1.systemPrompt, "clothing folds") {
		t.Fatalf("expected strengthened false-positive rejection warning in vision system prompt")
	}

	// Call 2 assertions: ONLY suspicious nodes, preserves configured reasoning_effort="none", draft & complete page draft populated.
	c2 := calls[1]
	if c2.reasoningEffort != "none" || !c2.hasDraft || len(c2.nodeIDs) != 2 || c2.nodeIDs[0] != "node_fp" || c2.nodeIDs[1] != "node_unbal" {
		t.Fatalf("unexpected micro-verify call 2 metadata: %+v", c2)
	}
	if !strings.Contains(c2.systemPrompt, "Complete page draft") {
		t.Fatalf("expected Complete page draft context in micro-verify system prompt")
	}

	if len(page.RejectedRegions) != 1 || page.RejectedRegions[0].ID != "node_fp" {
		t.Fatalf("expected node_fp rejected, got %+v", page.RejectedRegions)
	}
	if len(page.Nodes) != 2 || page.Nodes[0].ID != "node_good" || page.Nodes[1].ID != "node_unbal" {
		t.Fatalf("unexpected surviving nodes: %+v", page.Nodes)
	}
	if page.Nodes[1].TargetText != "（这可不妙）" {
		t.Fatalf("expected unbalanced punctuation fixed to （这可不妙）, got %q", page.Nodes[1].TargetText)
	}
}

func TestTargetedMicroVerificationSkippedOnCleanPageAndCrossNodeSplit(t *testing.T) {
	httpCalls := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		httpCalls++
		writeChatStream(w, `{"translations":{
			"sfx_ok":{"id":"sfx_ok","is_text":true,"source_text":"バッ","target_segments":["刷！"],"reason":"","rubies":[]},
			"split_open":{"id":"split_open","is_text":true,"source_text":"（まさか","target_segments":["（难道说"],"reason":"","rubies":[]},
			"split_close":{"id":"split_close","is_text":true,"source_text":"本当なのか）","target_segments":["是真的吗）"],"reason":"","rubies":[]}
		}}`)
	}))
	defer server.Close()

	st, err := store.Open(filepath.Join(t.TempDir(), "clean_skip.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	ctx := context.Background()
	work, err := st.EnsureWork(ctx, "CleanSkip", "ja", "zh-CN", "manga_rtl")
	if err != nil {
		t.Fatal(err)
	}

	page := dom.PageDOM{
		Nodes: []dom.TextNode{
			{
				ID:                 "sfx_ok",
				ReadingOrder:       0,
				BubbleKind:         "floating_sfx",
				BBox:               []int{10, 10, 30, 30},
				RecognitionPending: true,
				SourceReadings:     []dom.SourceReading{{LineIndex: 0, Recognizer: "a", Text: "バッ"}, {LineIndex: 0, Recognizer: "b", Text: "バッ"}},
				CharBudgetMax:      6,
			},
			{
				ID:                 "split_open",
				ReadingOrder:       1,
				BubbleKind:         "speech_oval",
				BBox:               []int{60, 10, 100, 100},
				RecognitionPending: true,
				CharBudgetMax:      10,
			},
			{
				ID:                 "split_close",
				ReadingOrder:       2,
				BubbleKind:         "speech_oval",
				BBox:               []int{180, 10, 100, 100},
				RecognitionPending: true,
				CharBudgetMax:      10,
			},
		},
	}
	cfg := config.Config{
		DataDir:            t.TempDir(),
		LLMBaseURL:         server.URL,
		LLMModel:           "mock-vision",
		LLMReasoningEffort: "none",
		LLMVision:          true,
		LLMBatchSize:       32,
		TargetLang:         "zh-CN",
	}
	if err := NewCompiler(cfg).CompilePageTranslations(ctx, st, work.ID, "1", 0, "zh-CN", &page, makeBlankVisionTestPNG(320, 200)); err != nil {
		t.Fatalf("CompilePageTranslations: %v", err)
	}
	if httpCalls != 1 {
		t.Fatalf("expected 0 extra micro-verification calls on clean/cross-split page, got %d total calls", httpCalls)
	}
}

func TestTargetedMicroVerificationTriggersWhenCrossNodeSplitDropsClosingBracket(t *testing.T) {
	var callIDs [][]string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Messages []struct {
				Content json.RawMessage `json:"content"`
			} `json:"messages"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatal(err)
		}
		var parts []struct {
			Text string `json:"text"`
		}
		if err := json.Unmarshal(req.Messages[1].Content, &parts); err != nil {
			t.Fatal(err)
		}
		var payload struct {
			Nodes []NodeTranslationRequest `json:"nodes"`
		}
		if err := json.Unmarshal([]byte(parts[0].Text), &payload); err != nil {
			t.Fatal(err)
		}
		var ids []string
		for _, n := range payload.Nodes {
			ids = append(ids, n.ID)
		}
		callIDs = append(callIDs, ids)
		if len(callIDs) == 1 {
			writeChatStream(w, `{"translations":{
				"split_open":{"id":"split_open","is_text":true,"source_text":"（まさか","target_segments":["（难道"],"reason":"","rubies":[]},
				"split_close":{"id":"split_close","is_text":true,"source_text":"本当なのか）","target_segments":["是真的吗"],"reason":"","rubies":[]}
			}}`)
			return
		}
		writeChatStream(w, `{"translations":{
			"split_open":{"id":"split_open","is_text":true,"source_text":"（まさか","target_segments":["（难道）"],"reason":"","rubies":[]}
		}}`)
	}))
	defer server.Close()

	st, err := store.Open(filepath.Join(t.TempDir(), "dropped_split.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	ctx := context.Background()
	work, err := st.EnsureWork(ctx, "DroppedSplit", "ja", "zh-CN", "manga_rtl")
	if err != nil {
		t.Fatal(err)
	}

	page := dom.PageDOM{
		Nodes: []dom.TextNode{
			{ID: "split_open", ReadingOrder: 0, BubbleKind: "speech_oval", BBox: []int{20, 10, 100, 100}, RecognitionPending: true, CharBudgetMax: 10},
			{ID: "split_close", ReadingOrder: 1, BubbleKind: "speech_oval", BBox: []int{140, 10, 100, 100}, RecognitionPending: true, CharBudgetMax: 10},
		},
	}
	cfg := config.Config{
		DataDir:            t.TempDir(),
		LLMBaseURL:         server.URL,
		LLMModel:           "mock-vision",
		LLMReasoningEffort: "none",
		LLMVision:          true,
		LLMBatchSize:       32,
		TargetLang:         "zh-CN",
	}
	if err := NewCompiler(cfg).CompilePageTranslations(ctx, st, work.ID, "1", 0, "zh-CN", &page, makeBlankVisionTestPNG(260, 140)); err != nil {
		t.Fatalf("CompilePageTranslations: %v", err)
	}
	if len(callIDs) != 2 || len(callIDs[1]) != 1 || callIDs[1][0] != "split_open" {
		t.Fatalf("expected micro-verification triggered on unmatched split_open, got calls=%v", callIDs)
	}
}
