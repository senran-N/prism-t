package engine

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"image"
	"image/color"
	"image/png"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"

	"prism-t/internal/archive"
	"prism-t/internal/config"
	"prism-t/internal/dom"
	"prism-t/internal/narrative"
	"prism-t/internal/store"
)

func makeEngineTestPNG(w, h int) []byte {
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.SetNRGBA(x, y, color.NRGBA{R: 245, G: 245, B: 245, A: 255})
		}
	}
	var buf bytes.Buffer
	_ = png.Encode(&buf, img)
	return buf.Bytes()
}

type overlappedBackend struct {
	onTranslateStart func()
	waitForRestore   <-chan struct{}
}

func (*overlappedBackend) TranscribeNodes(_ context.Context, requests []narrative.SourceNodeRequest) ([]dom.SourceTranscription, error) {
	return nil, nil
}

func (*overlappedBackend) OrderSourceLines(_ context.Context, request narrative.SourceNodeRequest, lines []string) (narrative.SourceLineOrder, error) {
	order := make([]int, len(lines))
	for i := range order {
		order[i] = i
	}
	return narrative.SourceLineOrder{ID: request.ID, LineOrder: order}, nil
}

func (b *overlappedBackend) TranslateNodes(
	_ context.Context,
	_ string,
	_ []store.DialogueTurn,
	_ []store.GlossaryTerm,
	nodes []narrative.NodeTranslationRequest,
) ([]narrative.NodeTranslationResult, error) {
	if b.onTranslateStart != nil {
		b.onTranslateStart()
	}
	if b.waitForRestore != nil {
		select {
		case <-b.waitForRestore:
		case <-time.After(3 * time.Second):
		}
	}
	yes := true
	no := false
	out := make([]narrative.NodeTranslationResult, 0, len(nodes))
	for _, n := range nodes {
		if n.ID == "node_002" {
			out = append(out, narrative.NodeTranslationResult{
				ID:     n.ID,
				IsText: &no,
				Reason: "uniform collar fold",
			})
			continue
		}
		out = append(out, narrative.NodeTranslationResult{
			ID:             n.ID,
			IsText:         &yes,
			SourceText:     "本当に？",
			TargetText:     "这是超长测试译文",
			TargetSegments: []string{"这是", "超长", "测试", "译文"},
		})
	}
	return out, nil
}

func TestDetectLayoutRestorePageOverlappedMerge(t *testing.T) {
	rawPNG := makeEngineTestPNG(120, 160)
	rawB64 := base64.StdEncoding.EncodeToString(rawPNG)
	roiB64 := base64.StdEncoding.EncodeToString(makeEngineTestPNG(24, 24))

	restoreStarted := make(chan struct{})
	translateStarted := make(chan struct{})
	var overlapped atomic.Bool
	var renderSeenNodes int

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Path {
		case "/v1/check_seam":
			_ = json.NewEncoder(w).Encode(map[string]bool{"needs_stitch": false})
		case "/v1/detect_layout":
			_ = json.NewEncoder(w).Encode(DetectLayoutResponse{
				LayoutToken: "tok-123",
				DOM: dom.PageDOM{
					SchemaVersion: 1,
					Width:         120,
					Height:        160,
					FormatProfile: "manga_rtl",
					Nodes: []dom.TextNode{
						{
							ID:                 "node_001",
							ReadingOrder:       0,
							Track:              "primary",
							BBox:               []int{10, 10, 40, 40},
							BubbleKind:         "speech_oval",
							RecognitionPending: true,
							CharBudgetMin:      1,
							CharBudgetMax:      4,
							Style: dom.StyleSpec{
								Orientation:    "vertical",
								FgColorRGBA:    []int{0, 0, 0, 255},
								OutlineWidthPx: 0,
								BlendAlpha:     1.0,
							},
						},
						{
							ID:                 "node_002",
							ReadingOrder:       1,
							Track:              "aside",
							BBox:               []int{60, 60, 20, 20},
							BubbleKind:         "floating_sfx",
							RecognitionPending: true,
							CharBudgetMin:      1,
							CharBudgetMax:      4,
						},
					},
					Metrics: dom.PageMetrics{TotalNodes: 2},
				},
			})
		case "/v1/restore_page":
			var req RestoreRequest
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				t.Errorf("decode restore_page: %v", err)
			}
			if req.LayoutToken != "tok-123" {
				t.Errorf("unexpected layout token: %q", req.LayoutToken)
			}
			close(restoreStarted)
			select {
			case <-translateStarted:
				overlapped.Store(true)
			case <-time.After(2 * time.Second):
			}
			_ = json.NewEncoder(w).Encode(AnalyzeResponse{
				CleanPreviewPNGB64: rawB64,
				DOM: dom.PageDOM{
					SchemaVersion: 1,
					Width:         120,
					Height:        160,
					FormatProfile: "manga_rtl",
					Nodes: []dom.TextNode{
						{
							ID:              "node_001",
							ReadingOrder:    0,
							Track:           "primary",
							BBox:            []int{10, 10, 40, 40},
							BubbleKind:      "speech_oval",
							RestorationPath: "lama_masked",
							CharBudgetMin:   1,
							CharBudgetMax:   4,
							Style: dom.StyleSpec{
								Orientation:      "vertical",
								FontCategory:     "mincho",
								FgColorRGBA:      []int{25, 25, 28, 255},
								OutlineColorRGBA: []int{255, 255, 255, 255},
								OutlineWidthPx:   2.5,
								BlendAlpha:       0.88,
								SourceFontSizePx: 26.0,
							},
							Patches: dom.PatchLayers{
								ROI:                     []int{10, 10, 40, 40},
								CleanZ0PNGB64:           roiB64,
								StrokeAlphaPNGB64:       roiB64,
								Z2OccluderMaskB64:       roiB64,
								ShadingMapPNGB64:        roiB64,
								SourceObservedInkPNGB64: "debug-ink-1",
							},
							Uncertainty: dom.UncertaintySpec{
								RestorationResidual:   0.04,
								AutomaticRepairPixels: 12,
							},
						},
						{
							ID:           "node_002",
							ReadingOrder: 1,
							Track:        "aside",
							BBox:         []int{60, 60, 20, 20},
							BubbleKind:   "floating_sfx",
							Patches:      dom.PatchLayers{CleanZ0PNGB64: roiB64},
						},
					},
					Metrics: dom.PageMetrics{TotalNodes: 2, DeepFusionNodes: 1},
				},
			})
		case "/v1/render":
			var req RenderRequest
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				t.Errorf("decode render: %v", err)
			}
			renderSeenNodes = len(req.DOM.Nodes)
			if renderSeenNodes != 1 || req.DOM.Nodes[0].ID != "node_001" {
				t.Errorf("expected rejected node_002 dropped before render, got %+v", req.DOM.Nodes)
			}
			if req.DOM.Nodes[0].Patches.SourceObservedInkPNGB64 != "" {
				t.Errorf("expected debug layer stripped before render")
			}
			req.DOM.Nodes[0].Patches.RenderedZ1PNGB64 = roiB64
			_ = json.NewEncoder(w).Encode(RenderResponse{
				DOM:             req.DOM,
				CompositePNGB64: rawB64,
			})
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	st, err := store.Open(filepath.Join(t.TempDir(), "overlap.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()

	cfg := config.Default()
	cfg.LLMVision = true
	cfg.MaxConcurrency = 1
	backend := &overlappedBackend{
		onTranslateStart: func() { close(translateStarted) },
		waitForRestore:   restoreStarted,
	}
	orch := NewOrchestrator(cfg, st, NewClient(srv.URL), narrative.NewCompilerWithBackend(cfg, backend))
	_, results, err := orch.TranslatePages(context.Background(), []archive.InputPage{
		{Filename: "001.png", PNGBytes: rawPNG, Width: 120, Height: 160},
	}, TranslateBatchOptions{WorkTitle: "Overlap", ChapterTitle: "Ch1"})
	if err != nil {
		t.Fatalf("TranslatePages: %v", err)
	}
	if !overlapped.Load() {
		t.Fatal("RestorePage and CompilePageTranslations did not run in parallel")
	}
	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}
	got := results[0].DOM
	if len(got.Nodes) != 1 || got.Nodes[0].ID != "node_001" {
		t.Fatalf("expected 1 merged node, got %+v", got.Nodes)
	}
	if len(got.RejectedRegions) != 1 || got.RejectedRegions[0].ID != "node_002" {
		t.Fatalf("expected node_002 in RejectedRegions, got %+v", got.RejectedRegions)
	}
	n := got.Nodes[0]
	if n.TargetText != "这是超长测试译文" || n.SourceText != "本当に？" || n.RecognitionPending {
		t.Fatalf("translation fields not merged: %+v", n)
	}
	if n.Style.FontCategory != "mincho" || len(n.Style.FgColorRGBA) != 4 || n.Style.FgColorRGBA[0] != 25 ||
		n.Style.OutlineWidthPx != 2.5 || n.Style.BlendAlpha != 0.88 || n.Style.SourceFontSizePx != 26.0 {
		t.Fatalf("restored typography/stroke Style was overwritten during merge: %+v", n.Style)
	}
	if n.Patches.CleanZ0PNGB64 != roiB64 || n.Patches.SourceObservedInkPNGB64 != "debug-ink-1" || n.Patches.RenderedZ1PNGB64 != roiB64 {
		t.Fatalf("restored/debug/rendered patches lost: %+v", n.Patches)
	}
	if !n.Uncertainty.NeedsReview || n.Uncertainty.AutomaticRepairPixels != 12 || got.Metrics.TotalNodes != 1 || got.Metrics.NeedsReviewNodes != 1 {
		t.Fatalf("merged uncertainty/metrics mismatch: uncertainty=%+v metrics=%+v", n.Uncertainty, got.Metrics)
	}
	rec, err := st.GetPage(context.Background(), results[0].PageID)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(rec.RawPNG, rawPNG) || rec.Summary.Status != "translated" {
		t.Fatalf("stored page raw_png/status mismatch: status=%q len(raw)=%d", rec.Summary.Status, len(rec.RawPNG))
	}
}

func TestDetectLayout404FallbackToAnalyze(t *testing.T) {
	rawPNG := makeEngineTestPNG(80, 80)
	rawB64 := base64.StdEncoding.EncodeToString(rawPNG)
	var detectCalls, analyzeCalls, renderCalls atomic.Int32

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/check_seam":
			_ = json.NewEncoder(w).Encode(map[string]bool{"needs_stitch": false})
		case "/v1/detect_layout":
			detectCalls.Add(1)
			http.NotFound(w, r)
		case "/v1/analyze":
			analyzeCalls.Add(1)
			_ = json.NewEncoder(w).Encode(AnalyzeResponse{
				CleanPreviewPNGB64: rawB64,
				DOM: dom.PageDOM{
					SchemaVersion: 1,
					Width:         80,
					Height:        80,
					FormatProfile: "manga_rtl",
					Nodes: []dom.TextNode{{
						ID:            "n1",
						BBox:          []int{10, 10, 20, 20},
						SourceText:    "おい",
						CharBudgetMax: 10,
					}},
					Metrics: dom.PageMetrics{TotalNodes: 1},
				},
			})
		case "/v1/render":
			renderCalls.Add(1)
			var req RenderRequest
			_ = json.NewDecoder(r.Body).Decode(&req)
			_ = json.NewEncoder(w).Encode(RenderResponse{DOM: req.DOM, CompositePNGB64: rawB64})
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	st, err := store.Open(filepath.Join(t.TempDir(), "fallback.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()

	cfg := config.Default()
	orch := NewOrchestrator(cfg, st, NewClient(srv.URL), narrative.NewCompilerWithBackend(cfg, &overlappedBackend{}))
	_, results, err := orch.TranslatePages(context.Background(), []archive.InputPage{
		{Filename: "001.png", PNGBytes: rawPNG, Width: 80, Height: 80},
	}, TranslateBatchOptions{WorkTitle: "Fallback", ChapterTitle: "Ch1"})
	if err != nil {
		t.Fatalf("TranslatePages fallback failed: %v", err)
	}
	if detectCalls.Load() != 1 || analyzeCalls.Load() != 1 || renderCalls.Load() != 1 {
		t.Fatalf("unexpected call counts: detect=%d analyze=%d render=%d", detectCalls.Load(), analyzeCalls.Load(), renderCalls.Load())
	}
	if len(results) != 1 || results[0].DOM.Nodes[0].TargetText != "这是超长测试译文" {
		t.Fatalf("unexpected fallback result: %+v", results)
	}
}

func TestRenderStripsAndRestoresDebugPatchLayers(t *testing.T) {
	rawPNG := makeEngineTestPNG(64, 64)
	rawB64 := base64.StdEncoding.EncodeToString(rawPNG)

	var seenPatches dom.PatchLayers
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req RenderRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatal(err)
		}
		seenPatches = req.DOM.Nodes[0].Patches
		req.DOM.Nodes[0].Patches.RenderedZ1PNGB64 = "rendered-z1"
		_ = json.NewEncoder(w).Encode(RenderResponse{
			DOM:             req.DOM,
			CompositePNGB64: rawB64,
		})
	}))
	defer srv.Close()

	inputDOM := dom.PageDOM{
		SchemaVersion: 1,
		Width:         64,
		Height:        64,
		FormatProfile: "manga_rtl",
		Nodes: []dom.TextNode{{
			ID: "n1",
			Patches: dom.PatchLayers{
				ROI:                      []int{0, 0, 16, 16},
				CleanZ0PNGB64:            "keep-z0",
				StrokeAlphaPNGB64:        "keep-alpha",
				Z2OccluderMaskB64:        "keep-z2",
				ShadingMapPNGB64:         "keep-shading",
				SourceObservedInkPNGB64:  "strip-1",
				SourceDomainPNGB64:       "strip-2",
				RefinedStrokePNGB64:      "strip-3",
				InpaintUnknownPNGB64:     "strip-4",
				InferredBackgroundPNGB64: "strip-5",
				BoundaryOpacityPNGB64:    "strip-6",
			},
		}},
	}

	resp, err := NewClient(srv.URL).Render(context.Background(), rawPNG, inputDOM, "zh-CN")
	if err != nil {
		t.Fatal(err)
	}
	if seenPatches.SourceObservedInkPNGB64 != "" || seenPatches.SourceDomainPNGB64 != "" ||
		seenPatches.RefinedStrokePNGB64 != "" || seenPatches.InpaintUnknownPNGB64 != "" ||
		seenPatches.InferredBackgroundPNGB64 != "" || seenPatches.BoundaryOpacityPNGB64 != "" {
		t.Fatalf("expected all 6 ledger-only debug layers stripped on wire, got %+v", seenPatches)
	}
	if seenPatches.CleanZ0PNGB64 != "keep-z0" || seenPatches.StrokeAlphaPNGB64 != "keep-alpha" ||
		seenPatches.Z2OccluderMaskB64 != "keep-z2" || seenPatches.ShadingMapPNGB64 != "keep-shading" {
		t.Fatalf("expected core render layers preserved on wire, got %+v", seenPatches)
	}
	outPatches := resp.DOM.Nodes[0].Patches
	if outPatches.SourceObservedInkPNGB64 != "strip-1" || outPatches.SourceDomainPNGB64 != "strip-2" ||
		outPatches.RefinedStrokePNGB64 != "strip-3" || outPatches.InpaintUnknownPNGB64 != "strip-4" ||
		outPatches.InferredBackgroundPNGB64 != "strip-5" || outPatches.BoundaryOpacityPNGB64 != "strip-6" ||
		outPatches.RenderedZ1PNGB64 != "rendered-z1" {
		t.Fatalf("expected all 6 debug layers restored onto RenderResponse DOM, got %+v", outPatches)
	}
	if inputDOM.Nodes[0].Patches.SourceObservedInkPNGB64 != "strip-1" {
		t.Fatal("input PageDOM was mutated by Render")
	}
}
