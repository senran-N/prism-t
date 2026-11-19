package server

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"image"
	"image/color"
	"image/png"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"prism-t/internal/config"
	"prism-t/internal/dom"
	"prism-t/internal/engine"
	"prism-t/internal/narrative"
	"prism-t/internal/store"
)

type testTranslationBackend struct{}

func (testTranslationBackend) OrderSourceLines(_ context.Context, request narrative.SourceNodeRequest, lines []string) (narrative.SourceLineOrder, error) {
	order := make([]int, len(lines))
	for i := range order {
		order[i] = i
	}
	return narrative.SourceLineOrder{ID: request.ID, LineOrder: order, Reason: "explicit server test order"}, nil
}

func (testTranslationBackend) TranscribeNodes(_ context.Context, requests []narrative.SourceNodeRequest) ([]dom.SourceTranscription, error) {
	results := make([]dom.SourceTranscription, len(requests))
	for i, request := range requests {
		isText := false
		order := make([]int, max(1, len(request.SourceLines)))
		for j := range order {
			order[j] = j
		}
		results[i] = dom.SourceTranscription{ID: request.ID, IsText: &isText, LineTexts: make([]string, len(order)), LineOrder: order, Reason: "explicit non-writing source fixture"}
	}
	return results, nil
}

func (testTranslationBackend) TranslateNodes(
	_ context.Context,
	_ string,
	_ []store.DialogueTurn,
	_ []store.GlossaryTerm,
	nodes []narrative.NodeTranslationRequest,
) ([]narrative.NodeTranslationResult, error) {
	res := make([]narrative.NodeTranslationResult, len(nodes))
	for i, n := range nodes {
		res[i] = narrative.NodeTranslationResult{
			ID:             n.ID,
			TargetText:     "你好棱镜！",
			TargetSegments: []string{"你好", "棱镜！"},
		}
	}
	return res, nil
}

func makeSmallPNG(w, h int) []byte {
	im := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			im.SetNRGBA(x, y, color.NRGBA{R: 250, G: 250, B: 250, A: 255})
		}
	}
	var buf bytes.Buffer
	_ = png.Encode(&buf, im)
	return buf.Bytes()
}

func TestEndToEndRouterTranslateRerenderAndPSDExport(t *testing.T) {
	samplePNG := makeSmallPNG(80, 100)
	sampleB64 := base64.StdEncoding.EncodeToString(samplePNG)
	roiPNG := makeSmallPNG(24, 24)
	roiB64 := base64.StdEncoding.EncodeToString(roiPNG)

	// Local stub engine server matching Python FastAPI wire schema
	engSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Path {
		case "/health":
			_ = json.NewEncoder(w).Encode(map[string]string{"status": "ok", "version": "0.1.0", "device": "cpu"})
		case "/v1/check_seam":
			_ = json.NewEncoder(w).Encode(map[string]bool{"needs_stitch": false})
		case "/v1/analyze":
			_ = json.NewEncoder(w).Encode(engine.AnalyzeResponse{
				CleanPreviewPNGB64: sampleB64,
				DOM: dom.PageDOM{
					SchemaVersion: 1,
					Width:         80,
					Height:        100,
					FormatProfile: "manga_rtl",
					Nodes: []dom.TextNode{
						{
							ID:            "node_001",
							BBox:          []int{20, 20, 24, 24},
							BubbleKind:    "speech_oval",
							FusionMode:    "clean_interior",
							SourceText:    "こんにちは",
							CharBudgetMin: 2,
							CharBudgetMax: 12,
							Patches: dom.PatchLayers{
								ROI:               []int{20, 20, 24, 24},
								CleanZ0PNGB64:     roiB64,
								StrokeAlphaPNGB64: roiB64,
							},
						},
					},
					Metrics: dom.PageMetrics{BitExactUnmaskedRatio: 1.0, TotalNodes: 1},
				},
			})
		case "/v1/render":
			var rReq engine.RenderRequest
			_ = json.NewDecoder(r.Body).Decode(&rReq)
			for i := range rReq.DOM.Nodes {
				rReq.DOM.Nodes[i].Patches.RenderedZ1PNGB64 = roiB64
			}
			_ = json.NewEncoder(w).Encode(engine.RenderResponse{
				DOM:             rReq.DOM,
				CompositePNGB64: sampleB64,
			})
		}
	}))
	defer engSrv.Close()

	cfg := config.Default()
	cfg.EngineURL = engSrv.URL
	cfg.AdminToken = "secret-test-token-123456"
	cfg.DBPath = filepath.Join(t.TempDir(), "router_test.db")

	st, err := store.Open(cfg.DBPath)
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()

	engClient := engine.NewClient(cfg.EngineURL)
	comp := narrative.NewCompilerWithBackend(cfg, testTranslationBackend{})
	orch := engine.NewOrchestrator(cfg, st, engClient, comp)
	srv := New(cfg, st, engClient, orch, comp, "")
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	// 1. Unauthorized POST must be rejected (401)
	unauthResp, err := http.Post(ts.URL+"/api/v1/translate", "application/json", nil)
	if err != nil {
		t.Fatal(err)
	}
	unauthResp.Body.Close()
	if unauthResp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("expected 401 unauthorized, got %d", unauthResp.StatusCode)
	}

	// 2. Upload page via multipart POST /api/v1/translate with token
	var body bytes.Buffer
	mw := multipart.NewWriter(&body)
	_ = mw.WriteField("work_title", "Prism Test Manga")
	_ = mw.WriteField("chapter_title", "Ch.01")
	part, _ := mw.CreateFormFile("files", "001.png")
	_, _ = part.Write(samplePNG)
	_ = mw.Close()

	req, _ := http.NewRequest(http.MethodPost, ts.URL+"/api/v1/translate", &body)
	req.Header.Set("Content-Type", mw.FormDataContentType())
	req.Header.Set("X-Prism-Token", cfg.AdminToken)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		rawErr, _ := io.ReadAll(resp.Body)
		t.Fatalf("translate returned %d: %s", resp.StatusCode, string(rawErr))
	}

	// 3. Fetch PageDOM, mutate TargetText, trigger live /rerender, and export multi-layer .psd!
	domResp, err := http.Get(ts.URL + "/api/v1/pages/1/dom")
	if err != nil {
		t.Fatal(err)
	}
	var domPayload struct {
		DOM dom.PageDOM `json:"dom"`
	}
	_ = json.NewDecoder(domResp.Body).Decode(&domPayload)
	domResp.Body.Close()
	if len(domPayload.DOM.Nodes) != 1 || domPayload.DOM.Nodes[0].TargetText != "你好棱镜！" {
		t.Fatalf("unexpected translated DOM: %+v", domPayload.DOM)
	}

	domPayload.DOM.Nodes[0].TargetText = "精修台词！"
	rerenderBody, _ := json.Marshal(map[string]any{
		"dom":           domPayload.DOM,
		"target_lang":   "zh-CN",
		"mark_reviewed": true,
	})
	reReq, _ := http.NewRequest(http.MethodPost, ts.URL+"/api/v1/pages/1/rerender", bytes.NewReader(rerenderBody))
	reReq.Header.Set("Content-Type", "application/json")
	reReq.Header.Set("X-Prism-Token", cfg.AdminToken)
	reResp, err := http.DefaultClient.Do(reReq)
	if err != nil {
		t.Fatal(err)
	}
	reResp.Body.Close()
	if reResp.StatusCode != http.StatusOK {
		t.Fatalf("rerender status %d", reResp.StatusCode)
	}

	psdResp, err := http.Get(ts.URL + "/api/v1/pages/1/export/psd")
	if err != nil {
		t.Fatal(err)
	}
	psdBytes, _ := io.ReadAll(psdResp.Body)
	psdResp.Body.Close()
	if psdResp.StatusCode != http.StatusOK || len(psdBytes) < 64 || string(psdBytes[:4]) != "8BPS" {
		t.Fatalf("expected valid 8BPS PSD binary, got status=%d len=%d", psdResp.StatusCode, len(psdBytes))
	}
}

func TestSettingsLoopbackBootstrapAndTestLLM(t *testing.T) {
	webDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(webDir, "index.html"), []byte("<!doctype html><html><head><title>Prism</title></head><body>ok</body></html>"), 0o644); err != nil {
		t.Fatal(err)
	}

	engSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.URL.Path == "/health" {
			_ = json.NewEncoder(w).Encode(map[string]any{
				"status": "ok",
				"device": "auto",
				"hardware": map[string]any{
					"configured_device": "auto",
					"active_device":     "cuda",
					"cuda_available":    true,
					"gpu_name":          "RTX Test",
					"vram_total_mb":     6141,
					"cpu_threads":       16,
					"inference_threads": 8,
				},
			})
		}
	}))
	defer engSrv.Close()

	var llmCalls atomic.Int32
	llmSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		llmCalls.Add(1)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"{\\\"ok\\\":true}\"},\"finish_reason\":\"stop\"}]}\n\n")
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
	defer llmSrv.Close()

	cfg := config.Default()
	cfg.DataDir = t.TempDir()
	cfg.DBPath = filepath.Join(cfg.DataDir, "test.db")
	cfg.EngineURL = engSrv.URL
	cfg.AdminToken = "local-token-abcdef123456"

	st, err := store.Open(cfg.DBPath)
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()

	engClient := engine.NewClient(cfg.EngineURL)
	comp := narrative.NewCompiler(cfg)
	orch := engine.NewOrchestrator(cfg, st, engClient, comp)
	srv := New(cfg, st, engClient, orch, comp, webDir)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	// 1. Loopback direct GET / injects meta token & no-store
	resp, err := http.Get(ts.URL + "/")
	if err != nil {
		t.Fatal(err)
	}
	htmlRaw, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if !strings.Contains(string(htmlRaw), `<meta name="prism-local-token" content="local-token-abcdef123456">`) {
		t.Fatalf("expected loopback token meta tag in index.html, got: %s", string(htmlRaw))
	}
	if resp.Header.Get("Cache-Control") != "no-store" {
		t.Fatalf("expected Cache-Control=no-store on index.html, got %q", resp.Header.Get("Cache-Control"))
	}

	// 2. Proxied request must NOT receive meta token
	fwdReq, _ := http.NewRequest(http.MethodGet, ts.URL+"/", nil)
	fwdReq.Header.Set("X-Forwarded-For", "203.0.113.10")
	fwdResp, err := http.DefaultClient.Do(fwdReq)
	if err != nil {
		t.Fatal(err)
	}
	fwdHTML, _ := io.ReadAll(fwdResp.Body)
	fwdResp.Body.Close()
	if strings.Contains(string(fwdHTML), "prism-local-token") {
		t.Fatalf("forwarded request must never receive auto-login token")
	}

	// 3. Settings requires auth, masks secret, and tests LLM connectivity
	unauthSettings, err := http.Get(ts.URL + "/api/v1/settings")
	if err != nil {
		t.Fatal(err)
	}
	unauthSettings.Body.Close()
	if unauthSettings.StatusCode != http.StatusUnauthorized {
		t.Fatalf("expected 401 for GET /api/v1/settings without token, got %d", unauthSettings.StatusCode)
	}

	testPayload, _ := json.Marshal(map[string]any{
		"device":               "cpu",
		"max_concurrency":      2,
		"target_lang":          "zh-CN",
		"llm_base_url":         llmSrv.URL + "/v1",
		"llm_api_key":          "sk-demo-key-9876",
		"llm_model":            "mock-vision",
		"llm_reasoning_effort": "none",
		"llm_timeout_seconds":  60,
		"llm_vision":           true,
		"llm_batch_size":       16,
	})

	probeReq, _ := http.NewRequest(http.MethodPost, ts.URL+"/api/v1/settings/test-llm", bytes.NewReader(testPayload))
	probeReq.Header.Set("Content-Type", "application/json")
	probeReq.Header.Set("X-Prism-Token", cfg.AdminToken)
	probeResp, err := http.DefaultClient.Do(probeReq)
	if err != nil {
		t.Fatal(err)
	}
	var probeOut map[string]any
	_ = json.NewDecoder(probeResp.Body).Decode(&probeOut)
	probeResp.Body.Close()
	if probeResp.StatusCode != http.StatusOK || probeOut["ok"] != true || llmCalls.Load() != 1 {
		t.Fatalf("expected probe ok, got status=%d body=%+v calls=%d", probeResp.StatusCode, probeOut, llmCalls.Load())
	}

	saveReq, _ := http.NewRequest(http.MethodPost, ts.URL+"/api/v1/settings", bytes.NewReader(testPayload))
	saveReq.Header.Set("Content-Type", "application/json")
	saveReq.Header.Set("X-Prism-Token", cfg.AdminToken)
	saveResp, err := http.DefaultClient.Do(saveReq)
	if err != nil {
		t.Fatal(err)
	}
	var saveOut map[string]any
	_ = json.NewDecoder(saveResp.Body).Decode(&saveOut)
	saveResp.Body.Close()
	if saveResp.StatusCode != http.StatusOK || saveOut["llm_configured"] != true || saveOut["llm_api_key_masked"] != "****9876" {
		t.Fatalf("unexpected save settings response: status=%d body=%+v", saveResp.StatusCode, saveOut)
	}
	if !comp.HasBackend() || orch.Config().MaxConcurrency != 2 {
		t.Fatalf("compiler/orchestrator did not update live")
	}

	loaded, exists, err := config.LoadSettingsFile(config.SettingsPath(cfg.DataDir))
	if err != nil || !exists || loaded.LLMModel != "mock-vision" || loaded.LLMAPIKey != "sk-demo-key-9876" {
		t.Fatalf("settings.json not persisted properly: exists=%v err=%v loaded=%+v", exists, err, loaded)
	}
}

func TestAsyncJobTranslateDownloadAndRetranslate(t *testing.T) {
	samplePNG := makeSmallPNG(80, 100)
	sampleB64 := base64.StdEncoding.EncodeToString(samplePNG)
	roiPNG := makeSmallPNG(24, 24)
	roiB64 := base64.StdEncoding.EncodeToString(roiPNG)

	var analyzeCalls, renderCalls atomic.Int32
	engSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Path {
		case "/health":
			_ = json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
		case "/v1/check_seam":
			_ = json.NewEncoder(w).Encode(map[string]bool{"needs_stitch": false})
		case "/v1/analyze":
			analyzeCalls.Add(1)
			_ = json.NewEncoder(w).Encode(engine.AnalyzeResponse{
				CleanPreviewPNGB64: sampleB64,
				DOM: dom.PageDOM{
					SchemaVersion: 1, Width: 80, Height: 100, FormatProfile: "manga_rtl",
					Nodes: []dom.TextNode{{
						ID: "node_001", BBox: []int{20, 20, 24, 24}, SourceText: "こんにちは",
						Patches: dom.PatchLayers{ROI: []int{20, 20, 24, 24}, CleanZ0PNGB64: roiB64, StrokeAlphaPNGB64: roiB64},
					}},
					Metrics: dom.PageMetrics{BitExactUnmaskedRatio: 1.0, TotalNodes: 1},
				},
			})
		case "/v1/render":
			renderCalls.Add(1)
			var rReq engine.RenderRequest
			_ = json.NewDecoder(r.Body).Decode(&rReq)
			_ = json.NewEncoder(w).Encode(engine.RenderResponse{DOM: rReq.DOM, CompositePNGB64: sampleB64})
		}
	}))
	defer engSrv.Close()

	cfg := config.Default()
	cfg.EngineURL = engSrv.URL
	cfg.AdminToken = "job-secret-123456"
	cfg.DBPath = filepath.Join(t.TempDir(), "jobs_test.db")

	st, err := store.Open(cfg.DBPath)
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()

	engClient := engine.NewClient(cfg.EngineURL)
	comp := narrative.NewCompilerWithBackend(cfg, testTranslationBackend{})
	orch := engine.NewOrchestrator(cfg, st, engClient, comp)
	srv := New(cfg, st, engClient, orch, comp, "")
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	var body bytes.Buffer
	mw := multipart.NewWriter(&body)
	_ = mw.WriteField("work_title", "Async Work")
	_ = mw.WriteField("chapter_title", "Ch.01")
	part, _ := mw.CreateFormFile("files", "page01.png")
	_, _ = part.Write(samplePNG)
	_ = mw.Close()

	req, _ := http.NewRequest(http.MethodPost, ts.URL+"/api/v1/jobs/translate", &body)
	req.Header.Set("Content-Type", mw.FormDataContentType())
	req.Header.Set("X-Prism-Token", cfg.AdminToken)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusAccepted {
		raw, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		t.Fatalf("expected 202 Accepted, got %d: %s", resp.StatusCode, string(raw))
	}
	var startOut struct {
		Job JobSnapshot `json:"job"`
	}
	_ = json.NewDecoder(resp.Body).Decode(&startOut)
	resp.Body.Close()

	deadline := time.Now().Add(5 * time.Second)
	var finalJob JobSnapshot
	for time.Now().Before(deadline) {
		jResp, jErr := http.Get(ts.URL + "/api/v1/jobs/" + startOut.Job.ID)
		if jErr != nil {
			t.Fatal(jErr)
		}
		var jWrap struct {
			Job JobSnapshot `json:"job"`
		}
		_ = json.NewDecoder(jResp.Body).Decode(&jWrap)
		jResp.Body.Close()
		finalJob = jWrap.Job
		if finalJob.Status == "completed" || finalJob.Status == "failed" {
			break
		}
		time.Sleep(25 * time.Millisecond)
	}
	if finalJob.Status != "completed" || !reflect.DeepEqual(finalJob.PageIDs, []int64{1}) {
		t.Fatalf("expected completed job with page_ids=[1], got %+v", finalJob)
	}

	dlResp, err := http.Get(ts.URL + "/api/v1/pages/1/composite?download=1")
	if err != nil {
		t.Fatal(err)
	}
	dlResp.Body.Close()
	if !strings.Contains(dlResp.Header.Get("Content-Disposition"), "page01_translated.png") {
		t.Fatalf("expected attachment filename page01_translated.png, got %q", dlResp.Header.Get("Content-Disposition"))
	}

	// Re-translate existing page without repeating /v1/analyze
	reReq, _ := http.NewRequest(http.MethodPost, ts.URL+"/api/v1/pages/1/translate", strings.NewReader(`{"target_lang":"zh-CN"}`))
	reReq.Header.Set("Content-Type", "application/json")
	reReq.Header.Set("X-Prism-Token", cfg.AdminToken)
	reResp, err := http.DefaultClient.Do(reReq)
	if err != nil {
		t.Fatal(err)
	}
	reResp.Body.Close()
	if reResp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200 from retranslate, got %d", reResp.StatusCode)
	}
	if analyzeCalls.Load() != 1 || renderCalls.Load() != 2 {
		t.Fatalf("expected analyze=1 and render=2 after retranslate, got analyze=%d render=%d", analyzeCalls.Load(), renderCalls.Load())
	}
}
