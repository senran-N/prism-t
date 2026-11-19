package server

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"prism-t/internal/config"
	"prism-t/internal/dom"
	"prism-t/internal/engine"
	"prism-t/internal/narrative"
	"prism-t/internal/source"
	"prism-t/internal/store"
)

func TestBookshelfDeleteRenameAndSourceStreamingTranslate(t *testing.T) {
	samplePNG := makeSmallPNG(80, 100)
	sampleB64 := base64.StdEncoding.EncodeToString(samplePNG)
	roiB64 := base64.StdEncoding.EncodeToString(makeSmallPNG(24, 24))

	// Stub remote comic image server
	imgSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/manifest.json":
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"title":"Stream Comic","pages":["/p1.png","/p2.png"]}`))
		case "/p1.png", "/p2.png":
			w.Header().Set("Content-Type", "image/png")
			_, _ = w.Write(samplePNG)
		default:
			http.NotFound(w, r)
		}
	}))
	defer imgSrv.Close()

	// Stub local Python engine
	engSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		pd := dom.PageDOM{
			Width: 80, Height: 100, FormatProfile: "manga_rtl",
			Nodes: []dom.TextNode{{
				ID: "n1", ReadingOrder: 0, BBox: []int{10, 10, 24, 24},
				SourceText: "テスト", TargetText: "测试",
				Patches: dom.PatchLayers{ROI: []int{10, 10, 24, 24}, CleanZ0PNGB64: roiB64},
			}},
			Metrics: dom.PageMetrics{BitExactUnmaskedRatio: 1.0, TotalNodes: 1},
		}
		switch r.URL.Path {
		case "/v1/detect_layout":
			_ = json.NewEncoder(w).Encode(engine.DetectLayoutResponse{DOM: pd, LayoutToken: "tok"})
		case "/v1/restore_page", "/v1/analyze":
			_ = json.NewEncoder(w).Encode(engine.AnalyzeResponse{DOM: pd, CleanPreviewPNGB64: sampleB64})
		case "/v1/render":
			_ = json.NewEncoder(w).Encode(engine.RenderResponse{DOM: pd, CompositePNGB64: sampleB64})
		default:
			http.NotFound(w, r)
		}
	}))
	defer engSrv.Close()

	dbPath := filepath.Join(t.TempDir(), "test.db")
	st, err := store.Open(dbPath)
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()

	cfg := config.Default()
	cfg.AdminToken = "test-secret-token"
	cfg.EngineURL = engSrv.URL
	cfg.LLMVision = false

	engClient := engine.NewClient(engSrv.URL)
	comp := narrative.NewCompilerWithBackend(cfg, testTranslationBackend{})
	orch := engine.NewOrchestrator(cfg, st, engClient, comp)
	srv := New(cfg, st, engClient, orch, comp, "")
	srv.srcClient = &source.Client{HTTP: imgSrv.Client()}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	authDo := func(method, path string, body any) *http.Response {
		t.Helper()
		var rdr *bytes.Reader
		if body != nil {
			b, _ := json.Marshal(body)
			rdr = bytes.NewReader(b)
		} else {
			rdr = bytes.NewReader(nil)
		}
		req, err := http.NewRequest(method, ts.URL+path, rdr)
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Authorization", "Bearer "+cfg.AdminToken)
		if body != nil {
			req.Header.Set("Content-Type", "application/json")
		}
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		return resp
	}

	// 1. Add and list a saved comic source
	resp := authDo(http.MethodPost, "/api/v1/sources", map[string]any{
		"name": "Local Test JSON", "kind": "json_manifest", "base_url": imgSrv.URL + "/manifest.json",
	})
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("create source status=%d", resp.StatusCode)
	}
	var createdSrc store.ComicSource
	_ = json.NewDecoder(resp.Body).Decode(&createdSrc)
	_ = resp.Body.Close()

	// 2. Probe source
	resp = authDo(http.MethodPost, "/api/v1/sources/probe", map[string]any{
		"kind": "json_manifest", "source_id": createdSrc.ID,
	})
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("probe source status=%d", resp.StatusCode)
	}
	var probed source.BrowseResult
	_ = json.NewDecoder(resp.Body).Decode(&probed)
	_ = resp.Body.Close()
	if probed.Kind != "pages" || len(probed.Pages) != 2 {
		t.Fatalf("probe unexpected: %+v", probed)
	}

	// 3. Start streaming download-while-translating job
	resp = authDo(http.MethodPost, "/api/v1/jobs/source-translate", map[string]any{
		"source_kind": "json_manifest",
		"work_title":  "流式下载测试漫",
		"chapter_title": "第1话",
		"pages":       probed.Pages,
	})
	if resp.StatusCode != http.StatusAccepted {
		t.Fatalf("start source job status=%d", resp.StatusCode)
	}
	var jobWrap struct {
		Job JobSnapshot `json:"job"`
	}
	_ = json.NewDecoder(resp.Body).Decode(&jobWrap)
	_ = resp.Body.Close()
	snap := jobWrap.Job

	// Wait for streaming job completion
	deadline := time.Now().Add(8 * time.Second)
	for time.Now().Before(deadline) {
		r2, err := http.Get(ts.URL + "/api/v1/jobs/" + snap.ID)
		if err != nil {
			t.Fatal(err)
		}
		_ = json.NewDecoder(r2.Body).Decode(&jobWrap)
		_ = r2.Body.Close()
		snap = jobWrap.Job
		if snap.Status == "completed" || snap.Status == "failed" {
			break
		}
		time.Sleep(30 * time.Millisecond)
	}
	if snap.Status != "completed" || snap.WorkID == 0 || len(snap.PageIDs) != 2 || snap.DownloadedUnits != 2 {
		t.Fatalf("streaming job did not complete cleanly: %+v", snap)
	}

	// 4. Verify work cover_page_id & rename & single-page delete & whole-work delete
	w, err := st.GetWork(context.Background(), snap.WorkID)
	if err != nil || w.CoverPageID == 0 || w.PageCount != 2 {
		t.Fatalf("work metadata mismatch: %+v err=%v", w, err)
	}

	resp = authDo(http.MethodPatch, "/api/v1/works/"+strconv.FormatInt(snap.WorkID, 10), map[string]any{"title": "已改名漫"})
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("rename status=%d", resp.StatusCode)
	}
	_ = resp.Body.Close()

	resp = authDo(http.MethodDelete, "/api/v1/pages/"+strconv.FormatInt(snap.PageIDs[0], 10), nil)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("delete page status=%d", resp.StatusCode)
	}
	_ = resp.Body.Close()
	w, _ = st.GetWork(context.Background(), snap.WorkID)
	if w.PageCount != 1 || w.Title != "已改名漫" {
		t.Fatalf("after page delete + rename: %+v", w)
	}

	resp = authDo(http.MethodDelete, "/api/v1/works/"+strconv.FormatInt(snap.WorkID, 10), nil)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("delete work status=%d", resp.StatusCode)
	}
	_ = resp.Body.Close()
	works, _ := st.ListWorks(context.Background())
	if len(works) != 0 {
		t.Fatalf("expected 0 works after delete, got %d", len(works))
	}

	// 5. Delete saved source
	resp = authDo(http.MethodDelete, "/api/v1/sources/"+strconv.FormatInt(createdSrc.ID, 10), nil)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("delete source status=%d", resp.StatusCode)
	}
	_ = resp.Body.Close()
	_ = strings.TrimSpace("")
}
