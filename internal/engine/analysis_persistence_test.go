package engine

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"

	"prism-t/internal/archive"
	"prism-t/internal/config"
	"prism-t/internal/dom"
	"prism-t/internal/narrative"
	"prism-t/internal/store"
)

func TestSuccessfulAnalysisSurvivesSiblingFailure(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/check_seam" {
			json.NewEncoder(w).Encode(map[string]bool{"needs_stitch": false})
			return
		}
		var request AnalyzeRequest
		json.NewDecoder(r.Body).Decode(&request)
		if request.ImagePNGB64 == base64.StdEncoding.EncodeToString([]byte("failed")) {
			http.Error(w, "actual operator failure", http.StatusInternalServerError)
			return
		}
		json.NewEncoder(w).Encode(AnalyzeResponse{DOM: dom.PageDOM{Width: 10, Height: 10}, CleanPreviewPNGB64: base64.StdEncoding.EncodeToString([]byte("preview"))})
	}))
	defer server.Close()
	st, err := store.Open(filepath.Join(t.TempDir(), "analysis.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	cfg := config.Config{MaxConcurrency: 2}
	orch := NewOrchestrator(cfg, st, NewClient(server.URL), narrative.NewCompiler(cfg))
	ctx := context.Background()
	_, _, err = orch.TranslatePages(ctx, []archive.InputPage{
		{Filename: "saved.png", PNGBytes: []byte("success"), Width: 10, Height: 10},
		{Filename: "failed.png", PNGBytes: []byte("failed"), Width: 10, Height: 10},
	}, TranslateBatchOptions{WorkTitle: "persisted", AnalyzeOnly: true})
	if err == nil {
		t.Fatal("operator failure must propagate")
	}
	works, err := st.ListWorks(ctx)
	if err != nil || len(works) != 1 {
		t.Fatalf("works: %+v %v", works, err)
	}
	pages, err := st.ListPages(ctx, works[0].ID, false)
	if err != nil || len(pages) != 1 || pages[0].Filename != "saved.png" || pages[0].Status != "analyzed" {
		t.Fatalf("successful analysis was discarded or failure marked successful: %+v %v", pages, err)
	}
}
