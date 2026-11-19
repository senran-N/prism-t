package main

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
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"prism-t/internal/dom"
	"prism-t/internal/engine"
	"prism-t/internal/store"
)

func TestRetryWorkKeepsNarrativeOrderAndEveryExport(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "work.db")
	st, err := store.Open(dbPath)
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	work, err := st.EnsureWork(ctx, "order", "en", "zh-CN", "comic_ltr")
	if err != nil {
		t.Fatal(err)
	}
	// Creation/completion order differs from chapter and page narrative order.
	for _, p := range []struct {
		chapter, name string
		index, value  int
	}{
		{"Chapter 2", "001.png", 0, 3},
		{"Chapter 1", "002.png", 1, 2},
		{"Chapter 1", "001.png", 0, 1},
	} {
		im := image.NewNRGBA(image.Rect(0, 0, 1, 1))
		im.SetNRGBA(0, 0, color.NRGBA{R: uint8(p.value), A: 255})
		var b bytes.Buffer
		png.Encode(&b, im)
		_, err = st.SavePage(ctx, work.ID, p.chapter, p.index, p.name, p.name, dom.PageDOM{Width: 1, Height: 1}, b.Bytes(), b.Bytes(), "translated")
		if err != nil {
			t.Fatal(err)
		}
	}
	st.Close()
	var order []uint8
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
			return
		}
		if r.URL.Path != "/v1/render" {
			t.Error("completed translations must not contact the model or repeat analysis")
			http.Error(w, "unexpected request", 500)
			return
		}
		var request engine.RenderRequest
		json.NewDecoder(r.Body).Decode(&request)
		raw, _ := base64.StdEncoding.DecodeString(request.RawImagePNGB64)
		im, err := png.Decode(bytes.NewReader(raw))
		if err != nil {
			t.Error(err)
			return
		}
		red, _, _, _ := im.At(0, 0).RGBA()
		order = append(order, uint8(red/257))
		json.NewEncoder(w).Encode(engine.RenderResponse{DOM: request.DOM, CompositePNGB64: request.RawImagePNGB64})
	}))
	defer server.Close()
	t.Setenv("PRISM_T_ENGINE_URL", server.URL)
	t.Setenv("PRISM_T_LLM_BASE_URL", server.URL)
	t.Setenv("PRISM_T_LLM_API_KEY", "test")
	out := t.TempDir()
	if err := runRetry([]string{"--work-id", "1", "--db", dbPath, "--out", out, "--psd"}); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(order, []uint8{1, 2, 3}) {
		t.Fatalf("wrong narrative order: %v", order)
	}
	for _, name := range []string{"000001_001.png", "000001_001.psd", "000002_002.png", "000002_002.psd", "000003_001.png", "000003_001.psd"} {
		if _, err := os.Stat(filepath.Join(out, name)); err != nil {
			t.Fatalf("missing export %s: %v", name, err)
		}
	}
}

func TestRetryRequiresExactlyOneStoredScope(t *testing.T) {
	for _, args := range [][]string{{"--out", "unused"}, {"--page-id", "1", "--work-id", "1", "--out", "unused"}} {
		if err := runRetry(args); err == nil {
			t.Fatal("ambiguous or absent retry scope was accepted")
		}
	}
}
