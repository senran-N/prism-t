package source

import (
	"archive/zip"
	"bytes"
	"context"
	"image"
	"image/color"
	"image/png"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
)

func makeTinyPNG(t *testing.T, r, g, b uint8) []byte {
	t.Helper()
	im := image.NewRGBA(image.Rect(0, 0, 12, 16))
	for y := 0; y < 16; y++ {
		for x := 0; x < 12; x++ {
			im.Set(x, y, color.RGBA{R: r, G: g, B: b, A: 255})
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, im); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func TestMangaDexSearchChaptersAndPages(t *testing.T) {
	const mangaID = "11111111-2222-3333-4444-555555555555"
	const chapID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
	var srv *httptest.Server
	srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch {
		case r.URL.Path == "/manga":
			_, _ = w.Write([]byte(`{"data":[{"id":"` + mangaID + `","attributes":{"title":{"ja":"テスト漫画"},"description":{"ja":"テスト説明"},"originalLanguage":"ja","availableTranslatedLanguages":["ja","en"],"tags":[]},"relationships":[{"type":"cover_art","attributes":{"fileName":"cover.jpg"}}]}]}`))
		case r.URL.Path == "/manga/"+mangaID:
			_, _ = w.Write([]byte(`{"data":{"attributes":{"title":{"ja":"テスト漫画"}}}}`))
		case r.URL.Path == "/manga/"+mangaID+"/feed":
			_, _ = w.Write([]byte(`{"data":[{"id":"` + chapID + `","attributes":{"title":"First","chapter":"1","volume":"1","translatedLanguage":"ja","pages":2,"publishAt":"2026-01-01T00:00:00Z","externalUrl":""}}]}`))
		case r.URL.Path == "/at-home/server/"+chapID:
			_, _ = w.Write([]byte(`{"baseUrl":"` + srv.URL + `","chapter":{"hash":"abc","data":["p1.png","p2.png"],"dataSaver":[]}}`))
		case strings.HasPrefix(r.URL.Path, "/data/abc/"):
			w.Header().Set("Content-Type", "image/png")
			_, _ = w.Write(makeTinyPNG(t, 200, 100, 50))
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	cl := &Client{HTTP: srv.Client(), MangaDexBase: srv.URL}
	ctx := context.Background()

	res, err := cl.SearchMangaDex(ctx, "test", 5)
	if err != nil || res.Kind != "comics" || len(res.Comics) != 1 || res.Comics[0].Title != "テスト漫画" {
		t.Fatalf("SearchMangaDex unexpected: res=%+v err=%v", res, err)
	}

	chRes, err := cl.ListMangaDexChapters(ctx, "https://mangadex.org/title/"+mangaID, "ja")
	if err != nil || chRes.Kind != "chapters" || len(chRes.Chapters) != 1 || chRes.Chapters[0].PageCount != 2 {
		t.Fatalf("ListMangaDexChapters unexpected: res=%+v err=%v", chRes, err)
	}

	pages, err := cl.ResolveMangaDexChapterPages(ctx, "https://mangadex.org/chapter/"+chapID, false)
	if err != nil || len(pages) != 2 {
		t.Fatalf("ResolveMangaDexChapterPages unexpected: pages=%+v err=%v", pages, err)
	}

	var dlCount atomic.Int32
	stream := cl.StreamPages(ctx, pages, nil, 2, func(downloaded, total int, filename string) {
		dlCount.Store(int32(downloaded))
	})
	var got []string
	for item := range stream {
		if item.Err != nil {
			t.Fatalf("stream err: %v", item.Err)
		}
		got = append(got, item.Page.Filename)
	}
	if len(got) != 2 || got[0] != "001.png" || got[1] != "002.png" || dlCount.Load() != 2 {
		t.Fatalf("stream result mismatch: got=%v dlCount=%d", got, dlCount.Load())
	}
}

func TestOPDSAndJSONManifestAndRemoteCBZ(t *testing.T) {
	png1 := makeTinyPNG(t, 10, 20, 30)
	png2 := makeTinyPNG(t, 40, 50, 60)

	var cbzBuf bytes.Buffer
	zw := zip.NewWriter(&cbzBuf)
	f1, _ := zw.Create("02.png")
	_, _ = f1.Write(png2)
	f2, _ := zw.Create("01.png")
	_, _ = f2.Write(png1)
	_ = zw.Close()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/opds":
			w.Header().Set("Content-Type", "application/atom+xml")
			_, _ = w.Write([]byte(`<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>My Komga Shelf</title>
  <entry>
    <id>ch1</id>
    <title>Vol 1 Ch 1</title>
    <link rel="http://opds-spec.org/acquisition" type="application/vnd.comicbook+zip" href="/dl/ch1.cbz"/>
  </entry>
</feed>`))
		case "/manifest.json":
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"title":"JSON Comic","pages":["/img/1.png","/img/2.png"]}`))
		case "/dl/ch1.cbz":
			w.Header().Set("Content-Type", "application/vnd.comicbook+zip")
			_, _ = w.Write(cbzBuf.Bytes())
		case "/img/1.png":
			w.Header().Set("Content-Type", "image/png")
			_, _ = w.Write(png1)
		case "/img/2.png":
			w.Header().Set("Content-Type", "image/png")
			_, _ = w.Write(png2)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	cl := &Client{HTTP: srv.Client()}
	ctx := context.Background()

	opdsRes, err := cl.BrowseOPDSOrJSON(ctx, srv.URL+"/opds", nil)
	if err != nil || opdsRes.Kind != "chapters" || len(opdsRes.Chapters) != 1 || !strings.HasSuffix(opdsRes.Chapters[0].ArchiveURL, "/dl/ch1.cbz") {
		t.Fatalf("OPDS browse failed: res=%+v err=%v", opdsRes, err)
	}

	cbzPages, err := cl.DownloadArchivePages(ctx, opdsRes.Chapters[0].ArchiveURL, nil, 0)
	if err != nil || len(cbzPages) != 2 || cbzPages[0].Filename != "01.png" || cbzPages[1].Filename != "02.png" {
		t.Fatalf("CBZ download failed: pages=%v err=%v", len(cbzPages), err)
	}

	jsonRes, err := cl.ResolveDirectInput(ctx, srv.URL+"/manifest.json", nil)
	if err != nil || jsonRes.Kind != "pages" || len(jsonRes.Pages) != 2 {
		t.Fatalf("JSON manifest failed: res=%+v err=%v", jsonRes, err)
	}
}
