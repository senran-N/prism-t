package store

import (
	"context"
	"encoding/base64"
	"encoding/binary"
	"fmt"
	"math"
	"path/filepath"
	"testing"

	"prism-t/internal/dom"
)

func makeTestVecB64(axis int) string {
	buf := make([]byte, 16*4)
	for i := 0; i < 16; i++ {
		val := float32(0)
		if i == axis {
			val = 1.0
		}
		binary.LittleEndian.PutUint32(buf[i*4:(i+1)*4], math.Float32bits(val))
	}
	return base64.StdEncoding.EncodeToString(buf)
}

func TestChaptersStayTogetherInNaturalReadingOrder(t *testing.T) {
	st, err := Open(filepath.Join(t.TempDir(), "chapters.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	ctx := context.Background()
	work, err := st.EnsureWork(ctx, "Chapter order", "ja", "zh-CN", "manga_rtl")
	if err != nil {
		t.Fatal(err)
	}
	for _, chapter := range []string{"Chapter 10", "Chapter 2", "Chapter 1"} {
		for _, index := range []int{1, 0} {
			_, err = st.SavePage(ctx, work.ID, chapter, index, fmt.Sprintf("%d.png", index), "hash", dom.PageDOM{Width: 1, Height: 1}, []byte("raw"), []byte("composite"), "translated")
			if err != nil {
				t.Fatal(err)
			}
		}
	}
	pages, err := st.ListPages(ctx, work.ID, false)
	if err != nil {
		t.Fatal(err)
	}
	for i, page := range pages {
		expectedChapter := []string{"Chapter 1", "Chapter 2", "Chapter 10"}[i/2]
		if page.ChapterTitle != expectedChapter || page.PageIndex != i%2 {
			t.Fatalf("position %d: %+v", i, page)
		}
	}
}

func TestStoreClusterKeyMonotonicSlimDialogueAndStats(t *testing.T) {
	st, err := Open(filepath.Join(t.TempDir(), "store_test.db"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer st.Close()
	ctx := context.Background()

	work, err := st.EnsureWork(ctx, "Store Test Work", "ja", "zh-CN", "manga_rtl")
	if err != nil {
		t.Fatalf("EnsureWork: %v", err)
	}

	c1, err := st.MatchOrRegisterSpeaker(ctx, work.ID, makeTestVecB64(0))
	if err != nil {
		t.Fatal(err)
	}
	c2, err := st.MatchOrRegisterSpeaker(ctx, work.ID, makeTestVecB64(1))
	if err != nil {
		t.Fatal(err)
	}
	// Delete first character; registering a 3rd distinct speaker must NOT collide with c2's cluster_key!
	if _, err := st.db.ExecContext(ctx, `DELETE FROM characters WHERE id = ?`, c1.ID); err != nil {
		t.Fatal(err)
	}
	c3, err := st.MatchOrRegisterSpeaker(ctx, work.ID, makeTestVecB64(2))
	if err != nil {
		t.Fatalf("MatchOrRegisterSpeaker after deletion collided: %v", err)
	}
	if c3.ClusterKey == c2.ClusterKey {
		t.Fatalf("cluster_key collision: c2=%s c3=%s", c2.ClusterKey, c3.ClusterKey)
	}

	pdom := dom.PageDOM{
		SchemaVersion: 1,
		Width:         100,
		Height:        120,
		FormatProfile: "manga_rtl",
		Nodes: []dom.TextNode{
			{
				ID:               "n1",
				Track:            "primary",
				SpeakerClusterID: c2.ClusterKey,
				SourceText:       "行くぞ！",
				TargetText:       "上吧！",
			},
		},
	}
	if _, err := st.SavePage(ctx, work.ID, "Ch1", 0, "001.png", "hash1", pdom, []byte("raw"), []byte("comp"), "translated"); err != nil {
		t.Fatalf("SavePage: %v", err)
	}
	turns, err := st.GetRecentChapterDialogue(ctx, work.ID, "Ch1", 1, 8)
	if err != nil {
		t.Fatalf("GetRecentChapterDialogue: %v", err)
	}
	if len(turns) != 1 || turns[0].TargetText != "上吧！" || turns[0].Speaker != c2.ClusterKey {
		t.Fatalf("unexpected dialogue turns: %+v", turns)
	}

	stats, err := st.GetStats(ctx)
	if err != nil {
		t.Fatalf("GetStats: %v", err)
	}
	if stats.Works != 1 || stats.Pages != 1 || stats.Characters != 2 {
		t.Fatalf("unexpected stats: %+v", stats)
	}
}

func TestUpdatePageDOMAndStatusPreservesRawPNG(t *testing.T) {
	st, err := Open(filepath.Join(t.TempDir(), "update_dom.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	ctx := context.Background()
	work, err := st.EnsureWork(ctx, "Update Work", "ja", "zh-CN", "manga_rtl")
	if err != nil {
		t.Fatal(err)
	}
	rawBytes := []byte("original-raw-bytes")
	previewBytes := []byte("preview-bytes")
	oldRawBytes := []byte("older-row-raw-bytes")
	oldPreviewBytes := []byte("older-row-preview")
	initDOM := dom.PageDOM{
		SchemaVersion: 1,
		Width:         100,
		Height:        120,
		FormatProfile: "manga_rtl",
		Nodes:         []dom.TextNode{{ID: "n1", SourceText: "テスト"}},
	}
	// Insert two distinct filenames at the same (work_id, chapter_title, page_index=0).
	oldID, err := st.SavePage(ctx, work.ID, "Ch1", 0, "old.png", "h0", initDOM, oldRawBytes, oldPreviewBytes, "analyzed")
	if err != nil {
		t.Fatal(err)
	}
	pageID, err := st.SavePage(ctx, work.ID, "Ch1", 0, "001.png", "h1", initDOM, rawBytes, previewBytes, "layout_detected")
	if err != nil {
		t.Fatal(err)
	}

	transDOM := initDOM
	transDOM.Nodes[0].TargetText = "测试"
	transDOM.Nodes[0].Uncertainty.NeedsReview = true
	if err := st.UpdatePageDOMAndStatus(ctx, pageID, transDOM, nil, "translated"); err != nil {
		t.Fatalf("UpdatePageDOMAndStatus(nil comp): err=%v", err)
	}
	oldRec, err := st.GetPage(ctx, oldID)
	if err != nil {
		t.Fatal(err)
	}
	if oldRec.Summary.Status != "analyzed" || string(oldRec.RawPNG) != string(oldRawBytes) || string(oldRec.CompositePNG) != string(oldPreviewBytes) || oldRec.DOM.Nodes[0].TargetText != "" {
		t.Fatalf("UpdatePageDOMAndStatus leaked into sibling row at same page_index: %+v", oldRec.Summary)
	}
	rec, err := st.GetPage(ctx, pageID)
	if err != nil {
		t.Fatal(err)
	}
	if string(rec.RawPNG) != string(rawBytes) || string(rec.CompositePNG) != string(previewBytes) {
		t.Fatalf("expected raw and preview bytes preserved, got raw=%q comp=%q", rec.RawPNG, rec.CompositePNG)
	}
	if rec.Summary.Status != "translated" || !rec.Summary.NeedsReview || rec.DOM.Nodes[0].TargetText != "测试" {
		t.Fatalf("unexpected updated summary/dom: %+v / %+v", rec.Summary, rec.DOM)
	}

	finalComp := []byte("final-composite")
	transDOM.Nodes[0].Uncertainty.NeedsReview = false
	if err := st.UpdatePageDOMAndStatus(ctx, pageID, transDOM, finalComp, "reviewed"); err != nil {
		t.Fatal(err)
	}
	rec2, err := st.GetPage(ctx, pageID)
	if err != nil {
		t.Fatal(err)
	}
	if string(rec2.RawPNG) != string(rawBytes) || string(rec2.CompositePNG) != string(finalComp) {
		t.Fatalf("expected raw preserved and composite updated, got raw=%q comp=%q", rec2.RawPNG, rec2.CompositePNG)
	}
	if rec2.Summary.Status != "reviewed" || rec2.Summary.NeedsReview {
		t.Fatalf("expected reviewed status with needs_review=false, got %+v", rec2.Summary)
	}
}
