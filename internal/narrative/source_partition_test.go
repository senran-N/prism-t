package narrative

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"path/filepath"
	"reflect"
	"testing"

	"prism-t/internal/config"
	"prism-t/internal/dom"
	"prism-t/internal/store"
)

type partitionBackend struct {
	stubBackend
	parts      []SourceNodeRequest
	failPart   int
	orderCalls int
	failOrder  bool
	badOrder   bool
}

func (b *partitionBackend) TranscribeNodes(_ context.Context, requests []SourceNodeRequest) ([]dom.SourceTranscription, error) {
	if len(requests) != 1 {
		return nil, errors.New("partition must own one node")
	}
	r := requests[0]
	b.parts = append(b.parts, r)
	if len(b.parts) == b.failPart {
		return nil, errors.New("interrupted source part")
	}
	yes := true
	result := dom.SourceTranscription{ID: r.ID, IsText: &yes, LineTexts: make([]string, len(r.SourceLines)), LineOrder: make([]int, len(r.SourceLines))}
	for i, box := range r.SourceLines {
		result.LineTexts[i] = fmt.Sprintf("源%d×！", box[0])
		result.LineOrder[i] = i
	}
	return []dom.SourceTranscription{result}, nil
}

func (b *partitionBackend) OrderSourceLines(_ context.Context, r SourceNodeRequest, lines []string) (SourceLineOrder, error) {
	b.orderCalls++
	if b.failOrder {
		return SourceLineOrder{}, errors.New("ordering interrupted")
	}
	order := make([]int, len(lines))
	for i := range order {
		order[i] = len(order) - 1 - i
	}
	if b.badOrder {
		order[0] = order[1]
	}
	return SourceLineOrder{ID: r.ID, LineOrder: order, Reason: "explicit reversed test paragraph"}, nil
}

func TestPartitionedSourcePersistsEveryLineThenResumesOrderingWithoutReReading(t *testing.T) {
	ctx := context.Background()
	st, err := store.Open(filepath.Join(t.TempDir(), "source.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	work, err := st.EnsureWork(ctx, "Source", "ja", "zh-CN", "manga_rtl")
	if err != nil {
		t.Fatal(err)
	}
	id, err := st.SavePage(ctx, work.ID, "1", 0, "actual.png", "", dom.PageDOM{}, []byte("raw"), []byte("preview"), "analyzed")
	if err != nil {
		t.Fatal(err)
	}
	request := NodeTranslationRequest{ID: "dense", BBox: []int{0, 0, 100, 100}}
	for i := 0; i < 7; i++ {
		request.SourceLines = append(request.SourceLines, []int{i, 0, 1, 10})
		request.SourceQuadrilaterals = append(request.SourceQuadrilaterals, [][]float64{{float64(i), 0}, {float64(i + 1), 0}, {float64(i + 1), 10}, {float64(i), 10}})
		request.SourceReadings = append(request.SourceReadings, dom.SourceReading{LineIndex: i, Recognizer: "explicit-fixture", Text: fmt.Sprint(i)})
	}
	cfg := config.Config{LLMSourceLineBatchSize: 2, LLMModel: "actual-model"}
	read := func(b *partitionBackend) (*dom.PageDOM, error) {
		saved, err := st.GetPage(ctx, id)
		if err != nil {
			t.Fatal(err)
		}
		err = NewCompilerWithBackend(cfg, b).transcribePage(ctx, st, work.ID, "1", 0, &saved.DOM, []byte("raw"), []NodeTranslationRequest{request})
		return &saved.DOM, err
	}
	first := &partitionBackend{failPart: 2}
	if _, err := read(first); err == nil {
		t.Fatal("part interruption hidden")
	}
	saved, _ := st.GetPage(ctx, id)
	if len(saved.DOM.RecognitionProgress.Results) != 0 || len(saved.DOM.RecognitionProgress.LineParts["dense"]) != 1 || saved.Summary.Status != "analyzed" {
		t.Fatal("partial lines lost or promoted to finished source")
	}
	second := &partitionBackend{failOrder: true}
	if _, err := read(second); err == nil {
		t.Fatal("global-order failure hidden")
	}
	if len(second.parts) != 3 || second.parts[0].SourceLines[0][0] != 2 || second.orderCalls != 1 {
		t.Fatal("saved real lines were re-read or remaining lines skipped")
	}
	for _, part := range second.parts {
		if !reflect.DeepEqual(part.BBox, request.BBox) || len(part.SourceQuadrilaterals) != len(part.SourceLines) || len(part.SourceReadings) != len(part.SourceLines) {
			t.Fatal("part lost complete-node context or native geometry")
		}
		for i, reading := range part.SourceReadings {
			if reading.LineIndex != i || reading.Text != fmt.Sprint(part.SourceLines[i][0]) {
				t.Fatal("local source observations were misindexed")
			}
		}
	}
	invalid := &partitionBackend{badOrder: true}
	if _, err := read(invalid); err == nil || len(invalid.parts) != 0 {
		t.Fatal("invalid order accepted or saved lines reread")
	}
	final := &partitionBackend{}
	page, err := read(final)
	if err != nil {
		t.Fatal(err)
	}
	if len(final.parts) != 0 || final.orderCalls != 1 || len(page.RecognitionProgress.Results) != 1 {
		t.Fatal("ordering resume lost independent parts")
	}
	var source dom.SourceTranscription
	if err := json.Unmarshal(page.RecognitionProgress.Results["dense"], &source); err != nil {
		t.Fatal(err)
	}
	for i, text := range source.LineTexts {
		if text != fmt.Sprintf("源%d×！", i) || source.LineOrder[i] != 6-i {
			t.Fatal("ordering rewrote or dropped locked source glyphs")
		}
	}
	completed := &partitionBackend{}
	if _, err := read(completed); err != nil || len(completed.parts) != 0 || completed.orderCalls != 0 {
		t.Fatal("complete source not reused")
	}
	cfg.LLMReasoningEffort = "high"
	changed := &partitionBackend{}
	if _, err := read(changed); err != nil || len(changed.parts) != 4 {
		t.Fatal("changed reasoning setting reused stale source parts")
	}
}
