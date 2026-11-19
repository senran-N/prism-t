package narrative

import (
	"context"
	"encoding/base64"
	"encoding/binary"
	"errors"
	"math"
	"path/filepath"
	"testing"

	"prism-t/internal/config"
	"prism-t/internal/dom"
	"prism-t/internal/store"
)

type stubBackend struct {
	fixtureSourceOrdering
	calls int
	last  []NodeTranslationRequest
}

// Explicit source fixtures belong only to mocked protocol tests.
func fixtureSources(requests []SourceNodeRequest, texts map[string]string) []dom.SourceTranscription {
	results := make([]dom.SourceTranscription, len(requests))
	for i, request := range requests {
		text := texts[request.ID]
		isText := text != ""
		lines := make([]string, max(1, len(request.SourceLines)))
		lines[0] = text
		order := make([]int, len(lines))
		for j := range order {
			order[j] = j
		}
		results[i] = dom.SourceTranscription{ID: request.ID, IsText: &isText, LineTexts: lines, LineOrder: order, Reason: "explicit test source fixture"}
	}
	return results
}

type fixtureSourceOrdering struct{}

func (fixtureSourceOrdering) OrderSourceLines(_ context.Context, request SourceNodeRequest, lines []string) (SourceLineOrder, error) {
	order := make([]int, len(lines))
	for i := range order {
		order[i] = i
	}
	return SourceLineOrder{ID: request.ID, LineOrder: order, Reason: "explicit test order"}, nil
}

func (*stubBackend) TranscribeNodes(_ context.Context, requests []SourceNodeRequest) ([]dom.SourceTranscription, error) {
	return fixtureSources(requests, nil), nil
}

func (sb *stubBackend) TranslateNodes(
	_ context.Context,
	_ string,
	_ []store.DialogueTurn,
	_ []store.GlossaryTerm,
	nodes []NodeTranslationRequest,
) ([]NodeTranslationResult, error) {
	sb.calls++
	sb.last = nodes
	out := make([]NodeTranslationResult, len(nodes))
	for i, n := range nodes {
		if n.ID == "node_1" {
			out[i] = NodeTranslationResult{
				ID:         n.ID,
				TargetText: "前辈小心！",
			}
		} else {
			out[i] = NodeTranslationResult{
				ID:         n.ID,
				TargetText: "笨蛋宿敌！",
				Rubies:     []RubyTranslation{{TargetBase: "宿敌", TargetRuby: "挚友"}},
			}
		}
		out[i].TargetSegments = []string{out[i].TargetText}
	}
	return out, nil
}

func makeUnitVecB64(axis int) string {
	buf := make([]byte, 16*4)
	for i := 0; i < 16; i++ {
		val := float32(0.0)
		if i == axis {
			val = 1.0
		}
		binary.LittleEndian.PutUint32(buf[i*4:(i+1)*4], math.Float32bits(val))
	}
	return base64.StdEncoding.EncodeToString(buf)
}

func TestPhonologicalMorphologyAndBudgetedPersonaCompilation(t *testing.T) {
	ph1 := ExtractPhonologicalFeatures("ドォォォンっ！")
	if ph1.ElongationCount < 2 || !ph1.GlottalStop {
		t.Fatalf("expected elongation and glottal stop, got %+v", ph1)
	}
	ph2 := ExtractPhonologicalFeatures("ドキドキ")
	if !ph2.Reduplicated {
		t.Fatalf("expected ABAB reduplication on ドキドキ")
	}

	dbPath := filepath.Join(t.TempDir(), "test_prism_t.db")
	st, err := store.Open(dbPath)
	if err != nil {
		t.Fatalf("open store: %v", err)
	}
	defer st.Close()
	ctx := context.Background()

	work, err := st.EnsureWork(ctx, "Test Manga", "ja", "zh-CN", "manga_rtl")
	if err != nil {
		t.Fatalf("ensure work: %v", err)
	}

	// Verify fail-honest behavior when no translation backend is configured!
	emptyComp := NewCompiler(config.Config{TargetLang: "zh-CN"})
	dummyDOM := dom.PageDOM{Nodes: []dom.TextNode{{ID: "n1", SourceText: "テスト"}}}
	if err := emptyComp.CompilePageTranslations(ctx, st, work.ID, "Ch1", 0, "zh-CN", &dummyDOM, nil); !errors.Is(err, ErrNoTranslationBackend) {
		t.Fatalf("expected ErrNoTranslationBackend, got %v", err)
	}

	// Pre-register two speakers via appearance Re-ID and set directed relation char_1 -> char_2 ("笨蛋前辈")
	c1, err := st.MatchOrRegisterSpeaker(ctx, work.ID, makeUnitVecB64(0))
	if err != nil {
		t.Fatal(err)
	}
	c2, err := st.MatchOrRegisterSpeaker(ctx, work.ID, makeUnitVecB64(1))
	if err != nil {
		t.Fatal(err)
	}
	c1.NameTarget = "学妹"
	c1.FirstPerson = "人家"
	_, _ = st.UpsertCharacter(ctx, c1)
	c2.NameTarget = "男主"
	_, _ = st.UpsertCharacter(ctx, c2)
	_ = st.UpsertRelation(ctx, store.CharacterRelation{
		WorkID:          work.ID,
		FromCharID:      c1.ID,
		ToCharID:        c2.ID,
		HonorificTarget: "笨蛋前辈",
	})
	_ = st.UpsertGlossaryTerm(ctx, store.GlossaryTerm{
		WorkID:       work.ID,
		SourceTerm:   "宿敵",
		TargetTerm:   "宿敌",
		RubyOverride: "挚友",
	})

	sb := &stubBackend{}
	comp := NewCompilerWithBackend(config.Config{TargetLang: "zh-CN"}, sb)

	page := dom.PageDOM{
		Nodes: []dom.TextNode{
			{
				ID:                   "node_1",
				ReadingOrder:         0,
				Track:                "primary",
				BubbleKind:           "shout_spiky",
				SpeakerClusterID:     "spk_a",
				SpeakerAppearanceB64: makeUnitVecB64(0),
				SourceText:           "危ない！",
				CharBudgetMin:        2,
				CharBudgetMax:        6,
			},
			{
				ID:                   "node_2",
				ReadingOrder:         1,
				Track:                "primary",
				BubbleKind:           "speech_oval",
				SpeakerClusterID:     "spk_b",
				SpeakerAppearanceB64: makeUnitVecB64(1),
				SourceText:           "バカ宿敵！",
				RubyPairs:            []dom.RubyPair{{Base: "宿敵", Ruby: "とも", CharStart: 2, CharEnd: 4}},
				CharBudgetMin:        2,
				CharBudgetMax:        10,
			},
		},
	}

	if err := comp.CompilePageTranslations(ctx, st, work.ID, "Ch1", 0, "zh-CN", &page, nil); err != nil {
		t.Fatalf("CompilePageTranslations: %v", err)
	}
	// Single-pass holistic compilation receives budgets upfront and avoids redundant pass-2 re-compression.
	if sb.calls != 1 {
		t.Fatalf("expected 1 single-pass backend call, got %d", sb.calls)
	}
	if page.Nodes[0].TargetText != "前辈小心！" {
		t.Fatalf("expected concise budgeted text on node_1, got %q", page.Nodes[0].TargetText)
	}
	if len(page.Nodes[1].TargetRubyPairs) != 1 || page.Nodes[1].TargetRubyPairs[0].Ruby != "挚友" {
		t.Fatalf("expected aligned Gikun Ruby on node_2, got %+v", page.Nodes[1].TargetRubyPairs)
	}
}

type alwaysOverflowBackend struct{ fixtureSourceOrdering }

func (*alwaysOverflowBackend) TranscribeNodes(_ context.Context, requests []SourceNodeRequest) ([]dom.SourceTranscription, error) {
	return fixtureSources(requests, nil), nil
}

func (b *alwaysOverflowBackend) TranslateNodes(
	ctx context.Context,
	targetLang string,
	priorTurns []store.DialogueTurn,
	glossary []store.GlossaryTerm,
	nodes []NodeTranslationRequest,
) ([]NodeTranslationResult, error) {
	out := make([]NodeTranslationResult, len(nodes))
	for i, n := range nodes {
		out[i] = NodeTranslationResult{
			ID:         n.ID,
			TargetText: "宿敌还是宿敌，太长无法压缩！",
			Rubies: []RubyTranslation{
				{TargetBase: "宿敌", TargetRuby: "对手A"},
				{TargetBase: "宿敌", TargetRuby: "挚友B"},
			},
		}
		out[i].TargetSegments = []string{out[i].TargetText}
	}
	return out, nil
}

func TestStrictBudgetOverflowMarksNeedsReviewAndUniqueRubies(t *testing.T) {
	extracted := extractJSONObject("Sure! Here is the JSON:\n```json\n{\"translations\":[{\"id\":\"n1\",\"target_text\":\"ok\"}]}\n```\nHope this helps.")
	if extracted != `{"translations":[{"id":"n1","target_text":"ok"}]}` {
		t.Fatalf("unexpected extracted JSON: %q", extracted)
	}

	dbPath := filepath.Join(t.TempDir(), "overflow.db")
	st, err := store.Open(dbPath)
	if err != nil {
		t.Fatalf("open store: %v", err)
	}
	defer st.Close()
	ctx := context.Background()
	work, err := st.EnsureWork(ctx, "Overflow Manga", "ja", "zh-CN", "manga_rtl")
	if err != nil {
		t.Fatal(err)
	}

	comp := NewCompilerWithBackend(config.Config{TargetLang: "zh-CN"}, &alwaysOverflowBackend{})
	page := dom.PageDOM{
		Nodes: []dom.TextNode{
			{
				ID:            "n_over",
				Track:         "primary",
				SourceText:    "短い",
				CharBudgetMax: 6,
			},
		},
	}
	if err := comp.CompilePageTranslations(ctx, st, work.ID, "Ch1", 0, "zh-CN", &page, nil); err != nil {
		t.Fatalf("CompilePageTranslations: %v", err)
	}
	nd := page.Nodes[0]
	if !nd.Uncertainty.NeedsReview {
		t.Fatalf("expected NeedsReview=true when translated text remains over CharBudgetMax")
	}
	if !containsString(nd.Uncertainty.Reasons, "budget_overflow") {
		t.Fatalf("expected budget_overflow reason, got %+v", nd.Uncertainty.Reasons)
	}
	if page.Metrics.NeedsReviewNodes != 1 {
		t.Fatalf("expected NeedsReviewNodes=1, got %d", page.Metrics.NeedsReviewNodes)
	}
	if len(nd.TargetRubyPairs) != 2 || nd.TargetRubyPairs[0].CharStart == nd.TargetRubyPairs[1].CharStart {
		t.Fatalf("expected non-overlapping ruby alignments for repeated base, got %+v", nd.TargetRubyPairs)
	}
}
