package narrative

import (
	"context"
	"prism-t/internal/dom"
	"prism-t/internal/store"
	"strings"
	"testing"
)

type fidelityBackend struct {
	fixtureSourceOrdering
	calls   [][]NodeTranslationRequest
	results [][]NodeTranslationResult
}

func (*fidelityBackend) TranscribeNodes(_ context.Context, requests []SourceNodeRequest) ([]dom.SourceTranscription, error) {
	return fixtureSources(requests, nil), nil
}

func (b *fidelityBackend) TranslateNodes(_ context.Context, _ string, _ []store.DialogueTurn, _ []store.GlossaryTerm, requests []NodeTranslationRequest) ([]NodeTranslationResult, error) {
	b.calls = append(b.calls, requests)
	return b.results[len(b.calls)-1], nil
}

func TestFidelityRepairsDroppedChapterNumberAndKeepsSibling(t *testing.T) {
	backend := &fidelityBackend{results: [][]NodeTranslationResult{
		{{ID: "caption", TargetText: "我想这就是原因。", TargetSegments: []string{"我想", "这就是", "原因。"}}, {ID: "other", TargetText: "你好！", TargetSegments: []string{"你好！"}}},
		{{ID: "caption", TargetText: "7 — 我想这就是原因。", TargetSegments: []string{"7 — ", "我想", "这就是", "原因。"}}},
	}}
	compiler := &Compiler{backend: backend}
	result, err := compiler.translateComplete(context.Background(), "zh-CN", nil, nil, []NodeTranslationRequest{{ID: "caption", SourceText: "7 - I suppose that explains it."}, {ID: "other", SourceText: "Hello!"}})
	if err != nil {
		t.Fatal(err)
	}
	if len(backend.calls) != 2 || len(backend.calls[1]) != 1 || backend.calls[1][0].Correction == "" || len(result) != 2 || result[0].TargetSegments[0] != "7 — " {
		t.Fatalf("Automatic fidelity repair lost context or a sibling: %+v / %+v", backend.calls, result)
	}
}

func TestFidelityRejectsRepeatedFailedRepair(t *testing.T) {
	bad := NodeTranslationResult{ID: "n", TargetText: "只有2个", TargetSegments: []string{"只有", "2个"}}
	backend := &fidelityBackend{results: [][]NodeTranslationResult{{bad}, {bad}}}
	_, err := (&Compiler{backend: backend}).translateComplete(context.Background(), "zh-CN", nil, nil, []NodeTranslationRequest{{ID: "n", SourceText: "2 and 2"}})
	if err == nil || !strings.Contains(err.Error(), "fidelity repair failed") || len(backend.calls) != 2 {
		t.Fatalf("Unrepaired omission was accepted: %v", err)
	}
}

func TestFidelityChecksExactSegmentationAndNumericRuns(t *testing.T) {
	cases := []struct {
		source, target string
		segments       []string
		valid          bool
	}{
		{"16:45 on 16/4", "16/4的16:45", []string{"16/4的", "16:45"}, true},
		{"12", "112", []string{"112"}, false},
		{"no digits", "新增3", []string{"新增", "3"}, false},
		{"１２", "１２", []string{"１２"}, true},
		{"a b", "甲 乙", []string{"甲", "乙"}, false},
		{"a b", "甲 乙", []string{"甲 ", "乙"}, true},
		{"source", "完整译文", nil, false},
	}
	for _, c := range cases {
		issue := translationFidelityIssue(NodeTranslationRequest{SourceText: c.source}, NodeTranslationResult{TargetText: c.target, TargetSegments: c.segments}, false)
		if (issue == "") != c.valid {
			t.Fatalf("%q => %q issue=%q", c.source, c.target, issue)
		}
	}
}

func TestAnonymousSourceSymbolsSurviveTranslationAndAutomaticRepair(t *testing.T) {
	for _, c := range []struct {
		source, target string
		valid          bool
	}{
		{"×××君、○○さん", "×××同学、○○先生", true},
		{"×××君", "某同学", false},
		{"×××君", "××同学", false},
		{"×××君と×××君", "×××同学", false},
		{"〇〇、●●", "●●和〇〇", true},
		{"○○", "〇〇", false},
		{"plain source", "新增×××", false},
		{"2×3", "2乘3", true},
		{"×××君", "×××君他", true}, // Symbol conservation alone cannot certify referent grammar.
	} {
		issue := translationFidelityIssue(NodeTranslationRequest{SourceText: c.source}, NodeTranslationResult{TargetText: c.target, TargetSegments: []string{c.target}}, false)
		if (issue == "") != c.valid {
			t.Fatalf("%q => %q: %s", c.source, c.target, issue)
		}
	}
	yes := true
	request := NodeTranslationRequest{ID: "anonymous", SourceText: "×××君", SourceTextLocked: true}
	backend := &fidelityBackend{results: [][]NodeTranslationResult{
		{{ID: request.ID, IsText: &yes, SourceText: request.SourceText, TargetText: "那位同学", TargetSegments: []string{"那位同学"}}},
		{{ID: request.ID, IsText: &yes, SourceText: request.SourceText, TargetText: "×××同学", TargetSegments: []string{"×××同学"}}},
	}}
	results, err := (&Compiler{backend: backend}).translateComplete(context.Background(), "zh-CN", nil, nil, []NodeTranslationRequest{request})
	if err != nil || len(backend.calls) != 2 || len(results) != 1 || results[0].TargetText != "×××同学" || !strings.Contains(backend.calls[1][0].Correction, "masking-symbol") {
		t.Fatalf("anonymous omission did not trigger grounded repair: %+v / %v", backend.calls, err)
	}
}

func TestNormalizeChineseEllipsesAndCompilerIntegration(t *testing.T) {
	cases := []struct {
		text     string
		segs     []string
		wantText string
		wantSegs []string
	}{
		{"等…", []string{"等…"}, "等……", []string{"等……"}},
		{"………真的", []string{"………", "真的"}, "……真的", []string{"……", "真的"}},
		{"……好", []string{"……好"}, "……好", []string{"……好"}},
		{"等……什么", []string{"等…", "…", "什么"}, "等……什么", []string{"等……", "什么"}},
		{"等……啊", []string{"等…", "…啊"}, "等……啊", []string{"等……", "啊"}},
		{"123…×××君…○○", []string{"123…", "×××君", "…○○"}, "123……×××君……○○", []string{"123……", "×××君", "……○○"}},
	}
	for _, tc := range cases {
		gotText, gotSegs := normalizeChineseEllipses(tc.text, tc.segs)
		if gotText != tc.wantText {
			t.Fatalf("normalizeChineseEllipses(%q) text=%q want %q", tc.text, gotText, tc.wantText)
		}
		if len(gotSegs) != len(tc.wantSegs) {
			t.Fatalf("normalizeChineseEllipses(%v) segs=%v want %v", tc.segs, gotSegs, tc.wantSegs)
		}
		for i := range gotSegs {
			if gotSegs[i] != tc.wantSegs[i] || gotSegs[i] == "" {
				t.Fatalf("segment %d=%q want %q", i, gotSegs[i], tc.wantSegs[i])
			}
		}
		if strings.Join(gotSegs, "") != gotText {
			t.Fatalf("joined segments mismatch: %v vs %q", gotSegs, gotText)
		}
		issue := translationFidelityIssue(
			NodeTranslationRequest{SourceText: "123 ×××君 ○○"},
			NodeTranslationResult{TargetText: gotText, TargetSegments: gotSegs},
			false,
		)
		if strings.Contains(tc.text, "123") && issue != "" {
			t.Fatalf("unexpected fidelity issue after normalization: %s", issue)
		}
	}

	req := NodeTranslationRequest{ID: "ell-1", SourceText: "…123 ×××君"}
	yes := true
	zhBackend := &fidelityBackend{results: [][]NodeTranslationResult{{
		{ID: req.ID, IsText: &yes, SourceText: req.SourceText, TargetText: "…123 ×××君", TargetSegments: []string{"…", "123 ", "×××君"}},
	}}}
	zhOut, err := (&Compiler{backend: zhBackend}).translateComplete(context.Background(), "zh-CN", nil, nil, []NodeTranslationRequest{req})
	if err != nil || len(zhOut) != 1 || zhOut[0].TargetText != "……123 ×××君" {
		t.Fatalf("zh-CN translateComplete did not normalize ellipsis: %+v / %v", zhOut, err)
	}

	enBackend := &fidelityBackend{results: [][]NodeTranslationResult{{
		{ID: req.ID, IsText: &yes, SourceText: req.SourceText, TargetText: "…123 ×××君", TargetSegments: []string{"…", "123 ", "×××君"}},
	}}}
	enOut, err := (&Compiler{backend: enBackend}).translateComplete(context.Background(), "en", nil, nil, []NodeTranslationRequest{req})
	if err != nil || len(enOut) != 1 || enOut[0].TargetText != "…123 ×××君" {
		t.Fatalf("en translateComplete altered non-Chinese ellipsis: %+v / %v", enOut, err)
	}
}

func TestAnnotateDiscourseTopologyHints(t *testing.T) {
	reqs := []NodeTranslationRequest{
		{ID: "close", ReadingOrder: 2, SourceText: "犯人だったのか」"},
		{ID: "mixed", ReadingOrder: 0, SourceText: "だから言ったのに（…まさか本当だとは）"},
		{ID: "open", ReadingOrder: 1, SourceText: "「君が……"},
	}
	annotateDiscourseTopology(reqs)
	if reqs[1].DiscourseHint != "mixed_outer_and_parenthetical_monologue" {
		t.Fatalf("mixed node DiscourseHint=%q", reqs[1].DiscourseHint)
	}
	if reqs[2].DiscourseHint != "unclosed_quote_or_paren_continues_to_next" {
		t.Fatalf("open node DiscourseHint=%q", reqs[2].DiscourseHint)
	}
	if reqs[0].DiscourseHint != "closes_quote_or_paren_from_prev" {
		t.Fatalf("close node DiscourseHint=%q", reqs[0].DiscourseHint)
	}

	split := []NodeTranslationRequest{
		{ID: "n1", ReadingOrder: 0, Track: "primary", SourceText: "範囲攻撃の余波で、"},
		{ID: "n2", ReadingOrder: 1, Track: "primary", SourceText: "死んでしまったか"},
	}
	annotateDiscourseTopology(split)
	if !strings.Contains(split[0].DiscourseHint, "clause_continues_to_next") {
		t.Fatalf("expected clause_continues_to_next on n1, got %q", split[0].DiscourseHint)
	}
	if !strings.Contains(split[1].DiscourseHint, "continues_clause_from_prev") {
		t.Fatalf("expected continues_clause_from_prev on n2, got %q", split[1].DiscourseHint)
	}
}
