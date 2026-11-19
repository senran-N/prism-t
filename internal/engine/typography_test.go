package engine

import (
	"testing"

	"prism-t/internal/dom"
)

func TestTargetTypographyKeepsObservedDirections(t *testing.T) {
	for _, profile := range []string{"manga_rtl", "comic_ltr", "webtoon_vertical"} {
		page := dom.PageDOM{FormatProfile: profile, Nodes: []dom.TextNode{
			{BBox: []int{0, 0, 700, 330}, Style: dom.StyleSpec{Orientation: "vertical"}},
			{BBox: []int{0, 0, 30, 100}, Style: dom.StyleSpec{Orientation: "horizontal"}},
		}}
		adaptTargetTypography(&page, "zh-CN")
		if page.Nodes[0].Style.Orientation != "vertical" || page.Nodes[1].Style.Orientation != "horizontal" {
			t.Fatalf("%s replaced measured writing directions: %+v", profile, page.Nodes)
		}
	}
}

func TestLatinTargetUsesHorizontalFont(t *testing.T) {
	page := dom.PageDOM{Nodes: []dom.TextNode{{Style: dom.StyleSpec{Orientation: "vertical", FontCategory: "gothic"}}}}
	adaptTargetTypography(&page, "en-US")
	if page.Nodes[0].Style.Orientation != "horizontal" || page.Nodes[0].Style.FontCategory != "comic_sans" {
		t.Fatal("Latin target must use horizontal typography")
	}
}
