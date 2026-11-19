package narrative

import (
	"bytes"
	"image"
	"image/color"
	"image/jpeg"
	"testing"
)

func TestVisionContactSheetDynamicPaddingAndExteriorFrame(t *testing.T) {
	src := image.NewRGBA(image.Rect(0, 0, 500, 500))
	// Fill background with cool blue-gray uniform tone; fill small box [100,100,40,30] with pure white.
	for y := 0; y < 500; y++ {
		for x := 0; x < 500; x++ {
			src.SetRGBA(x, y, color.RGBA{R: 30, G: 60, B: 120, A: 255})
		}
	}
	for y := 100; y < 130; y++ {
		for x := 100; x < 140; x++ {
			src.SetRGBA(x, y, color.RGBA{R: 255, G: 255, B: 255, A: 255})
		}
	}

	nodes := []NodeTranslationRequest{
		{ID: "small_box", BBox: []int{100, 100, 40, 30}},
		{ID: "large_box", BBox: []int{220, 220, 180, 140}},
	}
	jpegBytes, tiles, err := buildNodeCropSheetJPEG(src, nodes)
	if err != nil {
		t.Fatalf("buildNodeCropSheetJPEG: %v", err)
	}
	if len(tiles) != 2 {
		t.Fatalf("expected 2 tiles, got %d", len(tiles))
	}

	// Small box: max(40,30)=40 -> pad = max(12, min(40, round(0.6*40))) = 24 -> 40+48=88, 30+48=78
	smallTile := tiles[0]
	if smallTile.SourceRect != [4]int{76, 76, 88, 78} {
		t.Fatalf("expected expanded context source_rect [76 76 88 78], got %v", smallTile.SourceRect)
	}

	// Large box: keeps compact 4px padding -> 180+8=188, 140+8=148
	largeTile := tiles[1]
	if largeTile.SourceRect != [4]int{216, 216, 188, 148} {
		t.Fatalf("expected compact source_rect [216 216 188 148], got %v", largeTile.SourceRect)
	}

	sheetImg, err := jpeg.Decode(bytes.NewReader(jpegBytes))
	if err != nil {
		t.Fatalf("decode contact sheet JPEG: %v", err)
	}

	// Verify 1px #E11D48 exterior border around rawRect inside small_box's padded cell.
	// Note: JPEG 4:2:0 chroma subsampling blends 1px chroma with adjacent pixels, so R is
	// strongly elevated relative to the cool blue-gray background (30,60,120) and G/B channels.
	sr := smallTile.SheetRect
	frameX := sr[0] + 23 // 1px left of rawRect
	frameY := sr[1] + 24 + 15
	r, g, b, _ := sheetImg.At(frameX, frameY).RGBA()
	r8, g8, b8 := r>>8, g>>8, b>>8
	if r8 < 120 || r8 <= g8+50 || r8 <= b8 {
		t.Fatalf("expected reddish #E11D48 exterior frame at (%d,%d) over blue background, got RGB=(%d,%d,%d)", frameX, frameY, r8, g8, b8)
	}

	// Interior of rawRect must remain pure white (not overwritten by frame).
	inX := sr[0] + 24 + 20
	inY := sr[1] + 24 + 15
	ir, ig, ib, _ := sheetImg.At(inX, inY).RGBA()
	if (ir >> 8) < 240 || (ig >> 8) < 240 || (ib >> 8) < 240 {
		t.Fatalf("expected rawRect interior preserved white, got RGB=(%d,%d,%d)", ir>>8, ig>>8, ib>>8)
	}
}
