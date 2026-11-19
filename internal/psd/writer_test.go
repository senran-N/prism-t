package psd

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"image"
	"image/color"
	"image/png"
	"testing"

	"prism-t/internal/dom"
)

func encodeTestPNG(w, h int, c color.NRGBA) []byte {
	im := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			im.SetNRGBA(x, y, c)
		}
	}
	var buf bytes.Buffer
	_ = png.Encode(&buf, im)
	return buf.Bytes()
}

func TestPackBitsRoundtrip(t *testing.T) {
	cases := [][]byte{
		[]byte("A"),
		bytes.Repeat([]byte{0xFF}, 200),
		[]byte("ABCDEFGHHHHHHHHIJKLMNOOOOOOOOP"),
		make([]byte, 333),
	}
	for i := range cases[3] {
		cases[3][i] = byte((i * 37) & 0xFF)
	}
	for idx, src := range cases {
		enc := PackBitsEncode(src)
		dec, err := PackBitsDecode(enc, len(src))
		if err != nil {
			t.Fatalf("case %d decode error: %v", idx, err)
		}
		if !bytes.Equal(dec, src) {
			t.Fatalf("case %d mismatch", idx)
		}
	}
}

func TestBuildAndEncodeLayeredPSD(t *testing.T) {
	rawPNG := encodeTestPNG(64, 48, color.NRGBA{R: 245, G: 245, B: 245, A: 255})
	compPNG := encodeTestPNG(64, 48, color.NRGBA{R: 240, G: 240, B: 240, A: 255})
	patchClean := base64.StdEncoding.EncodeToString(encodeTestPNG(20, 16, color.NRGBA{R: 255, G: 255, B: 255, A: 255}))
	patchAlpha := base64.StdEncoding.EncodeToString(encodeTestPNG(20, 16, color.NRGBA{R: 255, G: 255, B: 255, A: 255}))
	patchZ1 := base64.StdEncoding.EncodeToString(encodeTestPNG(20, 16, color.NRGBA{R: 15, G: 15, B: 15, A: 220}))
	patchZ2 := base64.StdEncoding.EncodeToString(encodeTestPNG(20, 16, color.NRGBA{R: 255, G: 255, B: 255, A: 255}))

	pdom := dom.PageDOM{
		SchemaVersion: 1,
		Width:         64,
		Height:        48,
		Nodes: []dom.TextNode{
			{
				ID:         "node_001",
				TargetText: "宿敌对决",
				Style:      dom.StyleSpec{BlendMode: "multiply"},
				Patches: dom.PatchLayers{
					ROI:               []int{10, 12, 20, 16},
					CleanZ0PNGB64:     patchClean,
					StrokeAlphaPNGB64: patchAlpha,
					RenderedZ1PNGB64:  patchZ1,
					Z2OccluderMaskB64: patchZ2,
				},
			},
		},
	}

	doc, err := BuildFromComicDOM(rawPNG, compPNG, pdom)
	if err != nil {
		t.Fatalf("BuildFromComicDOM: %v", err)
	}
	// Expect 4 layers: Raw, Z0 Clean, Z1 Node, Z2 Occluders
	if len(doc.Layers) != 4 {
		t.Fatalf("expected 4 PSD layers, got %d", len(doc.Layers))
	}
	if doc.Layers[2].BlendMode != "mul " {
		t.Fatalf("expected multiply blend key 'mul ', got %q", doc.Layers[2].BlendMode)
	}

	psdBytes, err := Encode(doc)
	if err != nil {
		t.Fatalf("Encode PSD: %v", err)
	}
	if string(psdBytes[:4]) != "8BPS" {
		t.Fatalf("invalid PSD signature %q", string(psdBytes[:4]))
	}
	h := binary.BigEndian.Uint32(psdBytes[14:18])
	w := binary.BigEndian.Uint32(psdBytes[18:22])
	if w != 64 || h != 48 {
		t.Fatalf("unexpected PSD header dims %dx%d", w, h)
	}
	if !bytes.Contains(psdBytes, []byte("8BIMluni")) {
		t.Fatalf("expected Unicode luni layer name block in PSD")
	}

	// Verify ShadingMapPNGB64 modulation on diegetic_planar Z1 layer & explicit zero OpacitySet encoding
	shadePatch := base64.StdEncoding.EncodeToString(encodeTestPNG(20, 16, color.NRGBA{R: 128, G: 128, B: 128, A: 255}))
	z1Bright := base64.StdEncoding.EncodeToString(encodeTestPNG(20, 16, color.NRGBA{R: 200, G: 100, B: 50, A: 255}))
	pdomShaded := dom.PageDOM{
		SchemaVersion: 1,
		Width:         64,
		Height:        48,
		Nodes: []dom.TextNode{
			{
				ID:         "node_sign",
				FusionMode: "diegetic_planar",
				TargetText: "看板",
				Patches: dom.PatchLayers{
					ROI:              []int{4, 4, 20, 16},
					RenderedZ1PNGB64: z1Bright,
					ShadingMapPNGB64: shadePatch,
				},
			},
		},
	}
	docShaded, err := BuildFromComicDOM(rawPNG, compPNG, pdomShaded)
	if err != nil {
		t.Fatalf("BuildFromComicDOM shaded: %v", err)
	}
	z1Lyr := docShaded.Layers[2]
	// 200 * 128 / 255 ~= 100
	if z1Lyr.RGBA[0] < 98 || z1Lyr.RGBA[0] > 102 {
		t.Fatalf("expected shaded R around 100, got %d", z1Lyr.RGBA[0])
	}
	docZero := &Document{
		Width:     8,
		Height:    8,
		Composite: make([]byte, 8*8*3),
		Layers: []Layer{
			{
				Name:       "zero_op",
				Left:       0,
				Top:        0,
				Width:      8,
				Height:     8,
				RGBA:       make([]byte, 8*8*4),
				Opacity:    0,
				OpacitySet: true,
				BlendMode:  "norm",
			},
		},
	}
	if _, eErr := Encode(docZero); eErr != nil {
		t.Fatalf("Encode explicit zero opacity: %v", eErr)
	}
}
