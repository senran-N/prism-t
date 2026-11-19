package archive

import (
	"bytes"
	"image"
	"image/color"
	"image/png"
	"os"
	"path/filepath"
	"testing"

	"github.com/gen2brain/avif"
)

func makeStripPNG(w, h int, fill color.NRGBA) InputPage {
	im := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			im.SetNRGBA(x, y, fill)
		}
	}
	var buf bytes.Buffer
	_ = png.Encode(&buf, im)
	pBytes, hash, pw, ph, _ := NormalizeToPNG(buf.Bytes())
	return InputPage{
		PNGBytes:   pBytes,
		Blake3Hash: hash,
		Width:      pw,
		Height:     ph,
	}
}

func TestNaturalSortAndVirtualCanvasStitching(t *testing.T) {
	if !NaturalLess("page_2.png", "page_10.png") {
		t.Fatalf("expected page_2.png < page_10.png")
	}
	if NaturalLess("page_12.png", "page_3.png") {
		t.Fatalf("expected page_12.png > page_3.png")
	}

	p1 := makeStripPNG(200, 300, color.NRGBA{R: 100, G: 150, B: 200, A: 255})
	p1.Filename = "001.png"
	p2 := makeStripPNG(200, 350, color.NRGBA{R: 100, G: 150, B: 200, A: 255})
	p2.Filename = "002.png"
	p3 := makeStripPNG(200, 250, color.NRGBA{R: 255, G: 255, B: 255, A: 255})
	p3.Filename = "003.png"

	// Stitch 001 + 002 into one 650px Virtual Continuous Canvas; keep 003 separate
	callIdx := 0
	units, err := BuildVirtualUnits([]InputPage{p1, p2, p3}, 2000, func(_, _ []byte) bool {
		callIdx++
		return callIdx == 1
	})
	if err != nil {
		t.Fatalf("BuildVirtualUnits: %v", err)
	}
	if len(units) != 2 {
		t.Fatalf("expected 2 virtual units, got %d", len(units))
	}
	if units[0].Height != 650 || len(units[0].SeamSlices) != 2 {
		t.Fatalf("unexpected stitched unit: h=%d slices=%d", units[0].Height, len(units[0].SeamSlices))
	}

	cbzPath := filepath.Join(t.TempDir(), "out.cbz")
	if err := WriteCBZ(cbzPath, []OutputEntry{
		{Filename: "10.png", Data: p2.PNGBytes},
		{Filename: "2.png", Data: p1.PNGBytes},
	}); err != nil {
		t.Fatalf("WriteCBZ: %v", err)
	}
	readBack, err := ReadInputPages(cbzPath)
	if err != nil {
		t.Fatalf("ReadInputPages CBZ: %v", err)
	}
	if len(readBack) != 2 || readBack[0].Filename != "2.png" || readBack[1].Filename != "10.png" {
		t.Fatalf("unexpected CBZ natural order: %+v", readBack)
	}

	// Verify lossless SeamSlice re-slicing recovers exact constituent heights & filenames
	sliced, err := SliceCompositePNG(units[0].PNGBytes, units[0].SeamSlices, units[0].Name)
	if err != nil {
		t.Fatalf("SliceCompositePNG: %v", err)
	}
	if len(sliced) != 2 || sliced[0].Filename != "001.png" || sliced[1].Filename != "002.png" {
		t.Fatalf("unexpected sliced entries: len=%d", len(sliced))
	}
	for i, expH := range []int{300, 350} {
		dec, _, dErr := image.DecodeConfig(bytes.NewReader(sliced[i].Data))
		if dErr != nil {
			t.Fatalf("decode sliced[%d]: %v", i, dErr)
		}
		if dec.Width != 200 || dec.Height != expH {
			t.Fatalf("sliced[%d] dims %dx%d != 200x%d", i, dec.Width, dec.Height, expH)
		}
	}
}

func TestAVIFDecodeFileAndCBZ(t *testing.T) {
	if !IsSupportedImage("001.avif") || !IsSupportedImage("PAGE.AVIF") {
		t.Fatalf("expected .avif to be recognized as supported image")
	}

	// Encode a real AVIF image and verify NormalizeToPNG, ReadInputPages(file), and ReadInputPages(.cbz).
	im := image.NewNRGBA(image.Rect(0, 0, 48, 64))
	for y := 0; y < 64; y++ {
		for x := 0; x < 48; x++ {
			im.SetNRGBA(x, y, color.NRGBA{R: 220, G: 180, B: 90, A: 255})
		}
	}
	var avifBuf bytes.Buffer
	if err := avif.Encode(&avifBuf, im); err != nil {
		t.Fatalf("encode avif: %v", err)
	}
	avifBytes := avifBuf.Bytes()

	pngBytes, hash, w, h, err := NormalizeToPNG(avifBytes)
	if err != nil {
		t.Fatalf("NormalizeToPNG(avif): %v", err)
	}
	if w != 48 || h != 64 || len(pngBytes) == 0 || hash == "" {
		t.Fatalf("unexpected decoded AVIF metadata: %dx%d len=%d hash=%q", w, h, len(pngBytes), hash)
	}

	tmpDir := t.TempDir()
	avifPath := filepath.Join(tmpDir, "001.avif")
	if err := os.WriteFile(avifPath, avifBytes, 0o644); err != nil {
		t.Fatal(err)
	}
	single, err := ReadInputPages(avifPath)
	if err != nil || len(single) != 1 || single[0].Width != 48 || single[0].Height != 64 {
		t.Fatalf("ReadInputPages(avif): pages=%d err=%v", len(single), err)
	}

	cbzPath := filepath.Join(tmpDir, "avif_comic.cbz")
	if err := WriteCBZ(cbzPath, []OutputEntry{
		{Filename: "02.avif", Data: avifBytes},
		{Filename: "01.avif", Data: avifBytes},
	}); err != nil {
		t.Fatalf("WriteCBZ(avif): %v", err)
	}
	cbzPages, err := ReadInputPages(cbzPath)
	if err != nil || len(cbzPages) != 2 || cbzPages[0].Filename != "01.avif" || cbzPages[1].Filename != "02.avif" {
		t.Fatalf("ReadInputPages(avif cbz): pages=%d err=%v", len(cbzPages), err)
	}
}
