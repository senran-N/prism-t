package psd

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"fmt"
	"image"
	"image/draw"
	_ "image/jpeg"
	"image/png"
	"unicode/utf16"

	"prism-t/internal/dom"
)

type Layer struct {
	Name       string
	Left       int
	Top        int
	Width      int
	Height     int
	RGBA       []byte // Row-major 4 * Width * Height bytes
	Opacity    uint8
	OpacitySet bool   // When true, Opacity=0 means fully transparent rather than defaulting to 255
	BlendMode  string // "norm", "mul ", "scrn"
}

type Document struct {
	Width     int
	Height    int
	Composite []byte // Row-major 3 * Width * Height RGB bytes (or 4*W*H converted)
	Layers    []Layer
}

// PackBitsEncode compresses a single scanline using the exact Macintosh PackBits RLE algorithm.
func PackBitsEncode(src []byte) []byte {
	n := len(src)
	if n == 0 {
		return nil
	}
	out := make([]byte, 0, n+n/64+4)
	i := 0
	for i < n {
		// Check for run of >= 3 identical bytes (or 2 at end/run)
		runEnd := i + 1
		for runEnd < n && runEnd-i < 128 && src[runEnd] == src[i] {
			runEnd++
		}
		runLen := runEnd - i
		if runLen >= 3 {
			// Header byte = uint8(1 - runLen) in two's complement (i.e. 257 - runLen)
			out = append(out, byte(int8(1-runLen)), src[i])
			i = runEnd
			continue
		}

		// Literal run up to 128 bytes, stopping before any >= 3 identical byte run
		litStart := i
		for i < n && i-litStart < 128 {
			if i+2 < n && src[i] == src[i+1] && src[i] == src[i+2] {
				break
			}
			i++
		}
		litLen := i - litStart
		out = append(out, byte(litLen-1))
		out = append(out, src[litStart:i]...)
	}
	return out
}

// PackBitsDecode decodes a PackBits RLE buffer of expected length `expectedLen` (used for self-verification).
func PackBitsDecode(encoded []byte, expectedLen int) ([]byte, error) {
	out := make([]byte, 0, expectedLen)
	i := 0
	for i < len(encoded) && len(out) < expectedLen {
		hdr := int8(encoded[i])
		i++
		if hdr >= 0 {
			count := int(hdr) + 1
			if i+count > len(encoded) {
				return nil, fmt.Errorf("packbits literal truncated")
			}
			out = append(out, encoded[i:i+count]...)
			i += count
		} else if hdr != -128 {
			count := 1 - int(hdr)
			if i >= len(encoded) {
				return nil, fmt.Errorf("packbits repeat truncated")
			}
			val := encoded[i]
			i++
			for k := 0; k < count; k++ {
				out = append(out, val)
			}
		}
	}
	if len(out) != expectedLen {
		return nil, fmt.Errorf("packbits decoded %d bytes, expected %d", len(out), expectedLen)
	}
	return out, nil
}

func encodeChannelRLE(plane []byte, w, h int) ([]byte, error) {
	if w <= 0 || h <= 0 {
		var b [2]byte
		binary.BigEndian.PutUint16(b[:], 0)
		return b[:], nil
	}
	rows := make([][]byte, h)
	totalCompressed := 0
	for y := 0; y < h; y++ {
		r := PackBitsEncode(plane[y*w : (y+1)*w])
		if len(r) > 0xFFFF {
			return nil, fmt.Errorf("scanline RLE exceeds uint16 limit (%d bytes)", len(r))
		}
		rows[y] = r
		totalCompressed += len(r)
	}

	buf := bytes.NewBuffer(make([]byte, 0, 2+2*h+totalCompressed))
	_ = binary.Write(buf, binary.BigEndian, uint16(1)) // 1 = PackBits RLE
	for y := 0; y < h; y++ {
		_ = binary.Write(buf, binary.BigEndian, uint16(len(rows[y])))
	}
	for y := 0; y < h; y++ {
		buf.Write(rows[y])
	}
	return buf.Bytes(), nil
}

func encodeLayerExtraData(name string) []byte {
	var extra bytes.Buffer
	// 1. Layer mask data length = 0
	_ = binary.Write(&extra, binary.BigEndian, uint32(0))
	// 2. Layer blending ranges length = 0
	_ = binary.Write(&extra, binary.BigEndian, uint32(0))

	// 3. Pascal ASCII fallback string padded to 4-byte multiple
	ascii := make([]byte, 0, len(name))
	for _, r := range name {
		if r >= 0x20 && r <= 0x7E {
			ascii = append(ascii, byte(r))
		} else {
			ascii = append(ascii, '_')
		}
		if len(ascii) >= 31 {
			break
		}
	}
	if len(ascii) == 0 {
		ascii = []byte("Layer")
	}
	extra.WriteByte(byte(len(ascii)))
	extra.Write(ascii)
	totalPascal := 1 + len(ascii)
	if rem := totalPascal % 4; rem != 0 {
		for k := 0; k < 4-rem; k++ {
			extra.WriteByte(0)
		}
	}

	// 4. Additional Layer Info: Unicode layer name ("8BIM" + "luni")
	u16 := utf16.Encode([]rune(name))
	var luniData bytes.Buffer
	_ = binary.Write(&luniData, binary.BigEndian, uint32(len(u16)))
	for _, code := range u16 {
		_ = binary.Write(&luniData, binary.BigEndian, code)
	}
	if luniData.Len()%4 != 0 {
		pad := 4 - (luniData.Len() % 4)
		for k := 0; k < pad; k++ {
			luniData.WriteByte(0)
		}
	}
	extra.WriteString("8BIM")
	extra.WriteString("luni")
	_ = binary.Write(&extra, binary.BigEndian, uint32(luniData.Len()))
	extra.Write(luniData.Bytes())

	return extra.Bytes()
}

// Encode writes a spec-compliant multi-layer RGB 8-bit Adobe Photoshop .psd file.
func Encode(doc *Document) ([]byte, error) {
	if doc.Width <= 0 || doc.Height <= 0 {
		return nil, fmt.Errorf("invalid PSD dimensions %dx%d", doc.Width, doc.Height)
	}
	if len(doc.Composite) != doc.Width*doc.Height*3 {
		return nil, fmt.Errorf("composite RGB size mismatch: got %d, want %d", len(doc.Composite), doc.Width*doc.Height*3)
	}

	var out bytes.Buffer

	// 1. File Header (26 bytes)
	out.WriteString("8BPS")
	_ = binary.Write(&out, binary.BigEndian, uint16(1)) // Version 1
	out.Write(make([]byte, 6))                          // Reserved
	_ = binary.Write(&out, binary.BigEndian, uint16(3)) // 3 channels in merged composite
	_ = binary.Write(&out, binary.BigEndian, uint32(doc.Height))
	_ = binary.Write(&out, binary.BigEndian, uint32(doc.Width))
	_ = binary.Write(&out, binary.BigEndian, uint16(8)) // 8 bits per channel
	_ = binary.Write(&out, binary.BigEndian, uint16(3)) // RGB color mode

	// 2. Color Mode Data (4 bytes)
	_ = binary.Write(&out, binary.BigEndian, uint32(0))

	// 3. Image Resources (4 bytes)
	_ = binary.Write(&out, binary.BigEndian, uint32(0))

	// 4. Layer and Mask Information Section
	if len(doc.Layers) == 0 {
		_ = binary.Write(&out, binary.BigEndian, uint32(0))
	} else {
		type encodedLayer struct {
			rec      []byte
			channels [4][]byte // [-1 (A), 0 (R), 1 (G), 2 (B)]
		}
		encLayers := make([]encodedLayer, len(doc.Layers))
		chIDs := [4]int16{-1, 0, 1, 2}
		chOffsets := [4]int{3, 0, 1, 2} // Index inside RGBA pixel

		for idx, lyr := range doc.Layers {
			lw, lh := lyr.Width, lyr.Height
			if len(lyr.RGBA) != lw*lh*4 {
				return nil, fmt.Errorf("layer %q RGBA length mismatch", lyr.Name)
			}
			var el encodedLayer
			for c := 0; c < 4; c++ {
				plane := make([]byte, lw*lh)
				off := chOffsets[c]
				for p := 0; p < lw*lh; p++ {
					plane[p] = lyr.RGBA[p*4+off]
				}
				comp, err := encodeChannelRLE(plane, lw, lh)
				if err != nil {
					return nil, err
				}
				el.channels[c] = comp
			}

			var rec bytes.Buffer
			_ = binary.Write(&rec, binary.BigEndian, int32(lyr.Top))
			_ = binary.Write(&rec, binary.BigEndian, int32(lyr.Left))
			_ = binary.Write(&rec, binary.BigEndian, int32(lyr.Top+lh))
			_ = binary.Write(&rec, binary.BigEndian, int32(lyr.Left+lw))
			_ = binary.Write(&rec, binary.BigEndian, uint16(4))
			for c := 0; c < 4; c++ {
				_ = binary.Write(&rec, binary.BigEndian, chIDs[c])
				_ = binary.Write(&rec, binary.BigEndian, uint32(len(el.channels[c])))
			}
			rec.WriteString("8BIM")
			bkey := lyr.BlendMode
			if len(bkey) != 4 {
				bkey = "norm"
			}
			rec.WriteString(bkey)
			op := lyr.Opacity
			if op == 0 && !lyr.OpacitySet {
				op = 255
			}
			rec.WriteByte(op)
			rec.WriteByte(0) // Clipping base
			rec.WriteByte(0) // Flags (visible)
			rec.WriteByte(0) // Filler

			extra := encodeLayerExtraData(lyr.Name)
			_ = binary.Write(&rec, binary.BigEndian, uint32(len(extra)))
			rec.Write(extra)
			el.rec = rec.Bytes()
			encLayers[idx] = el
		}

		var layerInfoBody bytes.Buffer
		_ = binary.Write(&layerInfoBody, binary.BigEndian, int16(len(encLayers)))
		// Layer records and channel data share the document's bottom-to-top order.
		// Independently recomposing the exported stack must reproduce Composite.
		for i := 0; i < len(encLayers); i++ {
			layerInfoBody.Write(encLayers[i].rec)
		}
		for i := 0; i < len(encLayers); i++ {
			el := encLayers[i]
			for c := 0; c < 4; c++ {
				layerInfoBody.Write(el.channels[c])
			}
		}
		if layerInfoBody.Len()%2 != 0 {
			layerInfoBody.WriteByte(0)
		}

		var layerAndMask bytes.Buffer
		_ = binary.Write(&layerAndMask, binary.BigEndian, uint32(layerInfoBody.Len()))
		layerAndMask.Write(layerInfoBody.Bytes())
		// Global Layer Mask Info = 0
		_ = binary.Write(&layerAndMask, binary.BigEndian, uint32(0))

		_ = binary.Write(&out, binary.BigEndian, uint32(layerAndMask.Len()))
		out.Write(layerAndMask.Bytes())
	}

	// 5. Merged Composite Image Data Section (PackBits RLE across R, G, B planes)
	W, H := doc.Width, doc.Height
	rowCounts := make([]uint16, 3*H)
	compRows := make([][]byte, 3*H)
	planeRow := make([]byte, W)

	for c := 0; c < 3; c++ {
		for y := 0; y < H; y++ {
			base := y * W * 3
			for x := 0; x < W; x++ {
				planeRow[x] = doc.Composite[base+x*3+c]
			}
			rle := PackBitsEncode(planeRow)
			if len(rle) > 0xFFFF {
				return nil, fmt.Errorf("composite scanline RLE overflow")
			}
			idx := c*H + y
			rowCounts[idx] = uint16(len(rle))
			compRows[idx] = rle
		}
	}

	_ = binary.Write(&out, binary.BigEndian, uint16(1)) // PackBits RLE
	for _, rc := range rowCounts {
		_ = binary.Write(&out, binary.BigEndian, rc)
	}
	for _, cr := range compRows {
		out.Write(cr)
	}

	return out.Bytes(), nil
}

func decodeNRGBA(pngBytes []byte) (*image.NRGBA, error) {
	im, _, err := image.Decode(bytes.NewReader(pngBytes))
	if err != nil {
		return nil, err
	}
	b := im.Bounds()
	nrgba := image.NewNRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	draw.Draw(nrgba, nrgba.Bounds(), im, b.Min, draw.Src)
	return nrgba, nil
}

func decodeB64PNG(b64 string) (*image.NRGBA, error) {
	raw, err := base64.StdEncoding.DecodeString(b64)
	if err != nil {
		return nil, err
	}
	im, err := png.Decode(bytes.NewReader(raw))
	if err != nil {
		return nil, err
	}
	b := im.Bounds()
	nrgba := image.NewNRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	draw.Draw(nrgba, nrgba.Bounds(), im, b.Min, draw.Src)
	return nrgba, nil
}

// BuildFromComicDOM assembles independently editable Z0/Z1/Z2 layers.
func BuildFromComicDOM(rawImgBytes, compositeImgBytes []byte, pageDOM dom.PageDOM) (*Document, error) {
	raw, err := decodeNRGBA(rawImgBytes)
	if err != nil {
		return nil, fmt.Errorf("decode raw: %w", err)
	}
	composite, err := decodeNRGBA(compositeImgBytes)
	if err != nil {
		return nil, fmt.Errorf("decode composite: %w", err)
	}
	W, H := raw.Bounds().Dx(), raw.Bounds().Dy()
	if composite.Bounds() != raw.Bounds() {
		return nil, fmt.Errorf("raw/composite dimensions differ")
	}
	compRGB := make([]byte, W*H*3)
	for i := 0; i < W*H; i++ {
		copy(compRGB[i*3:i*3+3], composite.Pix[i*4:i*4+3])
	}
	clean := make([]byte, W*H*4)
	foreground := make([]byte, W*H*4)
	layers := []Layer{
		{Name: "00_Original_Raw", Width: W, Height: H, RGBA: raw.Pix, Opacity: 255, BlendMode: "norm"},
		{Name: "01_Restored_Clean_Z0", Width: W, Height: H, RGBA: clean, Opacity: 255, BlendMode: "norm"},
	}
	for _, node := range pageDOM.Nodes {
		if len(node.Patches.ROI) != 4 {
			return nil, fmt.Errorf("node %s: invalid ROI", node.ID)
		}
		b := node.Patches.ROI
		rx, ry, rw, rh := b[0], b[1], b[2], b[3]
		if rw <= 0 || rh <= 0 || rx < 0 || ry < 0 || rx+rw > W || ry+rh > H {
			return nil, fmt.Errorf("node %s: ROI is outside page", node.ID)
		}
		patch := func(payload string) (*image.NRGBA, error) {
			if payload == "" {
				return nil, nil
			}
			im, err := decodeB64PNG(payload)
			if err != nil {
				return nil, fmt.Errorf("node %s: decode patch: %w", node.ID, err)
			}
			if im.Bounds().Dx() != rw || im.Bounds().Dy() != rh {
				return nil, fmt.Errorf("node %s: patch dimensions differ from ROI", node.ID)
			}
			return im, nil
		}
		z0, err := patch(node.Patches.CleanZ0PNGB64)
		if err != nil {
			return nil, err
		}
		alpha, err := patch(node.Patches.StrokeAlphaPNGB64)
		if err != nil {
			return nil, err
		}
		z2, err := patch(node.Patches.Z2OccluderMaskB64)
		if err != nil {
			return nil, err
		}
		if (z0 == nil) != (alpha == nil) {
			return nil, fmt.Errorf("node %s: clean plate and stroke alpha must be supplied together", node.ID)
		}
		for y := 0; y < rh; y++ {
			for x := 0; x < rw; x++ {
				si, di := (y*rw+x)*4, ((ry+y)*W+rx+x)*4
				if z0 != nil && alpha.Pix[si] > 0 && (z2 == nil || z2.Pix[si] == 0) {
					copy(clean[di:di+3], z0.Pix[si:si+3])
					clean[di+3] = 255
				}
				if z2 != nil && z2.Pix[si] > 127 {
					copy(foreground[di:di+3], raw.Pix[di:di+3])
					foreground[di+3] = 255
				}
			}
		}
		z1, err := patch(node.Patches.RenderedZ1PNGB64)
		if err != nil {
			return nil, err
		}
		if z1 == nil {
			continue
		}
		if node.FusionMode == "diegetic_planar" && node.Patches.ShadingMapPNGB64 != "" {
			shading, err := patch(node.Patches.ShadingMapPNGB64)
			if err != nil {
				return nil, err
			}
			for i := 0; i < rw*rh; i++ {
				for c := 0; c < 3; c++ {
					j := i*4 + c
					z1.Pix[j] = uint8((uint16(z1.Pix[j])*uint16(shading.Pix[j]) + 127) / 255)
				}
			}
		}
		blend := "norm"
		switch node.Style.BlendMode {
		case "multiply":
			blend = "mul "
		case "screen":
			blend = "scrn"
		}
		label := node.TargetText
		if label == "" {
			label = node.SourceText
		}
		runes := []rune(label)
		if len(runes) > 14 {
			label = string(runes[:14]) + "…"
		}
		layers = append(layers, Layer{Name: fmt.Sprintf("%02d_Node_Z1_%s_%s", len(layers), node.ID, label), Left: rx, Top: ry, Width: rw, Height: rh, RGBA: z1.Pix, Opacity: 255, OpacitySet: true, BlendMode: blend})
	}
	layers = append(layers, Layer{Name: "99_Foreground_Occluders_Z2", Width: W, Height: H, RGBA: foreground, Opacity: 255, BlendMode: "norm"})
	return &Document{Width: W, Height: H, Composite: compRGB, Layers: layers}, nil
}
