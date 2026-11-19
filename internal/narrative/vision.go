package narrative

import (
	"bytes"
	"context"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/jpeg"
	_ "image/png"
	"math"
	"sort"
	"strconv"

	xdraw "golang.org/x/image/draw"
	"golang.org/x/image/font"
	"golang.org/x/image/font/basicfont"
	"golang.org/x/image/math/fixed"
)

const (
	sceneOverviewMaxDim         = 960
	sceneOverviewJPEGQuality    = 76
	nodeCropSheetJPEGQuality    = 88
	nodeCropContextPaddingPx    = 4
	nodeCropSheetCellGapPx      = 6
	nodeCropOversizeBoxPx       = 420
	nodeCropOversizeThickPx     = 36
	nodeCropTargetThickPx       = 28.0
	nodeCropSheetMaxTargetWidth = 1400
)

type parsedPageImageKey struct{}

type cropSheetTile struct {
	Index      int     `json:"index"`
	ID         string  `json:"id"`
	Scale      float64 `json:"scale"`
	SourceRect [4]int  `json:"source_rect"`
	SheetRect  [4]int  `json:"sheet_rect"`
}

func decodePageImage(ctx context.Context, raw []byte) (image.Image, error) {
	if img, ok := ctx.Value(parsedPageImageKey{}).(image.Image); ok && img != nil {
		return img, nil
	}
	img, _, err := image.Decode(bytes.NewReader(raw))
	return img, err
}

func fillSolidRect(dst *image.RGBA, r image.Rectangle, c color.Color) {
	clipped := r.Intersect(dst.Bounds())
	if clipped.Empty() {
		return
	}
	draw.Draw(dst, clipped, image.NewUniform(c), image.Point{}, draw.Src)
}

// drawExteriorFrame draws a 1px border immediately outside box, clipped to clip.
func drawExteriorFrame(dst *image.RGBA, box, clip image.Rectangle, c color.Color) {
	clippedBox := box.Intersect(clip)
	if clippedBox.Empty() {
		return
	}
	edges := [4]image.Rectangle{
		image.Rect(clippedBox.Min.X-1, clippedBox.Min.Y-1, clippedBox.Max.X+1, clippedBox.Min.Y),
		image.Rect(clippedBox.Min.X-1, clippedBox.Max.Y, clippedBox.Max.X+1, clippedBox.Max.Y+1),
		image.Rect(clippedBox.Min.X-1, clippedBox.Min.Y-1, clippedBox.Min.X, clippedBox.Max.Y+1),
		image.Rect(clippedBox.Max.X, clippedBox.Min.Y-1, clippedBox.Max.X+1, clippedBox.Max.Y+1),
	}
	for _, edge := range edges {
		if r := edge.Intersect(clip); !r.Empty() {
			fillSolidRect(dst, r, c)
		}
	}
}

func scaleBBox(b []int, scale float64, bounds image.Rectangle) (image.Rectangle, bool) {
	if len(b) != 4 {
		return image.Rectangle{}, false
	}
	x0 := int(math.Floor(float64(b[0]) * scale))
	y0 := int(math.Floor(float64(b[1]) * scale))
	x1 := int(math.Ceil(float64(b[0]+b[2]) * scale))
	y1 := int(math.Ceil(float64(b[1]+b[3]) * scale))
	if x1 <= x0 {
		x1 = x0 + 1
	}
	if y1 <= y0 {
		y1 = y0 + 1
	}
	r := image.Rect(x0, y0, x1, y1).Intersect(bounds)
	return r, !r.Empty()
}

func buildSceneOverviewJPEG(source image.Image, nodes []NodeTranslationRequest) ([]byte, error) {
	srcBounds := source.Bounds()
	srcW, srcH := srcBounds.Dx(), srcBounds.Dy()
	if srcW <= 0 || srcH <= 0 {
		return nil, fmt.Errorf("source image has empty bounds")
	}

	scale := 1.0
	dstW, dstH := srcW, srcH
	if maxDim := max(srcW, srcH); maxDim > sceneOverviewMaxDim {
		scale = float64(sceneOverviewMaxDim) / float64(maxDim)
		dstW = max(1, int(math.Round(float64(srcW)*scale)))
		dstH = max(1, int(math.Round(float64(srcH)*scale)))
	}

	overview := image.NewRGBA(image.Rect(0, 0, dstW, dstH))
	if dstW == srcW && dstH == srcH {
		draw.Draw(overview, overview.Bounds(), source, srcBounds.Min, draw.Src)
	} else {
		xdraw.CatmullRom.Scale(overview, overview.Bounds(), source, srcBounds, xdraw.Src, nil)
	}

	scaledBoxes := make([]image.Rectangle, len(nodes))
	hasBox := make([]bool, len(nodes))
	for i, node := range nodes {
		scaledBoxes[i], hasBox[i] = scaleBBox(node.BBox, scale, overview.Bounds())
	}

	borderColor := color.RGBA{R: 225, G: 29, B: 72, A: 255}
	pillBg := color.RGBA{R: 9, G: 9, B: 11, A: 255}
	pillFg := color.RGBA{R: 255, G: 255, B: 255, A: 255}
	face := basicfont.Face7x13
	pillH := face.Metrics().Height.Ceil() + 2
	ascent := face.Metrics().Ascent.Ceil()

	for i := range nodes {
		if !hasBox[i] {
			continue
		}
		drawExteriorFrame(overview, scaledBoxes[i], overview.Bounds(), borderColor)
	}

	// Render exterior index callouts after borders so tags stay legible.
	for i := range nodes {
		if !hasBox[i] {
			continue
		}
		r := scaledBoxes[i]
		tag := "#" + strconv.Itoa(i+1)
		pillW := font.MeasureString(face, tag).Ceil() + 6
		candidates := []image.Point{
			{X: r.Min.X, Y: r.Min.Y - pillH - 2},
			{X: r.Min.X, Y: r.Max.Y + 2},
			{X: r.Max.X + 2, Y: r.Min.Y},
			{X: r.Min.X - pillW - 2, Y: r.Min.Y},
		}
		chosen := image.Rectangle{}
		foundClean := false
		foundInBounds := false
		for _, pt := range candidates {
			cand := image.Rect(pt.X, pt.Y, pt.X+pillW, pt.Y+pillH)
			if !cand.In(overview.Bounds()) {
				continue
			}
			if !foundInBounds {
				chosen = cand
				foundInBounds = true
			}
			overlaps := false
			for j, other := range scaledBoxes {
				if hasBox[j] && cand.Overlaps(other) {
					overlaps = true
					break
				}
			}
			if !overlaps {
				chosen = cand
				foundClean = true
				break
			}
		}
		if !foundClean && !foundInBounds {
			px := min(max(0, r.Min.X), max(0, dstW-pillW))
			py := min(max(0, r.Min.Y-pillH), max(0, dstH-pillH))
			chosen = image.Rect(px, py, min(dstW, px+pillW), min(dstH, py+pillH))
		}
		if chosen.Empty() {
			continue
		}
		fillSolidRect(overview, chosen, pillBg)
		drawer := font.Drawer{
			Dst:  overview,
			Src:  image.NewUniform(pillFg),
			Face: face,
			Dot:  fixed.P(chosen.Min.X+3, chosen.Min.Y+1+ascent),
		}
		drawer.DrawString(tag)
	}

	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, overview, &jpeg.Options{Quality: sceneOverviewJPEGQuality}); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func medianLineThickness(lines [][]int) float64 {
	if len(lines) == 0 {
		return 0
	}
	thicknesses := make([]int, 0, len(lines))
	for _, box := range lines {
		if len(box) != 4 || box[2] <= 0 || box[3] <= 0 {
			continue
		}
		thicknesses = append(thicknesses, min(box[2], box[3]))
	}
	if len(thicknesses) == 0 {
		return 0
	}
	sort.Ints(thicknesses)
	mid := len(thicknesses) / 2
	if len(thicknesses)%2 == 1 {
		return float64(thicknesses[mid])
	}
	return float64(thicknesses[mid-1]+thicknesses[mid]) / 2.0
}

func buildNodeCropSheetJPEG(source image.Image, nodes []NodeTranslationRequest) ([]byte, []cropSheetTile, error) {
	if len(nodes) == 0 {
		return nil, nil, nil
	}
	srcBounds := source.Bounds()
	face := basicfont.Face7x13
	headerH := face.Metrics().Height.Ceil() + 2
	ascent := face.Metrics().Ascent.Ceil()

	type sheetItem struct {
		tile      cropSheetTile
		rawRect   image.Rectangle
		srcRect   image.Rectangle
		drawFrame bool
		dstW      int
		dstH      int
		cellW     int
		cellH     int
		labelText string
	}

	items := make([]sheetItem, len(nodes))
	order := make([]int, len(nodes))
	totalArea, widestCell := 0, 0

	for i, node := range nodes {
		if len(node.BBox) != 4 {
			return nil, nil, fmt.Errorf("node %s has invalid bbox", node.ID)
		}
		b := node.BBox
		rawRect := image.Rect(b[0], b[1], b[0]+b[2], b[1]+b[3]).Intersect(srcBounds)
		if rawRect.Empty() {
			return nil, nil, fmt.Errorf("node %s bbox is outside source image", node.ID)
		}
		cropW, cropH := rawRect.Dx(), rawRect.Dy()
		padPx := nodeCropContextPaddingPx
		isSmallCompact := min(cropW, cropH) < 72 || max(cropW, cropH) < 120
		if isSmallCompact {
			padPx = max(12, min(40, int(math.Round(0.6*float64(max(cropW, cropH))))))
		}
		padded := image.Rect(
			rawRect.Min.X-padPx,
			rawRect.Min.Y-padPx,
			rawRect.Max.X+padPx,
			rawRect.Max.Y+padPx,
		).Intersect(srcBounds)

		scale := 1.0
		if max(cropW, cropH) > nodeCropOversizeBoxPx {
			if thick := medianLineThickness(node.SourceLines); thick > nodeCropOversizeThickPx {
				scale = min(1.0, nodeCropTargetThickPx/thick)
			}
		}

		dstW := max(1, int(math.Round(float64(padded.Dx())*scale)))
		dstH := max(1, int(math.Round(float64(padded.Dy())*scale)))
		effectiveScale := float64(dstW) / float64(padded.Dx())

		label := fmt.Sprintf("#%d %s", i+1, node.ID)
		labelW := font.MeasureString(face, label).Ceil() + 4
		cellW := max(dstW, labelW)
		cellH := headerH + dstH

		items[i] = sheetItem{
			tile: cropSheetTile{
				Index:      i + 1,
				ID:         node.ID,
				Scale:      effectiveScale,
				SourceRect: [4]int{padded.Min.X, padded.Min.Y, padded.Dx(), padded.Dy()},
			},
			rawRect:   rawRect,
			srcRect:   padded,
			drawFrame: isSmallCompact,
			dstW:      dstW,
			dstH:      dstH,
			cellW:     cellW,
			cellH:     cellH,
			labelText: label,
		}
		totalArea += (cellW + nodeCropSheetCellGapPx) * (cellH + nodeCropSheetCellGapPx)
		widestCell = max(widestCell, cellW)
		order[i] = i
	}

	sort.SliceStable(order, func(i, j int) bool {
		if items[order[i]].cellH != items[order[j]].cellH {
			return items[order[i]].cellH > items[order[j]].cellH
		}
		return items[order[i]].cellW > items[order[j]].cellW
	})

	targetW := max(widestCell+2*nodeCropSheetCellGapPx, min(nodeCropSheetMaxTargetWidth, int(math.Ceil(math.Sqrt(float64(totalArea)*1.2)))))
	x, y, rowH := nodeCropSheetCellGapPx, nodeCropSheetCellGapPx, 0
	sheetW := widestCell + 2*nodeCropSheetCellGapPx

	for _, idx := range order {
		it := &items[idx]
		if x > nodeCropSheetCellGapPx && x+it.cellW+nodeCropSheetCellGapPx > targetW {
			x = nodeCropSheetCellGapPx
			y += rowH + nodeCropSheetCellGapPx
			rowH = 0
		}
		it.tile.SheetRect = [4]int{x, y + headerH, it.dstW, it.dstH}
		sheetW = max(sheetW, x+it.cellW+nodeCropSheetCellGapPx)
		x += it.cellW + nodeCropSheetCellGapPx
		rowH = max(rowH, it.cellH)
	}
	sheetH := y + rowH + nodeCropSheetCellGapPx

	sheet := image.NewRGBA(image.Rect(0, 0, sheetW, sheetH))
	draw.Draw(sheet, sheet.Bounds(), image.NewUniform(color.White), image.Point{}, draw.Src)

	tiles := make([]cropSheetTile, len(items))
	headerColor := color.RGBA{R: 24, G: 24, B: 27, A: 255}
	frameColor := color.RGBA{R: 225, G: 29, B: 72, A: 255}
	for i, it := range items {
		sr := it.tile.SheetRect
		dstRect := image.Rect(sr[0], sr[1], sr[0]+sr[2], sr[1]+sr[3])
		if it.dstW == it.srcRect.Dx() && it.dstH == it.srcRect.Dy() {
			draw.Draw(sheet, dstRect, source, it.srcRect.Min, draw.Src)
		} else {
			xdraw.CatmullRom.Scale(sheet, dstRect, source, it.srcRect, xdraw.Src, nil)
		}
		if it.drawFrame {
			relX0 := int(math.Floor(float64(it.rawRect.Min.X-it.srcRect.Min.X) * it.tile.Scale))
			relY0 := int(math.Floor(float64(it.rawRect.Min.Y-it.srcRect.Min.Y) * it.tile.Scale))
			relX1 := int(math.Ceil(float64(it.rawRect.Max.X-it.srcRect.Min.X) * it.tile.Scale))
			relY1 := int(math.Ceil(float64(it.rawRect.Max.Y-it.srcRect.Min.Y) * it.tile.Scale))
			cellRaw := image.Rect(sr[0]+relX0, sr[1]+relY0, sr[0]+relX1, sr[1]+relY1)
			drawExteriorFrame(sheet, cellRaw, dstRect, frameColor)
		}
		drawer := font.Drawer{
			Dst:  sheet,
			Src:  image.NewUniform(headerColor),
			Face: face,
			Dot:  fixed.P(sr[0], sr[1]-headerH+ascent+1),
		}
		drawer.DrawString(it.labelText)
		tiles[i] = it.tile
	}

	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, sheet, &jpeg.Options{Quality: nodeCropSheetJPEGQuality}); err != nil {
		return nil, nil, err
	}
	return buf.Bytes(), tiles, nil
}
