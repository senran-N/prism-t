package archive

import (
	"archive/zip"
	"bytes"
	"encoding/hex"
	"fmt"
	"image"
	"image/draw"
	_ "image/gif"
	_ "image/jpeg"
	"image/png"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"unicode"

	_ "github.com/gen2brain/avif"
	"github.com/zeebo/blake3"
	_ "golang.org/x/image/webp"

	"prism-t/internal/dom"
)

type InputPage struct {
	Filename   string
	PNGBytes   []byte
	Blake3Hash string
	Width      int
	Height     int
}

type VirtualUnit struct {
	Name       string
	PNGBytes   []byte
	Blake3Hash string
	Width      int
	Height     int
	SeamSlices []dom.SeamSlice
}

func IsSupportedImage(name string) bool {
	ext := strings.ToLower(filepath.Ext(name))
	switch ext {
	case ".png", ".jpg", ".jpeg", ".webp", ".gif", ".avif":
		return true
	default:
		return false
	}
}

// NaturalLess compares strings in natural human numeric order (e.g. "p2.png" < "p10.png").
func NaturalLess(a, b string) bool {
	ra, rb := []rune(strings.ToLower(a)), []rune(strings.ToLower(b))
	ia, ib := 0, 0
	for ia < len(ra) && ib < len(rb) {
		da, db := unicode.IsDigit(ra[ia]), unicode.IsDigit(rb[ib])
		if da && db {
			sa, sb := ia, ib
			for ia < len(ra) && unicode.IsDigit(ra[ia]) {
				ia++
			}
			for ib < len(rb) && unicode.IsDigit(rb[ib]) {
				ib++
			}
			na := strings.TrimLeft(string(ra[sa:ia]), "0")
			nb := strings.TrimLeft(string(rb[sb:ib]), "0")
			if len(na) != len(nb) {
				return len(na) < len(nb)
			}
			if na != nb {
				return na < nb
			}
			if (ia - sa) != (ib - sb) {
				return (ia - sa) < (ib - sb)
			}
			continue
		}
		if ra[ia] != rb[ib] {
			return ra[ia] < rb[ib]
		}
		ia++
		ib++
	}
	return len(ra) < len(rb)
}

func NormalizeToPNG(raw []byte) ([]byte, string, int, int, error) {
	im, fmtName, err := image.Decode(bytes.NewReader(raw))
	if err != nil {
		return nil, "", 0, 0, fmt.Errorf("decode image: %w", err)
	}
	b := im.Bounds()
	w, h := b.Dx(), b.Dy()
	var pngBytes []byte
	if fmtName == "png" {
		pngBytes = raw
	} else {
		var buf bytes.Buffer
		enc := png.Encoder{CompressionLevel: png.BestSpeed}
		if err := enc.Encode(&buf, im); err != nil {
			return nil, "", 0, 0, err
		}
		pngBytes = buf.Bytes()
	}
	sum := blake3.Sum256(pngBytes)
	return pngBytes, hex.EncodeToString(sum[:16]), w, h, nil
}

// ListDirectoryImagePaths returns all supported image paths directly inside dirPath sorted in natural order.
func ListDirectoryImagePaths(dirPath string) ([]string, error) {
	entries, err := os.ReadDir(dirPath)
	if err != nil {
		return nil, err
	}
	var names []string
	for _, e := range entries {
		if !e.IsDir() && IsSupportedImage(e.Name()) {
			names = append(names, e.Name())
		}
	}
	sort.Slice(names, func(i, j int) bool { return NaturalLess(names[i], names[j]) })
	paths := make([]string, len(names))
	for i, name := range names {
		paths[i] = filepath.Join(dirPath, name)
	}
	return paths, nil
}

// ReadInputPageFile reads and normalizes a single supported image file into an InputPage.
func ReadInputPageFile(filePath string) (InputPage, error) {
	if !IsSupportedImage(filePath) {
		return InputPage{}, fmt.Errorf("unsupported input file %q", filePath)
	}
	raw, err := os.ReadFile(filePath)
	if err != nil {
		return InputPage{}, err
	}
	pBytes, hash, w, h, err := NormalizeToPNG(raw)
	if err != nil {
		return InputPage{}, fmt.Errorf("%s: %w", filepath.Base(filePath), err)
	}
	return InputPage{
		Filename:   filepath.Base(filePath),
		PNGBytes:   pBytes,
		Blake3Hash: hash,
		Width:      w,
		Height:     h,
	}, nil
}

// ReadInputPages loads an image file, directory of images, or .cbz/.zip archive in natural reading order.
func ReadInputPages(inputPath string) ([]InputPage, error) {
	info, err := os.Stat(inputPath)
	if err != nil {
		return nil, err
	}
	if !info.IsDir() {
		ext := strings.ToLower(filepath.Ext(inputPath))
		if ext == ".cbz" || ext == ".zip" {
			return readZipPages(inputPath)
		}
		page, err := ReadInputPageFile(inputPath)
		if err != nil {
			return nil, err
		}
		return []InputPage{page}, nil
	}

	paths, err := ListDirectoryImagePaths(inputPath)
	if err != nil {
		return nil, err
	}
	pages := make([]InputPage, 0, len(paths))
	for _, p := range paths {
		page, err := ReadInputPageFile(p)
		if err != nil {
			return nil, err
		}
		pages = append(pages, page)
	}
	return pages, nil
}

func readZipPages(zipPath string) ([]InputPage, error) {
	zr, err := zip.OpenReader(zipPath)
	if err != nil {
		return nil, err
	}
	defer zr.Close()

	var files []*zip.File
	for _, f := range zr.File {
		if !f.FileInfo().IsDir() && !strings.HasPrefix(filepath.Base(f.Name), ".") && IsSupportedImage(f.Name) {
			files = append(files, f)
		}
	}
	sort.Slice(files, func(i, j int) bool { return NaturalLess(files[i].Name, files[j].Name) })

	pages := make([]InputPage, 0, len(files))
	for _, f := range files {
		rc, err := f.Open()
		if err != nil {
			return nil, err
		}
		raw, rErr := io.ReadAll(rc)
		_ = rc.Close()
		if rErr != nil {
			return nil, rErr
		}
		pBytes, hash, w, h, err := NormalizeToPNG(raw)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", f.Name, err)
		}
		pages = append(pages, InputPage{
			Filename:   filepath.Base(f.Name),
			PNGBytes:   pBytes,
			Blake3Hash: hash,
			Width:      w,
			Height:     h,
		})
	}
	return pages, nil
}

// BuildVirtualUnits groups consecutive same-width Webtoon slices into seamless Virtual Continuous Canvases
// whenever `shouldStitch(topPNG, bottomPNG)` indicates active cross-seam content, bounded by maxCanvasHeight.
func BuildVirtualUnits(
	pages []InputPage,
	maxCanvasHeight int,
	shouldStitch func(topPNG, bottomPNG []byte) bool,
) ([]VirtualUnit, error) {
	if maxCanvasHeight <= 0 {
		maxCanvasHeight = 7680
	}
	var units []VirtualUnit
	i := 0
	for i < len(pages) {
		group := []InputPage{pages[i]}
		totalH := pages[i].Height
		j := i + 1
		for j < len(pages) {
			prev := pages[j-1]
			curr := pages[j]
			if prev.Width != curr.Width || totalH+curr.Height > maxCanvasHeight {
				break
			}
			if shouldStitch == nil || !shouldStitch(prev.PNGBytes, curr.PNGBytes) {
				break
			}
			group = append(group, curr)
			totalH += curr.Height
			j++
		}

		if len(group) == 1 {
			units = append(units, VirtualUnit{
				Name:       group[0].Filename,
				PNGBytes:   group[0].PNGBytes,
				Blake3Hash: group[0].Blake3Hash,
				Width:      group[0].Width,
				Height:     group[0].Height,
			})
		} else {
			stitchedBytes, slices, err := StitchPagesPNG(group)
			if err != nil {
				return nil, err
			}
			sum := blake3.Sum256(stitchedBytes)
			units = append(units, VirtualUnit{
				Name:       fmt.Sprintf("%s__%s", strings.TrimSuffix(group[0].Filename, filepath.Ext(group[0].Filename)), group[len(group)-1].Filename),
				PNGBytes:   stitchedBytes,
				Blake3Hash: hex.EncodeToString(sum[:16]),
				Width:      group[0].Width,
				Height:     slices[len(slices)-1].YOffset + slices[len(slices)-1].Height,
				SeamSlices: slices,
			})
		}
		i = j
	}
	return units, nil
}

// StitchPagesPNG vertically concatenates a slice of same-width InputPages into one lossless PNG
// and returns the byte slice plus SeamSlice metadata for lossless re-slicing.
func StitchPagesPNG(group []InputPage) ([]byte, []dom.SeamSlice, error) {
	if len(group) == 0 {
		return nil, nil, fmt.Errorf("empty page group")
	}
	W := group[0].Width
	totalH := 0
	for _, sl := range group {
		if sl.Width != W {
			return nil, nil, fmt.Errorf("width mismatch in seam group: %d != %d", sl.Width, W)
		}
		totalH += sl.Height
	}
	canvas := image.NewNRGBA(image.Rect(0, 0, W, totalH))
	slices := make([]dom.SeamSlice, len(group))
	yOff := 0
	for idx, sl := range group {
		im, _, err := image.Decode(bytes.NewReader(sl.PNGBytes))
		if err != nil {
			return nil, nil, err
		}
		draw.Draw(canvas, image.Rect(0, yOff, W, yOff+sl.Height), im, im.Bounds().Min, draw.Src)
		slices[idx] = dom.SeamSlice{
			Index:    idx,
			YOffset:  yOff,
			Height:   sl.Height,
			Filename: sl.Filename,
		}
		yOff += sl.Height
	}
	var buf bytes.Buffer
	enc := png.Encoder{CompressionLevel: png.BestSpeed}
	if err := enc.Encode(&buf, canvas); err != nil {
		return nil, nil, err
	}
	return buf.Bytes(), slices, nil
}

type OutputEntry struct {
	Filename string
	Data     []byte
}

// EnsurePNGExt normalizes output entry filenames to end with `.png`.
func EnsurePNGExt(name string) string {
	dot := strings.LastIndexByte(name, '.')
	if dot > 0 {
		return name[:dot] + ".png"
	}
	return name + ".png"
}

// SliceCompositePNG re-slices a stitched composite PNG back into its constituent SeamSlice files.
// When slices is empty, it returns the single image under fallbackFilename.
func SliceCompositePNG(compositePNG []byte, slices []dom.SeamSlice, fallbackFilename string) ([]OutputEntry, error) {
	if len(slices) == 0 {
		return []OutputEntry{{
			Filename: EnsurePNGExt(fallbackFilename),
			Data:     compositePNG,
		}}, nil
	}
	img, _, err := image.Decode(bytes.NewReader(compositePNG))
	if err != nil {
		return nil, fmt.Errorf("decode composite PNG for slicing: %w", err)
	}
	bounds := img.Bounds()
	out := make([]OutputEntry, 0, len(slices))
	enc := png.Encoder{CompressionLevel: png.BestSpeed}
	for _, sl := range slices {
		y0 := bounds.Min.Y + sl.YOffset
		y1 := y0 + sl.Height
		if y0 < bounds.Min.Y {
			y0 = bounds.Min.Y
		}
		if y1 > bounds.Max.Y {
			y1 = bounds.Max.Y
		}
		if y1 <= y0 {
			return nil, fmt.Errorf("invalid seam slice bounds [%d,%d) for %s", y0, y1, sl.Filename)
		}
		rect := image.Rect(bounds.Min.X, y0, bounds.Max.X, y1)
		crop := image.NewNRGBA(image.Rect(0, 0, rect.Dx(), rect.Dy()))
		draw.Draw(crop, crop.Bounds(), img, rect.Min, draw.Src)
		var buf bytes.Buffer
		if err := enc.Encode(&buf, crop); err != nil {
			return nil, fmt.Errorf("encode seam slice %s: %w", sl.Filename, err)
		}
		out = append(out, OutputEntry{
			Filename: EnsurePNGExt(sl.Filename),
			Data:     buf.Bytes(),
		})
	}
	return out, nil
}

func WriteCBZ(outPath string, entries []OutputEntry) error {
	if err := os.MkdirAll(filepath.Dir(outPath), 0o755); err != nil {
		return err
	}
	f, err := os.Create(outPath)
	if err != nil {
		return err
	}
	defer f.Close()

	zw := zip.NewWriter(f)
	for _, e := range entries {
		hdr := &zip.FileHeader{
			Name:   filepath.ToSlash(e.Filename),
			Method: zip.Store, // PNGs are already compressed; Store enables zero-overhead random access in comic readers!
		}
		w, err := zw.CreateHeader(hdr)
		if err != nil {
			_ = zw.Close()
			return err
		}
		if _, err := w.Write(e.Data); err != nil {
			_ = zw.Close()
			return err
		}
	}
	return zw.Close()
}
