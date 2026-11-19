package narrative

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"prism-t/internal/dom"
)

func TestSourceAtlasKeepsEveryNativeLineAndIdentity(t *testing.T) {
	raw := image.NewNRGBA(image.Rect(0, 0, 100, 70))
	for y := 0; y < 70; y++ {
		for x := 0; x < 100; x++ {
			raw.SetNRGBA(x, y, color.NRGBA{uint8(x), uint8(y), uint8(x + y), 255})
		}
	}
	var encoded bytes.Buffer
	png.Encode(&encoded, raw)
	for _, count := range []int{0, 1, 2, 53} {
		t.Run(fmt.Sprint(count), func(t *testing.T) {
			lines := [][]int{{8, 12, 70, 5}, {8, 30, 65, 7}}
			for len(lines) < count {
				i := len(lines)
				lines = append(lines, []int{i % 20, i % 30, 1 + i%70, 1 + i%40})
			}
			lines = lines[:count]
			nodes := []NodeTranslationRequest{{ID: "tiny", BBox: []int{5, 10, 85, 45}, SourceLines: lines}}
			items, err := nodeImageEvidence(encoded.Bytes(), nodes)
			expected := 2
			if count > 0 {
				expected = 4
			}
			if err != nil || len(items) != expected {
				t.Fatalf("unexpected image inputs: %d, %v", len(items), err)
			}
			var images []image.Image
			for index := 1; index < len(items); index += 2 {
				item := items[index]["image_url"].(map[string]string)
				if item["detail"] != "high" {
					t.Fatal("source evidence must request high image detail")
				}
				decoded, err := base64.StdEncoding.DecodeString(strings.TrimPrefix(item["url"], "data:image/png;base64,"))
				if err != nil {
					t.Fatal(err)
				}
				img, err := png.Decode(bytes.NewReader(decoded))
				if err != nil {
					t.Fatal(err)
				}
				images = append(images, img)
			}
			checkPixels := func(img image.Image, rect, sourceRect [4]int) {
				t.Helper()
				if rect[2] != sourceRect[2] || rect[3] != sourceRect[3] {
					t.Fatal("native evidence was resampled")
				}
				for y := 0; y < sourceRect[3]; y++ {
					for x := 0; x < sourceRect[2]; x++ {
						if color.NRGBAModel.Convert(img.At(x+rect[0], y+rect[1])) != raw.NRGBAAt(x+sourceRect[0], y+sourceRect[1]) {
							t.Fatal("source evidence pixels changed or were overwritten by headers")
						}
					}
				}
			}
			checkPixels(images[0], [4]int{0, 0, 85, 45}, [4]int{5, 10, 85, 45})
			if count == 0 {
				return
			}
			label := items[2]["text"].(string)
			_, metadata, ok := strings.Cut(label, "Tiles: ")
			if !ok || !strings.Contains(label, "not source writing") || !strings.Contains(label, "not reading order") {
				t.Fatal("atlas ownership instructions are missing")
			}
			var tiles []struct {
				Index  int    `json:"line_index"`
				Source [4]int `json:"source_rect"`
				Atlas  [4]int `json:"atlas_rect"`
			}
			if err := json.Unmarshal([]byte(metadata), &tiles); err != nil || len(tiles) != count {
				t.Fatalf("missing source lines: %d, %v", len(tiles), err)
			}
			for index, tile := range tiles {
				b := lines[index]
				if tile.Index != index || tile.Source != [4]int{b[0], b[1], b[2], b[3]} {
					t.Fatal("line identity changed")
				}
				checkPixels(images[1], tile.Atlas, tile.Source)
			}
		})
	}
}

func TestBackendReceivesIndependentReadingsWithoutChoosingAnOCRWinner(t *testing.T) {
	probability := .9
	readings := []dom.SourceReading{{LineIndex: 0, Recognizer: "manga_ocr_nar_small", Text: "A"}, {LineIndex: 0, Recognizer: "rapidocr_ctc", Text: "B", Confidence: &probability}, {LineIndex: 1, Recognizer: "manga_ocr_nar_small", Text: "", Truncated: true}}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			Messages []struct {
				Content string `json:"content"`
			} `json:"messages"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatal(err)
		}
		var payload struct {
			Nodes []NodeTranslationRequest `json:"nodes"`
		}
		if err := json.Unmarshal([]byte(request.Messages[1].Content), &payload); err != nil {
			t.Fatal(err)
		}
		if len(payload.Nodes) != 1 || len(payload.Nodes[0].SourceReadings) != 3 || payload.Nodes[0].SourceReadings[2].Text != "" || payload.Nodes[0].SourceText != "" {
			t.Fatal("observations were filtered or promoted into source truth")
		}
		if payload.Nodes[0].SourceReadings[0].Confidence != nil || !payload.Nodes[0].SourceReadings[2].Truncated {
			t.Fatal("missing confidence or truncation state was fabricated")
		}
		if !strings.Contains(request.Messages[0].Content, "cannot be compared across models") {
			t.Fatal("missing evidence interpretation")
		}
		writeChatStream(w, `{"translations":{"n":{"id":"n","is_text":true,"source_text":"A","target_segments":["甲"],"reason":"","rubies":[]}}}`)
	}))
	defer server.Close()
	b := &OpenAICompatibleBackend{baseURL: server.URL, httpClient: server.Client(), diagnosticsPath: t.TempDir()}
	if _, err := b.TranslateNodes(context.Background(), "zh-CN", nil, nil, []NodeTranslationRequest{{ID: "n", SourceReadings: readings}}); err != nil {
		t.Fatal(err)
	}
}
