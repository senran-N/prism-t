package narrative

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"strings"
	"time"

	"prism-t/internal/dom"
	"prism-t/internal/store"
)

// SourceNodeRequest deliberately contains no translation, character persona,
// target language or narrative draft. Recognition must inspect source pixels.
type SourceNodeRequest struct {
	ID                   string              `json:"id"`
	BBox                 []int               `json:"bbox"`
	SourceLines          [][]int             `json:"source_lines,omitempty"`
	SourceQuadrilaterals [][][]float64       `json:"source_quadrilaterals,omitempty"`
	SourceReadings       []dom.SourceReading `json:"source_readings,omitempty"`
}

func sourceNodeRequests(nodes []NodeTranslationRequest) []SourceNodeRequest {
	requests := make([]SourceNodeRequest, len(nodes))
	for i, n := range nodes {
		requests[i] = SourceNodeRequest{n.ID, n.BBox, n.SourceLines, n.SourceQuadrilaterals, n.SourceReadings}
	}
	return requests
}

func sourceResultIssue(request SourceNodeRequest, result dom.SourceTranscription) string {
	if result.ID != request.ID || result.IsText == nil {
		return "source identity and explicit is_text decision are required"
	}
	if len(result.LineTexts) != max(1, len(request.SourceLines)) {
		return "line_texts must contain exactly one entry for each supplied source line (one bbox entry when no lines exist)"
	}
	if len(result.LineOrder) != len(result.LineTexts) {
		return "line_order must list every source line index exactly once"
	}
	seen := make([]bool, len(result.LineTexts))
	for _, index := range result.LineOrder {
		if index < 0 || index >= len(seen) || seen[index] {
			return "line_order must be a permutation of source line indices"
		}
		seen[index] = true
	}
	text := strings.TrimSpace(strings.Join(result.LineTexts, "\n"))
	if *result.IsText && text == "" {
		return "actual writing requires a nonempty transcription"
	}
	if !*result.IsText && (text != "" || strings.TrimSpace(result.Reason) == "") {
		return "non-writing requires empty line texts and a visual reason"
	}
	return ""
}

func transcribedSource(result dom.SourceTranscription) string {
	lines := make([]string, len(result.LineOrder))
	for i, index := range result.LineOrder {
		lines[i] = result.LineTexts[index]
	}
	return strings.TrimSpace(strings.Join(lines, "\n"))
}

func (c *Compiler) transcribeMegaNodes(ctx context.Context, st *store.Store, workID int64, chapter string, index int, page *dom.PageDOM, raw []byte, requests []NodeTranslationRequest) error {
	limit := c.cfg.LLMSourceLineBatchSize
	if limit <= 0 {
		return nil
	}
	var megaIndices []int
	var megaRequests []NodeTranslationRequest
	for i, req := range requests {
		if len(req.SourceLines) > limit {
			megaIndices = append(megaIndices, i)
			megaRequests = append(megaRequests, req)
		}
	}
	if len(megaRequests) == 0 {
		return nil
	}
	if err := c.transcribePage(ctx, st, workID, chapter, index, page, raw, megaRequests); err != nil {
		return err
	}
	for j, origIdx := range megaIndices {
		requests[origIdx] = megaRequests[j]
	}
	return nil
}

// Save successful source batches independently of translation. An upstream
// interruption must not force a second, potentially different reading of them.
func (c *Compiler) transcribePage(ctx context.Context, st *store.Store, workID int64, chapter string, index int, page *dom.PageDOM, raw []byte, requests []NodeTranslationRequest) error {
	sources := sourceNodeRequests(requests)
	contextBytes, err := json.Marshal(struct {
		Sources         []SourceNodeRequest
		Endpoint, Model string
		ReasoningEffort string
		LineBatchSize   int
		Image           [32]byte
		Implementation  [3][32]byte
	}{sources, c.cfg.LLMBaseURL, c.cfg.LLMModel, c.cfg.LLMReasoningEffort, c.cfg.LLMSourceLineBatchSize, sha256.Sum256(raw), [3][32]byte{sha256.Sum256(compilerImplementation), sha256.Sum256(transcriptionImplementation), sha256.Sum256(visionImplementation)}})
	if err != nil {
		return err
	}
	hash := fmt.Sprintf("%x", sha256.Sum256(contextBytes))
	if page.RecognitionProgress == nil || page.RecognitionProgress.ContextHash != hash {
		page.RecognitionProgress = &dom.RecognitionProgress{ContextHash: hash, Results: make(map[string]json.RawMessage), LineParts: make(map[string]map[int]dom.SourceTranscription)}
	}
	progress := page.RecognitionProgress
	results := make(map[string]dom.SourceTranscription, len(sources))
	var pending []SourceNodeRequest
	for _, request := range sources {
		if saved := progress.Results[request.ID]; len(saved) > 0 {
			var result dom.SourceTranscription
			if err := json.Unmarshal(saved, &result); err != nil {
				return err
			}
			if issue := sourceResultIssue(request, result); issue != "" {
				return fmt.Errorf("stored source %s: %s", request.ID, issue)
			}
			results[request.ID] = result
		} else {
			if c.cfg.LLMSourceLineBatchSize > 0 && len(request.SourceLines) > c.cfg.LLMSourceLineBatchSize {
				result, err := c.transcribePartitionedSource(ctx, request, progress, func() error {
					return st.SaveRecognitionProgress(ctx, workID, chapter, index, progress)
				})
				if err != nil {
					return err
				}
				results[request.ID] = result
			} else {
				pending = append(pending, request)
			}
		}
	}
	limit := c.cfg.LLMBatchSize
	if limit <= 0 {
		limit = max(1, len(pending))
	}
	for start := 0; start < len(pending); start += limit {
		batch := pending[start:min(start+limit, len(pending))]
		started := time.Now()
		log.Printf("[narrative] transcribing %d source nodes", len(batch))
		observed, err := c.backend.TranscribeNodes(ctx, batch)
		if err != nil {
			return err
		}
		expected := make(map[string]SourceNodeRequest, len(batch))
		for _, request := range batch {
			expected[request.ID] = request
		}
		seen := make(map[string]bool)
		for _, result := range observed {
			request, known := expected[result.ID]
			if !known || seen[result.ID] {
				return fmt.Errorf("source transcription returned unknown or duplicate node %q", result.ID)
			}
			seen[result.ID] = true
			if issue := sourceResultIssue(request, result); issue != "" {
				return fmt.Errorf("source transcription %s: %s", result.ID, issue)
			}
			results[result.ID] = result
			encoded, err := json.Marshal(result)
			if err != nil {
				return err
			}
			progress.Results[result.ID] = encoded
		}
		if err := st.SaveRecognitionProgress(ctx, workID, chapter, index, progress); err != nil {
			return err
		}
		if len(seen) != len(batch) {
			return fmt.Errorf("source transcription omitted %d nodes", len(batch)-len(seen))
		}
		log.Printf("[narrative] transcribed %d source nodes in %s", len(batch), time.Since(started).Round(time.Millisecond))
	}
	for i := range requests {
		result := results[requests[i].ID]
		if *result.IsText {
			requests[i].SourceText = transcribedSource(result)
			requests[i].SourceTextLocked = true
			requests[i].Phonology = ExtractPhonologicalFeatures(requests[i].SourceText)
		}
	}
	return nil
}

func sourceLinePart(source SourceNodeRequest, start, end int) (SourceNodeRequest, error) {
	part := SourceNodeRequest{ID: fmt.Sprintf("%s/lines/%d", source.ID, start), BBox: source.BBox,
		SourceLines: source.SourceLines[start:end]}
	if len(source.SourceQuadrilaterals) > 0 {
		if len(source.SourceQuadrilaterals) != len(source.SourceLines) {
			return part, fmt.Errorf("source %s quadrilaterals do not match owned lines", source.ID)
		}
		part.SourceQuadrilaterals = source.SourceQuadrilaterals[start:end]
	}
	for _, reading := range source.SourceReadings {
		if reading.LineIndex >= start && reading.LineIndex < end {
			reading.LineIndex -= start
			part.SourceReadings = append(part.SourceReadings, reading)
		}
	}
	return part, nil
}

// Read all owned lines in bounded inference tasks, checkpoint each real part,
// then ask only for global reading order. Ordering cannot rewrite any glyphs.
func (c *Compiler) transcribePartitionedSource(ctx context.Context, request SourceNodeRequest, progress *dom.RecognitionProgress, save func() error) (dom.SourceTranscription, error) {
	if progress.LineParts == nil {
		progress.LineParts = make(map[string]map[int]dom.SourceTranscription)
	}
	parts := progress.LineParts[request.ID]
	if parts == nil {
		parts = make(map[int]dom.SourceTranscription)
		progress.LineParts[request.ID] = parts
	}
	isText := false
	result := dom.SourceTranscription{ID: request.ID, IsText: &isText, LineTexts: make([]string, len(request.SourceLines))}
	var reasons []string
	for start := 0; start < len(request.SourceLines); start += c.cfg.LLMSourceLineBatchSize {
		end := min(start+c.cfg.LLMSourceLineBatchSize, len(request.SourceLines))
		part, err := sourceLinePart(request, start, end)
		if err != nil {
			return result, err
		}
		observed, saved := parts[start]
		if !saved {
			log.Printf("[narrative] transcribing source %s lines [%d,%d) of %d", request.ID, start, end, len(request.SourceLines))
			response, err := c.backend.TranscribeNodes(ctx, []SourceNodeRequest{part})
			if err != nil {
				return result, fmt.Errorf("source %s lines [%d,%d): %w", request.ID, start, end, err)
			}
			if len(response) != 1 {
				return result, fmt.Errorf("source %s part returned %d nodes, expected one", request.ID, len(response))
			}
			observed = response[0]
		}
		if issue := sourceResultIssue(part, observed); issue != "" {
			return result, fmt.Errorf("source %s lines [%d,%d): %s", request.ID, start, end, issue)
		}
		if !saved {
			parts[start] = observed
			if err := save(); err != nil {
				return result, err
			}
		}
		copy(result.LineTexts[start:end], observed.LineTexts)
		isText = isText || *observed.IsText
		if observed.Reason != "" {
			reasons = append(reasons, fmt.Sprintf("lines [%d,%d): %s", start, end, observed.Reason))
		}
	}
	log.Printf("[narrative] ordering all %d locked source lines for %s", len(result.LineTexts), request.ID)
	order, err := c.backend.OrderSourceLines(ctx, request, result.LineTexts)
	if err != nil {
		return result, fmt.Errorf("source %s global order: %w", request.ID, err)
	}
	if order.ID != request.ID {
		return result, fmt.Errorf("source order returned wrong node ID %s", order.ID)
	}
	result.LineOrder = order.LineOrder
	result.Reason = strings.Join(append(reasons, order.Reason), "\n")
	if issue := sourceResultIssue(request, result); issue != "" {
		return result, fmt.Errorf("assembled source %s: %s", request.ID, issue)
	}
	encoded, err := json.Marshal(result)
	if err != nil {
		return result, err
	}
	progress.Results[request.ID] = encoded
	if err := save(); err != nil {
		return result, err
	}
	return result, nil
}

type SourceLineOrder struct {
	ID        string `json:"id"`
	LineOrder []int  `json:"line_order"`
	Reason    string `json:"reason"`
}

func (b *OpenAICompatibleBackend) OrderSourceLines(ctx context.Context, request SourceNodeRequest, lineTexts []string) (SourceLineOrder, error) {
	raw, ok := ctx.Value(pageImageKey{}).([]byte)
	if !ok || len(raw) == 0 {
		return SourceLineOrder{}, errors.New("source ordering requires original page pixels")
	}
	source, err := decodePageImage(ctx, raw)
	if err != nil {
		return SourceLineOrder{}, err
	}
	geometry := []NodeTranslationRequest{{ID: request.ID, BBox: request.BBox, SourceLines: request.SourceLines}}
	overviewJPEG, err := buildSceneOverviewJPEG(source, geometry)
	if err != nil {
		return SourceLineOrder{}, err
	}
	payload, err := json.Marshal(map[string]any{"node": request, "locked_line_texts": lineTexts})
	if err != nil {
		return SourceLineOrder{}, err
	}
	content := []map[string]any{
		{"type": "text", "text": string(payload)},
		{"type": "image_url", "image_url": map[string]string{"url": "data:image/jpeg;base64," + base64.StdEncoding.EncodeToString(overviewJPEG), "detail": "auto"}},
	}
	crops, err := nodeImageEvidenceFromImage(source, geometry)
	if err != nil {
		return SourceLineOrder{}, err
	}
	content = append(content, crops...)
	count := len(lineTexts)
	format := map[string]any{"type": "json_schema", "json_schema": map[string]any{
		"name": "comic_source_reading_order", "strict": true,
		"schema": map[string]any{"type": "object", "additionalProperties": false,
			"required": []string{"id", "line_order", "reason"},
			"properties": map[string]any{
				"id":         map[string]any{"type": "string", "enum": []string{request.ID}},
				"line_order": map[string]any{"type": "array", "items": map[string]any{"type": "integer", "minimum": 0, "maximum": count - 1}, "minItems": count, "maxItems": count},
				"reason":     map[string]string{"type": "string"},
			},
		},
	}}
	prompt := `Determine only the global visual reading order of the supplied comic source lines. Source glyphs were independently transcribed in bounded groups, with original indices preserved. Do not translate, transcribe again, rewrite, omit or generate any line text. Use the complete source crop to resolve paragraph groups, titles, captions, columns and insets. Atlas packing and the detector's index order are not reading order. Return a permutation of EVERY zero-based source line_index, including empty/art-only/duplicate entries. locked_line_texts are immutable observations, not a new transcription task. Ground the order in actual page geometry and visible writing, preserving paragraph continuity. The reason should briefly explain the spatial grouping.`
	response, err := b.completeJSON(ctx, prompt, content, format, request)
	if err != nil {
		return SourceLineOrder{}, err
	}
	var result SourceLineOrder
	err = json.Unmarshal([]byte(extractJSONObject(response)), &result)
	return result, err
}

type keyedSources map[string]dom.SourceTranscription

func (s *keyedSources) UnmarshalJSON(raw []byte) error {
	results, err := decodeKeyedObject(raw, "sources", func(result dom.SourceTranscription) string { return result.ID })
	*s = results
	return err
}

func (b *OpenAICompatibleBackend) TranscribeNodes(ctx context.Context, requests []SourceNodeRequest) ([]dom.SourceTranscription, error) {
	raw, ok := ctx.Value(pageImageKey{}).([]byte)
	if !ok || len(raw) == 0 {
		return nil, errors.New("visual transcription requires original page pixels")
	}
	source, err := decodePageImage(ctx, raw)
	if err != nil {
		return nil, err
	}
	geometry := make([]NodeTranslationRequest, len(requests))
	for i, r := range requests {
		geometry[i] = NodeTranslationRequest{ID: r.ID, BBox: r.BBox, SourceLines: r.SourceLines}
	}
	overviewJPEG, err := buildSceneOverviewJPEG(source, geometry)
	if err != nil {
		return nil, err
	}
	payload, err := json.Marshal(map[string]any{"nodes": requests})
	if err != nil {
		return nil, err
	}
	content := []map[string]any{
		{"type": "text", "text": string(payload)},
		{"type": "image_url", "image_url": map[string]string{"url": "data:image/jpeg;base64," + base64.StdEncoding.EncodeToString(overviewJPEG), "detail": "auto"}},
	}
	crops, err := nodeImageEvidenceFromImage(source, geometry)
	if err != nil {
		return nil, err
	}
	content = append(content, crops...)
	prompt := `You are a source-only comic transcription operator. Read visible glyphs, never translate, explain, complete a story, or rewrite wording into fluent prose.
Return the sources object keyed by every supplied node ID, each with matching id, explicit is_text, line_texts, line_order and reason.
line_texts has exactly one entry per zero-based source line_index, in the supplied order; if no source_lines exist, return one entry for the bbox. Empty entries are allowed for regions without writing. A false is_text requires all entries empty and a visual reason. Any writing including labels, numbers, small notes and sound effects counts as text.
The detector's line_index order can be wrong. Keep line_texts indexed by the original supplied index so evidence remains traceable. Separately return line_order: a permutation of all these indices in the actual visual reading order (top to bottom for horizontal rows, right to left for Japanese columns, with paragraph groups resolved from the complete crop). Do not reorder line_texts themselves. Include indices of empty duplicate/art-only entries too; they contain no glyphs to translate.
source_lines and source_quadrilaterals define owned writing. bbox includes context: do not transcribe neighboring nodes. For overlapping owned crops record each visible glyph once, leaving duplicate-only entries empty. Inspect the complete crop to distinguish separate columns and punctuation from illustration.
source_readings are fallible independent OCR observations, not an answer key. Resolve disagreements using the native line pixels. Null confidence is unreported probability, not certainty; truncated means the observation is incomplete. Do not copy an OCR hallucination from a decorative mark. Preserve visible punctuation, anonymous symbols, small kana, digits and exact negation. Never add conventional endings, implied laughter, names, or explanatory words absent from the image. Do not silently drop difficult visible writing. This pass has no target language or translation draft.`
	response, err := b.completeJSON(ctx, prompt, content, sourceResponseFormat(requests), requests)
	if err != nil {
		return nil, err
	}
	var batch struct {
		Sources *keyedSources `json:"sources"`
	}
	if err := json.Unmarshal([]byte(extractJSONObject(response)), &batch); err != nil {
		return nil, fmt.Errorf("invalid source JSON: %w", err)
	}
	if batch.Sources == nil {
		return nil, errors.New("required sources object is missing or null")
	}
	results := make([]dom.SourceTranscription, 0, len(*batch.Sources))
	for _, result := range *batch.Sources {
		results = append(results, result)
	}
	return results, nil
}

func sourceResponseFormat(requests []SourceNodeRequest) map[string]any {
	properties := make(map[string]any, len(requests))
	ids := make([]string, len(requests))
	for i, r := range requests {
		ids[i] = r.ID
		count := max(1, len(r.SourceLines))
		properties[r.ID] = map[string]any{
			"type": "object", "additionalProperties": false,
			"required": []string{"id", "is_text", "line_texts", "line_order", "reason"},
			"properties": map[string]any{
				"id":         map[string]any{"type": "string", "enum": []string{r.ID}},
				"is_text":    map[string]string{"type": "boolean"},
				"reason":     map[string]string{"type": "string"},
				"line_texts": map[string]any{"type": "array", "items": map[string]string{"type": "string"}, "minItems": count, "maxItems": count},
				"line_order": map[string]any{"type": "array", "items": map[string]any{"type": "integer", "minimum": 0, "maximum": count - 1}, "minItems": count, "maxItems": count},
			},
		}
	}
	return map[string]any{"type": "json_schema", "json_schema": map[string]any{
		"name": "comic_source_transcription", "strict": true,
		"schema": map[string]any{"type": "object", "additionalProperties": false, "required": []string{"sources"},
			"properties": map[string]any{"sources": map[string]any{"type": "object", "additionalProperties": false, "required": ids, "properties": properties}},
		},
	}}
}
