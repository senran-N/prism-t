package narrative

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	_ "embed"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode"
	"unicode/utf8"

	"golang.org/x/image/font"
	"golang.org/x/image/font/basicfont"
	"golang.org/x/image/math/fixed"

	"prism-t/internal/config"
	"prism-t/internal/dom"
	"prism-t/internal/store"
)

var ErrNoTranslationBackend = errors.New("no LLM translation endpoint configured (set PRISM_T_LLM_BASE_URL and PRISM_T_LLM_API_KEY, or pass --analyze-only)")

type pageImageKey struct{}
type pageDraftKey struct{}
type translationCheckpointKey struct{}
type reasoningEffortOverrideKey struct{}

//go:embed compiler.go
var compilerImplementation []byte

//go:embed fidelity.go
var fidelityImplementation []byte

//go:embed transcription.go
var transcriptionImplementation []byte

//go:embed vision.go
var visionImplementation []byte

type translationCheckpoint struct {
	progress *dom.TranslationProgress
	save     func() error
}

// PhonologicalFeatures captures script-level acoustic morphology (elongation, geminate glottal stops,
// reduplication) without any hardcoded vocabulary table, enabling proportional expressive rendering.
type PhonologicalFeatures struct {
	ElongationCount int  `json:"elongation_count,omitempty"`
	GlottalStop     bool `json:"glottal_stop,omitempty"`
	Reduplicated    bool `json:"reduplicated,omitempty"`
}

func ExtractPhonologicalFeatures(text string) PhonologicalFeatures {
	runes := []rune(strings.TrimSpace(text))
	var feat PhonologicalFeatures
	if len(runes) == 0 {
		return feat
	}
	for i, r := range runes {
		if r == 'ー' || r == '〜' || r == '～' || r == '—' {
			feat.ElongationCount++
		} else if i > 0 && runes[i-1] == r && unicode.In(r, unicode.Hiragana, unicode.Katakana, unicode.Latin) {
			feat.ElongationCount++
		}
		if r == 'っ' || r == 'ッ' {
			feat.GlottalStop = true
		}
	}
	// Detect ABAB / AA reduplication structure common in mimetic/ideophonic expressions
	clean := make([]rune, 0, len(runes))
	for _, r := range runes {
		if !unicode.IsPunct(r) && !unicode.IsSpace(r) {
			clean = append(clean, r)
		}
	}
	if len(clean) >= 4 && len(clean)%2 == 0 {
		half := len(clean) / 2
		if string(clean[:half]) == string(clean[half:]) {
			feat.Reduplicated = true
		}
	}
	return feat
}

type NodeTranslationRequest struct {
	ID                   string               `json:"id"`
	ReadingOrder         int                  `json:"reading_order"`
	Track                string               `json:"track"`
	BubbleSemantics      string               `json:"bubble_semantics"`
	Speaker              string               `json:"speaker,omitempty"`
	Addressee            string               `json:"addressee,omitempty"`
	FirstPersonRule      string               `json:"first_person_rule,omitempty"`
	SpeechStyleRule      string               `json:"speech_style_rule,omitempty"`
	RelationHonorific    string               `json:"relation_honorific,omitempty"`
	DiscourseHint        string               `json:"discourse_hint,omitempty"`
	SourceText           string               `json:"source_text"`
	SourceTextLocked     bool                 `json:"source_text_locked"`
	DraftTranslation     string               `json:"draft_translation,omitempty"`
	DraftIsText          *bool                `json:"draft_is_text,omitempty"`
	DraftReason          string               `json:"draft_reason,omitempty"`
	Correction           string               `json:"correction,omitempty"`
	BBox                 []int                `json:"bbox"`
	SourceLines          [][]int              `json:"source_lines,omitempty"`
	SourceQuadrilaterals [][][]float64        `json:"source_quadrilaterals,omitempty"`
	SourceReadings       []dom.SourceReading  `json:"source_readings,omitempty"`
	RubyPairs            []dom.RubyPair       `json:"ruby_pairs,omitempty"`
	Phonology            PhonologicalFeatures `json:"phonology,omitempty"`
	CharBudgetMin        int                  `json:"char_budget_min"`
	CharBudgetMax        int                  `json:"char_budget_max"`
}

type RubyTranslation struct {
	TargetBase string `json:"target_base"`
	TargetRuby string `json:"target_ruby"`
}

type NodeTranslationResult struct {
	ID             string            `json:"id"`
	SourceText     string            `json:"source_text,omitempty"`
	IsText         *bool             `json:"is_text,omitempty"`
	Reason         string            `json:"reason,omitempty"`
	TargetText     string            `json:"target_text"`
	TargetSegments []string          `json:"target_segments"`
	Rubies         []RubyTranslation `json:"rubies,omitempty"`
}

// Response keys own node identity. Reject repeated keys instead of allowing
// encoding/json's last-value-wins rule to silently discard a translation.
type keyedTranslations map[string]NodeTranslationResult

func (results *keyedTranslations) UnmarshalJSON(raw []byte) error {
	type generatedTranslation struct {
		ID             string            `json:"id"`
		SourceText     string            `json:"source_text"`
		IsText         *bool             `json:"is_text"`
		Reason         string            `json:"reason"`
		TargetSegments []string          `json:"target_segments"`
		Rubies         []RubyTranslation `json:"rubies"`
	}
	parsed, err := decodeKeyedObject(raw, "translations", func(result generatedTranslation) string { return result.ID })
	if err != nil {
		return err
	}
	*results = make(keyedTranslations, len(parsed))
	for id, generated := range parsed {
		(*results)[id] = NodeTranslationResult{ID: id, SourceText: generated.SourceText, IsText: generated.IsText, Reason: generated.Reason,
			TargetText: strings.Join(generated.TargetSegments, ""), TargetSegments: generated.TargetSegments, Rubies: generated.Rubies}
	}
	return err
}

func decodeKeyedObject[T any](raw []byte, name string, identity func(T) string) (map[string]T, error) {
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	first, err := decoder.Token()
	if err != nil || first != json.Delim('{') {
		return nil, fmt.Errorf("%s must be an object keyed by node ID", name)
	}
	results := make(map[string]T)
	for decoder.More() {
		key, err := decoder.Token()
		if err != nil {
			return nil, err
		}
		id := key.(string)
		if _, duplicate := results[id]; duplicate {
			return nil, fmt.Errorf("%s returned duplicate key %q", name, id)
		}
		var result T
		if err := decoder.Decode(&result); err != nil {
			return nil, err
		}
		if identity(result) != id {
			return nil, fmt.Errorf("%s key %q disagrees with node ID %q", name, id, identity(result))
		}
		results[id] = result
	}
	_, err = decoder.Token()
	return results, err
}

// Backend abstracts the neural sequence localization model so tests and custom local engines
// plug in cleanly without ever polluting production code with hardcoded dictionaries.
type Backend interface {
	TranscribeNodes(context.Context, []SourceNodeRequest) ([]dom.SourceTranscription, error)
	OrderSourceLines(context.Context, SourceNodeRequest, []string) (SourceLineOrder, error)
	TranslateNodes(
		ctx context.Context,
		targetLang string,
		priorTurns []store.DialogueTurn,
		matchedGlossary []store.GlossaryTerm,
		nodes []NodeTranslationRequest,
	) ([]NodeTranslationResult, error)
}

type Compiler struct {
	mu            sync.RWMutex
	cfg           config.Config
	backend       Backend
	customBackend bool
}

func NewCompiler(cfg config.Config) *Compiler {
	c := &Compiler{cfg: cfg}
	if strings.TrimSpace(cfg.LLMBaseURL) != "" {
		c.backend = NewOpenAICompatibleBackend(cfg)
	}
	return c
}

func NewCompilerWithBackend(cfg config.Config, backend Backend) *Compiler {
	return &Compiler{
		cfg:           cfg,
		backend:       backend,
		customBackend: backend != nil,
	}
}

func (c *Compiler) Config() config.Config {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.cfg
}

func (c *Compiler) UpdateConfig(cfg config.Config) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.cfg = cfg
	if c.customBackend {
		return
	}
	if strings.TrimSpace(cfg.LLMBaseURL) != "" {
		c.backend = NewOpenAICompatibleBackend(cfg)
	} else {
		c.backend = nil
	}
}

func (c *Compiler) HasBackend() bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.backend != nil
}

func (c *Compiler) snapshot() *Compiler {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return &Compiler{
		cfg:           c.cfg,
		backend:       c.backend,
		customBackend: c.customBackend,
	}
}

// CompilePageTranslations grounds page speaker clusters to persistent Work Character entities via 64D
// cosine Re-ID, injects sliding cross-page chapter dialogue history + directed speaker->addressee
// honorific relations + relevant glossary hits, and enforces SDF-derived character budgets [N_min, N_max].
func (c *Compiler) CompilePageTranslations(
	ctx context.Context,
	st *store.Store,
	workID int64,
	chapterTitle string,
	pageIndex int,
	targetLang string,
	pageDOM *dom.PageDOM,
	rawPNG []byte,
) error {
	c = c.snapshot()
	if c.backend == nil {
		return ErrNoTranslationBackend
	}
	if len(pageDOM.Nodes) == 0 {
		return nil
	}
	if c.cfg.LLMVision && len(rawPNG) > 0 {
		ctx = context.WithValue(ctx, pageImageKey{}, rawPNG)
		if decoded, _, decErr := image.Decode(bytes.NewReader(rawPNG)); decErr == nil {
			ctx = context.WithValue(ctx, parsedPageImageKey{}, decoded)
		}
	}
	if targetLang == "" {
		targetLang = c.cfg.TargetLang
	}

	// 1. Ground page-local speaker visual vectors against persistent Work Characters in SQLite
	clusterToChar := make(map[string]store.Character)
	charByClusterKey := make(map[string]store.Character)
	existingChars, err := st.ListCharacters(ctx, workID)
	if err != nil {
		return fmt.Errorf("list characters: %w", err)
	}
	for _, ec := range existingChars {
		charByClusterKey[ec.ClusterKey] = ec
	}

	for i := range pageDOM.Nodes {
		n := &pageDOM.Nodes[i]
		if n.SpeakerAppearanceB64 != "" {
			if cached, ok := clusterToChar[n.SpeakerClusterID]; ok {
				n.SpeakerClusterID = cached.ClusterKey
			} else {
				matched, mErr := st.MatchOrRegisterSpeaker(ctx, workID, n.SpeakerAppearanceB64)
				if mErr != nil {
					return fmt.Errorf("match/register speaker: %w", mErr)
				}
				clusterToChar[n.SpeakerClusterID] = matched
				charByClusterKey[matched.ClusterKey] = matched
				n.SpeakerClusterID = matched.ClusterKey
			}
		}
	}

	// 2. Fetch cross-page sliding dialogue history, directed relation matrix, and active glossary subset
	priorTurns, err := st.GetRecentChapterDialogue(ctx, workID, chapterTitle, pageIndex, 16)
	if err != nil {
		return fmt.Errorf("load prior dialogue: %w", err)
	}
	allGlossary, err := st.ListGlossary(ctx, workID)
	if err != nil {
		return fmt.Errorf("load glossary: %w", err)
	}
	relations, err := st.ListRelations(ctx, workID)
	if err != nil {
		return fmt.Errorf("load character relations: %w", err)
	}

	relMap := make(map[[2]int64]store.CharacterRelation, len(relations))
	for _, r := range relations {
		relMap[[2]int64{r.FromCharID, r.ToCharID}] = r
	}

	// Filter glossary strictly to terms appearing in current page source text or rubies (context-window hygiene)
	var pageCorpus strings.Builder
	for _, n := range pageDOM.Nodes {
		pageCorpus.WriteString(n.SourceText)
		pageCorpus.WriteByte('\n')
		for _, rp := range n.RubyPairs {
			pageCorpus.WriteString(rp.Base)
			pageCorpus.WriteString(rp.Ruby)
		}
	}
	corpusLower := strings.ToLower(pageCorpus.String())
	var matchedGlossary []store.GlossaryTerm
	pendingRecognition := false
	for _, node := range pageDOM.Nodes {
		pendingRecognition = pendingRecognition || node.RecognitionPending
	}
	for _, g := range allGlossary {
		if g.SourceTerm != "" && (pendingRecognition || strings.Contains(corpusLower, strings.ToLower(g.SourceTerm))) {
			matchedGlossary = append(matchedGlossary, g)
		}
	}

	// 3. Build structured narrative nodes with inferred conversational addressee within adjacent primary turns
	var requests []NodeTranslationRequest
	for i, n := range pageDOM.Nodes {
		if strings.TrimSpace(n.SourceText) == "" && !n.RecognitionPending {
			continue
		}
		req := NodeTranslationRequest{
			ID:                   n.ID,
			ReadingOrder:         n.ReadingOrder,
			Track:                n.Track,
			BubbleSemantics:      n.BubbleKind,
			SourceText:           n.SourceText,
			SourceLines:          n.SourceLines,
			SourceQuadrilaterals: n.SourceQuadrilaterals,
			SourceReadings:       n.SourceReadings,
			BBox:                 n.BBox,
			RubyPairs:            n.RubyPairs,
			Phonology:            ExtractPhonologicalFeatures(n.SourceText),
			CharBudgetMin:        n.CharBudgetMin,
			CharBudgetMax:        n.CharBudgetMax,
		}

		if spk, ok := charByClusterKey[n.SpeakerClusterID]; ok {
			req.Speaker = spk.NameTarget
			req.FirstPersonRule = spk.FirstPerson
			req.SpeechStyleRule = spk.SpeechStyle

			// Infer conversational addressee from nearest preceding/succeeding primary node with a different speaker
			var addrChar *store.Character
			for delta := 1; delta <= 2 && addrChar == nil; delta++ {
				if i-delta >= 0 {
					prevKey := pageDOM.Nodes[i-delta].SpeakerClusterID
					if prevKey != "" && prevKey != n.SpeakerClusterID {
						if ac, ok2 := charByClusterKey[prevKey]; ok2 {
							addrChar = &ac
							break
						}
					}
				}
				if i+delta < len(pageDOM.Nodes) {
					nextKey := pageDOM.Nodes[i+delta].SpeakerClusterID
					if nextKey != "" && nextKey != n.SpeakerClusterID {
						if ac, ok2 := charByClusterKey[nextKey]; ok2 {
							addrChar = &ac
							break
						}
					}
				}
			}
			if addrChar != nil {
				req.Addressee = addrChar.NameTarget
				if rel, rOk := relMap[[2]int64{spk.ID, addrChar.ID}]; rOk {
					req.RelationHonorific = strings.TrimSpace(rel.HonorificTarget + " " + rel.RegisterNote)
				}
			}
		}
		requests = append(requests, req)
	}

	if len(requests) == 0 {
		return nil
	}
	if c.cfg.LLMVision && len(rawPNG) > 0 {
		if c.cfg.LLMSourcePrepass {
			if err := c.transcribePage(ctx, st, workID, chapterTitle, pageIndex, pageDOM, rawPNG, requests); err != nil {
				return fmt.Errorf("source transcription failed: %w", err)
			}
		} else {
			if err := c.transcribeMegaNodes(ctx, st, workID, chapterTitle, pageIndex, pageDOM, rawPNG, requests); err != nil {
				return fmt.Errorf("mega-node source transcription failed: %w", err)
			}
		}
	}
	annotateDiscourseTopology(requests)
	// Keep only actual, validated backend outputs. The exact input context and
	// compiler implementation bind interrupted batches to this page/run; a new
	// model, image, glossary, context or algorithm cannot inherit stale answers.
	contextBytes, err := json.Marshal(struct {
		Requests                  []NodeTranslationRequest
		Turns                     []store.DialogueTurn
		Glossary                  []store.GlossaryTerm
		Language, Endpoint, Model string
		ReasoningEffort           string
		Vision                    bool
		PreserveArtText           bool
		Image                     [32]byte
		Implementation            [4][32]byte
	}{requests, priorTurns, matchedGlossary, targetLang, c.cfg.LLMBaseURL, c.cfg.LLMModel, c.cfg.LLMReasoningEffort, c.cfg.LLMVision, c.cfg.PreserveArtText, sha256.Sum256(rawPNG), [4][32]byte{sha256.Sum256(compilerImplementation), sha256.Sum256(fidelityImplementation), sha256.Sum256(transcriptionImplementation), sha256.Sum256(visionImplementation)}})
	if err != nil {
		return err
	}
	contextHash := fmt.Sprintf("%x", sha256.Sum256(contextBytes))
	if pageDOM.TranslationProgress == nil || pageDOM.TranslationProgress.ContextHash != contextHash {
		pageDOM.TranslationProgress = &dom.TranslationProgress{ContextHash: contextHash, Results: make(map[string]json.RawMessage)}
	}
	checkpoint := &translationCheckpoint{progress: pageDOM.TranslationProgress}
	checkpoint.save = func() error {
		return st.SaveTranslationProgress(ctx, workID, chapterTitle, pageIndex, checkpoint.progress)
	}
	ctx = context.WithValue(ctx, translationCheckpointKey{}, checkpoint)

	// 4. Holistic structured neural compilation
	results, err := c.translateComplete(ctx, targetLang, priorTurns, matchedGlossary, requests)
	if err != nil {
		return fmt.Errorf("translation backend failed: %w", err)
	}

	resByID := make(map[string]NodeTranslationResult, len(results))
	for _, res := range results {
		resByID[res.ID] = res
	}

	// Optional second-pass image-grounded audit when explicitly enabled.
	if c.cfg.LLMVision && c.cfg.LLMVisionAudit && len(rawPNG) > 0 {
		var auditRequests []NodeTranslationRequest
		for _, request := range requests {
			result := resByID[request.ID]
			if result.SourceText != "" {
				request.SourceText = result.SourceText
			}
			request.DraftTranslation = result.TargetText
			request.DraftIsText = result.IsText
			request.DraftReason = result.Reason
			auditRequests = append(auditRequests, request)
		}
		if len(auditRequests) > 0 {
			annotateDiscourseTopology(auditRequests)
			// Every audit batch sees the complete actual page draft. Otherwise
			// names, references and split sentences can drift across batches.
			ctx = context.WithValue(ctx, pageDraftKey{}, auditRequests)
			audited, err := c.translateComplete(ctx, targetLang, priorTurns, matchedGlossary, auditRequests)
			if err != nil {
				return fmt.Errorf("image-grounded translation audit failed: %w", err)
			}
			for _, result := range audited {
				resByID[result.ID] = result
			}
		}
	} else if c.cfg.LLMVision && !c.cfg.LLMVisionAudit && len(rawPNG) > 0 {
		microRequests, pageDraft := collectSuspiciousMicroVerifyRequests(pageDOM.Nodes, requests, resByID)
		if len(microRequests) > 0 {
			microCtx := context.WithValue(ctx, translationCheckpointKey{}, (*translationCheckpoint)(nil))
			microCtx = context.WithValue(microCtx, pageDraftKey{}, pageDraft)
			verified, vErr := c.translateComplete(microCtx, targetLang, priorTurns, matchedGlossary, microRequests)
			if vErr != nil {
				return fmt.Errorf("targeted micro-verification failed: %w", vErr)
			}
			for _, result := range verified {
				resByID[result.ID] = result
			}
		}
	}
	needsReviewCount := 0
	verifiedNodes := make([]dom.TextNode, 0, len(pageDOM.Nodes))
	for i := range pageDOM.Nodes {
		n := &pageDOM.Nodes[i]
		if res, ok := resByID[n.ID]; ok {
			if res.IsText != nil && !*res.IsText {
				pageDOM.RejectedRegions = append(pageDOM.RejectedRegions, dom.RejectedRegion{ID: n.ID, BBox: n.BBox, Reason: res.Reason, SourceLines: n.SourceLines, SourceQuadrilaterals: n.SourceQuadrilaterals, SourceReadings: n.SourceReadings})
				continue
			}
			if strings.TrimSpace(res.SourceText) != "" && c.cfg.LLMVision {
				n.OCRText = n.SourceText
				n.SourceText = strings.TrimSpace(res.SourceText)
				n.RecognitionPending = false
			}
			n.TargetText = strings.TrimSpace(res.TargetText)
			n.TargetSegments = res.TargetSegments
			// Never erase and re-render floating/diegetic markings (e.g. curved body labels
			// like "SEX DOLL 01") when translation leaves the text identical to the original.
			if isUnchangedSurfaceText(n.BubbleKind, n.SourceText, n.TargetText) {
				pageDOM.RejectedRegions = append(pageDOM.RejectedRegions, dom.RejectedRegion{ID: n.ID, BBox: n.BBox, Reason: "unchanged_surface_text", SourceLines: n.SourceLines, SourceQuadrilaterals: n.SourceQuadrilaterals, SourceReadings: n.SourceReadings})
				continue
			}
			n.TargetRubyPairs = alignTargetRubies(n.TargetText, res.Rubies)
			if n.CharBudgetMax > 0 && utf8.RuneCountInString(n.TargetText) > n.CharBudgetMax {
				n.Uncertainty.NeedsReview = true
				if n.Uncertainty.Score < 0.7 {
					n.Uncertainty.Score = 0.7
				}
				if !containsString(n.Uncertainty.Reasons, "budget_overflow") {
					n.Uncertainty.Reasons = append(n.Uncertainty.Reasons, "budget_overflow")
				}
			}
		}
		if n.Uncertainty.NeedsReview {
			needsReviewCount++
		}
		verifiedNodes = append(verifiedNodes, *n)
	}
	pageDOM.Nodes = verifiedNodes
	pageDOM.Metrics.TotalNodes = len(verifiedNodes)
	pageDOM.Metrics.NeedsReviewNodes = needsReviewCount
	pageDOM.TranslationProgress = nil
	return nil
}

// translateComplete repairs actual missing/empty outputs with a focused model
// request. Unknown and duplicate IDs are protocol errors, never fabricated text.
func (c *Compiler) translateComplete(ctx context.Context, lang string, turns []store.DialogueTurn, glossary []store.GlossaryTerm, requests []NodeTranslationRequest) ([]NodeTranslationResult, error) {
	if limit := c.cfg.LLMBatchSize; limit > 0 && len(requests) > limit {
		var completed []NodeTranslationResult
		for start := 0; start < len(requests); start += limit {
			batch, err := c.translateComplete(ctx, lang, turns, glossary, requests[start:min(start+limit, len(requests))])
			if err != nil {
				return nil, err
			}
			completed = append(completed, batch...)
		}
		return completed, nil
	}
	accepted := make(map[string]NodeTranslationResult, len(requests))
	checkpoint, _ := ctx.Value(translationCheckpointKey{}).(*translationCheckpoint)
	resultKeys := make(map[string]string, len(requests))
	if checkpoint != nil {
		draft, _ := ctx.Value(pageDraftKey{}).([]NodeTranslationRequest)
		for _, req := range requests {
			payload, err := json.Marshal(struct {
				Request   NodeTranslationRequest
				PageDraft []NodeTranslationRequest
			}{req, draft})
			if err != nil {
				return nil, err
			}
			key := fmt.Sprintf("%x", sha256.Sum256(payload))
			resultKeys[req.ID] = key
			if data := checkpoint.progress.Results[key]; len(data) > 0 {
				var saved NodeTranslationResult
				if err := json.Unmarshal(data, &saved); err != nil {
					return nil, err
				}
				accepted[saved.ID] = saved
			}
		}
		if len(accepted) > 0 {
			log.Printf("[narrative] resumed %d/%d validated node results", len(accepted), len(requests))
		}
	}
	var pending []NodeTranslationRequest
	for _, req := range requests {
		if _, ok := accepted[req.ID]; !ok {
			pending = append(pending, req)
		}
	}
	repairAttempted := make(map[string]bool)
	for len(pending) > 0 {
		started := time.Now()
		log.Printf("[narrative] requesting %d nodes", len(pending))
		results, err := c.backend.TranslateNodes(ctx, lang, turns, glossary, pending)
		if err != nil {
			return nil, err
		}
		log.Printf("[narrative] received %d node results in %s", len(results), time.Since(started).Round(time.Millisecond))
		expected := make(map[string]NodeTranslationRequest, len(pending))
		for _, req := range pending {
			expected[req.ID] = req
		}
		seen := make(map[string]bool, len(results))
		repairs := make(map[string]NodeTranslationRequest)
		for _, result := range results {
			req, known := expected[result.ID]
			if !known {
				return nil, fmt.Errorf("translation returned unknown node %q", result.ID)
			}
			if seen[result.ID] {
				return nil, fmt.Errorf("translation returned duplicate node %q", result.ID)
			}
			seen[result.ID] = true
			rejected := result.IsText != nil && !*result.IsText
			if req.SourceTextLocked || (!rejected && strings.TrimSpace(result.TargetText) != "") {
				if issue := translationFidelityIssue(req, result, c.cfg.LLMVision); issue != "" {
					if repairAttempted[result.ID] {
						return nil, fmt.Errorf("translation fidelity repair failed for %s: %s", result.ID, issue)
					}
					repairAttempted[result.ID] = true
					req.DraftTranslation, req.Correction = result.TargetText, issue
					if result.SourceText != "" && c.cfg.LLMVision && !req.SourceTextLocked {
						req.SourceText = result.SourceText
					}
					repairs[result.ID] = req
					continue
				}
			}
			if (rejected && c.cfg.LLMVision && strings.TrimSpace(result.Reason) != "") || (!rejected && strings.TrimSpace(result.TargetText) != "" && (strings.TrimSpace(req.SourceText) != "" || strings.TrimSpace(result.SourceText) != "")) {
				if !rejected && isChineseTargetLang(lang) {
					result.TargetText, result.TargetSegments = normalizeChineseEllipses(result.TargetText, result.TargetSegments)
				}
				accepted[result.ID] = result
			}
		}
		var missing []NodeTranslationRequest
		for _, req := range pending {
			if _, ok := accepted[req.ID]; !ok {
				if repair, ok := repairs[req.ID]; ok {
					req = repair
				}
				missing = append(missing, req)
			}
		}
		if len(missing) == len(pending) && len(repairs) == 0 {
			return nil, fmt.Errorf("translation made no progress: %d nodes have no nonempty translation", len(missing))
		}
		if checkpoint != nil {
			for _, req := range requests {
				if res, ok := accepted[req.ID]; ok {
					data, err := json.Marshal(res)
					if err != nil {
						return nil, err
					}
					checkpoint.progress.Results[resultKeys[req.ID]] = data
				}
			}
			if err := checkpoint.save(); err != nil {
				return nil, fmt.Errorf("save translation progress: %w", err)
			}
		}
		pending = missing
	}
	out := make([]NodeTranslationResult, 0, len(requests))
	for _, req := range requests {
		out = append(out, accepted[req.ID])
	}
	return out, nil
}

func containsString(items []string, target string) bool {
	for _, it := range items {
		if it == target {
			return true
		}
	}
	return false
}

func sourceReadingsEmptyOrConflicting(readings []dom.SourceReading) bool {
	if len(readings) == 0 {
		return true
	}
	byLine := make(map[int]string)
	hasNonEmpty := false
	for _, r := range readings {
		txt := strings.TrimSpace(r.Text)
		if txt != "" {
			hasNonEmpty = true
		}
		if prev, seen := byLine[r.LineIndex]; seen {
			if prev != txt {
				return true
			}
		} else {
			byLine[r.LineIndex] = txt
		}
	}
	return !hasNonEmpty
}

var targetBracketPairs = [][2]rune{
	{'（', '）'},
	{'(', ')'},
	{'「', '」'},
	{'『', '』'},
	{'【', '】'},
	{'《', '》'},
	{'[', ']'},
}

// pageTargetUnbalancedNodes checks bracket balance across non-rejected nodes in
// reading order so valid cross-node split pairs (e.g. "（..." in node k and
// "...）" in node k+1) are not flagged as unbalanced.
func pageTargetUnbalancedNodes(requests []NodeTranslationRequest, resByID map[string]NodeTranslationResult) map[int]bool {
	order := make([]int, len(requests))
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool {
		return requests[order[a]].ReadingOrder < requests[order[b]].ReadingOrder
	})
	type openEntry struct {
		reqIdx  int
		pairIdx int
	}
	var stack []openEntry
	unbalanced := make(map[int]bool)
	for _, idx := range order {
		res, ok := resByID[requests[idx].ID]
		if !ok || (res.IsText != nil && !*res.IsText) {
			continue
		}
		for _, r := range res.TargetText {
			for pIdx, p := range targetBracketPairs {
				if r == p[0] {
					stack = append(stack, openEntry{reqIdx: idx, pairIdx: pIdx})
					break
				}
				if r == p[1] {
					matched := -1
					for k := len(stack) - 1; k >= 0; k-- {
						if stack[k].pairIdx == pIdx {
							matched = k
							break
						}
					}
					if matched < 0 {
						unbalanced[idx] = true
					} else {
						stack = append(stack[:matched], stack[matched+1:]...)
					}
					break
				}
			}
		}
	}
	for _, rem := range stack {
		unbalanced[rem.reqIdx] = true
	}
	return unbalanced
}

func collectSuspiciousMicroVerifyRequests(
	domNodes []dom.TextNode,
	requests []NodeTranslationRequest,
	resByID map[string]NodeTranslationResult,
) ([]NodeTranslationRequest, []NodeTranslationRequest) {
	domByID := make(map[string]dom.TextNode, len(domNodes))
	for _, n := range domNodes {
		domByID[n.ID] = n
	}

	// Build complete post-pass page draft enriched with first-pass SourceText/TargetText
	// so cross-node discourse continuity works even when initial SourceText was empty.
	postPass := make([]NodeTranslationRequest, len(requests))
	copy(postPass, requests)
	for i := range postPass {
		res, ok := resByID[postPass[i].ID]
		if !ok {
			continue
		}
		if res.IsText == nil || *res.IsText {
			if strings.TrimSpace(res.SourceText) != "" && !postPass[i].SourceTextLocked {
				postPass[i].SourceText = strings.TrimSpace(res.SourceText)
			}
		} else {
			postPass[i].SourceText = ""
		}
		postPass[i].DraftTranslation = res.TargetText
		postPass[i].DraftIsText = res.IsText
		postPass[i].DraftReason = res.Reason
	}
	annotateDiscourseTopology(postPass)
	unbalancedTarget := pageTargetUnbalancedNodes(postPass, resByID)

	var out []NodeTranslationRequest
	for i, req := range postPass {
		res, ok := resByID[req.ID]
		if !ok || (res.IsText != nil && !*res.IsText) {
			continue
		}
		domNode := domByID[req.ID]
		isFalsePositiveRisk := false
		if req.BubbleSemantics == "floating_sfx" {
			if containsString(domNode.Uncertainty.Reasons, "low_confidence_candidate") {
				isFalsePositiveRisk = true
			} else if utf8.RuneCountInString(strings.TrimSpace(res.TargetText)) <= 2 &&
				len(req.BBox) == 4 && min(req.BBox[2], req.BBox[3]) <= 64 &&
				sourceReadingsEmptyOrConflicting(req.SourceReadings) {
				isFalsePositiveRisk = true
			}
		}
		isUnbalanced := unbalancedTarget[i]
		if !isFalsePositiveRisk && !isUnbalanced {
			continue
		}
		micro := req
		var reasons []string
		if isFalsePositiveRisk {
			reasons = append(reasons, "MICRO-VERIFY FALSE-POSITIVE RISK: Inspect the padded crop (red 1px exterior frame marks exact candidate box) and scene overview. Is this genuine readable writing/SFX or a non-text artwork artifact (button, clothing fold/collar, hair highlight, panel/frame line)? If non-text artifact, return is_text:false with a visual reason and target_segments:[]. If genuine writing, verify exact source_text and accurate translation.")
		}
		if isUnbalanced {
			reasons = append(reasons, "MICRO-VERIFY UNBALANCED PUNCTUATION: target_text contains unmatched brackets/parentheses. Inspect the crop and fix missing characters or unmatched brackets/parentheses.")
		}
		micro.Correction = strings.Join(reasons, " ")
		out = append(out, micro)
	}
	return out, postPass
}

func alignTargetRubies(targetText string, rubies []RubyTranslation) []dom.RubyPair {
	if len(rubies) == 0 || targetText == "" {
		return nil
	}
	totalRunes := utf8.RuneCountInString(targetText)
	used := make([]bool, totalRunes)
	var out []dom.RubyPair
	for _, r := range rubies {
		if r.TargetBase == "" || r.TargetRuby == "" {
			continue
		}
		baseRunes := utf8.RuneCountInString(r.TargetBase)
		searchFrom := 0
		for searchFrom < len(targetText) {
			relIdx := strings.Index(targetText[searchFrom:], r.TargetBase)
			if relIdx < 0 {
				break
			}
			absByte := searchFrom + relIdx
			cStart := utf8.RuneCountInString(targetText[:absByte])
			cEnd := cStart + baseRunes
			overlap := false
			for pos := cStart; pos < cEnd && pos < totalRunes; pos++ {
				if used[pos] {
					overlap = true
					break
				}
			}
			if !overlap {
				for pos := cStart; pos < cEnd && pos < totalRunes; pos++ {
					used[pos] = true
				}
				out = append(out, dom.RubyPair{
					Base:      r.TargetBase,
					Ruby:      r.TargetRuby,
					CharStart: cStart,
					CharEnd:   cEnd,
				})
				break
			}
			_, runeSize := utf8.DecodeRuneInString(targetText[absByte:])
			if runeSize <= 0 {
				break
			}
			searchFrom = absByte + runeSize
		}
	}
	return out
}

// OpenAICompatibleBackend requires Chat Completions with strict JSON Schema
// structured output. Unsupported endpoints return an explicit upstream error.
type OpenAICompatibleBackend struct {
	diagnosticsPath      string
	baseURL              string
	apiKey               string
	model                string
	reasoningEffort      string
	preserveArtText      bool
	reasoningUnsupported atomic.Bool
	httpClient           *http.Client
}

func NewOpenAICompatibleBackend(cfg config.Config) *OpenAICompatibleBackend {
	return &OpenAICompatibleBackend{
		diagnosticsPath: filepath.Join(cfg.DataDir, "diagnostics", "llm"),
		baseURL:         strings.TrimRight(strings.TrimSpace(cfg.LLMBaseURL), "/"),
		apiKey:          strings.TrimSpace(cfg.LLMAPIKey),
		model:           strings.TrimSpace(cfg.LLMModel),
		reasoningEffort: config.NormalizeReasoningEffort(cfg.LLMReasoningEffort),
		preserveArtText: cfg.PreserveArtText,
		httpClient: &http.Client{
			Timeout: cfg.LLMRequestTimeout,
		},
	}
}

func (b *OpenAICompatibleBackend) TranslateNodes(
	ctx context.Context,
	targetLang string,
	priorTurns []store.DialogueTurn,
	matchedGlossary []store.GlossaryTerm,
	nodes []NodeTranslationRequest,
) ([]NodeTranslationResult, error) {
	rawImg, hasVision := ctx.Value(pageImageKey{}).([]byte)
	hasVision = hasVision && len(rawImg) > 0

	hasLockedSource := false
	hasDraftOrAudit := ctx.Value(pageDraftKey{}) != nil
	hasRubyPairs := false
	hasCorrection := false
	hasDiscourseHint := false
	hasPhonology := false
	hasPersonaRules := false
	for _, n := range nodes {
		if n.SourceTextLocked {
			hasLockedSource = true
		}
		if n.DraftTranslation != "" || n.DraftIsText != nil {
			hasDraftOrAudit = true
		}
		if len(n.RubyPairs) > 0 {
			hasRubyPairs = true
		}
		if n.Correction != "" {
			hasCorrection = true
		}
		if n.DiscourseHint != "" {
			hasDiscourseHint = true
		}
		if n.Phonology.ElongationCount > 0 || n.Phonology.GlottalStop || n.Phonology.Reduplicated {
			hasPhonology = true
		}
		if n.Speaker != "" || n.Addressee != "" || n.FirstPersonRule != "" || n.SpeechStyleRule != "" || n.RelationHonorific != "" {
			hasPersonaRules = true
		}
	}

	var sb strings.Builder
	fmt.Fprintf(&sb, "You are Prism-T's Narrative Localization Compiler translating comic/manga nodes into %s.\n", targetLang)
	sb.WriteString("Strict Invariants:\n")
	if hasLockedSource {
		sb.WriteString("source_text_locked:true means an independent source-only pass has already transcribed this node. Return source_text BYTE-FOR-BYTE unchanged, including newlines; do not re-read, normalize, expand punctuation, or replace its glyphs during translation, compression or audit. Translate the complete supplied source. Use the page to resolve referents and intent, never to rewrite a locked transcription. All later image/OCR correction instructions apply only to unlocked nodes. A transcription can still be a hallucination over illustration: independently inspect actual writing and reject such non-writing with a visual reason rather than inventing a translation.\n")
	}
	sb.WriteString("1. Distinguish `track`: 'primary' main narrative sequence vs 'aside' background margin commentary.\n")
	sb.WriteString("2. Match `bubble_semantics` ('shout_spiky'=explosive terse, 'thought_cloud'=inner monologue, 'narration_box'=literary caption, 'floating_sfx'=borderless text: determine dialogue/caption/sound effect from content and image, 'diegetic_surface'=in-world sign/label). Never turn borderless dialogue into a sound effect.\n")
	if hasPersonaRules {
		sb.WriteString("3. Enforce `first_person_rule`, `speech_style_rule`, and `relation_honorific` (how Speaker addresses Addressee).\n")
	}
	sb.WriteString("4. Translate completely and naturally while aiming concisely within [char_budget_min, char_budget_max] (measured space capacity at original font size) whenever natural phrasing allows. Prefer crisp spoken/narrative dialogue rather than verbose padding. Never omit meaning, propositions, negation, or speaker tone merely to shrink characters; the typesetter adjusts font size when a complete translation exceeds char_budget_max.\n")
	if hasPhonology {
		sb.WriteString("5. Preserve `phonology` elongation/reduplication intensity proportionally on sound effects.\n")
	}
	if hasRubyPairs {
		sb.WriteString("6. When `ruby_pairs` is present (Gikun Ruby dual-layer semantics), emit `rubies: [{\"target_base\": \"...\", \"target_ruby\": \"...\"}]` where `target_base` is an exact substring of `target_text`.\n")
	}
	sb.WriteString("7. Output strictly JSON with a translations object keyed by each supplied node ID. Each value must contain the same id as its key, is_text, source_text (exact transcription), target_segments (the complete translation in ordered linguistic units), reason and rubies. The application concatenates target_segments verbatim into the translation; do not output a separate target_text field.\n")
	sb.WriteString("8. Return EVERY supplied node exactly once with its original id. Actual text requires nonempty source_text and a nonempty target_segments array containing the complete translation. A region containing no writing requires is_text:false and a reason. Never omit a difficult node, merge IDs, or invent new IDs.\n")
	sb.WriteString("9. Preserve all meaning, negation, uncertainty, character intent, and emotional tone. A character budget is a layout preference, never permission to delete meaning. Prefer a faithful complete translation when no faithful compression fits.\n")
	if hasDraftOrAudit {
		sb.WriteString("10. If draft_translation is supplied, audit it against the actual image independently. For unlocked nodes verify each source glyph and line. Check every proposition, agent/patient/beneficiary direction, ally-vs-enemy stance, ownership, qualification, cross-node split-clause grammar, and character register against the draft and scene. Fix flipped roles, dangling split-clause fragments, literal source-kanji calques, stiff translationese, and wrong sentence-final punctuation. Return source_text and the full corrected translation as target_segments in the same JSON contract. Treat the draft as fallible evidence; do not approve it merely because it reads fluently.\n")
		if b.preserveArtText {
			sb.WriteString("11. Also audit draft_is_text:false decisions against the image. Dialogue, narration, sound effects, titles, and marginal notes must always be translated, while decorative body tattoos, skin stamps, clothing prints, and translucent status-stamp overlays integrated into character artwork should remain is_text:false with reason 'decorative_body_art'.\n")
		} else {
			sb.WriteString("11. Also audit draft_is_text:false decisions against the image. Any readable writing counts as text: dialogue, narration, sound effects, titles, marginal notes, package labels, signs and numbers. Never reject writing because it is not dialogue, is difficult to read, or was previously rejected. Return is_text:false only for a region with no actual writing and give a visual reason.\n")
		}
	}
	sb.WriteString("12. source_lines and source_quadrilaterals define this node's owned writing. bbox includes context and padding: do not copy adjacent titles, captions, or other nodes merely because their text is visible in the crop. Preserve line order and transcribe only the supplied node's own writing.\n")
	sb.WriteString("Independent source_readings are local recognizer observations for zero-based source line_index, not translations or an answer key. Read their corresponding line images and resolve disagreements from the actual glyph shapes. Scores are recognizer-specific and cannot be compared across models. Do not silently adopt a fluent guess, normalize visible source characters into a more likely expression, add laughter or explanatory text absent from the writing, or copy a draft over contradictory image evidence. Empty readings do not mean the region is not text.\n")
	sb.WriteString("A null recognition confidence means the recognizer does not expose probabilities; it is neither zero confidence nor certainty. truncated:true means the recognizer reached its output limit without an end token: recover the rest from the source image, never treat that partial reading as the complete line. Check the exact number and order of kana in negation against the native line image; adding, omitting or duplicating one changes the meaning.\n")
	sb.WriteString("13. Preserve every written digit run verbatim, including chapter numbers and repeated numbers. Do not spell digits out, add or discard numbers. target_segments is the sole output text: include every space, meaningful newline and punctuation within these strings; the application inserts nothing between units. Segment into short, natural linguistic units for line breaking: keep a lexical word, proper name, compound noun or number intact; attach closing punctuation to the preceding unit. Do not split every Han character. For is_text:false return target_segments:[].")
	if hasCorrection {
		sb.WriteString(" If correction is supplied, repair that specific defect from the image/source and return the entire corrected result.")
	}
	sb.WriteByte('\n')
	sb.WriteString("14. Preserve anonymous masking-symbol runs verbatim, with the same number of marks and occurrences. They denote withheld identities; do not invent a name or replace them with a pronoun. Distinguish a name/honorific suffix from a second-person pronoun using the clause and page context. In the target language, resolve each speaker, addressee and third-person referent consistently; do not concatenate a retained honorific and an incompatible pronoun or move an address into the subject of a different clause.")
	if hasDiscourseHint || hasDraftOrAudit {
		sb.WriteString(" When `discourse_hint` indicates `mixed_outer_and_parenthetical_monologue`, keep the spoken outer clause and enclosed `（...）` parenthetical inner monologue distinct in voice while preserving parentheses. When `discourse_hint` marks cross-node unclosed/closing quotes or parentheses or `clause_continues_to_next`/`continues_clause_from_prev`, translate split clauses as one grammatically seamless target sentence across neighboring `reading_order` nodes—keeping explicit causal/prepositional linkage and avoiding dangling fragments or duplicated subjects.")
	}
	sb.WriteString("\n15. Semantic & Scene Grounding: Verify each clause's agent (who acts), patient/target (who is affected), and ally-vs-adversary stance from the page overview, character posture/equipment, and `preceding_chapter_context`. Carefully resolve source transitive, causative, passive, directional, and benefactive constructions so defensive/protective statements toward allies are never inverted into threats, and vice versa. Maintain consistent terminology, character names, pronouns, and relationship stance with `preceding_chapter_context`.")
	sb.WriteString("\n16. Idiomatic Localization & Register: Translate into natural, vivid target-language prose matching the speaker's persona and register (e.g., haughty archaic antagonist, solemn prayer, urgent battle command, casual banter) rather than stiff word-for-word calques. Translate split sentences across consecutive `reading_order` nodes so the joined result is grammatically complete. Match closing punctuation to actual illocutionary force (use question marks for genuine questions rather than periods).")
	if isChineseTargetLang(targetLang) {
		sb.WriteString(" For Chinese targets (`zh*`), localize source-language kanji compounds, place-name morphemes, directional enumerations, and idioms into idiomatic Chinese rather than copying untranslated Japanese kanji verbatim, never leave untranslated Japanese kana (such as `っ`/`ッ`) inside Chinese text, avoid awkward Westernized/translationese syntax, and format all ellipses consistently as standard six-dot `……` rather than single `…` or triple `………`.")
	}
	if b.preserveArtText {
		sb.WriteString("\n17. Decorative Character/Body Artwork Preservation (`preserve_art_text` enabled): Do NOT translate decorative markings, tattoos, serial numbers, logos, clothing prints, armor engravings, or translucent glowing status-stamp overlays drawn directly onto a character's skin, body, outfit, or accessories (e.g., curved body prints like `SEX DOLL 01`, skin tattoos, T-shirt text, semi-transparent head stamps) unless they are actual spoken dialogue, narration, or manga sound effects. For such decorative in-art markings, return `is_text: false`, `target_segments: []`, and `reason: \"decorative_body_art\"` so the original illustration pixels stay completely untouched.")
	}
	if hasDraftOrAudit {
		sb.WriteString(" On audit, independently check these referents, agent/patient roles, and natural phrasing against the source and image, including split clauses in neighboring nodes.")
	}
	sb.WriteByte('\n')
	if draft, ok := ctx.Value(pageDraftKey{}).([]NodeTranslationRequest); ok {
		pageJSON, err := json.Marshal(draft)
		if err != nil {
			return nil, err
		}
		sb.WriteString("\nComplete page draft (fallible, not an answer key):\n")
		sb.Write(pageJSON)
		sb.WriteString("\nUse all page occurrences to resolve proper names, address forms, ownership, and references consistently. A recurring name must not alternate between a personal name and a literal common noun. Read split sentences across neighboring nodes, preserve their negation and scope, and independently correct the draft from the image. Output only the requested batch IDs, not all context IDs.\n")
	}

	if len(matchedGlossary) > 0 {
		sb.WriteString("\nActive Work Glossary:\n")
		for _, g := range matchedGlossary {
			fmt.Fprintf(&sb, "- %s => %s", g.SourceTerm, g.TargetTerm)
			if g.RubyOverride != "" {
				fmt.Fprintf(&sb, " [ruby: %s]", g.RubyOverride)
			}
			if g.Notes != "" {
				fmt.Fprintf(&sb, " (%s)", g.Notes)
			}
			sb.WriteByte('\n')
		}
	}

	wireNodes := nodes
	if hasVision {
		copied := make([]NodeTranslationRequest, len(nodes))
		copy(copied, nodes)
		for i := range copied {
			if !copied[i].SourceTextLocked && len(copied[i].SourceLines) > 0 {
				copied[i].SourceQuadrilaterals = nil
			}
		}
		wireNodes = copied
	}
	payloadMap := map[string]any{
		"nodes": wireNodes,
	}
	if len(priorTurns) > 0 {
		payloadMap["preceding_chapter_context"] = priorTurns
	}
	userJSON, err := json.Marshal(payloadMap)
	if err != nil {
		return nil, err
	}

	var userContent any = string(userJSON)
	if hasVision {
		source, err := decodePageImage(ctx, rawImg)
		if err != nil {
			return nil, fmt.Errorf("decode page image: %w", err)
		}
		overviewJPEG, err := buildSceneOverviewJPEG(source, nodes)
		if err != nil {
			return nil, fmt.Errorf("build scene overview: %w", err)
		}
		sheetJPEG, tiles, err := buildNodeCropSheetJPEG(source, nodes)
		if err != nil {
			return nil, fmt.Errorf("build node crop sheet: %w", err)
		}
		tilesJSON, err := json.Marshal(tiles)
		if err != nil {
			return nil, err
		}
		userContent = []map[string]any{
			{"type": "text", "text": string(userJSON)},
			{"type": "text", "text": "Annotated page overview (boxes and external #1..#N callouts locate each node on the page for panel layout, speaker, tone and cross-node flow):"},
			{"type": "image_url", "image_url": map[string]string{"url": "data:image/jpeg;base64," + base64.StdEncoding.EncodeToString(overviewJPEG), "detail": "auto"}},
			{"type": "text", "text": "Packed node glyph contact sheet (indexed headers #1..#N and node ID are printed strictly above each padded crop; inspect glyphs here without confusion from neighboring bubbles; source_rect maps onto original image): Tiles: " + string(tilesJSON)},
			{"type": "image_url", "image_url": map[string]string{"url": "data:image/jpeg;base64," + base64.StdEncoding.EncodeToString(sheetJPEG), "detail": "high"}},
		}
		sb.WriteString("\nThe supplied two images are structured page evidence: (1) an annotated page overview with external #1..#N callouts for scene semantics and reading context, and (2) a packed high-resolution contact sheet of padded node crops labeled #1..#N <node_id> above each crop for exact glyph reading. Small compact nodes include expanded surrounding context with a thin 1px red exterior frame marking the exact candidate box. Each bbox is [x,y,width,height] in original image pixels. For unlocked nodes read actual owned text and include its transcription as source_text.")
		if hasLockedSource {
			sb.WriteString(" Preserve locked source_text unchanged.")
		}
		sb.WriteString(" Preserve small kana, punctuation and column reading order. Resolve tone and speaker intent from the page. Do not invent dialogue outside the supplied nodes. Detector boxes on buttons, clothing folds, hair highlights, texture noise, or panel/frame lines with no genuine glyphs MUST be returned as is_text:false with a concise visual reason even if local source_readings contain noisy OCR guesses. Return is_text:true for actual text.\n")
	}
	response, err := b.completeJSON(ctx, sb.String(), userContent, translationResponseFormat(nodes), nodes)
	if err != nil {
		return nil, err
	}
	var batch struct {
		Translations *keyedTranslations `json:"translations"`
	}
	if err := json.Unmarshal([]byte(extractJSONObject(response)), &batch); err != nil {
		return nil, fmt.Errorf("invalid JSON from llm: %w", err)
	}
	if batch.Translations == nil {
		return nil, errors.New("invalid translation response: required translations object is missing or null")
	}
	results := make([]NodeTranslationResult, 0, len(*batch.Translations))
	for _, result := range *batch.Translations {
		results = append(results, result)
	}
	return results, nil
}

func isUnsupportedReasoningEffortResponse(status int, body []byte) bool {
	if status != http.StatusBadRequest && status != http.StatusUnprocessableEntity {
		return false
	}
	msg := strings.ToLower(string(body))
	if strings.Contains(msg, "reasoning_effort") {
		return true
	}
	if strings.Contains(msg, "reasoning") {
		for _, marker := range []string{"unsupported", "unknown", "invalid", "unrecognized"} {
			if strings.Contains(msg, marker) {
				return true
			}
		}
	}
	return false
}

func (b *OpenAICompatibleBackend) executeChatCompletionStream(
	ctx context.Context,
	prompt string,
	userContent any,
	format map[string]any,
) (chatStreamCompletion, error, string, time.Duration, error) {
	request := map[string]any{
		"model": b.model, "temperature": 0.2, "response_format": format, "stream": true,
		"messages": []map[string]any{
			{"role": "system", "content": prompt},
			{"role": "user", "content": userContent},
		},
	}

	endpoint := b.baseURL
	if !strings.HasSuffix(endpoint, "/chat/completions") {
		endpoint += "/chat/completions"
	}
	imageInputs := 0
	if content, ok := userContent.([]map[string]any); ok {
		for _, item := range content {
			if item["type"] == "image_url" {
				imageInputs++
			}
		}
	}

	var completion chatStreamCompletion
	var streamErr error
	var started time.Time
	effectiveEffort := ""
	effort := b.reasoningEffort
	overrideEffort, _ := ctx.Value(reasoningEffortOverrideKey{}).(string)
	if overrideEffort != "" {
		effort = overrideEffort
	}
	allowReasoning := effort != "" && (overrideEffort != "" || !b.reasoningUnsupported.Load())
	transientRetries := 0
	const maxTransientRetries = 3
	for attempt := 0; attempt < 2+maxTransientRetries; attempt++ {
		sendReasoning := allowReasoning
		if sendReasoning {
			effectiveEffort = effort
			request["reasoning_effort"] = effort
		} else {
			effectiveEffort = ""
			delete(request, "reasoning_effort")
		}
		reqBody, err := json.Marshal(request)
		if err != nil {
			return chatStreamCompletion{}, nil, effectiveEffort, 0, err
		}

		httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(reqBody))
		if err != nil {
			return chatStreamCompletion{}, nil, effectiveEffort, 0, err
		}
		httpReq.Header.Set("Content-Type", "application/json")
		if b.apiKey != "" {
			httpReq.Header.Set("Authorization", "Bearer "+b.apiKey)
		}
		started = time.Now()
		log.Printf("[narrative] LLM request: %d bytes, %d image inputs", len(reqBody), imageInputs)

		resp, err := b.httpClient.Do(httpReq)
		if err != nil {
			if ctx.Err() == nil && transientRetries < maxTransientRetries {
				transientRetries++
				backoff := time.Duration(transientRetries*2) * time.Second
				log.Printf("[narrative] transient LLM transport error (attempt %d/%d): %v; retrying in %s", transientRetries, maxTransientRetries, err, backoff)
				select {
				case <-ctx.Done():
					return chatStreamCompletion{}, nil, effectiveEffort, time.Since(started), ctx.Err()
				case <-time.After(backoff):
					continue
				}
			}
			return chatStreamCompletion{}, nil, effectiveEffort, time.Since(started), fmt.Errorf("LLM request (%d bytes): %w", len(reqBody), err)
		}
		if resp.StatusCode >= 300 {
			errBody, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
			resp.Body.Close()
			if sendReasoning && isUnsupportedReasoningEffortResponse(resp.StatusCode, errBody) {
				if overrideEffort == "" {
					b.reasoningUnsupported.Store(true)
				}
				allowReasoning = false
				log.Printf("[narrative] endpoint rejected reasoning_effort=%q; retrying without reasoning_effort", effort)
				continue
			}
			isTransientHTTP := resp.StatusCode == http.StatusTooManyRequests ||
				resp.StatusCode == http.StatusBadGateway ||
				resp.StatusCode == http.StatusServiceUnavailable ||
				resp.StatusCode == http.StatusGatewayTimeout ||
				resp.StatusCode == 524
			if isTransientHTTP && ctx.Err() == nil && transientRetries < maxTransientRetries {
				transientRetries++
				backoff := time.Duration(transientRetries*3) * time.Second
				log.Printf("[narrative] transient HTTP %d from LLM endpoint (attempt %d/%d); retrying in %s", resp.StatusCode, transientRetries, maxTransientRetries, backoff)
				select {
				case <-ctx.Done():
					return chatStreamCompletion{}, nil, effectiveEffort, time.Since(started), ctx.Err()
				case <-time.After(backoff):
					continue
				}
			}
			return chatStreamCompletion{}, nil, effectiveEffort, time.Since(started), fmt.Errorf("llm endpoint %s returned %d after %s (%d bytes, %d images): %s", endpoint, resp.StatusCode, time.Since(started).Round(time.Millisecond), len(reqBody), imageInputs, string(errBody))
		}

		completion, streamErr = readChatCompletionStream(resp.Body, func() {
			log.Printf("[narrative] LLM first stream event after %s", time.Since(started).Round(time.Millisecond))
		})
		resp.Body.Close()
		break
	}
	return completion, streamErr, effectiveEffort, time.Since(started), nil
}

func (b *OpenAICompatibleBackend) completeJSON(ctx context.Context, prompt string, userContent any, format map[string]any, nodes any) (string, error) {
	completion, streamErr, _, elapsed, err := b.executeChatCompletionStream(ctx, prompt, userContent, format)
	if err != nil {
		return "", err
	}
	diagnosticsDir := b.diagnosticsPath
	if err := os.MkdirAll(diagnosticsDir, 0o755); err != nil {
		return "", err
	}
	diagnostic, err := json.MarshalIndent(map[string]any{
		"model": b.model, "requested_nodes": nodes,
		"response": map[string]any{"choices": []any{map[string]any{
			"message": map[string]string{"content": completion.Content}, "finish_reason": completion.FinishReason,
		}}},
		"stream_events": completion.Events, "elapsed_ms": elapsed.Milliseconds(),
		"stream_error": fmt.Sprint(streamErr),
	}, "", "  ")
	if err != nil {
		return "", err
	}
	if err := os.WriteFile(filepath.Join(diagnosticsDir, fmt.Sprintf("%d.json", time.Now().UnixNano())), diagnostic, 0o600); err != nil {
		return "", err
	}
	if streamErr != nil {
		return "", fmt.Errorf("LLM stream after %s: %w", elapsed.Round(time.Millisecond), streamErr)
	}
	return completion.Content, nil
}

// ProbeOpenAICompatibleEndpoint verifies structured JSON Chat Completions connectivity without writing diagnostic files.
func ProbeOpenAICompatibleEndpoint(ctx context.Context, cfg config.Config) (int64, string, error) {
	if strings.TrimSpace(cfg.LLMBaseURL) == "" {
		return 0, "", ErrNoTranslationBackend
	}
	if strings.TrimSpace(cfg.LLMModel) == "" {
		return 0, "", errors.New("llm_model is required")
	}
	probeCfg := cfg
	if probeCfg.LLMRequestTimeout <= 0 || probeCfg.LLMRequestTimeout > 30*time.Second {
		probeCfg.LLMRequestTimeout = 30 * time.Second
	}
	b := NewOpenAICompatibleBackend(probeCfg)
	format := map[string]any{
		"type": "json_schema",
		"json_schema": map[string]any{
			"name":   "prism_t_connectivity_probe",
			"strict": true,
			"schema": map[string]any{
				"type":                 "object",
				"additionalProperties": false,
				"required":             []string{"ok"},
				"properties": map[string]any{
					"ok": map[string]any{"type": "boolean"},
				},
			},
		},
	}
	completion, streamErr, effectiveEffort, elapsed, err := b.executeChatCompletionStream(ctx, "Return {\"ok\":true}.", "ping", format)
	if err != nil {
		return elapsed.Milliseconds(), config.FormatReasoningEffortSetting(effectiveEffort), err
	}
	if streamErr != nil {
		return elapsed.Milliseconds(), config.FormatReasoningEffortSetting(effectiveEffort), fmt.Errorf("LLM stream error: %w", streamErr)
	}
	var out struct {
		OK bool `json:"ok"`
	}
	if uErr := json.Unmarshal([]byte(extractJSONObject(completion.Content)), &out); uErr != nil {
		return elapsed.Milliseconds(), config.FormatReasoningEffortSetting(effectiveEffort), fmt.Errorf("invalid probe JSON: %w", uErr)
	}
	if !out.OK {
		return elapsed.Milliseconds(), config.FormatReasoningEffortSetting(effectiveEffort), errors.New("probe returned ok=false")
	}
	return elapsed.Milliseconds(), config.FormatReasoningEffortSetting(effectiveEffort), nil
}

type chatStreamCompletion struct {
	Content      string
	FinishReason string
	Events       int
}

// Only a completed, normally stopped SSE response may reach JSON decoding.
// Partial output and reasoning deltas are never promoted to source/translation.
func readChatCompletionStream(input io.Reader, firstEvent func()) (chatStreamCompletion, error) {
	reader := bufio.NewReader(input)
	var result chatStreamCompletion
	var text, event strings.Builder
	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			result.Content = text.String()
			if err == io.EOF {
				err = io.ErrUnexpectedEOF
			}
			return result, err
		}
		line = strings.TrimSuffix(strings.TrimSuffix(line, "\n"), "\r")
		if strings.HasPrefix(line, "data:") {
			if event.Len() > 0 {
				event.WriteByte('\n')
			}
			event.WriteString(strings.TrimPrefix(strings.TrimPrefix(line, "data:"), " "))
			continue
		}
		if line != "" || event.Len() == 0 {
			continue
		}
		payload := event.String()
		event.Reset()
		result.Content = text.String()
		if payload == "[DONE]" {
			if result.FinishReason != "stop" || text.Len() == 0 {
				return result, fmt.Errorf("stream ended without a complete answer (finish_reason=%q)", result.FinishReason)
			}
			return result, nil
		}
		var chunk struct {
			Error   json.RawMessage `json:"error"`
			Choices []struct {
				Index int `json:"index"`
				Delta struct {
					Content string `json:"content"`
					Refusal string `json:"refusal"`
				} `json:"delta"`
				FinishReason *string `json:"finish_reason"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			return result, fmt.Errorf("invalid chat stream event: %w", err)
		}
		result.Events++
		if result.Events == 1 && firstEvent != nil {
			firstEvent()
		}
		if len(chunk.Error) > 0 && string(chunk.Error) != "null" {
			return result, fmt.Errorf("upstream stream error: %s", chunk.Error)
		}
		for _, choice := range chunk.Choices {
			if choice.Index != 0 || choice.Delta.Refusal != "" || (result.FinishReason != "" && choice.Delta.Content != "") {
				return result, fmt.Errorf("unexpected choice, refusal or content after completion")
			}
			text.WriteString(choice.Delta.Content)
			result.Content = text.String()
			if choice.FinishReason != nil {
				result.FinishReason = *choice.FinishReason
				if result.FinishReason != "stop" {
					return result, fmt.Errorf("incomplete answer (finish_reason=%q)", result.FinishReason)
				}
			}
		}
	}
}

func translationResponseFormat(nodes []NodeTranslationRequest) map[string]any {
	ids := make([]string, len(nodes))
	for i, node := range nodes {
		ids[i] = node.ID
	}
	text := map[string]string{"type": "string"}
	ruby := map[string]any{
		"type": "object", "additionalProperties": false,
		"properties": map[string]any{"target_base": text, "target_ruby": text},
		"required":   []string{"target_base", "target_ruby"},
	}
	properties := make(map[string]any, len(nodes))
	for _, node := range nodes {
		var source any = text
		if node.SourceTextLocked {
			source = map[string]any{"type": "string", "enum": []string{node.SourceText}}
		}
		properties[node.ID] = map[string]any{
			"type": "object", "additionalProperties": false,
			"properties": map[string]any{
				"id":          map[string]any{"type": "string", "enum": []string{node.ID}},
				"is_text":     map[string]string{"type": "boolean"},
				"source_text": source, "reason": text,
				"rubies":          map[string]any{"type": "array", "items": ruby},
				"target_segments": map[string]any{"type": "array", "items": text},
			},
			"required": []string{"id", "is_text", "source_text", "target_segments", "reason", "rubies"},
		}
	}
	return map[string]any{
		"type": "json_schema",
		"json_schema": map[string]any{
			"name": "comic_translation", "strict": true,
			"schema": map[string]any{
				"type": "object", "additionalProperties": false,
				"required": []string{"translations"},
				"properties": map[string]any{"translations": map[string]any{
					"type": "object", "additionalProperties": false, "required": ids, "properties": properties,
				}},
			},
		},
	}
}

type sourceAtlasTile struct {
	LineIndex  int    `json:"line_index"`
	SourceRect [4]int `json:"source_rect"`
	AtlasRect  [4]int `json:"atlas_rect"`
}

// Pack every owned line at native resolution. Headers are outside the source
// rectangles, so indexing adds no ink to the actual glyph evidence. Packing by
// height reduces image padding; line_index always keeps its original identity.
func sourceLineAtlas(source image.Image, node NodeTranslationRequest) (*image.RGBA, []sourceAtlasTile, error) {
	face := basicfont.Face7x13
	header := face.Metrics().Height.Ceil()
	tiles := make([]sourceAtlasTile, len(node.SourceLines))
	order := make([]int, len(tiles))
	area, widest := 0, 0
	cellWidth := func(index int) int {
		label := font.MeasureString(face, strconv.Itoa(index)).Ceil()
		return max(tiles[index].SourceRect[2], label)
	}
	for index, box := range node.SourceLines {
		if len(box) != 4 {
			return nil, nil, fmt.Errorf("node %s line %d has invalid bbox", node.ID, index)
		}
		rect := image.Rect(box[0], box[1], box[0]+box[2], box[1]+box[3]).Intersect(source.Bounds())
		if rect.Empty() {
			return nil, nil, fmt.Errorf("node %s line %d is outside source image", node.ID, index)
		}
		tiles[index] = sourceAtlasTile{LineIndex: index, SourceRect: [4]int{rect.Min.X, rect.Min.Y, rect.Dx(), rect.Dy()}}
		width := cellWidth(index)
		area += width * (rect.Dy() + header)
		widest = max(widest, width)
		order[index] = index
	}
	sort.SliceStable(order, func(i, j int) bool {
		return tiles[order[i]].SourceRect[3] > tiles[order[j]].SourceRect[3]
	})
	width := max(widest, int(math.Ceil(math.Sqrt(float64(area)))))
	x, y, rowHeight := 0, 0, 0
	for _, index := range order {
		tile := &tiles[index]
		cell := cellWidth(index)
		if x+cell > width {
			x, y, rowHeight = 0, y+rowHeight, 0
		}
		tile.AtlasRect = [4]int{x, y + header, tile.SourceRect[2], tile.SourceRect[3]}
		x += cell
		rowHeight = max(rowHeight, header+tile.SourceRect[3])
	}
	atlas := image.NewRGBA(image.Rect(0, 0, width, y+rowHeight))
	draw.Draw(atlas, atlas.Bounds(), image.NewUniform(color.White), image.Point{}, draw.Src)
	for _, tile := range tiles {
		r := tile.AtlasRect
		draw.Draw(atlas, image.Rect(r[0], r[1], r[0]+r[2], r[1]+r[3]), source, image.Pt(tile.SourceRect[0], tile.SourceRect[1]), draw.Src)
		drawer := font.Drawer{Dst: atlas, Src: image.NewUniform(color.Black), Face: face,
			Dot: fixed.P(r[0], r[1]-face.Metrics().Descent.Ceil())}
		drawer.DrawString(strconv.Itoa(tile.LineIndex))
	}
	return atlas, tiles, nil
}

// Preserve page and complete-node context, with all owned lines in an indexed
// native-pixel atlas. Many small crops no longer each consume an image input.
func nodeImageEvidence(raw []byte, nodes []NodeTranslationRequest) ([]map[string]any, error) {
	source, _, err := image.Decode(bytes.NewReader(raw))
	if err != nil {
		return nil, err
	}
	return nodeImageEvidenceFromImage(source, nodes)
}

func nodeImageEvidenceFromImage(source image.Image, nodes []NodeTranslationRequest) ([]map[string]any, error) {
	content := make([]map[string]any, 0, 2*len(nodes))
	for _, node := range nodes {
		if len(node.BBox) != 4 {
			return nil, fmt.Errorf("node %s has invalid bbox", node.ID)
		}
		b := node.BBox
		rect := image.Rect(b[0], b[1], b[0]+b[2], b[1]+b[3]).Intersect(source.Bounds())
		if rect.Empty() {
			return nil, fmt.Errorf("node %s bbox is outside source image", node.ID)
		}
		crop := image.NewRGBA(image.Rect(0, 0, rect.Dx(), rect.Dy()))
		draw.Draw(crop, crop.Bounds(), source, rect.Min, draw.Src)
		images := []image.Image{crop}
		labels := []string{"Complete source crop for node " + node.ID + "; use it to resolve actual reading order and join the lines. Preserve punctuation and inspect the actual glyph shapes."}
		if len(node.SourceLines) > 0 {
			atlas, tiles, err := sourceLineAtlas(source, node)
			if err != nil {
				return nil, err
			}
			metadata, err := json.Marshal(tiles)
			if err != nil {
				return nil, err
			}
			images = append(images, atlas)
			labels = append(labels, "Native source line atlas for node "+node.ID+". Printed numeric headers are indices, not source writing. All rectangles are [x,y,width,height]; source_rect uses page coordinates, atlas_rect locates unchanged source pixels below the header. Packing order is not reading order. Inspect every tile and return line_texts by original line_index. Tiles: "+string(metadata))
		}
		for index, img := range images {
			var encoded bytes.Buffer
			if err := png.Encode(&encoded, img); err != nil {
				return nil, err
			}
			content = append(content,
				map[string]any{"type": "text", "text": labels[index]},
				map[string]any{"type": "image_url", "image_url": map[string]string{"url": "data:image/png;base64," + base64.StdEncoding.EncodeToString(encoded.Bytes()), "detail": "high"}},
			)
		}
	}
	return content, nil
}

func extractJSONObject(content string) string {
	s := strings.TrimSpace(content)
	if start := strings.Index(s, "```"); start >= 0 {
		rest := s[start+3:]
		if nl := strings.IndexByte(rest, '\n'); nl >= 0 {
			rest = rest[nl+1:]
		}
		if end := strings.Index(rest, "```"); end >= 0 {
			s = strings.TrimSpace(rest[:end])
		}
	}
	openIdx := strings.IndexByte(s, '{')
	if openIdx < 0 {
		return s
	}
	depth := 0
	inStr := false
	escaped := false
	for i := openIdx; i < len(s); i++ {
		ch := s[i]
		if escaped {
			escaped = false
			continue
		}
		if inStr {
			if ch == '\\' {
				escaped = true
			} else if ch == '"' {
				inStr = false
			}
			continue
		}
		switch ch {
		case '"':
			inStr = true
		case '{':
			depth++
		case '}':
			depth--
			if depth == 0 {
				return s[openIdx : i+1]
			}
		}
	}
	return s[openIdx:]
}

func compactTextNoSpace(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	for _, r := range s {
		if unicode.IsSpace(r) {
			continue
		}
		b.WriteRune(r)
	}
	return b.String()
}

// isUnchangedSurfaceText reports whether a borderless/diegetic marking (e.g. a
// curved body print like "SEX DOLL 01") remained verbatim identical after
// translation (ignoring whitespace/newlines and ASCII letter case). Erasing and
// re-rendering identical text destroys the original curved artwork for zero gain.
func isUnchangedSurfaceText(bubbleKind, sourceText, targetText string) bool {
	if bubbleKind != "floating_sfx" && bubbleKind != "diegetic_surface" {
		return false
	}
	src := compactTextNoSpace(sourceText)
	tgt := compactTextNoSpace(targetText)
	if src == "" || tgt == "" {
		return false
	}
	return strings.EqualFold(src, tgt)
}
