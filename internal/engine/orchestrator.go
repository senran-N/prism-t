package engine

import (
	"context"
	"encoding/base64"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"

	"prism-t/internal/archive"
	"prism-t/internal/config"
	"prism-t/internal/dom"
	"prism-t/internal/narrative"
	"prism-t/internal/store"
)

type Orchestrator struct {
	mu       sync.RWMutex
	cfg      config.Config
	store    *store.Store
	client   *Client
	compiler *narrative.Compiler
}

type TranslateProgressEvent struct {
	Stage           string
	WorkID          int64
	TotalUnits      int
	DownloadedUnits int
	CompletedUnits  int
	UnitIndex       int
	Filename        string
	PageID          int64
	Message         string
}

type TranslateBatchOptions struct {
	WorkTitle       string
	ChapterTitle    string
	FormatProfile   string
	TargetLang      string
	AnalyzeOnly     bool
	RefPages        []archive.InputPage
	OnProgress      func(TranslateProgressEvent)
	OnPageDone      func(TranslatedUnitResult) error
	ContinueOnError bool
	OnPageError     func(index int, filename string, err error)
}

type TranslatedUnitResult struct {
	PageID       int64
	Filename     string
	CompositePNG []byte
	SliceEntries []archive.OutputEntry
	DOM          dom.PageDOM
}

func NewOrchestrator(cfg config.Config, st *store.Store, client *Client, comp *narrative.Compiler) *Orchestrator {
	return &Orchestrator{
		cfg:      cfg,
		store:    st,
		client:   client,
		compiler: comp,
	}
}

func (o *Orchestrator) Config() config.Config {
	o.mu.RLock()
	defer o.mu.RUnlock()
	return o.cfg
}

func (o *Orchestrator) UpdateConfig(cfg config.Config) {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.cfg = cfg
}

func emitProgress(opts TranslateBatchOptions, ev TranslateProgressEvent) {
	if opts.OnProgress != nil {
		opts.OnProgress(ev)
	}
}

func (o *Orchestrator) TranslatePages(
	ctx context.Context,
	inputPages []archive.InputPage,
	opts TranslateBatchOptions,
) (store.Work, []TranslatedUnitResult, error) {
	cfg := o.Config()
	if !opts.AnalyzeOnly && !o.compiler.HasBackend() {
		return store.Work{}, nil, narrative.ErrNoTranslationBackend
	}
	if len(inputPages) == 0 {
		return store.Work{}, nil, fmt.Errorf("no input pages provided")
	}
	if opts.WorkTitle == "" {
		opts.WorkTitle = "Untitled Comic"
	}
	if opts.ChapterTitle == "" {
		opts.ChapterTitle = "Chapter 1"
	}
	if opts.TargetLang == "" {
		opts.TargetLang = cfg.TargetLang
	}
	if opts.FormatProfile == "" {
		opts.FormatProfile = "auto"
	}

	work, err := o.store.EnsureWork(ctx, opts.WorkTitle, "auto", opts.TargetLang, opts.FormatProfile)
	if err != nil {
		return store.Work{}, nil, err
	}

	// Build seamless Virtual Continuous Canvases when Webtoon slice boundaries bisect active bubbles/panels!
	units, err := archive.BuildVirtualUnits(inputPages, 16384, func(topPNG, botPNG []byte) bool {
		stitch, sErr := o.client.CheckSeam(ctx, topPNG, botPNG)
		return sErr == nil && stitch
	})
	if err != nil {
		return store.Work{}, nil, fmt.Errorf("build virtual units: %w", err)
	}
	emitProgress(opts, TranslateProgressEvent{
		Stage:      "preparing",
		WorkID:     work.ID,
		TotalUnits: len(units),
		Message:    fmt.Sprintf("已切分并准备 %d 个画布单元", len(units)),
	})

	refPerUnit := resolveRefPNGsForUnits(inputPages, units, opts.RefPages)

	type unitPipelineState struct {
		layoutReady  chan struct{}
		restoreDone  chan struct{}
		usedFallback bool
		pageID       int64
		layoutDOM    dom.PageDOM
		layoutErr    error
		anaResp      AnalyzeResponse
		restoreResp  AnalyzeResponse
		restoreErr   error
	}
	states := make([]*unitPipelineState, len(units))
	for i := range units {
		states[i] = &unitPipelineState{
			layoutReady: make(chan struct{}),
			restoreDone: make(chan struct{}),
		}
	}

	maxConc := cfg.MaxConcurrency
	if maxConc < 1 {
		maxConc = 1
	}
	if maxConc > len(units) {
		maxConc = len(units)
	}

	recognizeText := opts.AnalyzeOnly || !cfg.LLMVision
	var completedUnits atomic.Int32
	sem := make(chan struct{}, maxConc)
	var wg sync.WaitGroup
	defer wg.Wait()
	for idx, unit := range units {
		st := states[idx]
		wg.Add(1)
		go func(i int, u archive.VirtualUnit, refPNG []byte, state *unitPipelineState) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			emitProgress(opts, TranslateProgressEvent{
				Stage:          "analyzing",
				TotalUnits:     len(units),
				CompletedUnits: int(completedUnits.Load()),
				UnitIndex:      i,
				Filename:       u.Name,
				Message:        fmt.Sprintf("正在分析与物理去字 (%d/%d): %s", i+1, len(units), u.Name),
			})

			runFallbackAnalyze := func() {
				resp, aErr := o.client.Analyze(ctx, u.PNGBytes, refPNG, opts.FormatProfile, u.SeamSlices, recognizeText)
				if aErr != nil {
					state.layoutErr = fmt.Errorf("analyze %s: %w", u.Name, aErr)
					close(state.layoutReady)
					close(state.restoreDone)
					return
				}
				pd := resp.DOM
				adaptTargetTypography(&pd, opts.TargetLang)
				preview, decodeErr := base64.StdEncoding.DecodeString(resp.CleanPreviewPNGB64)
				if decodeErr != nil {
					state.layoutErr = fmt.Errorf("decode analysis %s: %w", u.Name, decodeErr)
					close(state.layoutReady)
					close(state.restoreDone)
					return
				}
				pageID, saveErr := o.store.SavePage(ctx, work.ID, opts.ChapterTitle, i, u.Name, u.Blake3Hash, pd, u.PNGBytes, preview, "analyzed")
				if saveErr != nil {
					state.layoutErr = fmt.Errorf("save analysis %s: %w", u.Name, saveErr)
					close(state.layoutReady)
					close(state.restoreDone)
					return
				}
				state.usedFallback = true
				state.pageID = pageID
				state.anaResp = resp
				state.layoutDOM = pd
				close(state.layoutReady)
				close(state.restoreDone)
				emitProgress(opts, TranslateProgressEvent{
					Stage:          "analyzed",
					TotalUnits:     len(units),
					CompletedUnits: int(completedUnits.Load()),
					UnitIndex:      i,
					Filename:       u.Name,
					Message:        fmt.Sprintf("完成物理反编译 (%d/%d): %s", i+1, len(units), u.Name),
				})
			}

			if opts.AnalyzeOnly {
				runFallbackAnalyze()
				return
			}

			detResp, dErr := o.client.DetectLayout(ctx, u.PNGBytes, refPNG, opts.FormatProfile, u.SeamSlices, recognizeText)
			if dErr != nil {
				if IsEndpointNotFound(dErr) {
					runFallbackAnalyze()
					return
				}
				state.layoutErr = fmt.Errorf("detect layout %s: %w", u.Name, dErr)
				close(state.layoutReady)
				close(state.restoreDone)
				return
			}

			layoutDOM := detResp.DOM
			adaptTargetTypography(&layoutDOM, opts.TargetLang)
			pageID, saveErr := o.store.SavePage(ctx, work.ID, opts.ChapterTitle, i, u.Name, u.Blake3Hash, layoutDOM, u.PNGBytes, []byte{}, "layout_detected")
			if saveErr != nil {
				state.layoutErr = fmt.Errorf("save layout %s: %w", u.Name, saveErr)
				close(state.layoutReady)
				close(state.restoreDone)
				return
			}
			state.pageID = pageID
			state.layoutDOM = layoutDOM
			close(state.layoutReady)

			resResp, rErr := o.client.RestorePage(ctx, detResp.LayoutToken, u.PNGBytes, refPNG, opts.FormatProfile, u.SeamSlices, recognizeText, nil)
			if rErr != nil {
				state.restoreErr = fmt.Errorf("restore page %s: %w", u.Name, rErr)
			} else {
				adaptTargetTypography(&resResp.DOM, opts.TargetLang)
				state.restoreResp = resResp
			}
			close(state.restoreDone)
			if rErr == nil {
				emitProgress(opts, TranslateProgressEvent{
					Stage:          "analyzed",
					TotalUnits:     len(units),
					CompletedUnits: int(completedUnits.Load()),
					UnitIndex:      i,
					Filename:       u.Name,
					Message:        fmt.Sprintf("完成物理反编译 (%d/%d): %s", i+1, len(units), u.Name),
				})
			}
		}(idx, unit, refPerUnit[idx], st)
	}

	// Each page enters narrative compilation as soon as its layout is detected,
	// overlapping LLM translation with Z0/Z2 restoration in Python.
	results := make([]TranslatedUnitResult, 0, len(units))
	for idx, unit := range units {
		st := states[idx]
		<-st.layoutReady
		if st.layoutErr != nil {
			<-st.restoreDone
			return work, nil, st.layoutErr
		}
		pageDOM := st.layoutDOM
		pageID := st.pageID
		pageStatus := "analyzed"
		if !opts.AnalyzeOnly {
			emitProgress(opts, TranslateProgressEvent{
				Stage:          "translating",
				TotalUnits:     len(units),
				CompletedUnits: int(completedUnits.Load()),
				UnitIndex:      idx,
				Filename:       unit.Name,
				Message:        fmt.Sprintf("正在进行叙事视觉翻译 (%d/%d): %s", idx+1, len(units), unit.Name),
			})
			// Ground speakers in persistent SQLite Persona State Machine & compile geometry-budgeted translations sequentially for context continuity
			if err := o.compiler.CompilePageTranslations(ctx, o.store, work.ID, opts.ChapterTitle, idx, opts.TargetLang, &pageDOM, unit.PNGBytes); err != nil {
				return work, nil, fmt.Errorf("compile narrative %s: %w", unit.Name, err)
			}
			<-st.restoreDone
			if st.restoreErr != nil {
				return work, nil, st.restoreErr
			}
			if !st.usedFallback {
				restoredDOM := st.restoreResp.DOM
				mergeTranslatedLayout(&restoredDOM, pageDOM, opts.TargetLang)
				pageDOM = restoredDOM
			}
			pageStatus = "translated"
			if err := o.store.UpdatePageDOMAndStatus(ctx, pageID, pageDOM, nil, pageStatus); err != nil {
				return work, nil, err
			}
		} else {
			<-st.restoreDone
		}

		var compBytes []byte
		var sliceOutputs []archive.OutputEntry
		if opts.AnalyzeOnly {
			cleanBytes, dErr := base64.StdEncoding.DecodeString(st.anaResp.CleanPreviewPNGB64)
			if dErr != nil {
				return work, nil, fmt.Errorf("decode clean preview PNG: %w", dErr)
			}
			compBytes = cleanBytes
			sliceOutputs, err = archive.SliceCompositePNG(compBytes, unit.SeamSlices, unit.Name)
			if err != nil {
				return work, nil, fmt.Errorf("slice clean preview %s: %w", unit.Name, err)
			}
			if err := o.store.UpdatePageDOMAndStatus(ctx, pageID, pageDOM, compBytes, pageStatus); err != nil {
				return work, nil, fmt.Errorf("save page %s: %w", unit.Name, err)
			}
		} else {
			emitProgress(opts, TranslateProgressEvent{
				Stage:          "rendering",
				TotalUnits:     len(units),
				CompletedUnits: int(completedUnits.Load()),
				UnitIndex:      idx,
				Filename:       unit.Name,
				Message:        fmt.Sprintf("正在进行 SDF 嵌字与无缝回贴 (%d/%d): %s", idx+1, len(units), unit.Name),
			})
			renResp, rErr := o.client.Render(ctx, unit.PNGBytes, pageDOM, opts.TargetLang)
			if rErr != nil {
				return work, nil, fmt.Errorf("render %s: %w", unit.Name, rErr)
			}
			pageDOM = renResp.DOM
			var dErr error
			compBytes, dErr = base64.StdEncoding.DecodeString(renResp.CompositePNGB64)
			if dErr != nil {
				return work, nil, fmt.Errorf("decode composite PNG: %w", dErr)
			}
			if len(unit.SeamSlices) > 0 && len(renResp.SlicePNGsB64) == len(unit.SeamSlices) {
				for sIdx, sl := range unit.SeamSlices {
					sb, sErr := base64.StdEncoding.DecodeString(renResp.SlicePNGsB64[sIdx])
					if sErr != nil {
						return work, nil, fmt.Errorf("decode slice PNG %s: %w", sl.Filename, sErr)
					}
					sliceOutputs = append(sliceOutputs, archive.OutputEntry{
						Filename: archive.EnsurePNGExt(sl.Filename),
						Data:     sb,
					})
				}
			} else {
				sliceOutputs, err = archive.SliceCompositePNG(compBytes, unit.SeamSlices, unit.Name)
				if err != nil {
					return work, nil, fmt.Errorf("slice composite %s: %w", unit.Name, err)
				}
			}
			if err := o.store.UpdatePageDOMAndStatus(ctx, pageID, pageDOM, compBytes, pageStatus); err != nil {
				return work, nil, fmt.Errorf("save page %s: %w", unit.Name, err)
			}
		}

		doneCount := int(completedUnits.Add(1))
		emitProgress(opts, TranslateProgressEvent{
			Stage:          "page_done",
			WorkID:         work.ID,
			TotalUnits:     len(units),
			CompletedUnits: doneCount,
			UnitIndex:      idx,
			Filename:       unit.Name,
			PageID:         pageID,
			Message:        fmt.Sprintf("已完成 (%d/%d): %s", doneCount, len(units), unit.Name),
		})

		results = append(results, TranslatedUnitResult{
			PageID:       pageID,
			Filename:     unit.Name,
			CompositePNG: compBytes,
			SliceEntries: sliceOutputs,
			DOM:          pageDOM,
		})
	}

	return work, results, nil
}

// StreamPageItem is fed into TranslatePageStream as pages finish downloading in the background.
type StreamPageItem struct {
	Index int
	Total int
	Page  archive.InputPage
	Err   error
}

// TranslatePageStream translates pages one-by-one as they arrive from an upstream download stream,
// overlapping Z0 restoration with LLM narrative translation on every page and persisting each page
// immediately so users can start reading page 1 while pages 2..N are still downloading and translating.
func (o *Orchestrator) TranslatePageStream(
	ctx context.Context,
	totalExpected int,
	stream <-chan StreamPageItem,
	opts TranslateBatchOptions,
) (store.Work, []TranslatedUnitResult, error) {
	cfg := o.Config()
	if !opts.AnalyzeOnly && !o.compiler.HasBackend() {
		return store.Work{}, nil, narrative.ErrNoTranslationBackend
	}
	if opts.WorkTitle == "" {
		opts.WorkTitle = "Untitled Comic"
	}
	if opts.ChapterTitle == "" {
		opts.ChapterTitle = "Chapter 1"
	}
	if opts.TargetLang == "" {
		opts.TargetLang = cfg.TargetLang
	}
	if opts.FormatProfile == "" {
		opts.FormatProfile = "auto"
	}

	work, err := o.store.EnsureWork(ctx, opts.WorkTitle, "auto", opts.TargetLang, opts.FormatProfile)
	if err != nil {
		return store.Work{}, nil, err
	}
	emitProgress(opts, TranslateProgressEvent{
		Stage:      "downloading",
		WorkID:     work.ID,
		TotalUnits: totalExpected,
		Message:    fmt.Sprintf("已创建作品《%s》，正在边下载边翻译（共 %d 页）", work.Title, totalExpected),
	})

	recognizeText := opts.AnalyzeOnly || !cfg.LLMVision
	maxConc := cfg.MaxConcurrency
	if maxConc < 1 {
		maxConc = 1
	}

	streamCtx, cancelStream := context.WithCancel(ctx)
	defer cancelStream()

	var stateMu sync.Mutex
	var results []TranslatedUnitResult
	var doneCount atomic.Int32
	failedCount := 0
	var fatalErr error

	recordError := func(idx int, filename string, pageErr error) bool {
		stateMu.Lock()
		defer stateMu.Unlock()
		if opts.ContinueOnError {
			failedCount++
			if opts.OnPageError != nil {
				opts.OnPageError(idx, filename, pageErr)
			}
			return false
		}
		if fatalErr == nil {
			fatalErr = pageErr
			cancelStream()
		}
		return true
	}

	processItem := func(item StreamPageItem) bool {
		if streamCtx.Err() != nil {
			return true
		}
		if item.Err != nil {
			return recordError(item.Index, item.Page.Filename, fmt.Errorf("download page %d: %w", item.Index+1, item.Err))
		}
		total := totalExpected
		if item.Total > total {
			total = item.Total
		}
		u := item.Page
		idx := item.Index

		emitProgress(opts, TranslateProgressEvent{
			Stage:          "analyzing",
			WorkID:         work.ID,
			TotalUnits:     total,
			CompletedUnits: int(doneCount.Load()),
			UnitIndex:      idx,
			Filename:       u.Filename,
			Message:        fmt.Sprintf("正在分析与去字 (%d/%d): %s", idx+1, total, u.Filename),
		})

		var pageDOM dom.PageDOM
		var pageID int64
		var compBytes []byte
		pageStatus := "analyzed"

		if opts.AnalyzeOnly {
			resp, aErr := o.client.Analyze(streamCtx, u.PNGBytes, nil, opts.FormatProfile, nil, true)
			if aErr != nil {
				return recordError(idx, u.Filename, fmt.Errorf("analyze %s: %w", u.Filename, aErr))
			}
			pageDOM = resp.DOM
			adaptTargetTypography(&pageDOM, opts.TargetLang)
			preview, dErr := base64.StdEncoding.DecodeString(resp.CleanPreviewPNGB64)
			if dErr != nil {
				return recordError(idx, u.Filename, dErr)
			}
			compBytes = preview
			var sErr error
			pageID, sErr = o.store.SavePage(streamCtx, work.ID, opts.ChapterTitle, idx, u.Filename, u.Blake3Hash, pageDOM, u.PNGBytes, compBytes, pageStatus)
			if sErr != nil {
				return recordError(idx, u.Filename, sErr)
			}
		} else {
			detResp, dErr := o.client.DetectLayout(streamCtx, u.PNGBytes, nil, opts.FormatProfile, nil, recognizeText)
			usedFallback := false
			var fallbackResp AnalyzeResponse
			if dErr != nil {
				if !IsEndpointNotFound(dErr) {
					return recordError(idx, u.Filename, fmt.Errorf("detect layout %s: %w", u.Filename, dErr))
				}
				usedFallback = true
				var fErr error
				fallbackResp, fErr = o.client.Analyze(streamCtx, u.PNGBytes, nil, opts.FormatProfile, nil, recognizeText)
				if fErr != nil {
					return recordError(idx, u.Filename, fmt.Errorf("analyze %s: %w", u.Filename, fErr))
				}
				pageDOM = fallbackResp.DOM
			} else {
				pageDOM = detResp.DOM
			}
			adaptTargetTypography(&pageDOM, opts.TargetLang)
			var sErr error
			pageID, sErr = o.store.SavePage(streamCtx, work.ID, opts.ChapterTitle, idx, u.Filename, u.Blake3Hash, pageDOM, u.PNGBytes, []byte{}, "layout_detected")
			if sErr != nil {
				return recordError(idx, u.Filename, sErr)
			}

			// Overlap Z0/Z2 restoration with LLM narrative translation
			restoreDone := make(chan struct{})
			var restoreResp AnalyzeResponse
			var restoreErr error
			if !usedFallback {
				go func() {
					defer close(restoreDone)
					restoreResp, restoreErr = o.client.RestorePage(streamCtx, detResp.LayoutToken, u.PNGBytes, nil, opts.FormatProfile, nil, recognizeText, nil)
					if restoreErr == nil {
						adaptTargetTypography(&restoreResp.DOM, opts.TargetLang)
					}
				}()
			} else {
				close(restoreDone)
			}

			emitProgress(opts, TranslateProgressEvent{
				Stage:          "translating",
				WorkID:         work.ID,
				TotalUnits:     total,
				CompletedUnits: int(doneCount.Load()),
				UnitIndex:      idx,
				Filename:       u.Filename,
				Message:        fmt.Sprintf("正在视觉翻译 (%d/%d): %s", idx+1, total, u.Filename),
			})
			if cErr := o.compiler.CompilePageTranslations(streamCtx, o.store, work.ID, opts.ChapterTitle, idx, opts.TargetLang, &pageDOM, u.PNGBytes); cErr != nil {
				<-restoreDone
				return recordError(idx, u.Filename, fmt.Errorf("compile narrative %s: %w", u.Filename, cErr))
			}
			<-restoreDone
			if restoreErr != nil {
				return recordError(idx, u.Filename, fmt.Errorf("restore page %s: %w", u.Filename, restoreErr))
			}
			if !usedFallback {
				restoredDOM := restoreResp.DOM
				mergeTranslatedLayout(&restoredDOM, pageDOM, opts.TargetLang)
				pageDOM = restoredDOM
			}

			emitProgress(opts, TranslateProgressEvent{
				Stage:          "rendering",
				WorkID:         work.ID,
				TotalUnits:     total,
				CompletedUnits: int(doneCount.Load()),
				UnitIndex:      idx,
				Filename:       u.Filename,
				Message:        fmt.Sprintf("正在 SDF 嵌字 (%d/%d): %s", idx+1, total, u.Filename),
			})
			renResp, rErr := o.client.Render(streamCtx, u.PNGBytes, pageDOM, opts.TargetLang)
			if rErr != nil {
				return recordError(idx, u.Filename, fmt.Errorf("render %s: %w", u.Filename, rErr))
			}
			pageDOM = renResp.DOM
			var dErr2 error
			compBytes, dErr2 = base64.StdEncoding.DecodeString(renResp.CompositePNGB64)
			if dErr2 != nil {
				return recordError(idx, u.Filename, fmt.Errorf("decode composite %s: %w", u.Filename, dErr2))
			}
			pageStatus = "translated"
			if uErr := o.store.UpdatePageDOMAndStatus(streamCtx, pageID, pageDOM, compBytes, pageStatus); uErr != nil {
				return recordError(idx, u.Filename, fmt.Errorf("save page %s: %w", u.Filename, uErr))
			}
		}

		res := TranslatedUnitResult{
			PageID:       pageID,
			Filename:     u.Filename,
			CompositePNG: compBytes,
			DOM:          pageDOM,
		}
		if opts.OnPageDone != nil {
			stateMu.Lock()
			cbErr := opts.OnPageDone(res)
			if cbErr == nil {
				results = append(results, TranslatedUnitResult{
					PageID:   pageID,
					Filename: u.Filename,
				})
			}
			stateMu.Unlock()
			if cbErr != nil {
				return recordError(idx, u.Filename, cbErr)
			}
		} else {
			stateMu.Lock()
			results = append(results, res)
			stateMu.Unlock()
		}

		doneNow := int(doneCount.Add(1))
		emitProgress(opts, TranslateProgressEvent{
			Stage:          "page_done",
			WorkID:         work.ID,
			TotalUnits:     total,
			CompletedUnits: doneNow,
			UnitIndex:      idx,
			Filename:       u.Filename,
			PageID:         pageID,
			Message:        fmt.Sprintf("已完成 (%d/%d): %s", doneNow, total, u.Filename),
		})
		return false
	}

	var wg sync.WaitGroup
	for w := 0; w < maxConc; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for item := range stream {
				if processItem(item) {
					return
				}
			}
		}()
	}
	wg.Wait()

	if fatalErr != nil {
		return work, results, fatalErr
	}
	if ctx.Err() != nil {
		return work, results, ctx.Err()
	}
	if failedCount > 0 {
		return work, results, fmt.Errorf("%d page(s) failed during stream translation (completed %d)", failedCount, len(results))
	}
	if len(results) == 0 {
		return work, nil, fmt.Errorf("source returned 0 downloadable pages")
	}
	return work, results, nil
}

// RerenderStoredPage re-runs the typesetter & composite layer engine against
// the cached Z0/Z2/Shading patches in PageDOM, persisting updated edits and optional review approval.
func (o *Orchestrator) RerenderStoredPage(
	ctx context.Context,
	pageID int64,
	updatedDOM dom.PageDOM,
	targetLang string,
	markReviewed bool,
) (store.PageRecord, error) {
	cfg := o.Config()
	existing, err := o.store.GetPage(ctx, pageID)
	if err != nil {
		return store.PageRecord{}, err
	}
	if targetLang == "" {
		targetLang = cfg.TargetLang
	}
	renResp, err := o.client.Render(ctx, existing.RawPNG, updatedDOM, targetLang)
	if err != nil {
		return store.PageRecord{}, err
	}
	compBytes, err := base64.StdEncoding.DecodeString(renResp.CompositePNGB64)
	if err != nil {
		return store.PageRecord{}, err
	}
	if markReviewed {
		for i := range renResp.DOM.Nodes {
			renResp.DOM.Nodes[i].Uncertainty.NeedsReview = false
		}
		renResp.DOM.Metrics.NeedsReviewNodes = 0
	}
	if err := o.store.UpdatePageRender(ctx, pageID, renResp.DOM, compBytes, markReviewed); err != nil {
		return store.PageRecord{}, err
	}
	return o.store.GetPage(ctx, pageID)
}

// RetranslateStoredPage runs narrative compilation + rendering against cached Z0/Z2/Shading layers
// for an existing stored page without re-running visual analysis/restoration.
func (o *Orchestrator) RetranslateStoredPage(
	ctx context.Context,
	pageID int64,
	targetLang string,
) (store.PageRecord, error) {
	if !o.compiler.HasBackend() {
		return store.PageRecord{}, narrative.ErrNoTranslationBackend
	}
	rec, err := o.store.GetPage(ctx, pageID)
	if err != nil {
		return store.PageRecord{}, err
	}
	cfg := o.Config()
	lang := strings.TrimSpace(targetLang)
	if lang == "" {
		lang = cfg.TargetLang
	}
	adaptTargetTypography(&rec.DOM, lang)
	rec.DOM.TranslationProgress = nil
	if err := o.compiler.CompilePageTranslations(ctx, o.store, rec.Summary.WorkID, rec.Summary.ChapterTitle, rec.Summary.PageIndex, lang, &rec.DOM, rec.RawPNG); err != nil {
		return store.PageRecord{}, err
	}
	if err := o.store.UpdatePageDOMAndStatus(ctx, pageID, rec.DOM, nil, "translated"); err != nil {
		return store.PageRecord{}, err
	}
	return o.RerenderStoredPage(ctx, pageID, rec.DOM, lang, false)
}

func mergeTranslatedLayout(restored *dom.PageDOM, translated dom.PageDOM, targetLang string) {
	if restored == nil {
		return
	}
	adaptTargetTypography(restored, targetLang)

	rejectedIDs := make(map[string]bool, len(translated.RejectedRegions))
	existingRejected := make(map[string]bool, len(restored.RejectedRegions))
	for _, r := range restored.RejectedRegions {
		existingRejected[r.ID] = true
		rejectedIDs[r.ID] = true
	}
	for _, r := range translated.RejectedRegions {
		rejectedIDs[r.ID] = true
		if !existingRejected[r.ID] {
			restored.RejectedRegions = append(restored.RejectedRegions, r)
			existingRejected[r.ID] = true
		}
	}

	transByID := make(map[string]dom.TextNode, len(translated.Nodes))
	for _, n := range translated.Nodes {
		transByID[n.ID] = n
	}

	mergedNodes := make([]dom.TextNode, 0, len(restored.Nodes))
	needsReview := 0
	for _, rn := range restored.Nodes {
		if rejectedIDs[rn.ID] {
			continue
		}
		if tn, ok := transByID[rn.ID]; ok {
			rn.TargetText = tn.TargetText
			rn.TargetSegments = tn.TargetSegments
			rn.TargetRubyPairs = tn.TargetRubyPairs
			rn.SourceText = tn.SourceText
			rn.OCRText = tn.OCRText
			rn.RecognitionPending = tn.RecognitionPending
			rn.SpeakerClusterID = tn.SpeakerClusterID
			if tn.Uncertainty.NeedsReview {
				rn.Uncertainty.NeedsReview = true
			}
			if tn.Uncertainty.Score > rn.Uncertainty.Score {
				rn.Uncertainty.Score = tn.Uncertainty.Score
			}
			for _, reason := range tn.Uncertainty.Reasons {
				hasReason := false
				for _, existing := range rn.Uncertainty.Reasons {
					if existing == reason {
						hasReason = true
						break
					}
				}
				if !hasReason {
					rn.Uncertainty.Reasons = append(rn.Uncertainty.Reasons, reason)
				}
			}
		}
		if rn.Uncertainty.NeedsReview {
			needsReview++
		}
		mergedNodes = append(mergedNodes, rn)
	}

	restored.Nodes = mergedNodes
	restored.RecognitionProgress = translated.RecognitionProgress
	restored.TranslationProgress = translated.TranslationProgress
	restored.Metrics.TotalNodes = len(mergedNodes)
	restored.Metrics.NeedsReviewNodes = needsReview
}

func adaptTargetTypography(pageDOM *dom.PageDOM, targetLang string) {
	lang := strings.ToLower(targetLang)
	isLatinTarget := strings.HasPrefix(lang, "en") || strings.HasPrefix(lang, "es") || strings.HasPrefix(lang, "fr") || strings.HasPrefix(lang, "de")
	for i := range pageDOM.Nodes {
		n := &pageDOM.Nodes[i]
		if isLatinTarget {
			n.Style.Orientation = "horizontal"
			if n.Style.FontCategory == "gothic" {
				n.Style.FontCategory = "comic_sans"
			}
		}
		// Page reading order and paragraph extent do not determine writing
		// direction. Preserve the operator's observed row/column orientation
		// for CJK, including wide blocks made from many vertical columns.
	}
}

func resolveRefPNGsForUnits(inputPages []archive.InputPage, units []archive.VirtualUnit, refPages []archive.InputPage) [][]byte {
	out := make([][]byte, len(units))
	if len(refPages) == 0 {
		return out
	}
	if len(refPages) == len(inputPages) {
		cursor := 0
		for uIdx, unit := range units {
			span := len(unit.SeamSlices)
			if span <= 0 {
				span = 1
			}
			if cursor+span <= len(refPages) {
				if span == 1 {
					out[uIdx] = refPages[cursor].PNGBytes
				} else if stitched, _, err := archive.StitchPagesPNG(refPages[cursor : cursor+span]); err == nil {
					out[uIdx] = stitched
				}
			}
			cursor += span
		}
		return out
	}
	if len(refPages) == len(units) {
		for uIdx, unit := range units {
			if len(unit.SeamSlices) == 0 {
				out[uIdx] = refPages[uIdx].PNGBytes
			}
		}
	}
	return out
}
