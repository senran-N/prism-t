package server

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strconv"
	"strings"

	"prism-t/internal/dom"
	"prism-t/internal/engine"
	"prism-t/internal/narrative"
	"prism-t/internal/source"
	"prism-t/internal/store"
)

func (s *Server) sourceClient() *source.Client {
	s.mu.RLock()
	cl := s.srcClient
	s.mu.RUnlock()
	if cl != nil {
		return cl
	}
	return source.NewClient()
}

// ==================== Work & Page Deletion / Rename ====================

func (s *Server) handleDeleteWork(w http.ResponseWriter, r *http.Request) {
	workID, err := strconv.ParseInt(r.PathValue("id"), 10, 64)
	if err != nil {
		writeErr(w, http.StatusBadRequest, "invalid work id")
		return
	}
	if err := s.store.DeleteWork(r.Context(), workID); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			writeErr(w, http.StatusNotFound, "work not found")
			return
		}
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"deleted": true, "work_id": workID})
}

func (s *Server) handleRenameWork(w http.ResponseWriter, r *http.Request) {
	workID, err := strconv.ParseInt(r.PathValue("id"), 10, 64)
	if err != nil {
		writeErr(w, http.StatusBadRequest, "invalid work id")
		return
	}
	var req struct {
		Title string `json:"title"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeErr(w, http.StatusBadRequest, "invalid json: "+err.Error())
		return
	}
	work, err := s.store.RenameWork(r.Context(), workID, req.Title)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			writeErr(w, http.StatusNotFound, "work not found")
			return
		}
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, work)
}

func (s *Server) handleDeletePage(w http.ResponseWriter, r *http.Request) {
	pageID, err := strconv.ParseInt(r.PathValue("id"), 10, 64)
	if err != nil {
		writeErr(w, http.StatusBadRequest, "invalid page id")
		return
	}
	workID, err := s.store.DeletePage(r.Context(), pageID)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			writeErr(w, http.StatusNotFound, "page not found")
			return
		}
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"deleted": true, "page_id": pageID, "work_id": workID})
}

// handleTranslateSingleBox translates one user-drawn bounding box on a stored page using vision LLM.
func (s *Server) handleTranslateSingleBox(w http.ResponseWriter, r *http.Request) {
	pageID, err := strconv.ParseInt(r.PathValue("id"), 10, 64)
	if err != nil {
		writeErr(w, http.StatusBadRequest, "invalid page id")
		return
	}
	if !s.compiler.HasBackend() {
		writeErr(w, http.StatusPreconditionFailed, narrative.ErrNoTranslationBackend.Error())
		return
	}
	var req struct {
		BBox        []int  `json:"bbox"`
		Orientation string `json:"orientation"`
		TargetLang  string `json:"target_lang"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeErr(w, http.StatusBadRequest, "invalid json: "+err.Error())
		return
	}
	if len(req.BBox) != 4 || req.BBox[2] < 6 || req.BBox[3] < 6 {
		writeErr(w, http.StatusBadRequest, "bbox must be [x, y, w, h] with w,h >= 6")
		return
	}
	rec, err := s.store.GetPage(r.Context(), pageID)
	if err != nil {
		writeErr(w, http.StatusNotFound, err.Error())
		return
	}
	lang := strings.TrimSpace(req.TargetLang)
	if lang == "" {
		lang = s.currentConfig().TargetLang
	}
	orient := strings.TrimSpace(req.Orientation)
	if orient == "" {
		if req.BBox[3] > req.BBox[2] {
			orient = "vertical"
		} else {
			orient = "horizontal"
		}
	}
	tempDOM := rec.DOM
	tempDOM.Nodes = []dom.TextNode{{
		ID:           "manual_probe",
		ReadingOrder: 0,
		BBox:         req.BBox,
		BubbleKind:   "speech_oval",
		Style: dom.StyleSpec{
			Orientation:  orient,
			FontCategory: "gothic",
		},
	}}
	if err := s.compiler.CompilePageTranslations(
		r.Context(),
		s.store,
		rec.Summary.WorkID,
		rec.Summary.ChapterTitle,
		rec.Summary.PageIndex,
		lang,
		&tempDOM,
		rec.RawPNG,
	); err != nil {
		writeErr(w, http.StatusBadGateway, err.Error())
		return
	}
	if len(tempDOM.Nodes) == 0 {
		writeJSON(w, http.StatusOK, map[string]any{
			"is_text":     false,
			"source_text": "",
			"target_text": "",
		})
		return
	}
	n := tempDOM.Nodes[0]
	writeJSON(w, http.StatusOK, map[string]any{
		"is_text":         true,
		"source_text":     n.SourceText,
		"target_text":     n.TargetText,
		"target_segments": n.TargetSegments,
	})
}

// ==================== Comic Sources CRUD & Probe ====================

func (s *Server) handleListSources(w http.ResponseWriter, r *http.Request) {
	saved, err := s.store.ListComicSources(r.Context())
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	if saved == nil {
		saved = []store.ComicSource{}
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"sources": saved,
		"builtin": []map[string]any{
			{
				"kind":        "mangadex",
				"name":        "MangaDex 公开漫源 (API v5)",
				"base_url":    source.DefaultMangaDexBase,
				"description": "按书名搜索或粘贴 MangaDex 漫画/章节链接，支持边下载边翻译。",
			},
			{
				"kind":        "opds",
				"name":        "OPDS / JSON 自定义书源",
				"base_url":    "",
				"description": "兼容 Komga、Kavita、Suwayomi、LANraragi 或自定义 JSON 章节清单。",
			},
			{
				"kind":        "direct_urls",
				"name":        "直链 / 远程 CBZ / 图集网页",
				"base_url":    "",
				"description": "直接粘贴图片 URL 列表、远程 .cbz/.zip 链接或单话图集网页。",
			},
		},
	})
}

func (s *Server) handleCreateSource(w http.ResponseWriter, r *http.Request) {
	var req store.ComicSource
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeErr(w, http.StatusBadRequest, "invalid json: "+err.Error())
		return
	}
	if _, err := source.ValidateHTTPURL(req.BaseURL); err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	created, err := s.store.CreateComicSource(r.Context(), req)
	if err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, created)
}

func (s *Server) handleDeleteSource(w http.ResponseWriter, r *http.Request) {
	id, err := strconv.ParseInt(r.PathValue("id"), 10, 64)
	if err != nil {
		writeErr(w, http.StatusBadRequest, "invalid source id")
		return
	}
	if err := s.store.DeleteComicSource(r.Context(), id); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			writeErr(w, http.StatusNotFound, "source not found")
			return
		}
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"deleted": true, "id": id})
}

func (s *Server) handleProbeSource(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Kind      string            `json:"kind"` // "mangadex_search" | "mangadex_chapters" | "opds" | "direct"
		Query     string            `json:"query"`
		URL       string            `json:"url"`
		MangaID   string            `json:"manga_id"`
		Language  string            `json:"language"`
		Headers   map[string]string `json:"headers"`
		SourceID  int64             `json:"source_id"`
		DataSaver bool              `json:"data_saver"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeErr(w, http.StatusBadRequest, "invalid json: "+err.Error())
		return
	}

	headers := req.Headers
	targetURL := strings.TrimSpace(req.URL)
	if req.SourceID > 0 {
		saved, err := s.store.ListComicSources(r.Context())
		if err == nil {
			for _, cs := range saved {
				if cs.ID == req.SourceID {
					if targetURL == "" {
						targetURL = cs.BaseURL
					}
					if len(headers) == 0 && len(cs.Headers) > 0 {
						headers = cs.Headers
					}
					break
				}
			}
		}
	}

	cl := s.sourceClient()
	var res source.BrowseResult
	var err error
	switch strings.ToLower(strings.TrimSpace(req.Kind)) {
	case "mangadex_search", "mangadex":
		q := strings.TrimSpace(req.Query)
		if q == "" {
			q = targetURL
		}
		res, err = cl.SearchMangaDex(r.Context(), q, 12)
	case "mangadex_chapters":
		id := strings.TrimSpace(req.MangaID)
		if id == "" {
			id = targetURL
		}
		res, err = cl.ListMangaDexChapters(r.Context(), id, req.Language)
	case "opds", "json_manifest":
		if targetURL == "" {
			writeErr(w, http.StatusBadRequest, "url is required")
			return
		}
		res, err = cl.BrowseOPDSOrJSON(r.Context(), targetURL, headers)
	default:
		input := strings.TrimSpace(req.Query)
		if input == "" {
			input = targetURL
		}
		res, err = cl.ResolveDirectInput(r.Context(), input, headers)
	}
	if err != nil {
		writeErr(w, http.StatusBadGateway, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, res)
}

func (s *Server) handleProxyCover(w http.ResponseWriter, r *http.Request) {
	imgURL := strings.TrimSpace(r.URL.Query().Get("url"))
	if imgURL == "" {
		writeErr(w, http.StatusBadRequest, "url query parameter is required")
		return
	}
	data, ct, err := s.sourceClient().ProxyImage(r.Context(), imgURL, nil)
	if err != nil {
		writeErr(w, http.StatusBadGateway, err.Error())
		return
	}
	if ct == "" {
		ct = "image/jpeg"
	}
	w.Header().Set("Content-Type", ct)
	w.Header().Set("Cache-Control", "public, max-age=3600")
	_, _ = w.Write(data)
}

// ==================== Streaming Download-While-Translating Job ====================

func (s *Server) handleStartSourceTranslateJob(w http.ResponseWriter, r *http.Request) {
	var req struct {
		SourceKind        string             `json:"source_kind"` // "mangadex" | "opds" | "json_manifest" | "direct_urls" | "direct_archive" | "direct_gallery"
		WorkTitle         string             `json:"work_title"`
		ChapterTitle      string             `json:"chapter_title"`
		FormatProfile     string             `json:"format_profile"`
		TargetLang        string             `json:"target_lang"`
		AnalyzeOnly       bool               `json:"analyze_only"`
		MangaDexChapterID string             `json:"mangadex_chapter_id"`
		DataSaver         bool               `json:"data_saver"`
		ArchiveURL        string             `json:"archive_url"`
		DirectInput       string             `json:"direct_input"`
		Pages             []source.PageAsset `json:"pages"`
		Headers           map[string]string  `json:"headers"`
		SourceID          int64              `json:"source_id"`
		MaxPages          int                `json:"max_pages"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeErr(w, http.StatusBadRequest, "invalid json: "+err.Error())
		return
	}
	cfg := s.currentConfig()
	if !req.AnalyzeOnly && !s.compiler.HasBackend() {
		writeErr(w, http.StatusPreconditionFailed, narrative.ErrNoTranslationBackend.Error())
		return
	}
	if strings.TrimSpace(req.WorkTitle) == "" {
		req.WorkTitle = "在线漫源作品"
	}
	if strings.TrimSpace(req.ChapterTitle) == "" {
		req.ChapterTitle = "Chapter 1"
	}
	if strings.TrimSpace(req.FormatProfile) == "" {
		req.FormatProfile = "auto"
	}
	if strings.TrimSpace(req.TargetLang) == "" {
		req.TargetLang = cfg.TargetLang
	}

	headers := req.Headers
	if req.SourceID > 0 && len(headers) == 0 {
		if saved, err := s.store.ListComicSources(r.Context()); err == nil {
			for _, cs := range saved {
				if cs.ID == req.SourceID {
					headers = cs.Headers
					break
				}
			}
		}
	}

	cl := s.sourceClient()
	pages := req.Pages
	archiveURL := strings.TrimSpace(req.ArchiveURL)
	srcKind := strings.TrimSpace(req.SourceKind)

	// Resolve page URLs upfront (metadata-only, fast!) so we know exact total count before streaming downloads.
	if len(pages) == 0 && archiveURL == "" {
		switch {
		case strings.TrimSpace(req.MangaDexChapterID) != "":
			resolved, err := cl.ResolveMangaDexChapterPages(r.Context(), req.MangaDexChapterID, req.DataSaver)
			if err != nil {
				writeErr(w, http.StatusBadGateway, err.Error())
				return
			}
			pages = resolved
			if srcKind == "" {
				srcKind = "mangadex"
			}
		case strings.TrimSpace(req.DirectInput) != "":
			res, err := cl.ResolveDirectInput(r.Context(), req.DirectInput, headers)
			if err != nil {
				writeErr(w, http.StatusBadGateway, err.Error())
				return
			}
			if res.ArchiveURL != "" {
				archiveURL = res.ArchiveURL
				srcKind = res.SourceKind
			} else if len(res.Pages) > 0 {
				pages = res.Pages
				srcKind = res.SourceKind
			} else {
				writeErr(w, http.StatusBadRequest, "未能从该输入中解析出可下载的页面列表")
				return
			}
		default:
			writeErr(w, http.StatusBadRequest, "请提供章节 ID、图片列表或归档链接")
			return
		}
	}

	if req.MaxPages > 0 && len(pages) > req.MaxPages {
		pages = pages[:req.MaxPages]
	}

	totalExpected := len(pages)
	if totalExpected == 0 && archiveURL != "" {
		totalExpected = 1
	}

	jobCtx, cancel := context.WithCancel(context.Background())
	snap := s.jobs.Start(req.WorkTitle, req.ChapterTitle, totalExpected, cancel)
	if srcKind == "" {
		srcKind = "online_source"
	}
	s.jobs.SetSourceKind(snap.ID, srcKind)

	go func() {
		defer cancel()
		var work store.Work
		var results []engine.TranslatedUnitResult
		var runErr error

		if archiveURL != "" {
			s.jobs.Update(snap.ID, engine.TranslateProgressEvent{
				Stage:   "downloading",
				Message: fmt.Sprintf("正在下载远程漫画包: %s", archiveURL),
			})
			inputPages, dlErr := cl.DownloadArchivePages(jobCtx, archiveURL, headers, req.MaxPages)
			if dlErr != nil {
				s.jobs.Finish(snap.ID, 0, nil, dlErr)
				return
			}
			s.jobs.MarkDownloaded(snap.ID, len(inputPages), len(inputPages), inputPages[0].Filename)
			work, results, runErr = s.orch.TranslatePages(jobCtx, inputPages, engine.TranslateBatchOptions{
				WorkTitle:     req.WorkTitle,
				ChapterTitle:  req.ChapterTitle,
				FormatProfile: req.FormatProfile,
				TargetLang:    req.TargetLang,
				AnalyzeOnly:   req.AnalyzeOnly,
				OnProgress: func(ev engine.TranslateProgressEvent) {
					s.jobs.Update(snap.ID, ev)
				},
			})
		} else {
			// True streaming download-while-translating: download up to 3 pages ahead while page 1
			// enters layout detection, restoration, LLM translation, and rendering immediately!
			rawStream := cl.StreamPages(jobCtx, pages, headers, 3, func(downloaded, total int, filename string) {
				s.jobs.MarkDownloaded(snap.ID, downloaded, total, filename)
			})
			adapted := make(chan engine.StreamPageItem, 3)
			go func() {
				defer close(adapted)
				for item := range rawStream {
					select {
					case <-jobCtx.Done():
						return
					case adapted <- engine.StreamPageItem{
						Index: item.Index,
						Total: item.Total,
						Page:  item.Page,
						Err:   item.Err,
					}:
					}
				}
			}()
			work, results, runErr = s.orch.TranslatePageStream(jobCtx, len(pages), adapted, engine.TranslateBatchOptions{
				WorkTitle:     req.WorkTitle,
				ChapterTitle:  req.ChapterTitle,
				FormatProfile: req.FormatProfile,
				TargetLang:    req.TargetLang,
				AnalyzeOnly:   req.AnalyzeOnly,
				OnProgress: func(ev engine.TranslateProgressEvent) {
					s.jobs.Update(snap.ID, ev)
				},
			})
		}

		pageIDs := make([]int64, 0, len(results))
		for _, r := range results {
			pageIDs = append(pageIDs, r.PageID)
		}
		s.jobs.Finish(snap.ID, work.ID, pageIDs, runErr)
	}()

	writeJSON(w, http.StatusAccepted, map[string]any{"job": snap})
}
