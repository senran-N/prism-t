package server

import (
	"bytes"
	"context"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"fmt"
	"html"
	"io"
	"mime/multipart"
	"net"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"prism-t/internal/archive"
	"prism-t/internal/config"
	"prism-t/internal/dom"
	"prism-t/internal/engine"
	"prism-t/internal/narrative"
	"prism-t/internal/psd"
	"prism-t/internal/source"
	"prism-t/internal/store"
)

type Server struct {
	mu        sync.RWMutex
	cfg       config.Config
	store     *store.Store
	client    *engine.Client
	sup       *engine.Supervisor
	orch      *engine.Orchestrator
	compiler  *narrative.Compiler
	jobs      *jobManager
	srcClient *source.Client
	webDist   string
}

func New(cfg config.Config, st *store.Store, client *engine.Client, orch *engine.Orchestrator, comp *narrative.Compiler, webDist string) *Server {
	return &Server{
		cfg:       cfg,
		store:     st,
		client:    client,
		orch:      orch,
		compiler:  comp,
		jobs:      newJobManager(),
		srcClient: source.NewClient(),
		webDist:   webDist,
	}
}

// SetSupervisor attaches the engine process supervisor so Web Studio can hot-restart the engine on device switches.
func (s *Server) SetSupervisor(sup *engine.Supervisor) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.sup = sup
}

func (s *Server) currentConfig() config.Config {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.cfg
}

func (s *Server) setConfig(cfg config.Config) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.cfg = cfg
}

func (s *Server) supervisor() *engine.Supervisor {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.sup
}

func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()

	mux.HandleFunc("GET /api/v1/health", s.handleHealth)
	mux.HandleFunc("GET /api/v1/stats", s.handleStats)
	mux.HandleFunc("GET /api/v1/works", s.handleListWorks)
	mux.HandleFunc("GET /api/v1/pages", s.handleListPages)
	mux.HandleFunc("GET /api/v1/pages/{id}/dom", s.handleGetPageDOM)
	mux.HandleFunc("GET /api/v1/pages/{id}/raw", s.handleGetPageRaw)
	mux.HandleFunc("GET /api/v1/pages/{id}/composite", s.handleGetPageComposite)
	mux.HandleFunc("GET /api/v1/pages/{id}/export/psd", s.handleExportPagePSD)
	mux.HandleFunc("GET /api/v1/works/{id}/export/cbz", s.handleExportWorkCBZ)
	mux.HandleFunc("GET /api/v1/works/{id}/personas", s.handleGetPersonas)
	mux.HandleFunc("GET /api/v1/jobs", s.handleListJobs)
	mux.HandleFunc("GET /api/v1/jobs/{id}", s.handleGetJob)
	mux.HandleFunc("GET /api/v1/sources", s.handleListSources)
	mux.HandleFunc("GET /api/v1/sources/cover", s.handleProxyCover)

	// Protected write / studio mutation & settings routes (require admin token)
	mux.HandleFunc("GET /api/v1/admin/auth", s.requireAuth(s.handleAuthCheck))
	mux.HandleFunc("GET /api/v1/settings", s.requireAuth(s.handleGetSettings))
	mux.HandleFunc("POST /api/v1/settings", s.requireAuth(s.handleUpdateSettings))
	mux.HandleFunc("POST /api/v1/settings/test-llm", s.requireAuth(s.handleTestLLM))
	mux.HandleFunc("POST /api/v1/translate", s.requireAuth(s.handleTranslateUpload))
	mux.HandleFunc("POST /api/v1/jobs/translate", s.requireAuth(s.handleStartTranslateJob))
	mux.HandleFunc("POST /api/v1/jobs/source-translate", s.requireAuth(s.handleStartSourceTranslateJob))
	mux.HandleFunc("POST /api/v1/jobs/{id}/cancel", s.requireAuth(s.handleCancelJob))
	mux.HandleFunc("POST /api/v1/pages/{id}/translate", s.requireAuth(s.handleRetranslatePage))
	mux.HandleFunc("POST /api/v1/pages/{id}/translate-box", s.requireAuth(s.handleTranslateSingleBox))
	mux.HandleFunc("POST /api/v1/pages/{id}/rerender", s.requireAuth(s.handleRerenderPage))
	mux.HandleFunc("DELETE /api/v1/pages/{id}", s.requireAuth(s.handleDeletePage))
	mux.HandleFunc("PATCH /api/v1/works/{id}", s.requireAuth(s.handleRenameWork))
	mux.HandleFunc("DELETE /api/v1/works/{id}", s.requireAuth(s.handleDeleteWork))
	mux.HandleFunc("POST /api/v1/works/{id}/glossary", s.requireAuth(s.handleUpsertGlossary))
	mux.HandleFunc("POST /api/v1/works/{id}/characters", s.requireAuth(s.handleUpsertCharacter))
	mux.HandleFunc("POST /api/v1/works/{id}/relations", s.requireAuth(s.handleUpsertRelation))
	mux.HandleFunc("POST /api/v1/sources", s.requireAuth(s.handleCreateSource))
	mux.HandleFunc("DELETE /api/v1/sources/{id}", s.requireAuth(s.handleDeleteSource))
	mux.HandleFunc("POST /api/v1/sources/probe", s.requireAuth(s.handleProbeSource))

	mux.HandleFunc("/", s.handleStaticFrontend)
	return s.withCORS(mux)
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	cfg := s.currentConfig()
	compCfg := s.compiler.Config()
	engHealth, err := s.client.Health(r.Context())
	engOK := err == nil
	llmReady := s.compiler.HasBackend()
	llmModel := ""
	if llmReady {
		llmModel = compCfg.LLMModel
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"status":               "ok",
		"engine_connected":     engOK,
		"engine":               engHealth,
		"device":               cfg.Device,
		"max_concurrency":      cfg.MaxConcurrency,
		"llm_configured":       llmReady,
		"llm_model":            llmModel,
		"llm_vision":           compCfg.LLMVision,
		"llm_reasoning_effort": config.FormatReasoningEffortSetting(compCfg.LLMReasoningEffort),
		"default_target_lang":  cfg.TargetLang,
	})
}

func (s *Server) handleAuthCheck(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{"authorized": true})
}

func (s *Server) handleStats(w http.ResponseWriter, r *http.Request) {
	st, err := s.store.GetStats(r.Context())
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, st)
}

func (s *Server) handleListWorks(w http.ResponseWriter, r *http.Request) {
	works, err := s.store.ListWorks(r.Context())
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	if works == nil {
		works = []store.Work{}
	}
	writeJSON(w, http.StatusOK, map[string]any{"works": works})
}

func (s *Server) handleListPages(w http.ResponseWriter, r *http.Request) {
	workID, _ := strconv.ParseInt(r.URL.Query().Get("work_id"), 10, 64)
	onlyRev := r.URL.Query().Get("needs_review") == "1" || r.URL.Query().Get("needs_review") == "true"
	pages, err := s.store.ListPages(r.Context(), workID, onlyRev)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	if pages == nil {
		pages = []store.PageSummary{}
	}
	writeJSON(w, http.StatusOK, map[string]any{"pages": pages})
}

func (s *Server) handleGetPageDOM(w http.ResponseWriter, r *http.Request) {
	id, err := strconv.ParseInt(r.PathValue("id"), 10, 64)
	if err != nil || id <= 0 {
		writeErr(w, http.StatusBadRequest, "invalid page id")
		return
	}
	rec, err := s.store.GetPage(r.Context(), id)
	if err != nil {
		writeErr(w, http.StatusNotFound, "page not found")
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"summary": rec.Summary,
		"dom":     rec.DOM,
	})
}

func (s *Server) handleGetPageRaw(w http.ResponseWriter, r *http.Request) {
	id, err := strconv.ParseInt(r.PathValue("id"), 10, 64)
	if err != nil || id <= 0 {
		writeErr(w, http.StatusBadRequest, "invalid page id")
		return
	}
	rec, err := s.store.GetPage(r.Context(), id)
	if err != nil {
		writeErr(w, http.StatusNotFound, "page not found")
		return
	}
	w.Header().Set("Content-Type", "image/png")
	w.Header().Set("Cache-Control", "no-cache")
	_, _ = w.Write(rec.RawPNG)
}

func (s *Server) handleGetPageComposite(w http.ResponseWriter, r *http.Request) {
	id, err := strconv.ParseInt(r.PathValue("id"), 10, 64)
	if err != nil || id <= 0 {
		writeErr(w, http.StatusBadRequest, "invalid page id")
		return
	}
	rec, err := s.store.GetPage(r.Context(), id)
	if err != nil {
		writeErr(w, http.StatusNotFound, "page not found")
		return
	}
	w.Header().Set("Content-Type", "image/png")
	w.Header().Set("Cache-Control", "no-cache")
	if dl := r.URL.Query().Get("download"); dl == "1" || dl == "true" {
		dlName := strings.TrimSuffix(rec.Summary.Filename, filepath.Ext(rec.Summary.Filename)) + "_translated.png"
		w.Header().Set("Content-Disposition", fmt.Sprintf("attachment; filename=%q", dlName))
	}
	_, _ = w.Write(rec.CompositePNG)
}

func (s *Server) handleExportPagePSD(w http.ResponseWriter, r *http.Request) {
	id, err := strconv.ParseInt(r.PathValue("id"), 10, 64)
	if err != nil || id <= 0 {
		writeErr(w, http.StatusBadRequest, "invalid page id")
		return
	}
	rec, err := s.store.GetPage(r.Context(), id)
	if err != nil {
		writeErr(w, http.StatusNotFound, "page not found")
		return
	}
	doc, err := psd.BuildFromComicDOM(rec.RawPNG, rec.CompositePNG, rec.DOM)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	psdBytes, err := psd.Encode(doc)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	baseName := strings.TrimSuffix(rec.Summary.Filename, filepath.Ext(rec.Summary.Filename)) + ".psd"
	w.Header().Set("Content-Type", "image/vnd.adobe.photoshop")
	w.Header().Set("Content-Disposition", fmt.Sprintf("attachment; filename=%q", baseName))
	_, _ = w.Write(psdBytes)
}

func (s *Server) handleExportWorkCBZ(w http.ResponseWriter, r *http.Request) {
	workID, err := strconv.ParseInt(r.PathValue("id"), 10, 64)
	if err != nil || workID <= 0 {
		writeErr(w, http.StatusBadRequest, "invalid work id")
		return
	}
	summaries, err := s.store.ListPages(r.Context(), workID, false)
	if err != nil || len(summaries) == 0 {
		writeErr(w, http.StatusNotFound, "no pages found for work")
		return
	}
	tmpFile, err := os.CreateTemp("", "prism_t_export_*.cbz")
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	tmpPath := tmpFile.Name()
	_ = tmpFile.Close()
	defer os.Remove(tmpPath)

	entries := make([]archive.OutputEntry, 0, len(summaries))
	for ordinal, sm := range summaries {
		rec, gErr := s.store.GetPage(r.Context(), sm.ID)
		if gErr != nil {
			writeErr(w, http.StatusInternalServerError, gErr.Error())
			return
		}
		if len(rec.DOM.SeamSlices) > 0 {
			sliced, sErr := archive.SliceCompositePNG(rec.CompositePNG, rec.DOM.SeamSlices, sm.Filename)
			if sErr != nil {
				writeErr(w, http.StatusInternalServerError, sErr.Error())
				return
			}
			for sIdx, sl := range sliced {
				sl.Filename = fmt.Sprintf("%06d_%02d_%s", ordinal+1, sIdx+1, sl.Filename)
				entries = append(entries, sl)
			}
		} else {
			entries = append(entries, archive.OutputEntry{
				Filename: fmt.Sprintf("%06d_%s.png", ordinal+1, strings.TrimSuffix(sm.Filename, filepath.Ext(sm.Filename))),
				Data:     rec.CompositePNG,
			})
		}
	}
	if err := archive.WriteCBZ(tmpPath, entries); err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	cbzBytes, err := os.ReadFile(tmpPath)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	w.Header().Set("Content-Type", "application/vnd.comicbook+zip")
	w.Header().Set("Content-Disposition", fmt.Sprintf("attachment; filename=\"work_%d_localized.cbz\"", workID))
	_, _ = w.Write(cbzBytes)
}

type parsedTranslateForm struct {
	pages         []archive.InputPage
	refPages      []archive.InputPage
	workTitle     string
	chapterTitle  string
	formatProfile string
	targetLang    string
	analyzeOnly   bool
}

func (s *Server) parseTranslateUploadForm(w http.ResponseWriter, r *http.Request) (parsedTranslateForm, bool) {
	if err := r.ParseMultipartForm(256 << 20); err != nil {
		writeErr(w, http.StatusBadRequest, "invalid multipart upload: "+err.Error())
		return parsedTranslateForm{}, false
	}
	workTitle := strings.TrimSpace(r.FormValue("work_title"))
	chapterTitle := strings.TrimSpace(r.FormValue("chapter_title"))
	formatProfile := strings.TrimSpace(r.FormValue("format_profile"))
	targetLang := strings.TrimSpace(r.FormValue("target_lang"))
	analyzeOnly := r.FormValue("analyze_only") == "1" || r.FormValue("analyze_only") == "true"

	// If no LLM backend is configured and user didn't pass analyze_only, return a clear actionable 422!
	if !analyzeOnly && !s.compiler.HasBackend() {
		writeErr(w, http.StatusUnprocessableEntity, narrative.ErrNoTranslationBackend.Error())
		return parsedTranslateForm{}, false
	}

	files := r.MultipartForm.File["files"]
	if len(files) == 0 {
		writeErr(w, http.StatusBadRequest, "no files uploaded under `files`")
		return parsedTranslateForm{}, false
	}

	pages, err := extractUploadedPages(files)
	if err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return parsedTranslateForm{}, false
	}
	var refPages []archive.InputPage
	if refFiles := r.MultipartForm.File["ref_files"]; len(refFiles) > 0 {
		refPages, err = extractUploadedPages(refFiles)
		if err != nil {
			writeErr(w, http.StatusBadRequest, "reference files: "+err.Error())
			return parsedTranslateForm{}, false
		}
	}
	return parsedTranslateForm{
		pages:         pages,
		refPages:      refPages,
		workTitle:     workTitle,
		chapterTitle:  chapterTitle,
		formatProfile: formatProfile,
		targetLang:    targetLang,
		analyzeOnly:   analyzeOnly,
	}, true
}

func (s *Server) handleTranslateUpload(w http.ResponseWriter, r *http.Request) {
	form, ok := s.parseTranslateUploadForm(w, r)
	if !ok {
		return
	}
	work, results, err := s.orch.TranslatePages(r.Context(), form.pages, engine.TranslateBatchOptions{
		WorkTitle:     form.workTitle,
		ChapterTitle:  form.chapterTitle,
		FormatProfile: form.formatProfile,
		TargetLang:    form.targetLang,
		AnalyzeOnly:   form.analyzeOnly,
		RefPages:      form.refPages,
	})
	if err != nil {
		if errors.Is(err, narrative.ErrNoTranslationBackend) {
			writeErr(w, http.StatusUnprocessableEntity, err.Error())
			return
		}
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}

	pageIDs := make([]int64, len(results))
	for i, res := range results {
		pageIDs[i] = res.PageID
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"work":     work,
		"page_ids": pageIDs,
		"count":    len(results),
	})
}

func (s *Server) handleStartTranslateJob(w http.ResponseWriter, r *http.Request) {
	form, ok := s.parseTranslateUploadForm(w, r)
	if !ok {
		return
	}
	jobCtx, cancel := context.WithCancel(context.Background())
	title := form.workTitle
	if title == "" {
		title = "Untitled Comic"
	}
	chap := form.chapterTitle
	if chap == "" {
		chap = "Chapter 1"
	}
	snap := s.jobs.Start(title, chap, len(form.pages), cancel)
	jobID := snap.ID
	go func() {
		work, results, err := s.orch.TranslatePages(jobCtx, form.pages, engine.TranslateBatchOptions{
			WorkTitle:     form.workTitle,
			ChapterTitle:  form.chapterTitle,
			FormatProfile: form.formatProfile,
			TargetLang:    form.targetLang,
			AnalyzeOnly:   form.analyzeOnly,
			RefPages:      form.refPages,
			OnProgress: func(ev engine.TranslateProgressEvent) {
				s.jobs.Update(jobID, ev)
			},
		})
		pageIDs := make([]int64, len(results))
		for i, res := range results {
			pageIDs[i] = res.PageID
		}
		s.jobs.Finish(jobID, work.ID, pageIDs, err)
	}()
	writeJSON(w, http.StatusAccepted, map[string]any{"job": snap})
}

func (s *Server) handleListJobs(w http.ResponseWriter, _ *http.Request) {
	jobs := s.jobs.List()
	var active *JobSnapshot
	for i := range jobs {
		if jobs[i].Status == "running" {
			jCopy := jobs[i]
			active = &jCopy
			break
		}
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"jobs":   jobs,
		"active": active,
	})
}

func (s *Server) handleGetJob(w http.ResponseWriter, r *http.Request) {
	id := strings.TrimSpace(r.PathValue("id"))
	snap, ok := s.jobs.Get(id)
	if !ok {
		writeErr(w, http.StatusNotFound, "job not found")
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"job": snap})
}

func (s *Server) handleCancelJob(w http.ResponseWriter, r *http.Request) {
	id := strings.TrimSpace(r.PathValue("id"))
	snap, ok := s.jobs.Cancel(id)
	if !ok {
		writeErr(w, http.StatusNotFound, "job not found")
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"job": snap})
}

func (s *Server) handleRetranslatePage(w http.ResponseWriter, r *http.Request) {
	id, err := strconv.ParseInt(r.PathValue("id"), 10, 64)
	if err != nil || id <= 0 {
		writeErr(w, http.StatusBadRequest, "invalid page id")
		return
	}
	var req struct {
		TargetLang string `json:"target_lang"`
	}
	if r.Body != nil {
		if dErr := json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(&req); dErr != nil && !errors.Is(dErr, io.EOF) {
			writeErr(w, http.StatusBadRequest, "invalid JSON payload: "+dErr.Error())
			return
		}
	}
	updated, err := s.orch.RetranslateStoredPage(r.Context(), id, req.TargetLang)
	if err != nil {
		if errors.Is(err, narrative.ErrNoTranslationBackend) {
			writeErr(w, http.StatusUnprocessableEntity, err.Error())
			return
		}
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"summary": updated.Summary,
		"dom":     updated.DOM,
	})
}

type settingsUpdateRequest struct {
	config.RuntimeSettings
	PresetProvider string `json:"preset_provider"`
	PresetModel    string `json:"preset_model"`
	ClearAPIKey    bool   `json:"clear_api_key"`
}

func resolveCandidateSettings(curCfg config.Config, req settingsUpdateRequest) (config.Config, error) {
	rs := req.RuntimeSettings
	if strings.TrimSpace(rs.Device) == "" {
		rs.Device = curCfg.Device
	}
	if strings.TrimSpace(req.PresetProvider) != "" && strings.TrimSpace(req.PresetModel) != "" {
		url, key, model, _, err := config.ResolveLocalPresetCredentials(req.PresetProvider, req.PresetModel)
		if err != nil {
			return config.Config{}, err
		}
		if strings.TrimSpace(rs.LLMBaseURL) == "" {
			rs.LLMBaseURL = url
		}
		if strings.TrimSpace(rs.LLMModel) == "" {
			rs.LLMModel = model
		}
		if !req.ClearAPIKey && strings.TrimSpace(rs.LLMAPIKey) == "" {
			rs.LLMAPIKey = key
		}
	}
	if req.ClearAPIKey {
		rs.LLMAPIKey = ""
	} else if strings.TrimSpace(rs.LLMAPIKey) == "" {
		rs.LLMAPIKey = curCfg.LLMAPIKey
	}

	nextCfg := curCfg
	if err := config.ApplyRuntimeSettings(&nextCfg, rs); err != nil {
		return config.Config{}, err
	}
	return nextCfg, nil
}

func (s *Server) buildSettingsPayload(cfg config.Config, warning string) map[string]any {
	rs := config.SettingsFromConfig(cfg)
	hasKey := rs.LLMAPIKey != ""
	maskedKey := config.MaskSecret(rs.LLMAPIKey)
	rs.LLMAPIKey = ""
	return map[string]any{
		"settings":           rs,
		"llm_api_key_set":    hasKey,
		"llm_api_key_masked": maskedKey,
		"engine_managed":     s.supervisor().IsManaged(),
		"settings_file":      config.SettingsPath(cfg.DataDir),
		"local_presets":      config.DiscoverLocalModelPresets(),
		"warning":            warning,
	}
}

func (s *Server) handleGetSettings(w http.ResponseWriter, _ *http.Request) {
	cfg := s.currentConfig()
	writeJSON(w, http.StatusOK, s.buildSettingsPayload(cfg, ""))
}

func (s *Server) handleUpdateSettings(w http.ResponseWriter, r *http.Request) {
	var req settingsUpdateRequest
	if err := json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(&req); err != nil {
		writeErr(w, http.StatusBadRequest, "invalid settings JSON: "+err.Error())
		return
	}
	curCfg := s.currentConfig()
	nextCfg, err := resolveCandidateSettings(curCfg, req)
	if err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}

	needsEngineRestart := !strings.EqualFold(nextCfg.Device, curCfg.Device) || nextCfg.MaxConcurrency != curCfg.MaxConcurrency
	if needsEngineRestart && s.jobs.HasActive() {
		writeErr(w, http.StatusConflict, "当前有翻译任务正在运行，请等待完成或中止后再切换计算设备或并发数")
		return
	}

	warning := ""
	if needsEngineRestart {
		if sup := s.supervisor(); sup != nil {
			if rErr := sup.Restart(r.Context(), nextCfg); rErr != nil {
				if errors.Is(rErr, engine.ErrExternalEngineNotManaged) {
					warning = "计算引擎由外部进程托管，配置已保存，但需手动重启外部引擎后设备切换才生效。"
				} else {
					writeErr(w, http.StatusInternalServerError, "重启本地计算引擎失败: "+rErr.Error())
					return
				}
			}
		}
	}

	if err := config.SaveSettingsFile(config.SettingsPath(nextCfg.DataDir), config.SettingsFromConfig(nextCfg)); err != nil {
		writeErr(w, http.StatusInternalServerError, "保存配置文件失败: "+err.Error())
		return
	}
	s.setConfig(nextCfg)
	s.compiler.UpdateConfig(nextCfg)
	s.orch.UpdateConfig(nextCfg)

	engHealth, engErr := s.client.Health(r.Context())
	payload := s.buildSettingsPayload(nextCfg, warning)
	payload["engine_connected"] = engErr == nil
	payload["engine"] = engHealth
	payload["llm_configured"] = s.compiler.HasBackend()
	writeJSON(w, http.StatusOK, payload)
}

func (s *Server) handleTestLLM(w http.ResponseWriter, r *http.Request) {
	var req settingsUpdateRequest
	if err := json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(&req); err != nil {
		writeErr(w, http.StatusBadRequest, "invalid settings JSON: "+err.Error())
		return
	}
	candCfg, err := resolveCandidateSettings(s.currentConfig(), req)
	if err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()
	latencyMS, usedEffort, probeErr := narrative.ProbeOpenAICompatibleEndpoint(ctx, candCfg)
	if probeErr != nil {
		writeJSON(w, http.StatusUnprocessableEntity, map[string]any{
			"ok":         false,
			"error":      probeErr.Error(),
			"latency_ms": latencyMS,
		})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"ok":               true,
		"latency_ms":       latencyMS,
		"model":            candCfg.LLMModel,
		"reasoning_effort": usedEffort,
	})
}

func extractUploadedPages(headers []*multipart.FileHeader) ([]archive.InputPage, error) {
	var out []archive.InputPage
	for _, fh := range headers {
		ext := strings.ToLower(filepath.Ext(fh.Filename))
		f, err := fh.Open()
		if err != nil {
			return nil, err
		}
		raw, rErr := io.ReadAll(f)
		_ = f.Close()
		if rErr != nil {
			return nil, rErr
		}
		if ext == ".cbz" || ext == ".zip" {
			tmp, tErr := os.CreateTemp("", "prism_upload_*"+ext)
			if tErr != nil {
				return nil, tErr
			}
			tPath := tmp.Name()
			_, _ = tmp.Write(raw)
			_ = tmp.Close()
			zipPages, zErr := archive.ReadInputPages(tPath)
			_ = os.Remove(tPath)
			if zErr != nil {
				return nil, zErr
			}
			out = append(out, zipPages...)
			continue
		}
		pBytes, hash, w, h, err := archive.NormalizeToPNG(raw)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", fh.Filename, err)
		}
		out = append(out, archive.InputPage{
			Filename:   filepath.Base(fh.Filename),
			PNGBytes:   pBytes,
			Blake3Hash: hash,
			Width:      w,
			Height:     h,
		})
	}
	return out, nil
}

func (s *Server) handleRerenderPage(w http.ResponseWriter, r *http.Request) {
	id, err := strconv.ParseInt(r.PathValue("id"), 10, 64)
	if err != nil || id <= 0 {
		writeErr(w, http.StatusBadRequest, "invalid page id")
		return
	}
	var req struct {
		DOM          dom.PageDOM `json:"dom"`
		TargetLang   string      `json:"target_lang"`
		MarkReviewed bool        `json:"mark_reviewed"`
	}
	if err := json.NewDecoder(io.LimitReader(r.Body, 64<<20)).Decode(&req); err != nil {
		writeErr(w, http.StatusBadRequest, "invalid JSON payload: "+err.Error())
		return
	}
	updated, err := s.orch.RerenderStoredPage(r.Context(), id, req.DOM, req.TargetLang, req.MarkReviewed)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"summary": updated.Summary,
		"dom":     updated.DOM,
	})
}

func (s *Server) handleGetPersonas(w http.ResponseWriter, r *http.Request) {
	workID, err := strconv.ParseInt(r.PathValue("id"), 10, 64)
	if err != nil || workID <= 0 {
		writeErr(w, http.StatusBadRequest, "invalid work id")
		return
	}
	chars, _ := s.store.ListCharacters(r.Context(), workID)
	rels, _ := s.store.ListRelations(r.Context(), workID)
	gloss, _ := s.store.ListGlossary(r.Context(), workID)
	if chars == nil {
		chars = []store.Character{}
	}
	if rels == nil {
		rels = []store.CharacterRelation{}
	}
	if gloss == nil {
		gloss = []store.GlossaryTerm{}
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"characters": chars,
		"relations":  rels,
		"glossary":   gloss,
	})
}

func (s *Server) handleUpsertGlossary(w http.ResponseWriter, r *http.Request) {
	workID, err := strconv.ParseInt(r.PathValue("id"), 10, 64)
	if err != nil || workID <= 0 {
		writeErr(w, http.StatusBadRequest, "invalid work id")
		return
	}
	var term store.GlossaryTerm
	if err := json.NewDecoder(r.Body).Decode(&term); err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	term.WorkID = workID
	if strings.TrimSpace(term.SourceTerm) == "" || strings.TrimSpace(term.TargetTerm) == "" {
		writeErr(w, http.StatusBadRequest, "source_term and target_term are required")
		return
	}
	if err := s.store.UpsertGlossaryTerm(r.Context(), term); err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"ok": true})
}

func (s *Server) handleUpsertCharacter(w http.ResponseWriter, r *http.Request) {
	workID, err := strconv.ParseInt(r.PathValue("id"), 10, 64)
	if err != nil || workID <= 0 {
		writeErr(w, http.StatusBadRequest, "invalid work id")
		return
	}
	var ch store.Character
	if err := json.NewDecoder(r.Body).Decode(&ch); err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	ch.WorkID = workID
	id, err := s.store.UpsertCharacter(r.Context(), ch)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"id": id})
}

func (s *Server) handleUpsertRelation(w http.ResponseWriter, r *http.Request) {
	workID, err := strconv.ParseInt(r.PathValue("id"), 10, 64)
	if err != nil || workID <= 0 {
		writeErr(w, http.StatusBadRequest, "invalid work id")
		return
	}
	var rel store.CharacterRelation
	if err := json.NewDecoder(r.Body).Decode(&rel); err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	rel.WorkID = workID
	if err := s.store.UpsertRelation(r.Context(), rel); err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"ok": true})
}

func (s *Server) requireAuth(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		cfg := s.currentConfig()
		if cfg.AdminToken == "" {
			next(w, r)
			return
		}
		tok := strings.TrimSpace(strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer "))
		if tok == "" {
			tok = strings.TrimSpace(r.Header.Get("X-Prism-Token"))
		}
		if subtle.ConstantTimeCompare([]byte(tok), []byte(cfg.AdminToken)) != 1 {
			writeErr(w, http.StatusUnauthorized, "unauthorized: valid X-Prism-Token or Bearer token required")
			return
		}
		next(w, r)
	}
}

func (s *Server) withCORS(next http.Handler) http.Handler {
	cfg := s.currentConfig()
	allowed := make(map[string]struct{}, len(cfg.AllowedOrigins))
	for _, o := range cfg.AllowedOrigins {
		allowed[o] = struct{}{}
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		origin := r.Header.Get("Origin")
		if origin != "" {
			if _, ok := allowed[origin]; ok {
				w.Header().Set("Access-Control-Allow-Origin", origin)
				w.Header().Set("Vary", "Origin")
				w.Header().Set("Access-Control-Allow-Headers", "Authorization, Content-Type, X-Prism-Token")
				w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PATCH, DELETE, OPTIONS")
			}
		}
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
}

func isLoopbackRequest(r *http.Request) bool {
	if strings.TrimSpace(r.Header.Get("Origin")) != "" ||
		strings.TrimSpace(r.Header.Get("X-Forwarded-For")) != "" ||
		strings.TrimSpace(r.Header.Get("Forwarded")) != "" {
		return false
	}
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		host = r.RemoteAddr
	}
	host = strings.Trim(strings.TrimSpace(host), "[]")
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

func injectLocalTokenMeta(htmlBytes []byte, token string) []byte {
	meta := []byte(`<meta name="prism-local-token" content="` + html.EscapeString(token) + `">`)
	lower := bytes.ToLower(htmlBytes)
	if idx := bytes.Index(lower, []byte("</head>")); idx >= 0 {
		out := make([]byte, 0, len(htmlBytes)+len(meta)+1)
		out = append(out, htmlBytes[:idx]...)
		out = append(out, meta...)
		out = append(out, '\n')
		out = append(out, htmlBytes[idx:]...)
		return out
	}
	return append(meta, htmlBytes...)
}

func (s *Server) serveIndexHTML(w http.ResponseWriter, r *http.Request, indexFile string) {
	data, err := os.ReadFile(indexFile)
	if err != nil {
		http.NotFound(w, r)
		return
	}
	cfg := s.currentConfig()
	if cfg.AdminToken != "" && isLoopbackRequest(r) {
		data = injectLocalTokenMeta(data, cfg.AdminToken)
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Header().Set("Cache-Control", "no-store")
	_, _ = w.Write(data)
}

func (s *Server) handleStaticFrontend(w http.ResponseWriter, r *http.Request) {
	if s.webDist == "" {
		writeJSON(w, http.StatusOK, map[string]string{
			"service": "Prism-T Localization Factory",
			"hint":    "Build `web/dist` via `npm --prefix web run build` to mount Web Studio UI.",
		})
		return
	}
	reqPath := filepath.FromSlash(strings.TrimPrefix(path.Clean("/"+r.URL.Path), "/"))
	if reqPath != "" && reqPath != "." && !strings.EqualFold(reqPath, "index.html") {
		full := filepath.Join(s.webDist, reqPath)
		rel, rErr := filepath.Rel(s.webDist, full)
		if rErr == nil && rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator)) && !filepath.IsAbs(rel) {
			if info, err := os.Stat(full); err == nil && !info.IsDir() {
				http.ServeFile(w, r, full)
				return
			}
		}
	}
	indexFile := filepath.Join(s.webDist, "index.html")
	if _, err := os.Stat(indexFile); err == nil {
		s.serveIndexHTML(w, r, indexFile)
		return
	}
	http.NotFound(w, r)
}

func writeJSON(w http.ResponseWriter, status int, data any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(data)
}

func writeErr(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]string{"error": msg})
}
