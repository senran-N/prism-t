package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"time"

	"prism-t/internal/archive"
	"prism-t/internal/config"
	"prism-t/internal/engine"
	"prism-t/internal/narrative"
	"prism-t/internal/psd"
	"prism-t/internal/server"
	"prism-t/internal/store"
)

func main() {
	log.SetFlags(log.LstdFlags | log.Lmsgprefix)
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}

	sub := os.Args[1]
	args := os.Args[2:]

	var err error
	switch sub {
	case "serve":
		err = runServe(args)
	case "translate":
		err = runTranslate(args)
	case "retry":
		err = runRetry(args)
	case "export":
		err = runExport(args)
	case "glossary":
		err = runGlossary(args)
	case "stats":
		err = runStats(args)
	case "-h", "--help", "help":
		printUsage()
		return
	default:
		fmt.Fprintf(os.Stderr, "unknown subcommand %q\n\n", sub)
		printUsage()
		os.Exit(1)
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}

func printUsage() {
	fmt.Println(`Prism-T — Lossless Reversible Manga/Webtoon/Comic Localization Factory

Usage:
  prismtd <command> [flags]

Commands:
  serve      Start HTTP API, Web Studio, and local prism-t-engine supervisor
  translate  Localize an image, directory, Webtoon strip folder, or .cbz/.zip
  retry      Translate a stored analysis and render its cached restoration layers
  export     Export stored Comic DOM pages to native layered .psd or .cbz
  glossary   Import or inspect work-level terminology & Gikun Ruby glossary
  stats      Display catalog, persona state machine, and review queue metrics`)
}

func runServe(args []string) error {
	cfg := config.Default()
	fs := flag.NewFlagSet("serve", flag.ContinueOnError)
	fs.StringVar(&cfg.ListenAddr, "listen", cfg.ListenAddr, "HTTP listen address")
	fs.StringVar(&cfg.EngineURL, "engine-url", cfg.EngineURL, "Local prism-t-engine URL")
	fs.StringVar(&cfg.EngineDir, "engine-dir", cfg.EngineDir, "Path to engine Python directory")
	fs.StringVar(&cfg.DBPath, "db", cfg.DBPath, "Path to SQLite WAL database")
	fs.StringVar(&cfg.Device, "device", cfg.Device, "Compute device (auto, cpu, gpu, cuda, dml)")
	fs.StringVar(&cfg.LLMBaseURL, "llm-url", cfg.LLMBaseURL, "OpenAI-compatible LLM base URL")
	fs.StringVar(&cfg.LLMModel, "llm-model", cfg.LLMModel, "LLM model identifier")
	webDist := fs.String("web-dist", filepath.Join("web", "dist"), "Path to built frontend static assets")
	autoOpen := fs.Bool("open", false, "Open Web Studio in default browser once listening")
	if err := fs.Parse(args); err != nil {
		return err
	}

	visited := make(map[string]bool)
	fs.Visit(func(f *flag.Flag) {
		visited[f.Name] = true
	})
	flagDevice := cfg.Device
	flagLLMURL := cfg.LLMBaseURL
	flagLLMModel := cfg.LLMModel

	settingsPath := config.SettingsPath(cfg.DataDir)
	if saved, exists, sErr := config.LoadSettingsFile(settingsPath); sErr != nil {
		return fmt.Errorf("load settings %s: %w", settingsPath, sErr)
	} else if exists {
		if aErr := config.ApplyRuntimeSettings(&cfg, saved); aErr != nil {
			return fmt.Errorf("apply settings %s: %w", settingsPath, aErr)
		}
		if visited["device"] {
			cfg.Device = flagDevice
		}
		if visited["llm-url"] {
			cfg.LLMBaseURL = flagLLMURL
		}
		if visited["llm-model"] {
			cfg.LLMModel = flagLLMModel
		}
		log.Printf("[prismtd] loaded studio settings from %s (device=%s, llm=%s)", settingsPath, cfg.Device, cfg.LLMModel)
	}

	keyPath, err := config.EnsureAdminToken(&cfg)
	if err != nil {
		return err
	}
	if keyPath != "" {
		log.Printf("[prismtd] admin token ready at %s", keyPath)
	}

	st, err := store.Open(cfg.DBPath)
	if err != nil {
		return err
	}
	defer st.Close()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	engClient := engine.NewClient(cfg.EngineURL)
	sup, err := engine.EnsureRunning(ctx, cfg, engClient)
	if err != nil {
		return err
	}
	defer sup.Stop()

	comp := narrative.NewCompiler(cfg)
	orch := engine.NewOrchestrator(cfg, st, engClient, comp)
	srv := server.New(cfg, st, engClient, orch, comp, *webDist)
	srv.SetSupervisor(sup)

	ln, err := net.Listen("tcp", cfg.ListenAddr)
	if err != nil {
		return fmt.Errorf("listen %s: %w", cfg.ListenAddr, err)
	}

	httpSrv := &http.Server{
		Addr:              cfg.ListenAddr,
		Handler:           srv.Handler(),
		ReadHeaderTimeout: 10 * time.Second,
	}

	studioURL := "http://" + browserHostPort(ln.Addr().String())
	errCh := make(chan error, 1)
	go func() {
		log.Printf("[prismtd] listening on %s (engine: %s, device: %s, llm_ready: %v)", studioURL, cfg.EngineURL, cfg.Device, comp.HasBackend())
		if sErr := httpSrv.Serve(ln); sErr != nil && sErr != http.ErrServerClosed {
			errCh <- sErr
		}
	}()
	if *autoOpen {
		if bErr := openBrowserURL(studioURL); bErr != nil {
			log.Printf("[prismtd] unable to open browser automatically (%v); open %s manually", bErr, studioURL)
		}
	}

	select {
	case <-ctx.Done():
		log.Printf("[prismtd] shutting down...")
		shutCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		return httpSrv.Shutdown(shutCtx)
	case sErr := <-errCh:
		return sErr
	}
}

func browserHostPort(addr string) string {
	host, port, err := net.SplitHostPort(addr)
	if err != nil {
		return addr
	}
	if host == "" || host == "0.0.0.0" || host == "::" || host == "[::]" {
		host = "127.0.0.1"
	}
	return net.JoinHostPort(host, port)
}

func openBrowserURL(targetURL string) error {
	var cmd *exec.Cmd
	switch runtime.GOOS {
	case "windows":
		cmd = exec.Command("cmd", "/c", "start", "", targetURL)
	case "darwin":
		cmd = exec.Command("open", targetURL)
	default:
		cmd = exec.Command("xdg-open", targetURL)
	}
	return cmd.Start()
}

func runTranslate(args []string) error {
	cfg := config.Default()
	fs := flag.NewFlagSet("translate", flag.ContinueOnError)
	inPath := fs.String("in", "", "Input image, directory, or .cbz/.zip archive (required)")
	refPath := fs.String("ref", "", "Optional reference edition image/directory/.cbz for cross-edition clean-plate fusion")
	outPath := fs.String("out", "", "Output path (.cbz, directory, or .psd)")
	workTitle := fs.String("work", "Default Work", "Work title in Persona & Lore DB")
	chapterTitle := fs.String("chapter", "Chapter 1", "Chapter title")
	profile := fs.String("profile", "auto", "Format profile (auto, manga_rtl, webtoon_vertical, comic_ltr)")
	targetLang := fs.String("lang", cfg.TargetLang, "Target language (e.g. zh-CN, zh-TW, en, ko)")
	analyzeOnly := fs.Bool("analyze-only", false, "Run lossless de-rendering & clean-plate restoration without LLM translation")
	exportPSD := fs.Bool("psd", false, "Also emit multi-layer .psd files alongside output")
	streamMode := fs.Bool("stream", false, "Stream directory pages one-by-one with constant memory and immediate output persistence")
	resume := fs.Bool("resume", false, "Resume directory translation by skipping pages already completed in DB (implies --stream)")
	continueOnError := fs.Bool("continue-on-error", false, "Log per-page errors and continue translating remaining pages in stream mode")
	fs.StringVar(&cfg.Device, "device", cfg.Device, "Compute device (auto, cpu, gpu, cuda, dml)")
	fs.StringVar(&cfg.DBPath, "db", cfg.DBPath, "Path to SQLite WAL database")
	fs.StringVar(&cfg.EngineURL, "engine-url", cfg.EngineURL, "Local prism-t-engine URL")
	fs.StringVar(&cfg.EngineDir, "engine-dir", cfg.EngineDir, "Path to engine Python directory")
	fs.StringVar(&cfg.LLMBaseURL, "llm-url", cfg.LLMBaseURL, "OpenAI-compatible LLM base URL")
	fs.StringVar(&cfg.LLMModel, "llm-model", cfg.LLMModel, "LLM model identifier")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if strings.TrimSpace(*inPath) == "" {
		return fmt.Errorf("--in is required")
	}

	comp := narrative.NewCompiler(cfg)
	if !*analyzeOnly && !comp.HasBackend() {
		return narrative.ErrNoTranslationBackend
	}

	if *resume || *streamMode || *continueOnError {
		info, statErr := os.Stat(*inPath)
		if statErr != nil {
			return statErr
		}
		if !info.IsDir() {
			return fmt.Errorf("--stream / --resume requires --in to be a directory")
		}
		if strings.TrimSpace(*refPath) != "" {
			return fmt.Errorf("--stream / --resume does not support --ref")
		}
		outExt := strings.ToLower(filepath.Ext(*outPath))
		if outExt == ".cbz" || outExt == ".psd" {
			return fmt.Errorf("--stream / --resume requires --out to be a directory (or empty), got %q", *outPath)
		}
		return runTranslateStreamDirectory(cfg, comp, *inPath, *outPath, *workTitle, *chapterTitle, *profile, *targetLang, *analyzeOnly, *exportPSD, *resume, *continueOnError)
	}

	pages, err := archive.ReadInputPages(*inPath)
	if err != nil {
		return err
	}
	var refPages []archive.InputPage
	if strings.TrimSpace(*refPath) != "" {
		refPages, err = archive.ReadInputPages(*refPath)
		if err != nil {
			return fmt.Errorf("load --ref: %w", err)
		}
	}

	st, err := store.Open(cfg.DBPath)
	if err != nil {
		return err
	}
	defer st.Close()

	ctx := context.Background()
	engClient := engine.NewClient(cfg.EngineURL)
	sup, err := engine.EnsureRunning(ctx, cfg, engClient)
	if err != nil {
		return err
	}
	defer sup.Stop()

	orch := engine.NewOrchestrator(cfg, st, engClient, comp)
	work, results, err := orch.TranslatePages(ctx, pages, engine.TranslateBatchOptions{
		WorkTitle:     *workTitle,
		ChapterTitle:  *chapterTitle,
		FormatProfile: *profile,
		TargetLang:    *targetLang,
		AnalyzeOnly:   *analyzeOnly,
		RefPages:      refPages,
	})
	if err != nil {
		return err
	}

	log.Printf("[prismtd] processed %d virtual canvas units for work %q (id=%d)", len(results), work.Title, work.ID)
	if strings.TrimSpace(*outPath) == "" {
		return nil
	}

	ext := strings.ToLower(filepath.Ext(*outPath))
	if ext == ".cbz" {
		var allSlices []archive.OutputEntry
		for _, r := range results {
			allSlices = append(allSlices, r.SliceEntries...)
		}
		if err := archive.WriteCBZ(*outPath, allSlices); err != nil {
			return err
		}
		log.Printf("[prismtd] wrote CBZ archive to %s", *outPath)
		return nil
	}
	if ext == ".psd" && len(results) == 1 {
		rec, gErr := st.GetPage(ctx, results[0].PageID)
		if gErr != nil {
			return gErr
		}
		doc, bErr := psd.BuildFromComicDOM(rec.RawPNG, rec.CompositePNG, rec.DOM)
		if bErr != nil {
			return bErr
		}
		psdBytes, eErr := psd.Encode(doc)
		if eErr != nil {
			return eErr
		}
		if err := os.MkdirAll(filepath.Dir(*outPath), 0o755); err != nil {
			return err
		}
		return os.WriteFile(*outPath, psdBytes, 0o644)
	}

	// Output directory
	if err := os.MkdirAll(*outPath, 0o755); err != nil {
		return err
	}
	for _, r := range results {
		for _, sl := range r.SliceEntries {
			if err := os.WriteFile(filepath.Join(*outPath, sl.Filename), sl.Data, 0o644); err != nil {
				return err
			}
		}
		if *exportPSD {
			rec, gErr := st.GetPage(ctx, r.PageID)
			if gErr != nil {
				return gErr
			}
			doc, bErr := psd.BuildFromComicDOM(rec.RawPNG, rec.CompositePNG, rec.DOM)
			if bErr != nil {
				return bErr
			}
			psdBytes, eErr := psd.Encode(doc)
			if eErr != nil {
				return eErr
			}
			psdName := strings.TrimSuffix(r.Filename, filepath.Ext(r.Filename)) + ".psd"
			if err := os.WriteFile(filepath.Join(*outPath, psdName), psdBytes, 0o644); err != nil {
				return err
			}
		}
	}
	log.Printf("[prismtd] wrote output files to %s", *outPath)
	return nil
}

func writeFileAtomic(dst string, data []byte) error {
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return err
	}
	tmp := dst + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	if err := os.Rename(tmp, dst); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	return nil
}

func runTranslateStreamDirectory(
	cfg config.Config,
	comp *narrative.Compiler,
	inDir, outDir, workTitle, chapterTitle, profile, targetLang string,
	analyzeOnly, exportPSD, resume, continueOnError bool,
) error {
	paths, err := archive.ListDirectoryImagePaths(inDir)
	if err != nil {
		return err
	}
	if len(paths) == 0 {
		return fmt.Errorf("no supported image files found in %q", inDir)
	}
	if strings.TrimSpace(outDir) != "" {
		if err := os.MkdirAll(outDir, 0o755); err != nil {
			return err
		}
	}

	st, err := store.Open(cfg.DBPath)
	if err != nil {
		return err
	}
	defer st.Close()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	work, err := st.EnsureWork(ctx, workTitle, "auto", targetLang, profile)
	if err != nil {
		return err
	}

	type pendingEntry struct {
		index int
		path  string
	}
	var pending []pendingEntry
	skipped := 0
	backfilled := 0

	completedByFilename := make(map[string]store.PageSummary)
	if resume {
		existing, lErr := st.ListPages(ctx, work.ID, false)
		if lErr != nil {
			return lErr
		}
		for _, pg := range existing {
			if pg.ChapterTitle != chapterTitle {
				continue
			}
			isDone := pg.Status == "translated" || pg.Status == "reviewed" || (analyzeOnly && pg.Status == "analyzed")
			if isDone {
				completedByFilename[pg.Filename] = pg
			}
		}
	}

	for idx, p := range paths {
		base := filepath.Base(p)
		if pg, ok := completedByFilename[base]; ok {
			if strings.TrimSpace(outDir) != "" {
				outPNG := filepath.Join(outDir, archive.EnsurePNGExt(base))
				needPNG := true
				if fi, sErr := os.Stat(outPNG); sErr == nil && fi.Size() > 0 {
					needPNG = false
				}
				psdName := strings.TrimSuffix(base, filepath.Ext(base)) + ".psd"
				outPSD := filepath.Join(outDir, psdName)
				needPSD := exportPSD
				if needPSD {
					if fi, sErr := os.Stat(outPSD); sErr == nil && fi.Size() > 0 {
						needPSD = false
					}
				}
				if needPNG || needPSD {
					rec, gErr := st.GetPage(ctx, pg.ID)
					if gErr != nil {
						return fmt.Errorf("load completed page %s (id=%d): %w", base, pg.ID, gErr)
					}
					if needPNG && len(rec.CompositePNG) > 0 {
						if wErr := writeFileAtomic(outPNG, rec.CompositePNG); wErr != nil {
							return wErr
						}
						backfilled++
					} else if needPNG {
						// Stored page has no composite PNG; reprocess it.
						pending = append(pending, pendingEntry{index: idx, path: p})
						continue
					}
					if needPSD {
						doc, bErr := psd.BuildFromComicDOM(rec.RawPNG, rec.CompositePNG, rec.DOM)
						if bErr != nil {
							return bErr
						}
						psdBytes, eErr := psd.Encode(doc)
						if eErr != nil {
							return eErr
						}
						if wErr := writeFileAtomic(outPSD, psdBytes); wErr != nil {
							return wErr
						}
					}
				}
			}
			skipped++
			continue
		}
		pending = append(pending, pendingEntry{index: idx, path: p})
	}

	log.Printf("[prismtd] stream directory %q -> work %q (id=%d): total=%d, skipped=%d (backfilled=%d), pending=%d",
		inDir, work.Title, work.ID, len(paths), skipped, backfilled, len(pending))
	if len(pending) == 0 {
		log.Printf("[prismtd] all %d pages are already completed!", len(paths))
		return nil
	}

	engClient := engine.NewClient(cfg.EngineURL)
	sup, err := engine.EnsureRunning(ctx, cfg, engClient)
	if err != nil {
		return err
	}
	defer sup.Stop()

	stream := make(chan engine.StreamPageItem, 2)
	go func() {
		defer close(stream)
		for _, item := range pending {
			if ctx.Err() != nil {
				return
			}
			page, rErr := archive.ReadInputPageFile(item.path)
			out := engine.StreamPageItem{
				Index: item.index,
				Total: len(paths),
				Page:  page,
				Err:   rErr,
			}
			if out.Page.Filename == "" {
				out.Page.Filename = filepath.Base(item.path)
			}
			select {
			case <-ctx.Done():
				return
			case stream <- out:
			}
		}
	}()

	var failLogFile *os.File
	if continueOnError && strings.TrimSpace(outDir) != "" {
		f, fErr := os.OpenFile(filepath.Join(outDir, "_failed_pages.jsonl"), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
		if fErr != nil {
			return fmt.Errorf("open failure log: %w", fErr)
		}
		failLogFile = f
		defer failLogFile.Close()
	}

	orch := engine.NewOrchestrator(cfg, st, engClient, comp)
	startedAll := time.Now()
	writtenThisRun := 0
	_, results, runErr := orch.TranslatePageStream(ctx, len(paths), stream, engine.TranslateBatchOptions{
		WorkTitle:       workTitle,
		ChapterTitle:    chapterTitle,
		FormatProfile:   profile,
		TargetLang:      targetLang,
		AnalyzeOnly:     analyzeOnly,
		ContinueOnError: continueOnError,
		OnProgress: func(ev engine.TranslateProgressEvent) {
			if ev.Stage == "page_done" {
				log.Printf("[prismtd] [%d/%d] done page #%d (%s) page_id=%d elapsed=%s",
					skipped+ev.CompletedUnits, len(paths), ev.UnitIndex+1, ev.Filename, ev.PageID, time.Since(startedAll).Round(time.Second))
			}
		},
		OnPageDone: func(res engine.TranslatedUnitResult) error {
			if strings.TrimSpace(outDir) == "" {
				return nil
			}
			outPNG := filepath.Join(outDir, archive.EnsurePNGExt(res.Filename))
			if err := writeFileAtomic(outPNG, res.CompositePNG); err != nil {
				return fmt.Errorf("write %s: %w", outPNG, err)
			}
			if exportPSD {
				rec, gErr := st.GetPage(ctx, res.PageID)
				if gErr != nil {
					return gErr
				}
				doc, bErr := psd.BuildFromComicDOM(rec.RawPNG, rec.CompositePNG, rec.DOM)
				if bErr != nil {
					return bErr
				}
				psdBytes, eErr := psd.Encode(doc)
				if eErr != nil {
					return eErr
				}
				psdName := strings.TrimSuffix(res.Filename, filepath.Ext(res.Filename)) + ".psd"
				if err := writeFileAtomic(filepath.Join(outDir, psdName), psdBytes); err != nil {
					return err
				}
			}
			writtenThisRun++
			return nil
		},
		OnPageError: func(index int, filename string, pErr error) {
			log.Printf("[prismtd] FAILED page #%d/%d (%s): %v", index+1, len(paths), filename, pErr)
			if failLogFile != nil {
				rec := map[string]any{
					"index":    index,
					"filename": filename,
					"error":    pErr.Error(),
					"time":     time.Now().UTC().Format(time.RFC3339),
				}
				if line, mErr := json.Marshal(rec); mErr == nil {
					_, _ = failLogFile.Write(append(line, '\n'))
				}
			}
		},
	})
	log.Printf("[prismtd] stream finished: work_id=%d newly_completed=%d written=%d skipped=%d total=%d elapsed=%s",
		work.ID, len(results), writtenThisRun, skipped, len(paths), time.Since(startedAll).Round(time.Second))
	return runErr
}

func runRetry(args []string) error {
	cfg := config.Default()
	fs := flag.NewFlagSet("retry", flag.ContinueOnError)
	pageID := fs.Int64("page-id", 0, "Stored page ID")
	workID := fs.Int64("work-id", 0, "Resume all stored pages in narrative order")
	outPath := fs.String("out", "", "Output directory (required)")
	targetLang := fs.String("lang", cfg.TargetLang, "Target language")
	exportPSD := fs.Bool("psd", false, "Also export layered PSD")
	retranslate := fs.Bool("retranslate", false, "Request a fresh translation even when a completed translation is stored")
	fs.StringVar(&cfg.DBPath, "db", cfg.DBPath, "SQLite database")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if (*pageID > 0) == (*workID > 0) || *outPath == "" {
		return fmt.Errorf("exactly one of --page-id or --work-id, plus --out, is required")
	}
	comp := narrative.NewCompiler(cfg)
	if !comp.HasBackend() {
		return narrative.ErrNoTranslationBackend
	}
	st, err := store.Open(cfg.DBPath)
	if err != nil {
		return err
	}
	defer st.Close()
	ctx := context.Background()
	pageIDs := []int64{*pageID}
	if *workID > 0 {
		pages, err := st.ListPages(ctx, *workID, false)
		if err != nil {
			return err
		}
		if len(pages) == 0 {
			return fmt.Errorf("work %d has no stored pages", *workID)
		}
		pageIDs = make([]int64, len(pages))
		for i, page := range pages {
			pageIDs[i] = page.ID
		}
	}
	client := engine.NewClient(cfg.EngineURL)
	sup, err := engine.EnsureRunning(ctx, cfg, client)
	if err != nil {
		return err
	}
	defer sup.Stop()
	orch := engine.NewOrchestrator(cfg, st, client, comp)
	for ordinal, id := range pageIDs {
		prefix := ""
		if *workID > 0 {
			prefix = fmt.Sprintf("%06d_", ordinal+1)
		}
		if err := retryStoredPage(ctx, st, comp, orch, id, *targetLang, *outPath, prefix, *retranslate, *exportPSD); err != nil {
			return fmt.Errorf("retry page %d: %w", id, err)
		}
	}
	return nil
}

func retryStoredPage(ctx context.Context, st *store.Store, comp *narrative.Compiler, orch *engine.Orchestrator, pageID int64, targetLang, outPath, prefix string, retranslate, exportPSD bool) error {
	rec, err := st.GetPage(ctx, pageID)
	if err != nil {
		return err
	}
	if rec.Summary.Status == "analyzed" || retranslate {
		if err := comp.CompilePageTranslations(ctx, st, rec.Summary.WorkID, rec.Summary.ChapterTitle, rec.Summary.PageIndex, targetLang, &rec.DOM, rec.RawPNG); err != nil {
			return err
		}
		if _, err := st.SavePage(ctx, rec.Summary.WorkID, rec.Summary.ChapterTitle, rec.Summary.PageIndex, rec.Summary.Filename, rec.Blake3Hash, rec.DOM, rec.RawPNG, rec.CompositePNG, "translated"); err != nil {
			return err
		}
	}
	rec, err = orch.RerenderStoredPage(ctx, pageID, rec.DOM, targetLang, false)
	if err != nil {
		return err
	}
	entries, err := archive.SliceCompositePNG(rec.CompositePNG, rec.DOM.SeamSlices, rec.Summary.Filename)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(outPath, 0o755); err != nil {
		return err
	}
	for _, entry := range entries {
		if err := os.WriteFile(filepath.Join(outPath, prefix+entry.Filename), entry.Data, 0o644); err != nil {
			return err
		}
	}
	if exportPSD {
		doc, err := psd.BuildFromComicDOM(rec.RawPNG, rec.CompositePNG, rec.DOM)
		if err != nil {
			return err
		}
		data, err := psd.Encode(doc)
		if err != nil {
			return err
		}
		name := prefix + strings.TrimSuffix(rec.Summary.Filename, filepath.Ext(rec.Summary.Filename)) + ".psd"
		if err := os.WriteFile(filepath.Join(outPath, name), data, 0o644); err != nil {
			return err
		}
	}
	return nil
}

func runExport(args []string) error {
	cfg := config.Default()
	fs := flag.NewFlagSet("export", flag.ContinueOnError)
	pageID := fs.Int64("page-id", 0, "Page ID to export as layered .psd")
	workID := fs.Int64("work-id", 0, "Work ID to export all pages as .cbz or directory of .psd")
	outPath := fs.String("out", "", "Destination file (.psd / .cbz) or directory (required)")
	fs.StringVar(&cfg.DBPath, "db", cfg.DBPath, "Path to SQLite WAL database")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if strings.TrimSpace(*outPath) == "" {
		return fmt.Errorf("--out is required")
	}

	st, err := store.Open(cfg.DBPath)
	if err != nil {
		return err
	}
	defer st.Close()
	ctx := context.Background()

	if *pageID > 0 {
		rec, err := st.GetPage(ctx, *pageID)
		if err != nil {
			return err
		}
		doc, err := psd.BuildFromComicDOM(rec.RawPNG, rec.CompositePNG, rec.DOM)
		if err != nil {
			return err
		}
		b, err := psd.Encode(doc)
		if err != nil {
			return err
		}
		if err := os.MkdirAll(filepath.Dir(*outPath), 0o755); err != nil {
			return err
		}
		return os.WriteFile(*outPath, b, 0o644)
	}

	if *workID <= 0 {
		return fmt.Errorf("specify --page-id or --work-id")
	}
	summaries, err := st.ListPages(ctx, *workID, false)
	if err != nil {
		return err
	}
	if strings.HasSuffix(strings.ToLower(*outPath), ".cbz") {
		var entries []archive.OutputEntry
		for ordinal, sm := range summaries {
			rec, gErr := st.GetPage(ctx, sm.ID)
			if gErr != nil {
				return fmt.Errorf("load page %d: %w", sm.ID, gErr)
			}
			if len(rec.DOM.SeamSlices) > 0 {
				sliced, sErr := archive.SliceCompositePNG(rec.CompositePNG, rec.DOM.SeamSlices, sm.Filename)
				if sErr != nil {
					return fmt.Errorf("slice page %d: %w", sm.ID, sErr)
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
		return archive.WriteCBZ(*outPath, entries)
	}

	if err := os.MkdirAll(*outPath, 0o755); err != nil {
		return err
	}
	for ordinal, sm := range summaries {
		rec, gErr := st.GetPage(ctx, sm.ID)
		if gErr != nil {
			return fmt.Errorf("load page %d: %w", sm.ID, gErr)
		}
		doc, bErr := psd.BuildFromComicDOM(rec.RawPNG, rec.CompositePNG, rec.DOM)
		if bErr != nil {
			return fmt.Errorf("build page %d PSD: %w", sm.ID, bErr)
		}
		b, eErr := psd.Encode(doc)
		if eErr != nil {
			return fmt.Errorf("encode page %d PSD: %w", sm.ID, eErr)
		}
		name := fmt.Sprintf("%06d_%s.psd", ordinal+1, strings.TrimSuffix(sm.Filename, filepath.Ext(sm.Filename)))
		if err := os.WriteFile(filepath.Join(*outPath, name), b, 0o644); err != nil {
			return err
		}
	}
	return nil
}

func runGlossary(args []string) error {
	cfg := config.Default()
	fs := flag.NewFlagSet("glossary", flag.ContinueOnError)
	workTitle := fs.String("work", "", "Work title (required)")
	importPath := fs.String("import", "", "JSON file of glossary entries [{source_term, target_term, ruby_override, notes}]")
	fs.StringVar(&cfg.DBPath, "db", cfg.DBPath, "Path to SQLite WAL database")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if strings.TrimSpace(*workTitle) == "" {
		return fmt.Errorf("--work is required")
	}

	st, err := store.Open(cfg.DBPath)
	if err != nil {
		return err
	}
	defer st.Close()
	ctx := context.Background()

	work, err := st.EnsureWork(ctx, *workTitle, "auto", cfg.TargetLang, "auto")
	if err != nil {
		return err
	}

	if *importPath != "" {
		raw, err := os.ReadFile(*importPath)
		if err != nil {
			return err
		}
		var terms []store.GlossaryTerm
		if err := json.Unmarshal(raw, &terms); err != nil {
			return err
		}
		for _, t := range terms {
			t.WorkID = work.ID
			if err := st.UpsertGlossaryTerm(ctx, t); err != nil {
				return err
			}
		}
		fmt.Printf("Imported %d glossary entries into %q (work_id=%d)\n", len(terms), work.Title, work.ID)
		return nil
	}

	terms, err := st.ListGlossary(ctx, work.ID)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	return enc.Encode(terms)
}

func runStats(args []string) error {
	cfg := config.Default()
	fs := flag.NewFlagSet("stats", flag.ContinueOnError)
	fs.StringVar(&cfg.DBPath, "db", cfg.DBPath, "Path to SQLite WAL database")
	if err := fs.Parse(args); err != nil {
		return err
	}
	st, err := store.Open(cfg.DBPath)
	if err != nil {
		return err
	}
	defer st.Close()

	stats, err := st.GetStats(context.Background())
	if err != nil {
		return err
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	return enc.Encode(stats)
}
