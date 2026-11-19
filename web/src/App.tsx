import { useCallback, useEffect, useRef, useState } from "react";
import type {
  CharacterPersona,
  CharacterRelation,
  GlossaryTerm,
  HealthResponse,
  PageDOM,
  PageSummary,
  RuntimeSettingsResponse,
  TextNode,
  TranslationJob,
  WorkItem,
} from "./types";
import SettingsModal from "./SettingsModal";
import UploadModal from "./components/UploadModal";
import SourceModal from "./components/SourceModal";
import BookshelfView from "./components/BookshelfView";
import PageFilmstrip from "./components/PageFilmstrip";
import CanvasStage from "./components/CanvasStage";
import type { CanvasToolMode, CanvasViewMode } from "./components/CanvasStage";
import InspectorPanel from "./components/InspectorPanel";
import {
  IconBookshelf,
  IconClose,
  IconDownload,
  IconGlobe,
  IconHelp,
  IconPanelLeft,
  IconPanelRight,
  IconPlus,
  IconSettings,
  IconStudio,
} from "./components/Icons";
import { extractFilesFromDataTransfer } from "./utils/folderFiles";

const TOKEN_STORAGE_KEY = "prism_t_admin_token";

type AppView = "bookshelf" | "reader";

interface ToastItem {
  id: number;
  kind: "ok" | "err" | "info";
  text: string;
}

function readInitialToken(): string {
  try {
    const meta = document.querySelector('meta[name="prism-local-token"]');
    const metaTok = meta?.getAttribute("content")?.trim() || "";
    if (metaTok) {
      localStorage.setItem(TOKEN_STORAGE_KEY, metaTok);
      return metaTok;
    }
  } catch {
    // ignore storage failures
  }
  const hash = window.location.hash || "";
  const prefix = "#token=";
  if (hash.startsWith(prefix)) {
    const tok = decodeURIComponent(hash.slice(prefix.length)).trim();
    if (tok) {
      try {
        localStorage.setItem(TOKEN_STORAGE_KEY, tok);
      } catch {
        // ignore
      }
      window.history.replaceState(null, "", window.location.pathname + window.location.search);
      return tok;
    }
  }
  try {
    return localStorage.getItem(TOKEN_STORAGE_KEY) || "";
  } catch {
    return "";
  }
}

function compactDeviceLabel(health: HealthResponse | null): string {
  if (!health?.engine_connected) return "离线";
  const hw = health.engine?.hardware;
  if (!hw) return health.device.toUpperCase();
  if (hw.cuda_available && hw.vram_total_mb > 0) {
    return `${hw.active_device.toUpperCase()} ${(hw.vram_total_mb / 1024).toFixed(0)}G`;
  }
  return `${hw.active_device.toUpperCase()} ${hw.cpu_threads}C`;
}

async function parseJsonSafe(res: Response): Promise<Record<string, any>> {
  const txt = await res.text();
  if (!txt) return {};
  try {
    return JSON.parse(txt);
  } catch {
    return { error: txt.slice(0, 240) };
  }
}

function isEditableElement(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || target.isContentEditable;
}

export default function App() {
  const [token, setToken] = useState<string>(() => readInitialToken());
  const [authAuthorized, setAuthAuthorized] = useState<boolean>(false);
  const [health, setHealth] = useState<HealthResponse | null>(null);

  const [appView, setAppView] = useState<AppView>("bookshelf");
  const [works, setWorks] = useState<WorkItem[]>([]);
  const [selectedWorkId, setSelectedWorkId] = useState<number>(0);
  const [pages, setPages] = useState<PageSummary[]>([]);
  const [selectedPageId, setSelectedPageId] = useState<number>(0);
  const [reviewOnly, setReviewOnly] = useState<boolean>(false);

  const [pageDOM, setPageDOM] = useState<PageDOM | null>(null);
  const [savedDOMSnapshot, setSavedDOMSnapshot] = useState<string>("");
  const [selectedNodeIdx, setSelectedNodeIdx] = useState<number>(0);
  const [cacheBust, setCacheBust] = useState<number>(Date.now());

  const [viewMode, setViewMode] = useState<CanvasViewMode>("composite");
  const [toolMode, setToolMode] = useState<CanvasToolMode>("select");
  const [showOverlay, setShowOverlay] = useState<boolean>(false);
  const [peekingRaw, setPeekingRaw] = useState<boolean>(false);
  const [zenMode, setZenMode] = useState<boolean>(false);
  const [leftOpen, setLeftOpen] = useState<boolean>(true);
  const [rightOpen, setRightOpen] = useState<boolean>(true);

  const [glossary, setGlossary] = useState<GlossaryTerm[]>([]);
  const [personas, setPersonas] = useState<CharacterPersona[]>([]);
  const [relations, setRelations] = useState<CharacterRelation[]>([]);

  const [activeJob, setActiveJob] = useState<TranslationJob | null>(null);
  const [targetLang, setTargetLang] = useState<string>("zh-CN");
  const [busy, setBusy] = useState<boolean>(false);
  const [translatingNode, setTranslatingNode] = useState<boolean>(false);

  const [settingsOpen, setSettingsOpen] = useState<boolean>(false);
  const [uploadOpen, setUploadOpen] = useState<boolean>(false);
  const [sourcesOpen, setSourcesOpen] = useState<boolean>(false);
  const [shortcutsOpen, setShortcutsOpen] = useState<boolean>(false);
  const [droppedFiles, setDroppedFiles] = useState<File[]>([]);
  const [droppedFolderName, setDroppedFolderName] = useState<string>("");
  const [globalDragging, setGlobalDragging] = useState<boolean>(false);
  const [toasts, setToasts] = useState<ToastItem[]>([]);

  const lastAutoOpenedPageRef = useRef<number>(0);
  const lastCompletedJobRef = useRef<string>("");
  const dragCounterRef = useRef<number>(0);
  const toastIdRef = useRef<number>(1);

  const pushToast = useCallback((kind: ToastItem["kind"], text: string) => {
    const id = toastIdRef.current++;
    setToasts((prev) => [...prev.slice(-3), { id, kind, text }]);
    window.setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, 4200);
  }, []);

  const authHeaders = useCallback(
    (json = true): HeadersInit => {
      const h: Record<string, string> = {};
      if (json) h["Content-Type"] = "application/json";
      if (token) h["X-Prism-Token"] = token;
      return h;
    },
    [token]
  );

  const handleSaveToken = useCallback((next: string) => {
    setToken(next);
    try {
      if (next.trim()) {
        localStorage.setItem(TOKEN_STORAGE_KEY, next.trim());
      } else {
        localStorage.removeItem(TOKEN_STORAGE_KEY);
      }
    } catch {
      // ignore
    }
  }, []);

  const isDirty = Boolean(pageDOM && savedDOMSnapshot && JSON.stringify(pageDOM) !== savedDOMSnapshot);

  const fetchHealth = useCallback(async () => {
    try {
      const res = await fetch("/api/v1/health");
      if (res.ok) {
        const h = (await res.json()) as HealthResponse;
        setHealth(h);
        if (h.default_target_lang) {
          setTargetLang((prev) => prev || h.default_target_lang);
        }
      }
    } catch {
      // backend starting
    }
  }, []);

  const checkAuth = useCallback(async () => {
    try {
      const res = await fetch("/api/v1/admin/auth", { headers: authHeaders(false) });
      setAuthAuthorized(res.ok);
    } catch {
      setAuthAuthorized(false);
    }
  }, [authHeaders]);

  const fetchWorks = useCallback(async (): Promise<WorkItem[]> => {
    try {
      const res = await fetch("/api/v1/works");
      if (res.ok) {
        const data = await res.json();
        const raw = Array.isArray(data) ? data : data?.works;
        const safe = Array.isArray(raw) ? (raw as WorkItem[]) : [];
        setWorks(safe);
        return safe;
      }
    } catch {
      // ignore
    }
    return [];
  }, []);

  const fetchPages = useCallback(
    async (workId: number, revOnly: boolean) => {
      if (!workId) {
        setPages([]);
        setSelectedPageId(0);
        setPageDOM(null);
        setSavedDOMSnapshot("");
        return;
      }
      const qs = new URLSearchParams();
      qs.set("work_id", String(workId));
      if (revOnly) qs.set("needs_review", "true");
      try {
        const res = await fetch(`/api/v1/pages?${qs.toString()}`);
        if (res.ok) {
          const data = await res.json();
          const raw = Array.isArray(data) ? data : data?.pages;
          const safe = Array.isArray(raw) ? (raw as PageSummary[]) : [];
          setPages(safe);
          if (safe.length > 0 && !safe.some((p) => p.id === selectedPageId)) {
            setSelectedPageId(safe[0].id);
          } else if (safe.length === 0) {
            setSelectedPageId(0);
            setPageDOM(null);
            setSavedDOMSnapshot("");
          }
        }
      } catch {
        // ignore
      }
    },
    [selectedPageId]
  );

  const fetchPageDetail = useCallback(async (pageId: number) => {
    if (!pageId) return;
    try {
      const res = await fetch(`/api/v1/pages/${pageId}/dom`);
      if (res.ok) {
        const payload = await res.json();
        const data = (payload?.dom ? payload.dom : payload) as PageDOM;
        setPageDOM(data);
        setSavedDOMSnapshot(JSON.stringify(data));
        setSelectedNodeIdx(0);
      }
    } catch {
      // ignore
    }
  }, []);

  const fetchWorkPersonas = useCallback(async (workId: number) => {
    if (!workId) return;
    try {
      const res = await fetch(`/api/v1/works/${workId}/personas`);
      if (res.ok) {
        const data = await res.json();
        setPersonas(data.characters || []);
        setRelations(data.relations || []);
        setGlossary(data.glossary || []);
      }
    } catch {
      // ignore
    }
  }, []);

  const fetchActiveJobs = useCallback(async () => {
    try {
      const res = await fetch("/api/v1/jobs");
      if (!res.ok) return;
      const data = await res.json();
      const jobs = (data.jobs || []) as TranslationJob[];
      const running = jobs.find((j) => j.status === "running");
      if (running) setActiveJob(running);
    } catch {
      // ignore
    }
  }, []);

  useEffect(() => {
    void fetchHealth();
    void fetchWorks();
    void fetchActiveJobs();
  }, [fetchHealth, fetchWorks, fetchActiveJobs]);

  useEffect(() => {
    void checkAuth();
  }, [checkAuth]);

  useEffect(() => {
    if (selectedWorkId > 0) {
      void fetchPages(selectedWorkId, reviewOnly);
      void fetchWorkPersonas(selectedWorkId);
    }
  }, [selectedWorkId, reviewOnly, fetchPages, fetchWorkPersonas]);

  useEffect(() => {
    if (selectedPageId > 0) {
      void fetchPageDetail(selectedPageId);
    }
  }, [selectedPageId, fetchPageDetail]);

  // Poll live background job (supports streaming download-while-translating!)
  useEffect(() => {
    if (!activeJob || activeJob.status !== "running") return;
    let cancelled = false;
    const timer = window.setInterval(async () => {
      try {
        const res = await fetch(`/api/v1/jobs/${activeJob.id}`);
        if (!res.ok || cancelled) return;
        const data = await res.json();
        const job = data.job as TranslationJob;
        if (!job || cancelled) return;
        setActiveJob(job);

        const pageCount = job.page_ids ? job.page_ids.length : 0;
        if (pageCount > 0) {
          const latestPage = job.page_ids[pageCount - 1];
          if (latestPage !== lastAutoOpenedPageRef.current) {
            lastAutoOpenedPageRef.current = latestPage;
            await fetchWorks();
            if (job.work_id > 0) {
              if (selectedWorkId === job.work_id || appView === "reader") {
                await fetchPages(job.work_id, reviewOnly);
              }
              if (pageCount === 1) {
                setSelectedWorkId(job.work_id);
                setSelectedPageId(job.page_ids[0]);
                setCacheBust(Date.now());
                setAppView("reader");
                pushToast("ok", `《${job.work_title}》第 1 页已译完，现在即可边看边等！`);
              } else if (selectedWorkId === job.work_id) {
                setCacheBust(Date.now());
              }
            }
          }
        }

        if (job.status !== "running" && lastCompletedJobRef.current !== job.id) {
          lastCompletedJobRef.current = job.id;
          await fetchWorks();
          if (job.work_id > 0) {
            await fetchPages(job.work_id, reviewOnly);
          }
          setCacheBust(Date.now());
          if (job.status === "completed") {
            pushToast("ok", `《${job.work_title}》全部 ${job.page_ids?.length || 0} 页译制完成`);
          } else if (job.status === "failed") {
            pushToast("err", job.error || "翻译任务执行失败");
          } else if (job.status === "canceled") {
            pushToast("info", "后台任务已中止，已译页面已保留");
          }
        }
      } catch {
        // transient
      }
    }, 900);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [activeJob, appView, fetchPages, fetchWorks, pushToast, reviewOnly, selectedWorkId]);

  // Global window drag-and-drop -> UploadModal
  useEffect(() => {
    const onDragEnter = (e: DragEvent) => {
      if (!e.dataTransfer?.types?.includes("Files")) return;
      e.preventDefault();
      dragCounterRef.current += 1;
      setGlobalDragging(true);
    };
    const onDragOver = (e: DragEvent) => {
      if (!e.dataTransfer?.types?.includes("Files")) return;
      e.preventDefault();
    };
    const onDragLeave = (e: DragEvent) => {
      if (!e.dataTransfer?.types?.includes("Files")) return;
      e.preventDefault();
      dragCounterRef.current = Math.max(0, dragCounterRef.current - 1);
      if (dragCounterRef.current === 0) setGlobalDragging(false);
    };
    const onDrop = (e: DragEvent) => {
      if (!e.dataTransfer?.types?.includes("Files")) return;
      e.preventDefault();
      dragCounterRef.current = 0;
      setGlobalDragging(false);
      void extractFilesFromDataTransfer(e.dataTransfer).then(({ files: extracted, folderName }) => {
        if (extracted.length === 0) {
          pushToast("err", "拖入的内容中没有找到支持的漫画图片或 CBZ/ZIP 归档");
          return;
        }
        setDroppedFiles(extracted);
        setDroppedFolderName(folderName);
        setUploadOpen(true);
      });
    };
    window.addEventListener("dragenter", onDragEnter);
    window.addEventListener("dragover", onDragOver);
    window.addEventListener("dragleave", onDragLeave);
    window.addEventListener("drop", onDrop);
    return () => {
      window.removeEventListener("dragenter", onDragEnter);
      window.removeEventListener("dragover", onDragOver);
      window.removeEventListener("dragleave", onDragLeave);
      window.removeEventListener("drop", onDrop);
    };
  }, []);

  const currentPagePos = pages.findIndex((p) => p.id === selectedPageId);

  const goPrevPage = useCallback(() => {
    if (currentPagePos > 0) setSelectedPageId(pages[currentPagePos - 1].id);
  }, [currentPagePos, pages]);

  const goNextPage = useCallback(() => {
    if (currentPagePos >= 0 && currentPagePos < pages.length - 1) {
      setSelectedPageId(pages[currentPagePos + 1].id);
    }
  }, [currentPagePos, pages]);

  // Keyboard shortcuts
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        if (shortcutsOpen) return setShortcutsOpen(false);
        if (uploadOpen) return setUploadOpen(false);
        if (sourcesOpen) return setSourcesOpen(false);
        if (settingsOpen) return setSettingsOpen(false);
        if (toolMode === "draw") return setToolMode("select");
        if (zenMode) return setZenMode(false);
        return;
      }
      if (isEditableElement(e.target) || settingsOpen || uploadOpen || sourcesOpen) return;
      if (appView !== "reader") return;

      if (e.code === "Space") {
        e.preventDefault();
        setPeekingRaw(true);
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        goPrevPage();
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        goNextPage();
      } else if (e.key === "[") {
        setSelectedNodeIdx((i) => Math.max(0, i - 1));
      } else if (e.key === "]") {
        setSelectedNodeIdx((i) => Math.min((pageDOM?.nodes?.length || 1) - 1, i + 1));
      } else if (e.key === "o" || e.key === "O") {
        setShowOverlay((v) => !v);
      } else if (e.key === "f" || e.key === "F") {
        setZenMode((v) => !v);
      } else if (e.key === "v" || e.key === "V") {
        setToolMode("select");
      } else if (e.key === "a" || e.key === "A") {
        setToolMode("draw");
        setShowOverlay(true);
      } else if (e.key === "h" || e.key === "H") {
        setToolMode("pan");
      } else if (e.key === "1") {
        setViewMode("composite");
      } else if (e.key === "2") {
        setViewMode("split");
      } else if (e.key === "3") {
        setViewMode("side-by-side");
      } else if (e.key === "4") {
        setViewMode("raw");
      } else if (e.key === "?") {
        setShortcutsOpen((v) => !v);
      }
    };
    const onKeyUp = (e: KeyboardEvent) => {
      if (e.code === "Space") setPeekingRaw(false);
    };
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
    };
  }, [
    appView,
    goNextPage,
    goPrevPage,
    pageDOM?.nodes?.length,
    settingsOpen,
    shortcutsOpen,
    sourcesOpen,
    toolMode,
    uploadOpen,
    zenMode,
  ]);

  // ==================== Bookshelf & Page Management ====================

  const handleOpenWork = (workId: number) => {
    setSelectedWorkId(workId);
    setSelectedPageId(0);
    setReviewOnly(false);
    setAppView("reader");
  };

  const handleRenameWork = async (workId: number, title: string) => {
    try {
      const res = await fetch(`/api/v1/works/${workId}`, {
        method: "PATCH",
        headers: authHeaders(true),
        body: JSON.stringify({ title }),
      });
      const data = await parseJsonSafe(res);
      if (!res.ok) {
        pushToast("err", data.error || `重命名失败: HTTP ${res.status}`);
        return;
      }
      await fetchWorks();
      pushToast("ok", `已重命名为《${title}》`);
    } catch (err) {
      pushToast("err", String(err));
    }
  };

  const handleDeleteWork = async (workId: number) => {
    try {
      const res = await fetch(`/api/v1/works/${workId}`, {
        method: "DELETE",
        headers: authHeaders(false),
      });
      const data = await parseJsonSafe(res);
      if (!res.ok) {
        pushToast("err", data.error || `删除失败: HTTP ${res.status}`);
        return;
      }
      const nextWorks = await fetchWorks();
      if (selectedWorkId === workId) {
        setSelectedWorkId(nextWorks[0]?.id || 0);
        setSelectedPageId(0);
        setPageDOM(null);
        setSavedDOMSnapshot("");
        setAppView("bookshelf");
      }
      pushToast("info", "已永久删除该作品及关联页面");
    } catch (err) {
      pushToast("err", String(err));
    }
  };

  const handleDeletePage = async (pageId: number) => {
    try {
      const res = await fetch(`/api/v1/pages/${pageId}`, {
        method: "DELETE",
        headers: authHeaders(false),
      });
      const data = await parseJsonSafe(res);
      if (!res.ok) {
        pushToast("err", data.error || `删除页面失败: HTTP ${res.status}`);
        return;
      }
      await fetchWorks();
      if (selectedWorkId > 0) {
        await fetchPages(selectedWorkId, reviewOnly);
      }
      pushToast("info", "已删除该页面");
    } catch (err) {
      pushToast("err", String(err));
    }
  };

  // ==================== Upload & Job Handlers ====================

  const handleUploadSubmit = async (payload: {
    files: File[];
    refFiles: File[];
    workTitle: string;
    chapterTitle: string;
    formatProfile: string;
    targetLang: string;
    analyzeOnly: boolean;
  }): Promise<boolean> => {
    setBusy(true);
    try {
      const fd = new FormData();
      fd.append("work_title", payload.workTitle);
      fd.append("chapter_title", payload.chapterTitle);
      fd.append("format_profile", payload.formatProfile);
      fd.append("target_lang", payload.targetLang);
      fd.append("analyze_only", String(payload.analyzeOnly));
      payload.files.forEach((f) => fd.append("files", f));
      payload.refFiles.forEach((f) => fd.append("ref_files", f));

      const res = await fetch("/api/v1/jobs/translate", {
        method: "POST",
        headers: authHeaders(false),
        body: fd,
      });
      const data = await parseJsonSafe(res);
      if (!res.ok) {
        pushToast("err", data.error || `提交失败: HTTP ${res.status}`);
        if (res.status === 412 || res.status === 401) setSettingsOpen(true);
        return false;
      }
      if (data.job) {
        const job = data.job as TranslationJob;
        lastAutoOpenedPageRef.current = 0;
        lastCompletedJobRef.current = "";
        setActiveJob(job);
        pushToast("info", `已提交《${payload.workTitle}》(${payload.files.length} 文件)，后台加速处理中`);
        return true;
      }
      return false;
    } catch (err) {
      pushToast("err", String(err));
      return false;
    } finally {
      setBusy(false);
    }
  };

  const handleCancelJob = async (jobId: string) => {
    try {
      const res = await fetch(`/api/v1/jobs/${jobId}/cancel`, {
        method: "POST",
        headers: authHeaders(false),
      });
      if (res.ok) {
        const data = await res.json();
        if (data.job) setActiveJob(data.job);
        pushToast("info", "正在中止当前任务...");
      }
    } catch (err) {
      pushToast("err", String(err));
    }
  };

  // ==================== Interactive Canvas Node Operations ====================

  const updateNodeAt = useCallback((idx: number, mutator: (n: TextNode) => TextNode) => {
    setPageDOM((prev) => {
      if (!prev) return prev;
      const nodes = [...(prev.nodes || [])];
      if (!nodes[idx]) return prev;
      nodes[idx] = mutator(nodes[idx]);
      return { ...prev, nodes };
    });
  }, []);

  const handleAddManualBox = useCallback(
    (bbox: [number, number, number, number]) => {
      if (!pageDOM) return;
      const [bx, by, bw, bh] = bbox;
      const orient: "vertical" | "horizontal" = bh > bw * 1.15 ? "vertical" : "horizontal";
      const existing = pageDOM.nodes || [];
      const newNode: TextNode = {
        id: `manual_${Date.now().toString(36)}`,
        reading_order: existing.length,
        track: "primary",
        bbox: [bx, by, bw, bh],
        polygon: [],
        source_lines: [],
        source_quadrilaterals: [],
        detector_quadrilaterals: [],
        source_readings: [],
        bubble_polygon: [],
        spine_points: [],
        bubble_kind: "speech_oval",
        fusion_mode: "clean_interior",
        restoration_path: "lama_masked",
        speaker_cluster_id: "",
        speaker_appearance_b64: "",
        source_text: "",
        ocr_text: "",
        recognition_pending: false,
        ruby_pairs: [],
        target_text: "",
        target_segments: [],
        target_ruby_pairs: [],
        char_budget_min: 1,
        char_budget_max: Math.max(8, Math.round((bw * bh) / 360)),
        style: {
          orientation: orient,
          font_category: "gothic",
          stroke_width_px: 2,
          stroke_modulation: 0.15,
          tremor_amplitude: 0,
          slant_deg: 0,
          fg_color_rgba: [15, 15, 18, 255],
          outline_color_rgba: [255, 255, 255, 255],
          outline_width_px: 1.5,
          blend_alpha: 1,
          blend_mode: "normal",
          ink_density_orig: 0.12,
          hollow_mode: "solid",
          weight_scale: 1,
          source_font_size_px: Math.max(14, Math.min(36, Math.round(Math.min(bw, bh) * 0.24))),
        },
        layout: {
          font_size: 20,
          line_spacing: 1.15,
          char_spacing: 0,
          lines: [],
          silhouette_score: 1,
          ink_density_rendered: 0,
        },
        patches: {
          roi: [0, 0, 0, 0],
          clean_z0_png_b64: "",
          stroke_alpha_png_b64: "",
          source_observed_ink_png_b64: "",
          source_domain_png_b64: "",
          refined_stroke_png_b64: "",
          inpaint_unknown_png_b64: "",
          inferred_background_png_b64: "",
          boundary_opacity_png_b64: "",
          z2_occluder_mask_png_b64: "",
          shading_map_png_b64: "",
          exemplar_patch_png_b64: "",
          rendered_z1_png_b64: "",
          render_fingerprint: "",
        },
        uncertainty: {
          score: 0,
          ocr_entropy: 0,
          occlusion_ambiguity: 0,
          restoration_residual: 0,
          source_ink_uncovered_ratio: null,
          source_ink_residual_ratio: null,
          automatic_repair_pixels: 0,
          needs_review: true,
          reasons: ["manual_box"],
        },
      };
      const nextNodes = [...existing, newNode];
      setPageDOM({ ...pageDOM, nodes: nextNodes });
      setSelectedNodeIdx(nextNodes.length - 1);
      setShowOverlay(true);
      setRightOpen(true);
      pushToast("info", "已新增手动文本框，可点击「AI 识图翻译」或手填译文后点击「实时重排」");
    },
    [pageDOM, pushToast]
  );

  const handleMoveResizeNode = useCallback(
    (idx: number, bbox: [number, number, number, number]) => {
      updateNodeAt(idx, (n) => ({
        ...n,
        bbox,
        bubble_polygon: [],
      }));
    },
    [updateNodeAt]
  );

  const handleDeleteNode = useCallback(
    (idx: number) => {
      setPageDOM((prev) => {
        if (!prev) return prev;
        const nodes = (prev.nodes || [])
          .filter((_, i) => i !== idx)
          .map((n, i) => ({ ...n, reading_order: i }));
        setSelectedNodeIdx(Math.max(0, Math.min(idx, nodes.length - 1)));
        return { ...prev, nodes };
      });
      pushToast("info", "已移除该文本框，点击「实时重排」即可无损还原底图原画");
    },
    [pushToast]
  );

  const handleTranslateSingleNode = async (idx: number) => {
    if (!pageDOM || !selectedPageId) return;
    const node = (pageDOM.nodes || [])[idx];
    if (!node) return;
    setTranslatingNode(true);
    try {
      const res = await fetch(`/api/v1/pages/${selectedPageId}/translate-box`, {
        method: "POST",
        headers: authHeaders(true),
        body: JSON.stringify({
          bbox: node.bbox,
          orientation: node.style?.orientation || "vertical",
          target_lang: targetLang || "zh-CN",
        }),
      });
      const data = await parseJsonSafe(res);
      if (!res.ok) {
        pushToast("err", data.error || `单框识图失败: HTTP ${res.status}`);
        if (res.status === 412 || res.status === 401) setSettingsOpen(true);
        return;
      }
      if (!data.is_text || (!data.target_text && !data.source_text)) {
        pushToast("info", "模型未在该框内识别到有效文字，可直接手动输入译文");
        return;
      }
      updateNodeAt(idx, (n) => ({
        ...n,
        source_text: data.source_text || n.source_text,
        target_text: data.target_text || n.target_text,
        target_segments: Array.isArray(data.target_segments) ? data.target_segments : [],
      }));
      pushToast("ok", "单框 AI 识图翻译完成！点击「实时重排」立即擦除并嵌字");
    } catch (err) {
      pushToast("err", String(err));
    } finally {
      setTranslatingNode(false);
    }
  };

  const handleRevertChanges = () => {
    if (!savedDOMSnapshot) return;
    try {
      setPageDOM(JSON.parse(savedDOMSnapshot) as PageDOM);
      pushToast("info", "已还原未保存的节点改动");
    } catch {
      // ignore
    }
  };

  const handleRerender = async (markReviewed = false) => {
    if (!pageDOM || !selectedPageId) return;
    setBusy(true);
    try {
      const res = await fetch(`/api/v1/pages/${selectedPageId}/rerender`, {
        method: "POST",
        headers: authHeaders(true),
        body: JSON.stringify({ dom: pageDOM, target_lang: targetLang, mark_reviewed: markReviewed }),
      });
      const data = await parseJsonSafe(res);
      if (!res.ok) {
        pushToast("err", data.error || `重排失败: HTTP ${res.status}`);
        if (res.status === 401) setSettingsOpen(true);
        return;
      }
      setPageDOM(data.dom);
      setSavedDOMSnapshot(JSON.stringify(data.dom));
      setCacheBust(Date.now());
      await fetchWorks();
      if (selectedWorkId > 0) await fetchPages(selectedWorkId, reviewOnly);
      pushToast(
        "ok",
        markReviewed
          ? "已标记审校通过并保存成品"
          : `重排完成 (${Math.round(data.dom?.metrics?.render_ms || 0)}ms · L∞=${data.dom?.metrics?.unmasked_linf ?? 0})`
      );
    } catch (err) {
      pushToast("err", String(err));
    } finally {
      setBusy(false);
    }
  };

  const handleRetranslatePage = async () => {
    if (!selectedPageId) return;
    setBusy(true);
    try {
      const res = await fetch(`/api/v1/pages/${selectedPageId}/translate`, {
        method: "POST",
        headers: authHeaders(true),
        body: JSON.stringify({ target_lang: targetLang }),
      });
      const data = await parseJsonSafe(res);
      if (!res.ok) {
        pushToast("err", data.error || `AI 重译失败: HTTP ${res.status}`);
        if (res.status === 412 || res.status === 401) setSettingsOpen(true);
        return;
      }
      setPageDOM(data.dom);
      setSavedDOMSnapshot(JSON.stringify(data.dom));
      setCacheBust(Date.now());
      await fetchWorks();
      if (selectedWorkId > 0) await fetchPages(selectedWorkId, reviewOnly);
      pushToast("ok", `AI 重译完成 (${Math.round(data.elapsed_ms || 0)}ms)`);
    } catch (err) {
      pushToast("err", String(err));
    } finally {
      setBusy(false);
    }
  };

  const handleAddGlossary = async (src: string, tgt: string, ruby: string): Promise<boolean> => {
    if (!selectedWorkId) return false;
    try {
      const res = await fetch(`/api/v1/works/${selectedWorkId}/glossary`, {
        method: "POST",
        headers: authHeaders(true),
        body: JSON.stringify({ source_term: src, target_term: tgt, ruby_override: ruby, notes: "" }),
      });
      const data = await parseJsonSafe(res);
      if (!res.ok) {
        pushToast("err", data.error || "保存术语失败");
        return false;
      }
      await fetchWorkPersonas(selectedWorkId);
      pushToast("ok", `已加入术语「${src} → ${tgt}」`);
      return true;
    } catch (err) {
      pushToast("err", String(err));
      return false;
    }
  };

  const handleSavePersona = async (persona: CharacterPersona) => {
    if (!selectedWorkId) return;
    try {
      const res = await fetch(`/api/v1/works/${selectedWorkId}/characters`, {
        method: "POST",
        headers: authHeaders(true),
        body: JSON.stringify(persona),
      });
      if (!res.ok) {
        const data = await parseJsonSafe(res);
        pushToast("err", data.error || "保存角色设定失败");
        return;
      }
      await fetchWorkPersonas(selectedWorkId);
      pushToast("ok", `已保存角色「${persona.name_target || persona.cluster_key}」`);
    } catch (err) {
      pushToast("err", String(err));
    }
  };

  const handleSaveRelation = async (
    fromId: number,
    toId: number,
    honorific: string,
    reg: string
  ): Promise<boolean> => {
    if (!selectedWorkId) return false;
    try {
      const res = await fetch(`/api/v1/works/${selectedWorkId}/relations`, {
        method: "POST",
        headers: authHeaders(true),
        body: JSON.stringify({
          from_char_id: fromId,
          to_char_id: toId,
          honorific_target: honorific,
          register_note: reg,
        }),
      });
      if (!res.ok) {
        const data = await parseJsonSafe(res);
        pushToast("err", data.error || "保存称谓关系失败");
        return false;
      }
      await fetchWorkPersonas(selectedWorkId);
      pushToast("ok", "已绑定跨话称谓关系");
      return true;
    } catch (err) {
      pushToast("err", String(err));
      return false;
    }
  };

  const handleSettingsSaved = (resp: RuntimeSettingsResponse) => {
    if (resp.settings?.target_lang) setTargetLang(resp.settings.target_lang);
    void fetchHealth();
    void checkAuth();
    pushToast("ok", "运行与模型设置已生效");
  };

  const handleCopyText = async (text: string) => {
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
      pushToast("info", "已复制原文到剪贴板");
    } catch {
      pushToast("err", "浏览器拒绝写入剪贴板");
    }
  };

  const llmReady = Boolean(health?.llm_configured);
  const pageCounterLabel =
    pages.length > 0 && currentPagePos >= 0 ? `${currentPagePos + 1} / ${pages.length}` : "";

  const effectiveLeftOpen = !zenMode && leftOpen;
  const effectiveRightOpen = !zenMode && rightOpen;

  return (
    <div className={`studio-shell ${zenMode ? "zen-mode" : ""}`}>
      {!zenMode && (
        <header className="studio-topbar">
          <div className="topbar-left">
            <button
              type="button"
              className="brand-mark brand-btn"
              onClick={() => setAppView("bookshelf")}
              title="返回漫画书架"
            >
              PRISM-T <span className="brand-tag mono">STUDIO</span>
            </button>

            <div className="segmented-control" role="tablist" aria-label="主视图切换">
              <button
                type="button"
                className={`seg-btn ${appView === "bookshelf" ? "active" : ""}`}
                onClick={() => setAppView("bookshelf")}
              >
                <IconBookshelf size={14} /> 书架 ({works.length})
              </button>
              <button
                type="button"
                className={`seg-btn ${appView === "reader" ? "active" : ""}`}
                onClick={() => {
                  if (selectedWorkId === 0 && works.length > 0) {
                    setSelectedWorkId(works[0].id);
                  }
                  setAppView("reader");
                }}
              >
                <IconStudio size={14} /> 精修台
              </button>
            </div>

            <button
              type="button"
              className="status-chip mono"
              onClick={() => setSettingsOpen(true)}
              title="查看硬件与模型运行设置"
            >
              <span className={`dot ${health?.engine_connected ? "ok" : "err"}`} />
              {compactDeviceLabel(health)}
            </button>

            <button
              type="button"
              className={`status-chip mono ${llmReady ? "" : "warn"}`}
              onClick={() => setSettingsOpen(true)}
              title="配置大模型接口"
            >
              <span className={`dot ${llmReady ? "ok" : "warn"}`} />
              {llmReady ? health?.llm_model || "模型就绪" : "未配置模型"}
            </button>
          </div>

          <div className="topbar-right">
            <button
              type="button"
              className="btn btn-sm cyan"
              onClick={() => setSourcesOpen(true)}
              title="从在线漫画源边下载边翻译"
            >
              <IconGlobe size={14} /> 漫源
            </button>

            <button
              type="button"
              className="btn btn-sm primary"
              onClick={() => {
                setDroppedFiles([]);
                setDroppedFolderName("");
                setUploadOpen(true);
              }}
              title="导入本地图片或 CBZ/ZIP"
            >
              <IconPlus size={14} /> 导入
            </button>

            {appView === "reader" && selectedPageId > 0 && (
              <div className="export-group">
                <a
                  className="btn btn-sm"
                  href={`/api/v1/pages/${selectedPageId}/composite?download=1&t=${cacheBust}`}
                  download
                >
                  <IconDownload size={13} /> PNG
                </a>
                <a
                  className="btn btn-sm"
                  href={`/api/v1/pages/${selectedPageId}/export/psd`}
                  download
                >
                  <IconDownload size={13} /> PSD
                </a>
                {selectedWorkId > 0 && (
                  <a
                    className="btn btn-sm"
                    href={`/api/v1/works/${selectedWorkId}/export/cbz`}
                    download
                  >
                    <IconDownload size={13} /> CBZ
                  </a>
                )}
              </div>
            )}

            {appView === "reader" && (
              <>
                <button
                  type="button"
                  className="btn btn-sm"
                  onClick={() => setLeftOpen((v) => !v)}
                  title="显示/隐藏左侧页面栏"
                >
                  <IconPanelLeft size={14} />
                </button>
                <button
                  type="button"
                  className="btn btn-sm"
                  onClick={() => setRightOpen((v) => !v)}
                  title="显示/隐藏右侧精修栏"
                >
                  <IconPanelRight size={14} />
                </button>
              </>
            )}

            <button
              type="button"
              className="btn btn-sm"
              onClick={() => setShortcutsOpen(true)}
              title="快捷键速查 (?)"
            >
              <IconHelp size={14} />
            </button>

            <button
              type="button"
              className="btn btn-sm"
              onClick={() => setSettingsOpen(true)}
            >
              <IconSettings size={14} /> 设置
            </button>
          </div>
        </header>
      )}

      {appView === "bookshelf" ? (
        <BookshelfView
          works={works}
          cacheBust={cacheBust}
          activeJob={activeJob}
          onOpenWork={handleOpenWork}
          onRenameWork={handleRenameWork}
          onDeleteWork={handleDeleteWork}
          onOpenUpload={() => {
            setDroppedFiles([]);
            setUploadOpen(true);
          }}
          onOpenSources={() => setSourcesOpen(true)}
          onCancelJob={handleCancelJob}
        />
      ) : (
        <div
          className="studio-body"
          style={{
            gridTemplateColumns: `${effectiveLeftOpen ? "240px" : "0px"} minmax(0, 1fr) ${
              effectiveRightOpen ? "344px" : "0px"
            }`,
          }}
        >
          {effectiveLeftOpen ? (
            <PageFilmstrip
              works={works}
              selectedWorkId={selectedWorkId}
              onSelectWork={(id) => {
                setSelectedWorkId(id);
                setSelectedPageId(0);
              }}
              onBackToShelf={() => setAppView("bookshelf")}
              pages={pages}
              selectedPageId={selectedPageId}
              onSelectPage={setSelectedPageId}
              onDeletePage={handleDeletePage}
              reviewOnly={reviewOnly}
              onToggleReviewOnly={setReviewOnly}
              cacheBust={cacheBust}
              activeJob={activeJob}
              onCancelJob={handleCancelJob}
              onOpenUpload={() => {
                setDroppedFiles([]);
                setUploadOpen(true);
              }}
            />
          ) : (
            <div />
          )}

          <CanvasStage
            pageId={selectedPageId}
            pageDOM={pageDOM}
            cacheBust={cacheBust}
            viewMode={viewMode}
            onChangeViewMode={setViewMode}
            toolMode={toolMode}
            onChangeToolMode={(next) => {
              setToolMode(next);
              if (next === "draw") setShowOverlay(true);
            }}
            showOverlay={showOverlay}
            onToggleOverlay={() => setShowOverlay((v) => !v)}
            peekingRaw={peekingRaw}
            selectedNodeIdx={selectedNodeIdx}
            onSelectNode={(idx) => {
              setSelectedNodeIdx(idx);
              if (!zenMode && !rightOpen) setRightOpen(true);
            }}
            onAddManualBox={handleAddManualBox}
            onMoveResizeNode={handleMoveResizeNode}
            zenMode={zenMode}
            onToggleZen={() => setZenMode((v) => !v)}
            canPrevPage={currentPagePos > 0}
            canNextPage={currentPagePos >= 0 && currentPagePos < pages.length - 1}
            pageCounterLabel={pageCounterLabel}
            onPrevPage={goPrevPage}
            onNextPage={goNextPage}
            onOpenUpload={() => {
              setDroppedFiles([]);
              setUploadOpen(true);
            }}
          />

          {effectiveRightOpen ? (
            <InspectorPanel
              pageDOM={pageDOM}
              selectedNodeIdx={selectedNodeIdx}
              onSelectNode={setSelectedNodeIdx}
              onUpdateNodeAt={updateNodeAt}
              onDeleteNode={handleDeleteNode}
              onTranslateSingleNode={handleTranslateSingleNode}
              onStartDrawBox={() => {
                setToolMode("draw");
                setShowOverlay(true);
                if (viewMode === "split") setViewMode("composite");
              }}
              translatingNode={translatingNode}
              isDirty={isDirty}
              onRevertChanges={handleRevertChanges}
              busy={busy}
              onRerender={handleRerender}
              onRetranslatePage={handleRetranslatePage}
              selectedWorkId={selectedWorkId}
              glossary={glossary}
              personas={personas}
              relations={relations}
              onAddGlossary={handleAddGlossary}
              onSavePersona={handleSavePersona}
              onChangePersonaDraft={(idx, next) =>
                setPersonas((prev) => prev.map((p, i) => (i === idx ? next : p)))
              }
              onSaveRelation={handleSaveRelation}
              onCopyText={handleCopyText}
            />
          ) : (
            <div />
          )}
        </div>
      )}

      {globalDragging && (
        <div className="global-drop-overlay">
          <div className="global-drop-box">
            <div style={{ fontSize: 22, fontWeight: 900 }}>松开鼠标立即导入漫画</div>
            <div className="mono" style={{ fontSize: 13 }}>
              支持整个漫画文件夹、多张 PNG / JPG / WebP / AVIF，或整卷 .cbz / .zip 归档
            </div>
          </div>
        </div>
      )}

      {shortcutsOpen && (
        <div className="modal-backdrop" onClick={() => setShortcutsOpen(false)}>
          <div className="shortcuts-dialog panel-card" onClick={(e) => e.stopPropagation()}>
            <div className="panel-title">
              <span>快捷键速查</span>
              <button type="button" className="btn btn-sm" onClick={() => setShortcutsOpen(false)}>
                <IconClose size={13} /> 关闭
              </button>
            </div>
            <div className="shortcuts-grid mono">
              <kbd>Space (按住)</kbd>
              <span>瞬看原图生肉，松开秒回译图</span>
              <kbd>V / A / H</kbd>
              <span>选框移动 / 拖拽框选新增 / 抓手平移</span>
              <kbd>← / →</kbd>
              <span>上一页 / 下一页</span>
              <kbd>[ / ]</kbd>
              <span>上一个 / 下一个文本气泡</span>
              <kbd>1 / 2 / 3 / 4</kbd>
              <span>成品 / 拉帘 / 并排 / 原图</span>
              <kbd>O</kbd>
              <span>显示或隐藏文本检测框线</span>
              <kbd>F</kbd>
              <span>进入或退出沉浸阅读模式</span>
              <kbd>Ctrl + 滚轮</kbd>
              <span>画布自由缩放</span>
              <kbd>Ctrl + Enter</kbd>
              <span>译文编辑框内一键实时重排渲染</span>
            </div>
          </div>
        </div>
      )}

      <UploadModal
        open={uploadOpen}
        onClose={() => setUploadOpen(false)}
        initialFiles={droppedFiles}
        initialFolderName={droppedFolderName}
        defaultTargetLang={targetLang}
        llmReady={llmReady}
        busy={busy}
        jobRunning={activeJob?.status === "running"}
        onSubmit={handleUploadSubmit}
      />

      <SourceModal
        open={sourcesOpen}
        onClose={() => setSourcesOpen(false)}
        token={token}
        defaultTargetLang={targetLang}
        llmReady={llmReady}
        jobRunning={activeJob?.status === "running"}
        onJobStarted={(job) => {
          lastAutoOpenedPageRef.current = 0;
          lastCompletedJobRef.current = "";
          setActiveJob(job);
        }}
        onToast={pushToast}
      />

      <SettingsModal
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        token={token}
        onChangeToken={handleSaveToken}
        authAuthorized={authAuthorized}
        health={health}
        hasActiveJob={activeJob?.status === "running"}
        onSaved={handleSettingsSaved}
      />

      {toasts.length > 0 && (
        <div className="toast-stack" aria-live="polite">
          {toasts.map((t) => (
            <div key={t.id} className={`toast-item ${t.kind}`}>
              {t.text}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
