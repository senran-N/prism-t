import { useCallback, useEffect, useState } from "react";
import type {
  BrowseSourceResult,
  ChapterEntry,
  ComicEntry,
  ComicSourceItem,
  PageAsset,
  TranslationJob,
} from "../types";
import {
  IconBookshelf,
  IconClose,
  IconGlobe,
  IconLink,
  IconSearch,
  IconZap,
} from "./Icons";

type SourceTab = "mangadex" | "opds" | "direct";

interface SourceModalProps {
  open: boolean;
  onClose: () => void;
  token: string;
  defaultTargetLang: string;
  llmReady: boolean;
  jobRunning: boolean;
  onJobStarted: (job: TranslationJob) => void;
  onToast: (kind: "ok" | "err" | "info", text: string) => void;
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

export default function SourceModal({
  open,
  onClose,
  token,
  defaultTargetLang,
  llmReady,
  jobRunning,
  onJobStarted,
  onToast,
}: SourceModalProps) {
  const [tab, setTab] = useState<SourceTab>("mangadex");
  const [savedSources, setSavedSources] = useState<ComicSourceItem[]>([]);

  // MangaDex state
  const [mdQuery, setMdQuery] = useState("");
  const [mdLang, setMdLang] = useState("");
  const [mdDataSaver, setMdDataSaver] = useState(false);

  // OPDS / JSON source manager state
  const [newSrcName, setNewSrcName] = useState("");
  const [newSrcKind, setNewSrcKind] = useState("opds");
  const [newSrcUrl, setNewSrcUrl] = useState("");
  const [newSrcAuthHeader, setNewSrcAuthHeader] = useState("");
  const [activeSourceId, setActiveSourceId] = useState<number>(0);

  // Direct URL / gallery / CBZ state
  const [directInput, setDirectInput] = useState("");

  // Shared browse result & launch options
  const [loading, setLoading] = useState(false);
  const [launching, setLaunching] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");
  const [browse, setBrowse] = useState<BrowseSourceResult | null>(null);
  const [selectedComic, setSelectedComic] = useState<ComicEntry | null>(null);
  const [workTitle, setWorkTitle] = useState("");
  const [chapterTitle, setChapterTitle] = useState("第01话");
  const [formatProfile, setFormatProfile] = useState("auto");
  const [targetLang, setTargetLang] = useState(defaultTargetLang || "zh-CN");
  const [maxPages, setMaxPages] = useState<number>(0);

  const authHeaders = useCallback((): HeadersInit => {
    const h: Record<string, string> = { "Content-Type": "application/json" };
    if (token) h["X-Prism-Token"] = token;
    return h;
  }, [token]);

  const loadSources = useCallback(async () => {
    try {
      const res = await fetch("/api/v1/sources");
      const data = await parseJsonSafe(res);
      if (res.ok && Array.isArray(data.sources)) {
        setSavedSources(data.sources as ComicSourceItem[]);
      }
    } catch {
      // ignore transient error
    }
  }, []);

  useEffect(() => {
    if (open) {
      setTargetLang(defaultTargetLang || "zh-CN");
      setErrorMsg("");
      void loadSources();
    }
  }, [open, defaultTargetLang, loadSources]);

  if (!open) return null;

  const probeSource = async (payload: Record<string, any>) => {
    setLoading(true);
    setErrorMsg("");
    try {
      const res = await fetch("/api/v1/sources/probe", {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify(payload),
      });
      const data = await parseJsonSafe(res);
      if (!res.ok) {
        setErrorMsg(data.error || `解析失败: HTTP ${res.status}`);
        return;
      }
      const typed = data as BrowseSourceResult;
      setBrowse(typed);
      if (typed.title && !workTitle) {
        setWorkTitle(typed.title);
      }
    } catch (err) {
      setErrorMsg(String(err));
    } finally {
      setLoading(false);
    }
  };

  const handleSaveSource = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newSrcName.trim() || !newSrcUrl.trim()) return;
    setLoading(true);
    setErrorMsg("");
    try {
      const headers: Record<string, string> = {};
      if (newSrcAuthHeader.trim()) {
        headers["Authorization"] = newSrcAuthHeader.trim();
      }
      const res = await fetch("/api/v1/sources", {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({
          name: newSrcName.trim(),
          kind: newSrcKind,
          base_url: newSrcUrl.trim(),
          headers,
          default_profile: formatProfile,
        }),
      });
      const data = await parseJsonSafe(res);
      if (!res.ok) {
        setErrorMsg(data.error || `添加书源失败: HTTP ${res.status}`);
        return;
      }
      setNewSrcName("");
      setNewSrcUrl("");
      setNewSrcAuthHeader("");
      await loadSources();
      onToast("ok", `已保存漫画源「${data.name}」`);
    } catch (err) {
      setErrorMsg(String(err));
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteSource = async (id: number) => {
    try {
      const res = await fetch(`/api/v1/sources/${id}`, {
        method: "DELETE",
        headers: authHeaders(),
      });
      if (res.ok) {
        await loadSources();
        if (activeSourceId === id) setActiveSourceId(0);
        onToast("info", "已删除漫画源");
      }
    } catch {
      // ignore
    }
  };

  const startStreamingJob = async (opts: {
    source_kind: string;
    work_title: string;
    chapter_title: string;
    mangadex_chapter_id?: string;
    archive_url?: string;
    pages?: PageAsset[];
    direct_input?: string;
    source_id?: number;
  }) => {
    if (!llmReady) {
      setErrorMsg("请先在右上角「设置」中配置大模型 API Key 后再启动边下边译。");
      return;
    }
    setLaunching(true);
    setErrorMsg("");
    try {
      const res = await fetch("/api/v1/jobs/source-translate", {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({
          ...opts,
          format_profile: formatProfile,
          target_lang: targetLang || "zh-CN",
          data_saver: mdDataSaver,
          max_pages: maxPages > 0 ? maxPages : 0,
        }),
      });
      const data = await parseJsonSafe(res);
      if (!res.ok) {
        setErrorMsg(data.error || `启动失败: HTTP ${res.status}`);
        return;
      }
      const job = data.job as TranslationJob;
      onJobStarted(job);
      onToast("ok", `已启动《${job.work_title}》边下边译！第 1 页译完即可开看`);
      onClose();
    } catch (err) {
      setErrorMsg(String(err));
    } finally {
      setLaunching(false);
    }
  };

  const handleOpenComic = (comic: ComicEntry) => {
    setSelectedComic(comic);
    setWorkTitle(comic.title);
    if (comic.format_profile) setFormatProfile(comic.format_profile);
    if (comic.source_kind === "mangadex") {
      void probeSource({
        kind: "mangadex_chapters",
        manga_id: comic.id,
        language: mdLang,
      });
    } else if (comic.url) {
      void probeSource({
        kind: comic.source_kind || "opds",
        url: comic.url,
        source_id: activeSourceId || undefined,
      });
    }
  };

  const handleLaunchChapter = (ch: ChapterEntry) => {
    const title = workTitle.trim() || selectedComic?.title || browse?.title || "在线漫画";
    if (ch.source_kind === "mangadex") {
      void startStreamingJob({
        source_kind: "mangadex",
        work_title: title,
        chapter_title: ch.title,
        mangadex_chapter_id: ch.id,
      });
    } else if (ch.archive_url) {
      void startStreamingJob({
        source_kind: ch.source_kind || "opds",
        work_title: title,
        chapter_title: ch.title,
        archive_url: ch.archive_url,
        source_id: activeSourceId || undefined,
      });
    } else if (ch.pages && ch.pages.length > 0) {
      void startStreamingJob({
        source_kind: ch.source_kind || "json_manifest",
        work_title: title,
        chapter_title: ch.title,
        pages: ch.pages,
        source_id: activeSourceId || undefined,
      });
    } else if (ch.url) {
      void startStreamingJob({
        source_kind: ch.source_kind || "direct_urls",
        work_title: title,
        chapter_title: ch.title,
        direct_input: ch.url,
        source_id: activeSourceId || undefined,
      });
    }
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal-dialog modal-dialog-wide modal-dialog-flex panel-card"
        role="dialog"
        aria-modal="true"
        aria-label="漫画源导入与边下边译"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="panel-title">
          <span className="inline-icon-text">
            <IconGlobe size={16} /> 导入漫画源 · 边下载边翻译
          </span>
          <button type="button" className="btn btn-sm" onClick={onClose}>
            <IconClose size={13} /> 关闭
          </button>
        </div>

        <div className="source-tabs">
          <button
            type="button"
            className={`tab-btn ${tab === "mangadex" ? "active" : ""}`}
            onClick={() => {
              setTab("mangadex");
              setErrorMsg("");
            }}
          >
            <IconSearch size={13} /> MangaDex 漫源
          </button>
          <button
            type="button"
            className={`tab-btn ${tab === "opds" ? "active" : ""}`}
            onClick={() => {
              setTab("opds");
              setErrorMsg("");
            }}
          >
            <IconBookshelf size={13} /> OPDS / JSON 书源 ({savedSources.length})
          </button>
          <button
            type="button"
            className={`tab-btn ${tab === "direct" ? "active" : ""}`}
            onClick={() => {
              setTab("direct");
              setErrorMsg("");
            }}
          >
            <IconLink size={13} /> 直链 / 远程 CBZ / 图集
          </button>
        </div>

        {errorMsg && <div className="error-banner">{errorMsg}</div>}
        {!llmReady && (
          <div className="warn-banner">
            当前未配置大模型密钥；请先在「设置」完成模型接入，再启动边下边译。
          </div>
        )}

        <div className="modal-scroll-body">
          {/* Shared streaming options bar */}
          <div className="source-options-bar">
            <div className="field">
              <label>入库书名</label>
              <input
                type="text"
                value={workTitle}
                placeholder="自动取漫源书名，可自定义"
                onChange={(e) => setWorkTitle(e.target.value)}
              />
            </div>
            <div className="field">
              <label>版式</label>
              <select value={formatProfile} onChange={(e) => setFormatProfile(e.target.value)}>
                <option value="auto">自动识别</option>
                <option value="manga_rtl">日漫右翻</option>
                <option value="comic_ltr">美漫左翻</option>
                <option value="webtoon_vertical">竖滑条漫</option>
              </select>
            </div>
            <div className="field">
              <label>目标语言</label>
              <input
                type="text"
                value={targetLang}
                onChange={(e) => setTargetLang(e.target.value)}
              />
            </div>
            <div className="field">
              <label>页数限制 (0=整话)</label>
              <input
                type="number"
                min={0}
                max={200}
                value={maxPages}
                onChange={(e) => setMaxPages(Math.max(0, Number(e.target.value) || 0))}
                title="可先设 3~5 页快速试看，设 0 则整话边下边译"
              />
            </div>
          </div>

          {tab === "mangadex" && (
            <div className="source-pane">
              <form
                className="source-search-row"
                onSubmit={(e) => {
                  e.preventDefault();
                  setSelectedComic(null);
                  void probeSource({ kind: "mangadex_search", query: mdQuery });
                }}
              >
                <input
                  type="text"
                  style={{ flex: 1 }}
                  value={mdQuery}
                  placeholder="输入漫画名搜索（如 Yotsuba / Solo Leveling），或直接粘贴 MangaDex 漫画/章节链接..."
                  onChange={(e) => setMdQuery(e.target.value)}
                />
                <select
                  style={{ width: 118 }}
                  value={mdLang}
                  onChange={(e) => setMdLang(e.target.value)}
                  title="章节源语言筛选"
                >
                  <option value="">全部语言</option>
                  <option value="ja">日语 (ja)</option>
                  <option value="en">英语 (en)</option>
                  <option value="ko">韩语 (ko)</option>
                </select>
                <label className="mono" style={{ fontSize: 11, display: "flex", alignItems: "center", gap: 4 }}>
                  <input
                    type="checkbox"
                    checked={mdDataSaver}
                    onChange={(e) => setMdDataSaver(e.target.checked)}
                  />
                  省流图
                </label>
                <button type="submit" className="btn primary" disabled={loading}>
                  {loading ? "查询中..." : "搜索 / 解析"}
                </button>
              </form>
            </div>
          )}

          {tab === "opds" && (
            <div className="source-pane">
              <div className="saved-source-list">
                {savedSources.map((cs) => (
                  <div key={cs.id} className="saved-source-row">
                    <div style={{ minWidth: 0, flex: 1 }}>
                      <strong>{cs.name}</strong>{" "}
                      <span className="mono count-pill">{cs.kind}</span>
                      <div className="mono" style={{ fontSize: 11, color: "var(--zinc-note)", overflow: "hidden", textOverflow: "ellipsis" }}>
                        {cs.base_url}
                      </div>
                    </div>
                    <div className="btn-row">
                      <button
                        type="button"
                        className="btn btn-sm cyan"
                        disabled={loading}
                        onClick={() => {
                          setActiveSourceId(cs.id);
                          setSelectedComic(null);
                          void probeSource({ kind: cs.kind, source_id: cs.id });
                        }}
                      >
                        浏览书库
                      </button>
                      <button
                        type="button"
                        className="btn btn-sm danger-btn"
                        onClick={() => void handleDeleteSource(cs.id)}
                      >
                        删除
                      </button>
                    </div>
                  </div>
                ))}
                {savedSources.length === 0 && (
                  <div className="empty-inspector">
                    尚未添加自定义书源。支持接入 Komga / Kavita / Suwayomi 的 OPDS 地址，或任意 JSON 章节清单。
                  </div>
                )}
              </div>

              <form onSubmit={handleSaveSource} className="style-card">
                <div className="style-card-title">+ 添加新漫画源（OPDS 1.2 / JSON 清单）</div>
                <div className="settings-grid-2">
                  <input
                    type="text"
                    placeholder="书源名称（如 我的 Komga 书库）"
                    value={newSrcName}
                    onChange={(e) => setNewSrcName(e.target.value)}
                  />
                  <select value={newSrcKind} onChange={(e) => setNewSrcKind(e.target.value)}>
                    <option value="opds">OPDS 1.2 Atom XML (Komga/Kavita/Suwayomi)</option>
                    <option value="json_manifest">JSON 漫画清单 (comics / chapters / pages)</option>
                  </select>
                </div>
                <div className="settings-grid-2">
                  <input
                    type="url"
                    placeholder="书源 URL（如 http://127.0.0.1:25600/opds/v1.2/catalog）"
                    value={newSrcUrl}
                    onChange={(e) => setNewSrcUrl(e.target.value)}
                  />
                  <input
                    type="text"
                    placeholder="可选 Authorization 头（如 Basic ... 或 Bearer ...）"
                    value={newSrcAuthHeader}
                    onChange={(e) => setNewSrcAuthHeader(e.target.value)}
                  />
                </div>
                <div className="btn-row" style={{ justifyContent: "flex-end" }}>
                  <button type="submit" className="btn btn-sm primary" disabled={loading}>
                    保存漫画源
                  </button>
                </div>
              </form>
            </div>
          )}

          {tab === "direct" && (
            <div className="source-pane">
              <div className="field">
                <label>粘贴远程 .cbz/.zip 直链、单话图集网页 URL，或每行一个图片直链</label>
                <textarea
                  rows={4}
                  value={directInput}
                  placeholder={"https://example.com/chapter-01/001.jpg\nhttps://example.com/chapter-01/002.jpg\n或粘贴单个 .cbz / MangaDex 章节 / JSON 清单链接"}
                  onChange={(e) => setDirectInput(e.target.value)}
                />
              </div>
              <div className="btn-row" style={{ justifyContent: "space-between" }}>
                <input
                  type="text"
                  style={{ width: 180 }}
                  value={chapterTitle}
                  placeholder="章节名称（如 第01话）"
                  onChange={(e) => setChapterTitle(e.target.value)}
                />
                <div className="btn-row">
                  <button
                    type="button"
                    className="btn"
                    disabled={loading || !directInput.trim()}
                    onClick={() => void probeSource({ kind: "direct", query: directInput })}
                  >
                    {loading ? "解析中..." : "先预览页数"}
                  </button>
                  <button
                    type="button"
                    className="btn primary"
                    disabled={launching || jobRunning || !directInput.trim()}
                    onClick={() =>
                      void startStreamingJob({
                        source_kind: "direct_urls",
                        work_title: workTitle.trim() || "在线直链漫画",
                        chapter_title: chapterTitle.trim() || "第01话",
                        direct_input: directInput,
                      })
                    }
                  >
                    <IconZap size={13} /> 直接边下边译
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Unified Browse Results (Comics / Chapters / Pages / Archive) */}
          {browse && (
            <div className="browse-results">
              <div className="browse-results-head">
                <strong>
                  {browse.title || "漫源结果"}
                  {selectedComic ? ` · ${selectedComic.title}` : ""}
                </strong>
                <span className="mono count-pill">
                  {browse.kind === "comics"
                    ? `${browse.comics?.length || 0} 部作品`
                    : browse.kind === "chapters"
                    ? `${browse.chapters?.length || 0} 个章节`
                    : browse.kind === "pages"
                    ? `${browse.pages?.length || 0} 页图片`
                    : "远程归档"}
                </span>
              </div>

              {browse.comics && browse.comics.length > 0 && (
                <div className="source-comic-grid">
                  {browse.comics.map((c) => (
                    <button
                      key={`${c.source_kind}-${c.id}-${c.url}`}
                      type="button"
                      className="source-comic-card"
                      onClick={() => handleOpenComic(c)}
                    >
                      <div className="source-comic-cover">
                        {c.cover_url ? (
                          <img
                            src={`/api/v1/sources/cover?url=${encodeURIComponent(c.cover_url)}`}
                            alt={c.title}
                            loading="lazy"
                          />
                        ) : (
                          <span className="mono">COMIC</span>
                        )}
                      </div>
                      <div className="source-comic-info">
                        <div className="source-comic-title">{c.title}</div>
                        {c.description && <div className="source-comic-desc">{c.description}</div>}
                        <div className="mono source-comic-meta">
                          {c.languages && c.languages.length > 0 ? c.languages.slice(0, 5).join(" / ") : c.source_kind}
                        </div>
                      </div>
                    </button>
                  ))}
                </div>
              )}

              {browse.chapters && browse.chapters.length > 0 && (
                <div className="chapter-list">
                  {browse.chapters.map((ch) => (
                    <div key={`${ch.id}-${ch.url}`} className="chapter-row">
                      <div style={{ minWidth: 0, flex: 1 }}>
                        <div className="chapter-title">{ch.title}</div>
                        <div className="mono chapter-sub">
                          {ch.language ? `${ch.language.toUpperCase()} · ` : ""}
                          {ch.page_count ? `${ch.page_count} 页` : ch.archive_url ? "归档直下" : "在线章节"}
                        </div>
                      </div>
                      <button
                        type="button"
                        className="btn btn-sm primary"
                        disabled={launching || jobRunning}
                        onClick={() => handleLaunchChapter(ch)}
                      >
                        <IconZap size={13} /> {launching ? "启动中..." : "边下边译"}
                      </button>
                    </div>
                  ))}
                </div>
              )}

              {browse.pages && browse.pages.length > 0 && (
                <div className="direct-ready-box">
                  <div>
                    已解析到 <strong className="mono">{browse.pages.length}</strong> 页在线漫画图片
                    {maxPages > 0 && browse.pages.length > maxPages
                      ? `（将按设置先译前 ${maxPages} 页）`
                      : ""}
                    ，启动后第 1 页下完立即进入翻译，其余页面后台并发预下载。
                  </div>
                  <button
                    type="button"
                    className="btn primary"
                    disabled={launching || jobRunning}
                    onClick={() =>
                      void startStreamingJob({
                        source_kind: browse.source_kind,
                        work_title: workTitle.trim() || browse.title || "在线漫源作品",
                        chapter_title: chapterTitle.trim() || "第01话",
                        pages: browse.pages,
                        source_id: activeSourceId || undefined,
                      })
                    }
                  >
                    <IconZap size={13} /> {launching ? "启动中..." : "立即开始边下边译"}
                  </button>
                </div>
              )}

              {browse.archive_url && (
                <div className="direct-ready-box">
                  <div className="mono" style={{ wordBreak: "break-all" }}>
                    远程归档: {browse.archive_url}
                  </div>
                  <button
                    type="button"
                    className="btn primary"
                    disabled={launching || jobRunning}
                    onClick={() =>
                      void startStreamingJob({
                        source_kind: browse.source_kind || "direct_archive",
                        work_title: workTitle.trim() || browse.title || "远程归档漫画",
                        chapter_title: chapterTitle.trim() || "第01话",
                        archive_url: browse.archive_url,
                        source_id: activeSourceId || undefined,
                      })
                    }
                  >
                    <IconZap size={13} /> {launching ? "启动中..." : "下载归档并立即翻译"}
                  </button>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
