import { useState } from "react";
import type { TranslationJob, WorkItem } from "../types";
import {
  IconCheck,
  IconClose,
  IconDownload,
  IconEdit,
  IconGlobe,
  IconPlus,
  IconSparkles,
  IconTrash,
} from "./Icons";

interface BookshelfViewProps {
  works: WorkItem[];
  cacheBust: number;
  activeJob: TranslationJob | null;
  onOpenWork: (workId: number) => void;
  onRenameWork: (workId: number, title: string) => Promise<void>;
  onDeleteWork: (workId: number) => Promise<void>;
  onOpenUpload: () => void;
  onOpenSources: () => void;
  onCancelJob: (jobId: string) => void;
}

function profileLabel(p: string): string {
  switch (p) {
    case "manga_rtl":
      return "日漫右翻";
    case "comic_ltr":
      return "美漫左翻";
    case "webtoon_vertical":
      return "竖滑条漫";
    default:
      return "自动版式";
  }
}

export default function BookshelfView({
  works,
  cacheBust,
  activeJob,
  onOpenWork,
  onRenameWork,
  onDeleteWork,
  onOpenUpload,
  onOpenSources,
  onCancelJob,
}: BookshelfViewProps) {
  const [query, setQuery] = useState("");
  const [confirmDeleteId, setConfirmDeleteId] = useState<number | null>(null);
  const [editingId, setEditingId] = useState<number | null>(null);
  const [editingTitle, setEditingTitle] = useState("");
  const [busyId, setBusyId] = useState<number | null>(null);

  const filtered = works.filter((w) => {
    const q = query.trim().toLowerCase();
    if (!q) return true;
    return w.title.toLowerCase().includes(q) || w.format_profile.toLowerCase().includes(q);
  });

  const isRunning = activeJob?.status === "running";
  const jobTotal = Math.max(1, activeJob?.total_units || 1);
  const dlPct = activeJob?.downloaded_units
    ? Math.min(100, Math.round((activeJob.downloaded_units / jobTotal) * 100))
    : 0;
  const donePct = activeJob
    ? Math.min(100, Math.round((activeJob.completed_units / jobTotal) * 100))
    : 0;

  return (
    <main className="bookshelf-page">
      <div className="bookshelf-header">
        <div className="bookshelf-title-block">
          <h1>我的漫画书架</h1>
          <span className="mono count-pill">{works.length} 本作品</span>
        </div>
        <div className="bookshelf-actions">
          <input
            type="text"
            className="bookshelf-search"
            placeholder="搜索书名或版式..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <button type="button" className="btn cyan" onClick={onOpenSources}>
            <IconGlobe size={14} /> 漫源 · 边下边译
          </button>
          <button type="button" className="btn primary" onClick={onOpenUpload}>
            <IconPlus size={14} /> 本地导入漫画
          </button>
        </div>
      </div>

      {activeJob && (isRunning || activeJob.status === "failed") && (
        <div className={`shelf-job-banner ${activeJob.status === "failed" ? "is-error" : ""}`}>
          <div className="shelf-job-top">
            <div>
              <strong>{activeJob.work_title}</strong>
              {activeJob.chapter_title ? ` · ${activeJob.chapter_title}` : ""}
              <span className="mono" style={{ marginLeft: 10, fontSize: 11 }}>
                {activeJob.message}
              </span>
            </div>
            <div className="btn-row">
              {activeJob.work_id > 0 && (
                <button
                  type="button"
                  className="btn btn-sm cyan"
                  onClick={() => onOpenWork(activeJob.work_id)}
                >
                  立即进本边看边等 →
                </button>
              )}
              {isRunning && (
                <button
                  type="button"
                  className="btn btn-sm"
                  onClick={() => onCancelJob(activeJob.id)}
                >
                  中止任务
                </button>
              )}
            </div>
          </div>
          {isRunning && (
            <div className="dual-progress-stack">
              {typeof activeJob.downloaded_units === "number" && activeJob.downloaded_units > 0 && (
                <div className="progress-line">
                  <span className="mono progress-lbl">
                    <IconDownload size={12} /> 下载 {activeJob.downloaded_units}/{activeJob.total_units}
                  </span>
                  <div className="progress-track">
                    <div className="progress-fill cyan" style={{ width: `${Math.max(4, dlPct)}%` }} />
                  </div>
                </div>
              )}
              <div className="progress-line">
                <span className="mono progress-lbl">
                  <IconSparkles size={12} /> 译制 {activeJob.completed_units}/{activeJob.total_units}
                </span>
                <div className="progress-track">
                  <div className="progress-fill" style={{ width: `${Math.max(4, donePct)}%` }} />
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {filtered.length === 0 ? (
        <div className="bookshelf-empty">
          <div className="hero-card">
            <h2>{works.length === 0 ? "书架还是空的" : "没有找到匹配的漫画"}</h2>
            <p>
              {works.length === 0
                ? "直接把本地漫画图片、文件夹或 .cbz/.zip 拖进窗口，或点击上方「漫源 · 边下边译」从在线漫画源边下边看。"
                : "换个关键词试试，或清空搜索框查看全部作品。"}
            </p>
            <div className="btn-row">
              <button type="button" className="btn cyan" onClick={onOpenSources}>
                <IconGlobe size={14} /> 从漫源边下边译
              </button>
              <button type="button" className="btn primary" onClick={onOpenUpload}>
                <IconPlus size={14} /> 导入本地漫画
              </button>
            </div>
          </div>
        </div>
      ) : (
        <div className="bookshelf-grid">
          {filtered.map((w) => {
            const reviewed = w.reviewed_count ?? 0;
            const totalPages = w.page_count || 0;
            const chapters = w.chapter_count || (totalPages > 0 ? 1 : 0);
            const coverUrl =
              w.cover_page_id && w.cover_page_id > 0
                ? `/api/v1/pages/${w.cover_page_id}/composite?t=${cacheBust}`
                : "";
            const isConfirming = confirmDeleteId === w.id;
            const isEditing = editingId === w.id;
            const isBusy = busyId === w.id;

            return (
              <article key={w.id} className="book-card">
                <div
                  className="book-cover"
                  onClick={() => onOpenWork(w.id)}
                  role="button"
                  tabIndex={0}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      onOpenWork(w.id);
                    }
                  }}
                >
                  {coverUrl ? (
                    <img src={coverUrl} alt={w.title} loading="lazy" />
                  ) : (
                    <div className="book-cover-placeholder mono">NO COVER</div>
                  )}
                  <span className="book-badge-tl mono">{profileLabel(w.format_profile)}</span>
                  <span
                    className={`book-badge-tr mono ${
                      w.review_count > 0 ? "warn" : totalPages > 0 ? "ok" : ""
                    }`}
                  >
                    {w.review_count > 0 ? (
                      `${w.review_count} 待审`
                    ) : totalPages > 0 ? (
                      <>
                        <IconCheck size={11} /> 就绪
                      </>
                    ) : (
                      "空本"
                    )}
                  </span>
                  <div className="book-cover-hover">打开阅读 / 精修 →</div>
                </div>

                <div className="book-body">
                  {isEditing ? (
                    <form
                      className="book-rename-row"
                      onSubmit={async (e) => {
                        e.preventDefault();
                        if (!editingTitle.trim()) return;
                        setBusyId(w.id);
                        try {
                          await onRenameWork(w.id, editingTitle.trim());
                          setEditingId(null);
                        } finally {
                          setBusyId(null);
                        }
                      }}
                    >
                      <input
                        type="text"
                        value={editingTitle}
                        autoFocus
                        onChange={(e) => setEditingTitle(e.target.value)}
                      />
                      <button type="submit" className="btn btn-sm primary" disabled={isBusy}>
                        保存
                      </button>
                      <button
                        type="button"
                        className="btn btn-sm"
                        onClick={() => setEditingId(null)}
                        title="取消"
                      >
                        <IconClose size={13} />
                      </button>
                    </form>
                  ) : (
                    <div className="book-title-row">
                      <h3 title={w.title} onClick={() => onOpenWork(w.id)}>
                        {w.title}
                      </h3>
                      <button
                        type="button"
                        className="icon-mini-btn"
                        title="重命名作品"
                        onClick={() => {
                          setEditingId(w.id);
                          setEditingTitle(w.title);
                          setConfirmDeleteId(null);
                        }}
                      >
                        <IconEdit size={12} />
                      </button>
                    </div>
                  )}

                  <div className="book-meta mono">
                    <span>
                      {chapters} 话 · {totalPages} 页
                    </span>
                    <span>
                      → {w.target_lang}
                      {reviewed > 0 ? ` · 已审 ${reviewed}/${totalPages}` : ""}
                    </span>
                  </div>

                  {isConfirming ? (
                    <div className="book-delete-confirm">
                      <span className="mono">确认永久删除整本？</span>
                      <div className="btn-row">
                        <button
                          type="button"
                          className="btn btn-sm primary"
                          disabled={isBusy}
                          onClick={async () => {
                            setBusyId(w.id);
                            try {
                              await onDeleteWork(w.id);
                              setConfirmDeleteId(null);
                            } finally {
                              setBusyId(null);
                            }
                          }}
                        >
                          {isBusy ? "删除中..." : "确认删除"}
                        </button>
                        <button
                          type="button"
                          className="btn btn-sm"
                          onClick={() => setConfirmDeleteId(null)}
                        >
                          取消
                        </button>
                      </div>
                    </div>
                  ) : (
                    <div className="book-footer">
                      <button
                        type="button"
                        className="btn btn-sm primary"
                        style={{ flex: 1 }}
                        onClick={() => onOpenWork(w.id)}
                      >
                        阅读 / 精修
                      </button>
                      {totalPages > 0 && (
                        <a
                          className="btn btn-sm"
                          href={`/api/v1/works/${w.id}/export/cbz`}
                          download
                          title="导出整本 CBZ"
                        >
                          <IconDownload size={13} /> CBZ
                        </a>
                      )}
                      <button
                        type="button"
                        className="btn btn-sm danger-btn"
                        title="删除整本漫画"
                        onClick={() => setConfirmDeleteId(w.id)}
                      >
                        <IconTrash size={13} />
                      </button>
                    </div>
                  )}
                </div>
              </article>
            );
          })}
        </div>
      )}
    </main>
  );
}
