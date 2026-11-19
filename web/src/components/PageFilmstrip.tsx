import { useState } from "react";
import type { PageSummary, TranslationJob, WorkItem } from "../types";
import { IconArrowLeft, IconCheck, IconClose, IconDownload, IconPlus, IconSparkles } from "./Icons";

interface PageFilmstripProps {
  works: WorkItem[];
  selectedWorkId: number;
  onSelectWork: (id: number) => void;
  onBackToShelf: () => void;
  pages: PageSummary[];
  selectedPageId: number;
  onSelectPage: (id: number) => void;
  onDeletePage: (pageId: number) => Promise<void>;
  reviewOnly: boolean;
  onToggleReviewOnly: (val: boolean) => void;
  cacheBust: number;
  activeJob: TranslationJob | null;
  onCancelJob: (jobId: string) => void;
  onOpenUpload: () => void;
}

function formatDuration(ms: number): string {
  const s = Math.max(0, Math.round(ms / 1000));
  if (s < 60) return `${s}s`;
  return `${Math.floor(s / 60)}m${String(s % 60).padStart(2, "0")}s`;
}

function stageLabel(stage: string): string {
  switch (stage) {
    case "queued":
      return "排队中";
    case "downloading":
      return "边下边译";
    case "preparing":
      return "切分画布";
    case "analyzing":
      return "物理去字";
    case "analyzed":
      return "底图完成";
    case "translating":
      return "视觉翻译";
    case "rendering":
      return "SDF 嵌字";
    case "page_done":
      return "整页完成";
    case "done":
      return "全部完成";
    case "canceled":
      return "已中止";
    case "failed":
      return "失败";
    default:
      return stage || "处理中";
  }
}

export default function PageFilmstrip({
  works,
  selectedWorkId,
  onSelectWork,
  onBackToShelf,
  pages,
  selectedPageId,
  onSelectPage,
  onDeletePage,
  reviewOnly,
  onToggleReviewOnly,
  cacheBust,
  activeJob,
  onCancelJob,
  onOpenUpload,
}: PageFilmstripProps) {
  const [confirmDelPageId, setConfirmDelPageId] = useState<number | null>(null);
  const [deletingId, setDeletingId] = useState<number | null>(null);

  const reviewCount = pages.filter((p) => p.needs_review).length;
  const isRunning = activeJob?.status === "running";
  const jobTotal = Math.max(1, activeJob?.total_units || 1);
  const dlPct = activeJob?.downloaded_units
    ? Math.min(100, Math.round((activeJob.downloaded_units / jobTotal) * 100))
    : 0;
  const jobPct = activeJob
    ? Math.min(
        100,
        Math.round(
          ((activeJob.status === "completed" ? jobTotal : activeJob.completed_units) / jobTotal) *
            100
        )
      )
    : 0;

  // Group pages by chapter when multiple chapters exist
  const chapterNames = Array.from(new Set(pages.map((p) => p.chapter_title || "默认章节")));
  const showChapterHeaders = chapterNames.length > 1;

  return (
    <aside className="studio-sidebar left-sidebar">
      <div className="sidebar-section">
        <div className="sidebar-label-row">
          <button type="button" className="btn btn-sm" onClick={onBackToShelf}>
            <IconArrowLeft size={13} /> 书架
          </button>
          <span className="mono count-pill">{pages.length} 页</span>
        </div>
        <select
          aria-label="切换漫画作品"
          value={selectedWorkId}
          onChange={(e) => onSelectWork(Number(e.target.value))}
        >
          {works.map((w) => (
            <option key={w.id} value={w.id}>
              {w.title} ({w.page_count}页)
            </option>
          ))}
        </select>

        <div className="filter-pills">
          <button
            type="button"
            className={`pill-btn ${!reviewOnly ? "active" : ""}`}
            onClick={() => onToggleReviewOnly(false)}
          >
            全部 ({pages.length})
          </button>
          <button
            type="button"
            className={`pill-btn ${reviewOnly ? "active" : ""}`}
            onClick={() => onToggleReviewOnly(true)}
          >
            待审{reviewCount > 0 ? ` (${reviewCount})` : ""}
          </button>
        </div>
      </div>

      {activeJob && (isRunning || activeJob.status === "failed") && (
        <div className="job-card-compact">
          <div className="job-card-head">
            <span className="job-card-title">{activeJob.work_title}</span>
            <span className="mono" style={{ fontSize: 11 }}>
              {formatDuration(activeJob.elapsed_ms)}
            </span>
          </div>
          <div className="job-card-stage">
            {stageLabel(activeJob.current_stage)} · {activeJob.message}
          </div>
          {typeof activeJob.downloaded_units === "number" && activeJob.downloaded_units > 0 && (
            <div className="progress-track" title={`已下载 ${activeJob.downloaded_units}/${activeJob.total_units}`}>
              <div className="progress-fill cyan" style={{ width: `${Math.max(4, dlPct)}%` }} />
            </div>
          )}
          <div className="progress-track">
            <div className="progress-fill" style={{ width: `${Math.max(4, jobPct)}%` }} />
          </div>
          <div className="job-card-foot mono">
            <span className="inline-icon-text">
              {typeof activeJob.downloaded_units === "number" && activeJob.downloaded_units > 0 ? (
                <>
                  <IconDownload size={11} /> {activeJob.downloaded_units} <IconSparkles size={11} />{" "}
                  {activeJob.completed_units}/{activeJob.total_units}
                </>
              ) : (
                `${activeJob.completed_units}/${activeJob.total_units} 页`
              )}
            </span>
            {isRunning && (
              <button
                type="button"
                className="link-btn danger"
                onClick={() => onCancelJob(activeJob.id)}
              >
                中止
              </button>
            )}
          </div>
        </div>
      )}

      <div className="filmstrip-scroll">
        {pages.length === 0 && (
          <div className="empty-filmstrip">
            <div style={{ fontSize: 12, color: "var(--zinc-note)", marginBottom: 10 }}>
              {reviewOnly ? "当前没有待审页面" : "本书还没有漫画页面"}
            </div>
            {!reviewOnly && (
              <button type="button" className="btn primary btn-sm" onClick={onOpenUpload}>
                <IconPlus size={13} /> 导入页面
              </button>
            )}
          </div>
        )}

        {pages.map((p, idx) => {
          const active = p.id === selectedPageId;
          const prevChap = idx > 0 ? pages[idx - 1].chapter_title : "";
          const showHeader = showChapterHeaders && (idx === 0 || p.chapter_title !== prevChap);
          const isConfirmDel = confirmDelPageId === p.id;
          return (
            <div key={p.id}>
              {showHeader && (
                <div className="chapter-divider mono">{p.chapter_title || "默认章节"}</div>
              )}
              <div className={`thumb-card ${active ? "active" : ""}`}>
                <div
                  className="thumb-img-wrap"
                  role="button"
                  tabIndex={0}
                  onClick={() => onSelectPage(p.id)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      onSelectPage(p.id);
                    }
                  }}
                >
                  <img
                    src={`/api/v1/pages/${p.id}/composite?t=${cacheBust}`}
                    alt={p.filename}
                    loading="lazy"
                  />
                  <span className="thumb-index mono">#{idx + 1}</span>
                  <span className={`thumb-status mono ${p.needs_review ? "warn" : "ok"}`}>
                    {p.status === "reviewed" ? (
                      <IconCheck size={11} />
                    ) : p.needs_review ? (
                      "待审"
                    ) : (
                      "已译"
                    )}
                  </span>
                </div>
                <div className="thumb-meta">
                  <div className="thumb-meta-row">
                    <div
                      className="thumb-name"
                      title={p.filename}
                      onClick={() => onSelectPage(p.id)}
                    >
                      {p.filename}
                    </div>
                    {isConfirmDel ? (
                      <div className="btn-row" style={{ gap: 3 }}>
                        <button
                          type="button"
                          className="mini-del-confirm"
                          disabled={deletingId === p.id}
                          onClick={async (e) => {
                            e.stopPropagation();
                            setDeletingId(p.id);
                            try {
                              await onDeletePage(p.id);
                              setConfirmDelPageId(null);
                            } finally {
                              setDeletingId(null);
                            }
                          }}
                        >
                          删
                        </button>
                        <button
                          type="button"
                          className="mini-del-cancel"
                          onClick={(e) => {
                            e.stopPropagation();
                            setConfirmDelPageId(null);
                          }}
                        >
                          取消
                        </button>
                      </div>
                    ) : (
                      <button
                        type="button"
                        className="thumb-del-btn"
                        title="删除此页"
                        onClick={(e) => {
                          e.stopPropagation();
                          setConfirmDelPageId(p.id);
                        }}
                      >
                        <IconClose size={12} />
                      </button>
                    )}
                  </div>
                  <div className="thumb-sub mono">
                    {p.total_nodes} 框 · {p.chapter_title}
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </aside>
  );
}
