import React, { useEffect, useMemo, useRef, useState } from "react";
import { IconClose, IconFolder, IconUpload } from "./Icons";
import {
  extractFilesFromDataTransfer,
  filterAndSortComicFiles,
  inferFolderOrFileTitle,
} from "../utils/folderFiles";

interface UploadModalProps {
  open: boolean;
  onClose: () => void;
  initialFiles: File[];
  initialFolderName?: string;
  defaultTargetLang: string;
  llmReady: boolean;
  busy: boolean;
  jobRunning: boolean;
  onSubmit: (payload: {
    files: File[];
    refFiles: File[];
    workTitle: string;
    chapterTitle: string;
    formatProfile: string;
    targetLang: string;
    analyzeOnly: boolean;
  }) => Promise<boolean>;
}

export default function UploadModal({
  open,
  onClose,
  initialFiles,
  initialFolderName = "",
  defaultTargetLang,
  llmReady,
  busy,
  jobRunning,
  onSubmit,
}: UploadModalProps) {
  const [files, setFiles] = useState<File[]>([]);
  const [refFiles, setRefFiles] = useState<File[]>([]);
  const [workTitle, setWorkTitle] = useState<string>("未命名漫画");
  const [workTitleEdited, setWorkTitleEdited] = useState<boolean>(false);
  const [chapterTitle, setChapterTitle] = useState<string>("第01话");
  const [formatProfile, setFormatProfile] = useState<string>("auto");
  const [targetLang, setTargetLang] = useState<string>(defaultTargetLang || "zh-CN");
  const [analyzeOnly, setAnalyzeOnly] = useState<boolean>(!llmReady);
  const [dragActive, setDragActive] = useState<boolean>(false);
  const [localError, setLocalError] = useState<string>("");

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const folderInputRef = useRef<HTMLInputElement | null>(null);
  const refInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    if (!open) return;
    setLocalError("");
    setTargetLang(defaultTargetLang || "zh-CN");
    if (!llmReady) setAnalyzeOnly(true);
    if (initialFiles.length > 0) {
      const cleaned = filterAndSortComicFiles(initialFiles);
      setFiles(cleaned);
      const inferred = inferFolderOrFileTitle(cleaned, initialFolderName);
      if (inferred && !workTitleEdited) {
        setWorkTitle(inferred);
      }
    }
  }, [open, initialFiles, initialFolderName, defaultTargetLang, llmReady, workTitleEdited]);

  const previewUrls = useMemo(() => {
    return files.slice(0, 6).map((f) => {
      const isImage =
        f.type.startsWith("image/") || /\.(png|jpe?g|webp|gif|avif)$/i.test(f.name);
      return {
        name: f.name,
        isImage,
        url: isImage ? URL.createObjectURL(f) : "",
      };
    });
  }, [files]);

  useEffect(() => {
    return () => {
      previewUrls.forEach((p) => {
        if (p.url) URL.revokeObjectURL(p.url);
      });
    };
  }, [previewUrls]);

  if (!open) return null;

  const acceptFiles = (incoming: FileList | File[] | null, explicitFolderName = "") => {
    if (!incoming || incoming.length === 0) return;
    const cleaned = filterAndSortComicFiles(Array.from(incoming));
    if (cleaned.length === 0) {
      setLocalError("未在所选内容中找到支持的漫画文件（PNG / JPG / WebP / GIF / AVIF / CBZ / ZIP）");
      return;
    }
    setFiles(cleaned);
    setLocalError("");
    const inferred = inferFolderOrFileTitle(cleaned, explicitFolderName);
    if (inferred && !workTitleEdited) {
      setWorkTitle(inferred);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (files.length === 0) {
      setLocalError("请先拖拽或选择至少一张漫画图片 / .cbz / .zip 归档");
      return;
    }
    setLocalError("");
    const ok = await onSubmit({
      files,
      refFiles,
      workTitle: workTitle.trim() || "未命名漫画",
      chapterTitle: chapterTitle.trim() || "第01话",
      formatProfile,
      targetLang: targetLang.trim() || "zh-CN",
      analyzeOnly,
    });
    if (ok) {
      setFiles([]);
      setRefFiles([]);
      setWorkTitleEdited(false);
      onClose();
    }
  };

  return (
    <div
      className="modal-backdrop"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <form
        className="panel-card modal-dialog upload-modal"
        onSubmit={(e) => void handleSubmit(e)}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="panel-title">
          <span className="inline-icon-text">
            <IconUpload size={15} /> 导入漫画与一键无损翻译
          </span>
          <button type="button" className="btn btn-sm" onClick={onClose}>
            <IconClose size={13} /> 关闭
          </button>
        </div>

        {localError && <div className="error-banner">{localError}</div>}

        <div
          className={`dropzone upload-dropzone ${dragActive ? "drag-active" : ""}`}
          onClick={() => fileInputRef.current?.click()}
          onDragOver={(e) => {
            e.preventDefault();
            setDragActive(true);
          }}
          onDragLeave={() => setDragActive(false)}
          onDrop={(e) => {
            e.preventDefault();
            e.stopPropagation();
            setDragActive(false);
            void extractFilesFromDataTransfer(e.dataTransfer).then(({ files: dropped, folderName }) => {
              acceptFiles(dropped, folderName);
            });
          }}
        >
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept="image/*,.avif,.cbz,.zip"
            style={{ display: "none" }}
            onChange={(e) => acceptFiles(e.target.files)}
          />
          <input
            ref={folderInputRef}
            type="file"
            multiple
            // @ts-expect-error webkitdirectory is supported by all modern desktop browsers
            webkitdirectory=""
            directory=""
            style={{ display: "none" }}
            onChange={(e) => acceptFiles(e.target.files)}
          />
          <div className="dropzone-icon">
            <IconUpload size={22} />
          </div>
          <div className="dropzone-title">
            {files.length > 0
              ? `已选中 ${files.length} 个漫画文件（可继续点击或拖入替换）`
              : "拖拽漫画文件夹、图片 (PNG/JPG/WebP/AVIF) 或 .cbz/.zip 到此处"}
          </div>
          <div
            className="btn-row"
            style={{ justifyContent: "center", marginTop: 6 }}
            onClick={(e) => e.stopPropagation()}
          >
            <button
              type="button"
              className="btn btn-sm"
              onClick={() => fileInputRef.current?.click()}
            >
              <IconUpload size={13} /> 选择图片 / CBZ
            </button>
            <button
              type="button"
              className="btn btn-sm cyan"
              onClick={() => folderInputRef.current?.click()}
            >
              <IconFolder size={13} /> 选择漫画文件夹
            </button>
          </div>
          <div className="dropzone-sub mono">
            自动递归读取文件夹内全部漫画页并按自然页序排列 · 自动提取文件夹名作为书名
          </div>
        </div>

        {previewUrls.length > 0 && (
          <div className="upload-preview-strip">
            {previewUrls.map((p, idx) => (
              <div key={`${p.name}-${idx}`} className="upload-thumb-card" title={p.name}>
                {p.isImage ? (
                  <img src={p.url} alt={p.name} />
                ) : (
                  <div className="upload-archive-icon mono">CBZ/ZIP</div>
                )}
                <span className="mono">{p.name}</span>
              </div>
            ))}
            {files.length > 6 && (
              <div className="upload-thumb-more mono">+{files.length - 6} 页</div>
            )}
          </div>
        )}

        <div className="settings-grid-2">
          <div className="field">
            <label>作品名称（跨章节绑定角色人设与术语）</label>
            <input
              type="text"
              value={workTitle}
              onChange={(e) => {
                setWorkTitleEdited(true);
                setWorkTitle(e.target.value);
              }}
              placeholder="如：电锯人 / Always Human"
            />
          </div>
          <div className="field">
            <label>章节标题</label>
            <input
              type="text"
              value={chapterTitle}
              onChange={(e) => setChapterTitle(e.target.value)}
              placeholder="如：第01话"
            />
          </div>
        </div>

        <div className="settings-grid-2">
          <div className="field">
            <label>版式预设</label>
            <select value={formatProfile} onChange={(e) => setFormatProfile(e.target.value)}>
              <option value="auto">Auto · 自动识别日漫 / 条漫 / 美漫</option>
              <option value="manga_rtl">Manga RTL · 竖排日漫（网点 / ルビ）</option>
              <option value="webtoon_vertical">Webtoon · 长条漫跨切片无缝画布</option>
              <option value="comic_ltr">Comic LTR · 横排全彩 / 欧美漫</option>
            </select>
          </div>
          <div className="field">
            <label>目标语言</label>
            <select value={targetLang} onChange={(e) => setTargetLang(e.target.value)}>
              <option value="zh-CN">简体中文 (zh-CN)</option>
              <option value="zh-TW">繁體中文 (zh-TW)</option>
              <option value="en-US">English (en-US)</option>
              <option value="ja-JP">日本語 (ja-JP)</option>
              <option value="ko-KR">한국어 (ko-KR)</option>
            </select>
          </div>
        </div>

        <details className="subtle-details">
          <summary>高级选项：异版无字参考图借底（USAC_MAGSAC 亚像素配准）</summary>
          <div className="field" style={{ marginTop: 8 }}>
            <input
              ref={refInputRef}
              type="file"
              multiple
              accept="image/*,.avif,.cbz,.zip"
              onChange={(e) => setRefFiles(e.target.files ? Array.from(e.target.files) : [])}
            />
            {refFiles.length > 0 && (
              <span className="mono" style={{ fontSize: 11, color: "var(--zinc-note)" }}>
                已附加 {refFiles.length} 个参考图源
              </span>
            )}
          </div>
        </details>

        <div className="upload-footer">
          <label className="check-inline">
            <input
              type="checkbox"
              checked={analyzeOnly}
              onChange={(e) => setAnalyzeOnly(e.target.checked)}
            />
            <span>仅物理去字与分层反编译（不调用大模型）</span>
          </label>
          <div className="btn-row">
            {files.length > 0 && (
              <button
                type="button"
                className="btn"
                onClick={() => {
                  setFiles([]);
                  if (fileInputRef.current) fileInputRef.current.value = "";
                }}
              >
                清空文件
              </button>
            )}
            <button
              type="submit"
              className="btn primary"
              disabled={busy || jobRunning || files.length === 0}
            >
              {jobRunning ? "后台翻译进行中..." : busy ? "提交中..." : "开始翻译与无损合成"}
            </button>
          </div>
        </div>
      </form>
    </div>
  );
}
