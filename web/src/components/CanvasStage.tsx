import React, { useCallback, useEffect, useRef, useState } from "react";
import type { PageDOM, TextNode } from "../types";
import {
  IconBoxSelect,
  IconChevronLeft,
  IconChevronRight,
  IconCursor,
  IconHand,
  IconLayers,
  IconMaximize,
  IconMinus,
  IconPlus,
} from "./Icons";

export type CanvasViewMode = "composite" | "split" | "side-by-side" | "raw";
export type ZoomMode = "fit-h" | "fit-w" | "custom";
export type CanvasToolMode = "select" | "draw" | "pan";

interface CanvasStageProps {
  pageId: number;
  pageDOM: PageDOM | null;
  cacheBust: number;
  viewMode: CanvasViewMode;
  onChangeViewMode: (m: CanvasViewMode) => void;
  toolMode: CanvasToolMode;
  onChangeToolMode: (t: CanvasToolMode) => void;
  showOverlay: boolean;
  onToggleOverlay: () => void;
  peekingRaw: boolean;
  selectedNodeIdx: number;
  onSelectNode: (idx: number) => void;
  onAddManualBox: (bbox: [number, number, number, number]) => void;
  onMoveResizeNode: (idx: number, bbox: [number, number, number, number]) => void;
  zenMode: boolean;
  onToggleZen: () => void;
  canPrevPage: boolean;
  canNextPage: boolean;
  pageCounterLabel: string;
  onPrevPage: () => void;
  onNextPage: () => void;
  onOpenUpload: () => void;
}

type DragAction =
  | { kind: "draw"; startX: number; startY: number; curX: number; curY: number }
  | {
      kind: "move";
      idx: number;
      startX: number;
      startY: number;
      origBox: [number, number, number, number];
    }
  | {
      kind: "resize";
      idx: number;
      corner: "nw" | "ne" | "sw" | "se";
      startX: number;
      startY: number;
      origBox: [number, number, number, number];
    };

export default function CanvasStage({
  pageId,
  pageDOM,
  cacheBust,
  viewMode,
  onChangeViewMode,
  toolMode,
  onChangeToolMode,
  showOverlay,
  onToggleOverlay,
  peekingRaw,
  selectedNodeIdx,
  onSelectNode,
  onAddManualBox,
  onMoveResizeNode,
  zenMode,
  onToggleZen,
  canPrevPage,
  canNextPage,
  pageCounterLabel,
  onPrevPage,
  onNextPage,
  onOpenUpload,
}: CanvasStageProps) {
  const [zoomMode, setZoomMode] = useState<ZoomMode>("fit-h");
  const [zoomPct, setZoomPct] = useState<number>(100);
  const [splitPos, setSplitPos] = useState<number>(50);
  const [draggingSplit, setDraggingSplit] = useState<boolean>(false);
  const [hoveredNode, setHoveredNode] = useState<TextNode | null>(null);
  const [dragAction, setDragAction] = useState<DragAction | null>(null);
  const [panState, setPanState] = useState<{
    startX: number;
    startY: number;
    scrollLeft: number;
    scrollTop: number;
  } | null>(null);

  const splitBoxRef = useRef<HTMLDivElement | null>(null);
  const viewportRef = useRef<HTMLDivElement | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);

  // Auto-select comfortable zoom mode for webtoons vs manga pages
  useEffect(() => {
    if (!pageDOM) return;
    if (pageDOM.height > pageDOM.width * 1.9) {
      setZoomMode("fit-w");
    } else {
      setZoomMode("fit-h");
    }
  }, [pageId, pageDOM?.width, pageDOM?.height]);

  const applyZoomDelta = useCallback((delta: number) => {
    setZoomMode("custom");
    setZoomPct((prev) => Math.max(25, Math.min(300, Math.round(prev + delta))));
  }, []);

  const handleWheel = (e: React.WheelEvent) => {
    if (e.ctrlKey || e.metaKey) {
      e.preventDefault();
      applyZoomDelta(e.deltaY < 0 ? 15 : -15);
    }
  };

  const updateSplitFromClientX = useCallback((clientX: number) => {
    const el = splitBoxRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    if (rect.width <= 0) return;
    const pct = ((clientX - rect.left) / rect.width) * 100;
    setSplitPos(Math.max(2, Math.min(98, pct)));
  }, []);

  useEffect(() => {
    if (!draggingSplit) return;
    const onMove = (e: MouseEvent) => updateSplitFromClientX(e.clientX);
    const onTouchMove = (e: TouchEvent) => {
      if (e.touches[0]) updateSplitFromClientX(e.touches[0].clientX);
    };
    const onUp = () => setDraggingSplit(false);
    window.addEventListener("mousemove", onMove);
    window.addEventListener("touchmove", onTouchMove, { passive: true });
    window.addEventListener("mouseup", onUp);
    window.addEventListener("touchend", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("touchmove", onTouchMove);
      window.removeEventListener("mouseup", onUp);
      window.removeEventListener("touchend", onUp);
    };
  }, [draggingSplit, updateSplitFromClientX]);

  // Smooth pan dragging on zoomed viewport
  useEffect(() => {
    if (!panState) return;
    const onMove = (e: MouseEvent) => {
      const vp = viewportRef.current;
      if (!vp) return;
      vp.scrollLeft = panState.scrollLeft - (e.clientX - panState.startX);
      vp.scrollTop = panState.scrollTop - (e.clientY - panState.startY);
    };
    const onUp = () => setPanState(null);
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, [panState]);

  // Map mouse client coordinates -> image pixel coordinates
  const clientToImg = useCallback(
    (clientX: number, clientY: number): [number, number] | null => {
      const svg = svgRef.current;
      if (!svg || !pageDOM) return null;
      const rect = svg.getBoundingClientRect();
      if (rect.width <= 0 || rect.height <= 0) return null;
      const ix = ((clientX - rect.left) / rect.width) * pageDOM.width;
      const iy = ((clientY - rect.top) / rect.height) * pageDOM.height;
      return [
        Math.max(0, Math.min(pageDOM.width, Math.round(ix))),
        Math.max(0, Math.min(pageDOM.height, Math.round(iy))),
      ];
    },
    [pageDOM]
  );

  // Interactive box draw / move / resize tracking
  useEffect(() => {
    if (!dragAction || !pageDOM) return;
    const onMove = (e: MouseEvent) => {
      const pt = clientToImg(e.clientX, e.clientY);
      if (!pt) return;
      const [ix, iy] = pt;
      if (dragAction.kind === "draw") {
        setDragAction((prev) =>
          prev && prev.kind === "draw" ? { ...prev, curX: ix, curY: iy } : prev
        );
      } else if (dragAction.kind === "move") {
        const dx = ix - dragAction.startX;
        const dy = iy - dragAction.startY;
        const [ox, oy, ow, oh] = dragAction.origBox;
        const nx = Math.max(0, Math.min(pageDOM.width - ow, ox + dx));
        const ny = Math.max(0, Math.min(pageDOM.height - oh, oy + dy));
        onMoveResizeNode(dragAction.idx, [nx, ny, ow, oh]);
      } else if (dragAction.kind === "resize") {
        const dx = ix - dragAction.startX;
        const dy = iy - dragAction.startY;
        const [ox, oy, ow, oh] = dragAction.origBox;
        let x0 = ox;
        let y0 = oy;
        let x1 = ox + ow;
        let y1 = oy + oh;
        if (dragAction.corner.includes("w")) x0 = Math.min(x1 - 14, Math.max(0, ox + dx));
        if (dragAction.corner.includes("e")) x1 = Math.max(x0 + 14, Math.min(pageDOM.width, ox + ow + dx));
        if (dragAction.corner.includes("n")) y0 = Math.min(y1 - 14, Math.max(0, oy + dy));
        if (dragAction.corner.includes("s")) y1 = Math.max(y0 + 14, Math.min(pageDOM.height, oy + oh + dy));
        onMoveResizeNode(dragAction.idx, [Math.round(x0), Math.round(y0), Math.round(x1 - x0), Math.round(y1 - y0)]);
      }
    };
    const onUp = () => {
      if (dragAction.kind === "draw") {
        const x0 = Math.min(dragAction.startX, dragAction.curX);
        const y0 = Math.min(dragAction.startY, dragAction.curY);
        const w = Math.abs(dragAction.curX - dragAction.startX);
        const h = Math.abs(dragAction.curY - dragAction.startY);
        if (w >= 14 && h >= 14) {
          onAddManualBox([x0, y0, w, h]);
          onChangeToolMode("select");
        }
      }
      setDragAction(null);
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, [dragAction, pageDOM, clientToImg, onAddManualBox, onMoveResizeNode, onChangeToolMode]);

  const rawUrl = pageId > 0 ? `/api/v1/pages/${pageId}/raw?t=${cacheBust}` : "";
  const compUrl = pageId > 0 ? `/api/v1/pages/${pageId}/composite?t=${cacheBust}` : "";
  const effectiveMode: CanvasViewMode = peekingRaw ? "raw" : viewMode;

  const imgStyle: React.CSSProperties = (() => {
    if (!pageDOM) return {};
    if (zoomMode === "fit-h") {
      const hOffset = viewMode === "side-by-side" ? (zenMode ? 116 : 186) : (zenMode ? 76 : 148);
      return {
        maxHeight: `calc(100vh - ${hOffset}px)`,
        maxWidth: "100%",
        width: "auto",
        height: "auto",
      };
    }
    if (zoomMode === "fit-w") {
      return {
        width: viewMode === "side-by-side" ? "100%" : "min(100%, 960px)",
        maxWidth: "100%",
        height: "auto",
      };
    }
    const targetW = Math.round((pageDOM.width * zoomPct) / 100);
    return { width: `${targetW}px`, height: "auto", maxWidth: "none" };
  })();

  const interactiveOverlayVisible =
    Boolean(pageDOM) && (showOverlay || toolMode === "draw") && effectiveMode !== "split";

  const renderOverlay = () => {
    if (!interactiveOverlayVisible || !pageDOM) return null;
    const handleSize = Math.max(8, Math.round(Math.min(pageDOM.width, pageDOM.height) * 0.012));

    return (
      <svg
        ref={svgRef}
        className={`overlay-svg ${toolMode === "draw" ? "cursor-crosshair" : ""}`}
        viewBox={`0 0 ${pageDOM.width} ${pageDOM.height}`}
        preserveAspectRatio="none"
        onMouseDown={(e) => {
          if (toolMode === "pan" || e.button === 1) return;
          if (toolMode === "draw" && e.button === 0) {
            e.stopPropagation();
            const pt = clientToImg(e.clientX, e.clientY);
            if (!pt) return;
            setDragAction({ kind: "draw", startX: pt[0], startY: pt[1], curX: pt[0], curY: pt[1] });
          }
        }}
      >
        {(pageDOM.nodes || []).map((n, idx) => {
          const [bx, by, bw, bh] = Array.isArray(n.bbox) && n.bbox.length === 4 ? n.bbox : [0, 0, 0, 0];
          const isSel = idx === selectedNodeIdx;
          const strokeCol = isSel
            ? "#E11D48"
            : n.uncertainty?.needs_review
            ? "#D97706"
            : "#0284C7";
          const pillW = (n.reading_order ?? idx) + 1 >= 10 ? 30 : 24;
          const corners: { key: "nw" | "ne" | "sw" | "se"; cx: number; cy: number; cursor: string }[] = [
            { key: "nw", cx: bx, cy: by, cursor: "nwse-resize" },
            { key: "ne", cx: bx + bw, cy: by, cursor: "nesw-resize" },
            { key: "sw", cx: bx, cy: by + bh, cursor: "nesw-resize" },
            { key: "se", cx: bx + bw, cy: by + bh, cursor: "nwse-resize" },
          ];
          return (
            <g
              key={n.id}
              onClick={(e) => {
                if (toolMode === "draw") return;
                e.stopPropagation();
                onSelectNode(idx);
              }}
              onMouseDown={(e) => {
                if (toolMode !== "select" || e.button !== 0) return;
                e.stopPropagation();
                onSelectNode(idx);
                const pt = clientToImg(e.clientX, e.clientY);
                if (!pt) return;
                setDragAction({
                  kind: "move",
                  idx,
                  startX: pt[0],
                  startY: pt[1],
                  origBox: [bx, by, bw, bh],
                });
              }}
              onMouseEnter={() => setHoveredNode(n)}
              onMouseLeave={() => setHoveredNode((prev) => (prev?.id === n.id ? null : prev))}
              style={{ cursor: toolMode === "select" ? "move" : "crosshair" }}
            >
              {Array.isArray(n.bubble_polygon) && n.bubble_polygon.length >= 3 && (
                <polygon
                  points={n.bubble_polygon.map((p) => `${p[0]},${p[1]}`).join(" ")}
                  fill={isSel ? "rgba(225, 29, 72, 0.08)" : "rgba(2, 132, 199, 0.05)"}
                  stroke={strokeCol}
                  strokeWidth={isSel ? 2.5 : 1.2}
                  strokeDasharray="5 3"
                />
              )}
              <rect
                x={bx}
                y={by}
                width={bw}
                height={bh}
                fill={isSel ? "rgba(225, 29, 72, 0.14)" : "rgba(2, 132, 199, 0.03)"}
                stroke={strokeCol}
                strokeWidth={isSel ? 3 : 1.6}
              />
              <rect x={bx} y={Math.max(0, by - 20)} width={pillW} height={20} fill={strokeCol} />
              <text
                x={bx + 6}
                y={Math.max(14, by - 6)}
                fill="#FFFFFF"
                fontSize="13"
                fontWeight="800"
                fontFamily="JetBrains Mono, monospace"
              >
                {(n.reading_order ?? idx) + 1}
              </text>
              {isSel &&
                toolMode === "select" &&
                corners.map((c) => (
                  <rect
                    key={c.key}
                    x={c.cx - handleSize / 2}
                    y={c.cy - handleSize / 2}
                    width={handleSize}
                    height={handleSize}
                    fill="#FFFFFF"
                    stroke="#E11D48"
                    strokeWidth={2.2}
                    style={{ cursor: c.cursor }}
                    onMouseDown={(e) => {
                      e.stopPropagation();
                      const pt = clientToImg(e.clientX, e.clientY);
                      if (!pt) return;
                      setDragAction({
                        kind: "resize",
                        idx,
                        corner: c.key,
                        startX: pt[0],
                        startY: pt[1],
                        origBox: [bx, by, bw, bh],
                      });
                    }}
                  />
                ))}
            </g>
          );
        })}

        {dragAction?.kind === "draw" && (
          <rect
            x={Math.min(dragAction.startX, dragAction.curX)}
            y={Math.min(dragAction.startY, dragAction.curY)}
            width={Math.abs(dragAction.curX - dragAction.startX)}
            height={Math.abs(dragAction.curY - dragAction.startY)}
            fill="rgba(225, 29, 72, 0.16)"
            stroke="#E11D48"
            strokeWidth={3}
            strokeDasharray="6 4"
          />
        )}
      </svg>
    );
  };

  return (
    <section className="canvas-column">
      <div className="canvas-toolbar">
        <div className="segmented-control" role="tablist" aria-label="画布视图模式">
          <button
            type="button"
            className={`seg-btn ${viewMode === "composite" ? "active" : ""}`}
            onClick={() => onChangeViewMode("composite")}
            title="译制成品（快捷键 1，按住 Space 瞬看原图）"
          >
            成品
          </button>
          <button
            type="button"
            className={`seg-btn ${viewMode === "split" ? "active" : ""}`}
            onClick={() => onChangeViewMode("split")}
            title="左右拉帘对照（快捷键 2）"
          >
            拉帘
          </button>
          <button
            type="button"
            className={`seg-btn ${viewMode === "side-by-side" ? "active" : ""}`}
            onClick={() => onChangeViewMode("side-by-side")}
            title="左右并排对照（快捷键 3）"
          >
            并排
          </button>
          <button
            type="button"
            className={`seg-btn ${viewMode === "raw" ? "active" : ""}`}
            onClick={() => onChangeViewMode("raw")}
            title="原始生肉图（快捷键 4）"
          >
            原图
          </button>
        </div>

        <div className="segmented-control" role="group" aria-label="画布交互工具">
          <button
            type="button"
            className={`seg-btn ${toolMode === "select" ? "active-cyan" : ""}`}
            onClick={() => onChangeToolMode("select")}
            title="选择 / 拖动移动 / 拖拽四角缩放文本框 (V)"
          >
            <IconCursor size={13} /> 选框
          </button>
          <button
            type="button"
            className={`seg-btn ${toolMode === "draw" ? "active" : ""}`}
            onClick={() => onChangeToolMode("draw")}
            title="在漏检文字处拖拽框选新增文本节点 (A)"
          >
            <IconBoxSelect size={13} /> 框选
          </button>
          <button
            type="button"
            className={`seg-btn ${toolMode === "pan" ? "active-cyan" : ""}`}
            onClick={() => onChangeToolMode("pan")}
            title="抓手拖动画布平移 (H)"
          >
            <IconHand size={13} /> 平移
          </button>
        </div>

        <div className="toolbar-group">
          <button
            type="button"
            className={`btn btn-sm ${showOverlay ? "cyan" : ""}`}
            onClick={onToggleOverlay}
            title="显示或隐藏文本检测框线 (O)"
          >
            <IconLayers size={13} /> 框线:{showOverlay ? "开" : "关"}
          </button>

          <div className="zoom-controls">
            <button
              type="button"
              className={`btn btn-sm ${zoomMode === "fit-h" ? "ink-active" : ""}`}
              onClick={() => setZoomMode("fit-h")}
              title="自适应窗口高度"
            >
              适高
            </button>
            <button
              type="button"
              className={`btn btn-sm ${zoomMode === "fit-w" ? "ink-active" : ""}`}
              onClick={() => setZoomMode("fit-w")}
              title="自适应宽度（条漫推荐）"
            >
              适宽
            </button>
            <button
              type="button"
              className="btn btn-sm"
              onClick={() => applyZoomDelta(-20)}
              title="缩小"
            >
              <IconMinus size={13} />
            </button>
            <button
              type="button"
              className={`btn btn-sm mono ${zoomMode === "custom" ? "ink-active" : ""}`}
              onClick={() => {
                setZoomMode("custom");
                setZoomPct(100);
              }}
              title="100% 像素原大"
            >
              {zoomMode === "custom" ? `${zoomPct}%` : "1:1"}
            </button>
            <button
              type="button"
              className="btn btn-sm"
              onClick={() => applyZoomDelta(20)}
              title="放大"
            >
              <IconPlus size={13} />
            </button>
          </div>
        </div>

        <div className="toolbar-group">
          {pageCounterLabel && (
            <div className="pager-pill">
              <button
                type="button"
                className="btn btn-sm"
                disabled={!canPrevPage}
                onClick={onPrevPage}
                title="上一页 (←)"
              >
                <IconChevronLeft size={13} />
              </button>
              <span className="mono pager-text">{pageCounterLabel}</span>
              <button
                type="button"
                className="btn btn-sm"
                disabled={!canNextPage}
                onClick={onNextPage}
                title="下一页 (→)"
              >
                <IconChevronRight size={13} />
              </button>
            </div>
          )}

          <button
            type="button"
            className={`btn btn-sm ${zenMode ? "primary" : ""}`}
            onClick={onToggleZen}
            title="沉浸阅读模式 (F)"
          >
            <IconMaximize size={13} /> {zenMode ? "退出沉浸" : "沉浸"}
          </button>
        </div>
      </div>

      <div
        ref={viewportRef}
        className={`canvas-viewport ${draggingSplit ? "is-dragging" : ""} ${
          toolMode === "pan" ? (panState ? "is-panning" : "can-pan") : ""
        }`}
        onWheel={handleWheel}
        onMouseDown={(e) => {
          if (toolMode === "pan" || e.button === 1) {
            const vp = viewportRef.current;
            if (!vp) return;
            e.preventDefault();
            setPanState({
              startX: e.clientX,
              startY: e.clientY,
              scrollLeft: vp.scrollLeft,
              scrollTop: vp.scrollTop,
            });
          }
        }}
      >
        {peekingRaw && (
          <div className="peek-banner mono">正在瞬看原图 · 松开 Space 返回译制成品</div>
        )}

        {toolMode === "draw" && !peekingRaw && (
          <div className="tool-hint-banner mono">
            <IconBoxSelect size={14} /> 框选模式：直接在图上漏检文字区域拖拽画框，松开即添加新节点（Esc 退出）
          </div>
        )}

        {hoveredNode && interactiveOverlayVisible && !draggingSplit && !dragAction && (
          <div className="hover-bubble-tip">
            <span className="mono tip-id">#{(hoveredNode.reading_order ?? 0) + 1}</span>
            <span className="tip-text">{hoveredNode.target_text || hoveredNode.source_text || "（待填译文）"}</span>
          </div>
        )}

        {pageId > 0 && pageDOM ? (
          effectiveMode === "split" ? (
            <div
              ref={splitBoxRef}
              className="stage-frame split-stage"
              onMouseDown={(e) => {
                setDraggingSplit(true);
                updateSplitFromClientX(e.clientX);
              }}
              onTouchStart={(e) => {
                setDraggingSplit(true);
                if (e.touches[0]) updateSplitFromClientX(e.touches[0].clientX);
              }}
            >
              <img src={compUrl} alt="Translated Composite" style={imgStyle} draggable={false} />
              <div
                className="split-raw-layer"
                style={{ clipPath: `inset(0 ${100 - splitPos}% 0 0)` }}
              >
                <img src={rawUrl} alt="Original Raw" style={imgStyle} draggable={false} />
              </div>
              <div className="split-divider" style={{ left: `${splitPos}%` }}>
                <div className="split-handle mono">◀ ▶</div>
              </div>
              <span className="split-tag left mono">原图 RAW</span>
              <span className="split-tag right mono">译制成品</span>
            </div>
          ) : effectiveMode === "side-by-side" ? (
            <div className="side-by-side-wrap">
              <div className="side-pane">
                <div className="side-pane-label mono">原始生肉 RAW</div>
                <div className="stage-frame">
                  <img src={rawUrl} alt="Original Raw" style={imgStyle} draggable={false} />
                </div>
              </div>
              <div className="side-pane">
                <div className="side-pane-label mono accent">译制成品 Z0+Z1+Z2</div>
                <div className="stage-frame">
                  <img src={compUrl} alt="Translated Composite" style={imgStyle} draggable={false} />
                  {renderOverlay()}
                </div>
              </div>
            </div>
          ) : (
            <div className="stage-frame">
              <img
                src={effectiveMode === "raw" ? rawUrl : compUrl}
                alt={effectiveMode === "raw" ? "Original Raw" : "Translated Composite"}
                style={imgStyle}
                draggable={false}
              />
              {renderOverlay()}
            </div>
          )
        ) : (
          <div className="canvas-empty-hero">
            <div className="hero-card">
              <h2>当前作品暂无页面</h2>
              <p>拖拽任意漫画图片或 .cbz/.zip 到窗口中，即可全自动去字、视觉翻译与无缝嵌字。</p>
              <button type="button" className="btn primary" onClick={onOpenUpload}>
                + 导入漫画页面
              </button>
            </div>
          </div>
        )}
      </div>

      {pageDOM && (
        <div className="canvas-statusbar mono">
          <span>
            {pageDOM.width}×{pageDOM.height} · {pageDOM.format_profile}
          </span>
          <span>
            {(pageDOM.nodes || []).length} 框 · 原画零损{" "}
            <strong>{((pageDOM.metrics?.bit_exact_unmasked_ratio ?? 1) * 100).toFixed(2)}%</strong> (L∞=
            {pageDOM.metrics?.unmasked_linf ?? 0})
          </span>
          <span>
            分析 {((pageDOM.metrics?.analyze_ms ?? 0) / 1000).toFixed(2)}s · 渲染{" "}
            {Math.round(pageDOM.metrics?.render_ms ?? 0)}ms · A 框选 / Space 瞬看原图
          </span>
        </div>
      )}
    </section>
  );
}
