import { useState } from "react";
import type {
  CharacterPersona,
  CharacterRelation,
  GlossaryTerm,
  PageDOM,
  TextNode,
} from "../types";
import {
  IconAlert,
  IconBoxSelect,
  IconCheck,
  IconRefresh,
  IconSparkles,
  IconTrash,
} from "./Icons";

export type InspectorTab = "node" | "script" | "lore";

interface InspectorPanelProps {
  pageDOM: PageDOM | null;
  selectedNodeIdx: number;
  onSelectNode: (idx: number) => void;
  onUpdateNodeAt: (idx: number, mutator: (n: TextNode) => TextNode) => void;
  onDeleteNode: (idx: number) => void;
  onTranslateSingleNode: (idx: number) => Promise<void>;
  onStartDrawBox: () => void;
  translatingNode: boolean;
  isDirty: boolean;
  onRevertChanges: () => void;
  busy: boolean;
  onRerender: (markReviewed: boolean) => void;
  onRetranslatePage: () => void;
  selectedWorkId: number;
  glossary: GlossaryTerm[];
  personas: CharacterPersona[];
  relations: CharacterRelation[];
  onAddGlossary: (src: string, tgt: string, ruby: string) => Promise<boolean>;
  onSavePersona: (persona: CharacterPersona) => Promise<void>;
  onChangePersonaDraft: (idx: number, next: CharacterPersona) => void;
  onSaveRelation: (fromId: number, toId: number, honorific: string, reg: string) => Promise<boolean>;
  onCopyText: (text: string) => void;
}

function rgbaToHex(rgba?: number[]): string {
  if (!rgba || rgba.length < 3) return "#09090b";
  const toHex = (n: number) => Math.max(0, Math.min(255, Math.round(n))).toString(16).padStart(2, "0");
  return `#${toHex(rgba[0])}${toHex(rgba[1])}${toHex(rgba[2])}`;
}

function hexToRgba(hex: string, alpha = 255): number[] {
  const clean = hex.replace("#", "");
  if (clean.length !== 6) return [12, 12, 14, alpha];
  return [
    parseInt(clean.slice(0, 2), 16),
    parseInt(clean.slice(2, 4), 16),
    parseInt(clean.slice(4, 6), 16),
    alpha,
  ];
}

export default function InspectorPanel({
  pageDOM,
  selectedNodeIdx,
  onSelectNode,
  onUpdateNodeAt,
  onDeleteNode,
  onTranslateSingleNode,
  onStartDrawBox,
  translatingNode,
  isDirty,
  onRevertChanges,
  busy,
  onRerender,
  onRetranslatePage,
  selectedWorkId,
  glossary,
  personas,
  relations,
  onAddGlossary,
  onSavePersona,
  onChangePersonaDraft,
  onSaveRelation,
  onCopyText,
}: InspectorPanelProps) {
  const [tab, setTab] = useState<InspectorTab>("node");
  const [srcTerm, setSrcTerm] = useState("");
  const [tgtTerm, setTgtTerm] = useState("");
  const [rubyHint, setRubyHint] = useState("");
  const [relFromId, setRelFromId] = useState<number>(0);
  const [relToId, setRelToId] = useState<number>(0);
  const [relHonorific, setRelHonorific] = useState("");
  const [relRegister, setRelRegister] = useState("");

  const nodes = Array.isArray(pageDOM?.nodes) ? pageDOM!.nodes : [];
  const activeNode = nodes[selectedNodeIdx] || null;
  const charCount = activeNode ? Array.from((activeNode.target_text ?? "").trim()).length : 0;
  const overBudget =
    activeNode && (activeNode.char_budget_max ?? 0) > 0 && charCount > activeNode.char_budget_max;

  const effectiveFrom = relFromId || personas[0]?.id || 0;
  const effectiveTo = relToId || personas[1]?.id || 0;

  return (
    <aside className="studio-sidebar right-sidebar">
      <div className="inspector-tabs">
        <button
          type="button"
          className={`tab-btn ${tab === "node" ? "active" : ""}`}
          onClick={() => setTab("node")}
        >
          节点精修
          {isDirty && <span className="dirty-dot" title="有未渲染改动" />}
        </button>
        <button
          type="button"
          className={`tab-btn ${tab === "script" ? "active" : ""}`}
          onClick={() => setTab("script")}
        >
          全页台词 ({nodes.length})
        </button>
        <button
          type="button"
          className={`tab-btn ${tab === "lore" ? "active" : ""}`}
          onClick={() => setTab("lore")}
        >
          人设术语
        </button>
      </div>

      <div className="inspector-scroll">
        {tab === "node" && (
          <div className="inspector-body">
            {activeNode ? (
              <>
                <div className="node-stepper">
                  <button
                    type="button"
                    className="btn btn-sm"
                    disabled={selectedNodeIdx <= 0}
                    onClick={() => onSelectNode(selectedNodeIdx - 1)}
                    title="上一个文本块 ([)"
                  >
                    ‹ 上一框
                  </button>
                  <span className="mono stepper-label">
                    #{activeNode.reading_order + 1} / {nodes.length} · {activeNode.id}
                  </span>
                  <button
                    type="button"
                    className="btn btn-sm"
                    disabled={selectedNodeIdx >= nodes.length - 1}
                    onClick={() => onSelectNode(selectedNodeIdx + 1)}
                    title="下一个文本块 (])"
                  >
                    下一框 ›
                  </button>
                </div>

                <div className="node-quick-actions">
                  <button
                    type="button"
                    className="btn btn-sm"
                    onClick={onStartDrawBox}
                    title="在漏检文字处拖拽画框新增节点 (快捷键 A)"
                  >
                    <IconBoxSelect size={13} /> 框选新增
                  </button>
                  <button
                    type="button"
                    className="btn btn-sm cyan"
                    disabled={busy || translatingNode}
                    onClick={() => void onTranslateSingleNode(selectedNodeIdx)}
                    title="用视觉大模型单独识别并翻译当前框（适合手动新增框）"
                  >
                    <IconSparkles size={13} /> {translatingNode ? "识图中..." : "AI 识图翻译"}
                  </button>
                  <button
                    type="button"
                    className="btn btn-sm danger-btn"
                    disabled={busy}
                    onClick={() => onDeleteNode(selectedNodeIdx)}
                    title="删除当前文本框（点击实时重排后将完全还原底图原画）"
                  >
                    <IconTrash size={13} /> 删框
                  </button>
                </div>

                <div className="source-quote-box">
                  <div className="quote-head">
                    <span>原文识别</span>
                    <button
                      type="button"
                      className="link-btn"
                      onClick={() => onCopyText(activeNode.source_text)}
                    >
                      复制原文
                    </button>
                  </div>
                  <div className="quote-text">{activeNode.source_text || "（视觉直译，无独立原文）"}</div>
                </div>

                <div className="field">
                  <div className="sidebar-label-row">
                    <label>本地化译文 (Ctrl+Enter)</label>
                    <span className={`mono budget-pill ${overBudget ? "over" : ""}`}>
                      {charCount} 字
                      {activeNode.char_budget_max > 0
                        ? ` / 建议 ${activeNode.char_budget_min}-${activeNode.char_budget_max}`
                        : ""}
                    </span>
                  </div>
                  <textarea
                    rows={4}
                    className="target-editor"
                    value={activeNode.target_text}
                    onKeyDown={(e) => {
                      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
                        e.preventDefault();
                        onRerender(false);
                      }
                    }}
                    onChange={(e) =>
                      onUpdateNodeAt(selectedNodeIdx, (n) => ({
                        ...n,
                        target_text: e.target.value,
                        target_segments: [],
                      }))
                    }
                  />
                </div>

                <div className="style-card">
                  <div className="style-card-title">字体与排版样式</div>
                  <div className="settings-grid-2">
                    <div className="field">
                      <label>书写方向</label>
                      <select
                        value={activeNode.style.orientation}
                        onChange={(e) =>
                          onUpdateNodeAt(selectedNodeIdx, (n) => ({
                            ...n,
                            style: {
                              ...n.style,
                              orientation: e.target.value as "vertical" | "horizontal",
                            },
                          }))
                        }
                      >
                        <option value="vertical">竖排 (JIS 标点)</option>
                        <option value="horizontal">横排</option>
                      </select>
                    </div>
                    <div className="field">
                      <label>字体情绪</label>
                      <select
                        value={activeNode.style.font_category}
                        onChange={(e) =>
                          onUpdateNodeAt(selectedNodeIdx, (n) => ({
                            ...n,
                            style: {
                              ...n.style,
                              font_category: e.target.value as TextNode["style"]["font_category"],
                            },
                          }))
                        }
                      >
                        <option value="gothic">黑体 · 标准台词</option>
                        <option value="mincho">明朝 · 独白/严肃</option>
                        <option value="rounded">圆体 · 柔和/心声</option>
                        <option value="brush">毛笔 · 怒吼/特效</option>
                        <option value="comic_sans">美漫手写体</option>
                      </select>
                    </div>
                  </div>

                  <div className="color-row">
                    <label className="color-chip">
                      <span>字色</span>
                      <input
                        type="color"
                        value={rgbaToHex(activeNode.style.fg_color_rgba)}
                        onChange={(e) =>
                          onUpdateNodeAt(selectedNodeIdx, (n) => ({
                            ...n,
                            style: {
                              ...n.style,
                              fg_color_rgba: hexToRgba(e.target.value, n.style.fg_color_rgba?.[3] ?? 255),
                            },
                          }))
                        }
                      />
                      <span className="mono">{rgbaToHex(activeNode.style.fg_color_rgba)}</span>
                    </label>
                    <label className="color-chip">
                      <span>描边</span>
                      <input
                        type="color"
                        value={rgbaToHex(activeNode.style.outline_color_rgba)}
                        onChange={(e) =>
                          onUpdateNodeAt(selectedNodeIdx, (n) => ({
                            ...n,
                            style: {
                              ...n.style,
                              outline_color_rgba: hexToRgba(
                                e.target.value,
                                n.style.outline_color_rgba?.[3] ?? 255
                              ),
                            },
                          }))
                        }
                      />
                      <span className="mono">{rgbaToHex(activeNode.style.outline_color_rgba)}</span>
                    </label>
                  </div>

                  <div className="settings-grid-2">
                    <div className="field">
                      <label>基准字号 (px)</label>
                      <input
                        type="number"
                        step="1"
                        min="8"
                        max="120"
                        value={Math.round(
                          activeNode.style?.source_font_size_px || activeNode.layout?.font_size || 20
                        )}
                        onChange={(e) =>
                          onUpdateNodeAt(selectedNodeIdx, (n) => ({
                            ...n,
                            style: {
                              ...n.style,
                              source_font_size_px: Math.max(8, Number(e.target.value) || 20),
                            },
                          }))
                        }
                      />
                    </div>
                    <div className="field">
                      <label>描边粗细 (px)</label>
                      <input
                        type="number"
                        step="0.5"
                        min="0"
                        max="8"
                        value={Number((activeNode.style?.outline_width_px ?? 0).toFixed(1))}
                        onChange={(e) =>
                          onUpdateNodeAt(selectedNodeIdx, (n) => ({
                            ...n,
                            style: {
                              ...n.style,
                              outline_width_px: Math.max(0, Number(e.target.value) || 0),
                            },
                          }))
                        }
                      />
                    </div>
                  </div>

                  <div className="field">
                    <label>光学墨量守恒模式</label>
                    <select
                      value={activeNode.style.hollow_mode}
                      onChange={(e) =>
                        onUpdateNodeAt(selectedNodeIdx, (n) => ({
                          ...n,
                          style: {
                            ...n.style,
                            hollow_mode: e.target.value as TextNode["style"]["hollow_mode"],
                          },
                        }))
                      }
                    >
                      <option value="solid">实心填充 (Solid)</option>
                      <option value="double_outline_hollow">镂空透底 (Brentq 守恒)</option>
                      <option value="halftone_core">网点镂空芯</option>
                      <option value="stroke_thinned">SDF 瘦笔重</option>
                    </select>
                  </div>
                </div>

                <details className="subtle-details">
                  <summary>物理算子与质量遥测</summary>
                  <div className="kv-grid mono" style={{ marginTop: 8 }}>
                    <div>修复路径: {activeNode.restoration_path || "—"}</div>
                    <div>融合模式: {activeNode.fusion_mode || "—"}</div>
                    <div>气泡类型: {activeNode.bubble_kind || "—"}</div>
                    <div>说话人簇: {activeNode.speaker_cluster_id || "—"}</div>
                    <div>原墨密度: {((activeNode.style?.ink_density_orig ?? 0) * 100).toFixed(1)}%</div>
                    <div>现墨密度: {((activeNode.layout?.ink_density_rendered ?? 0) * 100).toFixed(1)}%</div>
                    <div>实排字号: {(activeNode.layout?.font_size ?? 0).toFixed(1)}px</div>
                    <div>不确定度: {((activeNode.uncertainty?.score ?? 0) * 100).toFixed(0)}%</div>
                  </div>
                  {Array.isArray(activeNode.uncertainty?.reasons) && activeNode.uncertainty.reasons.length > 0 && (
                    <div className="reason-tags mono">
                      {activeNode.uncertainty.reasons.map((r) => (
                        <span key={r} className="reason-tag">
                          {r}
                        </span>
                      ))}
                    </div>
                  )}
                </details>
              </>
            ) : (
              <div className="empty-inspector">
                <div style={{ marginBottom: 10 }}>
                  当前页面未选中节点，可直接点选画布文本框或手动框选漏检区域
                </div>
                <button type="button" className="btn btn-sm primary" onClick={onStartDrawBox}>
                  + 在画布上框选新增文本框
                </button>
              </div>
            )}
          </div>
        )}

        {tab === "script" && (
          <div className="script-list">
            {nodes.map((n, idx) => {
              const isSel = idx === selectedNodeIdx;
              return (
                <div
                  key={n.id}
                  className={`script-card ${isSel ? "active" : ""}`}
                  onClick={() => onSelectNode(idx)}
                >
                  <div className="script-card-head mono">
                    <span>
                      #{n.reading_order + 1} · {n.bubble_kind}
                    </span>
                    <span className={n.uncertainty?.needs_review ? "txt-warn" : "txt-ok"}>
                      {n.uncertainty?.needs_review ? (
                        <>
                          <IconAlert size={12} /> 待审
                        </>
                      ) : (
                        <IconCheck size={12} />
                      )}
                    </span>
                  </div>
                  {n.source_text && <div className="script-src">{n.source_text}</div>}
                  <textarea
                    rows={2}
                    value={n.target_text}
                    onFocus={() => onSelectNode(idx)}
                    onKeyDown={(e) => {
                      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
                        e.preventDefault();
                        onRerender(false);
                      }
                    }}
                    onChange={(e) =>
                      onUpdateNodeAt(idx, (node) => ({
                        ...node,
                        target_text: e.target.value,
                        target_segments: [],
                      }))
                    }
                  />
                </div>
              );
            })}
            {nodes.length === 0 && <div className="empty-inspector">当前页面暂无可编辑文本节点</div>}
          </div>
        )}

        {tab === "lore" && (
          <div className="inspector-body">
            <div className="style-card">
              <div className="style-card-title">专有名词与義訓ルビ术语库</div>
              <form
                style={{ display: "flex", flexDirection: "column", gap: 8 }}
                onSubmit={async (e) => {
                  e.preventDefault();
                  if (!srcTerm.trim() || !tgtTerm.trim()) return;
                  const ok = await onAddGlossary(srcTerm.trim(), tgtTerm.trim(), rubyHint.trim());
                  if (ok) {
                    setSrcTerm("");
                    setTgtTerm("");
                    setRubyHint("");
                  }
                }}
              >
                <div className="settings-grid-2">
                  <input
                    type="text"
                    placeholder="原文 (如: 宿敵)"
                    value={srcTerm}
                    onChange={(e) => setSrcTerm(e.target.value)}
                  />
                  <input
                    type="text"
                    placeholder="译文 (如: 宿敌)"
                    value={tgtTerm}
                    onChange={(e) => setTgtTerm(e.target.value)}
                  />
                </div>
                <div className="btn-row">
                  <input
                    type="text"
                    placeholder="可选: 伴生注音 ルビ"
                    value={rubyHint}
                    onChange={(e) => setRubyHint(e.target.value)}
                    style={{ flex: 1 }}
                  />
                  <button className="btn btn-sm" type="submit" disabled={selectedWorkId <= 0}>
                    写入词库
                  </button>
                </div>
              </form>
              <div className="lore-list">
                {glossary.map((g) => (
                  <div key={g.source_term} className="lore-item mono">
                    <span>{g.source_term}</span>
                    <span>→</span>
                    <strong>{g.target_term}</strong>
                    {g.ruby_override && <span className="badge">[{g.ruby_override}]</span>}
                  </div>
                ))}
                {glossary.length === 0 && (
                  <div className="lore-empty">暂无专有术语，录入后自动注入后续章节翻译</div>
                )}
              </div>
            </div>

            <div className="style-card">
              <div className="style-card-title">跨章节角色声纹设定 ({personas.length})</div>
              {personas.map((ch, cIdx) => (
                <div key={ch.cluster_key} className="persona-row">
                  <div className="btn-row" style={{ alignItems: "center" }}>
                    <span className="mono badge">{ch.cluster_key}</span>
                    <input
                      type="text"
                      placeholder="角色译名"
                      value={ch.name_target}
                      onChange={(e) => onChangePersonaDraft(cIdx, { ...ch, name_target: e.target.value })}
                      style={{ flex: 1 }}
                    />
                    <input
                      type="text"
                      placeholder="自称"
                      value={ch.first_person}
                      onChange={(e) => onChangePersonaDraft(cIdx, { ...ch, first_person: e.target.value })}
                      style={{ width: 76 }}
                    />
                  </div>
                  <div className="btn-row">
                    <input
                      type="text"
                      placeholder="口吻/语癖（如：傲娇/敬语）"
                      value={ch.speech_style}
                      onChange={(e) => onChangePersonaDraft(cIdx, { ...ch, speech_style: e.target.value })}
                      style={{ flex: 1 }}
                    />
                    <button
                      type="button"
                      className="btn btn-sm"
                      onClick={() => void onSavePersona(ch)}
                    >
                      保存
                    </button>
                  </div>
                </div>
              ))}
              {personas.length === 0 && (
                <div className="lore-empty">翻译含对话气泡的作品后自动聚类角色</div>
              )}
            </div>

            {personas.length >= 2 && (
              <div className="style-card">
                <div className="style-card-title">双向称谓与敬语关系</div>
                <form
                  style={{ display: "flex", flexDirection: "column", gap: 8 }}
                  onSubmit={async (e) => {
                    e.preventDefault();
                    const ok = await onSaveRelation(
                      effectiveFrom,
                      effectiveTo,
                      relHonorific.trim(),
                      relRegister.trim()
                    );
                    if (ok) {
                      setRelHonorific("");
                      setRelRegister("");
                    }
                  }}
                >
                  <div className="btn-row" style={{ alignItems: "center" }}>
                    <select
                      value={effectiveFrom}
                      onChange={(e) => setRelFromId(Number(e.target.value))}
                      style={{ flex: 1 }}
                    >
                      {personas.map((p) => (
                        <option key={`f-${p.cluster_key}`} value={p.id || 0}>
                          {p.name_target || p.cluster_key}
                        </option>
                      ))}
                    </select>
                    <span className="mono">→</span>
                    <select
                      value={effectiveTo}
                      onChange={(e) => setRelToId(Number(e.target.value))}
                      style={{ flex: 1 }}
                    >
                      {personas.map((p) => (
                        <option key={`t-${p.cluster_key}`} value={p.id || 0}>
                          {p.name_target || p.cluster_key}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div className="btn-row">
                    <input
                      type="text"
                      placeholder="称呼（如：前辈）"
                      value={relHonorific}
                      onChange={(e) => setRelHonorific(e.target.value)}
                      style={{ flex: 1 }}
                    />
                    <input
                      type="text"
                      placeholder="语域备注"
                      value={relRegister}
                      onChange={(e) => setRelRegister(e.target.value)}
                      style={{ flex: 1 }}
                    />
                    <button className="btn btn-sm" type="submit">
                      绑定
                    </button>
                  </div>
                </form>
                <div className="lore-list">
                  {relations.map((r, idx) => {
                    const fn =
                      personas.find((p) => p.id === r.from_char_id)?.name_target || `#${r.from_char_id}`;
                    const tn =
                      personas.find((p) => p.id === r.to_char_id)?.name_target || `#${r.to_char_id}`;
                    return (
                      <div key={idx} className="lore-item mono">
                        {fn} → {tn}: <strong>{r.honorific_target || "直呼"}</strong> {r.register_note}
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {pageDOM && (
        <div className="inspector-footer">
          {isDirty && (
            <div className="dirty-banner">
              <span className="mono">● 有未渲染改动</span>
              <button type="button" className="link-btn" disabled={busy} onClick={onRevertChanges}>
                还原改动
              </button>
            </div>
          )}
          <div className="inspector-footer-grid">
            <button
              type="button"
              className="btn cyan"
              disabled={busy}
              onClick={() => onRerender(false)}
              title="仅重排文字层 (Z1)，零重算底图，毫秒级生效 (Ctrl+Enter)"
            >
              {busy ? "渲染中..." : "实时重排"}
            </button>
            <button
              type="button"
              className="btn primary"
              disabled={busy}
              onClick={() => onRerender(true)}
            >
              审校通过
            </button>
            <button
              type="button"
              className="btn"
              disabled={busy}
              onClick={onRetranslatePage}
              title="保留物理擦除底图，用大模型重新翻译本页"
            >
              <IconRefresh size={13} /> 重译
            </button>
          </div>
        </div>
      )}
    </aside>
  );
}
