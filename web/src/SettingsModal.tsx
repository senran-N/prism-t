import React, { useCallback, useEffect, useState } from "react";
import type {
  HealthResponse,
  ModelPreset,
  RuntimeSettings,
  RuntimeSettingsResponse,
  RuntimeSettingsUpdate,
  TestLLMResponse,
} from "./types";
import { IconClose, IconFileText, IconSearch, IconSettings, IconZap } from "./components/Icons";

interface SettingsModalProps {
  open: boolean;
  onClose: () => void;
  token: string;
  onChangeToken: (next: string) => void;
  authAuthorized: boolean;
  health: HealthResponse | null;
  hasActiveJob: boolean;
  onSaved: (resp: RuntimeSettingsResponse) => void;
}

const DEFAULT_FORM: RuntimeSettings = {
  device: "auto",
  max_concurrency: 1,
  target_lang: "zh-CN",
  llm_base_url: "",
  llm_model: "",
  llm_reasoning_effort: "none",
  llm_timeout_seconds: 120,
  llm_vision: true,
  llm_source_prepass: false,
  llm_vision_audit: false,
  llm_batch_size: 16,
};

const STANDARD_PROVIDERS = [
  { label: "OpenAI", base_url: "https://api.openai.com/v1", model: "gpt-4.1-mini", vision: true },
  { label: "DeepSeek", base_url: "https://api.deepseek.com/v1", model: "deepseek-chat", vision: false },
  { label: "Gemini 兼容", base_url: "https://generativelanguage.googleapis.com/v1beta/openai", model: "gemini-2.5-flash", vision: true },
  { label: "OpenRouter", base_url: "https://openrouter.ai/api/v1", model: "google/gemini-2.5-flash", vision: true },
  { label: "本地 Ollama / vLLM", base_url: "http://127.0.0.1:11434/v1", model: "qwen2.5vl:7b", vision: true },
];

async function parseJsonSafe(res: Response): Promise<Record<string, any>> {
  const txt = await res.text();
  if (!txt) return {};
  try {
    return JSON.parse(txt);
  } catch {
    return { error: txt.slice(0, 240) };
  }
}

export default function SettingsModal({
  open,
  onClose,
  token,
  onChangeToken,
  authAuthorized,
  health,
  hasActiveJob,
  onSaved,
}: SettingsModalProps) {
  const [form, setForm] = useState<RuntimeSettings>(DEFAULT_FORM);
  const [apiKeyInput, setApiKeyInput] = useState<string>("");
  const [clearApiKey, setClearApiKey] = useState<boolean>(false);
  const [presetProvider, setPresetProvider] = useState<string>("");
  const [presetModel, setPresetModel] = useState<string>("");

  const [maskedKey, setMaskedKey] = useState<string>("");
  const [apiKeySet, setApiKeySet] = useState<boolean>(false);
  const [engineManaged, setEngineManaged] = useState<boolean>(false);
  const [settingsFile, setSettingsFile] = useState<string>("data/settings.json");
  const [presets, setPresets] = useState<ModelPreset[]>([]);
  const [presetFilter, setPresetFilter] = useState<string>("");
  const [visionPresetsOnly, setVisionPresetsOnly] = useState<boolean>(true);

  const [loading, setLoading] = useState<boolean>(false);
  const [saving, setSaving] = useState<boolean>(false);
  const [testing, setTesting] = useState<boolean>(false);
  const [unauthorized, setUnauthorized] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string>("");
  const [warningMsg, setWarningMsg] = useState<string>("");
  const [testResult, setTestResult] = useState<TestLLMResponse | null>(null);

  const loadSettings = useCallback(async (overrideToken?: string) => {
    const activeToken = overrideToken !== undefined ? overrideToken : token;
    setLoading(true);
    setErrorMsg("");
    setWarningMsg("");
    setTestResult(null);
    try {
      const headers: HeadersInit = activeToken ? { "X-Prism-Token": activeToken } : {};
      const res = await fetch("/api/v1/settings", { headers });
      if (res.status === 401) {
        setUnauthorized(true);
        return;
      }
      const data = await parseJsonSafe(res);
      if (!res.ok) {
        setErrorMsg(data.error || `加载配置失败: HTTP ${res.status}`);
        return;
      }
      const typed = data as RuntimeSettingsResponse;
      setUnauthorized(false);
      setForm({ ...DEFAULT_FORM, ...typed.settings });
      setApiKeyInput("");
      setClearApiKey(false);
      setPresetProvider("");
      setPresetModel("");
      setMaskedKey(typed.llm_api_key_masked || "");
      setApiKeySet(Boolean(typed.llm_api_key_set));
      setEngineManaged(Boolean(typed.engine_managed));
      setSettingsFile(typed.settings_file || "data/settings.json");
      setPresets(typed.local_presets || []);
      if (typed.warning) {
        setWarningMsg(typed.warning);
      }
    } catch (err) {
      setErrorMsg(String(err));
    } finally {
      setLoading(false);
    }
  }, [token]);

  useEffect(() => {
    if (open) {
      void loadSettings();
    }
  }, [open, loadSettings]);

  if (!open) return null;

  const buildPayload = (): RuntimeSettingsUpdate => {
    const payload: RuntimeSettingsUpdate = {
      ...form,
      max_concurrency: Math.max(1, Math.min(8, Number(form.max_concurrency) || 1)),
      llm_batch_size: Math.max(1, Math.min(64, Number(form.llm_batch_size) || 16)),
      llm_timeout_seconds: Math.max(10, Math.min(600, Number(form.llm_timeout_seconds) || 120)),
    };
    if (apiKeyInput.trim() !== "") {
      payload.llm_api_key = apiKeyInput.trim();
    }
    if (clearApiKey) {
      payload.clear_api_key = true;
    }
    if (presetProvider && presetModel) {
      payload.preset_provider = presetProvider;
      payload.preset_model = presetModel;
    }
    return payload;
  };

  const handleSelectPreset = (p: ModelPreset) => {
    setForm((prev) => ({
      ...prev,
      llm_base_url: p.base_url,
      llm_model: p.model_id,
      llm_vision: p.supports_vision,
    }));
    setPresetProvider(p.provider);
    setPresetModel(p.model_id);
    setApiKeyInput("");
    setClearApiKey(false);
    setTestResult(null);
    setErrorMsg("");
  };

  const applyPipelineMode = (mode: "fast_vision" | "audit_vision" | "text_only") => {
    if (mode === "fast_vision") {
      setForm((prev) => ({
        ...prev,
        llm_vision: true,
        llm_reasoning_effort: "none",
        llm_source_prepass: false,
        llm_vision_audit: false,
      }));
    } else if (mode === "audit_vision") {
      setForm((prev) => ({
        ...prev,
        llm_vision: true,
        llm_reasoning_effort: "low",
        llm_source_prepass: false,
        llm_vision_audit: true,
      }));
    } else {
      setForm((prev) => ({
        ...prev,
        llm_vision: false,
        llm_reasoning_effort: "none",
        llm_source_prepass: false,
        llm_vision_audit: false,
      }));
    }
  };

  const handleTestLLM = async () => {
    setTesting(true);
    setTestResult(null);
    setErrorMsg("");
    try {
      const headers: HeadersInit = {
        "Content-Type": "application/json",
        ...(token ? { "X-Prism-Token": token } : {}),
      };
      const res = await fetch("/api/v1/settings/test-llm", {
        method: "POST",
        headers,
        body: JSON.stringify(buildPayload()),
      });
      const data = await parseJsonSafe(res);
      if (res.status === 401) {
        setUnauthorized(true);
        setErrorMsg("访问未授权 (401)，请先验证 Admin Token");
        return;
      }
      setTestResult(data as TestLLMResponse);
    } catch (err) {
      setTestResult({ ok: false, latency_ms: 0, error: String(err) });
    } finally {
      setTesting(false);
    }
  };

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault();
    setSaving(true);
    setErrorMsg("");
    setWarningMsg("");
    try {
      const headers: HeadersInit = {
        "Content-Type": "application/json",
        ...(token ? { "X-Prism-Token": token } : {}),
      };
      const res = await fetch("/api/v1/settings", {
        method: "POST",
        headers,
        body: JSON.stringify(buildPayload()),
      });
      const data = await parseJsonSafe(res);
      if (res.status === 401) {
        setUnauthorized(true);
        setErrorMsg("访问未授权 (401)，请先填写正确的 Admin Token");
        return;
      }
      if (!res.ok) {
        setErrorMsg(data.error || `保存失败: HTTP ${res.status}`);
        return;
      }
      const typed = data as RuntimeSettingsResponse;
      setMaskedKey(typed.llm_api_key_masked || "");
      setApiKeySet(Boolean(typed.llm_api_key_set));
      setApiKeyInput("");
      setClearApiKey(false);
      setPresetProvider("");
      setPresetModel("");
      if (typed.warning) {
        setWarningMsg(typed.warning);
      }
      onSaved(typed);
      if (!typed.warning) {
        onClose();
      }
    } catch (err) {
      setErrorMsg(String(err));
    } finally {
      setSaving(false);
    }
  };

  const hw = health?.engine?.hardware;
  const activeMode =
    !form.llm_vision
      ? "text_only"
      : form.llm_vision_audit
      ? "audit_vision"
      : "fast_vision";

  const concurrencyHint = (() => {
    if (!hw || !hw.cuda_available || hw.vram_total_mb <= 0) return "CPU 模式建议 1";
    if (hw.vram_total_mb < 9000) return "当前显存建议 1–2";
    return "当前显存建议 2–4";
  })();

  const filteredPresets = presets.filter((p) => {
    if (visionPresetsOnly && !p.supports_vision) return false;
    const q = presetFilter.trim().toLowerCase();
    if (!q) return true;
    return (
      p.display_name.toLowerCase().includes(q) ||
      p.provider.toLowerCase().includes(q) ||
      p.model_id.toLowerCase().includes(q)
    );
  });

  const selectedPresetKey =
    presetProvider && presetModel ? `${presetProvider}::${presetModel}` : "";

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal-dialog modal-dialog-flex panel-card"
        role="dialog"
        aria-modal="true"
        aria-label="运行方式与模型配置中心"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="panel-title">
          <span className="inline-icon-text">
            <IconSettings size={16} /> 运行方式与大模型配置中心
          </span>
          <button type="button" className="btn btn-sm" onClick={onClose}>
            <IconClose size={13} /> 关闭
          </button>
        </div>

        {/* Compact generic hardware telemetry bar */}
        <div className="kv-grid mono" style={{ flexShrink: 0 }}>
          <div>
            <strong>算子状态:</strong>{" "}
            {health?.engine_connected
              ? hw
                ? `${hw.active_device.toUpperCase()} · ${hw.cuda_available ? hw.gpu_name || "GPU" : "CPU"}`
                : health.device.toUpperCase()
              : "离线 / 未连接"}
          </div>
          <div>
            <strong>硬件资源:</strong>{" "}
            {hw
              ? `${hw.vram_total_mb > 0 ? `${(hw.vram_total_mb / 1024).toFixed(1)} GB VRAM · ` : ""}${hw.cpu_threads} 核`
              : "等待算子响应"}
          </div>
        </div>

        {errorMsg && <div className="error-banner">{errorMsg}</div>}
        {warningMsg && <div className="warn-banner">{warningMsg}</div>}

        {unauthorized ? (
          <div className="panel-card" style={{ background: "#FEF3C7", boxShadow: "none" }}>
            <div style={{ fontWeight: 700, fontSize: 13 }}>
              当前浏览器未完成本地管理授权（可能通过非回环地址或开发端口访问）
            </div>
            <div style={{ fontSize: 12, color: "var(--zinc-note)" }}>
              直接访问本机服务地址会自动授权；若远程访问，请粘贴服务器 <span className="mono">data/admin.key</span> 令牌：
            </div>
            <div className="btn-row">
              <input
                type="password"
                value={token}
                placeholder="输入 Admin Token"
                onChange={(e) => onChangeToken(e.target.value)}
                style={{ flex: 1 }}
              />
              <button type="button" className="btn primary" onClick={() => void loadSettings(token)}>
                验证并加载设置
              </button>
            </div>
          </div>
        ) : loading ? (
          <div className="mono" style={{ fontSize: 12, padding: 12 }}>
            正在读取实时配置...
          </div>
        ) : (
          <form onSubmit={handleSave} className="modal-scroll-form">
            <div className="modal-scroll-body">
              {/* Section 1: Clean LLM Endpoint & Standard Providers */}
              <div className="settings-section">
                <div className="settings-section-title">1. 大模型接入（兼容任意 OpenAI 协议接口）</div>

                <div className="field">
                  <label>常用服务商快捷模板</label>
                  <div className="btn-row">
                    {STANDARD_PROVIDERS.map((sp) => (
                      <button
                        key={sp.label}
                        type="button"
                        className={`btn btn-sm ${form.llm_base_url === sp.base_url ? "primary" : ""}`}
                        onClick={() => {
                          setPresetProvider("");
                          setPresetModel("");
                          setForm((prev) => ({
                            ...prev,
                            llm_base_url: sp.base_url,
                            llm_model: sp.model,
                            llm_vision: sp.vision,
                          }));
                          setTestResult(null);
                        }}
                      >
                        {sp.label}
                      </button>
                    ))}
                  </div>
                </div>

                {presets.length > 0 && (
                  <div className="local-preset-box">
                    <div className="sidebar-label-row">
                      <label style={{ fontSize: 11, fontWeight: 800 }}>
                        从本机已配置模型下拉选择（{filteredPresets.length} / {presets.length}）
                      </label>
                      <label style={{ fontSize: 11, display: "flex", alignItems: "center", gap: 4 }}>
                        <input
                          type="checkbox"
                          checked={visionPresetsOnly}
                          onChange={(e) => setVisionPresetsOnly(e.target.checked)}
                        />
                        仅显示视觉多模态模型（推荐）
                      </label>
                    </div>
                    <div className="settings-grid-2" style={{ gap: 8 }}>
                      <input
                        type="text"
                        placeholder="关键词筛选模型（如 flash / gpt / gemini）..."
                        value={presetFilter}
                        onChange={(e) => setPresetFilter(e.target.value)}
                      />
                      <select
                        value={selectedPresetKey}
                        onChange={(e) => {
                          const val = e.target.value;
                          if (!val) return;
                          const [prov, mod] = val.split("::");
                          const found = presets.find((p) => p.provider === prov && p.model_id === mod);
                          if (found) handleSelectPreset(found);
                        }}
                      >
                        <option value="">-- 选择要填入的模型预设 --</option>
                        {filteredPresets.map((p) => (
                          <option key={`${p.provider}::${p.model_id}`} value={`${p.provider}::${p.model_id}`}>
                            [{p.provider}] {p.model_id} {p.supports_vision ? "(视觉)" : "(纯文本)"}
                          </option>
                        ))}
                      </select>
                    </div>
                  </div>
                )}

                <div className="settings-grid-2">
                  <div className="field">
                    <label>API Base URL</label>
                    <input
                      type="text"
                      value={form.llm_base_url}
                      placeholder="https://api.openai.com/v1"
                      onChange={(e) => {
                        setPresetProvider("");
                        setPresetModel("");
                        setForm({ ...form, llm_base_url: e.target.value });
                      }}
                    />
                  </div>
                  <div className="field">
                    <label>模型名称 (Model ID)</label>
                    <input
                      type="text"
                      value={form.llm_model}
                      placeholder="输入模型 ID"
                      onChange={(e) => {
                        setPresetProvider("");
                        setPresetModel("");
                        setForm({ ...form, llm_model: e.target.value });
                      }}
                    />
                  </div>
                </div>

                <div className="settings-grid-2">
                  <div className="field">
                    <label>
                      API Key{" "}
                      {apiKeySet && !clearApiKey && (
                        <span className="mono" style={{ color: "var(--emerald-ok)" }}>
                          (已保存: {maskedKey})
                        </span>
                      )}
                    </label>
                    <input
                      type="password"
                      value={apiKeyInput}
                      disabled={clearApiKey}
                      placeholder={
                        presetProvider && presetModel
                          ? "已绑定本机预设密钥，保存时自动注入"
                          : apiKeySet
                          ? "留空保留当前已保存密钥"
                          : "输入 sk-..."
                      }
                      onChange={(e) => {
                        setPresetProvider("");
                        setPresetModel("");
                        setApiKeyInput(e.target.value);
                      }}
                    />
                  </div>
                  <div className="field" style={{ justifyContent: "flex-end" }}>
                    <div className="btn-row" style={{ alignItems: "center" }}>
                      <button
                        type="button"
                        className="btn cyan"
                        disabled={testing}
                        onClick={() => void handleTestLLM()}
                      >
                        <IconZap size={13} /> {testing ? "探测中..." : "测试模型连通"}
                      </button>
                      {apiKeySet && (
                        <label style={{ fontSize: 12, display: "flex", alignItems: "center", gap: 4 }}>
                          <input
                            type="checkbox"
                            checked={clearApiKey}
                            onChange={(e) => {
                              setClearApiKey(e.target.checked);
                              if (e.target.checked) {
                                setApiKeyInput("");
                                setPresetProvider("");
                                setPresetModel("");
                              }
                            }}
                          />
                          清除密钥
                        </label>
                      )}
                    </div>
                  </div>
                </div>

                {testResult && (
                  <div className={testResult.ok ? "ok-banner mono" : "error-banner mono"}>
                    {testResult.ok
                      ? `连通成功 (${testResult.latency_ms} ms) · 模型=${testResult.model}`
                      : `连通测试失败 (${testResult.latency_ms} ms): ${testResult.error}`}
                  </div>
                )}
              </div>

              {/* Section 2: Pipeline Strategy */}
              <div className="settings-section">
                <div className="settings-section-title">2. 翻译质量策略</div>
                <div className="mode-cards">
                  <button
                    type="button"
                    className={`mode-card ${activeMode === "fast_vision" ? "active" : ""}`}
                    onClick={() => applyPipelineMode("fast_vision")}
                  >
                    <div className="mode-card-title inline-icon-text">
                      <IconZap size={14} /> 极速单轮视觉 (推荐)
                    </div>
                    <div className="mode-card-desc">整页上下文直译 + 关闭冗长思考，速度与准确度最均衡</div>
                  </button>
                  <button
                    type="button"
                    className={`mode-card ${activeMode === "audit_vision" ? "active" : ""}`}
                    onClick={() => applyPipelineMode("audit_vision")}
                  >
                    <div className="mode-card-title inline-icon-text">
                      <IconSearch size={14} /> 精校视觉审计
                    </div>
                    <div className="mode-card-desc">开启轻量思考与贴字二次复核，适合高复杂度精修</div>
                  </button>
                  <button
                    type="button"
                    className={`mode-card ${activeMode === "text_only" ? "active" : ""}`}
                    onClick={() => applyPipelineMode("text_only")}
                  >
                    <div className="mode-card-title inline-icon-text">
                      <IconFileText size={14} /> 纯文本 OCR 兼容
                    </div>
                    <div className="mode-card-desc">仅发送识别文本，适用于无视觉能力的纯文本模型</div>
                  </button>
                </div>

                <div className="settings-grid-2">
                  <div className="field">
                    <label>默认目标语言</label>
                    <input
                      type="text"
                      list="prism-target-langs"
                      value={form.target_lang}
                      onChange={(e) => setForm({ ...form, target_lang: e.target.value })}
                    />
                    <datalist id="prism-target-langs">
                      <option value="zh-CN">简体中文 (zh-CN)</option>
                      <option value="zh-TW">繁體中文 (zh-TW)</option>
                      <option value="en">English (en)</option>
                      <option value="ja">日本語 (ja)</option>
                      <option value="ko">한국어 (ko)</option>
                    </datalist>
                  </div>
                  <div className="field">
                    <label>推理思考预算 (reasoning_effort)</label>
                    <select
                      value={form.llm_reasoning_effort}
                      onChange={(e) => setForm({ ...form, llm_reasoning_effort: e.target.value })}
                    >
                      <option value="none">none · 关闭思考 (最快)</option>
                      <option value="omit">omit · 不发送该字段 (兼容旧接口)</option>
                      <option value="low">low · 轻量思考</option>
                      <option value="medium">medium · 中等思考</option>
                      <option value="high">high · 深度思考</option>
                    </select>
                  </div>
                </div>
              </div>

              {/* Section 3: Compute Device & Concurrency */}
              <div className="settings-section">
                <div className="settings-section-title">3. 本地计算后端与并发</div>
                {hasActiveJob && (
                  <div className="warn-banner" style={{ fontSize: 12 }}>
                    当前有后台任务正在运行，切换计算设备需等待任务完成或中止后生效。
                  </div>
                )}
                <div className="settings-grid-2">
                  <div className="field">
                    <label>推理计算设备</label>
                    <select
                      value={form.device}
                      onChange={(e) => setForm({ ...form, device: e.target.value })}
                    >
                      <option value="auto">auto · 自动检测最佳可用硬件</option>
                      <option value="cuda">cuda · NVIDIA CUDA GPU 加速</option>
                      <option value="dml">dml · DirectML (Windows 通用 GPU)</option>
                      <option value="cpu">cpu · 仅使用 CPU（高兼容）</option>
                    </select>
                  </div>
                  <div className="field">
                    <label>并发处理页数上限 (1–8，{concurrencyHint})</label>
                    <input
                      type="number"
                      min={1}
                      max={8}
                      value={form.max_concurrency}
                      onChange={(e) => setForm({ ...form, max_concurrency: Number(e.target.value) })}
                    />
                  </div>
                </div>
              </div>

              {/* Advanced collapsible options */}
              <details className="subtle-details">
                <summary>高级调参、访问令牌与持久化路径 ({settingsFile})</summary>
                <div style={{ display: "flex", flexDirection: "column", gap: 10, marginTop: 10 }}>
                  <div className="settings-grid-2">
                    <div className="field">
                      <label>单批节点数 (1–64)</label>
                      <input
                        type="number"
                        min={1}
                        max={64}
                        value={form.llm_batch_size}
                        onChange={(e) => setForm({ ...form, llm_batch_size: Number(e.target.value) })}
                      />
                    </div>
                    <div className="field">
                      <label>请求超时 (秒)</label>
                      <input
                        type="number"
                        min={10}
                        max={600}
                        value={form.llm_timeout_seconds}
                        onChange={(e) => setForm({ ...form, llm_timeout_seconds: Number(e.target.value) })}
                      />
                    </div>
                  </div>
                  <div className="btn-row" style={{ gap: 14 }}>
                    <label style={{ fontSize: 12, display: "flex", alignItems: "center", gap: 5 }}>
                      <input
                        type="checkbox"
                        checked={form.llm_vision}
                        onChange={(e) => setForm({ ...form, llm_vision: e.target.checked })}
                      />
                      整页视觉上下文
                    </label>
                    <label style={{ fontSize: 12, display: "flex", alignItems: "center", gap: 5 }}>
                      <input
                        type="checkbox"
                        checked={form.llm_source_prepass}
                        onChange={(e) => setForm({ ...form, llm_source_prepass: e.target.checked })}
                      />
                      源语言前置审校
                    </label>
                    <label style={{ fontSize: 12, display: "flex", alignItems: "center", gap: 5 }}>
                      <input
                        type="checkbox"
                        checked={form.llm_vision_audit}
                        onChange={(e) => setForm({ ...form, llm_vision_audit: e.target.checked })}
                      />
                      合成图视觉复核
                    </label>
                  </div>
                  <div className="field">
                    <label>
                      管理访问令牌（{authAuthorized ? "已认证" : "未认证"} · 托管模式: {engineManaged ? "自动" : "外部"}）
                    </label>
                    <div className="btn-row">
                      <input
                        type="password"
                        value={token}
                        placeholder="Admin Token (data/admin.key)"
                        onChange={(e) => onChangeToken(e.target.value)}
                        style={{ flex: 1 }}
                      />
                      <button type="button" className="btn btn-sm" onClick={() => void loadSettings(token)}>
                        同步
                      </button>
                    </div>
                  </div>
                </div>
              </details>
            </div>

            <div className="modal-sticky-footer">
              <button type="button" className="btn" onClick={onClose}>
                取消
              </button>
              <button type="submit" className="btn primary" disabled={saving}>
                {saving ? "保存应用中..." : "保存设置并立即生效"}
              </button>
            </div>
          </form>
        )}
      </div>
    </div>
  );
}
