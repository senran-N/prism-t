# 系统架构与技术原理

Prism-T 采用「Go 控制平面 + Python 算子平面」双进程本地架构，将业务编排、状态持久化、LLM 叙事编译与重型视觉/数值计算清晰解耦。

## 1. 双平面职责边界

```text
+-------------------------------------------------------------------+
| prismtd (Go, CGO_ENABLED=0, :3090)                                |
|  - Archive I/O (dir / .cbz / .zip / PNG / JPEG / WebP / AVIF)     |
|  - SQLite WAL Store (works / pages / characters / glossary)       |
|  - Narrative Compiler (Vision packet / JSON Schema / Audit)       |
|  - Native Binary PSD v1 Writer (PackBits RLE + UTF-16BE luni)     |
|  - Engine Supervisor (auto-spawn / health probe / auto-restart)   |
+---------------------------------+---------------------------------+
                                  | HTTP Loopback (127.0.0.1:18775)
                                  v
+---------------------------------+---------------------------------+
| prism-t-engine (Python, FastAPI, stateless)                       |
|  - /health                                                        |
|  - /v1/check_seam       (Webtoon boundary continuity test)        |
|  - /v1/detect_layout    (CTD + OCR geometry + bubble topology)    |
|  - /v1/restore_page     (Z0 clean-plate + Z2 occluders + style)   |
|  - /v1/analyze          (Combined layout + restoration)           |
|  - /v1/render           (SDF Knuth-Plass typeset + Z0/Z1/Z2 comp) |
+-------------------------------------------------------------------+
```

- **两阶段流水线重叠**：翻译单页时，`prismtd` 先调用 `/v1/detect_layout` 获取版面与节点框，随后并行发起 `/v1/restore_page`（GPU 去字修复）与远端 LLM 视觉翻译，将 GPU 修复耗时隐藏在网络等待中。
- **毫秒级局部重排**：用户在 Web Studio 修改译文、字号或位置时，仅调用 `/v1/render` 复用缓存的 `clean_z0`、`z2_occluder_mask` 与 `shading_map`，不重复执行 OCR 与去字修复。

## 2. Comic DOM 与三层可逆合成

每页在 SQLite 中以 `PageDOM`（`schema_version = 1`）持久化，包含：

- **`00_Original_Raw`**：无损原始输入图像；
- **`01_Restored_Clean_Z0`**：仅在已记录 `stroke_alpha > 0` 掩膜支持内写回的洁净底图；
- **`02..N_Node_Z1`**：逐节点独立排版渲染的新译文透明图层（附带 `render_fingerprint` 实现未改动节点缓存复用）；
- **`99_Foreground_Occluders_Z2`**：从外部画作连续穿入文字区域的前景遮挡层（如发丝、武器、边框）。

合成严格遵守 **非掩膜原画逐字节无损不变式（`L_inf == 0`）**：对所有 `stroke_alpha == 0` 且 `rendered_z1_alpha == 0` 的像素，最终输出与原始输入保持 `uint8` 逐字节完全一致。

## 3. 视觉与数学算子关键机制

### 3.1 预检与长条漫跨切片缝合
- 使用非重叠二维 Haar 小波高频子带的下四分位 MAD 估计传感器噪声 $\hat{\sigma}_n$，结合 CIE Lab 统计与几何方向自动判定版式（`manga_rtl` / `webtoon_vertical` / `comic_ltr`）。
- 对连续同宽切片边界执行经验分位数连续性检验，自动缝合被物理切分的气泡与文字，渲染后再按原切片偏移无损裁回。

### 3.2 闭式抠图、多层描边回收与振假名剥离
- **完整四周背景环采样**：从扩展页裁图采样多边形完整外周背景，按真实外周亮度与色差显著性判定前景，避免跨「纯黑画框 + 彩色背景」长竖列发生极性反转或丢弃与背景同灰度的彩色字芯。
- **外层描边壳回收（`_recover_outer_outline_shell`）**：以实测笔画半径 Tukey 上界为尺度，联合 Laplace 外背景与最近可见外像素双重对比，并对满足拓扑包绕占比的外层不透明描边（如三层双描边字幕的粗黑外框）做连通回收与 BH-FDR 抗锯齿羽化。
- **汉字—振假名分离**：基于对数面积 2-GMM 与 BIC 模型选择剥离正文与旁注振假名（`ルビ` / `義訓ルビ`），并通过 Bookstein TPS 展平弯曲文本脊线。

### 3.3 多路物理门控 `Z0` 底图修复
- **跨版本无字生肉光流融合**：提供 `--ref` 时以 `CLAHE + SIFT + USAC_MAGSAC` 粗配准叠加稠密 `DIS Optical Flow` 移植真实原画。
- **纯色拉普拉斯调和插值**：纯色气泡直接求解稀疏 Dirichlet 边值方程 $\Delta u = 0$。
- **二维傅里叶晶格网点重构**：以 Fisher's $g$-test 检出周期网点基频并拟合实三角级数相位。
- **Weickert 各向异性扩散 / 双调和方程 / 局部掩膜 LaMa**：在复杂画作上通过隐藏可观测背景环的交叉验证误差择优，仅写回实测掩膜支持。

### 3.4 4D 光学解混与 SDF 排版
- **半透明涂层反演**：沿笔画内外边界对做协方差线性回归求解不透明度与前景墨色。
- **`Z2` 前景穿字提取**：结合外背景中位/75 分位异质性检验（避免双区背景误判）、颜色段外偏离度与神经文本概率否决，防止将原文残影误提为 `Z2` 盖回新译文。
- **气泡 SDF 动态规划与墨量守恒**：计算气泡内腔欧氏 SDF 与不可达极点，以广义 Knuth-Plass 动态规划折行，内置 JIS X 4051 / GB/T 15834 避头尾禁则与 Brent 根求解描边/字重守恒。

## 4. 叙事编译与并发流式编排

- **严格输出契约**：使用 Chat Completions SSE 流配合 `response_format.type=json_schema`，按节点 ID 键控输出 `target_segments`，校验数字串、伏字符号串与分段拼接一致性，并对瞬时网络/限流错误做有限次指数退避重试。
- **语义上下文增强**：注入滑动章节上下文（`preceding_chapter_context`）、64 维外观聚类角色人设、称谓关系图、作品术语表与跨节点分句续接拓扑提示；支持可选轻量推理档位与整页二审复核。
- **恒定内存流式续译**：`TranslatePageStream` 支持 `PRISM_T_CONCURRENCY` 多路并发工作池与每页完成回调（`OnPageDone`），在百万像素级长篇目录翻译中保持恒定内存占用与断点续译。
