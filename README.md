# Prism-T

中文 | [English](README_EN.md)

Prism-T 是一个面向日式漫画、彩色长条漫（Webtoon）与横排美漫的本地化翻译与嵌字工具链。系统以可逆的 `Comic DOM` 组织页面：独立保存 `Z0` 洁净底图、`Z1` 译文排版层与 `Z2` 前景穿字遮挡层，在掩膜外严格保持原图像素逐字节无损（`L_inf == 0`），并支持 Web Studio 交互改稿以及多图层 `.psd` / `.cbz` 导出。

---

## 功能概览

- **无损预检与长条漫无缝缝合**
  - 基于二维 Haar 小波下四分位 MAD 估计噪声基准 $\hat{\sigma}_n$，结合 CIE Lab 统计与几何方向分布自动判定版式（日漫右翻、条漫竖滑、美漫左翻）；
  - 对长条漫切片边界执行连续性检验，自动缝合被物理切分的气泡与跨页文字，完成渲染后按原切片偏移无损裁回。

- **闭式笔画抠图与多层描边擦除**
  - 结合 Comic Text Detector 神经分割、扩展四周背景环的色差显著性提取与拓扑包绕检验，分离字芯、内描边与外层粗描边（含跨纯黑画框与彩色背景的双描边文字）；
  - 基于 2-GMM 与 BIC 准则自动剥离正文汉字与振假名（`ルビ` / `義訓ルビ`），支持弯曲脊线薄板样条（Bookstein TPS）展平识别。

- **物理门控多路底图修复（`Z0` Clean-Plate）与 4D 光学合成**
  - 支持跨版本无字生肉光流融合（`CLAHE + SIFT + USAC_MAGSAC + DIS Optical Flow`）、纯色气泡拉普拉斯调和插值、二维傅里叶晶格网点重构（Fisher's $g$-test）、Weickert 各向异性扩散与原生 FFT 局部掩膜 LaMa 修复；
  - 仅在已记录的文字掩膜内写回修复像素，绝不将整页送入 VAE 或扩散重编码器；
  - 支持半透明涂层逆解混、`Z2` 前景穿字提层回压与局部漫反射照度分解（`Albedo × Shading`）。

- **气泡 SDF 动态规划排版与光学墨量守恒**
  - 计算气泡内腔精确欧氏距离场（SDF），使用广义 Knuth-Plass 动态规划贴合气泡轮廓折行；
  - 内置中日文避头尾禁则（JIS X 4051 / GB/T 15834）、竖排标点映射与 `義訓ルビ` 双层排版，并通过 Brent 根求解动态调整字重与描边以守恒原画墨量密度。

- **上下文叙事编译器与大批量流式续译**
  - 支持视觉模式（整页缩略概览 + 原尺寸节点图集）与纯文本 OCR 模式，严格校验 JSON Schema 结构化输出、数字串与伏字符号守恒；
  - 内置跨章节角色人设状态机（64 维外观聚类、双向称谓关系图）、专有名词术语表、跨节点分句续接拓扑、可选轻量推理（`PRISM_T_LLM_REASONING_EFFORT`）与整页二审（`PRISM_T_LLM_VISION_AUDIT`）；
  - 支持目录流式断点续译（`--stream --resume --continue-on-error`）与多路并发（`PRISM_T_CONCURRENCY`），逐页原子落盘并同步写入 Web Studio 数据库。

- **Web Studio 与原生多图层导出**
  - 内置本地 Web Studio（书架管理、边译边看、画布交互调字/调框、局部重排无需重跑去字、流式图源与本地目录导入）；
  - 原生纯 Go 实现 Adobe Photoshop `.psd` v1 编码（PackBits RLE + UTF-16BE `8BIMluni` 图层名），固定分层为 `00_Original_Raw` → `01_Restored_Clean_Z0` → `02..N_Node_Z1` → `99_Foreground_Occluders_Z2`。

### 适用边界与非目标

- **诚实翻译后端边界**：Prism-T 绝不内置硬编码词表或伪造离线假译文。端到端翻译需配置支持严格 JSON Schema（`response_format.type=json_schema`）的 OpenAI 兼容 Chat Completions 端点；未配置端点时请使用 `--analyze-only` 执行纯视觉解构、去字修复与分层 `.psd` 导出。
- **单节点本地架构**：状态存储基于单机 SQLite WAL（`data/prism_t.db`），算子平面运行于本地回环 `127.0.0.1:18775`，面向单机工作站设计。
- **自动化质量边界**：极细手写拟声词漏检、极端复杂遮挡与高度依赖上下文的语义润色仍可能需在 Web Studio 或导出的分层 `.psd` 中微调。

---

## 系统架构

```text
[CLI / Web Studio] ---> prismtd (Go, CGO_ENABLED=0, :3090) ---> data/prism_t.db (SQLite WAL)
                              |---> OpenAI-Compatible LLM Endpoint (Narrative Compiler & Audit)
                              |---> Native Multi-Layer PSD v1 & Seamless CBZ Exporter
                              |
                              v  HTTP loopback (127.0.0.1:18775, auto-supervised)
                        prism-t-engine (Python, FastAPI / SciPy / OpenCV / ONNX / PyTorch)
                              |---> Scale-Invariant Preprocessor & Webtoon Seam Stitcher
                              |---> CTD + Closed-Form Matting + Outer-Rim Shell Recovery + TPS OCR
                              |---> Multi-Path Z0 Clean-Plate Solver (Flow / Laplace / FFT / PDE / LaMa)
                              +---> 4D Optical Unmixing (Alpha / Z2 / Shading) + SDF Knuth-Plass Typesetter
```

---

## 快速开始

环境要求：**Go 1.24+**、**Python 3.11+**（含 [`uv`](https://docs.astral.sh/uv/)）、**Node.js 20+**。支持 NVIDIA CUDA GPU 加速或纯 CPU 运行。

```powershell
# 1. 安装 Python 视觉与数学算子依赖
uv --directory engine sync --extra dev

# 2. 下载漫画文字分割、背景修复与日文源字识别权重（含 SHA-256 校验）
uv --directory engine run python -m prism_t_engine.models

# 3. 构建前端静态资源并编译纯 Go 零 CGO 单二进制
npm --prefix web ci
npm --prefix web run build
$env:CGO_ENABLED="0"; go build -ldflags="-s -w" -o prismtd.exe ./cmd/prismtd

# 4. 启动 Web Studio（自动拉起并监督本地 prism-t-engine 子进程）
./prismtd.exe serve --listen 127.0.0.1:3090
```

> Linux / macOS 下请使用 `CGO_ENABLED=0 go build -ldflags="-s -w" -o prismtd ./cmd/prismtd`。首次启动若未设置 `PRISM_T_ADMIN_TOKEN`，管理员令牌将自动生成于 `data/admin.key`。

### 常用命令行操作

```powershell
# 纯视觉解构、去字修复与分层 .psd 导出（无需配置 LLM）
./prismtd.exe translate --in ./page_01.png --out ./page_01_layered.psd --analyze-only

# 配置 OpenAI 兼容端点后执行端到端翻译（支持单图、目录、.cbz/.zip 与 --ref 无字生肉）
$env:PRISM_T_LLM_BASE_URL="https://your-openai-compatible-endpoint/v1"
$env:PRISM_T_LLM_API_KEY="your-api-key"
$env:PRISM_T_LLM_MODEL="your-vision-model"
$env:PRISM_T_LLM_VISION="true"
./prismtd.exe translate --in ./raw_chapter.cbz --out ./zh_cn.cbz --work "Sample Manga" --chapter "第01话" --lang zh-CN

# 大批量图片目录流式并发断点续译（恒定内存、逐页原子落盘并同步写入 Web Studio）
$env:PRISM_T_CONCURRENCY="4"
$env:PRISM_T_LLM_REASONING_EFFORT="low"
$env:PRISM_T_LLM_VISION_AUDIT="true"
./prismtd.exe translate --in ./raw_pages --out ./out_zh --work "Sample Manga" --stream --resume --continue-on-error

# 复用已保存的去字底图重试翻译 / 仅重新排版
./prismtd.exe retry --page-id 1 --out ./retry --psd
./prismtd.exe retry --work-id 1 --out ./retry-work --retranslate

# 导出单页分层 .psd、整部作品 .cbz 或管理术语表
./prismtd.exe export --page-id 1 --out ./page_01.psd
./prismtd.exe export --work-id 1 --out ./work_01.cbz
./prismtd.exe glossary --work "Sample Manga" --import ./glossary_terms.json
./prismtd.exe stats
```

---

## 开发者文档

详细文档位于 [`docs/`](docs/) 目录：

- **[快速开始与源码构建](docs/getting-started.md)**：环境准备、模型下载校验、CUDA/CPU 配置与首次运行
- **[系统架构与技术原理](docs/architecture.md)**：双平面控制/算子职责、`Comic DOM` 分层、去字/修复与排版原理
- **[CLI 与配置参考](docs/cli-and-configuration.md)**：全部子命令参数、流式断点续译与 `PRISM_T_*` 环境变量对照表
- **[HTTP API 参考](docs/api-reference.md)**：Web Studio 接口、鉴权机制、异步任务与多格式导出协议

---

## 仓库目录结构

```text
.
├── cmd/prismtd/         # CLI 入口（serve, translate, retry, export, glossary, stats）
├── internal/
│   ├── archive/         # 目录、.cbz/.zip 自然序读取（含 AVIF/WebP 支持）与条漫切片缝合
│   ├── config/          # 环境变量、CLI 标志、Web Studio 运行时配置与管理员密钥
│   ├── dom/             # Comic DOM 结构与渲染/识别进度契约
│   ├── engine/          # prism-t-engine 客户端、子进程守护监督与流式并发编排器
│   ├── narrative/       # 叙事编译器、分句拓扑、结构化 JSON Schema 校验与视觉打包
│   ├── psd/             # 纯 Go 原生 Photoshop .psd v1 编码器（PackBits RLE + UTF-16BE）
│   ├── server/          # HTTP API、鉴权、异步任务队列、流式图源与 Web Studio 静态托管
│   ├── source/          # 远程漫画图源客户端
│   └── store/           # SQLite WAL 存储（作品、页面、角色人设簇、称谓图与术语表）
├── engine/              # Python 视觉与数学算子服务（FastAPI / SciPy / OpenCV / ONNX / PyTorch）
├── web/                 # React 19 + TypeScript + Vite 6 前端工作台源码
├── scripts/             # 本地启动与结果/PSD 验收脚本
└── docs/                # 开发者文档
```

---

## 测试验证

```bash
# Go 单元测试与路由/编排集成测试
CGO_ENABLED=0 go test -count=1 ./...

# Python 算子平面数学不变量与接口测试
uv --directory engine run pytest -v

# Web 前端类型检查与生产构建
npm --prefix web run build
```

---

## 模型与第三方致谢

模型权重按需下载至本地 `data/models/`，不随代码仓库分发，下载器会记录来源 URL 与 SHA-256 校验信息：

- [Comic Text Detector](https://github.com/dmMaze/comic-text-detector)（漫画文字检测与分割）
- [AnimeMangaInpainting](https://huggingface.co/dreMaz/AnimeMangaInpainting)（基于 [LaMa](https://github.com/advimman/lama) 的漫画背景修复模型）
- [manga-ocr-nar-small](https://huggingface.co/muscgab/manga-ocr-nar-small)（日文源字识别 ONNX 权重，其权重受 CC BY-NC-SA 4.0 许可约束，详见下载目录 `NOTICE`）

---

## 参与贡献与安全反馈

- 贡献流程与代码规范请参阅 [CONTRIBUTING.md](CONTRIBUTING.md)。
- 安全边界说明与漏洞报告方式请参阅 [SECURITY.md](SECURITY.md)。

## 许可证

本项目源代码基于 [GNU General Public License v3.0](LICENSE) 开源。
