# 快速开始与源码构建

本指南说明如何在本地搭建 Prism-T 的 Go 控制平面、Python 视觉算子平面与 React Web Studio。

## 1. 环境要求

| 组件 | 版本要求 | 说明 |
| --- | --- | --- |
| **Go** | `1.24+` | 支持 `CGO_ENABLED=0` 静态构建 |
| **Python** | `3.11+` | 推荐使用 [`uv`](https://docs.astral.sh/uv/) 管理虚拟环境 |
| **Node.js** | `20+` | 构建 `web/` 前端静态资源 |
| **GPU（可选）** | NVIDIA CUDA | 自动检测 CUDA 加速 LaMa 与 ONNX；无 GPU 时自动回退 CPU |

## 2. 安装 Python 算子依赖与模型权重

```bash
# 同步 Python 算子环境（含开发与测试依赖）
uv --directory engine sync --extra dev

# 下载并校验漫画文字分割、背景修复与源字识别权重（约 301 MB）
uv --directory engine run python -m prism_t_engine.models
```

下载器会将模型保存至 `data/models/` 并记录 `*.provenance.json`（含下载 URL 与 SHA-256 哈希）。若缺少权重，引擎会显式报错而不静默降级。

## 3. 构建前端与 Go 单二进制

Linux / macOS：
```bash
npm --prefix web ci
npm --prefix web run build
CGO_ENABLED=0 go build -ldflags="-s -w" -o prismtd ./cmd/prismtd
```

Windows PowerShell：
```powershell
npm --prefix web ci
npm --prefix web run build
$env:CGO_ENABLED="0"; go build -ldflags="-s -w" -o prismtd.exe ./cmd/prismtd
```

## 4. 启动 Web Studio

```bash
./prismtd serve --listen 127.0.0.1:3090 --open
```

Windows 下也可直接双击仓库根目录的 `start-web.bat` 或运行 `./scripts/start-web.ps1`。

- `prismtd serve` 会自动拉起并监督 `127.0.0.1:18775` 上的 `prism-t-engine` 子进程。
- 首次启动若未设置 `PRISM_T_ADMIN_TOKEN`，程序会自动生成随机管理员密钥并保存至 `data/admin.key`。
- 可在 Web Studio 的设置面板中配置 OpenAI 兼容 LLM 端点、模型、计算设备（`auto` / `cuda` / `cpu` / `dml`）并测试连通性。

## 5. 首次命令行测试

### 5.1 纯去字修复与 PSD 导出（无需 LLM）

```bash
./prismtd translate --in ./page_01.png --out ./page_01_clean.psd --analyze-only
```

### 5.2 配置 LLM 执行端到端翻译

```bash
export PRISM_T_LLM_BASE_URL="https://api.example.com/v1"
export PRISM_T_LLM_API_KEY="sk-..."
export PRISM_T_LLM_MODEL="your-vision-model"
export PRISM_T_LLM_VISION="true"

./prismtd translate --in ./raw_pages --out ./out_zh --work "Sample" --stream --resume --continue-on-error
```

## 6. 独立验收脚本

项目在 `scripts/` 下提供独立验收工具（不依赖运行态内部状态）：

```bash
# 导出原图、去字底图、译制合成图、差异图与掩膜外最大像素误差
uv --directory engine run python ../scripts/inspect_result.py --db ../data/prism_t.db --out ./inspect_report

# 独立解码导出的 .psd 图层并校验合成误差与图层顺序
uv run --with psd-tools --with numpy --with pillow python scripts/inspect_psd.py ./page_01.psd
```
