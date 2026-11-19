# 贡献指南 (Contributing Guide)

欢迎参与 Prism-T 的开发与改进。提交 Issue 或 Pull Request 前，请参考以下本地构建、测试与工程约束。

## 1. 开发环境要求

- **Go**：`1.24` 及以上（控制平面要求支持 `CGO_ENABLED=0` 零依赖静态构建）
- **Python**：`3.11` 及以上，使用 [`uv`](https://docs.astral.sh/uv/) 管理虚拟环境与依赖锁文件
- **Node.js**：`20` 及以上（用于构建 `web/` 前端静态资源）

## 2. 本地测试与构建验证

提交代码变更前，请确保以下三套检查全部通过：

### Go 控制平面测试与静态编译
```bash
CGO_ENABLED=0 go test -count=1 ./...
CGO_ENABLED=0 go build ./cmd/prismtd
```

Windows PowerShell：
```powershell
$env:CGO_ENABLED="0"
go test -count=1 ./...
go build -o prismtd.exe ./cmd/prismtd
```

### Python 算子平面测试
```bash
uv --directory engine run pytest -v
```

### Web 前端类型检查与生产构建
```bash
npm --prefix web ci
npm --prefix web run build
```

## 3. 核心工程约束

1. **保持 `CGO_ENABLED=0` 兼容性**：`cmd/` 与 `internal/` 下的 Go 代码不得引入依赖 CGO 的第三方包（SQLite 使用纯 Go 实现的 `modernc.org/sqlite`，AVIF/WebP 解码均采用纯 Go 库）。
2. **严禁硬编码译文词表与静默回退**：`internal/narrative/` 与 `engine/` 禁止内置固定台词/拟声词对照表或伪造离线假译文；未配置兼容端点且未传 `--analyze-only` 时必须返回 `ErrNoTranslationBackend`。
3. **非掩膜原画逐字节无损（`L_inf == 0`）**：禁止将整页送入 VAE 或扩散重编码器；在 `stroke_alpha == 0` 且新译文 `rendered_z1_alpha == 0` 的所有像素上，合成图与原图必须保持 `uint8` 逐字节完全一致。
4. **统计自标定，避免分辨率魔数**：图像噪声、前景/背景极性、笔画与描边外扩尺度、网点周期及条漫接缝判定应由图像自身统计量（如 Haar MAD、Otsu、χ² 检验、EDT 笔画半径分位数与 BIC）推导，避免硬编码固定像素阈值。
5. **文档同步**：修改 CLI 标志、环境变量、HTTP API 或 `Comic DOM` 契约时，请同步更新 `README.md`、`README_EN.md` 及 [`docs/`](docs/) 下的对应文档。

## 4. 提交规范

建议使用清晰的 Conventional Commits 前缀，例如：
- `feat(engine): ...`
- `fix(narrative): ...`
- `docs: ...`
- `test(psd): ...`
