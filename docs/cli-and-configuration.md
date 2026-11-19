# CLI 与配置参考

`prismtd` 是 Prism-T 的统一单二进制入口，涵盖本地服务、批量翻译、断点续译、分层导出、术语管理与统计查询。

## 1. 子命令总览

```text
prismtd <command> [flags]
```

| 子命令 | 说明 |
| --- | --- |
| `serve` | 启动 HTTP API、Web Studio 静态前端与本地 `prism-t-engine` 守护监督器 |
| `translate` | 翻译单图、图片目录、条漫切片目录或 `.cbz` / `.zip` 归档 |
| `retry` | 复用数据库中已保存的视觉分析与去字底图，重新翻译或重新排版渲染 |
| `export` | 将数据库中的单页或整部作品导出为原生多图层 `.psd` 或 `.cbz` |
| `glossary` | 导入或查看作品级专有名词与双关振假名术语表 |
| `stats` | 输出当前数据库的作品、页面、待复核队列、角色簇与术语统计 |

---

## 2. 子命令详细标志

### `prismtd serve`

| 标志 | 默认值 | 说明 |
| --- | --- | --- |
| `--listen` | `127.0.0.1:3090` | HTTP 监听地址 |
| `--engine-url` | `http://127.0.0.1:18775` | 本地算子服务 URL |
| `--engine-dir` | `engine` | Python 算子目录路径 |
| `--db` | `data/prism_t.db` | SQLite WAL 数据库路径 |
| `--device` | `auto` | 计算设备（`auto`、`cpu`、`gpu`、`cuda`、`dml`） |
| `--llm-url` | 环境变量 | OpenAI 兼容 LLM Base URL |
| `--llm-model` | 环境变量 | LLM 模型标识 |
| `--web-dist` | `web/dist` | 前端构建产物目录 |
| `--open` | `false` | 监听就绪后自动在默认浏览器打开 Web Studio |

### `prismtd translate`

| 标志 | 默认值 | 说明 |
| --- | --- | --- |
| `--in` | （必填） | 输入图片、图片目录或 `.cbz` / `.zip` 归档 |
| `--ref` | `""` | 可选无字/异语种参考版路径（用于跨版本光流底图融合） |
| `--out` | `""` | 输出路径（`.cbz`、单页 `.psd` 或输出目录；留空则仅写入数据库） |
| `--work` | `Default Work` | 作品名（写入角色人设与术语库） |
| `--chapter` | `Chapter 1` | 章节名 |
| `--profile` | `auto` | 版式配置（`auto`、`manga_rtl`、`webtoon_vertical`、`comic_ltr`） |
| `--lang` | `zh-CN` | 目标语言（如 `zh-CN`、`zh-TW`、`en`、`ko`） |
| `--analyze-only` | `false` | 仅执行去字修复与版面反编译，不调用 LLM |
| `--psd` | `false` | 输出目录模式下同步为每页生成分层 `.psd` |
| `--stream` | `false` | 目录流式逐页翻译，恒定内存并逐页立即原子写盘 |
| `--resume` | `false` | 断点续译（自动启用 `--stream`）：跳过库中已译页，缺失输出文件直接从库回补 |
| `--continue-on-error` | `false` | 流式模式下单页失败不中断整批，记录失败明细至 `<out>/_failed_pages.jsonl` |
| `--device` | `auto` | 计算设备（`auto`、`cpu`、`gpu`、`cuda`、`dml`） |
| `--db` / `--engine-url` / `--engine-dir` / `--llm-url` / `--llm-model` | 同环境变量 | 覆盖默认路径与后端配置 |

### `prismtd retry`

| 标志 | 默认值 | 说明 |
| --- | --- | --- |
| `--page-id` | `0` | 重试指定单页（与 `--work-id` 二选一） |
| `--work-id` | `0` | 按阅读顺序恢复整部作品所有已存页面 |
| `--out` | （必填） | 输出目录 |
| `--lang` | `zh-CN` | 目标语言 |
| `--psd` | `false` | 同步导出分层 `.psd` |
| `--retranslate` | `false` | 即使该页已存在完成译文，仍显式重新请求 LLM 翻译 |
| `--db` | `data/prism_t.db` | SQLite 数据库路径 |

### `prismtd export` / `glossary` / `stats`

- `prismtd export --page-id <ID> --out ./page.psd`：导出单页原生多图层 `.psd`。
- `prismtd export --work-id <ID> --out ./work.cbz`（或传目录路径批量导出 `.psd`）。
- `prismtd glossary --work "<作品名>" [--import ./terms.json]`：JSON 格式为 `[{"source_term":"...","target_term":"...","ruby_override":"...","notes":"..."}]`。
- `prismtd stats [--db ./data/prism_t.db]`：输出 JSON 统计摘要。

---

## 3. 环境变量参考

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `PRISM_T_LISTEN` | `127.0.0.1:3090` | HTTP 服务监听地址 |
| `PRISM_T_ENGINE_URL` | `http://127.0.0.1:18775` | 本地算子平面地址 |
| `PRISM_T_ENGINE_DIR` | `engine` | Python 算子工程目录 |
| `PRISM_T_DATA_DIR` | `data` | 数据与密钥目录 |
| `PRISM_T_DB_PATH` | `data/prism_t.db` | SQLite WAL 数据库路径 |
| `PRISM_T_ADMIN_TOKEN` | 自动生成至 `data/admin.key` | 管理接口鉴权令牌 |
| `PRISM_T_CORS_ORIGINS` | `""`（同源） | 逗号分隔的允许跨域源白名单 |
| `PRISM_T_AUTO_ENGINE` | `true` | 是否由 `prismtd` 自动拉起并监督 `prism-t-engine` |
| `PRISM_T_DEVICE` | `auto` | 计算设备（`auto` / `cpu` / `gpu` / `cuda` / `dml`） |
| `PRISM_T_TARGET_LANG` | `zh-CN` | 默认目标语言 |
| `PRISM_T_LLM_BASE_URL` | `""` | OpenAI 兼容 Chat Completions Base URL |
| `PRISM_T_LLM_API_KEY` | `""` | LLM API 密钥 |
| `PRISM_T_LLM_MODEL` | `gpt-4o-mini` | LLM 模型名称 |
| `PRISM_T_LLM_VISION` | `false` | 设为 `true` 启用整页+节点图集视觉翻译模式 |
| `PRISM_T_LLM_REASONING_EFFORT` | `none` | 推理档位（如 `none`、`low`、`medium`、`high`、`auto`） |
| `PRISM_T_LLM_VISION_AUDIT` | `false` | 设为 `true` 启用整页第二遍视觉语义复核 |
| `PRISM_T_LLM_SOURCE_PREPASS` | `false` | 设为 `true` 启用独立的纯源文预转录通道 |
| `PRISM_T_LLM_TIMEOUT_SECONDS` | `300` | 单次 LLM 流式请求超时时间（秒） |
| `PRISM_T_LLM_BATCH_SIZE` | `32` | 单次 LLM 请求包含的最大节点数 |
| `PRISM_T_SOURCE_LINES_PER_REQUEST` | `24` | 密集节点预转录单批最大行数 |
| `PRISM_T_SKIP_SOURCE_OCR` | `false` | 视觉模式下设为 `true` 跳过本地逐行 `source_readings` 识别，显著降低 `DetectLayout` 耗时与显存占用 |
| `PRISM_T_CONCURRENCY` | `1` | 并发处理页数（目录流式翻译推荐 `4`–`8`） |
