# HTTP API 参考

`prismtd serve` 默认在 `http://127.0.0.1:3090` 提供 JSON API、无损图片/导出流以及 Web Studio 静态界面。

## 1. 鉴权规则

- **只读浏览与导出接口**：默认可直接在本地访问。
- **写入、翻译、重排、删除与配置接口**：采取 Fail-Closed 策略，必须在请求头携带管理员令牌：
  - `X-Prism-Admin-Token: <token>`，或
  - `Authorization: Bearer <token>`
- 令牌来自环境变量 `PRISM_T_ADMIN_TOKEN`，未设置时保存在 `data/admin.key`。

---

## 2. 只读与导出接口

| 方法与路径 | 说明 |
| --- | --- |
| `GET /api/v1/health` | 控制平面、算子平面（含硬件设备与模型就绪状态）及 LLM 配置探针 |
| `GET /api/v1/stats` | 已收录作品数、总页数、待复核页数、角色簇数与术语表规模 |
| `GET /api/v1/works` | 作品书架列表（含页数、章节数、封面 `cover_page_id` 与待复核计数） |
| `GET /api/v1/pages?work_id={id}&needs_review=1` | 查询作品下所有页面摘要或按待复核状态筛选 |
| `GET /api/v1/pages/{id}/dom` | 获取单页完整 `Comic DOM` JSON（含节点框、译文、样式与诊断指标） |
| `GET /api/v1/pages/{id}/raw` | 下载单页无损原始 PNG |
| `GET /api/v1/pages/{id}/composite` | 下载单页当前合成译制 PNG |
| `GET /api/v1/pages/{id}/export/psd` | 实时编码下载单页原生多图层 `.psd`（`00_Original_Raw` / `01_Restored_Clean_Z0` / `Z1` / `99_Foreground_Occluders_Z2`） |
| `GET /api/v1/works/{id}/export/cbz` | 按阅读顺序打包下载整部作品 `.cbz` 归档 |
| `GET /api/v1/works/{id}/personas` | 获取作品角色人设簇、双向称谓关系图与术语表 |
| `GET /api/v1/jobs` | 列出当前与近期异步翻译任务快照 |
| `GET /api/v1/jobs/{id}` | 查询指定异步任务的实时进度（下载数、完成页数、阶段与错误信息） |
| `GET /api/v1/sources` | 列出已注册的远程漫画图源 |
| `GET /api/v1/sources/cover` | 代理已注册图源的封面栅格图像 |

---

## 3. 受保护的写入与工作台接口（需管理员令牌）

| 方法与路径 | 说明 |
| --- | --- |
| `GET /api/v1/admin/auth` | 校验管理员令牌是否有效 |
| `GET /api/v1/settings` | 读取当前运行时配置（计算设备、LLM 端点/模型/推理档位/复核开关，API Key 脱敏返回） |
| `POST /api/v1/settings` | 更新运行时配置并持久化至 `data/settings.json`（切换设备时自动热重启本地算子引擎） |
| `POST /api/v1/settings/test-llm` | 实时探测当前 LLM 端点连通性与延迟 |
| `POST /api/v1/translate` | `multipart/form-data` 同步上传单图、多图或 `.cbz`（可选 `ref_files`）执行解构/翻译 |
| `POST /api/v1/jobs/translate` | `multipart/form-data` 异步提交批量上传或本地目录图片翻译任务，立即返回 `202 Accepted` 与 `job` ID |
| `POST /api/v1/jobs/source-translate` | 从已注册远程图源启动流式边下载边翻译任务 |
| `POST /api/v1/jobs/{id}/cancel` | 取消运行中的异步翻译任务 |
| `POST /api/v1/pages/{id}/translate` | 对指定页面重新执行翻译与嵌字 |
| `POST /api/v1/pages/{id}/translate-box` | 对用户在画布上手动框选的单个区域执行定向识别、去字、翻译与嵌字 |
| `POST /api/v1/pages/{id}/rerender` | 提交编辑后的 `Comic DOM` 节点文本/排版/框位置，跳过 OCR 与底图修复即时重排合成 |
| `DELETE /api/v1/pages/{id}` | 删除单页记录 |
| `PATCH /api/v1/works/{id}` | 重命名作品 |
| `DELETE /api/v1/works/{id}` | 级联删除整部作品及其所有页面、角色与术语记录 |
| `POST /api/v1/works/{id}/glossary` | 新增或更新作品级专有名词及振假名覆盖规则 |
| `POST /api/v1/works/{id}/characters` | 更新角色簇名称、第一人称与口癖风格 |
| `POST /api/v1/works/{id}/relations` | 更新角色间双向称谓与语域规则 |
| `POST /api/v1/sources` | 注册远程漫画图源 |
| `DELETE /api/v1/sources/{id}` | 删除已注册图源 |
| `POST /api/v1/sources/probe` | 探测远程图源章节与分页目录 |
