# 安全说明与漏洞反馈 (Security Policy)

本文档说明 Prism-T 的安全边界、默认防护策略以及安全问题反馈方式。

## 1. 安全模型与信任边界

### 管理接口鉴权
- 所有写入、翻译任务触发、页面重排、作品删除、人设/术语修改以及运行时配置读写接口均采取默认拒绝（Fail-Closed）策略，要求请求携带有效管理员令牌：
  - `X-Prism-Admin-Token: <token>`，或
  - `Authorization: Bearer <token>`
- 服务端使用常量时间比对（`crypto/subtle.ConstantTimeCompare`）校验令牌。
- 若未通过环境变量 `PRISM_T_ADMIN_TOKEN` 显式指定，`prismtd serve` 启动时会在数据目录生成随机令牌并写入 `data/admin.key`（文件权限 `0600`）。

### LLM 密钥与本地配置
- LLM API 密钥仅从环境变量或本地 `data/settings.json` 读取，不写入 SQLite 数据库或导出的 `.psd` / `.cbz` 文件。
- 切勿将 `data/admin.key`、`data/settings.json`、`.env` 或包含个人模型凭据的本地脚本提交至版本控制系统。

### 进程隔离与本地算子边界
- `prism-t-engine` 默认仅绑定本地回环地址 `127.0.0.1:18775`，仅接受本机 `prismtd` 进程调用，不直接对外暴露端口，也不主动发起外部 LLM 请求。
- 远程图源拉取与封面代理仅对已鉴权管理员配置的来源生效。

### 跨域资源共享（CORS）
- 默认同源策略（`PRISM_T_CORS_ORIGINS` 为空时不返回跨域放行头）。仅在显式配置白名单源时按源精确放行。

## 2. 部署建议

1. 如需在局域网或公网访问 Web Studio，请将 `prismtd` 置于具备 TLS 终止与访问控制的反向代理（如 Caddy 或 Nginx）之后，并设置强随机 `PRISM_T_ADMIN_TOKEN`。
2. 保持 `prism-t-engine` 仅监听 `127.0.0.1`。
3. 如无需跨站调用 API，保持 `PRISM_T_CORS_ORIGINS` 为空。

## 3. 报告安全问题

如发现潜在的安全漏洞（例如鉴权绕过、路径遍历、任意文件读取或 SSRF），请勿在公开 Issue 中披露利用细节。请通过 GitHub 仓库的 [Private Vulnerability Reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability) 提交复现步骤与影响范围说明。
