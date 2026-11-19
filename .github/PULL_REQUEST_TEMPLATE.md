## 变更说明
<!-- 请简述本次 PR 解决的问题及核心改动 -->

## 验证清单
- [ ] `CGO_ENABLED=0 go test -count=1 ./...` 通过
- [ ] `uv --directory engine run pytest -v` 通过
- [ ] `npm --prefix web run build` 通过
- [ ] 未提交任何内部文档、运行数据、密钥或本地测试媒体文件
- [ ] 如变更 CLI / 环境变量 / API，已同步更新 `README.md`、`README_EN.md` 与 `docs/`
