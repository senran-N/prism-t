---
name: Bug Report
about: 反馈去字修复、排版渲染、语义编译或 Web Studio 中的缺陷
title: "[Bug] "
labels: bug
---

## 问题描述
请简要描述遇到的问题与预期行为。

## 复现步骤
1. 运行命令或 Web Studio 操作：
2. 输入样本特征（分辨率、版式、是否含双描边/网点/条漫切片）：
3. 使用的 LLM 端点类型、模型与参数（如 `PRISM_T_LLM_VISION`、`PRISM_T_LLM_REASONING_EFFORT`）：

## 诊断信息
- 操作系统与计算设备（CUDA / CPU）：
- `prismtd` / Python 算子日志（请务必移除 API Key 与个人路径）：
- 若涉及去字或合成误差，建议附上 `scripts/inspect_result.py` 摘要指标。
