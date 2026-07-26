# 阶段执行索引

## 使用方式

每次新 Codex 会话只选择 `STATUS.md` 指定的一个任务。开始前复制当前阶段文件中的“Codex 执行指令”，把 `<任务 ID>`替换为实际任务，例如 `M0-T01`。执行者必须在任务结束时提交测试、证据和状态更新，不能只给建议。

## 阶段

| 阶段 | 文件 | 核心输出 |
|---|---|---|
| M0 | [M0-product-and-evaluation-contracts.md](M0-product-and-evaluation-contracts.md) | 数据、状态机、评测和证据契约 |
| M1 | [M1-real-first-round-retrieval.md](M1-real-first-round-retrieval.md) | arXiv 首轮真实召回 |
| M2 | [M2-ranking-and-round1-selection.md](M2-ranking-and-round1-selection.md) | 首轮排序与 15-20 篇选择 |
| M3 | [M3-feedback-and-round2-calibration.md](M3-feedback-and-round2-calibration.md) | 反馈驱动第二轮检索 |
| M4 | [M4-ten-question-evaluation.md](M4-ten-question-evaluation.md) | 十题 Go/No-Go 评测 |
| M5 | [M5-openai-compatible-service.md](M5-openai-compatible-service.md) | OpenAI 兼容服务与平台探测 |
| M6 | [M6-cnki-deployment-and-demo.md](M6-cnki-deployment-and-demo.md) | 合规中文导入、部署和比赛演示 |

## 通用门禁

任何阶段都必须：

- 先执行失败验证，再做最小实现。
- 使用确定性 fixture 测核心逻辑。
- 把 mock、录制外部响应、真实外部调用和人工判断分开报告。
- 运行阶段测试和完整回归。
- 更新 `STATUS.md`，但只在全部硬标准通过后解锁下一阶段。
- 保留失败证据；不得用叙述替代机器可读报告。
