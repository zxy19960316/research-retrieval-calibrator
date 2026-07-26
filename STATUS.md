# Project Status

更新日期：2026-07-26

## 当前门禁

- 当前阶段：`M0`
- 当前状态：`IN_PROGRESS`
- 当前入口：`docs/phases/M0-product-and-evaluation-contracts.md`
- 允许执行：仅 M0 范围与契约任务
- 禁止执行：M1-M6 实现、模型下载、平台部署、CNKI 自动化

## 阶段状态

| 阶段 | 状态 | 已通过任务 | 阶段证据 |
|---|---|---:|---|
| M0 产品与评测契约 | IN_PROGRESS | 2/4 | M0-T01 domain contracts and M0-T02 state-machine guard validated |
| M1 首轮真实召回 | BLOCKED_BY_M0 | 0/4 | 尚未生成 |
| M2 首轮排序与选择 | BLOCKED_BY_M1 | 0/5 | 尚未生成 |
| M3 反馈与第二轮校准 | BLOCKED_BY_M2 | 0/5 | 尚未生成 |
| M4 十题离线评测 | BLOCKED_BY_M3 | 0/4 | 尚未生成 |
| M5 OpenAI 兼容服务 | BLOCKED_BY_M4 | 0/4 | 尚未生成 |
| M6 CNKI、部署与演示 | BLOCKED_BY_M5 | 0/5 | 尚未生成 |

状态值只允许使用：`READY`、`IN_PROGRESS`、`BLOCKED_BY_M<n>`、`NO_GO`、`COMPLETE`。

## 基线变更记录

| 日期 | 版本 | 说明 |
|---|---|---|
| 2026-07-26 | v0.1 | 建立独立 RRC 仓库、风险优先阶段顺序和 M0-M6 执行门禁 |

## 下一动作

`M0-T01` 与 `M0-T02` 已完成：领域枚举、Pydantic 数据契约与状态机的先红后绿验证已通过。下一动作：`M0-T03`；不得在该任务开始前执行其他任务。
