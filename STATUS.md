# Project Status

更新日期：2026-07-26

## 当前门禁

- 当前阶段：`M1`
- 当前状态：`IN_PROGRESS`
- 当前入口：`docs/phases/M1-real-first-round-retrieval.md`
- 允许执行：仅 M1 范围内且从 M1-T02 开始的任务
- 禁止执行：M2-M6 实现、模型下载、平台部署、CNKI 自动化

## 阶段状态

| 阶段 | 状态 | 已通过任务 | 阶段证据 |
|---|---|---:|---|
| M0 产品与评测契约 | COMPLETE | 4/4 | M0-T01 domain contracts, M0-T02 state-machine guard, M0-T03 evaluation contracts, and M0-T04 evidence gate validated |
| M1 首轮真实召回 | IN_PROGRESS | 1/4 | M1-T01 意图 IR、澄清问题与四路线查询规划已验证 |
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

M0-T01、M0-T02、M0-T03、M0-T03R 与 M0-T04 已完成：领域契约、状态机、十题评测模板与指标、负向抑制修正，以及机器可读的证据报告和阶段自校验均已通过。M1-T01 已完成：意图 IR、澄清问题、严格 LLM fake 边界和四路线三宽度查询规划均已验证。下一动作：`M1-T02`；继续遵守 M1 的真实来源与证据边界。

## M1-T01R3 repair evidence

PR #6 now contains the M1-T01R3 breadth and exclusion closure evidence: each of the four branches has serializable NARROW/MEDIUM/WIDE required-group expressions, with WIDE strictly using fewer mandatory groups than MEDIUM while retaining the branch anchor. Former `framework`, `application`, `methodology`, and `benchmark` WIDE-only mandatory terms are removed; their individual exclusions preserve a complete twelve-query plan. Exclusions retain original audit text and canonical English comparison values, duplicate canonical exclusions are rejected, and exhausted bridge vocabulary reports `INVALID_QUERY_PLAN` with a stable reason. M1 remains `IN_PROGRESS 1/4`; M1-T02 is explicitly not started and outside this repair scope.
