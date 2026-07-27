# Project Status

更新日期：2026-07-27

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
| M1 首轮真实召回 | IN_PROGRESS | 3/4 | M1-T01 意图 IR、澄清问题与四路线查询规划、M1-T02 arXiv adapter contract，以及 M1-T03 确定性标准化和保守去重均已验证 |
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

M0-T01、M0-T02、M0-T03、M0-T03R 与 M0-T04 已完成：领域契约、状态机、十题评测模板与指标、负向抑制修正，以及机器可读的证据报告和阶段自校验均已通过。M1-T01、M1-T02 与 M1-T03 已完成：意图 IR、澄清问题、严格 LLM fake 边界、四路线三宽度查询规划、bounded arXiv adapter contract，以及离线的确定性标准化和保守去重均已验证。下一动作：`M1-T04`；其 real arXiv live-success gate 仍未满足，且在 M1-T03 PR 最终审查和合并前不开始实现。

## M1-T03 normalization and deduplication evidence

`evaluation/reports/m1-t03-normalization-dedup.json` records M1-T03R1 implementation commit `32bad697f626cfc1d1ad73fb697a9e62a36fe74f`, a 34-test focused suite, and a 247-test full regression. It proves recorded adapter-to-dedup integration, same-`paper_id` retrieval-path coalescing, stable `DUPLICATE_PAPER_ID_CONFLICT` failures, globally unique cluster IDs, and title/author transitive-bridge blocks, while retaining frozen negative `false_auto_merge_count = 0`, complete retrieval-path preservation, input-order invariance, and idempotence. This task is deterministic and offline; no arXiv smoke was run. M1 is `IN_PROGRESS 3/4` on this PR branch; M1-T04 is `NOT_STARTED`, its live-success gate remains unsatisfied, M2 remains `BLOCKED_BY_M1`, and main remains M1 `2/4` until PR #8 is merged.

## M1-T02 arXiv adapter evidence

M1-T02R3 adds canonical legacy arXiv identifier compatibility: optional uppercase two-letter subject classes such as `math.GT`, `cs.SE`, and `nlin.CD` parse without casefolding; only `v1+` is stripped and `v0` remains rejected. The 47 offline adapter tests and 213-test full regression passed. Transport, total budgets, Retry-After scheduling, Atom feed-root validation, cache provenance, smoke argument JSON, and external evidence were not changed. M1-T02 adapter contract complete. The previously attempted `real_external` smoke failed honestly with `attempt_count = 3`, `http_status = null`, and `ARXIV_TRANSPORT_ERROR`; its timestamp and observation were preserved, no extra diagnostic or smoke was run, and causal link remains `not_established`. M1-T04 live-success acceptance gate remains unsatisfied. M1 is `IN_PROGRESS 2/4`; M1-T03 is the next action but is not started here, and M2 remains `BLOCKED_BY_M1`.

## M1-T01R3 repair evidence

PR #6 contains the M1-T01R3 breadth and exclusion closure evidence: each of the four branches has serializable NARROW/MEDIUM/WIDE required-group expressions, with WIDE strictly using fewer mandatory groups than MEDIUM while retaining the branch anchor. Former `framework`, `application`, `methodology`, and `benchmark` WIDE-only mandatory terms are removed; their individual exclusions preserve a complete twelve-query plan. Exclusions retain original audit text and canonical English comparison values, duplicate canonical exclusions are rejected, and exhausted bridge vocabulary reports `INVALID_QUERY_PLAN` with a stable reason. M1 is now `IN_PROGRESS 2/4`; M1-T02 is complete, while M1-T03 remains outside this change scope.
