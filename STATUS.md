# Project Status

更新日期：2026-07-27

## 当前门禁

- 当前阶段：`M2`
- 当前状态：`READY`
- 当前入口：`docs/phases/M2-ranking-and-round1-selection.md`
- 允许执行：仅 M2 范围内且从 M2-T01 开始的任务
- 禁止执行：M3-M6 实现、模型下载、平台部署、CNKI 自动化

## 阶段状态

| 阶段 | 状态 | 已通过任务 | 阶段证据 |
|---|---|---:|---|
| M0 产品与评测契约 | COMPLETE | 4/4 | M0-T01 domain contracts, M0-T02 state-machine guard, M0-T03 evaluation contracts, and M0-T04 evidence gate validated |
| M1 首轮真实召回 | COMPLETE | 4/4 | M1-T01 意图 IR、澄清问题与四路线查询规划、M1-T02 arXiv adapter contract、M1-T03 确定性标准化和保守去重，以及 M1-T04 real arXiv CLI retrieval、缓存回放和证据均已验证 |
| M2 首轮排序与选择 | READY | 0/5 | M1 已完成；可从 M2-T01 开始 |
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

M0-T01、M0-T02、M0-T03、M0-T03R 与 M0-T04 已完成：领域契约、状态机、十题评测模板与指标、负向抑制修正，以及机器可读的证据报告和阶段自校验均已通过。M1-T01 至 M1-T04 已完成：M1-T04R1 的 fresh-cache real arXiv smoke 返回 12 个 HTTP 200，产生 51 条原始记录和 33 条去重候选，source-ID/URL coverage 均为 100%，metadata projection mismatch count 为 0、Metadata Hallucination Rate 为 0；真实模式最小请求间隔为 3.0 秒，观测到的最小请求起始间隔为 4.016 秒，累计 11 次等待共 32.984 秒；同配置 real-cache 回放产生 0 次传输、12 次缓存命中且候选一致。前两次 real 尝试各有一次传输失败，未被作为完成证据。下一动作：`M2-T01`；本次变更不开始 M2 实现。

## M1-T03 normalization and deduplication evidence

`evaluation/reports/m1-t03-normalization-dedup.json` records M1-T03R2 implementation commit `fa2c9151e34ad4bb06716eebdebe933b5e71a2f8`, a 38-test focused suite, and a 251-test full regression. It proves immutable all-pair base classification before union, complete-link closure for three-record exact-title and high-similarity cliques, retained `2+1` `TRANSITIVE_BRIDGE_RISK` blocks for genuine bridges, and deterministic raw-representative selection for normalization-equivalent repeated observations. It retains same-`paper_id` retrieval-path coalescing, stable `DUPLICATE_PAPER_ID_CONFLICT` failures, globally unique cluster IDs, frozen negative `false_auto_merge_count = 0`, complete retrieval-path preservation, input-order invariance, and idempotence. This task is deterministic and offline; no arXiv smoke was run. Its historical phase state is retained in the corresponding evidence JSON; the current phase table above is authoritative.

## M1-T02 arXiv adapter evidence

M1-T02R3 adds canonical legacy arXiv identifier compatibility: optional uppercase two-letter subject classes such as `math.GT`, `cs.SE`, and `nlin.CD` parse without casefolding; only `v1+` is stripped and `v0` remains rejected. The 47 offline adapter tests and 213-test full regression passed. Transport, total budgets, Retry-After scheduling, Atom feed-root validation, cache provenance, smoke argument JSON, and external evidence were not changed. M1-T02 adapter contract complete. Its earlier external observations and historical phase state remain recorded in the corresponding evidence JSON; the current phase table above is authoritative.

## M1-T01R3 repair evidence

PR #6 contains the M1-T01R3 breadth and exclusion closure evidence: each of the four branches has serializable NARROW/MEDIUM/WIDE required-group expressions, with WIDE strictly using fewer mandatory groups than MEDIUM while retaining the branch anchor. Former `framework`, `application`, `methodology`, and `benchmark` WIDE-only mandatory terms are removed; their individual exclusions preserve a complete twelve-query plan. Exclusions retain original audit text and canonical English comparison values, duplicate canonical exclusions are rejected, and exhausted bridge vocabulary reports `INVALID_QUERY_PLAN` with a stable reason. Its historical phase state remains recorded in the corresponding evidence JSON; the current phase table above is authoritative.
