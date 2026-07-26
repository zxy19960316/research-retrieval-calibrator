# M0：产品与评测契约

## 目标

把产品规格转成可执行、可校验的数据契约和阶段门禁。M0 不调用 arXiv、不下载模型、不启动 Web 服务。

## 前置条件

- `STATUS.md` 当前阶段为 `M0`。
- 已阅读 `agent.md`、`PRODUCT_SPEC.md`、`docs/data_model.md` 和 `docs/evaluation.md`。

## 任务

### M0-T01：领域枚举与数据契约

本任务只交付 `Query` 和 `QueryRevision` 的最小溯源接口；完整 `QueryPlan` 属于 M1、`RankedPaper` 属于 M2、完整 `QueryRevision` 属于 M3。这些后续对象不在 M0-T01 实现范围内。

创建：

- `pyproject.toml`
- `app/models/enums.py`
- `app/models/project.py`
- `app/models/paper.py`
- `app/models/feedback.py`
- `app/models/query.py`
- `tests/contract/test_domain_models.py`

先用无效 fixture 证明以下不变量会失败，再实现 Pydantic 模型：

- 非法阶段、查询分支、证据槽位和支持级别被拒绝。
- `PARTIAL` 反馈没有 aspect 被拒绝。
- 用户可见论文缺少真实 `source_id` 或 URL 被拒绝。
- 第二轮查询没有变化来源被拒绝。
- 首轮选择数量不在 15-20，最终选择超过 10 被拒绝。

通过标准：契约测试全部通过，模型 JSON Schema 可导出且无不稳定字段。

### M0-T02：状态机

创建：

- `app/core/state_machine.py`
- `tests/unit/test_state_machine.py`

只允许：

```text
INIT
→ CLARIFYING
→ ROUND1_SEARCHING
→ WAITING_FOR_FEEDBACK
→ ROUND2_SEARCHING
→ WAITING_FOR_DIRECTION_CONFIRMATION
→ FINALIZED
```

失败测试至少覆盖跳过澄清、反馈不足进入第二轮、从 FINALIZED 回退、并发重复迁移。

通过标准：合法迁移返回明确新状态；非法迁移不修改原状态并返回稳定错误码。

### M0-T03：十题评测模板与指标契约

创建：

- `evaluation/datasets/questions.v0.1.yaml`
- `evaluation/datasets/questions.schema.json`
- `evaluation/metrics/retrieval.py`
- `tests/contract/test_evaluation_dataset.py`
- `tests/unit/test_retrieval_metrics.py`

模板必须包含 10 个稳定 question ID，其中 5 个标记 `nuclear_engineering`，5 个标记 `cross_domain_stem`。每项包含中英文问题、范围、排除项、判断指南和 `label_status: unjudged`。`unjudged` 是 M0 的合法状态，M4 冻结后不得保留。

指标测试使用手算小样本验证 Strict/Inclusive Precision@10、NDCG@10、Evidence Coverage、Negative Suppression、New Useful Papers 和 Metadata Hallucination Rate。

通过标准：10 题结构校验通过，所有指标与手算结果一致。

### M0-T04：证据报告与阶段自校验

创建：

- `evaluation/report.schema.json`
- `scripts/validate_phase.py`
- `tests/contract/test_evidence_report.py`
- `evaluation/reports/m0-validation.json`

报告必须区分 `automated`、`mock`、`recorded_external`、`real_external`、`human_judged` 和 `not_run`，记录 commit、命令、退出码、版本与输入哈希。

通过标准：

- `python scripts/validate_phase.py M0` 返回 0。
- 完整测试通过。
- 报告明确声明 arXiv、真实模型、平台和 CNKI 均为 `not_run`。
- `STATUS.md` 只有在报告验证成功后把 M0 改为 `COMPLETE`，M1 改为 `READY`。

## 阶段验收

- 4/4 任务完成。
- JSON Schema 与 Pydantic 模型语义一致。
- 状态机非法跳转有测试。
- 10 题模板可校验，指标有手算测试。
- M0 证据报告通过自身 schema 校验。
- 没有任何真实网络或模型成功的虚假声明。

## Codex 执行指令

```text
在当前仓库只执行 <任务 ID>。
先阅读 AGENTS.md、agent.md、STATUS.md、PROJECT_PLAN.md、
docs/phases/M0-product-and-evaluation-contracts.md 和该任务引用的文件。
从最新 main 创建 agent/<任务 ID 小写>-<短名> 分支。
先添加能证明目标不变量缺失的失败测试并运行，保存真实红灯结果；
再做最小实现，运行聚焦测试、M0 全部测试和完整回归。
生成或更新机器可读证据，明确 automated/mock/recorded_external/
real_external/human_judged/not_run。
只在通过标准全部满足时更新 STATUS.md。
提交、推送并创建草稿 PR；报告分支、短 SHA、完整 SHA、测试统计和未运行项。
不要开始 M1，不要下载模型，不要调用外部数据源。
```
