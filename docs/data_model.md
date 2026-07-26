# Data Model

## 1. 核心枚举

```text
ProjectStage:
  INIT
  CLARIFYING
  ROUND1_SEARCHING
  WAITING_FOR_FEEDBACK
  ROUND2_SEARCHING
  WAITING_FOR_DIRECTION_CONFIRMATION
  FINALIZED

QueryBranch:
  DIRECT_INTERSECTION
  PROBLEM_DOMAIN
  METHOD_DOMAIN
  BRIDGE_DOMAIN

QueryBreadth:
  NARROW
  MEDIUM
  WIDE

Relevance:
  HIGH
  PARTIAL
  IRRELEVANT

FeedbackAspect:
  OBJECT
  TASK
  METHOD
  PHYSICS
  BRIDGE
  EVALUATION

EvidenceSlot:
  PROBLEM_EXISTENCE
  CURRENT_METHODS
  METHOD_TRANSFERABILITY
  IMPLEMENTATION_PATH
  EVALUATION_BASIS

SupportLevel:
  DIRECT
  INDIRECT
  HYPOTHETICAL
```

## 2. 领域对象

### RetrievalProject

- `project_id`: `RRC-YYYY-NNNN` 格式
- `stage`: `ProjectStage`
- `round_number`: `0 | 1 | 2`
- `original_input`: 非空字符串
- `current_intent`: `ResearchIntent | null`
- `created_at`, `updated_at`: UTC 时间

### ResearchIntent

- `object_terms`, `task_terms`, `method_terms`, `scope_terms`: 中英文词列表
- `exclusions`: 中英文排除词列表
- `method_constraint`: `REQUIRED | PREFERRED | OPEN`
- `accepted_paper_roles`: 非空角色集合
- `revision`: 从 1 开始递增
- `frozen_at`: UTC 时间

### QueryPlan / Query

- 轮次、分支、宽度、语言、查询文本、预算权重
- 每个查询有稳定 `query_id`
- 第二轮查询必须关联至少一个 `QueryRevision`

### PaperRecord

- 真实 `source` 和 `source_id`
- 标题、摘要、作者、年份、DOI、URL、语言、分类
- 一个或多个 `retrieval_paths`
- 缺失摘要允许保留在原始池，但不得伪造摘要

### FeedbackRecord

- 项目、论文、相关性、选定维度、原始用户文本、时间
- `PARTIAL` 必须至少有一个维度
- `HIGH` 和 `IRRELEVANT` 不接受维度化正样本语义

### QueryRevision

- `from_round=1`、`to_round=2`
- 提升/降低术语
- 移除/放宽约束
- 分支预算变化
- 每项变化的来源论文或系统规则

### RankedPaper

- 论文 ID、轮次、分项分数、总分、模型/配置版本
- 证据槽位、支持级别、基于标题摘要的理由
- 选择或未选择原因

## 3. 最小持久化表

- `retrieval_projects`
- `query_rounds`
- `queries`
- `papers`
- `paper_retrievals`
- `feedback`
- `query_revisions`
- `ranked_papers`
- `evaluation_runs`

## 4. 跨记录不变量

- 项目阶段与轮次一致。
- 一个项目最多有一份活动的意图 revision。
- 第一轮查询不能引用反馈。
- 第二轮查询必须引用至少一条变化来源。
- 反馈论文必须出现在该项目首轮候选中。
- `PARTIAL` 反馈必须有 aspect；其他相关性不得伪装成部分正样本。
- 所有用户可见论文必须有真实来源标识。
- 总分必须能由同一配置版本的分项分数重算。
- 最终选择不得超过 10 篇，首轮选择必须在 15-20 篇范围。
