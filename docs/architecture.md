# Architecture

## 1. 分层

```text
CLI / OpenAI-compatible API
            ↓
Conversation Router and Project State Machine
            ↓
Retrieval Orchestrator
            ↓
Source Adapters ── arXiv / user-provided CNKI records
            ↓
Paper Normalization and Deduplication
            ↓
Embedding / Reranking / Evidence Selection
            ↓
Feedback Calibration and Query Revision
            ↓
Project Store and Evaluation Reports
```

## 2. 边界

- `app/api/` 只做协议映射、鉴权和错误格式，不实现检索算法。
- `app/core/` 保存状态机、意图、查询规划、排序、反馈和选择的纯业务逻辑。
- `app/adapters/` 隔离 arXiv、导入格式、模型供应商和 LLM 供应商。
- `app/models/` 是跨模块数据契约。
- `app/storage/` 负责事务和持久化，不决定相关性。
- `evaluation/` 可在没有 Web 服务时直接运行核心算法。

## 3. 供应商接口

目标接口：

```python
class LLMProvider:
    async def generate_structured(self, request: StructuredRequest) -> dict: ...
    async def generate_text(self, request: TextRequest) -> str: ...

class EmbeddingProvider:
    def encode(self, texts: list[str]) -> list[list[float]]: ...

class RerankerProvider:
    def score(self, query: str, documents: list[str]) -> list[float]: ...

class PaperSource:
    async def search(self, query: SourceQuery) -> list[PaperRecord]: ...
```

业务测试使用确定性 fake；真实模型和真实网络测试单独标记。

## 4. 一致性与失败

- 项目状态迁移和对应数据写入同一事务。
- 外部请求使用稳定 `query_id` 实现幂等。
- 重试不重复创建论文或检索关系。
- 排序保存配置版本、模型版本和输入快照。
- 外部源失败时保留上一稳定状态，并记录可重试错误。
- 第二轮报告引用不可变的首轮快照。

## 5. 初始仓库结构

```text
app/
  api/
  core/
  adapters/
  models/
  storage/
  prompts/
evaluation/
  datasets/
  baselines/
  metrics/
  reports/
tests/
  unit/
  contract/
  integration/
scripts/
docker/
docs/
  phases/
  superpowers/plans/
```

目录在对应阶段需要时创建，M0 不提前搭建未使用的运行模块。
