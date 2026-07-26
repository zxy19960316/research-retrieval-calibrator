# Research Retrieval Calibrator

跨学科文献检索校准智能体：通过可追溯的用户反馈，让第二轮文献检索稳定优于第一轮。

本仓库是独立的轻量比赛项目，不依赖 `adaptive-research-graph`。首版只围绕一条主链建设：

```text
研究问题
→ 多路线文献召回
→ 向量化、重排与多样性选择
→ 至少 6 条人工反馈
→ 查询校准与第二轮检索
→ 前后指标对比与检索演化报告
```

## 当前状态

当前处于 **M0：产品与评测契约冻结** 的可执行起点。实现者应先阅读：

- [`agent.md`](agent.md)：项目总览与 Codex 工作规则
- [`PRODUCT_SPEC.md`](PRODUCT_SPEC.md)：产品规格 v0.1
- [`PROJECT_PLAN.md`](PROJECT_PLAN.md)：唯一里程碑顺序
- [`STATUS.md`](STATUS.md)：当前门禁和证据状态
- [`docs/phases/`](docs/phases/)：M0-M6 独立阶段执行文件

## 核心原则

- 大模型理解、扩展和解释，不充当论文数据库或最终相关性裁判。
- 所有论文必须来自真实数据源或用户合法导入的题录；元数据幻觉率必须为 0。
- 每次查询变化都要能追溯到原始意图、用户反馈或明确的系统推断。
- 找不到足够论文时如实报告，不凑数。
- 先证明离线双轮闭环有效，再建设平台协议、CNKI 支线和比赛包装。

## 技术基线

- Python 3.12
- FastAPI、Pydantic、SQLAlchemy
- httpx、feedparser
- NumPy、scikit-learn
- sentence-transformers 或 FlagEmbedding
- PostgreSQL 用于正式部署；MVP 可先使用 SQLite
- Redis 只在缓存、限速或共享临时状态确有需要时引入

仓库公开可见不等于自动授予开源许可。在项目负责人选择许可证前，第三方不得假定拥有复制、修改或再分发权。
