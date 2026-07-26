# M1：首轮真实召回

## 目标

从研究问题生成四路线查询，通过 arXiv 获取真实元数据，标准化、去重并用 CLI 输出可追溯候选池。M1 不做 embedding、reranker 或第二轮反馈。

## 前置门禁

- M0 状态为 `COMPLETE`，M1 为 `READY`。
- M0 validation report schema 校验通过。

## 任务

### M1-T01：意图 IR、澄清与四路线查询规划

创建意图解析、缺失度计算、最多 3 个澄清问题和四分支三宽度查询规划器。测试必须证明：

- 明确字段不会被重复提问。
- 高歧义字段优先。
- 澄清问题最多 3 个。
- 每个查询关联 IR revision、分支、宽度和语言。
- 排除词不会被重新作为正向扩展词。

通过标准：固定输入产生确定性计划；LLM fake 返回无效结构时系统拒绝而非猜测。

### M1-T02：arXiv 适配器

实现分页、请求节流、缓存、超时、指数退避、相同查询去重和 Atom 解析。使用录制 fixture 覆盖正常、空结果、限流、服务器错误、畸形 Atom 和重复页。

通过标准：

- 单元测试不依赖网络。
- 真实 smoke test 单独标记。
- 失败重试不重复创建结果。
- User-Agent、超时和节流策略可配置。

### M1-T03：标准化与去重

实现 DOI、arXiv ID、标准化标题、标题相似度加作者重叠的分层去重；中英文疑似同文只标记人工检查。

通过标准：冻结 fixture 上没有错误自动合并；每个合并结果保留所有 retrieval paths。

### M1-T04：首轮 CLI 垂直切片

提供命令行入口：

```text
研究问题 → 意图 IR → 查询计划 → arXiv → 标准化去重 → 候选 JSON/Markdown
```

通过标准：

- 至少一个真实问题完成实时 arXiv smoke run。
- 用户可见候选真实来源 ID 与 URL 覆盖率 100%。
- Metadata Hallucination Rate 为 0。
- 相同查询回放命中缓存。
- 候选不足或 arXiv 不可用时返回诚实失败报告。

## 阶段证据

`evaluation/reports/m1-validation.json` 至少记录：

- mock/fixture 测试统计
- recorded arXiv 解析统计
- real arXiv 请求时间、查询、结果数和来源 ID 样本
- 缓存命中与限速证据
- 去重前后数量
- 网络未运行时的明确状态

## Codex 执行指令

```text
在当前仓库只执行 <任务 ID>，不得进入 M2。
先核对 M0 COMPLETE 和 M1 READY，再阅读本阶段文件。
先写失败测试并保存红灯；外部调用先使用 fixture/recording，
完成确定性测试后才运行独立的 real_external smoke test。
不得让 LLM 生成论文元数据，不得抓取 arXiv 网页，不得宣称引用网络。
运行聚焦测试、M1 测试和完整回归，更新机器可读证据与 STATUS.md。
提交、推送并创建草稿 PR，分别报告 automated、recorded_external、
real_external 和 not_run 结果。
```
