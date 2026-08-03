# M2：首轮排序与选择

## 目标

在 M1 的真实候选快照上实现多语言 embedding、Cross-Encoder 重排、分项评分、证据槽位分类和多样性选择，输出 15-20 篇首轮候选。

## 前置门禁

- M1 状态为 `COMPLETE`，M2 为 `READY`。
- 至少有一个带真实来源 ID 的冻结候选快照。

## 任务

### M2-T01：EmbeddingProvider 与冻结向量

实现可替换 provider、批处理、模型版本记录、输入文本拼接和向量缓存。确定性 fake 测排序逻辑，真实 BGE-M3 运行单独记录。

通过标准：同一输入/模型版本命中缓存；空摘要不伪造；真实向量维度和记录数一致。

### M2-T02：RerankerProvider

只对 embedding 前 50-100 篇精排，保存原始分数与归一化方法。

通过标准：批次切分不改变结果顺序；模型失败时不把未运行分数当 0 分继续静默排序。

### M2-T03：证据槽位分类

只根据论文标题和摘要进行分类，不得读取正文、引用次数、作者声誉、期刊等级、排序分数、用户反馈或网络补充内容。输出至少包括：

- evidence slot
- support level
- grounded reason
- supporting source text or excerpt
- source text hash
- classifier/prompt/schema version

固定槽位为 `PROBLEM_EXISTENCE`、`CURRENT_METHODS`、`METHOD_TRANSFERABILITY`、`IMPLEMENTATION_PATH` 和 `EVALUATION_BASIS`；固定支持级别为 `DIRECT`、`INDIRECT` 和 `HYPOTHETICAL`。未知槽位或支持级别必须拒绝，不得映射为默认项。

通过标准至少包括：

- 只能依据标题和摘要，不得从正文外推断结论。
- 理由超出来源文本时必须拒绝；无摘要时不得伪造摘要内容。
- 无摘要只能在标题足以支持时明确标记 title-only，摘录只能来自标题；否则拒绝。
- `INDIRECT` 必须明确说明该证据为间接迁移依据，仍需在目标问题中验证。
- deterministic fake、真实模型和人工抽查必须分开报告，不得把解释性输出直接当作来源依据。

### M2-T04：六分项评分

实现并分别保存：

- reranker score
- dense similarity
- concept match
- query branch score
- evidence slot score（由更早的 M2-T03 提供）
- metadata quality

通过标准保持：总分可由配置和分项精确重算；缺少任一必需分项时明确失败；权重、计算方法和模型/规则版本进入快照；不得使用隐藏默认值。

### M2-T05：多样性选择与首轮报告

实现贪心选择：高分优先、重复惩罚、未覆盖槽位奖励、问题域/方法域保底和有限不确定性探索。

通过标准：

- 结果数量为 15-20。
- 所有入选论文有真实来源和分项解释。
- 高度重复论文受到惩罚。
- 有可用候选时，问题域与方法域均至少入选 1 篇。
- 固定输入、配置和随机种子产生相同顺序。

## 阶段证据

`evaluation/reports/m2-validation.json` 区分 fake、真实模型、人工抽查和未运行项，并记录模型 revision、候选快照哈希、配置、运行时间和硬件。

## Codex 执行指令

```text
在当前仓库只执行 <任务 ID>，不得实现反馈或第二轮检索。
先用确定性 fake 添加失败测试，证明当前排序、分数或选择不满足任务不变量；
再实现最小逻辑。真实模型测试独立运行和报告，不得用 fake 通过声称 BGE 通过。
保留 M1 原始候选快照，不修改真实来源字段。
运行聚焦测试、M2 测试和完整回归，生成机器可读证据。
只在 5/5 任务和阶段标准全部通过时把 M2 标为 COMPLETE。
提交、推送并创建草稿 PR，报告模型版本、输入哈希、短/完整 SHA 和未运行项。
```
