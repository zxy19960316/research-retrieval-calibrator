# Evaluation Contract

## 1. 数据集

M4 冻结 10 个真实问题：

- 5 个核工程或辐射屏蔽交叉问题
- 5 个另一理工交叉领域问题

每个问题必须包含问题文本、范围、排除项、判断指南、判断人、判断版本和来源池快照。M0 只建立可校验模板，M4 才冻结人工标签。

## 2. 四组比较

1. 用户原始查询
2. 通用大模型一次性生成查询
3. RRC 第一轮
4. RRC 第二轮

所有组使用相同来源快照或记录清楚时间差，不能把数据源变化误当算法改善。

## 3. 指标

- `Strict Precision@10`：只有 `HIGH` 计 1。
- `Inclusive Precision@10`：`HIGH` 和 `PARTIAL` 计 1。
- `NDCG@10`：`HIGH=2`、`PARTIAL=1`、`IRRELEVANT=0`。
- `Evidence Coverage`：5 个固定槽位中至少被一篇有效论文覆盖的比例。
- `Negative Suppression`：负反馈方向在第二轮前 10 中的暴露下降。
- `New Useful Papers`：第二轮前 10 中未在首轮前 20 且被判断为 HIGH/PARTIAL 的数量。
- `Metadata Hallucination Rate`：无真实来源确认的用户可见记录数 / 用户可见记录总数。

## 4. M3 单问题门禁

必须同时满足：

- 至少 6 条人工反馈，且包含正、部分或负信号中的至少两类。
- 第二轮 Inclusive Precision@10 高于第一轮。
- 第二轮 NDCG@10 不低于第一轮。
- Evidence Coverage 不下降。
- 至少 1 篇 `New Useful Paper`。
- Metadata Hallucination Rate 为 0。
- 所有查询变化都能追溯。

不满足时状态为 `NO_GO`，返回 M2 或 M3 的明确任务，不进入十题评测。

## 5. M4 Go 门禁

必须同时满足：

- 至少 7/10 问题的第二轮 Inclusive Precision@10 高于第一轮。
- 平均 Inclusive Precision@10 绝对提升至少 0.05。
- 平均 NDCG@10 不下降。
- 平均 Evidence Coverage 不下降。
- Metadata Hallucination Rate 为 0。
- 10/10 问题的查询变化可追溯。
- 候选不足案例能报告真实数量和缺失槽位。

## 6. 证据类型

报告中的每个检查必须标注：

- `automated`
- `mock`
- `recorded_external`
- `real_external`
- `human_judged`
- `not_run`

这些类别不得合并为一个“通过”数字。

## 7. 防止评测污染

- Go 阈值先于正式标签冻结。
- 模型和权重调优只使用开发集或交叉验证，不直接针对最终 10 题逐题手调。
- 每次评测保存配置、代码 commit、数据快照哈希和随机种子。
- 失败结果保留，不覆盖为后续成功报告。
