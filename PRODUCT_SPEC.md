# Research Retrieval Calibrator 产品规格 v0.1

## 1. 产品定义

Research Retrieval Calibrator 是一个反馈驱动的双轮文献检索系统。用户用自然语言描述跨学科研究问题，系统在真实文献源中形成首轮候选；用户标注至少 6 篇后，系统把反馈转化为可审计的查询变化，重新检索并比较前后质量。

目标用户是需要寻找直接论文、问题域论文、方法域论文和桥接论文的科研人员。首版比赛场景以核工程或辐射屏蔽交叉问题为深度案例，同时用另一类理工交叉问题验证泛化。

## 2. 成功定义

首版成功必须同时满足：

- 10 个真实问题中，第二轮至少在 7 个问题上提高 Inclusive Precision@10。
- 第二轮平均 Inclusive Precision@10 比第一轮至少提高 `0.05` 绝对值。
- 第二轮平均 NDCG@10 和 Evidence Coverage 均不低于第一轮。
- Metadata Hallucination Rate 为 `0`。
- 每个第二轮新增/删除术语、约束和路线预算变化均可追溯。
- 系统在候选不足时明确报告缺口，不凑数。

这些是 v0.1 的预注册 Go 阈值。首次正式 M4 标注开始后，阈值变更必须创建新的评测版本，不得覆盖旧结果。

## 3. 用户工作流

### 3.1 创建项目

输入支持：

- 研究方向或研究问题
- 已失败的关键词
- 可选种子论文
- 可选排除范围
- 中文或英文描述

系统返回稳定的 `project_id`，并在服务端保存状态。

### 3.2 澄清

系统先解析意图字段的明确程度，只针对信息缺口最大的字段提出最多 3 个问题。不得用开放式闲聊替代结构化澄清。

### 3.3 冻结 Research Intent IR

IR 至少包含：

- 中英文对象词
- 中英文任务词
- 中英文方法词
- 中英文物理或应用范围词
- 排除词
- 方法是否为硬约束
- 可接受论文角色

任何后续变化都记录变化内容、原因、来源和确认类型。

### 3.4 首轮查询

固定四个查询分支：

1. `DIRECT_INTERSECTION`：直接交叉
2. `PROBLEM_DOMAIN`：问题域
3. `METHOD_DOMAIN`：方法域
4. `BRIDGE_DOMAIN`：桥接域

每个分支可生成 `NARROW`、`MEDIUM`、`WIDE` 查询。首轮合并候选池目标为 100-300 篇，避免无上限抓取。

### 3.5 首轮排序

两阶段排序：

1. 多语言 bi-encoder 产生快速语义分数。
2. Cross-Encoder 只重排前 50-100 篇。

初始候选模型为 `BAAI/bge-m3` 和 `BAAI/bge-reranker-v2-m3`，但模型名必须配置化并通过离线评测决定是否保留。

首轮分数分项保存：

```text
0.35 × reranker_score
+ 0.25 × dense_similarity
+ 0.15 × concept_match
+ 0.10 × query_branch_score
+ 0.10 × evidence_slot_score
+ 0.05 × metadata_quality
```

权重是 v0.1 baseline，不是不可修改的产品常数。

### 3.6 首轮选择

输出 15-20 篇，兼顾：

- 相关性
- 查询路线多样性
- 论文去重
- 证据槽位覆盖
- 少量高不确定潜力候选

用户至少标注 6 篇后才允许进入第二轮。

### 3.7 反馈

相关性枚举：

- `HIGH`
- `PARTIAL`
- `IRRELEVANT`

部分相关可选择：

- `OBJECT`
- `TASK`
- `METHOD`
- `PHYSICS`
- `BRIDGE`
- `EVALUATION`

部分相关论文只在用户选定维度上贡献正反馈。

### 3.8 第二轮校准

语义向量 baseline：

```text
q2 = 1.00 × original_intent
   + 0.75 × high_relevance_centroid
   + 0.35 × partial_selected_dimension_centroid
   - 0.15 × irrelevant_centroid
```

同时执行：

- 从高相关论文中提升区分性术语。
- 从不相关论文中降低偏移术语。
- 根据各查询分支的高相关率调整第二轮预算。
- 保留低表现分支的最小探索预算，不彻底关闭。
- 为每个改写后的查询记录反馈来源。

### 3.9 最终组合

目标最多 10 篇：

- 保留 3-5 篇首轮高相关论文。
- 加入 5-7 篇第二轮新论文。
- 最大化相关性和证据槽位覆盖。
- 保持问题域与方法域多样性。
- 抑制重复和负反馈方向。

候选不足时返回实际数量、缺失槽位和原因。

## 4. 数据源

### 4.1 arXiv

- 使用公开 API 返回的真实元数据。
- 支持分页、缓存、限速、重试和相同查询去重。
- 真实请求的节流策略必须以实现时核对的官方要求为准。
- 不承诺完整引用网络。

### 4.2 CNKI

首版只支持：

```text
系统生成中文检索式
→ 用户通过合法权限检索并导出题录
→ 用户上传或粘贴
→ 系统解析、标准化、向量化和重排
```

支持目标格式：RIS、EndNote、NoteExpress、CSV、纯文本题录及标题摘要列表。

## 5. 论文真实性与去重

所有来源统一为 `PaperRecord`。去重顺序：

1. DOI 完全一致
2. arXiv ID 一致
3. 标准化标题完全一致
4. 标题高相似度且作者重叠
5. 中英文疑似同文只标记人工检查

每篇输出必须至少有 `source`、`source_id`、`title`、`url`。无法从真实来源确认的记录不得进入用户结果。

## 6. 证据槽位

固定槽位：

- `PROBLEM_EXISTENCE`
- `CURRENT_METHODS`
- `METHOD_TRANSFERABILITY`
- `IMPLEMENTATION_PATH`
- `EVALUATION_BASIS`

支持级别：

- `DIRECT`
- `INDIRECT`
- `HYPOTHETICAL`

模型只能基于标题和摘要生成分类理由。间接迁移证据必须明确标注“仍需验证”。

## 7. 项目状态机

```text
INIT
→ CLARIFYING
→ ROUND1_SEARCHING
→ WAITING_FOR_FEEDBACK
→ ROUND2_SEARCHING
→ WAITING_FOR_DIRECTION_CONFIRMATION
→ FINALIZED
```

非法跳转必须拒绝并返回当前允许动作。

## 8. 平台接口

通过核心评测后提供：

- `GET /v1/models`
- `POST /v1/chat/completions`
- 流式和非流式响应
- Bearer 鉴权
- `project_id` 关联的服务端状态

核心业务依赖 `LLMProvider` 抽象，不绑定单一模型供应商。

## 9. 非功能要求

- 可复现：同一数据快照、配置和随机种子产生一致排序。
- 可审计：查询、反馈、分数、选择和版本全链记录。
- 可恢复：外部源超时或模型失败不损坏项目状态。
- 安全：密钥不入库，日志不记录 Bearer token 或用户 Cookie。
- 合规：只访问允许的公开接口和用户合法提供的数据。
- 诚实：实时、录制、mock、人工标注和未运行证据分开报告。

## 10. 明确不做

- 完整知识图谱和长期记忆
- Scheme 自动生成
- 动态 Schema 演化
- 复杂全文解析器竞赛
- arXiv 引文网络承诺
- 自动化知网抓取或登录
- 在核心评测通过前制作完整聊天前端
