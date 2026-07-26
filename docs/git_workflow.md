# Git Workflow

## 1. 分支

- 默认分支：`main`
- 任务分支：`agent/m<阶段>-t<任务号>-<短名>`
- 文档基线：`docs/<短名>`
- 紧急修复：`fix/<短名>`

从最新 `main` 创建分支。一个分支只完成一个可独立审查的任务。

## 2. 提交

提交格式：

```text
<type>: <imperative summary>
```

允许的 type：`docs`、`test`、`feat`、`fix`、`refactor`、`chore`、`eval`。

每个实现任务优先形成：

1. 能证明缺口的测试或 fixture 提交
2. 最小实现提交
3. 必要的证据与状态提交

不 amend、不 rebase 已共享分支、不强制推送。

## 3. Pull Request

PR 描述至少包含：

- 阶段和任务 ID
- 范围内与范围外
- 红灯命令及失败摘要
- 绿灯命令及通过摘要
- 真实外部证据状态
- 数据/模型/接口风险
- 回滚方式

合并使用 squash 或普通 merge 由仓库维护者决定；不得为了整理历史重写已发布 commit。

## 4. 阶段完成

阶段最后一个 PR 必须同步：

- `STATUS.md`
- 阶段证据索引
- 依赖锁文件
- 用户可见文档
- 下一阶段入口

合并后验证本地 `main`、远端 `main` 和阶段标签指向同一目标提交。

## 5. 禁止提交

- API key、Bearer token、Cookie、`.env`
- 用户受限题录原文件，除非确认可公开
- 模型权重、运行缓存、SQLite 运行库
- 未确认再分发许可的论文 PDF
- 将 mock 或录制数据伪装成实时结果的报告
