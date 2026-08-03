# M2 Task Order and Status Handoff Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair the dynamic status handoff and make evidence-slot classification precede the M2 six-factor scoring task.

**Architecture:** Keep STATUS.md as the sole live-state source. Extend the existing offline documentation validator with pure text checks for delegation, next-task ownership, M2 task order, the T03-to-T04 dependency, and the fixed 2/5 handoff; update only the allowed documentation and test files.

**Tech Stack:** Markdown, Python 3.12, pytest, the existing project-doc validator, Git, and GitHub Actions.

## Global Constraints

- Only execute PR #11 post-merge acceptance and the M2 documentation baseline repair.
- Do not implement M2-T03 business logic, M2-T04 scoring, or M2-T05 selection.
- Do not run model downloads, model inference, M2-T02 replay, or new external searches.
- Preserve M2 IN_PROGRESS 2/5, M3 BLOCKED_BY_M2 0/5, and M2-T03 as not started.
- Modify only agent.md, STATUS.md, docs/phases/M2-ranking-and-round1-selection.md, scripts/validate_project_docs.py, related documentation tests, and this plan.
- Preserve all M2-T01/M2-T02 evidence and the known candidate/result/report SHA-256 values.
- Tests must demonstrate the documentation contract failure before the implementation and document repair make it pass.

---

### Task 1: Add red documentation-contract tests

**Files:**
- Modify: tests/contract/test_project_docs_validation.py

**Interfaces:**
- Consumes: validator functions for dynamic delegation, status handoff, and the M2 contract.
- Produces: Regression tests for all seven requested invariants.

- [ ] **Step 1: Add a future-state dynamic-status regression**

Read README.md and agent.md, call validate_dynamic_status_delegation, and assert no errors for the repaired documents. Replace the agent delegation sentence with the future-state text 当前活动阶段为 M1。 and assert an error containing agent.md and dynamic phase. Replace the README delegation sentence with 当前阶段为 M1。 and assert an error requiring STATUS.md. This proves the test is not a one-off M0 ban.

- [ ] **Step 2: Add next-task and M2-order regressions**

Call validate_status_handoff with STATUS.md and the M2 phase text and assert no errors. Change the explicit next-task marker to M1-T01 and assert rejection. Call validate_m2_task_contract and assert that it accepts the repaired T01-T05/2/5 baseline, rejects swapped T03/T04 headings, rejects removal of the T03 provider statement from the T04 section, and rejects M2 3/5.

- [ ] **Step 3: Run the focused tests and preserve the red result**

Run:

```powershell
python -m pytest -q tests/contract/test_project_docs_validation.py
```

Expected: failure because the validator functions are not yet implemented and the current documents still contain the stale M0 handoff and reversed M2 definitions.

### Task 2: Implement pure validator checks

**Files:**
- Modify: scripts/validate_project_docs.py

**Interfaces:**
- Consumes: README, agent, STATUS, and M2 phase text.
- Produces: validate_dynamic_status_delegation(readme, agent, errors), validate_status_handoff(status, phase_text, errors), and validate_m2_task_contract(status, phase_text, errors).

- [ ] **Step 1: Centralize phase task extraction**

Add extract_phase_task_ids(phase, text) and reuse it from validate_phase_tasks. Add an explicit next-task regex that reads the task ID after 下一动作：, rather than taking the first task ID anywhere in STATUS.md.

- [ ] **Step 2: Validate stable dynamic-state delegation**

Require README.md and agent.md to mention STATUS.md and 唯一权威来源. Reject any agent text matching 当前活动阶段为/是 M followed by digits, and reject M followed by digits combined with 尚未完成, 已完成, or 为当前. Require agent.md to state that it does not duplicate dynamic phase numbering. The regex must work for M1, M2, and future M7, not only M0.

- [ ] **Step 3: Validate handoff ownership and dependency**

validate_status_handoff must reject a next task absent from the current phase file. validate_m2_task_contract must require exactly M2-T01 through M2-T05 in order, require the M2-T04 section to contain evidence slot score and a statement that earlier M2-T03 supplies it, and require M2 IN_PROGRESS 2/5 plus M3 BLOCKED_BY_M2 0/5.

- [ ] **Step 4: Wire the checks into main**

Read README.md, agent.md, STATUS.md, and the phase file once, call the new checks alongside the existing required-file, link, phase-task, and generic status checks, and retain the existing PASS/ERROR output contract.

- [ ] **Step 5: Run focused validation**

Run:

```powershell
python -m pytest -q tests/contract/test_project_docs_validation.py
python scripts/validate_project_docs.py
```

Expected: the mutation tests pass; the real documents fail only on the document repairs still required in Task 3.

### Task 3: Repair the document baseline

**Files:**
- Modify: agent.md
- Modify: STATUS.md
- Modify: docs/phases/M2-ranking-and-round1-selection.md

**Interfaces:**
- Consumes: the validator contract from Task 2.
- Produces: stable state delegation, correct M2-T03/T04 order, and the unchanged 2/5 handoff.

- [ ] **Step 1: Replace the stale agent handoff**

Replace the current M0 section with exactly:

```markdown
## 10. 动态项目状态

当前活动阶段、当前任务、允许执行范围和阶段门禁，
均以 STATUS.md 为唯一权威来源。

开始任何任务前必须读取 STATUS.md；
agent.md 不重复维护动态阶段编号或任务进度。
```

Do not write M2 or any other current phase number into agent.md.

- [ ] **Step 2: Make M2-T03 evidence-slot classification**

Define M2-T03 as title-and-abstract-only classification producing evidence slot, support level, grounded reason, supporting source text or excerpt, source text hash, and classifier/prompt/schema version. Keep closed vocabularies, fail-closed unknown values, source-text-only reasons, no fabricated abstracts, explicit title-only handling, explicit INDIRECT still-requires-validation language, and separate fake/real/human evidence.

- [ ] **Step 3: Make M2-T04 six-factor scoring**

Define M2-T04 as separate reranker score, dense similarity, concept match, query branch score, evidence slot score, and metadata quality. State that evidence slot score is supplied by earlier M2-T03. Preserve exact recomputation, explicit missing-component failure, versioned weights/calculation/model-rule snapshots, and no hidden defaults. Keep M2-T05 unchanged.

- [ ] **Step 4: Update STATUS without advancing T03**

Keep M2 IN_PROGRESS 2/5 and M3 BLOCKED_BY_M2 0/5. Set the next action to M2-T03：证据槽位分类. Add this baseline note:

```text
修正 M2-T03/T04 的执行依赖顺序：证据槽位分类先于依赖该结果的六分项评分。不改变阶段任务总数、已有完成状态或产品指标。
```

Do not mark M2-T03 started or complete.

- [ ] **Step 5: Run the green focused checks**

Run:

```powershell
python -m pytest -q tests/contract/test_project_docs_validation.py
python scripts/validate_project_docs.py
```

Expected: focused tests and project-doc validation pass.

### Task 4: Verify scope, hashes, and required offline gates

**Files:**
- Read-only verification; no additional files may change.

- [ ] **Step 1: Audit the diff**

Run git diff --name-only and git diff --stat. Reject every path outside the five allowed repository files plus this plan.

- [ ] **Step 2: Recompute protected hashes**

Recompute hashes for the M2-T02 candidate result, completion report, M1 candidate snapshot and manifest, and M2-T02 preflight evidence and receipt. Candidate result must remain 0a751dbc35bfa8d07433796113939412bfbdffd4c5c13043c90390bccfc5c35b; completion report must remain 009f4ad441b80a597b955c40ff6d31aac13476adf6574540bf3f4a790192361a; all other protected values must equal the pre-edit baseline.

- [ ] **Step 3: Run the required commands without model/network activity**

Run exactly:

```powershell
python -m pytest -q -m "not packaging"
python -m pytest -q -m packaging
python -m ruff check app evaluation scripts tests
python -m ruff check --select I app evaluation scripts tests
python -m mypy app evaluation scripts
python scripts/validate_project_docs.py
python scripts/validate_phase.py M0
python scripts/validate_m1_evidence.py
python scripts/validate_m2_t01_evidence.py
python scripts/validate_m2_t02_evidence.py
python -m pip check
git diff --check
```

Record pass counts, exit codes, and not-run real-model/human evidence separately.

### Task 5: Commit, publish, and open the Draft PR

**Files:**
- Stage only the allowed documentation, validator, test, and plan files.

**Interfaces:**
- Consumes: green local gates and unchanged protected hashes.
- Produces: branch agent/m2-task-order-and-handoff-fix, commit docs: fix M2 task dependency and status handoff, and a Draft PR with the same title.

- [ ] **Step 1: Create the branch from verified main**

Run:

```powershell
git switch -c agent/m2-task-order-and-handoff-fix
```

- [ ] **Step 2: Stage explicit paths and commit**

Run:

```powershell
git add agent.md STATUS.md docs/phases/M2-ranking-and-round1-selection.md scripts/validate_project_docs.py tests/contract/test_project_docs_validation.py docs/superpowers/plans/2026-08-03-m2-task-order-and-handoff-fix.md
git diff --cached --check
git commit -m "docs: fix M2 task dependency and status handoff"
```

- [ ] **Step 3: Push and create the Draft PR**

Push agent/m2-task-order-and-handoff-fix and create a Draft PR titled docs: fix M2 task dependency and status handoff. The body must report merge commit ef2c5b10f5cba847bdf9dbc73626dd9da4199f75, main HEAD, all four ancestor exit codes, the dynamic-state repair, T03/T04 rationale, M2 2/5, T03 not started, complete test statistics, unchanged M2-T02 hashes, and the explicit no-model/no-product-metric/no-historical-evidence-change boundary. Do not merge it.

## Self-Review Checklist

- Merge gate, file scope, dynamic handoff, T03/T04 dependency, STATUS baseline, regression tests, offline gates, hash preservation, and Draft PR are all covered.
- No task implements M2-T03 business logic, M2-T04 scoring, M2-T05 selection, or model/network activity.
- The regression uses M1 to prevent future dynamic phase drift, not only M0.
- Next-task validation checks the task belongs to the current phase file.
