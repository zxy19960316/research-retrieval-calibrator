# M2-T03 Human Adjudication Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close PR #15's M2-T03 human-review infrastructure while preserving the pending evidence boundary at M2 `IN_PROGRESS 2/5`, without creating any human conclusions or M2-T04 artifacts.

**Architecture:** Keep the existing machine-advisory bundle immutable and bind its pending history to the commit that generated it. Split current-worktree M2 gate evaluation into an explicit function, add a protocol-hashed blind template, and put completed human judgments in a separate Pydantic contract. Publish pending bundle, receipt, blind template, and report through a preflight/stage/atomic-publish/rollback transaction with the report last.

**Tech Stack:** Python 3.12, Pydantic v2, pytest, Ruff, mypy, JSON UTF-8 artifacts, Git commit/`git show` history validation, and POSIX-relative artifact bindings on Windows.

## Global Constraints

- This turn only closes M2-T03 human-adjudication infrastructure; do not record real human judgments, implement M2-T04, or change M2 from `IN_PROGRESS 2/5`.
- Pending validation must use `bundle.generated_from_commit` and `git show <commit>:STATUS.md`; it must never use the current worktree STATUS bytes as historical proof.
- Current status validation is separate: no completed result requires M2 `2/5`; M2 `3/5` requires a completed-result validator to pass.
- Blind review may use only `paper_id`, `title`, `abstract`, `source`, `source_id`, `url`, and `ResearchIntent`.
- Human input is only `SUPPORTED`/`REJECTED` plus the declared human fields; `CONFIRM`/`REVISE`/`REJECT` is a post-review derived comparison, never an input field.
- No completed result, receipt, or report files may be created in this PR.
- All artifact paths are non-empty relative POSIX paths with no backslashes, `..`, Windows drives/UNC roots, or absolute POSIX roots.
- Diagnostics and validation errors remain closed and sanitized; no credentials, authorization headers, tokens, passwords, tracebacks, raw exceptions, URLs, or local paths are emitted as raw failure detail.

## File Map

- Modify `app/models/m2_t03_human_adjudication.py`: tighten artifact path safety, change empty pending human fields to the protocol's `verdict` contract, bind the review protocol, and close pending policy fields.
- Create `app/models/m2_t03_human_adjudication_result.py`: define `HumanJudgment`, `CompletedHumanAdjudicationResult`, `HumanAdjudicationResultReceipt`, and `HumanAdjudicationResultReport`, including model-level completeness and source-excerpt invariants.
- Modify `scripts/build_m2_t03_human_adjudication_bundle.py`: load historical STATUS from Git, generate the protocol-bound blind template, and make publication rollback-safe with the report last.
- Modify `scripts/validate_m2_t03_human_adjudication_bundle.py`: validate historical STATUS/report bindings, policy/protocol/template/report closure, safe paths, and expose `validate_current_m2_gate` separately.
- Create `docs/reviews/m2-t03-human-adjudication-protocol.md`: frozen source boundary, slot/support vocabulary, and human-vs-derived decision rules.
- Create `evaluation/source-artifacts/m2-t03-human-adjudication-review-template.json`: generated blind review surface without machine advisory fields.
- Modify `tests/contract/test_m2_t03_human_adjudication.py`: alternate-root status transition, historical commit/status tamper tests, policy/template/path/privacy adversarial tests, and current-gate tests.
- Modify `tests/unit/test_m2_t03_human_adjudication_bundle.py`: update pending field expectations and add publication failure/rollback cases.
- Create `tests/unit/test_m2_t03_human_adjudication_result.py`: exercise supported/rejected judgment validation and exact 33-item result invariants.
- Refresh only pending evidence files after implementation: `evaluation/source-artifacts/m2-t03-human-adjudication-bundle.json`, `evaluation/source-artifacts/m2-t03-human-adjudication-bundle-receipt.json`, and `evaluation/reports/m2-t03-human-adjudication.json`; the blind template is a fourth pending source artifact.

### Task 1: Add the frozen review protocol and red tests for the new boundaries

**Files:**
- Create: `docs/reviews/m2-t03-human-adjudication-protocol.md`
- Modify: `tests/contract/test_m2_t03_human_adjudication.py`
- Modify: `tests/unit/test_m2_t03_human_adjudication_bundle.py`

**Interfaces:**
- The protocol is bound by `ArtifactBinding(path="docs/reviews/m2-t03-human-adjudication-protocol.md", sha256=<raw UTF-8 SHA-256>)`.
- Contract tests call `validate_bundle(bundle_path, repository_root=...)` and `validate_current_m2_gate(repository_root=..., completed_result_validator=...)`.

- [ ] **Step 1: Write the failing protocol and status-transition tests.**

  Add tests that (a) copy the pending artifacts into an alternate Git root, commit a historical `STATUS.md` at M2 `2/5`, then mutate only the current `STATUS.md` to M2 `3/5` and expect `validate_bundle` to remain valid, (b) expect `validate_current_m2_gate` to reject the same current root without a passing completed validator, and (c) mutate the generated commit, historical STATUS, report STATUS hash, and historical STATUS boundary one at a time and expect failure.

  Add assertions for the protocol text containing the seven allowed sources, all five `EvidenceSlot` values, all three `SupportLevel` values, and the derived `CONFIRM`/`REVISE`/`REJECT` rules.

- [ ] **Step 2: Run the focused red tests.**

  Run:

  ```text
  python -m pytest -q tests/unit/test_m2_t03_human_adjudication_bundle.py tests/contract/test_m2_t03_human_adjudication.py
  ```

  Expected: the new historical-status and protocol-binding assertions fail against the current validator/builder; existing pending tests remain the characterization baseline.

- [ ] **Step 3: Write the protocol file.**

  Freeze the following source boundary and decision grammar in Markdown:

  ```text
  Allowed: paper_id, title, abstract, source, source_id, url, ResearchIntent
  Forbidden: full text, citation count, author reputation, journal rank,
  reranker/dense scores, final selection, network supplementation, user feedback
  Human verdict: SUPPORTED or REJECTED
  Derived comparison: CONFIRM when verdict/slot/support agree with advisory;
  REVISE when SUPPORTED disagrees on slot/support; REJECT when verdict is REJECTED.
  ```

  State that `INDIRECT` must explicitly say target-problem validation remains required and that excerpts must be exact title/abstract substrings.

- [ ] **Step 4: Commit the protocol/red-test checkpoint.**

  ```text
  git add docs/reviews/m2-t03-human-adjudication-protocol.md tests/contract/test_m2_t03_human_adjudication.py tests/unit/test_m2_t03_human_adjudication_bundle.py
  git commit -m "test: harden M2-T03 human review transition gates"
  ```

### Task 2: Bind pending history, protocol policy, and the blind template

**Files:**
- Modify: `app/models/m2_t03_human_adjudication.py`
- Modify: `scripts/build_m2_t03_human_adjudication_bundle.py`
- Modify: `scripts/validate_m2_t03_human_adjudication_bundle.py`
- Create: `evaluation/source-artifacts/m2-t03-human-adjudication-review-template.json`
- Modify: `tests/contract/test_m2_t03_human_adjudication.py`

**Interfaces:**
- Builder constants: `PROTOCOL_PATH`, `REVIEW_TEMPLATE_PATH`, `EXPECTED_COMMANDS`, and `EXPECTED_EXIT_CODES`.
- Builder helper: `_historical_status_bytes(repository_root: Path, generated_from_commit: str) -> bytes`.
- Validator helper: `_historical_status_bytes(repository_root: Path, generated_from_commit: str) -> bytes | None`.
- Validator entry points: `validate_bundle(...)` and `validate_current_m2_gate(...)`.

- [ ] **Step 1: Add red assertions for exact policy/template closure.**

  Assert that `review_policy.source_fields` is exactly `("paper_id", "title", "abstract", "source", "source_id", "url", "ResearchIntent")`, that `disallowed_sources` is the protocol order, that the bundle contains a protocol binding, and that the template top-level binding contains pending-bundle path/hash, protocol path/hash, candidate count `33`, and the exact paper-ID order.

  Assert every template item has exactly `paper_id`, `title`, `abstract`, `source_identity`, `source_text_sha256`, and `human_adjudication`; assert no serialized key/value contains machine advisory content or score/rank/selection fields.

- [ ] **Step 2: Implement model-level safe artifact paths and pending field names.**

  Add a `field_validator("path")` to `ArtifactBinding` using `PurePosixPath`/`PureWindowsPath`: reject empty strings, backslashes, `..`, `.`, POSIX absolute paths, Windows drive paths, and UNC paths. Change empty `HumanAdjudicationFields` to `verdict: Literal["SUPPORTED", "REJECTED"] | None` and bind `ReviewPolicy.protocol: ArtifactBinding`.

- [ ] **Step 3: Replace current STATUS reads with historical Git reads.**

  Resolve the commit with `git merge-base --is-ancestor`, then read exact bytes using `git show <commit>:STATUS.md` with `text=False`. Parse only those bytes for M2 `IN_PROGRESS 2/5` and M3 `BLOCKED_BY_M2 0/5`; bind that exact hash in `report.input_artifacts`.

  Remove the pending validator's current-worktree `_validate_status(repository_root, errors)` call. `validate_bundle` must validate pending history only. Keep current status logic in `validate_current_m2_gate`, with this behavior:

  ```python
  if current_m2 == "IN_PROGRESS 2/5":
      return valid
  if current_m2 == "IN_PROGRESS 3/5" and completed_result_validator is not None:
      return completed_result_validator(repository_root)
  return invalid
  ```

  Require M3 `BLOCKED_BY_M2 0/5` in both current branches. The default completed-result path remains absent in PR #15.

- [ ] **Step 4: Generate and validate the blind template.**

  Build template JSON from pending context only; copy no `machine_advisory` object or advisory-derived value. Use the new `verdict` field names and the exact source hash from each context item. Include the protocol and bundle bindings, candidate count, and order.

  Extend receipt/report bindings to include the template and protocol. Require `bundle.generated_at_utc == report.generated_at_utc`, exact fixed command tuple, exact `{"build_bundle": 0}` exit-code map, equal generated commits, and historical STATUS hash equality.

- [ ] **Step 5: Run the focused tests and commit the boundary implementation.**

  ```text
  python -m pytest -q tests/unit/test_m2_t03_human_adjudication_bundle.py tests/contract/test_m2_t03_human_adjudication.py
  git add app/models/m2_t03_human_adjudication.py scripts/build_m2_t03_human_adjudication_bundle.py scripts/validate_m2_t03_human_adjudication_bundle.py evaluation/source-artifacts/m2-t03-human-adjudication-review-template.json
  git commit -m "fix: preserve pending adjudication evidence across status changes"
  ```

### Task 3: Add the independent completed-human-result contracts

**Files:**
- Create: `app/models/m2_t03_human_adjudication_result.py`
- Create: `tests/unit/test_m2_t03_human_adjudication_result.py`

**Interfaces:**
- `HumanJudgment` contains `paper_id`, `title`, `abstract`, `source_text_sha256`, `verdict`, `evidence_slot`, `support_level`, `grounded_reason`, `supporting_excerpt`, `reviewer_id`, `reviewed_at_utc`, and `notes`.
- `CompletedHumanAdjudicationResult` contains fixed M2-T03 metadata, pending/protocol artifact bindings, `candidate_count=33`, `paper_id_order`, and a 33-item `judgments` tuple.
- `HumanAdjudicationResultReceipt` and `HumanAdjudicationResultReport` bind the reserved result paths but are not instantiated as files in PR #15.

- [ ] **Step 1: Write failing Pydantic contract tests.**

  Cover: supported judgments require slot/support/nonblank reason/nonblank exact title-or-abstract excerpt/nonblank reviewer/UTC timestamp; rejected judgments require null slot/support/nonblank reason and may omit the excerpt; non-UTC timestamps fail; wrong source hashes fail; extra fields fail; duplicate/missing/unknown paper IDs and non-33 result lengths fail.

- [ ] **Step 2: Implement `HumanJudgment` with a `model_validator(mode="after")`.**

  Normalize zero-offset timestamps to `UTC`, reject non-aware/non-zero-offset values, compute the exact title/abstract source hash, and enforce the two verdict branches in one after-model validator. Keep `extra="forbid"` and require all reviewer text to be non-blank after stripping.

- [ ] **Step 3: Implement the completed result/receipt/report models.**

  Use frozen extra-forbid models, fixed version literals, `Literal[33]`, exact 33-item bounds, and an after-model validator that checks unique IDs, exact `paper_id_order`, and absence of partial judgments. Define the reserved result paths as constants without writing those artifacts.

- [ ] **Step 4: Run and commit the independent contract tests.**

  ```text
  python -m pytest -q tests/unit/test_m2_t03_human_adjudication_result.py
  git add app/models/m2_t03_human_adjudication_result.py tests/unit/test_m2_t03_human_adjudication_result.py
  git commit -m "feat: define completed human adjudication contracts"
  ```

### Task 4: Make four-artifact publication rollback-safe

**Files:**
- Modify: `scripts/build_m2_t03_human_adjudication_bundle.py`
- Modify: `tests/unit/test_m2_t03_human_adjudication_bundle.py`

**Interfaces:**
- `_publish_conflict_safe(repository_root: Path, targets: Mapping[Path, bytes]) -> None`.
- Publication order is bundle, receipt, blind template, report; report is always last.
- Failure surface remains `BundleError("RESULT_CONFLICT")` or `BundleError("RESULT_PUBLISH_FAILED")` with no raw exception text.

- [ ] **Step 1: Add failure-injection tests.**

  For first-, second-, and third-publish `os.replace` failures, assert all newly created formal files are absent, all pre-existing files retain exact bytes and mtimes, and all owned temp files are removed. Add same-byte idempotent reuse, different-byte zero-modification conflict, symlink target rejection, and parent-file conflict tests.

- [ ] **Step 2: Implement complete preflight.**

  Validate every relative target path, reject symlink/non-regular destinations and symlink/non-directory parents, read every existing destination before staging, and fail on any different byte before creating any staging file.

- [ ] **Step 3: Stage and verify every byte.**

  Create unique sibling staging files in the destination's parent, write all target bytes, re-read each staging file through the regular-file guard, and compare exact bytes before the first `os.replace`.

- [ ] **Step 4: Publish and roll back.**

  Replace only previously absent destinations in insertion order; skip exact-byte existing files. Track created destinations and owned staging files. On any exception, delete only created destinations whose bytes equal this invocation's bytes, remove all staging files, and remove only empty directories created by this invocation. Never delete pre-existing files.

- [ ] **Step 5: Run tests and commit the transaction fix.**

  ```text
  python -m pytest -q tests/unit/test_m2_t03_human_adjudication_bundle.py
  git add scripts/build_m2_t03_human_adjudication_bundle.py tests/unit/test_m2_t03_human_adjudication_bundle.py
  git commit -m "fix: make adjudication bundle publication rollback-safe"
  ```

### Task 5: Close validator security/policy checks and adversarial coverage

**Files:**
- Modify: `scripts/validate_m2_t03_human_adjudication_bundle.py`
- Modify: `tests/contract/test_m2_t03_human_adjudication.py`

**Interfaces:**
- `_validate_privacy(value: object, errors: list[str])` rejects score/rank/selection keys and credential/error/path text using fixed sanitized error messages.
- `validate_bundle(...) -> ValidationResult` validates pending artifacts, protocol/template bindings, historical status, fixed policy, fixed commands/exit codes, and runner/report/receipt hashes.
- `validate_current_m2_gate(...) -> ValidationResult` is the only current-worktree transition gate.

- [ ] **Step 1: Add adversarial mutation tests.**

  Cover fixed M1 first-run/repair-manifest/candidate/classification hashes; missing/duplicate/reordered candidates; source-text hash and advisory drift; any human prefill; policy source-field missing/duplicate/reordered and disallowed-source drift; generated-commit mismatch; command/exit-code drift; runner hash drift; historical STATUS drift; current 2/5-to-3/5 transition; Windows/UNC/POSIX absolute paths; credential-style text; extra score/rank/selection fields; and alternate repository roots.

- [ ] **Step 2: Implement exact report/policy derivation.**

  Compare the bundle policy against protocol constants, compare the report's generated time/commit/input/artifact bindings, require exact command/exit maps, verify the historical STATUS hash from Git, verify template structure and no advisory fields, and verify all current formal files are regular non-symlink files.

- [ ] **Step 3: Harden privacy/path checks.**

  Replace the limited `/Users|home|tmp` absolute-path regex with the model-level path validator and a value scan that catches credentials, authorization headers, tokens, passwords, tracebacks, raw exceptions, and forbidden absolute-path forms without printing the offending text.

- [ ] **Step 4: Run focused adversarial tests and commit.**

  ```text
  python -m pytest -q tests/contract/test_m2_t03_human_adjudication.py tests/unit/test_m2_t03_human_adjudication_bundle.py
  git add scripts/validate_m2_t03_human_adjudication_bundle.py tests/contract/test_m2_t03_human_adjudication.py
  git commit -m "test: expand M2-T03 human review adversarial coverage"
  ```

### Task 6: Refresh pending evidence and run the PR #15 gates

**Files:**
- Modify: `evaluation/source-artifacts/m2-t03-human-adjudication-bundle.json`
- Modify: `evaluation/source-artifacts/m2-t03-human-adjudication-bundle-receipt.json`
- Create/modify: `evaluation/source-artifacts/m2-t03-human-adjudication-review-template.json`
- Modify: `evaluation/reports/m2-t03-human-adjudication.json`

- [ ] **Step 1: Generate only pending artifacts.**

  Run:

  ```text
  python scripts/build_m2_t03_human_adjudication_bundle.py --execute-offline
  python scripts/validate_m2_t03_human_adjudication_bundle.py
  ```

  Confirm no completed result/receipt/report exists and the resulting bundle/report retain `M2 IN_PROGRESS 2/5`, `human_review=pending`, `M2-T04=not_started`, and `scoring_eligible=false`.

- [ ] **Step 2: Run the required PR gates.**

  ```text
  python -m pytest -q tests/unit/test_m2_t03_human_adjudication_bundle.py tests/contract/test_m2_t03_human_adjudication.py
  python -m pytest -q -m "not packaging"
  python -m pytest -q -m packaging
  python -m ruff check app evaluation scripts tests
  python -m ruff check --select I app evaluation scripts tests
  python -m mypy app evaluation scripts
  python scripts/validate_project_docs.py
  python scripts/validate_phase.py M0
  python scripts/validate_m1_evidence.py
  python scripts/validate_m1_frozen_intent_replay_repair.py
  python scripts/validate_m2_t01_evidence.py
  python scripts/validate_m2_t02_evidence.py
  python scripts/validate_m2_t03_evidence.py
  python scripts/validate_m2_t03_human_adjudication_bundle.py
  python -m pip check
  git diff --check
  ```

- [ ] **Step 3: Audit scope and status.**

  Verify `git status --short`, the changed-path list, exact current branch/HEAD, no completed-result files, no M2/M3 status advancement, and no real-model/network/human-judged claim. Keep failures/not-run gates explicit in the final handoff.

- [ ] **Step 4: Commit the pending evidence refresh.**

  ```text
  git add evaluation/source-artifacts/m2-t03-human-adjudication-bundle.json evaluation/source-artifacts/m2-t03-human-adjudication-bundle-receipt.json evaluation/source-artifacts/m2-t03-human-adjudication-review-template.json evaluation/reports/m2-t03-human-adjudication.json
  git commit -m "chore: refresh pending human review evidence"
  ```

## Self-Review Checklist

- [ ] The pending validator never reads current `STATUS.md` for historical proof.
- [ ] The current gate is exposed independently and does not make pending validation fail after a worktree-only `2/5` to `3/5` edit.
- [ ] The template contains no machine advisory fields or derived comparison values.
- [ ] `HumanJudgment` uses `model_validator` and checks all supported/rejected branches.
- [ ] Publication failure leaves no partial formal bundle and preserves all pre-existing files.
- [ ] Protocol, template, receipt, report, commands, exit codes, commits, timestamps, paths, and hashes are mutually bound.
- [ ] PR #15 does not create completed result artifacts or update STATUS to M2 `3/5`.
