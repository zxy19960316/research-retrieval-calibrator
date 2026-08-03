# M1 Replay Repair Evidence Gates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Execute this plan inline because the user explicitly authorized implementation in the existing Draft PR #14.

**Goal:** Harden the existing M1 frozen-intent replay repair so source-bundle inventory, raw replay candidate evolution, closed evidence schemas, alternate-root validation, and conflict-safe append-only publication are all enforced without changing historical M1 or protected M2 evidence.

**Architecture:** Keep the core replay contract unchanged and remove the runner's candidate projection. Add a public source-bundle inventory audit and a candidate audit that binds raw replay output to the protected M2 candidate snapshot. Represent the repair manifest/report as closed schemas, validate every target before publication, stage all three artifacts, and publish them conflict-safely as one bundle. Regenerate only the existing dated repair bundle and report after the new gates pass.

**Tech Stack:** Python 3.12, Pydantic, pytest, JSON/UTF-8 byte hashing, pathlib, Git subprocess checks, Ruff, mypy, GitHub Actions, PowerShell.

## Global Constraints

- Work only on the existing `agent/m1-frozen-intent-replay-repair` branch and existing Draft PR #14; do not create PR #15, merge PR #14, rebase, squash, amend, or force-push.
- Do not modify the historical M1 rebaseline bundle or reports, the M2 candidate snapshot/manifest, M2-T01/T02/T03 formal evidence, `STATUS.md` M2 `IN_PROGRESS 2/5`, or protected hashes.
- Do not make real arXiv requests, run/download models, create a human review bundle, start M2-T04, or claim external/model/human completion; keep `scoring_eligible=false`.
- Preserve `frozen_at` in canonical intent identity and classify the repaired run as `offline_cache_replay=real_cache_replay` with `new_real_arxiv_requests=not_run`.
- Preserve the historical first-run bytes exactly (`ed4a4d89a247c535e6138644069094d3c59a046bdefd44a79e48a4f861f95afa`) and keep the existing `.gitattributes` binary rule.
- Use new commits only, stage explicit paths, and avoid destructive cleanup. Temporary mutation fixtures must stay under pytest `tmp_path`.
- The fixed source manifest hash is `5dad9e6070edececceeead3f8bb3e7f43302f7405fee811dcc7ddb34c84f0849`; the protected M2 candidate snapshot hash is `4a2aec0fd0a1d22adc801fd3bc506e5da89895d1276cd572e2ac64014c162448`.

---

### Task 1: Record the evidence-gate design and establish a red baseline

**Files:**
- Create: `docs/superpowers/plans/2026-08-03-m1-replay-repair-evidence-gates.md`
- Read: `agent.md`, `STATUS.md`, `PROJECT_PLAN.md`, `docs/phases/M2-ranking-and-round1-selection.md`, `docs/git_workflow.md`
- Test: `tests/contract/test_m1_frozen_intent_replay_repair_runner.py`
- Test: `tests/contract/test_m1_frozen_intent_repair_validation.py`

**Interfaces:**
- Runner tests will call the CLI through `scripts.repair_m1_frozen_intent_replay.main()` and the execution API with a temporary repository root/path configuration.
- Validator tests will call `validate_m1_frozen_intent_replay_repair(report_path=..., repository_root=...)` and `validate_source_bundle_inventory(repository_root)`.

- [ ] **Step 1: Add runner contract tests before implementation.** Cover refusal without `--execute-offline-repair`, fixed source manifest/first-run/replay hash mutations, exact intent drift, cache miss with a forbidden transport, three-artifact first run, byte-identical second run, no temp residue, and an existing-target conflict that leaves all three targets unchanged.
- [ ] **Step 2: Add alternate-root validator contract tests before implementation.** Build isolated repository fixtures under `tmp_path` and cover exact manifest/report fields including nested `candidate_audit` and `zero_transport_replay`, canonical/replay hash drift, `STATUS.md` `3/5`, protected M2 drift, absolute/credential-shaped report text, extra cache, missing replay, non-ancestor commit, and unknown nested fields.
- [ ] **Step 3: Run the two new files red-first.** Run `python -m pytest tests/contract/test_m1_frozen_intent_repair_runner.py tests/contract/test_m1_frozen_intent_repair_validation.py -q`; record that the new public APIs/schema gates are not yet present and do not weaken the tests.
- [ ] **Step 4: Commit only the new failing contract tests.**

```powershell
git add tests/contract/test_m1_frozen_intent_repair_runner.py tests/contract/test_m1_frozen_intent_repair_validation.py
git commit -m "test: harden M1 replay repair evidence gates"
```

### Task 2: Add closed repair evidence models

**Files:**
- Create: `app/models/m1_replay_repair.py`
- Modify: `app/models/__init__.py` only if this repository exports model symbols there
- Test: `tests/contract/test_m1_frozen_intent_repair_validation.py`

**Interfaces:**
- Define `M1ReplayCandidateAudit`, `M1ReplayTransportAudit`, `M1ReplayRepairManifest`, and `M1ReplayRepairReport` as Pydantic models with `ConfigDict(extra="forbid")`.
- `M1ReplayCandidateAudit` exposes `candidate_count`, `candidate_order_equal`, `candidate_identity_equal`, `candidate_payload_equal`, `candidate_delta_fields`, `protected_candidate_snapshot_path`, `protected_candidate_snapshot_sha256`, and `replay_candidate_array_sha256`.
- `M1ReplayTransportAudit` exposes exactly `transport_requests`, `cache_hits`, and `query_count`.
- The manifest version is `m1-frozen-intent-replay-repair.v2`; the report version is `m1-frozen-intent-replay-repair.v2`.

- [ ] **Step 1: Define the exact top-level and nested field sets in the models.** Reject unknown fields and preserve list ordering for drift fields, candidate delta fields, commands, and not-run gates.
- [ ] **Step 2: Validate the model tests red-to-green in isolation.** Run the nested extra-field, missing-field, and type/value contract tests.
- [ ] **Step 3: Commit the closed-schema models.**

```powershell
git add app/models/m1_replay_repair.py app/models/__init__.py tests/contract/test_m1_frozen_intent_repair_validation.py
git commit -m "refactor: close M1 replay repair schemas"
```

### Task 3: Add complete fixed source-bundle inventory auditing

**Files:**
- Modify: `scripts/validate_m1_frozen_intent_replay_repair.py`
- Modify: `scripts/repair_m1_frozen_intent_replay.py`
- Test: `tests/contract/test_m1_frozen_intent_repair_runner.py`
- Test: `tests/contract/test_m1_frozen_intent_repair_validation.py`

**Interfaces:**
- Add public `validate_source_bundle_inventory(repository_root: Path = ROOT) -> SourceBundleAudit` (or an equivalent public function returning a typed audit with validity, verified file count, payload count, cache count, and errors).
- Audit `evaluation/source-artifacts/m1-rebaseline-2026-07-28/bundle-manifest.json` before any cache-only replay.
- Enforce `payload_file_count=16`, `cache_file_count=12`, `file_inventory_excludes=["bundle-manifest.json"]`, exact inventory set, safe relative paths, regular non-symlink files, byte sizes, and SHA-256 hashes.

- [ ] **Step 1: Implement the public inventory audit against actual filesystem bytes.** Keep the existing fixed hashes and Git tracked-byte comparison as additional checks; do not replace working-tree validation with `git show HEAD`.
- [ ] **Step 2: Make runner execution invoke the inventory audit before constructing the cache-only adapter.** Map an audit failure to a stable fail-closed error and publish no repair artifact.
- [ ] **Step 3: Add mutations for missing/extra files, symlinks/directories, unsafe paths, size drift, hash drift, and manifest count/field drift.** Assert the audit reports failure and the runner does not call transport or publish files.
- [ ] **Step 4: Run the source-inventory-focused tests and commit.**

```powershell
python -m pytest tests/contract/test_m1_frozen_intent_repair_validation.py -k "inventory or source_bundle" -q
git add scripts/validate_m1_frozen_intent_replay_repair.py scripts/repair_m1_frozen_intent_replay.py tests/contract/test_m1_frozen_intent_repair_validation.py tests/contract/test_m1_frozen_intent_repair_runner.py
git commit -m "fix: validate complete M1 source bundle inventory"
```

### Task 4: Preserve raw replay candidate evidence and bind it to M2

**Files:**
- Modify: `scripts/repair_m1_frozen_intent_replay.py`
- Modify: `scripts/validate_m1_frozen_intent_replay_repair.py`
- Test: `tests/contract/test_m1_frozen_intent_repair_runner.py`
- Test: `tests/contract/test_m1_frozen_intent_repair_validation.py`

**Interfaces:**
- Delete `_align_historical_candidate_contract()` and do not replace `repaired.candidates` with `first_run.candidates`.
- Compare candidate fields exactly for `paper_id`, `source`, `source_id`, `title`, `authors`, `year`, `doi`, `url`, `retrieval_paths`, `cluster_id`, `member_source_identities`, and `merge_reasons`.
- Bind replay `title`, `abstract`, `source`, and `source_id` by exact paper ID to `evaluation/snapshots/m2/m1-candidates.v1.json` and its fixed SHA-256.
- Emit `candidate_audit` with computed `candidate_count`, order/identity/payload equality, delta fields, protected snapshot binding, and replay candidate array hash. Set `candidate_projection_applied=false` in the report.

- [ ] **Step 1: Remove candidate normalization and replace equality checks with a computed audit.** Preserve `abstract`; permit only an explicit schema-evolution delta such as `abstract` between historical first-run and raw replay.
- [ ] **Step 2: Implement protected snapshot validation.** Require exactly 33 paper IDs and exact title, abstract, source, and source_id values; fail closed on any mismatch before publication.
- [ ] **Step 3: Add tests for candidate order/identity/payload changes, snapshot drift, missing IDs, and the absence of projection.**
- [ ] **Step 4: Run candidate-audit tests and commit.**

```powershell
python -m pytest tests/contract/test_m1_frozen_intent_repair_runner.py tests/contract/test_m1_frozen_intent_repair_validation.py -k "candidate" -q
git add scripts/repair_m1_frozen_intent_replay.py scripts/validate_m1_frozen_intent_replay_repair.py tests/contract/test_m1_frozen_intent_repair_runner.py tests/contract/test_m1_frozen_intent_repair_validation.py
git commit -m "fix: preserve raw replay candidate evidence"
```

### Task 5: Make publication atomic/conflict-safe and enforce runner fail-closed behavior

**Files:**
- Modify: `scripts/repair_m1_frozen_intent_replay.py`
- Test: `tests/contract/test_m1_frozen_intent_replay_runner.py`

**Interfaces:**
- Replace per-file `_write_if_unchanged()` publication with a three-target preflight/stage/validate/publish routine.
- The routine must return all three SHA-256 values, reuse identical existing bytes without rewriting, reject any conflicting existing target with `REPAIR_TARGET_CONFLICT`, and leave no partial bundle or temporary residue after a conflict.

- [ ] **Step 1: Implement target preflight.** Read every existing target before creating any new target; reject directory/symlink targets and any byte conflict.
- [ ] **Step 2: Stage all three bytes in a unique sibling staging directory.** Validate staged bytes and manifest bindings before publication, use `os.replace` only for absent targets, and clean only the task-owned staging directory after success or failure.
- [ ] **Step 3: Add cache-miss, first-run, second-run, and conflict tests.** Assert no network transport request, exactly three files, byte identity on rerun, no overwrite, no `.tmp` residue, and no partial bundle on conflict.
- [ ] **Step 4: Run the full runner contract file and commit.**

```powershell
python -m pytest tests/contract/test_m1_frozen_intent_repair_runner.py -q
git add scripts/repair_m1_frozen_intent_replay.py tests/contract/test_m1_frozen_intent_repair_runner.py
git commit -m "fix: publish M1 replay repair atomically"
```

### Task 6: Harden alternate-root validator and regenerate v2 evidence

**Files:**
- Modify: `scripts/validate_m1_frozen_intent_replay_repair.py`
- Modify: `scripts/repair_m1_frozen_intent_replay.py`
- Create/modify: `evaluation/source-artifacts/m1-intent-replay-repair-2026-08-03/replay/first-round.json`
- Create/modify: `evaluation/source-artifacts/m1-intent-replay-repair-2026-08-03/repair-manifest.json`
- Create/modify: `evaluation/reports/m1-frozen-intent-replay-repair-2026-08-03.json`
- Test: `tests/contract/test_m1_frozen_intent_repair_validation.py`

**Interfaces:**
- Validator must require exact closed manifest/report fields, exact nested fields, corrected replay and canonical hashes, source inventory counts, candidate audit, protected hashes, `STATUS.md` `IN_PROGRESS 2/5`, `scoring_eligible=false`, human review/M2-T04 not started, and implementation commit ancestry.
- Report must include `source_bundle_inventory_verified=true`, `source_bundle_verified_file_count=16`, `source_cache_verified_file_count=12`, `candidate_projection_applied=false`, `offline_cache_replay=real_cache_replay`, and `new_real_arxiv_requests=not_run`.

- [ ] **Step 1: Update validator mutation tests to use `repository_root=temp_repository` for every alternate-root case.** Cover missing/extra manifest/report fields, canonical intent/replay/zero-transport drift, status/protected drift, unsafe report text, extra cache/missing replay, and non-ancestor commit.
- [ ] **Step 2: Run the validator tests red-first against the current v1 artifact and confirm the expected schema/version failures.**
- [ ] **Step 3: Update the validator and runner to emit/consume v2 manifest/report fields and exact schemas.** Keep fixed historical and protected hash checks unchanged.
- [ ] **Step 4: Run `python scripts/repair_m1_frozen_intent_replay.py --execute-offline-repair` to regenerate only the existing dated repair paths.** Verify the first-run byte copy remains unchanged and record manifest/replay/report hashes.
- [ ] **Step 5: Run `python scripts/validate_m1_frozen_intent_replay_repair.py` and the complete new validation file.**
- [ ] **Step 6: Commit regenerated evidence and validator changes.**

```powershell
python -m pytest tests/contract/test_m1_frozen_intent_repair_validation.py -q
python scripts/repair_m1_frozen_intent_replay.py --execute-offline-repair
python scripts/validate_m1_frozen_intent_replay_repair.py
git add scripts/repair_m1_frozen_intent_replay.py scripts/validate_m1_frozen_intent_replay_repair.py evaluation/source-artifacts/m1-intent-replay-repair-2026-08-03/replay/first-round.json evaluation/source-artifacts/m1-intent-replay-repair-2026-08-03/repair-manifest.json evaluation/reports/m1-frozen-intent-replay-repair-2026-08-03.json tests/contract/test_m1_frozen_intent_repair_validation.py
git commit -m "chore: refresh transparent replay repair evidence"
```

### Task 7: Run local release gates, update PR #14, and verify remote CI

**Files:**
- Modify: `.github/workflows/docs-validation.yml` only if the existing M1 repair validation step needs its command or placement corrected
- No changes to `STATUS.md`, historical M1 artifacts/reports, protected M2 artifacts, or PR number/state

**Interfaces:**
- CI continues to run `python scripts/validate_m1_frozen_intent_replay_repair.py` after M1 validation and before M2 validators.
- PR #14 remains open and Draft; its body is updated only after the final pushed HEAD and remote CI run are known.

- [ ] **Step 1: Verify old and protected bytes independently.** Compare filesystem SHA-256, Git blob SHA-1 where applicable, manifest inventory, and semantic payloads; assert the old source/reports and protected M2 hashes are unchanged.
- [ ] **Step 2: Run focused, full non-packaging, packaging, Ruff including `I001`, mypy, and existing M0/M1/M2-T01/T02/T03 validators.** Record all commands and exit codes in the v2 report; do not reinterpret offline/replay results as real external/model evidence.
- [ ] **Step 3: Commit any CI-only adjustment as a new commit.**
- [ ] **Step 4: Review `git diff --check`, status, and explicit staged paths; push the existing branch without force.**
- [ ] **Step 5: Wait for the latest PR #14 CI run to pass.** Capture its remote run ID and keep the PR Draft/unmerged.
- [ ] **Step 6: Update PR #14 body with HEAD, source inventory counts, candidate audit, hashes, tests, remote CI run ID, protected hashes, M2 `IN_PROGRESS 2/5`, human review not started, and M2-T04 not started.**
- [ ] **Step 7: Final handoff reports only successful Git action directives and confirms no merge/new PR/model/network work occurred.**

```powershell
git diff --check
git status --short --branch
git push origin agent/m1-frozen-intent-replay-repair
```

## Self-review checklist

- [ ] Every attachment requirement maps to a task and an exact test or validation gate.
- [ ] No task edits historical M1 evidence, protected M2 evidence, `STATUS.md`, or the existing `.gitattributes` binary rule.
- [ ] No candidate alignment/replacement remains; raw `abstract` evidence and M2 snapshot binding are explicit.
- [ ] The runner audits the source bundle before transport construction and cannot leave a partial three-file bundle after conflict.
- [ ] Alternate-root tests exercise working-tree mutations rather than relying only on `git show HEAD`.
- [ ] Manifest/report and nested objects reject unknown fields.
