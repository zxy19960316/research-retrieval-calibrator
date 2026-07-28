# M2-T01R3 Publication Boundaries Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use inline red-green execution. The user-authorized delivery boundary is one implementation commit, A4.

**Goal:** Close M2 frozen-input publication edge cases while restoring the explicit production 33-candidate gate and validating all snapshot/manifest provenance.

**Architecture:** The single freeze gate will model existing snapshot and manifest files independently, stage/verify both byte streams, and roll back only targets actually published in the current call. The low-level snapshot validator will require an explicit expected count, cross-check every provenance field between snapshot and manifest, and retain exact-byte SHA-256 validation. The high-level production embedding loader will retain its explicit default of 33; synthetic callers must state their alternate count and evidence-report label.

**Tech Stack:** Python 3.12, Pydantic v2, pytest, existing M1 arXiv cache/deduplicator, Ruff, mypy, GitHub Actions.

## Global Constraints

- A3 head is `0f2a3a6fd3cf5a9440bf403b0dd990f9d33b9bfe`; retain `A -> A1 -> A2 -> A3 -> A4` and do not amend A3.
- Deliver only A4: `fix: close M2 freeze publication edge cases`.
- Do not create M1 re-baseline data, B1/B2/C, real snapshots/caches, vector artifacts, evidence reports, `STATUS.md` changes, or M2-T02 work.
- Do not issue arXiv requests or run/download BGE-M3.
- PR #10 remains Draft; the real missing-input CLI remains code `M2_T01_FROZEN_INPUT_MISSING`, exit 2, no output directory.

---

### Task 1: Make paired publication a complete state machine

**Files:**

- Modify: `scripts/freeze_m2_candidates.py`
- Test: `tests/contract/test_m2_frozen_input_replay.py`

**Interfaces:**

- Produces `ExistingPublicationState(snapshot_bytes, manifest_bytes, snapshot_needs_publish, manifest_needs_publish)`.
- Produces `_publish_snapshot_pair(..., expected_candidate_count: int, replace: Callable = os.replace) -> None`.

- [ ] **Step 1: Write red publication-state tests**

```python
_publish_snapshot_pair(snapshot_missing, same_manifest_exists, expected_candidate_count=12)
assert snapshot_path.read_bytes() == snapshot_bytes
assert manifest_path.read_bytes() == manifest_bytes
```

Cover absent/absent, snapshot-only, manifest-only, both-same, either-conflicting, both-absent manifest publish failure, pre-existing snapshot preservation on manifest failure, pre-existing manifest preservation on snapshot failure, staging write/read-back failures, no residue, cross-directory rejection, same-path rejection, and directory targets.

- [ ] **Step 2: Run the red state suite**

Run: `py -3.12 -m pytest tests/contract/test_m2_frozen_input_replay.py tests/contract/test_embedding_contracts.py tests/integration/test_embedding_pipeline.py tests/unit/test_freeze_m2_candidates.py -q`

Expected: FAIL because A3 rolls back a newly created snapshot whenever the manifest was already present.

- [ ] **Step 3: Implement per-run publish tracking**

```python
snapshot_published_this_run = False
manifest_published_this_run = False
if state.snapshot_needs_publish:
    replace(staged_snapshot, snapshot_path)
    snapshot_published_this_run = True
if state.manifest_needs_publish:
    replace(staged_manifest, manifest_path)
    manifest_published_this_run = True
```

Require one resolved parent directory, reject existing directories and byte conflicts, validate staged exact bytes, read back both final targets, and remove only `*_published_this_run` targets after a failure.

- [ ] **Step 4: Run the state suite green**

Run: same command as Step 2.

Expected: all legal state combinations preserve matching existing bytes and every failure returns `M2_T01_SNAPSHOT_PUBLICATION_FAILED` without `.m2-freeze-*` residue.

### Task 2: Require caller-selected counts and closed provenance

**Files:**

- Modify: `scripts/freeze_m2_candidates.py`
- Modify: `scripts/embed_frozen_candidates.py`
- Test: `tests/contract/test_m2_frozen_input_replay.py`
- Test: `tests/contract/test_embedding_contracts.py`
- Test: `tests/integration/test_embedding_pipeline.py`

**Interfaces:**

- Changes `validate_frozen_snapshot_bytes(snapshot_bytes, manifest, *, expected_candidate_count) -> str | None`.
- Changes `load_validated_frozen_inputs(snapshot_path, manifest_path, *, expected_candidate_count=33)`.

- [ ] **Step 1: Write red count and provenance tests**

```python
assert validate_frozen_snapshot_bytes(snapshot_bytes, manifest, expected_candidate_count=12) is None
assert validate_frozen_snapshot_bytes(snapshot_bytes, manifest, expected_candidate_count=33) == (
    "frozen snapshot must contain exactly 33 candidates"
)
assert load_validated_frozen_inputs(snapshot_path, manifest_path).candidate_count == 33
```

Parameterize mutations of completion merge, evidence baseline, validated implementation, ancestry, all source/identity hashes, coverage, metadata mismatch count, transport/cache/query counts, and snapshot/manifest evidence-report label or SHA.

- [ ] **Step 2: Implement closed checks**

```python
if expected_candidate_count < 1:
    return "expected candidate count must be positive"
if len(frozen) != expected_candidate_count or snapshot.get("count") != expected_candidate_count:
    return f"frozen snapshot must contain exactly {expected_candidate_count} candidates"
```

Require matching valid 40-character completion/baseline/implementation commits; non-empty ancestry containing the implementation; valid 64-character source, identity, and snapshot hashes; 1.0 coverage; zero mismatch; and exactly 12 zero-transport replay queries.

- [ ] **Step 3: Run count/provenance tests green**

Run: same focused command.

Expected: 12-candidate synthetic snapshots pass only with 12; production loaders reject them with 33; every cross-field mutation fails.

### Task 3: Record the actual accepted report label

**Files:**

- Modify: `scripts/freeze_m2_candidates.py`
- Test: `tests/contract/test_m2_frozen_input_replay.py`

**Interfaces:**

- Produces `_repository_relative_posix_path(path: Path) -> str`.
- Extends `_replay_and_project(..., source_evidence_report: str)` and `freeze_candidates(..., source_evidence_report_label: str | None = None)`.

- [ ] **Step 1: Write red dynamic-path tests**

```python
result = freeze_candidates(..., source_evidence_report_label="evaluation/reports/synthetic-m1-validation.json")
assert snapshot["source_evidence_report"] == "evaluation/reports/synthetic-m1-validation.json"
assert manifest["source_evidence_report"] == snapshot["source_evidence_report"]
```

Cover default production label, custom legal label, report-label mismatch, report-SHA mismatch, rejected absolute label, and Windows backslash normalization.

- [ ] **Step 2: Implement canonical labels**

```python
relative = path.resolve().relative_to(ROOT.resolve())
return relative.as_posix()
```

Production derives only a repository-relative label. Synthetic tests supply an explicit non-absolute normalized label; both snapshot and manifest receive the same label and evidence report SHA.

- [ ] **Step 3: Run dynamic-path tests green**

Run: same focused command.

Expected: snapshots never contain an absolute Windows path and all report-label cross-checks pass.

### Task 4: Validate and publish one A4 commit

**Files:**

- Create: `docs/superpowers/plans/2026-07-28-m2-t01r3-publication-boundaries.md`
- Modify: only Task 1-3 files and their listed tests.

- [ ] **Step 1: Run required checks**

Run: focused A4 suite; prescribed M2-T01 suite; `py -3.12 -m pytest -q`; Ruff; mypy; docs; M0/M1/M2-T01 validators; and `pip check`.

Expected: each exits 0.

- [ ] **Step 2: Verify missing-input fail-close**

Run: the task-provided A4 missing-input CLI command.

Expected: JSON error code `M2_T01_FROZEN_INPUT_MISSING`, exit 2, and no output directory.

- [ ] **Step 3: Commit and push only A4**

```powershell
git add scripts/freeze_m2_candidates.py scripts/embed_frozen_candidates.py tests/contract/test_m2_frozen_input_replay.py tests/contract/test_embedding_contracts.py tests/integration/test_embedding_pipeline.py tests/unit/test_freeze_m2_candidates.py docs/superpowers/plans/2026-07-28-m2-t01r3-publication-boundaries.md
git commit -m "fix: close M2 freeze publication edge cases"
git push origin agent/m2-t01-embedding-provider
```

Update existing PR #10 with the A4 SHA and CI state while retaining Draft. Stop as soon as A4 CI is green.

## Self-Review

- Covers the four authorized repairs: one-sided publication, production count gate, provenance cross-validation, and dynamic report paths.
- Covers every requested red test and all publication state-matrix rollback cases.
- Does not claim a re-baseline, recovered historical artifact, B1/B2/C, BGE result, or M2 completion.
