# M2-T01R1 Frozen M1 Input Replay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore a verifiable M1 candidate freeze path that succeeds only with the accepted real M1 output and a complete zero-network real-cache replay.

**Architecture:** The freeze CLI derives its accepted hashes and counts from the immutable M1 evidence report, validates the historical `FirstRoundRun`, and reuses the existing arXiv cache, query-plan, and deduplication components. It writes a deterministic M2 candidate snapshot only after every cache hit, metadata audit, and canonical record projection succeeds; missing artifacts remain a stable no-write failure.

**Tech Stack:** Python 3.12, Pydantic v2, existing arXiv adapter/cache, existing M1 deduplication, pytest, Ruff, mypy.

## Global Constraints

- Never issue an arXiv request during the freeze replay; a transport object raises on every attempted request.
- Accept only `real_external` M1 evidence with 33 candidates and source-ID/URL coverage of `1.0`.
- Reuse `FirstRoundRun.query_plan`, `ArxivAdapter`, and `deduplicate_papers`; do not create a parallel M1 parser or deduplicator.
- Create `evaluation/snapshots/m2/m1-candidates.v1*.json` only after exact artifact recovery; never commit a real cache, vector snapshot, evidence C, or `STATUS.md` update.

---

### Task 1: Evidence-derived provenance

**Files:**
- Modify: `scripts/freeze_m2_candidates.py`
- Test: `tests/contract/test_m2_frozen_input_replay.py`

- [ ] Write red tests for `load_accepted_m1_provenance(Path)` using `evaluation/reports/m1-validation.json`, missing hash fields, altered hashes, and `recorded_external` classifications.
- [ ] Implement an immutable `AcceptedM1Provenance` loader that reads the accepted output/candidate hashes, count, coverage, and baseline commit from the report.
- [ ] Make `freeze_candidates` and `--m1-evidence-report` use that loader by default while preserving explicit synthetic hashes in pure validation tests.
- [ ] Run `py -3.12 -m pytest tests/contract/test_m2_frozen_input_replay.py -q`.

### Task 2: Zero-transport replay and source projection

**Files:**
- Modify: `scripts/freeze_m2_candidates.py`
- Modify: `app/models/first_round.py`
- Modify: `app/core/first_round.py`
- Test: `tests/contract/test_m2_frozen_input_replay.py`

- [ ] Add red synthetic-cache tests for 12 real-namespace cache hits, zero transport, wrong endpoint/schema/namespace, corrupt or missing cache entries, and raw/deduplicated counts.
- [ ] Reuse the stored query plan, `ArxivAdapter`, and `deduplicate_papers`; compare the replayed clusters to the historical visible candidates by all legacy provenance fields.
- [ ] Add only nullable `CandidateOutput.abstract` and project it directly from the canonical source record; normalize blank abstracts to `None` and never synthesize categories.
- [ ] Run the focused replay contracts.

### Task 3: Deterministic snapshot success path

**Files:**
- Modify: `scripts/freeze_m2_candidates.py`
- Test: `tests/contract/test_m2_frozen_input_replay.py`

- [ ] Add red tests proving ordered candidates, byte-identical repeated output, direct/null abstract handling, manifest byte SHA, and atomic paired writes.
- [ ] Implement the success result and deterministic snapshot/manifest writer only after provenance, replay, and metadata checks pass.
- [ ] Keep missing historical artifacts as `M2_T01_FROZEN_INPUT_MISSING` with no output directory creation.

### Task 4: Verify and publish A2 only

**Files:**
- Modify: `docs/superpowers/plans/2026-07-28-m2-t01r1-frozen-input-replay.md`

- [ ] Run the specified focused suite, full regression, Ruff, mypy, project-doc, M0, M1, M2-T01, and pip validators.
- [ ] Re-run the real freeze CLI only if exact historical artifacts are recovered; otherwise preserve the stable missing-input result.
- [ ] Commit only implementation, tests, and this plan as `fix: complete M2 frozen-input replay gate`; do not create B1 when recovery is not exact.
