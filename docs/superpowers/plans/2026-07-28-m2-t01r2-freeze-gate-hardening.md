# M2-T01R2 Freeze Gate Hardening Implementation Plan

> **For agentic workers:** Execute the listed red-green checks inline; the user-authorized delivery boundary is one implementation commit, A3.

**Goal:** Close the M2-T01 frozen-input success path so accepted M1 evidence, a complete cache-only replay, exact-byte snapshot validation, and paired publication are independently testable and fail closed.

**Architecture:** `scripts/freeze_m2_candidates.py` remains the one freeze implementation. It loads explicit M1 completion/evidence/implementation provenance, replays the existing `ArxivAdapter` cache and `deduplicate_papers`, then renders one snapshot before a verified paired publish. Typed `FreezeGateError` codes carry expected failure classification.

**Tech Stack:** Python 3.12, Pydantic v2, pytest, existing M1 arXiv adapter/cache and deduplicator, Ruff, mypy.

## Global Constraints

- A2 head: `43a9fb3028fbc9bb115d78f24a108329cd51069e`; retain `A -> A1 -> A2 -> A3` without amending A2.
- The only production hash source is `evaluation/reports/m1-validation.json`; remove hard-coded M1 output and candidate hashes.
- M1 completion merge is `0eb45fc22d10adb72cb66aa45494333057080bc1`, distinct from evidence baseline `f0f167766589e3321821b0caf7793b00c8ff7291`.
- Synthetic tests may use an explicit lower candidate count; CLI uses the report's accepted count of 33.
- Never issue arXiv requests during replay. Do not create B1/B2/C, real cache data, BGE vectors, M2-T02 code, or a `STATUS.md` change.
- The real missing-artifact CLI remains no-output-dir `M2_T01_FROZEN_INPUT_MISSING`, exit 2.

## File Structure

- `scripts/freeze_m2_candidates.py`: provenance loader, typed errors, replay/projection, byte validator, paired publisher.
- `scripts/embed_frozen_candidates.py`: consume the snapshot's file-byte hash.
- `tests/contract/test_m2_frozen_input_replay.py`: 12-query cache, provenance, error, metadata, bytes, and publication contracts.
- `tests/unit/test_freeze_m2_candidates.py`: pure validation and malformed evidence coverage.
- `tests/contract/test_embedding_contracts.py`: exact snapshot-byte validation compatibility.

---

### Task 1: Explicit M1 provenance and evidence-derived expected hashes

**Files:**

- Modify: `scripts/freeze_m2_candidates.py`
- Test: `tests/contract/test_m2_frozen_input_replay.py`
- Test: `tests/unit/test_freeze_m2_candidates.py`

**Interfaces:**

- Produces `AcceptedM1Provenance(m1_completion_merge_commit, m1_evidence_baseline_commit, validated_implementation_commit, implementation_ancestry, output_sha256, candidate_array_sha256, candidate_count, source_id_coverage, url_coverage)`.
- Produces `validate_m1_freeze_input(raw_output, *, expected_output_sha256, expected_candidate_array_sha256, expected_candidate_count) -> str | None`.

- [ ] **Step 1: Write failing provenance tests**

```python
assert provenance.m1_completion_merge_commit == M1_COMPLETION_MERGE_COMMIT
assert provenance.m1_evidence_baseline_commit == "f0f167766589e3321821b0caf7793b00c8ff7291"
assert provenance.validated_implementation_commit in provenance.implementation_ancestry
with pytest.raises(ValueError, match="SHA-256"):
    load_accepted_m1_provenance(malformed_hash_report)
```

Cover rejected old output hash, missing output/candidate hashes, 64-character non-hex hashes, wrong count, and non-1.0 coverage.

- [ ] **Step 2: Run red tests**

Run: `py -3.12 -m pytest tests/contract/test_m2_frozen_input_replay.py tests/unit/test_freeze_m2_candidates.py -q`

Expected: FAIL before new provenance fields and required count argument exist.

- [ ] **Step 3: Implement the explicit contract**

```python
M1_COMPLETION_MERGE_COMMIT = "0eb45fc22d10adb72cb66aa45494333057080bc1"
def validate_m1_freeze_input(raw_output: bytes, *, expected_output_sha256: str,
                             expected_candidate_array_sha256: str,
                             expected_candidate_count: int) -> str | None: ...
```

Validate 40/64-character hexadecimal values, preserve baseline only as evidence provenance, and use completion merge for the rendered snapshot.

- [ ] **Step 4: Run green tests**

Run: `py -3.12 -m pytest tests/contract/test_m2_frozen_input_replay.py tests/unit/test_freeze_m2_candidates.py -q`

Expected: all selected tests pass.

### Task 2: Full synthetic cache-only replay with stable error mapping

**Files:**

- Modify: `scripts/freeze_m2_candidates.py`
- Test: `tests/contract/test_m2_frozen_input_replay.py`

**Interfaces:**

- Produces `FreezeGateError(code: str, reason: str)` for each declared M2-T01 freeze code.
- Extends `freeze_candidates(..., expected_candidate_count: int | None = None) -> FreezeResult` for explicit synthetic evidence only.

- [ ] **Step 1: Write a failing 12-query success-path test**

```python
result = freeze_candidates(m1_output=output, m1_cache_dir=cache, output_dir=target,
                           m1_evidence_report=synthetic_report)
assert result.status == "success"
assert manifest["zero_transport_replay"] == {
    "cache_hits": 12, "query_count": 12, "transport_requests": 0
}
```

Generate entries through `ArxivAdapter`, then invoke the actual freeze replay, cache reads, deduplication, and projection without mocking those core components.

- [ ] **Step 2: Run red test**

Run: `py -3.12 -m pytest tests/contract/test_m2_frozen_input_replay.py -q`

Expected: FAIL because errors are inferred from text and success-path coverage is incomplete.

- [ ] **Step 3: Implement typed errors and exact projection**

```python
raise FreezeGateError("M2_T01_REAL_CACHE_INCOMPLETE", "cache entry missing for query ...")
except FreezeGateError as error:
    return FreezeResult(status="blocked", error_code=error.code, reason=error.reason)
```

Map absent cache to incomplete; bad namespace/schema/endpoint or corrupt JSON to provenance mismatch; invoked transport to zero-transport failure. Parameterize all 12 legacy metadata fields, copy abstract directly with whitespace normalized to `None`, and set categories to `[]`.

- [ ] **Step 4: Run green replay suite**

Run: `py -3.12 -m pytest tests/contract/test_m2_frozen_input_replay.py -q`

Expected: 12 hits, zero transport, raw/dedup checks, metadata mismatch, and cache-provenance cases pass.

### Task 3: Exact-byte SHA and paired publication

**Files:**

- Modify: `scripts/freeze_m2_candidates.py`
- Modify: `scripts/embed_frozen_candidates.py`
- Test: `tests/contract/test_m2_frozen_input_replay.py`
- Test: `tests/contract/test_embedding_contracts.py`

**Interfaces:**

- Produces `validate_frozen_snapshot_bytes(snapshot_bytes: bytes, manifest: Mapping[str, object]) -> str | None`.
- Produces `_publish_snapshot_pair(snapshot_path, snapshot_bytes, manifest_path, manifest_bytes) -> None`.

- [ ] **Step 1: Write failing byte and publication tests**

```python
assert validate_frozen_snapshot_bytes(snapshot_bytes + b" ", manifest) == SNAPSHOT_INVALID
with pytest.raises(FreezeGateError, match="M2_T01_SNAPSHOT_PUBLICATION_FAILED"):
    _publish_snapshot_pair(...)
assert not any(path.rglob("*.tmp"))
```

Cover snapshot/manifest temporary write failure, manifest publication failure after snapshot publication, rollback, same-byte idempotence, conflict rejection, no residue, alternate formatting rejection, and byte-identical repeats.

- [ ] **Step 2: Run red tests**

Run: `py -3.12 -m pytest tests/contract/test_m2_frozen_input_replay.py tests/contract/test_embedding_contracts.py -q`

Expected: FAIL because canonical-object hash and independent writes are still accepted.

- [ ] **Step 3: Implement the one byte contract and pair publisher**

```python
actual = hashlib.sha256(snapshot_bytes).hexdigest()
if manifest.get("snapshot_sha256") != actual:
    return "frozen snapshot manifest SHA-256 does not match exact snapshot bytes"
```

Stage and read-back verify both files, reject conflicting existing targets, restore any target changed by failed publish, and remove all temporary/staging artifacts.

- [ ] **Step 4: Run green tests**

Run: `py -3.12 -m pytest tests/contract/test_m2_frozen_input_replay.py tests/contract/test_embedding_contracts.py -q`

Expected: only exact rendered bytes validate and paired publication is atomic from the caller's visible state.

### Task 4: Validate and publish the sole A3 commit

**Files:**

- Modify: only Task 1-3 files plus this plan; exclude `STATUS.md`, snapshots, vectors, evidence reports, and M2-T02 files.

- [ ] **Step 1: Run all specified verification**

Run: focused freeze tests, prescribed M2-T01 tests, full `pytest -q`, Ruff, mypy, project-doc, M0/M1/M2-T01 validators, and `pip check`.

Expected: every command exits 0.

- [ ] **Step 2: Confirm missing-input behavior**

Run: `py -3.12 scripts/freeze_m2_candidates.py --m1-evidence-report evaluation/reports/m1-validation.json --m1-output evaluation/runs/m1-live-r1-final/first-round.json --m1-cache-dir evaluation/runs/m1-t04r1-cache-final --output-dir evaluation/snapshots/m2/a3-missing-input-check`

Expected: `M2_T01_FROZEN_INPUT_MISSING`, exit 2, and no output directory.

- [ ] **Step 3: Commit and update the existing draft PR**

```powershell
git add scripts/freeze_m2_candidates.py scripts/embed_frozen_candidates.py tests/contract/test_m2_frozen_input_replay.py tests/unit/test_freeze_m2_candidates.py tests/contract/test_embedding_contracts.py docs/superpowers/plans/2026-07-28-m2-t01r2-freeze-gate-hardening.md
git commit -m "fix: harden M2 frozen-input success path"
git push origin agent/m2-t01-embedding-provider
```

Update PR #10 with the A3 SHA and verification summary while retaining Draft state. Stop after CI is green; do not re-search artifacts, create B1/B2/C, run BGE/arXiv, change status, or start M2-T02.

## Self-Review

- The plan closes all six audit blockers: provenance, default hashes, synthetic success replay, stable errors, paired publication, and one byte-hash convention.
- Each task supplies its files, interfaces, test code, commands, and expected result.
- The single A3 commit boundary deliberately overrides the skill's usual frequent-commit recommendation.
