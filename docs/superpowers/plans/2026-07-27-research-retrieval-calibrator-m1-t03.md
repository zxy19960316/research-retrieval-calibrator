# Research Retrieval Calibrator M1-T03 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deterministically normalize source-backed `PaperRecord` objects and conservatively deduplicate them while preserving every original record, source identity, and retrieval path, with zero false automatic merges in the frozen negative fixture.

**Architecture:** Add immutable derived normalization and deduplication contracts alongside the existing source-backed `PaperRecord` model. A dependency-free normalization module canonicalizes only values already present in each record; a deterministic deduplication module evaluates stable record pairs through strict identity and similarity tiers, collects auditable decisions, and materializes stable clusters without mutating inputs. Fixtures and tests make the false-positive boundary, manual-review boundary, and input-order invariants executable.

**Tech Stack:** Python 3.12 standard library (`unicodedata`, `re`, `difflib`, `hashlib`), Pydantic v2, pytest, Ruff, mypy.

## Global Constraints

- Work only on M1-T03; do not add M1-T04 CLI, M2 ranking, embeddings, rerankers, LLM judgments, databases, network requests, CNKI, or deployment.
- Do not use `sentence-transformers`, an embedding API, LLM judgment, network translation, Crossref/Semantic Scholar, `fuzzywuzzy`, or any new dependency.
- Preserve `PaperRecord` fields and objects unchanged; every normalized value is a derived field in `NormalizedPaper`.
- Give every automatic merge a machine-readable `DedupDecision`; insufficient evidence must remain separate.
- Exact identity takes precedence over fuzzy comparison, except that contradictory DOI/arXiv identities force manual review.
- Chinese/English possible duplicates may be marked for manual review but must never auto-merge without matching DOI or arXiv identity.
- All cluster membership, decisions, canonical selection, IDs, source identities, and merged retrieval paths must be independent of input order.
- Sort merged retrieval paths lexicographically after removing duplicates; reject no valid nonblank path and discard no member path.
- Keep the frozen negative fixture at `false_auto_merge_count = 0`.
- Commit implementation/tests/fixtures/this plan as implementation commit A; do not include `STATUS.md` or final evidence in A.
- Record evidence and `STATUS.md` only in commit B, and make B reference the full hash of A.

---

## File Structure

- Create `app/models/dedup.py`: Pydantic contracts for normalized records, source identities, reasons, decisions, clusters, and a complete deduplication result.
- Create `app/core/paper_normalization.py`: pure DOI, arXiv, title, author, language, and record normalization functions; no I/O or mutation.
- Create `app/core/paper_dedup.py`: deterministic pair classification, metrics, conservative unioning, canonical record selection, and cluster materialization.
- Create `tests/fixtures/dedup/positive.json`: source-backed successful merge cases and expected counts.
- Create `tests/fixtures/dedup/negative.json`: frozen non-merge cases with `false_auto_merge_count: 0`.
- Create `tests/fixtures/dedup/manual_review.json`: conservative manual-review cases.
- Create `tests/unit/test_paper_normalization.py`: focused normalization and non-mutation behavior.
- Create `tests/unit/test_paper_dedup.py`: decisions, cluster contents, path/source preservation, determinism, and idempotence.
- Create `tests/contract/test_m1_t03_contracts.py`: exact public model shapes and fixture acceptance boundary.
- Create `evaluation/reports/m1-t03-normalization-dedup.json`: post-implementation validation evidence that references A.
- Modify `STATUS.md`: only after all validation succeeds, move M1 to `IN_PROGRESS 3/4` and set the next action to M1-T04 while retaining the unmet M1-T04 live-success gate.

### Task 1: Add the derived contracts and red tests

**Files:**
- Create: `app/models/dedup.py`
- Create: `tests/contract/test_m1_t03_contracts.py`
- Create: `tests/unit/test_paper_normalization.py`
- Create: `tests/unit/test_paper_dedup.py`

**Interfaces:**
- Consumes: `app.models.paper.PaperRecord`.
- Produces: `SourceIdentity(source, source_id, url)`, `NormalizedPaper(record, canonical_doi, canonical_arxiv_id, normalized_title, title_tokens, normalized_authors)`, `DedupReason`, `DedupDecision`, `DedupCluster`, and `DeduplicationResult(clusters, decisions)`.
- Produces: `DedupDecision.action` constrained to `auto_merge`, `manual_review`, or `keep_separate`; `DedupReason` includes the six specified reasons plus `IDENTITY_CONFLICT` for contradictory otherwise-exact DOI/arXiv evidence.

- [ ] **Step 1: Write failing contract and behavior tests**

```python
def test_normalized_paper_keeps_original_record_and_only_derived_fields() -> None:
    record = _paper(paper_id="p-1", doi="doi:10.1000/XYZ")
    normalized = normalize_paper(record)
    assert normalized.record is record
    assert normalized.canonical_doi == "10.1000/xyz"
    assert normalized.model_dump(exclude={"record"}) == {
        "canonical_doi": "10.1000/xyz",
        "canonical_arxiv_id": None,
        "normalized_title": "graph based retrieval a study",
        "title_tokens": ("graph", "based", "retrieval", "a", "study"),
        "normalized_authors": ("ada author",),
    }


def test_decision_and_cluster_contracts_forbid_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        DedupDecision(
            left_paper_id="p-1", right_paper_id="p-2", action="auto_merge",
            reason=DedupReason.EXACT_DOI, title_similarity=None,
            author_overlap=None, year_difference=None, unexpected=True,
        )
```

- [ ] **Step 2: Run the focused tests to verify a real red light**

Run:

```powershell
py -3.12 -m pytest tests/unit/test_paper_normalization.py tests/unit/test_paper_dedup.py tests/contract/test_m1_t03_contracts.py -q
```

Expected: collection/import failure because `app.models.dedup`, `normalize_paper`, and `deduplicate_papers` do not exist. Record the exact failing-test count and exit code for evidence.

- [ ] **Step 3: Implement strict Pydantic contracts**

```python
class DedupReason(str, Enum):
    EXACT_DOI = "EXACT_DOI"
    EXACT_ARXIV_ID = "EXACT_ARXIV_ID"
    EXACT_NORMALIZED_TITLE_WITH_AUTHOR = "EXACT_NORMALIZED_TITLE_WITH_AUTHOR"
    HIGH_TITLE_SIMILARITY_WITH_AUTHOR = "HIGH_TITLE_SIMILARITY_WITH_AUTHOR"
    CROSS_LANGUAGE_POSSIBLE_DUPLICATE = "CROSS_LANGUAGE_POSSIBLE_DUPLICATE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    IDENTITY_CONFLICT = "IDENTITY_CONFLICT"


class DedupDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    left_paper_id: str = Field(min_length=1)
    right_paper_id: str = Field(min_length=1)
    action: Literal["auto_merge", "manual_review", "keep_separate"]
    reason: DedupReason
    title_similarity: float | None = Field(default=None, ge=0, le=1)
    author_overlap: float | None = Field(default=None, ge=0, le=1)
    year_difference: int | None = Field(default=None, ge=0)
```

Give every model `ConfigDict(extra="forbid")`; make `SourceIdentity` nonblank and `DedupCluster` require nonempty members, nonblank sorted unique retrieval paths, and `canonical_record` that is also retained in `member_records`.

- [ ] **Step 4: Re-run the focused tests and static checks**

Run:

```powershell
py -3.12 -m pytest tests/contract/test_m1_t03_contracts.py -q
py -3.12 -m ruff check app/models/dedup.py tests/contract/test_m1_t03_contracts.py
py -3.12 -m mypy app/models/dedup.py tests/contract/test_m1_t03_contracts.py
```

Expected: all pass; later behavioral tests remain red until Tasks 2 and 3.

### Task 2: Implement immutable deterministic normalization

**Files:**
- Create: `app/core/paper_normalization.py`
- Modify: `tests/unit/test_paper_normalization.py`

**Interfaces:**
- Consumes: `PaperRecord` and only its existing `source`, `source_id`, `title`, `authors`, `doi`, and `language` fields.
- Produces: `canonicalize_doi(value: str | None) -> str | None`, `canonicalize_arxiv_id(record: PaperRecord) -> str | None`, `normalize_title(value: str) -> tuple[str, tuple[str, ...]]`, `normalize_author(value: str) -> str`, and `normalize_paper(record: PaperRecord) -> NormalizedPaper`.

- [ ] **Step 1: Add failing normalization tests**

```python
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("10.1000/XYZ", "10.1000/xyz"),
        ("doi:10.1000/xyz", "10.1000/xyz"),
        ("https://doi.org/10.1000/XYZ", "10.1000/xyz"),
        ("not a doi", None),
    ],
)
def test_canonicalize_doi_never_guesses(raw: str, expected: str | None) -> None:
    assert canonicalize_doi(raw) == expected


def test_title_and_authors_use_nfkc_and_preserve_meaningful_unicode() -> None:
    assert normalize_title(" Graph-Based Retrieval: A Study ")[0] == normalize_title(
        " graph based retrieval — a study "
    )[0]
    assert normalize_author("Ａda,  Author") == "ada author"


def test_arxiv_identity_requires_confirmed_arxiv_source_and_removes_only_version() -> None:
    assert canonicalize_arxiv_id(_paper(source="arxiv", source_id="math.GT/0309136v2")) == "math.GT/0309136"
    assert canonicalize_arxiv_id(_paper(source="crossref", source_id="2401.00001v2")) is None
```

- [ ] **Step 2: Run normalization tests and record the red light**

Run:

```powershell
py -3.12 -m pytest tests/unit/test_paper_normalization.py -q
```

Expected: FAIL because the pure normalization functions are absent or incomplete.

- [ ] **Step 3: Implement pure canonicalization functions**

Use `unicodedata.normalize("NFKC", value)`, `strip()`, `casefold()`, and compiled `re` patterns. Strip only the explicit DOI prefixes, then terminal citation punctuation; require a conservative DOI shape beginning with `10.` and a nonblank suffix, returning `None` on invalid input. Do not inspect title, abstract, or URL to manufacture a DOI.

For arXiv, return an ID only when `record.source == "arxiv"`; accept the M1-T02 modern and canonical legacy forms and strip a terminal `v1+`, preserving source-case rules already enforced by the adapter. Do not parse title text.

For title and author values, normalize NFKC, whitespace, and case; translate dash/punctuation runs into token boundaries; retain Unicode letters, numbers, and meaningful non-ASCII characters; neither translate, stem, reorder, nor infer names. Build tuples in source order after omitting blank normalized authors. `normalize_paper` must only retain the original `record` reference and derived values.

- [ ] **Step 4: Prove green behavior and immutability**

Run:

```powershell
py -3.12 -m pytest tests/unit/test_paper_normalization.py -q
py -3.12 -m ruff check app/core/paper_normalization.py tests/unit/test_paper_normalization.py
py -3.12 -m mypy app/core/paper_normalization.py tests/unit/test_paper_normalization.py
```

Expected: all pass; add an assertion comparing `record.model_dump()` before and after normalization.

### Task 3: Implement conservative pair decisions and stable clusters

**Files:**
- Create: `app/core/paper_dedup.py`
- Modify: `tests/unit/test_paper_dedup.py`

**Interfaces:**
- Consumes: `Sequence[PaperRecord]` through `deduplicate_papers(records: Sequence[PaperRecord]) -> DeduplicationResult`.
- Produces: every sorted pair's `DedupDecision`, auto-merge clusters, and non-mutating singleton clusters.
- Produces: module constants `TITLE_TOKEN_JACCARD_THRESHOLD = 0.92`, `TITLE_SEQUENCE_RATIO_THRESHOLD = 0.96`, and `AUTHOR_JACCARD_THRESHOLD = 0.50`.

- [ ] **Step 1: Add failing decision and determinism tests**

```python
def test_exact_doi_and_arxiv_merge_and_preserve_all_paths_and_sources() -> None:
    result = deduplicate_papers([_paper("b", doi="doi:10.1000/X", paths=["Q2"]), _paper("a", doi="10.1000/x", paths=["Q1", "Q2"])])
    cluster = result.clusters[0]
    assert cluster.retrieval_paths == ["Q1", "Q2"]
    assert {(identity.source, identity.source_id) for identity in cluster.source_identities} == {("crossref", "a"), ("arxiv", "b")}
    assert cluster.merge_reasons[0].reason is DedupReason.EXACT_DOI


def test_equal_title_without_author_overlap_never_auto_merges() -> None:
    decision = deduplicate_papers([_paper("a", authors=["Ada"]), _paper("b", authors=["Ben"])]) .decisions[0]
    assert (decision.action, decision.reason) == ("keep_separate", DedupReason.INSUFFICIENT_EVIDENCE)


def test_chinese_english_and_identity_conflicts_need_manual_review() -> None:
    result = deduplicate_papers([_paper("zh", language="zh", authors=["Ada"], year=2024), _paper("en", language="en", authors=["Ada"], year=2024)])
    assert result.decisions[0].action == "manual_review"


def test_reversed_input_has_identical_clusters_decisions_and_ids() -> None:
    records = [_paper("b", doi="10.1000/x"), _paper("a", doi="doi:10.1000/X")]
    assert deduplicate_papers(records) == deduplicate_papers(list(reversed(records)))
```

- [ ] **Step 2: Run pair/cluster tests to prove red**

Run:

```powershell
py -3.12 -m pytest tests/unit/test_paper_dedup.py -q
```

Expected: FAIL because deduplication and cluster materialization do not yet exist.

- [ ] **Step 3: Implement ordered tiers without heuristic expansion**

Sort records by stable `paper_id` before pair generation. For each pair, normalize both and calculate token Jaccard, `difflib.SequenceMatcher(..., autojunk=False).ratio()`, author Jaccard, and absolute year difference (or `None` when either year is absent).

Classify in this exact order: (1) contradictory non-null DOI/arXiv identity evidence => `manual_review/IDENTITY_CONFLICT`; (2) equal valid DOI => `auto_merge/EXACT_DOI`; (3) equal valid arXiv ID => `auto_merge/EXACT_ARXIV_ID`; (4) equal normalized title, an exactly equal author, compatible year, and equal language => `auto_merge/EXACT_NORMALIZED_TITLE_WITH_AUTHOR`; (5) all three named threshold constants, compatible year, and equal language => `auto_merge/HIGH_TITLE_SIMILARITY_WITH_AUTHOR`; (6) cross-language possible duplicate signaled by shared normalized author plus compatible year, high similarity with insufficient authors, or year conflict alongside otherwise close metadata => `manual_review/CROSS_LANGUAGE_POSSIBLE_DUPLICATE`; (7) otherwise `keep_separate/INSUFFICIENT_EVIDENCE`.

Before unioning an auto-merge edge, reject it into the same `IDENTITY_CONFLICT` manual decision if its resulting component would contain more than one distinct non-null DOI or more than one distinct non-null arXiv ID. This prevents transitive false merges.

Materialize each component with all original member records sorted by `paper_id`; choose canonical record by the tuple: valid DOI, nonblank source ID, HTTP(S) URL, nonblank abstract, author count, year present, normalized-title information length, then inverted lexical `paper_id` so the smallest ID wins. Build `cluster_id` as `doi:<doi>`, else `arxiv:<id>`, else `paper:<smallest-paper-id>`. Build unique source identities and nonblank retrieval paths sorted lexicographically. Do not mutate a record or call `model_copy(update=...)` on its fields.

- [ ] **Step 4: Run focused green tests and threshold checks**

Run:

```powershell
py -3.12 -m pytest tests/unit/test_paper_normalization.py tests/unit/test_paper_dedup.py tests/contract/test_m1_t03_contracts.py -q
py -3.12 -m ruff check app/core/paper_normalization.py app/core/paper_dedup.py app/models/dedup.py tests/unit tests/contract/test_m1_t03_contracts.py
py -3.12 -m mypy app/core/paper_normalization.py app/core/paper_dedup.py app/models/dedup.py tests/unit/test_paper_normalization.py tests/unit/test_paper_dedup.py tests/contract/test_m1_t03_contracts.py
```

Expected: all pass. Add tests for exact arXiv with a version suffix, high-similarity author merge, year conflict, only-equal-initial authors, part I/part II, comment/erratum, idempotent rerun, and input record deep snapshots.

### Task 4: Freeze and execute the positive, negative, and manual-review fixtures

**Files:**
- Create: `tests/fixtures/dedup/positive.json`
- Create: `tests/fixtures/dedup/negative.json`
- Create: `tests/fixtures/dedup/manual_review.json`
- Modify: `tests/unit/test_paper_dedup.py`
- Modify: `tests/contract/test_m1_t03_contracts.py`

**Interfaces:**
- Consumes: JSON objects with `records` containing valid `PaperRecord` payloads and `expected` aggregate counts.
- Produces: fixture tests that load each payload with `PaperRecord.model_validate`, deduplicate it, and compare expected decision counts, cluster counts, path preservation, and false automatic merge count.

- [ ] **Step 1: Add failing fixture-driven tests**

```python
@pytest.mark.parametrize("fixture_name", ["positive", "negative", "manual_review"])
def test_frozen_fixture_has_expected_dedup_boundary(fixture_name: str) -> None:
    fixture = _load_fixture(fixture_name)
    before = copy.deepcopy(fixture["records"])
    result = deduplicate_papers([PaperRecord.model_validate(record) for record in fixture["records"]])
    assert _summary(result) == fixture["expected"]
    assert fixture["records"] == before


def test_negative_fixture_has_zero_false_auto_merges() -> None:
    fixture = _load_fixture("negative")
    result = deduplicate_papers([PaperRecord.model_validate(record) for record in fixture["records"]])
    assert _false_auto_merge_count(result, fixture["protected_pairs"]) == 0
```

- [ ] **Step 2: Run fixture tests to prove red**

Run:

```powershell
py -3.12 -m pytest tests/unit/test_paper_dedup.py tests/contract/test_m1_t03_contracts.py -q
```

Expected: FAIL because the three frozen JSON files and expected summaries do not exist.

- [ ] **Step 3: Create complete frozen fixtures**

Use only offline source-backed `PaperRecord` payloads. `positive.json` must cover same arXiv ID through distinct query paths, same DOI across sources, title casing/punctuation variation, high title/author similarity, version-free arXiv identity, and union of multiple paths. `negative.json` must include all ten required non-merge categories: same title/different authors, `A Review`, same authors/different titles, distant years, different DOI, different arXiv IDs, equal initials only, adjacent series, comment/erratum/part I/part II, and same topic/different experimental objects. Set `expected.false_auto_merge_count` to `0` and enumerate every protected pair. `manual_review.json` must cover Chinese/English possible duplicate, high title similarity without authors, DOI/arXiv conflict, and strong year conflict with otherwise similar fields.

- [ ] **Step 4: Verify fixtures, full focused suite, and repeatability**

Run:

```powershell
py -3.12 -m pytest tests/unit/test_paper_normalization.py tests/unit/test_paper_dedup.py tests/contract/test_m1_t03_contracts.py -q
py -3.12 -m pytest tests/unit/test_paper_dedup.py -q --count=2
```

Expected: all pass. If `pytest-repeat` is unavailable, replace the second command with the explicit two-call idempotence test and do not add the dependency.

### Task 5: Commit A, generate evidence, update status, and commit B

**Files:**
- Create: `evaluation/reports/m1-t03-normalization-dedup.json`
- Modify: `STATUS.md`

**Interfaces:**
- Consumes: implementation commit A, recorded red/green command results, fixture summaries, and SHA-256 of each fixture input.
- Produces: machine-readable M1-T03 evidence with `validated_implementation_commit` set to A and status transition to M1 `IN_PROGRESS 3/4`.

- [ ] **Step 1: Commit implementation A only after focused tests pass**

Run:

```powershell
git status --short
git add app/models/dedup.py app/core/paper_normalization.py app/core/paper_dedup.py tests/fixtures/dedup tests/unit/test_paper_normalization.py tests/unit/test_paper_dedup.py tests/contract/test_m1_t03_contracts.py docs/superpowers/plans/2026-07-27-research-retrieval-calibrator-m1-t03.md
git commit -m "feat: implement M1 paper normalization and deduplication"
git rev-parse HEAD
```

Expected: A contains only implementation, tests, fixtures, and plan; it excludes `STATUS.md` and the evidence report.

- [ ] **Step 2: Run the required final validation commands against A**

Run:

```powershell
py -3.12 -m pytest tests/unit/test_paper_normalization.py tests/unit/test_paper_dedup.py tests/contract/test_m1_t03_contracts.py -q
py -3.12 -m pytest -q
py -3.12 -m ruff check app evaluation scripts tests
py -3.12 -m mypy app evaluation scripts
py -3.12 scripts/validate_project_docs.py
py -3.12 scripts/validate_phase.py M0
py -3.12 -m pip check
```

Expected: every command exits 0. Do not run the arXiv smoke command: set `real_external` to `not_run` with reason `M1-T03 is deterministic offline normalization and deduplication`.

- [ ] **Step 3: Write reproducible evidence and update only task state**

Write an `evaluation/reports/m1-t03-normalization-dedup.json` conforming to the existing evidence-report schema. Include `task_id: "M1-T03"`, `phase: "M1"`, the full A hash, UTC time, Python/dependency versions, every red/green command and exit code, focused/full test counts, Ruff/mypy/docs/M0/pip results, fixture input SHA-256 values, positive/negative/manual aggregate summaries, input/cluster/auto/manual/separate counts, `false_auto_merge_count`, retrieval-path preservation, deterministic rerun, input-order invariance, input non-mutation, and the explicit `real_external: "not_run"` boundary.

Update `STATUS.md` to M0 `COMPLETE 4/4`, M1 `IN_PROGRESS 3/4`, M1-T01/T02/T03 complete, M1-T04 not started, M2 blocked by M1, and next action M1-T04. Retain the exact statement that M1-T04's live-success gate is unsatisfied.

- [ ] **Step 4: Commit B and perform final clean-tree validation**

Run:

```powershell
git add evaluation/reports/m1-t03-normalization-dedup.json STATUS.md
git commit -m "chore: record M1-T03 validation evidence"
git rev-parse HEAD
git status --short
```

Expected: B contains only the evidence report and `STATUS.md`; its evidence references A; `git status --short` is empty.

### Task 6: Push and open the required draft PR

**Files:**
- No additional files.

**Interfaces:**
- Consumes: pushed A and B with a clean `agent/m1-t03-normalization-dedup` branch.
- Produces: one Draft PR against `main` titled `feat: implement M1 paper normalization and deduplication`.

- [ ] **Step 1: Push the two intentional commits**

Run:

```powershell
git push
git status --short --branch
```

Expected: branch tracks `origin/agent/m1-t03-normalization-dedup` with no uncommitted changes.

- [ ] **Step 2: Create Draft PR with complete boundary evidence**

Create a Draft PR with head `agent/m1-t03-normalization-dedup`, base `main`, and title `feat: implement M1 paper normalization and deduplication`. Its body must state the main baseline `3bb5da48912b43a2cc76ec8dd82ec553c66d549d`, commits A/B, DOI/title/author rules, tier order, manual-review rules, frozen false-auto-merge count, path preservation, order invariance, focused/full test counts, `real_external = not_run`, M1 `IN_PROGRESS 3/4`, and rollback commands:

```powershell
git revert <B>
git revert <A>
```

- [ ] **Step 3: Verify the published PR and CI state**

Run:

```powershell
gh pr view <number> --json number,isDraft,headRefName,baseRefName,headRefOid,url,statusCheckRollup
git status --short
```

Expected: the PR is Draft, targets `main`, points to B, and the worktree is clean. Do not begin M1-T04 before this PR's final review and merge.

## Self-Review

1. **Spec coverage:** Tasks 1-4 cover derived contracts, canonicalization, all six automatic/manual/separate tiers, stable canonical/cluster/path behavior, all required fixtures, red lights, non-mutation, order invariance, and idempotence. Task 5 isolates A from evidence/status B and records every required check plus the explicit offline boundary. Task 6 creates and verifies the required Draft PR without beginning M1-T04.
2. **Placeholder scan:** No task uses TODO/TBD or defers behavior; threshold values, data shapes, commands, expected results, and commit boundaries are concrete.
3. **Type consistency:** `normalize_paper` returns `NormalizedPaper`; `deduplicate_papers` consumes `Sequence[PaperRecord]` and returns `DeduplicationResult`; its `DedupDecision` and `DedupCluster` use the contracts defined in Task 1.

## M1-T03R1 integration-safe conservative clustering

**Goal:** Repair integration boundaries exposed by repeated arXiv observations without relaxing the M1-T03 conservative false-merge contract.

### Required behavior

1. Before pairwise clustering, deterministically coalesce records with the same `paper_id` only when `paper_id`, source, canonical source ID, normalized title and abstract, normalized authors, year, canonical DOI, URL, language, and `user_visible` agree. The only permitted difference is `retrieval_paths`; emit a deep-copied derived observation with lexicographically sorted unique paths and never mutate either input record.
2. If a same-`paper_id` group differs on any protected field, fail closed with `PaperDeduplicationError("DUPLICATE_PAPER_ID_CONFLICT")`. Do not choose a record, fill metadata, or silently overwrite a source identity.
3. `DeduplicationResult` validates globally unique `cluster_id` values. Cluster generation must avoid conflicts: clusters involved in an `IDENTITY_CONFLICT` use `paper:<minimum-paper-id>` instead of a disputed DOI/arXiv identity.
4. DOI/arXiv exact identity may remain transitive. Before unioning an automatic title/author similarity edge, however, apply a complete-link check: every cross-component pair must already be `auto_merge` and have no identity conflict. Otherwise preserve the existing components and downgrade the candidate edge to `manual_review/TRANSITIVE_BRIDGE_RISK`.

### Acceptance tests and fixtures

- Recorded `ArxivAdapter` output from two query IDs must flow directly into `deduplicate_papers()` as repeated `arxiv:<id>` observations, producing one cluster with both paths, no input mutation, order invariance, and idempotence.
- Cover duplicate IDs with differing source ID, DOI, title, and URL; only retrieval-path differences may coalesce.
- Cover same DOI with different arXiv IDs, requiring a manual `IDENTITY_CONFLICT`, two stable distinct fallback IDs, and no DOI-derived collision.
- Cover high-title and shared-author three-record bridges. Each must retain a two-record cluster plus singleton rather than merge all three, and label the blocked edge `TRANSITIVE_BRIDGE_RISK`.
- Keep `false_auto_merge_count = 0`, add repeated arXiv `paper_id` observations to the positive fixture, and record the additional coalescing, conflict, unique-ID, and bridge statistics in refreshed M1-T03 evidence.

## M1-T03R2 deterministic complete-link closure

**Goal:** Make title/author complete-link clustering independent of pair-processing order while retaining conservative rejection of genuine transitive bridges and deterministic selection of a repeated-observation representative.

**Scope:** Modify `app/core/paper_dedup.py`, `app/models/dedup.py`, focused unit/contract tests, the dedup fixtures when aggregate counts change, this plan, then refresh only `evaluation/reports/m1-t03-normalization-dedup.json` and `STATUS.md` in a separate evidence commit. Do not start M1-T04 or run any real network request.

### Task R2.1: Capture the order-dependent failures

- [ ] Add tests for a three-record exact-title-and-author clique and a three-record high-similarity clique. Each test must assert one three-member cluster, exactly three automatic pair decisions, reverse-input equality, and repeat-run equality.
- [ ] Retain the existing two-auto-edge/one-rejected-edge bridge tests and assert the blocked candidate remains `manual_review/TRANSITIVE_BRIDGE_RISK` with a 2+1 clustering result.
- [ ] Add a same-`paper_id` observation test whose title, author casing, and abstract whitespace differ only in normalization-equivalent ways. Assert forward and reverse calls return byte-for-byte equal Pydantic results, the chosen canonical raw record is stable, retrieval paths are sorted/complete, and both inputs are unchanged.
- [ ] Run the focused suite before implementation and record the expected failures for the two cliques and representative-order test.

### Task R2.2: Precompute base decisions and materialize deterministic clusters

- [ ] In `deduplicate_papers`, normalize all coalesced records and precompute `_classify_pair(left, right, config)` for every stable unique paper-ID pair before any union. Keep the mapping keyed by sorted `(left_paper_id, right_paper_id)` and never rewrite it.
- [ ] Process exact DOI, exact arXiv, exact normalized title/author, then high-similarity title/author edges. Exact identity edges retain the existing component identity-conflict guard.
- [ ] For a title/author candidate, inspect every cross-component pair in the immutable base-decision mapping. Union only when all cross pairs are `auto_merge` and none is an identity conflict; otherwise publish a final `manual_review/TRANSITIVE_BRIDGE_RISK` for the candidate edge without changing its base decision.
- [ ] Emit one final decision per pair in stable pair order. Preserve base decisions for accepted clique edges; use only the bridge downgrade where a candidate edge cannot satisfy complete-link closure.
- [ ] Select the repeated-observation reference using the lexicographically minimal JSON representation of `record.model_dump(mode="json", exclude={"retrieval_paths"})`, with `ensure_ascii=False`, sorted keys, and compact separators. Keep `_observation_key()` solely as the semantic coalescing eligibility check.

### Task R2.3: Harden result invariants and verify

- [ ] Validate unique cluster member paper IDs in `DedupCluster` and unique, stably sorted pair identifiers in `DeduplicationResult`; retain the existing unique cluster-ID and sorted unique retrieval-path checks.
- [ ] Add contract tests for these rejected model shapes and focused tests for deterministic decision order.
- [ ] Run the focused tests, full regression, Ruff, mypy, project-doc validation, historical M0 validation, and `pip check`. Record actual test totals, fixture SHA-256 values, clique/bridge counts, and offline boundaries.
- [ ] Commit implementation/tests/fixtures/plan as `fix: make M1 complete-link clustering deterministic`; then commit only evidence plus `STATUS.md` as `chore: refresh M1-T03 deterministic evidence`, with the evidence report referencing the first full hash.
