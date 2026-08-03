# M2-T03 Evidence Slot Classification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a closed, deterministic, title/abstract-only evidence-slot classification contract and a fixed 33-candidate M2-T03 runner with auditable, atomic, privacy-safe evidence.

**Architecture:** Keep M2-T03 independent from reranker scores and selection logic. Pydantic models define the source hash, classifier descriptor, closed enum/state, grounded excerpt, and batch invariants; an adapter Protocol separates providers from core orchestration; a deterministic fake is the only executable provider in this task. The runner verifies immutable M1/M2-T02 input bytes and paper-id order, classifies only normalized title/abstract inputs, and publishes a closed artifact through a sibling temporary file and `os.replace`.

**Tech Stack:** Python 3.12, Pydantic 2.9+, `enum.StrEnum`, pytest, mypy, Ruff, JSON, `hashlib`, `tempfile`, and `os.replace`.

## Global Constraints

- Execute only `M2-T03：证据槽位分类`; do not implement M2-T04 six-item scoring, M2-T05 diversity selection, feedback, second-round retrieval, or modify M2-T01/M2-T02 formal results.
- Classification inputs are only `paper_id`, normalized `title`, normalized optional `abstract`, the exact source-text SHA-256, and the required classifier descriptor/version.
- Never read or use paper full text, citations, author reputation, journal rank, reranker score, dense score, final selection, user feedback, or network supplementation.
- Reuse `EvidenceSlot` and `SupportLevel` from `app.models.enums`; unknown values fail closed.
- All Pydantic contracts use `extra="forbid"`; trimmed text must be non-empty; reasons and excerpts are bounded; excerpts must be exact title/abstract substrings; source hashes must be recomputable.
- `INDIRECT` reasons must state that the evidence is indirect transfer evidence and still needs validation in the target problem; hypothetical output must not claim verified results.
- Real model execution is not approved in this repository state; deterministic fake evidence, real-model evidence, and human review remain separate, with `human_judged = false`.
- Fixed evidence artifacts must use relative POSIX paths, SHA-256 input binding, no credentials/absolute paths/raw exception text, and sibling-temp-file plus `os.replace` publication with byte-identical reuse and conflict rejection.

---

### Task 1: Define failing M2-T03 contracts and provider-boundary tests

**Files:**
- Create: `tests/contract/test_m2_t03_evidence_classification.py`
- Create: `tests/unit/test_m2_t03_evidence_classification_provider.py`

**Interfaces:**
- Consumes: the existing `EvidenceSlot`, `SupportLevel`, Pydantic, and Protocol conventions.
- Produces: executable red tests defining the public M2-T03 model names, deterministic fake behavior, fail-closed error codes, input-order invariance, provider failure behavior, and forbidden score/selection fields.

- [ ] **Step 1: Write the failing model-contract tests**

  Cover `EvidenceClassificationInput`, `EvidenceClassifierDescriptor`, `EvidenceClassificationRecord`, `EvidenceClassificationBatch`, `EvidenceClassificationError`, and `EvidenceClassifierProvider`. Include tests for direct, indirect, hypothetical, title-only, insufficient source text, unknown enums, blank reason, non-source excerpts, ungrounded reasons, duplicate/conflicting paper IDs, source hash recomputation, and `extra="forbid"`.

  Use a fixed helper shape so later implementation has one unambiguous source serialization:

  ```python
  def make_input(title: str, abstract: str | None) -> EvidenceClassificationInput:
      return EvidenceClassificationInput(
          paper_id="arxiv:test-1",
          title=title,
          abstract=abstract,
          source_text_sha256=source_text_sha256(title, abstract),
          classifier_descriptor=deterministic_fake_descriptor(),
          classification_version=EVIDENCE_CLASSIFICATION_VERSION,
      )
  ```

- [ ] **Step 2: Write the failing provider/core tests**

  Define a fake provider fixture and assert that `classify_evidence_batch` returns sorted, paper-id-keyed results; repeated calls are byte/datum identical; reversed inputs preserve per-paper outputs; provider exceptions become `PROVIDER_UNAVAILABLE`; malformed provider outputs become `INVALID_OUTPUT`; and no output dictionary may contain `score`, `weight`, `total_score`, `selection_rank`, or `rank`.

- [ ] **Step 3: Run only the new tests to verify the red baseline**

  Run:

  ```powershell
  python -m pytest -q tests/contract/test_m2_t03_evidence_classification.py tests/unit/test_m2_t03_evidence_classification_provider.py
  ```

  Expected: collection or test failures because the M2-T03 modules and contracts do not yet exist. Do not weaken the tests to make the baseline pass.

### Task 2: Implement the closed classification models and deterministic fake provider

**Files:**
- Create: `app/models/evidence_classification.py`
- Create: `app/adapters/evidence_classification.py`
- Modify: `app/models/__init__.py` only if the repository’s public model export convention requires it
- Modify: `app/adapters/__init__.py` only to expose the Protocol/fake without importing optional model runtimes

**Interfaces:**
- Consumes: `EvidenceSlot` and `SupportLevel` from `app.models.enums`.
- Produces: `source_text_sha256`, `serialize_source_text`, `EvidenceClassificationInput`, `EvidenceClassifierDescriptor`, `EvidenceClassificationRecord`, `EvidenceClassificationBatch`, `EvidenceClassificationError`, `EvidenceClassifierProvider`, `DeterministicFakeEvidenceClassifier`, and the fixed `EVIDENCE_CLASSIFICATION_VERSION`.

- [ ] **Step 1: Add canonical source normalization and closed enums/states**

  Normalize CRLF/CR to LF and trim outer whitespace. Serialize exactly the normalized title and optional abstract as sorted, compact UTF-8 JSON before hashing. Define explicit states for `CLASSIFIED`, `TITLE_ONLY`, `REJECTED`, and `INSUFFICIENT_SOURCE_TEXT`; no unknown state maps to a default.

- [ ] **Step 2: Add Pydantic descriptor/input/record/batch contracts**

  Use `ConfigDict(extra="forbid", frozen=True)`. Enforce lowercase 64-hex SHA-256 values, exact recomputation from normalized title/abstract, bounded non-empty reason/excerpt fields, exact excerpt membership in title/abstract through a source-validation method, state-consistent nullable fields, one descriptor/version per batch, unique paper IDs, and no score/selection fields.

- [ ] **Step 3: Add stable fail-closed errors and Protocol**

  Expose stable codes for invalid input, provider unavailable, invalid output, duplicate paper ID, source hash mismatch, and result conflict/publication failure. The Protocol accepts a sequence of validated inputs and returns a sequence of structured outputs without importing any model runtime.

- [ ] **Step 4: Add the deterministic fake with fixed descriptor identity**

  Classify only normalized title/abstract text using deterministic keyword/sentence rules. Prefer explicit evaluation/method/implementation/problem/transfer markers for the slot; distinguish direct evidence from indirect transfer wording and hypothetical-only wording; use title-only state only when abstract is absent and the title supports a rule; otherwise return an explicit rejection. Construct reasons through the bounded grounded-reason helper so the excerpt is an exact source substring and indirect reasons include the required validation caveat.

- [ ] **Step 5: Run the contract tests and verify they are green**

  Run the focused command from Task 1. Expected: all new model/provider tests pass while the rest of the repository remains unchanged.

### Task 3: Implement fail-closed core batch orchestration

**Files:**
- Create: `app/core/evidence_classification.py`
- Modify: `tests/contract/test_m2_t03_evidence_classification.py`
- Modify: `tests/unit/test_m2_t03_evidence_classification_provider.py`

**Interfaces:**
- Consumes: the Task 2 contracts and `EvidenceClassifierProvider`.
- Produces: `build_classification_input`, `classify_evidence_batch`, `validate_classification_record`, and deterministic batch serialization helpers.

- [ ] **Step 1: Add a test for canonical input construction from frozen candidates**

  Assert that `FrozenCandidate` title/abstract values are the only source fields used, missing abstracts are not invented, the hash is recomputable, and reranker records/scores are not accepted as function inputs.

- [ ] **Step 2: Add a test for all-or-nothing provider output validation**

  Assert that wrong paper IDs, duplicate IDs, missing records, unknown enums, invalid state fields, invalid excerpts, ungrounded reasons, and mixed descriptors raise stable errors without returning partial records.

- [ ] **Step 3: Implement validation and deterministic ordering**

  Validate all inputs before provider invocation, reject duplicate IDs, canonicalize provider invocation order by `paper_id`, validate every returned record against its corresponding input, reject any set/order mismatch, and return records sorted by `paper_id` with explicit batch state. Never catch an error and synthesize a default slot.

- [ ] **Step 4: Run core-focused tests**

  Run:

  ```powershell
  python -m pytest -q tests/contract/test_m2_t03_evidence_classification.py tests/unit/test_m2_t03_evidence_classification_provider.py
  ```

  Expected: green, including deterministic repeatability and reversed-input invariance.

### Task 4: Add the fixed-input M2-T03 runner and atomic artifact validator

**Files:**
- Create: `scripts/run_m2_t03_evidence_classification.py`
- Create: `scripts/validate_m2_t03_evidence.py`
- Create: `tests/unit/test_m2_t03_evidence_classification_runner.py`
- Create: `tests/contract/test_m2_t03_evidence_classification_evidence.py`

**Interfaces:**
- Consumes: the immutable M1 snapshot/manifest and M2-T02 result/receipt at their committed relative paths; Task 2/3 classification APIs.
- Produces: `run_m2_t03_evidence_classification`, `validate_classification_run_report`, `publish_classification_run`, `EvidenceClassificationRunPaths`, and a closed formal JSON artifact/receipt.

- [ ] **Step 1: Add fixed-input and publication red tests**

  Cover snapshot/manifest/result/receipt SHA drift, paper-id set/order drift, changed reranker score not affecting classification, provider failure leaving no formal artifact, byte-identical `REUSED`, different-content `RESULT_CONFLICT`, sibling temp cleanup, closed JSON fields, safe relative paths, and no absolute paths/raw exceptions/credentials.

- [ ] **Step 2: Implement safe fixed-input loading**

  Read regular files without following links/junctions, verify the exact expected SHA-256 values and closed JSON structures, ensure all 33 fixed IDs and full abstracts are present, validate the M2-T02 result/receipt provenance, and use the M1 manifest order as the only runner order. Read reranker records only to validate identity; never use score/rank values in classification.

- [ ] **Step 3: Implement deterministic-fake execution and report schema**

  Build one input per frozen candidate, invoke the provider through the Protocol, emit a closed run object with input hashes, descriptor/version, records, counts, and explicit evidence type `deterministic_fake`. Keep real model and human evidence out of the formal result rather than substituting fake output for either.

- [ ] **Step 4: Implement sibling-temp plus replace publication**

  Serialize with UTF-8, sorted compact JSON plus one newline; reuse identical bytes without replacing or changing mtime; reject different existing bytes; stage next to the final file; flush/fsync; publish with `os.replace`; and remove only owned temporary files after success or failure.

- [ ] **Step 5: Implement the no-runtime evidence validator**

  Validate the summary report, formal artifact, receipt, all input hashes, distributions, automated check records, `human_judged = false`, real-model `not_run`, and M2/M3 status transition. Keep validation independent from optional torch/transformers imports.

- [ ] **Step 6: Run runner/validator focused tests and the validator command**

  Run:

  ```powershell
  python -m pytest -q tests/contract/test_m2_t03_evidence_classification_evidence.py tests/unit/test_m2_t03_evidence_classification_runner.py
  python scripts/validate_m2_t03_evidence.py
  ```

  Expected: green after the formal fake artifact and summary report are created.

### Task 5: Generate evidence, update status, and verify all repository gates

**Files:**
- Create: `evaluation/source-artifacts/m2-t03-evidence-classification-run.json`
- Create: `evaluation/source-artifacts/m2-t03-evidence-classification-run-receipt.json`
- Create: `evaluation/reports/m2-t03-evidence-classification.json`
- Modify: `STATUS.md` to `M2 IN_PROGRESS 3/5` and next task `M2-T04`; retain `M3 BLOCKED_BY_M2 0/5`

**Interfaces:**
- Consumes: the fixed runner and all focused/full gate outputs.
- Produces: machine-readable fake classification evidence for all 33 candidates, explicit not-run boundaries, and status handoff to M2-T04.

- [ ] **Step 1: Run the controlled deterministic fake over all 33 candidates**

  Run:

  ```powershell
  python scripts/run_m2_t03_evidence_classification.py --execute-deterministic-fake
  ```

  Expected: exactly 33 explicit results, zero rejected records, source hashes/excerpts bound, and no score/selection keys.

- [ ] **Step 2: Capture hashes and machine-readable execution evidence**

  Record task/implementation/validated commits, UTC timestamp, Python/dependency versions, all input artifact paths and hashes, record count, slot/support distributions, title-only/rejected counts, fake/real/human evidence types, command exit codes, test totals, not-run items, `human_judged = false`, and residual risks without credentials, absolute paths, or raw exception text.

- [ ] **Step 3: Run the complete required validation matrix**

  Run the focused suite, non-packaging suite, packaging suite, Ruff and import-order Ruff, mypy, project-doc validation, M0/M1/M2-T01/M2-T02/M2-T03 validators, `pip check`, and `git diff --check`. Preserve any intentional red gate as not-run/failed evidence rather than weakening it.

- [ ] **Step 4: Update STATUS.md only after all M2-T03 gates pass**

  Change only the M2 row, next-task text, and M2-T03 evidence section needed for the direct handoff; do not mark M2 complete and do not alter M1/M2-T01/M2-T02 formal result files.

### Task 6: Commit independently, push, and open the Draft PR

**Files:**
- Git commits only over the files listed in Tasks 1–5.

**Interfaces:**
- Consumes: green tests, validators, evidence hashes, and status transition.
- Produces: independent contract, implementation, runner/evidence, and documentation commits plus a pushed branch and Draft PR titled `feat: classify evidence slots for M2 candidates`.

- [ ] **Step 1: Inspect the final diff and scope**

  Run `git status --short`, `git diff --check`, and `git diff --stat`; verify no M2-T04/M2-T05 files, model weights, secrets, caches, or M2-T01/M2-T02 formal results changed.

- [ ] **Step 2: Create bounded commits without amend/rebase**

  Use messages:

  ```text
  test: define M2-T03 evidence classification contracts
  feat: implement deterministic evidence slot classification
  feat: add controlled M2-T03 classification runner
  chore: record validated M2-T03 evidence
  docs: mark M2-T03 complete
  ```

- [ ] **Step 3: Push the task branch**

  Run `git push -u origin agent/m2-t03-evidence-slot-classification` only after the local gates pass.

- [ ] **Step 4: Create a Draft PR**

  Use title `feat: classify evidence slots for M2 candidates`. The body must start with M2-T03 status, M2 progress `3/5`, next task M2-T04, M3 blocked, current/implementation/evidence commits, input and artifact hashes, test totals, fake/real/human evidence status, and not-run items. Do not auto-merge.

## Self-Review Checklist

- [ ] Every requirement in the task attachment maps to a model, fake/provider test, runner test, validator check, artifact field, or status handoff above.
- [ ] No step introduces `score`, `weight`, `total_score`, `selection_rank`, or diversity-selection behavior.
- [ ] Unknown enums, ungrounded reasons, missing text, provider exceptions, partial results, duplicate IDs, and output conflicts fail closed.
- [ ] Real model execution and human judgment remain explicit `not_run`/`false`; deterministic fake evidence is not presented as real-model or human evidence.
- [ ] The final branch remains limited to M2-T03 and preserves the committed M1/M2-T01/M2-T02 evidence bytes.
