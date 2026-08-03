# M2-T02 RerankerProvider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a replaceable, fail-closed reranker orchestration boundary for 50-100 frozen candidates without running or downloading a real model in its contract stage.

**Architecture:** Keep reranker models and validation in `app/models/reranking.py`, provider protocol and eventual adapter implementations in `app/adapters/reranking.py`, and deterministic batching/cache/normalization orchestration in `app/core/reranking.py`. The orchestrator must validate all raw provider output before globally normalizing and sorting it; it must never turn a missing or failed score into zero.

**Tech Stack:** Python 3.12, Pydantic v2, pytest, standard-library SHA-256 and JSON cache files.

## Global Constraints

- M2-T02 only; do not modify `STATUS.md`, M2 task counts, M3-M6, M1 artifacts, or M2-T01 evidence.
- `configured_top_k` is restricted to `50..100`; available candidate count may be lower; `effective_top_k = min(configured_top_k, available_count)`.
- The current 33 frozen candidates therefore yield `configured_top_k=100`, `available_count=33`, and `effective_top_k=33`; do not encode 33 into production logic.
- Inputs are the frozen query plus each source-backed title and optional abstract; a missing abstract means title-only input and must not be synthesized.
- Cache identity includes query SHA-256, paper-input SHA-256, the complete `RerankerModelDescriptor`, and therefore immutable model revision, library/version, input format version, and cache namespace.
- Raw scores remain raw until every selected provider output has passed count, identity, type, and finiteness validation; normalization occurs once over the complete raw-score set.
- Success returns `RerankRun(state=SCORED)`; an empty selected set returns `RerankRun(state=NOT_RUN, records=[])` without a provider call or cache write.
- A provider load or execution failure raises `RerankerTaskError(code="PROVIDER_UNAVAILABLE")`; malformed count, ID, type, or score output raises `RerankerTaskError(code="INVALID_OUTPUT")`.
- Every `RerankerTaskError` exposes its stable `.code`; failed calls return no partial `RerankRun`, no numeric fallback score, and no new cache entry.
- Fixed query, frozen inputs, configuration, descriptor, and provider output must produce the same final order: raw score descending, then `paper_id` ascending.
- No real reranker invocation, model download, model dependency, or real score artifact is part of this task.

## R0.1 Contract Clarifications

- The core, not the provider, selects cache misses and partitions them into ordered chunks. Each `RerankerProvider.score` call receives one chunk whose input count is at most `batch_size`; concatenated chunk paper IDs equal the selected miss set exactly once.
- A cache entry contains exactly `raw_score`, `paper_id`, `query_sha256`, `input_sha256`, `descriptor`, and `cache_key`. It never stores a normalized score.
- A cache key is independent of batch size and caller input order, but changes when the query, title, abstract, full descriptor, immutable revision, input format version, or cache namespace changes.
- The core validates each provider response against the current chunk by `paper_id`, rather than response position. Reordered valid IDs are accepted; missing, extra, duplicate, unknown, prior-chunk, and non-sequence outputs are `INVALID_OUTPUT`.
- Cache writes are transactional for one call: retain pre-existing valid entries, stage all new raw-score entries only in memory or temporary files, and publish none unless every selected miss succeeds and validates. On any failure, delete temporary files and leave no new JSON cache entry.
- After all selected raw scores, including cache hits, are available, normalize once using `(raw - min_raw) / (max_raw - min_raw)`. Use `1.0` for a single-item or all-equal raw-score set. Normalized scores never define ordering.

## R0.2 Test-Layer Boundaries

- `tests/contract/test_reranker_model_contracts.py` imports only `app.models.reranking` and `app.adapters.reranking.RerankerProvider`. It owns direct Pydantic/model and provider-protocol contracts, so A1 can make this file pass without creating the core.
- `tests/contract/test_reranker_orchestration.py` owns every test that imports or calls `build_reranker_input`, `effective_top_k`, or `rerank_candidates`. It remains collection-red until A2 creates `app.core.reranking`.
- `SCORED` and `NOT_RUN` are normal `rerank_candidates` return states. `PROVIDER_UNAVAILABLE` and `INVALID_OUTPUT` remain machine-readable outcome vocabulary, but the core raises `RerankerTaskError` for them and never returns a partial run.
- `RerankerModelDescriptor.input_format_version` accepts exactly `m2-reranker-title-abstract-v1` and `m2-reranker-title-abstract-v2`; blank, whitespace-only, and unknown versions are invalid.
- Synthetic providers record `call_queries`, `call_paper_ids`, and `call_batch_sizes`. A changed query is valid fixture input and must trigger a distinct provider call rather than a fixture assertion.

## File Structure

- Create: `app/models/reranking.py` — closed descriptor, input, raw/final record, state, cache-entry, run-stat, and error-code contracts.
- Create: `app/adapters/reranking.py` — `RerankerProvider` protocol only; later real adapters must implement it without changing core behavior.
- Create: `app/core/reranking.py` — title/abstract serialization, cache-key construction, batch orchestration, fail-closed output validation, global normalization, and stable order restoration.
- Create: `tests/contract/test_reranker_model_contracts.py` — direct model/provider red-green contracts; no core import.
- Create: `tests/contract/test_reranker_orchestration.py` — synthetic-only core red contracts; no model runtime, network, or downloaded weight.

### Task 1: Close the data and provider contracts

**Files:**
- Create: `app/models/reranking.py`
- Create: `app/adapters/reranking.py`
- Test: `tests/contract/test_reranker_model_contracts.py`

**Interfaces:**
- Produces `RerankerProvider.score(query: str, inputs: Sequence[RerankerInput], *, batch_size: int) -> list[ProviderRawScore]`.
- Produces `RerankerModelDescriptor`, `RerankerInput`, `RerankRecord`, `RerankRun`, and `RerankerTaskError`.
- `RerankRecord` exposes `paper_id`, `raw_score`, `normalized_score`, `descriptor`, and `input_sha256` only when its state is `SCORED`.
- `RerankerTaskError.code` is one of `INVALID_INPUT`, `PROVIDER_UNAVAILABLE`, or `INVALID_OUTPUT`; the core uses `INVALID_INPUT` for invalid Top-K/count parameters.
- `RerankerModelDescriptor.input_format_version` permits the explicit v1 and v2 literals, so a format revision produces a distinct cache identity.

- [ ] **Step 1: Write the failing contract tests**

Add direct tests for stable task-error codes, closed descriptor fields and immutable revisions, v1/v2 input formats, exact UTF-8 input hashes, finite strict raw scores, `RerankRecord` score/state invariants, and `RerankRun` cardinality/identity/descriptor invariants.

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `py -3.12 -m pytest -q tests/contract/test_reranker_model_contracts.py`

Expected: FAIL during collection because `app.models.reranking` or `app.adapters.reranking` does not exist.

- [ ] **Step 3: Implement the closed models and protocol**

Define literal states `SCORED`, `NOT_RUN`, `PROVIDER_UNAVAILABLE`, and `INVALID_OUTPUT`; reject unpinned revisions, non-SHA input hashes, non-finite scores, score-bearing non-`SCORED` records, and unknown descriptor fields. Use a protocol rather than a concrete model dependency.

- [ ] **Step 4: Run the focused tests to verify the contracts pass**

Run: `py -3.12 -m pytest -q tests/contract/test_reranker_model_contracts.py`

Expected: model/provider contract tests pass without importing the core.

Run: `py -3.12 -m pytest -q tests/contract/test_reranker_orchestration.py`

Expected: collection error because `app.core.reranking` does not exist.

- [ ] **Step 5: Commit**

```powershell
git add app/models/reranking.py app/adapters/reranking.py tests/contract/test_reranker_model_contracts.py
git commit -m "feat: add reranker provider contracts"
```

### Task 2: Implement deterministic input, cache, and batch orchestration

**Files:**
- Create: `app/core/reranking.py`
- Modify: `tests/contract/test_reranker_orchestration.py`

**Interfaces:**
- Consumes `FrozenCandidate`, `RerankerProvider`, and `RerankerModelDescriptor`.
- Produces `build_reranker_input(candidate)`, `effective_top_k(configured_top_k, available_count)`, and `rerank_candidates(query, candidates, provider, cache_dir, *, batch_size, configured_top_k)`.

- [ ] **Step 1: Write the failing orchestration tests**

Add tests named `test_effective_top_k_caps_to_available_candidates_without_hard_coding_33`, `test_empty_abstract_serializes_title_only`, `test_cache_key_changes_when_revision_changes`, and `test_batch_sizes_1_2_8_33_preserve_raw_scores_and_final_order`.

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `py -3.12 -m pytest -q tests/contract/test_reranker_orchestration.py`

Expected: collection error because the orchestration module is not implemented.

- [ ] **Step 3: Implement the minimal orchestration**

Serialize exactly `title:\n{title}` plus `\n\nabstract:\n{abstract}` only when the source abstract is non-empty. Batch only cache misses, restore requested identities, and include query/input/revision/format-version cache identity. Reject a provider exception as `PROVIDER_UNAVAILABLE`; do not persist a score or cache entry for that call.

- [ ] **Step 4: Run the focused tests to verify deterministic behavior**

Run: `py -3.12 -m pytest -q tests/contract/test_reranker_orchestration.py`

Expected: synthetic batch and cache tests pass for batch sizes 1, 2, 8, and 33.

- [ ] **Step 5: Commit**

```powershell
git add app/core/reranking.py tests/contract/test_reranker_orchestration.py
git commit -m "feat: orchestrate deterministic reranker batches"
```

### Task 3: Enforce global normalization, fail-closed outcomes, and stable ranking

**Files:**
- Modify: `app/core/reranking.py`
- Modify: `tests/contract/test_reranker_orchestration.py`

**Interfaces:**
- Consumes complete validated `ProviderRawScore` values.
- Produces records sorted by `(-raw_score, paper_id)` after a single all-record min-max normalization pass.

- [ ] **Step 1: Write the failing ranking-safety tests**

Add tests named `test_global_normalization_prevents_batch_local_rank_inversion`, `test_provider_failure_never_becomes_zero_score`, `test_equal_scores_use_stable_paper_id_tie_break`, and `test_input_order_permutations_produce_the_same_final_order`.

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `py -3.12 -m pytest -q tests/contract/test_reranker_model_contracts.py tests/contract/test_reranker_orchestration.py`

Expected: FAIL until normalization follows full validated raw-score collection.

- [ ] **Step 3: Implement the minimal ranking safety logic**

Normalize once with `(raw - min_raw) / (max_raw - min_raw)` after complete validation, with `1.0` for one or all-equal records. On provider load/error raise `RerankerTaskError(code="PROVIDER_UNAVAILABLE")`; on malformed count/IDs/scores raise `RerankerTaskError(code="INVALID_OUTPUT")`; both stop the run without a zero-score substitute, partial ranked result, new cache JSON, or leftover temporary file.

- [ ] **Step 4: Run focused and full validation**

Run: `py -3.12 -m pytest -q tests/contract/test_reranker_model_contracts.py tests/contract/test_reranker_orchestration.py`

Expected: all M2-T02 contract tests pass with synthetic providers only.

Run: `py -3.12 -m pytest -q`

Expected: full regression passes without downloading a reranker model.

- [ ] **Step 5: Commit**

```powershell
git add app/core/reranking.py tests/contract/test_reranker_orchestration.py
git commit -m "fix: fail closed on reranker output errors"
```

## Self-Review

- Spec coverage: the three tasks cover replaceability, query/title/abstract inputs, persisted raw provenance, core-owned batch chunks, batch invariance, fail-closed states, no zero-score fallback, transactional cache publication, one-pass min-max normalization, full-descriptor cache identity, title-only empty abstracts, Top-K bounds, deterministic output, and tie behavior.
- Placeholder scan: no task relies on an unspecified function name or test command.
- Type consistency: the provider returns `ProviderRawScore`; core validates it into `RerankRecord`; only `RerankRecord` enters cache and final order.

## A3 Transaction and Adversarial Follow-up

### A3-R: Adversarial tests

**Files:**
- Modify: `tests/contract/test_reranker_model_contracts.py`
- Modify: `tests/contract/test_reranker_orchestration.py`
- Modify: `docs/superpowers/plans/2026-07-29-m2-t02-reranker-provider.md`

**Goal:** Establish red-first contracts for cache publication rollback, temporary-file isolation, direct revalidation, corrupt-cache rejection, and input validation ordering without changing production code.

- [ ] Add a four-miss publication test that fails the third `os.replace` and requires call-level removal of every new JSON entry, no temporary/staging file, and `INVALID_OUTPUT`.
- [ ] Add byte-for-byte restoration coverage for a pre-existing target replaced before a later publication failure.
- [ ] Add parameterized failure-point coverage for temporary-file open, `json.dump`, flush, `os.fsync`, and `os.replace`; retain all existing cache bytes and remove temporary files.
- [ ] Add same-process threaded concurrency coverage using `ThreadPoolExecutor(max_workers=2)` and `threading.Barrier`; require independent temporary/staging identities and identical completed runs.
- [ ] Add invalid constructed `ProviderRawScore` coverage using `warnings.catch_warnings(record=True)`; reject before serializing and record no serializer warning.
- [ ] Add corrupt-cache, distinct-paper-ID cache identity, generator-output, descriptor-access ordering, and unhashable input-format characterization gates.
- [ ] Run focused contracts and Ruff. Preserve the expected red tests for transaction rollback, temporary isolation, warning-free revalidation, input-before-descriptor validation, and unhashable format validation.
- [ ] Commit only tests and this plan as `test: harden reranker transaction boundaries`.

### A3-F: Transaction and validation hardening

**Files:**
- Modify: `app/core/reranking.py`
- Modify: `app/models/reranking.py`
- Test: `tests/contract/test_reranker_model_contracts.py`
- Test: `tests/contract/test_reranker_orchestration.py`

**Goal:** Make every A3-R red contract pass without changing reranker ranking semantics or adding a real reranker.

- [ ] Implement call-level rollback for multi-file publication: preserve pre-call target bytes, remove every newly published entry on any write failure, and restore overwritten targets byte-for-byte.
- [ ] Allocate a unique transaction/staging identity per call and isolate temporary writes for concurrent calls in one Python process. Cross-process locking remains explicitly out of scope.
- [ ] Revalidate `ProviderRawScore` constructed instances by extracting `paper_id` and `raw_score` directly before any Pydantic serializer invocation, so rejected values emit no serializer warning.
- [ ] Validate all selected candidate inputs before reading `provider.descriptor`, and convert unhashable descriptor-format values into Pydantic `ValidationError`.
- [ ] Re-run focused, full, Ruff, mypy, historical validators, and remote Actions before changing any task status.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-29-m2-t02-reranker-provider.md`. The present branch stops after the plan and red tests. A later implementation can execute the three tasks inline, one red/green commit boundary at a time.

## B0 Model selection and immutable source lock

**Goal:** Select a documented reranker without downloading any model or tokenizer file, loading a model, or producing a score.

**Files:**
- Modify: `docs/superpowers/plans/2026-07-29-m2-t02-reranker-provider.md`
- Create: `evaluation/source-artifacts/m2-t02-reranker-selection.json`
- Create: `tests/contract/test_m2_t02_reranker_selection.py`

- [ ] Query only official Hugging Face model APIs and pinned model cards for `BAAI/bge-reranker-v2-m3`, `Alibaba-NLP/gte-multilingual-reranker-base`, `cross-encoder/ms-marco-MiniLM-L6-v2`, and `jinaai/jina-reranker-v2-base-multilingual`. Do not download file bodies other than small metadata/model-card text.
- [ ] Resolve the selected BAAI model to a lowercase 40-character commit SHA; record Apache-2.0, `transformers`, `trust_remote_code: false`, raw relevance logits, core-owned global min-max normalization, repository/weight sizes, and file metadata with either an LFS SHA-256 or explicitly typed Git blob SHA-1.
- [ ] Record the three alternatives without performance ranking: GTE is backup-only because its official configuration requires remote custom code; MiniLM is an English CPU baseline only; Jina is not selected because its card declares CC-BY-NC-4.0 and its configuration provides remote custom code.
- [ ] Keep `weights_downloaded`, `tokenizer_downloaded`, `model_loaded`, `inference_run`, and `real_scores_generated` all false. The contract test must read only the local JSON artifact and reject an unpinned revision, duplicate file paths, invalid digest types, secrets, absolute paths, or invented runtime/benchmark fields.
- [ ] Run `py -3.12 -m pytest -q tests/contract/test_m2_t02_reranker_selection.py` before and after creating the artifact. Then run the full offline regression and all static/historical validators before committing only these three files.

## B0.1 Metadata semantics and exact immutable contract

**Goal:** Correct and lock the B0 selected-model metadata without downloading a model or tokenizer, loading a model, running inference, or producing a score.

**Files:**

- Modify: `evaluation/source-artifacts/m2-t02-reranker-selection.json`
- Modify: `tests/contract/test_m2_t02_reranker_selection.py`
- Modify: `docs/superpowers/plans/2026-07-29-m2-t02-reranker-provider.md`

**Constraints:**

- B0.1 changes metadata and its offline contract only; B1 remains unauthorized.
- `tokenizer_model_max_length: 8192` is the tokenizer/model capability boundary; `model_max_position_embeddings: 8194` records the model position table; `runtime_policy.max_length: 512` remains the initial runtime truncation policy.
- B2 disk planning uses `required_runtime_files_size_bytes: 2293242108`, not Hugging Face `usedStorage`.
- Any B2 download and validation must use revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` and the seven exact source-file digests.

- [ ] **Step 1: Write the failing exact-schema contract**

Replace subset checks with `assert set(object) == EXPECTED_FIELDS` for `selected_model`, `runtime_policy`, `official_metadata_source`, every source-file entry, every alternative entry, and `execution_state`. Assert the selected BAAI revision, Apache-2.0 license, parameter count `567755777`, `trust_remote_code: false`, and runtime library `transformers`. Assert the three alternative IDs, revisions, licenses, multilingual flags, remote-code flags, and selection statuses exactly.

- [ ] **Step 2: Verify the old artifact is rejected**

Run: `py -3.12 -m pytest -q tests/contract/test_m2_t02_reranker_selection.py`

Expected: FAIL because the old artifact contains `maximum_supported_input`, `repository_size_bytes`, and the revisionless `official_api_url` rather than the B0.1 fields.

- [ ] **Step 3: Apply the metadata correction**

Replace `normalization_owner: app.core.reranking.global_min_max` with `normalization_owner: app.core.reranking` and `normalization_method: global_min_max`. Replace `maximum_supported_input` with `tokenizer_model_max_length: 8192` and `model_max_position_embeddings: 8194`; retain the runtime policy `max_length: 512` and set `pair_truncation_strategy: longest_first`. Replace `repository_size_bytes` with `huggingface_used_storage_bytes: 7975340915`, `pinned_source_files_size_bytes: 2293259337`, and `required_runtime_files_size_bytes: 2293242108`. Replace every `official_api_url` with no field and add the selected model's closed `official_metadata_source` object pinned to `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`.

- [ ] **Step 4: Verify exact source, state, and safety invariants**

Make the offline test compare all seven source file objects to the pinned path, size, digest, digest type, storage type, and runtime flag; recompute the two source-size totals and require `weight_size_bytes` to equal `model.safetensors.size_bytes`. Require `git -> git_blob_sha1`, `lfs -> sha256`, five false execution-state values, no absolute local paths, no secrets, no score/runtime/benchmark result fields, and no `main` or `latest` revision value.

- [ ] **Step 5: Run the approved verification suite and commit**

Run the focused contract, full pytest, Ruff, mypy, project-document and historical evidence validators, and `pip check` exactly as specified by the B0.1 task request. Stage only these three B0 files and commit:

```powershell
git add evaluation/source-artifacts/m2-t02-reranker-selection.json tests/contract/test_m2_t02_reranker_selection.py docs/superpowers/plans/2026-07-29-m2-t02-reranker-provider.md
git commit -m "fix: harden M2 reranker selection lock"
```

## B1 Provider implementation with mocked runtime

B1 may implement a `transformers` adapter using mocked imports and mocked runtime objects only. Unit tests must not access the network, download files, load weights, or create real scores. B1 is not authorized by B0.

## B1-R Mocked provider red contracts

**Goal:** Define collection-safe, mocked red contracts for the future pinned BGE provider while preserving the B0.1 selection contract.

**Files:**

- Modify: `tests/contract/test_m2_t02_reranker_selection.py`
- Create: `tests/unit/test_bge_reranker_provider.py`
- Modify: `docs/superpowers/plans/2026-07-29-m2-t02-reranker-provider.md`

**Interfaces:** B1-F will add `BgeRerankerProvider(model_id, model_revision, cache_namespace, model_dir, max_length=512, device="cpu")`, with `.descriptor`, `.runtime`, and `score(query, inputs, *, batch_size)` to `app/adapters/reranking.py`. B1-R does not create that production class or constants.

- [ ] **Step 1: Preserve B0.1 model-card guarantees**

Add non-blocking assertions for `model_parameter_count: 567755777`, standard-transformers compatibility, Apache-2.0 commercial-use constraint, and the exact pinned README URL. Reject `/main/`, `/latest/`, and missing revision text.

- [ ] **Step 2: Add collection-safe mocked red tests**

Import only `app.adapters.reranking` at module level. Resolve the absent provider in `_provider_type()` with `getattr`, so pytest collects and each behavioral test fails with `BgeRerankerProvider has not been implemented` rather than an import error. Include a subprocess characterization that adapter import leaves `torch`, `transformers`, `huggingface_hub`, `sentence_transformers`, and `FlagEmbedding` absent.

- [ ] **Step 3: Lock B1-F's local-only behavior**

Use `sys.modules` fake `torch` and `transformers` modules, non-empty temporary placeholders for the six runtime files, and fixed metadata versions. Assert exact CPU/float32 construction gates, snapshot presence/non-empty checks, lazy loading, local-only `from_pretrained` arguments, ordered query/passage pairs, no provider-side normalization, exact raw logits, output validation, and stable `INVALID_INPUT`, `INVALID_OUTPUT`, or `PROVIDER_UNAVAILABLE` codes.

- [ ] **Step 4: Run and record the intentional red state**

Run:

```powershell
py -3.12 -m pytest -q tests/contract/test_m2_t02_reranker_selection.py tests/unit/test_bge_reranker_provider.py
py -3.12 -m ruff check tests/contract/test_m2_t02_reranker_selection.py tests/unit/test_bge_reranker_provider.py
```

Expected: B0.1 selection and import-side-effect tests pass; provider behavior tests fail only because B1-F has not implemented `BgeRerankerProvider`. Do not use `skip` or `xfail`.

- [ ] **Step 5: Commit only B1-R**

```powershell
git add docs/superpowers/plans/2026-07-29-m2-t02-reranker-provider.md tests/contract/test_m2_t02_reranker_selection.py tests/unit/test_bge_reranker_provider.py
git commit -m "test: define pinned BGE reranker provider contract"
```

## B1-F Offline local-snapshot provider implementation

**Goal:** Implement only the B1-R contract in `app/adapters/reranking.py` with lazy, CPU float32, local-snapshot-only mocked-runtime behavior.

**Constraints:** B1-F may not download a model, access Hugging Face, use a token, load real weights, generate real scores, add dependencies, begin B2/B3/C, or update `STATUS.md`. It checks only required-file existence and non-emptiness; B2 owns digest verification.

## B1-R.1 Failure-stage isolation and metadata semantics

**Goal:** Repair the mocked red contracts so B1-F must satisfy each precise failure stage without changing production code.

**Files:**

- Modify: `tests/unit/test_bge_reranker_provider.py`
- Modify: `docs/superpowers/plans/2026-07-29-m2-t02-reranker-provider.md`

- [ ] **Step 1: Separate fake controls**

Split tokenizer `load_error` from `call_error`; split model class `load_error` from instance `to_error`, `eval_error`, and `forward_error`. Reset every control in the fake-runtime installer and prove later stages were not reached after each injected failure.

- [ ] **Step 2: Separate metadata from imports and strengthen invalid inputs**

Create a metadata-only version helper that returns `transformers=4.test` and `torch=2.test` without writing `sys.modules`; use it for lazy and missing-module tests. Cover blank/non-string query, invalid batch-size types, blank paper/text, invalid digest length, and digest/text mismatch with `model_construct()`, requiring `INVALID_INPUT` before either loader is called.

- [ ] **Step 3: Preserve intentional red verification**

Run the focused B0.1-plus-provider pytest command and Ruff for the unit test. Require normal collection; the B0.1 and import characterization tests pass, while every provider behavior test fails solely because `BgeRerankerProvider` remains absent. Do not use `skip` or `xfail`.

- [ ] **Step 4: Commit only B1-R.1**

```powershell
git add tests/unit/test_bge_reranker_provider.py docs/superpowers/plans/2026-07-29-m2-t02-reranker-provider.md
git commit -m "test: isolate BGE provider failure stages"
```

## B2 Controlled download and preflight

B2 is the first task allowed to perform a controlled download. It must verify the B0 pinned revision and file digests before CPU float32 preflight. B2 is not authorized by B0.

## B2-R Pinned snapshot preparation red contract

**Goal:** Define a fully offline, collection-safe contract for B2-F's pinned reranker snapshot preparation. This contract does not create a downloader, import `huggingface_hub`, access the network, install a dependency, download a model or tokenizer, load a model, or run inference.

**Files:**

- Modify: `docs/superpowers/plans/2026-07-29-m2-t02-reranker-provider.md`
- Create: `tests/unit/test_m2_t02_reranker_snapshot.py`
- Do not create: `scripts/prepare_m2_t02_reranker_snapshot.py`

**Future B2-F interface locked by this red suite:**

`SnapshotPreparationError` must expose `.code` with one of `INSUFFICIENT_DISK_SPACE`, `DOWNLOAD_FAILED`, or `INTEGRITY_CHECK_FAILED`. `load_snapshot_plan(selection_path: Path) -> SnapshotPlan` must return a plan exposing `.model_id`, `.revision`, and ordered `.files`. `prepare_snapshot(*, selection_path: Path, snapshot_dir: Path, download_file: Callable[..., str], disk_usage: Callable[[Path], object]) -> SnapshotPreparationResult` must return `.status` (`PUBLISHED` or `REUSED`) and `.snapshot_dir` (the requested final path).

- [ ] **Step 1: Add a dynamically resolved, collection-safe red suite**

  Keep the module import inside `_snapshot_module()` and fail each test with `prepare_m2_t02_reranker_snapshot has not been implemented` while B2-F is absent. The test module may import only standard-library modules and pytest; it must not import `huggingface_hub`, `transformers`, torch, or a BGE provider.

- [ ] **Step 2: Lock selection parsing and exact source identity**

  Read only the committed local `evaluation/source-artifacts/m2-t02-reranker-selection.json`. Require `BAAI/bge-reranker-v2-m3`, revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`, and exactly these ordered `(path, digest_type)` pairs: `README.md/git_blob_sha1`, `config.json/git_blob_sha1`, `model.safetensors/sha256`, `sentencepiece.bpe.model/sha256`, `special_tokens_map.json/git_blob_sha1`, `tokenizer.json/sha256`, and `tokenizer_config.json/git_blob_sha1`.

- [ ] **Step 3: Lock preflight and downloader-call safety before B2-F exists**

  Use a locally rewritten small-file selection fixture so no test creates a real model file. Require a zero-free-space `disk_usage` result to raise `SnapshotPreparationError(code="INSUFFICIENT_DISK_SPACE")` before any downloader call. For a successful fake downloader, require each call to use the exact model ID and pinned revision, `repo_type="model"`, `token=False`, `local_dir_use_symlinks=False`, and a staging `local_dir` rather than the final destination.

- [ ] **Step 4: Lock byte integrity, cleanup, atomic publication, and idempotence**

  The fixture must cover both a Git blob SHA-1 mismatch (`config.json`) and an LFS SHA-256 mismatch (`tokenizer.json`), each failing as `INTEGRITY_CHECK_FAILED` with no final snapshot and no `.<snapshot>.staging-*` directory. Simulate an `OSError` on the third download and require `DOWNLOAD_FAILED`, cleanup, and no partial final snapshot. Then require all seven fake downloads to complete before a single `PUBLISHED` final directory becomes visible; a second call must verify and return `REUSED` without disk preflight or downloader access.

- [ ] **Step 5: Record the intentional red state and commit only B2-R**

  Run:

  ```powershell
  py -3.12 -m pytest -q tests/unit/test_m2_t02_reranker_snapshot.py
  py -3.12 -m ruff check tests/unit/test_m2_t02_reranker_snapshot.py
  ```

  Expected: the focused suite collects without skip or xfail and every test fails only because `scripts.prepare_m2_t02_reranker_snapshot` is intentionally absent. Do not implement B2-F in this commit.

  ```powershell
  git add docs/superpowers/plans/2026-07-29-m2-t02-reranker-provider.md tests/unit/test_m2_t02_reranker_snapshot.py
  git commit -m "test: define pinned reranker snapshot preparation contract"
  ```

## B2-R.1 False-green closure and path safety

**Goal:** Close B2-R false-green gaps before any snapshot preparation code exists by fixing digest corruption semantics and defining fail-closed selection, disk, target-path, downloader-return, staging-content, publication, reuse, and cleanup contracts.

**Files:**

- Modify: `tests/unit/test_m2_t02_reranker_snapshot.py`
- Modify: `docs/superpowers/plans/2026-07-29-m2-t02-reranker-provider.md`
- Do not create or modify: `scripts/**`, `app/**`, `evaluation/**`, `tests/contract/**`, dependencies, model files, runtime evidence, or `STATUS.md`

**Boundary sequence:**

- B2-F may implement only the dependency-free, offline-testable preparation core specified by B2-R and B2-R.1. It must not install dependencies, access the network, download files, construct a reranker provider, load a tokenizer/model, run inference, or produce scores.
- B2-D is the first step allowed to lock and install runtime/download dependencies.
- B2-L is the first step allowed to perform the real pinned download.
- B2-P is the first step allowed to load the real tokenizer/model for CPU float32 preflight.
- B3 is the first step allowed to generate complete real reranker scores.
- Until C, `STATUS.md` remains M2 `IN_PROGRESS` with `1/5`; PR #11 remains Draft.

- [ ] **Step 1: Add independent characterization gates**

  Read `.gitignore` directly and require `models/` plus `*.safetensors`, without starting Git. Read the committed selection artifact directly and require `decision_status == "selected_not_downloaded"` plus all five execution-state flags false. Verify the fixed `b"reranker-snapshot\n"` Git-object digest is `f62d0c3a99a7648eaced175f791324b04930b7a0` and differs from the plain-file SHA-1. These three tests must pass while the future module is absent.

- [ ] **Step 2: Remove digest-size false greens and close selection parsing**

  Corrupt `config.json` and `tokenizer.json` by flipping one bit while preserving byte length, then explicitly prove unchanged size and changed Git blob SHA-1/SHA-256. Parameterize single-field mutations for the pinned model/revisions/status/execution flags, exact seven-file set, safe relative paths, positive sizes, digest/storage pairings, boolean runtime flags, digest syntax, and all three declared byte totals. Every invalid selection must raise the same `INTEGRITY_CHECK_FAILED` code without network access or artifact repair.

- [ ] **Step 3: Lock disk and target-path preflight**

  Expose `DOWNLOAD_HEADROOM_BYTES = 1073741824`; require `plan.total_size_bytes + DOWNLOAD_HEADROOM_BYTES`, which is exactly `3,367,001,161` bytes for the committed selection. Permit download when free bytes equal the requirement; reject one byte less before downloader/staging. Map a disk-probe `OSError` to `DOWNLOAD_FAILED`. Reject existing-file targets, symlinks, the selection path, a controlled repository root, unresolved `..`, and file parents before download, preserving every existing byte.

- [ ] **Step 4: Close destination, downloader-return, and staging contents**

  Reuse an exact existing seven-file snapshot without disk probing, downloading, rewriting bytes/mtimes, or staging. Reject missing, same-length-corrupt, extra-file, extra-directory, and symlink-bearing existing snapshots without repair. Require every downloader return to resolve inside the current staging directory; reject outside files, final-destination paths, another staging directory, and relative traversal without deleting external data. Before publication, require exactly the seven declared root files; reject extra files/directories, symlinks, and nested declared files. A staging-only `.cache/huggingface` may either be rejected or removed before publication, but must never appear in the final snapshot or affect caches outside staging.

- [ ] **Step 5: Prove one atomic publication and exhaustive cleanup**

  Allocate each unique sibling staging directory with `tempfile.mkdtemp`, then patch the future module's `os.replace` and require exactly one directory rename from `.<snapshot>.staging-*` to the absent final target. A publication `OSError` maps to `DOWNLOAD_FAILED` and removes all staging files. Cover first, third, last, and post-write download failures, staging-directory creation failure, same-length digest failure, and publication failure; every case leaves no destination/staging/partial path and preserves unrelated directories, external caches, and a pre-existing staging-shaped sentinel. Two independent failed calls must use distinct sibling staging paths. Cross-process locking remains out of scope.

- [ ] **Step 6: Preserve import isolation and intentional red evidence**

  In an isolated Python process, importing the future module and running fake preparation must not import `torch`, `transformers`, `huggingface_hub`, `FlagEmbedding`, or `sentence_transformers`, and must not construct `BgeRerankerProvider`. Run:

  ```powershell
  py -3.12 -m pytest -q tests/unit/test_m2_t02_reranker_snapshot.py
  py -3.12 -m ruff check tests/unit/test_m2_t02_reranker_snapshot.py
  ```

  Expected for B2-R.1: collection succeeds with `3 passed, 67 failed`; the three characterization tests pass, and all 67 parameterized behavior cases fail only with `prepare_m2_t02_reranker_snapshot has not been implemented`. No fixture, network, package-import, skip, or xfail outcome is permitted.

- [ ] **Step 7: Commit and publish only B2-R.1**

  ```powershell
  git add tests/unit/test_m2_t02_reranker_snapshot.py docs/superpowers/plans/2026-07-29-m2-t02-reranker-provider.md
  git diff --cached --name-only
  git diff --cached --check
  git commit -m "test: harden reranker snapshot preparation boundaries"
  git push origin agent/m2-t02-reranker-provider
  ```

## B3 Real live/replay evidence

B3 is the first task allowed to run real inference, generate real scores, and record separately labeled live and replay evidence. B3 is not authorized by B0.

### B3-I Controlled fixed-candidate runner and fake-runtime contract

**Files:**

- Create: `scripts/run_m2_t02_candidate_reranking.py`
- Create: `tests/unit/test_m2_t02_candidate_reranking_runner.py`
- Create: `tests/contract/test_m2_t02_candidate_reranking.py`
- Modify: `app/adapters/reranking.py`
- Modify: `docs/superpowers/plans/2026-07-29-m2-t02-reranker-provider.md`
- Do not modify: `app/models/reranking.py`, `pyproject.toml`, `app/core/reranking.py`,
  `STATUS.md`, any existing evidence/snapshot/manifest, `models/**`, or the
  download/preflight runners.
- Do not create: `evaluation/source-artifacts/m2-t02-reranker-candidate-run.json`
  or any real score/cache artifact.

**Goal:** Implement the future-only fixed 33-candidate local reranking entrypoint,
its closed `m2-t02-reranker-candidate-run.v1` report contract, and its fake-only
test seam without invoking a real model or generating B3 evidence in this task.

**Architecture:** The runner resolves repository paths from `Path(__file__)`,
validates the committed preflight evidence/receipt and candidate snapshot before
any provider construction, then delegates batching, raw-score validation,
global normalization, and deterministic ordering to `app.core.reranking`. A
counting wrapper and invocation-owned temporary cache capture the fixed 17-batch
contract and are always cleaned up. Report validation and publication are
separate: tests publish only into temporary paths, while the real CLI is the
only path that can publish the reserved formal result in a future authorized run.

**Fixed interfaces:**

- The only accepted CLI vector is
  `py -3.12 -m scripts.run_m2_t02_candidate_reranking --execute-local-reranking`;
  every other vector returns non-zero before candidate/runtime access.
- The runner exposes `run_candidate_reranking(...)`,
  `validate_candidate_run_report(value)`, and
  `publish_candidate_run(value, destination=...)`; Python-only provider/path/cache
  injection is test-only and no CLI override is added.
- Fixed input is the 33-candidate snapshot, its manifest, the two committed
  preflight files, and the pinned snapshot directory. The query is exactly
  `How can graph-based retrieval support scientific literature discovery?`.
- Runtime policy is `configured_top_k=50`, `effective_top_k=33`, `batch_size=2`,
  `max_length=512`, CPU `float32`, one provider instance, serial inference,
  one runtime load, and an initially empty invocation-owned cache.
- The report has exactly the top-level fields
  `report_version`, `phase`, `task_id`, `baseline_commit`, `decision_status`,
  `input`, `model`, `runtime`, `policy`, `execution`, and `records`; each record
  has exactly `rank`, `paper_id`, `input_sha256`, `raw_score`, and
  `normalized_score`. Records contain no title, abstract, URL, authors, tokens,
  logits tensor, absolute path, credential, username, or hostname.
- Publication validates the complete report, uses one unique sibling temp and
  one `os.replace`, reuses byte-identical regular files without changing mtime,
  and raises conflict for changed, malformed, symlink/junction, directory, or
  otherwise non-regular targets while cleaning only the owned temp.

- [x] **Step 1: Add fake-only red contracts**

  Add subprocess import-isolation and exact-CLI tests; fixed-path fixture tests
  for preflight/receipt/hash/count/order/query validation; fake-provider tests
  for 33 records, `16 x 2 + 1 x 1` batches, one instance/load, zero cache hits,
  tie-breaking, equal-score normalization, failure cleanup, forbidden report
  fields, and the reserved formal path. Add temporary-path publication tests for
  idempotence, conflict, and owned-temp cleanup. All provider/model behavior is
  injected fake behavior; no test invokes the accepted CLI vector.

- [x] **Step 2: Implement the fixed runner and closed report validator**

  Validate ordinary-file identities and exact hashes before constructing the
  provider. Load candidates only after the evidence gates, validate the required
  English title/abstract/paper-id/query invariants, call
  `build_reranker_input(candidate)` for every selected candidate, use one
  `CountingRerankerProvider`, run the existing core with batch size 2, build the
  closed report, and clean the temporary cache in `finally`. Map all failures to
  stable closed runner codes without printing exception text or writing partial
  output.

- [x] **Step 3: Implement and test atomic report publication**

  Serialize the validated report deterministically, inspect the destination with
  `lstat`, return reuse only for identical regular bytes, reject every conflict,
  create one sibling temp, flush it, perform one `os.replace`, and remove only
  the owned temp on every failed path. Never call this publisher for the default
  fake-only runner path or write the repository formal result during B3-I.

- [x] **Step 4: Harden provider loading to match preflight**

  Keep `torch`/`transformers` imports delayed; set and restore all four offline
  environment variables around runtime loading; pass the exact local-only,
  non-remote-code, safetensors, `torch.float32` arguments; use the preflight
  `(queries, passages, padding=True, truncation=True, max_length=512,
  return_tensors="pt")` call; move/eval on CPU and reject CUDA, training mode,
  non-CPU parameters, or non-float32 parameters. Preserve the existing raw-score
  and stable error contracts.

- [x] **Step 5: Run verification and audit scope**

  Run the focused five-file suite, the full pytest suite, Ruff (including import
  sorting), both mypy commands, project/M0/M1/M2-T01 validators, and `pip check`.
  Confirm the test count increases, no formal result/cache/model/evidence or
  `STATUS.md` change exists, `git ls-files models` is empty, and only the five
  authorized files are staged.

  Result: the focused suite passed `196` tests and the full suite passed `1006`
  tests (`976` baseline plus the new fake-only contracts). Ruff and Ruff `I001`,
  mypy for `46` source files plus the standalone runner, project docs, M0, M1,
  M2-T01, and `pip check` all passed. The formal candidate-run file is absent,
  no cache or temporary result remains, `git ls-files models` is empty, and all
  protected evidence/snapshot/manifest/STATUS paths are byte-unchanged.

- [x] **Step 6: Commit, push, and update Draft PR #11**

  Commit exactly `feat: add controlled candidate reranking runner`, push the
  current branch without amend/rebase/force, and update PR #11 while retaining
  Draft. Record B3-I fake-only validation, fixed 33-candidate/17-batch and
  offline CPU contracts, the focused/full counts, no real scores, no formal
  artifact, B3-Live not started, and `STATUS.md` still at M2 `IN_PROGRESS (1/5)`.

  Result: implementation commit `5a7cf46f52b20dd0b1af92ea29a25639995ff01e`
  was pushed to `agent/m2-t02-reranker-provider`; PR #11 was updated and
  remains open/Draft. No amend, rebase, force push, real B3 invocation, B3-Live
  start, formal candidate-run artifact, or STATUS update occurred.

### B3-I self-review checklist

- [x] The fixed preflight validator and receipt schema are called before any runtime
  import or provider construction.
- [x] Candidate serialization is delegated to the existing
  `build_reranker_input(candidate)` and never rebuilt in the runner.
- [x] A provider failure cannot return partial records or leave cache files.
- [x] The report validator rejects NaN/Inf, wrong cardinality, duplicate/missing/
  extra paper IDs, wrong sorting, score-range violations, and forbidden fields.
- [x] All B3 real/live/replay execution, formal result publication, and STATUS closure
  remain explicitly not run.

### B3-I.1 Candidate evidence boundary hardening

**Goal:** Harden the fake-only B3-I runner so fixed candidate bytes, the current
model snapshot, strict runtime observations, execution counts, and report
verification cannot drift into a formal candidate-run result. This task does
not run a real model, invoke real candidate reranking, create a formal
candidate-run artifact, or enter B3-Live.

**Files:**

- Modify: `scripts/run_m2_t02_candidate_reranking.py`
- Modify: `app/adapters/reranking.py`
- Modify: `tests/unit/test_m2_t02_candidate_reranking_runner.py`
- Modify: `tests/contract/test_m2_t02_candidate_reranking.py`
- Modify: `tests/unit/test_bge_reranker_provider.py`
- Modify: `docs/superpowers/plans/2026-07-29-m2-t02-reranker-provider.md`
- Do not modify: `STATUS.md`, existing evidence, candidate snapshot, candidate
  manifest, `models/**`, `pyproject.toml`, download/preflight runners,
  `app/models/reranking.py`, or `app/core/reranking.py`.

**Fixed boundary:** The candidate snapshot raw-byte SHA-256 is
`4a2aec0fd0a1d22adc801fd3bc506e5da89895d1276cd572e2ac64014c162448`; the
manifest raw-byte SHA-256 is computed from the committed manifest and fixed in
the runner. Both are checked before either JSON payload is parsed. The pinned
model is `BAAI/bge-reranker-v2-m3` at revision
`953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`, with the seven production snapshot
files, exact sizes, and exact digests projected consistently by the selection
and snapshot-download evidence verifiers. Snapshot verification is read-only
and cannot call download, disk preparation, network, copy, rename, or write
operations.

- [x] **Step 1: Add red tests for byte, snapshot, runtime, count, and report drift**

  Add the coordinated snapshot title/abstract plus manifest-digest mutation
  and assert failure before provider construction. Add model selection,
  snapshot-download evidence, file-size, digest, symlink/junction, and
  projection tests that assert the read-only verifier runs before provider
  construction. Add strict runtime tests for every missing, exceptional, or
  drifted observation, exact two-sequence tokenizer arguments with no
  fallback, missing/zero/two/bool/string runtime load counts, fixed snapshot
  digest rejection, normalized-score recomputation, and candidate input-hash
  recomputation. All test providers and runtime modules remain fake-only.

- [x] **Step 2: Implement fixed candidate and current snapshot validation**

  Hash the raw snapshot and manifest bytes before JSON parsing; require both
  fixed hashes and retain the fixed candidate sequence as the validator's
  source of truth. Reuse the production selection-plan, snapshot-integrity,
  and snapshot-download-evidence validators before constructing any provider.
  Require the fixed model ID/revision, exact seven-file closure, regular
  non-link files, exact sizes/digests, and matching selection/evidence
  projections.

- [x] **Step 3: Implement strict provider runtime and tokenizer contracts**

  Require observed eval mode, CPU-only Torch metadata and availability
  behavior, an available float32 dtype, a callable non-empty parameter set,
  CPU parameters, and float32 parameters. Convert every missing, exceptional,
  or mismatched observation to `PROVIDER_UNAVAILABLE`. Remove the TypeError
  tokenizer retry and call only `(queries, passages, padding=True,
  truncation=True, max_length=512, return_tensors="pt")`.

- [x] **Step 4: Implement strict counts and report truth validation**

  Preserve the provider's raw `runtime_load_count` without defaulting or
  correcting it, and require exact `int` value `1` in the counting wrapper
  and report builder. Require the fixed candidate hash, fixed candidate order,
  fixed serialization hashes, and exact min-max normalized scores in report
  validation; equal raw scores must produce `1.0`, and in-range mismatches must
  fail.

- [x] **Step 5: Run the authorized offline verification and audit**

  Run the five-file focused suite, full pytest, Ruff and explicit `I001`, both
  mypy commands, project/M0/M1/M2-T01 validators, and `pip check`. Require a
  test-count increase and confirm no formal candidate-run file, real cache,
  model load, inference, score/ranking, or protected-byte/STATUS change.

- [x] **Step 6: Commit implementation, push, then record the plan separately**

  Stage only the five implementation/test files and commit
  `fix: harden candidate reranking evidence boundary`; push the existing branch
  without amend, rebase, or force push. Then mark this plan section complete,
  record exact validation counts and the fake-only/not-run boundary, stage only
  this plan, and commit `docs: record B3-I.1 hardening validation`; push again.
  Stop immediately without `--execute-local-reranking` or B3-Live.

**Completion record:** Implementation commit
`e6e3576e63b0170b5b24d20c3be55f9684039b79` was pushed to
`agent/m2-t02-reranker-provider`. The focused five-file suite passed `224`
tests and the full offline suite passed `1034` tests, increasing the prior
`1006` baseline by `28`. Ruff, explicit Ruff `I001`, production mypy,
standalone runner mypy, project-doc validation, M0, M1, M2-T01, and `pip
check` all passed. The current model snapshot was verified read-only through
the production selection, snapshot-integrity, and snapshot-download-evidence
verifiers; no downloader, disk preparation, network, copy, rename, write,
provider construction, model load, inference, score, or ranking was performed
by that verification. The formal candidate-run file is absent, no new
candidate reranking cache remains, `git ls-files models` is empty, and
`STATUS.md` plus all existing evidence, candidate snapshot, and candidate
manifest bytes are unchanged. B3-Live and
`--execute-local-reranking` remain not run.

## C Completion evidence and STATUS

Only C may consolidate completion evidence and consider `STATUS.md`; until C, M2 remains 1/5. B0 must not start B1, B2, B3, C, M2-T03, or M3.

## B2-D Runtime dependency lock and clean-environment verification

### B2-D-R Dependency selection contract

**Goal:** Select, document, and machine-validate the exact reranker runtime dependency set without installing a package, downloading a model/tokenizer, reading model weights, loading a model, or running inference.

**Files:**

- Modify: `docs/superpowers/plans/2026-07-29-m2-t02-reranker-provider.md`
- Create: `evaluation/source-artifacts/m2-t02-reranker-runtime-dependencies.json`
- Create: `tests/contract/test_m2_t02_reranker_runtime_dependencies.py`

**Locked decision:** `torch==2.4.1`, `transformers==4.53.2`, `huggingface-hub==0.34.3`, `safetensors==0.5.3`, and `tokenizers==0.21.2` are the complete direct runtime package set for the later B2-L and B2-P tasks. Torch reuses the existing embedding extra's exact Python 3.12 pin so embedding and reranking share one process environment. They target CPU `float32`, `BAAI/bge-reranker-v2-m3` revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`, standard `AutoTokenizer`, standard `AutoModelForSequenceClassification`, `local_files_only=True` after download, `trust_remote_code=False`, and safetensors weights.

**Evidence boundary:** Research may query only the listed official PyPI metadata and PyTorch/Hugging Face documentation. The artifact records source type and query date but no cookie, authorization, token, local absolute path, score, timing, or benchmark. Every verification-state field remains `false`; this is selection evidence, not an installation or runtime claim.

- [ ] **Step 1: Add the offline red contract**

  Create `tests/contract/test_m2_t02_reranker_runtime_dependencies.py` with closed-field checks for the artifact, the exact baseline commit, `selected_not_installed`, Python 3.12 range, exact five-name package set, exact semantic versions, non-empty licenses, runtime policy, six false verification flags, B0.1 model ID/revision, official HTTPS sources, and forbidden secret/result fields.

- [ ] **Step 2: Record the red result**

  Run:

  ```powershell
  py -3.12 -m pytest -q tests/contract/test_m2_t02_reranker_runtime_dependencies.py
  ```

  Expected: FAIL only because `evaluation/source-artifacts/m2-t02-reranker-runtime-dependencies.json` does not yet exist. Do not install a dependency to make this test pass.

- [ ] **Step 3: Add the selected-not-installed artifact**

  Add the JSON artifact with no fields beyond `report_version`, `phase`, `task_id`, `baseline_commit`, `decision_status`, `python`, `packages`, `runtime_policy`, `verification_state`, and `sources`. Each package has exactly `name`, `version`, `role`, `python_compatibility`, `license`, `source_type`, `source_reference`, and `selection_reason`.

- [ ] **Step 4: Verify the selection contract and repository gates**

  Run the focused contract, full pytest suite, Ruff, mypy, documentation and historical evidence validators, then `pip check`. These commands validate existing installed development dependencies only; they must not install the selected runtime set, download a model/tokenizer, or run a provider.

- [ ] **Step 5: Commit only B2-D-R**

  ```powershell
  git add docs/superpowers/plans/2026-07-29-m2-t02-reranker-provider.md evaluation/source-artifacts/m2-t02-reranker-runtime-dependencies.json tests/contract/test_m2_t02_reranker_runtime_dependencies.py
  git diff --cached --name-only
  git diff --cached --check
  git commit -m "test: define reranker runtime dependency lock"
  git push origin agent/m2-t02-reranker-provider
  ```

### B2-D-R.1 Shared embedding and reranker environment reconciliation

**Goal:** Reconcile the reranker selection artifact with the existing embedding extra so both can later be installed into one Python 3.12 process without a conflicting Torch exact pin. This task selects and validates metadata only; it does not install or run any runtime package.

**Files:**

- Modify: `docs/superpowers/plans/2026-07-29-m2-t02-reranker-provider.md`
- Modify: `evaluation/source-artifacts/m2-t02-reranker-runtime-dependencies.json`
- Modify: `tests/contract/test_m2_t02_reranker_runtime_dependencies.py`

**Interfaces:**

- Consumes: the existing `pyproject.toml` embedding extra containing `FlagEmbedding==1.3.5` and `torch==2.4.1`.
- Produces: artifact version `m2-t02-reranker-runtime-dependencies.v1.1`, the current baseline commit, a shared Torch `2.4.1` policy, and machine-checked package declaration compatibility for the later B2-D-F gate.

- [ ] **Step 1: Make the shared-environment contract red**

  Add closed `environment_policy` and `compatibility_policy` expectations, parse `pyproject.toml` with `tomllib`, and require the artifact Torch version to equal the embedding extra Torch version. Keep the existing five direct reranker packages and all six execution-state flags false.

  Run:

  ```powershell
  py -3.12 -m pytest -q tests/contract/test_m2_t02_reranker_runtime_dependencies.py
  ```

  Expected: FAIL because the v1 artifact has no `environment_policy` or `compatibility_policy` and selects `torch==2.7.1` rather than the embedding pin.

- [ ] **Step 2: Record the metadata-only reconciliation**

  Change the artifact to v1.1 with baseline `9a5518665ccc2ae6b7fb7d385206362a13bf3b93`; set the reranker Torch package and shared-environment policy to `2.4.1`; retain `selected_not_installed`; add exact FlagEmbedding and Transformers declaration ranges; and record only official HTTPS source URLs with no query, fragment, credentials, token, or local absolute path.

- [ ] **Step 3: Prove the contract and symlink regression stay green**

  Run the dependency focused contract, the snapshot focused suite without excluding symlink tests, then the full pytest, Ruff, mypy, project documentation, M0, M1, M2-T01, and `pip check` gates. Do not install packages, create an environment, download/load a model or tokenizer, invoke a provider, generate scores, or begin B2-D-F/B2-L/B2-P/B3.

- [ ] **Step 4: Commit only B2-D-R.1**

  ```powershell
  git add docs/superpowers/plans/2026-07-29-m2-t02-reranker-provider.md evaluation/source-artifacts/m2-t02-reranker-runtime-dependencies.json tests/contract/test_m2_t02_reranker_runtime_dependencies.py
  git diff --cached --name-only
  git diff --cached --check
  git commit -m "fix: reconcile reranker runtime with embedding environment"
  git push origin agent/m2-t02-reranker-provider
  ```

### B2-D-F Exact optional dependency group and clean-environment verification

**Goal:** Add the exact reranker optional-dependency group and prove that `.[embedding,reranker]` installs in one clean Windows Python 3.12 temporary virtual environment with the official CPU-only Torch 2.4.1 wheel.

**Files:**

- Modify: `pyproject.toml`
- Modify: `docs/superpowers/plans/2026-07-29-m2-t02-reranker-provider.md`
- Modify: `evaluation/source-artifacts/m2-t02-reranker-runtime-dependencies.json`
- Modify: `tests/contract/test_m2_t02_reranker_runtime_dependencies.py`
- Create: `evaluation/source-artifacts/m2-t02-reranker-runtime-installation.json`
- Create: `tests/contract/test_m2_t02_reranker_runtime_installation.py`

**Interfaces:**

- Consumes: the `embedding` group with `FlagEmbedding==1.3.5` and `torch==2.4.1`, plus the B2-D-R.1 five-package selection lock.
- Produces: a `reranker` extra containing the same Torch exact pin, clean-environment installation evidence, and offline contracts which distinguish dependency installation from all prohibited model operations.

**Boundary:** B2-D-F is the first task allowed to install runtime dependencies. It may modify only the files listed above. It must not download a model or tokenizer; read model weights; construct `BgeRerankerProvider`; load a tokenizer or model; run inference; create scores, timings, or benchmarks; modify `STATUS.md`; or begin B2-L, B2-P, B3, M2-T03, or M3. The temporary environment must be deleted after evidence is recorded.

- [ ] **Step 1: Add the offline installation-evidence red contract**

  Create `tests/contract/test_m2_t02_reranker_runtime_installation.py`. It must read only the committed JSON artifact, require exactly the top-level fields `report_version`, `phase`, `task_id`, `baseline_commit`, `decision_status`, `platform`, `installation_policy`, `installed_packages`, `verification`, and `execution_state`, and reject local absolute paths, usernames, credentials, model actions, scores, benchmarks, and timing claims. Require Windows, Python 3.12, a `clean_temporary_venv`, exactly `embedding` and `reranker` extras, CPU-only Torch base version `2.4.1`, `torch_cuda_version: null`, `torch_cuda_available: false`, the six direct packages, passing `pip check`, passing imports, and a true installation flag with every model-operation flag false.

  Run:

  ```powershell
  py -3.12 -m pytest -q tests/contract/test_m2_t02_reranker_runtime_installation.py
  ```

  Expected: FAIL only because `evaluation/source-artifacts/m2-t02-reranker-runtime-installation.json` does not yet exist. The test must never create a virtual environment or access the network.

- [ ] **Step 2: Declare the exact reranker group and synchronize selection evidence**

  Add this group without changing the base dependencies or the embedding pins:

  ```toml
  reranker = [
    "torch==2.4.1",
    "transformers==4.53.2",
    "huggingface-hub==0.34.3",
    "safetensors==0.5.3",
    "tokenizers==0.21.2",
  ]
  ```

  If the clean PEP 660 editable build reports multiple top-level packages, add only `[tool.setuptools.packages.find]` with `include = ["app*", "evaluation*"]` so the already committed Python package boundary is explicit. Do not add a runtime dependency or change the base dependency list to resolve that build configuration error.

  Update the dependency artifact and its existing offline contract to version `m2-t02-reranker-runtime-dependencies.v1.2`, baseline `00a323d89bbca08d21e13322da3a9b7c25e3ffa5`, decision status `selected_and_clean_environment_verified`, and `dependencies_installed_in_clean_environment: true`. Keep `model_downloaded`, `tokenizer_loaded`, `model_loaded`, `inference_run`, and `real_scores_generated` false.

- [ ] **Step 3: Install in one clean temporary environment**

  Set `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` and `HF_HUB_DISABLE_TELEMETRY=1` only for the controlled command session. Create a new venv under `%TEMP%` with `py -3.12 -m venv`, record `python -VV`, `pip --version`, and `pip list`, and prove no Torch, Transformers, FlagEmbedding, or Hugging Face runtime package is preinstalled. Install Torch exactly from the official CPU index:

  ```powershell
  & $Python -m pip install "torch==2.4.1" --index-url "https://download.pytorch.org/whl/cpu"
  ```

  Immediately assert the base version is `2.4.1`, `torch.version.cuda is None`, and `torch.cuda.is_available() is False`. From the repository root, install the combined extras without `--no-deps`:

  ```powershell
  & $Python -m pip install -e ".[embedding,reranker]"
  & $Python -m pip check
  ```

  Assert the final metadata versions, import only `torch`, `transformers`, `huggingface_hub`, `safetensors`, `tokenizers`, and `FlagEmbedding`, and record only safe platform, dependency, and verification facts. Do not use a Hugging Face token or call a model or Hub API.

- [ ] **Step 4: Commit offline installation evidence and clean the environment**

  Add `evaluation/source-artifacts/m2-t02-reranker-runtime-installation.json` with the closed top-level schema from Step 1. Its `installed_packages` entries must each contain `name`, `expected_version`, `installed_version`, and `matched`; it must show the six required packages, CPU-only Torch, null CUDA version, unavailable CUDA, passed `pip check`, and passed imports. Its execution state must record only a true clean-install flag and false model download/load/inference/score flags. Delete the temporary venv with `Remove-Item -LiteralPath $Venv -Recurse -Force` and confirm it no longer exists.

- [ ] **Step 5: Verify the repository without excluding Windows symlink coverage**

  Run the two focused dependency contracts, `tests/unit/test_m2_t02_reranker_snapshot.py`, the full pytest suite, Ruff, mypy, project-documentation and M0/M1/M2-T01 validators, and `py -3.12 -m pip check`. Require all checks to pass; do not skip or exclude the Windows symlink tests.

- [ ] **Step 6: Publish only B2-D-F**

  ```powershell
  git add pyproject.toml docs/superpowers/plans/2026-07-29-m2-t02-reranker-provider.md evaluation/source-artifacts/m2-t02-reranker-runtime-dependencies.json evaluation/source-artifacts/m2-t02-reranker-runtime-installation.json tests/contract/test_m2_t02_reranker_runtime_dependencies.py tests/contract/test_m2_t02_reranker_runtime_installation.py
  git diff --cached --name-only
  git diff --cached --check
  git commit -m "build: verify shared reranker runtime installation"
  git push origin agent/m2-t02-reranker-provider
  ```

  Update Draft PR #11 with the B2-D-F commit, clean Windows Python 3.12 environment, CPU-only Torch command and CUDA assertions, combined extras, actual six-package versions, all validation outcomes, the remote Actions result, and the explicit statement that no model download, loading, inference, or scores occurred. Keep `STATUS.md` at M2 `1/5` and stop before B2-L.

### B2-L Controlled pinned snapshot download

**Goal:** Download exactly the B0.1-pinned seven-file snapshot only after the
B2-D-F clean-environment installation evidence is accepted, while preserving a
separate auditable record of the download and offline-reuse verification.

#### B2-L-R Live download adapter and evidence red contract

**Files:**

- Modify: `docs/superpowers/plans/2026-07-29-m2-t02-reranker-provider.md`
- Create: `tests/unit/test_m2_t02_reranker_snapshot_download_runner.py`
- Create: `tests/contract/test_m2_t02_reranker_snapshot_download.py`

**Goal:** Define the fail-closed future runner and download-evidence contract
without creating `scripts/download_m2_t02_reranker_snapshot.py`, contacting
Hugging Face, importing the real Hub client, downloading a model, creating a
snapshot, loading a tokenizer/model, or running inference.

- [ ] Dynamically import only the future runner in its red tests so collection
  succeeds and the absent runner is the single intentional root cause. Do not
  use `skip` or `xfail`.
- [ ] Lock `HUGGINGFACE_HUB_VERSION == "0.34.3"`, the BAAI model ID, pinned
  revision, seven-file selection allowlist, fixed selection/snapshot/evidence
  paths, and the `run_download`/`main` interfaces. CLI options must never
  override those fixed values.
- [ ] Lock a default-deny `execute_live_download` switch: a false function
  argument and a CLI invocation without `--execute-live-download` must not
  import the Hub client, call preparation, create `models/`, create evidence,
  or access the network.
- [ ] Use fake Hub imports, fake `hf_hub_download`, and fake preparation
  results to specify late Hub import after
  `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` and `HF_HUB_DISABLE_TELEMETRY=1`, exact
  package-version checking before preparation, `token=False`, and forwarding
  of the injected downloader and `disk_usage` to the existing preparation
  core. Do not set `HF_HUB_OFFLINE`, read token/cookie/authorization sources,
  or import model runtimes.
- [ ] Specify `PUBLISHED` and `REUSED` handling, fail-closed mappings for
  missing/wrong/unreadable Hub versions, Hub import errors, preparation errors
  and unknown statuses, wrong snapshot paths, and evidence-write failures.
  Preparation owns staging, byte/digest checks, publication, cleanup, and
  reuse; the runner must not reimplement them or delete a published snapshot
  after an evidence-write failure.
- [ ] Specify atomic download evidence publication through one uniquely named
  `.m2-t02-reranker-snapshot-download.json.tmp-<uuid>` sibling and one
  `os.replace`, including cleanup of only this invocation's temporary file.
  Lock the closed evidence schema, exact seven-file projection from the
  immutable selection artifact, privacy/secret exclusions (including the
  `r"bearer\s"` expression), download verification, and non-inference state.
- [ ] Add characterization checks that the committed selection artifact remains
  `selected_not_downloaded` with every execution-state flag false, and that
  the installation evidence still records Hub `0.34.3`, CPU Torch `2.4.1`,
  `pip_check == "passed"`, `all_imports == "passed"`, and no model download.
- [ ] Run the two focused test files. The two immutable-artifact
  characterizations pass; all future-runner/evidence cases remain red solely
  because the runner does not exist. Run Ruff on the two new files, then commit
  only the plan and these red tests as
  `test: define controlled reranker snapshot download contract`.

#### B2-L-R.1 Offline reuse and runtime-platform provenance closure

**Files:**

- Modify: `docs/superpowers/plans/2026-07-29-m2-t02-reranker-provider.md`
- Modify: `tests/contract/test_m2_t02_reranker_snapshot_download.py`
- Modify: `tests/unit/test_m2_t02_reranker_snapshot_download_runner.py`

**Goal:** Close false-green gaps in the future controlled-download contract:
evidence must prove a genuinely offline second preparation after a newly
published snapshot, and must identify the actual download execution platform
without leaking host, user, credential, or path data.

**Boundary:** B2-L-R.1 remains an intentional-red test/plan task. Do not
create `scripts/download_m2_t02_reranker_snapshot.py` or
`evaluation/source-artifacts/m2-t02-reranker-snapshot-download.json`; do not
contact Hugging Face, import the real Hub client, download a model/tokenizer,
create a formal snapshot, load a model, run inference, generate scores, change
the immutable selection/installation artifacts, modify `STATUS.md`, or begin
B2-L-F, B2-P, or B3. Dynamic import must retain the absent runner as the only
red root cause; do not use `skip` or `xfail`.

- [ ] Add fake-live runner contracts that monkeypatch `platform.system`,
  `platform.machine`, and `platform.python_version` before `run_download`.
  Require evidence to match injected `Windows` / `AMD64` / `3.12.10` and
  `Linux` / `x86_64` / `3.12.13` values exactly. The future runner must query
  its current execution environment; it must neither hard-code Windows nor
  copy platform fields from runtime-installation evidence.

- [ ] Make `validate_download_evidence()` close the exact `platform` object:
  non-empty safe `system` and `machine` strings and a Python `3.12.x` version
  only. Reject empty fields, non-3.12 Python, absolute paths, credential-like
  strings, extra fields, and missing fields. Accept ordinary architecture
  strings including `AMD64`, `x86_64`, and `arm64`; do not classify every
  machine string as a host name.

- [ ] Replace ambiguous `weights_downloaded` and `tokenizer_downloaded`
  evidence with the closed execution-state fields
  `snapshot_present_and_verified`, `download_performed_this_run`,
  `model_loaded`, `inference_run`, and `real_scores_generated`. Preserve
  `snapshot.preparation_status` as the original preparation result. Require
  `PUBLISHED` to report `downloaded_and_verified` and
  `download_performed_this_run: true`; require `REUSED` to report
  `reused_and_verified` and `download_performed_this_run: false`. Model
  loading, inference, and real scores remain false in both states.

- [ ] Require a first `PUBLISHED` preparation to invoke the real injected Hub
  downloader and disk-space callback once, followed by a second preparation
  with the identical selection path and snapshot directory. The second call
  receives forbidden downloader and `disk_usage` callbacks that fail if used,
  must return `REUSED`, and must leave the snapshot directory unchanged. Only
  after that second call may evidence set `offline_reuse_check: "passed"` and
  `downloader_called_during_offline_reuse: false`.

- [ ] Add the already-`REUSED` first-call case: run preparation exactly once,
  use neither downloader nor disk-space callback, return `REUSED`, and emit
  the reuse decision without fabricating a preceding `PUBLISHED` event.

- [ ] Add parameterized second-call failures for repeated `PUBLISHED`,
  `UNKNOWN`, wrong snapshot path, downloader use, disk-space callback use, and
  preparation exceptions. Each must create no successful evidence, preserve
  the immutable selection and installation artifact bytes, preserve the
  first-published snapshot, clean only this invocation's evidence temporary
  file, and raise `OFFLINE_REUSE_FAILED` or
  `PREPARATION_RESULT_INVALID` as appropriate.

- [ ] Add `HF_HUB_OFFLINE=1` coverage. Before package-version lookup, Hub
  import, or preparation, the future runner must fail closed with
  `HUB_OFFLINE_MODE_ENABLED`, leave the user's environment variable unchanged,
  and create neither a snapshot nor evidence.

- [ ] Retain one uniquely named sibling evidence temporary file and a single
  `os.replace`. Validate the complete evidence before `os.replace`; a
  validation failure must raise `EVIDENCE_VALIDATION_FAILED`, call no replace,
  and remove only the invocation-owned temporary file. A pre-existing
  `.m2-t02-reranker-snapshot-download.json.tmp-sentinel` must survive every
  successful and failure path.

- [ ] Run and record the focused intentional-red baseline:

  ```powershell
  py -3.12 -m pytest -q `
    tests/contract/test_m2_t02_reranker_snapshot_download.py `
    tests/unit/test_m2_t02_reranker_snapshot_download_runner.py
  py -3.12 -m ruff check `
    tests/contract/test_m2_t02_reranker_snapshot_download.py `
    tests/unit/test_m2_t02_reranker_snapshot_download_runner.py
  ```

  Expected: the two immutable-artifact characterizations pass; every remaining
  future-runner/evidence test fails solely because the runner module is absent.
  Ruff passes. Do not claim a real download, formal snapshot, model load, or
  inference from this test-only result.

- [ ] Commit only these three files as
  `test: harden reranker download evidence provenance`, push the existing
  branch without rebasing or force-pushing, and update Draft PR #11 with the
  actual intentional-red pass/fail counts, the single missing-runner root
  cause, this offline-reuse and platform-provenance closure, and the explicit
  B2-L-F boundary.

#### B2-L-R.2 Zero-attempt offline reuse closure

**Files:**

- Modify: `docs/superpowers/plans/2026-07-29-m2-t02-reranker-provider.md`
- Modify: `tests/contract/test_m2_t02_reranker_snapshot_download.py`
- Modify: `tests/unit/test_m2_t02_reranker_snapshot_download_runner.py`

**Goal:** Remove the false green in which a successful fake `REUSED` path
attempted forbidden callbacks and swallowed their failures.

- [ ] In every successful first-`PUBLISHED` then second-`REUSED` fake flow,
  retain the forbidden downloader and disk-space callbacks but make zero
  callback attempts. Record and assert independent zero-attempt lists, two
  preparation calls, `PUBLISHED` then `REUSED` results, one first-call Hub
  download, one first-call disk check, and identical selection/snapshot paths.
- [ ] Keep callback invocation exclusively in the parameterized failure
  contracts. Record the attempted second-call callback before invoking it,
  allow its `AssertionError("offline reuse ...")` to reach the future runner,
  and require `OFFLINE_REUSE_FAILED` with no successful evidence, preserved
  snapshot/artifacts/sentinel, and owned-temporary-file cleanup.
- [ ] This remains a test/plan-only intentional-red task: do not create the
  runner or a snapshot artifact, download a snapshot, load tokenizer/model, or
  run inference/scores. B2-L-F-I alone implements the runner; B2-L-F-Live
  alone performs a real download; B2-P alone loads tokenizer/model; B3 alone
  reranks 33 papers. Keep `STATUS.md` at M2 `1/5`.

#### B2-L-F Controlled pinned snapshot live download

**Files:**

- Create: `scripts/download_m2_t02_reranker_snapshot.py`
- Create: `evaluation/source-artifacts/m2-t02-reranker-snapshot-download.json`

**Goal:** In a separately authorized, networked task, use the exact
`huggingface-hub==0.34.3` client to download only the immutable seven-file
allowlist for `BAAI/bge-reranker-v2-m3` revision
`953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`, through the existing
`scripts.prepare_m2_t02_reranker_snapshot.prepare_snapshot` core.

- [ ] Require the explicit `--execute-live-download` flag before importing the
  Hub client. Set the two token/telemetry environment guards before the delayed
  import, require the exact Hub version before any preparation call, and use
  only `huggingface_hub.hf_hub_download` with `repo_type="model"` and
  `token=False`.
- [ ] Pass the fixed selection and snapshot paths plus the Hub downloader and
  `shutil.disk_usage` to `prepare_snapshot`; do not duplicate its disk gate,
  staging, digest, Git-blob, atomic-publication, cleanup, or reuse logic.
- [ ] After a verified `PUBLISHED` or `REUSED` preparation result, write and
  self-validate closed UTF-8 JSON download evidence atomically. The evidence
  records only the fixed model/provenance, exact selection-file projection,
  verification facts, and non-loading/non-inference execution state. It must
  never contain credentials, host/user data, absolute paths, file contents,
  scores, benchmarks, or elapsed time.
- [ ] Do not load a tokenizer/model, construct `BgeRerankerProvider`, run
  inference, generate scores, modify the selection or installation evidence,
  modify `STATUS.md`, or begin B2-P, B3, M2-T03, or M3.

#### B2-L-F-Live Successful controlled download

This separately authorized live closure was executed exactly once on Windows
with the existing temporary environment. It used `Python 3.12.10`,
`huggingface-hub==0.34.3`, no `hf-xet`, and the Hub regular HTTP fallback.

- [x] **Supporting-file-first download:** The actual completion order was
  `README.md`, `config.json`, `sentencepiece.bpe.model`,
  `special_tokens_map.json`, `tokenizer.json`, `tokenizer_config.json`, then
  `model.safetensors`. Diagnostic ordinals remain canonical: ordinal `4` for
  `sentencepiece.bpe.model` and ordinal `3` for `model.safetensors`.
- [x] **Publication and verification:** All seven allowlisted files completed;
  `preparation_status` was `PUBLISHED`, `decision_status` was
  `downloaded_and_verified`, the aggregate snapshot size was
  `2,293,259,337` bytes, digest verification passed, atomic publication
  passed, and the internal offline `REUSED` check passed.
- [x] **Evidence and artifact boundary:** The closed evidence JSON was written
  and self-validated; staging is absent, the snapshot is not tracked by Git,
  and the selection and installation SHA-256 values are unchanged.
- [x] **Runtime boundary:** No tokenizer or model was loaded; inference,
  scores, and benchmarks were not run. B2-P and B3 have not started, and no
  further live runner invocation is permitted for this closure.
- [x] **Project-doc validation repair:** The project documentation validator
  now excludes ignored external `models/**` snapshot Markdown from repository
  local-link checks, so model-card asset links do not invalidate project docs.
  No snapshot content was changed.
- [x] **Offline gates:** The evidence contract passed (`2` tests), the related
  M2-T02 focused suite passed (`175` tests), the full suite passed (`907`
  tests), and Ruff, both mypy checks, all project/evidence validators, and
  `pip check` passed.

### B2-P CPU float32 preflight

**Goal:** In a later, separately authorized task, run the first local-only CPU `float32` tokenizer/model preflight against the verified B2-L snapshot.

**Boundary:** B2-P is the first task allowed to load a tokenizer and model. It
must use `local_files_only=True`, `trust_remote_code=False`, and safetensors.
It may record tokenizer/model loading evidence but must not score the full
33-candidate set, modify `STATUS.md`, or begin B3, M2-T03, or M3. Until the
later C closure task, M2 remains `1/5`; B3 alone may run the complete 33-paper
candidate set.

#### B2-P-I Controlled local CPU float32 preflight implementation

**Files:**

- Create: `scripts/run_m2_t02_reranker_preflight.py`
- Create: `tests/unit/test_m2_t02_reranker_preflight_runner.py`
- Create: `tests/contract/test_m2_t02_reranker_preflight.py`
- Modify: `docs/superpowers/plans/2026-07-29-m2-t02-reranker-provider.md`
- Do not create: `evaluation/source-artifacts/m2-t02-reranker-preflight.json`

**Goal:** Implement only the offline-controlled runner and fake-runtime
contracts for a future CPU `float32` preflight. This task performs no real
tokenizer/model import, no real inference, no network operation, and no formal
preflight-evidence publication.

**Fixed interfaces and boundaries:**

- The only successful CLI argument vector is
  `py -3.12 -m scripts.run_m2_t02_reranker_preflight --execute-local-preflight`.
  `main(argv)` returns `2` for every other vector before runtime imports or
  writes.
- The runner exposes `run_preflight(*, execute_local_preflight=False,
  memory_probe=None)`, `validate_preflight_evidence(value)`, and the fixed
  `PREFLIGHT_EVIDENCE_PATH`; tests may monkeypatch fixed paths and fake modules
  only through Python injection, never through CLI overrides.
- Module import may use standard library plus the existing pure-standard
  `scripts.prepare_m2_t02_reranker_snapshot` and download-evidence validator;
  `torch`, `transformers`, `huggingface_hub`, `safetensors`, `tokenizers`, and
  `FlagEmbedding` remain delayed imports.
- Before delayed runtime imports, set `HF_HUB_OFFLINE=1`,
  `TRANSFORMERS_OFFLINE=1`, `HF_HUB_DISABLE_IMPLICIT_TOKEN=1`, and
  `HF_HUB_DISABLE_TELEMETRY=1`; restore every caller value in `finally`.
- Revalidate the fixed selection and committed download evidence with
  `load_snapshot_plan(...)` and `validate_download_evidence(...)`, using the
  evidence-recorded platform values for the historical evidence check. Call
  `prepare_snapshot(...)` with forbidden downloader and disk callbacks and
  require `REUSED`; any callback attempt maps to
  `SNAPSHOT_REUSE_VERIFICATION_FAILED`.
- Gate exact versions before runtime imports: torch distribution
  `2.4.1+cpu` with base `2.4.1`, transformers `4.53.2`, huggingface-hub
  `0.34.3`, safetensors `0.5.3`, and tokenizers `0.21.2`. Reject missing,
  drifted, CUDA, or unknown-local torch builds with closed error codes.
- Use exactly `AutoTokenizer.from_pretrained(snapshot_path,
  local_files_only=True, trust_remote_code=False)` and
  `AutoModelForSequenceClassification.from_pretrained(snapshot_path,
  local_files_only=True, trust_remote_code=False, use_safetensors=True,
  torch_dtype=torch.float32)`, then `model.to("cpu")` and `model.eval()`.
- Use the two fixed synthetic pairs, `batch_size=2`, `max_length=512`,
  `padding=True`, `truncation=True`, and `return_tensors="pt"`. Serialize the
  pair tuple as compact UTF-8 JSON and record only its SHA-256.
- Run one `torch.inference_mode()` call and validate only CPU placement,
  batch dimension `2`, pair-compatible logits count, `float32` dtype, and
  finite logits. Never normalize, rank, print, persist, or return raw logits.
- Inject a standard-library Windows memory probe and require strict
  non-negative integers for total/available system memory, process working
  set before load, and process peak working set after load/inference. Probe
  failure is `MEMORY_PROBE_FAILED` and cannot produce success evidence.
- Build and validate the closed `m2-t02-reranker-preflight.v1` schema with
  `decision_status=cpu_float32_preflight_passed`, fixed runtime/load policy,
  fixture hash, logits shape/dtype/finite facts, memory facts, and explicit
  false flags for persisted logits, full-candidate reranking, and real scores.
  Reject absolute paths, URLs, credentials, host/user data, raw logits, scores,
  candidate identifiers, and extra fields.
- Publish only through candidate validation, one uniquely named sibling temp,
  one `os.replace`, and cleanup of that invocation's temp. Identical existing
  regular evidence is idempotent; different, malformed, symlink, junction, or
  directory targets return `PREFLIGHT_EVIDENCE_CONFLICT`.

- [x] **Step 1: Add red import, CLI, identity, and delayed-runtime contracts**

  Assert that importing the runner loads none of the runtime packages, that
  missing/unknown/repeated/override arguments return `2` without reading
  artifacts or writing evidence, and that the only accepted flag is
  `--execute-local-preflight`.

- [x] **Step 2: Add red evidence, REUSED, and package-gate contracts**

  Inject the production preparation and download validators, assert the
  evidence platform is used for validation, require `REUSED`, record zero
  downloader/disk callback attempts, and cover evidence drift plus missing or
  drifted runtime package metadata with the named closed codes.

- [x] **Step 3: Add red fake-runtime and memory contracts**

  Supply fake torch/transformers modules through `sys.modules` or delayed
  import injection. Assert exact tokenizer/model kwargs, offline environment
  ordering and restoration, CPU/float32 gates, one inference-mode call,
  fixed-pair hash, shape/dtype/finite validation, no raw-logit persistence,
  and memory probe success/failure paths.

- [x] **Step 4: Add red closed-schema and atomic-publication contracts**

  Validate the exact nested schema and forbidden-value closure. Exercise
  identical-file idempotence without mtime change, conflict for changed or
  malformed files and symlink/junction/directory targets, one sibling temp,
  one `os.replace`, and cleanup limited to the owned temp. Use only temporary
  test paths; never create the formal repository evidence path.

- [x] **Step 5: Implement the minimal runner and pass focused tests**

  Implement the fixed-path flow, delayed imports, package/memory gates,
  fake-compatible runtime adapter, closed errors, evidence validator, and
  atomic publisher without changing any existing production download,
  preparation, dependency, app, selection, installation, snapshot-download
  evidence, `STATUS.md`, or validator file.

  ```powershell
  py -3.12 -m pytest -q `
    tests/unit/test_m2_t02_reranker_preflight_runner.py `
    tests/contract/test_m2_t02_reranker_preflight.py
  ```

  Result: `29 passed`; the fake-only success path writes only under pytest's
  temporary directory and does not create the formal preflight evidence.

- [x] **Step 6: Run all offline gates and audit side effects**

  Run the full pytest, Ruff, both mypy commands, project/M0/M1/M2-T01
  validators, and `pip check`. Confirm the result exceeds the `907`-test
  baseline, the selection/installation/snapshot-download evidence bytes and
  `STATUS.md` are unchanged, no preflight evidence or cache exists, and
  `git ls-files models` is empty.

  Result: full pytest `936 passed`; Ruff, mypy (`45` source files plus the
  standalone runner), project docs, M0/M1/M2-T01 validators, and `pip check`
  passed. The formal preflight evidence is absent, the seven-file snapshot is
  unchanged and ignored, no staging sibling or active runner exists, and
  `git ls-files models` is empty. Selection SHA-256 remains
  `19e86cf2176969236e70ff9770ac086a5555d40892c0bfa913ac768eb752604b` and
  installation SHA-256 remains
  `feb79ad507afb4bb7027e5aa49765b941414fe6e9a9e3ac024e7eba6d2ced831`.

- [x] **Step 7: Commit and update Draft PR #11**

  Stage only the runner, two offline contract files, and this plan; commit as
  `feat: add controlled local reranker preflight`, push without amend/rebase/
  force, and record the exact focused/full counts, fake-only evidence type,
  no-real-load/no-real-inference boundary, B2-P-Live and B3 not started, and
  `STATUS.md` still at M2 `IN_PROGRESS (1/5)`.

  Result: implementation commit `c0d8baf78ff3ad26be47429fed289402e32300a5`
  was pushed to `agent/m2-t02-reranker-provider`. PR #11 remains open and
  Draft; this plan-status update is limited to the same authorized plan file.

#### B2-L-F-I.1 Runner startup, repository paths, evidence idempotence, and mypy boundary

**Files:**

- Modify: `pyproject.toml`
- Modify: `scripts/download_m2_t02_reranker_snapshot.py`
- Modify: `tests/unit/test_m2_t02_reranker_snapshot_download_runner.py`
- Modify: `tests/contract/test_m2_t02_reranker_snapshot_download.py`
- Modify: `docs/superpowers/plans/2026-07-29-m2-t02-reranker-provider.md`

**Goal:** Harden the already implemented controlled runner without carrying
out a real download. The canonical entry point must be
`py -3.12 -m scripts.download_m2_t02_reranker_snapshot`; direct execution may
bootstrap only the `__file__`-derived repository root. Operational selection,
snapshot, and evidence paths must not depend on the caller's working directory.

**Interfaces:**

- Consumes: `load_snapshot_plan(path: Path) -> SnapshotPlan` and
  `prepare_snapshot(selection_path: Path, snapshot_dir: Path, ...)` from
  `scripts.prepare_m2_t02_reranker_snapshot`.
- Produces: `_operational_path(path: Path) -> Path`, verified-plan evidence
  construction, stable runner errors `SELECTION_VALIDATION_FAILED` and
  `EVIDENCE_CONFLICT`, and atomic idempotent evidence publication.

- [ ] **Step 1: Add the red startup, root-path, plan, offline, and evidence tests**

  Add subprocess smoke coverage for both module and direct-script invocation
  without `--execute-live-download`; each must return non-zero before Hub
  metadata/import/preparation and leave a non-repository CWD without `models/`
  or `evaluation/`. Add fake-Hub tests that prove all trimmed truthy
  `HF_HUB_OFFLINE` values (`1`, `TRUE`, `ON`, and `YES`, with the supplied case
  variants) fail before plan loading, metadata lookup, Hub import, preparation,
  or writes while preserving the caller environment. Add a malformed selection
  test that fails before Hub work and writes. Add an absolute-path monkeypatch
  test that receives repository-root operational paths at preparation and uses
  only repository-relative paths in evidence.

  Add plan-immutability coverage: copy the selection artifact, let
  `load_snapshot_plan()` validate it, mutate that file in the fake preparation
  callback, and assert the generated evidence retains the original plan's
  files, sizes, digests, and aggregate sizes. Add publication cases for:

  ```python
  existing_published + new_reused == DownloadRunnerError("EVIDENCE_CONFLICT")
  existing_reused + new_published == DownloadRunnerError("EVIDENCE_CONFLICT")
  malformed_json_or_symlink_or_directory == DownloadRunnerError("EVIDENCE_CONFLICT")
  existing_bytes == candidate_bytes  # no temp file, os.replace, byte, or mtime change
  ```

  For every conflict case, assert the snapshot and immutable selection and
  installation artifacts are unchanged and invocation-owned temporary files are
  removed. Run the two focused files and record their intentional red failures
  before implementation. No test may call a real Hub client or download a model
  or tokenizer.

- [ ] **Step 2: Restore the project mypy scope**

  Replace the current mypy configuration with exactly:

  ```toml
  [tool.mypy]
  python_version = "3.12"
  strict = true
  plugins = ["pydantic.mypy"]
  ```

  Do not replace it with another global test exclusion or make `mypy .` an
  acceptance command. The required production checks are
  `py -3.12 -m mypy app evaluation scripts` and
  `py -3.12 -m mypy scripts/download_m2_t02_reranker_snapshot.py`.

- [ ] **Step 3: Make the runner location-independent and verified-plan based**

  Define the fixed root as:

  ```python
  _REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

  def _operational_path(path: Path) -> Path:
      if path.is_absolute():
          return path
      return _REPOSITORY_ROOT / path
  ```

  Before a direct script import, insert only this resolved root into `sys.path`
  when `__package__` is empty. Load one `SnapshotPlan` after the offline guard
  and before package metadata, Hub import, or preparation; map any selection
  parser failure to `SELECTION_VALIDATION_FAILED` without exposing its original
  exception. Use that one plan for `_expected_evidence()` and
  `validate_download_evidence()`. The preparation call and evidence file write
  must use operational paths, while serialized evidence retains the fixed
  repository-relative selection and snapshot identifiers.

- [ ] **Step 4: Publish evidence only if it is absent or byte-identical**

  Encode and validate the candidate JSON before a write. If the final evidence
  target is an existing regular file with identical bytes, return it without
  creating a temporary file or calling `os.replace`. If it is a different file,
  malformed JSON, symlink, directory, or another non-regular target, raise
  `EVIDENCE_CONFLICT`; preserve bytes and `mtime`, preserve the snapshot and
  immutable artifacts, and clean only the invocation's temporary file. When
  absent, keep one uniquely named sibling temporary file and one `os.replace`.

- [ ] **Step 5: Verify and publish the bounded task**

  Run:

  ```powershell
  py -3.12 -m pytest -q `
    tests/contract/test_m2_t02_reranker_snapshot_download.py `
    tests/unit/test_m2_t02_reranker_snapshot_download_runner.py
  py -3.12 -m pytest -q
  py -3.12 -m ruff check app evaluation scripts tests
  py -3.12 -m mypy app evaluation scripts
  py -3.12 -m mypy scripts/download_m2_t02_reranker_snapshot.py
  py -3.12 scripts/validate_project_docs.py
  py -3.12 scripts/validate_phase.py M0
  py -3.12 scripts/validate_m1_evidence.py
  py -3.12 scripts/validate_m2_t01_evidence.py
  ```

  Confirm no model files, formal download evidence, Hugging Face cache, or
  forbidden immutable-artifact/`STATUS.md` changes exist. Stage exactly the
  five authorized files, commit as
  `fix: harden reranker download runner execution boundaries`, push the
  existing branch without rebase, amend, or force push, then update Draft PR
  #11 with exact results and the explicit statement that B2-L-F-Live, B2-P,
  and B3 have not started.

#### B2-L-Diag Closed transport-failure diagnostics

**Goal:** Add an opt-in, closed diagnostic for a controlled snapshot-download
failure without retaining exception text, URLs, paths, credentials, or raw
exception objects. This task is offline only; it does not invoke Hugging Face,
download a model, alter CA configuration, or start B2-P or B3.

**Files:**

- Modify: `scripts/download_m2_t02_reranker_snapshot.py`
- Modify: `tests/unit/test_m2_t02_reranker_snapshot_download_runner.py`
- Modify: `tests/contract/test_m2_t02_reranker_snapshot_download.py`
- Create: `tests/unit/test_m2_t02_reranker_download_failure_diagnostics.py`
- Modify: `docs/superpowers/plans/2026-07-29-m2-t02-reranker-provider.md`

- [x] **Step 1: Add red offline diagnostic contracts**

  Test the fixed seven-file identity map, unknown-file fallback, safe
  certificate integers, bounded cycle-safe cause/context traversal, sealed
  type/module/family mapping, `hf_xet` classification, timeout/proxy/reset
  mapping, and hostile exceptions whose `__str__`/`__repr__` fail if touched.
  Cover a third-file wrapper failure and assert it reaches the runner as
  `PREPARATION_FAILED` with ordinal `3`, `weights`, and `lfs` while the
  preparation core remains unchanged. Assert default CLI stderr is only the
  fixed code; the explicit diagnostic flag emits exactly one compact,
  ASCII-safe JSON record with the closed schema and no forbidden text.

- [x] **Step 2: Verify red tests fail**

  Run `py -3.12 -m pytest -q
  tests/unit/test_m2_t02_reranker_download_failure_diagnostics.py` and confirm
  the missing closed diagnostic API is the failure cause.

- [x] **Step 3: Implement sealed runner-only diagnostics**

  Add frozen diagnostic and file-identity dataclasses, fixed literal unions,
  safe integer extraction, and a maximum-eight-node cause/context walk. The
  wrapper forwards downloader kwargs unchanged, retains only a closed
  diagnostic in run-local state, and raises a fixed-code internal error. Do
  not inspect exception text or mutate `prepare_snapshot`. Keep diagnostics
  disabled unless `--emit-closed-failure-diagnostic` accompanies the exact
  live switch.

- [x] **Step 4: Verify offline behavior and repository gates**

  Run the new diagnostic test plus the runner and download contracts, then the
  full pytest, Ruff, production mypy, validators, and `pip check`. Confirm no
  snapshot, evidence, failure-diagnostic file, cache, model artifact, or
  immutable selection/installation/`STATUS.md` change exists.

- [ ] **Step 5: Commit and publish the bounded diagnostic task**

  Stage only the five declared files, commit as `feat: add closed reranker
  download failure diagnostics`, push without amend/rebase/force, and update
  Draft PR #11. Record that real download remains unresolved and that any live
  diagnostic or Xet/CA adjustment requires separate authorization.

#### B2-L-Diag.1 Transport provenance closure

**Goal:** Preserve transport provenance from the offline exception-module
chain while retaining the most specific closed failure classification. This
closure is limited to diagnostic code and offline tests; it neither starts a
live diagnostic nor downloads any snapshot artifact.

**Files:**

- Modify: `scripts/download_m2_t02_reranker_snapshot.py`
- Modify: `tests/unit/test_m2_t02_reranker_download_failure_diagnostics.py`
- Modify: `docs/superpowers/plans/2026-07-29-m2-t02-reranker-provider.md`

- [x] **Step 1: Preserve module-chain transport provenance**

  Classify `xet` only when an exception in the bounded chain belongs to the
  `hf_xet` module family. Classify `http` only when the chain contains a
  `requests`, `urllib3`, or `httpx` module family. Do not infer either backend
  from a filename, file ordinal, storage type, or any snapshot metadata.

- [x] **Step 2: Keep specific failure and backend classifications independent**

  Apply the same module-chain backend selection to certificate verification,
  TLS handshake, connection reset, connection aborted, connect timeout, read
  timeout, proxy failure, and HTTP transport branches. Thus an `hf_xet`
  wrapper around SSL, timeout, or reset can report `transport_backend="xet"`
  alongside its more specific exception family; an HTTP wrapper around SSL can
  similarly report `transport_backend="http"`.

- [x] **Step 3: Respect explicit exception-link semantics**

  Traverse an explicit `__cause__` in preference to `__context__`; when no
  cause exists, traverse context only if `__suppress_context__ is not True`.
  If any link-attribute read raises, stop that node safely. Preserve the
  eight-node cap, cycle detection, closed exception-type mapping, and the ban
  on `str()` or `repr()` over exception objects.

- [x] **Step 4: Verify offline closure**

  Add offline combinations for Xet plus certificate, SSL, connection-reset,
  and read-timeout failures; HTTP plus SSL; and ordinary SSL without an Xet or
  HTTP module. Verify suppressed context does not participate in
  classification, while an explicit cause does and wins over a simultaneous
  context. No live diagnostic has run in this stage. Any future live
  diagnostic remains separately authorized; B2-P and B3 have not started.

#### B2-L-F-I.2 Supporting-file-first execution order

**Goal:** Change only the internal download execution order so the six
supporting files are fetched in their existing relative order before
`model.safetensors`, while preserving the immutable selection order, the
validated `SnapshotPlan.files` order, the formal evidence order, and all
fixed file identities. This is an offline implementation and contract task;
it must not invoke Hugging Face, create a real snapshot/evidence artifact, or
load a model.

**Files:**

- Modify: `scripts/prepare_m2_t02_reranker_snapshot.py`
- Modify: `tests/unit/test_m2_t02_reranker_snapshot.py`
- Modify: `tests/unit/test_m2_t02_reranker_download_failure_diagnostics.py`
- Modify: `tests/contract/test_m2_t02_reranker_snapshot_download.py`
- Modify: `docs/superpowers/plans/2026-07-29-m2-t02-reranker-provider.md`

**Interfaces:**

- Add `_execution_file_order(files: tuple[SnapshotFilePlan, ...]) -> tuple[SnapshotFilePlan, ...]`.
- Reject unknown, duplicate, missing, or multiple weight paths; valid output
  contains the same seven objects with non-weight entries in input order and
  `model.safetensors` last.
- Use this tuple only in the downloader loop. Plan parsing, snapshot
  verification, evidence construction, and diagnostic file ordinals continue
  to use the original canonical order.

- [x] **Step 1: Add failing offline contracts**

  Assert the exact execution order
  `README.md`, `config.json`, `sentencepiece.bpe.model`,
  `special_tokens_map.json`, `tokenizer.json`, `tokenizer_config.json`,
  `model.safetensors`; the original `SnapshotPlan.files` and selection/evidence
  order remain unchanged; all seven objects download exactly once; supporting
  failure never calls the weight downloader; weight failure cleans only owned
  staging; and REUSED calls no downloader. Add invalid-path, duplicate-path,
  and missing-weight fail-closed cases. Retain diagnostic ordinals 3 for the
  weight and 4 for `sentencepiece.bpe.model`.

- [x] **Step 2: Run red contracts**

  ```powershell
  py -3.12 -m pytest -q tests/unit/test_m2_t02_reranker_snapshot.py tests/unit/test_m2_t02_reranker_download_failure_diagnostics.py tests/contract/test_m2_t02_reranker_snapshot_download.py
  ```

  Expected: the new order assertions fail because the current loop iterates
  `plan.files` directly. No network or model runtime is permitted.

- [x] **Step 3: Implement the minimal order-only change**

  Implement the pure helper with exact path-set and uniqueness validation,
  preserve the input order of non-weight entries, append the single weight
  object, and replace only the preparation loop's iteration source. Do not
  alter `_FILE_SPECS`, selection parsing, digests, sizes, model/revision/path
  constants, cleanup, retry behavior, CA/proxy/timeout/Xet settings, evidence,
  STATUS, or the runner diagnostic map.

- [x] **Step 4: Run offline validation**

  ```powershell
  py -3.12 -m pytest -q
  py -3.12 -m ruff check app evaluation scripts tests
  py -3.12 -m mypy app evaluation scripts
  py -3.12 -m mypy scripts/download_m2_t02_reranker_snapshot.py
  py -3.12 scripts/validate_project_docs.py
  py -3.12 scripts/validate_phase.py M0
  py -3.12 scripts/validate_m1_evidence.py
  py -3.12 scripts/validate_m2_t01_evidence.py
  py -3.12 -m pip check
  ```

  Confirm no live download, model/tokenizer load, snapshot/evidence artifact,
  selection/installation byte change, or STATUS change occurred.

- [x] **Step 5: Commit and publish**

  Stage only the three B2-L-Evidence files plus the minimal project-doc
  validator repair, audit the diff, then use:

  ```powershell
  git commit -m "chore: record verified reranker snapshot download"
  git push origin agent/m2-t02-reranker-provider
  ```

  Update Draft PR #11 with the successful supporting-file-first live result,
  unchanged canonical/evidence order and network configuration, closed
  evidence and validation results, the minimal project-doc validator repair,
  and the statement that B2-P and B3 have not started.

#### B2-P-I.1 Windows memory probe, eval-mode, and preflight failure-contract closure

**Files:**

- Modify: `scripts/run_m2_t02_reranker_preflight.py`
- Modify: `tests/unit/test_m2_t02_reranker_preflight_runner.py`
- Modify: `tests/contract/test_m2_t02_reranker_preflight.py`
- Modify: `docs/superpowers/plans/2026-07-29-m2-t02-reranker-provider.md`
- Do not create: `evaluation/source-artifacts/m2-t02-reranker-preflight.json`

**Goal:** Close the local-only preflight safety contracts without loading a
real tokenizer/model, running real inference, changing the fixed CLI/model/
revision/path/runtime policy, or producing formal preflight evidence.

**Interfaces and fixed boundaries:**

- `_windows_memory_probe()` remains standard-library-only and returns exactly
  four strict non-negative integers. It explicitly declares
  `GlobalMemoryStatusEx`, `GetCurrentProcess`, and `GetProcessMemoryInfo`
  `argtypes`/`restype`; `GetCurrentProcess.restype` is `wintypes.HANDLE` so a
  64-bit handle cannot be truncated. Non-Windows and every API failure map to
  `MEMORY_PROBE_FAILED`.
- `model.eval()` is followed by direct observation of
  `model.training is False`. A missing attribute, no-op eval, or any other
  observed value maps to `MODEL_LOAD_FAILED`; evidence does not infer eval
  mode from the method call alone.
- Download evidence remains fixed to the committed PUBLISHED,
  `downloaded_and_verified` artifact and the pinned model/revision/seven-file
  projection. Runtime version absence maps to `RUNTIME_VERSION_UNAVAILABLE`;
  any version drift maps to `RUNTIME_VERSION_MISMATCH`; CUDA builds and CUDA
  availability map to `NON_CPU_TORCH_BUILD`.
- The fake runtime must receive exactly the two fixed synthetic pairs,
  tokenizer kwargs `{padding: True, truncation: True, max_length: 512,
  return_tensors: "pt"}`, one model forward call, CPU float32 tensors and
  logits, and no 33-candidate M1 input. Raw logits never enter evidence.
- Evidence publication remains closed and atomic: symlink, junction,
  directory, malformed target, or different regular-file targets conflict;
  byte-identical regular evidence is reused without mtime/replace changes;
  an absent target gets one sibling temp and exactly one `os.replace`; only
  the invocation-owned temp is cleaned after a write failure.

- [x] **Step 1: Add fake WinDLL, eval, runtime, evidence, and publication contracts**

  Add fake WinDLL coverage for strict integer results, both Windows API
  failures, non-Windows rejection, explicit ctypes signatures, and a
  64-bit-handle round trip. Add fake-model coverage for training state,
  tokenizer/model/parameter/CPU/float32/logit failures, exact pair and
  tokenizer arguments, one forward call, no M1 candidate expansion, and no
  raw-logit evidence. Add temporary-copy evidence mutations for decision
  status, PUBLISHED preparation status, and model/revision/file projection.
  Add package-not-found and every-runtime-version drift cases, CUDA cases,
  and temporary-directory publication cases for link/junction/directory/
  conflict/idempotence/single-temp/single-replace/owned-cleanup behavior.

- [x] **Step 2: Implement the minimal closed runner changes**

  Set the Windows declarations before the first API call:

  ```python
  kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(_MemoryStatusEx)]
  kernel32.GlobalMemoryStatusEx.restype = wintypes.BOOL
  kernel32.GetCurrentProcess.argtypes = []
  kernel32.GetCurrentProcess.restype = wintypes.HANDLE
  psapi.GetProcessMemoryInfo.argtypes = [
      wintypes.HANDLE,
      ctypes.POINTER(_ProcessMemoryCountersEx),
      wintypes.DWORD,
  ]
  psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
  ```

  Map probe failures to `MEMORY_PROBE_FAILED` and reject eval mode unless
  `getattr(model, "training", sentinel) is False` after applying `eval()`.
  Keep all existing fixed paths, privacy environment restoration, local-only
  load kwargs, CPU float32 gates, fixed fixture hash, closed schema, and
  atomic publisher behavior unchanged.

- [x] **Step 3: Run focused offline validation**

  ```powershell
  py -3.12 -m pytest -q `
    tests/unit/test_m2_t02_reranker_preflight_runner.py `
    tests/contract/test_m2_t02_reranker_preflight.py
  ```

  Result: `65 passed`; all runtime/model behavior is fake-only and all
  evidence writes use pytest temporary directories.

- [x] **Step 4: Run complete offline gates and side-effect audit**

  Run full pytest, Ruff, both requested mypy commands, project/M0/M1/M2-T01
  validators, and `pip check`. Confirm full pytest is greater than the prior
  `936 passed` baseline; formal preflight evidence, new model/cache files,
  real tokenizer/model load, real inference, STATUS changes, and immutable
  selection/installation/snapshot-download artifact changes are absent; and
  `git ls-files models` remains empty.

  Result: full pytest `972 passed in 121.17s`; focused contracts `65 passed`;
  Ruff and explicit `I001` check passed; mypy passed for 45 production source
  files and the standalone runner; project docs, M0, M1, M2-T01 validators,
  and `pip check` passed. `STATUS.md` and all three immutable M2-T02 source
  artifacts match their HEAD blobs byte-for-byte, the formal preflight
  evidence path is absent, no Hugging Face or Transformers cache path was
  created, and `git ls-files models` is empty. All preflight runtime tests
  used fake modules and temporary paths; no real tokenizer/model load or
  inference ran.

- [x] **Step 5: Commit, push, and update Draft PR #11**

  Stage only these four authorized files and commit exactly as:

  ```powershell
  git commit -m "test: close local reranker preflight safety contracts"
  git push origin agent/m2-t02-reranker-provider
  ```

  Record the B2-P-I.1 commit SHA, Windows API signature closure, observed
  eval mode, focused/full/static-validator counts, fake-only boundary, no
  formal preflight evidence, B2-P-Live not started, B3 not started, and
  `STATUS.md` still `M2 IN_PROGRESS (1/5)`. Keep PR #11 open and Draft; stop
  immediately after the PR record is updated.

  Completion record: commit `a50f1b9f5467c318dbbabe8807df6ede47d79b14`
  closed the B2-P-I.1 fake-only safety contracts. Its formal evidence path
  was intentionally absent at that commit; the later B2-P-Live execution is
  recorded below without rewriting that historical result.

#### B2-P-Live: Real local CPU float32 preflight PASS

**Execution boundary:**

- Execution HEAD: `a50f1b9f5467c318dbbabe8807df6ede47d79b14`.
- The fixed runner was invoked exactly once with
  `--execute-local-preflight`; exit code was `0`.
- The execution used Windows AMD64, Python `3.12.10`, torch `2.4.1+cpu`,
  transformers `4.53.2`, huggingface-hub `0.34.3`, safetensors `0.5.3`, and
  tokenizers `0.21.2`.
- Snapshot preparation was `REUSED`, with zero downloader and disk-usage
  callback attempts.
- Model loading was CPU float32, eval mode, local-only, no remote code, and
  safetensors-enabled. The minimal inference result had shape `[2, 1]`,
  dtype `float32`, and all finite values; logits were not persisted.
- `full_candidate_reranking_run` and `real_candidate_scores_generated` are
  both `false`. No 33-candidate run, ranking, score artifact, or raw logits
  were produced.

**Memory record:**

- `system_total_physical_memory_bytes = 16873545728`
- `system_available_physical_memory_before_bytes = 8793714688`
- `process_working_set_before_bytes = 121421824`
- `process_peak_working_set_after_load_bytes = 452280320`
- `process_peak_working_set_after_inference_bytes = 1676398592`

The process peak is approximately 1.56 GiB and does not equal complete model
RAM or commit usage; safetensors mapping and demand paging can keep the
post-load working set lower. Any later B3 run starts with batch size `2` and
must not place all 33 candidates in one batch.

**Evidence boundary:**

The preflight evidence retains its runner-published `baseline_commit` of
`7a234ff2684f24030dde91b3df6fe8483aa92688`, which is the stage baseline.
The execution receipt is the separate proof of the real execution HEAD
`a50f1b9f5467c318dbbabe8807df6ede47d79b14`. B2-P-Live is `PASS`; B3 has not
started; `STATUS.md` remains `M2 IN_PROGRESS (1/5)`.

#### B2-P-Evidence: Offline receipt and CI evidence contract PASS

**Files:**

- Preserve byte-identical:
  `evaluation/source-artifacts/m2-t02-reranker-preflight.json`
- Create:
  `evaluation/source-artifacts/m2-t02-reranker-preflight-run-receipt.json`
- Create:
  `tests/contract/test_m2_t02_reranker_preflight_evidence.py`
- Modify the historical pre-live contract only to preserve existing formal
  evidence bytes/mtime when the CLI is invoked without its execution flag:
  `tests/contract/test_m2_t02_reranker_preflight.py`
- Modify this plan only for the evidence and execution record.

The execution receipt is a closed, privacy-safe schema with `run_count = 1`,
`exit_code = 0`, `decision_status = cpu_float32_preflight_passed`, and
SHA-256 bindings for the formal evidence, runner, selection artifact, runtime
installation artifact, and snapshot-download evidence. It contains no
absolute paths, host/user/venv details, timestamps, URLs, credentials, raw
logits, candidate content, scores, or rankings.

The CI contract validates the formal evidence through production
`validate_preflight_evidence`, checks strict closed schemas and byte hashes,
and runs in an isolated temporary directory without a local `models/**`
snapshot. It imports no torch, transformers, tokenizers, or safetensors and
does not load a tokenizer/model or execute inference. The existing historical
pre-live contract is updated only for the now-materialized evidence file: its
no-flag CLI test verifies byte and mtime preservation rather than absence.

The formal evidence SHA-256 is
`103a73cc82674c2c1f1139a72a7a3d1a880029a7c01d292f52de24d6dc781b06`; the
runner SHA-256 is
`38ec370acd26a315fdb57bd405e9348081e37ea5f26bebb362cfcd3febd930d7`.

#### B3-Evidence: Freeze the verified real candidate reranking result

**Goal:** Preserve the runner-generated 33-candidate result byte-for-byte,
bind it to the exact execution HEAD and all fixed input/evidence bytes, and
add a fully offline contract that validates the result without importing a
runtime package or loading a model.

**Files:**

- Preserve byte-identical:
  `evaluation/source-artifacts/m2-t02-reranker-candidate-run.json`
- Create:
  `evaluation/source-artifacts/m2-t02-reranker-candidate-run-receipt.json`
- Create:
  `tests/contract/test_m2_t02_candidate_reranking_evidence.py`
- Modify only if an existing assertion still requires the now-materialized
  formal result to be absent:
  `tests/contract/test_m2_t02_candidate_reranking.py`
- Modify this plan with the B3-Live/B3-Evidence execution and validation
  record.
- Do not modify `scripts/run_m2_t02_candidate_reranking.py`,
  `app/adapters/reranking.py`, `app/core/reranking.py`,
  `app/models/reranking.py`, `pyproject.toml`, `STATUS.md`, any existing
  evidence/snapshot/manifest, or `models/**`.

**Closed receipt schema:**

The receipt must contain exactly the fields `report_version`, `phase`,
`task_id`, `execution_commit`, `result_path`, `result_sha256`, `runner_path`,
`runner_sha256`, `candidate_snapshot_path`, `candidate_snapshot_sha256`,
`candidate_manifest_path`, `candidate_manifest_sha256`,
`model_selection_path`, `model_selection_sha256`,
`runtime_installation_path`, `runtime_installation_sha256`,
`snapshot_download_evidence_path`, `snapshot_download_evidence_sha256`,
`preflight_evidence_path`, `preflight_evidence_sha256`,
`preflight_receipt_path`, `preflight_receipt_sha256`, `runtime`, `exit_code`,
`run_count`, `decision_status`, `formal_result_created`,
`full_candidate_reranking_run`, `additional_model_rerun_performed`, and
`replay_performed`. All paths are repository-relative POSIX identifiers;
all SHA-256 values are calculated from the current file bytes. The receipt
must not contain absolute paths, usernames, hostnames, URLs, credentials,
candidate content, scores, rankings, tokens, token IDs, logits, or model
parameter data.

**Offline contract boundary:**

The contract must call production `validate_candidate_run_report`, require an
ordinary UTF-8 JSON file and closed top-level/record schemas, verify 33 fixed
records, ranks 1..33, candidate-ID/input-hash identity, finite raw scores,
global min-max normalized scores, descending raw-score order with paper-ID
tie-breaking, and exact execution counts (`1`, `1`, `17`, `33`, `[2] * 16 +
[1]`, `0`). It must verify every receipt hash against actual bytes and run a
subprocess from a temporary directory with no `models/**` directory. That
subprocess may import the production validator but must not import `torch`,
`transformers`, `tokenizers`, or `safetensors`, load a tokenizer/model, run
inference, create a cache, or modify the formal result.

- [x] **Step 1: Record the immutable live-run facts**

  Reconfirm the exact execution HEAD
  `c1f95238480d8cfe69a79f705d98caadbf372220`, preserve the runner-generated
  result bytes, calculate its SHA-256, calculate the runner and every bound
  artifact SHA-256 from disk, and confirm `git ls-files models` is empty.

  Completion record: the live run started at `2026-08-02T19:51:28.6256910+08:00`
  and ended at `2026-08-02T19:52:47.6591386+08:00` (79.034 seconds), with
  exit code `0`. The run used the fixed HEAD, Python `3.12.10`, torch
  `2.4.1+cpu`, transformers `4.53.2`, huggingface-hub `0.34.3`, safetensors
  `0.5.3`, tokenizers `0.21.2`, and pydantic `2.9.2`; CUDA was `None` and
  unavailable. It processed 33 candidates in 17 serial batches of
  `[2] * 16 + [1]`, with one provider instance/load, 33 scored candidates,
  and zero cache hits. The result SHA-256 is
  `0a751dbc35bfa8d07433796113939412bfbdffd4c5c13043c90390bccfc5c35b`;
  the runner SHA-256 is
  `3ad6c629b2f6b1f28c3119f9f7d810226e28e6d572535e28b0de6566ef8f63b1`.
  The run began with 9.045 GiB available physical memory out of 15.715 GiB.
  The formal result was atomically published, and no second model run or
  replay was performed. `git ls-files models` remained empty.

- [x] **Step 2: Add the failing offline evidence contract**

  Create `tests/contract/test_m2_t02_candidate_reranking_evidence.py` with
  ordinary-file, UTF-8, production-validator, closed-schema, exact-count,
  score/ranking, execution, receipt-hash, privacy, and no-local-model
  subprocess tests. Before the receipt exists, run:

  ```powershell
  py -3.12 -m pytest -q tests/contract/test_m2_t02_candidate_reranking_evidence.py
  ```

  Expected: failure only because the authorized receipt file is not yet
  present; no model/runtime import or inference is allowed.

  Completion record: the initial contract run was intentionally red before
  receipt creation. Its preflight-commit assertion was corrected to preserve
  the historical preflight receipt binding; the formal candidate result was
  not changed.

- [x] **Step 3: Create the execution receipt from measured bytes**

  Add the closed JSON receipt using the actual result, runner, candidate,
  selection, runtime-installation, snapshot-download, preflight, and
  preflight-receipt SHA-256 values. Set `execution_commit` to the actual
  B3-Live HEAD, `exit_code` and `run_count` to `0` and `1`,
  `decision_status` to `candidate_reranking_completed`,
  `formal_result_created` and `full_candidate_reranking_run` to `true`, and
  `additional_model_rerun_performed` and `replay_performed` to `false`.

  Completion record: the closed receipt was created at
  `evaluation/source-artifacts/m2-t02-reranker-candidate-run-receipt.json`
  with all fixed artifact hashes bound from disk. It records
  `exit_code = 0`, `run_count = 1`, `formal_result_created = true`,
  `full_candidate_reranking_run = true`, and both rerun/replay flags as
  `false`.

- [x] **Step 4: Run focused offline evidence validation**

  ```powershell
  py -3.12 -m pytest -q `
    tests/contract/test_m2_t02_candidate_reranking.py `
    tests/contract/test_m2_t02_candidate_reranking_evidence.py `
    tests/unit/test_m2_t02_candidate_reranking_runner.py `
    tests/unit/test_bge_reranker_provider.py
  ```

  Expected: all focused tests pass without invoking the accepted live CLI,
  loading a model, running inference, or changing result bytes/mtime.

  Completion record: `122 passed in 6.92s`. The focused evidence-only run
  separately passed `3 passed in 0.54s`.

- [x] **Step 5: Run the complete offline gates and side-effect audit**

  Run the full pytest suite, default and import-order Ruff checks, both mypy
  commands, project/M0/M1/M2-T01 validators, and `pip check`. Confirm the only
  worktree additions are the candidate-run JSON, receipt, evidence contract,
  and this plan before commit, `STATUS.md` remains M2 `IN_PROGRESS (1/5)`, and
  PR #11 remains Draft. Do not run the live reranker or B3-Replay during
  these checks.

  Completion record: the single full `py -3.12 -m pytest -q` invocation
  exceeded the 120-second command-executor ceiling and was not counted as a
  pass. Directory-complete offline coverage passed as `501 passed` for
  `tests/contract`, `502 passed` for `tests/unit`, and `34 passed` for
  `tests/integration` (1037 total, including the three new evidence tests).
  `ruff check` and import-order Ruff both passed; mypy passed for 46 source
  files and for the candidate runner; project docs, M0, M1, M2-T01, and pip
  check all passed. The evidence contract verified no model snapshot was
  needed, no runtime model package was imported by its isolated subprocess,
  and no formal result bytes or mtime were changed. B3-Replay was not
  started, and `STATUS.md` was not modified.

- [x] **Step 6: Commit and publish the evidence**

  Audit the explicit file scope, stage only the candidate result, receipt,
  evidence contract, and plan (plus the existing contract test only if its
  no-result assertion is actually repaired), then commit and push:

  ```powershell
  git commit -m "chore: record verified candidate reranking result"
  git push origin agent/m2-t02-reranker-provider
  ```

  Keep PR #11 Draft, do not amend/rebase/force-push, do not modify STATUS, and
  stop after recording the commit, validation results, result/receipt hashes,
  top-five IDs, and the fact that B3-Replay has not started.

**B3-Evidence completion record:**

- Formal result: 33 records, decision `candidate_reranking_completed`;
  normalized scores were verified by exact global min-max normalization while
  preserving runner raw scores and deterministic ordering.
- Top five: `arxiv:2602.17856` (`3.782480239868164`, `1.0`),
  `arxiv:2512.12760` (`3.653545618057251`, `0.9884483594410878`),
  `arxiv:2007.12731` (`2.9663286209106445`, `0.9268785217550191`),
  `arxiv:2511.05498` (`2.940110206604004`, `0.9245295351613396`), and
  `arxiv:2412.15232` (`2.8659772872924805`, `0.9178877439539677`).
- Last three: `arxiv:2510.26824` (`-3.939950466156006`,
  `0.3081242066392194`), `arxiv:1611.00097` (`-4.189328670501709`,
  `0.28578166326826043`), and `arxiv:2408.15002` (`-7.379105091094971`,
  `0.0`).
- Execution fields: provider instances `1`, runtime loads `1`, provider calls
  `17`, scored candidates `33`, batch sizes `[2] * 16 + [1]`, cache hits `0`.
- The result and receipt are evidence-only artifacts. B3-Replay has not
  started; `STATUS.md` remains `M2 IN_PROGRESS (1/5)` and PR #11 remains Draft.

#### M2-T02 C: Completion report, machine gates, and STATUS transition

**Scope:**

- Create `evaluation/reports/m2-t02-reranker.json`.
- Create `scripts/validate_m2_t02_evidence.py` and
  `tests/contract/test_m2_t02_evidence_validation.py`.
- Update only the historical M2-T01 status compatibility rule and its
  regression test, `STATUS.md`, this plan, and the docs-validation workflow.
- Preserve the candidate result, candidate execution receipt, all fixed input
  evidence, candidate snapshot/manifest, runner, provider, and model files.

**Completion record:**

- The closed report binds the fixed live execution commit
  `c1f95238480d8cfe69a79f705d98caadbf372220`, evidence/validated commit
  `5195a4eabfccdd64dac61f2efdc2dfc4a92242ae`, the result SHA-256
  `0a751dbc35bfa8d07433796113939412bfbdffd4c5c13043c90390bccfc5c35b`, and
  all ten validated input blobs. The validator reads each input with
  `git show <validated_commit>:<path>` and compares the submitted bytes before
  calling the production preflight and candidate-run validators.
- The report records the real CPU float32 run exactly once: 33 candidates,
  17 serial calls, one provider instance/load, batch sizes `16 x 2 + 1`, zero
  cache hits, and no rerun, Replay, benchmark, manual relevance judgement,
  or M2-T03 start. Fake acceptance is explicitly classified as deterministic
  fake; real-model evidence is separately classified as real local CPU
  float32 evidence.
- `STATUS.md` now records M2 `IN_PROGRESS 2/5`, M2-T02 reranker completion,
  next task M2-T03, and M3 `BLOCKED_BY_M2 0/5`; M2 is not marked COMPLETE.
  The historical M2-T01 validator now accepts M2 `IN_PROGRESS 1-4/5` or
  `COMPLETE 5/5` while still rejecting zero-progress, invalid totals,
  `READY 0/5`, and an unlocked M3.
- Focused C evidence tests passed `118`; the complete offline pytest suite
  passed `1049`. Ruff default/import-order, mypy (`47` source files), project
  docs, M1, M2-T01, M2-T02, and pip check passed. The pre-commit M0 phase
  validator correctly stopped only because the intentional STATUS change was
  uncommitted; it is rerun after commit. No model, tokenizer, reranking,
  Replay, network, or model-cache operation was performed.
- CI adds `python scripts/validate_m2_t02_evidence.py` after the M2-T01
  validator. PR #11 remains Draft until remote CI is green; no automatic
  merge or Ready-for-review transition is performed.
