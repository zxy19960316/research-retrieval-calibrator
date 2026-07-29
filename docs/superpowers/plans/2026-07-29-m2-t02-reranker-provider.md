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

## B2 Controlled download and preflight

B2 is the first task allowed to perform a controlled download. It must verify the B0 pinned revision and file digests before CPU float32 preflight. B2 is not authorized by B0.

## B3 Real live/replay evidence

B3 is the first task allowed to run real inference, generate real scores, and record separately labeled live and replay evidence. B3 is not authorized by B0.

## C Completion evidence and STATUS

Only C may consolidate completion evidence and consider `STATUS.md`; until C, M2 remains 1/5. B0 must not start B1, B2, B3, C, M2-T03, or M3.
