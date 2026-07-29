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

## File Structure

- Create: `app/models/reranking.py` — closed descriptor, input, raw/final record, state, cache-entry, run-stat, and error-code contracts.
- Create: `app/adapters/reranking.py` — `RerankerProvider` protocol only; later real adapters must implement it without changing core behavior.
- Create: `app/core/reranking.py` — title/abstract serialization, cache-key construction, batch orchestration, fail-closed output validation, global normalization, and stable order restoration.
- Create: `tests/contract/test_reranker_contracts.py` — synthetic red-first contract tests; no model runtime, network, or downloaded weight.

### Task 1: Close the data and provider contracts

**Files:**
- Create: `app/models/reranking.py`
- Create: `app/adapters/reranking.py`
- Test: `tests/contract/test_reranker_contracts.py`

**Interfaces:**
- Produces `RerankerProvider.score(query: str, inputs: Sequence[RerankerInput], *, batch_size: int) -> list[ProviderRawScore]`.
- Produces `RerankerModelDescriptor`, `RerankerInput`, `RerankRecord`, `RerankRun`, and `RerankerTaskError`.
- `RerankRecord` exposes `paper_id`, `raw_score`, `normalized_score`, `descriptor`, and `input_sha256` only when its state is `SCORED`.
- `RerankerTaskError.code` is one of `INVALID_INPUT`, `PROVIDER_UNAVAILABLE`, or `INVALID_OUTPUT`; the core uses `INVALID_INPUT` for invalid Top-K/count parameters.
- `RerankerModelDescriptor.input_format_version` is a non-blank versioned string, not a one-value literal, so a format revision produces a distinct cache identity.

- [ ] **Step 1: Write the failing contract tests**

Add tests named `test_reranker_records_preserve_raw_score_descriptor_and_input_sha`, `test_provider_output_must_match_each_requested_unique_paper_id`, and `test_non_finite_or_non_numeric_scores_are_invalid_output`.

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `py -3.12 -m pytest -q tests/contract/test_reranker_contracts.py`

Expected: FAIL during collection because `app.models.reranking` and `app.core.reranking` do not exist.

- [ ] **Step 3: Implement the closed models and protocol**

Define literal states `SCORED`, `NOT_RUN`, `PROVIDER_UNAVAILABLE`, and `INVALID_OUTPUT`; reject unpinned revisions, non-SHA input hashes, non-finite scores, score-bearing non-`SCORED` records, and unknown descriptor fields. Use a protocol rather than a concrete model dependency.

- [ ] **Step 4: Run the focused tests to verify the contracts pass**

Run: `py -3.12 -m pytest -q tests/contract/test_reranker_contracts.py`

Expected: contract tests pass using only synthetic providers.

- [ ] **Step 5: Commit**

```powershell
git add app/models/reranking.py app/adapters/reranking.py tests/contract/test_reranker_contracts.py
git commit -m "feat: add reranker provider contracts"
```

### Task 2: Implement deterministic input, cache, and batch orchestration

**Files:**
- Create: `app/core/reranking.py`
- Modify: `tests/contract/test_reranker_contracts.py`

**Interfaces:**
- Consumes `FrozenCandidate`, `RerankerProvider`, and `RerankerModelDescriptor`.
- Produces `build_reranker_input(candidate)`, `effective_top_k(configured_top_k, available_count)`, and `rerank_candidates(query, candidates, provider, cache_dir, *, batch_size, configured_top_k)`.

- [ ] **Step 1: Write the failing orchestration tests**

Add tests named `test_effective_top_k_caps_to_available_candidates_without_hard_coding_33`, `test_empty_abstract_serializes_title_only`, `test_cache_key_changes_when_revision_changes`, and `test_batch_sizes_1_2_8_33_preserve_raw_scores_and_final_order`.

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `py -3.12 -m pytest -q tests/contract/test_reranker_contracts.py`

Expected: FAIL because the orchestration functions are not implemented.

- [ ] **Step 3: Implement the minimal orchestration**

Serialize exactly `title:\n{title}` plus `\n\nabstract:\n{abstract}` only when the source abstract is non-empty. Batch only cache misses, restore requested identities, and include query/input/revision/format-version cache identity. Reject a provider exception as `PROVIDER_UNAVAILABLE`; do not persist a score or cache entry for that call.

- [ ] **Step 4: Run the focused tests to verify deterministic behavior**

Run: `py -3.12 -m pytest -q tests/contract/test_reranker_contracts.py`

Expected: synthetic batch and cache tests pass for batch sizes 1, 2, 8, and 33.

- [ ] **Step 5: Commit**

```powershell
git add app/core/reranking.py tests/contract/test_reranker_contracts.py
git commit -m "feat: orchestrate deterministic reranker batches"
```

### Task 3: Enforce global normalization, fail-closed outcomes, and stable ranking

**Files:**
- Modify: `app/core/reranking.py`
- Modify: `tests/contract/test_reranker_contracts.py`

**Interfaces:**
- Consumes complete validated `ProviderRawScore` values.
- Produces records sorted by `(-raw_score, paper_id)` after a single all-record min-max normalization pass.

- [ ] **Step 1: Write the failing ranking-safety tests**

Add tests named `test_global_normalization_prevents_batch_local_rank_inversion`, `test_provider_failure_never_becomes_zero_score`, `test_equal_scores_use_stable_paper_id_tie_break`, and `test_input_order_permutations_produce_the_same_final_order`.

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `py -3.12 -m pytest -q tests/contract/test_reranker_contracts.py`

Expected: FAIL until normalization follows full validated raw-score collection.

- [ ] **Step 3: Implement the minimal ranking safety logic**

Normalize once with `(raw - min_raw) / (max_raw - min_raw)` after complete validation, with `1.0` for one or all-equal records. On provider load/error raise `RerankerTaskError(code="PROVIDER_UNAVAILABLE")`; on malformed count/IDs/scores raise `RerankerTaskError(code="INVALID_OUTPUT")`; both stop the run without a zero-score substitute, partial ranked result, new cache JSON, or leftover temporary file.

- [ ] **Step 4: Run focused and full validation**

Run: `py -3.12 -m pytest -q tests/contract/test_reranker_contracts.py`

Expected: all M2-T02 contract tests pass with synthetic providers only.

Run: `py -3.12 -m pytest -q`

Expected: full regression passes without downloading a reranker model.

- [ ] **Step 5: Commit**

```powershell
git add app/core/reranking.py tests/contract/test_reranker_contracts.py
git commit -m "fix: fail closed on reranker output errors"
```

## Self-Review

- Spec coverage: the three tasks cover replaceability, query/title/abstract inputs, persisted raw provenance, core-owned batch chunks, batch invariance, fail-closed states, no zero-score fallback, transactional cache publication, one-pass min-max normalization, full-descriptor cache identity, title-only empty abstracts, Top-K bounds, deterministic output, and tie behavior.
- Placeholder scan: no task relies on an unspecified function name or test command.
- Type consistency: the provider returns `ProviderRawScore`; core validates it into `RerankRecord`; only `RerankRecord` enters cache and final order.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-29-m2-t02-reranker-provider.md`. The present branch stops after the plan and red tests. A later implementation can execute the three tasks inline, one red/green commit boundary at a time.
