# M2-T01 EmbeddingProvider and Frozen Vectors Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze the verified M1 real candidate set, build deterministic embedding inputs, providers, per-input caching, and a separately evidenced BGE-M3 dense-vector run without implementing any ranking, reranking, scoring, selection, feedback, or second-round retrieval behavior.

**Architecture:** Add closed M2 Pydantic contracts for frozen candidates, inputs, descriptors, vectors, cache entries, and evidence. Keep model execution behind an `EmbeddingProvider` protocol: the standard-library deterministic fake is the CI-only implementation, while BGE-M3 is a lazy optional adapter. A core orchestrator validates frozen inputs, constructs stable chunks, restores caller order after cache hit/miss mixing, and persists only JSON cache entries using atomic replacement.

**Tech Stack:** Python 3.12; Pydantic 2; standard-library `hashlib`/`json`/`math`/`pathlib`; pytest; Ruff; mypy; optional `FlagEmbedding`, `torch`, and their transitive model dependencies for the separately run BGE-M3 gate.

## Global Constraints

- Baseline before branch creation is `0eb45fc22d10adb72cb66aa45494333057080bc1` (PR #9 normal merge); do not develop from another commit.
- Scope is **M2-T01 only**. Do not add Cross-Encoder/reranking, similarity scoring, Top-K ranking, evidence slots, diversity selection, user feedback, second-round queries, Rocchio, CNKI, HTTP service, deployment, or LLM metadata generation.
- Preserve every M1 source field byte-for-byte in value: source, source ID, URL, title, authors, year, DOI, language, retrieval paths, cluster ID, and source identities. Never infer or manufacture metadata.
- The only text transformation for embedding input is CRLF-to-LF followed by trimming the title and abstract boundaries. Preserve internal text, Unicode, case, punctuation, and language.
- Use `m2-title-abstract-v1`; an absent or whitespace-only abstract yields title-only text and never a prose placeholder.
- CI runs only fake embedding tests. It must not download BGE-M3, contact Hugging Face, or require GPU hardware.
- A real descriptor must have an immutable revision, must not use `latest`, `main`, or an empty revision, and must use `embedding_mode="dense"`. Fake and real cache namespaces must differ.
- Cache identity is the canonical descriptor fields plus `text_sha256`; batch size is deliberately excluded. Cache entries are UTF-8 JSON with sorted keys and atomic writes, never pickle.
- If the historical M1 run output and the `first-round:real` replay cache are unavailable, return `M2_T01_FROZEN_INPUT_MISSING`. Fake implementation/tests may proceed, but no candidate/vector snapshots, BGE claim, evidence-completion commit, or M2-T01 completion may be produced.
- Real BGE success requires a fresh real cache run with zero hits and a same-config replay with zero provider calls, zero misses, and byte-identical vector arrays. A provider failure must not fall back to fake.
- Keep the three-commit boundary: implementation A, data snapshot B, evidence/status C. Roll back as `git revert <C>`, then `<B>`, then `<A>`; never rewrite history or force-push.

## Current Preflight Findings

- The required baseline, clean worktree, M0 validator, and M1 validator were verified on 2026-07-28 before branch creation.
- `evaluation/reports/m1-validation.json` records the accepted real M1 run: 51 raw records, 33 deduplicated candidates, candidate-array SHA-256 `e51eb84d4e772bba324a199478caa5f0983edef0b0dbf4c58fce5f9da5803209`, and `first-round.json` SHA-256 `069cd8c94d294c68bb051898d4c262f324b7e1f20eff10eabcf22d9bc435d178`.
- At planning time the workspace has neither `evaluation/runs/m1-live-r1-final/first-round.json` nor `evaluation/runs/m1-t04r1-cache-final/`. Therefore Task 2 must first take the stable blocked path unless the original, provenance-preserving artifacts are restored; recorded fixtures and a fresh network retrieval are explicitly invalid replacements.

## File Structure

| Path | Responsibility |
| --- | --- |
| `app/models/embedding.py` | Closed contracts, stable M2 error type/codes, descriptor and frozen/vector/cache models. |
| `app/adapters/embedding.py` | Provider protocol, deterministic fake provider, and lazy BGE-M3 dense provider. |
| `app/core/embedding.py` | Snapshot validation, embedding-text protocol, cache identity/read/write, batching orchestration, and vector validation. |
| `scripts/freeze_m2_candidates.py` | Creates or deterministically refuses the candidate snapshot from the M1 real output/cache replay. |
| `scripts/embed_frozen_candidates.py` | Embeds frozen candidate plus query inputs; exposes explicit fake/real modes and reports structured failures. |
| `scripts/validate_m2_t01_evidence.py` | Validates committed provenance, hashes, fake/real separation, and all completed real-run gates. |
| `tests/contract/test_embedding_contracts.py` | Contract, snapshot, provenance, and immutable-descriptor coverage. |
| `tests/unit/test_embedding_text.py` | Text-protocol and input-identity tests. |
| `tests/unit/test_embedding_cache.py` | Cache identity, corruption, atomic-write, and namespace-isolation tests. |
| `tests/unit/test_embedding_provider.py` | Fake/BGE provider, descriptor, count, dimension, finite-value, and error-surface tests. |
| `tests/integration/test_embedding_pipeline.py` | Batch/cache/mixed-hit order invariants and fake end-to-end snapshot tests. |
| `tests/contract/test_m2_t01_evidence_validation.py` | Evidence validator tests for incomplete main and completed-evidence paths. |
| `evaluation/snapshots/m2/m1-candidates.v1.json` | Conditional, source-reconstructible frozen 33-candidate snapshot. |
| `evaluation/snapshots/m2/m1-candidates.v1.manifest.json` | Conditional snapshot hash, coverage, order, and M1 provenance. |
| `evaluation/snapshots/m2/bge-m3-dense-v1.json` | Conditional real-only query and candidate vectors; never fake-labelled BGE data. |
| `evaluation/snapshots/m2/bge-m3-dense-v1.manifest.json` | Conditional real-run/replay metadata and vector-snapshot hash. |
| `evaluation/reports/m2-t01-red-tests.json` | Honest initial red-test command, exit code, failure, UTC time, and test-file hashes. |
| `evaluation/reports/m2-t01-embedding.json` | Conditional final evidence pointing to commits A/B and the real/vector gates. |
| `.github/workflows/docs-validation.yml` | Adds evidence validation only after the validator safely accepts the pre-completion state. |
| `pyproject.toml` | Adds an `embedding` optional dependency group only; base dependencies remain lightweight. |

### Task 1: Define the closed M2 embedding contracts and record honest red tests

**Files:**
- Create: `app/models/embedding.py`
- Create: `tests/contract/test_embedding_contracts.py`
- Create: `tests/unit/test_embedding_text.py`
- Create: `tests/unit/test_embedding_cache.py`
- Create: `tests/unit/test_embedding_provider.py`
- Create: `tests/integration/test_embedding_pipeline.py`
- Create: `tests/contract/test_m2_t01_evidence_validation.py`
- Create: `evaluation/reports/m2-t01-red-tests.json`

**Interfaces:**
- Produces `EmbeddingTaskError(code: str)`, `EmbeddingModelDescriptor`, `FrozenCandidate`, `FrozenCandidateSnapshot`, `EmbeddingInput`, `EmbeddingVectorRecord`, `EmbeddingCacheEntry`, and `EmbeddingRunStats`.
- The error-code literal set is exactly `FROZEN_SNAPSHOT_MISSING`, `FROZEN_SNAPSHOT_HASH_MISMATCH`, `INVALID_EMBEDDING_INPUT`, `DUPLICATE_EMBEDDING_ID`, `MODEL_REVISION_UNPINNED`, `EMBEDDING_PROVIDER_UNAVAILABLE`, `EMBEDDING_PROVIDER_FAILED`, `EMBEDDING_COUNT_MISMATCH`, `EMBEDDING_DIMENSION_MISMATCH`, `EMBEDDING_NON_FINITE`, `EMBEDDING_CACHE_CORRUPT`, and `EMBEDDING_OUTPUT_WRITE_FAILED`.

- [ ] **Step 1: Write the six named test modules before production embedding modules exist.**

  Each fixture must use two source-backed records with realistic HTTP(S) URLs, one title-only record (`abstract=None`), a query, and this shared descriptor:

  ```python
  EmbeddingModelDescriptor(
      provider_name="deterministic_fake",
      model_id="sha256-vector",
      model_revision="fake-v1",
      provider_library="stdlib",
      provider_library_version="3.12",
      embedding_mode="dense",
      input_format_version="m2-title-abstract-v1",
      normalized=True,
      dimension=16,
      cache_namespace="embedding:fake",
  )
  ```

  Contract tests must assert `extra="forbid"`, reject placeholder abstracts, duplicate paper/source identities, non-HTTP(S) URLs, an unpinned real revision, and fake/real namespace reuse. Unit and integration tests must assert all required text, cache, fake, failure, batch size `1/2/3/N`, and mixed hit/miss invariants specified in this plan's Global Constraints.

- [ ] **Step 2: Run the focused tests and record the genuine failure.**

  Run:

  ```powershell
  py -3.12 -m pytest tests/contract/test_embedding_contracts.py tests/unit/test_embedding_text.py tests/unit/test_embedding_cache.py tests/unit/test_embedding_provider.py tests/integration/test_embedding_pipeline.py tests/contract/test_m2_t01_evidence_validation.py -q
  ```

  Expected: non-zero exit because `app.models.embedding`, provider/core modules, snapshot scripts, and the M2 evidence validator do not yet exist. Write that exact command, UTC time, non-zero exit code, concise first failure, and SHA-256 values of the six test files into `m2-t01-red-tests.json`; do not manufacture a red result after implementation exists.

- [ ] **Step 3: Implement the contracts with strict validation.**

  `EmbeddingModelDescriptor` must be frozen/closed and validate the rules above. `FrozenCandidate` must include `paper_id`, `source`, `source_id`, `title`, nullable `abstract`, `authors`, `year`, `doi`, `url`, `language`, `categories`, `retrieval_paths`, `cluster_id`, and `member_source_identities`. Use an empty list only for genuinely unavailable `categories`; do not fabricate category text. `EmbeddingInput` must carry `input_id`, `input_kind`, exactly one of `paper_id`/`query_id`, version, text, `text_sha256`, and source snapshot hash. Vector and cache models must require positive, matching dimensions and finite floats.

- [ ] **Step 4: Re-run the focused contract tests.**

  Run the command from Step 2. Expected: remaining imports fail only for Tasks 2-7; every test that imports only the new model contracts passes.

### Task 2: Implement the M1 real-source freeze gate and candidate snapshot validator

**Files:**
- Create: `scripts/freeze_m2_candidates.py`
- Modify: `app/models/first_round.py` only if the real cache audit proves that adding `CandidateOutput.abstract: str | None` is necessary to copy the canonical `PaperRecord.abstract`
- Modify: `app/core/first_round.py` only for that additive, non-transforming projection and provenance audit
- Modify: `tests/contract/test_m1_t04_contracts.py` and `tests/integration/test_first_round_pipeline.py` only if the additive M1 field is required
- Modify: `tests/contract/test_embedding_contracts.py`
- Conditional create: `evaluation/snapshots/m2/m1-candidates.v1.json`
- Conditional create: `evaluation/snapshots/m2/m1-candidates.v1.manifest.json`

**Interfaces:**
- Consumes the expected M1 real paths `evaluation/runs/m1-live-r1-final/first-round.json` and `evaluation/runs/m1-t04r1-cache-final/` only when their hashes and real-mode provenance validate.
- Produces a JSON result with either `status="blocked", error_code="M2_T01_FROZEN_INPUT_MISSING"` or validated snapshot/manifest paths and SHA-256 values.

- [ ] **Step 1: Test all freeze-gate branches before writing the script.**

  Add fixtures for: both real artifacts absent; a recorded fixture substituted for real data; wrong output SHA; wrong candidate-array SHA; fewer/more than 33 candidates; duplicate paper ID; duplicate source identity; empty source ID; non-HTTP URL; placeholder abstract; and a valid cache-backed projection. Assert only the valid case writes the snapshot.

- [ ] **Step 2: Implement a zero-network source audit.**

  The script must hash the exact `first-round.json`, load it through the existing closed M1 run contract, verify `mode="real"`, exactly 33 candidates, 100% source-ID/URL coverage, the expected output and candidate-array hashes, and no user-visible field mismatch. It must then replay the existing M1 query plan/deduplication only through `first-round:real` cached responses, require `transport_requests == 0`, and align canonical `PaperRecord` values with candidates by stable paper/source identity.

- [ ] **Step 3: Implement the snapshot projection and manifest.**

  Serialize candidates in a documented stable `paper_id` ascending order using UTF-8 JSON, `ensure_ascii=False`, sorted keys, and a trailing newline. The snapshot must carry `snapshot_version`, `source_phase="M1"`, `source_merge_commit="0eb45fc22d10adb72cb66aa45494333057080bc1"`, source evidence report/hash values, source candidate-array SHA, source output SHA, question, count, and candidates. The manifest must record snapshot SHA, count 33, coverage 1.0/1.0, order strategy, source identity set/hash, abstract provenance counts, and the zero-transport replay audit.

- [ ] **Step 4: Keep M1 corrections minimal and conditional.**

  If M1 JSON lacks abstracts but the audited real cache has them, add only `abstract: str | None = None` to `CandidateOutput` and directly copy `cluster.canonical_record.abstract`. Do not alter any old evidence file or field values. `categories` must remain a source-absence representation (`[]`) unless the canonical source record genuinely provides categories; do not infer them from title, query, or URL.

- [ ] **Step 5: Execute the actual freeze command and preserve its truthful state.**

  Run:

  ```powershell
  py -3.12 scripts/freeze_m2_candidates.py --m1-output evaluation/runs/m1-live-r1-final/first-round.json --m1-cache-dir evaluation/runs/m1-t04r1-cache-final --output-dir evaluation/snapshots/m2
  ```

  Expected in the current workspace: stable structured `M2_T01_FROZEN_INPUT_MISSING`, no snapshot files, no network request, and no completion claim. If the original artifacts are later restored and pass every check, rerun to create the two conditional snapshot files and then execute the full M1 tests plus `py -3.12 scripts/validate_m1_evidence.py`.

### Task 3: Build deterministic input texts and the provider boundary

**Files:**
- Create: `app/adapters/embedding.py`
- Create: `app/core/embedding.py`
- Modify: `app/models/embedding.py`
- Modify: `tests/unit/test_embedding_text.py`
- Modify: `tests/unit/test_embedding_provider.py`

**Interfaces:**
- Produces `EmbeddingProvider` with `descriptor` and `embed(texts: Sequence[str], *, batch_size: int) -> list[list[float]]`.
- Produces `build_embedding_text(record: FrozenCandidate, source_snapshot_sha256: str) -> EmbeddingInput` and `build_query_embedding_input(question: str, source_snapshot_sha256: str) -> EmbeddingInput`.

- [ ] **Step 1: Implement the exact text protocol.**

  Implement title-plus-abstract text as follows:

  ```python
  normalized_title = record.title.replace("\r\n", "\n").strip()
  normalized_abstract = (record.abstract or "").replace("\r\n", "\n").strip()
  text = f"title:\n{normalized_title}"
  if normalized_abstract:
      text += f"\n\nabstract:\n{normalized_abstract}"
  ```

  Reject a blank normalized title and any abstract equal, after trimming/casefolding only for rejection, to `no abstract`, `not provided`, or `no abstract available`. Hash UTF-8 text bytes with SHA-256; do not alter the stored text itself.

- [ ] **Step 2: Implement the deterministic fake provider.**

  Derive sixteen finite floats from consecutive SHA-256 blocks over UTF-8 `text + canonical_descriptor_manifest`; use no Python `hash`, randomness, network, or external dependency. Its descriptor must exactly identify `deterministic_fake`, `sha256-vector`, `fake-v1`, `dense`, `m2-title-abstract-v1`, dimension 16, and `embedding:fake`; fake output/evidence is always labelled `deterministic_fake`.

- [ ] **Step 3: Implement the lazy real provider without silently substituting fake.**

  `BgeM3DenseProvider` must validate a non-floating revision before construction, import its optional package only within `embed`, set eval/inference/no-gradient behavior, execute dense-only encoding, and convert output to finite Python floats. Missing package/device/model load failures map to `EMBEDDING_PROVIDER_UNAVAILABLE`; execution failures map to `EMBEDDING_PROVIDER_FAILED`. No core-module import may load a model.

- [ ] **Step 4: Run text/provider unit tests.**

  Run:

  ```powershell
  py -3.12 -m pytest tests/unit/test_embedding_text.py tests/unit/test_embedding_provider.py -q
  ```

  Expected: all fake and error-surface tests pass without installing optional embedding dependencies or using a network.

### Task 4: Add per-input cache, vector validation, and batch orchestration

**Files:**
- Modify: `app/core/embedding.py`
- Modify: `tests/unit/test_embedding_cache.py`
- Modify: `tests/integration/test_embedding_pipeline.py`

**Interfaces:**
- Produces `embed_inputs(inputs, provider, cache_dir, batch_size) -> tuple[list[EmbeddingVectorRecord], EmbeddingRunStats]`.
- Cache key is SHA-256 of canonical JSON containing descriptor identity fields and `text_sha256`, excluding input ID and batch size.

- [ ] **Step 1: Implement cache read/write validation.**

  Reject manifest/text-hash/dimension/vector mismatches and JSON decoding failures as `EMBEDDING_CACHE_CORRUPT`, increment `cache_corrupt_count`, and treat them as misses only after recording the rejection. Write `*.tmp` in the destination directory, flush, then replace the exact key path atomically; serialize UTF-8 JSON with sorted keys.

- [ ] **Step 2: Implement stable misses and restored order.**

  Reject duplicate input IDs before cache lookup. Visit inputs in caller order; call the provider only for cache misses in stable chunks; immediately reject count mismatch, inconsistent/non-positive dimension, NaN, or Infinity. Join hits and new records by input ID and return the original input sequence exactly.

- [ ] **Step 3: Prove all cache/batch invariants.**

  Run:

  ```powershell
  py -3.12 -m pytest tests/unit/test_embedding_cache.py tests/integration/test_embedding_pipeline.py -q
  ```

  Expected: batch sizes `1`, `2`, `3`, and larger than N return identical input order/vector values; partial hits call only for misses; replay calls the provider zero times; changed text/revision/format/normalization/namespace misses; corrupt JSON cannot become a hit.

### Task 5: Provide the embedding CLI and real-run freeze/replay gates

**Files:**
- Create: `scripts/embed_frozen_candidates.py`
- Modify: `pyproject.toml`
- Modify: `tests/integration/test_embedding_pipeline.py`
- Conditional create: `evaluation/snapshots/m2/bge-m3-dense-v1.json`
- Conditional create: `evaluation/snapshots/m2/bge-m3-dense-v1.manifest.json`

**Interfaces:**
- Consumes only a validated candidate snapshot and its manifest; creates one query input from the original M1 question plus one paper input per frozen candidate.
- Produces structured stdout/JSON failures, vector snapshots, and manifests with live/replay cache statistics.

- [ ] **Step 1: Add optional-only model dependencies.**

  Add an `embedding` optional dependency group for the selected documented BGE-M3 inference library and torch, including version constraints determined from the official model card/library documentation during execution. Do not add either package to base `dependencies`; record package versions, model ID, immutable revision, device/hardware, dtype, normalization, and batch size in the real manifest.

- [ ] **Step 2: Implement CLI mode gates.**

  Require `--provider fake|bge-m3`, `--snapshot`, `--manifest`, `--cache-dir`, and `--batch-size`. `fake` is accepted for tests only and cannot write either `bge-m3-dense-v1.*` artifact. `bge-m3` requires a validated snapshot, explicitly supplied immutable revision, a cache namespace distinct from `embedding:fake`, and an installed optional provider; otherwise emit the stable structured code without traceback.

- [ ] **Step 3: Test the live/replay acceptance logic using only fake test doubles.**

  Assert a fresh namespace reports `hits=0`, `misses=candidate_count+1`, and calls greater than zero; a second identical run reports zero calls/misses, full hits, and exactly equal vector arrays. Assert a changed descriptor/input format/revision cannot replay an old entry.

- [ ] **Step 4: Run the real model only after the source snapshot exists.**

  First read the official BGE-M3 model card and inference-library documentation, then pin the exact resolved revision. Run once with a newly empty real namespace and once with the identical namespace/config. Persist real vectors only if candidate count is 33, query count is 1, all dimensions match, every value is finite, live hits are zero, replay calls/misses are zero, and arrays compare equal. If the source freeze remains blocked, mark every real-model item `not_run` and create no BGE-labelled snapshot.

### Task 6: Add evidence schema/validator and CI integration

**Files:**
- Create: `scripts/validate_m2_t01_evidence.py`
- Modify: `.github/workflows/docs-validation.yml`
- Modify: `tests/contract/test_m2_t01_evidence_validation.py`
- Conditional create: `evaluation/reports/m2-t01-embedding.json`
- Conditional modify: `STATUS.md`

**Interfaces:**
- Validator exits zero on pre-completion main when no M2-T01 report/status claim exists, and strictly validates a report when present.

- [ ] **Step 1: Implement strict completed-evidence validation.**

  Validate report version/task ID/baseline, A/B/head ancestry, Git-blob SHA-256s, candidate and vector snapshot hashes, 33 candidates, 1.0 coverage, exact fake/real classifications, distinct namespaces, immutable BGE revision, positive uniform dimension, zero non-finite values, zero live hits, zero replay provider calls/misses, equal replay arrays, M1 metadata mismatch zero, and M2/M3 status consistency.

- [ ] **Step 2: Add safe CI coverage.**

  Add a `Validate completed M2-T01 evidence` step after M1 validation. It must only inspect checked-in files and run no model download/network command. Its test must demonstrate that baseline `main` remains valid before the task is complete.

- [ ] **Step 3: Gate the final report and status.**

  Create `m2-t01-embedding.json` and change `STATUS.md` to `M2 IN_PROGRESS 1/5`, `M2-T01 COMPLETE`, next `M2-T02`, and `M3 BLOCKED_BY_M2` only after every Task 2-5 real condition passes. Otherwise retain `M2 IN_PROGRESS 0/5`, identify `M2-T01 real model gate` as next, retain `M3 BLOCKED_BY_M2`, and do not create a misleading final-completion report.

### Task 7: Run acceptance checks, preserve evidence classes, and commit within the required boundary

**Files:**
- Modify only files created/changed by Tasks 1-6; never add runtime caches, model weights, tokens, or recorded data disguised as real evidence.

**Interfaces:**
- Produces implementation commit A, conditional data commit B, and conditional evidence commit C with full SHA values recorded in the final report/PR.

- [ ] **Step 1: Run all required offline checks.**

  Run:

  ```powershell
  py -3.12 -m pytest tests/contract/test_embedding_contracts.py tests/unit/test_embedding_text.py tests/unit/test_embedding_cache.py tests/unit/test_embedding_provider.py tests/integration/test_embedding_pipeline.py tests/contract/test_m2_t01_evidence_validation.py -q
  py -3.12 -m pytest -q
  py -3.12 -m ruff check app evaluation scripts tests
  py -3.12 -m mypy app evaluation scripts
  py -3.12 scripts/validate_project_docs.py
  py -3.12 scripts/validate_phase.py M0
  py -3.12 scripts/validate_m1_evidence.py
  py -3.12 scripts/validate_m2_t01_evidence.py
  py -3.12 -m pip check
  ```

  Record each exact command, exit code, test count, UTC time, and `deterministic_fake`/`real`/`not_run` classification in the appropriate evidence file.

- [ ] **Step 2: Commit implementation A only after its offline tests pass.**

  Stage the plan, implementation, tests, red-test report, validator, CI, and optional dependency definition; exclude `STATUS.md`, final evidence, BGE snapshot, cache directories, and model files. Commit:

  ```powershell
  git add docs/superpowers/plans/2026-07-28-research-retrieval-calibrator-m2-t01.md app/models/embedding.py app/adapters/embedding.py app/core/embedding.py scripts/freeze_m2_candidates.py scripts/embed_frozen_candidates.py scripts/validate_m2_t01_evidence.py tests evaluation/reports/m2-t01-red-tests.json pyproject.toml .github/workflows/docs-validation.yml
  git commit -m "feat: implement M2 embedding provider and vector cache"
  ```

- [ ] **Step 3: Create data B only after the original M1 source and real BGE gates pass.**

  Commit only the four frozen candidate/vector snapshot files with:

  ```powershell
  git add evaluation/snapshots/m2/m1-candidates.v1.json evaluation/snapshots/m2/m1-candidates.v1.manifest.json evaluation/snapshots/m2/bge-m3-dense-v1.json evaluation/snapshots/m2/bge-m3-dense-v1.manifest.json
  git commit -m "chore: freeze M2 candidate and embedding snapshots"
  ```

  If either gate is blocked, do not create B and do not label fake vectors as BGE-M3.

- [ ] **Step 4: Create evidence C only when A/B and every real acceptance condition pass.**

  Commit exactly `evaluation/reports/m2-t01-embedding.json` and `STATUS.md`:

  ```powershell
  git add evaluation/reports/m2-t01-embedding.json STATUS.md
  git commit -m "chore: record M2-T01 embedding evidence"
  ```

- [ ] **Step 5: Push and open a Draft PR only at the truthful task boundary.**

  Push `agent/m2-t01-embedding-provider` and open the requested Draft PR title, `feat: implement M2 embedding provider and frozen vectors`. The description must report baseline, A/B/C/full HEAD SHAs, snapshot/hash/count/coverage/abstract counts, input/query hashes, fake and real descriptors, pinned library/torch/device/hardware/dtype/batch/dimension, live/replay cache stats, equality, every required check/CI result, M2/M3 state, not-run items, and the rollback order.

## Self-Review

1. **Spec coverage:** Tasks 1-2 cover the frozen M1 source, real-cache replay, closed snapshot schema, provenance, hashes, and blocked outcome; Tasks 3-4 cover protocol/provider/fake/batching/cache; Task 5 covers lazy BGE and real freeze/replay; Task 6 covers report/validator/CI/status; Task 7 covers all required verification, commits, PR, and rollback. Nothing implements M2-T02 through M3.
2. **Placeholder scan:** This plan names every required path, command, stable error code, conditional gate, and commit boundary. The only conditional artifacts are explicitly forbidden until their real provenance preconditions are satisfied.
3. **Type consistency:** `FrozenCandidate` feeds `build_embedding_text`; `EmbeddingInput` feeds `EmbeddingProvider.embed` through `embed_inputs`; `EmbeddingVectorRecord` and `EmbeddingRunStats` feed the CLI, snapshot manifests, and evidence validator. Fake/real cache namespaces remain descriptor-owned throughout.
