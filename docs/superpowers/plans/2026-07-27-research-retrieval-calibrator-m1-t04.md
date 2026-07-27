# Research Retrieval Calibrator M1-T04 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide a bounded, replayable, auditable first-round CLI that turns a research question into a deterministic arXiv query plan, source-backed normalized/deduplicated candidates, and JSON/Markdown outputs.

**Architecture:** Add an M1-only orchestration layer between the existing deterministic intent/query-plan modules and the existing bounded arXiv adapter. The orchestrator owns input limits, per-query outcome capture, global candidate and attempt budgets, conservative deduplication, deterministic rendering, and structured failures; the thin script only parses arguments, builds dependencies, writes outputs, and sets an exit code. Recorded transports are injected for all automated end-to-end tests, while the only real-network path is an explicitly invoked, bounded smoke command.

**Tech Stack:** Python 3.12; Pydantic 2 contracts; stdlib argparse/json/pathlib/hashlib; existing `ArxivAdapter`, intent/query planner, and deterministic paper deduplication; pytest, Ruff, mypy.

## Global Constraints

- Start from merged main `f0f167766589e3321821b0caf7793b00c8ff7291`; preserve M1 `IN_PROGRESS 3/4` and M2 `BLOCKED_BY_M1` unless every M1-T04 acceptance gate succeeds.
- Scope is M1-T04 only. Do not implement ranking/selection, embeddings, reranking, second-round feedback, CNKI acquisition, web/API serving, deployment, or LLM-generated paper metadata.
- Invoke the existing Query IR and `build_query_plan`; do not hand-write arXiv query strings in the CLI.
- Use only source-provided `PaperRecord` fields. Every visible candidate must have nonblank `source_id`, an HTTP(S) URL, preserved retrieval paths, cluster identity, source identities, and actual merge reasons.
- Defaults are `max_results_per_query=5` and `max_total_candidates=60`; exactly 12 queries therefore cannot collect more than 60 raw records. Validate every input bound before creating a transport.
- Bound arXiv total attempts, per-query results, raw candidates, timeout, and run observation time. When the raw-candidate budget is met, stop issuing further requests and record `candidate_budget_reached=true`.
- All failure paths emit a JSON result with a stable error code and never a traceback. Recorded data is `recorded_external`, never `real_external`.
- Keep cache identity sensitive to query text, max results, and adapter schema/config version. A replay must make zero transport calls and preserve candidates exactly.
- The implementation commit G includes production code, tests, fixtures, and this plan only; it must not modify `STATUS.md` or `evaluation/reports/m1-validation.json`. Evidence commit H may modify only those two files.

---

## File Structure

- Create: `app/models/first_round.py` — immutable Pydantic contracts for configuration, query outcomes, candidate output, metrics, failures, and the complete run result.
- Create: `app/core/first_round.py` — deterministic question-to-intent adapter, orchestration, error mapping, candidate budgeting, source coverage calculation, and conversion of `DedupCluster` to visible output.
- Create: `app/cli/__init__.py` — marks the CLI package.
- Create: `app/cli/first_round.py` — argument-independent CLI boundary: parse/validate inputs, select recorded or real transport, invoke the orchestrator, serialize JSON/Markdown, and return a stable exit status.
- Create: `scripts/first_round_retrieval.py` — repository-root bootstrap and `app.cli.first_round.main()` entrypoint.
- Create: `tests/fixtures/arxiv/first_round_graph.xml` and `tests/fixtures/arxiv/first_round_empty.xml` — recorded Atom responses, including repeated source identities across query paths.
- Create: `tests/unit/test_first_round_cli.py` — argument validation, render/write failure behavior, and traceback-free structured output tests.
- Create: `tests/integration/test_first_round_pipeline.py` — recorded full vertical slice, cache replay, budgets, and all/partial source-failure scenarios using a fake transport and injected clock.
- Create: `tests/contract/test_m1_t04_contracts.py` — contract rejection and JSON/Markdown field-provenance assertions.
- Create: `evaluation/reports/m1-t04-red-tests.json` — immutable record of the initially failing focused test commands and their exit status; include this in G because the task requires preserving the red result.
- Create later: `evaluation/reports/m1-validation.json` — evidence-only H, produced only after G and all non-network gates are green.
- Modify: `app/models/__init__.py` — export only the public M1-T04 contracts.

## Task 1: Define first-round contracts and deterministic request limits

**Files:**
- Create: `app/models/first_round.py`
- Modify: `app/models/__init__.py`
- Test: `tests/contract/test_m1_t04_contracts.py`

**Interfaces:**
- Consumes: existing `app.models.paper.PaperRecord`, `app.models.planning.QueryPlan`, `app.models.query.Query`, and `app.models.dedup.SourceIdentity`.
- Produces: `FirstRoundConfig`, `FirstRoundStatus`, `QueryExecutionResult`, `CandidateOutput`, `FailureReport`, `RunMetrics`, and `FirstRoundRun` consumed by `app.core.first_round` and `app.cli.first_round`.

- [ ] **Step 1: Write the failing contract tests**

```python
def test_config_rejects_unbounded_or_inconsistent_limits() -> None:
    with pytest.raises(ValidationError):
        FirstRoundConfig(max_results_per_query=0)
    with pytest.raises(ValidationError):
        FirstRoundConfig(max_total_candidates=61)
    with pytest.raises(ValidationError):
        FirstRoundConfig(timeout_seconds=121)


def test_visible_candidate_requires_source_identity_and_http_url() -> None:
    with pytest.raises(ValidationError):
        CandidateOutput.model_validate({"paper_id": "arxiv:1", "source": "arxiv"})


def test_failure_and_run_contracts_forbid_unknown_fields() -> None:
    failure = FailureReport(error_code="ARXIV_UNAVAILABLE", scope="run")
    with pytest.raises(ValidationError):
        FailureReport.model_validate({**failure.model_dump(), "unexpected": True})
```

- [ ] **Step 2: Run the focused contract test and preserve the red result**

Run: `py -3.12 -m pytest tests/contract/test_m1_t04_contracts.py -q`

Expected: collection failure because `app.models.first_round` does not yet exist. Capture the command, UTC time, exit code, and a SHA-256 of the test file in `evaluation/reports/m1-t04-red-tests.json`; do not manufacture a passing result.

- [ ] **Step 3: Implement the contracts with closed schemas**

```python
class FirstRoundStatus(StrEnum):
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"


class FirstRoundConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    max_results_per_query: int = Field(default=5, ge=1, le=5)
    max_total_candidates: int = Field(default=60, ge=1, le=60)
    max_total_attempts: int = Field(default=20, ge=1, le=100)
    timeout_seconds: float = Field(default=20.0, gt=0, le=120)
    cache_dir: Path
    mode: Literal["recorded", "real"]
    adapter_schema_version: Literal["m1-t04.v1"] = "m1-t04.v1"


class CandidateOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    paper_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    authors: list[str]
    year: int | None
    doi: str | None
    url: str = Field(min_length=1)
    retrieval_paths: list[str] = Field(min_length=1)
    cluster_id: str = Field(min_length=1)
    member_source_identities: list[SourceIdentity] = Field(min_length=1)
    merge_reasons: list[DedupDecision]

    @model_validator(mode="after")
    def require_visible_source_identity(self) -> Self:
        if not self.url.startswith(("http://", "https://")):
            raise ValueError("Visible candidates require an HTTP(S) URL")
        return self
```

Define `QueryExecutionResult` with `query_id`, `status` (`success` or `failed`), `candidate_count`, `attempt_count`, `http_status`, `cache_hit`, and optional stable `error_code`. Define `FailureReport` with `error_code`, `scope` (`run` or `query`), optional `query_id`, and non-secret reason text. Define `RunMetrics` with all raw/deduplicated/manual-review counts, source/URL coverage rates, metadata hallucination rate, candidate-budget state, transport requests, cache hits, and elapsed seconds. Define `FirstRoundRun` with every JSON field required by the user request, plus `error_code` for all-run failures; reject status/count contradictions in an `after` validator.

- [ ] **Step 4: Export public contracts and make the tests pass**

Add the new names to `app/models/__init__.py`, then run:

Run: `py -3.12 -m pytest tests/contract/test_m1_t04_contracts.py -q`

Expected: PASS; schemas reject extra fields, invalid bounds, blank source IDs, and non-HTTP(S) visible URLs.

## Task 2: Build the bounded first-round orchestration core

**Files:**
- Create: `app/core/first_round.py`
- Test: `tests/integration/test_first_round_pipeline.py`
- Test: `tests/contract/test_m1_t04_contracts.py`

**Interfaces:**
- Consumes: `FirstRoundConfig`, existing `freeze_research_intent`, `build_query_plan`, `ArxivAdapter.search`, and `deduplicate_papers`.
- Produces: `run_first_round(question: str, *, config: FirstRoundConfig, adapter: ArxivAdapter, now: Callable[[], datetime], monotonic: Callable[[], float]) -> FirstRoundRun`.

- [ ] **Step 1: Write failing recorded-pipeline tests**

```python
def test_recorded_question_runs_ir_plan_arxiv_dedup_and_outputs_source_backed_candidates() -> None:
    run = run_first_round(QUESTION, config=_config(), adapter=_recorded_adapter())

    assert run.status is FirstRoundStatus.SUCCESS
    assert len(run.query_plan.queries) == 12
    assert run.raw_candidate_count == 3
    assert run.deduplicated_candidate_count == 2
    assert run.metrics.source_id_coverage == 1.0
    assert run.metrics.url_coverage == 1.0
    assert run.metrics.metadata_hallucination_rate == 0.0
    assert run.candidates[0].retrieval_paths == sorted(run.candidates[0].retrieval_paths)


def test_all_queries_unavailable_returns_honest_failure_without_candidates() -> None:
    run = run_first_round(QUESTION, config=_config(), adapter=_failing_adapter())

    assert run.status is FirstRoundStatus.FAILED
    assert run.error_code == "ARXIV_UNAVAILABLE"
    assert run.candidates == []


def test_one_query_failure_with_records_is_partial_success() -> None:
    run = run_first_round(QUESTION, config=_config(), adapter=_partially_failing_adapter())

    assert run.status is FirstRoundStatus.PARTIAL_SUCCESS
    assert {item.query_id for item in run.query_results if item.status == "failed"}
    assert run.candidates
```

Add tests for blank question (`INTENT_PARSE_FAILED`), a `max_total_candidates=2` budget (no further transport after two raw records and `candidate_budget_reached=True`), deduplication errors (`DEDUPLICATION_FAILED`), empty recorded responses (`INSUFFICIENT_CANDIDATES`), preserved paths, deterministic JSON when clocks are fixed, and no adapter calls for invalid configuration.

- [ ] **Step 2: Run the integration tests and preserve their red result**

Run: `py -3.12 -m pytest tests/integration/test_first_round_pipeline.py -q`

Expected: collection failure because `app.core.first_round` does not yet exist. Append the exact command and nonzero exit to the same red-test evidence file without replacing the earlier record.

- [ ] **Step 3: Implement the deterministic IR bridge and orchestration**

Implement a deliberately narrow, deterministic parser for the supported question form `How can <method> support <task>?`; it must create an `IntentDraft` from captured source text, use fixed accepted role `{\"method\"}`, set `MethodConstraint.PREFERRED`, freeze via `freeze_research_intent`, then pass that intent to `build_query_plan`. Any nonmatching, blank, or incomplete question returns `FailureReport(error_code=\"INTENT_PARSE_FAILED\", scope=\"run\")`; do not guess terms or call an LLM.

```python
def _candidate_from_cluster(cluster: DedupCluster) -> CandidateOutput:
    record = cluster.canonical_record.model_copy(update={"user_visible": True})
    return CandidateOutput(
        paper_id=record.paper_id,
        source=record.source,
        source_id=record.source_id,
        title=record.title,
        authors=record.authors,
        year=record.year,
        doi=record.doi,
        url=record.url,
        retrieval_paths=cluster.retrieval_paths,
        cluster_id=cluster.cluster_id,
        member_source_identities=cluster.source_identities,
        merge_reasons=cluster.merge_reasons,
    )


def _map_arxiv_error(error: ArxivAdapterError) -> str:
    return "ARXIV_PARTIAL_FAILURE" if error.code.startswith("ARXIV_") else "ARXIV_UNAVAILABLE"
```

Loop over the 12 generated `Query` values in their existing stable order. Before each call, calculate `remaining = config.max_total_candidates - len(raw_records)` and stop when it is zero. Call `adapter.search(query, max_results=min(config.max_results_per_query, remaining))`; record `adapter.last_observation` immediately for both success and failure. Treat an `ArxivAdapterError` as an isolated query failure and continue only while budget remains. If every attempted query fails, return the no-candidate `ARXIV_UNAVAILABLE` run. Otherwise pass raw records exactly once to `deduplicate_papers`, derive candidates from its stable clusters, and report insufficient candidates honestly rather than filling data.

Build `run_id` from a SHA-256 of the canonical request manifest (question, config model dump, and query-plan ID), not the wall clock. Use injected UTC/monotonic clocks for timestamps and elapsed time in tests. Compute source/URL coverage from the emitted candidates only; compute metadata hallucination rate as `0.0` because each displayed value is copied from a `PaperRecord`/dedup artifact, and assert this invariant rather than inferring metadata.

- [ ] **Step 4: Verify focused core behavior**

Run: `py -3.12 -m pytest tests/integration/test_first_round_pipeline.py tests/contract/test_m1_t04_contracts.py -q`

Expected: PASS. The recorded fixture provides the only paper data; all-unavailable and insufficient outcomes carry structured JSON-ready stable error codes.

## Task 3: Add CLI, recorded transport, deterministic renderers, and replay tests

**Files:**
- Create: `app/cli/__init__.py`
- Create: `app/cli/first_round.py`
- Create: `scripts/first_round_retrieval.py`
- Create: `tests/fixtures/arxiv/first_round_graph.xml`
- Create: `tests/fixtures/arxiv/first_round_empty.xml`
- Create: `tests/unit/test_first_round_cli.py`
- Modify: `tests/integration/test_first_round_pipeline.py`

**Interfaces:**
- Consumes: `run_first_round`, `FirstRoundRun`, `ArxivAdapterConfig`, and the `ArxivTransport` protocol.
- Produces: `main(argv: Sequence[str] | None = None) -> int`, `render_json(run: FirstRoundRun) -> str`, `render_markdown(run: FirstRoundRun) -> str`, and exactly two files named `first-round.json` and `first-round.md` in the requested output directory.

- [ ] **Step 1: Write failing CLI tests**

```python
@pytest.mark.parametrize("argv", [
    [],
    ["--question", "   ", "--output-dir", "out"],
    ["--question", QUESTION, "--output-dir", "out", "--max-results-per-query", "6"],
])
def test_invalid_cli_arguments_emit_only_structured_json(argv, capsys) -> None:
    assert main(argv) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "failed"
    assert payload["error_code"] == "INVALID_CLI_ARGUMENT"
    assert "Traceback" not in capsys.readouterr().err


def test_recorded_cli_writes_json_and_markdown_without_invented_metadata(tmp_path) -> None:
    assert main(["--question", QUESTION, "--output-dir", str(tmp_path), "--mode", "recorded"]) == 0
    payload = json.loads((tmp_path / "first-round.json").read_text(encoding="utf-8"))
    markdown = (tmp_path / "first-round.md").read_text(encoding="utf-8")
    assert payload["metrics"]["source_id_coverage"] == 1.0
    assert "Abstract:" not in markdown
    assert "DOI:" not in markdown or all(item["doi"] for item in payload["candidates"])
```

Also test `OUTPUT_WRITE_FAILED` by injecting a writer that raises `OSError`, and test a cache replay with the same temp cache directory: first run has `transport_requests > 0`, second run has `transport_requests == 0`, `cache_hit=True`, and byte-identical candidate lists.

- [ ] **Step 2: Run the CLI test before implementation**

Run: `py -3.12 -m pytest tests/unit/test_first_round_cli.py -q`

Expected: collection failure because the CLI module does not exist. Add the command and exit code to `evaluation/reports/m1-t04-red-tests.json`.

- [ ] **Step 3: Implement a thin, safe CLI boundary**

```python
parser.add_argument("--question", required=True)
parser.add_argument("--output-dir", required=True, type=Path)
parser.add_argument("--user-agent", default="research-retrieval-calibrator/0.1 (M1-T04)")
parser.add_argument("--max-results-per-query", type=int, default=5)
parser.add_argument("--max-total-candidates", type=int, default=60)
parser.add_argument("--timeout-seconds", type=float, default=20.0)
parser.add_argument("--cache-dir", type=Path)
parser.add_argument("--mode", choices=("recorded", "real"), default="recorded")
```

Reject a blank question/user-agent, missing output directory parent, values outside the contract, or an unrecognized mode with `INVALID_CLI_ARGUMENT` before constructing a network transport. In `recorded` mode, construct an `ArxivTransport` backed only by the two checked-in fixture files and record its response count. In `real` mode, use `UrllibArxivTransport` with nonempty `--user-agent`; never scrape HTML.

Serialize JSON with `run.model_dump(mode=\"json\")`, `ensure_ascii=False`, `sort_keys=True`, and fixed indentation. Render Markdown from candidate fields only: title, authors, year, source, source ID, URL, retrieval paths, merge reasons, and whether manual review is required. Omit abstract, translated titles, guessed DOI/author/year, rankings, relevance scores, and any selection language. On successful serialization write JSON first, then Markdown; if either write fails, print an `OUTPUT_WRITE_FAILED` run JSON and return `1` without a traceback. Return `0` only for `success`, and `1` for `partial_success` or `failed`.

Use a persistent cache representation keyed by `sha256(canonical_json({\"query_text\": ..., \"max_results\": ..., \"adapter_schema_version\": ...}))`. Store and restore only validated `PaperRecord.model_dump(mode=\"json\")` values; atomically replace each cache file through a same-directory temporary path. Do not cache failures or partial pages.

- [ ] **Step 4: Run unit and recorded end-to-end verification**

Run: `py -3.12 -m pytest tests/unit/test_first_round_cli.py tests/integration/test_first_round_pipeline.py tests/contract/test_m1_t04_contracts.py -q`

Expected: PASS. The recorded test demonstrates `question → IR → 12-query plan → recorded Atom → dedup → JSON/Markdown`, 100% source/URL coverage, zero metadata hallucination, structured failures, and zero-request cache replay.

## Task 4: Verify, commit implementation G, then record only truthful external evidence

**Files:**
- Create: `evaluation/reports/m1-validation.json` (only after implementation commit G)
- Modify: `STATUS.md` (only if every completion gate succeeds)

**Interfaces:**
- Consumes: commit G, all automated/recorded results, the bounded real smoke output, and Git SHA-256 inputs.
- Produces: evidence-only commit H and (only after all gates) M1 `COMPLETE 4/4` / M2 `READY` state.

- [ ] **Step 1: Run the required pre-real gates**

Run, in this order:

```powershell
py -3.12 -m pytest tests/unit/test_first_round_cli.py tests/integration/test_first_round_pipeline.py tests/contract/test_m1_t04_contracts.py -q
py -3.12 -m pytest -q
py -3.12 -m ruff check app evaluation scripts tests
py -3.12 -m mypy app evaluation scripts
py -3.12 scripts/validate_project_docs.py
py -3.12 scripts/validate_phase.py M0
py -3.12 -m pip check
```

Expected: every command exits 0. Stop before real networking if any command fails; fix only M1-T04 scope and rerun the failed gate.

- [ ] **Step 2: Commit the implementation as G**

Run:

```powershell
git add app/cli app/core/first_round.py app/models/first_round.py app/models/__init__.py scripts/first_round_retrieval.py tests/fixtures/arxiv/first_round_graph.xml tests/fixtures/arxiv/first_round_empty.xml tests/unit/test_first_round_cli.py tests/integration/test_first_round_pipeline.py tests/contract/test_m1_t04_contracts.py evaluation/reports/m1-t04-red-tests.json docs/superpowers/plans/2026-07-27-research-retrieval-calibrator-m1-t04.md
git commit -m "feat: implement M1 first-round retrieval CLI"
git rev-parse HEAD
```

Expected: record the full returned SHA as G. Confirm `git show --name-only --format=` contains neither `STATUS.md` nor `evaluation/reports/m1-validation.json`.

- [ ] **Step 3: Execute one bounded real-network smoke only after the above is green**

Run:

```powershell
py -3.12 scripts/first_round_retrieval.py --question "How can graph-based retrieval support scientific literature discovery?" --output-dir evaluation/runs/m1-live --mode real --user-agent "research-retrieval-calibrator/0.1 (M1-T04 validation)"
```

Expected: preserve the exact UTC time, 12 query IDs, each observation's attempts and HTTP status, raw/deduplicated counts, source-ID and URL samples, cache state, candidate-budget state, and elapsed seconds. A nonzero exit or source failure is honest `real_external` failure evidence, not a reason to alter fixtures or claim completion.

- [ ] **Step 4: Replay real cache only after a successful real smoke**

Run the identical command and output to an isolated evidence directory while preserving the same cache directory. Expected: first run transport requests > 0; second run transport requests = 0; candidate JSON arrays exactly match. If real smoke failed, record cache replay as `not_run` with the causal reason; do not run a second network request.

- [ ] **Step 5: Write evidence H and update state only when warranted**

Create `evaluation/reports/m1-validation.json` with baseline SHA, G, Python/dependency versions, red/focused/full/static/doc/M0/pip commands and exit codes, fixture and source-file SHA-256s, recorded statistics, real or `not_run` evidence, cache/attempt/budget data, raw/dedup counts, coverage rates, and metadata hallucination rate. If and only if real live success, real cache replay, 100% source/URL coverage, zero hallucination, honest unavailable-source test, and all automated gates pass, change `STATUS.md` to M1 `COMPLETE 4/4`, M2 `READY`, next action `M2-T01`.

Run:

```powershell
git add evaluation/reports/m1-validation.json STATUS.md
git commit -m "chore: record M1 first-round validation evidence"
git rev-parse HEAD
```

Expected: record SHA H. If any completion condition is absent, H must instead preserve M1 `IN_PROGRESS 3/4`, M2 `BLOCKED_BY_M1`, and the explicit failed/not-run causal state.

- [ ] **Step 6: Push and open the required Draft PR**

Push the branch and create a Draft PR titled `feat: implement M1 first-round retrieval CLI`. Its description must name the baseline, G, H, final head, automated/recorded/real/not-run classifications, raw/deduplicated counts, source/URL coverage, hallucination rate, request budgets, focused/full test counts, CI state, M1/M2 state, and rollback order:

```powershell
git revert <H>
git revert <G>
```

Do not merge the PR or begin M2.

## Self-Review

- **Spec coverage:** Task 1 covers bounded schemas and structured failure contracts. Task 2 forces IR/planning, bounded arXiv invocation, source-only candidate construction, deduplication, all/partial failure policy, and deterministic metrics. Task 3 covers CLI arguments, JSON/Markdown provenance, output failures, and cache replay. Task 4 preserves red/green/recorded/real evidence, G/H separation, completion gates, and draft-PR reporting.
- **Placeholder scan:** The plan contains no unresolved implementation markers or implicit validation behavior; all error codes, paths, commands, limits, and external-evidence branches are explicit.
- **Type consistency:** `run_first_round` returns `FirstRoundRun`; its `CandidateOutput` is derived only from `DedupCluster`; `main` serializes that same run. The existing `ArxivAdapter` remains the sole arXiv HTTP boundary.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-27-research-retrieval-calibrator-m1-t04.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?

---

## M1-T04R1: provenance-safe live closure

**Goal:** Close the already-implemented M1-T04 path only after a fresh live arXiv run proves rate-limit behaviour, cache-mode isolation, exact metadata provenance, and a machine-validated evidence report.

**Constraints:** Continue on `agent/m1-t04-first-round-cli` / Draft PR #9; do not create a branch or PR, do not begin M2, use `m1-t04.v2`, default real interval `3.0`, permit recorded interval `0.0`, and keep `STATUS.md` plus `evaluation/reports/m1-validation.json` out of the implementation commit.

### Task R1.1: Red tests for rate limiting and cache identity

**Files:** Modify `tests/unit/test_arxiv_adapter.py`, `tests/unit/test_first_round_cli.py`, `tests/integration/test_first_round_pipeline.py`, and `tests/contract/test_m1_t04_contracts.py`.

- [ ] Add fake-clock/fake-transport tests that prove real mode rejects `0.0` (and cannot receive a no-op sleeper), makes the second request wait at least the configured positive interval, and chooses the larger of Retry-After and the interval.
- [ ] Add tests that use one cache root, equal query/max-results, and fixture then fake-live transports: recorded data must not satisfy the first real request; the second real run must use only the real cache. Add a differing-endpoint cache miss case.
- [ ] Add candidate mutations for title, authors, source ID, URL, year/DOI, and retrieval paths. Each must be detected by the provenance audit; an unmodified cluster projection must have zero mismatches.
- [ ] Run `py -3.12 -m pytest tests/unit/test_arxiv_adapter.py tests/unit/test_first_round_cli.py tests/integration/test_first_round_pipeline.py tests/contract/test_m1_t04_contracts.py -q` and preserve the nonzero red result before implementation.

### Task R1.2: Implement mode-safe configuration, request observations, and cache manifests

**Files:** Modify `app/adapters/arxiv.py`, `app/models/first_round.py`, `app/cli/first_round.py`, `app/core/first_round.py`.

- [ ] Add `cache_namespace` to `ArxivAdapterConfig`; include `adapter_schema_version`, namespace, endpoint, query text, and max results in the cache manifest and use the namespace as a directory boundary. Reject a cache payload whose manifest does not exactly match.
- [ ] Make `FirstRoundConfig` expose `min_request_interval_seconds`, validate `recorded >= 0` and `real >= 1`, and default the CLI to `0.0` for recorded / `3.0` for real unless `--min-request-interval-seconds` is explicitly supplied.
- [ ] In real mode construct the adapter with its default `time.sleep`; only recorded mode may receive a no-op sleeper. Record request-start offsets, observed minimum start delta, wait count, and wait seconds without recording complete request URLs.
- [ ] Raise the schema version to `m1-t04.v2`, pass `first-round:recorded` or `first-round:real` from the CLI, and verify that endpoint changes never reuse cache.

### Task R1.3: Replace the hallucination constant with an audit

**Files:** Modify `app/core/first_round.py` and `app/models/first_round.py`; add any focused helper tests to `tests/integration/test_first_round_pipeline.py`.

- [ ] Implement a helper that compares every `CandidateOutput` field to the canonical record and its `DedupCluster`: `paper_id`, source, source ID, title, authors, year, DOI, URL, retrieval paths, cluster ID, source identities, and merge reasons.
- [ ] Emit `metadata_projection_mismatch_count` and compute `metadata_hallucination_rate = mismatch_candidate_count / candidate_count`; do not set it through a constant or assertion. The normal emitted projection must audit to `0`.

### Task R1.4: Machine-validate evidence, CI, live closure, and separate commits

**Files:** Create `scripts/validate_m1_evidence.py` and `tests/contract/test_m1_evidence_validation.py`; modify `.github/workflows/docs-validation.yml`; then, only after implementation commit, modify `evaluation/reports/m1-validation.json` and `STATUS.md`.

- [ ] The validator must reject old reports and require current-head ancestry for G, I, and K; matching input SHA-256s; all automated exits zero; a successful live run with requests > 0 and zero live-cache hits; zero-request replay with cache hits; equal candidates; non-empty live-derived source-ID/URL samples at coverage 1; audit rate 0; positive rate limit evidence; distinct cache namespaces; and `M1 COMPLETE 4/4` / `M2 READY` in STATUS.
- [ ] Add `Validate completed M1 evidence` after the retained M0 validator, using `actions/checkout@v4` with `fetch-depth: 0`.
- [ ] Commit implementation and tests as `fix: harden M1 live provenance and closure gates` (K), without STATUS or `m1-validation.json`; run focused tests, full pytest, Ruff, mypy, docs validation, M0 validation, and pip check.
- [ ] Run a fresh-cache real smoke for the specified graph-retrieval question, then an identical real-cache replay. Write v3 evidence from the observed output and hashes; only if every validator gate passes, commit exactly STATUS and report as `chore: finalize M1 validated closure evidence` (L).
