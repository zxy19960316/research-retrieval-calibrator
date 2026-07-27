# M1-T02 arXiv Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Execute inline task-by-task with a red/green test cycle. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retrieve source-backed arXiv metadata through a deterministic, paginated, cached adapter with recorded fixtures and separately labelled real-network smoke evidence.

**Architecture:** `app/adapters/arxiv.py` owns HTTP policy, pagination, retry/backoff, cache and Atom-to-`PaperRecord` conversion behind an injectable transport. Tests supply response sequences, a monotonic clock and sleeper so no unit test opens a socket or sleeps. A dedicated smoke script is the only real-network entry point and reports `real_external` separately from automated fixtures.

**Tech Stack:** Python 3.12 standard library (`urllib`, `xml.etree.ElementTree`), Pydantic v2, pytest, Ruff, mypy.

## Global Constraints

- Implement M1-T02 only on `agent/m1-t02-arxiv-adapter`; do not begin M1-T03/M1-T04, M2-M6, model download, platform deployment, CNKI automation, or a real model call.
- Use the official public arXiv API Atom response only; never scrape arXiv HTML and never claim recorded fixtures are real-network evidence.
- All ordinary tests are offline; User-Agent, timeout, page start/max-results, rate limit, retry/backoff, cache and malformed Atom behavior must be deterministic.
- Commit tests/implementation/plan separately from evidence and `STATUS.md`; preserve the actual red command, exits and summaries. Real smoke is independently run and labelled `real_external` or `not_run` with its reason.

---

### Task 1: Define adapter contracts and record an offline red suite

**Files:**
- Create: `app/adapters/arxiv.py`
- Create: `tests/unit/test_arxiv_adapter.py`
- Create: `tests/fixtures/arxiv/normal.xml`
- Create: `tests/fixtures/arxiv/empty.xml`
- Create: `tests/fixtures/arxiv/malformed.xml`

**Interfaces:**

```python
class ArxivResponse(NamedTuple):
    status_code: int
    body: bytes
    headers: Mapping[str, str]

class ArxivTransport(Protocol):
    def get(self, url: str, *, headers: Mapping[str, str], timeout_seconds: float) -> ArxivResponse: ...

class ArxivAdapter:
    def search(self, query: Query, *, max_results: int) -> list[PaperRecord]: ...
```

- [ ] Add a fake transport that records URL/headers/timeout and returns the selected recorded response or raises a selected transport error.
- [ ] Add failing tests for Atom records, empty responses, `start=0` then `start=<page_size>`, URL-encoded query text, User-Agent, timeout, cache hit, duplicate arXiv IDs across pages, 429/5xx retry, exponential delays, and malformed XML.
- [ ] Run `py -3.12 -m pytest tests/unit/test_arxiv_adapter.py -q`; expect collection/import failure because the adapter does not exist.

### Task 2: Implement strict Atom conversion and one-page HTTP boundary

**Files:**
- Create: `app/adapters/arxiv.py`
- Modify: `app/adapters/__init__.py`
- Test: `tests/unit/test_arxiv_adapter.py`

**Interfaces:**

```python
class ArxivAdapterError(ValueError):
    code: str

class ArxivAdapterConfig(BaseModel):
    endpoint: str = "https://export.arxiv.org/api/query"
    user_agent: str
    timeout_seconds: float
    page_size: int
    min_request_interval_seconds: float
    max_attempts: int
    initial_backoff_seconds: float
```

- [ ] Parse Atom namespace entries with a non-empty `id`, title and summary; retain only source-confirmed title, summary, authors, published year and HTTP(S) arXiv URL in `PaperRecord`.
- [ ] Refuse malformed XML, non-200 responses and entries missing source identity with stable `ArxivAdapterError` codes; do not fabricate titles, abstracts, IDs or URLs.
- [ ] Generate `search_query`, `start` and `max_results` with `urllib.parse.urlencode`; pass exactly the configured User-Agent and timeout to transport.
- [ ] Run the targeted Atom/HTTP tests and confirm they pass without network access.

### Task 3: Add pagination, rate, retry, cache and recorded-response guarantees

**Files:**
- Modify: `app/adapters/arxiv.py`
- Modify: `tests/unit/test_arxiv_adapter.py`
- Modify: `tests/fixtures/arxiv/normal.xml`

**Interfaces:** `ArxivAdapter.search()` advances `start` by page size until `max_results` is satisfied, an empty page arrives, or a short page ends the result stream. Its cache key is the canonical query text plus requested maximum; cached results return defensive copies.

- [ ] Before each uncached request, compute `max(0, min_interval - elapsed)` with the injected monotonic clock and call injected sleeper only for a positive delay.
- [ ] Retry only configured transient transport errors and HTTP `429`/`500`/`502`/`503`/`504`; sleep `initial_backoff * 2**attempt` between attempts and never duplicate parsed records.
- [ ] Deduplicate pages by arXiv source ID while retaining the initial query ID in each `retrieval_paths`; keep page order stable.
- [ ] Cache only successful complete searches; a cache hit performs no transport, sleep or retry operation.
- [ ] Run `py -3.12 -m pytest tests/unit/test_arxiv_adapter.py -q`; expected all adapter unit tests pass offline.

### Task 4: Add independent real-network smoke and task evidence

**Files:**
- Create: `scripts/arxiv_smoke.py`
- Create: `evaluation/reports/m1-t02-arxiv-adapter.json`
- Modify: `STATUS.md`

- [ ] The smoke script accepts an explicit canonical query, sends one bounded request using the adapter, emits JSON with URL-free safe command metadata, source IDs and outcome, and exits nonzero on no confirmed result or transport failure. It must not run as part of pytest.
- [ ] Run focused adapter tests, full pytest, Ruff, mypy, project-doc validation, historical M0 validation and pip check. Then run one explicit smoke command; record its UTC time, query, result count and source-ID sample only if the network call succeeds. If it cannot run, record `not_run` with the observed reason.
- [ ] Commit implementation/tests/fixtures/plan as `feat: add M1 arXiv retrieval adapter`; write evidence that references that full commit and hashes its changed blobs. Commit only evidence and `STATUS.md` as `chore: record M1-T02 adapter evidence`.
- [ ] Push the existing M1-T02 branch and open/update one Draft PR only after all local checks are green. Roll back using `git revert <evidence>` then `git revert <implementation>`.

### M1-T02R transport, Atom-error and cache-provenance closure

- [x] Detect arXiv Atom error feeds before creating a `PaperRecord`; report `ARXIV_API_ERROR` with an optional source reason, and reject malformed source identities with `INVALID_ARXIV_ENTRY`.
- [x] Keep HTTP response status/body/headers at the urllib transport boundary, but translate DNS, proxy, TLS and timeout failures to a retryable `ArxivTransportFailure`. The adapter alone maps exhausted attempts to `ARXIV_TRANSPORT_ERROR`.
- [x] Schedule every transport attempt through one stateful boundary. The wait is the maximum of the remaining three-second minimum interval, exponential backoff and a valid `Retry-After`; cache hits do not sleep.
- [x] Cache source metadata independently of query provenance, return deep copies, and bind every returned record's `retrieval_paths` to the current `query_id`.
- [x] Add recorded error Atom, modern/legacy ID, default-urllib transport retry, bounded retry, per-attempt interval, `Retry-After` and cache-isolation tests. The independent smoke reports URL-free `attempt_count`, HTTP status, error code, retry-after and elapsed time.
- [x] Preserve scope: this closure does not implement M1-T03 normalization/deduplication, M1-T04 orchestration, ranking, models, CNKI, or platform work. Real smoke evidence reports adapter observation separately from connectivity diagnostics; a causal link is not established.

## Self-Review

- Coverage: HTTP boundary, Atom parsing, pagination, User-Agent, timeout, rate limit, retry/backoff, cache, recorded fixture, automated/recorded/real evidence separation and independent smoke each have a named task and test path.
- Placeholder scan: no HTML scraping, retrieval ranking, normalization/deduplication phase work, CNKI automation, model work or M1-T03/M1-T04 behavior is included.
- Type consistency: the adapter consumes M1-T01 `Query` and produces only existing `PaperRecord` contracts; transport, time and sleeper seams remain private implementation inputs.
