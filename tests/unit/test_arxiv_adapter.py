from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from pathlib import Path
from typing import Self
from urllib.error import URLError
from urllib.parse import parse_qs, urlparse

import pytest

from app.adapters.arxiv import (
    ArxivAdapter,
    ArxivAdapterConfig,
    ArxivAdapterError,
    ArxivResponse,
)
from app.models.enums import QueryBranch, QueryBreadth
from app.models.query import Query
from scripts import arxiv_smoke

FIXTURES = Path(__file__).parents[1] / "fixtures" / "arxiv"


class FakeTransport:
    def __init__(self, responses: list[ArxivResponse | Exception]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, Mapping[str, str], float]] = []

    def get(self, url: str, *, headers: Mapping[str, str], timeout_seconds: float) -> ArxivResponse:
        self.calls.append((url, headers, timeout_seconds))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _response(name: str, status_code: int = 200) -> ArxivResponse:
    return ArxivResponse(status_code, (FIXTURES / name).read_bytes(), {})


def _atom_record(source_id: str, *, title: str = "Recorded title", summary: str = "Recorded abstract", published: str = "2024-01-01") -> bytes:
    return f'''<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"><entry>
      <id>https://arxiv.org/abs/{source_id}</id><title>{title}</title><summary>{summary}</summary>
      <published>{published}</published>
    </entry></feed>'''.encode()


def _query() -> Query:
    return Query(
        query_id="Q1-fixture",
        round_number=1,
        branch=QueryBranch.DIRECT_INTERSECTION,
        breadth=QueryBreadth.MEDIUM,
        language="en",
        query_text='all:"radiation shielding" AND all:"transfer learning"',
        weight=1.0,
    )


def _adapter(
    responses: list[ArxivResponse | Exception], **overrides: object
) -> tuple[ArxivAdapter, FakeTransport, list[float]]:
    clock = [0.0]
    sleeps: list[float] = []
    transport = FakeTransport(responses)
    config_values: dict[str, object] = {
        "user_agent": "rrc-tests/1.0",
        "timeout_seconds": 4.5,
        "page_size": 2,
        "min_request_interval_seconds": 1.0,
        "max_attempts": 3,
        "max_total_results": 100,
        "max_total_attempts": 20,
        "max_retry_after_seconds": 60.0,
        "initial_backoff_seconds": 0.25,
    }
    config_values.update(overrides)
    config = ArxivAdapterConfig(
        **config_values,
    )
    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock[0] += seconds

    return (
        ArxivAdapter(
            config,
            transport=transport,
            monotonic=lambda: clock[0],
            sleeper=sleep,
            utc_now=lambda: datetime(2026, 7, 27, tzinfo=UTC),
        ),
        transport,
        sleeps,
    )


def test_parses_recorded_atom_and_sends_configured_http_boundary() -> None:
    adapter, transport, _ = _adapter([_response("normal.xml")])

    papers = adapter.search(_query(), max_results=2)

    assert [paper.source_id for paper in papers] == ["2401.00001", "2401.00002"]
    assert papers[0].title == "Radiation Shielding with Transfer Learning"
    assert papers[0].authors == ["Ada Author", "Ben Author"]
    assert papers[0].retrieval_paths == ["Q1-fixture"]
    url, headers, timeout = transport.calls[0]
    assert parse_qs(urlparse(url).query) == {
        "search_query": [_query().query_text],
        "start": ["0"],
        "max_results": ["2"],
    }
    assert headers["User-Agent"] == "rrc-tests/1.0"
    assert timeout == 4.5


def test_paginates_deduplicates_and_caches_successful_result() -> None:
    adapter, transport, _ = _adapter([_response("normal.xml"), _response("normal.xml"), _response("empty.xml")])

    first = adapter.search(_query(), max_results=3)
    second = adapter.search(_query(), max_results=3)

    assert [paper.source_id for paper in first] == ["2401.00001", "2401.00002"]
    assert first == second and first is not second
    assert [parse_qs(urlparse(call[0]).query)["start"] for call in transport.calls] == [["0"], ["2"]]


def test_transient_status_retries_with_exponential_backoff_and_no_duplicate_records() -> None:
    adapter, transport, sleeps = _adapter([_response("empty.xml", 429), _response("normal.xml")])

    papers = adapter.search(_query(), max_results=2)

    assert len(papers) == 2
    assert len(transport.calls) == 2
    assert sleeps == [1.0]


def test_rate_limit_applies_before_uncached_second_request() -> None:
    adapter, _, sleeps = _adapter([_response("empty.xml"), _response("empty.xml")])

    adapter.search(_query(), max_results=1)
    adapter.search(
        _query().model_copy(
            update={"query_id": "Q1-other", "query_text": 'all:"nuclear engineering"'}
        ),
        max_results=1,
    )

    assert sleeps == [1.0]


def test_rate_limit_observation_records_real_request_starts_and_waits() -> None:
    adapter, transport, sleeps = _adapter(
        [_response("empty.xml"), _response("empty.xml")],
        min_request_interval_seconds=3.0,
    )

    adapter.search(_query(), max_results=1)
    adapter.search(
        _query().model_copy(
            update={"query_id": "Q1-other", "query_text": 'all:"different query"'}
        ),
        max_results=1,
    )

    observation = adapter.rate_limit_observation
    assert len(transport.calls) == 2
    assert sleeps == [3.0]
    assert observation.configured_min_request_interval_seconds == 3.0
    assert observation.request_start_offsets_seconds == (0.0, 3.0)
    assert observation.minimum_observed_request_start_delta_seconds == 3.0
    assert observation.rate_limit_wait_count == 1
    assert observation.rate_limit_wait_seconds == 3.0


def test_persistent_cache_does_not_cross_namespace_or_endpoint(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    first, first_transport, _ = _adapter(
        [_response("normal.xml")],
        cache_dir=cache_dir,
        cache_schema_version="m1-t04.v2",
        cache_namespace="first-round:recorded",
    )
    second, second_transport, _ = _adapter(
        [_response("empty.xml")],
        cache_dir=cache_dir,
        cache_schema_version="m1-t04.v2",
        cache_namespace="first-round:real",
    )
    real_replay, replay_transport, _ = _adapter(
        [],
        cache_dir=cache_dir,
        cache_schema_version="m1-t04.v2",
        cache_namespace="first-round:real",
    )
    endpoint_changed, endpoint_transport, _ = _adapter(
        [_response("empty.xml")],
        cache_dir=cache_dir,
        cache_schema_version="m1-t04.v2",
        cache_namespace="first-round:real",
        endpoint="https://example.test/arxiv",
    )

    assert first.search(_query(), max_results=1)
    assert second.search(_query(), max_results=1) == []
    assert real_replay.search(_query(), max_results=1) == []
    assert endpoint_changed.search(_query(), max_results=1) == []
    assert len(first_transport.calls) == 1
    assert len(second_transport.calls) == 1
    assert replay_transport.calls == []
    assert len(endpoint_transport.calls) == 1


def test_malformed_atom_and_non_transient_response_fail_closed() -> None:
    malformed, _, _ = _adapter([_response("malformed.xml")])
    rejected, _, _ = _adapter([_response("empty.xml", 400)])

    with pytest.raises(ArxivAdapterError, match="MALFORMED_ARXIV_ATOM"):
        malformed.search(_query(), max_results=1)
    with pytest.raises(ArxivAdapterError, match="ARXIV_HTTP_400"):
        rejected.search(_query(), max_results=1)


def test_error_atom_is_not_a_paper_record() -> None:
    adapter, _, _ = _adapter([_response("error.xml")])

    with pytest.raises(ArxivAdapterError, match="ARXIV_API_ERROR"):
        adapter.search(_query(), max_results=1)


@pytest.mark.parametrize(
    "source_url",
    [
        "https://example.test/abs/2401.00001v2",
        "https://arxiv.org/pdf/2401.00001v2",
        "https://arxiv.org/abs/not-an-arxiv-id",
        "https://arxiv.org/abs/",
    ],
)
def test_invalid_arxiv_source_identity_is_rejected(source_url: str) -> None:
    body = f'''<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"><entry>
      <id>{source_url}</id><title>Recorded title</title><summary>Recorded abstract</summary>
    </entry></feed>'''.encode()
    adapter, _, _ = _adapter([ArxivResponse(200, body, {})])

    with pytest.raises(ArxivAdapterError, match="INVALID_ARXIV_ENTRY"):
        adapter.search(_query(), max_results=1)


def test_validates_new_and_legacy_arxiv_ids() -> None:
    body = b'''<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">
      <entry><id>https://arxiv.org/abs/2401.00001v2</id><title>New</title><summary>New</summary></entry>
      <entry><id>https://arxiv.org/abs/hep-ex/0307015v1</id><title>Legacy</title><summary>Legacy</summary></entry>
    </feed>'''
    adapter, _, _ = _adapter([ArxivResponse(200, body, {})])

    papers = adapter.search(_query(), max_results=2)

    assert [paper.source_id for paper in papers] == ["2401.00001", "hep-ex/0307015"]


@pytest.mark.parametrize(
    ("source_url", "expected_source_id"),
    [
        ("https://arxiv.org/abs/math.GT/0309136", "math.GT/0309136"),
        ("https://arxiv.org/abs/math.GT/0309136v2", "math.GT/0309136"),
        ("https://arxiv.org/abs/cs.SE/0501001", "cs.SE/0501001"),
        ("https://arxiv.org/abs/nlin.CD/0101001v1", "nlin.CD/0101001"),
        ("https://arxiv.org/abs/hep-ex/0307015v1", "hep-ex/0307015"),
        ("https://arxiv.org/abs/astro-ph/9901001", "astro-ph/9901001"),
    ],
)
def test_accepts_canonical_legacy_arxiv_identifier_forms(
    source_url: str, expected_source_id: str
) -> None:
    body = f'''<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"><entry>
      <id>{source_url}</id><title>Recorded title</title><summary>Recorded abstract</summary>
    </entry></feed>'''.encode()
    adapter, _, _ = _adapter([ArxivResponse(200, body, {})])

    papers = adapter.search(_query(), max_results=1)

    assert [paper.source_id for paper in papers] == [expected_source_id]
    assert papers[0].retrieval_paths == ["Q1-fixture"]


def test_mixed_modern_and_canonical_legacy_atom_page_preserves_order() -> None:
    body = b'''<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">
      <entry><id>https://arxiv.org/abs/2401.00001v2</id><title>Modern</title><summary>Modern</summary></entry>
      <entry><id>https://arxiv.org/abs/math.GT/0309136</id><title>Math</title><summary>Math</summary></entry>
      <entry><id>https://arxiv.org/abs/cs.SE/0501001v3</id><title>Computer science</title><summary>Computer science</summary></entry>
    </feed>'''
    adapter, _, _ = _adapter([ArxivResponse(200, body, {})])

    papers = adapter.search(_query(), max_results=3)

    assert [paper.source_id for paper in papers] == [
        "2401.00001",
        "math.GT/0309136",
        "cs.SE/0501001",
    ]
    assert [paper.retrieval_paths for paper in papers] == [["Q1-fixture"]] * 3


@pytest.mark.parametrize(
    "source_url",
    [
        "https://arxiv.org/abs/math.gt/0309136",
        "https://arxiv.org/abs/math.GT/0309136v0",
        "https://arxiv.org/abs/cs.S/0501001",
        "https://arxiv.org/abs/cs.TOOLONG/0501001",
        "https://arxiv.org/abs/math.GT/030913",
        "https://arxiv.org/abs/math.GT/03091360",
        "https://arxiv.org/abs/2401.00001v0",
    ],
)
def test_rejects_noncanonical_legacy_or_zero_version_identifier(source_url: str) -> None:
    body = f'''<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"><entry>
      <id>{source_url}</id><title>Recorded title</title><summary>Recorded abstract</summary>
    </entry></feed>'''.encode()
    adapter, _, _ = _adapter([ArxivResponse(200, body, {})])

    with pytest.raises(ArxivAdapterError) as raised:
        adapter.search(_query(), max_results=1)

    assert raised.value.code == "INVALID_ARXIV_ENTRY"


@pytest.mark.parametrize("failure", [URLError("temporary DNS failure"), TimeoutError("timeout")])
def test_transport_failure_retries_through_default_urllib_boundary(
    monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    from app.adapters import arxiv

    class Response:
        def __init__(self) -> None:
            self.status = 200
            self.headers: dict[str, str] = {}

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def read(self) -> bytes:
            return (FIXTURES / "normal.xml").read_bytes()

    calls = 0

    def urlopen_once_then_succeed(*_: object, **__: object) -> Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise failure
        return Response()

    monkeypatch.setattr(arxiv, "urlopen", urlopen_once_then_succeed)
    adapter, _, sleeps = _adapter([], min_request_interval_seconds=0.0)
    adapter._transport = arxiv.UrllibArxivTransport()  # type: ignore[assignment]

    papers = adapter.search(_query(), max_results=1)

    assert len(papers) == 1
    assert calls == 2
    assert sleeps == [0.25]


def test_transport_failure_retries_until_the_configured_attempt_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.adapters import arxiv

    calls = 0

    def always_fail(*_: object, **__: object) -> None:
        nonlocal calls
        calls += 1
        raise URLError("persistent DNS failure")

    monkeypatch.setattr(arxiv, "urlopen", always_fail)
    adapter, _, sleeps = _adapter([], min_request_interval_seconds=0.0)
    adapter._transport = arxiv.UrllibArxivTransport()  # type: ignore[assignment]

    with pytest.raises(ArxivAdapterError, match="ARXIV_TRANSPORT_ERROR"):
        adapter.search(_query(), max_results=1)

    assert calls == 3
    assert sleeps == [0.25, 0.5]


def test_every_attempt_obeys_minimum_interval_over_shorter_backoff() -> None:
    adapter, transport, sleeps = _adapter(
        [_response("empty.xml", 429), _response("normal.xml")],
        min_request_interval_seconds=3.0,
        initial_backoff_seconds=1.0,
    )

    adapter.search(_query(), max_results=1)

    assert len(transport.calls) == 2
    assert sleeps == [3.0]


def test_retry_after_overrides_shorter_backoff_and_interval() -> None:
    adapter, _, sleeps = _adapter(
        [ArxivResponse(429, b"", {"Retry-After": "7"}), _response("normal.xml")],
        min_request_interval_seconds=3.0,
        initial_backoff_seconds=1.0,
    )

    adapter.search(_query(), max_results=1)

    assert sleeps == [7.0]


def test_invalid_retry_after_falls_back_to_exponential_backoff() -> None:
    adapter, _, sleeps = _adapter(
        [ArxivResponse(429, b"", {"Retry-After": "not-a-delay"}), _response("normal.xml")],
        min_request_interval_seconds=0.0,
        initial_backoff_seconds=1.0,
    )

    adapter.search(_query(), max_results=1)

    assert sleeps == [1.0]


def test_cache_rebinds_provenance_and_defensively_copies() -> None:
    adapter, transport, sleeps = _adapter([_response("normal.xml")])
    first = adapter.search(_query(), max_results=1)
    second_query = _query().model_copy(update={"query_id": "Q1-second"})
    second = adapter.search(second_query, max_results=1)
    second[0].retrieval_paths.append("caller-mutation")
    third = adapter.search(_query(), max_results=1)

    assert len(transport.calls) == 1
    assert sleeps == []
    assert first[0].retrieval_paths == ["Q1-fixture"]
    assert second[0].retrieval_paths == ["Q1-second", "caller-mutation"]
    assert third[0].retrieval_paths == ["Q1-fixture"]


def test_accepts_max_results_at_total_result_limit() -> None:
    adapter, _, _ = _adapter([_response("normal.xml")], max_total_results=2)

    assert len(adapter.search(_query(), max_results=2)) == 2


def test_rejects_max_results_over_total_result_limit() -> None:
    adapter, transport, _ = _adapter([], max_total_results=2)

    with pytest.raises(ArxivAdapterError, match="INVALID_ARXIV_MAX_RESULTS"):
        adapter.search(_query(), max_results=3)

    assert transport.calls == []


def test_total_attempt_budget_spans_pages_and_retries_without_caching_partial_results() -> None:
    adapter, transport, _ = _adapter(
        [
            ArxivResponse(200, _atom_record("2401.00001"), {}),
            ArxivResponse(429, b"", {}),
            ArxivResponse(200, _atom_record("2401.00002"), {}),
        ],
        page_size=1,
        max_total_attempts=3,
        min_request_interval_seconds=0.0,
    )

    with pytest.raises(ArxivAdapterError, match="ARXIV_REQUEST_BUDGET_EXHAUSTED"):
        adapter.search(_query(), max_results=3)

    assert len(transport.calls) == 3
    assert adapter.last_observation.attempt_count == 3
    assert adapter._cache == {}


def test_retry_after_is_capped_and_observation_records_the_cap() -> None:
    adapter, _, sleeps = _adapter(
        [ArxivResponse(429, b"", {"Retry-After": "86400"}), _response("normal.xml")],
        min_request_interval_seconds=0.0,
        initial_backoff_seconds=1.0,
        max_retry_after_seconds=60.0,
    )

    adapter.search(_query(), max_results=1)

    assert sleeps == [60.0]
    assert adapter.last_observation.retry_after_seconds == 60.0
    assert adapter.last_observation.retry_after_was_capped is True


def test_http_date_retry_after_uses_injected_wall_clock_and_cap() -> None:
    now = datetime(2026, 7, 27, tzinfo=UTC)
    retry_after = format_datetime(now + timedelta(seconds=120), usegmt=True)
    adapter, _, sleeps = _adapter(
        [ArxivResponse(429, b"", {"Retry-After": retry_after}), _response("normal.xml")],
        min_request_interval_seconds=0.0,
        max_retry_after_seconds=30.0,
    )

    adapter.search(_query(), max_results=1)

    assert sleeps == [30.0]
    assert adapter.last_observation.retry_after_seconds == 30.0
    assert adapter.last_observation.retry_after_was_capped is True


def test_negative_retry_after_still_obeys_minimum_interval() -> None:
    adapter, _, sleeps = _adapter(
        [ArxivResponse(429, b"", {"Retry-After": "-4"}), _response("normal.xml")],
        min_request_interval_seconds=3.0,
        initial_backoff_seconds=1.0,
    )

    adapter.search(_query(), max_results=1)

    assert sleeps == [3.0]


def test_well_formed_non_atom_root_fails_closed_without_cache() -> None:
    adapter, transport, _ = _adapter([_response("wrong-root.xml")])

    with pytest.raises(ArxivAdapterError, match="INVALID_ARXIV_ATOM"):
        adapter.search(_query(), max_results=1)

    assert len(transport.calls) == 1
    assert adapter._cache == {}
    assert adapter.last_observation.final_error_code == "INVALID_ARXIV_ATOM"


@pytest.mark.parametrize(
    "body",
    [
        _atom_record("2401.00001", published="1899-01-01"),
        _atom_record("2401.00001", title=""),
        _atom_record("2401.00001", summary=""),
        _atom_record("not-an-arxiv-id"),
    ],
)
def test_entry_validation_errors_are_wrapped_in_a_stable_adapter_error(body: bytes) -> None:
    adapter, _, _ = _adapter([ArxivResponse(200, body, {})])

    with pytest.raises(ArxivAdapterError) as raised:
        adapter.search(_query(), max_results=1)

    assert raised.value.code == "INVALID_ARXIV_ENTRY"


@pytest.mark.parametrize(
    "arguments",
    [
        ["--max-results", "0"],
        ["--max-results", "101"],
        ["--user-agent", "   "],
        ["--query", "   "],
    ],
)
def test_smoke_invalid_arguments_emit_json_failure_without_a_traceback(
    arguments: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert arxiv_smoke.main(arguments) == 2

    payload = __import__("json").loads(capsys.readouterr().out)
    assert payload == {
        "evidence_type": "real_external",
        "status": "failed",
        "error_code": "INVALID_SMOKE_ARGUMENT",
    }
