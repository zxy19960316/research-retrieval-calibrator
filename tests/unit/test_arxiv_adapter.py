from __future__ import annotations

from collections.abc import Mapping
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
        ArxivAdapter(config, transport=transport, monotonic=lambda: clock[0], sleeper=sleep),
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
