from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
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


def _adapter(responses: list[ArxivResponse | Exception], **overrides: object) -> tuple[ArxivAdapter, FakeTransport, list[float]]:
    clock = [0.0]
    sleeps: list[float] = []
    transport = FakeTransport(responses)
    config = ArxivAdapterConfig(
        user_agent="rrc-tests/1.0",
        timeout_seconds=4.5,
        page_size=2,
        min_request_interval_seconds=1.0,
        max_attempts=3,
        initial_backoff_seconds=0.25,
        **overrides,
    )
    return ArxivAdapter(config, transport=transport, monotonic=lambda: clock[0], sleeper=sleeps.append), transport, sleeps


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
    assert sleeps == [0.25]


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
