"""Recorded integration coverage for the bounded M1-T04 orchestration core."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.adapters.arxiv import ArxivAdapter, ArxivAdapterConfig, ArxivResponse
from app.core.first_round import audit_candidate_provenance, run_first_round
from app.models.dedup import DedupCluster
from app.models.first_round import (
    CandidateOutput,
    FirstRoundConfig,
    FirstRoundRun,
    FirstRoundStatus,
)
from app.models.paper import PaperRecord

QUESTION = "How can graph-based retrieval support scientific literature discovery?"
FIXED_NOW = datetime(2026, 7, 27, 12, 0, tzinfo=UTC)


class RecordedTransport:
    """Test-only recorded Atom response sequence; it never opens the network."""

    def __init__(self, responses: list[ArxivResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[str] = []

    def get(self, url: str, *, headers: Mapping[str, str], timeout_seconds: float) -> ArxivResponse:
        del headers, timeout_seconds
        self.calls.append(url)
        return self._responses.pop(0)


def _response(*entries: tuple[str, str]) -> ArxivResponse:
    records = "".join(
        f"""<entry><id>https://arxiv.org/abs/{source_id}</id><title>{title}</title>
        <summary>Recorded source metadata.</summary><published>2024-01-01</published>
        <author><name>Ada Author</name></author></entry>"""
        for source_id, title in entries
    )
    body = f'<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">{records}</feed>'.encode()
    return ArxivResponse(200, body, {})


def _adapter(
    responses: list[ArxivResponse], *, cache_dir: Path | None = None
) -> tuple[ArxivAdapter, RecordedTransport]:
    transport = RecordedTransport(responses)
    return (
        ArxivAdapter(
            ArxivAdapterConfig(
                user_agent="rrc-m1-t04-tests/1.0",
                timeout_seconds=1.0,
                page_size=5,
                min_request_interval_seconds=0.0,
                max_attempts=1,
                max_total_results=5,
                max_total_attempts=20,
                initial_backoff_seconds=0.0,
                cache_dir=cache_dir,
                cache_schema_version="m1-t04.v1",
            ),
            transport=transport,
            monotonic=lambda: 10.0,
            sleeper=lambda _: None,
            utc_now=lambda: FIXED_NOW,
        ),
        transport,
    )


def _config(tmp_path: Path, **overrides: object) -> FirstRoundConfig:
    values: dict[str, object] = {
        "cache_dir": tmp_path / "cache",
        "mode": "recorded",
        "max_results_per_query": 5,
        "max_total_candidates": 60,
        "max_total_attempts": 20,
        "timeout_seconds": 1.0,
    }
    values.update(overrides)
    return FirstRoundConfig.model_validate(values)


def _run(adapter: ArxivAdapter, config: FirstRoundConfig) -> FirstRoundRun:
    return run_first_round(
        QUESTION,
        config=config,
        adapter=adapter,
        now=lambda: FIXED_NOW,
        monotonic=lambda: 100.0,
    )


def _cluster_for_candidate(source: CandidateOutput) -> DedupCluster:
    record = PaperRecord(
        paper_id=source.paper_id,
        source=source.source,
        source_id=source.source_id,
        title=source.title,
        authors=source.authors,
        year=source.year,
        doi=source.doi,
        url=source.url,
        language="en",
        retrieval_paths=source.retrieval_paths,
    )
    return DedupCluster(
        cluster_id=source.cluster_id,
        canonical_record=record,
        member_records=[record],
        retrieval_paths=source.retrieval_paths,
        source_identities=source.member_source_identities,
        merge_reasons=source.merge_reasons,
    )


def test_recorded_question_runs_ir_plan_arxiv_dedup_and_outputs_source_backed_candidates(
    tmp_path: Path,
) -> None:
    adapter, _ = _adapter(
        [
            _response(("2401.00001", "Graph Retrieval"), ("2401.00002", "Literature Discovery")),
            _response(("2401.00001", "Graph Retrieval")),
            *[_response() for _ in range(10)],
        ]
    )

    run = _run(adapter, _config(tmp_path))

    assert run.status is FirstRoundStatus.SUCCESS
    assert run.query_plan is not None
    assert len(run.query_plan.queries) == 12
    assert run.raw_candidate_count == 3
    assert run.deduplicated_candidate_count == 2
    assert run.metrics.source_id_coverage == 1.0
    assert run.metrics.url_coverage == 1.0
    assert run.metrics.metadata_hallucination_rate == 0.0
    assert run.candidates[0].retrieval_paths == sorted(run.candidates[0].retrieval_paths)
    assert run.candidates[0].retrieval_paths == [
        run.query_plan.queries[0].query_id,
        run.query_plan.queries[1].query_id,
    ]


def test_all_queries_unavailable_returns_honest_failure_without_candidates(tmp_path: Path) -> None:
    adapter, _ = _adapter([ArxivResponse(503, b"unavailable", {}) for _ in range(12)])

    run = _run(adapter, _config(tmp_path))

    assert run.status is FirstRoundStatus.FAILED
    assert run.error_code == "ARXIV_UNAVAILABLE"
    assert run.candidates == []
    assert len(run.query_results) == 12
    assert all(item.status == "failed" for item in run.query_results)


def test_one_query_failure_with_records_is_partial_success(tmp_path: Path) -> None:
    adapter, _ = _adapter(
        [
            ArxivResponse(503, b"unavailable", {}),
            _response(("2401.00001", "Graph Retrieval")),
            *[_response() for _ in range(10)],
        ]
    )

    run = _run(adapter, _config(tmp_path))

    assert run.status is FirstRoundStatus.PARTIAL_SUCCESS
    assert {item.query_id for item in run.query_results if item.status == "failed"}
    assert run.candidates


def test_blank_question_returns_intent_parse_failure_without_transport(tmp_path: Path) -> None:
    adapter, transport = _adapter([_response()])

    run = run_first_round(
        "   ",
        config=_config(tmp_path),
        adapter=adapter,
        now=lambda: FIXED_NOW,
        monotonic=lambda: 100.0,
    )

    assert run.status is FirstRoundStatus.FAILED
    assert run.error_code == "INTENT_PARSE_FAILED"
    assert transport.calls == []


def test_candidate_budget_stops_before_further_transport(tmp_path: Path) -> None:
    adapter, transport = _adapter(
        [_response(("2401.00001", "Graph Retrieval"), ("2401.00002", "Literature Discovery"))]
    )

    run = _run(adapter, _config(tmp_path, max_total_candidates=2))

    assert run.raw_candidate_count == 2
    assert run.metrics.candidate_budget_reached is True
    assert len(transport.calls) == 1
    assert len(run.query_results) == 1


def test_empty_recorded_responses_are_insufficient_candidates(tmp_path: Path) -> None:
    adapter, _ = _adapter([_response() for _ in range(12)])

    run = _run(adapter, _config(tmp_path))

    assert run.status is FirstRoundStatus.FAILED
    assert run.error_code == "INSUFFICIENT_CANDIDATES"
    assert run.candidates == []


def test_deduplication_conflicts_return_structured_failure(tmp_path: Path) -> None:
    adapter, _ = _adapter(
        [
            _response(("2401.00001", "Graph Retrieval")),
            _response(("2401.00001", "Conflicting Source Title")),
            *[_response() for _ in range(10)],
        ]
    )

    run = _run(adapter, _config(tmp_path))

    assert run.status is FirstRoundStatus.FAILED
    assert run.error_code == "DEDUPLICATION_FAILED"
    assert run.candidates == []


def test_fixed_clocks_produce_deterministic_json(tmp_path: Path) -> None:
    responses = [_response(("2401.00001", "Graph Retrieval")), *[_response() for _ in range(11)]]
    first_adapter, _ = _adapter(responses)
    second_adapter, _ = _adapter(list(responses))

    first = _run(first_adapter, _config(tmp_path))
    second = _run(second_adapter, _config(tmp_path))

    assert first.model_dump(mode="json") == second.model_dump(mode="json")


def test_invalid_constructed_configuration_does_not_call_adapter(tmp_path: Path) -> None:
    adapter, transport = _adapter([_response()])
    invalid_values = _config(tmp_path).model_dump()
    invalid_values["max_total_candidates"] = 0
    invalid = FirstRoundConfig.model_construct(**invalid_values)

    with pytest.raises(ValueError, match="INVALID_CONFIGURATION"):
        _run(adapter, invalid)

    assert transport.calls == []


def test_attempt_budget_prevents_an_adapter_call_that_could_exceed_it(tmp_path: Path) -> None:
    transport = RecordedTransport([ArxivResponse(503, b"unavailable", {})])
    adapter = ArxivAdapter(
        ArxivAdapterConfig(
            user_agent="rrc-m1-t04-tests/1.0",
            timeout_seconds=1.0,
            page_size=1,
            min_request_interval_seconds=0.0,
            max_attempts=4,
            max_total_results=5,
            max_total_attempts=20,
            initial_backoff_seconds=0.0,
        ),
        transport=transport,
        monotonic=lambda: 10.0,
        sleeper=lambda _: None,
        utc_now=lambda: FIXED_NOW,
    )

    run = _run(adapter, _config(tmp_path, max_total_attempts=3))

    assert transport.calls == []
    assert run.status is FirstRoundStatus.FAILED
    assert run.error_code == "ARXIV_ATTEMPT_BUDGET_EXHAUSTED"
    assert run.query_results[0].attempt_count == 0
    assert run.query_results[0].error_code == "ARXIV_ATTEMPT_BUDGET_EXHAUSTED"
    assert run.metrics.transport_requests == 0
    assert run.metrics.transport_requests <= run.config.max_total_attempts


def test_persistent_cache_replay_has_zero_transport_requests(tmp_path: Path) -> None:
    cache_dir = tmp_path / "persistent-cache"
    first_adapter, first_transport = _adapter(
        [_response(("2401.00001", "Graph Retrieval")) for _ in range(12)],
        cache_dir=cache_dir,
    )
    second_adapter, second_transport = _adapter(
        [_response(("2401.00001", "Graph Retrieval")) for _ in range(12)],
        cache_dir=cache_dir,
    )
    config = _config(tmp_path, cache_dir=cache_dir)

    first = _run(first_adapter, config)
    second = _run(second_adapter, config)

    assert first.metrics.transport_requests > 0
    assert first_transport.calls
    assert second.metrics.transport_requests == 0
    assert second.metrics.cache_hits == len(second.query_results)
    assert not second_transport.calls
    assert first.candidates == second.candidates


@pytest.mark.parametrize(
    "field,value",
    [
        ("title", "Altered title"),
        ("authors", ["Altered Author"]),
        ("source_id", "9999.99999"),
        ("url", "https://arxiv.org/abs/9999.99999"),
        ("year", 2025),
        ("doi", "10.1000/altered"),
        ("retrieval_paths", ["Q-altered"]),
    ],
)
def test_metadata_provenance_audit_detects_candidate_mutations(
    tmp_path: Path, field: str, value: object
) -> None:
    adapter, _ = _adapter([_response(("2401.00001", "Graph Retrieval")), *[_response() for _ in range(11)]])
    run = _run(adapter, _config(tmp_path))

    mutated = run.candidates[0].model_copy(update={field: value})
    assert audit_candidate_provenance([mutated], [_cluster_for_candidate(run.candidates[0])]) == 1


def test_metadata_provenance_audit_accepts_exact_cluster_projection(tmp_path: Path) -> None:
    adapter, _ = _adapter([_response(("2401.00001", "Graph Retrieval")), *[_response() for _ in range(11)]])
    run = _run(adapter, _config(tmp_path))

    assert audit_candidate_provenance(run.candidates, [_cluster_for_candidate(run.candidates[0])]) == 0
