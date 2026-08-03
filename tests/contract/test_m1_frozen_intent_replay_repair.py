"""Contract tests for preserving the frozen M1 intent during replay."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.adapters.arxiv import (
    ArxivAdapter,
    ArxivAdapterConfig,
    ArxivResponse,
    ArxivTransportFailure,
)
from app.core.first_round import run_first_round
from app.core.intent import canonical_research_intent_bytes
from app.models.enums import MethodConstraint
from app.models.first_round import FirstRoundConfig, FirstRoundStatus
from app.models.project import ResearchIntent

QUESTION = "How can graph-based retrieval support scientific literature discovery?"
FIRST_STARTED_AT = datetime(2026, 7, 28, 7, 8, 40, 329494, tzinfo=UTC)
REPLAY_STARTED_AT = datetime(2026, 7, 28, 7, 10, 58, 105607, tzinfo=UTC)


class RecordedTransport:
    """Test-only Atom transport; it never opens a network connection."""

    def __init__(self, responses: list[ArxivResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[str] = []

    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> ArxivResponse:
        del headers, timeout_seconds
        self.calls.append(url)
        return self._responses.pop(0)


class ForbiddenTransport:
    """A transport that turns any cache miss into an explicit offline failure."""

    def __init__(self) -> None:
        self.calls = 0

    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> ArxivResponse:
        del url, headers, timeout_seconds
        self.calls += 1
        raise ArxivTransportFailure("OFFLINE_CACHE_MISS")


def test_normal_run_uses_started_at_for_intent_and_query_plan(tmp_path: Path) -> None:
    config = _config(tmp_path)
    adapter, _ = _adapter(_responses(), config)

    run = _run(adapter, config, FIRST_STARTED_AT)

    assert run.status is FirstRoundStatus.SUCCESS
    assert run.intent is not None
    assert run.intent.frozen_at == FIRST_STARTED_AT
    assert run.query_plan is not None
    assert run.query_plan.generated_at_utc == FIRST_STARTED_AT


def test_replay_preserves_original_frozen_intent_and_canonical_identity(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    first_adapter, first_transport = _adapter(_responses(), config)
    first = _run(first_adapter, config, FIRST_STARTED_AT)
    assert first.intent is not None
    assert first.query_plan is not None

    forbidden = ForbiddenTransport()
    replay_adapter = _adapter_from_transport(forbidden, config)
    replay = _run(
        replay_adapter,
        config,
        REPLAY_STARTED_AT,
        frozen_intent=first.intent,
    )

    assert first_transport.calls
    assert replay.status is FirstRoundStatus.SUCCESS
    assert replay.intent == first.intent
    assert replay.intent is not None
    assert canonical_research_intent_bytes(replay.intent) == canonical_research_intent_bytes(
        first.intent
    )
    assert replay.query_plan == first.query_plan
    assert replay.candidates == first.candidates
    assert replay.metrics.transport_requests == 0
    assert replay.metrics.cache_hits == 12
    assert forbidden.calls == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("object_terms", ["changed object"]),
        ("task_terms", ["changed task"]),
        ("method_terms", ["changed method"]),
        ("scope_terms", ["changed scope"]),
        ("exclusions", ["changed exclusion"]),
        ("method_constraint", MethodConstraint.REQUIRED),
        ("accepted_paper_roles", {"method", "background"}),
        ("revision", 2),
    ],
)
def test_any_intent_field_mutation_fails_closed(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    config = _config(tmp_path)
    first_adapter, _ = _adapter(_responses(), config)
    first = _run(first_adapter, config, FIRST_STARTED_AT)
    assert first.intent is not None

    mutated = first.intent.model_copy(update={field: value})
    replay_transport = ForbiddenTransport()
    replay = _run(
        _adapter_from_transport(replay_transport, config),
        config,
        REPLAY_STARTED_AT,
        frozen_intent=mutated,
    )

    assert replay.status is FirstRoundStatus.FAILED
    assert replay.error_code == "INTENT_REPLAY_MISMATCH"
    assert replay_transport.calls == 0


def test_non_utc_frozen_at_is_rejected_before_replay_transport(tmp_path: Path) -> None:
    config = _config(tmp_path)
    first_adapter, _ = _adapter(_responses(), config)
    first = _run(first_adapter, config, FIRST_STARTED_AT)
    assert first.intent is not None

    non_utc = first.intent.model_copy(
        update={"frozen_at": FIRST_STARTED_AT.replace(tzinfo=None)}
    )
    forbidden = ForbiddenTransport()
    replay = _run(
        _adapter_from_transport(forbidden, config),
        config,
        REPLAY_STARTED_AT,
        frozen_intent=non_utc,
    )

    assert replay.status is FirstRoundStatus.FAILED
    assert replay.error_code == "INTENT_REPLAY_MISMATCH"
    assert forbidden.calls == 0


def test_canonical_intent_identity_sorts_roles_and_includes_frozen_at() -> None:
    intent = ResearchIntent(
        object_terms=["scientific literature", "scientific", "literature"],
        task_terms=["discovery"],
        method_terms=["graph-based retrieval", "graph-based", "retrieval"],
        scope_terms=["support", "scientific literature discovery"],
        exclusions=[],
        method_constraint=MethodConstraint.PREFERRED,
        accepted_paper_roles={"method", "background"},
        revision=1,
        frozen_at=FIRST_STARTED_AT,
    )
    reordered = intent.model_copy(update={"accepted_paper_roles": {"background", "method"}})

    first_bytes = canonical_research_intent_bytes(intent)
    second_bytes = canonical_research_intent_bytes(reordered)
    changed_timestamp_bytes = canonical_research_intent_bytes(
        intent.model_copy(update={"frozen_at": REPLAY_STARTED_AT})
    )

    assert first_bytes == second_bytes
    assert first_bytes != changed_timestamp_bytes
    assert b'"frozen_at":"2026-07-28T07:08:40.329494Z"' in first_bytes


def test_frozen_intent_cache_miss_fails_closed_with_forbidden_transport(tmp_path: Path) -> None:
    config = _config(tmp_path)
    seed_adapter, _ = _adapter(_responses(), config)
    first = _run(seed_adapter, config, FIRST_STARTED_AT)
    assert first.intent is not None

    empty_cache = tmp_path / "empty-cache"
    empty_config = config.model_copy(update={"cache_dir": empty_cache})
    forbidden = ForbiddenTransport()
    replay = _run(
        _adapter_from_transport(forbidden, empty_config),
        empty_config,
        REPLAY_STARTED_AT,
        frozen_intent=first.intent,
    )

    assert replay.status is FirstRoundStatus.FAILED
    assert replay.error_code == "ARXIV_UNAVAILABLE"
    assert forbidden.calls == 12
    assert replay.metrics.transport_requests == 12


def _responses() -> list[ArxivResponse]:
    return [_response(("2401.00001", "Graph Retrieval"))] + [
        _response() for _ in range(11)
    ]


def _response(*entries: tuple[str, str]) -> ArxivResponse:
    records = "".join(
        f"""<entry><id>https://arxiv.org/abs/{source_id}</id><title>{title}</title>
        <summary>Recorded source metadata.</summary><published>2024-01-01</published>
        <author><name>Ada Author</name></author></entry>"""
        for source_id, title in entries
    )
    body = f'<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">{records}</feed>'.encode()
    return ArxivResponse(200, body, {})


def _config(tmp_path: Path) -> FirstRoundConfig:
    return FirstRoundConfig(
        cache_dir=tmp_path / "cache",
        mode="recorded",
        cache_namespace="first-round:recorded",
        max_results_per_query=5,
        max_total_candidates=60,
        max_total_attempts=20,
        timeout_seconds=1.0,
        min_request_interval_seconds=0.0,
    )


def _adapter(
    responses: list[ArxivResponse], config: FirstRoundConfig
) -> tuple[ArxivAdapter, RecordedTransport]:
    transport = RecordedTransport(responses)
    return _adapter_from_transport(transport, config), transport


def _adapter_from_transport(
    transport: object,
    config: FirstRoundConfig,
) -> ArxivAdapter:
    return ArxivAdapter(
        ArxivAdapterConfig(
            user_agent="rrc-m1-frozen-intent-tests/1.0",
            timeout_seconds=config.timeout_seconds,
            page_size=config.max_results_per_query,
            min_request_interval_seconds=0.0,
            max_attempts=1,
            max_total_results=config.max_results_per_query,
            max_total_attempts=config.max_total_attempts,
            initial_backoff_seconds=0.0,
            cache_dir=config.cache_dir,
            cache_schema_version=config.adapter_schema_version,
            cache_namespace=config.cache_namespace,
        ),
        transport=transport,  # type: ignore[arg-type]
        monotonic=lambda: 10.0,
        sleeper=lambda _: None,
        utc_now=lambda: FIRST_STARTED_AT,
    )


def _run(
    adapter: ArxivAdapter,
    config: FirstRoundConfig,
    now: datetime,
    *,
    frozen_intent: ResearchIntent | None = None,
):
    return run_first_round(
        QUESTION,
        config=config,
        adapter=adapter,
        now=lambda: now,
        monotonic=lambda: 100.0,
        frozen_intent=frozen_intent,
    )
