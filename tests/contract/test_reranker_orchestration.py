"""Red-first orchestration contracts for M2-T02 reranking."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from app.core.reranking import build_reranker_input, effective_top_k, rerank_candidates
from app.models.dedup import SourceIdentity
from app.models.embedding import FrozenCandidate
from app.models.reranking import (
    ProviderRawScore,
    RerankerModelDescriptor,
    RerankerRunState,
    RerankerTaskError,
)

if TYPE_CHECKING:
    from app.adapters.reranking import RerankerProvider

_QUERY = "Which papers calibrate retrieval?"


def _candidate(index: int, *, abstract: str | None = "Source-backed abstract.") -> FrozenCandidate:
    source_id = f"2401.{index:05d}"
    return FrozenCandidate(
        paper_id=f"arxiv:{source_id}",
        source="arxiv",
        source_id=source_id,
        title=f"Source-backed title {index:02d}",
        abstract=abstract,
        authors=["Ada Author"],
        year=2024,
        doi=None,
        url=f"https://arxiv.org/abs/{source_id}",
        language="en",
        categories=["cs.IR"],
        retrieval_paths=["Q1"],
        cluster_id=f"cluster:{source_id}",
        member_source_identities=[
            SourceIdentity(
                source="arxiv",
                source_id=source_id,
                url=f"https://arxiv.org/abs/{source_id}",
            )
        ],
    )


def _descriptor(
    revision: str = "a" * 40,
    *,
    model_id: str = "synthetic-cross-encoder",
    provider_library: str = "tests",
    provider_library_version: str = "1",
    input_format_version: str = "m2-reranker-title-abstract-v1",
    cache_namespace: str = "reranker:synthetic",
) -> RerankerModelDescriptor:
    return RerankerModelDescriptor(
        provider_name="synthetic",
        model_id=model_id,
        model_revision=revision,
        provider_library=provider_library,
        provider_library_version=provider_library_version,
        input_format_version=input_format_version,
        cache_namespace=cache_namespace,
    )


class _SyntheticProvider:
    def __init__(
        self,
        descriptor: RerankerModelDescriptor,
        scores: dict[str, float],
        *,
        output: Callable[[Sequence[Any], int], object] | None = None,
        error: Exception | None = None,
        fail_on_call: int | None = None,
    ) -> None:
        self.descriptor = descriptor
        self.scores = scores
        self.output = output
        self.error = error
        self.fail_on_call = fail_on_call
        self.calls = 0
        self.call_paper_ids: list[list[str]] = []
        self.call_batch_sizes: list[int] = []
        self.call_queries: list[str] = []

    def score(self, query: str, inputs: Sequence[Any], *, batch_size: int) -> object:
        assert isinstance(query, str)
        assert query.strip()
        assert batch_size >= 1
        self.calls += 1
        self.call_queries.append(query)
        self.call_paper_ids.append([item.paper_id for item in inputs])
        self.call_batch_sizes.append(len(inputs))
        if self.error is not None or self.fail_on_call == self.calls:
            raise self.error or RuntimeError("synthetic provider failure")
        if self.output is not None:
            return self.output(inputs, self.calls)
        return [
            ProviderRawScore(paper_id=item.paper_id, raw_score=self.scores[item.paper_id])
            for item in inputs
        ]


def _run(
    candidates: Sequence[FrozenCandidate],
    provider: RerankerProvider,
    cache_dir: Path,
    *,
    batch_size: int = 2,
    configured_top_k: int = 100,
    query: str = _QUERY,
) -> Any:
    return rerank_candidates(
        query,
        candidates,
        provider,
        cache_dir,
        batch_size=batch_size,
        configured_top_k=configured_top_k,
    )


def _assert_error_code(action: Callable[[], object], expected_code: str) -> None:
    with pytest.raises(RerankerTaskError) as captured:
        action()
    assert captured.value.code == expected_code


def _cache_payloads(cache_dir: Path) -> dict[str, dict[str, object]]:
    return {
        path.name: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(cache_dir.glob("*.json"))
    }


@pytest.mark.parametrize(("configured_top_k", "available_count"), [(49, 0), (101, 0), (50, -1)])
def test_effective_top_k_rejects_invalid_config_or_available_count(
    configured_top_k: int, available_count: int
) -> None:
    _assert_error_code(
        lambda: effective_top_k(
            configured_top_k=configured_top_k,
            available_count=available_count,
        ),
        "INVALID_INPUT",
    )


def test_effective_top_k_caps_to_available_candidates_without_hard_coding_33() -> None:
    assert effective_top_k(configured_top_k=100, available_count=33) == 33
    assert effective_top_k(configured_top_k=100, available_count=87) == 87
    assert effective_top_k(configured_top_k=50, available_count=87) == 50


@pytest.mark.parametrize("batch_size", [0, -1])
def test_rerank_candidates_rejects_non_positive_batch_size(tmp_path: Path, batch_size: int) -> None:
    candidate = _candidate(1)
    provider = _SyntheticProvider(_descriptor(), {candidate.paper_id: 1.0})

    _assert_error_code(lambda: _run([candidate], provider, tmp_path, batch_size=batch_size), "INVALID_INPUT")


def test_rerank_candidates_rejects_blank_query_and_duplicate_paper_id(tmp_path: Path) -> None:
    candidate = _candidate(1)
    provider = _SyntheticProvider(_descriptor(), {candidate.paper_id: 1.0})

    _assert_error_code(lambda: _run([candidate], provider, tmp_path, query="  "), "INVALID_INPUT")
    _assert_error_code(lambda: _run([candidate, candidate], provider, tmp_path), "INVALID_INPUT")
    assert provider.calls == 0


def test_empty_candidate_set_returns_not_run_without_provider_or_cache(tmp_path: Path) -> None:
    provider = _SyntheticProvider(_descriptor(), {})

    run = _run([], provider, tmp_path)

    assert run.state is RerankerRunState.NOT_RUN
    assert run.records == []
    assert provider.calls == 0
    assert _cache_payloads(tmp_path) == {}


def test_reranker_records_preserve_raw_score_descriptor_and_input_sha(tmp_path: Path) -> None:
    candidates = [_candidate(1), _candidate(2)]
    provider = _SyntheticProvider(
        _descriptor(),
        {candidate.paper_id: float(index + 1) for index, candidate in enumerate(candidates)},
    )

    run = _run(candidates, provider, tmp_path)

    assert run.state is RerankerRunState.SCORED
    assert [(record.paper_id, record.raw_score) for record in run.records] == [
        (candidates[1].paper_id, 2.0),
        (candidates[0].paper_id, 1.0),
    ]
    for record in run.records:
        candidate = next(item for item in candidates if item.paper_id == record.paper_id)
        assert record.descriptor == provider.descriptor
        assert record.input_sha256 == build_reranker_input(candidate).input_sha256


@pytest.mark.parametrize("batch_size, expected_call_count", [(1, 33), (2, 17), (8, 5), (33, 1)])
def test_core_chunks_all_selected_misses_before_calling_provider(
    tmp_path: Path, batch_size: int, expected_call_count: int
) -> None:
    candidates = [_candidate(index) for index in range(33)]
    scores = {candidate.paper_id: float(1000 - index) for index, candidate in enumerate(candidates)}
    provider = _SyntheticProvider(_descriptor(), scores)

    run = _run(candidates, provider, tmp_path, batch_size=batch_size)

    assert run.state is RerankerRunState.SCORED
    assert provider.calls == expected_call_count
    assert provider.call_batch_sizes == [
        min(batch_size, 33 - start) for start in range(0, 33, batch_size)
    ]
    assert all(size <= batch_size for size in provider.call_batch_sizes)
    assert [paper_id for chunk in provider.call_paper_ids for paper_id in chunk] == [
        candidate.paper_id for candidate in candidates
    ]
    assert len({paper_id for chunk in provider.call_paper_ids for paper_id in chunk}) == 33


def test_top_k_limits_provider_input_records_and_new_cache_entries(tmp_path: Path) -> None:
    candidates = [_candidate(index) for index in range(87)]
    scores = {candidate.paper_id: float(1000 - index) for index, candidate in enumerate(candidates)}
    provider = _SyntheticProvider(_descriptor(), scores)

    run = _run(candidates, provider, tmp_path, batch_size=8, configured_top_k=50)

    assert len(run.records) == 50
    assert [paper_id for chunk in provider.call_paper_ids for paper_id in chunk] == [
        candidate.paper_id for candidate in candidates[:50]
    ]
    assert {record.paper_id for record in run.records} == {candidate.paper_id for candidate in candidates[:50]}
    assert len(_cache_payloads(tmp_path)) == 50
    assert not {candidate.paper_id for candidate in candidates[50:]} & {
        record.paper_id for record in run.records
    }


def test_thirty_three_candidates_with_configured_100_processes_every_candidate(tmp_path: Path) -> None:
    candidates = [_candidate(index) for index in range(33)]
    provider = _SyntheticProvider(
        _descriptor(),
        {candidate.paper_id: float(index) for index, candidate in enumerate(candidates)},
    )

    run = _run(candidates, provider, tmp_path, batch_size=8, configured_top_k=100)

    assert len(run.records) == 33
    assert [paper_id for chunk in provider.call_paper_ids for paper_id in chunk] == [
        candidate.paper_id for candidate in candidates
    ]


def test_provider_output_is_mapped_by_paper_id_even_when_a_chunk_is_reversed(tmp_path: Path) -> None:
    candidates = [_candidate(1), _candidate(2)]
    scores = {candidates[0].paper_id: 1.0, candidates[1].paper_id: 2.0}
    provider = _SyntheticProvider(
        _descriptor(),
        scores,
        output=lambda inputs, _call: [
            ProviderRawScore(paper_id=item.paper_id, raw_score=scores[item.paper_id])
            for item in reversed(inputs)
        ],
    )

    run = _run(candidates, provider, tmp_path)

    assert [(record.paper_id, record.raw_score) for record in run.records] == [
        (candidates[1].paper_id, 2.0),
        (candidates[0].paper_id, 1.0),
    ]


@pytest.mark.parametrize(
    "output",
    [
        lambda inputs, _call: [ProviderRawScore(paper_id=inputs[0].paper_id, raw_score=1.0)],
        lambda inputs, _call: [
            ProviderRawScore(paper_id=inputs[0].paper_id, raw_score=1.0),
            ProviderRawScore(paper_id=inputs[1].paper_id, raw_score=0.5),
            ProviderRawScore(paper_id="arxiv:extra", raw_score=0.25),
        ],
        lambda inputs, _call: [
            ProviderRawScore(paper_id=inputs[0].paper_id, raw_score=1.0),
            ProviderRawScore(paper_id=inputs[0].paper_id, raw_score=0.5),
        ],
        lambda inputs, _call: [
            ProviderRawScore(paper_id="arxiv:unknown", raw_score=1.0),
            ProviderRawScore(paper_id=inputs[1].paper_id, raw_score=0.5),
        ],
        lambda _inputs, _call: object(),
    ],
    ids=["too_few", "extra", "duplicate", "unknown", "non_sequence"],
)
def test_invalid_provider_output_fails_closed_without_new_cache(
    tmp_path: Path, output: Callable[[Sequence[Any], int], object]
) -> None:
    candidates = [_candidate(1), _candidate(2)]
    provider = _SyntheticProvider(_descriptor(), {}, output=output)

    _assert_error_code(lambda: _run(candidates, provider, tmp_path), "INVALID_OUTPUT")

    assert _cache_payloads(tmp_path) == {}
    assert list(tmp_path.rglob("*.tmp")) == []


def test_previous_chunk_ids_are_invalid_and_leave_no_partial_cache(tmp_path: Path) -> None:
    candidates = [_candidate(index) for index in range(4)]
    scores = {candidate.paper_id: float(index) for index, candidate in enumerate(candidates)}

    def output(inputs: Sequence[Any], call: int) -> object:
        if call == 1:
            return [ProviderRawScore(paper_id=item.paper_id, raw_score=scores[item.paper_id]) for item in inputs]
        return [
            ProviderRawScore(paper_id=candidates[0].paper_id, raw_score=1.0),
            ProviderRawScore(paper_id=candidates[1].paper_id, raw_score=0.0),
        ]

    provider = _SyntheticProvider(_descriptor(), scores, output=output)

    _assert_error_code(lambda: _run(candidates, provider, tmp_path, batch_size=2), "INVALID_OUTPUT")

    assert provider.calls == 2
    assert _cache_payloads(tmp_path) == {}
    assert list(tmp_path.rglob("*.tmp")) == []


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), "1.0", True, None])
def test_non_finite_or_non_numeric_scores_are_invalid_output(tmp_path: Path, value: object) -> None:
    candidate = _candidate(1)
    provider = _SyntheticProvider(
        _descriptor(),
        {},
        output=lambda inputs, _call: [
            ProviderRawScore.model_construct(paper_id=inputs[0].paper_id, raw_score=value)
        ],
    )

    _assert_error_code(lambda: _run([candidate], provider, tmp_path), "INVALID_OUTPUT")

    assert _cache_payloads(tmp_path) == {}


def test_provider_failure_never_becomes_zero_score_or_partial_run(tmp_path: Path) -> None:
    candidates = [_candidate(1), _candidate(2)]
    provider = _SyntheticProvider(_descriptor(), {}, error=RuntimeError("model load failed"))

    _assert_error_code(lambda: _run(candidates, provider, tmp_path), "PROVIDER_UNAVAILABLE")

    assert provider.calls == 1
    assert _cache_payloads(tmp_path) == {}


def test_empty_abstract_serializes_title_only_without_fabrication() -> None:
    candidate = _candidate(1, abstract=None)

    reranker_input = build_reranker_input(candidate)

    assert reranker_input.text == "title:\nSource-backed title 01"
    assert "abstract:" not in reranker_input.text
    assert reranker_input.input_sha256 == hashlib.sha256(reranker_input.text.encode("utf-8")).hexdigest()


def test_cache_entry_contains_only_raw_provenance_and_never_normalized_score(tmp_path: Path) -> None:
    candidate = _candidate(1)
    provider = _SyntheticProvider(_descriptor(), {candidate.paper_id: 7.0})

    _run([candidate], provider, tmp_path)

    payloads = _cache_payloads(tmp_path)
    assert len(payloads) == 1
    assert set(next(iter(payloads.values()))) == {
        "raw_score",
        "paper_id",
        "query_sha256",
        "input_sha256",
        "descriptor",
        "cache_key",
    }


def test_complete_cache_identity_and_batch_order_invariance(tmp_path: Path) -> None:
    first, second = _candidate(1), _candidate(2)
    scores = {first.paper_id: 1.0, second.paper_id: 2.0}
    provider = _SyntheticProvider(_descriptor(), scores)

    _run([first, second], provider, tmp_path, batch_size=1)
    _run([second, first], provider, tmp_path, batch_size=33)
    assert provider.calls == 2

    changed_query = "A changed query"
    _run([first], provider, tmp_path, query=changed_query)
    assert provider.calls == 3
    assert provider.call_queries[-1] == changed_query

    changed_title = first.model_copy(update={"title": "A changed title"})
    _run([changed_title], provider, tmp_path)
    assert provider.calls == 4

    changed_abstract = first.model_copy(update={"abstract": "A changed abstract"})
    _run([changed_abstract], provider, tmp_path)
    assert provider.calls == 5

    for descriptor in (
        _descriptor("b" * 40),
        _descriptor(model_id="synthetic-cross-encoder-v2"),
        _descriptor(provider_library="other-tests"),
        _descriptor(provider_library_version="2"),
        _descriptor(input_format_version="m2-reranker-title-abstract-v2"),
        _descriptor(cache_namespace="reranker:synthetic-v2"),
    ):
        changed_descriptor_provider = _SyntheticProvider(descriptor, scores)
        _run([first], changed_descriptor_provider, tmp_path)
        assert changed_descriptor_provider.calls == 1


@pytest.mark.parametrize(
    ("raw_scores", "expected_normalized"),
    [
        ([100.0, 90.0, 89.0, 0.0], [1.0, 0.9, 0.89, 0.0]),
        ([-4.0, -2.0, 0.0], [0.0, 0.5, 1.0]),
        ([3.0, 3.0, 3.0], [1.0, 1.0, 1.0]),
        ([7.0], [1.0]),
    ],
)
def test_global_min_max_normalization_uses_the_complete_raw_score_set(
    tmp_path: Path, raw_scores: list[float], expected_normalized: list[float]
) -> None:
    candidates = [_candidate(index) for index in range(len(raw_scores))]
    provider = _SyntheticProvider(
        _descriptor(),
        {candidate.paper_id: score for candidate, score in zip(candidates, raw_scores, strict=True)},
    )

    run = _run(candidates, provider, tmp_path, batch_size=2)

    normalized_by_paper_id = {record.paper_id: record.normalized_score for record in run.records}
    assert [normalized_by_paper_id[candidate.paper_id] for candidate in candidates] == pytest.approx(
        expected_normalized
    )


def test_normalized_scores_are_identical_across_batch_sizes_and_mixed_cache(tmp_path: Path) -> None:
    candidates = [_candidate(index) for index in range(4)]
    scores = {candidate.paper_id: score for candidate, score in zip(candidates, [100.0, 90.0, 89.0, 0.0])}
    normalized_by_batch_size: dict[int, dict[str, float]] = {}
    for batch_size in (1, 2, 8, 33):
        run = _run(
            candidates,
            _SyntheticProvider(_descriptor(), scores),
            tmp_path / f"batch-{batch_size}",
            batch_size=batch_size,
        )
        normalized_by_batch_size[batch_size] = {
            record.paper_id: record.normalized_score for record in run.records
        }
    assert len({tuple(sorted(item.items())) for item in normalized_by_batch_size.values()}) == 1

    first_provider = _SyntheticProvider(_descriptor(), scores)
    _run([candidates[0]], first_provider, tmp_path / "mixed", batch_size=1)
    mixed_provider = _SyntheticProvider(_descriptor(), scores)
    mixed = _run(candidates, mixed_provider, tmp_path / "mixed", batch_size=2)
    assert mixed_provider.call_paper_ids == [
        [candidates[1].paper_id, candidates[2].paper_id],
        [candidates[3].paper_id],
    ]
    assert [record.normalized_score for record in mixed.records] == pytest.approx([1.0, 0.9, 0.89, 0.0])


def test_final_order_uses_raw_score_descending_then_paper_id_ascending(tmp_path: Path) -> None:
    candidates = [_candidate(7), _candidate(2), _candidate(4)]
    provider = _SyntheticProvider(_descriptor(), {candidate.paper_id: 3.0 for candidate in candidates})

    run = _run(candidates, provider, tmp_path)

    assert [record.paper_id for record in run.records] == sorted(candidate.paper_id for candidate in candidates)
    assert all(math.isfinite(record.normalized_score) for record in run.records)


def test_input_order_only_selects_top_k_and_never_changes_final_order(tmp_path: Path) -> None:
    candidates = [_candidate(1), _candidate(2), _candidate(3)]
    scores = {candidates[0].paper_id: 1.0, candidates[1].paper_id: 2.0, candidates[2].paper_id: 1.0}
    forward = _run(candidates, _SyntheticProvider(_descriptor(), scores), tmp_path / "forward")
    reversed_run = _run(
        list(reversed(candidates)),
        _SyntheticProvider(_descriptor(), scores),
        tmp_path / "reversed",
    )

    assert [record.paper_id for record in forward.records] == [record.paper_id for record in reversed_run.records]


def test_provider_failure_after_two_chunks_is_transactional_for_new_cache_entries(tmp_path: Path) -> None:
    candidates = [_candidate(index) for index in range(33)]
    provider = _SyntheticProvider(
        _descriptor(),
        {candidate.paper_id: float(index) for index, candidate in enumerate(candidates)},
        fail_on_call=3,
    )

    _assert_error_code(lambda: _run(candidates, provider, tmp_path, batch_size=8), "PROVIDER_UNAVAILABLE")

    assert provider.call_batch_sizes == [8, 8, 8]
    assert _cache_payloads(tmp_path) == {}
    assert list(tmp_path.rglob("*.tmp")) == []


def test_existing_cache_survives_failed_miss_run_without_new_partial_entries(tmp_path: Path) -> None:
    candidates = [_candidate(index) for index in range(33)]
    scores = {candidate.paper_id: float(index) for index, candidate in enumerate(candidates)}
    _run([candidates[0]], _SyntheticProvider(_descriptor(), scores), tmp_path)
    before = {path.name: path.read_bytes() for path in tmp_path.glob("*.json")}
    failing_provider = _SyntheticProvider(_descriptor(), scores, fail_on_call=3)

    _assert_error_code(
        lambda: _run(candidates, failing_provider, tmp_path, batch_size=8),
        "PROVIDER_UNAVAILABLE",
    )

    after = {path.name: path.read_bytes() for path in tmp_path.glob("*.json")}
    assert failing_provider.call_batch_sizes == [8, 8, 8]
    assert after == before
    assert list(tmp_path.rglob("*.tmp")) == []
