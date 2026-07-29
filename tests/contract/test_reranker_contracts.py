"""Red-first, synthetic-only contracts for M2-T02 reranking."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

from app.adapters.reranking import RerankerProvider
from app.core.reranking import build_reranker_input, effective_top_k, rerank_candidates
from app.models.dedup import SourceIdentity
from app.models.embedding import FrozenCandidate
from app.models.reranking import (
    ProviderRawScore,
    RerankerModelDescriptor,
    RerankerRunState,
    RerankerTaskError,
)


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
            SourceIdentity(source="arxiv", source_id=source_id, url=f"https://arxiv.org/abs/{source_id}")
        ],
    )


def _descriptor(revision: str = "a" * 40) -> RerankerModelDescriptor:
    return RerankerModelDescriptor(
        provider_name="synthetic",
        model_id="synthetic-cross-encoder",
        model_revision=revision,
        provider_library="tests",
        provider_library_version="1",
        input_format_version="m2-reranker-title-abstract-v1",
        cache_namespace="reranker:synthetic",
    )


class _SyntheticProvider:
    def __init__(
        self,
        descriptor: RerankerModelDescriptor,
        scores: dict[str, float],
        *,
        output: Callable[[Sequence[Any]], list[Any]] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.descriptor = descriptor
        self.scores = scores
        self.output = output
        self.error = error
        self.calls = 0

    def score(self, query: str, inputs: Sequence[Any], *, batch_size: int) -> list[Any]:
        assert query == "Which papers calibrate retrieval?"
        assert batch_size >= 1
        self.calls += 1
        if self.error is not None:
            raise self.error
        if self.output is not None:
            return self.output(inputs)
        return [ProviderRawScore(paper_id=item.paper_id, raw_score=self.scores[item.paper_id]) for item in inputs]


def _run(
    candidates: Sequence[FrozenCandidate], provider: RerankerProvider, cache_dir: Path, *, batch_size: int = 2
) -> Any:
    return rerank_candidates(
        "Which papers calibrate retrieval?",
        candidates,
        provider,
        cache_dir,
        batch_size=batch_size,
        configured_top_k=100,
    )


def test_declares_only_the_required_fail_closed_states() -> None:
    assert {state.value for state in RerankerRunState} == {
        "SCORED",
        "NOT_RUN",
        "PROVIDER_UNAVAILABLE",
        "INVALID_OUTPUT",
    }


def test_effective_top_k_caps_to_available_candidates_without_hard_coding_33() -> None:
    assert effective_top_k(configured_top_k=100, available_count=33) == 33
    assert effective_top_k(configured_top_k=100, available_count=87) == 87
    assert effective_top_k(configured_top_k=50, available_count=87) == 50


def test_reranker_records_preserve_raw_score_descriptor_and_input_sha(tmp_path: Path) -> None:
    candidates = [_candidate(1), _candidate(2)]
    provider = _SyntheticProvider(
        _descriptor(), {candidate.paper_id: float(index + 1) for index, candidate in enumerate(candidates)}
    )

    run = _run(candidates, provider, tmp_path)

    assert run.state is RerankerRunState.SCORED
    assert [(record.paper_id, record.raw_score) for record in run.records] == [
        (candidates[1].paper_id, 2.0),
        (candidates[0].paper_id, 1.0),
    ]
    for record in run.records:
        expected_input = build_reranker_input(next(item for item in candidates if item.paper_id == record.paper_id))
        assert record.descriptor == provider.descriptor
        assert record.input_sha256 == expected_input.input_sha256
        assert record.raw_score != record.normalized_score or record.raw_score in {0.0, 1.0}


@pytest.mark.parametrize(
    "output_factory",
    [
        lambda inputs: [ProviderRawScore(paper_id=inputs[0].paper_id, raw_score=1.0)],
        lambda inputs: [
            ProviderRawScore(paper_id=inputs[0].paper_id, raw_score=1.0),
            ProviderRawScore(paper_id=inputs[0].paper_id, raw_score=0.5),
        ],
        lambda inputs: [
            ProviderRawScore(paper_id="arxiv:unknown", raw_score=1.0),
            ProviderRawScore(paper_id=inputs[1].paper_id, raw_score=0.5),
        ],
    ],
    ids=["too_few", "duplicate_paper_id", "unknown_paper_id"],
)
def test_provider_output_must_match_each_requested_unique_paper_id(
    tmp_path: Path, output_factory: Callable[[Sequence[Any]], list[Any]]
) -> None:
    candidates = [_candidate(1), _candidate(2)]
    provider = _SyntheticProvider(_descriptor(), {}, output=output_factory)

    with pytest.raises(RerankerTaskError, match="INVALID_OUTPUT"):
        _run(candidates, provider, tmp_path)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), "1.0", True, None])
def test_non_finite_or_non_numeric_scores_are_invalid_output(tmp_path: Path, value: object) -> None:
    candidates = [_candidate(1)]
    provider = _SyntheticProvider(
        _descriptor(),
        {},
        output=lambda inputs: [ProviderRawScore.model_construct(paper_id=inputs[0].paper_id, raw_score=value)],
    )

    with pytest.raises(RerankerTaskError, match="INVALID_OUTPUT"):
        _run(candidates, provider, tmp_path)


@pytest.mark.parametrize("batch_size", [1, 2, 8, 33])
def test_batch_sizes_1_2_8_33_preserve_raw_scores_and_final_order(tmp_path: Path, batch_size: int) -> None:
    candidates = [_candidate(index) for index in range(33)]
    scores = {candidate.paper_id: float(1000 - index) for index, candidate in enumerate(candidates)}
    provider = _SyntheticProvider(_descriptor(), scores)

    run = _run(candidates, provider, tmp_path / str(batch_size), batch_size=batch_size)

    assert [(record.paper_id, record.raw_score) for record in run.records] == [
        (candidate.paper_id, scores[candidate.paper_id]) for candidate in candidates
    ]


def test_global_normalization_prevents_batch_local_rank_inversion(tmp_path: Path) -> None:
    candidates = [_candidate(index) for index in range(4)]
    scores = {candidate.paper_id: score for candidate, score in zip(candidates, [100.0, 90.0, 89.0, 0.0])}
    provider = _SyntheticProvider(_descriptor(), scores)

    run = _run(candidates, provider, tmp_path, batch_size=2)

    assert [record.paper_id for record in run.records] == [candidate.paper_id for candidate in candidates]
    assert [record.normalized_score for record in run.records] == pytest.approx([1.0, 0.9, 0.89, 0.0])


def test_provider_failure_never_becomes_zero_score(tmp_path: Path) -> None:
    candidates = [_candidate(1), _candidate(2)]
    provider = _SyntheticProvider(_descriptor(), {}, error=RuntimeError("model load failed"))

    with pytest.raises(RerankerTaskError, match="PROVIDER_UNAVAILABLE"):
        _run(candidates, provider, tmp_path)
    assert provider.calls == 1


def test_empty_abstract_serializes_title_only_without_fabrication() -> None:
    candidate = _candidate(1, abstract=None)

    reranker_input = build_reranker_input(candidate)

    assert reranker_input.text == "title:\nSource-backed title 01"
    assert "abstract:" not in reranker_input.text
    assert reranker_input.input_sha256 == hashlib.sha256(reranker_input.text.encode("utf-8")).hexdigest()


def test_cache_key_changes_when_revision_changes(tmp_path: Path) -> None:
    candidates = [_candidate(1)]
    scores = {candidates[0].paper_id: 0.5}
    provider_v1 = _SyntheticProvider(_descriptor("a" * 40), scores)
    provider_v2 = _SyntheticProvider(_descriptor("b" * 40), scores)

    _run(candidates, provider_v1, tmp_path)
    _run(candidates, provider_v1, tmp_path)
    _run(candidates, provider_v2, tmp_path)

    assert provider_v1.calls == 1
    assert provider_v2.calls == 1


def test_equal_scores_use_stable_paper_id_tie_break(tmp_path: Path) -> None:
    candidates = [_candidate(7), _candidate(2), _candidate(4)]
    provider = _SyntheticProvider(_descriptor(), {candidate.paper_id: 3.0 for candidate in candidates})

    run = _run(candidates, provider, tmp_path)

    assert [record.paper_id for record in run.records] == sorted(candidate.paper_id for candidate in candidates)
    assert all(math.isfinite(record.normalized_score) for record in run.records)


def test_input_order_permutations_produce_the_same_final_order(tmp_path: Path) -> None:
    candidates = [_candidate(1), _candidate(2), _candidate(3)]
    scores = {candidates[0].paper_id: 1.0, candidates[1].paper_id: 2.0, candidates[2].paper_id: 1.0}
    forward = _run(candidates, _SyntheticProvider(_descriptor(), scores), tmp_path / "forward")
    reversed_run = _run(list(reversed(candidates)), _SyntheticProvider(_descriptor(), scores), tmp_path / "reversed")

    assert [record.paper_id for record in forward.records] == [record.paper_id for record in reversed_run.records]
