from __future__ import annotations

from math import log2

import pytest

from app.models.enums import EvidenceSlot, Relevance
from evaluation.metrics.retrieval import (
    JudgedPaper,
    evidence_coverage,
    metadata_hallucination_rate,
    ndcg_at_k,
    negative_suppression,
    new_useful_papers,
    precision_at_k,
)


def test_precision_variants_use_fixed_denominator() -> None:
    labels = [
        Relevance.HIGH,
        Relevance.HIGH,
        Relevance.HIGH,
        Relevance.PARTIAL,
        Relevance.PARTIAL,
    ] + [Relevance.IRRELEVANT] * 5

    assert precision_at_k(labels, 10) == 0.3
    assert precision_at_k(labels, 10, inclusive=True) == 0.5


def test_precision_at_k_treats_missing_positions_as_irrelevant() -> None:
    assert precision_at_k([Relevance.HIGH, Relevance.PARTIAL], 10) == 0.1
    assert precision_at_k([Relevance.HIGH, Relevance.PARTIAL], 10, inclusive=True) == 0.2


def test_ndcg_perfect_ranking_is_one() -> None:
    assert ndcg_at_k([Relevance.HIGH, Relevance.PARTIAL, Relevance.IRRELEVANT], 3) == 1.0


def test_ndcg_nonideal_ranking_uses_explicit_ideal_relevances() -> None:
    observed = [Relevance.PARTIAL, Relevance.HIGH, Relevance.IRRELEVANT]
    ideal = [Relevance.HIGH, Relevance.PARTIAL, Relevance.IRRELEVANT]
    expected = (1 / log2(2) + 3 / log2(3)) / (3 / log2(2) + 1 / log2(3))

    assert ndcg_at_k(observed, 3, ideal_relevances=ideal) == pytest.approx(expected)


def test_ndcg_returns_zero_when_idcg_is_zero() -> None:
    assert ndcg_at_k([Relevance.IRRELEVANT], 10, ideal_relevances=[Relevance.IRRELEVANT]) == 0.0


def test_evidence_coverage_uses_five_fixed_slots_and_deduplicates() -> None:
    assert evidence_coverage(
        [
            EvidenceSlot.PROBLEM_EXISTENCE,
            EvidenceSlot.CURRENT_METHODS,
            EvidenceSlot.IMPLEMENTATION_PATH,
            EvidenceSlot.CURRENT_METHODS,
        ]
    ) == 0.6


def test_evidence_coverage_rejects_unknown_slot() -> None:
    with pytest.raises(ValueError, match="Unknown evidence slot"):
        evidence_coverage(["NOT_A_SLOT"])


def test_negative_suppression_reports_improvement() -> None:
    first = [Relevance.IRRELEVANT] * 4 + [Relevance.HIGH] * 6
    second = [Relevance.IRRELEVANT] + [Relevance.HIGH] * 9

    assert negative_suppression(first, second, 10) == pytest.approx(0.3)


def test_negative_suppression_allows_negative_regression() -> None:
    first = [Relevance.IRRELEVANT] + [Relevance.HIGH] * 9
    second = [Relevance.IRRELEVANT] * 4 + [Relevance.HIGH] * 6

    assert negative_suppression(first, second, 10) == pytest.approx(-0.3)


def test_new_useful_papers_counts_unique_new_relevant_papers() -> None:
    first = [JudgedPaper("old-high", Relevance.HIGH)]
    second = [
        JudgedPaper("old-high", Relevance.HIGH),
        JudgedPaper("new-high", Relevance.HIGH),
        JudgedPaper("new-partial", Relevance.PARTIAL),
        JudgedPaper("new-irrelevant", Relevance.IRRELEVANT),
    ]

    assert new_useful_papers(first, second, 10) == 2


def test_new_useful_papers_rejects_duplicate_or_empty_paper_ids() -> None:
    with pytest.raises(ValueError, match="duplicate paper_id"):
        new_useful_papers(
            [],
            [JudgedPaper("duplicate", Relevance.HIGH), JudgedPaper("duplicate", Relevance.PARTIAL)],
            10,
        )
    with pytest.raises(ValueError, match="non-empty paper_id"):
        new_useful_papers([], [JudgedPaper(" ", Relevance.HIGH)], 10)


def test_metadata_hallucination_rate_uses_only_source_confirmation_booleans() -> None:
    assert metadata_hallucination_rate([True, True, False, True]) == 0.25
    with pytest.raises(TypeError, match="bool"):
        metadata_hallucination_rate(["confirmed"])


def test_metadata_hallucination_rate_is_zero_without_visible_records() -> None:
    assert metadata_hallucination_rate([]) == 0.0


@pytest.mark.parametrize(
    "metric_call",
    [
        lambda: precision_at_k([], 0),
        lambda: ndcg_at_k([], 0),
        lambda: negative_suppression([], [], 0),
        lambda: new_useful_papers([], [], 0),
    ],
)
def test_metrics_reject_nonpositive_k(metric_call: object) -> None:
    with pytest.raises(ValueError, match="k must be greater than 0"):
        metric_call()  # type: ignore[operator]
