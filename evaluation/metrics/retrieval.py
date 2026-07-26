"""Frozen, hand-verifiable retrieval evaluation metrics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import log2

from app.models.enums import EvidenceSlot, Relevance


@dataclass(frozen=True, slots=True)
class JudgedPaper:
    """The minimal paper judgement needed before M2's RankedPaper exists."""

    paper_id: str
    relevance: Relevance

    def __post_init__(self) -> None:
        if not self.paper_id.strip():
            raise ValueError("JudgedPaper requires a non-empty paper_id")


_GRADES: dict[Relevance, int] = {
    Relevance.HIGH: 2,
    Relevance.PARTIAL: 1,
    Relevance.IRRELEVANT: 0,
}


def _require_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be greater than 0")


def _as_relevance(value: Relevance | str) -> Relevance:
    try:
        return Relevance(value)
    except ValueError as error:
        raise ValueError(f"Unknown relevance: {value!r}") from error


def _relevance_grade(value: Relevance | str) -> int:
    return _GRADES[_as_relevance(value)]


def precision_at_k(
    relevances: Sequence[Relevance | str], k: int, *, inclusive: bool = False
) -> float:
    """Return strict or inclusive precision with a denominator fixed at ``k``."""

    _require_k(k)
    relevant = {Relevance.HIGH, Relevance.PARTIAL} if inclusive else {Relevance.HIGH}
    return sum(_as_relevance(value) in relevant for value in relevances[:k]) / k


def _dcg(relevances: Sequence[Relevance | str], k: int) -> float:
    total = 0.0
    for rank, value in enumerate(relevances[:k], start=1):
        total += (2**_GRADES[_as_relevance(value)] - 1) / log2(rank + 1)
    return total


def ndcg_at_k(
    relevances: Sequence[Relevance | str],
    k: int,
    *,
    ideal_relevances: Sequence[Relevance | str] | None = None,
) -> float:
    """Return NDCG using an explicit ideal sequence when one is supplied."""

    _require_k(k)
    dcg = _dcg(relevances, k)
    if ideal_relevances is None:
        default_ideal: list[Relevance] = [_as_relevance(value) for value in relevances[:k]]
        ideal_relevances = sorted(
            default_ideal,
            key=_relevance_grade,
            reverse=True,
        )
    idcg = _dcg(ideal_relevances, k)
    return 0.0 if idcg == 0 else dcg / idcg


def evidence_coverage(slots: Sequence[EvidenceSlot | str]) -> float:
    """Return unique fixed EvidenceSlot coverage out of all five slots."""

    normalized: set[EvidenceSlot] = set()
    for slot in slots:
        try:
            normalized.add(EvidenceSlot(slot))
        except ValueError as error:
            raise ValueError(f"Unknown evidence slot: {slot!r}") from error
    return len(normalized) / len(EvidenceSlot)


def negative_suppression(
    round_one: Sequence[Relevance | str], round_two: Sequence[Relevance | str], k: int
) -> float:
    """Return round-one minus round-two negative exposure; regressions stay negative."""

    _require_k(k)
    first_exposure = sum(_as_relevance(value) is Relevance.IRRELEVANT for value in round_one[:k]) / k
    second_exposure = sum(_as_relevance(value) is Relevance.IRRELEVANT for value in round_two[:k]) / k
    return first_exposure - second_exposure


def _validate_unique_papers(papers: Sequence[JudgedPaper]) -> None:
    paper_ids = [paper.paper_id for paper in papers]
    if any(not paper_id.strip() for paper_id in paper_ids):
        raise ValueError("JudgedPaper requires a non-empty paper_id")
    if len(paper_ids) != len(set(paper_ids)):
        raise ValueError("duplicate paper_id in evaluation input")


def new_useful_papers(round_one: Sequence[JudgedPaper], round_two: Sequence[JudgedPaper], k: int) -> int:
    """Count unique relevant round-two papers not present in round one's top 20."""

    _require_k(k)
    _validate_unique_papers(round_one)
    _validate_unique_papers(round_two)
    first_round_ids = {paper.paper_id for paper in round_one[:20]}
    useful = {Relevance.HIGH, Relevance.PARTIAL}
    return sum(
        paper.paper_id not in first_round_ids and paper.relevance in useful for paper in round_two[:k]
    )


def metadata_hallucination_rate(source_confirmed: Sequence[bool]) -> float:
    """Return unconfirmed user-visible records divided by visible records."""

    if any(type(confirmed) is not bool for confirmed in source_confirmed):
        raise TypeError("source confirmation values must be bool")
    if not source_confirmed:
        return 0.0
    return sum(not confirmed for confirmed in source_confirmed) / len(source_confirmed)
