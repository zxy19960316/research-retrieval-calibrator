from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.models.dedup import (
    DedupCluster,
    DedupDecision,
    DeduplicationResult,
    DedupReason,
    NormalizedPaper,
    SourceIdentity,
)
from app.models.paper import PaperRecord


def _paper() -> PaperRecord:
    return PaperRecord(
        paper_id="paper-1",
        source="arxiv",
        source_id="2401.00001",
        title="Graph-Based Retrieval: A Study",
        abstract="A source-backed abstract.",
        authors=["Ada Author"],
        year=2024,
        url="https://arxiv.org/abs/2401.00001",
        language="en",
        retrieval_paths=["Q1"],
    )


def test_normalized_paper_contract_keeps_record_and_derived_values() -> None:
    normalized = NormalizedPaper(
        record=_paper(),
        canonical_doi=None,
        canonical_arxiv_id="2401.00001",
        normalized_title="graph based retrieval a study",
        title_tokens=("graph", "based", "retrieval", "a", "study"),
        normalized_authors=("ada author",),
    )

    assert normalized.record.paper_id == "paper-1"
    assert normalized.title_tokens == ("graph", "based", "retrieval", "a", "study")


def test_decision_and_cluster_contracts_forbid_unknown_or_invalid_fields() -> None:
    record = _paper()
    decision = DedupDecision(
        left_paper_id="paper-1",
        right_paper_id="paper-2",
        action="auto_merge",
        reason=DedupReason.EXACT_ARXIV_ID,
        title_similarity=None,
        author_overlap=None,
        year_difference=None,
    )
    cluster = DedupCluster(
        cluster_id="arxiv:2401.00001",
        canonical_record=record,
        member_records=[record],
        retrieval_paths=["Q1"],
        source_identities=[
            SourceIdentity(
                source="arxiv",
                source_id="2401.00001",
                url="https://arxiv.org/abs/2401.00001",
            )
        ],
        merge_reasons=[],
    )

    assert decision.action == "auto_merge"
    assert cluster.member_records == [record]
    with pytest.raises(ValidationError):
        DedupDecision.model_validate({**decision.model_dump(), "unexpected": True})
    with pytest.raises(ValidationError):
        DedupCluster.model_validate({**cluster.model_dump(), "retrieval_paths": ["", "Q1"]})
    with pytest.raises(ValidationError, match="Cluster IDs must be unique"):
        DeduplicationResult(clusters=[cluster, cluster], decisions=[])
