"""Unit tests for the independent completed M2-T03 human-result contracts."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.models.enums import EvidenceSlot, SupportLevel
from app.models.evidence_classification import source_text_sha256
from app.models.m2_t03_human_adjudication import ArtifactBinding
from app.models.m2_t03_human_adjudication_result import (
    CompletedHumanAdjudicationResult,
    HumanJudgment,
    derive_adjudication_comparison,
)


def _judgment(
    *,
    paper_id: str = "paper-1",
    verdict: str = "SUPPORTED",
    evidence_slot: EvidenceSlot | None = EvidenceSlot.CURRENT_METHODS,
    support_level: SupportLevel | None = SupportLevel.DIRECT,
    excerpt: str | None = "The method supports retrieval.",
) -> HumanJudgment:
    title = "A retrieval method"
    abstract = "The method supports retrieval."
    return HumanJudgment(
        paper_id=paper_id,
        title=title,
        abstract=abstract,
        source_text_sha256=source_text_sha256(title, abstract),
        verdict=verdict,  # type: ignore[arg-type]
        evidence_slot=evidence_slot,
        support_level=support_level,
        grounded_reason="The abstract states the method directly.",
        supporting_excerpt=excerpt,
        reviewer_id="reviewer-a",
        reviewed_at_utc=datetime(2026, 8, 3, 12, 0, tzinfo=UTC),
    )


def test_supported_judgment_requires_an_exact_source_excerpt() -> None:
    judgment = _judgment()
    assert judgment.reviewed_at_utc.tzinfo is UTC

    with pytest.raises(ValidationError):
        _judgment(excerpt="Not in the source")


def test_rejected_judgment_requires_null_slot_and_support() -> None:
    rejected = _judgment(
        verdict="REJECTED",
        evidence_slot=None,
        support_level=None,
        excerpt=None,
    )
    assert rejected.evidence_slot is None
    assert rejected.support_level is None

    with pytest.raises(ValidationError):
        _judgment(verdict="REJECTED", evidence_slot=EvidenceSlot.CURRENT_METHODS)


def test_completed_result_requires_exactly_33_ordered_judgments() -> None:
    judgments = tuple(
        _judgment(paper_id=f"paper-{index}") for index in range(33)
    )
    result = CompletedHumanAdjudicationResult(
        result_version="m2-t03-human-adjudication-result.v1",
        phase="M2",
        task_id="M2-T03",
        generated_from_commit="a" * 40,
        generated_at_utc=datetime(2026, 8, 3, 12, 0, tzinfo=UTC),
        pending_bundle={"path": "pending.json", "sha256": "b" * 64},
        review_protocol={"path": "protocol.md", "sha256": "c" * 64},
        candidate_count=33,
        paper_id_order=tuple(f"paper-{index}" for index in range(33)),
        judgments=judgments,
        human_review="completed",
        m2_t04="not_started",
        scoring_eligible=False,
    )
    assert len(result.judgments) == 33

    with pytest.raises(ValidationError):
        CompletedHumanAdjudicationResult(
            **{
                **result.model_dump(mode="python"),
                "paper_id_order": tuple(f"paper-{index}" for index in range(32)),
            }
        )


def test_comparison_is_derived_after_human_judgment() -> None:
    judgment = _judgment()
    assert (
        derive_adjudication_comparison(
            judgment, EvidenceSlot.CURRENT_METHODS, SupportLevel.DIRECT
        )
        == "CONFIRM"
    )
    assert (
        derive_adjudication_comparison(
            judgment, EvidenceSlot.PROBLEM_EXISTENCE, SupportLevel.DIRECT
        )
        == "REVISE"
    )


@pytest.mark.parametrize("path", ["/tmp/result.json", "C:/result.json", "\\\\server\\share", "../result.json", "result\\file.json"])
def test_artifact_binding_rejects_absolute_and_non_posix_paths(path: str) -> None:
    with pytest.raises(ValidationError):
        ArtifactBinding(path=path, sha256="a" * 64)
