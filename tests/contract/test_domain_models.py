from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.models.enums import (
    EvidenceSlot,
    FeedbackAspect,
    ProjectStage,
    QueryBranch,
    Relevance,
    SupportLevel,
)
from app.models.feedback import FeedbackRecord
from app.models.paper import PaperRecord
from app.models.project import ResearchIntent, RetrievalProject
from app.models.query import Query, QueryRevision


def test_closed_enums_reject_unknown_values() -> None:
    with pytest.raises(ValueError):
        ProjectStage("SEARCHING")
    with pytest.raises(ValueError):
        QueryBranch("UNKNOWN_BRANCH")
    with pytest.raises(ValueError):
        FeedbackAspect("WHOLE_PAPER")
    with pytest.raises(ValueError):
        EvidenceSlot("UNSUPPORTED_SLOT")
    with pytest.raises(ValueError):
        SupportLevel("WEAK")


def test_partial_feedback_requires_aspect() -> None:
    with pytest.raises(ValidationError):
        FeedbackRecord(
            project_id="RRC-2026-0001",
            paper_id="arxiv:2401.00001",
            relevance=Relevance.PARTIAL,
            aspects=[],
            raw_text="method is only partially relevant",
        )


@pytest.mark.parametrize("relevance", [Relevance.HIGH, Relevance.IRRELEVANT])
def test_non_partial_feedback_rejects_aspects(relevance: Relevance) -> None:
    with pytest.raises(ValidationError):
        FeedbackRecord(
            project_id="RRC-2026-0001",
            paper_id="arxiv:2401.00001",
            relevance=relevance,
            aspects=[FeedbackAspect.METHOD],
            raw_text="feedback",
        )


def test_visible_paper_requires_source_identity_and_url() -> None:
    with pytest.raises(ValidationError):
        PaperRecord(
            paper_id="arxiv:2401.00001",
            source="arxiv",
            source_id="",
            title="A real title",
            url="",
            language="en",
            user_visible=True,
        )


def test_paper_requires_at_least_one_retrieval_path() -> None:
    with pytest.raises(ValidationError):
        PaperRecord(
            paper_id="arxiv:2401.00001",
            source="arxiv",
            source_id="2401.00001",
            title="A real title",
            url="https://arxiv.org/abs/2401.00001",
            language="en",
            retrieval_paths=[],
        )


def test_round_two_query_requires_revision_source() -> None:
    with pytest.raises(ValidationError):
        Query(
            query_id="Q2-M-01",
            round_number=2,
            branch=QueryBranch.METHOD_DOMAIN,
            breadth="MEDIUM",
            language="en",
            query_text="graph surrogate",
            weight=0.25,
            revision_sources=[],
        )


def test_round_one_query_rejects_revision_source() -> None:
    with pytest.raises(ValidationError):
        Query(
            query_id="Q1-M-01",
            round_number=1,
            branch=QueryBranch.METHOD_DOMAIN,
            breadth="MEDIUM",
            language="en",
            query_text="graph surrogate",
            weight=0.25,
            revision_sources=[QueryRevision(rule="term boost", reason="feedback")],
        )


@pytest.mark.parametrize("count", [14, 21])
def test_round_one_selection_requires_15_to_20_papers(count: int) -> None:
    with pytest.raises(ValidationError):
        RetrievalProject(
            project_id="RRC-2026-0001",
            stage=ProjectStage.WAITING_FOR_FEEDBACK,
            round_number=1,
            original_input="Find transfer learning methods for shielding design.",
            round1_selection_ids=[f"paper-{index}" for index in range(count)],
        )


def test_final_selection_rejects_more_than_ten_papers() -> None:
    with pytest.raises(ValidationError):
        RetrievalProject(
            project_id="RRC-2026-0001",
            stage=ProjectStage.FINALIZED,
            round_number=2,
            original_input="Find transfer learning methods for shielding design.",
            final_selection_ids=[f"paper-{index}" for index in range(11)],
        )


def test_project_rejects_stage_and_round_mismatch() -> None:
    with pytest.raises(ValidationError):
        RetrievalProject(
            project_id="RRC-2026-0001",
            stage=ProjectStage.ROUND2_SEARCHING,
            round_number=1,
            original_input="Find transfer learning methods for shielding design.",
        )


def test_valid_models_export_stable_json_schema() -> None:
    intent = ResearchIntent(
        object_terms=["shielding"],
        task_terms=["design"],
        method_terms=["surrogate model"],
        scope_terms=["nuclear engineering"],
        exclusions=["medical imaging"],
        method_constraint="PREFERRED",
        accepted_paper_roles={"METHOD", "BRIDGE"},
        revision=1,
        frozen_at=datetime.now(UTC),
    )
    project = RetrievalProject(
        project_id="RRC-2026-0001",
        stage=ProjectStage.WAITING_FOR_FEEDBACK,
        round_number=1,
        original_input="Find transfer learning methods for shielding design.",
        current_intent=intent,
        round1_selection_ids=[f"paper-{index}" for index in range(15)],
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )

    assert project.project_id == "RRC-2026-0001"
    assert RetrievalProject.model_json_schema() == RetrievalProject.model_json_schema()
    assert set(RetrievalProject.model_json_schema()["properties"]) == {
        "project_id",
        "stage",
        "round_number",
        "original_input",
        "current_intent",
        "round1_selection_ids",
        "final_selection_ids",
        "created_at",
        "updated_at",
    }
