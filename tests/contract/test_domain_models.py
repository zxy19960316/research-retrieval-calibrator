import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

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


@pytest.mark.parametrize(
    ("model", "payload", "field"),
    [
        (
            RetrievalProject,
            {
                "project_id": "RRC-2026-0001",
                "stage": "SEARCHING",
                "round_number": 0,
                "original_input": "Find shielding design methods.",
            },
            "stage",
        ),
        (
            FeedbackRecord,
            {
                "project_id": "RRC-2026-0001",
                "paper_id": "arxiv:2401.00001",
                "relevance": "UNKNOWN_RELEVANCE",
                "raw_text": "feedback",
            },
            "relevance",
        ),
        (
            Query,
            {
                "query_id": "Q1-M-01",
                "round_number": 1,
                "branch": "UNKNOWN_BRANCH",
                "breadth": "MEDIUM",
                "language": "en",
                "query_text": "graph surrogate",
                "weight": 0.25,
            },
            "branch",
        ),
    ],
)
def test_pydantic_models_reject_unknown_enum_values(
    model: type[RetrievalProject] | type[FeedbackRecord] | type[Query],
    payload: dict[str, object],
    field: str,
) -> None:
    with pytest.raises(ValidationError) as error:
        model.model_validate(payload)

    assert error.value.errors()[0]["loc"] == (field,)
    assert error.value.errors()[0]["type"] == "enum"


@pytest.mark.parametrize("enum_type", [EvidenceSlot, SupportLevel])
def test_type_adapter_rejects_unknown_public_contract_enum_values(
    enum_type: type[EvidenceSlot] | type[SupportLevel],
) -> None:
    with pytest.raises(ValidationError) as error:
        TypeAdapter(enum_type).validate_python("NOT_A_VALID_VALUE")

    assert error.value.errors()[0]["type"] == "enum"


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
    with pytest.raises(ValidationError) as error:
        PaperRecord(
            paper_id="arxiv:2401.00001",
            source="arxiv",
            source_id="",
            title="A real title",
            url="",
            language="en",
            retrieval_paths=["DIRECT_INTERSECTION"],
            user_visible=True,
        )

    assert error.value.errors()[0]["msg"] == (
        "Value error, User-visible papers require source_id and HTTP(S) URL"
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


def test_public_m0_t01_contracts_export_stable_json_schemas() -> None:
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
    snapshot_path = Path(__file__).parent / "fixtures" / "m0_t01_public_contracts.schema.json"
    expected_schema = json.loads(snapshot_path.read_text(encoding="utf-8"))
    public_models = {
        "FeedbackRecord": FeedbackRecord,
        "PaperRecord": PaperRecord,
        "Query": Query,
        "QueryRevision": QueryRevision,
        "ResearchIntent": ResearchIntent,
        "RetrievalProject": RetrievalProject,
    }
    normalized_actual = json.loads(
        json.dumps(
            {name: model.model_json_schema() for name, model in sorted(public_models.items())},
            sort_keys=True,
        )
    )
    normalized_expected = json.loads(json.dumps(expected_schema, sort_keys=True))
    assert normalized_actual == normalized_expected
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
