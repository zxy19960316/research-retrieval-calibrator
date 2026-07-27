from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.adapters.llm import validate_llm_candidate
from app.core.intent import freeze_research_intent
from app.core.query_planner import build_query_plan
from app.models.enums import MethodConstraint, QueryBranch, QueryBreadth
from app.models.planning import (
    ClarificationQuestion,
    IntentDraft,
    IntentField,
    IntentGap,
    PlanningError,
    QueryPlan,
    RetrievalIntentField,
    TermEvidence,
    TermSource,
)
from app.models.query import Query


def _draft_payload() -> dict[str, object]:
    return {
        "original_input": "Find radiation shielding transfer learning methods.",
        "object_terms": ["radiation shielding"],
        "task_terms": ["design"],
        "method_terms": ["transfer learning"],
        "scope_terms": ["nuclear engineering"],
        "method_constraint": MethodConstraint.PREFERRED,
        "accepted_paper_roles": {"method"},
        "source_language": "en",
        "revision": 1,
        "field_evidence": {
            IntentField.OBJECT: [TermEvidence(term="radiation shielding", source=TermSource.ORIGINAL_INPUT)],
            IntentField.TASK: [TermEvidence(term="design", source=TermSource.ORIGINAL_INPUT)],
            IntentField.METHOD: [TermEvidence(term="transfer learning", source=TermSource.ORIGINAL_INPUT)],
            IntentField.SCOPE: [TermEvidence(term="nuclear engineering", source=TermSource.ORIGINAL_INPUT)],
            IntentField.ACCEPTED_PAPER_ROLES: [TermEvidence(term="method", source=TermSource.DETERMINISTIC_RULE)],
        },
    }


def test_intent_draft_rejects_blank_terms_and_extra_fields() -> None:
    payload = _draft_payload()
    payload["object_terms"] = ["  "]
    with pytest.raises(ValidationError):
        IntentDraft.model_validate(payload)

    payload = _draft_payload()
    payload["unknown"] = "not allowed"
    with pytest.raises(ValidationError):
        IntentDraft.model_validate(payload)


def test_gap_and_question_scores_are_bounded() -> None:
    with pytest.raises(ValidationError):
        IntentGap(
            field=IntentField.OBJECT,
            missingness_score=1.1,
            ambiguity_score=0.0,
            priority_score=0.0,
            reason="bad score",
            already_explicit=False,
        )
    with pytest.raises(ValidationError):
        ClarificationQuestion(
            question_id="CQ-OBJECT",
            target_field=IntentField.OBJECT,
            question_zh="对象是什么？",
            question_en="What is the object?",
            reason="missing",
            priority_score=-0.1,
        )


def test_query_plan_rejects_non_first_round_and_wrong_source() -> None:
    query = Query(
        query_id="Q1-test",
        round_number=1,
        branch=QueryBranch.DIRECT_INTERSECTION,
        breadth=QueryBreadth.NARROW,
        language="en",
        query_text='"radiation shielding"',
        weight=1.0,
    )
    payload = {
        "plan_id": "plan-test",
        "project_id": "RRC-2026-0001",
        "round_number": 2,
        "intent_revision": 1,
        "source": "other",
        "planning_config_version": "m1-t01.v1",
        "branch_weights": {branch: 0.25 for branch in QueryBranch},
        "queries": [query] * 12,
        "generated_at_utc": datetime(2026, 1, 1, tzinfo=UTC),
    }
    with pytest.raises(ValidationError):
        QueryPlan.model_validate(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {"candidate_terms": {"object": ["shielding"]}, "unknown": []},
        {"candidate_terms": {"object": ["shielding"]}, "field_confidences": {"object": 2}},
        {"candidate_terms": {"object": ["shielding"]}, "source_language": "fr"},
    ],
)
def test_invalid_llm_fake_is_rejected_without_repair(payload: dict[str, object]) -> None:
    with pytest.raises(PlanningError, match="INVALID_INTENT_STRUCTURE"):
        validate_llm_candidate(payload)


def test_intent_draft_requires_exact_nonduplicated_evidence_per_term() -> None:
    payload = _draft_payload()
    payload["object_terms"] = ["radiation shielding", "radiation protection"]
    payload["field_evidence"] = {
        **payload["field_evidence"],
        IntentField.OBJECT: [
            TermEvidence(term="radiation shielding", source=TermSource.ORIGINAL_INPUT)
        ],
    }
    with pytest.raises(ValidationError):
        IntentDraft.model_validate(payload)


def test_intent_evidence_uses_nfkc_for_matching_and_duplicate_detection() -> None:
    payload = _draft_payload()
    payload["object_terms"] = ["ＡＢＣ"]
    payload["field_evidence"] = {
        **payload["field_evidence"],
        IntentField.OBJECT: [TermEvidence(term="ABC", source=TermSource.ORIGINAL_INPUT)],
    }
    assert IntentDraft.model_validate(payload).object_terms == ["ＡＢＣ"]

    payload["object_terms"] = ["ＡＢＣ", "ABC"]
    payload["field_evidence"] = {
        **payload["field_evidence"],
        IntentField.OBJECT: [
            TermEvidence(term="ＡＢＣ", source=TermSource.ORIGINAL_INPUT),
            TermEvidence(term="ABC", source=TermSource.ORIGINAL_INPUT),
        ],
    }
    with pytest.raises(ValidationError):
        IntentDraft.model_validate(payload)


def _complete_intent_for_query_contract() -> object:
    return freeze_research_intent(IntentDraft.model_validate(_draft_payload()), datetime(2026, 1, 1, tzinfo=UTC))


def test_query_plan_enforces_version_text_branch_weights_and_expansion_fields() -> None:
    plan = build_query_plan("RRC-2026-0001", _complete_intent_for_query_contract())
    payload = plan.model_dump(mode="python")
    payload["planning_config_version"] = "m1-t01.v1"
    with pytest.raises(ValidationError):
        QueryPlan.model_validate(payload)

    payload = plan.model_dump(mode="python")
    payload["queries"][1] = payload["queries"][0]
    with pytest.raises(ValidationError):
        QueryPlan.model_validate(payload)

    payload = plan.model_dump(mode="python")
    payload["queries"][0]["weight"] = 0.2
    with pytest.raises(ValidationError):
        QueryPlan.model_validate(payload)

    payload = plan.model_dump(mode="python")
    payload["positive_expansions"] = [
        {"target_field": "accepted_paper_roles", "term_en": "survey", "source": "llm_fake"}
    ]
    with pytest.raises(ValidationError):
        QueryPlan.model_validate(payload)

    assert RetrievalIntentField.OBJECT.value == "object"

    payload = _draft_payload()
    payload["field_evidence"] = {
        **payload["field_evidence"],
        IntentField.OBJECT: [
            TermEvidence(term="radiation shielding", source=TermSource.ORIGINAL_INPUT),
            TermEvidence(term="radiation shielding", source=TermSource.LLM_FAKE),
        ],
    }
    with pytest.raises(ValidationError):
        IntentDraft.model_validate(payload)
