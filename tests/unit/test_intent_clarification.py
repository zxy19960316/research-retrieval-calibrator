from datetime import UTC, datetime

import pytest

from app.adapters.llm import StructuredRequest, validate_llm_candidate
from app.core.intent import (
    build_clarification_questions,
    build_intent_gaps,
    freeze_research_intent,
    prepare_intent,
)
from app.models.enums import MethodConstraint
from app.models.planning import IntentDraft, IntentField, PlanningError, TermEvidence, TermSource


def _draft(**overrides: object) -> IntentDraft:
    payload: dict[str, object] = {
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
            "object": [TermEvidence(term="radiation shielding", source=TermSource.ORIGINAL_INPUT)],
            "task": [TermEvidence(term="design", source=TermSource.ORIGINAL_INPUT)],
            "method": [TermEvidence(term="transfer learning", source=TermSource.ORIGINAL_INPUT)],
            "scope": [TermEvidence(term="nuclear engineering", source=TermSource.ORIGINAL_INPUT)],
            "accepted_paper_roles": [TermEvidence(term="method", source=TermSource.DETERMINISTIC_RULE)],
        },
    }
    payload.update(overrides)
    if payload["field_evidence"] == {}:
        payload["field_evidence"] = {
            "object": [TermEvidence(term=term, source=TermSource.ORIGINAL_INPUT) for term in payload["object_terms"]],
            "task": [TermEvidence(term=term, source=TermSource.ORIGINAL_INPUT) for term in payload["task_terms"]],
            "method": [TermEvidence(term=term, source=TermSource.ORIGINAL_INPUT) for term in payload["method_terms"]],
            "scope": [TermEvidence(term=term, source=TermSource.ORIGINAL_INPUT) for term in payload["scope_terms"]],
            "accepted_paper_roles": [TermEvidence(term=term, source=TermSource.DETERMINISTIC_RULE) for term in payload["accepted_paper_roles"]],
        }
    return IntentDraft.model_validate(payload)


def test_explicit_object_and_method_are_not_reasked() -> None:
    draft = _draft(task_terms=[], scope_terms=[], field_evidence={})
    questions = build_clarification_questions(draft, build_intent_gaps(draft))

    assert {question.target_field for question in questions}.isdisjoint(
        {IntentField.OBJECT, IntentField.METHOD}
    )


def test_missing_high_ambiguity_field_has_priority_and_questions_cap_at_three() -> None:
    draft = _draft(
        object_terms=[],
        task_terms=[],
        method_terms=[],
        scope_terms=[],
        accepted_paper_roles=set(),
        method_constraint=None,
        field_evidence={},
    )
    gaps = build_intent_gaps(draft, {IntentField.METHOD: 0.0})
    questions = build_clarification_questions(draft, gaps)

    assert gaps[0].field is IntentField.METHOD
    assert len(questions) == 3


def test_gap_ties_have_fixed_field_order() -> None:
    draft = _draft(
        object_terms=[], task_terms=[], method_terms=[], scope_terms=[], field_evidence={}
    )
    gaps = build_intent_gaps(draft)

    assert [gap.field for gap in gaps[:4]] == [
        IntentField.OBJECT,
        IntentField.TASK,
        IntentField.METHOD,
        IntentField.SCOPE,
    ]


def test_incomplete_draft_cannot_freeze_research_intent() -> None:
    draft = _draft(method_terms=[], method_constraint=None, field_evidence={})
    with pytest.raises(PlanningError, match="INCOMPLETE_INTENT"):
        freeze_research_intent(draft, datetime(2026, 1, 1, tzinfo=UTC))


def test_valid_fake_keeps_provenance_and_invalid_fake_is_not_guessed() -> None:
    candidate = validate_llm_candidate(
        {
            "candidate_terms": {"object": [" radiation shielding "]},
            "candidate_synonyms": {"method": ["domain adaptation"]},
            "field_confidences": {"object": 0.9},
            "source_language": "en",
        }
    )
    assert candidate.candidate_terms[IntentField.OBJECT] == ["radiation shielding"]

    with pytest.raises(PlanningError, match="INVALID_INTENT_STRUCTURE"):
        validate_llm_candidate({"candidate_terms": {"object": ["shielding"]}, "unexpected": True})


class _FakeProvider:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.calls = 0

    def generate_structured(self, request: StructuredRequest) -> dict[str, object]:
        assert request.original_input
        self.calls += 1
        return self.payload


def _complete_candidate(*, chinese: bool = False) -> dict[str, object]:
    if chinese:
        object_term = "\u8f90\u5c04\u5c4f\u853d"
        task_term = "\u8bbe\u8ba1"
        method_term = "\u8fc1\u79fb\u5b66\u4e60"
        scope_term = "\u6838\u5de5\u7a0b"
        language = "zh"
    else:
        object_term = "radiation shielding"
        task_term = "design"
        method_term = "transfer learning"
        scope_term = "nuclear engineering"
        language = "en"
    return {
        "candidate_terms": {
            "object": [object_term],
            "task": [task_term],
            "method": [method_term],
            "scope": [scope_term],
            "accepted_paper_roles": ["method"],
        },
        "candidate_method_constraint": "PREFERRED",
        "field_confidences": {"object": 0.9},
        "source_language": language,
    }


def test_prepare_intent_calls_fake_once_and_freezes_complete_english_draft() -> None:
    provider = _FakeProvider(_complete_candidate())

    result = prepare_intent(
        StructuredRequest(
            original_input="Find radiation shielding transfer learning methods.",
            target_fields=tuple(IntentField),
        ),
        provider,
    )

    assert provider.calls == 1
    assert result.research_intent is not None
    assert result.clarification_questions == []
    assert all(term.source is TermSource.LLM_FAKE for term in result.retrieval_terms)


def test_prepare_intent_keeps_chinese_original_and_mapped_english_retrieval_term() -> None:
    provider = _FakeProvider(_complete_candidate(chinese=True))

    result = prepare_intent(
        StructuredRequest(
            original_input="\u7528\u8fc1\u79fb\u5b66\u4e60\u8bbe\u8ba1\u8f90\u5c04\u5c4f\u853d",
            target_fields=tuple(IntentField),
        ),
        provider,
    )

    object_term = next(term for term in result.retrieval_terms if term.target_field is IntentField.OBJECT)
    assert object_term.original_text == "\u8f90\u5c04\u5c4f\u853d"
    assert object_term.retrieval_text_en == "radiation shielding"


def test_prepare_intent_returns_clarification_not_frozen_intent_for_incomplete_draft() -> None:
    candidate = _complete_candidate()
    candidate["candidate_terms"] = {
        key: value
        for key, value in candidate["candidate_terms"].items()
        if key not in {"scope", "accepted_paper_roles"}
    }
    candidate.pop("candidate_method_constraint")
    provider = _FakeProvider(candidate)

    result = prepare_intent(
        StructuredRequest(original_input="Find shielding methods.", target_fields=tuple(IntentField)),
        provider,
    )

    assert result.research_intent is None
    assert 1 <= len(result.clarification_questions) <= 3


def test_prepare_intent_rejects_invalid_fake_output_without_repair() -> None:
    provider = _FakeProvider({"candidate_terms": {"object": ["shielding"]}, "unexpected": True})

    with pytest.raises(PlanningError, match="INVALID_INTENT_STRUCTURE"):
        prepare_intent(
            StructuredRequest(original_input="Find shielding.", target_fields=tuple(IntentField)),
            provider,
        )
