"""Pure M1-T01 intent normalisation, clarification, and freeze functions."""

import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Literal, cast

from app.adapters.llm import LLMProvider, StructuredRequest, validate_llm_candidate
from app.core.text_normalization import normalise_text
from app.models.planning import (
    ClarificationQuestion,
    IntentDraft,
    IntentField,
    IntentGap,
    IntentPreparationResult,
    PlanningError,
    QueryExpansion,
    RetrievalIntentField,
    RetrievalTerm,
    TermEvidence,
    TermSource,
)
from app.models.project import ResearchIntent

_FIELD_ORDER = (
    IntentField.OBJECT,
    IntentField.TASK,
    IntentField.METHOD,
    IntentField.SCOPE,
    IntentField.ACCEPTED_PAPER_ROLES,
    IntentField.EXCLUSIONS,
)
_QUESTION_TEMPLATES = {
    IntentField.OBJECT: ("研究对象具体是什么？", "What is the exact research object?"),
    IntentField.TASK: ("希望完成什么研究任务？", "What research task should the papers support?"),
    IntentField.METHOD: ("方法是必需、偏好还是开放？", "Is the method required, preferred, or open?"),
    IntentField.SCOPE: ("适用范围或场景是什么？", "What application scope or setting is required?"),
    IntentField.ACCEPTED_PAPER_ROLES: ("可接受哪些论文角色？", "Which paper roles are acceptable?"),
    IntentField.EXCLUSIONS: ("有哪些需要排除的方向？", "Which directions should be excluded?"),
}
_RETRIEVAL_TRANSLATIONS = {
    "\u8f90\u5c04\u5c4f\u853d": "radiation shielding",
    "\u6838\u53cd\u5e94\u5806": "nuclear reactor",
    "\u6838\u5de5\u7a0b": "nuclear engineering",
    "\u8fc1\u79fb\u5b66\u4e60": "transfer learning",
    "\u6df1\u5ea6\u5b66\u4e60": "deep learning",
    "\u673a\u5668\u5b66\u4e60": "machine learning",
    "\u8bbe\u8ba1": "design",
    "\u8bc4\u4f30": "evaluation",
    "\u65b9\u6cd5": "method",
    "\u6846\u67b6": "framework",
}
_CJK_PATTERN = re.compile(r"[\u3400-\u9fff]")


def normalise_term(value: str) -> str:
    """Canonical comparison form required by the M1-T01 exclusion contract."""

    return normalise_text(value)


def build_retrieval_term(
    original_text: str,
    *,
    source: TermSource,
    target_field: IntentField,
) -> RetrievalTerm:
    """Map a required term to English or fail closed without content loss."""

    retrieval_text = normalise_text(original_text)
    for chinese, english in sorted(_RETRIEVAL_TRANSLATIONS.items(), key=lambda item: -len(item[0])):
        retrieval_text = retrieval_text.replace(chinese, english)
    if _CJK_PATTERN.search(retrieval_text):
        raise PlanningError("UNMAPPED_RETRIEVAL_TERM")
    retrieval_text = " ".join(retrieval_text.split()).casefold()
    if not retrieval_text:
        raise PlanningError("UNMAPPED_RETRIEVAL_TERM")
    return RetrievalTerm(
        original_text=original_text,
        retrieval_text_en=retrieval_text,
        source=source,
        target_field=target_field,
    )


def _field_is_explicit(draft: IntentDraft, field: IntentField) -> bool:
    if field is IntentField.OBJECT:
        return bool(draft.object_terms)
    if field is IntentField.TASK:
        return bool(draft.task_terms)
    if field is IntentField.METHOD:
        return bool(draft.method_terms) and draft.method_constraint is not None
    if field is IntentField.SCOPE:
        return bool(draft.scope_terms)
    if field is IntentField.ACCEPTED_PAPER_ROLES:
        return bool(draft.accepted_paper_roles)
    return bool(draft.exclusions)


def build_intent_gaps(
    draft: IntentDraft,
    field_confidences: Mapping[IntentField, float] | None = None,
) -> list[IntentGap]:
    """Score known intent fields without random values or an external call."""

    confidences = field_confidences or {}
    gaps: list[IntentGap] = []
    for field in _FIELD_ORDER:
        explicit = _field_is_explicit(draft, field)
        if explicit or field is IntentField.EXCLUSIONS:
            missingness = 0.0
            ambiguity = 0.0
        else:
            missingness = 1.0
            confidence = confidences.get(field, 0.5)
            if not 0 <= confidence <= 1:
                raise PlanningError("INVALID_INTENT_STRUCTURE")
            ambiguity = 1.0 - confidence
        priority = 0.65 * missingness + 0.35 * ambiguity
        gaps.append(
            IntentGap(
                field=field,
                missingness_score=missingness,
                ambiguity_score=ambiguity,
                priority_score=priority,
                reason="already explicit" if explicit else "missing or ambiguous information",
                already_explicit=explicit,
            )
        )
    order = {field: index for index, field in enumerate(_FIELD_ORDER)}
    return sorted(gaps, key=lambda gap: (-gap.priority_score, order[gap.field]))


def build_clarification_questions(
    draft: IntentDraft, gaps: Sequence[IntentGap]
) -> list[ClarificationQuestion]:
    """Return no more than three stable bilingual questions for unresolved fields."""

    del draft  # The gaps carry explicitness; preserve a simple, pure interface.
    order = {field: index for index, field in enumerate(_FIELD_ORDER)}
    unanswered = [gap for gap in gaps if not gap.already_explicit and gap.priority_score > 0]
    ordered = sorted(unanswered, key=lambda gap: (-gap.priority_score, order[gap.field]))[:3]
    return [
        ClarificationQuestion(
            question_id=f"CQ-{gap.field.value.upper()}",
            target_field=gap.field,
            question_zh=_QUESTION_TEMPLATES[gap.field][0],
            question_en=_QUESTION_TEMPLATES[gap.field][1],
            reason=gap.reason,
            priority_score=gap.priority_score,
        )
        for gap in ordered
    ]


def freeze_research_intent(draft: IntentDraft, frozen_at: datetime) -> ResearchIntent:
    """Create M0's immutable intent only once every required field is explicit."""

    required = (
        bool(draft.object_terms),
        bool(draft.task_terms),
        bool(draft.method_terms),
        bool(draft.scope_terms),
        draft.method_constraint is not None,
        bool(draft.accepted_paper_roles),
    )
    if not all(required):
        raise PlanningError("INCOMPLETE_INTENT")
    return ResearchIntent(
        object_terms=draft.object_terms,
        task_terms=draft.task_terms,
        method_terms=draft.method_terms,
        scope_terms=draft.scope_terms,
        exclusions=draft.exclusions,
        method_constraint=draft.method_constraint,
        accepted_paper_roles=draft.accepted_paper_roles,
        revision=draft.revision,
        frozen_at=frozen_at,
    )


def prepare_intent(
    request: StructuredRequest,
    provider: LLMProvider,
) -> IntentPreparationResult:
    """Call a fake provider once, validate strictly, then clarify or freeze its draft."""

    candidate = validate_llm_candidate(provider.generate_structured(request))
    terms_by_field = {
        field: list(candidate.candidate_terms.get(field, [])) for field in IntentField
    }
    retrieval_terms = [
        build_retrieval_term(term, source=TermSource.LLM_FAKE, target_field=field)
        for field, terms in terms_by_field.items()
        for term in terms
    ]
    field_evidence = {
        field: [TermEvidence(term=term, source=TermSource.LLM_FAKE) for term in terms]
        for field, terms in terms_by_field.items()
        if terms
    }
    query_expansions: list[QueryExpansion] = []
    for field, synonyms in candidate.candidate_synonyms.items():
        if field.value not in {item.value for item in RetrievalIntentField}:
            raise PlanningError("INVALID_QUERY_PLAN")
        target_field = RetrievalIntentField(field.value)
        for synonym in synonyms:
            mapped = build_retrieval_term(
                synonym,
                source=TermSource.LLM_FAKE,
                target_field=field,
            )
            query_expansions.append(
                QueryExpansion(
                    target_field=target_field,
                    term_en=mapped.retrieval_text_en,
                    source=TermSource.LLM_FAKE,
                )
            )
    source_language = cast(Literal["zh", "en", "mixed"], candidate.source_language)
    draft = IntentDraft(
        original_input=request.original_input,
        object_terms=terms_by_field[IntentField.OBJECT],
        task_terms=terms_by_field[IntentField.TASK],
        method_terms=terms_by_field[IntentField.METHOD],
        scope_terms=terms_by_field[IntentField.SCOPE],
        exclusions=terms_by_field[IntentField.EXCLUSIONS],
        method_constraint=candidate.candidate_method_constraint,
        accepted_paper_roles=set(terms_by_field[IntentField.ACCEPTED_PAPER_ROLES]),
        source_language=source_language,
        revision=1,
        field_evidence=field_evidence,
    )
    gaps = build_intent_gaps(draft, candidate.field_confidences)
    questions = build_clarification_questions(draft, gaps)
    if questions:
        return IntentPreparationResult(
            draft=draft,
            retrieval_terms=retrieval_terms,
            query_expansions=query_expansions,
            clarification_questions=questions,
        )
    return IntentPreparationResult(
        draft=draft,
        retrieval_terms=retrieval_terms,
        query_expansions=query_expansions,
        research_intent=freeze_research_intent(draft, datetime.now(UTC)),
    )
