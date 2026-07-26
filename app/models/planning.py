"""Strict M1 intent-clarification and first-round query-planning contracts."""

from datetime import UTC, datetime
from enum import StrEnum
from math import isclose
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.models.enums import MethodConstraint, QueryBranch, QueryBreadth
from app.models.project import ResearchIntent
from app.models.query import Query


class IntentField(StrEnum):
    OBJECT = "object"
    TASK = "task"
    METHOD = "method"
    SCOPE = "scope"
    ACCEPTED_PAPER_ROLES = "accepted_paper_roles"
    EXCLUSIONS = "exclusions"


class TermSource(StrEnum):
    ORIGINAL_INPUT = "original_input"
    USER_CLARIFICATION = "user_clarification"
    DETERMINISTIC_RULE = "deterministic_rule"
    LLM_FAKE = "llm_fake"


def _trim_term(value: str) -> str:
    return " ".join(value.strip().split())


def _clean_term_collection(value: object) -> object:
    if not isinstance(value, (list, set, tuple)):
        return value
    cleaned: list[str] = []
    for term in value:
        if not isinstance(term, str):
            return value
        normalised = _trim_term(term)
        if not normalised:
            raise ValueError("Terms must be non-blank after trimming")
        cleaned.append(normalised)
    return cleaned


class PlanningError(ValueError):
    """Stable planning error that refuses malformed or incomplete input."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class TermEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    term: str = Field(min_length=1)
    source: TermSource

    @field_validator("term")
    @classmethod
    def validate_term(cls, value: str) -> str:
        normalised = _trim_term(value)
        if not normalised:
            raise ValueError("Evidence terms must be non-blank after trimming")
        return normalised


class RetrievalTerm(BaseModel):
    """A source-traceable term with its explicit English retrieval mapping."""

    model_config = ConfigDict(extra="forbid")

    original_text: str = Field(min_length=1)
    retrieval_text_en: str = Field(min_length=1)
    source: TermSource
    target_field: IntentField

    @field_validator("original_text", "retrieval_text_en")
    @classmethod
    def validate_retrieval_text(cls, value: str) -> str:
        normalised = _trim_term(value)
        if not normalised:
            raise ValueError("Retrieval terms must be non-blank after trimming")
        if '"' in normalised or "%" in normalised:
            raise ValueError("Retrieval terms cannot contain query syntax or URL encoding")
        return normalised


class QueryExpansion(BaseModel):
    """A provenance-bearing English term usable only in its declared field."""

    model_config = ConfigDict(extra="forbid")

    target_field: IntentField
    term_en: str = Field(min_length=1)
    source: TermSource

    @field_validator("term_en")
    @classmethod
    def validate_term_en(cls, value: str) -> str:
        normalised = _trim_term(value)
        if not normalised or not normalised.isascii() or '"' in normalised or "%" in normalised:
            raise ValueError("Query expansion must be plain ASCII retrieval text")
        return normalised.casefold()


class IntentDraft(BaseModel):
    """Pre-freeze input; this is deliberately not a ResearchIntent subtype."""

    model_config = ConfigDict(extra="forbid")

    original_input: str = Field(min_length=1)
    object_terms: list[str] = Field(default_factory=list)
    task_terms: list[str] = Field(default_factory=list)
    method_terms: list[str] = Field(default_factory=list)
    scope_terms: list[str] = Field(default_factory=list)
    exclusions: list[str] = Field(default_factory=list)
    method_constraint: MethodConstraint | None = None
    accepted_paper_roles: set[str] = Field(default_factory=set)
    source_language: Literal["zh", "en", "mixed"]
    revision: int = Field(ge=1)
    field_evidence: dict[IntentField, list[TermEvidence]] = Field(default_factory=dict)

    @field_validator(
        "object_terms",
        "task_terms",
        "method_terms",
        "scope_terms",
        "exclusions",
        "accepted_paper_roles",
        mode="before",
    )
    @classmethod
    def validate_term_collections(cls, value: object) -> object:
        return _clean_term_collection(value)

    @field_validator("original_input")
    @classmethod
    def validate_original_input(cls, value: str) -> str:
        normalised = _trim_term(value)
        if not normalised:
            raise ValueError("original_input must be non-blank after trimming")
        return normalised

    @model_validator(mode="after")
    def validate_evidence_matches_terms(self) -> "IntentDraft":
        terms = {
            IntentField.OBJECT: self.object_terms,
            IntentField.TASK: self.task_terms,
            IntentField.METHOD: self.method_terms,
            IntentField.SCOPE: self.scope_terms,
            IntentField.ACCEPTED_PAPER_ROLES: list(self.accepted_paper_roles),
            IntentField.EXCLUSIONS: self.exclusions,
        }
        for field, evidence_items in self.field_evidence.items():
            declared = [_trim_term(term).casefold() for term in terms[field]]
            evidence = [_trim_term(item.term).casefold() for item in evidence_items]
            if len(declared) != len(set(declared)):
                raise ValueError("Declared terms must not contain normalized duplicates")
            if len(evidence) != len(set(evidence)) or set(evidence) != set(declared):
                raise ValueError(
                    "Field evidence must exactly cover declared terms without duplicates"
                )
        for field, field_terms in terms.items():
            if field_terms and not self.field_evidence.get(field):
                raise ValueError(f"Non-empty {field.value} terms require field evidence")
        return self


class IntentGap(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: IntentField
    missingness_score: float = Field(ge=0, le=1)
    ambiguity_score: float = Field(ge=0, le=1)
    priority_score: float = Field(ge=0, le=1)
    reason: str = Field(min_length=1)
    already_explicit: bool


class ClarificationQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(min_length=1)
    target_field: IntentField
    question_zh: str = Field(min_length=1)
    question_en: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    priority_score: float = Field(ge=0, le=1)


class IntentPreparationResult(BaseModel):
    """The result of one strict fake-provider intent-preparation call."""

    model_config = ConfigDict(extra="forbid")

    draft: IntentDraft
    retrieval_terms: list[RetrievalTerm]
    clarification_questions: list[ClarificationQuestion] = Field(default_factory=list)
    research_intent: ResearchIntent | None = None

    @model_validator(mode="after")
    def validate_result_state(self) -> "IntentPreparationResult":
        if self.research_intent is None and not self.clarification_questions:
            raise ValueError("An unfreezable draft must return clarification questions")
        if self.research_intent is not None and self.clarification_questions:
            raise ValueError("A frozen intent cannot retain clarification questions")
        return self


class TermConflict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    term: str = Field(min_length=1)
    positive_source: TermSource
    exclusion_source: TermSource


class QueryPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan_id: str = Field(min_length=1)
    project_id: str = Field(pattern=r"^RRC-\d{4}-\d{4}$")
    round_number: int
    intent_revision: int = Field(ge=1)
    source: Literal["arxiv"]
    planning_config_version: str = Field(min_length=1)
    branch_weights: dict[QueryBranch, float]
    queries: list[Query]
    generated_at_utc: datetime
    exclusions: list[str] = Field(default_factory=list)
    positive_expansions: list[QueryExpansion] = Field(default_factory=list)
    excluded_term_conflicts: list[TermConflict] = Field(default_factory=list)

    @field_validator("generated_at_utc")
    @classmethod
    def validate_utc_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
            raise ValueError("generated_at_utc must be UTC-aware")
        return value

    @model_validator(mode="after")
    def validate_first_round_structure(self) -> "QueryPlan":
        if self.round_number != 1:
            raise ValueError("M1-T01 QueryPlan round_number must be 1")
        if set(self.branch_weights) != set(QueryBranch):
            raise ValueError("QueryPlan must declare all four branch weights")
        if any(weight <= 0 for weight in self.branch_weights.values()):
            raise ValueError("QueryPlan branch weights must be positive")
        if not isclose(sum(self.branch_weights.values()), 1.0, abs_tol=1e-9):
            raise ValueError("QueryPlan branch weights must sum to 1")
        if len(self.queries) != 12:
            raise ValueError("QueryPlan must contain exactly 12 queries")
        if len({query.query_id for query in self.queries}) != 12:
            raise ValueError("QueryPlan query IDs must be unique")
        if any(
            query.round_number != 1
            or query.language != "en"
            or query.revision_sources
            or query.weight <= 0
            for query in self.queries
        ):
            raise ValueError("QueryPlan queries must be positive English round-one queries")
        if not isclose(sum(query.weight for query in self.queries), 1.0, abs_tol=1e-9):
            raise ValueError("QueryPlan query weights must sum to 1")
        for branch in QueryBranch:
            breadths = {query.breadth for query in self.queries if query.branch is branch}
            if breadths != set(QueryBreadth):
                raise ValueError("Each branch must contain all three query breadths")
        return self
