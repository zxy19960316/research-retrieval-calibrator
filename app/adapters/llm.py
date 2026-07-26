"""LLM protocol and strict structured-candidate validation for M1-T01."""

from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.models.planning import IntentField, PlanningError, _clean_term_collection


class StructuredRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    original_input: str = Field(min_length=1)
    target_fields: tuple[IntentField, ...]


class LLMProvider(Protocol):
    def generate_structured(self, request: StructuredRequest) -> dict[str, object]: ...


class LLMCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_terms: dict[IntentField, list[str]] = Field(default_factory=dict)
    candidate_synonyms: dict[IntentField, list[str]] = Field(default_factory=dict)
    field_confidences: dict[IntentField, float] = Field(default_factory=dict)
    source_language: str = "mixed"

    @field_validator("candidate_terms", "candidate_synonyms", mode="before")
    @classmethod
    def validate_term_mappings(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        return {key: _clean_term_collection(terms) for key, terms in value.items()}

    @field_validator("source_language")
    @classmethod
    def validate_source_language(cls, value: str) -> str:
        if value not in {"zh", "en", "mixed"}:
            raise ValueError("source_language must be zh, en, or mixed")
        return value

    @field_validator("field_confidences")
    @classmethod
    def validate_confidences(cls, value: dict[IntentField, float]) -> dict[IntentField, float]:
        if any(confidence < 0 or confidence > 1 for confidence in value.values()):
            raise ValueError("field confidences must be between 0 and 1")
        return value


def validate_llm_candidate(payload: dict[str, object]) -> LLMCandidate:
    """Validate a fake candidate exactly once; never attempt speculative repair."""

    try:
        return LLMCandidate.model_validate(payload)
    except ValidationError as error:
        raise PlanningError("INVALID_INTENT_STRUCTURE") from error
