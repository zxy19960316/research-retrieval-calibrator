"""Closed data contracts for the M2-T02 reranker boundary."""

from __future__ import annotations

import hashlib
import math
import re
from enum import Enum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

RerankerErrorCode = Literal[
    "INVALID_INPUT",
    "PROVIDER_UNAVAILABLE",
    "INVALID_OUTPUT",
]

RERANKER_ERROR_CODES: tuple[RerankerErrorCode, ...] = (
    "INVALID_INPUT",
    "PROVIDER_UNAVAILABLE",
    "INVALID_OUTPUT",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_INPUT_FORMATS = frozenset(
    {"m2-reranker-title-abstract-v1", "m2-reranker-title-abstract-v2"}
)


class RerankerTaskError(ValueError):
    """A stable, machine-readable failure at the reranker task boundary."""

    def __init__(self, code: RerankerErrorCode | str) -> None:
        if code not in RERANKER_ERROR_CODES:
            raise ValueError(f"Unknown reranker task error code: {code}")
        self.code: RerankerErrorCode = code
        super().__init__(code)


class RerankerRunState(str, Enum):
    """States that may appear in a reranker run's machine-readable record."""

    SCORED = "SCORED"
    NOT_RUN = "NOT_RUN"
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"
    INVALID_OUTPUT = "INVALID_OUTPUT"


class RerankerModelDescriptor(BaseModel):
    """Immutable identity and input format of a reranking provider."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider_name: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_revision: str
    provider_library: str = Field(min_length=1)
    provider_library_version: str = Field(min_length=1)
    input_format_version: Literal[
        "m2-reranker-title-abstract-v1", "m2-reranker-title-abstract-v2"
    ]
    cache_namespace: str = Field(min_length=1)

    @field_validator(
        "provider_name",
        "model_id",
        "provider_library",
        "provider_library_version",
        "cache_namespace",
    )
    @classmethod
    def reject_blank_identity_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Reranker descriptor identity fields must be non-blank")
        return value

    @field_validator("model_revision")
    @classmethod
    def require_immutable_lowercase_revision(cls, value: str) -> str:
        if not _REVISION_RE.fullmatch(value):
            raise ValueError("Reranker model revision must be an immutable lowercase 40-character hex")
        return value

    @field_validator("input_format_version", mode="before")
    @classmethod
    def require_known_input_format(cls, value: object) -> object:
        if value not in _INPUT_FORMATS:
            raise ValueError("Reranker input format must be a declared title-abstract format")
        return value


class RerankerInput(BaseModel):
    """A pre-serialized reranker input with its exact UTF-8 text digest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    paper_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    input_sha256: str = Field(min_length=64, max_length=64)

    @field_validator("paper_id", "text")
    @classmethod
    def reject_blank_input_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Reranker input paper_id and text must be non-blank")
        return value

    @field_validator("input_sha256")
    @classmethod
    def require_lowercase_input_sha256(cls, value: str) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError("Reranker input_sha256 must be lowercase 64-character hex")
        return value

    @model_validator(mode="after")
    def require_exact_text_hash(self) -> Self:
        if hashlib.sha256(self.text.encode("utf-8")).hexdigest() != self.input_sha256:
            raise ValueError("Reranker input_sha256 must hash the UTF-8 text exactly")
        return self


class ProviderRawScore(BaseModel):
    """One finite, unnormalized score emitted by a reranking provider."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    paper_id: str = Field(min_length=1)
    raw_score: float

    @field_validator("paper_id")
    @classmethod
    def reject_blank_score_paper_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Provider raw score paper_id must be non-blank")
        return value

    @field_validator("raw_score", mode="before")
    @classmethod
    def require_numeric_raw_score(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("Provider raw score must not be boolean")  # noqa: TRY004
        if not isinstance(value, (int, float)):
            raise ValueError("Provider raw score must be a number")  # noqa: TRY004
        return value

    @field_validator("raw_score")
    @classmethod
    def require_finite_raw_score(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("Provider raw score must be finite")
        return value


class RerankRecord(BaseModel):
    """One reranking result, with failures prohibited from carrying a score."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    state: RerankerRunState
    paper_id: str = Field(min_length=1)
    raw_score: float | None
    normalized_score: float | None
    descriptor: RerankerModelDescriptor
    input_sha256: str = Field(min_length=64, max_length=64)

    @field_validator("paper_id")
    @classmethod
    def reject_blank_record_paper_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Rerank record paper_id must be non-blank")
        return value

    @field_validator("input_sha256")
    @classmethod
    def require_record_input_sha256(cls, value: str) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError("Rerank record input_sha256 must be lowercase 64-character hex")
        return value

    @field_validator("raw_score", "normalized_score", mode="before")
    @classmethod
    def require_numeric_optional_scores(cls, value: object) -> object:
        if value is None:
            return value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Rerank scores must be numeric, not boolean or text")  # noqa: TRY004
        return value

    @model_validator(mode="after")
    def require_state_consistent_scores(self) -> Self:
        if self.state is RerankerRunState.SCORED:
            if self.raw_score is None or self.normalized_score is None:
                raise ValueError("SCORED records require raw and normalized scores")
            if not math.isfinite(self.raw_score):
                raise ValueError("SCORED raw score must be finite")
            if not math.isfinite(self.normalized_score) or not 0 <= self.normalized_score <= 1:
                raise ValueError("SCORED normalized score must be finite and within [0, 1]")
        elif self.raw_score is not None or self.normalized_score is not None:
            raise ValueError("non-SCORED records must not carry scores")
        return self


class RerankRun(BaseModel):
    """A closed reranker run: either fully scored or explicitly scoreless."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    state: RerankerRunState
    records: list[RerankRecord]

    @model_validator(mode="after")
    def require_closed_run_state(self) -> Self:
        if self.state is RerankerRunState.SCORED:
            if not self.records:
                raise ValueError("SCORED runs require non-empty records")
            if any(record.state is not RerankerRunState.SCORED for record in self.records):
                raise ValueError("SCORED runs require only SCORED records")
            paper_ids = [record.paper_id for record in self.records]
            if len(paper_ids) != len(set(paper_ids)):
                raise ValueError("SCORED run paper_id values must be unique")
            descriptors = {record.descriptor for record in self.records}
            if len(descriptors) != 1:
                raise ValueError("SCORED run records must share one descriptor")
        elif self.records:
            if self.state in {
                RerankerRunState.PROVIDER_UNAVAILABLE,
                RerankerRunState.INVALID_OUTPUT,
            }:
                raise ValueError("Failed reranker runs must not retain partial records")
            raise ValueError("NOT_RUN reranker runs must have empty records")
        return self
