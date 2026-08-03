"""Closed, source-grounded contracts for the M2-T03 evidence boundary."""

from __future__ import annotations

import hashlib
import json
import re
from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.models.enums import EvidenceSlot, SupportLevel

EVIDENCE_CLASSIFICATION_VERSION = "m2-t03-evidence-classification.v1"
EVIDENCE_CLASSIFIER_SCHEMA_VERSION = EVIDENCE_CLASSIFICATION_VERSION
MAX_CLASSIFICATION_REASON_LENGTH = 1200
MAX_SUPPORTING_EXCERPT_LENGTH = 1000

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

EvidenceClassificationErrorCode = Literal[
    "INVALID_INPUT",
    "PROVIDER_UNAVAILABLE",
    "INVALID_OUTPUT",
    "DUPLICATE_PAPER_ID",
    "PAPER_ID_MISMATCH",
    "SOURCE_HASH_MISMATCH",
    "INSUFFICIENT_SOURCE_TEXT",
    "RESULT_CONFLICT",
    "RESULT_PUBLICATION_FAILED",
    "FIXED_INPUT_INVALID",
]

EVIDENCE_CLASSIFICATION_ERROR_CODES: tuple[EvidenceClassificationErrorCode, ...] = (
    "INVALID_INPUT",
    "PROVIDER_UNAVAILABLE",
    "INVALID_OUTPUT",
    "DUPLICATE_PAPER_ID",
    "PAPER_ID_MISMATCH",
    "SOURCE_HASH_MISMATCH",
    "INSUFFICIENT_SOURCE_TEXT",
    "RESULT_CONFLICT",
    "RESULT_PUBLICATION_FAILED",
    "FIXED_INPUT_INVALID",
)


class EvidenceClassificationError(ValueError):
    """A stable, privacy-safe failure at the M2-T03 boundary."""

    def __init__(self, code: EvidenceClassificationErrorCode | str) -> None:
        if code not in EVIDENCE_CLASSIFICATION_ERROR_CODES:
            raise ValueError(f"Unknown evidence classification error code: {code}")
        self.code: EvidenceClassificationErrorCode = code  # type: ignore[assignment]
        super().__init__(code)


class EvidenceClassificationState(StrEnum):
    """Terminal state of one paper's classification record."""

    CLASSIFIED = "CLASSIFIED"
    TITLE_ONLY = "TITLE_ONLY"
    REJECTED = "REJECTED"
    INSUFFICIENT_SOURCE_TEXT = "INSUFFICIENT_SOURCE_TEXT"


class EvidenceClassificationBatchState(StrEnum):
    """Terminal state of a batch after provider output validation."""

    COMPLETE = "COMPLETE"
    PARTIAL_FAILURE = "PARTIAL_FAILURE"


def normalize_source_text(value: str, *, field_name: str) -> str:
    """Normalize one source field without inventing or silently dropping text."""

    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-blank after trimming")
    return normalized


def normalize_optional_source_text(value: str | None, *, field_name: str) -> str | None:
    """Normalize an optional source field while rejecting supplied blank text."""

    if value is None:
        return None
    return normalize_source_text(value, field_name=field_name)


def serialize_source_text(title: str, abstract: str | None) -> str:
    """Serialize only normalized title/abstract fields for the provenance hash."""

    normalized_title = normalize_source_text(title, field_name="title")
    normalized_abstract = normalize_optional_source_text(abstract, field_name="abstract")
    return json.dumps(
        {"title": normalized_title, "abstract": normalized_abstract},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def source_text_sha256(title: str, abstract: str | None) -> str:
    """Return the exact UTF-8 SHA-256 for canonical title/abstract serialization."""

    return hashlib.sha256(serialize_source_text(title, abstract).encode("utf-8")).hexdigest()


def build_grounded_reason(
    support_level: SupportLevel,
    supporting_excerpt: str,
    *,
    title_only: bool = False,
) -> str:
    """Build the bounded reason grammar accepted at the classification boundary."""

    excerpt = supporting_excerpt.strip()
    if not excerpt:
        raise ValueError("supporting_excerpt must be non-blank")
    if title_only:
        return f"title-only 来源摘录支持该证据槽位：{excerpt}。"
    if support_level is SupportLevel.DIRECT:
        return f"来源摘录直接支持该证据槽位：{excerpt}。"
    if support_level is SupportLevel.INDIRECT:
        return f"来源摘录提供间接迁移依据，仍需在目标问题中验证：{excerpt}。"
    if support_level is SupportLevel.HYPOTHETICAL:
        return f"来源摘录仅支持可能适用的假设探索，不能声称目标问题已有结论：{excerpt}。"
    raise ValueError("unknown support level")


class EvidenceClassifierDescriptor(BaseModel):
    """Closed identity of the classifier and schema revision producing a record."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider_name: str = Field(min_length=1, max_length=100)
    classifier_id: str = Field(min_length=1, max_length=200)
    classifier_revision: str = Field(min_length=1, max_length=200)
    prompt_revision: str = Field(min_length=1, max_length=200)
    schema_version: Literal["m2-t03-evidence-classification.v1"]
    evidence_type: Literal["deterministic_fake", "real_model", "human"]

    @field_validator(
        "provider_name",
        "classifier_id",
        "classifier_revision",
        "prompt_revision",
    )
    @classmethod
    def reject_blank_descriptor_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Classifier descriptor fields must be non-blank")
        return normalized


class EvidenceClassificationInput(BaseModel):
    """One classifier input derived solely from a paper ID, title, and abstract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    paper_id: str = Field(min_length=1, max_length=200)
    title: str = Field(min_length=1, max_length=2000)
    abstract: str | None = Field(default=None, max_length=30000)
    source_text_sha256: str = Field(min_length=64, max_length=64)
    classifier_descriptor: EvidenceClassifierDescriptor
    classification_version: Literal["m2-t03-evidence-classification.v1"]

    @field_validator("paper_id")
    @classmethod
    def normalize_paper_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("paper_id must be non-blank")
        return normalized

    @field_validator("title")
    @classmethod
    def normalize_title(cls, value: str) -> str:
        return normalize_source_text(value, field_name="title")

    @field_validator("abstract")
    @classmethod
    def normalize_abstract(cls, value: str | None) -> str | None:
        return normalize_optional_source_text(value, field_name="abstract")

    @field_validator("source_text_sha256")
    @classmethod
    def require_lowercase_sha256(cls, value: str) -> str:
        if _SHA256_RE.fullmatch(value) is None:
            raise ValueError("source_text_sha256 must be lowercase 64-character hex")
        return value

    @model_validator(mode="after")
    def require_exact_source_hash_and_version(self) -> Self:
        if self.classification_version != EVIDENCE_CLASSIFICATION_VERSION:
            raise ValueError("classification_version is not the declared M2-T03 version")
        if self.classifier_descriptor.schema_version != self.classification_version:
            raise ValueError("classifier descriptor schema_version must match classification_version")
        expected = source_text_sha256(self.title, self.abstract)
        if self.source_text_sha256 != expected:
            raise ValueError("source_text_sha256 must hash normalized title/abstract serialization")
        return self


class EvidenceClassificationRecord(BaseModel):
    """One source-grounded classification or explicit fail-closed rejection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    paper_id: str = Field(min_length=1, max_length=200)
    evidence_slot: EvidenceSlot | None = None
    support_level: SupportLevel | None = None
    reason: str = Field(min_length=1, max_length=MAX_CLASSIFICATION_REASON_LENGTH)
    supporting_excerpt: str | None = Field(
        default=None, max_length=MAX_SUPPORTING_EXCERPT_LENGTH
    )
    source_text_sha256: str = Field(min_length=64, max_length=64)
    classifier_descriptor: EvidenceClassifierDescriptor
    classification_version: Literal["m2-t03-evidence-classification.v1"]
    state: EvidenceClassificationState

    @field_validator("paper_id")
    @classmethod
    def normalize_record_paper_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("paper_id must be non-blank")
        return normalized

    @field_validator("reason")
    @classmethod
    def normalize_reason(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("reason must be non-blank after trimming")
        return normalized

    @field_validator("supporting_excerpt")
    @classmethod
    def normalize_excerpt(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("supporting_excerpt must be non-blank when present")
        return normalized

    @field_validator("source_text_sha256")
    @classmethod
    def require_record_sha256(cls, value: str) -> str:
        if _SHA256_RE.fullmatch(value) is None:
            raise ValueError("source_text_sha256 must be lowercase 64-character hex")
        return value

    @model_validator(mode="after")
    def require_state_consistent_fields(self) -> Self:
        if self.classification_version != EVIDENCE_CLASSIFICATION_VERSION:
            raise ValueError("classification_version is not the declared M2-T03 version")
        if self.classifier_descriptor.schema_version != self.classification_version:
            raise ValueError("classifier descriptor schema_version must match classification_version")
        is_classified = self.state in {
            EvidenceClassificationState.CLASSIFIED,
            EvidenceClassificationState.TITLE_ONLY,
        }
        if is_classified:
            if self.evidence_slot is None or self.support_level is None:
                raise ValueError("classified records require evidence_slot and support_level")
            if self.supporting_excerpt is None:
                raise ValueError("classified records require supporting_excerpt")
            if (
                self.state is EvidenceClassificationState.TITLE_ONLY
                and "title-only" not in self.reason.casefold()
                and "仅标题" not in self.reason
            ):
                raise ValueError("TITLE_ONLY records must be explicitly marked title-only")
        elif (
            self.evidence_slot is not None
            or self.support_level is not None
            or self.supporting_excerpt is not None
        ):
            raise ValueError("rejected records must not carry classification or source excerpts")
        return self

    def validate_against_input(self, source: EvidenceClassificationInput) -> None:
        """Validate source binding and exact excerpt membership for one input."""

        if self.paper_id != source.paper_id:
            raise ValueError("classification paper_id does not match input")
        if self.source_text_sha256 != source.source_text_sha256:
            raise ValueError("classification source_text_sha256 does not match input")
        if self.classifier_descriptor != source.classifier_descriptor:
            raise ValueError("classification descriptor does not match input")
        if self.classification_version != source.classification_version:
            raise ValueError("classification version does not match input")
        if self.state is EvidenceClassificationState.TITLE_ONLY and source.abstract is not None:
            raise ValueError("TITLE_ONLY requires a missing abstract")
        if self.state in {
            EvidenceClassificationState.CLASSIFIED,
            EvidenceClassificationState.TITLE_ONLY,
        }:
            assert self.supporting_excerpt is not None
            source_texts = (source.title,) if source.abstract is None else (
                source.title,
                source.abstract,
            )
            if not any(self.supporting_excerpt in text for text in source_texts):
                raise ValueError("supporting_excerpt must be an exact title/abstract substring")


class EvidenceClassificationBatch(BaseModel):
    """A closed batch with one descriptor and explicit partial failures."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    state: EvidenceClassificationBatchState
    records: list[EvidenceClassificationRecord] = Field(min_length=1)
    classifier_descriptor: EvidenceClassifierDescriptor
    classification_version: Literal["m2-t03-evidence-classification.v1"]

    @model_validator(mode="after")
    def require_closed_batch(self) -> Self:
        if self.classification_version != EVIDENCE_CLASSIFICATION_VERSION:
            raise ValueError("classification_version is not the declared M2-T03 version")
        if self.classifier_descriptor.schema_version != self.classification_version:
            raise ValueError("batch descriptor schema_version must match classification_version")
        paper_ids = [record.paper_id for record in self.records]
        if len(paper_ids) != len(set(paper_ids)):
            raise ValueError("batch paper_id values must be unique")
        if any(record.classifier_descriptor != self.classifier_descriptor for record in self.records):
            raise ValueError("batch records must share one classifier descriptor")
        if any(record.classification_version != self.classification_version for record in self.records):
            raise ValueError("batch records must share one classification version")
        has_failure = any(
            record.state
            in {
                EvidenceClassificationState.REJECTED,
                EvidenceClassificationState.INSUFFICIENT_SOURCE_TEXT,
            }
            for record in self.records
        )
        if self.state is EvidenceClassificationBatchState.COMPLETE and has_failure:
            raise ValueError("COMPLETE batches cannot contain rejected records")
        if self.state is EvidenceClassificationBatchState.PARTIAL_FAILURE and not has_failure:
            raise ValueError("PARTIAL_FAILURE batches require an explicit rejected record")
        return self
