"""Independent contracts for completed M2-T03 human judgments."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.models.enums import EvidenceSlot, SupportLevel
from app.models.evidence_classification import source_text_sha256 as build_source_text_sha256
from app.models.m2_t03_human_adjudication import ArtifactBinding

M2_T03_HUMAN_RESULT_VERSION = "m2-t03-human-adjudication-result.v1"
M2_T03_HUMAN_RESULT_RECEIPT_VERSION = "m2-t03-human-adjudication-result-receipt.v1"
M2_T03_HUMAN_RESULT_REPORT_VERSION = "m2-t03-human-adjudication-result-report.v1"
M2_T03_HUMAN_RESULT_PATH = (
    "evaluation/source-artifacts/m2-t03-human-adjudication-result.json"
)
M2_T03_HUMAN_RESULT_RECEIPT_PATH = (
    "evaluation/source-artifacts/m2-t03-human-adjudication-result-receipt.json"
)
M2_T03_HUMAN_RESULT_REPORT_PATH = "evaluation/reports/m2-t03-human-adjudication-result.json"


class HumanJudgment(BaseModel):
    """One complete, blind human judgment grounded in title/abstract text."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    paper_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    abstract: str = Field(min_length=1)
    source_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    verdict: Literal["SUPPORTED", "REJECTED"]
    evidence_slot: EvidenceSlot | None = None
    support_level: SupportLevel | None = None
    grounded_reason: str = Field(min_length=1, max_length=1200)
    supporting_excerpt: str | None = Field(default=None, max_length=1000)
    reviewer_id: str = Field(min_length=1, max_length=200)
    reviewed_at_utc: datetime
    notes: str | None = Field(default=None, max_length=2000)

    @field_validator("reviewed_at_utc")
    @classmethod
    def normalize_reviewed_at_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
            raise ValueError("reviewed_at_utc must be an aware UTC timestamp")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_grounded_judgment(self) -> Self:
        if not self.paper_id.strip():
            raise ValueError("paper_id must be non-blank")
        if not self.grounded_reason.strip():
            raise ValueError("grounded_reason must be non-blank")
        if not self.reviewer_id.strip():
            raise ValueError("reviewer_id must be non-blank")
        if self.source_text_sha256 != build_source_text_sha256(self.title, self.abstract):
            raise ValueError("source_text_sha256 must match title and abstract")
        excerpt = self.supporting_excerpt
        if self.verdict == "SUPPORTED":
            if self.evidence_slot is None:
                raise ValueError("SUPPORTED judgments require evidence_slot")
            if self.support_level is None:
                raise ValueError("SUPPORTED judgments require support_level")
            if not excerpt or (excerpt not in self.title and excerpt not in self.abstract):
                raise ValueError("SUPPORTED supporting_excerpt must be an exact title/abstract substring")
        else:
            if self.evidence_slot is not None or self.support_level is not None:
                raise ValueError("REJECTED judgments require null evidence_slot/support_level")
            if excerpt is not None and excerpt not in self.title and excerpt not in self.abstract:
                raise ValueError("REJECTED supporting_excerpt must be an exact title/abstract substring")
        return self


class CompletedHumanAdjudicationResult(BaseModel):
    """Sealed 33-item human result, intentionally separate from pending advisory data."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    result_version: Literal["m2-t03-human-adjudication-result.v1"]
    phase: Literal["M2"]
    task_id: Literal["M2-T03"]
    generated_from_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    generated_at_utc: datetime
    pending_bundle: ArtifactBinding
    review_protocol: ArtifactBinding
    candidate_count: Literal[33]
    paper_id_order: tuple[str, ...] = Field(min_length=33, max_length=33)
    judgments: tuple[HumanJudgment, ...] = Field(min_length=33, max_length=33)
    human_review: Literal["completed"]
    m2_t04: Literal["not_started"]
    scoring_eligible: Literal[False]

    @field_validator("generated_at_utc")
    @classmethod
    def normalize_generated_at_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
            raise ValueError("generated_at_utc must be an aware UTC timestamp")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_complete_order(self) -> Self:
        ids = tuple(judgment.paper_id for judgment in self.judgments)
        if len(ids) != 33 or len(set(ids)) != 33:
            raise ValueError("completed result must contain 33 unique paper IDs")
        if ids != self.paper_id_order:
            raise ValueError("completed result paper ID order must match judgments")
        return self


class HumanAdjudicationResultReceipt(BaseModel):
    """Receipt contract reserved for the PR #16 completed-result workflow."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    receipt_version: Literal["m2-t03-human-adjudication-result-receipt.v1"]
    phase: Literal["M2"]
    task_id: Literal["M2-T03"]
    generated_from_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    result: ArtifactBinding
    runner: ArtifactBinding
    candidate_count: Literal[33]
    human_review: Literal["completed"]
    m2_t04: Literal["not_started"]
    scoring_eligible: Literal[False]
    exit_code: Literal[0]


class HumanAdjudicationResultReport(BaseModel):
    """Report contract reserved for post-review comparison diagnostics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    report_version: Literal["m2-t03-human-adjudication-result-report.v1"]
    phase: Literal["M2"]
    task_id: Literal["M2-T03"]
    generated_from_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    generated_at_utc: datetime
    input_artifacts: tuple[ArtifactBinding, ...] = Field(min_length=1)
    artifact: tuple[ArtifactBinding, ...] = Field(min_length=1)
    candidate_count: Literal[33]
    judgment_count: Literal[33]
    human_review: Literal["completed"]
    m2_t04: Literal["not_started"]
    scoring_eligible: Literal[False]
    commands: tuple[str, ...] = Field(min_length=1)
    exit_codes: dict[str, Literal[0]]


def derive_adjudication_comparison(
    judgment: HumanJudgment,
    machine_evidence_slot: EvidenceSlot | None,
    machine_support_level: SupportLevel | None,
) -> Literal["CONFIRM", "REVISE", "REJECT"]:
    """Derive a post-review comparison without exposing it as human input."""

    if judgment.verdict == "REJECTED":
        return "REJECT"
    if (
        judgment.evidence_slot == machine_evidence_slot
        and judgment.support_level == machine_support_level
    ):
        return "CONFIRM"
    return "REVISE"
